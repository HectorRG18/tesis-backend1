from models import get_db_connection
import bcrypt
from pydantic import BaseModel
from typing import List, Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO
from services.file_validation import validate_file, validate_excel_columns
import unicodedata
from src.preprocessing import limpiar_texto, mover_distrito  # Funciones de limpieza
from src.evaluate_quality import evaluar_calidad 
from src.limpiar_direccion import extraer_direccion_limpia 
from src.predecir_distanciaClasificacion import predecir_distancia , cargar_modelos
from src.limpiar_referencia import extraer_referencia_limpia 
from src.ubicacion_finder import CorredorLocationFinder
from src.predecir_problema import clasificar_direccion
from typing import Optional, List
from pydantic import BaseModel

location_finder = CorredorLocationFinder('AIzaSyBQSAOoJY09KmIHvzbblYIEFQTZhNIc1MY')
import pandas as pd

router = APIRouter()


# Pydantic models
class UserBase(BaseModel):
    username: str
    email: str
    password: str
    rol_id: int

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    rol: str

class UserResponseUnit(BaseModel):
    id: int
    username: str
    email: str
    rol_id: int

class LoginRequest(BaseModel):
    username: str
    password: str

class EntradaDireccionProblema(BaseModel):
    direcciones: List[str]

class SalidaDireccionProblema(BaseModel):
    direccion_final: str
    clasificacion_codigo: int
    mensaje: str

class EntradaDireccionReferencia(BaseModel):
    indexNum: int
    direccion: str
    referencia: str
    distrito: int

class DireccionResultado(BaseModel):
    indexNum: int
    direccion_original: str
    referencia_original: str
    distrito: int
    coordenadas: Tuple[float, float]
    distancia: float

class DireccionesRequest(BaseModel):
    entradas: List[EntradaDireccionReferencia]

class DireccionesResponse(BaseModel):
    resultados: List[DireccionResultado]
    
class ResultadoDireccion(BaseModel):
    indexNum: int
    direccion_original: str
    referencia_original: str
    coordenadas: Optional[List[float]] = None  # Acepta None
    distancia: float
    estado: str  # "éxito", "no_localizable" o "error"
    mensaje: Optional[str] = None  # Mensaje descriptivo opcional

class DireccionesResponse(BaseModel):
    resultados: List[ResultadoDireccion]

class DireccionGeocodificableUpdate(BaseModel):
    archivo_id: int
    direccion: str
    direccion_geocodificable: str  # "sí" o "no"

class DireccionesGeocodificablesRequest(BaseModel):
    actualizaciones: List[DireccionGeocodificableUpdate]


# Metricas
class GeocodificacionStats(BaseModel):
    si: int
    no: int
    nulos: int
    total: int
distritos_por_ubigeo = {
            150101: "Lima",
            150102: "Ancón",
            150103: "ATE",
            150104: "Barranco",
            150105: "Breña",
            150106: "Carabayllo",
            150107: "Chaclacayo",
            150108: "Chorrillos",
            150109: "Cieneguilla",
            150110: "Comas",
            150111: "El Agustino",
            150112: "Independencia",
            150113: "Jesús María",
            150114: "La Molina",
            150115: "La Victoria",
            150116: "Lince",
            150117: "Los Olivos",
            150118: "Lurigancho",
            150119: "Lurín",
            150120: "Magdalena del Mar",
            150121: "Pueblo Libre",
            150122: "Miraflores",
            150123: "Pachacamac",
            150124: "Pucusana",
            150125: "Puente Piedra",
            150126: "Punta Hermosa",
            150127: "Punta Negra",
            150128: "Rímac",
            150129: "San Bartolo",
            150130: "San Borja",
            150131: "San Isidro",
            150132: "San Juan de Lurigancho",
            150133: "San Juan de Miraflores",
            150134: "San Luis",
            150135: "San Martín de Porres",
            150136: "San Miguel",
            150137: "Santa Anita",
            150138: "Santa María del Mar",
            150139: "Santa Rosa",
            150140: "Santiago de Surco",
            150141: "Surquillo",
            150142: "Villa El Salvador",
            150143: "Villa María del Triunfo"
        }
        
@router.get("/estadisticas-geocodificacion")
async def get_geocodificacion_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Consulta para contar solo los valores 'si' y 'no' (excluyendo nulos)
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN direccion_geocodificable = 'si' THEN 1 END) as si,
                COUNT(CASE WHEN direccion_geocodificable = 'no' THEN 1 END) as no,
                COUNT(CASE WHEN direccion_geocodificable IS NOT NULL THEN 1 END) as total
            FROM addresses
        """)
        
        stats = cursor.fetchone()
        
        return {
            "si": stats[0],
            "no": stats[1],
            "total": stats[0] + stats[1]  
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()



@router.get("/estadisticas-distritos")
async def get_distritos_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT distrito, COUNT(*) 
            FROM addresses a
            WHERE distrito IS NOT NULL
            AND  a.direccion_geocodificable = 'si'
            GROUP BY distrito
        """)
        
        resultados = cursor.fetchall()

        labels = []
        valores = []

        for ubigeo, conteo in resultados:
            nombre = distritos_por_ubigeo.get(ubigeo, "Desconocido")
            labels.append(nombre) 
            valores.append(conteo)

        return {
            "labels": labels,
            "valores": valores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@router.get("/estadisticas-distritos-error")
async def get_distritos_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT distrito, COUNT(*) 
            FROM addresses a
            WHERE distrito IS NOT NULL
            AND  a.direccion_geocodificable = 'no'
            GROUP BY distrito
        """)
        
        resultados = cursor.fetchall()

        labels = []
        valores = []

        for ubigeo, conteo in resultados:
            nombre = distritos_por_ubigeo.get(ubigeo, "Desconocido")
            labels.append(nombre) 
            valores.append(conteo)

        return {
            "labels": labels,
            "valores": valores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
      
## fin de metricas   



# Ruta para login
@router.post("/login")
async def login(request: LoginRequest):
    username = request.username
    password = request.password

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()

    if user is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    hashed_password = user[0]

    if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
        cursor.execute("""
            SELECT u.id, u.username, r.rol AS rol
            FROM users u
            INNER JOIN rol r ON u.rol_id = r.id
            WHERE u.username = %s
        """, (username,))
        user_data = cursor.fetchone()
        return {
            "message": "Inicio de sesión exitoso",
            "user": {
                "id": user_data[0],
                "username": user_data[1],
                "rol": user_data[2]
            }
        }
    else:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
# Ruta para obtener usuarios
@router.get("/users", response_model=List[UserResponse])
async def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT u.id, u.username, u.email, r.rol AS rol
            FROM users u
            JOIN rol r ON u.rol_id = r.id
        """)
        users = cursor.fetchall()
        return [{"id": row[0], "username": row[1], "email": row[2], "rol": row[3]} for row in users]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

# Ruta para crear usuario
@router.post("/users/new")
async def create_user(user: UserBase):
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (username, email, password, rol_id) VALUES (%s, %s, %s, %s)", 
                       (user.username, user.email, hashed_password, user.rol_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

    return {"message": "Usuario creado correctamente"}

# Ruta para eliminar usuario
@router.delete("/users/{id}")
async def delete_user(id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM users WHERE id = %s", (id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

    return {"message": f"Usuario con ID {id} eliminado correctamente"}

# Ruta para actualizar usuario
@router.put("/users/{id}")
async def update_user(id: int, user: UserBase):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if user.password:
            hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
                UPDATE users SET username = %s, email = %s, password = %s, rol_id = %s WHERE id = %s
            """, (user.username, user.email, hashed_password, user.rol_id, id))
        else:
            cursor.execute("""
                UPDATE users SET username = %s, email = %s, rol_id = %s WHERE id = %s
            """, (user.username, user.email, user.rol_id, id))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

    return {"message": "Usuario actualizado correctamente"}

# Ruta para obtener usuario por ID
@router.get("/users/{id}", response_model=UserResponseUnit)
async def get_user_by_id(id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, username, email, rol_id FROM users WHERE id = %s", (id,))
    user = cursor.fetchone()

    if user:
        return {"id": user[0], "username": user[1], "email": user[2], "rol_id": user[3]}
    else:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

def normalizar_texto(texto):
    # Convierte a mayúsculas, elimina tildes, asteriscos y espacios extra
    texto = texto.upper().replace("*", "").strip()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto

@router.post("/validar-columnas")
async def validar_columnas(file: UploadFile = File(...)):
    required_columns = ["NUMERO GUIA", "VEHICULO", "DIRECCION", "LATITUD", "LONGITUD", "REFERENCIA"]

    contents = await file.read()
    import io
    file_io = io.BytesIO(contents)

    validate_file(file_io)

    import pandas as pd
    from fastapi import HTTPException

    try:
        df = pd.read_excel(file_io)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo Excel: {str(e)}")

    # Normalizamos los nombres de columnas del archivo
    normalized_columns = [normalizar_texto(col) for col in df.columns]
    required_normalized = [normalizar_texto(col) for col in required_columns]

    # Verificamos columnas faltantes
    missing = [col for col in required_normalized if col not in normalized_columns]

    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas: {missing}")

    return {"message": "Validación exitosa"}

@router.post("/procesar-direcciones")
async def procesar_direcciones(data: EntradaDireccionProblema):
    resultados = []

    for direccion in data.direcciones:
        direccion_final = mover_distrito(direccion)
        codigo, mensaje = clasificar_direccion(direccion_final)

        resultados.append({
            "direccion_final": direccion_final,
            "clasificacion_codigo": int(codigo),
            "clasificacion_mensaje": mensaje
        })

    return {"message": "Direcciones procesadas con éxito", "resultados": resultados}


@router.post("/limpiar-direcciones", response_model=DireccionesResponse)
async def limpiar_direcciones(request: DireccionesRequest):
    resultados = []
    modelos = cargar_modelos()

    for entrada in request.entradas:
        distancia = 0
        try:
            direccion_limpia = extraer_direccion_limpia(entrada.direccion)
            referencia_limpia = extraer_referencia_limpia(entrada.referencia)
            distancia = predecir_distancia(entrada.referencia, modelos)
            distrito = entrada.distrito

            resultado_finder = location_finder.encontrar_punto_final(
                direccion_limpia, 
                referencia_limpia, 
                distrito,
                distancia
            )

            # Construir objeto resultado según el estado
            resultado = {
                "indexNum": entrada.indexNum,
                "direccion_original": entrada.direccion,
                "referencia_original": entrada.referencia,
                "estado": resultado_finder.get('status', 'error'),
                "mensaje": resultado_finder.get('message'),
                "distancia": distancia
            }

            # Solo añadir coordenadas si fue exitoso
            if resultado_finder.get('status') == 'success':
                resultado["coordenadas"] = resultado_finder['punto_final']
            else:
                resultado["coordenadas"] = None

            resultados.append(resultado)

        except Exception as e:
            # Manejar errores inesperados para esta entrada específica
            resultados.append({
                "indexNum": entrada.indexNum,
                "direccion_original": entrada.direccion,
                "referencia_original": entrada.referencia,
                "coordenadas": None,
                "distancia": distancia,
                "estado": "error",
                "mensaje": f"Error inesperado: {str(e)}"
            })

    return {"resultados": resultados}
    


##Guardar en la base datos
from database import get_db_connection
from datetime import datetime

@router.post("/guardar-excel")
def guardar_excel(datos: list[dict]):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Crear registro en excel_files
        fecha_creacion = datetime.now().strftime("%Y-%m-%d")
        tiempo = datetime.now().strftime("%H:%M") 

        cursor.execute("""
            INSERT INTO excel_files ( fecha_creacion, tiempo)
            VALUES (%s, %s)
            RETURNING id;
        """, (fecha_creacion, tiempo))
        excel_file_id = cursor.fetchone()[0]

        for fila in datos:
            # 2. Insertar dirección
            cursor.execute("""
                INSERT INTO addresses (direccion, latitud, longitud, referencia, distrito, direccion_geocodificable)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                fila.get('direccion'),
                fila.get('latitud'),
                fila.get('longitud'),
                fila.get('referencia'),
                fila.get('distrito'),
                ''  # dirección geocodificable aún vacío
            ))
            address_id = cursor.fetchone()[0]

            # 3. Insertar entrega
            cursor.execute("""
                INSERT INTO deliveries (
                    excel_file_id, address_id,
                    numero_guia, vehiculo,
                    nombre_item, cantidad, codigo_item,
                    identificador_contacto, nombre_contacto, telefono, email_contacto,
                    fecha_min_entrega, fecha_max_entrega
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                excel_file_id,
                address_id,
                fila.get('numero_guia'),
                fila.get('vehiculo'),
                fila.get('nombre_item'),
                fila.get('cantidad'),
                fila.get('codigo_item'),
                fila.get('identificador_contacto'),
                fila.get('nombre_contacto'),
                fila.get('telefono'),
                fila.get('email_contacto'),
                fila.get('fecha_min_entrega'),
                fila.get('fecha_max_entrega')
            ))

        conn.commit()
        return {"message": "Datos guardados correctamente", "archivo_id": excel_file_id}

    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

@router.get("/obtener-archivos/{archivo_id}")
def obtener_entregas(archivo_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT 
                d.numero_guia,
                d.vehiculo,
                d.nombre_item,
                d.cantidad,
                d.codigo_item,
                d.identificador_contacto,
                d.nombre_contacto,
                d.telefono,
                d.email_contacto,
                a.direccion,
                a.latitud,
                a.longitud,
                d.fecha_min_entrega,
                d.fecha_max_entrega,
                a.referencia,
                a.distrito
            FROM deliveries d
            JOIN addresses a ON d.address_id = a.id
            WHERE d.excel_file_id = %s
        """
        cursor.execute(query, (archivo_id,))
        rows = cursor.fetchall()
        columnas = [desc[0] for desc in cursor.description]

        resultado = [dict(zip(columnas, fila)) for fila in rows]
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cursor.close()
        conn.close()

@router.put("/actualizar-coordenadas")
def actualizar_coordenadas_lote(items: list[dict]):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for item in items:
            archivo_id = item.get("archivo_id")
            direccion = item.get("direccion")
            referencia = item.get("referencia")
            latitud = item.get("latitud")
            longitud = item.get("longitud")


            cursor.execute("""
                UPDATE addresses
                SET latitud = %s,
                    longitud = %s
                WHERE id IN (
                    SELECT a.id
                    FROM addresses a
                    JOIN deliveries d ON d.address_id = a.id
                    WHERE d.excel_file_id = %s
                    AND a.direccion = %s
                    AND a.referencia = %s
                )
            """, (latitud, longitud, archivo_id, direccion, referencia))

        conn.commit()
        return {"message": "Coordenadas actualizadas correctamente"}

    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

@router.get("/fecha-archivo/{archivo_id}")
def obtener_fecha_formateada(archivo_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Obtener fecha de la base de datos (asumiendo tabla 'excel_files')
        cursor.execute("""
            SELECT fecha_creacion 
            FROM excel_files 
            WHERE id = %s
        """, (archivo_id,))
        
        fecha_db = cursor.fetchone()
        if not fecha_db:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # 2. Formatear fecha
        fecha = fecha_db[0]  # Obtener el valor datetime
        dia_semana = fecha.strftime("%A")  # Día en inglés (ej. "Wednesday")
        dia_mes = fecha.strftime("%d")     # Día numérico (ej. "11")
        
        # 3. Traducir día a español
        dias_traduccion = {
            "Monday": "Lunes",
            "Tuesday": "Martes",
            "Wednesday": "Miércoles",
            "Thursday": "Jueves",
            "Friday": "Viernes",
            "Saturday": "Sábado",
            "Sunday": "Domingo"
        }
        
        dia_es = dias_traduccion.get(dia_semana, dia_semana)
        
        # 4. Devolver formato "Miércoles/11"
        return {"fecha_formateada": f"{dia_es}/{dia_mes}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        
    finally:
        cursor.close()
        conn.close()        

@router.get("/lista-archivos")
def obtener_archivos_con_fechas() -> List[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. Obtener todos los archivos
        cursor.execute("""
            SELECT id, fecha_creacion, tiempo
            FROM excel_files 
            ORDER BY fecha_creacion DESC
        """)
        archivos = cursor.fetchall()

        # 2. Validar si hay resultados
        if not archivos:
            return []

        # 3. Procesar resultados
        resultado = []
        for archivo_id, fecha_creacion, tiempo in archivos:
            resultado.append({
                "id": archivo_id,
                "fecha_creacion": fecha_creacion.strftime("%Y-%m-%d"),  # o "%d/%m/%Y" si prefieres
                "tiempo": tiempo
            })


        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener archivos: {str(e)}")

    finally:
        cursor.close()
        conn.close()

@router.put("/actualizar-direccion-geocodificable")
def actualizar_direccion_geocodificable_lote(items: list[dict]):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for item in items:
            archivo_id = item.get("archivo_id")
            direccion = item.get("direccion")
            referencia = item.get("referencia")
            estado = item.get("estado")

            direccion_geocodificable = "si" if estado == "success" else "no"

            cursor.execute("""
                UPDATE addresses
                SET direccion_geocodificable = %s
                WHERE id IN (
                    SELECT a.id
                    FROM addresses a
                    JOIN deliveries d ON d.address_id = a.id
                    WHERE d.excel_file_id = %s
                    AND a.direccion = %s
                    AND a.referencia = %s
                )
            """, (direccion_geocodificable, archivo_id, direccion, referencia))

        conn.commit()
        return {"message": "Campo direccion_geocodificable actualizado correctamente"}

    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()


