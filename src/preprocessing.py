import re
import unicodedata

# Diccionario de normalizaciones para tipos de vía
normalizaciones = {
    r"\bavenida\b|\bav\b|\bavd\b|\bavenid\b": "avenida",
    r"\bcalle\b|\bca\b": "calle",
    r"\bpasaje\b|\bpsj\b|\bpsje\b": "psje",
    r"\bcarretera\b|\bctra\b": "ctra",
    r"\bmanzana\b|\bmz\b": "mz",
    r"\bjiron\b|\bjr\b": "jr",  # Añadido para cubrir "Jirón" o "JR"
    r"\burbanizacion\b|\burb\b": "urb",
    r"\bcooperativa\b|\bcoop\b": "coop"
}

# Lista de tipos de vía reconocidos
tipos_via = ["avenida", "calle", "psje", "ctra", "mz", "jr", "urb", "coop", "sector"]

# Lista de distritos de Lima
distritos_lima = [
    "ancón", "ate", "barranco", "breña", "callao", "cercado de lima", "chaclacayo", "chorrillos", 
    "comas", "el agustino", "independencia", "la molina", "la victoria", "lince", "los olivos", 
    "magdalena del mar", "miraflores", "monterrico", "puente piedra", "pueblo libre", "rimac", 
    "san borja", "san isidro", "san juan de lurigancho", "san juan de miraflores", "san martin de porres",
    "san miguel", "santiago de surco", "surquillo", "villa el salvador", "villa maría del triunfo", "lima"
]


def mover_distrito(texto):
    texto_lower = texto.lower()
    distrito_encontrado = None

    # Buscar y remover distrito
    for distrito in distritos_lima:
        if distrito in texto_lower:
            distrito_encontrado = distrito
            texto = re.sub(rf"\b{re.escape(distrito)}\b", "", texto, flags=re.IGNORECASE)
            break

    # Si se encontró distrito, eliminar "peru"
    if distrito_encontrado:
        texto = re.sub(r"\bperu\b", "", texto, flags=re.IGNORECASE)
        texto = texto.strip() + f" {distrito_encontrado}"
    else:
        # Si no hay distrito, mantener "peru" si existe
        texto = texto.strip()

    # Limpiar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def agregar_tipo_via_implicito(texto):
    palabras = texto.lower().split()
    if palabras and palabras[0] not in tipos_via:
        texto = "avenida " + texto  # asumimos que es una avenida si no se especifica
    return texto

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""

    # Paso 1: minúsculas y sin tildes
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")

    # Paso 2: eliminar signos raros
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    # Paso 3: normalizar tipos de vía
    for patron, reemplazo in normalizaciones.items():
        texto = re.sub(patron, reemplazo, texto)

    # Paso 4: mover distrito al final (o antes de "Perú" si está presente)
    texto = mover_distrito(texto)

    # Paso 5: agregar tipo de vía si no hay
    texto = agregar_tipo_via_implicito(texto)

    # Paso 65: eliminar menciones de departamentos
    texto = re.sub(r"\bdpto\.?\s*\w+\b", "", texto)
    texto = re.sub(r"\bdepto\.?\s*\w+\b", "", texto)
    texto = re.sub(r"\bdep\.?\s*\w+\b", "", texto)
    texto = re.sub(r"\bdepartamento\s*\w+\b", "", texto)

    # Paso 7: quitar palabras duplicadas
    palabras = texto.split()
    palabras_unicas = []
    for palabra in palabras:
        if palabra not in palabras_unicas:
            palabras_unicas.append(palabra)
    texto = " ".join(palabras_unicas)

    return texto

def limpiar_dataframe(df):
    df["direccion_limpia"] = df["DIRECCION"].apply(limpiar_texto)
    df["referencia_limpia"] = df["referencia"].apply(limpiar_texto)
    return df
