from fastapi import HTTPException
import pandas as pd

def validate_file(file_io, max_size_mb=5, allowed_types=None):
    if allowed_types is None:
        allowed_types = ["xlsx"]

    file_io.seek(0, 2)  # Ir al final del archivo para obtener tamaño
    file_size = file_io.tell()
    file_io.seek(0)  # Volver al inicio

    size_mb = file_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise HTTPException(status_code=400, detail=f"El archivo supera los {max_size_mb}MB permitidos")

    # Comprobar extensión
    if hasattr(file_io, "name"):
        filename = file_io.name
    else:
        filename = "archivo.xlsx"  # por si no tiene nombre

    extension = filename.split(".")[-1].lower()
    if extension not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: .{extension}")

def validate_excel_columns(file_io, required_columns: list[str]) -> None:
    try:
        df = pd.read_excel(file_io)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo Excel: {str(e)}")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas obligatorias: {', '.join(missing)}"
        )
