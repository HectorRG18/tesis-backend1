import re

def evaluar_calidad(direccion):
    """
    Evalúa si una dirección es problemática usando reglas heurísticas.
    Devuelve un diccionario con 'es_problematica' y 'motivo'.
    """

    direccion = direccion.strip().lower()

    if not direccion:
        return {"es_problematica": 1, "motivo_problema": "vacia"}

    if len(direccion) < 10:
        return {"es_problematica": 1, "motivo_problema": "direccion_muy_corta"}

    tipos_via = ["av", "avenida", "jr", "jiron", "ca", "calle", "psj", "pasaje", "carretera", "sector", "urb"]
    if not any(tipo in direccion for tipo in tipos_via):
        return {"es_problematica": 1, "motivo_problema": "sin_tipo_via"}

    if re.search(r"\bmz\b|\blt\b", direccion):
        return {"es_problematica": 1, "motivo_problema": "contiene_mz_lt"}

    if direccion.replace(" ", "").isdigit():
        return {"es_problematica": 1, "motivo_problema": "solo_numeros"}

    palabras = direccion.split()
    if len(set(palabras)) < len(palabras):
        return {"es_problematica": 1, "motivo_problema": "palabras_duplicadas"}

    # Más reglas pueden añadirse luego

    return {"es_problematica": 0, "motivo_problema": "ninguno"}
