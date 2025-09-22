import re
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extractor import FeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    modelo_path = os.path.join(BASE_DIR, '..', 'modelos', 'randomForest_problematica', 'modelo_problemas.pkl')
    modelo = joblib.load(modelo_path)

except FileNotFoundError:
    modelo = None
    print("Advertencia: Modelo no encontrado. Ejecuta primero modelo.py")

def clasificar_direccion(direccion):

    if not modelo:
        raise ValueError("Modelo no cargado. Verifica que modelo_direcciones.pkl exista")
    
    # Limpieza básica
    direccion = re.sub(r'\s+', ' ', direccion).strip()
    
    # Aplicar reglas en orden de prioridad
    if re.search(r'\b(?:Mz|manzana|Lt|lote|int|dep|DPTO)\b', direccion, flags=re.IGNORECASE):
        return 1, "⚠️ Contiene 'Mz'/'Lt / interiores'"
    if len(direccion) < 15:
        return 3, "⚠️ Muy corta (<20 caracteres)"
    if not re.search(r'\d+', direccion):
        return 2, "⚠️ Sin numeración"
    
    # Si pasa todas las reglas, usar el modelo para confirmar
    prob = modelo.predict([direccion])[0]

    # Mapear códigos a mensajes
    problemas = {
        0: "✅ Correcta",
        1: "⚠️ Contiene 'Mz'/'Lt'",
        2: "⚠️ Sin numeración",
        3: "⚠️ Muy corta"
    }
    
    return prob, problemas.get(prob, "Desconocido")

def obtener_explicacion_problema(codigo):
    """Devuelve el mensaje descriptivo para cada código de problema"""
    explicaciones = {
        0: "La dirección es correcta y completa",
        1: "Contiene referencias a manzana (Mz) y/o lote (Lt)",
        2: "Carece de numeración o número identificable",
        3: "Es demasiado corta (menos de 20 caracteres)"
    }
    return explicaciones.get(codigo, "Código de problema desconocido")