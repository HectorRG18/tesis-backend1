import pandas as pd
import re
import joblib
import sys
import os
# ✅ 2. Añadir el path a la carpeta src
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_extractor import FeatureExtractor 


def entrenar_modelo(ruta_datos):
    # Cargar datos
    df = pd.read_excel(ruta_datos)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Aplicar reglas de prioridad
    df['problema'] = df['direccion'].apply(aplicar_reglas)
    
    # Dividir datos
    X = df['direccion']
    y = df['problema']
    
    # Crear pipeline
    model = Pipeline([
        ('features', FeatureUnion([
            ('text', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(1, 4),
                max_features=500,
                lowercase=False
            )),
            ('custom', FeatureExtractor())
        ])),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight={0:1, 1:2, 2:2, 3:3},
            random_state=42
        ))
    ])
    
    # Entrenar
    model.fit(X, y)
    
    # Guardar modelo
    joblib.dump(model, '../modelos/randomForest_problematica/modelo_problemas.pkl')
    print("Modelo entrenado y guardado correctamente")

def aplicar_reglas(direccion):
    direccion = re.sub(r'\s+', ' ', direccion).strip()
    if len(direccion) < 20:
        return 3
    if re.search(r'\bMz\bmanzana\b|\bLt\b', direccion, flags=re.IGNORECASE):
        return 1
    if not re.search(r'\d+', direccion):
        return 2
    return 0

if __name__ == "__main__":
    entrenar_modelo("../data/processed/dataset_limpio.xlsx")