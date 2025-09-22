# src/feature_extractor.py
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            contiene_mz = 1 if re.search(r'\bMz\b', text, flags=re.IGNORECASE) else 0
            contiene_lt = 1 if re.search(r'\bLt\b', text, flags=re.IGNORECASE) else 0
            tiene_numeros = 1 if re.search(r'\d+', text) else 0
            longitud = len(text)
            es_corta = 1 if longitud < 20 else 0

            features.append([contiene_mz, contiene_lt, tiene_numeros, longitud, es_corta])

        return pd.DataFrame(features, columns=[
            'contiene_mz', 'contiene_lt', 'tiene_numeros', 'longitud', 'es_corta'
        ])
