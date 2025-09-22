import torch
import joblib
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch import nn
from scipy.sparse import csr_matrix
import re
from scipy.sparse import csr_matrix, hstack  # Añadir hstack aquí
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClassificationModel(nn.Module):
    def __init__(self, n_classes, bert_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

def cargar_modelos():
    """Carga todos los modelos necesarios."""
    # 1. Cargar modelo BERT de clasificación
    bert_model = ClassificationModel(n_classes=3, bert_model_name='dccuchile/bert-base-spanish-wwm-cased').to(DEVICE)

    bert_model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, '..', 'modelos', 'bert_clasificacion_rf', 'best_classification_model.pth'),
        map_location=DEVICE
    ))
    bert_model.eval()
    
    # 2. Cargar Random Forest y vectorizador TF-IDF
    rf_model = joblib.load(os.path.join(BASE_DIR, '..', 'modelos', 'randomForest_distancia', 'rf_balanced.pkl'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, '..', 'modelos', 'randomForest_distancia', 'tfidf_vectorizer.pkl'))
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    return {
        'bert_model': bert_model,
        'rf_model': rf_model,
        'vectorizer': vectorizer,
        'tokenizer': tokenizer,
        'expected_features': rf_model.n_features_in_  # Para verificar dimensionalidad
    }


def predecir_distancia(texto, modelos):
    """Predice la distancia para una sola referencia de texto."""
    # 1. Asegurar que tenemos una lista aunque sea un solo texto
    textos = [texto] if isinstance(texto, str) else texto
    
    # 2. Clasificación con BERT
    inputs = modelos['tokenizer'](
        textos, 
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(DEVICE)
    
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    with torch.no_grad():
        logits = modelos['bert_model'](**inputs)
        probas = torch.softmax(logits, dim=1)
    
    grupos_idx = torch.argmax(logits, dim=1).cpu().numpy()
    grupos = ['cercania_corta', 'cercania_inmediata', 'distancia_media']
    
    # 3. Preparar features
    tfidf_features = modelos['vectorizer'].transform(textos)
    
    # Features manuales
    def extract_features(text):
        text = str(text).lower()
        return [
            len(re.findall(r'(\d+)\s*cuadra', text)),
            int('frente' in text),
            int('metros' in text)
        ]
    
    manual_features = np.array([extract_features(t) for t in textos])
    
    # One-hot encoding
    grupo_features = np.zeros((len(textos), 3))
    for i, idx in enumerate(grupos_idx):
        grupo_features[i, idx] = 1
    
    # Combinar features
    X = np.hstack([
        tfidf_features.toarray()[:, :modelos['expected_features']-6],
        grupo_features,
        manual_features
    ])
    
    # 4. Predecir y retornar solo el primer valor (si es un solo texto)
    preds = modelos['rf_model'].predict(X)
    return float(preds[0])  # Convertir a float simple


