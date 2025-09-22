# limpiar_referencia.py

import torch
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración
MODEL_PATH = os.path.join(BASE_DIR, '..', 'modelos', 'bert_crf_referencia', 'modelo_bert_softmax.pt')
TOKENIZER_PATH = os.path.join(BASE_DIR, '..', 'modelos', 'bert_crf_referencia', 'modelo_bert_softmax_tokenizer')

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo BERT para extracción de entidades
class BertForEntityExtraction(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertForEntityExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss
        else:
            preds = torch.argmax(logits, dim=-1)
            return preds

# Cargar modelo y tokenizer
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = BertForEntityExtraction(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# Nueva función: Extrae la referencia usando el modelo
def extraer_referencia_limpia(texto_original):
    tokenizer, model = load_model()

    encoding = tokenizer(texto_original, return_offsets_mapping=True, max_length=MAX_LEN,
                         truncation=True, padding='max_length', return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    offsets = encoding['offset_mapping'][0].cpu().numpy()

    with torch.no_grad():
        preds = model(input_ids, attention_mask)[0]

    active_preds = preds[:attention_mask.sum().item()]
    active_offsets = offsets[:attention_mask.sum().item()]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:attention_mask.sum().item()]

    entity_tokens = []
    for pred, (start, end), token in zip(active_preds, active_offsets, tokens):
        if pred == 1 and start != end:
            entity_tokens.append((start, end))

    if not entity_tokens:
        return ""

    start = entity_tokens[0][0]
    end = entity_tokens[-1][1]

    referencia_extraida = texto_original[start:end]
    return referencia_extraida.strip()
