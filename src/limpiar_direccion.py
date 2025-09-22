import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# 2. Cargar modelo y tokenizer
def load_model(model_dir):
    """Carga el modelo y tokenizer desde el directorio especificado"""
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        model = BertForTokenClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

# 3. Función para predecir y alinear sin post-procesamiento
def extraer_direccion_limpia(text):
    model_dir = os.path.join(BASE_DIR, "..", "modelos", "bert_direcciones")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_model(model_dir)
    model.to(device)
    model.eval()

    max_len = 128
    words = text.split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predicción
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predictions = predictions[0].cpu().numpy()
    pred_labels = [model.config.id2label[pred] for pred in predictions]
    word_ids = encoding.word_ids()

    word_level_preds = []
    current_word_id = None
    for word_id, label in zip(word_ids, pred_labels):
        if word_id is not None and word_id != current_word_id:
            word_level_preds.append(label)
            current_word_id = word_id

    predicted_address = ' '.join([
        word for word, label in zip(words, word_level_preds) if label != 'O'
    ])

    return predicted_address.strip()
