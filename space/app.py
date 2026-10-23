import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from transformers import logging as hf_logging

ROOT = Path(__file__).resolve().parent
# Локально модель лежит в ../Models; на сервере (Hugging Face Spaces) путь задается переменной окружения
MODELS_DIR = Path(os.environ.get('MODELS_DIR', ROOT.parent / 'Models'))
sys.path.insert(0, str(MODELS_DIR))
from model_cyberbullying import BERT_NAME, MAX_LEN, CustomBERTCNNAttentionModel  # noqa: E402
from preprocessing import clean_text  # noqa: E402

# Обученные веса: torch.save(model.state_dict(), ...) и список классов в том же порядке, что и выходы модели
CHECKPOINT = MODELS_DIR / 'checkpoints' / 'model.pt'
CLASSES_FILE = MODELS_DIR / 'checkpoints' / 'classes.json'
DEFAULT_CLASSES = ['не кибербуллинг', 'кибербуллинг']
NEGATIVE_NAMES = {'не кибербуллинг', 'not_cyberbullying', 'not cyberbullying', '0'}
THRESHOLD = 0.5

hf_logging.set_verbosity_error()
state = {'status': 'loading', 'error': None, 'trained': False, 'classes': DEFAULT_CLASSES}
model = tokenizer = None


def load_model():
    global model, tokenizer
    try:
        classes = json.loads(CLASSES_FILE.read_text('utf-8')) if CLASSES_FILE.exists() else DEFAULT_CLASSES
        torch.manual_seed(0)  # стабильный вывод необученной головы между перезапусками
        m = CustomBERTCNNAttentionModel(num_classes=len(classes))
        if CHECKPOINT.exists():
            m.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
            state['trained'] = True
        tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
        model = m.eval()
        state.update(status='ready', classes=classes)
    except Exception as e:
        state.update(status='error', error=str(e))


@asynccontextmanager
async def lifespan(app):
    # Загрузка BERT занимает время, поэтому сервер стартует сразу, а модель грузится в фоне
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(title='Детектор кибербуллинга — демо', lifespan=lifespan)
# Страница на GitHub Pages обращается к API с другого домена
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get('/api/status')
def status():
    return {**state, 'model': 'CustomBERTCNNAttentionModel', 'encoder': BERT_NAME, 'max_len': MAX_LEN, 'threshold': THRESHOLD}


@app.post('/api/predict')
def predict(req: PredictRequest):
    if state['status'] != 'ready':
        raise HTTPException(503, 'Модель ещё загружается' if state['status'] == 'loading' else state['error'])

    cleaned = clean_text(req.text)
    enc = tokenizer([cleaned], max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.softmax(model(enc['input_ids'], enc['attention_mask']), dim=-1)[0].tolist()

    classes = state['classes']
    # Вероятность кибербуллинга = 1 − P(класса «нет кибербуллинга»); подходит и для многоклассовой модели
    neg = next((i for i, c in enumerate(classes) if str(c).lower() in NEGATIVE_NAMES), 0)
    probability = 1 - probs[neg]
    return {
        'probability': probability,
        'is_cyberbullying': probability >= THRESHOLD,
        'classes': [{'name': c, 'probability': p} for c, p in zip(classes, probs)],
        'clean_text': cleaned,
        'tokens': int(enc['attention_mask'].sum()),
        'truncated': len(tokenizer(cleaned)['input_ids']) > MAX_LEN,
        'trained': state['trained'],
    }


app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='static')
