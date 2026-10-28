import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from classifier_errors import ClassifierError

ROOT = Path(__file__).resolve().parent
# Значения из .env важнее унаследованных переменных окружения
load_dotenv(ROOT / '.env', override=True)

# local – собственная модель CustomBERTCNNAttentionModel; claude – определение через Claude API
ENGINE = os.environ.get('DETECTOR_ENGINE', 'local').strip().lower()
if ENGINE == 'claude':
    from classifier_claude import ClaudeClassifier as Classifier
else:
    from classifier_local import LocalModelClassifier as Classifier

THRESHOLD = 0.5
# Запросы к Claude API платные; собственная модель дешевле, а карта влияния слов отправляет запрос на каждое нажатие
RATE_LIMIT_PER_MIN = int(os.environ.get('RATE_LIMIT_PER_MIN', '20' if ENGINE == 'claude' else '60'))

classifier = Classifier()
state = {'status': 'loading', 'error': None}


def load_classifier():
    try:
        classifier.load()
        state['status'] = 'ready'
    except Exception as e:
        state.update(status='error', error=str(e))


@asynccontextmanager
async def lifespan(app):
    # Локальная модель грузится долго, поэтому сервер стартует сразу, а движок готовится в фоне
    threading.Thread(target=load_classifier, daemon=True).start()
    yield


app = FastAPI(title='Детектор кибербуллинга', lifespan=lifespan)
# Страница на GitHub Pages обращается к API с другого домена
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','),
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

# Ограничение запросов с одного адреса: каждый запрос к Claude API оплачивается
_requests = defaultdict(deque)
_requests_lock = threading.Lock()


def check_rate_limit(ip):
    now = time.monotonic()
    with _requests_lock:
        q = _requests[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(429, 'Слишком много запросов, повторите попытку через минуту')
        q.append(now)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    explain: bool = True  # считать влияние каждого слова (только для собственной модели)


@app.get('/api/status')
def status():
    return {**state, **classifier.info(), 'engine': classifier.name, 'threshold': THRESHOLD}


@app.post('/api/predict')
def predict(req: PredictRequest, request: Request):
    if state['status'] != 'ready':
        raise HTTPException(503, 'Сервис ещё загружается' if state['status'] == 'loading' else state['error'])
    check_rate_limit(request.client.host if request.client else 'unknown')
    try:
        result = classifier.classify(req.text, explain=req.explain)
    except ClassifierError as e:
        raise HTTPException(e.status, str(e))
    return {**result, 'engine': classifier.name}


app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='static')
