"""Локальная модель CustomBERTCNNAttentionModel (режим DETECTOR_ENGINE=local, по умолчанию)."""
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers import logging as hf_logging

# Локально модель лежит в ../Models; на сервере путь задается переменной окружения
MODELS_DIR = Path(os.environ.get('MODELS_DIR', Path(__file__).resolve().parent.parent / 'Models'))
sys.path.insert(0, str(MODELS_DIR))
from model_cyberbullying import BERT_NAME, MAX_LEN, CustomBERTCNNAttentionModel  # noqa: E402
from preprocessing import clean_text  # noqa: E402

# Обученные веса: torch.save(model.state_dict(), ...) и список классов в том же порядке, что и выходы модели
CHECKPOINT = MODELS_DIR / 'checkpoints' / 'model.pt'
CLASSES_FILE = MODELS_DIR / 'checkpoints' / 'classes.json'
DEFAULT_CLASSES = ['не кибербуллинг', 'кибербуллинг']
NEGATIVE_NAMES = {'не кибербуллинг', 'not_cyberbullying', 'not cyberbullying', '0'}
# Для скольких первых слов считать влияние на оценку (каждое слово – еще один прогон модели)
MAX_EXPLAIN_WORDS = 60

hf_logging.set_verbosity_error()


class LocalModelClassifier:
    name = 'local'

    def __init__(self):
        self.model = self.tokenizer = None
        self.classes = DEFAULT_CLASSES
        self.trained = False

    def load(self):
        self.classes = json.loads(CLASSES_FILE.read_text('utf-8')) if CLASSES_FILE.exists() else DEFAULT_CLASSES
        torch.manual_seed(0)  # стабильный вывод необученной головы между перезапусками
        m = CustomBERTCNNAttentionModel(num_classes=len(self.classes))
        if CHECKPOINT.exists():
            m.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
            self.trained = True
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
        self.model = m.eval()

    def info(self):
        info = {'model': 'CustomBERTCNNAttentionModel', 'encoder': BERT_NAME, 'max_len': MAX_LEN,
                'trained': self.trained, 'classes': self.classes}
        # Сведения об обучении, которые сохраняет Models/train.py
        try:
            cfg = json.loads((CHECKPOINT.parent / 'train_config.json').read_text('utf-8'))
            test = json.loads((CHECKPOINT.parent / 'metrics.json').read_text('utf-8'))['test']
            info['training'] = {'dataset': Path(cfg['csv']).name, 'n_texts': cfg['n_texts'],
                                'test_accuracy': test['accuracy'], 'binary_f1': test.get('binary', {}).get('f1')}
        except (OSError, KeyError, ValueError):
            pass
        return info

    def classify(self, text, threshold=0.5, explain=True):
        # Слова исходного текста вместе с пробелами. Очистка по словам дает тот же результат, что clean_text(text),
        # поэтому каждому слову соответствует не больше одного слова очищенного текста
        segments = [{'text': s, 'clean': clean_text(s), 'space': s.isspace()} for s in re.split(r'(\s+)', text) if s]
        words = [s['clean'] for s in segments if s['clean']]
        cleaned = ' '.join(words)
        n_explain = min(len(words), MAX_EXPLAIN_WORDS) if explain else 0
        # Влияние слова – насколько упадет вероятность кибербуллинга, если его убрать; все варианты считаются одним пакетом
        variants = [cleaned] + [' '.join(words[:i] + words[i + 1:]) for i in range(n_explain)]
        enc = self.tokenizer(variants, max_length=MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            probs = torch.softmax(self.model(enc['input_ids'], enc['attention_mask']), dim=-1).tolist()
        # Вероятность кибербуллинга = 1 − P(класса «нет кибербуллинга»); подходит и для многоклассовой модели
        neg = next((i for i, c in enumerate(self.classes) if str(c).lower() in NEGATIVE_NAMES), 0)
        probability = 1 - probs[0][neg]
        result = {
            'probability': probability,
            'is_cyberbullying': probability >= threshold,
            'classes': [{'name': c, 'probability': p} for c, p in zip(self.classes, probs[0])],
            'clean_text': cleaned,
            'tokens': int(enc['attention_mask'][0].sum()),
            'truncated': len(self.tokenizer(cleaned)['input_ids']) > MAX_LEN,
            'trained': self.trained,
        }
        if explain:
            impacts = iter(probability - (1 - p[neg]) for p in probs[1:])
            result['words'] = []
            for s in segments:
                if s['space']:
                    result['words'].append({'text': s['text'], 'type': 'space'})
                elif not s['clean']:
                    result['words'].append({'text': s['text'], 'type': 'ignored'})
                else:
                    result['words'].append({'text': s['text'], 'type': 'word', 'impact': next(impacts, None)})
        return result
