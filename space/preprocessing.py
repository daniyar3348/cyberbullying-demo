import re
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from model_cyberbullying import BERT_NAME, MAX_LEN


def clean_text(text):
    # Шаги предобработки по Fig. 1 статьи
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URL удаляем до пунктуации
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)               # слова, содержащие цифры
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_data(csv_path, text_col='text', label_col='label', bert_name=BERT_NAME, max_len=MAX_LEN):
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).map(clean_text).tolist()

    # Метки (строки или числа) -> индексы классов 0..N-1
    classes = sorted(df[label_col].unique())
    labels = df[label_col].map({c: i for i, c in enumerate(classes)})

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    enc = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    dataset = TensorDataset(enc['input_ids'], enc['attention_mask'], torch.tensor(labels.to_numpy()))
    return dataset, classes, tokenizer

# Пример использования:
# dataset, classes, tokenizer = prepare_data('../Dataset/data.csv')
# model = CustomBERTCNNAttentionModel(num_classes=len(classes))
