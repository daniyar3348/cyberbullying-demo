---
title: Cyberbullying Detector Demo
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Детектор кибербуллинга — сервер демо

Серверная часть демо гибридной модели CustomBERTCNNAttentionModel (BERT + Multihead Self-Attention + CNN), проект AP23488900.

- `GET /api/status` — состояние модели и сведения об обучении
- `POST /api/predict` с JSON `{"text": "..."}` — вероятность кибербуллинга и вероятности по классам

Модель обучена на англоязычном наборе cyberbullying_tweets.csv (44 255 текстов, 6 классов). На тестовой выборке: точность 89,6 % по шести классам, F1 «кибербуллинг / нет» 95,5 %. Тексты на казахском и русском языках эта версия модели не распознает.
