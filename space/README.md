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

- `GET /api/status` — состояние модели
- `POST /api/predict` с JSON `{"text": "..."}` — вероятность кибербуллинга

Обученные веса пока не загружены, поэтому модель работает в демо-режиме: выходной слой инициализирован случайно, вероятность около 50 %.
