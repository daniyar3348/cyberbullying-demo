FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    MODELS_DIR=/home/user/app
WORKDIR /home/user/app

# CPU-сборка PyTorch: на бесплатном сервере нет GPU, а CUDA-версия весит несколько гигабайт
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Энкодер BERT скачивается при сборке, чтобы сервер не тратил на это время при запуске
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased'); AutoTokenizer.from_pretrained('bert-base-uncased')"

# Код и веса копируются последними: при их изменении не приходится заново ставить библиотеки и скачивать BERT
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
