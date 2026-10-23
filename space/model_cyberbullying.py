import torch
import torch.nn as nn
from transformers import AutoModel

# Параметры (размерности по Fig. 2 статьи)
BERT_NAME = 'bert-base-uncased'
MAX_LEN = 128
NUM_HEADS = 8        # в статье не указано
CONV_CHANNELS = 64
KERNEL_SIZE = 3      # в статье не указано; padding сохраняет длину 128
HIDDEN_DIM = 256
DROPOUT = 0.1


class CustomBERTCNNAttentionModel(nn.Module):
    def __init__(self, num_classes=2, bert_name=BERT_NAME, num_heads=NUM_HEADS,
                 conv_channels=CONV_CHANNELS, kernel_size=KERNEL_SIZE,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        # BERT: [B, 128] -> [B, 128, 768]
        self.bert = AutoModel.from_pretrained(bert_name)
        embed_dim = self.bert.config.hidden_size

        # Multihead self-attention: [B, 128, 768] -> [B, 128, 768]
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # CNN: [B, 768, 128] -> [B, 64, 128] -> [B, 64, 1]
        self.conv1 = nn.Conv1d(embed_dim, conv_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Классификатор: [B, 64] -> [B, 256] -> [B, num_classes]
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Паддинг-токены не используются как ключи внимания
        h, _ = self.attention(h, h, h, key_padding_mask=attention_mask == 0, need_weights=False)

        x = h.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)

        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        # Логиты; softmax из формулы (3) применяется в CrossEntropyLoss / при предсказании
        return self.fc2(x)


if __name__ == '__main__':
    model = CustomBERTCNNAttentionModel(num_classes=2)
    print(model)
    print(f'Параметров: {sum(p.numel() for p in model.parameters()):,}')

    # Проверка размерностей на фиктивном входе [1, 128]
    input_ids = torch.randint(0, model.bert.config.vocab_size, (1, MAX_LEN))
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    print('Выход:', tuple(logits.shape))
