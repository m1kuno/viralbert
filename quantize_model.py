import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path

# Архитектура модели
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_path, num_emotions=28, hidden_size=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        self.popularity_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, input_ids, attention_mask, task='both'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        pop_out = None
        emo_out = None
        
        if task in ['popularity', 'both']:
            pop_out = self.popularity_head(cls_embedding)
        if task in ['emotion', 'both']:
            emo_out = self.emotion_head(cls_embedding)
        
        return pop_out, emo_out

# Пути
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "rubert-base")  # str() для абсолютного пути
CLASSIFIER_PATH = str(PROJECT_ROOT / "models" / "multitask_best.pth")

print(f"📥 Загружаю модель из {MODEL_PATH}...")
model = MultiTaskClassifier(MODEL_PATH, num_emotions=28)

print(f"📥 Загружаю веса из {CLASSIFIER_PATH}...")
checkpoint = torch.load(CLASSIFIER_PATH, map_location='cpu')
