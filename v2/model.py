import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_path, num_emotions=28, hidden_size=768):
        super().__init__()
        
        # Загружаем RuBERT-base
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Head 1: Популярность (binary classification)
        self.popularity_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # sigmoid будет в loss
        )
        
        # Head 2: Эмоции (multi-label classification)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)  # sigmoid будет в loss
        )
    
    def forward(self, input_ids, attention_mask, task='both'):
        """
        Args:
            input_ids: токены текста
            attention_mask: маска внимания
            task: 'popularity', 'emotion', или 'both'
        
        Returns:
            (pop_out, emo_out) - один из них может быть None
        """
        # Получаем эмбеддинги от BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        pop_out = None
        emo_out = None
        
        if task in ['popularity', 'both']:
            pop_out = self.popularity_head(cls_embedding)
        
        if task in ['emotion', 'both']:
            emo_out = self.emotion_head(cls_embedding)
        
        return pop_out, emo_out
    
    def freeze_bert(self):
        """Заморозить все веса BERT (для первых эпох обучения)"""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("🔒 BERT заморожен")
    
    def unfreeze_bert(self, num_layers=2):
        """Разморозить верхние N слоёв BERT для fine-tuning"""
        # Размораживаем последние N encoder layers
        layers = self.bert.encoder.layer
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"🔓 Разморожено верхних {num_layers} слоёв BERT")
    
    def count_parameters(self):
        """Считаем количество обучаемых параметров"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Тест модели
    print("🧪 Тестирую архитектуру модели...")
    
    model_path = "./models/rubert-base"
    model = MultiTaskClassifier(model_path, num_emotions=28)
    model.freeze_bert()
    
    total, trainable = model.count_parameters()
    print(f"📊 Всего параметров: {total:,}")
    print(f"📊 Обучаемых: {trainable:,}")
    
    # Тестовый forward pass
    batch_size = 4
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        pop_out, emo_out = model(dummy_input_ids, dummy_attention_mask, task='both')
    
    print(f"✅ Popularity output shape: {pop_out.shape}")
    print(f"✅ Emotion output shape: {emo_out.shape}")
    print(f"✅ Модель работает корректно!")
