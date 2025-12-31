# test_viralbert.py - Тесты для ViralBERT v2

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import time

# ========== АРХИТЕКТУРА МОДЕЛИ ==========
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

# ========== ЭМОЦИИ ==========
EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

EMOTION_WEIGHTS = {
    'amusement': 0.15, 'excitement': 0.12, 'joy': 0.10,
    'surprise': 0.08, 'admiration': 0.07, 'love': 0.06,
    'anger': 0.05, 'fear': 0.04, 'disgust': 0.03,
    'sadness': -0.02, 'neutral': -0.05
}

# ========== ЗАГРУЗКА МОДЕЛИ ==========
print("📥 Загружаю модель...")

# ВАЖНО: Укажи правильные пути (из твоего ShortsBot проекта)
MODEL_PATH = r"C:\Users\konst\OneDrive\Рабочий стол\ShortsBot-main\ShortsBot-main\models\rubert-base"
CLASSIFIER_PATH = r"C:\Users\konst\OneDrive\Рабочий стол\ShortsBot-main\ShortsBot-main\models\multitask_best.pth"

# Проверка существования файлов
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"❌ Не найдена папка модели: {MODEL_PATH}")
if not Path(CLASSIFIER_PATH).exists():
    raise FileNotFoundError(f"❌ Не найден файл весов: {CLASSIFIER_PATH}")

# Загрузка токенизатора (игнорируем предупреждение)
import warnings
warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Загрузка модели
model = MultiTaskClassifier(MODEL_PATH, num_emotions=28, hidden_size=768)

checkpoint = torch.load(CLASSIFIER_PATH, map_location='cpu', weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Модель загружена!\n")

# ========== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ==========
def predict(text: str, show_details=False):
    """Предсказывает вирусность и эмоции для текста"""
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        pop_logits, emo_logits = model(encoding['input_ids'], encoding['attention_mask'], task='both')
        
        # Вирусность
        base_viral = torch.sigmoid(pop_logits).item()
        
        # Эмоции
        emo_probs = torch.sigmoid(emo_logits).squeeze(0).numpy()
    
    # Топ эмоции
    top_emotions = []
    emotion_boost = 0.0
    
    for i, (name, score) in enumerate(zip(EMOTION_NAMES, emo_probs)):
        if score >= 0.3:
            top_emotions.append((name, float(score)))
            emotion_boost += EMOTION_WEIGHTS.get(name, 0) * score
    
    top_emotions = sorted(top_emotions, key=lambda x: x[1], reverse=True)
    
    # Финальный скор
    final_score = base_viral * 0.8 + emotion_boost * 0.2
    final_score = max(0.0, min(1.0, final_score))
    
    # Вывод
    if show_details:
        print(f"📊 Базовая вирусность: {base_viral:.2%}")
        print(f"🎭 Эмоции (топ-3):")
        for emo, score in top_emotions[:3]:
            print(f"   • {emo}: {score:.2%}")
        print(f"⚡ Emotion boost: {emotion_boost:+.3f}")
        print(f"🎯 Итоговая вирусность: {final_score:.2%}\n")
    
    return {
        'viral_score': final_score,
        'base_viral': base_viral,
        'emotions': top_emotions,
        'emotion_boost': emotion_boost
    }

# ========== ТЕСТОВЫЕ ПРИМЕРЫ ==========
print("=" * 60)
print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
print("=" * 60)

test_cases = [
    # Высокая вирусность
    {
        "text": "ЭТО НЕВЕРОЯТНО! Я просто в шоке от результатов! 😱",
        "expected": "high",
        "description": "Эмоциональный заголовок с капсом"
    },
    {
        "text": "Как я заработал миллион за месяц: СЕКРЕТНЫЙ метод",
        "expected": "high",
        "description": "Кликбейт про деньги"
    },
    {
        "text": "Врачи в ШОКЕ! Этот продукт творит чудеса",
        "expected": "high",
        "description": "Сенсационный заголовок"
    },
    
    # Средняя вирусность
    {
        "text": "Интересный факт о космосе, который вы не знали",
        "expected": "medium",
        "description": "Познавательный контент"
    },
    {
        "text": "5 лайфхаков для повышения продуктивности",
        "expected": "medium",
        "description": "Практические советы"
    },
    
    # Низкая вирусность
    {
        "text": "Сегодня я ходил в магазин и купил молоко",
        "expected": "low",
        "description": "Обычная бытовая фраза"
    },
    {
        "text": "Отчёт о работе за второй квартал 2024 года",
        "expected": "low",
        "description": "Формальный текст"
    },
    {
        "text": "Инструкция по установке программного обеспечения",
        "expected": "low",
        "description": "Техническая документация"
    },
]

# ========== ЗАПУСК ТЕСТОВ ==========
results = {"high": 0, "medium": 0, "low": 0}
correct = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n[ТЕСТ {i}] {test['description']}")
    print(f"📝 Текст: \"{test['text']}\"")
    print(f"🎯 Ожидаем: {test['expected'].upper()}")
    print("-" * 60)
    
    result = predict(test['text'], show_details=True)
    
    # Классификация по порогам
    score = result['viral_score']
    if score >= 0.7:
        predicted = "high"
    elif score >= 0.4:
        predicted = "medium"
    else:
        predicted = "low"
    
    results[predicted] += 1
    
    # Проверка корректности
    if predicted == test['expected']:
        print("✅ ПРАВИЛЬНО")
        correct += 1
    else:
        print(f"❌ НЕПРАВИЛЬНО (получили: {predicted.upper()})")

# ========== ИТОГИ ==========
print("\n" + "=" * 60)
print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
print("=" * 60)
print(f"✅ Правильных ответов: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
print(f"\nРаспределение предсказаний:")
print(f"  • HIGH:   {results['high']}")
print(f"  • MEDIUM: {results['medium']}")
print(f"  • LOW:    {results['low']}")

# ========== КАСТОМНЫЕ ТЕКСТЫ ==========
print("\n" + "=" * 60)
print("🎨 ТЕСТ НА СВОИХ ТЕКСТАХ")
print("=" * 60)
print("Введите текст для анализа (или 'exit' для выхода):\n")

while True:
    try:
        user_text = input("📝 Ваш текст: ").strip()
        if user_text.lower() == 'exit':
            break
        if not user_text:
            continue
        
        print()
        start_time = time.time()
        result = predict(user_text, show_details=True)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Время обработки: {elapsed:.3f}с\n")
        
    except KeyboardInterrupt:
        break

print("\n👋 Тестирование завершено!")
