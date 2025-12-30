# ViralBERT v2 — multi-task модель для детекции вирусного контента

Multi-task RuBERT-модель для предсказания **вирусности текста** и распознавания **28 эмоций** (multi-label) на русском языке.[1]

***

## Результаты обучения

### Конфигурация

- Модель: `ai-forever/ruBert-base` (≈178M параметров)[1]
- Задачи: вирусность (binary) + эмоции (28 классов, multi-label)[1]
- Датасет: ~54K примеров (5.6K вирусность + 48K эмоции)[1]
- Optimizer: AdamW (`lr=2e-5`, `weight_decay=0.01`)[1]
- Scheduler: Cosine + warmup (500 шагов)[1]
- Техники: Gradient Surgery (PCGrad), Weighted Loss, BERT freezing[1]

### Динамика обучения по эпохам

| Эпоха | Train Loss | Val Loss | Val Pop | Val Emo | Статус |
|------:|-----------:|---------:|--------:|--------:|--------|
| 1 | 0.64  | 0.36  |  0.26 | 0.1482 | BERT заморожен |
| 2 | ~0.48 | ~0.32 | ~0.23 | ~0.14 | Head'ы обучены |
| 3 | ~0.40 | ~0.30 | ~0.21 | ~0.13 | Разморозка BERT (2 слоя) |
| 4 | 0.32  | 0.28  |  0.18 | 0.1342 | Fine-tuning |
| 5 | ~0.29 | ~0.27 | ~0.17 | ~0.12 | Финальная эпоха |

Итоговая модель: `./models/multitask_best.pth` (val_loss ≈ 0.2828, ~794 MB).[1]

***

## Качество модели

- Val Loss: 0.28 — сильный результат для multi-task (в README указан топ-диапазон 0.25–0.35).[1]
- Популярность: Val loss 0.18 ≈ ~85–90% accuracy (оценка из README).[1]
- Эмоции: Val loss 0.13 ≈ F1 ~0.50–0.55 (оценка из README).[1]
- Нет переобучения: Train/Val разница ~0.04 (по README).[1]

***

## Архитектура

    RuBERT-base (768d)
      ↓ [CLS] token
      ├─ Popularity Head: Linear(768→256→128→1) + Sigmoid
      └─ Emotion Head:    Linear(768→256→28)   + Sigmoid
[1]

Popularity Head (вирусность):

    nn.Linear(768, 256) → ReLU → Dropout(0.3)
    → nn.Linear(256, 128) → ReLU → Dropout(0.2)
    → nn.Linear(128, 1)  # BCEWithLogitsLoss
[1]

Emotion Head (28 эмоций):

    nn.Linear(768, 256) → ReLU → Dropout(0.3)
    → nn.Linear(256, 28)  # BCEWithLogitsLoss с pos_weight
[1]

***

## Датасеты

### 1) Популярность (кастомный)

- Размер: 5,676 примеров[1]
- Формат: CSV с колонками `text`, `label` (0/1)[1]
- Источник: собственная разметка коротких текстов/заголовков/фраз из вирусных видео[1]

### 2) Эмоции (ru_go_emotions)

- Размер: 48,836 примеров (train + val)[1]
- Классы: 28 эмоций (admiration, amusement, anger, joy, surprise, neutral, ...)[1]
- Формат: multi-label[1]
- Источник: русскоязычный аналог GoEmotions[1]

### 3) Объединённый датасет

    python v2/prepare_dataset.py

Создаёт: `v2/multitask_train.csv` (~54,512 примеров, перемешанные задачи).[1]

***

## Быстрый старт

### 1) Установка зависимостей

    pip install torch transformers pandas numpy scikit-learn tqdm
[1]

### 2) Скачивание RuBERT

    python v2/download_rubert.py

Сохраняет в `./models/rubert-base/`.[1]

### 3) Подготовка датасета

    python v2/prepare_dataset.py

Требует:[1]
- `final_train_dataset.csv` (популярность)
- `ru_go_emotions_dataset/` (эмоции)

### 4) Обучение

    python v2/train.py

Обучение ~40–50 минут на Tesla T4 (оценка из README).[1]

Сохраняет:[1]
- `./models/multitask_best.pth` — лучшая модель по val_loss
- `./models/multitask_final.pth` — финальная эпоха

***

## Использование модели

### Инференс

    from v2.inference import ViralBERTInference
    predictor = ViralBERTInference(
        model_path="./models/multitask_best.pth",
        rubert_path="./models/rubert-base",
    )
    text = "Это просто невероятно! Лучший день в моей жизни! 😍"
    result = predictor.predict(text)
    print(f"Вирусность: {result['viral_score']:.2%}")
    print(f"Вирусный: {result['is_viral']}")
    print(f"Эмоции: {result['emotions']}")
[1]

Пример вывода (пример из README):[1]

    Вирусность: 87.3%
    Вирусный: True
    Эмоции: ['excitement', 'joy', 'admiration']
[1]

### Поиск вирусных моментов в видео

    from v2.viral_detector_v2 import ViralDetectorV2
    detector = ViralDetectorV2()

    # subtitles = [{"text": "...", "start": 12.5}, ...]
    moments = await detector.find_viral_moments_from_subtitles(subtitles)

    for m in moments:
        print(f"{m['start']} → {m['end']} | Score: {m['score']:.2%}")
        print(f"  Эмоции: {m['emotions']}")
[1]

***

## Структура проекта

    viralbert/
    ├── v2/
    │   ├── model.py              # MultiTaskClassifier
    │   ├── train.py              # Обучение с Gradient Surgery
    │   ├── prepare_dataset.py    # Подготовка датасета
    │   ├── download_rubert.py    # Загрузка RuBERT
    │   ├── inference.py          # Инференс класс
    │   └── viral_detector_v2.py  # Детектор вирусных моментов
    ├── models/
    │   ├── rubert-base/          # RuBERT токенизатор и веса
    │   ├── multitask_best.pth    # Лучшая модель (794 MB)
    │   └── multitask_final.pth   # Финальная модель (682 MB)
    ├── .gitignore
    └── README.md
[1]

***

## Технические детали

### Gradient Surgery (PCGrad)

При обучении multi-task модели градиенты от разных задач могут конфликтовать; PCGrad уменьшает конфликт и стабилизирует обучение.[1]

### Weighted Loss для эмоций

В README указано, что некоторые эмоции встречаются значительно реже neutral, поэтому используется `pos_weight` для балансировки.[1]

    weights = total_samples / (num_classes * class_counts)
    weights = clip(weights, 0.1, 10.0)
    criterion = BCEWithLogitsLoss(pos_weight=weights)
[1]

### Прогрессивная разморозка BERT

- Эпохи 1–2: BERT заморожен, обучаются только head'ы[1]
- Эпохи 3–5: размораживаются верхние 2 слоя, LR уменьшается в 10 раз[1]

***

## Требования

    python >= 3.10
    torch >= 2.0.0
    transformers >= 4.30.0
    pandas
    numpy
    scikit-learn
    tqdm
[1]

***

## Лицензия

MIT License — свободное использование с указанием авторства.[1]
