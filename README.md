# ViralBERT 🚀

Multi-task BERT модель для детекции вирусного контента и распознавания эмоций.

## Возможности

- ✅ Предсказание вирусности текста (популярность)
- ✅ Распознавание 28 эмоций (multi-label)
- ✅ Gradient Surgery для multi-task learning
- ✅ Weighted Loss для несбалансированных классов
- ✅ Cosine LR Scheduler с warmup

## Архитектура

- **Base Model**: RuBERT-base (ai-forever/ruBert-base)
- **Parameters**: 178M (trainable: ~2.5M initially)
- **Tasks**: Binary classification (popularity) + Multi-label (emotions)

## Обучение

\\\ash
# Скачай RuBERT
python v2/download_rubert.py

# Подготовь датасет
python v2/prepare_dataset.py

# Обучение
python v2/train.py
\\\

## Датасеты

- **Популярность**: Custom dataset (~5.6K примеров)
- **Эмоции**: ru_go_emotions (~48K примеров, 28 классов)
- **Total**: ~54K примеров

## Результаты

| Метрика | Train | Validation |
|---------|-------|------------|
| Total Loss | 0.65 | 0.36 |
| Popularity | 0.51 | 0.26 |
| Emotions | 0.22 | 0.15 |

*После 1 эпохи*

## Требования

\\\
torch>=2.0.0
transformers>=4.30.0
pandas
numpy
scikit-learn
tqdm
\\\

## Автор

[@m1kuno](https://github.com/m1kuno)
