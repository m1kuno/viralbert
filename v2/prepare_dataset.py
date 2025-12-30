import pandas as pd
import os

print("📦 Объединяю датасеты...")

# Определяем корневую директорию проекта
if os.path.basename(os.getcwd()) == 'v2':
    base_path = '..'
else:
    base_path = '.'

# Функция для парсинга labels
def parse_labels(labels_str):
    if pd.isna(labels_str):
        return []
    
    labels_str = str(labels_str).strip()
    labels_str = labels_str.replace('[', '').replace(']', '').strip()
    
    if not labels_str:
        return []
    
    try:
        labels = [int(x) for x in labels_str.replace(',', ' ').split() if x.strip()]
        return labels
    except:
        return []

# 1. Загружаем популярность (твой датасет)
print("\n📂 Загружаю датасет популярности...")
pop_df = pd.read_csv(f"{base_path}/final_train_dataset.csv")

# Создаём чистый датафрейм для популярности
pop_clean = pd.DataFrame({
    'text': pop_df['text'],
    'label': pop_df['label'],
    'task': 'popularity',
    'emotion_labels': None
})
print(f"✅ Популярность: {len(pop_clean)} примеров")

# 2. Загружаем эмоции (ru_go_emotions)
print("\n📂 Загружаю датасет эмоций...")
emo_train = pd.read_csv(f"{base_path}/ru_go_emotions_dataset/ru_go_emotions_train.csv")
emo_val = pd.read_csv(f"{base_path}/ru_go_emotions_dataset/ru_go_emotions_validation.csv")
emo_test = pd.read_csv(f"{base_path}/ru_go_emotions_dataset/ru_go_emotions_test.csv")

# Объединяем train+val
emo_combined = pd.concat([emo_train, emo_val], ignore_index=True)

# Находим колонку с текстом
text_col = None
for col in ['ru_text', 'text', 'comment_text']:
    if col in emo_combined.columns:
        text_col = col
        break

if text_col is None:
    print("❌ Не найдена колонка с текстом!")
    print("Доступные колонки:", emo_combined.columns.tolist())
    exit(1)

# Узнаём количество классов эмоций
all_labels = []
for labels_str in emo_combined['labels']:
    labels = parse_labels(labels_str)
    all_labels.extend(labels)

num_emotions = max(all_labels) + 1 if all_labels else 28
print(f"📊 Найдено эмоций: {num_emotions}")

# Создаём чистый датафрейм для эмоций
emo_clean = pd.DataFrame({
    'text': emo_combined[text_col],
    'label': None,
    'task': 'emotion',
    'emotion_labels': emo_combined['labels'].apply(lambda x: str(parse_labels(x)))
})
print(f"✅ Эмоции train: {len(emo_clean)} примеров")

# 3. Объединяем
print("\n🔗 Объединяю датасеты...")
combined = pd.concat([pop_clean, emo_clean], ignore_index=True)

# Перемешиваем
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем
os.makedirs('v2', exist_ok=True)
combined.to_csv("v2/multitask_train.csv", index=False)
print(f"✅ Сохранено {len(combined)} примеров в v2/multitask_train.csv")

# 4. Подготавливаем test
print("\n📂 Подготавливаю test датасет...")
emo_test_clean = pd.DataFrame({
    'text': emo_test[text_col],
    'label': None,
    'task': 'emotion',
    'emotion_labels': emo_test['labels'].apply(lambda x: str(parse_labels(x)))
})
emo_test_clean.to_csv("v2/multitask_test.csv", index=False)

print(f"✅ Test: {len(emo_test_clean)} примеров")
print(f"\n📊 Итоговая статистика:")
print(f"   Популярность: {len(pop_clean)} примеров")
print(f"   Эмоции train: {len(emo_clean)} примеров")
print(f"   Эмоции test: {len(emo_test_clean)} примеров")
print(f"   Всего train: {len(combined)} примеров")
print(f"   Число классов эмоций: {num_emotions}")

# Сохраняем конфиг
with open("v2/config.txt", "w") as f:
    f.write(f"NUM_EMOTIONS={num_emotions}\n")
    f.write(f"TRAIN_SIZE={len(combined)}\n")
    f.write(f"TEST_SIZE={len(emo_test_clean)}\n")

print(f"\n✅ Конфиг сохранён в v2/config.txt")
print(f"\n🎯 Следующий шаг: python v2/train.py")
