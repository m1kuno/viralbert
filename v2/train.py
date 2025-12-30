import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast
from model import MultiTaskClassifier

# ========== CONFIG ==========
MODEL_PATH = "./models/rubert-base"
NUM_EMOTIONS = 28
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {device}")

# ========== DATASET ==========
class MultiTaskDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, num_emotions=28):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_emotions = num_emotions
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        task = row['task']
        
        # Токенизация
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Метки
        pop_label = torch.tensor(0.0)
        emo_labels = torch.zeros(self.num_emotions)
        
        if task == 'popularity':
            pop_label = torch.tensor(float(row['label']))
        elif task == 'emotion':
            # Парсим список эмоций
            labels_str = row['emotion_labels']
            if pd.notna(labels_str) and labels_str != 'None':
                try:
                    labels = ast.literal_eval(labels_str)
                    for label_idx in labels:
                        if label_idx < self.num_emotions:
                            emo_labels[label_idx] = 1.0
                except:
                    pass
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task': task,
            'pop_label': pop_label,
            'emo_labels': emo_labels
        }

# ========== LOAD DATA ==========
print("📂 Загружаю датасет...")
df = pd.read_csv("v2/multitask_train.csv")
print(f"✅ {len(df)} примеров")
print(f"   Популярность: {len(df[df['task']=='popularity'])}")
print(f"   Эмоции: {len(df[df['task']=='emotion'])}")

# Split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LEN, NUM_EMOTIONS)
val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LEN, NUM_EMOTIONS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== MODEL ==========
print("\n🤖 Создаю модель...")
model = MultiTaskClassifier(MODEL_PATH, NUM_EMOTIONS)
model.freeze_bert()  # Замораживаем BERT на первые эпохи
model = model.to(device)

total_params, trainable_params = model.count_parameters()
print(f"📊 Параметров: {total_params:,} (обучаемых: {trainable_params:,})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss functions
criterion_pop = nn.BCEWithLogitsLoss()
criterion_emo = nn.BCEWithLogitsLoss()

# ========== TRAINING ==========
print(f"\n🏋️ Начинаю обучение на {EPOCHS} эпох...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # === TRAIN ===
    model.train()
    total_loss = 0
    pop_loss_sum = 0
    emo_loss_sum = 0
    pop_count = 0
    emo_count = 0
    
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tasks = batch['task']
        
        optimizer.zero_grad()
        
        # Разделяем батч по задачам
        pop_mask = torch.tensor([t == 'popularity' for t in tasks])
        emo_mask = torch.tensor([t == 'emotion' for t in tasks])
        
        loss = torch.tensor(0.0).to(device)
        batch_loss_pop = 0
        batch_loss_emo = 0
        
        # Популярность
        if pop_mask.any():
            pop_indices = pop_mask.to(device)
            pop_out, _ = model(
                input_ids[pop_indices], 
                attention_mask[pop_indices], 
                task='popularity'
            )
            pop_labels = batch['pop_label'][pop_indices].to(device)
            loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
            loss = loss + loss_pop
            batch_loss_pop = loss_pop.item()
            pop_loss_sum += batch_loss_pop
            pop_count += 1
        
        # Эмоции
        if emo_mask.any():
            emo_indices = emo_mask.to(device)
            _, emo_out = model(
                input_ids[emo_indices], 
                attention_mask[emo_indices], 
                task='emotion'
            )
            emo_labels = batch['emo_labels'][emo_indices].to(device)
            loss_emo = criterion_emo(emo_out, emo_labels)
            loss = loss + loss_emo
            batch_loss_emo = loss_emo.item()
            emo_loss_sum += batch_loss_emo
            emo_count += 1
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pop': f'{batch_loss_pop:.4f}',
            'emo': f'{batch_loss_emo:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pop = pop_loss_sum / max(pop_count, 1)
    avg_emo = emo_loss_sum / max(emo_count, 1)
    
    print(f"\nEpoch {epoch+1} Train - Loss: {avg_loss:.4f} (Pop: {avg_pop:.4f}, Emo: {avg_emo:.4f})")
    
    # === VALIDATION ===
    model.eval()
    val_loss = 0
    val_pop_loss = 0
    val_emo_loss = 0
    val_pop_count = 0
    val_emo_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            
            pop_mask = torch.tensor([t == 'popularity' for t in tasks])
            emo_mask = torch.tensor([t == 'emotion' for t in tasks])
            
            if pop_mask.any():
                pop_indices = pop_mask.to(device)
                pop_out, _ = model(input_ids[pop_indices], attention_mask[pop_indices], task='popularity')
                pop_labels = batch['pop_label'][pop_indices].to(device)
                loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
                val_loss += loss_pop.item()
                val_pop_loss += loss_pop.item()
                val_pop_count += 1
            
            if emo_mask.any():
                emo_indices = emo_mask.to(device)
                _, emo_out = model(input_ids[emo_indices], attention_mask[emo_indices], task='emotion')
                emo_labels = batch['emo_labels'][emo_indices].to(device)
                loss_emo = criterion_emo(emo_out, emo_labels)
                val_loss += loss_emo.item()
                val_emo_loss += loss_emo.item()
                val_emo_count += 1
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_pop = val_pop_loss / max(val_pop_count, 1)
    avg_val_emo = val_emo_loss / max(val_emo_count, 1)
    
    print(f"Epoch {epoch+1} Val   - Loss: {avg_val_loss:.4f} (Pop: {avg_val_pop:.4f}, Emo: {avg_val_emo:.4f})")
    
    # Сохраняем лучшую модель
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), './models/multitask_best.pth')
        print(f"💾 Сохранена лучшая модель (val_loss: {best_val_loss:.4f})")
    
    # После 2 эпох размораживаем верхние слои BERT
    if epoch == 1:
        print("\n🔓 Размораживаю верхние 2 слоя BERT...")
        model.unfreeze_bert(num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE/10)

# ========== SAVE FINAL ==========
print("\n💾 Сохраняю финальную модель...")
torch.save(model.state_dict(), './models/multitask_final.pth')
print("✅ Модель сохранена в ./models/multitask_final.pth")
print(f"✅ Лучшая модель: ./models/multitask_best.pth (val_loss: {best_val_loss:.4f})")
