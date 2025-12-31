# continue_training.py - Дообучение ViralBERT с 3 до 10 эпох
# Оптимизировано для датасета 54K примеров

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import pandas as pd
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ========== АРХИТЕКТУРА МОДЕЛИ ==========
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_path, num_emotions=28, hidden_size=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Popularity head (регрессия 0-1)
        self.popularity_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Emotion head (multi-label 28 классов)
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
        
        # Инициализация меток
        pop_label = torch.tensor(0.0)
        emo_labels = torch.zeros(self.num_emotions)
        
        # Popularity task
        if task == 'popularity':
            pop_label = torch.tensor(float(row['label']))
        
        # Emotion task
        elif task == 'emotion':
            labels_str = row.get('emotion_labels', '[]')
            if pd.notna(labels_str) and labels_str != '[]':
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

# ========== КОНФИГУРАЦИЯ ==========
print("=" * 70)
print("🚀 ДООБУЧЕНИЕ VIRALBERT v2")
print("=" * 70)

# Пути
MODEL_PATH = './models/rubert-base'
CHECKPOINT_PATH = './models/multitask_best.pth'
DATA_PATH = './v2/multitask_train.csv'

# Гиперпараметры
START_EPOCH = 4              # Продолжаем с 4-й эпохи
NUM_EPOCHS = 10              # До 10-й эпохи
PATIENCE = 4                 # Early stopping
BATCH_SIZE = 16              # Оптимально для 54K
LEARNING_RATE = 1e-5         # Понижен для fine-tuning
MAX_LEN = 128
NUM_EMOTIONS = 28

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ========== ЗАГРУЗКА МОДЕЛИ ==========
print("📥 Загружаю модель с Epoch 3...")
model = MultiTaskClassifier(MODEL_PATH, NUM_EMOTIONS)
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

print(f"✅ Модель загружена:")
print(f"   • Epoch: {checkpoint['epoch']}")
print(f"   • Val Loss: {checkpoint['val_loss']:.4f}")
print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("📊 Загружаю датасет...")
df = pd.read_csv(DATA_PATH)

print(f"✅ Датасет загружен:")
print(f"   • Всего примеров: {len(df):,}")
print(f"   • Popularity: {len(df[df['task']=='popularity']):,}")
print(f"   • Emotion: {len(df[df['task']=='emotion']):,}")
print()

# Split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['task'])
print(f"📂 Split:")
print(f"   • Train: {len(train_df):,} примеров")
print(f"   • Val: {len(val_df):,} примеров")
print()

# Datasets & Loaders
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LEN, NUM_EMOTIONS)
val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LEN, NUM_EMOTIONS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

print(f"🔄 DataLoaders готовы:")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Train batches: {len(train_loader)}")
print(f"   • Val batches: {len(val_loader)}")
print()

# ========== ОПТИМИЗАТОР И LOSS ==========
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Восстанавливаем состояние оптимизатора
if 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✅ Состояние оптимизатора восстановлено")
else:
    print("⚠️  Оптимизатор инициализирован заново")
print()

criterion_pop = nn.BCEWithLogitsLoss()
criterion_emo = nn.BCEWithLogitsLoss()

# ========== ОБУЧЕНИЕ ==========
print("=" * 70)
print(f"🚀 НАЧИНАЮ ОБУЧЕНИЕ: Epoch {START_EPOCH} → {NUM_EPOCHS}")
print(f"⏱️  Примерное время: ~{(NUM_EPOCHS - START_EPOCH + 1) * 4} минут на GPU")
print("=" * 70)
print()

best_val_loss = checkpoint['val_loss']
no_improve_epochs = 0
best_epoch = checkpoint['epoch']

for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
    print(f"{'='*70}")
    print(f"📈 EPOCH {epoch}/{NUM_EPOCHS}")
    print(f"{'='*70}")
    
    # ========== TRAINING ==========
    model.train()
    total_loss = 0
    pop_loss_sum = 0
    emo_loss_sum = 0
    pop_count = 0
    emo_count = 0
    
    progress = tqdm(train_loader, desc=f"Training", ncols=100)
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
        
        # Popularity task
        if pop_mask.any():
            pop_indices = pop_mask.to(device)
            pop_out, _ = model(input_ids[pop_indices], attention_mask[pop_indices], task='popularity')
            pop_labels = batch['pop_label'][pop_indices].to(device)
            loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
            loss = loss + loss_pop
            batch_loss_pop = loss_pop.item()
            pop_loss_sum += batch_loss_pop
            pop_count += 1
        
        # Emotion task
        if emo_mask.any():
            emo_indices = emo_mask.to(device)
            _, emo_out = model(input_ids[emo_indices], attention_mask[emo_indices], task='emotion')
            emo_labels = batch['emo_labels'][emo_indices].to(device)
            loss_emo = criterion_emo(emo_out, emo_labels)
            loss = loss + loss_emo
            batch_loss_emo = loss_emo.item()
            emo_loss_sum += batch_loss_emo
            emo_count += 1
        
        # Backward pass
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({
            'loss': f"{loss.item():.4f}",
            'pop': f"{batch_loss_pop:.4f}",
            'emo': f"{batch_loss_emo:.4f}"
        })
    
    # Train metrics
    avg_loss = total_loss / len(train_loader)
    avg_pop = pop_loss_sum / max(pop_count, 1)
    avg_emo = emo_loss_sum / max(emo_count, 1)
    
    print(f"\n📊 Train - Loss: {avg_loss:.4f} | Pop: {avg_pop:.4f} | Emo: {avg_emo:.4f}")
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    val_pop_loss = 0
    val_emo_loss = 0
    val_pop_count = 0
    val_emo_count = 0
    
    with torch.no_grad():
        progress_val = tqdm(val_loader, desc=f"Validation", ncols=100)
        for batch in progress_val:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            
            pop_mask = torch.tensor([t == 'popularity' for t in tasks])
            emo_mask = torch.tensor([t == 'emotion' for t in tasks])
            
            # Popularity task
            if pop_mask.any():
                pop_indices = pop_mask.to(device)
                pop_out, _ = model(input_ids[pop_indices], attention_mask[pop_indices], task='popularity')
                pop_labels = batch['pop_label'][pop_indices].to(device)
                loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
                val_loss += loss_pop.item()
                val_pop_loss += loss_pop.item()
                val_pop_count += 1
            
            # Emotion task
            if emo_mask.any():
                emo_indices = emo_mask.to(device)
                _, emo_out = model(input_ids[emo_indices], attention_mask[emo_indices], task='emotion')
                emo_labels = batch['emo_labels'][emo_indices].to(device)
                loss_emo = criterion_emo(emo_out, emo_labels)
                val_loss += loss_emo.item()
                val_emo_loss += loss_emo.item()
                val_emo_count += 1
            
            progress_val.set_postfix({'loss': f"{val_loss / max(val_pop_count + val_emo_count, 1):.4f}"})
    
    # Validation metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_pop = val_pop_loss / max(val_pop_count, 1)
    avg_val_emo = val_emo_loss / max(val_emo_count, 1)
    
    print(f"📊 Val   - Loss: {avg_val_loss:.4f} | Pop: {avg_val_pop:.4f} | Emo: {avg_val_emo:.4f}")
    
    # ========== EARLY STOPPING ==========
    if avg_val_loss < best_val_loss:
        improvement = best_val_loss - avg_val_loss
        best_val_loss = avg_val_loss
        best_epoch = epoch
        no_improve_epochs = 0
        
        # Сохранение лучшей модели
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'train_loss': avg_loss
        }, CHECKPOINT_PATH)
        
        print(f"💾 Сохранена лучшая модель! Улучшение: {improvement:.4f}")
    else:
        no_improve_epochs += 1
        print(f"⚠️  Val Loss не улучшается: {no_improve_epochs}/{PATIENCE} эпох")
    
    print(f"🏆 Best: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    
    # Early stopping
    if no_improve_epochs >= PATIENCE:
        print(f"\n{'='*70}")
        print(f"🛑 EARLY STOPPING на эпохе {epoch}!")
        print(f"🏆 Лучшая модель: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        print(f"{'='*70}")
        break
    
    print()

# ========== ЗАВЕРШЕНИЕ ==========
print("\n" + "=" * 70)
print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 70)
print(f"✅ Лучшая модель:")
print(f"   • Epoch: {best_epoch}")
print(f"   • Val Loss: {best_val_loss:.4f}")
print(f"   • Улучшение с 3 эпохи: {0.2828 - best_val_loss:.4f}")
print(f"\n📂 Модель сохранена: {CHECKPOINT_PATH}")
print(f"\n📋 Следующие шаги:")
print(f"   1. Скопируй модель в ShortsBot:")
print(f"      copy models\\multitask_best.pth \"C:\\Users\\konst\\OneDrive\\Рабочий стол\\ShortsBot-main\\ShortsBot-main\\models\\multitask_best.pth\"")
print(f"   2. Запусти тесты:")
print(f"      python test_viralbert.py")
print("=" * 70)
