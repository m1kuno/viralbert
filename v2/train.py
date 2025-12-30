import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast
import numpy as np
from model import MultiTaskClassifier

# ========== CONFIG ==========
MODEL_PATH = "./models/rubert-base"
NUM_EMOTIONS = 28
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 128
WARMUP_STEPS = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {device}")

# ========== GRADIENT SURGERY ==========
def gradient_surgery(losses, shared_params):
    """PCGrad: проецируем конфликтующие градиенты"""
    grads = []
    for loss in losses:
        loss.backward(retain_graph=True)
        grad = []
        for p in shared_params:
            if p.grad is not None:
                grad.append(p.grad.clone().flatten())
        if grad:
            grads.append(torch.cat(grad))
        for p in shared_params:
            if p.grad is not None:
                p.grad.zero_()
    
    if len(grads) < 2:
        return
    
    g1, g2 = grads[0], grads[1]
    dot_product = torch.dot(g1, g2)
    
    if dot_product < 0:
        g2 = g2 - (dot_product / (torch.norm(g1) ** 2 + 1e-8)) * g1
    
    avg_grad = (g1 + g2) / 2
    
    idx = 0
    for p in shared_params:
        if p.grad is not None:
            numel = p.grad.numel()
            p.grad.copy_(avg_grad[idx:idx+numel].view_as(p.grad))
            idx += numel

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
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pop_label = torch.tensor(0.0)
        emo_labels = torch.zeros(self.num_emotions)
        
        if task == 'popularity':
            pop_label = torch.tensor(float(row['label']))
        elif task == 'emotion':
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

# ========== WEIGHTED LOSS ==========
def calculate_emotion_weights(df, num_emotions=28):
    counts = np.zeros(num_emotions)
    total = 0
    
    for labels_str in df[df['task']=='emotion']['emotion_labels']:
        if pd.notna(labels_str) and labels_str != 'None':
            try:
                labels = ast.literal_eval(labels_str)
                for l in labels:
                    if l < num_emotions:
                        counts[l] += 1
                total += 1
            except:
                pass
    
    counts = np.where(counts == 0, 1, counts)
    weights = total / (num_emotions * counts)
    weights = np.clip(weights, 0.1, 10.0)
    
    return torch.FloatTensor(weights)

# ========== LOAD DATA ==========
print("📂 Загружаю датасет...")
df = pd.read_csv("v2/multitask_train.csv")
print(f"✅ {len(df)} примеров")
print(f"   Популярность: {len(df[df['task']=='popularity'])}")
print(f"   Эмоции: {len(df[df['task']=='emotion'])}")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['task'])
print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
train_dataset = MultiTaskDataset(train_df, tokenizer, MAX_LEN, NUM_EMOTIONS)
val_dataset = MultiTaskDataset(val_df, tokenizer, MAX_LEN, NUM_EMOTIONS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

# ========== MODEL ==========
print("\n🤖 Создаю модель...")
model = MultiTaskClassifier(MODEL_PATH, NUM_EMOTIONS)
model.freeze_bert()
model = model.to(device)

total_params, trainable_params = model.count_parameters()
print(f"📊 Параметров: {total_params:,} (обучаемых: {trainable_params:,})")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

num_training_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps
)

criterion_pop = nn.BCEWithLogitsLoss()

print("\n⚖️ Вычисляю веса классов эмоций...")
emotion_weights = calculate_emotion_weights(train_df, NUM_EMOTIONS).to(device)
print(f"   Min weight: {emotion_weights.min():.2f}, Max: {emotion_weights.max():.2f}")
criterion_emo = nn.BCEWithLogitsLoss(pos_weight=emotion_weights)

# ========== TRAINING ==========
print(f"\n🏋️ Начинаю обучение на {EPOCHS} эпох...\n")
best_val_loss = float('inf')
use_gradient_surgery = True

for epoch in range(EPOCHS):
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
        
        pop_mask = torch.tensor([t == 'popularity' for t in tasks])
        emo_mask = torch.tensor([t == 'emotion' for t in tasks])
        
        losses_for_surgery = []
        batch_loss_pop = 0
        batch_loss_emo = 0
        
        if pop_mask.any():
            pop_out, _ = model(
                input_ids[pop_mask],
                attention_mask[pop_mask],
                task='popularity'
            )
            pop_labels = batch['pop_label'][pop_mask].to(device)
            loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
            losses_for_surgery.append(loss_pop)
            batch_loss_pop = loss_pop.item()
            pop_loss_sum += batch_loss_pop
            pop_count += 1
        
        if emo_mask.any():
            _, emo_out = model(
                input_ids[emo_mask],
                attention_mask[emo_mask],
                task='emotion'
            )
            emo_labels = batch['emo_labels'][emo_mask].to(device)
            loss_emo = criterion_emo(emo_out, emo_labels)
            losses_for_surgery.append(loss_emo)
            batch_loss_emo = loss_emo.item()
            emo_loss_sum += batch_loss_emo
            emo_count += 1
        
        if len(losses_for_surgery) == 2 and use_gradient_surgery:
            shared_params = list(model.bert.parameters())
            gradient_surgery(losses_for_surgery, shared_params)
            for loss in losses_for_surgery:
                loss.backward()
        else:
            if losses_for_surgery:
                total_batch_loss = sum(losses_for_surgery)
                total_batch_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += batch_loss_pop + batch_loss_emo
        
        progress.set_postfix({
            'loss': f'{(batch_loss_pop + batch_loss_emo):.4f}',
            'pop': f'{batch_loss_pop:.3f}',
            'emo': f'{batch_loss_emo:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pop = pop_loss_sum / max(pop_count, 1)
    avg_emo = emo_loss_sum / max(emo_count, 1)
    
    print(f"\nEpoch {epoch+1} Train - Loss: {avg_loss:.4f} (Pop: {avg_pop:.4f}, Emo: {avg_emo:.4f})")
    
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
                pop_out, _ = model(input_ids[pop_mask], attention_mask[pop_mask], task='popularity')
                pop_labels = batch['pop_label'][pop_mask].to(device)
                loss_pop = criterion_pop(pop_out.squeeze(-1), pop_labels)
                val_loss += loss_pop.item()
                val_pop_loss += loss_pop.item()
                val_pop_count += 1
            
            if emo_mask.any():
                _, emo_out = model(input_ids[emo_mask], attention_mask[emo_mask], task='emotion')
                emo_labels = batch['emo_labels'][emo_mask].to(device)
                loss_emo = criterion_emo(emo_out, emo_labels)
                val_loss += loss_emo.item()
                val_emo_loss += loss_emo.item()
                val_emo_count += 1
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_pop = val_pop_loss / max(val_pop_count, 1)
    avg_val_emo = val_emo_loss / max(val_emo_count, 1)
    
    print(f"Epoch {epoch+1} Val   - Loss: {avg_val_loss:.4f} (Pop: {avg_val_pop:.4f}, Emo: {avg_val_emo:.4f})")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, './models/multitask_best.pth')
        print(f"💾 Сохранена лучшая модель (val_loss: {best_val_loss:.4f})")
    
    if epoch == 1:
        print("\n🔓 Размораживаю верхние 2 слоя BERT...")
        model.unfreeze_bert(num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE/10, weight_decay=0.01)
        remaining_steps = len(train_loader) * (EPOCHS - epoch - 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100,
            num_training_steps=remaining_steps
        )

print("\n💾 Сохраняю финальную модель...")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'num_emotions': NUM_EMOTIONS,
        'model_path': MODEL_PATH,
    }
}, './models/multitask_final.pth')

print("✅ Модель сохранена в ./models/multitask_final.pth")
print(f"✅ Лучшая модель: ./models/multitask_best.pth (val_loss: {best_val_loss:.4f})")
