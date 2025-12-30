from transformers import AutoTokenizer, AutoModel
import torch

print("📥 Скачиваю rubert-base с safetensors...")

model_name = "ai-forever/ruBert-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(
        model_name,
        use_safetensors=True  # Используем безопасный формат
    )
    
    print(f"✅ Модель скачана!")
    print(f"📊 Hidden size: {base_model.config.hidden_size}")
    
    save_dir = "./models/rubert-base"
    tokenizer.save_pretrained(save_dir)
    base_model.save_pretrained(save_dir, safe_serialization=True)
    
    print(f"💾 Сохранено в {save_dir}")
    
    # Проверка
    test = "Это тест"
    inputs = tokenizer(test, return_tensors='pt')
    with torch.no_grad():
        outputs = base_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
    
    print(f"✅ Вектор: {emb.shape[1]} dims")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
