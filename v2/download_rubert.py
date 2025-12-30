from transformers import AutoTokenizer, AutoModel
import torch

print("📥 Скачиваю rubert-base-cased...")

model_name = "ai-forever/ruBert-base"  # или "cointegrated/rubert-base-cased"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    
    print(f"✅ Модель скачана!")
    print(f"📊 Размер эмбеддингов: {base_model.config.hidden_size}")
    print(f"📊 Параметров: ~180M")
    
    save_dir = "./models/rubert-base"
    tokenizer.save_pretrained(save_dir)
    base_model.save_pretrained(save_dir)
    
    print(f"💾 Сохранено в {save_dir}")
    
    # Тест
    test_text = "Это невероятная история!"
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = base_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    
    print(f"✅ Вектор размерности: {embeddings.shape[1]}")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
