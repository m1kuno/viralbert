import torch
from pathlib import Path

models_dir = Path('.')
model_files = ['multitask_best.pth', 'multitask_quantized.pth', 'viral_classifier.pth']

print("=" * 70)
print("🔍 ПРОВЕРКА ВСЕХ МОДЕЛЕЙ")
print("=" * 70)

for model_file in model_files:
    model_path = models_dir / model_file

    if not model_path.exists():
        print(f"\n❌ {model_file} - НЕ НАЙДЕН")
        continue

    print(f"\n{'=' * 70}")
    print(f"📦 {model_file}")
    print("=" * 70)

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Размер файла
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"💾 Размер: {size_mb:.1f} MB")

        # Тип данных
        print(f"📊 Тип: {type(checkpoint).__name__}")

        # Если это словарь с метаданными
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("✅ Формат: Словарь с метаданными")
                print(f"   Ключи: {list(checkpoint.keys())}")

                if 'epoch' in checkpoint:
                    print(f"   🔢 Epoch: {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"   📉 Val Loss: {checkpoint['val_loss']:.4f}")
                if 'train_loss' in checkpoint:
                    print(f"   📉 Train Loss: {checkpoint['train_loss']:.4f}")

                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                print("✅ Формат: state_dict (только веса)")
        else:
            state_dict = checkpoint
            print("✅ Формат: OrderedDict (только веса)")

        # Анализ архитектуры
        print(f"\n🏗️  АРХИТЕКТУРА:")

        # Проверка BERT
        bert_keys = [k for k in state_dict.keys() if k.startswith('bert.')]
        if bert_keys:
            # Определяем hidden_size
            emb_weight = state_dict.get('bert.embeddings.word_embeddings.weight')
            if emb_weight is not None:
                hidden_size = emb_weight.shape[1]
                print(f"   • BERT hidden_size: {hidden_size}")

                if hidden_size == 768:
                    print(f"   • Модель: RuBERT-base ✅")
                elif hidden_size == 312:
                    print(f"   • Модель: RuBERT-tiny2 ⚠️")

            # Количество слоёв
            layer_nums = set()
            for key in bert_keys:
                if 'encoder.layer.' in key:
                    layer_num = int(key.split('encoder.layer.')[1].split('.')[0])
                    layer_nums.add(layer_num)

            if layer_nums:
                num_layers = max(layer_nums) + 1
                print(f"   • Количество слоёв BERT: {num_layers}")

        # Проверка классификаторов
        print(f"\n🎯 КЛАССИФИКАТОРЫ:")

        # popularity_head (MultiTask)
        pop_weight = state_dict.get('popularity_head.0.weight')
        if pop_weight is not None:
            print(f"   ✅ popularity_head:")
            print(f"      • Shape: {pop_weight.shape}")
            print(f"      • Mean: {pop_weight.mean():.6f}")
            print(f"      • Std: {pop_weight.std():.6f}")

        # emotion_head (MultiTask)
        emo_weight = state_dict.get('emotion_head.0.weight')
        if emo_weight is not None:
            print(f"   ✅ emotion_head:")
            print(f"      • Shape: {emo_weight.shape}")
            print(f"      • Mean: {emo_weight.mean():.6f}")
            print(f"      • Std: {emo_weight.std():.6f}")

        # classifier (Single task v1)
        class_weight = state_dict.get('classifier.0.weight')
        if class_weight is not None:
            print(f"   ✅ classifier (v1):")
            print(f"      • Shape: {class_weight.shape}")
            print(f"      • Mean: {class_weight.mean():.6f}")
            print(f"      • Std: {class_weight.std():.6f}")

        # Проверка на квантизацию
        quantized_keys = [k for k in state_dict.keys() if 'packed_params' in k or 'scale' in k]
        if quantized_keys:
            print(f"\n⚡ КВАНТИЗАЦИЯ: Обнаружена ({len(quantized_keys)} квантизированных слоёв)")

        # Итоговая оценка
        print(f"\n🎯 ОЦЕНКА:")
        if pop_weight is not None and emo_weight is not None:
            if pop_weight.shape[1] == 768:
                print(f"   ✅ Подходит для ShortsBot (MultiTask RuBERT-base)")
            else:
                print(f"   ⚠️  Неправильная архитектура (hidden_size != 768)")
        elif class_weight is not None:
            print(f"   ❌ Старая версия v1 (Single Task) - НЕ ПОДХОДИТ")
        else:
            print(f"   ❓ Неизвестная архитектура")

    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")

print(f"\n{'=' * 70}")
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 70)