# pip install datasets pandas

from datasets import load_dataset
import pandas as pd
import os

def save_ru_go_emotions(output_dir="ru_go_emotions_dataset"):
    os.makedirs(output_dir, exist_ok=True)

    # Указываем конфиг 'simplified' (есть ещё 'raw' с 28 эмоциями)
    ds = load_dataset("seara/ru_go_emotions", "simplified")  # [web:74]

    for split in ds.keys():  # 'train', 'validation', 'test'
        df = ds[split].to_pandas()
        
        out_path = os.path.join(output_dir, f"ru_go_emotions_{split}.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"Saved {split}: {len(df)} examples to {out_path}")

if __name__ == "__main__":
    save_ru_go_emotions()
