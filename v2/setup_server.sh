#!/bin/bash
set -e

echo "Настройка сервера для обучения..."

apt update && apt upgrade -y
apt install python3.11 python3-pip git wget -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt

nvidia-smi

echo "Сервер готов!"
echo "Следующие шаги:"
echo "  python3 v2/download_rubert.py"
echo "  python3 v2/download_emo.py"
echo "  Загрузи final_train_dataset.csv через WinSCP"
echo "  python3 v2/prepare_dataset.py"
echo "  python3 v2/train.py"
