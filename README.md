whisper-finetune/
├── configs/
│   └── config.yaml          # File cấu hình (Hyperparameters, đường dẫn)
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Xử lý dataset, audio preprocessing, data collator
│   └── metrics.py           # Tính toán WER (Word Error Rate)
├── train.py                 # File chính để chạy training
├── requirements.txt         # Các thư viện cần thiết
└── README.md

pip install transformers datasets evaluate jiwer pyyaml soundfile
pip install torch==2.8.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

chú ý comment cho kaggle ở train.py và data_loader.py

set HF_TOKEN for higher download rate