"""
Kerangka Model Jalur CNN - NIM 123450014
Menjalankan pelatihan, evaluasi, dan penyimpanan hasil prediksi (.npy) menggunakan data simulasi.
"""
from __future__ import annotations
import sys
import pathlib

# Menambahkan direktori root proyek ke sistem path
root_dir = pathlib.Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import modul protokol tim
from src.protokol import data
from src.common import evaluate

class CNNModel(nn.Module):
    """Arsitektur CNN 2D untuk masukan Log-mel 64 pita."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Ukuran 64xT menjadi 32x(T/2)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Ukuran menjadi 16x(T/4)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def latih_dan_prediksi(seed: int, final_run: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Memuat data via protokol tim (bisa diganti data simulasi jika dataset belum ada)
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Pastikan bentuk masukan sesuai (Batch, Channel=1, Tinggi=64 pita, Lebar waktu)
    # Jika menggunakan data dummy/simulasi, sesuaikan tipenya ke float32
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Inisialisasi Model, Loss, dan Optimizer
    model = CNNModel(num_classes=len(np.unique(ytr)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training singkat (1 epoch untuk pengujian kerangka)
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 4. Evaluasi / Prediksi Validasi
    model.eval()
    with torch.no_grad():
        val_outputs = model(Xva_t)
        pred_val = torch.softmax(val_outputs, dim=1).numpy()

    # Prediksi Test (jika tersedia)
    pred_test = None
    if Xte is not None:
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        with torch.no_grad():
            test_outputs = model(Xte_t)
            pred_test = torch.softmax(test_outputs, dim=1).numpy()

    return pred_val, pred_test

def main():
    parser = argparse.ArgumentParser(description="CNN Model Runner - NIM 123450014")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--final", action="store_true", help="Run final evaluation")
    args = parser.parse_args()

    NIM = "123450014"
    print(f"[*] Menjalankan kerangka CNN untuk NIM {NIM} (Seed: {args.seed})...")
    
    pred_val, pred_test = latih_dan_prediksi(args.seed, args.final)

    # 5. Menyimpan hasil keluaran melalui modul evaluasi / direktori target
    out_dir = root_dir / "results" / NIM
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "pred_val.npy", np.asarray(pred_val))
    print(f"[+] Sukses menyimpan: {out_dir / 'pred_val.npy'}")

    if pred_test is not None:
        np.save(out_dir / "pred_test.npy", np.asarray(pred_test))
        print(f"[+] Sukses menyimpan: {out_dir / 'pred_test.npy'}")

if __name__ == "__main__":
    main()