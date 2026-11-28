"""Kerangka run.py anggota jalur Transfer Learning (TL) — NIM 123450089.

    python models/123450089/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # 123450089
JALUR = "tl"                                   # jalur Transfer Learning


def bangun_model_tl(jumlah_kelas: int):
    """Membangun model pralatih MobileNetV2 yang diadaptasi untuk spektrogram Log-mel (64, 101)."""
    import torch.nn as nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        # Cadangan jika lingkungan offline dan bobot belum ter-cache
        model = mobilenet_v2(weights=None)

    # Ganti lapisan klasifikasi terakhir agar sesuai jumlah kelas tim (11 kelas: 0-9 + lain)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, jumlah_kelas),
    )
    return model


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None) menggunakan adaptasi model pralatih."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    jumlah_kelas = int(max(np.max(ytr), np.max(yva)) + 1)

    def ke_tensor_3kanal(X: np.ndarray) -> torch.Tensor:
        # (N, 64, 101) -> (N, 3, 64, 101) agar sesuai masukan 3-kanal backbone pralatih
        t = torch.tensor(np.asarray(X, dtype=np.float32))
        if t.ndim == 3:
            t = t.unsqueeze(1).repeat(1, 3, 1, 1)
        return t

    X_train_t = ke_tensor_3kanal(Xtr)
    y_train_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    X_val_t = ke_tensor_3kanal(Xva)

    loader_latih = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    model = bangun_model_tl(jumlah_kelas).to(perangkat)

    # Tahap 1: Bekukan backbone (feature extractor), latih classifier head selama 5 epoch
    for param in model.features.parameters():
        param.requires_grad = False

    opt_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(5):
        model.train()
        for xb, yb in loader_latih:
            xb, yb = xb.to(perangkat), yb.to(perangkat)
            opt_head.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt_head.step()

    # Tahap 2: Buka pembekuan (fine-tuning) seluruh layer dengan learning rate kecil (10 epoch)
    for param in model.features.parameters():
        param.requires_grad = True

    opt_ft = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_val_acc = -1.0
    best_state = None

    for _ in range(10):
        model.train()
        for xb, yb in loader_latih:
            xb, yb = xb.to(perangkat), yb.to(perangkat)
            opt_ft.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt_ft.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t.to(perangkat))
            pred_v = logits_val.argmax(dim=1).cpu().numpy()
            val_acc = float((pred_v == np.asarray(yva)).mean())
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val_t.to(perangkat)).argmax(dim=1).cpu().numpy()
        pred_test = None
        if final and Xte is not None:
            X_test_t = ke_tensor_3kanal(Xte)
            pred_test = model(X_test_t.to(perangkat)).argmax(dim=1).cpu().numpy()

    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()