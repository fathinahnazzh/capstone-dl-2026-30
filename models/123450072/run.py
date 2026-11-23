"""run.py jalur Transformer (trf): encoder Transformer kecil atas patch waktu log-mel.

    python models/<nim>/run.py --seed 1 [--final]
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

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "trf"                                   # cnn | tl | rnn | trf | ae

N_MEL = 64
PATCH = 4          # jumlah frame waktu per patch
D_MODEL = 64
N_HEAD = 4
N_LAYER = 2
D_FF = 128
DROPOUT = 0.1
EPOCH = 15
BATCH = 64
LR = 1e-3


def siapkan(X) -> np.ndarray:
    """Ubah ke (N, T, 64) float32, apa pun urutan sumbunya."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:                      # (N, 1, 64, T) -> (N, 64, T)
        X = X[:, 0]
    if X.ndim == 2:                      # sudah diratakan -> (N, 64, T)
        X = X.reshape(len(X), N_MEL, -1)
    if X.shape[1] == N_MEL and X.shape[2] != N_MEL:
        X = X.transpose(0, 2, 1)         # (N, 64, T) -> (N, T, 64)
    assert X.shape[2] == N_MEL, f"bentuk masukan tak terduga: {X.shape}"
    return X


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Kembalikan (pred_val, pred_test|None)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    Xtr, Xva = siapkan(Xtr), siapkan(Xva)
    Xte = siapkan(Xte) if (final and Xte is not None) else None
    ytr = np.asarray(ytr).astype(np.int64)
    n_kelas = int(ytr.max()) + 1         # 10 angka (11 bila ada kelas 'lain')

    # normalisasi per pita mel, statistik dari data latih saja
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    if Xte is not None:
        Xte = (Xte - mu) / sd

    T = Xtr.shape[1]
    T_pad = int(np.ceil(T / PATCH) * PATCH)
    n_patch = T_pad // PATCH

    class TransformerAngka(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(PATCH * N_MEL, D_MODEL)          # patch waktu -> token
            self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
            self.pos = nn.Parameter(torch.zeros(1, n_patch + 1, D_MODEL))
            nn.init.trunc_normal_(self.pos, std=0.02)
            lapis = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,
                dropout=DROPOUT, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(lapis, num_layers=N_LAYER)
            self.norm = nn.LayerNorm(D_MODEL)
            self.kepala = nn.Linear(D_MODEL, n_kelas)

        def forward(self, x):                                       # x: (B, T, 64)
            x = F.pad(x, (0, 0, 0, T_pad - x.shape[1]))
            x = x.reshape(x.shape[0], n_patch, PATCH * N_MEL)
            x = self.embed(x)
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
            x = self.enc(x)
            return self.kepala(self.norm(x[:, 0]))                  # token CLS

    model = TransformerAngka()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCH)

    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(EPOCH):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    def prediksi(X):
        model.eval()
        keluar = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                keluar.append(model(torch.tensor(X[i:i + 256])).argmax(1).numpy())
        return np.concatenate(keluar)

    pred_val = prediksi(Xva)
    pred_test = prediksi(Xte) if Xte is not None else None
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
