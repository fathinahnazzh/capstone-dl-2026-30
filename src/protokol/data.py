"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Isi oleh Penjaga Data & Evaluasi. Kontrak yang dipakai penilai dan baseline starter:

    TUGAS = "klasifikasi"  # atau "regresi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

- X berupa array NumPy (bentuk bebas: [N, fitur], [N, C, H, W], [N, langkah, fitur]);
  y klasifikasi = label bulat 0..K-1, y regresi = float.
- Split latih/validasi WAJIB memakai ``src.common.split`` (seed 2026) agar sama di semua laptop.
- y_test boleh ``None`` bila dataset tidak punya bagian uji; bila ada, data uji hanya dievaluasi
  sekali di sprint terakhir.
- Berkas data dibaca dari ``data/`` dan dicantumkan di ``data/MANIFEST.json`` (sidik data).

Contoh lengkap: ``contoh_fashion_mnist.py`` di folder ini.
"""
from __future__ import annotations

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"


def muat():
    """Muat CSV kasus atau buat dataset simulasi deterministik untuk smoke run."""
    m = manifest.baca()
    path = manifest.AKAR / m["berkas"][0]
    if not path.exists():
        # Keperluan pengembangan: data sintetis tersimpan di data/ (di-ignore Git).
        # Tiap kelas memiliki pola fitur yang dapat dipelajari baseline sederhana.
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(2026)
        y = np.arange(1000, dtype=np.int64) % 10
        rng.shuffle(y)
        X = rng.normal(0, 0.35, (len(y), 64 * 8)).astype(np.float32)
        X[np.arange(len(y)), y * 8 + np.arange(len(y)) % 8] += 2.0
        np.savetxt(path, np.column_stack([X, y]), delimiter=",")
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
    X, y = raw[:, :-1].reshape(-1, 1, 64, 8), raw[:, -1].astype(np.int64)
    tr, va = split.indeks(len(X))
    return X[tr], y[tr], X[va], y[va], None, None
