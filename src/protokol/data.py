"""Pemuat data tim 30: angka lisan 0-9 + kelas 'lain' -> log-mel 64 pita, 1 detik[cite: 1, 3, 5].

muat() -> (X_train, y_train, X_val, y_val, X_test, y_test), X berbentuk (N, 64, 101)[cite: 5].

Dua mode:
  1. NYATA   : bila data/label.csv ada. Membaca data/audio_index.csv + data/audio/*.wav
               (rekaman web, laju asli mikrofon), memotong per kondisi, resample ke 16 kHz,
               membuang 1 detik pertama, memusatkan jendela 1 detik pada ucapan,
               mengambil jendela tanpa ucapan untuk kelas 'lain', log-mel, dan normalisasi[cite: 3, 5].
  2. TIRUAN  : bila label.csv tidak ada. Data acak berstruktur HANYA untuk uji asap run.py.
               Hasilnya tidak bermakna. Peringatan dicetak ke stderr[cite: 5].
"""
from __future__ import annotations

import sys
import wave
from math import gcd
from pathlib import Path

import numpy as np

from src.common import manifest

TUGAS = "klasifikasi"

DATA = manifest.AKAR / "data"
KOLOM = dict(partisipan="partisipan", angka="angka", kipas="kipas",
             t0="t_mulai_host", t1="t_selesai_host")
KATA = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "lain"]

SR = 16000
N_MEL, N_FFT, HOP = 64, 512, 160
BUANG_AWAL = 1.0      # detik pertama kondisi dibuang (peralihan)
MARGIN = 0.25         # detik; menutup ketidaktelitian jam (~250 ms) dan latensi mikrofon
DURASI_KONDISI = 2.5
N_UJI, N_VAL = 2, 1   # jumlah partisipan untuk uji dan validasi
SEED_SPLIT = 2026


# ----------------------------------------------------------------- log-mel
def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_inv(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def _bank(sr=SR, n_fft=N_FFT, n_mel=N_MEL, fmin=50.0, fmax=8000.0) -> np.ndarray:
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    hz = _mel_inv(np.linspace(_mel(fmin), _mel(fmax), n_mel + 2))
    fb = np.zeros((n_mel, len(f)), np.float32)
    for i in range(n_mel):
        lo, c, up = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0, np.minimum((f - lo) / (c - lo), (up - f) / (up - c)))
    return fb


_FB = _bank()
_WIN = np.hanning(N_FFT).astype(np.float32)


def log_mel(x: np.ndarray) -> np.ndarray:
    """x: 16000 sampel float -> (64, 101) log-mel."""
    x = np.pad(x.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spek = np.abs(np.fft.rfft(x[idx] * _WIN, axis=1)) ** 2
    return np.log(spek @ _FB.T + 1e-6).T.astype(np.float32)


# ------------------------------------------------------------------- audio
def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def _baca_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, f"{p.name}: harus mono 16 bit"
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


class _Audio:
    """Potongan dari audio_index.csv; ambil(a, b) mengembalikan sinyal 16 kHz."""

    def __init__(self):
        import pandas as pd
        self.idx = pd.read_csv(DATA / "audio_index.csv")
        self.cache: dict[str, np.ndarray] = {}

    def ambil(self, a: float, b: float) -> np.ndarray:
        bagian, sr_pakai = [], None
        for r in self.idx.itertuples():
            sr, n = int(r.sample_rate), int(r.jumlah_sampel)
            ts = float(r.t_mulai_host)
            if ts + n / sr < a or ts > b:
                continue
            if r.berkas not in self.cache:
                self.cache[r.berkas] = _baca_wav(DATA / "audio" / r.berkas)
            i0, i1 = max(0, round((a - ts) * sr)), min(n, round((b - ts) * sr))
            if i1 > i0:
                bagian.append(self.cache[r.berkas][i0:i1])
                sr_pakai = sr
        if not bagian:
            return np.zeros(0, np.float32)
        return _resample(np.concatenate(bagian), sr_pakai)


def _jendela_ucapan(x: np.ndarray) -> np.ndarray | None:
    """Jendela 1 detik yang dipusatkan pada energi ucapan; None bila sinyal terlalu pendek."""
    if len(x) < 0.8 * SR:
        return None
    if len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    fr = 800  # 50 ms
    n = len(x) // fr
    rms = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1))
    aktif = rms > 0.5 * rms.max()
    pusat = (np.arange(n)[aktif] * fr + fr // 2).mean()
    mulai = int(np.clip(pusat - SR // 2, 0, len(x) - SR))
    return x[mulai: mulai + SR]


def _jendela_tanpa_ucapan(x: np.ndarray) -> np.ndarray | None:
    """Potongan 1 detik berenergi terendah (tanpa ucapan) untuk kelas 'lain'."""
    if len(x) < SR:
        return None
    fr = 800  # 50 ms
    n_jendela = SR // fr
    n = len(x) // fr
    rms = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1))
    energi_blok = np.convolve(rms, np.ones(n_jendela), mode="valid")
    mulai = int(np.argmin(energi_blok)) * fr
    return x[mulai: mulai + SR]


# ---------------------------------------------------------------- mode nyata
def _muat_nyata():
    import pandas as pd
    lab = pd.read_csv(DATA / "label.csv")
    k = KOLOM
    audio = _Audio()
    X, y, g = [], [], []
    dibuang = 0
    for idx_baris, r in enumerate(lab.to_dict("records")):
        t0 = float(r[k["t0"]])
        t1 = float(r.get(k["t1"], t0 + DURASI_KONDISI))
        # Buang 1 detik pertama setiap kondisi (peralihan)
        x = audio.ambil(t0 + BUANG_AWAL - MARGIN, t1 + MARGIN)
        w = _jendela_ucapan(x)
        if w is None:
            dibuang += 1
            continue
        a = r[k["angka"]]
        a = KATA.index(str(a).strip().lower()) if str(a).strip().lower() in KATA else int(a)
        partisipan = str(r[k["partisipan"]])
        X.append(log_mel(w)); y.append(a); g.append(partisipan)

        # Kelas 'lain' diambil dari potongan tanpa ucapan (1 tiap 10 kondisi agar seimbang)
        if idx_baris % 10 == 0:
            w_lain = _jendela_tanpa_ucapan(x)
            if w_lain is not None:
                X.append(log_mel(w_lain)); y.append(KATA.index("lain")); g.append(partisipan)

    if dibuang:
        print(f"[data.py] {dibuang} kondisi dibuang (audio tidak cukup).", file=sys.stderr)
    return np.stack(X), np.asarray(y, np.int64), np.asarray(g)


# -------------------------------------------------------------- mode tiruan
def _muat_tiruan():
    print("[data.py] PERINGATAN: label.csv tidak ada -> memakai DATA TIRUAN (hanya uji asap).",
          file=sys.stderr)
    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for p in range(8):
        for ang in range(10):
            for _ in range(6):                     # 10 angka x 2 kipas x 3 ulangan = 60 per partisipan
                x = rng.normal(size=(N_MEL, 101)).astype(np.float32)
                x[ang * 6: ang * 6 + 6, 30:70] += 2.0
                X.append(x); y.append(ang); g.append(f"P{p + 1:02d}")
        for _ in range(6):                         # 6 sampel kelas 'lain' per partisipan
            x_lain = rng.normal(scale=0.5, size=(N_MEL, 101)).astype(np.float32)
            X.append(x_lain); y.append(10); g.append(f"P{p + 1:02d}")
    return np.stack(X), np.asarray(y, np.int64), np.asarray(g)


# -------------------------------------------------------------------- muat
_CACHE: dict = {}


def _semua():
    if "d" not in _CACHE:
        _CACHE["d"] = _muat_nyata() if (DATA / "label.csv").exists() else _muat_tiruan()
    return _CACHE["d"]


def _masker(g):
    orang = np.array(sorted(set(g)))
    orang = orang[np.random.default_rng(SEED_SPLIT).permutation(len(orang))]
    uji, val = orang[:N_UJI], orang[N_UJI: N_UJI + N_VAL]
    m_uji, m_val = np.isin(g, uji), np.isin(g, val)
    m_tr = ~(m_uji | m_val)
    assert m_tr.any() and m_val.any() and m_uji.any(), "partisipan terlalu sedikit untuk split"

    # Pemeriksaan ketat: tidak ada satuan pemisahan yang muncul di latih dan uji sekaligus
    set_tr, set_val, set_uji = set(g[m_tr]), set(g[m_val]), set(g[m_uji])
    assert set_tr.isdisjoint(set_uji), f"Kebocoran penutur latih & uji: {set_tr & set_uji}"
    assert set_tr.isdisjoint(set_val), f"Kebocoran penutur latih & val: {set_tr & set_val}"
    assert set_val.isdisjoint(set_uji), f"Kebocoran penutur val & uji: {set_val & set_uji}"
    return m_tr, m_val, m_uji


def muat():
    X, y, g = _semua()
    m_tr, m_val, m_uji = _masker(g)

    X_train, y_train = X[m_tr], y[m_tr]
    X_val, y_val = X[m_val], y[m_val]
    X_test, y_test = X[m_uji], y[m_uji]

    # Normalisasi dihitung HANYA dari data latih (per pita mel)
    mean_tr = X_train.mean(axis=(0, 2), keepdims=True)
    std_tr = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

    X_train = (X_train - mean_tr) / std_tr
    X_val = (X_val - mean_tr) / std_tr
    X_test = (X_test - mean_tr) / std_tr

    return X_train, y_train, X_val, y_val, X_test, y_test


def penutur():
    """ID penutur tiap sampel (latih, val, uji); urutan sama dengan muat()."""
    _, _, g = _semua()
    m_tr, m_val, m_uji = _masker(g)
    return g[m_tr], g[m_val], g[m_uji]


def akurasi_per_penutur(y_true: np.ndarray, y_pred: np.ndarray, g_test: np.ndarray) -> float:
    """Metrik utama: rata-rata akurasi yang dihitung terpisah per penutur uji."""
    skor = []
    for p in sorted(set(g_test)):
        mask = (g_test == p)
        skor.append(float((y_true[mask] == y_pred[mask]).mean()))
    return float(np.mean(skor))


if __name__ == "__main__":
    X_tr, y_tr, X_va, y_va, X_te, y_te = muat()
    g_tr, g_va, g_te = penutur()
    assert set(g_tr).isdisjoint(set(g_te)), "Gagal: penutur bocor!"
    print(f"OK! Latih: {X_tr.shape} {sorted(set(g_tr))} | Val: {X_va.shape} {sorted(set(g_va))} | Uji: {X_te.shape} {sorted(set(g_te))}")