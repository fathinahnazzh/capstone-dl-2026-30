# Catatan model 123450072 (jalur Transformer, trf)

Status: kode diuji pada data TIRUAN (data.py sementara, tidak di-commit) karena data.py tim dan label.csv sesi meja belum tersedia. Semua angka hasil di bawah diisi setelah data meja dipakai.

## Alasan optimizer

AdamW (lr 1e-3, weight decay 1e-2) dengan jadwal cosine annealing selama 15 epoch dan gradient clipping 1.0. Alasannya:
- Transformer peka terhadap skala gradien, dan Adam mengadaptasi laju belajar per parameter, sehingga lebih stabil daripada SGD biasa pada data sekecil ini.
- AdamW memisahkan weight decay dari pembaruan gradien, sehingga regularisasinya benar-benar bekerja (pada Adam biasa, L2 ikut diskalakan oleh statistik gradien).
- Cosine annealing menurunkan lr mulus ke mendekati nol, sehingga tidak perlu menebak epoch berhenti.
- Clipping 1.0 menjaga pembaruan tetap stabil di awal pelatihan, ketika attention belum terlatih.

## Regularisasi

- Dropout 0.1 di setiap layer encoder.
- Weight decay 1e-2 (AdamW).
- Gradient clipping 1.0.
- Normalisasi per pita mel memakai statistik data latih saja, agar tidak ada kebocoran ke validasi/uji.
- Pemisahan data per partisipan, sehingga penutur yang sama tidak ada di latih dan uji.
- Belum ada augmentasi. SpecAugment direncanakan sebagai ablation.

## Pencarian hyperparameter (tabel)

Arsitektur dasar: patch 4 frame x 64 pita (sekitar 25 token), d_model 64, 4 head, 2 layer, FF 128, token CLS, positional embedding dipelajari.

| PATCH | N_LAYER | D_MODEL | DROPOUT | LR | Akurasi val | Durasi |
|---|---|---|---|---|---|---|
| 4 | 2 | 64 | 0.1 | 1e-3 | (isi) | (isi) |
| 2 | 2 | 64 | 0.1 | 1e-3 | (isi) | (isi) |
| 8 | 2 | 64 | 0.1 | 1e-3 | (isi) | (isi) |
| 4 | 3 | 64 | 0.1 | 1e-3 | (isi) | (isi) |
| 4 | 2 | 64 | 0.2 | 1e-3 | (isi) | (isi) |

Dipilih berdasarkan akurasi validasi, bukan data uji. Anggaran: maksimal 20 menit CPU laptop per run.

## Ablation (>= 2)

| Ablation | Perubahan | Akurasi val | Kesimpulan |
|---|---|---|---|
| A1 | Tanpa SpecAugment vs dengan SpecAugment | (isi) | (isi) |
| A2 | Tanpa positional embedding | (isi) | (isi) |
| A3 | Token CLS vs rata-rata token | (isi) | (isi) |

## Hasil 3 seed konfigurasi akhir

| Seed | Akurasi val | Akurasi per penutur (uji) | Durasi |
|---|---|---|---|
| 1 | (isi) | (isi) | (isi) |
| 2 | (isi) | (isi) | (isi) |
| 3 | (isi) | (isi) | (isi) |

Rata-rata dan simpangan: (isi).

Belum dapat diisi karena belum pakai data asli.