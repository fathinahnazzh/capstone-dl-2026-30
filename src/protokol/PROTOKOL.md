# Protokol evaluasi tim — Tim 30 (Pengenalan Angka Lisan 0–9)

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Pengenalan angka lisan 0–9 untuk menyetel kecepatan/durasi alat (PWM kipas 0–255) dengan suara |
| Sumber dataset | Rekaman web meja 5 V Tim 30, sesi 23 Nov 2026 (`tim30/web-20261123-100451-1bd7`, 33 potongan audio 60 detik mono PCM 16 bit 48 kHz) dan `label.csv` dari sesi komputer meja (`8 partisipan × 60 kondisi = 480 kondisi`) |
| Jenis tugas (`TUGAS`) | `klasifikasi` (11 kelas: `nol`, `satu`, `dua`, `tiga`, `empat`, `lima`, `enam`, `tujuh`, `delapan`, `sembilan`, serta kelas `lain` dari potongan tanpa ucapan) |
| Target / label | Kolom `angka` dari `data/label.csv` (`0`–`9`) dan kelas `lain` (`10`)[cite: 4] |
| Ukuran data latih / validasi / uji | 5 partisipan latih (~330 jendela) / 1 partisipan validasi (~66 jendela) / 2 partisipan uji (~132 jendela) |
| Cara split | Pemisahan per partisipan (`P01`–`P08`) menggunakan `SEED_SPLIT = 2026` di `src/protokol/data.py` (`N_UJI = 2`, `N_VAL = 1`, sisa latih). Seluruh jendela satu penutur berada pada lipatan yang sama (tidak pernah terbagi antara latih dan uji) |
| Metrik utama untuk komparasi | **Akurasi per penutur** (rata-rata akurasi klasifikasi yang dihitung terpisah pada setiap penutur uji) |
| Praproses yang sama untuk semua model | 1. Resample dari 48 kHz ke 16 kHz mono[cite: 4]; 2. Buang 1,0 detik pertama tiap kondisi (masa peralihan); 3. Ambil 1 jendela berdurasi 1,0 detik yang dipusatkan pada energi ucapan (dan potongan energi terendah untuk kelas `lain`); 4. Ekstraksi Log-mel 64 pita (`N_FFT=512`, `HOP=160` -> `(64, 101)`); 5. Normalisasi standar ($\mu, \sigma$) dihitung **hanya dari data latih** lalu diterapkan ke validasi dan uji |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | *(Diisi dari `results/baseline/metrics.json` setelah baseline dijalankan)* |

Disetujui pembimbing pada: *(Tautan Pull Request)*