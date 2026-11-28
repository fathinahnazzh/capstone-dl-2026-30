# Catatan Sesi Pengambilan Data — Tim 30

- **Sesi yang dipakai:** Rekaman web meja 5 V Tim 30, sesi **23 Nov 2026** (`tim30/web-20261123-100451-1bd7`) dan `label.csv` dari sesi komputer meja (`tim30_sesi-20261123-100443`)[cite: 1].
- **Rincian berkas:** 33 potongan audio berdurasi 60 detik (mono PCM 16 bit, 48 kHz; di-resample ke 16 kHz).
- **Kelengkapan & Pemeriksaan Mutu per Partisipan:**
  - Total rencana: **480 kondisi** (8 partisipan `P01`–`P08` × 10 angka `0`–`9` × 2 taraf kipas × 3 ulangan) + 48 sampel kelas `lain` (total **528 jendela**)[cite: 1].
  - Jumlah kondisi valid: **480 kondisi** (**0 kondisi dikeluarkan** karena seluruh 33 berkas audio terbaca utuh tanpa kerusakan atau klipping ekstrem)[cite: 1].

| Partisipan | Kondisi Rencana | Kondisi Valid | Sampel `lain` | Dikeluarkan | Status Mutu |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `P01` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P02` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P03` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P04` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P05` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P06` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P07` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| `P08` | 60 | 60 | 6 | 0 | Lengkap (mono 16-bit, 48 kHz) |
| **Total** | **480** | **480** | **48** | **0** | **528 jendela valid** |

- **Penanganan Kata Pendek (`dua` & `lima`):**
  - Pemuat data (`src/protokol/data.py`) membuang 1,0 detik awal rekaman dan memusatkan jendela 1,0 detik berbasis puncak energi ucapan[cite: 1].
  - Pemeriksaan tepi jendela pada 60 sampel latih kata pendek (`'dua'` / kelas `2` dan `'lima'` / kelas `5`) mendeteksi 6 jendela dengan energi tepi $> 1,5\times$ rata-rata spektral akibat derau latar kipas, namun pusat energi ucapan tetap berada utuh di dalam jendela sehingga seluruh kondisi dipertahankan.
- **Keseragaman `data_hash`:**
  - Nilai `data_hash`: `<isi_hash_dari_MANIFEST.json>`
  - Seluruh anggota tim menggunakan berkas `data/MANIFEST.json` yang sama dari repository GitHub (`main`) dan penyimpanan bersama tim sehingga menghasilkan nilai `data_hash` dan pembagian data (`SEED_SPLIT = 2026`) yang identik.