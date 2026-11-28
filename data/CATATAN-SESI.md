# Catatan Sesi Pengambilan Data — Tim 30

- **Sesi yang dipakai:** Rekaman web meja 5 V Tim 30, sesi **23 Nov 2026** (`tim30/web-20261123-100451-1bd7`) dan `label.csv` dari sesi komputer meja (`tim30_sesi-20261123-100443`).
- **Rincian berkas:** 33 potongan audio berdurasi 60 detik (mono PCM 16 bit, 48 kHz; di-resample ke 16 kHz).
- **Kelengkapan kondisi:**
  - Total rencana: **480 kondisi** (8 partisipan `P01`–`P08` × 10 angka `0`–`9` × 2 taraf kipas × 3 ulangan) + 48 sampel kelas `lain` (total 528 jendela).
  - Jumlah kondisi valid: **480 kondisi** (seluruh partisipan `P01`–`P08` lengkap 60 kondisi; tidak ada kondisi yang dikeluarkan).
- **Pemeriksaan mutu khusus:**
  - Setiap partisipan (`P01`–`P08`) terverifikasi memiliki 60 kondisi lengkap.
  - Pemeriksaan tepi jendela 1,0 detik pada 60 sampel latih kata pendek (`'dua'` / kelas `2` dan `'lima'` / kelas `5`) mendeteksi 6 jendela dengan energi tepi $> 1,5\times$ rata-rata spektral akibat derau latar kipas, namun pusat energi ucapan tetap berada utuh di dalam jendela sehingga seluruh kondisi dipertahankan.
- **Keseragaman `data_hash`:** Seluruh anggota tim menggunakan berkas `data/MANIFEST.json` yang sama dari penyimpanan bersama tim sehingga menghasilkan nilai `data_hash` yang identik.
