# Program Deteksi Tanda Tangan pada Ijazah

## Deskripsi

Program ini digunakan untuk mendeteksi keberadaan tanda tangan pada gambar ijazah menggunakan Python dan OpenCV. Proses deteksi dilakukan melalui pemotongan area tertentu (ROI), konversi gambar ke grayscale, thresholding, dan perhitungan rasio piksel foreground.

## Persiapan

Pastikan Python sudah terinstal di komputer. Buka terminal pada folder proyek, kemudian instal library yang diperlukan:

```bash
pip install opencv-python numpy pandas
```

## Cara Menjalankan Program

1. Buka folder proyek di Visual Studio Code.
2. Pastikan gambar ijazah sudah ditempatkan pada folder `images/ada_ttd/` dan `images/tidak_ada_ttd/`.
3. Buka terminal di VS Code melalui menu **Terminal → New Terminal**.
4. Jalankan perintah berikut:

```bash
python main.py
```

5. Program akan menampilkan gambar untuk memilih area tanda tangan (ROI). Pilih area yang sama pada setiap gambar, lalu tekan **Enter** atau **Space** untuk melanjutkan.
6. Tunggu hingga seluruh gambar selesai diproses.
7. Hasil deteksi dan perhitungan akurasi akan ditampilkan oleh program.

## Hasil Program

Hasil deteksi disimpan dalam file CSV bernama `hasil_deteksi_tanda_tangan.csv` di dalam folder `hasil/`. File tersebut berisi label asli, rasio foreground, hasil prediksi, dan status benar atau salah.

## Teknologi

* Python
* OpenCV
* NumPy
* Pandas
