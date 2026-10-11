
import cv2
import os
import numpy as np
import pandas as pd

# =====================================================
# PENGATURAN
# =====================================================

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "hasil"

BATAS_FOREGROUND = 0.10

METODE_KLASIFIKASI = "global"

KELAS = {
    "ada_ttd": "SIGNATURE PRESENT",
    "tidak_ada_ttd": "SIGNATURE ABSENT"
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SUBFOLDERS = [
    "crop",
    "grayscale",
    "global_threshold",
    "adaptive_threshold",
    "opening",
    "closing"
]

for folder in SUBFOLDERS:
    for kelas in KELAS:
        os.makedirs(
            os.path.join(OUTPUT_FOLDER, folder, kelas),
            exist_ok=True
        )

# =====================================================
# PROSES GAMBAR
# =====================================================

hasil_data = []

for folder_kelas, label_asli in KELAS.items():

    folder_input = os.path.join(INPUT_FOLDER, folder_kelas)

    if not os.path.isdir(folder_input):
        print(f"Folder tidak ditemukan: {folder_input}")
        continue

    daftar_file = sorted([
        f for f in os.listdir(folder_input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for filename in daftar_file:

        path = os.path.join(folder_input, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"Gagal membaca: {filename}")
            continue

        print("\n================================")
        print("Gambar:", filename)
        print("Label asli:", label_asli)
        print("Pilih area tanda tangan, lalu tekan ENTER.")

        tinggi, lebar = image.shape[:2]
        skala = min(1.0, 1200 / lebar)

        if skala < 1.0:
            tampilan = cv2.resize(
                image, None, fx=skala, fy=skala
            )
        else:
            tampilan = image.copy()

        # Pilih ROI secara manual
        x, y, w, h = cv2.selectROI(
            "Pilih Area Tanda Tangan",
            tampilan,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyAllWindows()

        if w == 0 or h == 0:
            print("Gambar dilewati.")
            continue

        # Kembalikan koordinat ke ukuran asli
        x = int(x / skala)
        y = int(y / skala)
        w = int(w / skala)
        h = int(h / skala)

        crop = image[y:y+h, x:x+w]

        # Menyimpan hasil crop
        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "crop", folder_kelas, filename
            ),
            crop
        )

        # =================================================
        # 1. GRAYSCALE
        # =================================================

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "grayscale",
                folder_kelas, filename
            ),
            gray
        )

        # =================================================
        # 2. GLOBAL THRESHOLD (OTSU)
        # =================================================

        _, global_thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # =================================================
        # 3. ADAPTIVE THRESHOLD
        # =================================================

        if min(gray.shape) >= 31:
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31, 10
            )
        else:
            adaptive_thresh = global_thresh.copy()

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "global_threshold",
                folder_kelas, filename
            ),
            global_thresh
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "adaptive_threshold",
                folder_kelas, filename
            ),
            adaptive_thresh
        )

        # =================================================
        # 4. MORPHOLOGICAL OPENING DAN CLOSING
        # =================================================

        kernel = np.ones((2, 2), np.uint8)

        def proses_morfologi(binary):
            opening = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, kernel
            )
            closing = cv2.morphologyEx(
                opening, cv2.MORPH_CLOSE, kernel
            )
            return opening, closing

        global_opening, global_closing = (
            proses_morfologi(global_thresh)
        )

        adaptive_opening, adaptive_closing = (
            proses_morfologi(adaptive_thresh)
        )

        if METODE_KLASIFIKASI == "adaptive":
            opening = adaptive_opening
            closing = adaptive_closing
        else:
            opening = global_opening
            closing = global_closing

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "opening",
                folder_kelas, filename
            ),
            opening
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER, "closing",
                folder_kelas, filename
            ),
            closing
        )

        # =================================================
        # 5. HITUNG FOREGROUND
        # =================================================

        jumlah_foreground = cv2.countNonZero(closing)
        total_pixel = closing.shape[0] * closing.shape[1]

        rasio = (
            jumlah_foreground / total_pixel
            if total_pixel > 0 else 0
        )

        # =================================================
        # 6. DETEKSI TANDA TANGAN
        # =================================================

        if rasio < BATAS_FOREGROUND:
            prediksi = "SIGNATURE PRESENT"
        else:
            prediksi = "SIGNATURE ABSENT"

        benar = prediksi == label_asli

        print("Metode:", METODE_KLASIFIKASI)
        print("Foreground:", jumlah_foreground)
        print(f"Rasio foreground: {rasio:.2%}")
        print("Label asli:", label_asli)
        print("Hasil deteksi:", prediksi)
        print("Prediksi benar:", "Ya" if benar else "Tidak")

        hasil_data.append({
            "Gambar": filename,
            "Label Asli": label_asli,
            "Metode Threshold": METODE_KLASIFIKASI,
            "Jumlah Foreground": jumlah_foreground,
            "Total Pixel": total_pixel,
            "Rasio Foreground (%)": round(rasio * 100, 2),
            "Hasil Deteksi": prediksi,
            "Prediksi Benar": benar
        })

# =====================================================
# 7. SIMPAN LAPORAN DAN EVALUASI
# =====================================================

df = pd.DataFrame(hasil_data)

csv_path = os.path.join(
    OUTPUT_FOLDER, "hasil_deteksi_tanda_tangan.csv"
)
df.to_csv(csv_path, index=False)

print("\n================================")
print("PENGUJIAN SELESAI")
print("Jumlah gambar diproses:", len(df))

if not df.empty:
    jumlah_benar = int(df["Prediksi Benar"].sum())
    jumlah_total = len(df)
    akurasi = jumlah_benar / jumlah_total * 100

    print(f"Prediksi benar: {jumlah_benar}/{jumlah_total}")
    print(f"Akurasi deteksi: {akurasi:.2f}%")
    print("Laporan tersimpan:", csv_path)

    print("\nRingkasan per label asli:")
    print(
        df.groupby("Label Asli")["Prediksi Benar"]
        .agg(["sum", "count"])
    )