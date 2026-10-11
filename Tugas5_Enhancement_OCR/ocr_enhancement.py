import cv2
import os
import pytesseract
import pandas as pd
import re
import numpy as np

# =========================================================
# PENGATURAN
# =========================================================

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "hasil"

# Teks asli nomor ijazah
GROUND_TRUTH = "NOMORIJAZAH571012022000056"

# Lokasi Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# FUNGSI MEMBERSIHKAN HASIL OCR
# =========================================================

def clean_ocr(text):
    text = text.upper()

    # Hanya menyimpan huruf dan angka
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# =========================================================
# FUNGSI LEVENSHTEIN DISTANCE
# =========================================================

def levenshtein_distance(a, b):

    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]


# =========================================================
# HITUNG AKURASI OCR
# =========================================================

def calculate_accuracy(ocr_text):

    error = levenshtein_distance(
        GROUND_TRUTH,
        ocr_text
    )

    karakter_benar = max(
        0,
        len(GROUND_TRUTH) - error
    )

    akurasi = (
        karakter_benar / len(GROUND_TRUTH)
    ) * 100

    return karakter_benar, akurasi


# =========================================================
# FOLDER HASIL
# =========================================================

folders = [
    "crop",
    "original",
    "mean_filter",
    "median_filter",
    "gaussian_filter",
    "sharpening"
]

for folder in folders:

    os.makedirs(
        os.path.join(
            OUTPUT_FOLDER,
            folder
        ),
        exist_ok=True
    )


# =========================================================
# MEMBACA SEMUA GAMBAR
# =========================================================

files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

files.sort()


# =========================================================
# MENYIMPAN HASIL OCR
# =========================================================

hasil_ocr = []


# =========================================================
# PROSES SETIAP GAMBAR
# =========================================================

for filename in files:

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print(
            f"Gagal membaca gambar: {filename}"
        )

        continue


    # =====================================================
    # PILIH AREA NOMOR IJAZAH
    # =====================================================

    display_image = image.copy()

    max_width = 1200

    scale = 1

    if display_image.shape[1] > max_width:

        scale = (
            max_width /
            display_image.shape[1]
        )

        display_image = cv2.resize(
            display_image,
            None,
            fx=scale,
            fy=scale
        )


    print("\n======================================")
    print(f"Gambar: {filename}")
    print("Silakan pilih area nomor ijazah.")
    print("Klik dan drag pada area nomor ijazah.")
    print("Tekan ENTER setelah selesai.")
    print("Tekan ESC jika ingin membatalkan.")

    x, y, w, h = cv2.selectROI(
        "Pilih Area Nomor Ijazah",
        display_image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()


    # Jika tidak memilih area

    if w == 0 or h == 0:

        print(
            f"Area tidak dipilih: {filename}"
        )

        continue


    # =====================================================
    # KEMBALIKAN KOORDINAT KE UKURAN ASLI
    # =====================================================

    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)


    # =====================================================
    # CROP NOMOR IJAZAH
    # =====================================================

    crop = image[
        y:y+h,
        x:x+w
    ]


    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "crop",
            filename
        ),
        crop
    )


    # =====================================================
    # GRAYSCALE
    # =====================================================

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # 1. ORIGINAL
    # =====================================================

    original = gray.copy()


    # =====================================================
    # 2. MEAN FILTER
    # =====================================================

    mean_filter = cv2.blur(
        gray,
        (5, 5)
    )


    # =====================================================
    # 3. MEDIAN FILTER
    # =====================================================

    median_filter = cv2.medianBlur(
        gray,
        5
    )


    # =====================================================
    # 4. GAUSSIAN FILTER
    # =====================================================

    gaussian_filter = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # =====================================================
    # 5. SHARPENING
    # =====================================================

    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpening = cv2.filter2D(
        gray,
        -1,
        sharpening_kernel
    )


    # =====================================================
    # KUMPULKAN SEMUA METODE
    # =====================================================

    methods = {

        "Original": original,

        "Mean Filter": mean_filter,

        "Median Filter": median_filter,

        "Gaussian Filter": gaussian_filter,

        "Sharpening": sharpening

    }


    # =====================================================
    # PROSES SETIAP METODE
    # =====================================================

    for method, result in methods.items():

        # -----------------------------------------------
        # Simpan hasil enhancement
        # -----------------------------------------------

        folder_name = {

            "Original": "original",

            "Mean Filter": "mean_filter",

            "Median Filter": "median_filter",

            "Gaussian Filter": "gaussian_filter",

            "Sharpening": "sharpening"

        }[method]


        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                folder_name,
                filename
            ),
            result
        )


        # -----------------------------------------------
        # Threshold untuk OCR
        # -----------------------------------------------

        _, threshold = cv2.threshold(
            result,
            0,
            255,
            cv2.THRESH_BINARY +
            cv2.THRESH_OTSU
        )


        # -----------------------------------------------
        # OCR
        # -----------------------------------------------

        text = pytesseract.image_to_string(
            threshold,
            config="--psm 7"
        )


        text = clean_ocr(text)


        # -----------------------------------------------
        # Hitung karakter benar & akurasi
        # -----------------------------------------------

        karakter_benar, akurasi = (
            calculate_accuracy(text)
        )


        # -----------------------------------------------
        # Simpan hasil
        # -----------------------------------------------

        hasil_ocr.append({

            "Gambar": filename,

            "Metode": method,

            "Hasil OCR": text,

            "Karakter Benar": karakter_benar,

            "Akurasi": round(
                akurasi,
                2
            )

        })


        print(
            f"{method}: {text} | "
            f"Benar: {karakter_benar} | "
            f"Akurasi: {akurasi:.2f}%"
        )


# =========================================================
# DATAFRAME HASIL
# =========================================================

df = pd.DataFrame(hasil_ocr)


# =========================================================
# SIMPAN HASIL OCR LENGKAP
# =========================================================

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_ocr.csv"
    ),
    index=False
)


# =========================================================
# BUAT TABEL PER GAMBAR
# =========================================================

for filename in df["Gambar"].unique():

    data_gambar = df[
        df["Gambar"] == filename
    ]

    nama_file = os.path.splitext(
        filename
    )[0]

    data_gambar[
        [
            "Metode",
            "Hasil OCR",
            "Karakter Benar",
            "Akurasi"
        ]
    ].to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            f"tabel_{nama_file}.csv"
        ),

        index=False
    )


# =========================================================
# RINGKASAN AKURASI
# =========================================================

ringkasan = df[
    [
        "Gambar",
        "Metode",
        "Karakter Benar",
        "Akurasi"
    ]
]

ringkasan.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "ringkasan_akurasi.csv"
    ),
    index=False
)


# =========================================================
# SELESAI
# =========================================================

print("\n======================================")
print("SEMUA PROSES SELESAI")
print("======================================")

print(
    f"Hasil tersimpan di folder: "
    f"{OUTPUT_FOLDER}"
)

print("\nGround Truth:")
print(GROUND_TRUTH)