import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# ============================================================
# PENGATURAN
# ============================================================

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "hasil"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# DAFTAR 9 CITRA IJAZAH
# ============================================================

nama_citra = [
    "ijazah1.jpg",
    "ijazah2.jpg",
    "ijazah3.jpg",
    "ijazah4.jpg",
    "ijazah5.jpg",
    "ijazah6.jpg",
    "ijazah7.jpg",
    "ijazah8.jpg",
    "ijazah9.jpg"
]


# ============================================================
# FUNGSI GAMMA CORRECTION
# ============================================================

def gamma_correction(image, gamma):

    tabel = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, tabel)


# ============================================================
# FUNGSI CLAHE
# ============================================================

def clahe_enhancement(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


# ============================================================
# FUNGSI MEDIAN FILTER UNTUK NOISE
# ============================================================

def median_filter(image):

    return cv2.medianBlur(image, 3)


# ============================================================
# FUNGSI KOREKSI PENCAHAYAAN
# ============================================================

def illumination_correction(image):

    # Membuat perkiraan background/pencahayaan
    background = cv2.GaussianBlur(
        image,
        (0, 0),
        25
    )

    # Hindari pembagian dengan nol
    background = background.astype(np.float32) + 1

    # Koreksi pencahayaan
    corrected = (
        image.astype(np.float32) /
        background
    ) * 255

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    return corrected


# ============================================================
# ANALISIS KONDISI CITRA
# ============================================================

def analisis_kondisi(gray):

    # Rata-rata intensitas
    mean = np.mean(gray)

    # Standar deviasi = ukuran penyebaran/kontras
    std = np.std(gray)

    # Intensitas minimum dan maksimum
    minimum = np.min(gray)
    maksimum = np.max(gray)

    # Persentase piksel gelap
    piksel_gelap = np.sum(gray < 50)
    persen_gelap = (
        piksel_gelap / gray.size
    ) * 100

    # Persentase piksel terang
    piksel_terang = np.sum(gray > 200)
    persen_terang = (
        piksel_terang / gray.size
    ) * 100

    # --------------------------------------------------------
    # DETEKSI PENCAHAYAAN TIDAK MERATA
    # --------------------------------------------------------

    # Membagi gambar menjadi 4 bagian
    tinggi, lebar = gray.shape

    bagian = [
        gray[0:tinggi//2, 0:lebar//2],
        gray[0:tinggi//2, lebar//2:lebar],
        gray[tinggi//2:tinggi, 0:lebar//2],
        gray[tinggi//2:tinggi, lebar//2:lebar]
    ]

    rata_bagian = [
        np.mean(x)
        for x in bagian
    ]

    perbedaan_pencahayaan = (
        max(rata_bagian) -
        min(rata_bagian)
    )

    # --------------------------------------------------------
    # DETEKSI NOISE
    # --------------------------------------------------------

    median = cv2.medianBlur(gray, 3)

    noise_residual = np.mean(
        np.abs(
            gray.astype(np.float32) -
            median.astype(np.float32)
        )
    )

    # --------------------------------------------------------
    # MENENTUKAN KONDISI
    # --------------------------------------------------------

    if noise_residual > 8:
        kondisi = "Noise"
        metode = "Median Filter"

    elif perbedaan_pencahayaan > 35:
        kondisi = "Pencahayaan tidak merata"
        metode = "Illumination Correction"

    elif mean < 80:
        kondisi = "Terlalu gelap"
        metode = "Gamma Correction"

    elif mean > 190:
        kondisi = "Terlalu terang"
        metode = "Gamma Correction"

    elif std < 40:
        kondisi = "Kontras rendah"
        metode = "CLAHE"

    else:
        kondisi = "Relatif normal"
        metode = "Tidak perlu enhancement khusus"

    return {
        "mean": mean,
        "std": std,
        "minimum": minimum,
        "maksimum": maksimum,
        "persen_gelap": persen_gelap,
        "persen_terang": persen_terang,
        "perbedaan_pencahayaan": perbedaan_pencahayaan,
        "noise_residual": noise_residual,
        "kondisi": kondisi,
        "metode": metode
    }


# ============================================================
# MEMBUAT HISTOGRAM
# ============================================================

def buat_histogram(rgb, gray, nama_file):

    plt.figure(figsize=(12, 5))

    # --------------------------------------------------------
    # HISTOGRAM RGB
    # --------------------------------------------------------

    plt.subplot(1, 2, 1)

    warna = ["r", "g", "b"]

    for i in range(3):

        histogram = cv2.calcHist(
            [rgb],
            [i],
            None,
            [256],
            [0, 256]
        )

        plt.plot(
            histogram,
            color=warna[i]
        )

    plt.title("Histogram Citra RGB")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim([0, 256])

    # --------------------------------------------------------
    # HISTOGRAM GRAYSCALE
    # --------------------------------------------------------

    plt.subplot(1, 2, 2)

    histogram_gray = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.plot(
        histogram_gray,
        color="black"
    )

    plt.title("Histogram Citra Grayscale")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim([0, 256])

    plt.tight_layout()

    nama_output = (
        os.path.splitext(nama_file)[0]
        + "_histogram.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            nama_output
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MEMBUAT GAMBAR PERBANDINGAN
# ============================================================

def buat_perbandingan(
    rgb,
    gray,
    enhanced,
    nama_file,
    kondisi,
    metode
):

    plt.figure(figsize=(15, 5))

    # --------------------------------------------------------
    # CITRA ASLI
    # --------------------------------------------------------

    plt.subplot(1, 3, 1)

    plt.imshow(rgb)
    plt.title("Citra Asli")
    plt.axis("off")

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    plt.subplot(1, 3, 2)

    plt.imshow(
        gray,
        cmap="gray"
    )

    plt.title("Grayscale")
    plt.axis("off")

    # --------------------------------------------------------
    # HASIL ENHANCEMENT
    # --------------------------------------------------------

    plt.subplot(1, 3, 3)

    plt.imshow(
        enhanced,
        cmap="gray"
    )

    plt.title(
        "Enhancement\n"
        + kondisi
        + " - "
        + metode
    )

    plt.axis("off")

    plt.tight_layout()

    nama_output = (
        os.path.splitext(nama_file)[0]
        + "_perbandingan.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            nama_output
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# PROSES 9 CITRA
# ============================================================

hasil_data = []


for nama_file in nama_citra:

    print("\n===================================")
    print("Memproses:", nama_file)
    print("===================================")

    path = os.path.join(
        INPUT_FOLDER,
        nama_file
    )

    # --------------------------------------------------------
    # BACA CITRA
    # --------------------------------------------------------

    image = cv2.imread(path)

    if image is None:

        print(
            "ERROR: File tidak ditemukan:",
            nama_file
        )

        continue

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # RGB -> GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # ANALISIS
    # --------------------------------------------------------

    analisis = analisis_kondisi(gray)

    mean = analisis["mean"]
    std = analisis["std"]
    minimum = analisis["minimum"]
    maksimum = analisis["maksimum"]
    persen_gelap = analisis["persen_gelap"]
    persen_terang = analisis["persen_terang"]
    perbedaan_pencahayaan = (
        analisis["perbedaan_pencahayaan"]
    )
    noise_residual = (
        analisis["noise_residual"]
    )
    kondisi = analisis["kondisi"]
    metode = analisis["metode"]

    # --------------------------------------------------------
    # MENAMPILKAN HASIL ANALISIS DI TERMINAL
    # --------------------------------------------------------

    print(
        f"Rata-rata intensitas : {mean:.2f}"
    )

    print(
        f"Standar deviasi      : {std:.2f}"
    )

    print(
        f"Intensitas minimum   : {minimum}"
    )

    print(
        f"Intensitas maksimum  : {maksimum}"
    )

    print(
        f"Piksel gelap         : {persen_gelap:.2f}%"
    )

    print(
        f"Piksel terang        : {persen_terang:.2f}%"
    )

    print(
        f"Perbedaan pencahayaan: "
        f"{perbedaan_pencahayaan:.2f}"
    )

    print(
        f"Indikasi noise       : "
        f"{noise_residual:.2f}"
    )

    print(
        f"Kondisi citra        : {kondisi}"
    )

    print(
        f"Metode enhancement   : {metode}"
    )

    # --------------------------------------------------------
    # SIMPAN GRAYSCALE
    # --------------------------------------------------------

    nama_gray = (
        os.path.splitext(nama_file)[0]
        + "_grayscale.jpg"
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            nama_gray
        ),
        gray
    )

    # --------------------------------------------------------
    # BUAT HISTOGRAM
    # --------------------------------------------------------

    buat_histogram(
        rgb,
        gray,
        nama_file
    )

    # --------------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------------

    if kondisi == "Terlalu gelap":

        enhanced = gamma_correction(
            gray,
            0.5
        )

    elif kondisi == "Terlalu terang":

        enhanced = gamma_correction(
            gray,
            1.8
        )

    elif kondisi == "Kontras rendah":

        enhanced = clahe_enhancement(
            gray
        )

    elif kondisi == "Pencahayaan tidak merata":

        enhanced = illumination_correction(
            gray
        )

    elif kondisi == "Noise":

        enhanced = median_filter(
            gray
        )

    else:

        enhanced = gray.copy()

    # --------------------------------------------------------
    # SIMPAN HASIL ENHANCEMENT
    # --------------------------------------------------------

    nama_enhanced = (
        os.path.splitext(nama_file)[0]
        + "_enhanced.jpg"
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            nama_enhanced
        ),
        enhanced
    )

    # --------------------------------------------------------
    # BUAT PERBANDINGAN
    # --------------------------------------------------------

    buat_perbandingan(
        rgb,
        gray,
        enhanced,
        nama_file,
        kondisi,
        metode
    )

    # --------------------------------------------------------
    # SIMPAN DATA ANALISIS
    # --------------------------------------------------------

    hasil_data.append({

        "Citra":
            nama_file,

        "Rata-rata Intensitas":
            round(mean, 2),

        "Standar Deviasi":
            round(std, 2),

        "Minimum":
            int(minimum),

        "Maksimum":
            int(maksimum),

        "Piksel Gelap (%)":
            round(persen_gelap, 2),

        "Piksel Terang (%)":
            round(persen_terang, 2),

        "Perbedaan Pencahayaan":
            round(
                perbedaan_pencahayaan,
                2
            ),

        "Indikasi Noise":
            round(
                noise_residual,
                2
            ),

        "Kondisi Citra":
            kondisi,

        "Metode Enhancement":
            metode
    })


# ============================================================
# MEMBUAT TABEL HASIL
# ============================================================

df = pd.DataFrame(
    hasil_data
)


# ============================================================
# SIMPAN CSV
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_analisis.csv"
    ),
    index=False
)


# ============================================================
# SELESAI
# ============================================================

print("\n")
print("======================================")
print("SEMUA CITRA SELESAI DIANALISIS")
print("======================================")

print(
    "\nHasil tersimpan di folder:",
    OUTPUT_FOLDER
)

print(
    "\nFile tabel:",
    "hasil/hasil_analisis.csv"
)

print("\nRingkasan:")

print(
    df[
        [
            "Citra",
            "Kondisi Citra",
            "Metode Enhancement"
        ]
    ].to_string(index=False)
)