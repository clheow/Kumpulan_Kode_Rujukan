import cv2
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. DAFTAR CITRA
# =========================================================

image_files = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg"
]

# =========================================================
# MEMBUAT FOLDER HASIL
# =========================================================

os.makedirs("hasil/brightness", exist_ok=True)
os.makedirs("hasil/contrast", exist_ok=True)
os.makedirs("hasil/histogram_equalization", exist_ok=True)
os.makedirs("hasil/perbandingan", exist_ok=True)
os.makedirs("hasil/histogram", exist_ok=True)
# =========================================================
# 2. MEMILIH AREA NOMOR IJAZAH
# =========================================================

first_image = cv2.imread(
    os.path.join("images", image_files[0])
)

if first_image is None:
    print("Gambar pertama tidak ditemukan!")
    exit()

# Ukuran gambar asli
original_height, original_width = first_image.shape[:2]

# Ukuran maksimal gambar yang ditampilkan
max_width = 1200
max_height = 700

# Hitung skala agar gambar muat di layar
scale = min(
    max_width / original_width,
    max_height / original_height
)

# Resize untuk ditampilkan
display_width = int(original_width * scale)
display_height = int(original_height * scale)

display_image = cv2.resize(
    first_image,
    (display_width, display_height)
)

print("Silakan pilih area nomor ijazah.")
print("Klik dan drag pada area nomor ijazah.")
print("Tekan ENTER setelah selesai.")

# Pilih area pada gambar yang sudah diperkecil
x, y, w, h = cv2.selectROI(
    "Pilih Area Nomor Ijazah",
    display_image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

# Jika tidak memilih area
if w == 0 or h == 0:
    print("Area nomor ijazah belum dipilih.")
    exit()

# Kembalikan koordinat ke ukuran gambar asli
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

print("\nKoordinat crop gambar asli:")
print("x =", x)
print("y =", y)
print("width =", w)
print("height =", h)
# =========================================================
# 4. MEMPROSES SEMUA CITRA
# =========================================================

for filename in image_files:

    print("\nMemproses:", filename)

    # Baca citra
    path = os.path.join("images", filename)
    img = cv2.imread(path)

    if img is None:
        print("Gagal membaca:", filename)
        continue

    # -----------------------------------------------------
    # Crop area nomor ijazah
    # -----------------------------------------------------

    crop = img[y:y+h, x:x+w]

    # Ubah crop menjadi grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # 1. Brightness Adjustment
    # -----------------------------------------------------

    brightness = cv2.convertScaleAbs(
        gray,
        alpha=1.0,
        beta=30
    )

    # -----------------------------------------------------
    # 2. Contrast Stretching
    # -----------------------------------------------------

    contrast = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # -----------------------------------------------------
    # 3. Histogram Equalization
    # -----------------------------------------------------

    hist_eq = cv2.equalizeHist(gray)

    # Nama file tanpa ekstensi
    name = os.path.splitext(filename)[0]

    # =====================================================
    # 5. SIMPAN HASIL ENHANCEMENT
    # =====================================================

    cv2.imwrite(
        f"hasil/brightness/{name}_brightness.jpg",
        brightness
    )

    cv2.imwrite(
        f"hasil/contrast/{name}_contrast.jpg",
        contrast
    )

    cv2.imwrite(
        f"hasil/histogram_equalization/{name}_hist_eq.jpg",
        hist_eq
    )

    # =====================================================
    # 6. TAMPILKAN PERBANDINGAN
    # =====================================================

    plt.figure(figsize=(12, 8))

    # Citra sebelum enhancement
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Sebelum Enhancement")
    plt.axis("off")

    # Brightness
    plt.subplot(2, 2, 2)
    plt.imshow(brightness, cmap="gray")
    plt.title("Brightness Adjustment")
    plt.axis("off")

    # Contrast
    plt.subplot(2, 2, 3)
    plt.imshow(contrast, cmap="gray")
    plt.title("Contrast Stretching")
    plt.axis("off")

    # Histogram Equalization
    plt.subplot(2, 2, 4)
    plt.imshow(hist_eq, cmap="gray")
    plt.title("Histogram Equalization")
    plt.axis("off")

    plt.suptitle(name)

    plt.tight_layout()

    plt.savefig(
        f"hasil/perbandingan/{name}_comparison.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # 7. HISTOGRAM SEBELUM DAN SESUDAH
    # =====================================================

    plt.figure(figsize=(12, 8))

    # Sebelum enhancement
    plt.subplot(2, 2, 1)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title("Histogram Sebelum Enhancement")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Pixel")

    # Brightness
    plt.subplot(2, 2, 2)
    plt.hist(brightness.ravel(), 256, [0, 256])
    plt.title("Histogram Brightness")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Pixel")

    # Contrast Stretching
    plt.subplot(2, 2, 3)
    plt.hist(contrast.ravel(), 256, [0, 256])
    plt.title("Histogram Contrast Stretching")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Pixel")

    # Histogram Equalization
    plt.subplot(2, 2, 4)
    plt.hist(hist_eq.ravel(), 256, [0, 256])
    plt.title("Histogram Equalization")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah Pixel")

    plt.suptitle(name)

    plt.tight_layout()

    plt.savefig(
        f"hasil/histogram/{name}_histogram.png",
        dpi=300
    )

    plt.close()

    print("Selesai.")

# =========================================================
# 8. SELESAI
# =========================================================

print("\n======================================")
print("SEMUA CITRA BERHASIL DIPROSES")
print("Hasil tersedia di folder: hasil/")
print("======================================")