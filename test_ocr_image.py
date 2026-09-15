import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = input("Enter image path: ").strip()

image = Image.open(image_path).convert("RGB")

# Enlarge image
scale = 3
image = image.resize(
    (image.width * scale, image.height * scale)
)

# Improve contrast and sharpness
image = ImageEnhance.Contrast(image).enhance(1.5)
image = ImageEnhance.Sharpness(image).enhance(2.0)

# Try multiple OCR layouts
configs = [
    "--psm 6",
    "--psm 11",
    "--psm 12"
]

results = []

for config in configs:
    text = pytesseract.image_to_string(
        image,
        config=config
    )

    results.append(text)

print("\n" + "=" * 60)
print("OCR RESULTS")
print("=" * 60)

for i, text in enumerate(results, 1):

    print(f"\n--- OCR configuration {i} ---")
    print(text)