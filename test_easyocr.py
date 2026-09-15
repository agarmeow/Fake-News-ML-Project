import easyocr

image_path = input("Enter image path: ").strip()

# gpu=True if you want to use your RTX 2050; first run downloads
# the detection+recognition models (a few hundred MB, one-time).
reader = easyocr.Reader(["en"], gpu=True)

results = reader.readtext(image_path)

print("\n" + "=" * 60)
print("EASYOCR RESULTS (each detected text region)")
print("=" * 60)

detected_pieces = []

for (bbox, text, confidence) in results:
    print(f"\nText: {text}")
    print(f"Confidence: {confidence:.4f}")
    detected_pieces.append(text)

print("\n" + "=" * 60)
print("COMBINED TEXT (all regions joined)")
print("=" * 60)
print(" ".join(detected_pieces))