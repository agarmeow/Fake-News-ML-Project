import sys
import os

# Add src/ to the path so we can import fact_checker exactly
# the way it imports its own sibling modules.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

import easyocr
from fact_check.fact_checker import fact_check


def extract_text(image_path):
    """
    Run EasyOCR on the image and return every detected text region
    joined together, unfiltered. We deliberately don't drop low-confidence
    detections here -- as seen on the Jamie Chastain test, a correct-but-
    garbled phrase can score LOWER confidence than a wrong guess would,
    so confidence isn't a safe filter for what to keep.
    """

    reader = easyocr.Reader(["en"], gpu=True)
    results = reader.readtext(image_path)

    pieces = [text for (_, text, _) in results]
    return " ".join(pieces)


def main():

    image_path = input("Enter image path: ").strip()

    print("\nRunning OCR...")
    extracted_text = extract_text(image_path)

    print("\n" + "=" * 60)
    print("OCR EXTRACTED TEXT (raw)")
    print("=" * 60)
    print(extracted_text)

    print("\n" + "=" * 60)
    print(
        "Clean this up into a single factual claim "
        "(drop site names/boilerplate, fix obvious spelling), "
        "or press Enter to use it exactly as-is:"
    )
    edited = input("> ").strip()

    claim = edited if edited else extracted_text

    print(f"\nSending to verifier:\n{claim}")

    fact_check(claim)


if __name__ == "__main__":
    main()