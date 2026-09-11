import zipfile
from pathlib import Path
import shutil

# Main folders
raw_folder = Path("data/raw")
extracted_folder = Path("data/extracted")

# Create extracted folder if it doesn't exist
extracted_folder.mkdir(parents=True, exist_ok=True)

# Find all ZIP files
zip_files = list(raw_folder.glob("*.zip"))

print("=" * 60)
print("EXTRACTING FAKE NEWS DATASETS")
print("=" * 60)

for zip_path in zip_files:

    print(f"\nProcessing: {zip_path.name}")

    # Determine language from filename
    name = zip_path.name.lower()

    if "gujarati" in name:
        language = "gujarati"
    elif "hindi" in name:
        language = "hindi"
    elif "marathi" in name:
        language = "marathi"
    elif "telugu" in name:
        language = "telugu"
    else:
        language = zip_path.stem.lower()

    # Create language-specific folder
    language_folder = extracted_folder / language
    language_folder.mkdir(parents=True, exist_ok=True)

    # Open ZIP
    with zipfile.ZipFile(zip_path, "r") as z:

        extracted_count = 0

        for member in z.infolist():

            # Ignore directories
            if member.is_dir():
                continue

            # Ignore Mac metadata
            if "__MACOSX" in member.filename:
                continue

            # Only extract TXT files
            if not member.filename.lower().endswith(".txt"):
                continue

            # Get just the filename
            filename = Path(member.filename).name

            # Extract directly into language folder
            destination = language_folder / filename

            with z.open(member) as source:
                with open(destination, "wb") as target:
                    shutil.copyfileobj(source, target)

            extracted_count += 1

    print(f"Language: {language}")
    print(f"Articles extracted: {extracted_count}")

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)