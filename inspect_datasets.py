import zipfile
from pathlib import Path

data_folder = Path("data/raw")
zip_files = list(data_folder.glob("*.zip"))

print("=" * 70)
print("FAKE NEWS ML PROJECT - DATASET STRUCTURE")
print("=" * 70)

for zip_path in zip_files:

    print("\n" + "-" * 70)
    print(f"DATASET: {zip_path.name}")
    print("-" * 70)

    with zipfile.ZipFile(zip_path, "r") as z:

        # Ignore Mac metadata and folders
        article_files = [
            name for name in z.namelist()
            if name.lower().endswith(".txt")
            and "__MACOSX" not in name
        ]

        fake_files = [
            name for name in article_files
            if "_fake_news" in name.lower()
        ]

        real_files = [
            name for name in article_files
            if "_real_news" in name.lower()
        ]

        print(f"Total actual text files : {len(article_files)}")
        print(f"Fake articles           : {len(fake_files)}")
        print(f"Real articles           : {len(real_files)}")

        # Show one fake article
        if fake_files:
            sample_fake = fake_files[0]

            print("\n--- SAMPLE FAKE ARTICLE ---")
            print(f"File: {sample_fake}")

            with z.open(sample_fake) as f:
                content = f.read().decode("utf-8", errors="ignore")

            print(content[:500])

        # Show one real article
        if real_files:
            sample_real = real_files[0]

            print("\n--- SAMPLE REAL ARTICLE ---")
            print(f"File: {sample_real}")

            with z.open(sample_real) as f:
                content = f.read().decode("utf-8", errors="ignore")

            print(content[:500])