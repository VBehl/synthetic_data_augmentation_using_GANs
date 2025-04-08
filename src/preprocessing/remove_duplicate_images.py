import os
import hashlib
from PIL import Image

def calculate_hash(image_path):
    with Image.open(image_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicates(folder_path):
    print(f"📂 Scanning folder: {folder_path}")
    hashes = {}
    duplicates = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                img_hash = calculate_hash(filepath)
                if img_hash in hashes:
                    duplicates.append(filepath)
                else:
                    hashes[img_hash] = filepath
            except Exception as e:
                print(f"⚠️ Could not process {filename}: {e}")

    print(f"🗑️ Found {len(duplicates)} duplicate images. Removing...")
    for dup in duplicates:
        os.remove(dup)
        print(f"✅ Removed: {dup}")

    print("🎉 Cleanup complete.")

if __name__ == "__main__":
    target_folder = "data/classifier_data/train/defect"
    remove_duplicates(target_folder)


