import cv2
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
CATEGORY = "bottle"   # Change to "cable", "capsule", etc. if needed
IMAGE_SIZE = (128, 128)  # Resize resolution
INPUT_DIR = Path("data/raw") / CATEGORY
OUTPUT_DIR = Path("data/processed") / CATEGORY


def make_dirs():
    """Create required directories if not exist"""
    (OUTPUT_DIR / "train" / "good").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "good").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "defective").mkdir(parents=True, exist_ok=True)


def resize_and_save(src_path, dest_path):
    """Resize image and save to destination"""
    image = cv2.imread(str(src_path))
    if image is None:
        print(f"[!] Could not read {src_path}")
        return
    image = cv2.resize(image, IMAGE_SIZE)
    cv2.imwrite(str(dest_path), image)
    print("Successful")


def process_category():
    print(f"🔄 Processing category: {CATEGORY}")
    make_dirs()

    # === TRAIN GOOD ===
    train_good = INPUT_DIR / "train" / "good"
    for img_path in tqdm(train_good.glob("*.png"), desc="Train good"):
        dest = OUTPUT_DIR / "train" / "good" / img_path.name
        resize_and_save(img_path, dest)

    # === TEST GOOD ===
    test_good = INPUT_DIR / "test" / "good"
    for img_path in tqdm(test_good.glob("*.png"), desc="Test good"):
        dest = OUTPUT_DIR / "test" / "good" / img_path.name
        resize_and_save(img_path, dest)

    # === TEST DEFECTIVE ===
    test_root = INPUT_DIR / "test"
    for subdir in test_root.iterdir():
        if subdir.name == "good" or not subdir.is_dir():
            continue  # skip the good folder
        for img_path in tqdm(subdir.glob("*.png"), desc=f"Defective: {subdir.name}"):
            dest_name = f"{subdir.name}_{img_path.name}"
            dest = OUTPUT_DIR / "test" / "defective" / dest_name
            resize_and_save(img_path, dest)

    print("✅ Finished preprocessing!")


if __name__ == "__main__":
    process_category()
