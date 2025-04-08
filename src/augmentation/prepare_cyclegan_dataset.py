import shutil
from pathlib import Path

CATEGORY = "bottle"
SRC_DIR = Path("data/processed") / CATEGORY
DEST_DIR = Path("models/cyclegan/datasets") / CATEGORY

def copy_images(src_folder, dest_folder):
    dest_folder.mkdir(parents=True, exist_ok=True)
    for img in src_folder.glob("*.png"):
        shutil.copy(img, dest_folder / img.name)

def prepare():
    print(f"📦 Preparing CycleGAN dataset for: {CATEGORY}")

    # Domain A = good (normal)
    copy_images(SRC_DIR / "train" / "good", DEST_DIR / "trainA")
    copy_images(SRC_DIR / "test" / "good", DEST_DIR / "testA")

    # Domain B = defect (real defects)
    copy_images(SRC_DIR / "test" / "defective", DEST_DIR / "trainB")
    copy_images(SRC_DIR / "test" / "defective", DEST_DIR / "testB")

    print("✅ Dataset ready!")

if __name__ == "__main__":
    prepare()


