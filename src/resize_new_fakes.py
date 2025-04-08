
from PIL import Image
from pathlib import Path

# Source and target paths
source_dir = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan/train_latest/images")
target_dir = Path("data/classifier_data/train/defect")
target_dir.mkdir(parents=True, exist_ok=True)

# Resize and save
count = 0
for img_path in source_dir.glob("*_fake_B.png"):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))
    save_path = target_dir / f"new_fake_{count:03d}.png"
    img.save(save_path)
    count += 1

print(f"✅ Resized and moved {count} new synthetic defect images to {target_dir}")


