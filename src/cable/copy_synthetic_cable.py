import shutil
from pathlib import Path

# Source: where CycleGAN put the generated images
source = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/cable_cyclegan/test_60/images")

# Target: where classifier will train on defect images
target = Path("data/classifier_data/cable/train/defect")
target.mkdir(parents=True, exist_ok=True)

# Get all fake_B images
fake_images = sorted(source.glob("*_fake_B.png"))

# Copy and rename from 030.png to 079.png
for i, img_path in enumerate(fake_images[:50]):
    new_name = f"{i + 30:03}.png"
    shutil.copy(img_path, target / new_name)
    print(f"✅ Copied {img_path.name} → {new_name}")

print("\n🎉 50 synthetic defect images copied and renamed successfully.")