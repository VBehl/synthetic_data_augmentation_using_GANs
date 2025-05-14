import shutil
from pathlib import Path

# Paths
source = Path(r"C:\Users\nitro\defect_detection_project\models\cyclegan\pytorch-CycleGAN-and-pix2pix\results\bottle_cyclegan\test_60\images")
target = Path(r"C:\Users\nitro\defect_detection_project\data\classifier_data\bottle\train\defect")
target.mkdir(parents=True, exist_ok=True)

# Get all fake_B images
fake_images = sorted(source.glob("*_fake_B.png"))

# Copy and rename from 030.png to 079.png
for i, img_path in enumerate(fake_images[:50]):
    new_name = f"{i + 30:03}.png"
    shutil.copy(img_path, target / new_name)
    print(f"✅ Copied {img_path.name} → {new_name}")

print("\n🎉 50 synthetic bottle defect images copied successfully.")


