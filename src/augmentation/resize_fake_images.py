from PIL import Image
from pathlib import Path

# Input directory where CycleGAN outputs are stored
input_dir = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan/test_latest/images")

# Output directory where resized fake_B images will go
output_dir = Path("data/synthetic/bottle/defect")
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through all fake_B images
count = 0
for img_path in input_dir.glob("*_fake_B.png"):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))
    output_file = output_dir / img_path.name
    img.save(output_file)
    count += 1

print(f"✅ Resized and saved {count} synthetic defect images to {output_dir}")