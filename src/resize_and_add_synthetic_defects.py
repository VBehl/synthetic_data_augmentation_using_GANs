import os
from PIL import Image

# Source: fake_B images generated from trainA
source_dir = r"models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan/train_latest/images"
# Destination: Classifier training defect folder
dest_dir = r"data/classifier_data/train/defect"

os.makedirs(dest_dir, exist_ok=True)

count = 0
for fname in os.listdir(source_dir):
    if fname.endswith("_fake_B.png"):
        src_path = os.path.join(source_dir, fname)
        dest_path = os.path.join(dest_dir, f"synthetic_{fname}")

        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((128, 128))
            img.save(dest_path)
            count += 1
        except Exception as e:
            print(f"❌ Error with {fname}: {e}")

print(f"✅ Resized and added {count} synthetic defect images to {dest_dir}")
