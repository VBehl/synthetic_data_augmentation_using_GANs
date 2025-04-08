import os
import shutil

# Paths
source_dir = r"models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan_balanced_v2/train_latest/images"
dest_dir = r"data/classifier_data/train/defect"

# Ensure destination exists
os.makedirs(dest_dir, exist_ok=True)

# Counter for renamed files
counter = 0

for filename in os.listdir(source_dir):
    if filename.endswith("_fake_B.png"):
        src_path = os.path.join(source_dir, filename)
        new_name = f"syn_{counter:03d}.png"
        dest_path = os.path.join(dest_dir, new_name)
        shutil.copyfile(src_path, dest_path)
        counter += 1

print(f"✅ Copied {counter} fakeB images to {dest_dir}")


