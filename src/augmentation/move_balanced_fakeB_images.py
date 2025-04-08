import os
import shutil
from PIL import Image

source_dir = 'models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan_balanced/train_latest/images'
target_dir = 'data/classifier_data/train/defect'

# Resize config
resize_to = (128, 128)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

fakeB_images = [f for f in os.listdir(source_dir) if f.endswith('_fake_B.png')]

for i, img_name in enumerate(fakeB_images):
    img_path = os.path.join(source_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(resize_to)
    
    new_name = f"balanced_fakeB_{i:03d}.png"
    img.save(os.path.join(target_dir, new_name))

print(f"✅ Moved and resized {len(fakeB_images)} new synthetic defect images to {target_dir}")


