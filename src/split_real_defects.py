import os
import random
import shutil

# Set the source directory (update this if your defect images are in a different folder)
source_dir = r"C:\Users\nitro\defect_detection_project\models\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets\bottle\trainB"

# Set destination directories for splitting the real defect images
dest_train = r"C:\Users\nitro\defect_detection_project\datasets\bottle\real_defect_train"
dest_test  = r"C:\Users\nitro\defect_detection_project\datasets\bottle\real_defect_test"

os.makedirs(dest_train, exist_ok=True)
os.makedirs(dest_test, exist_ok=True)

# List all valid image files (adjust extensions if needed)
all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle and split: 40 for training, remainder for testing
random.shuffle(all_images)
train_imgs = all_images[:40]
test_imgs  = all_images[40:]

# Copy the training images
for img in train_imgs:
    shutil.copy(os.path.join(source_dir, img), os.path.join(dest_train, img))

# Copy the test images
for img in test_imgs:
    shutil.copy(os.path.join(source_dir, img), os.path.join(dest_test, img))

print(f"✅ Real defect training images: {len(train_imgs)} copied to {dest_train}")
print(f"✅ Real defect testing images: {len(test_imgs)} copied to {dest_test}")


