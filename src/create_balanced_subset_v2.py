import os
import random
import shutil

# Set paths for the original dataset
full_good_dir = r"C:\Users\nitro\defect_detection_project\models\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets\bottle\trainA"
real_defect_train_dir = r"C:\Users\nitro\defect_detection_project\datasets\bottle\real_defect_train"

# Define the new balanced subset destination folder
balanced_dir = r"C:\Users\nitro\defect_detection_project\models\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets\bottle\balanced_v2"
balanced_trainA = os.path.join(balanced_dir, "trainA")  # For good images
balanced_trainB = os.path.join(balanced_dir, "trainB")  # For defect images

os.makedirs(balanced_trainA, exist_ok=True)
os.makedirs(balanced_trainB, exist_ok=True)

# Set number of good images to select (e.g., 80)
num_good = 80

# List all good images (adjust extensions if necessary)
all_good_imgs = [f for f in os.listdir(full_good_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(all_good_imgs) < num_good:
    print(f"Warning: Only found {len(all_good_imgs)} good images. Using all available images.")
    selected_good_imgs = all_good_imgs
else:
    selected_good_imgs = random.sample(all_good_imgs, num_good)

# Copy selected good images to balanced_v2/trainA
for img in selected_good_imgs:
    src = os.path.join(full_good_dir, img)
    dst = os.path.join(balanced_trainA, img)
    shutil.copyfile(src, dst)

# List all defect images from the real defect training folder
all_defect_imgs = [f for f in os.listdir(real_defect_train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img in all_defect_imgs:
    src = os.path.join(real_defect_train_dir, img)
    dst = os.path.join(balanced_trainB, img)
    shutil.copyfile(src, dst)

print(f"✅ Balanced subset created at '{balanced_dir}':")
print(f"    Good images in trainA: {len(selected_good_imgs)}")
print(f"    Defect images in trainB: {len(all_defect_imgs)}")



