import os
import random
import shutil

# ✅ Set the original dataset directory (full path)
original_data_dir = r'C:\Users\nitro\defect_detection_project\models\cyclegan\pytorch-CycleGAN-and-pix2pix\datasets\bottle'

# ✅ Set the balanced output directory
balanced_dir = os.path.join(original_data_dir, 'balanced')

# ✅ Create balanced/trainA and balanced/trainB folders
trainA_balanced = os.path.join(balanced_dir, 'trainA')
trainB_balanced = os.path.join(balanced_dir, 'trainB')
os.makedirs(trainA_balanced, exist_ok=True)
os.makedirs(trainB_balanced, exist_ok=True)

# ✅ Source folders
full_trainA_dir = os.path.join(original_data_dir, 'trainA')  # Good images
full_trainB_dir = os.path.join(original_data_dir, 'trainB')  # Defect images

# ✅ Randomly sample 80 images from trainA
all_good_imgs = [f for f in os.listdir(full_trainA_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
sampled_good_imgs = random.sample(all_good_imgs, 80)

# ✅ Copy selected good images to trainA_balanced
for img_name in sampled_good_imgs:
    src = os.path.join(full_trainA_dir, img_name)
    dst = os.path.join(trainA_balanced, img_name)
    shutil.copyfile(src, dst)

# ✅ Copy all defect images to trainB_balanced
all_defect_imgs = [f for f in os.listdir(full_trainB_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_name in all_defect_imgs:
    src = os.path.join(full_trainB_dir, img_name)
    dst = os.path.join(trainB_balanced, img_name)
    shutil.copyfile(src, dst)

print(f"✅ Copied 80 good images to {trainA_balanced}")
print(f"✅ Copied {len(all_defect_imgs)} defect images to {trainB_balanced}")


