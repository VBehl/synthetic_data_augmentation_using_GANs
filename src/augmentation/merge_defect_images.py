import os
import shutil

# Paths for synthetic defect images (resized) and real defect images
synthetic_dir = "data/classifier_data/train/defect_resized"
real_defect_dir = r"C:\Users\nitro\defect_detection_project\datasets\bottle\real_defect_train"  # Update if necessary

# Destination folder for the unified defect training set
destination_dir = "data/classifier_data/train/defect_final"
os.makedirs(destination_dir, exist_ok=True)

# Copy synthetic defect images to the destination
synthetic_files = [f for f in os.listdir(synthetic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for f in synthetic_files:
    src = os.path.join(synthetic_dir, f)
    dst = os.path.join(destination_dir, f)
    shutil.copyfile(src, dst)

# Copy real defect images to the destination
real_files = [f for f in os.listdir(real_defect_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for f in real_files:
    src = os.path.join(real_defect_dir, f)
    dst = os.path.join(destination_dir, f)
    shutil.copyfile(src, dst)

print(f"✅ Merged {len(synthetic_files)} synthetic and {len(real_files)} real defect images into '{destination_dir}'")


