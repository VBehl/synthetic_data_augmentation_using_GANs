import os
import shutil

# Paths
src_folder = r"C:\Users\nitro\defect_detection_project\datasets\bottle\real_defect_test"
dst_folder = r"C:\Users\nitro\defect_detection_project\data\classifier_data\test\defect"

# Make sure destination exists
os.makedirs(dst_folder, exist_ok=True)

# Step 1: Remove old images in destination
removed = 0
for file in os.listdir(dst_folder):
    file_path = os.path.join(dst_folder, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
        removed += 1

# Step 2: Copy new test defect images
copied = 0
for file in os.listdir(src_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        shutil.copy(os.path.join(src_folder, file), os.path.join(dst_folder, file))
        copied += 1

print(f"🗑️  Removed {removed} old images from test/defect")
print(f"✅ Copied {copied} new images to test/defect")


