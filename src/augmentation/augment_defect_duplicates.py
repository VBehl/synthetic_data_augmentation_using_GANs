import os
import shutil

# Path to original defect images
source_folder = "data/classifier_data/train/defect"

# How many times to duplicate each image
num_copies = 10

# Get all original defect images
original_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Start counter after existing files
start_index = len(original_images)

print(f"🧪 Found {len(original_images)} original defect images.")
print(f"🛠️  Duplicating each {num_copies} times...\n")

for i in range(num_copies):
    for img in original_images:
        original_path = os.path.join(source_folder, img)
        new_name = f"copy_{i}_{img}"
        new_path = os.path.join(source_folder, new_name)
        shutil.copyfile(original_path, new_path)

# Final count
final_count = len(os.listdir(source_folder))
print(f"✅ Done. Total images in 'train/defect': {final_count}")
