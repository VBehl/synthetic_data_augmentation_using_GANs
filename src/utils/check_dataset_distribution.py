import os

def count_images(folder_path):
    return len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0

paths = {
    "Train - Defect": "data/classifier_data/train/defect_final",
    "Train - Good":   "data/classifier_data/train/good",
    "Test - Defect":  "data/classifier_data/test/defect",
    "Test - Good":    "data/classifier_data/test/good"
}

print("📊 Dataset Image Distribution:\n")
for label, path in paths.items():
    print(f"{label:<20}: {count_images(path)} images")



