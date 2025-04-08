from pathlib import Path
import shutil

# CycleGAN dataset source paths (adjusted to your real paths)
REAL_GOOD_TRAIN = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/datasets/bottle/trainA")
REAL_GOOD_TEST  = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/datasets/bottle/testA")
REAL_DEFECT_TEST = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/datasets/bottle/testB")
SYNTHETIC_DEFECTS = Path("data/synthetic/bottle/defect")

# Target classifier dataset structure
TRAIN_GOOD = Path("data/classifier_data/train/good")
TRAIN_DEFECT = Path("data/classifier_data/train/defect")
TEST_GOOD = Path("data/classifier_data/test/good")
TEST_DEFECT = Path("data/classifier_data/test/defect")

# Ensure all required directories exist
for folder in [TRAIN_GOOD, TRAIN_DEFECT, TEST_GOOD, TEST_DEFECT]:
    folder.mkdir(parents=True, exist_ok=True)

# Step 1: Copy synthetic defects → train/defect
for i, f in enumerate(SYNTHETIC_DEFECTS.glob("*.png")):
    shutil.copy(f, TRAIN_DEFECT / f"synthetic_defect_{i:03d}.png")

# Step 2: Copy good images from trainA → train/good
for i, f in enumerate(REAL_GOOD_TRAIN.glob("*.png")):
    shutil.copy(f, TRAIN_GOOD / f"good_train_{i:03d}.png")

# Step 3: Copy good images from testA → test/good
for i, f in enumerate(REAL_GOOD_TEST.glob("*.png")):
    shutil.copy(f, TEST_GOOD / f"good_test_{i:03d}.png")

# Step 4: Copy real defect images from testB → test/defect
for i, f in enumerate(REAL_DEFECT_TEST.glob("*.png")):
    shutil.copy(f, TEST_DEFECT / f"real_defect_{i:03d}.png")

print("✅ Classifier dataset prepared at data/classifier_data/")