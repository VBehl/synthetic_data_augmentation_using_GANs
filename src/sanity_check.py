from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Path to generated images
img_dir = Path("models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan/test_latest/images")

# Choose an index (like '000', '001', '005' etc.)
index = "004"

# Load real and fake images
real = Image.open(img_dir / f"{index}_real_A.png")
fake = Image.open(img_dir / f"{index}_fake_B.png")

# Display side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(real)
plt.title("Real A (Defect-Free)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(fake)
plt.title("Fake B (Synthetic Defect)")
plt.axis('off')

plt.show()