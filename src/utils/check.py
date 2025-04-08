# Quick script to count how many fakeB images were actually generated:
import os

folder = "models/cyclegan/pytorch-CycleGAN-and-pix2pix/results/bottle_cyclegan_balanced_v2/train_latest/images"
fakeb_files = [f for f in os.listdir(folder) if f.endswith("_fake_B.png")]
print(f"Total fakeB images: {len(fakeb_files)}")