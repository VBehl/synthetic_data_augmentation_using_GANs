import os
import random
from PIL import Image
import torch
from torchvision import transforms

# Add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

# Full augmentation (all in tensor space)
augmentation = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),                       # Convert to tensor
    AddGaussianNoise(0., 0.05),                  # Add noise
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value='random'),  # Random erase
    transforms.ToPILImage()                      # Back to PIL for saving
])

def augment_images(input_dir, output_dir, augmentations_per_image=5):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    total = 0
    for img_file in image_files:
        path = os.path.join(input_dir, img_file)
        img = Image.open(path).convert('RGB')
        base_name = os.path.splitext(img_file)[0]

        for i in range(augmentations_per_image):
            aug_img = augmentation(img)
            aug_name = f"{base_name}_aug{i}.png"
            aug_img.save(os.path.join(output_dir, aug_name))
            total += 1

    print(f"✅ Saved {total} augmented images to {output_dir}")

# Paths
input_folder = "data/classifier_data/train/defect"
output_folder = "data/classifier_data/train/defect_augmented"

# Run
augment_images(input_folder, output_folder, augmentations_per_image=5)


