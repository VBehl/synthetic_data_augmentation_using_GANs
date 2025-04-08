import os
from PIL import Image

# Define input and output directories
input_dir = "data/classifier_data/train/defect"
output_dir = "data/classifier_data/train/defect_resized"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Target size
target_size = (128, 128)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        try:
            with Image.open(img_path) as img:
                # Convert image to RGB (in case it's not)
                img = img.convert("RGB")
                # Resize the image
                img_resized = img.resize(target_size)
                # Save the resized image to the output directory with the same filename
                img_resized.save(os.path.join(output_dir, filename))
                print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("✅ All images have been resized to 128x128 and saved to", output_dir)


