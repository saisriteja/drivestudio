import os
import shutil
from PIL import Image

# Paths
original_images_dir = "/data/teja/3d_flare_removal/scripts/teja_ds_v2/drivestudio/data/waymo/processed/training/010/images"
original_masks_dir = "/data/teja/3d_flare_removal/scripts/teja_ds_v2/drivestudio/data/waymo/processed/training/010/sky_masks"
target_dir = "/data/teja/3d_flare_removal/scripts/teja_ds_v2/drivestudio/data/waymo/processed/training/010/sky_masks_refined"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Mapping between position names and numbers
position_to_number = {
    "front": "0",
    "front_left": "1",
    "side_left": "2", 
    "front_right": "3",
    "side_right": "4"
}

# Get list of files in both directories
image_files = os.listdir(original_images_dir)
mask_files = os.listdir(original_masks_dir)

# Create mapping from original mask names to target names
mapping = {}
for mask_file in mask_files:
    # Parse the position and number
    parts = mask_file.split('_')
    
    if len(parts) == 2:  # front_XXXX.png
        position = "front"
        number = parts[1].split('.')[0]
    else:  # position_direction_XXXX.png
        position = f"{parts[0]}_{parts[1]}"
        number = parts[2].split('.')[0]
    
    # Remove leading zeros from the number
    number_int = int(number)
    
    # Format: XXX_Y.png
    target_name = f"{number_int:03d}_{position_to_number[position]}.png"
    mapping[mask_file] = target_name

# Copy and rename files
for original_name, new_name in mapping.items():
    src_path = os.path.join(original_masks_dir, original_name)
    dst_path = os.path.join(target_dir, new_name)
    
    # Read the image to get its dimensions
    img = Image.open(src_path)
    # Get corresponding image file
    img_file = new_name.replace('.png', '.jpg')
    
    if img_file in image_files:
        # Get dimensions of the corresponding image file
        img_path = os.path.join(original_images_dir, img_file)
        target_img = Image.open(img_path)
        target_size = target_img.size
        
        # Resize the mask to match the image
        resized_img = img.resize(target_size)
        
        # Save the resized image to the new path
        resized_img.save(dst_path)
        print(f"Processed: {original_name} → {new_name}")
    else:
        print(f"Warning: Corresponding image for {original_name} not found")

print("Processing complete!")