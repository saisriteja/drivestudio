import os
import cv2
from tqdm import tqdm

base_dir = "/data/teja/3d_flare_removal/scripts/teja_ds_v2/teja_drive_studio/drivestudio/data/waymo/processed/training"

for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir)
    images_path = os.path.join(subdir_path, "images")
    flare_mask_path = os.path.join(subdir_path, "flare_mask")

    if not os.path.exists(images_path) or not os.path.exists(flare_mask_path):
        continue

    image_files = sorted(os.listdir(images_path))

    print(f"\nProcessing {subdir} ({len(image_files)} images)...")

    for img_name in tqdm(image_files, desc=f"Resizing {subdir}", unit="img"):
        img_path = os.path.join(images_path, img_name)
        mask_path = os.path.join(flare_mask_path, img_name)

        if not os.path.exists(mask_path):
            print(f"Skipping {mask_path}, as it does not exist.")
            continue

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # Read and resize mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Failed to read mask: {mask_path}")
            continue
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Save the resized mask
        cv2.imwrite(mask_path, resized_mask)

print("\nAll resizing operations completed.")
