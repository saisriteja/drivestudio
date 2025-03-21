import os
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from tqdm import tqdm

# Load SegFormer model and feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
model.eval()

# Define the label ID for "sky"
SKY_LABEL = 10  # From id2label mapping

def process_image(image_path):
    """Process an image and return a binary sky mask."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Save original image size
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # (1, num_labels, H/4, W/4)
    mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # (H/4, W/4)
    
    # Resize mask back to original image size
    sky_mask = (mask == SKY_LABEL).astype(np.uint8) * 255  # Binary mask (0 or 255)
    sky_mask = Image.fromarray(sky_mask).resize(original_size, Image.NEAREST)
    return sky_mask

def process_directory(base_dir):
    """Iterate over image directories, process images, and save sky masks."""
    for scene_id in os.listdir(base_dir):
        scene_path = os.path.join(base_dir, scene_id)
        images_dir = os.path.join(scene_path, "images")
        sky_masks_dir = os.path.join(scene_path, "sky_masks")
        
        if not os.path.exists(images_dir) or not os.path.exists(sky_masks_dir):
            continue  # Skip if necessary directories don't exist
        
        os.makedirs(sky_masks_dir, exist_ok=True)
        
        for image_name in tqdm(os.listdir(images_dir), desc=f"Processing {scene_id}"):
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(sky_masks_dir, image_name)
            
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files
            
            sky_mask = process_image(image_path)
            sky_mask.save(mask_path)
            
            # print(f"Saved sky mask: {mask_path}")

if __name__ == "__main__":
    base_directory = "/data/teja/3d_flare_removal/scripts/teja_ds_v2/teja_drive_studio/drivestudio/data/waymo/processed/training"
    process_directory(base_directory)