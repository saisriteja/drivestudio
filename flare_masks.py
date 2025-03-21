import os
import shutil
import glob

# Source directory where flare masks are stored
source_base = '/data/teja/3d_flare_removal/scripts/night_dataset_copy/training_unet/simplified_pix2pixHD/flare_free_pix/flare'

# Target directory where we need to add flare_mask folders
target_base = '/data/teja/3d_flare_removal/scripts/teja_ds_v2/teja_drive_studio/drivestudio/data/waymo/processed/training'

# Get list of sequence folders (010, 030, etc.)
sequence_folders = [f for f in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, f))]

from tqdm import tqdm

for seq in tqdm(sequence_folders):
    # Source path for this sequence's flare masks
    source_path = os.path.join(source_base, seq)
    
    # Target path where we'll create the flare_mask directory
    target_path = os.path.join(target_base, seq, 'flare_mask')
    
    # Create flare_mask directory if it doesn't exist
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Created directory: {target_path}")
    
    # Find all files in the source directory
    mask_files = glob.glob(os.path.join(source_path, '*'))
    
    # Copy each file to the target directory
    for file_path in mask_files:
        file_name = os.path.basename(file_path)
        target_file = os.path.join(target_path, file_name)
        shutil.copy2(file_path, target_file)
        # print(f"Copied {file_name} to {target_path}")
    
    # print(f"Processed sequence {seq}: {len(mask_files)} files copied")
    # break

print("All flare masks have been copied to their respective flare_mask directories.")