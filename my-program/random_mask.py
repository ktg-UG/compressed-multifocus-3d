import os
import argparse
import numpy as np
from PIL import Image

def create_random_mask(image_size, mask_percentage):
    """Create a random mask with specified percentage of masked pixels"""
    mask = np.ones(image_size)
    num_pixels = image_size[0] * image_size[1]
    num_mask_pixels = int(num_pixels * mask_percentage / 100)
    
    # Generate random indices
    mask_indices = np.random.choice(num_pixels, num_mask_pixels, replace=False)
    mask.ravel()[mask_indices] = 0
    
    return mask

def apply_mask(image_path, output_dir, mask_percentage):
    """Apply random mask to an image and save it"""
    # Read image
    img = Image.open(image_path)
    img_array = np.array(img).astype(np.float32)
    
    # 値を0-1に正規化
    img_array = img_array / 255.0
    
    # Create mask
    mask = create_random_mask(img_array.shape[:2], mask_percentage)
    
    # Apply mask
    if len(img_array.shape) == 3:  # Color image
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)
    
    masked_img = img_array * mask
    
    # 値の範囲を確保
    masked_img = np.clip(masked_img, 0.0, 1.0)
    
    # 保存前に0-255にスケール
    masked_img = (masked_img * 255).astype(np.uint8)
    
    # Create output directory
    mask_dir = os.path.join(output_dir, f"masked_{mask_percentage}")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Save masked image
    output_filename = f"masked_{mask_percentage}_{os.path.basename(image_path)}"
    output_path = os.path.join(mask_dir, output_filename)
    
    Image.fromarray(masked_img).save(output_path)
    print(f"Saved masked image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply random masks to images")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for masked images")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Mask percentages
    # mask_percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    mask_percentages = [96, 97, 98, 99]    
    # Process each image in the input directory
    for image_name in os.listdir(args.input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(args.input_dir, image_name)
            
            # Apply masks with different percentages
            for percentage in mask_percentages:
                apply_mask(image_path, args.output_dir, percentage)

if __name__ == "__main__":
    main()
