#!/usr/bin/env python
"""
Preprocess medical images for MAISI training.
Standardizes spacing, dimensions, and intensity ranges.
"""

import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import argparse

def preprocess_image(input_file, output_file, target_spacing=(3.0, 0.5, 0.5), target_shape=(24, 512, 512)):
    """
    Preprocess a single medical image to standard spacing and dimensions.
    
    Args:
        input_file: Path to input .nii.gz file
        output_file: Path to output .nii.gz file  
        target_spacing: Target voxel spacing (z, y, x) in mm
        target_shape: Target image dimensions (z, y, x)
    """
    print(f"Processing: {os.path.basename(input_file)}")
    
    # Load image
    img = nib.load(input_file)
    data = img.get_fdata()
    current_spacing = img.header.get_zooms()[:3]
    
    print(f"  Original: shape={data.shape}, spacing={current_spacing}")
    
    # Calculate zoom factors for resampling
    zoom_factors = [
        current_spacing[i] / target_spacing[i] 
        for i in range(3)
    ]
    
    # Resample to target spacing
    resampled_data = zoom(data, zoom_factors, order=1, mode='constant', cval=0)
    print(f"  Resampled: shape={resampled_data.shape}")
    
    # Crop or pad to target shape
    final_data = resize_to_target_shape(resampled_data, target_shape)
    print(f"  Final: shape={final_data.shape}")
    
    # Normalize intensity to [0, 1] range
    final_data = normalize_intensity(final_data)
    
    # Create new NIfTI image with target spacing
    new_affine = img.affine.copy()
    # Update spacing in affine matrix
    for i in range(3):
        new_affine[i, i] = target_spacing[i] if i == 0 else target_spacing[i]
    
    new_img = nib.Nifti1Image(final_data, affine=new_affine, header=img.header)
    
    # Update header spacing
    new_img.header.set_zooms(target_spacing)
    
    # Save preprocessed image
    nib.save(new_img, output_file)
    print(f"  Saved: {output_file}")

def resize_to_target_shape(data, target_shape):
    """Crop or pad image to target shape."""
    current_shape = data.shape
    
    # Calculate padding/cropping for each dimension
    result = data.copy()
    
    for i in range(3):
        current_size = current_shape[i]
        target_size = target_shape[i]
        
        if current_size > target_size:
            # Crop (take center)
            start = (current_size - target_size) // 2
            end = start + target_size
            if i == 0:
                result = result[start:end, :, :]
            elif i == 1:
                result = result[:, start:end, :]
            else:
                result = result[:, :, start:end]
        elif current_size < target_size:
            # Pad with zeros
            pad_before = (target_size - current_size) // 2
            pad_after = target_size - current_size - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[i] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode='constant', constant_values=0)
    
    return result

def normalize_intensity(data):
    """Normalize intensity to [0, 1] range."""
    # Remove extreme outliers (99.5th percentile)
    p99_5 = np.percentile(data[data > 0], 99.5)
    data = np.clip(data, 0, p99_5)
    
    # Normalize to [0, 1]
    if p99_5 > 0:
        data = data / p99_5
    
    return data.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Preprocess medical images for MAISI")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .nii.gz files")
    parser.add_argument("--output_dir", required=True, help="Directory for preprocessed files")
    parser.add_argument("--target_spacing", nargs=3, type=float, default=[3.0, 0.5, 0.5],
                       help="Target voxel spacing (z y x) in mm")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[24, 512, 512],
                       help="Target image dimensions (z y x)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .nii.gz files
    input_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    
    if not input_files:
        print(f"No .nii.gz files found in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    print(f"Target spacing: {args.target_spacing}")
    print(f"Target shape: {args.target_shape}")
    print()
    
    # Process each file
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, filename)
        
        try:
            preprocess_image(
                input_file, 
                output_file, 
                target_spacing=args.target_spacing,
                target_shape=args.target_shape
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nPreprocessing complete! Processed files saved to: {args.output_dir}")
    print("\nNext steps:")
    print(f"1. Verify preprocessed data quality")
    print(f"2. Re-run MAISI setup with preprocessed data:")
    print(f"   python setup_maisi.py --data_path {args.output_dir} --epochs 100")

if __name__ == "__main__":
    main()
