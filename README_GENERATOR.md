# MAISI Medical Image Generator

This script generates synthetic medical images for various anatomical structures using the MAISI (Medical AI for Synthetic Imaging) model.

## Quick Start

### 1. List all available options:
```bash
python generate_medical_images.py --list-options
```

### 2. Generate prostate images (default):
```bash
python generate_medical_images.py
```

### 3. Generate other anatomical structures:
```bash
# Generate liver images
python generate_medical_images.py --anatomy liver

# Generate brain images  
python generate_medical_images.py --anatomy brain --body_region head

# Generate heart images with custom settings
python generate_medical_images.py --anatomy heart --body_region chest --num_samples 2 --output_size 512 512 256
```

## Available Anatomical Structures

The script supports over 100 anatomical structures including:
- **Organs**: liver, spleen, pancreas, kidneys, heart, brain, prostate, bladder
- **Vessels**: aorta, portal vein, pulmonary vein, carotid arteries
- **Bones**: vertebrae, ribs, skull, femur, humerus
- **Muscles**: gluteus muscles, iliopsoas
- **Tumors**: lung tumor, pancreatic tumor, hepatic tumor
- **And many more...**

Use `--list-options` to see the complete list organized by body region.

## Command Line Options

```
--anatomy ANATOMY         Anatomy to generate (default: prostate)
--body_region REGION       Body region (auto-detected if not specified)
--num_samples N            Number of samples to generate (default: 1)
--output_size W H D        Output size in voxels (default: 256 256 256)
--spacing X Y Z            Voxel spacing in mm (default: 1.5 1.5 2.0)
--output_dir DIR           Output directory (default: output)
--maisi_version VERSION    MAISI version: maisi3d-rflow (fast) or maisi3d-ddpm (default: maisi3d-rflow)
--list-options             Show all available options and exit
```

## Output

The script generates two files per sample:
- **Image file**: `synthetic_image_XXX.nii.gz` - The generated CT image
- **Mask file**: `synthetic_mask_XXX.nii.gz` - The segmentation mask

Files are saved in NIfTI format (.nii.gz) which can be viewed with medical imaging software like:
- 3D Slicer
- ITK-SNAP
- ImageJ with NIfTI plugin
- ParaView

## System Requirements

- **GPU**: CUDA-compatible GPU recommended (CPU mode available but slower)
- **Memory**: 16GB+ RAM, 15GB+ GPU memory for 256³ images
- **Storage**: ~10GB for model weights and generated data
- **Python**: 3.8+

## Examples by Body Region

### Head Region
```bash
python generate_medical_images.py --anatomy brain --body_region head
python generate_medical_images.py --anatomy skull --body_region head
```

### Chest Region  
```bash
python generate_medical_images.py --anatomy heart --body_region chest
python generate_medical_images.py --anatomy "lung tumor" --body_region chest
```

### Abdomen Region
```bash
python generate_medical_images.py --anatomy liver --body_region abdomen
python generate_medical_images.py --anatomy pancreas --body_region abdomen
python generate_medical_images.py --anatomy "right kidney" --body_region abdomen
```

### Pelvis Region
```bash
python generate_medical_images.py --anatomy prostate --body_region pelvis
python generate_medical_images.py --anatomy bladder --body_region pelvis
```

## Performance Notes

- **maisi3d-rflow**: Faster inference (33x speedup), better for small volumes
- **maisi3d-ddpm**: More traditional diffusion, comparable quality for large volumes
- Generation time: ~3-8 seconds for 256³ images on A100 GPU
- Larger images (512³) take proportionally longer

## Troubleshooting

If you encounter package import errors, the script will automatically install required packages:
- monai-weekly[nibabel,tqdm]
- matplotlib
- torch

Make sure you have sufficient disk space for model downloads (~10GB total).
