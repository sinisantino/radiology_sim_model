# MAISI 3D Medical Image Generation Setup Guide

## Overview

This guide provides step-by-step instructions for setting up and running the MAISI 3D diffusion model for synthetic medical image generation. The setup script (`setup_maisi.py`) prepares your data, configurations, and provides the exact commands needed to train and run inference.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- Medical imaging data in NIfTI format (.nii.gz files)
- Sufficient disk space for training data and model checkpoints

## Quick Start

### 1. Basic Setup Command

```bash
python setup_maisi.py --data_path /path/to/your/medical/images
```

### 2. Advanced Setup with Options

```bash
python setup_maisi.py \
    --data_path /path/to/your/medical/images \
    --maisi_version maisi3d-rflow \
    --num_gpus 4 \
    --epochs 100 \
    --num_images 1 \
    --work_dir ./my_maisi_project \
    --body_region chest_abdomen
```

## Command-Line Arguments

### Required Arguments

- `--data_path`: Path to directory containing medical imaging data (.nii.gz files)

### Model Configuration

- `--maisi_version`: Choose between `maisi3d-ddpm` or `maisi3d-rflow` (default: maisi3d-rflow)
  - `maisi3d-rflow`: 33x faster inference, no body region requirement
  - `maisi3d-ddpm`: Requires body region specification, original version

### Training Parameters

- `--num_gpus`: Number of GPUs to use (default: 1)
- `--epochs`: Number of training epochs (default: 50, recommend 100+ for production)
- `--work_dir`: Working directory for outputs (default: ./maisi_work_dir)

### Inference Parameters

- `--num_images`: Number of synthetic images to generate during inference (default: 1)
  - **Start with 1**: Medical images are complex and memory-intensive
  - **Scale carefully**: Each image can take significant GPU memory and time
  - **Typical usage**: Generate 1-5 images per run for evaluation

### Body Region Control (maisi3d-ddpm only)

- `--body_region`: Target anatomical region (default: chest_abdomen)
  - `head_neck`: Head and neck region (brain, skull, cervical spine)
  - `chest`: Chest region (lungs, heart, thoracic spine)
  - `abdomen`: Abdominal region (liver, kidneys, lumbar spine)
  - `lower_body`: Lower body region (pelvis, hips, legs)
  - `chest_abdomen`: Chest to abdomen (thorax and upper abdomen)

### Advanced Options

- `--no_amp`: Disable automatic mixed precision (useful for H100 GPUs)
- `--num_splits`: Number of splits for autoencoder (default: 2, reduces GPU memory)
- `--create_json_only`: Only create JSON metadata files (run after training data creation)

## Complete Workflow

### Phase 0: Data Preprocessing (CRITICAL)

**⚠️ IMPORTANT**: Medical images must have consistent spacing and dimensions for successful training. If your images have different shapes or voxel spacing, you MUST preprocess them first.

#### Check Your Data Consistency

First, verify if your data needs preprocessing:

```bash
# Check consistency of your medical images
python -c "
import nibabel as nib
import glob
files = glob.glob('./your_data_path/*.nii.gz')
for f in files[:5]:
    img = nib.load(f)
    print(f'{f}: shape={img.shape}, spacing={img.header.get_zooms()[:3]}')
"
```

#### When Preprocessing is Required

Preprocess if you see:
- **Different image dimensions** (e.g., some 512x512, others 320x320)
- **Varying voxel spacing** (e.g., spacing from 0.35mm to 0.625mm)
- **Different slice counts** (e.g., some 23 slices, others 26 slices)

#### Run Data Preprocessing

```bash
# Install required packages
pip install scipy

# Preprocess your data to consistent spacing and dimensions
python preprocess_data.py \
    --input_dir ./your_original_data \
    --output_dir ./preprocessed_data \
    --target_spacing 3.0 0.5 0.5 \
    --target_shape 24 512 512
```

**Preprocessing Parameters:**
- `--target_spacing`: Voxel spacing in mm (z, y, x) - adjust for your anatomy
- `--target_shape`: Image dimensions (z, y, x) - ensure adequate resolution

**Recommended Settings by Anatomy:**
- **Prostate**: `--target_spacing 3.0 0.5 0.5 --target_shape 24 512 512`
- **Brain**: `--target_spacing 1.0 1.0 1.0 --target_shape 128 256 256`
- **Chest**: `--target_spacing 2.0 0.7 0.7 --target_shape 64 512 512`

#### Verify Preprocessing Results

```bash
# Confirm all images now have consistent properties
python -c "
import nibabel as nib
import glob
files = glob.glob('./preprocessed_data/*.nii.gz')
for f in files[:5]:
    img = nib.load(f)
    print(f'{f}: shape={img.shape}, spacing={img.header.get_zooms()[:3]}')
"
```

All images should now have **identical** shapes and spacing.

### Phase 1: Data Preparation

1. **Run the setup script with preprocessed data:**
   ```bash
   python setup_maisi.py --data_path ./preprocessed_data --num_gpus 4 --epochs 100
   ```

2. **What this does:**
   - Copies your medical images to the working directory
   - Downloads the pre-trained autoencoder model
   - Creates configuration files for training
   - Displays the exact torchrun commands to execute

### Phase 2: Training Pipeline

After setup completes, run these commands in sequence:

#### Step 1: Create Training Data
```bash
torchrun --nproc_per_node=4 --nnodes=1 \
    -m scripts.diff_model_create_training_data \
    --env_config ./maisi_work_dir/environment_maisi_diff_model.json \
    --model_config ./maisi_work_dir/config_maisi_diff_model.json \
    --model_def ./maisi_work_dir/config_maisi.json \
    --num_gpus 4
```

#### Step 2: Create JSON Metadata (After Step 1 completes)

**For maisi3d-rflow (default - no body region needed):**
```bash
python setup_maisi.py --data_path /path/to/medical/images --work_dir ./maisi_work_dir --create_json_only
```

**For maisi3d-ddpm (body region required):**
```bash
python setup_maisi.py --data_path /path/to/medical/images --work_dir ./maisi_work_dir --create_json_only --body_region chest_abdomen
```

**Important Notes:**
- **maisi3d-rflow**: Body region specification is NOT required
- **maisi3d-ddpm**: Body region specification IS required (`--body_region` flag)
- Use the same `--body_region` value you used in the initial setup (if using maisi3d-ddpm)

#### Step 3: Train the Model
**Do not train parallelly. Use only 1 GPU due to known Maisi error**
```bash
torchrun --nproc_per_node=1 --nnodes=1 \
    -m scripts.diff_model_train \
    --env_config ./maisi_work_dir/environment_maisi_diff_model.json \
    --model_config ./maisi_work_dir/config_maisi_diff_model.json \
    --model_def ./maisi_work_dir/config_maisi.json \
    --num_gpus 1
```

#### Step 4: Run Inference
```bash
torchrun --nproc_per_node=4 --nnodes=1 \
    -m scripts.diff_model_infer \
    --env_config ./maisi_work_dir/environment_maisi_diff_model.json \
    --model_config ./maisi_work_dir/config_maisi_diff_model.json \
    --model_def ./maisi_work_dir/config_maisi.json \
    --num_gpus 4
```

## Understanding the Output

After setup completes, you'll find:

### Directory Structure
```
maisi_work_dir/
├── dataroot/                    # Your medical images
├── embedding/                   # Training embeddings (created in Step 1)
├── models/                      # Model checkpoints
├── outputs/                     # Generated images
├── datalist.json               # Data list configuration
├── environment_maisi_diff_model.json  # Environment config
├── config_maisi_diff_model.json       # Model training config
└── config_maisi.json                  # Model definition
```

### Key Files
- **Training data**: `embedding/` directory contains processed embeddings
- **Model checkpoints**: `models/` directory contains saved model weights
- **Generated images**: `outputs/` directory contains inference results
- **Configuration files**: JSON files control all training parameters

## Version Differences

### maisi3d-rflow (Recommended)
- ✅ 33x faster inference
- ✅ No body region requirement
- ✅ Better quality for head region and small volumes
- ✅ Easier data preparation

### maisi3d-ddpm (Original)
- ⚠️ Requires body region specification
- ⚠️ Slower inference (DDPM scheduler)
- ✅ Proven performance for large volumes

## Tips and Best Practices

### For Production Use
- **ALWAYS preprocess data first** to ensure consistency
- Use 100+ epochs for real medical data
- Consider multiple GPUs for faster training
- Monitor GPU memory usage and adjust `num_splits` if needed

### For Development/Testing
- **Check data consistency before training**
- Start with 50 epochs and smaller datasets
- Use `maisi3d-rflow` for faster iteration
- Test with single GPU first

### Data Quality Requirements
- **Consistent voxel spacing** across all images
- **Identical image dimensions** for all cases
- **Appropriate resolution** for target anatomy
- **Normalized intensity ranges** (handled by preprocessing)

### Memory Management
- Use `--no_amp` for H100 GPUs if encountering issues
- Increase `--num_splits` to reduce GPU memory usage
- Monitor disk space during training data creation

### Troubleshooting

#### Common Issues

**1. AttributeError: 'DistributedDataParallel' object has no attribute 'include_top_region_index_input'**

This error occurs with multi-GPU training. **Solution**: Use single GPU for training:

```bash
# Instead of --nproc_per_node=2, use --nproc_per_node=1
torchrun --nproc_per_node=1 --nnodes=1 \
    -m scripts.diff_model_train \
    --env_config ./maisi_work_dir/environment_maisi_diff_model.json \
    --model_config ./maisi_work_dir/config_maisi_diff_model.json \
    --model_def ./maisi_work_dir/config_maisi.json \
    --num_gpus 1
```

**2. Missing embedding files**
- Ensure Step 1 (create training data) completed successfully
- Check that `embeddings/` directory contains `.nii.gz` files

**3. Missing JSON metadata files**
- Run Step 2 (create JSON metadata) after Step 1 completes
- Verify `.json` files exist alongside `.nii.gz` files in `embeddings/`

**4. Memory Issues**
- Ensure all .nii.gz files are valid medical images
- Check that CUDA drivers are compatible
- Verify sufficient disk space in working directory
- Use `--no_amp` for H100 GPUs if encountering issues
- Increase `--num_splits` to reduce GPU memory usage
- Monitor disk space during training data creation

**5. Poor Training Performance (High Loss)**

If your loss remains high (>0.5) after 25+ epochs:

```bash
# Check data consistency - this is the most common cause
python -c "
import nibabel as nib
import glob
files = glob.glob('./your_data/*.nii.gz')
shapes = set()
spacings = set()
for f in files:
    img = nib.load(f)
    shapes.add(img.shape)
    spacings.add(tuple(round(x, 3) for x in img.header.get_zooms()[:3]))
print(f'Unique shapes: {shapes}')
print(f'Unique spacings: {spacings}')
if len(shapes) > 1 or len(spacings) > 1:
    print('❌ INCONSISTENT DATA - Preprocessing required!')
else:
    print('✅ Data is consistent')
"
```

**Solution**: If data is inconsistent, use `preprocess_data.py` to standardize before training.

## Example Use Cases

### Research Setup (Small Dataset)
```bash
# First preprocess data
python preprocess_data.py \
    --input_dir ./research_data \
    --output_dir ./research_data_preprocessed \
    --target_spacing 2.0 0.7 0.7 \
    --target_shape 32 256 256

# Then run MAISI setup
python setup_maisi.py \
    --data_path ./research_data_preprocessed \
    --maisi_version maisi3d-rflow \
    --epochs 50 \
    --num_gpus 1
```

### Production Setup (Large Dataset)
```bash
# First preprocess data
python preprocess_data.py \
    --input_dir /data/medical_images \
    --output_dir /data/medical_images_preprocessed \
    --target_spacing 1.5 0.5 0.5 \
    --target_shape 48 512 512

# Then run MAISI setup
python setup_maisi.py \
    --data_path /data/medical_images_preprocessed \
    --maisi_version maisi3d-rflow \
    --epochs 200 \
    --num_gpus 8 \
    --work_dir /workspace/maisi_production
```

### Specific Anatomy (Head/Neck)
```bash
# Preprocess with brain-appropriate settings
python preprocess_data.py \
    --input_dir ./head_neck_data \
    --output_dir ./head_neck_preprocessed \
    --target_spacing 1.0 1.0 1.0 \
    --target_shape 128 256 256

# Then run MAISI setup
python setup_maisi.py \
    --data_path ./head_neck_preprocessed \
    --maisi_version maisi3d-ddpm \
    --body_region head_neck \
    --epochs 100 \
    --num_gpus 4
```

## Support and Resources

- **Configuration files**: All JSON configs are automatically generated
- **Model weights**: Pre-trained autoencoder downloaded automatically
- **Documentation**: See individual script help with `--help` flag
- **Logs**: Training progress logged to console and files

## Next Steps

1. Run the setup script with your data
2. Follow the printed torchrun commands exactly
3. Monitor training progress and adjust parameters as needed
4. Use generated images for your research or application

For advanced cluster deployment, see `README_PARALLEL.md` and `MAISI_WORKFLOW_GUIDE.md`.
