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

### Phase 1: Data Preparation

1. **Run the setup script:**
   ```bash
   python setup_maisi.py --data_path /path/to/medical/images --num_gpus 4 --epochs 100
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
```bash
python setup_maisi.py --data_path /path/to/medical/images --work_dir ./maisi_work_dir --create_json_only
```

#### Step 3: Train the Model
```bash
torchrun --nproc_per_node=4 --nnodes=1 \
    -m scripts.diff_model_train \
    --env_config ./maisi_work_dir/environment_maisi_diff_model.json \
    --model_config ./maisi_work_dir/config_maisi_diff_model.json \
    --model_def ./maisi_work_dir/config_maisi.json \
    --num_gpus 4
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
- Use 100+ epochs for real medical data
- Consider multiple GPUs for faster training
- Monitor GPU memory usage and adjust `num_splits` if needed

### For Development/Testing
- Start with 50 epochs and smaller datasets
- Use `maisi3d-rflow` for faster iteration
- Test with single GPU first

### Memory Management
- Use `--no_amp` for H100 GPUs if encountering issues
- Increase `--num_splits` to reduce GPU memory usage
- Monitor disk space during training data creation

### Troubleshooting
- Ensure all .nii.gz files are valid medical images
- Check that CUDA drivers are compatible
- Verify sufficient disk space in working directory

## Example Use Cases

### Research Setup (Small Dataset)
```bash
python setup_maisi.py \
    --data_path ./research_data \
    --maisi_version maisi3d-rflow \
    --epochs 50 \
    --num_gpus 1
```

### Production Setup (Large Dataset)
```bash
python setup_maisi.py \
    --data_path /data/medical_images \
    --maisi_version maisi3d-rflow \
    --epochs 200 \
    --num_gpus 8 \
    --work_dir /workspace/maisi_production
```

### Specific Anatomy (Head/Neck)
```bash
python setup_maisi.py \
    --data_path ./head_neck_data \
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
