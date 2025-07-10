# MAISI 3D Diffusion Model Workflow Guide

## Overview

This guide provides comprehensive instructions for using the MAISI 3D diffusion model for synthetic medical image generation. The workflow has been converted from Jupyter notebooks to production-ready Python scripts with command-line interfaces.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [MAISI Versions](#maisi-versions)
4. [Body Region Specification](#body-region-specification)
5. [Workflow Steps](#workflow-steps)
6. [Command Reference](#command-reference)
7. [Advanced Options](#advanced-options)
8. [Troubleshooting](#troubleshooting)
9. [File Structure](#file-structure)

## Prerequisites

- Python 3.8+
- CUDA-capable GPU(s)
- Medical imaging data in NIfTI format (`.nii.gz`)
- Sufficient disk space for training data and models

## Quick Start

### 1. Basic Setup and Training

```bash
# Setup data preparation and configuration
python setup_maisi.py --data_path /path/to/your/medical/images --work_dir ./my_maisi_work

# Follow the printed torchrun commands to train and generate images
```

### 2. Complete Workflow Example

```bash
# Step 1: Data preparation
python setup_maisi.py \
    --data_path /path/to/medical/images \
    --work_dir ./maisi_work \
    --maisi_version maisi3d-rflow \
    --epochs 100 \
    --num_gpus 2

# Step 2: Create training data (run the printed torchrun command)
torchrun --nproc_per_node=2 --nnodes=1 \
    -m scripts.diff_model_create_training_data \
    --env_config ./maisi_work/environment_maisi_diff_model.json \
    --model_config ./maisi_work/config_maisi_diff_model.json \
    --model_def ./maisi_work/config_maisi.json \
    --num_gpus 2

# Step 3: Create JSON metadata
python setup_maisi.py \
    --data_path /path/to/medical/images \
    --work_dir ./maisi_work \
    --create_json_only

# Step 4: Train the model
torchrun --nproc_per_node=2 --nnodes=1 \
    -m scripts.diff_model_train \
    --env_config ./maisi_work/environment_maisi_diff_model.json \
    --model_config ./maisi_work/config_maisi_diff_model.json \
    --model_def ./maisi_work/config_maisi.json \
    --num_gpus 2

# Step 5: Generate synthetic images
torchrun --nproc_per_node=2 --nnodes=1 \
    -m scripts.diff_model_infer \
    --env_config ./maisi_work/environment_maisi_diff_model.json \
    --model_config ./maisi_work/config_maisi_diff_model.json \
    --model_def ./maisi_work/config_maisi.json \
    --num_gpus 2
```

## MAISI Versions

### maisi3d-rflow (Recommended)
- **Advantages**: 33x faster inference, easier data preparation, better head region quality
- **Requirements**: No body region labeling required
- **Best for**: Most use cases, especially when speed is important

### maisi3d-ddpm (Traditional)
- **Advantages**: More control over anatomical regions
- **Requirements**: Body region specification required
- **Best for**: When precise anatomical control is needed

## Body Region Specification

### Available Body Regions (maisi3d-ddpm only)

| Region | Description | Use Case |
|--------|-------------|----------|
| `head_neck` | Head and neck region (brain, skull, cervical spine) | Neurological studies |
| `chest` | Chest region (lungs, heart, thoracic spine) | Pulmonary/cardiac studies |
| `abdomen` | Abdominal region (liver, kidneys, lumbar spine) | Abdominal imaging |
| `lower_body` | Lower body region (pelvis, hips, legs) | Orthopedic studies |
| `chest_abdomen` | Chest to abdomen (thorax and upper abdomen) | General body imaging |

### Body Region Encoding

Each region is encoded as one-hot vectors:
- `head_neck`: `[1, 0, 0, 0]`
- `chest`: `[0, 1, 0, 0]`
- `abdomen`: `[0, 0, 1, 0]`
- `lower_body`: `[0, 0, 0, 1]`
- `chest_abdomen`: `[0, 1, 0, 0]` (top) → `[0, 0, 1, 0]` (bottom)

### Example with Body Region

```bash
# Setup for chest imaging with DDPM
python setup_maisi.py \
    --data_path /path/to/chest/images \
    --maisi_version maisi3d-ddpm \
    --body_region chest \
    --work_dir ./chest_maisi_work
```

## Workflow Steps

### Step 1: Data Preparation (`setup_maisi.py`)

**Purpose**: Prepare your medical imaging data and create configuration files.

**Key Actions**:
- Copy medical images to working directory
- Create data list JSON
- Download pre-trained autoencoder
- Generate configuration files
- Set up directory structure

**Required Parameters**:
- `--data_path`: Path to your `.nii.gz` medical images
- `--work_dir`: Working directory for the project

### Step 2: Create Training Data (`diff_model_create_training_data`)

**Purpose**: Generate embeddings from your medical images using the autoencoder.

**Key Actions**:
- Load medical images
- Generate latent space embeddings
- Save embeddings for training

**GPU Memory**: High usage - uses autoencoder to process images

### Step 3: Create JSON Metadata (`--create_json_only`)

**Purpose**: Create metadata files for each embedding with spatial information.

**Key Actions**:
- Generate dimension and spacing metadata
- Add body region information (if using DDPM)
- Create `.json` files for each embedding

### Step 4: Train Diffusion Model (`diff_model_train`)

**Purpose**: Train the 3D diffusion model on the generated embeddings.

**Key Actions**:
- Load embeddings and metadata
- Train diffusion model
- Save model checkpoints

**Duration**: Longest step - depends on epochs and data size

### Step 5: Generate Synthetic Images (`diff_model_infer`)

**Purpose**: Generate new synthetic medical images using the trained model.

**Key Actions**:
- Load trained diffusion model
- Generate new latent codes
- Decode to medical images
- Save synthetic images

## Command Reference

### Setup Command

```bash
python setup_maisi.py [OPTIONS]
```

**Essential Options**:
- `--data_path PATH`: Path to medical images (required)
- `--work_dir PATH`: Working directory (default: ./maisi_work_dir)
- `--maisi_version {maisi3d-ddpm,maisi3d-rflow}`: MAISI version (default: maisi3d-rflow)

**Training Options**:
- `--epochs N`: Training epochs (default: 50)
- `--num_gpus N`: Number of GPUs (default: 1)

**Body Region Options** (DDPM only):
- `--body_region {head_neck,chest,abdomen,lower_body,chest_abdomen}`: Target region

**Advanced Options**:
- `--no_amp`: Disable automatic mixed precision
- `--num_splits N`: Autoencoder memory splits (default: 2)
- `--create_json_only`: Only create JSON metadata files

### Torchrun Commands

**Pattern**:
```bash
torchrun --nproc_per_node=N --nnodes=1 \
    -m scripts.SCRIPT_NAME \
    --env_config ENV_CONFIG.json \
    --model_config MODEL_CONFIG.json \
    --model_def MODEL_DEF.json \
    --num_gpus N [--no_amp]
```

**Scripts**:
- `diff_model_create_training_data`: Create embeddings
- `diff_model_train`: Train diffusion model
- `diff_model_infer`: Generate synthetic images

## Advanced Options

### GPU Configuration

**Single GPU**:
```bash
python setup_maisi.py --num_gpus 1 --data_path /path/to/data
```

**Multi-GPU**:
```bash
python setup_maisi.py --num_gpus 4 --data_path /path/to/data
```

### Memory Optimization

**For Limited GPU Memory**:
```bash
python setup_maisi.py \
    --num_splits 4 \
    --no_amp \
    --data_path /path/to/data
```

**For High-Memory GPUs (H100)**:
```bash
python setup_maisi.py \
    --num_splits 1 \
    --no_amp \
    --data_path /path/to/data
```

### Production Training

**Long Training Run**:
```bash
python setup_maisi.py \
    --epochs 200 \
    --num_gpus 8 \
    --data_path /path/to/large/dataset
```

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Solutions:
--num_splits 4    # Increase splits
--no_amp          # Disable mixed precision
# Use fewer GPUs or smaller batch sizes
```

**2. No Medical Images Found**
```bash
# Check:
ls /path/to/data/*.nii.gz
# Ensure files have .nii.gz extension
```

**3. JSON Creation Fails**
```bash
# Run after training data creation:
python setup_maisi.py --create_json_only --data_path /path --work_dir /work
```

**4. Body Region Errors (DDPM)**
```bash
# Valid regions only:
--body_region chest  # Valid
--body_region invalid_region  # Error
```

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Monitor GPU utilization** with `nvidia-smi`
3. **Start with fewer epochs** for testing
4. **Use multiple GPUs** for faster training
5. **Choose maisi3d-rflow** for faster inference

## File Structure

After running the setup, your working directory will contain:

```
maisi_work_dir/
├── dataroot/                    # Copied medical images
│   ├── image001.nii.gz
│   ├── image002.nii.gz
│   └── ...
├── embedding/                   # Generated embeddings
│   ├── image001.nii.gz
│   ├── image001.nii.gz.json
│   └── ...
├── models/                      # Model checkpoints
│   ├── autoencoder_epoch273.pt
│   └── diffusion_model_*.pt
├── output/                      # Generated images
│   ├── synthetic_001.nii.gz
│   └── ...
├── datalist.json               # Data list configuration
├── environment_maisi_diff_model.json  # Environment config
├── config_maisi_diff_model.json       # Model training config
└── config_maisi.json                  # Model definition
```

## Examples for Different Use Cases

### Neurological Imaging
```bash
python setup_maisi.py \
    --data_path /data/brain_scans \
    --maisi_version maisi3d-ddpm \
    --body_region head_neck \
    --epochs 150
```

### Chest Imaging
```bash
python setup_maisi.py \
    --data_path /data/chest_ct \
    --maisi_version maisi3d-rflow \
    --epochs 100
```

### Multi-Region Body Imaging
```bash
python setup_maisi.py \
    --data_path /data/body_scans \
    --maisi_version maisi3d-ddpm \
    --body_region chest_abdomen \
    --epochs 200 \
    --num_gpus 4
```

## Best Practices

1. **Start Small**: Test with a small dataset and few epochs first
2. **Monitor Training**: Check GPU utilization and training progress
3. **Version Control**: Keep track of configurations and model versions
4. **Backup Models**: Save important model checkpoints
5. **Document Settings**: Record successful configurations for reproduction

## Support and Resources

- **Configuration Files**: Located in `./configs/` directory
- **Scripts**: Located in `./scripts/` directory
- **Logs**: Check terminal output for detailed progress
- **Model Weights**: Automatically downloaded during setup

This workflow provides a robust foundation for synthetic medical image generation using the MAISI 3D diffusion model. Adjust parameters based on your specific requirements and computational resources.
