# MAISI 3D Diffusion Model - Parallel Computing Guide

## Overview

The MAISI (Medical AI for Synthetic Imaging) parallel training script enables distributed training and inference of 3D diffusion models for generating medical images across multiple GPUs and compute nodes. This version is optimized for research clusters and high-performance computing environments.

## Features

- **Multi-GPU Support**: Leverage multiple GPUs on a single node
- **Multi-Node Clustering**: Scale across multiple compute nodes 
- **Parallel Image Generation**: Generate one unique medical image per GPU rank simultaneously
- **Distributed Training**: Efficient distributed training across all available GPUs
- **Flexible Configuration**: Command-line interface for easy parameter adjustment
- **Real Data Support**: Works with real medical imaging datasets (.nii.gz files)

## Files

- `maisi_train_diff_unet_parallel.py` - Main parallel training script
- `maisi_train_diff_unet.py` - Single-user version (for comparison)
- `configs/` - Configuration files for different model versions
- `scripts/` - Core training and inference modules

## Quick Start

### Single Node (4 GPUs)
```bash
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py --real-data --data-path /path/to/medical/data
```

### Multi-Node Cluster
**Master Node (Node 0):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=12355 \
    maisi_train_diff_unet_parallel.py --num-nodes 2 --node-rank 0 --master-addr 192.168.1.100 \
    --real-data --data-path /shared/medical/data
```

**Worker Node (Node 1):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=12355 \
    maisi_train_diff_unet_parallel.py --num-nodes 2 --node-rank 1 --master-addr 192.168.1.100 \
    --real-data --data-path /shared/medical/data
```

## Command Line Arguments

### Training Configuration
| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--epochs` | `-e` | int | Auto | Number of training epochs (2 for simulated, 50 for real data) |
| `--batch-size` | `-b` | int | 1 | Training batch size per GPU |

### Data Configuration
| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--real-data` | | flag | False | Use real medical imaging data instead of simulated |
| `--data-path` | `-d` | str | `/path/to/your/medical/imaging/data` | Path to directory containing .nii.gz files |

### Parallel Computing Configuration
| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--num-gpus` | `-g` | int | 4 | Number of GPUs per node |
| `--num-nodes` | `-n` | int | 1 | Number of compute nodes in cluster |
| `--node-rank` | | int | 0 | Rank of current node (0-based) |
| `--master-addr` | | str | `localhost` | Address of master node |
| `--master-port` | | str | `12355` | Port of master node |

### Model Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model-version` | str | `maisi3d-rflow` | MAISI version (`maisi3d-rflow` or `maisi3d-ddpm`) |
| `--base-seed` | int | 42 | Base random seed (each GPU uses base_seed + rank) |

## Usage Examples

### Development and Testing
```bash
# Quick test with simulated data (2 epochs, 4 GPUs)
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py

# Test with custom epochs
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py --epochs 5
```

### Production Training
```bash
# Real data training with 8 GPUs
torchrun --nproc_per_node=8 maisi_train_diff_unet_parallel.py \
    --num-gpus 8 \
    --epochs 100 \
    --batch-size 2 \
    --real-data \
    --data-path /data/medical_scans \
    --model-version maisi3d-rflow

# High-performance setup with custom seed
torchrun --nproc_per_node=8 maisi_train_diff_unet_parallel.py \
    --num-gpus 8 \
    --epochs 200 \
    --real-data \
    --data-path /shared/ct_scans \
    --base-seed 123 \
    --batch-size 1
```

### Multi-Node Cluster Examples

**2-Node Cluster (8 GPUs total):**

Node 0:
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
    maisi_train_diff_unet_parallel.py \
    --num-nodes 2 --node-rank 0 --master-addr 10.0.0.1 --master-port 29500 \
    --num-gpus 4 --epochs 100 --real-data --data-path /shared/medical_data
```

Node 1:
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 \
    maisi_train_diff_unet_parallel.py \
    --num-nodes 2 --node-rank 1 --master-addr 10.0.0.1 --master-port 29500 \
    --num-gpus 4 --epochs 100 --real-data --data-path /shared/medical_data
```

**4-Node Cluster (16 GPUs total):**

Node 0 (Master):
```bash
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr=master.cluster --master_port=29500 \
    maisi_train_diff_unet_parallel.py \
    --num-nodes 4 --node-rank 0 --master-addr master.cluster --master-port 29500 \
    --num-gpus 4 --epochs 150 --real-data --data-path /nfs/medical_imaging
```

Nodes 1-3 (Workers):
```bash
# Node 1
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=1 --master_addr=master.cluster --master_port=29500 \
    maisi_train_diff_unet_parallel.py \
    --num-nodes 4 --node-rank 1 --master-addr master.cluster --master-port 29500 \
    --num-gpus 4 --epochs 150 --real-data --data-path /nfs/medical_imaging

# Node 2 (change --node_rank to 2)
# Node 3 (change --node_rank to 3)
```

## Output Structure

The script generates the following outputs:

```
output_work_dir/
├── dataroot/                          # Training data
│   ├── tr_image_001.nii.gz
│   └── tr_image_002.nii.gz
├── embedding/                         # Latent embeddings
│   ├── tr_image_001.nii.gz
│   ├── tr_image_001.nii.gz.json
│   └── ...
├── models/                            # Trained model checkpoints
│   ├── autoencoder_epoch273.pt
│   ├── diffusion_unet_checkpoint.pt
│   └── ...
├── output/                            # Generated images
│   ├── image_seed42_rank0.nii.gz     # GPU 0 output
│   ├── image_seed43_rank1.nii.gz     # GPU 1 output
│   ├── image_seed44_rank2.nii.gz     # GPU 2 output
│   └── image_seed45_rank3.nii.gz     # GPU 3 output
└── config files...
```

### Generated Image Naming Convention
- Format: `image_seed{SEED}_rank{RANK}.nii.gz`
- Each GPU generates one unique image with a different seed
- Rank corresponds to the GPU's position in the cluster
- Example with 4 GPUs and base_seed=42:
  - GPU 0: `image_seed42_rank0.nii.gz`
  - GPU 1: `image_seed43_rank1.nii.gz`
  - GPU 2: `image_seed44_rank2.nii.gz`
  - GPU 3: `image_seed45_rank3.nii.gz`

## System Requirements

### Hardware
- **GPUs**: NVIDIA GPUs with 8GB+ VRAM recommended
- **Memory**: 16GB+ RAM per node
- **Storage**: Fast SSD storage for medical imaging data
- **Network**: High-speed interconnect for multi-node setups (InfiniBand recommended)

### Software
- **CUDA**: 11.0 or later
- **Python**: 3.8+
- **PyTorch**: Latest version with CUDA support
- **MONAI**: Latest weekly build
- **torchrun**: Included with PyTorch

### Recommended GPU Configurations
| GPU Model | VRAM | Recommended Batch Size | Notes |
|-----------|------|----------------------|-------|
| RTX 3090 | 24GB | 1-2 | Good for development |
| RTX 4090 | 24GB | 1-2 | Excellent performance |
| A100 | 40/80GB | 2-4 | Best for production |
| H100 | 80GB | 4-8 | Highest performance |
| V100 | 16/32GB | 1 | Budget option |

## Performance Expectations

### Training Time Estimates (per epoch)
- **Single GPU**: ~30 minutes per epoch (with real data)
- **4 GPUs**: ~8-10 minutes per epoch
- **8 GPUs**: ~4-6 minutes per epoch
- **16 GPUs**: ~2-4 minutes per epoch

### Scaling Efficiency
- **Single Node**: ~90% efficiency up to 8 GPUs
- **Multi-Node**: ~75-85% efficiency (depends on network)
- **Image Generation**: Linear scaling (N GPUs = N images simultaneously)

## Data Preparation

### Real Medical Data
1. **Format**: NIfTI files (.nii.gz)
2. **Structure**: Place all .nii.gz files in a single directory
3. **Naming**: Any naming convention is supported
4. **Size**: Typical medical images (e.g., 512x512x128 voxels)

### Example Data Directory
```
/data/medical_scans/
├── patient_001_ct.nii.gz
├── patient_002_ct.nii.gz
├── patient_003_mri.nii.gz
└── ...
```

### Data Quality Requirements
- **Intensity normalization**: Consistent intensity ranges
- **Spatial alignment**: Similar anatomical orientations
- **Resolution**: Consistent voxel spacing preferred
- **Dataset size**: 100+ images for meaningful training

## Cluster Setup Guide

### Network Configuration
1. **Ensure all nodes can communicate**:
   ```bash
   # Test connectivity from each worker to master
   ping master_node_ip
   telnet master_node_ip 29500
   ```

2. **Configure SSH key-based authentication** (for ease of deployment)

3. **Set up shared storage** (NFS, Lustre, or similar):
   ```bash
   # Mount shared storage on all nodes
   sudo mount -t nfs master:/shared/data /data
   ```

### Environment Setup
1. **Install identical environments on all nodes**:
   ```bash
   # Install CUDA, PyTorch, MONAI on each node
   pip install torch torchvision monai-weekly[pillow,tqdm]
   ```

2. **Synchronize code across nodes**:
   ```bash
   # Copy scripts to all nodes or use shared filesystem
   rsync -av maisi_scripts/ node1:/path/to/maisi/
   ```

### Firewall Configuration
```bash
# Open required ports on all nodes
sudo ufw allow 29500  # Default master port
sudo ufw allow 29501  # Alternative port
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch size to 1
--batch-size 1
```

**2. Connection Timeout in Multi-Node**
```
Check: Network connectivity, firewall settings, master address
Verify: ping master_addr && telnet master_addr master_port
```

**3. Training Data Creation Fails**
```
Check: Data path exists, .nii.gz files present, file permissions
Verify: ls /path/to/data/*.nii.gz
```

**4. Model Loading Errors**
```
Solution: Ensure autoencoder weights download correctly
Check: Internet connection, storage space
```

**5. Poor Image Quality**
```
Causes: Too few epochs, insufficient training data, simulated data
Solutions: Increase epochs (100+), use more real data, longer training
```

### Performance Optimization

**1. Maximize GPU Utilization**
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Optimal settings for A100 80GB
--batch-size 4 --num-gpus 8
```

**2. Network Optimization**
```bash
# For InfiniBand networks
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# For Ethernet networks
export NCCL_SOCKET_IFNAME=eth0
```

**3. Storage Optimization**
```bash
# Use local SSD for temporary files
export TMPDIR=/local/ssd/tmp

# Preload data to memory if possible
```

## Model Versions

### maisi3d-rflow (Recommended)
- **Speed**: 33x faster inference than DDPM
- **Quality**: Better for head region and small volumes
- **Training**: Easier data preparation (no body region requirements)
- **Use case**: Production environments, fast inference

### maisi3d-ddpm (Legacy)
- **Speed**: Standard DDPM scheduler (1000 steps)
- **Quality**: Comparable quality for most cases
- **Training**: Requires body region labeling
- **Use case**: Research, when specific DDPM behavior is needed

## Advanced Usage

### Custom Configurations
```bash
# Create custom model configuration
cp configs/config_maisi_diff_model.json configs/my_config.json
# Edit my_config.json as needed

# Use custom configuration
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py \
    --model-config configs/my_config.json
```

### Checkpoint Management
```bash
# Resume from checkpoint
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py \
    --resume-from-checkpoint output_work_dir/models/checkpoint_epoch_50.pt
```

### Monitoring and Logging
```bash
# Enhanced logging
torchrun --nproc_per_node=4 maisi_train_diff_unet_parallel.py \
    --log-level DEBUG --log-file training.log
```

## Citation

If you use this parallel MAISI implementation in your research, please cite:

```bibtex
@article{maisi2024,
  title={MAISI: Medical AI for Synthetic Imaging},
  author={MONAI Consortium},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

- **Documentation**: [MONAI Documentation](https://docs.monai.io/)
- **Issues**: [GitHub Issues](https://github.com/Project-MONAI/MONAI)
- **Community**: [MONAI Discord](https://discord.gg/monai)

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
