# MAISI Parallel Training - Python Scripts

This directory contains Python scripts for automated MAISI 3D diffusion model parallel training setup and execution.

## 📁 Files Overview

- **`maisi_parallel_setup_and_run.py`** - Main comprehensive script (replaces the notebook)
- **`quick_launch_maisi.py`** - Simple launcher for quick 2-GPU testing
- **`maisi_train_diff_unet_parallel.py`** - Original MAISI parallel training script
- **`maisi_train_diff_unet.py`** - Single-GPU training script

## 🚀 Quick Start (Recommended)

For your 2-GPU setup, the fastest way to get started:

```bash
# Quick test with simulated data
python quick_launch_maisi.py
```

This will:
- ✅ Install dependencies automatically
- ✅ Detect your 2 GPUs
- ✅ Create test medical data
- ✅ Run 5-epoch training on both GPUs
- ✅ Generate 2 unique medical images
- ✅ Handle port conflicts and GPU mismatches

## 🛠️ Advanced Usage

### Use Real Medical Data

```bash
python maisi_parallel_setup_and_run.py --data-path /path/to/your/medical/data
```

### Customize Training Parameters

```bash
python maisi_parallel_setup_and_run.py \
    --data-path /your/data \
    --epochs 25 \
    --batch-size 2 \
    --nproc-per-node 2
```

### Production Training (Long Run)

```bash
python maisi_parallel_setup_and_run.py \
    --data-path /your/medical/data \
    --epochs 100 \
    --batch-size 2 \
    --nproc-per-node 2 \
    --model-version maisi3d
```

### Skip Dependency Installation

```bash
python maisi_parallel_setup_and_run.py --skip-install
```

## 📋 Command Line Options

### Data Configuration
- `--data-path PATH` - Path to medical imaging data directory
- `--use-simulated-data` - Create and use test data
- `--num-test-images N` - Number of test images to create (default: 3)

### Training Parameters
- `--epochs N` - Number of training epochs (default: 5)
- `--batch-size N` - Batch size per GPU (default: 1)
- `--model-version VERSION` - Model version (default: maisi3d-rflow)
- `--base-seed N` - Random seed (default: 42)

### Parallel Configuration
- `--nproc-per-node N` - Number of GPUs to use (auto-detects by default)
- `--script-path PATH` - Path to MAISI script (default: maisi_train_diff_unet_parallel.py)

### Setup Options
- `--skip-install` - Skip dependency installation
- `--continue-on-error` - Continue even if some steps fail
- `--debug` - Enable debug output

## 🔧 What the Scripts Do

### 1. Environment Setup
- Checks Python version and virtual environment
- Installs required packages (torch, monai, nibabel, etc.)
- Verifies GPU availability and memory

### 2. Data Validation
- Searches common directories for medical data (.nii.gz files)
- Creates simulated test data if no real data found
- Validates data paths and file accessibility

### 3. GPU Configuration
- Detects available GPUs and their specifications
- Configures optimal parallel training parameters
- Handles GPU count mismatches automatically

### 4. Port Management
- Checks for available ports to avoid conflicts
- Resolves common `EADDRINUSE` errors
- Cleans up existing torchrun processes

### 5. Training Execution
- Builds and executes the correct torchrun command
- Monitors training output in real-time
- Captures and analyzes errors

### 6. Results Analysis
- Inspects output directory structure
- Lists generated medical images
- Provides viewing recommendations
- Saves configuration for reference

## 🎯 Your 2-GPU Setup

For your specific RTX 4090 setup:

```bash
# Optimal command for your hardware
python maisi_parallel_setup_and_run.py \
    --use-simulated-data \
    --epochs 5 \
    --batch-size 1 \
    --nproc-per-node 2
```

Expected output:
- 🖥️ Uses both RTX 4090 GPUs
- 🖼️ Generates 2 unique medical images
- ⚡ ~5-10 minutes for test run
- 📁 Results in `output_work_dir/`

## 📊 Output Structure

After successful training:

```
output_work_dir/
├── generated_image_seed42_rank0.nii.gz  # GPU 0 output
├── generated_image_seed42_rank1.nii.gz  # GPU 1 output
├── model_checkpoints/
└── logs/
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"No GPUs detected"**
   ```bash
   nvidia-smi  # Check GPU status
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **"Port already in use"**
   ```bash
   pkill -f torchrun  # Kill existing processes
   # Script handles this automatically
   ```

3. **"CUDA out of memory"**
   ```bash
   python maisi_parallel_setup_and_run.py --batch-size 1 --nproc-per-node 1
   ```

4. **"Script not found"**
   ```bash
   # Make sure maisi_train_diff_unet_parallel.py is in current directory
   ls -la maisi_train_diff_unet_parallel.py
   ```

### Debug Mode

For detailed troubleshooting:

```bash
python maisi_parallel_setup_and_run.py --debug --continue-on-error
```

## 🔄 Manual Fallback

If automated script fails, you can run manually:

```bash
# Clean up first
pkill -f torchrun
sleep 3

# Run with your 2 GPUs
torchrun --nproc_per_node=2 --master-port=29500 maisi_train_diff_unet_parallel.py \
    --epochs 5 --batch-size 1 --model-version maisi3d-rflow --base-seed 42
```

## 💡 Tips for Better Results

1. **Use Real Data**: Replace `--use-simulated-data` with `--data-path /your/real/data`
2. **Increase Epochs**: Use `--epochs 50` or higher for production quality
3. **Optimize Batch Size**: Try `--batch-size 2` if you have enough GPU memory
4. **Save Configuration**: The script saves settings to `maisi_training_config.json`

## 🎉 Success Indicators

Look for these in the output:
- ✅ "CUDA is available! Found 2 GPU(s)"
- ✅ "Training completed successfully!"
- ✅ "Generated images saved"
- ✅ Files in `output_work_dir/`

## 📧 Support

If you encounter issues:
1. Run with `--debug` flag
2. Check the generated `maisi_training_config.json`
3. Verify `nvidia-smi` shows both GPUs
4. Ensure `maisi_train_diff_unet_parallel.py` exists in current directory

---

**Happy Training! 🚀**
