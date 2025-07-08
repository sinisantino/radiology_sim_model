#!/usr/bin/env python
# coding: utf-8

# Copyright (c) MONAI Consortium  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
# &nbsp;&nbsp;&nbsp;&nbsp;http://www.apache.org/licenses/LICENSE-2.0  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.

# # Training a 3D Diffusion Model for Generating 3D Images with Various Sizes and Spacings
# 
# ![Generated image examples](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/image3.png)
# 
# In this notebook, we detail the procedure for training a 3D latent diffusion model to generate high-dimensional 3D medical images. Due to the potential for out-of-memory issues on most GPUs when generating large images (e.g., those with dimensions of 512 x 512 x 512 or greater), we have structured the training process into two primary steps: 1) generating image embeddings and 2) training 3D latent diffusion models. The subsequent sections will demonstrate the entire process using a simulated dataset.
# 
# `[Release Note (March 2025)]:` We are excited to announce the new MAISI Version `'maisi3d-rflow'`. Compared with the previous version `'maisi3d-ddpm'`, it accelerated latent diffusion model inference by 33x. Please see the detailed difference in the following section.

# ## Command Line Arguments

import argparse

def parse_arguments():
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a 3D Diffusion Model for Generating Medical Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs. If not specified, uses 2 for simulated data and 50 for real data."
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Training batch size (number of medical images processed together). "
             "Default=1 is safe for most GPUs. Increase to 2-4 if you have: "
             "high-end GPU (24GB+ VRAM), smaller image dimensions, or want faster training. "
             "Keep at 1 if you have: limited GPU memory (8-16GB), very large 3D medical images, "
             "or get GPU out-of-memory errors."
    )
    
    # Data configuration
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real medical imaging data instead of simulated data."
    )
    
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="/path/to/your/medical/imaging/data",
        help="Path to directory containing real medical imaging data (.nii.gz files)."
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=1,
        help="Number of GPUs to use for training."
    )
    
    parser.add_argument(
        "--validate-gpu",
        action="store_true",
        help="Validate GPU availability before starting training. "
             "Recommended for: new machines, multiple GPUs, cloud environments, "
             "debugging GPU issues, or when unsure about hardware compatibility."
    )
    
    # MAISI version
    parser.add_argument(
        "--model-version",
        type=str,
        default="maisi3d-rflow",
        choices=["maisi3d-rflow", "maisi3d-ddpm"],
        help="MAISI model version to use."
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# GPU Validation (if requested)
if args.validate_gpu:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU validation: Found {gpu_count} GPU(s)")
            
            # Check if requested GPUs exceed available
            if args.gpus > gpu_count:
                print(f"⚠️  Warning: Requested {args.gpus} GPUs but only {gpu_count} available")
                args.gpus = gpu_count
                print(f"   Adjusting to use {args.gpus} GPUs")
            
            # Validate each GPU and check for potential issues
            insufficient_memory_gpus = []
            low_memory_gpus = []
            
            for i in range(min(args.gpus, gpu_count)):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_memory = gpu_props.total_memory / 1024**3
                    
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                    # Check for insufficient memory (less than 6 GB)
                    if gpu_memory < 6.0:
                        insufficient_memory_gpus.append((i, gpu_name, gpu_memory))
                    # Check for low memory warning (6-12 GB)
                    elif gpu_memory < 12.0:
                        low_memory_gpus.append((i, gpu_name, gpu_memory))
                    
                    # Check GPU memory usage
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    
                    if memory_allocated > 0.1:  # More than 100MB already in use
                        print(f"   ⚠️  GPU {i} has {memory_allocated:.1f} GB already allocated")
                        print(f"      Consider freeing GPU memory or restarting Python")
                        
                except Exception as e:
                    print(f"   ❌ Error accessing GPU {i}: {str(e)}")
            
            # Report memory issues
            if insufficient_memory_gpus:
                print(f"\n❌ INSUFFICIENT GPU MEMORY:")
                for gpu_id, name, memory in insufficient_memory_gpus:
                    print(f"   • GPU {gpu_id} ({name}): {memory:.1f} GB - TOO LOW for 3D medical imaging")
                print(f"   RECOMMENDATION: Use GPUs with 8GB+ memory for this workload")
                print(f"   Your training may fail with out-of-memory errors")
            
            if low_memory_gpus:
                print(f"\n⚠️  LOW GPU MEMORY WARNING:")
                for gpu_id, name, memory in low_memory_gpus:
                    print(f"   • GPU {gpu_id} ({name}): {memory:.1f} GB - Limited for large 3D images")
                print(f"   RECOMMENDATION: Keep batch_size=1 and monitor memory usage")
            
            # Check CUDA version compatibility
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"   CUDA version: {cuda_version}")
                # Basic version check - warn if very old
                major_version = int(cuda_version.split('.')[0])
                if major_version < 11:
                    print(f"   ⚠️  Old CUDA version detected. Consider upgrading to CUDA 11.0+")
            
        else:
            print("❌ No CUDA GPUs available. Training will be very slow on CPU.")
            print("   POSSIBLE CAUSES:")
            print("   • No NVIDIA GPU installed")
            print("   • CUDA drivers not installed or outdated")
            print("   • PyTorch CPU-only version installed")
            print("   • GPU not detected by system")
            print("   SOLUTIONS:")
            print("   • Install NVIDIA drivers: sudo apt install nvidia-driver-xxx")
            print("   • Install CUDA toolkit")
            print("   • Reinstall PyTorch with CUDA support")
            print("   • Check GPU with: nvidia-smi")
            
    except ImportError as e:
        print("⚠️  PyTorch not available for GPU validation")
        print(f"   Error: {str(e)}")
        print("   SOLUTION: Install PyTorch with: pip install torch torchvision")
    except Exception as e:
        print(f"❌ Unexpected error during GPU validation: {str(e)}")
        print("   This may indicate a serious GPU or driver issue")

# Display configuration
print("="*60)
print("MAISI TRAINING CONFIGURATION:")
print("="*60)
print(f"• Model version: {args.model_version}")
print(f"• Using real data: {args.real_data}")
if args.real_data:
    print(f"• Data path: {args.data_path}")
if args.epochs:
    print(f"• Training epochs: {args.epochs} (from command line)")
else:
    print(f"• Training epochs: Auto-determined based on data type")
print(f"• Number of GPUs: {args.gpus}")
print(f"• Batch size: {args.batch_size}")

# Display batch size guidance
if args.batch_size > 1:
    print(f"  ℹ️  Using batch size {args.batch_size} - ensure you have sufficient GPU memory")
    print(f"  💡 Monitor GPU usage with: nvidia-smi")
else:
    print(f"  ℹ️  Using batch size 1 (safe default for most GPUs)")
    print(f"  💡 Consider increasing to 2-4 if you have 24GB+ GPU memory")

# Add tip about GPU validation
if not args.validate_gpu:
    print(f"\n💡 TIP: Add --validate-gpu to check GPU compatibility and avoid training issues")

print("\n📋 BATCH SIZE GUIDANCE:")
print("• INCREASE batch size (to 2-4) if you have:")
print("  - High-end GPU with 24GB+ VRAM (RTX 4090, A100, H100)")
print("  - Smaller medical image dimensions")
print("  - Want potentially faster training")
print("• KEEP batch size at 1 if you have:")
print("  - Limited GPU memory (8-16GB)")
print("  - Very large 3D medical images (512³+ voxels)")
print("  - Getting GPU out-of-memory (OOM) errors")
print("• NOTE: 3D medical images are memory-intensive, batch_size=1 is often optimal")

print("="*60)

# ## Setup environment

# In[1]:

import subprocess
import sys

try:
    import monai
except ImportError:
    print("MONAI not found. Installing monai-weekly[pillow, tqdm]...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai-weekly[pillow, tqdm]"])
    import monai


# ## Setup imports

# In[2]:


import copy
import os
import json
import numpy as np
import nibabel as nib
import subprocess
# Note: IPython.display imports removed as they don't work in regular Python scripts

from monai.apps import download_url
from monai.data import create_test_image_3d
from monai.config import print_config

from scripts.diff_model_setting import setup_logging

print_config()

logger = setup_logging("notebook")


# ## Set up the MAISI version
# 
# Choose between `'maisi3d-ddpm'` and `'maisi3d-rflow'`. The differences are:
# - The maisi version `'maisi3d-ddpm'` uses basic noise scheduler DDPM. `'maisi3d-rflow'` uses Rectified Flow scheduler, can be 33 times faster during inference.
# - The maisi version `'maisi3d-ddpm'` requires training images to be labeled with body regions (`"top_region_index"` and `"bottom_region_index"`), while `'maisi3d-rflow'` does not have such requirement. In other words, it is easier to prepare training data for `'maisi3d-rflow'`.
# - For the released model weights, `'maisi3d-rflow'` can generate images with better quality for head region and small output volumes, and comparable quality for other cases compared with `'maisi3d-ddpm'`.

# In[3]:


maisi_version = args.model_version  # Use command line argument
if maisi_version == "maisi3d-ddpm":
    model_def_path = "./configs/config_maisi3d-ddpm.json"
elif maisi_version == "maisi3d-rflow":
    model_def_path = "./configs/config_maisi3d-rflow.json"
else:
    raise ValueError(f"maisi_version has to be chosen from ['maisi3d-ddpm', 'maisi3d-rflow'], yet got {maisi_version}.")
with open(model_def_path, "r") as f:
    model_def = json.load(f)
include_body_region = model_def["include_body_region"]
logger.info(f"MAISI version is {maisi_version}, whether to use body_region is {include_body_region}")


# ### Dataset Configuration
# 
# Choose between simulated data (for demo) or real medical imaging data (for production)
# Now controlled by command line arguments: --real-data and --data-path

# In[4]:

# =============================================================================
# CONFIGURATION: Use command line arguments
# =============================================================================

USE_REAL_DATA = args.real_data  # Set by --real-data flag

if USE_REAL_DATA:
    # =============================================================================
    # OPTION 2: REAL MEDICAL IMAGING DATA (Recommended for production)
    # =============================================================================
    
    # Path from command line argument
    REAL_DATA_PATH = args.data_path
    
    # Create datalist from real data
    import glob
    real_image_files = glob.glob(os.path.join(REAL_DATA_PATH, "*.nii.gz"))
    
    if not real_image_files:
        raise ValueError(f"No .nii.gz files found in {REAL_DATA_PATH}. Please check your data path.")
    
    # Create datalist with relative paths
    sim_datalist = {
        "training": [{"image": os.path.basename(f)} for f in real_image_files]
    }
    
    # For real data, dimensions will be read from actual files
    # Load first image to get typical dimensions
    import nibabel as nib
    first_img = nib.load(real_image_files[0])
    sim_dim = first_img.shape[:3]
    
    print(f"Found {len(real_image_files)} real medical images")
    print(f"Example dimensions from first image: {sim_dim}")
    print("Real image files:")
    for i, f in enumerate(real_image_files[:5]):  # Show first 5
        print(f"  {i+1}. {os.path.basename(f)}")
    if len(real_image_files) > 5:
        print(f"  ... and {len(real_image_files) - 5} more files")

else:
    # =============================================================================
    # OPTION 1: SIMULATED DATA (Demo only - produces noise-like results)
    # =============================================================================
    
    print("WARNING: Using simulated data. Results will look like noise/static!")
    print("For real medical images, use: --real-data --data-path /your/data/path")
    
    sim_datalist = {"training": [{"image": "tr_image_001.nii.gz"}, {"image": "tr_image_002.nii.gz"}]}
    sim_dim = (224, 224, 96)


# ### Generate/Copy Data
# 
# Now we either generate simulated images or copy real medical images to the working directory

# In[5]:


work_dir = "./output_work_dir"  # Fixed output directory
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

dataroot_dir = os.path.join(work_dir, "dataroot")
if not os.path.isdir(dataroot_dir):
    os.makedirs(dataroot_dir)

datalist_file = os.path.join(work_dir, "datalist.json")
with open(datalist_file, "w") as f:
    json.dump(sim_datalist, f)

if USE_REAL_DATA:
    # Copy real medical images to working directory
    import shutil
    
    print("Copying real medical images to working directory...")
    for i, source_file in enumerate(real_image_files):
        dest_file = os.path.join(dataroot_dir, os.path.basename(source_file))
        shutil.copy2(source_file, dest_file)
        print(f"  Copied {i+1}/{len(real_image_files)}: {os.path.basename(source_file)}")
    
    print(f"Successfully copied {len(real_image_files)} real medical images")
    
else:
    # Generate simulated images (original demo behavior)
    print("Generating simulated images (will produce noise-like results)...")
    for d in sim_datalist["training"]:
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(dataroot_dir, d["image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)
    
    print("Generated simulated images.")

logger.info(f"Data prepared in: {dataroot_dir}")


# ### Set up directories and configurations
# 
# To optimize the demonstration for time efficiency, we have adjusted the training epochs to 2. Additionally, we modified the `num_splits` parameter in [AutoencoderKlMaisi](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/generation/maisi/networks/autoencoderkl_maisi.py#L881) from its default value of 16 to 4. This adjustment reduces the spatial splitting of feature maps in convolutions, which is particularly beneficial given the smaller input size. (This change helps convert convolutions to a for-loop based approach, thereby conserving GPU memory resources.)

# In[6]:


env_config_path = "./configs/environment_maisi_diff_model.json"
model_config_path = "./configs/config_maisi_diff_model.json"

# Load environment configuration, model configuration and model definition
with open(env_config_path, "r") as f:
    env_config = json.load(f)

with open(model_config_path, "r") as f:
    model_config = json.load(f)

env_config_out = copy.deepcopy(env_config)
model_config_out = copy.deepcopy(model_config)
model_def_out = copy.deepcopy(model_def)

# Set up directories based on configurations
env_config_out["data_base_dir"] = dataroot_dir
env_config_out["embedding_base_dir"] = os.path.join(work_dir, env_config_out["embedding_base_dir"])
env_config_out["json_data_list"] = datalist_file
env_config_out["model_dir"] = os.path.join(work_dir, env_config_out["model_dir"])
env_config_out["output_dir"] = os.path.join(work_dir, env_config_out["output_dir"])
trained_autoencoder_path = os.path.join(work_dir, "models/autoencoder_epoch273.pt")
env_config_out["trained_autoencoder_path"] = trained_autoencoder_path
trained_autoencoder_path_url = (
    "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
    "model_zoo/model_maisi_autoencoder_epoch273_alternative.pt"
)
if not os.path.exists(trained_autoencoder_path):
    download_url(url=trained_autoencoder_path_url, filepath=trained_autoencoder_path)

# Create necessary directories
os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)
os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)

env_config_filepath = os.path.join(work_dir, "environment_maisi_diff_model.json")
with open(env_config_filepath, "w") as f:
    json.dump(env_config_out, f, sort_keys=True, indent=4)

# Update model configuration for training
if args.epochs:
    # Use user-specified epochs from command line
    max_epochs = args.epochs
    print(f"Using {max_epochs} epochs (from command line)")
elif USE_REAL_DATA:
    # Use more realistic training settings for real data
    max_epochs = 50  # Increased for real data (still conservative)
    print(f"Using {max_epochs} epochs for real data training")
    print("Note: For production, consider 100+ epochs")
else:
    # Keep minimal settings for demo
    max_epochs = 2
    print(f"Using {max_epochs} epochs for simulated data demo")

model_config_out["diffusion_unet_train"]["n_epochs"] = max_epochs

# Update batch size if specified
if hasattr(args, 'batch_size') and args.batch_size != 1:
    model_config_out["diffusion_unet_train"]["batch_size"] = args.batch_size
    print(f"Updated batch size to: {args.batch_size}")

model_config_filepath = os.path.join(work_dir, "config_maisi_diff_model.json")
with open(model_config_filepath, "w") as f:
    json.dump(model_config_out, f, sort_keys=True, indent=4)

# Adjust based on the number of GPUs you want to use
num_gpus = args.gpus  # Use command line argument instead of hardcoded 1
logger.info(f"number of GPUs: {num_gpus}.")

# =============================================================================
# TRAINING TIME ESTIMATION
# =============================================================================

def estimate_training_time(num_images, epochs, num_gpus=1):
    """
    Estimate training time based on typical performance benchmarks.
    These are rough estimates and actual time may vary significantly.
    """
    
    # Base time per image per epoch (in minutes) - varies by GPU and image size
    # These are conservative estimates for safety
    time_estimates = {
        "V100": 0.5,      # ~30 seconds per image per epoch
        "A100": 0.25,     # ~15 seconds per image per epoch  
        "H100": 0.15,     # ~9 seconds per image per epoch
        "RTX4090": 0.4,   # ~24 seconds per image per epoch
        "RTX3090": 0.6,   # ~36 seconds per image per epoch
        "average": 0.4    # Conservative average
    }
    
    base_time_per_image_epoch = time_estimates["average"]
    
    # Adjust for number of GPUs (not perfectly linear due to overhead)
    gpu_efficiency = {1: 1.0, 2: 0.85, 4: 0.75, 8: 0.65}
    efficiency = gpu_efficiency.get(num_gpus, 0.6)
    
    # Total time calculation
    total_time_minutes = (num_images * epochs * base_time_per_image_epoch) / (num_gpus * efficiency)
    
    # Add overhead time (data loading, validation, checkpointing)
    overhead_multiplier = 1.3
    total_time_minutes *= overhead_multiplier
    
    return total_time_minutes

def format_time(minutes):
    """Convert minutes to human-readable format."""
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes / 60
        return f"{hours:.1f} hours"
    else:  # Days
        days = minutes / 1440
        return f"{days:.1f} days"

# Calculate and display time estimates
if USE_REAL_DATA:
    num_training_images = len(real_image_files)
else:
    num_training_images = len(sim_datalist["training"])

estimated_minutes = estimate_training_time(num_training_images, max_epochs, num_gpus)

print("\n" + "="*60)
print("TRAINING TIME ESTIMATION:")
print("="*60)
print(f"• Number of training images: {num_training_images}")
print(f"• Number of epochs: {max_epochs}")
print(f"• Number of GPUs: {num_gpus}")
print(f"• Estimated total training time: {format_time(estimated_minutes)}")

if USE_REAL_DATA:
    print(f"\n📊 SCALING EXAMPLES with {num_training_images} images:")
    for epochs in [25, 50, 100, 200]:
        time_est = estimate_training_time(num_training_images, epochs, num_gpus)
        print(f"   • {epochs:3d} epochs: {format_time(time_est)}")
else:
    print(f"\n⚠️  DEMO MODE: Very fast but poor results expected")

print(f"\n🚀 GPU PERFORMANCE COMPARISON (estimated for {max_epochs} epochs, {num_training_images} images):")
gpu_types = ["RTX3090", "RTX4090", "V100", "A100", "H100"]
for gpu in gpu_types:
    time_est = estimate_training_time(num_training_images, max_epochs, num_gpus)
    if gpu == "A100":
        time_est *= 0.6  # A100 is faster
    elif gpu == "H100":
        time_est *= 0.4  # H100 is much faster
    elif gpu == "V100":
        time_est *= 1.2  # V100 is slower
    elif gpu == "RTX4090":
        time_est *= 1.0  # Baseline
    elif gpu == "RTX3090":
        time_est *= 1.5  # RTX3090 is slower
    
    print(f"   • {gpu:8s}: {format_time(time_est)}")

print(f"\n💡 FACTORS AFFECTING ACTUAL TIME:")
print("   • Image dimensions (larger = slower)")
print("   • GPU memory (affects batch size)")
print("   • System I/O speed (SSD vs HDD)")
print("   • CPU performance (data loading)")
print("   • Network storage vs local storage")
print("   • Other processes running on the system")

print("\n⏰ MONITORING TIPS:")
print("   • Training will show progress and ETA during execution")
print("   • Check GPU utilization: nvidia-smi")
print("   • Monitor training logs for loss curves")
print("   • First epoch is often slower (data loading setup)")

print("="*60)
# Update model definition for demo
model_def_out["autoencoder_def"]["num_splits"] = 2
model_def_filepath = os.path.join(work_dir, "config_maisi.json")
with open(model_def_filepath, "w") as f:
    json.dump(model_def_out, f, sort_keys=True, indent=4)

# Print files and folders under work_dir
logger.info(f"files and folders under work_dir: {os.listdir(work_dir)}.")


# In[7]:


def run_torchrun(module, module_args, num_gpus=1):
    # Define the arguments for torchrun
    num_nodes = 1

    # Build the torchrun command
    torchrun_command = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "-m",
        module,
    ] + module_args

    # Set the OMP_NUM_THREADS environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Execute the command
    process = subprocess.Popen(torchrun_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Print the output in real-time
    try:
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Capture and print any remaining output
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return


# ## Step 1: Create Training Data
# 
# To train the latent diffusion model, we first store the latent features produced by the autoencoder's encoder in local storage. This allows the latent diffusion model to directly utilize these features, thereby conserving both time and GPU memory during the training process. Additionally, we have provided the script for multi-GPU processing to save latent features from all training images, significantly accelerating the creation of the entire training set.
# 
# The diffusion model utilizes a U-shaped convolutional neural network architecture, requiring matching input and output dimensions. Therefore, it is advisable to resample the input image dimensions to be multiples of 2 for compatibility. In this case, we have chosen dimensions that are multiples of 128.

# In[8]:


logger.info("Creating training data...")

# Define the arguments for torchrun
module = "scripts.diff_model_create_training_data"
module_args = [
    "--env_config",
    env_config_filepath,
    "--model_config",
    model_config_filepath,
    "--model_def",
    model_def_filepath,
    "--num_gpus",
    str(num_gpus),
]

run_torchrun(module, module_args, num_gpus=num_gpus)


# ### Create .json files for embedding files
# 
# The diffusion model necessitates additional input attributes, including output dimension, output spacing, and top/bottom body region. These dimensions and spacing can be extracted from the header information of the training images. The top and bottom body region inputs can be determined through manual examination or by utilizing segmentation masks from tools such as [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) or [MONAI VISTA](https://github.com/Project-MONAI/VISTA). The body regions are formatted as 4-dimensional one-hot vectors: the head and neck region is represented by [1,0,0,0], the chest region by [0,1,0,0], the abdomen region by [0,0,1,0], and the lower body region (below the abdomen) by [0,0,0,1]. The additional input attributes are saved in a separate .json file. In the following example, we assume that the images cover the chest and abdomen regions.

# In[9]:


def list_gz_files(folder_path):
    """List all .gz files in the folder and its subfolders."""
    gz_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".gz"):
                gz_files.append(os.path.join(root, file))
    return gz_files


def create_json_files(gz_files):
    """Create .json files for each .gz file with the specified keys and values."""
    for gz_file in gz_files:
        # Load the NIfTI image
        img = nib.load(gz_file)

        # Get the dimensions and spacing
        dimensions = img.shape
        dimensions = dimensions[:3]
        spacing = img.header.get_zooms()[:3]
        spacing = spacing[:3]
        spacing = [float(_item) for _item in spacing]

        # Create the dictionary with the specified keys and values
        data = {"dim": dimensions, "spacing": spacing}
        if include_body_region:
            # The region can be selected from one of four regions from top to bottom.
            # [1,0,0,0] is the head and neck, [0,1,0,0] is the chest region, [0,0,1,0]
            # is the abdomen region, and [0,0,0,1] is the lower body region.
            data["top_region_index"] = [0, 1, 0, 0]  # chest region
            data["bottom_region_index"] = [0, 0, 1, 0]  # abdomen region
        logger.info(f"data: {data}.")

        # Create the .json filename
        json_filename = gz_file + ".json"

        # Write the dictionary to the .json file
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Save json file to {json_filename}")


folder_path = env_config_out["embedding_base_dir"]
gz_files = list_gz_files(folder_path)
create_json_files(gz_files)

logger.info("Completed creating .json files for all embedding files.")


# ## Step 2: Train the Model
# 
# After all latent features have been created, we will initiate the multi-GPU script to train the latent diffusion model.
# 
# The image generation process utilizes the [DDPM scheduler](https://arxiv.org/pdf/2006.11239) with 1,000 iterative steps. The diffusion model is optimized using L1 loss and a decayed learning rate scheduler. The batch size for this process is set to 1.
# 
# Please be aware that using the H100 GPU may occasionally result in random segmentation faults. To avoid this issue, you can disable AMP by setting the `--no_amp` flag.

# In[10]:


logger.info("Training the model...")

# Define the arguments for torchrun
module = "scripts.diff_model_train"
module_args = [
    "--env_config",
    env_config_filepath,
    "--model_config",
    model_config_filepath,
    "--model_def",
    model_def_filepath,
    "--num_gpus",
    str(num_gpus),
]

run_torchrun(module, module_args, num_gpus=num_gpus)


# ## Step 3: Infer using the Trained Model
# 
# Upon completing the training of the latent diffusion model, we can employ the multi-GPU script to perform inference. By integrating the diffusion model with the autoencoder's decoder, this process will generate 3D images with specified top/bottom body regions, spacing, and dimensions.

# In[11]:


logger.info("Running inference...")

# Define the arguments for torchrun
module = "scripts.diff_model_infer"
module_args = [
    "--env_config",
    env_config_filepath,
    "--model_config",
    model_config_filepath,
    "--model_def",
    model_def_filepath,
    "--num_gpus",
    str(num_gpus),
]

run_torchrun(module, module_args, num_gpus=num_gpus)

logger.info("Completed all steps.")


# Upon completing the full training with the actual CT datasets, users can expect output images similar to the examples below, which present the generated images in axial, sagittal, and coronal views. The specific content may vary depending on the distribution of body regions in the training set. It is advisable to use tools such as [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) or [3D Slicer](https://www.slicer.org/) to visualize the entire volume for a comprehensive evaluation, rather than relying solely on the three different views to assess the quality of the checkpoints.

# In[12]:

# Display information about where to find your actual results
print("Training completed!")
print(f"Your trained model checkpoints are saved in: {env_config_out['model_dir']}")
print(f"Your generated images and inference results are saved in: {env_config_out['output_dir']}")
print(f"Working directory with all outputs: {work_dir}")
print("\nTo visualize your results, use tools like ITK-SNAP or 3D Slicer to open the generated .nii.gz files.")
print("Look for files with names like 'generated_*.nii.gz' or 'inferred_*.nii.gz' in the output directory.")

print("\n" + "="*60)

print("\n3. COMMAND LINE TOOLS (Alternative):")
print("   • Install FSL: sudo apt-get install fsl")
print("   • View with fsleyes: fsleyes your_image.nii.gz")
print("   • Or use fslview: fslview your_image.nii.gz")

print(f"\n4. YOUR SPECIFIC FILES TO CHECK:")
print(f"   • Output directory: {env_config_out['output_dir']}")
print(f"   • Model directory: {env_config_out['model_dir']}")
print(f"   • Look for files ending in .nii.gz")

print("\nTIP: Start with 3D Slicer if you're new to medical imaging - it's more user-friendly!")
print("="*60)

print("\n" + "="*60)
print("TROUBLESHOOTING: IF YOUR IMAGES LOOK LIKE STATIC/NOISE")
print("="*60)

print("\nIf your generated images look like static or noise, this is EXPECTED with this demo setup!")
print("Here's why and what you can do:")

print("\n1. WHY THE IMAGES LOOK LIKE STATIC:")
print("   • This demo only trains for 2 epochs (very short training)")
print("   • Only 2 training images are used (extremely small dataset)")
print("   • The simulated images are simple test patterns, not real medical data")
print("   • Diffusion models need MUCH more training to generate realistic images")

print("\n2. WHAT REAL TRAINING REQUIRES:")
print("   • 100+ epochs minimum (this demo uses only 2)")
print("   • Thousands of real medical images (this demo uses 2 simulated ones)")
print("   • High-quality, diverse training data")
print("   • Much longer training time (hours to days, not minutes)")

print("\n3. HOW TO GET BETTER RESULTS:")
print("   • Increase max_epochs from 2 to 100+ in the script")
print("   • Use real medical imaging datasets (CT scans, MRI, etc.)")
print("   • Add more training images to your dataset")
print("   • Train on a powerful GPU for longer periods")

print("\n4. WHAT TO CHECK IN YOUR CURRENT RESULTS:")
print("   • Look at the training loss curves - are they decreasing?")
print("   • Check if the 'static' has any structure or patterns")
print("   • Compare multiple generated samples - are they different?")
print("   • Verify the intensity values are in a reasonable range")

print("\n5. QUICK FIXES TO TRY:")
print("   • Adjust window/level settings in Slicer (try different contrast)")
print("   • Check different slices - some may show more structure")
print("   • Look at the original training images to compare")

print(f"\n6. YOUR TRAINING SETUP (CURRENT LIMITATIONS):")
print(f"   • Training epochs: {max_epochs} (recommended: 100+)")
print(f"   • Training images: 2 simulated (recommended: 1000+ real)")
print(f"   • Image dimensions: {sim_dim} (smaller than typical clinical images)")

print("\nREMEMBER: This is a DEMONSTRATION script. For better results, you need real data and extensive training.")
print("="*60)

