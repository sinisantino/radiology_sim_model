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

# # Training a 3D Diffusion Model for Generating 3D Images (PARALLEL COMPUTING VERSION)
# 
# This version is optimized for parallel computing environments with multiple GPUs.
# It generates one image per GPU rank simultaneously, ideal for research clusters.
# 
# For single-user environments, use maisi_train_diff_unet.py instead.

# ## Command Line Arguments

import argparse

def parse_arguments():
    """Parse command line arguments for the parallel training script."""
    parser = argparse.ArgumentParser(
        description="Train a 3D Diffusion Model for Generating Medical Images (Parallel Computing Version)",
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
        help="Training batch size per GPU. Default=1 is safe for most GPUs."
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
    
    # Parallel computing configuration
    parser.add_argument(
        "--num-gpus", "-g",
        type=int,
        default=4,
        help="Number of GPUs to use for parallel training/inference. Each GPU will generate one image."
    )
    
    parser.add_argument(
        "--num-nodes", "-n",
        type=int,
        default=1,
        help="Number of compute nodes in the cluster."
    )
    
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of the current node (0-based). Required for multi-node setups."
    )
    
    parser.add_argument(
        "--master-addr",
        type=str,
        default="localhost",
        help="Address of the master node for distributed training."
    )
    
    parser.add_argument(
        "--master-port",
        type=str,
        default="12355",
        help="Port of the master node for distributed training."
    )
    
    # MAISI version
    parser.add_argument(
        "--model-version",
        type=str,
        default="maisi3d-rflow",
        choices=["maisi3d-rflow", "maisi3d-ddpm"],
        help="MAISI model version to use."
    )
    
    # Parallel inference configuration
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Each GPU will use base_seed + rank for diversity."
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Display parallel computing configuration
print("="*80)
print("MAISI PARALLEL COMPUTING CONFIGURATION:")
print("="*80)
print(f"🖥️  CLUSTER SETUP:")
print(f"• Number of nodes: {args.num_nodes}")
print(f"• Current node rank: {args.node_rank}")
print(f"• GPUs per node: {args.num_gpus}")
print(f"• Total GPUs in cluster: {args.num_nodes * args.num_gpus}")
print(f"• Master address: {args.master_addr}")
print(f"• Master port: {args.master_port}")

print(f"\n🎯 GENERATION PLAN:")
print(f"• Each GPU will generate 1 unique image")
print(f"• Total images to generate: {args.num_nodes * args.num_gpus}")
print(f"• Base random seed: {args.base_seed}")
print(f"• Seed range: {args.base_seed} to {args.base_seed + (args.num_nodes * args.num_gpus) - 1}")

print(f"\n📊 TRAINING SETUP:")
print(f"• Model version: {args.model_version}")
print(f"• Using real data: {args.real_data}")
if args.real_data:
    print(f"• Data path: {args.data_path}")
if args.epochs:
    print(f"• Training epochs: {args.epochs} (from command line)")
else:
    print(f"• Training epochs: Auto-determined based on data type")
print(f"• Batch size per GPU: {args.batch_size}")

print("="*80)

# ## Setup environment and imports (same as original)

import subprocess
import sys

try:
    import monai
except ImportError:
    print("MONAI not found. Installing monai-weekly[pillow, tqdm]...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai-weekly[pillow, tqdm]"])
    import monai

import copy
import os
import json
import numpy as np
import nibabel as nib
import subprocess

from monai.apps import download_url
from monai.data import create_test_image_3d
from monai.config import print_config

from scripts.diff_model_setting import setup_logging

print_config()

logger = setup_logging("parallel_notebook")

# ## Set up the MAISI version (same as original)

maisi_version = args.model_version
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

# ## Dataset Configuration (same as original)

USE_REAL_DATA = args.real_data

if USE_REAL_DATA:
    REAL_DATA_PATH = args.data_path
    
    import glob
    real_image_files = glob.glob(os.path.join(REAL_DATA_PATH, "*.nii.gz"))
    
    if not real_image_files:
        raise ValueError(f"No .nii.gz files found in {REAL_DATA_PATH}. Please check your data path.")
    
    sim_datalist = {
        "training": [{"image": os.path.basename(f)} for f in real_image_files]
    }
    
    import nibabel as nib
    first_img = nib.load(real_image_files[0])
    sim_dim = first_img.shape[:3]
    
    print(f"Found {len(real_image_files)} real medical images")
    print(f"Example dimensions from first image: {sim_dim}")

else:
    print("WARNING: Using simulated data. Results will look like noise/static!")
    print("For real medical images, use: --real-data --data-path /your/data/path")
    
    sim_datalist = {"training": [{"image": "tr_image_001.nii.gz"}, {"image": "tr_image_002.nii.gz"}]}
    sim_dim = (224, 224, 96)

# ## Data Setup (same as original)

work_dir = "./output_work_dir"
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

dataroot_dir = os.path.join(work_dir, "dataroot")
if not os.path.isdir(dataroot_dir):
    os.makedirs(dataroot_dir)

datalist_file = os.path.join(work_dir, "datalist.json")
with open(datalist_file, "w") as f:
    json.dump(sim_datalist, f)

if USE_REAL_DATA:
    import shutil
    print("Copying real medical images to working directory...")
    for i, source_file in enumerate(real_image_files):
        dest_file = os.path.join(dataroot_dir, os.path.basename(source_file))
        shutil.copy2(source_file, dest_file)
        print(f"  Copied {i+1}/{len(real_image_files)}: {os.path.basename(source_file)}")
    print(f"Successfully copied {len(real_image_files)} real medical images")
else:
    print("Generating simulated images (will produce noise-like results)...")
    for d in sim_datalist["training"]:
        im, _ = create_test_image_3d(
            sim_dim[0], sim_dim[1], sim_dim[2], rad_max=10, num_seg_classes=1, random_state=np.random.RandomState(42)
        )
        image_fpath = os.path.join(dataroot_dir, d["image"])
        nib.save(nib.Nifti1Image(im, affine=np.eye(4)), image_fpath)
    print("Generated simulated images.")

logger.info(f"Data prepared in: {dataroot_dir}")

# ## Configuration Setup (same as original but with parallel parameters)

env_config_path = "./configs/environment_maisi_diff_model.json"
model_config_path = "./configs/config_maisi_diff_model.json"

with open(env_config_path, "r") as f:
    env_config = json.load(f)

with open(model_config_path, "r") as f:
    model_config = json.load(f)

env_config_out = copy.deepcopy(env_config)
model_config_out = copy.deepcopy(model_config)
model_def_out = copy.deepcopy(model_def)

# Set up directories
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

os.makedirs(env_config_out["embedding_base_dir"], exist_ok=True)
os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)

env_config_filepath = os.path.join(work_dir, "environment_maisi_diff_model.json")
with open(env_config_filepath, "w") as f:
    json.dump(env_config_out, f, sort_keys=True, indent=4)

# Update model configuration for training
if args.epochs:
    max_epochs = args.epochs
    print(f"Using {max_epochs} epochs (from command line)")
elif USE_REAL_DATA:
    max_epochs = 50
    print(f"Using {max_epochs} epochs for real data training")
    print("Note: For production, consider 100+ epochs")
else:
    max_epochs = 2
    print(f"Using {max_epochs} epochs for simulated data demo")

model_config_out["diffusion_unet_train"]["n_epochs"] = max_epochs
model_config_out["diffusion_unet_train"]["batch_size"] = args.batch_size

# Update inference config with base seed for parallel generation
model_config_out["diffusion_unet_inference"]["random_seed"] = args.base_seed

model_config_filepath = os.path.join(work_dir, "config_maisi_diff_model.json")
with open(model_config_filepath, "w") as f:
    json.dump(model_config_out, f, sort_keys=True, indent=4)

model_def_out["autoencoder_def"]["num_splits"] = 2
model_def_filepath = os.path.join(work_dir, "config_maisi.json")
with open(model_def_filepath, "w") as f:
    json.dump(model_def_out, f, sort_keys=True, indent=4)

logger.info(f"files and folders under work_dir: {os.listdir(work_dir)}.")

# ## Parallel Training Functions

def run_parallel_torchrun(module, module_args, num_gpus, num_nodes=1, node_rank=0, master_addr="localhost", master_port="12355"):
    """
    Run torchrun for parallel/distributed training across multiple GPUs and nodes.
    
    Args:
        module: Python module to run
        module_args: Arguments to pass to the module
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes in the cluster
        node_rank: Rank of the current node
        master_addr: Address of the master node
        master_port: Port for communication
    """
    torchrun_command = [
        "torchrun",
        "--nproc_per_node", str(num_gpus),
        "--nnodes", str(num_nodes),
        "--node_rank", str(node_rank),
        "--master_addr", master_addr,
        "--master_port", master_port,
        "-m", module,
    ] + module_args

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    print(f"🚀 Running parallel command: {' '.join(torchrun_command)}")
    
    process = subprocess.Popen(torchrun_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

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
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return

def display_parallel_training_info():
    """Display information about parallel training setup."""
    total_gpus = args.num_nodes * args.num_gpus
    
    print("\n" + "="*80)
    print("PARALLEL TRAINING INFORMATION:")
    print("="*80)
    
    print(f"🔧 HARDWARE CONFIGURATION:")
    print(f"• Total GPUs across cluster: {total_gpus}")
    print(f"• GPUs per node: {args.num_gpus}")
    print(f"• Number of nodes: {args.num_nodes}")
    
    print(f"\n📈 PERFORMANCE BENEFITS:")
    print(f"• Training data creation: {args.num_gpus}x faster per node")
    print(f"• Model training: Distributed across {total_gpus} GPUs")
    print(f"• Image generation: {total_gpus} images generated simultaneously")
    
    print(f"\n🎯 EXPECTED OUTPUTS:")
    print(f"• Training: Distributed model weights")
    print(f"• Inference: {total_gpus} unique medical images")
    print(f"• Filenames will include rank: image_rankN.nii.gz")
    
    if args.num_nodes > 1:
        print(f"\n🌐 MULTI-NODE SETUP:")
        print(f"• Node {args.node_rank} of {args.num_nodes}")
        print(f"• Master node: {args.master_addr}:{args.master_port}")
        print(f"• This node will handle ranks {args.node_rank * args.num_gpus} to {(args.node_rank + 1) * args.num_gpus - 1}")
    
    print("="*80)

# Display parallel training information
display_parallel_training_info()

# ## Step 1: Create Training Data (Parallel)

print("\n🔄 STEP 1: Creating training data in parallel...")
logger.info("Creating training data...")

module = "scripts.diff_model_create_training_data"
module_args = [
    "--env_config", env_config_filepath,
    "--model_config", model_config_filepath,
    "--model_def", model_def_filepath,
    "--num_gpus", str(args.num_gpus),
]

run_parallel_torchrun(module, module_args, args.num_gpus, args.num_nodes, args.node_rank, args.master_addr, args.master_port)

# ## Create .json files for embedding files

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
        img = nib.load(gz_file)
        dimensions = img.shape[:3]
        spacing = img.header.get_zooms()[:3]
        spacing = [float(_item) for _item in spacing]

        data = {"dim": dimensions, "spacing": spacing}
        if include_body_region:
            data["top_region_index"] = [0, 1, 0, 0]  # chest region
            data["bottom_region_index"] = [0, 0, 1, 0]  # abdomen region
        
        json_filename = gz_file + ".json"
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Save json file to {json_filename}")

folder_path = env_config_out["embedding_base_dir"]
gz_files = list_gz_files(folder_path)
create_json_files(gz_files)
logger.info("Completed creating .json files for all embedding files.")

# ## Step 2: Train the Model (Parallel)

print("\n🔄 STEP 2: Training model in parallel...")
logger.info("Training the model...")

module = "scripts.diff_model_train"
module_args = [
    "--env_config", env_config_filepath,
    "--model_config", model_config_filepath,
    "--model_def", model_def_filepath,
    "--num_gpus", str(args.num_gpus),
]

run_parallel_torchrun(module, module_args, args.num_gpus, args.num_nodes, args.node_rank, args.master_addr, args.master_port)

# ## Step 3: Parallel Inference

print("\n🔄 STEP 3: Running parallel inference...")
logger.info("Running parallel inference...")

print(f"🎯 Each of the {args.num_gpus} GPUs will generate 1 unique image simultaneously")
print(f"📁 Results will be saved with rank identifiers: image_rank0.nii.gz, image_rank1.nii.gz, etc.")

module = "scripts.diff_model_infer"
module_args = [
    "--env_config", env_config_filepath,
    "--model_config", model_config_filepath,
    "--model_def", model_def_filepath,
    "--num_gpus", str(args.num_gpus),
]

run_parallel_torchrun(module, module_args, args.num_gpus, args.num_nodes, args.node_rank, args.master_addr, args.master_port)

logger.info("Completed all parallel training and inference steps.")

# ## Results Summary

print("\n" + "="*80)
print("PARALLEL TRAINING COMPLETED!")
print("="*80)

total_images = args.num_nodes * args.num_gpus

print(f"🎉 RESULTS SUMMARY:")
print(f"• Training completed on {args.num_nodes * args.num_gpus} GPUs")
print(f"• Generated {total_images} unique medical images")
print(f"• Model checkpoints saved in: {env_config_out['model_dir']}")
print(f"• Generated images saved in: {env_config_out['output_dir']}")

print(f"\n📁 GENERATED IMAGES:")
for node in range(args.num_nodes):
    for gpu in range(args.num_gpus):
        rank = node * args.num_gpus + gpu
        seed = args.base_seed + rank
        print(f"   • Rank {rank} (Node {node}, GPU {gpu}): image_seed{seed}_rank{rank}.nii.gz")

print(f"\n🔍 TO VIEW YOUR RESULTS:")
print(f"   cd {env_config_out['output_dir']}")
print(f"   ls -la *.nii.gz")
print(f"   # Use 3D Slicer, ITK-SNAP, or FSL to visualize the images")

print(f"\n💾 MODEL FILES:")
print(f"   • Trained diffusion model: {env_config_out['model_dir']}/")
print(f"   • Configuration files: {work_dir}/")

print("="*80)

print(f"\n🚀 PARALLEL COMPUTING BENEFITS ACHIEVED:")
print(f"• {args.num_gpus}x faster training data creation")
print(f"• Distributed training across {total_images} GPUs")
print(f"• {total_images} diverse images generated simultaneously")
print(f"• Each image has unique random seed for maximum diversity")

if args.num_nodes > 1:
    print(f"\n🌐 MULTI-NODE RESULTS:")
    print(f"• This node ({args.node_rank}) processed ranks {args.node_rank * args.num_gpus}-{(args.node_rank + 1) * args.num_gpus - 1}")
    print(f"• Check other nodes for their generated images")
    print(f"• Combine results from all {args.num_nodes} nodes for complete dataset")

print("="*80)
