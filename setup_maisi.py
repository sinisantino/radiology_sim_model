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

import argparse

# # Training a 3D Diffusion Model for Generating 3D Images with Various Sizes and Spacings
# 
# ![Generated image examples](https://developer-blogs.nvidia.com/wp-content/uploads/2024/06/image3.png)
# 
# In this notebook, we detail the procedure for training a 3D latent diffusion model to generate high-dimensional 3D medical images. Due to the potential for out-of-memory issues on most GPUs when generating large images (e.g., those with dimensions of 512 x 512 x 512 or greater), we have structured the training process into two primary steps: 1) generating image embeddings and 2) training 3D latent diffusion models. The subsequent sections will demonstrate the entire process using a simulated dataset.
# 
# `[Release Note (March 2025)]:` We are excited to announce the new MAISI Version `'maisi3d-rflow'`. Compared with the previous version `'maisi3d-ddpm'`, it accelerated latent diffusion model inference by 33x. Please see the detailed difference in the following section.

# ## Setup environment

# ## Setup imports

import copy
import os
import json
import sys
import glob
import shutil
import subprocess
import numpy as np
import nibabel as nib

# Check and install MONAI if needed
try:
    import monai
    print("MONAI is already installed")
except ImportError:
    print("Installing MONAI...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "monai-weekly[pillow,tqdm]"])

from monai.apps import download_url
from monai.config import print_config

from scripts.diff_model_setting import setup_logging


def get_body_region_encoding(region_name):
    """
    Get the one-hot encoding for different body regions.
    
    Args:
        region_name: String specifying the body region
        
    Returns:
        tuple: (top_region, bottom_region) as one-hot encoded lists
    """
    region_mappings = {
        "head_neck": ([1, 0, 0, 0], [1, 0, 0, 0]),      # Head/neck only
        "chest": ([0, 1, 0, 0], [0, 1, 0, 0]),          # Chest only  
        "abdomen": ([0, 0, 1, 0], [0, 0, 1, 0]),        # Abdomen only
        "lower_body": ([0, 0, 0, 1], [0, 0, 0, 1]),     # Lower body only
        "chest_abdomen": ([0, 1, 0, 0], [0, 0, 1, 0]),  # Chest to abdomen (default)
    }
    
    if region_name not in region_mappings:
        raise ValueError(f"Unknown body region: {region_name}. Available: {list(region_mappings.keys())}")
    
    return region_mappings[region_name]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MAISI 3D Diffusion Model Data Preparation and Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--maisi_version", 
        choices=["maisi3d-ddpm", "maisi3d-rflow"], 
        default="maisi3d-rflow", 
        help="MAISI version to use"
    )
    
    # Training parameters
    parser.add_argument(
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs"
    )
    
    # Directory configuration
    parser.add_argument(
        "--work_dir", 
        default="./maisi_work_dir", 
        help="Working directory for outputs and temporary files"
    )
    
    # Data configuration (required)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing medical imaging data (.nii.gz files)"
    )
    
    # Advanced options
    parser.add_argument(
        "--no_amp", 
        action="store_true", 
        help="Disable automatic mixed precision (useful for H100 GPUs)"
    )
    parser.add_argument(
        "--num_splits", 
        type=int, 
        default=2, 
        help="Number of splits for autoencoder (reduces GPU memory usage)"
    )
    parser.add_argument(
        "--create_json_only",
        action="store_true",
        help="Only create JSON metadata files (run after training data creation)"
    )
    
    # Body region specification (for maisi3d-ddpm only)
    parser.add_argument(
        "--body_region",
        choices=["head_neck", "chest", "abdomen", "lower_body", "chest_abdomen"],
        default="chest_abdomen",
        help="Target body region for image generation (maisi3d-ddpm only). Options: head_neck, chest, abdomen, lower_body, chest_abdomen"
    )
    
    # Inference parameters
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of synthetic images to generate during inference (default: 1). Medical images are complex - start with 1 and increase carefully."
    )
    
    return parser.parse_args()


print_config()

logger = setup_logging("notebook")


# ## Set up the MAISI version
# 
# Choose between `'maisi3d-ddpm'` and `'maisi3d-rflow'`. The differences are:
# - The maisi version `'maisi3d-ddpm'` uses basic noise scheduler DDPM. `'maisi3d-rflow'` uses Rectified Flow scheduler, can be 33 times faster during inference.
# - The maisi version `'maisi3d-ddpm'` requires training images to be labeled with body regions (`"top_region_index"` and `"bottom_region_index"`), while `'maisi3d-rflow'` does not have such requirement. In other words, it is easier to prepare training data for `'maisi3d-rflow'`.
# - For the released model weights, `'maisi3d-rflow'` can generate images with better quality for head region and small output volumes, and comparable quality for other cases compared with `'maisi3d-ddpm'`.

# In[3]:


def main():
    """Main function to run the MAISI training pipeline."""
    args = parse_args()
    
    logger.info(f"Starting MAISI training pipeline with arguments: {vars(args)}")
    
    maisi_version = args.maisi_version
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
    
    # Display body region information
    if include_body_region:
        top_region, bottom_region = get_body_region_encoding(args.body_region)
        logger.info(f"Target body region: {args.body_region}")
        logger.info(f"  Top region encoding: {top_region}")
        logger.info(f"  Bottom region encoding: {bottom_region}")
        
        # Provide helpful information about body regions
        region_descriptions = {
            "head_neck": "Head and neck region (brain, skull, cervical spine)",
            "chest": "Chest region (lungs, heart, thoracic spine)",
            "abdomen": "Abdominal region (liver, kidneys, lumbar spine)",
            "lower_body": "Lower body region (pelvis, hips, legs)",
            "chest_abdomen": "Chest to abdomen (thorax and upper abdomen)"
        }
        logger.info(f"  Description: {region_descriptions[args.body_region]}")
    else:
        logger.info("Body region specification not required for this MAISI version")

    # ### Load and validate input data
    data_path = args.data_path
    
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    image_files = glob.glob(os.path.join(data_path, "*.nii.gz"))
    
    if not image_files:
        raise ValueError(f"No .nii.gz files found in {data_path}. Please check your data path.")
    
    datalist = {
        "training": [{"image": os.path.basename(f)} for f in image_files]
    }
    
    # Get dimensions from first image
    first_img = nib.load(image_files[0])
    image_dims = first_img.shape[:3]
    
    logger.info(f"Found {len(image_files)} medical images")
    logger.info(f"Using dimensions from first image: {image_dims}")
    logger.info(f"Training epochs: {args.epochs}")

    # Handle JSON-only mode
    if args.create_json_only:
        logger.info("Creating JSON metadata files only...")
        
        # Check if embedding directory exists
        embedding_dir = os.path.join(args.work_dir, "embeddings")
        if not os.path.exists(embedding_dir):
            raise ValueError(f"Embedding directory not found: {embedding_dir}. Run data creation step first.")
        
        # Load model definition to check if body region is needed
        with open(model_def_path, "r") as f:
            model_def = json.load(f)
        include_body_region = model_def["include_body_region"]
        
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
                    # Get body region encoding based on user specification
                    top_region, bottom_region = get_body_region_encoding(args.body_region)
                    data["top_region_index"] = top_region
                    data["bottom_region_index"] = bottom_region
                    logger.info(f"Body region '{args.body_region}': top={top_region}, bottom={bottom_region}")
                
                logger.info(f"data: {data}.")

                # Create the .json filename
                json_filename = gz_file + ".json"

                # Write the dictionary to the .json file
                with open(json_filename, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                logger.info(f"Save json file to {json_filename}")

        gz_files = list_gz_files(embedding_dir)
        create_json_files(gz_files)
        logger.info("Completed creating .json files for all embedding files.")
        return


    # ### Prepare data
    work_dir = args.work_dir
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    dataroot_dir = os.path.join(work_dir, "dataroot")
    if not os.path.isdir(dataroot_dir):
        os.makedirs(dataroot_dir)

    datalist_file = os.path.join(work_dir, "datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(datalist, f)

    logger.info("Copying medical images to working directory...")
    for i, source_file in enumerate(image_files):
        dest_file = os.path.join(dataroot_dir, os.path.basename(source_file))
        shutil.copy2(source_file, dest_file)
        logger.info(f"  Copied {i+1}/{len(image_files)}: {os.path.basename(source_file)}")
    logger.info(f"Successfully copied {len(image_files)} medical images")

    logger.info(f"Data prepared in: {dataroot_dir}")


    # ### Set up directories and configurations
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

    # Update model configuration
    epochs = args.epochs
    model_config_out["diffusion_unet_train"]["n_epochs"] = epochs
    
    # Configure inference parameters
    if "diffusion_unet_infer" in model_config_out:
        model_config_out["diffusion_unet_infer"]["num_images"] = args.num_images
    
    logger.info(f"Using {epochs} epochs for training")
    logger.info(f"Will generate {args.num_images} images during inference")
    if epochs < 100:
        logger.info("Note: For production, consider 100+ epochs for real data")

    model_config_filepath = os.path.join(work_dir, "config_maisi_diff_model.json")
    with open(model_config_filepath, "w") as f:
        json.dump(model_config_out, f, sort_keys=True, indent=4)

    # Update model definition for demo
    model_def_out["autoencoder_def"]["num_splits"] = args.num_splits
    model_def_filepath = os.path.join(work_dir, "config_maisi.json")
    with open(model_def_filepath, "w") as f:
        json.dump(model_def_out, f, sort_keys=True, indent=4)

    # Print files and folders under work_dir
    logger.info(f"files and folders under work_dir: {os.listdir(work_dir)}.")

    # Adjust based on the number of GPUs you want to use
    num_gpus = args.num_gpus
    logger.info(f"number of GPUs: {num_gpus}.")

    # ### Create .json files for embedding files (must be done after data creation step)
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
                # Get body region encoding based on user specification
                top_region, bottom_region = get_body_region_encoding(args.body_region)
                data["top_region_index"] = top_region
                data["bottom_region_index"] = bottom_region
                logger.info(f"Body region '{args.body_region}': top={top_region}, bottom={bottom_region}")
            
            logger.info(f"data: {data}.")

            # Create the .json filename
            json_filename = gz_file + ".json"

            # Write the dictionary to the .json file
            with open(json_filename, "w") as json_file:
                json.dump(data, json_file, indent=4)
            logger.info(f"Save json file to {json_filename}")

    logger.info("="*80)
    logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"✅ Copied {len(image_files)} medical images")
    logger.info(f"✅ Created configuration files")
    logger.info(f"✅ Downloaded autoencoder model")
    logger.info(f"✅ Working directory: {work_dir}")
    
    # Print the torchrun commands for manual execution
    print("\n" + "="*80)
    print("NEXT STEPS: Run these torchrun commands manually")
    print("="*80)
    
    # Prepare common arguments
    no_amp_arg = " --no_amp" if args.no_amp else ""
    
    print("\n# Step 1: Create training data")
    print(f"torchrun --nproc_per_node={num_gpus} --nnodes=1 \\")
    print(f"    -m scripts.diff_model_create_training_data \\")
    print(f"    --env_config {env_config_filepath} \\")
    print(f"    --model_config {model_config_filepath} \\")
    print(f"    --model_def {model_def_filepath} \\")
    print(f"    --num_gpus {num_gpus}{no_amp_arg}")
    
    print(f"\n# After step 1 completes, create metadata files:")
    print(f"python {sys.argv[0]} --data_path {data_path} --work_dir {work_dir} --create_json_only --body_region {args.body_region}")
    
    print(f"\n# Step 2: Train the model")
    print(f"torchrun --nproc_per_node={num_gpus} --nnodes=1 \\")
    print(f"    -m scripts.diff_model_train \\")
    print(f"    --env_config {env_config_filepath} \\")
    print(f"    --model_config {model_config_filepath} \\")
    print(f"    --model_def {model_def_filepath} \\")
    print(f"    --num_gpus {num_gpus}{no_amp_arg}")
    
    print(f"\n# Step 3: Run inference (generate {args.num_images} images)")
    print(f"torchrun --nproc_per_node={num_gpus} --nnodes=1 \\")
    print(f"    -m scripts.diff_model_infer \\")
    print(f"    --env_config {env_config_filepath} \\")
    print(f"    --model_config {model_config_filepath} \\")
    print(f"    --model_def {model_def_filepath} \\")
    print(f"    --num_gpus {num_gpus}{no_amp_arg}")
    
    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("="*80)
    print(f"• Run each command in sequence")
    print(f"• Wait for each step to complete before running the next")
    print(f"• After Step 1, JSON metadata files will be created automatically")
    print(f"• Results will be saved in: {env_config_out['output_dir']}")
    print(f"• Model checkpoints will be in: {env_config_out['model_dir']}")
    if num_gpus > 1:
        print(f"• Using {num_gpus} GPUs for distributed training")
    print("="*80)
    
    # Note about JSON files creation
    print(f"\n📝 After Step 1 completes, run this script again with a special flag to create JSON metadata:")
    print(f"python {sys.argv[0]} --data_path {data_path} --work_dir {work_dir} --create_json_only")
    
    logger.info("Setup complete. Follow the printed torchrun commands above.")


if __name__ == "__main__":
    main()

