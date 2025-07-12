#!/usr/bin/env python3
"""
MAISI Medical Image Generation Script
Converted from maisi_inference_tutorial.ipynb for easier command-line usage.
Generate synthetic medical images for various anatomical structures.
"""

import argparse
import json
import os
import tempfile
import sys

# Available anatomical structures from label_dict.json
AVAILABLE_ANATOMIES = [
    "liver", "spleen", "pancreas", "right kidney", "left kidney", "aorta", 
    "inferior vena cava", "right adrenal gland", "left adrenal gland", "gallbladder",
    "esophagus", "stomach", "duodenum", "bladder", "portal vein and splenic vein",
    "small bowel", "brain", "lung tumor", "pancreatic tumor", "hepatic vessel",
    "hepatic tumor", "colon cancer primaries", "left lung upper lobe", "left lung lower lobe",
    "right lung upper lobe", "right lung middle lobe", "right lung lower lobe",
    "vertebrae L5", "vertebrae L4", "vertebrae L3", "vertebrae L2", "vertebrae L1",
    "vertebrae T12", "vertebrae T11", "vertebrae T10", "vertebrae T9", "vertebrae T8",
    "vertebrae T7", "vertebrae T6", "vertebrae T5", "vertebrae T4", "vertebrae T3",
    "vertebrae T2", "vertebrae T1", "vertebrae C7", "vertebrae C6", "vertebrae C5",
    "vertebrae C4", "vertebrae C3", "vertebrae C2", "vertebrae C1", "trachea",
    "left iliac artery", "right iliac artery", "left iliac vena", "right iliac vena",
    "colon", "left rib 1", "left rib 2", "left rib 3", "left rib 4", "left rib 5",
    "left rib 6", "left rib 7", "left rib 8", "left rib 9", "left rib 10",
    "left rib 11", "left rib 12", "right rib 1", "right rib 2", "right rib 3",
    "right rib 4", "right rib 5", "right rib 6", "right rib 7", "right rib 8",
    "right rib 9", "right rib 10", "right rib 11", "right rib 12", "left humerus",
    "right humerus", "left scapula", "right scapula", "left clavicula", "right clavicula",
    "left femur", "right femur", "left hip", "right hip", "sacrum", "left gluteus maximus",
    "right gluteus maximus", "left gluteus medius", "right gluteus medius",
    "left gluteus minimus", "right gluteus minimus", "left autochthon", "right autochthon",
    "left iliopsoas", "right iliopsoas", "left atrial appendage", "brachiocephalic trunk",
    "left brachiocephalic vein", "right brachiocephalic vein", "left common carotid artery",
    "right common carotid artery", "costal cartilages", "heart", "left kidney cyst",
    "right kidney cyst", "prostate", "pulmonary vein", "skull", "spinal cord",
    "sternum", "left subclavian artery", "right subclavian artery", "superior vena cava",
    "thyroid gland", "vertebrae S1", "bone lesion", "airway"
]

# Available body regions (commonly used combinations)
AVAILABLE_BODY_REGIONS = [
    "head", "neck", "chest", "abdomen", "pelvis", "head_neck", "chest_abdomen", 
    "abdomen_pelvis", "chest_abdomen_pelvis"
]

# Mapping of anatomies to their typical body regions
ANATOMY_TO_REGION = {
    # Head region
    "brain": "head", "skull": "head",
    
    # Neck region  
    "trachea": "neck", "thyroid gland": "neck", "esophagus": "neck",
    "left common carotid artery": "neck", "right common carotid artery": "neck",
    "left subclavian artery": "neck", "right subclavian artery": "neck",
    
    # Chest region
    "heart": "chest", "left atrial appendage": "chest", "pulmonary vein": "chest",
    "superior vena cava": "chest", "brachiocephalic trunk": "chest",
    "left brachiocephalic vein": "chest", "right brachiocephalic vein": "chest",
    "left lung upper lobe": "chest", "left lung lower lobe": "chest",
    "right lung upper lobe": "chest", "right lung middle lobe": "chest", 
    "right lung lower lobe": "chest", "lung tumor": "chest", "sternum": "chest",
    "costal cartilages": "chest", "airway": "chest",
    
    # Abdomen region
    "liver": "abdomen", "spleen": "abdomen", "pancreas": "abdomen", 
    "right kidney": "abdomen", "left kidney": "abdomen", "aorta": "abdomen",
    "inferior vena cava": "abdomen", "right adrenal gland": "abdomen", 
    "left adrenal gland": "abdomen", "gallbladder": "abdomen", "stomach": "abdomen",
    "duodenum": "abdomen", "portal vein and splenic vein": "abdomen", 
    "small bowel": "abdomen", "pancreatic tumor": "abdomen", "hepatic vessel": "abdomen",
    "hepatic tumor": "abdomen", "colon": "abdomen", "left kidney cyst": "abdomen",
    "right kidney cyst": "abdomen",
    
    # Pelvis region
    "bladder": "pelvis", "prostate": "pelvis", "left hip": "pelvis", "right hip": "pelvis",
    "sacrum": "pelvis", "left gluteus maximus": "pelvis", "right gluteus maximus": "pelvis",
    "left gluteus medius": "pelvis", "right gluteus medius": "pelvis",
    "left gluteus minimus": "pelvis", "right gluteus minimus": "pelvis",
    "left iliopsoas": "pelvis", "right iliopsoas": "pelvis", "left femur": "pelvis",
    "right femur": "pelvis", "colon cancer primaries": "pelvis"
}

def get_recommended_body_region(anatomy):
    """Get the recommended body region for a given anatomy."""
    return ANATOMY_TO_REGION.get(anatomy, "abdomen")

def print_available_options():
    """Print all available anatomies and body regions."""
    print("\n" + "="*60)
    print("MAISI MEDICAL IMAGE GENERATOR - Available Options")
    print("="*60)
    
    print("\nAVAILABLE BODY REGIONS:")
    print("-" * 30)
    for i, region in enumerate(AVAILABLE_BODY_REGIONS, 1):
        print(f"{i:2d}. {region}")
    
    print("\nAVAILABLE ANATOMICAL STRUCTURES:")
    print("-" * 40)
    
    # Group by body region for better organization
    regions_groups = {
        "head": [], "neck": [], "chest": [], "abdomen": [], "pelvis": [], "other": []
    }
    
    for anatomy in AVAILABLE_ANATOMIES:
        region = ANATOMY_TO_REGION.get(anatomy, "other")
        if region in ["head_neck", "chest_abdomen", "abdomen_pelvis", "chest_abdomen_pelvis"]:
            region = "other"
        regions_groups[region].append(anatomy)
    
    for region, anatomies in regions_groups.items():
        if anatomies:
            print(f"\n{region.upper()} region:")
            for i, anatomy in enumerate(sorted(anatomies), 1):
                print(f"  {i:2d}. {anatomy}")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("-" * 20)
    print("# Generate prostate image:")
    print("python generate_medical_images.py --anatomy prostate")
    print("\n# Generate liver image with custom settings:")
    print("python generate_medical_images.py --anatomy liver --body_region abdomen --num_samples 2")
    print("\n# Generate brain image:")
    print("python generate_medical_images.py --anatomy brain --body_region head")
    print("\n# High quality generation (more inference steps):")
    print("python generate_medical_images.py --anatomy liver --num_inference_steps 100")
    print("="*60)

def check_and_install_packages():
    """Check and install required packages if needed."""
    try:
        import monai
        import torch
        import matplotlib
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Installing required packages...")
        os.system("pip install -q 'monai-weekly[nibabel, tqdm]' matplotlib")
        


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic medical images for various anatomical structures using MAISI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --anatomy prostate
  %(prog)s --anatomy liver --body_region abdomen --num_samples 2
  %(prog)s --anatomy brain --body_region head
  %(prog)s --anatomy liver --num_inference_steps 100  # High quality
  %(prog)s --list-options  # Show all available options
        """)
    
    parser.add_argument('--anatomy', default='prostate', 
                       help='Anatomy to generate (default: prostate). Use --list-options to see all available anatomies.')
    parser.add_argument('--body_region', 
                       help='Body region (auto-detected if not specified). Use --list-options to see available regions.')
    parser.add_argument('--num_samples', type=int, default=1, 
                       help='Number of samples to generate (default: 1)')
    parser.add_argument('--output_size', nargs=3, type=int, default=[256, 256, 256], 
                       help='Output size in voxels (default: 256 256 256)')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.5, 1.5, 2.0],
                       help='Voxel spacing in mm (default: 1.5 1.5 2.0)')
    parser.add_argument('--output_dir', default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--maisi_version', default='maisi3d-rflow', 
                       choices=['maisi3d-ddpm', 'maisi3d-rflow'], 
                       help='MAISI version (default: maisi3d-rflow - faster)')
    parser.add_argument('--list-options', action='store_true',
                       help='List all available anatomies and body regions, then exit')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results (default: random each time)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use fixed seed (0) for reproducible results')
    parser.add_argument('--num_inference_steps', type=int, default=30,
                       help='Number of inference steps for diffusion (default: 30, higher=better quality but slower)')
    
    args = parser.parse_args()
    
    # Handle list options request
    if args.list_options:
        print_available_options()
        return
    
    # Validate anatomy
    if args.anatomy not in AVAILABLE_ANATOMIES:
        print(f"Error: '{args.anatomy}' is not a valid anatomy.")
        print("Use --list-options to see all available anatomies.")
        sys.exit(1)
    
    # Auto-detect body region if not specified
    if args.body_region is None:
        args.body_region = get_recommended_body_region(args.anatomy)
        print(f"Auto-detected body region: {args.body_region}")
    
    # Validate body region
    if args.body_region not in AVAILABLE_BODY_REGIONS:
        print(f"Error: '{args.body_region}' is not a valid body region.")
        print("Use --list-options to see all available body regions.")
        sys.exit(1)
    
    # Check and install packages
    check_and_install_packages()
    
    # Now import after ensuring packages are installed
    import monai
    import torch
    from monai.apps import download_url
    from monai.config import print_config
    from monai.transforms import LoadImage, Orientation
    from monai.utils import set_determinism
    from scripts.sample import LDMSampler, check_input
    from scripts.utils import define_instance
    from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image
    from scripts.diff_model_setting import setup_logging
    
    print("MAISI Medical Image Generation")
    print("=" * 50)
    print(f"Anatomy: {args.anatomy}")
    print(f"Body Region: {args.body_region}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Output Size: {args.output_size}")
    print(f"Voxel Spacing: {args.spacing}")
    print(f"MAISI Version: {args.maisi_version}")
    print("=" * 50)
    print_config()
    
    logger = setup_logging("script")
    
    # Set up MAISI version
    maisi_version = args.maisi_version
    if maisi_version == "maisi3d-ddpm":
        model_def_path = "./configs/config_maisi3d-ddpm.json"
    elif maisi_version == "maisi3d-rflow":
        model_def_path = "./configs/config_maisi3d-rflow.json"
    else:
        raise ValueError(f"maisi_version must be 'maisi3d-ddpm' or 'maisi3d-rflow', got {maisi_version}")
    
    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    include_body_region = model_def["include_body_region"]
    logger.info(f"MAISI version: {maisi_version}, body_region required: {include_body_region}")
    
    # Setup data directory
    os.environ["MONAI_DATA_DIRECTORY"] = "temp_work_dir_inference_demo"
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    
    # Download required files
    logger.info("Downloading model weights and configuration files...")
    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
        },
        {
            "path": "models/mask_generation_autoencoder.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/mask_generation_autoencoder.pt",
        },
        {
            "path": "models/mask_generation_diffusion_unet.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_mask_generation_diffusion_unet_v2.pt",
        },
        {
            "path": "configs/all_anatomy_size_condtions.json",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_anatomy_size_condtions.json",
        },
        {
            "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_masks_flexible_size_and_spacing_4000.zip",
        },
    ]
    
    if maisi_version == "maisi3d-rflow":
        files += [
            {
                "path": "models/diff_unet_3d_rflow.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/diff_unet_ckpt_rflow_epoch19350.pt",
            },
            {
                "path": "models/controlnet_3d_rflow.pt",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/controlnet_rflow_epoch60.pt",
            },
            {
                "path": "configs/candidate_masks_flexible_size_and_spacing_4000.json",
                "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/candidate_masks_flexible_size_and_spacing_4000.json",
            },
        ]
    
    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        download_url(url=file["url"], filepath=file["path"])
    
    # Setup arguments
    args_ns = argparse.Namespace()
    
    # Load environment settings
    if maisi_version == "maisi3d-ddpm":
        environment_file = "./configs/environment_maisi3d-ddpm.json"
    elif maisi_version == "maisi3d-rflow":
        environment_file = "./configs/environment_maisi3d-rflow.json"
    
    with open(environment_file, "r") as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        setattr(args_ns, k, val)
    
    # Load model definition
    for k, v in model_def.items():
        setattr(args_ns, k, v)
    
    # Set inference configuration
    setattr(args_ns, "num_output_samples", args.num_samples)
    setattr(args_ns, "body_region", [args.body_region])
    setattr(args_ns, "anatomy_list", [args.anatomy])
    setattr(args_ns, "controllable_anatomy_size", [])
    setattr(args_ns, "num_inference_steps", args.num_inference_steps)
    setattr(args_ns, "mask_generation_num_inference_steps", 1000)
    setattr(args_ns, "output_size", args.output_size)
    setattr(args_ns, "image_output_ext", ".nii.gz")
    setattr(args_ns, "label_output_ext", ".nii.gz")
    setattr(args_ns, "spacing", args.spacing)
    setattr(args_ns, "autoencoder_sliding_window_infer_size", [48, 48, 48])
    setattr(args_ns, "autoencoder_sliding_window_infer_overlap", 0.6666)
    setattr(args_ns, "modality", 1)
    setattr(args_ns, "output_dir", args.output_dir)
    
    # Validate inputs
    check_input(
        args_ns.body_region,
        args_ns.anatomy_list,
        args_ns.label_dict_json,
        args_ns.output_size,
        args_ns.spacing,
        args_ns.controllable_anatomy_size,
    )
    
    # Set deterministic seed
    if args.deterministic:
        seed = 0
        print("Using deterministic mode (seed=0) for reproducible results")
    elif args.seed is not None:
        seed = args.seed
        print(f"Using custom seed: {seed}")
    else:
        import random
        import time
        seed = int(time.time()) % 10000  # Use current time as seed
        print(f"Using random seed: {seed}")
    
    set_determinism(seed=seed)
    args_ns.random_seed = seed
    
    # Initialize networks and load weights
    logger.info("Loading model weights...")
    latent_shape = [args_ns.latent_channels, args_ns.output_size[0] // 4, args_ns.output_size[1] // 4, args_ns.output_size[2] // 4]
    
    noise_scheduler = define_instance(args_ns, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args_ns, "mask_generation_noise_scheduler")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    autoencoder = define_instance(args_ns, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(args_ns.trained_autoencoder_path, weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)
    
    diffusion_unet = define_instance(args_ns, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(args_ns.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=True)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)
    
    controlnet = define_instance(args_ns, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args_ns.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)
    
    mask_generation_autoencoder = define_instance(args_ns, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_generation_autoencoder = torch.load(args_ns.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)
    
    mask_generation_diffusion_unet = define_instance(args_ns, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_generation_diffusion_unet = torch.load(args_ns.trained_mask_generation_diffusion_path, weights_only=True)
    mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
    mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]
    
    logger.info("All model weights loaded successfully.")
    
    # Create LDM Sampler
    ldm_sampler = LDMSampler(
        args_ns.body_region,
        args_ns.anatomy_list,
        args_ns.all_mask_files_json,
        args_ns.all_anatomy_size_conditions_json,
        args_ns.all_mask_files_base_dir,
        args_ns.label_dict_json,
        args_ns.label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        args_ns.mask_generation_latent_shape,
        args_ns.output_size,
        args_ns.output_dir,
        args_ns.controllable_anatomy_size,
        image_output_ext=args_ns.image_output_ext,
        label_output_ext=args_ns.label_output_ext,
        spacing=args_ns.spacing,
        modality=args_ns.modality,
        num_inference_steps=args_ns.num_inference_steps,
        mask_generation_num_inference_steps=args_ns.mask_generation_num_inference_steps,
        random_seed=args_ns.random_seed,
        autoencoder_sliding_window_infer_size=args_ns.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args_ns.autoencoder_sliding_window_infer_overlap,
    )
    
    # Generate images
    logger.info(f"Generating {args.num_samples} synthetic {args.anatomy} image(s)...")
    logger.info(f"Output will be saved to: {args_ns.output_dir}")
    output_filenames = ldm_sampler.sample_multiple_images(args_ns.num_output_samples)
    logger.info("Image generation completed!")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Successfully generated {len(output_filenames)} {args.anatomy} image/mask pair(s)")
    print(f"Anatomy: {args.anatomy}")
    print(f"Body Region: {args.body_region}")
    print(f"Output Size: {args.output_size} voxels")
    print(f"Voxel Spacing: {args.spacing} mm")
    print()
    print("Generated files:")
    for i, (img_file, mask_file) in enumerate(output_filenames):
        print(f"  Sample {i+1}:")
        print(f"    Image: {img_file}")
        print(f"    Mask:  {mask_file}")
    print(f"\nAll files saved in: {args_ns.output_dir}")
    print(f"Use medical imaging software (e.g., 3D Slicer, ITK-SNAP) to view .nii.gz files")
    print("=" * 60)

if __name__ == "__main__":
    main()
