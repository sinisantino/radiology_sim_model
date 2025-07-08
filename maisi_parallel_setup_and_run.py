#!/usr/bin/env python3
"""
MAISI Parallel Training Setup and Execution Script

This script automatically:
1. Checks environment and installs dependencies
2. Detects GPU configuration and validates setup
3. Sets up data paths (real or simulated)
4. Configures and runs parallel training with optimal parameters
5. Handles common errors (port conflicts, GPU mismatches)
6. Analyzes output and provides troubleshooting guidance

Usage:
    python maisi_parallel_setup_and_run.py [options]

Author: GitHub Copilot
Date: July 2025
"""

import sys
import subprocess
import os
import glob
import time
import socket
import argparse
from pathlib import Path
from datetime import datetime
import json


class MAISIParallelTrainer:
    """Main class for MAISI parallel training setup and execution."""
    
    def __init__(self, args):
        self.args = args
        self.config = {}
        self.training_output = []
        self.data_files = []
        
    def print_header(self, title, char="="):
        """Print formatted section header."""
        print(f"\n{title}")
        print(char * len(title))
        
    def check_environment(self):
        """Check Python environment and install dependencies."""
        self.print_header("🔍 ENVIRONMENT CHECK")
        
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Running in virtual environment")
        else:
            print("⚠️  Not in virtual environment - consider using one")
            
        # Install required packages
        if not self.args.skip_install:
            self.print_header("📦 INSTALLING REQUIRED PACKAGES")
            packages = [
                "torch",
                "torchvision", 
                "monai-weekly[pillow,tqdm]",
                "nibabel",
                "numpy",
                "scipy",
                "psutil"
            ]
            
            for package in packages:
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "-q", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
                    if not self.args.continue_on_error:
                        sys.exit(1)
        else:
            print("⚠️  Skipping package installation (--skip-install)")
            
    def check_gpu_availability(self):
        """Check GPU availability and configuration."""
        self.print_header("🖥️  GPU AVAILABILITY CHECK")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✅ CUDA is available! Found {gpu_count} GPU(s)")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                # Test GPU functionality
                print(f"\n🧪 Testing GPU functionality...")
                x = torch.randn(1000, 1000).cuda()
                y = torch.matmul(x, x.T)
                print(f"✅ GPU computation test passed!")
                del x, y  # Free memory
                
                self.config['gpu_count'] = gpu_count
                self.config['cuda_available'] = True
                
            else:
                print("❌ CUDA not available. Training will be slow on CPU.")
                print("   Make sure you have:")
                print("   • NVIDIA GPU installed")
                print("   • CUDA drivers installed") 
                print("   • PyTorch with CUDA support")
                self.config['gpu_count'] = 0
                self.config['cuda_available'] = False
                
        except ImportError:
            print("❌ PyTorch not available. Please install it first.")
            self.config['gpu_count'] = 0
            self.config['cuda_available'] = False
            if not self.args.continue_on_error:
                sys.exit(1)
                
        # System resources
        try:
            import psutil
            print(f"\n📊 SYSTEM RESOURCES:")
            print(f"CPU cores: {psutil.cpu_count()}")
            print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
            print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        except ImportError:
            print("psutil not available for system info")
            
    def setup_data_paths(self):
        """Set up data paths and create test data if needed."""
        self.print_header("📁 DATA DIRECTORY CONFIGURATION")
        
        # Use command line data path or search common locations
        if self.args.data_path:
            data_path = os.path.expanduser(self.args.data_path)
            if os.path.exists(data_path):
                nii_files = glob.glob(os.path.join(data_path, "*.nii.gz"))
                if nii_files:
                    print(f"✅ Using provided data path: {data_path}")
                    print(f"   Found {len(nii_files)} .nii.gz files")
                    self.config['data_path'] = data_path
                    self.config['use_real_data'] = True
                    self.data_files = nii_files
                    return
                else:
                    print(f"❌ No .nii.gz files found in provided path: {data_path}")
            else:
                print(f"❌ Provided data path does not exist: {data_path}")
        
        # Search common data directories
        common_paths = [
            "/home/santino/medical_data",
            "/home/santino/data/medical",
            "/data/medical",
            "/mnt/medical_data",
            "/shared/medical_data",
            "~/Documents/medical_data",
            "~/Desktop/medical_data"
        ]
        
        print("🔍 Searching for medical data in common locations...")
        found_data = False
        
        for path in common_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                nii_files = glob.glob(os.path.join(expanded_path, "*.nii.gz"))
                if nii_files:
                    print(f"✅ Found {len(nii_files)} .nii.gz files in: {expanded_path}")
                    self.config['data_path'] = expanded_path
                    self.config['use_real_data'] = True
                    self.data_files = nii_files
                    found_data = True
                    break
                else:
                    print(f"📂 Directory exists but no .nii.gz files: {expanded_path}")
            else:
                print(f"❌ Directory not found: {expanded_path}")
                
        if not found_data:
            print(f"\n⚠️  NO MEDICAL DATA FOUND!")
            if self.args.use_simulated_data:
                self.create_simulated_data()
            else:
                print("Please either:")
                print("1. Provide data path with --data-path /your/path")
                print("2. Place .nii.gz files in a common directory")
                print("3. Use --use-simulated-data for testing")
                sys.exit(1)
                
    def create_simulated_data(self):
        """Create simulated medical data for testing."""
        self.print_header("🏗️  CREATING TEST DATA DIRECTORY")
        
        test_data_dir = os.path.join(os.getcwd(), "test_medical_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        try:
            import nibabel as nib
            import numpy as np
            
            print(f"📁 Creating test directory: {test_data_dir}")
            
            # Create dummy medical images
            num_images = self.args.num_test_images
            for i in range(num_images):
                # Create random 3D image data (simulating medical scan)
                img_data = np.random.randint(0, 255, (64, 64, 32), dtype=np.uint8)
                
                # Create NIfTI image
                nii_img = nib.Nifti1Image(img_data, affine=np.eye(4))
                
                # Save as .nii.gz file
                filename = f"test_patient_{i+1:03d}.nii.gz"
                filepath = os.path.join(test_data_dir, filename)
                nib.save(nii_img, filepath)
                
                print(f"✅ Created: {filename}")
                
            # Update configuration
            self.config['data_path'] = test_data_dir
            self.config['use_real_data'] = False
            self.data_files = glob.glob(os.path.join(test_data_dir, "*.nii.gz"))
            
            print(f"\n🎉 Test data ready! Created {len(self.data_files)} files in {test_data_dir}")
            
        except ImportError:
            print("❌ nibabel not available. Cannot create test data.")
            print("   Please install nibabel or provide real medical data.")
            sys.exit(1)
            
    def check_port_availability(self):
        """Check for available ports and resolve conflicts."""
        self.print_header("🔍 PORT AVAILABILITY CHECK")
        
        def check_port(port):
            """Check if a port is in use."""
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    return result == 0  # 0 means connection successful (port in use)
            except:
                return False
                
        ports_to_check = [12355, 29500, 29501, 23456, 25000]
        available_port = None
        
        for port in ports_to_check:
            if check_port(port):
                print(f"   ❌ Port {port}: IN USE")
            else:
                print(f"   ✅ Port {port}: Available")
                if available_port is None:
                    available_port = port
                    
        if available_port:
            self.config['master_port'] = available_port
            print(f"\n💡 Will use port {available_port} for training")
        else:
            print("\n⚠️  All checked ports are in use. Will try default port.")
            self.config['master_port'] = 29500
            
    def cleanup_existing_processes(self):
        """Clean up existing torchrun processes."""
        self.print_header("🧹 CLEANING UP EXISTING PROCESSES")
        
        try:
            import psutil
            
            # Find and kill torchrun processes
            killed_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'torchrun' in ' '.join(proc.info['cmdline'] or []):
                        proc.terminate()
                        killed_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            if killed_processes:
                print(f"✅ Terminated {len(killed_processes)} torchrun processes")
                time.sleep(3)  # Wait for cleanup
            else:
                print("✅ No existing torchrun processes found")
                
        except ImportError:
            # Fallback method using pkill
            try:
                subprocess.run(["pkill", "-f", "torchrun"], 
                             capture_output=True, timeout=10)
                print("✅ Cleaned up processes with pkill")
                time.sleep(3)
            except:
                print("⚠️  Could not clean up processes (may not exist)")
                
    def configure_training_parameters(self):
        """Configure all training parameters."""
        self.print_header("⚙️  TRAINING CONFIGURATION")
        
        # Use actual GPU count or override
        gpu_count = self.args.nproc_per_node or self.config.get('gpu_count', 1)
        if gpu_count > self.config.get('gpu_count', 0):
            print(f"⚠️  Requested {gpu_count} GPUs but only {self.config.get('gpu_count', 0)} available")
            gpu_count = max(1, self.config.get('gpu_count', 1))
            
        # Configure parameters
        training_config = {
            # Torchrun parameters
            "nproc_per_node": gpu_count,
            "nnodes": 1,
            "master_port": self.config.get('master_port', 29500),
            
            # MAISI script parameters
            "data_path": self.config['data_path'],
            "use_real_data": self.config['use_real_data'],
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "model_version": self.args.model_version,
            "base_seed": self.args.base_seed,
            
            # Script location
            "script_path": self.args.script_path
        }
        
        self.config.update(training_config)
        
        print("Configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
            
        # Build torchrun command
        self.build_torchrun_command()
        
    def build_torchrun_command(self):
        """Build the torchrun command with all parameters."""
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config['nproc_per_node']}",
            f"--nnodes={self.config['nnodes']}",
            f"--master-port={self.config['master_port']}",
            self.config["script_path"]
        ]
        
        # Add MAISI script arguments
        if self.config["use_real_data"]:
            cmd.extend(["--real-data", "--data-path", self.config["data_path"]])
            
        cmd.extend([
            "--epochs", str(self.config["epochs"]),
            "--batch-size", str(self.config["batch_size"]),
            "--model-version", self.config["model_version"],
            "--base-seed", str(self.config["base_seed"])
        ])
        
        self.config['torchrun_command'] = cmd
        
        print(f"\n🚀 TORCHRUN COMMAND:")
        print(" ".join(cmd))
        
        print(f"\n💡 EXPECTED BEHAVIOR:")
        print(f"• Will use {self.config['nproc_per_node']} GPU(s)")
        print(f"• Will generate {self.config['nproc_per_node']} unique medical images")
        print(f"• Training will run for {self.config['epochs']} epochs")
        print(f"• Each GPU gets batch_size={self.config['batch_size']} images")
        print(f"• Using {'real' if self.config['use_real_data'] else 'simulated'} data")
        print(f"• Using port {self.config['master_port']}")
        
    def run_training(self):
        """Execute the MAISI parallel training."""
        self.print_header("🚀 STARTING MAISI PARALLEL TRAINING")
        
        # Check if script exists
        script_path = self.config["script_path"]
        if not os.path.exists(script_path):
            print(f"❌ ERROR: Script not found: {script_path}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Files in directory: {os.listdir('.')}")
            print(f"\n   Please make sure {script_path} is in your current directory.")
            return False
            
        print(f"✅ Found script: {script_path}")
        print(f"\n📋 Command: {' '.join(self.config['torchrun_command'])}")
        print(f"\n⏰ Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            if self.config.get('cuda_available'):
                gpu_list = ",".join(str(i) for i in range(self.config['nproc_per_node']))
                env["CUDA_VISIBLE_DEVICES"] = gpu_list
                
            # Start the process
            process = subprocess.Popen(
                self.config['torchrun_command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # Read output in real-time
            for line in process.stdout:
                print(line, end='')
                self.training_output.append(line)
                
            # Wait for completion
            return_code = process.wait()
            
            print("=" * 60)
            print(f"⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if return_code == 0:
                print("🎉 SUCCESS: Training completed successfully!")
                return True
            else:
                print(f"❌ FAILED: Training failed with return code {return_code}")
                return False
                
        except FileNotFoundError:
            print("❌ ERROR: torchrun not found. Please install PyTorch properly.")
            return False
        except Exception as e:
            print(f"❌ ERROR: Unexpected error during training: {e}")
            return False
            
    def analyze_output(self):
        """Analyze training output and provide troubleshooting guidance."""
        self.print_header("🔍 OUTPUT ANALYSIS")
        
        if not self.training_output:
            print("⚠️  No training output available to analyze")
            return
            
        output_text = "".join(self.training_output)
        
        # Check for common error patterns
        error_patterns = {
            "No .nii.gz files found": "Data path issue - no medical images found",
            "CUDA out of memory": "GPU memory insufficient - reduce batch size",
            "ConnectionTimeout": "Multi-node communication issue",
            "FileNotFoundError": "Missing file or script",
            "ImportError": "Missing Python package",
            "Permission denied": "File permission issue",
            "Address already in use": "Port conflict in multi-node setup",
            "EADDRINUSE": "Port conflict - address already in use",
            "device ordinal": "GPU count mismatch or invalid GPU access"
        }
        
        errors_found = []
        for pattern, description in error_patterns.items():
            if pattern.lower() in output_text.lower():
                errors_found.append((pattern, description))
                
        if errors_found:
            print("❌ ERRORS DETECTED:")
            for pattern, description in errors_found:
                print(f"   • {pattern}: {description}")
                
            self.print_troubleshooting_guide(errors_found)
        else:
            print("✅ No obvious errors detected in output")
            
        # Check for success indicators
        success_patterns = [
            "Training completed successfully",
            "inference completed",
            "Generated images saved",
            "MAISI PARALLEL COMPUTING CONFIGURATION"
        ]
        
        success_found = any(pattern.lower() in output_text.lower() 
                           for pattern in success_patterns)
        
        if success_found:
            print("\n🎉 SUCCESS INDICATORS FOUND:")
            for pattern in success_patterns:
                if pattern.lower() in output_text.lower():
                    print(f"   ✅ {pattern}")
                    
    def print_troubleshooting_guide(self, errors_found):
        """Print specific troubleshooting guidance based on errors."""
        print(f"\n🔧 TROUBLESHOOTING GUIDE:")
        
        error_types = [error[0] for error in errors_found]
        
        if any("No .nii.gz files found" in error for error in error_types):
            print(f"📁 Data Path Issue:")
            print(f"   • Check data path: {self.config.get('data_path', 'Not set')}")
            print(f"   • Verify files exist: ls {self.config.get('data_path', '')}/*.nii.gz")
            print(f"   • Use --data-path to specify correct path")
            print(f"   • Or use --use-simulated-data for testing")
            
        if any("CUDA out of memory" in error for error in error_types):
            print(f"💾 Memory Issue:")
            print(f"   • Reduce batch size: --batch-size 1")
            print(f"   • Use fewer GPUs: --nproc-per-node 1")
            print(f"   • Close other GPU applications")
            
        if any("EADDRINUSE" in error or "Address already in use" in error for error in error_types):
            print(f"🔌 Port Conflict:")
            print(f"   • Use different port: --master-port 29501")
            print(f"   • Kill existing processes: pkill -f torchrun")
            print(f"   • Wait before retrying")
            
        if any("device ordinal" in error for error in error_types):
            print(f"🖥️  GPU Issue:")
            print(f"   • Check GPU count: nvidia-smi")
            print(f"   • Match nproc_per_node to actual GPU count")
            print(f"   • Current setting: {self.config.get('nproc_per_node', 'Not set')}")
            
    def inspect_output_directory(self):
        """Inspect generated output directory and files."""
        self.print_header("📂 OUTPUT DIRECTORY INSPECTION")
        
        output_dir = "./output_work_dir"
        
        if os.path.exists(output_dir):
            print(f"✅ Found output directory: {output_dir}")
            
            # Show directory structure
            print(f"\n📁 Directory Structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
                    
            # Look for generated images
            generated_images = glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True)
            
            if generated_images:
                print(f"\n🖼️  GENERATED MEDICAL IMAGES:")
                print(f"Found {len(generated_images)} generated images:")
                
                for img_path in generated_images:
                    rel_path = os.path.relpath(img_path, output_dir)
                    file_size = os.path.getsize(img_path) / 1024**2  # MB
                    print(f"   📄 {rel_path} ({file_size:.1f} MB)")
                    
                print(f"\n💡 VIEWING YOUR RESULTS:")
                print("You can view these medical images with:")
                print("• 3D Slicer: https://www.slicer.org/")
                print("• ITK-SNAP: http://www.itksnap.org/")
                print("• FSL tools: fsleyes your_image.nii.gz")
                print("• Python: nibabel.load('your_image.nii.gz')")
                
            else:
                print("❌ No generated .nii.gz images found")
                print("   Training may not have completed successfully")
                
            # Summary
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                           for dirpath, dirnames, filenames in os.walk(output_dir)
                           for filename in filenames) / 1024**2
            print(f"\n📊 SUMMARY:")
            print(f"• Total output size: {total_size:.1f} MB")
            print(f"• Output location: {os.path.abspath(output_dir)}")
            
        else:
            print(f"❌ Output directory not found: {output_dir}")
            print("   Training may not have started or failed early")
            
    def save_configuration(self):
        """Save configuration to JSON file for reference."""
        config_file = "maisi_training_config.json"
        
        # Prepare config for JSON serialization
        save_config = self.config.copy()
        if 'torchrun_command' in save_config:
            save_config['torchrun_command'] = ' '.join(save_config['torchrun_command'])
            
        try:
            with open(config_file, 'w') as f:
                json.dump(save_config, f, indent=2)
            print(f"\n💾 Configuration saved to: {config_file}")
        except Exception as e:
            print(f"⚠️  Could not save configuration: {e}")
            
    def run_complete_workflow(self):
        """Run the complete MAISI parallel training workflow."""
        print("🚀 MAISI PARALLEL TRAINING - AUTOMATED SETUP AND EXECUTION")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Environment setup
            self.check_environment()
            
            # Step 2: GPU detection
            self.check_gpu_availability()
            
            # Step 3: Data setup
            self.setup_data_paths()
            
            # Step 4: Port checking
            self.check_port_availability()
            
            # Step 5: Process cleanup
            self.cleanup_existing_processes()
            
            # Step 6: Configuration
            self.configure_training_parameters()
            
            # Step 7: Save configuration
            self.save_configuration()
            
            # Step 8: Run training
            success = self.run_training()
            
            # Step 9: Analyze output
            self.analyze_output()
            
            # Step 10: Inspect results
            self.inspect_output_directory()
            
            # Final summary
            self.print_header("🎉 WORKFLOW COMPLETE")
            if success:
                print("✅ MAISI parallel training completed successfully!")
                print("📁 Check the output_work_dir for generated medical images")
            else:
                print("❌ Training encountered errors - check output above")
                print("💡 Try troubleshooting suggestions or manual command")
                
            print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Workflow interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error in workflow: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MAISI Parallel Training Setup and Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with auto-detection
  python maisi_parallel_setup_and_run.py
  
  # Use specific data path
  python maisi_parallel_setup_and_run.py --data-path /path/to/medical/data
  
  # Use simulated data for testing
  python maisi_parallel_setup_and_run.py --use-simulated-data
  
  # Customize training parameters
  python maisi_parallel_setup_and_run.py --epochs 10 --batch-size 2 --nproc-per-node 4
  
  # Skip dependency installation
  python maisi_parallel_setup_and_run.py --skip-install
        """
    )
    
    # Data configuration
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to medical imaging data directory")
    parser.add_argument("--use-simulated-data", action="store_true",
                       help="Create and use simulated data for testing")
    parser.add_argument("--num-test-images", type=int, default=3,
                       help="Number of test images to create (default: 3)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU (default: 1)")
    parser.add_argument("--model-version", type=str, default="maisi3d-rflow",
                       help="MAISI model version (default: maisi3d-rflow)")
    parser.add_argument("--base-seed", type=int, default=42,
                       help="Base seed for random number generation (default: 42)")
    
    # Parallel configuration
    parser.add_argument("--nproc-per-node", type=int, default=None,
                       help="Number of processes per node (auto-detect by default)")
    parser.add_argument("--script-path", type=str, default="maisi_train_diff_unet_parallel.py",
                       help="Path to MAISI parallel training script")
    
    # Setup options
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue workflow even if some steps fail")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create and run trainer
    trainer = MAISIParallelTrainer(args)
    trainer.run_complete_workflow()


if __name__ == "__main__":
    main()
