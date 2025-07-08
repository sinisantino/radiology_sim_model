#!/usr/bin/env python3
"""
Quick MAISI Parallel Training Launcher

This is a simple wrapper script that demonstrates how to use 
maisi_parallel_setup_and_run.py with your 2-GPU setup.

Usage:
    python quick_launch_maisi.py
"""

import subprocess
import sys
import os


def main():
    print("🚀 MAISI Parallel Training - Quick Launch for 2 GPUs")
    print("=" * 60)
    
    # Check if main script exists
    main_script = "maisi_parallel_setup_and_run.py"
    if not os.path.exists(main_script):
        print(f"❌ ERROR: {main_script} not found in current directory")
        print(f"   Current directory: {os.getcwd()}")
        return 1
    
    # Build command for your 2-GPU setup
    cmd = [
        sys.executable,
        main_script,
        "--use-simulated-data",        # Use test data for quick demo
        "--epochs", "5",               # Short training for testing
        "--batch-size", "1",           # Safe batch size
        "--nproc-per-node", "2",       # Use your 2 GPUs
        "--num-test-images", "3",      # Create 3 test images
        "--continue-on-error"          # Continue even if minor issues
    ]
    
    print("📋 Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Execute the main script
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n🎉 SUCCESS: Quick launch completed!")
            print("📁 Check output_work_dir/ for generated medical images")
        else:
            print(f"\n❌ FAILED: Process returned code {result.returncode}")
            print("💡 Check the output above for troubleshooting guidance")
            
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Quick launch interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
