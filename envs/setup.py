#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Setup Script (only tested on Linux platform, not test on Win or Mac)

This script automatically sets up a conda environment and installs pip dependencies
from environment.yml and requirements.txt files.

Example Usage:
--------------
# Basic usage with default settings (uses current directory for config files)
python setup.py --conda-dir /path/to/miniconda3 --env-name my-project

# Specify a custom config directory containing environment.yml and requirements.txt
python setup.py --conda-dir /path/to/miniconda3 --env-name my-project --config-dir /path/to/config

# Complete example with all options
python setup.py --conda-dir /home/user/miniconda3 --env-name tHPM-LDM --config-dir ./

Output:
-------
The script will:
1. Check if conda and pip are installed
2. Create a conda environment with the specified name in the standard location
3. Install pip dependencies from requirements.txt
4. Show the command to activate the new environment
"""

import sys
import argparse
import subprocess
from pathlib import Path


def check_requirements():
    """Check if necessary tools (conda and pip) are installed"""
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Conda is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Conda not found. Please install conda: https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)

    try:
        subprocess.run(["pip", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Pip is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Pip not found. Please make sure pip is installed")
        sys.exit(1)


def setup_conda_environment(conda_dir, config_dir, env_name):
    """Set up conda environment using environment.yml"""
    yml_path = Path(config_dir) / "environment.yml"
    
    if not yml_path.exists():
        print(f"❌ environment.yml file not found in {config_dir}")
        return False
    
    # Create conda environment
    print(f"🔄 Creating conda environment '{env_name}'...")
    
    # First check if environment already exists
    result = subprocess.run(["conda", "env", "list"], stdout=subprocess.PIPE, text=True, check=False)
    if f"{env_name} " in result.stdout:
        print(f"⚠️ Environment '{env_name}' already exists. Would you like to remove it first? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            subprocess.run(["conda", "env", "remove", "-n", env_name], check=False)
        else:
            print(f"❌ Cannot proceed with existing environment. Please choose a different name or remove the environment.")
            return False
    
    # Create environment using -n flag (standard name-based approach)
    cmd = ["conda", "env", "create", "-n", env_name, "-f", str(yml_path)]
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Failed to create conda environment")
        return False
    
    print(f"✅ Successfully created conda environment: {env_name}")
    return True


def install_pip_requirements(conda_dir, config_dir, env_name):
    """Install pip packages using requirements.txt"""
    req_path = Path(config_dir) / "requirements.txt"
    
    if not req_path.exists():
        print(f"⚠️ requirements.txt file not found in {config_dir}, skipping pip installation")
        return True
    
    print("🔄 Installing pip dependencies...")
    
    # Use conda run to execute pip in the environment
    cmd = ["conda", "run", "-n", env_name, "pip", "install", "-r", str(req_path)]
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print("❌ Failed to install pip dependencies")
        return False
    
    print("✅ Successfully installed pip dependencies")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Set up conda environment and install pip dependencies')
    parser.add_argument('--conda-dir', type=str, required=True, 
                        help='Path to the Conda installation directory (e.g., /home/user/miniconda3)')
    parser.add_argument('--config-dir', type=str, default="./", 
                        help='Directory containing environment.yml and requirements.txt files')
    parser.add_argument('--env-name', type=str, default="tHPM-LDM", 
                        help='Environment name to create')
    
    args = parser.parse_args()
    
    print("🔍 Checking environment requirements...")
    check_requirements()
    
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"❌ Config directory {config_dir} does not exist")
        sys.exit(1)
        
    print(f"\n🔍 Checking configuration files in {config_dir}...")
    if not (config_dir / "environment.yml").exists() and not (config_dir / "requirements.txt").exists():
        print(f"❌ Neither environment.yml nor requirements.txt found in {config_dir}")
        sys.exit(1)
    
    # Set up conda environment
    if (config_dir / "environment.yml").exists():
        success = setup_conda_environment(args.conda_dir, args.config_dir, args.env_name)
        if not success:
            sys.exit(1)
    else:
        print("⚠️ environment.yml file not found, creating basic environment")
        # Create basic environment
        cmd = ["conda", "create", "--yes", "-n", args.env_name, "python"]
        
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print("❌ Failed to create basic conda environment")
            sys.exit(1)
    
    # Install pip dependencies
    if (config_dir / "requirements.txt").exists():
        success = install_pip_requirements(args.conda_dir, args.config_dir, args.env_name)
        if not success:
            sys.exit(1)
    
    # Show activation command
    print("\n🎉 Environment setup complete!")
    print(f"To activate this environment, run:")
    print(f"  conda activate {args.env_name}")
    print("\n🤗 Wish you have fun in exploring tHPM-LDM!")

if __name__ == "__main__":
    main()