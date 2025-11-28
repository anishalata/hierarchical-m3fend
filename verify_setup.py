#!/usr/bin/env python3
"""
Quick verification script to check if the repository is set up correctly.
Run this after cloning and installing dependencies.
"""

import os
import sys

def check_file_exists(path, description):
    """Check if a file exists and print status."""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} - NOT FOUND")
        return False

def check_directory_exists(path, description):
    """Check if a directory exists and print status."""
    if os.path.isdir(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} - NOT FOUND")
        return False

def check_import(module_name, description):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError:
        print(f"✗ {description}: {module_name} - NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("Hierarchical M3FEND Setup Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print("Python Version Check:")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 6:
        print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"✗ Python version: {python_version.major}.{python_version.minor}.{python_version.micro} - Need Python 3.6+")
        all_checks_passed = False
    print()
    
    # Check required Python packages
    print("Required Python Packages:")
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'TQDM'),
    ]
    
    for module, name in packages:
        if not check_import(module, name):
            all_checks_passed = False
    print()
    
    # Check core files
    print("Core Files:")
    core_files = [
        ('main.py', 'Main entry point'),
        ('grid_search.py', 'Grid search module'),
        ('hierarchical_english_innovation.py', 'Hierarchical innovation'),
        ('requirements.txt', 'Dependencies file'),
        ('README.md', 'Main README'),
    ]
    
    for file, desc in core_files:
        if not check_file_exists(file, desc):
            all_checks_passed = False
    print()
    
    # Check directory structure
    print("Directory Structure:")
    directories = [
        ('data', 'Data directory'),
        ('data/ch', 'Chinese data directory'),
        ('data/en', 'English data directory'),
        ('models', 'Models directory'),
        ('utils', 'Utils directory'),
    ]
    
    for dir_path, desc in directories:
        if not check_directory_exists(dir_path, desc):
            all_checks_passed = False
    print()
    
    # Check data files (optional - may not exist if data needs to be downloaded)
    print("Data Files (optional - may need to be downloaded/extracted):")
    data_files = [
        ('data/en/train.pkl', 'English training data'),
        ('data/en/val.pkl', 'English validation data'),
        ('data/en/test.pkl', 'English test data'),
        ('data/ch/train.pkl', 'Chinese training data'),
        ('data/ch/val.pkl', 'Chinese validation data'),
        ('data/ch/test.pkl', 'Chinese test data'),
    ]
    
    data_files_exist = True
    for file, desc in data_files:
        exists = check_file_exists(file, desc)
        if not exists:
            data_files_exist = False
    
    if not data_files_exist:
        print("\n⚠ Warning: Some data files are missing.")
        print("   If you see .zip files in data/ch/ or data/en/, extract them first:")
        print("   cd data/ch && unzip ch.zip && cd ../en && unzip en.zip")
        print("   Otherwise, you may need to download the data files.")
    print()
    
    # Final summary
    print("=" * 60)
    if all_checks_passed:
        print("✓ All core checks passed!")
        if data_files_exist:
            print("✓ Data files are present - ready to run!")
        else:
            print("⚠ Data files missing - please extract/download data before running")
        print("\nYou can now run:")
        print("  python main.py --gpu 0 --model_name m3fend --dataset en --domain_num 3")
        print("  python hierarchical_english_innovation.py")
    else:
        print("✗ Some checks failed - please fix the issues above")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check that all files are present in the repository")
    print("=" * 60)

if __name__ == "__main__":
    main()

