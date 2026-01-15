#!/usr/bin/env python3
"""
Build script to compile VRM2 Arm regressor C files into a shared library.

This script compiles the CasADi-generated C files into a shared library (.so)
that can be loaded by the Python wrapper.

Usage:
    python build_regressor_lib.py [--arm left|right] [--output-dir <dir>]
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def find_c_files(arm="right"):
    """Find the C source files.
    
    Args:
        arm: "left" or "right" to specify which arm to build
    """
    script_dir = Path(__file__).parent.absolute()
    
    # Path to C files
    arm_name = "VRM2LeftArm" if arm == "left" else "VRM2RightArm"
    prefix = "left_arm" if arm == "left" else "right_arm"
    c_dir = script_dir.parent.parent / "c" / "autogen" / arm_name
    
    if not c_dir.exists():
        raise FileNotFoundError(f"C files directory not found: {c_dir}")
    
    c_files = [
        c_dir / f"{prefix}_regressor.c",
        c_dir / f"{prefix}_params_to_X.c",
        c_dir / f"{prefix}_tau_from_regressor.c",
        c_dir / f"{prefix}_regressor_base.c",
        c_dir / f"{prefix}_params_to_X_base.c",
        c_dir / f"{prefix}_tau_from_regressor_base.c",
    ]
    
    missing = [f for f in c_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing C files: {missing}")
    
    return c_files, prefix


def compile_shared_library(c_files, lib_name, output_dir=None):
    """Compile C files into a shared library.
    
    Args:
        c_files: List of C source file paths
        lib_name: Name of the shared library (e.g., "libleft_arm_regressor.so")
        output_dir: Output directory for the shared library. If None, uses script directory.
    
    Returns:
        Path to the compiled shared library
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.absolute()
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lib_path = output_dir / lib_name
    
    # Compiler command
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O3",
        "-Wall",
        "-o", str(lib_path),
    ]
    
    # Add source files
    cmd.extend([str(f) for f in c_files])
    
    # Add math library
    cmd.append("-lm")
    
    print(f"Compiling shared library: {lib_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully compiled {lib_path}")
        return lib_path
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("✗ Error: gcc not found. Please install gcc.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Build VRM2 Arm regressor shared library")
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right"],
        default="right",
        help="Which arm to build (left or right, default: right)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the shared library (default: script directory)"
    )
    args = parser.parse_args()
    
    try:
        # Find C files
        print(f"Finding C source files for {args.arm} arm...")
        c_files, prefix = find_c_files(args.arm)
        print(f"Found {len(c_files)} C files")
        
        # Determine library name
        lib_name = f"lib{prefix}_regressor.so"
        
        # Compile
        lib_path = compile_shared_library(c_files, lib_name, args.output_dir)
        
        print(f"\nBuild complete!")
        print(f"  Library: {lib_path}")
        print(f"\nYou can now use the Python wrapper:")
        wrapper_name = "LeftArmRegressor" if args.arm == "left" else "RightArmRegressor"
        wrapper_module = f"{prefix}_regressor_wrapper"
        print(f"  from {wrapper_module} import {wrapper_name}")
        print(f"  regressor = {wrapper_name}()")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

