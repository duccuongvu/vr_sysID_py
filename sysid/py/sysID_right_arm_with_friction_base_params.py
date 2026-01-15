#!/usr/bin/env python3
"""
System Identification for Right Arm with Friction using Base Parameters

This program:
1. Reads q, dq, ddq, and tau from JSON file
2. Filters velocity and acceleration data
3. Computes base regressor matrix Phi_base for each sample (7x45) using C functions
4. Extends regressor to include friction: Phi_ext = [Phi_base, diag(dq), diag(sign(dq))] (7x66)
5. Stacks all regressors: A = [Phi_ext_1; Phi_ext_2; ...] (DOF*n_sample x 66)
6. Stacks all torques: b = [tau_1; tau_2; ...] (DOF*n_sample x 1)
7. Solves least squares using pseudo-inverse: minimize ||A*x - b||^2
   where x = [X_base(45); Dv(7); Dc(7); Armature(7)] (66x1)
8. Plots data and compares identified model with measured torques

Usage:
    python sysID_right_arm_with_friction_base_params.py <json_file> [--urdf <urdf_path>] [--cutoff_vel <freq>] [--cutoff_acc <freq>]
    python sysID_right_arm_with_friction_base_params.py ../json/motion_data_sysid_right_arm_2025_12_16_151359.json
    python sysID_right_arm_with_friction_base_params.py ../json/motion_data_sysid_right_arm_2025_12_16_151359.json --t_start 10.0 --t_end 30.0
"""

import numpy as np
import pinocchio as pin
import argparse
import os


from regressors.right_arm_regressor_wrapper import RightArmRegressor
from filter.filter import *
from utils.utils import *

# Base parameter constants
N_PARAMS_BASE = 45  # Base parameters (reduced from 70)
N_TOTAL_BASE = N_PARAMS_BASE + N_FRICTION + N_ARMATURE  # 45 + 14 + 7 = 66


def build_regressor_matrix(model, data, q, dq_filt, ddq_filt, tau, g):
    """
    Build extended regressor matrix with friction using base parameters.
    
    Returns:
        A: (DOF*n_samples x 66) matrix (using base parameters)
        b: (DOF*n_samples x 1) vector
    """
    print("\nBuilding regressor matrices with base parameters...")
    
    n_samples = q.shape[1]
    M = DOF * n_samples
    
    # Initialize matrices
    A = np.zeros((M, N_TOTAL_BASE))
    b = np.zeros((M, 1))
    
    # Initialize regressor wrapper
    regressor = RightArmRegressor()
    
    # For each sample, compute base regressor and extend with friction
    for s in range(n_samples):
        q_sample = q[:, s]
        dq_sample = dq_filt[:, s]
        ddq_sample = ddq_filt[:, s]
        
        # Compute base regressor Phi_base (7x45) using C functions
        Phi_base = regressor.compute_regressor_base(q_sample, dq_sample, ddq_sample, g)

        # Extend regressor: Phi_ext = [Phi_base, diag(dq), diag(sign(dq)), diag(ddq)] (7x66)
        # For each row i (joint):
        for i in range(DOF):
            row_idx = s * DOF + i  # Row index in stacked matrix A
            
            # Copy Phi_base row (45 columns)
            A[row_idx, :N_PARAMS_BASE] = Phi_base[i, :]
            
            if N_FRICTION > 0:
                # Add diag(dq): only diagonal element for joint i
                A[row_idx, N_PARAMS_BASE + i] = dq_sample[i]
                
                # Add diag(sign(dq)): only diagonal element for joint i
                A[row_idx, N_PARAMS_BASE + DOF + i] = sign_func(dq_sample[i])

            if N_ARMATURE > 0:
                # Add armature: only diagonal element for joint i
                A[row_idx, N_PARAMS_BASE + 2*DOF + i] = ddq_sample[i]
        
        # Stack torques
        for i in range(DOF):
            row_idx = s * DOF + i
            b[row_idx, 0] = tau[i, s]
    
    print(f"Regressor matrix A: {M} x {N_TOTAL_BASE} (using base parameters)")
    print(f"Torque vector b: {M} x 1")
    
    return A, b


def solve_least_squares(A, b, x_init=None):
    """Solve least squares problem using pseudo-inverse (no constraints).
    
    Args:
        A: Regressor matrix (M x N_TOTAL_BASE)
        b: Torque vector (M x 1)
        x_init: Initial guess (optional, for comparison only)
    
    Returns:
        x_opt: Optimized parameter vector (N_TOTAL_BASE,)
        residual_norm: ||A*x_opt - b||
    """
    print("\nSolving least squares problem using pseudo-inverse...")
    
    # Solve using pseudo-inverse: x = pinv(A) @ b
    # This is equivalent to: x = (A^T @ A)^(-1) @ A^T @ b
    x_opt = np.linalg.pinv(A) @ b
    x_opt = x_opt.flatten()
    
    # Compute residual for verification
    residual_opt = A @ x_opt.reshape(-1, 1) - b
    residual_norm = np.linalg.norm(residual_opt)
    residual_norm_sq = residual_norm ** 2
    
    print(f"\nResidual norm: ||A*x - b|| = {residual_norm:.6e}")
    print(f"Residual squared norm: ||A*x - b||^2 = {residual_norm_sq:.6e}")
    
    return x_opt, residual_norm


def print_results_base(x_opt, residual_norm, X_base_init=None):
    """Print identification results for base parameters."""
    lines = []
    
    # Extract solution components
    X_base_opt = x_opt[:N_PARAMS_BASE]
    Dv = x_opt[N_PARAMS_BASE:N_PARAMS_BASE + DOF]
    Dc = x_opt[N_PARAMS_BASE + DOF:N_PARAMS_BASE + DOF + DOF]
    armature = x_opt[N_PARAMS_BASE + 2*DOF : N_PARAMS_BASE + 2*DOF + N_ARMATURE]
    
    lines.append("\n" + "="*70)
    lines.append("System Identification Results (Base Parameters)")
    lines.append("="*70)
    lines.append(f"\nResidual norm: ||A*x - b|| = {residual_norm:.6e}")
    
    if X_base_init is not None:
        lines.append("\nBase Parameter Comparison (Initial vs Optimized):")
        lines.append(f"{'Index':<8} {'Initial':<20} {'Optimized':<20} {'Difference':<20} {'Rel Diff %':<12}")
        lines.append("-"*80)
        for i in range(N_PARAMS_BASE):
            diff = X_base_opt[i] - X_base_init[i]
            rel_diff = (diff / abs(X_base_init[i]) * 100) if abs(X_base_init[i]) > 1e-10 else 0.0
            lines.append(f"{i:<8} {X_base_init[i]:>19.6e} {X_base_opt[i]:>19.6e} {diff:>19.6e} {rel_diff:>11.2f}%")
    
    lines.append("\nOptimized Base Parameters X_base (45 parameters):")
    for i in range(N_PARAMS_BASE):
        lines.append(f"  X_base[{i:2d}] = {X_base_opt[i]:.6e}")
    
    if N_FRICTION > 0:
        lines.append("\nViscous Friction Dv:")
        for i in range(DOF):
            lines.append(f"  Dv[{i}] = {Dv[i]:.6e}")
        
        lines.append("\nCoulomb Friction Dc:")
        for i in range(DOF):
            lines.append(f"  Dc[{i}] = {Dc[i]:.6e}")
    
    if N_ARMATURE > 0:
        lines.append("\nArmature (motor inertia) parameters:")
        for i in range(DOF):
            lines.append(f"  Armature[{i}] = {armature[i]:.6e}")
    
    lines.append("\n" + "="*70)
    print("\n".join(lines))


def save_results_to_file_base(x_opt, residual_norm, X_base_init, json_file, urdf_path, 
                              t_start, t_end, output_dir=None):
    """Save all results to a text file for base parameters."""
    from datetime import datetime
    
    # Determine output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"sysID_results_base_{timestamp}.txt")
    
    # Build the complete report
    lines = []
    lines.append("="*80)
    lines.append("System Identification Results - Right Arm with Friction (Base Parameters)")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"JSON file: {json_file}")
    lines.append(f"URDF file: {urdf_path}")
    lines.append(f"Time range: [{t_start}, {t_end}] seconds")
    lines.append("\n" + "="*80)
    
    # Extract solution components
    X_base_opt = x_opt[:N_PARAMS_BASE]
    Dv = x_opt[N_PARAMS_BASE:N_PARAMS_BASE + DOF]
    Dc = x_opt[N_PARAMS_BASE + DOF:N_PARAMS_BASE + DOF + DOF]
    armature = x_opt[N_PARAMS_BASE + 2*DOF : N_PARAMS_BASE + 2*DOF + N_ARMATURE]
    
    lines.append(f"\nResidual norm: ||A*x - b|| = {residual_norm:.6e}")
    
    if X_base_init is not None:
        lines.append("\nBase Parameter Comparison (Initial vs Optimized):")
        lines.append(f"{'Index':<8} {'Initial':<20} {'Optimized':<20} {'Difference':<20} {'Rel Diff %':<12}")
        lines.append("-"*80)
        for i in range(N_PARAMS_BASE):
            diff = X_base_opt[i] - X_base_init[i]
            rel_diff = (diff / abs(X_base_init[i]) * 100) if abs(X_base_init[i]) > 1e-10 else 0.0
            lines.append(f"{i:<8} {X_base_init[i]:>19.6e} {X_base_opt[i]:>19.6e} {diff:>19.6e} {rel_diff:>11.2f}%")
    
    lines.append("\nOptimized Base Parameters X_base (45 parameters):")
    for i in range(N_PARAMS_BASE):
        lines.append(f"  X_base[{i:2d}] = {X_base_opt[i]:.6e}")
    
    if N_FRICTION > 0:
        lines.append("\nViscous Friction Dv:")
        for i in range(DOF):
            lines.append(f"  Dv[{i}] = {Dv[i]:.6e}")
        
        lines.append("\nCoulomb Friction Dc:")
        for i in range(DOF):
            lines.append(f"  Dc[{i}] = {Dc[i]:.6e}")
    
    if N_ARMATURE > 0:
        lines.append("\nArmature (motor inertia) parameters:")
        for i in range(DOF):
            lines.append(f"  Armature[{i}] = {armature[i]:.6e}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def plot_results_base(q, dq, dq_filt, ddq_filt, tau, time_data, model, data, x_opt, g, urdf_path, X_base_init=None):
    """Plot data and compare identified model with measured torques using base parameters."""
    print("\nGenerating plots...")
    
    # Extract solution components
    X_base_opt = x_opt[:N_PARAMS_BASE]
    Dv = x_opt[N_PARAMS_BASE:N_PARAMS_BASE + DOF]
    Dc = x_opt[N_PARAMS_BASE + DOF:N_PARAMS_BASE + DOF + DOF]
    armature = x_opt[N_PARAMS_BASE + 2*DOF : N_PARAMS_BASE + 2*DOF + N_ARMATURE]
    
    # Initialize regressor wrapper
    regressor = RightArmRegressor()
    
    # Compute torques from identified model
    n_samples = q.shape[1]
    tau_identified = np.zeros((DOF, n_samples))
    tau_id_no_armature = np.zeros((DOF, n_samples))
    tau_opt_no_friction = np.zeros((DOF, n_samples))
    tau_original = None
    tau_orig_with_friction = None
    
    if X_base_init is not None:
        tau_original = np.zeros((DOF, n_samples))
        tau_orig_with_friction = np.zeros((DOF, n_samples))
    
    for s in range(n_samples):
        q_sample = q[:, s]
        dq_sample = dq_filt[:, s]
        ddq_sample = ddq_filt[:, s]
        
        # Compute base regressor
        Phi_base = regressor.compute_regressor_base(q_sample, dq_sample, ddq_sample, g)
        
        # Compute torque from identified base parameters
        tau_base = Phi_base @ X_base_opt
        
        # Add friction
        if N_FRICTION > 0:
            tau_friction = Dv * dq_sample + Dc * sign_func(dq_sample)
        else:
            tau_friction = 0
        
        # Add armature
        if N_ARMATURE > 0:
            tau_armature = armature * ddq_sample
        else:
            tau_armature = 0

        # Total torque (identified)
        tau_identified[:, s] = tau_base + tau_friction + tau_armature
        tau_id_no_armature[:, s] = tau_base + tau_friction
        tau_opt_no_friction[:, s] = tau_base
        
        # Compute torque from original base parameters (no friction)
        if X_base_init is not None:
            tau_original[:, s] = Phi_base @ X_base_init
            # Torque from original parameters with optimized friction
            tau_orig_with_friction[:, s] = tau_original[:, s] + tau_friction
    
    # ===== FIRST FIGURE: Position, Velocity, Acceleration, Torque (All Joints) =====
    import matplotlib.pyplot as plt
    
    fig1 = plt.figure(figsize=(16, 10))
    
    # Plot 1: Joint positions
    ax1 = plt.subplot(2, 2, 1)
    for i in range(DOF):
        ax1.plot(time_data, q[i, :], label=f'q{i+1}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Joint Positions')
    ax1.legend(loc='center right')
    ax1.grid(True)
    
    # Plot 2: Joint velocities (raw and filtered)
    ax2 = plt.subplot(2, 2, 2)
    for i in range(DOF):
        ax2.plot(time_data, dq[i, :], alpha=0.5, label=f'dq{i+1} (raw)')
        ax2.plot(time_data, dq_filt[i, :], label=f'dq{i+1} (filt)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Joint Velocities (Raw and Filtered)')
    ax2.legend(loc='center right')
    ax2.grid(True)
    
    # Plot 3: Joint accelerations
    ax3 = plt.subplot(2, 2, 3)
    for i in range(DOF):
        ax3.plot(time_data, ddq_filt[i, :], label=f'ddq{i+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Joint Accelerations (Filtered)')
    ax3.legend(loc='center right')
    ax3.grid(True)
    
    # Plot 4: Torques (measured vs identified vs original) - All joints
    ax4 = plt.subplot(2, 2, 4)
    for i in range(DOF):
        ax4.plot(time_data, tau[i, :], alpha=0.7, label=f'τ{i+1} (meas)', linewidth=0.5)
        ax4.plot(time_data, tau_identified[i, :], '--', label=f'τ{i+1} (id)', linewidth=0.75)
        ax4.plot(time_data, tau_id_no_armature[i, :], 'orange', linestyle='--', label=f'τ{i+1} (id no armature)', linewidth=0.75, alpha=0.7)
        ax4.plot(time_data, tau_opt_no_friction[i, :], '-.', label=f'τ{i+1} (opt no fric)', linewidth=0.75, alpha=0.7)
        if tau_original is not None:
            ax4.plot(time_data, tau_original[i, :], ':', label=f'τ{i+1} (orig)', linewidth=0.75, alpha=0.8)
            ax4.plot(time_data, tau_orig_with_friction[i, :], ':', label=f'τ{i+1} (orig+fric)', linewidth=0.75, alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Torques: Measured vs Identified vs Original (All Joints)')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # ===== SECOND FIGURE: Individual Joint Torque Comparisons =====
    fig2 = plt.figure(figsize=(16, 14))
    
    # Plot 5-11: Torques comparison for each joint
    for joint_idx in range(DOF):
        ax = plt.subplot(4, 2, joint_idx + 1)
        ax.plot(time_data, tau[joint_idx, :], 'b-', linewidth=1.0, label='Measured', alpha=0.4)
        ax.plot(time_data, tau_identified[joint_idx, :], 'r--', linewidth=1, label='Identified')
        ax.plot(time_data, tau_id_no_armature[joint_idx, :], 'orange', linestyle='--', linewidth=1, label='Id (no armature)')
        ax.plot(time_data, tau_opt_no_friction[joint_idx, :], 'm-.', linewidth=1, label='Opt Params (no friction)', alpha=0.8)
        if tau_original is not None:
            ax.plot(time_data, tau_original[joint_idx, :], 'g:', linewidth=2, label='Original (URDF)', alpha=0.8)
            ax.plot(time_data, tau_orig_with_friction[joint_idx, :], 'c:', linewidth=2, label='Orig Params + Friction', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(f'Joint {joint_idx+1}: Measured vs Identified vs Original Torque')
        ax.legend(loc='center right', fontsize=7)
        ax.grid(True)
        
        # Compute errors
        error_id = tau[joint_idx, :] - tau_identified[joint_idx, :]
        rmse_id = np.sqrt(np.mean(error_id**2))
        max_error_id = np.max(np.abs(error_id))
        
        error_text = f'RMSE (id): {rmse_id:.4f} Nm\nMax Error (id): {max_error_id:.4f} Nm'
        
        # Error for identified params without armature
        error_id_no_armature = tau[joint_idx, :] - tau_id_no_armature[joint_idx, :]
        rmse_id_no_armature = np.sqrt(np.mean(error_id_no_armature**2))
        max_error_id_no_armature = np.max(np.abs(error_id_no_armature))
        error_text += f'\nRMSE (id no armature): {rmse_id_no_armature:.4f} Nm\nMax Error (id no armature): {max_error_id_no_armature:.4f} Nm'
        
        # Error for optimized params without friction
        error_opt_no_fric = tau[joint_idx, :] - tau_opt_no_friction[joint_idx, :]
        rmse_opt_no_fric = np.sqrt(np.mean(error_opt_no_fric**2))
        max_error_opt_no_fric = np.max(np.abs(error_opt_no_fric))
        error_text += f'\nRMSE (opt no fric): {rmse_opt_no_fric:.4f} Nm\nMax Error (opt no fric): {max_error_opt_no_fric:.4f} Nm'
        
        if tau_original is not None:
            error_orig = tau[joint_idx, :] - tau_original[joint_idx, :]
            rmse_orig = np.sqrt(np.mean(error_orig**2))
            max_error_orig = np.max(np.abs(error_orig))
            error_text += f'\nRMSE (orig): {rmse_orig:.4f} Nm\nMax Error (orig): {max_error_orig:.4f} Nm'
            
            # Error for original params with friction
            error_orig_fric = tau[joint_idx, :] - tau_orig_with_friction[joint_idx, :]
            rmse_orig_fric = np.sqrt(np.mean(error_orig_fric**2))
            max_error_orig_fric = np.max(np.abs(error_orig_fric))
            error_text += f'\nRMSE (orig+fric): {rmse_orig_fric:.4f} Nm\nMax Error (orig+fric): {max_error_orig_fric:.4f} Nm'
        
        ax.text(0.02, 0.98, error_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=7)
    
    plt.tight_layout()
    
    # Show plots
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='System Identification for Right Arm with Friction')
    parser.add_argument('json_file', type=str, help='Path to JSON file with robot data')
    parser.add_argument('--urdf', type=str, default=None,
                       help='Path to URDF file (default: ../sysid/matlab/vr_m2_right_arm.urdf)')
    parser.add_argument('--cutoff_vel', type=float, default=50.0,
                       help='Cutoff frequency for velocity filter (Hz, default: 50.0)')
    parser.add_argument('--cutoff_acc', type=float, default=30.0,
                       help='Cutoff frequency for acceleration filter (Hz, default: 30.0)')
    parser.add_argument('--t_start', type=float, default=10.0,
                       help='Start time for data filtering (seconds, default: 10.0)')
    parser.add_argument('--t_end', type=float, default=30.0,
                       help='End time for data filtering (seconds, default: 30.0)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    # Determine URDF path
    if args.urdf is None:
        # Try default paths
        default_paths = [
            # '../sysid/matlab/vr_m2_right_arm.urdf',
            # '../../sysid/matlab/vr_m2_right_arm.urdf',
            # '../models/urdf/origin/vr_m2_right_arm_updated.urdf',
            '../../models/urdf/origin/vr_m2_right_arm_updated.urdf'
        ]
        urdf_path = None
        for path in default_paths:
            if os.path.exists(path):
                urdf_path = path
                break
        if urdf_path is None:
            raise FileNotFoundError("URDF file not found. Please specify with --urdf")
    else:
        urdf_path = args.urdf
    
    print(f"Using URDF: {urdf_path}")
    
    # Load model
    print("\nLoading robot model...")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    print(f"Model loaded: nq={model.nq}, nv={model.nv}, nbodies={model.nbodies}")
    
    # Read JSON data
    q, dq, tau, time_data, dt = read_json_data(args.json_file)
    
    # Filter data by time range
    q, dq, tau, time_data = filter_by_time_range(q, dq, tau, time_data, args.t_start, args.t_end)
    
    # Filter velocity data
    dq_filt = filter_velocity_data(dq, dt, args.cutoff_vel)
    
    # Compute acceleration from filtered velocity
    ddq = compute_acceleration(dq_filt, dt)
    
    # Filter acceleration data
    ddq_filt = filter_acceleration_data(ddq, dt, args.cutoff_acc)
    
    # Build regressor matrix
    A, b = build_regressor_matrix(model, data, q, dq_filt, ddq_filt, tau, DEFAULT_GRAVITY)
    
    # Get initial condition from model using base parameters
    regressor = RightArmRegressor()
    X_init_full = extract_params(model)  # Full 70 parameters from URDF
    # X_init_full = pinocchio_standard_params(model)

    # Convert full parameters to base parameters using params_to_X_base
    # params_to_X_base expects 70 individual parameters
    X_base_init = regressor.params_to_X_base(*X_init_full)
    
    # Initialize x_init with base parameters
    x_init = np.zeros(N_TOTAL_BASE)
    x_init[:N_PARAMS_BASE] = X_base_init
    # Set the armature (motor inertia) parameters to 0.5 for each joint in the initial guess
    x_init[N_PARAMS_BASE + 2*DOF : N_PARAMS_BASE + 2*DOF + N_ARMATURE] = 0.0
    # Friction parameters initialized to zero
    
    # Print initial X_base for comparison
    print("\n" + "="*70)
    print("Initial X_base (from URDF parameters):")
    print("="*70)
    for i in range(N_PARAMS_BASE):
        print(f"  X_base_init[{i:2d}] = {X_base_init[i]:.6e}")
    print("="*70)
    
    # Solve least squares using pseudo-inverse
    x_opt, residual_norm = solve_least_squares(A, b, x_init)
    
    # Extract optimized X_base
    X_base_opt = x_opt[:N_PARAMS_BASE]
    
    # Print optimized X_base and compare with initial
    print("\n" + "="*70)
    print("Optimized X_base:")
    print("="*70)
    for i in range(N_PARAMS_BASE):
        diff = X_base_opt[i] - X_base_init[i]
        rel_diff = (diff / abs(X_base_init[i]) * 100) if abs(X_base_init[i]) > 1e-10 else 0.0
        print(f"  X_base_opt[{i:2d}] = {X_base_opt[i]:.6e}  (init: {X_base_init[i]:.6e}, diff: {diff:.6e}, rel: {rel_diff:.2f}%)")
    print("="*70)
    
    # Print results with comparison
    print_results_base(x_opt, residual_norm, X_base_init)
    
    # Save results to file
    save_results_to_file_base(x_opt, residual_norm, X_base_init, args.json_file, urdf_path, 
                              args.t_start, args.t_end)
    
    # Note: Not saving to URDF file as requested (base parameters don't map directly to physical params)
    
    # Plot results
    if not args.no_plot:
        plot_results_base(q, dq, dq_filt, ddq_filt, tau, time_data, model, data, x_opt, DEFAULT_GRAVITY, urdf_path, X_base_init)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
