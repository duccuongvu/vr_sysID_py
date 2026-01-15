#!/usr/bin/env python3
"""
Generate optimal trajectory for system identification using Fourier series.

This script:
1. Generates Fourier series trajectories for 7-DOF right arm robot
2. Computes base regressor matrix Phi_base for all trajectory samples
3. Optimizes Fourier coefficients to minimize condition number of stacked Phi_base
4. Constraints: zero pos/vel/acc at t=0 and t=T, position limits from URDF, velocity limits ±5 rad/s

Usage:
    python generate_trajectory.py [--urdf <urdf_path>] [--T_traj <time>] [--dt <sampling_time>] [--N_fourier <n>]
"""

import numpy as np
import pinocchio as pin
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
import time
import matplotlib.pyplot as plt

from regressors.right_arm_regressor_wrapper import RightArmRegressor
from utils.utils import DEFAULT_GRAVITY, DOF

# Constants
N_PARAMS_BASE = 45  # Base parameters (reduced from 70)
DEFAULT_T_TRAJ = 20.0  # Default trajectory duration (seconds)
DEFAULT_DT = 0.1  # Default sampling time (seconds)
DEFAULT_N_FOURIER = 10  # Default number of Fourier terms
JOINT2_OFFSET = - np.pi / 4.0  # 45 degrees offset for joint 2 (index 1) in radians


def extract_joint_limits_from_urdf(urdf_path):
    """Extract joint position limits from URDF file.
    
    Returns:
        q_min: Lower joint limits (7,)
        q_max: Upper joint limits (7,)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Right arm joint names in order
    joint_names = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint"
    ]
    
    q_min = np.zeros(DOF)
    q_max = np.zeros(DOF)
    
    # Find all joints
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        if joint_name in joint_names:
            idx = joint_names.index(joint_name)
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = limit_elem.get('lower')
                upper = limit_elem.get('upper')
                if lower is not None:
                    q_min[idx] = float(lower)
                if upper is not None:
                    q_max[idx] = float(upper)
    
    return q_min*0.8, q_max*0.8


def fourier_series_trajectory(t, A, B, T_traj, N_fourier, joint2_offset=0.0):
    """
    Generate Fourier series trajectory for all joints.
    
    For each joint i:
        q_i(t) = sum_{j=1}^{N_fourier} [a_ij * sin(2*pi*j*t/T) + b_ij * cos(2*pi*j*t/T)]
    
    For joint 2 (index 1), an offset is added to shift the trajectory.
    
    Args:
        t: Time (scalar or array)
        A: Coefficient matrix (DOF x N_fourier) - sine coefficients
        B: Coefficient matrix (DOF x N_fourier) - cosine coefficients
        T_traj: Trajectory duration
        N_fourier: Number of Fourier terms
        joint2_offset: Offset for joint 2 (default: 0.0, typically π/4 for 45 degrees)
    
    Returns:
        q: Position (DOF,) or (DOF, n_samples)
        dq: Velocity (DOF,) or (DOF, n_samples)
        ddq: Acceleration (DOF,) or (DOF, n_samples)
    """
    t = np.asarray(t)
    is_scalar = t.ndim == 0
    if is_scalar:
        t = np.array([t])
    
    n_samples = len(t)
    q = np.zeros((DOF, n_samples))
    dq = np.zeros((DOF, n_samples))
    ddq = np.zeros((DOF, n_samples))
    
    omega = 2 * np.pi / T_traj  # Fundamental frequency
    
    for i in range(DOF):
        for j in range(1, N_fourier + 1):
            omega_j = j * omega
            sin_term = np.sin(omega_j * t)
            cos_term = np.cos(omega_j * t)
            
            # Position
            q[i, :] += A[i, j-1] * sin_term + B[i, j-1] * cos_term
            
            # Velocity (derivative of sin is cos, derivative of cos is -sin)
            dq[i, :] += omega_j * (A[i, j-1] * cos_term - B[i, j-1] * sin_term)
            
            # Acceleration (second derivative)
            ddq[i, :] += -omega_j**2 * (A[i, j-1] * sin_term + B[i, j-1] * cos_term)
    
    # Apply offset to joint 2 (index 1)
    if joint2_offset != 0.0:
        q[1, :] += joint2_offset
        # Note: velocity and acceleration are unchanged since offset is constant
    
    if is_scalar:
        return q[:, 0], dq[:, 0], ddq[:, 0]
    return q, dq, ddq




def compute_condition_number(Phi_stacked):
    """
    Compute condition number of a matrix.
    
    Condition number = max(eigenvalue) / min(eigenvalue) of Phi^T @ Phi
    """
    # Compute Phi^T @ Phi
    PhiT_Phi = Phi_stacked.T @ Phi_stacked
    
    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(PhiT_Phi)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Filter out near-zero eigenvalues
    
    if len(eigenvals) == 0:
        return np.inf
    
    max_eval = np.max(eigenvals)
    min_eval = np.min(eigenvals)
    
    if min_eval < 1e-12:
        return np.inf
    
    return max_eval / min_eval


def optimize_trajectory(urdf_path, T_traj, dt, N_fourier, q_min, q_max, g):
    """
    Optimize Fourier coefficients to minimize condition number of Phi_base.
    
    Args:
        urdf_path: Path to URDF file
        T_traj: Trajectory duration
        dt: Sampling time
        N_fourier: Number of Fourier terms
        q_min: Lower joint limits (DOF,)
        q_max: Upper joint limits (DOF,)
        g: Gravity vector (3,)
    
    Returns:
        A_opt: Optimized sine coefficients (DOF x N_fourier)
        B_opt: Optimized cosine coefficients (DOF x N_fourier)
        cond_num: Final condition number
    """
    print("\n" + "="*70)
    print("Optimizing Trajectory for System Identification")
    print("="*70)
    print(f"Trajectory duration: {T_traj} s")
    print(f"Sampling time: {dt} s")
    print(f"Number of Fourier terms: {N_fourier}")
    print(f"Number of samples: {int(T_traj / dt) + 1}")
    
    # Generate time samples
    t_samples = np.arange(0, T_traj + dt, dt)
    n_samples = len(t_samples)
    print(f"Actual number of samples: {n_samples}")
    
    # Initialize regressor wrapper
    regressor = RightArmRegressor()
    
    # Since CasADi can't directly call the C regressor function, we'll use scipy.optimize
    # with a callback that evaluates Phi_base numerically
    print("\nUsing numerical optimization with callback...")
    
    # Initial guess: small random values
    x0 = np.random.randn(DOF * N_fourier * 2) * 0.1
    
    # Bounds: reasonable range for coefficients
    lb = np.full(DOF * N_fourier * 2, -10.0)
    ub = np.full(DOF * N_fourier * 2, 10.0)
    
    # Constraints: zero pos/vel/acc at t=0 and t=T_traj
    def constraint_zero_boundary(x):
        """Constraint: q, dq, ddq = 0 at t=0 and t=T_traj"""
        A_reshaped = x[:DOF * N_fourier].reshape(DOF, N_fourier)
        B_reshaped = x[DOF * N_fourier:].reshape(DOF, N_fourier)
        
        constraints = []
        
        # At t=0: all sin terms are 0, all cos terms are 1
        # q(0) = sum_j b_ij = 0 for all i
        # dq(0) = sum_j j*omega * a_ij = 0 for all i
        # ddq(0) = -sum_j (j*omega)^2 * b_ij = 0 for all i
        omega = 2 * np.pi / T_traj
        for i in range(DOF):
            # q(0) = sum_j b_ij = 0
            constraints.append(np.sum(B_reshaped[i, :]))
            # dq(0) = sum_j j*omega * a_ij = 0
            j_vals = np.arange(1, N_fourier + 1)
            constraints.append(np.sum(j_vals * omega * A_reshaped[i, :]))
            # ddq(0) = -sum_j (j*omega)^2 * b_ij = 0
            constraints.append(-np.sum((j_vals * omega)**2 * B_reshaped[i, :]))
        
        # Note: At t=T_traj, sin(2*pi*j*T_traj/T_traj) = sin(2*pi*j) = 0,
        #       and cos(2*pi*j*T_traj/T_traj) = cos(2*pi*j) = 1
        # So the boundary conditions at t=T_traj are identical to t=0,
        # and we only need to enforce them once (which we did above)
        
        return np.array(constraints)
    
    # Objective function: condition number of stacked Phi_base
    def objective(x):
        """Compute condition number of stacked Phi_base matrix"""
        A_reshaped = x[:DOF * N_fourier].reshape(DOF, N_fourier)
        B_reshaped = x[DOF * N_fourier:].reshape(DOF, N_fourier)
        
        # Compute trajectory for all samples (with joint 2 offset)
        q_all, dq_all, ddq_all = fourier_series_trajectory(t_samples, A_reshaped, B_reshaped, T_traj, N_fourier, JOINT2_OFFSET)
        
        # Stack Phi_base for all samples
        Phi_stacked = []
        for s in range(n_samples):
            q_sample = q_all[:, s]
            dq_sample = dq_all[:, s]
            ddq_sample = ddq_all[:, s]
            
            # Compute Phi_base using regressor
            Phi_base = regressor.compute_regressor_base(q_sample, dq_sample, ddq_sample, g)
            Phi_stacked.append(Phi_base)
        
        # Stack vertically: (n_samples * DOF) x N_PARAMS_BASE
        Phi_stacked = np.vstack(Phi_stacked)
        
        # Compute condition number
        cond_num = compute_condition_number(Phi_stacked)
        
        return cond_num
    
    # Constraint function for position and velocity limits
    def constraint_limits(x):
        """Constraint: position and velocity limits (inequality: g(x) >= 0)"""
        A_reshaped = x[:DOF * N_fourier].reshape(DOF, N_fourier)
        B_reshaped = x[DOF * N_fourier:].reshape(DOF, N_fourier)
        
        # Compute trajectory for all samples (with joint 2 offset)
        q_all, dq_all, _ = fourier_series_trajectory(t_samples, A_reshaped, B_reshaped, T_traj, N_fourier, JOINT2_OFFSET)
        
        constraints = []
        
        # Position limits: q_min <= q <= q_max
        # Constraint: q - q_min >= 0 and q_max - q >= 0
        for i in range(DOF):
            constraints.extend(q_all[i, :] - q_min[i])  # q - q_min >= 0
            constraints.extend(q_max[i] - q_all[i, :])  # q_max - q >= 0
        
        # Velocity limits: -5 <= dq <= 5
        # Constraint: dq + 5 >= 0 and 5 - dq >= 0
        vel_limit = 5.0
        for i in range(DOF):
            constraints.extend(dq_all[i, :] + vel_limit)  # dq + 5 >= 0
            constraints.extend(vel_limit - dq_all[i, :])  # 5 - dq >= 0
        
        return np.array(constraints)
    
    # Use scipy.optimize for optimization
    from scipy.optimize import minimize
    
    print("\nSetting up optimization problem...")
    
    # Combine constraints
    def constraint_boundary(x):
        return constraint_zero_boundary(x)
    
    def constraint_all_limits(x):
        return constraint_limits(x)
    
    constraints = [
        {'type': 'eq', 'fun': constraint_boundary},  # Equality: zero at boundaries
        {'type': 'ineq', 'fun': constraint_all_limits}  # Inequality: limits
    ]
    
    print("Starting optimization...")
    print("This may take a while...")
    
    # Start timing
    tic = time.perf_counter()
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=list(zip(lb, ub)),
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )
    
    # End timing
    toc = time.perf_counter()
    solving_time = toc - tic
    
    if not result.success:
        print(f"\nWarning: Optimization did not converge: {result.message}")
    
    # Extract optimized coefficients
    A_opt = result.x[:DOF * N_fourier].reshape(DOF, N_fourier)
    B_opt = result.x[DOF * N_fourier:].reshape(DOF, N_fourier)
    
    # Compute final condition number
    cond_num_final = objective(result.x)
    
    print(f"\nOptimization completed!")
    print(f"Solving time: {solving_time:.2f} seconds ({solving_time/60:.2f} minutes)")
    print(f"Final condition number: {cond_num_final:.6e}")
    print(f"Optimization status: {result.message}")
    
    return A_opt, B_opt, cond_num_final


def plot_trajectory(t_samples, q_traj, dq_traj, ddq_traj, q_min, q_max):
    """Plot the generated trajectory.
    
    Args:
        t_samples: Time samples (n_samples,)
        q_traj: Position trajectory (DOF, n_samples)
        dq_traj: Velocity trajectory (DOF, n_samples)
        ddq_traj: Acceleration trajectory (DOF, n_samples)
        q_min: Lower joint limits (DOF,)
        q_max: Upper joint limits (DOF,)
    """
    n_samples = len(t_samples)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot positions
    ax1 = plt.subplot(3, 1, 1)
    for i in range(DOF):
        ax1.plot(t_samples, q_traj[i, :], label=f'Joint {i+1}', linewidth=1.5)
        # Plot limits as horizontal lines
        ax1.axhline(y=q_min[i], color=f'C{i}', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=q_max[i], color=f'C{i}', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Joint Positions')
    ax1.legend(loc='upper right', ncol=4, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot velocities
    ax2 = plt.subplot(3, 1, 2)
    for i in range(DOF):
        ax2.plot(t_samples, dq_traj[i, :], label=f'Joint {i+1}', linewidth=1.5)
        # Plot velocity limits
        ax2.axhline(y=-5.0, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.axhline(y=5.0, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Joint Velocities')
    ax2.legend(loc='upper right', ncol=4, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot accelerations
    ax3 = plt.subplot(3, 1, 3)
    for i in range(DOF):
        ax3.plot(t_samples, ddq_traj[i, :], label=f'Joint {i+1}', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Joint Accelerations')
    ax3.legend(loc='upper right', ncol=4, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(script_dir, 'trajectory_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    # Create individual joint plots
    fig2 = plt.figure(figsize=(16, 14))
    
    for joint_idx in range(DOF):
        # Position
        ax = plt.subplot(DOF, 3, joint_idx * 3 + 1)
        ax.plot(t_samples, q_traj[joint_idx, :], 'b-', linewidth=1.5, label='Position')
        ax.axhline(y=q_min[joint_idx], color='r', linestyle='--', alpha=0.7, linewidth=1, label='Limits')
        ax.axhline(y=q_max[joint_idx], color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.set_title(f'Joint {joint_idx+1}: Position')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Velocity
        ax = plt.subplot(DOF, 3, joint_idx * 3 + 2)
        ax.plot(t_samples, dq_traj[joint_idx, :], 'g-', linewidth=1.5, label='Velocity')
        ax.axhline(y=-5.0, color='r', linestyle='--', alpha=0.7, linewidth=1, label='Limits')
        ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title(f'Joint {joint_idx+1}: Velocity')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Acceleration
        ax = plt.subplot(DOF, 3, joint_idx * 3 + 3)
        ax.plot(t_samples, ddq_traj[joint_idx, :], 'm-', linewidth=1.5, label='Acceleration')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (rad/s²)')
        ax.set_title(f'Joint {joint_idx+1}: Acceleration')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save individual plots
    plot_file2 = os.path.join(script_dir, 'trajectory_plot_individual.png')
    plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
    print(f"Individual joint plots saved to: {plot_file2}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate optimal trajectory for system identification')
    parser.add_argument('--urdf', type=str, default=None,
                       help='Path to URDF file (default: auto-detect)')
    parser.add_argument('--T_traj', type=float, default=DEFAULT_T_TRAJ,
                       help=f'Trajectory duration in seconds (default: {DEFAULT_T_TRAJ})')
    parser.add_argument('--dt', type=float, default=DEFAULT_DT,
                       help=f'Sampling time in seconds (default: {DEFAULT_DT})')
    parser.add_argument('--N_fourier', type=int, default=DEFAULT_N_FOURIER,
                       help=f'Number of Fourier terms (default: {DEFAULT_N_FOURIER})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for trajectory coefficients (default: trajectory_coeffs.npz)')
    
    args = parser.parse_args()
    
    # Determine URDF path
    if args.urdf is None:
        default_paths = [
            '../../models/urdf/origin/vr_m2_right_arm_updated_sysID.urdf',
            '../models/urdf/origin/vr_m2_right_arm_updated_sysID.urdf',
            'models/urdf/origin/vr_m2_right_arm_updated_sysID.urdf'
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
    
    # Extract joint limits
    print("\nExtracting joint limits from URDF...")
    q_min, q_max = extract_joint_limits_from_urdf(urdf_path)
    
    # Store original limits for display (before offset adjustment)
    q_min_original = q_min.copy()
    q_max_original = q_max.copy()
    
    print("Joint limits (original):")
    for i in range(DOF):
        print(f"  Joint {i+1}: [{q_min_original[i]:.4f}, {q_max_original[i]:.4f}] rad")
    print(f"\nNote: Joint 2 will have a {np.degrees(JOINT2_OFFSET):.1f}° ({JOINT2_OFFSET:.4f} rad) offset applied.")
    
    # Adjust limits for optimization (accounting for offset)
    q_min_opt = q_min.copy()
    q_max_opt = q_max.copy()
    # q_min_opt[1] = q_min_opt[1] - JOINT2_OFFSET
    # q_max_opt[1] = q_max_opt[1] - JOINT2_OFFSET
    
    # Optimize trajectory (using adjusted limits)
    A_opt, B_opt, cond_num = optimize_trajectory(
        urdf_path, args.T_traj, args.dt, args.N_fourier, q_min_opt, q_max_opt, DEFAULT_GRAVITY
    )
    
    # Generate final trajectory for visualization (with joint 2 offset)
    t_samples = np.arange(0, args.T_traj + args.dt, args.dt)
    q_traj, dq_traj, ddq_traj = fourier_series_trajectory(t_samples, A_opt, B_opt, args.T_traj, args.N_fourier, JOINT2_OFFSET)
    
    # Verify constraints
    print("\n" + "="*70)
    print("Verifying Constraints")
    print("="*70)
    
    # Check boundary conditions (with joint 2 offset)
    q_0, dq_0, ddq_0 = fourier_series_trajectory(0.0, A_opt, B_opt, args.T_traj, args.N_fourier, JOINT2_OFFSET)
    q_T, dq_T, ddq_T = fourier_series_trajectory(args.T_traj, A_opt, B_opt, args.T_traj, args.N_fourier, JOINT2_OFFSET)
    
    print("\nBoundary conditions:")
    print(f"  q(0): {q_0}")
    print(f"  dq(0): {dq_0}")
    print(f"  ddq(0): {ddq_0}")
    print(f"  q(T): {q_T}")
    print(f"  dq(T): {dq_T}")
    print(f"  ddq(T): {ddq_T}")
    print(f"\nNote: Joint 2 (index 1) has an offset of {JOINT2_OFFSET:.4f} rad ({np.degrees(JOINT2_OFFSET):.1f}°),")
    print(f"      so q(0)[1] and q(T)[1] should be ~{JOINT2_OFFSET:.4f}, not ~0.")
    print(f"      Velocities and accelerations should still be ~0 for all joints.")
    
    # Check limits (using original limits for display)
    print("\nPosition limits (should be within bounds):")
    for i in range(DOF):
        q_min_actual = np.min(q_traj[i, :])
        q_max_actual = np.max(q_traj[i, :])
        if i == 1:  # Joint 2
            print(f"  Joint {i+1}: [{q_min_actual:.4f}, {q_max_actual:.4f}] (limits: [{q_min_original[i]:.4f}, {q_max_original[i]:.4f}], offset: {JOINT2_OFFSET:.4f} rad)")
        else:
            print(f"  Joint {i+1}: [{q_min_actual:.4f}, {q_max_actual:.4f}] (limits: [{q_min_original[i]:.4f}, {q_max_original[i]:.4f}])")
    
    print("\nVelocity limits (should be within [-5, 5]):")
    for i in range(DOF):
        dq_min_actual = np.min(dq_traj[i, :])
        dq_max_actual = np.max(dq_traj[i, :])
        print(f"  Joint {i+1}: [{dq_min_actual:.4f}, {dq_max_actual:.4f}]")
    
    # Save results
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'trajectory_coeffs.npz')
    else:
        output_file = args.output
    
    np.savez(
        output_file,
        A=A_opt,
        B=B_opt,
        T_traj=args.T_traj,
        dt=args.dt,
        N_fourier=args.N_fourier,
        q_min=q_min_original,
        q_max=q_max_original,
        q_min_opt=q_min_opt,
        q_max_opt=q_max_opt,
        joint2_offset=JOINT2_OFFSET,
        cond_num=cond_num,
        t_samples=t_samples,
        q_traj=q_traj,
        dq_traj=dq_traj,
        ddq_traj=ddq_traj
    )
    
    print(f"\nTrajectory coefficients saved to: {output_file}")
    
    # Save trajectory to CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'trajectory_data.csv')
    
    print(f"\nSaving trajectory to CSV: {csv_file}")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['time']
        for i in range(DOF):
            header.append(f'q{i+1}')
        for i in range(DOF):
            header.append(f'dq{i+1}')
        for i in range(DOF):
            header.append(f'ddq{i+1}')
        writer.writerow(header)
        
        # Data rows
        for s in range(len(t_samples)):
            row = [t_samples[s]]
            row.extend(q_traj[:, s])
            row.extend(dq_traj[:, s])
            row.extend(ddq_traj[:, s])
            writer.writerow(row)
    
    print(f"Trajectory data saved to CSV: {csv_file}")
    
    # Plot trajectory (using original limits for display)
    print("\nGenerating plots...")
    plot_trajectory(t_samples, q_traj, dq_traj, ddq_traj, q_min_original, q_max_original)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

