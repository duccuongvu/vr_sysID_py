#!/usr/bin/env python3
"""
System Identification for Right Arm with Friction

This program:
1. Reads q, dq, ddq, and tau from JSON file
2. Filters velocity and acceleration data
3. Computes regressor matrix Phi for each sample (7x70) using Pinocchio
4. Extends regressor to include friction: Phi_ext = [Phi, diag(dq), diag(sign(dq))] (7x84)
5. Stacks all regressors: A = [Phi_ext_1; Phi_ext_2; ...] (DOF*n_sample x 84)
6. Stacks all torques: b = [tau_1; tau_2; ...] (DOF*n_sample x 1)
7. Solves least squares: minimize ||A*x - b||^2
   where x = [X(70); Dv(7); Dc(7)] (84x1)
8. Plots data and compares identified model with measured torques

Usage:
    python sysID_right_arm_with_friction.py <json_file> [--urdf <urdf_path>] [--cutoff_vel <freq>] [--cutoff_acc <freq>]
    python sysID_right_arm_with_friction.py ../json/motion_right_arm.json
    python sysID_right_arm_with_friction.py ../json/motion_right_arm.json --t_start 10.0 --t_end 30.0
"""

import numpy as np
import pinocchio as pin
import casadi as ca
import argparse
import os


from regressors.right_arm_regressor_wrapper import RightArmRegressor
from filter.filter import *
from utils.utils import *


def build_regressor_matrix(model, data, q, dq_filt, ddq_filt, tau, g):
    """
    Build extended regressor matrix with friction.
    
    Returns:
        A: (DOF*n_samples x 84) matrix
        b: (DOF*n_samples x 1) vector
    """
    print("\nBuilding regressor matrices...")
    
    n_samples = q.shape[1]
    M = DOF * n_samples
    
    # Initialize matrices
    A = np.zeros((M, N_TOTAL))
    b = np.zeros((M, 1))
    
    # Set gravity in model
    model.gravity.linear = g
    
    # For each sample, compute regressor and extend with friction
    for s in range(n_samples):
        q_sample = q[:, s]
        dq_sample = dq_filt[:, s]
        ddq_sample = ddq_filt[:, s]
        
        # Compute regressor Phi (7x70) using Pinocchio
        # Phi = pin.computeJointTorqueRegressor(model, data, q_sample, dq_sample, ddq_sample)
        Phi = RightArmRegressor().compute_regressor(q_sample, dq_sample, ddq_sample, DEFAULT_GRAVITY)

        # Extend regressor: Phi_ext = [Phi, diag(dq), diag(sign(dq))] (7x84)
        # For each row i (joint):
        for i in range(DOF):
            row_idx = s * DOF + i  # Row index in stacked matrix A
            
            # Copy Phi row (70 columns)
            A[row_idx, :N_PARAMS] = Phi[i, :]
            
            if N_FRICTION > 0:
                # Add diag(dq): only diagonal element for joint i
                A[row_idx, N_PARAMS + i] = dq_sample[i]
                
                # Add diag(sign(dq)): only diagonal element for joint i
                A[row_idx, N_PARAMS + DOF + i] = sign_func(dq_sample[i])

            if N_ARMATURE > 0:
                # Add armature: only diagonal element for joint i
                A[row_idx, N_PARAMS + 2*DOF + i] = ddq_sample[i]
        
        # Stack torques
        for i in range(DOF):
            row_idx = s * DOF + i
            b[row_idx, 0] = tau[i, s]
    
    print(f"Regressor matrix A: {M} x {N_TOTAL}")
    print(f"Torque vector b: {M} x 1")
    
    return A, b


def solve_least_squares(A, b, x_init=None, add_physical_constraints=True):
    """Solve least squares problem using CasADi with physical feasibility constraints.
    
    Physical constraints applied to inertia tensor for each link:
    1. Triangle inequalities for principal moments of inertia
    2. Upper bound on inertia anisotropy
    3. Bounds on cross inertia terms
    4. Positivity of principal inertias
    """
    print("\nSetting up optimization problem...")
    
    # Convert to CasADi DM
    A_ca = ca.DM(A)
    b_ca = ca.DM(b)
    
    # Create optimization problem
    opti = ca.Opti()
    
    # Define optimization variable x (84x1)
    x = opti.variable(N_TOTAL, 1)
    
    # Extract X (first N_PARAMS elements) - these are the base parameters
    X = x[:N_PARAMS]
    
    # Compute residual: r = A*x - b
    residual = ca.mtimes(A_ca, x) - b_ca
    
    # Objective: minimize ||r||^2 = r^T * r
    objective = ca.dot(residual, residual)
    opti.minimize(objective)
    
    # Add physical feasibility constraints
    # if add_physical_constraints:
    #     print("Adding physical feasibility constraints...")
        
    #     # For each link, extract inertia tensor components
    #     # Pinocchio format per link: [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
    #     for link_idx in range(DOF):
    #         link_start = link_idx * 10
            
    #         # Extract mass
    #         m = X[link_start + 0]
            
    #         # Extract inertia tensor components
    #         Ixx = X[link_start + 4]
    #         Ixy = X[link_start + 5]
    #         Iyy = X[link_start + 6]
    #         Ixz = X[link_start + 7]
    #         Iyz = X[link_start + 8]
    #         Izz = X[link_start + 9]
            
    #         # 0. Mass positivity: m > 0
    #         opti.subject_to(-m + 1e-4 <= 0)  # m >= 1e-4
            
    #         # 0. Inertia matrix positive definiteness (eigenvalues > 0)
    #         # For a 3x3 symmetric matrix to be positive definite, all leading principal minors must be positive
    #         # First minor: Ixx > 0 (already enforced below)
    #         # Second minor: Ixx*Iyy - Ixy^2 > 0
    #         opti.subject_to(-(Ixx * Iyy - Ixy * Ixy) + 1e-6 <= 0)  # Ixx*Iyy - Ixy^2 >= 1e-6
    #         # Third minor (determinant): det(I) > 0
    #         # det(I) = Ixx*Iyy*Izz + 2*Ixy*Ixz*Iyz - Ixx*Iyz^2 - Iyy*Ixz^2 - Izz*Ixy^2
    #         det_I = (Ixx * Iyy * Izz + 
    #                  2 * Ixy * Ixz * Iyz - 
    #                  Ixx * Iyz * Iyz - 
    #                  Iyy * Ixz * Ixz - 
    #                  Izz * Ixy * Ixy)
    #         opti.subject_to(-det_I + 1e-8 <= 0)  # det(I) >= 1e-8
            

    #         # armature constraints
    #         for i in range(DOF):
    #             opti.subject_to(-X[N_PARAMS + 2*DOF + i] + 0.1 <= 0)  # armature >= 0.1
    #             opti.subject_to(-X[N_PARAMS + 2*DOF + i] - 1.0 <= 0)  # armature <= 1.0

    #         # 1. Triangle inequalities for principal moments of inertia
    #         # Izz - (Ixx + Iyy) <= 0
    #         opti.subject_to(Izz - (Ixx + Iyy) <= 0)
    #         # Iyy - (Ixx + Izz) <= 0
    #         opti.subject_to(Iyy - (Ixx + Izz) <= 0)
    #         # Ixx - (Iyy + Izz) <= 0
    #         opti.subject_to(Ixx - (Iyy + Izz) <= 0)
            
    #         # 2. Upper bound on inertia anisotropy
    #         # max(Ixx, Iyy, Izz) - 100 * min(Ixx, Iyy, Izz) <= 0
    #         # This is equivalent to: each inertia <= 100 * each other inertia
    #         # Ixx <= 100 * Iyy, Ixx <= 100 * Izz
    #         opti.subject_to(Ixx - 100 * Iyy <= 0)
    #         opti.subject_to(Ixx - 100 * Izz <= 0)
    #         # # Iyy <= 100 * Ixx, Iyy <= 100 * Izz
    #         opti.subject_to(Iyy - 100 * Ixx <= 0)
    #         opti.subject_to(Iyy - 100 * Izz <= 0)
    #         # # Izz <= 100 * Ixx, Izz <= 100 * Iyy
    #         opti.subject_to(Izz - 100 * Ixx <= 0)
    #         opti.subject_to(Izz - 100 * Iyy <= 0)
            
    #         # # 3. Bounds on cross inertia terms
    #         # # 3 * abs(Ixy) - min(Ixx, Iyy, Izz) <= 0
    #         # # This is equivalent to: 3*|Ixy| <= Ixx, 3*|Ixy| <= Iyy, 3*|Ixy| <= Izz
    #         # opti.subject_to(30 * ca.fabs(Ixy) - Ixx <= 0)
    #         # opti.subject_to(30 * ca.fabs(Ixy) - Iyy <= 0)
    #         # opti.subject_to(30 * ca.fabs(Ixy) - Izz <= 0)
    #         # # Same for Ixz
    #         # opti.subject_to(30 * ca.fabs(Ixz) - Ixx <= 0)
    #         # opti.subject_to(30 * ca.fabs(Ixz) - Iyy <= 0)
    #         # opti.subject_to(30 * ca.fabs(Ixz) - Izz <= 0)
    #         # # Same for Iyz
    #         # opti.subject_to(30 * ca.fabs(Iyz) - Ixx <= 0)
    #         # opti.subject_to(30 * ca.fabs(Iyz) - Iyy <= 0)
    #         # opti.subject_to(30 * ca.fabs(Iyz) - Izz <= 0)
            
    #         # 4. Positivity of principal inertias
    #         # -Ixx + 1e-4 <= 0  =>  Ixx >= 1e-4
    #         opti.subject_to(-Ixx + 1e-4 <= 0)
    #         # -Iyy + 1e-4 <= 0  =>  Iyy >= 1e-4
    #         opti.subject_to(-Iyy + 1e-4 <= 0)
    #         # -Izz + 1e-4 <= 0  =>  Izz >= 1e-4
    #         opti.subject_to(-Izz + 1e-4 <= 0)
        # print(f"  Added {DOF * 24} physical constraints ({DOF} links × 24 constraints per link)")
    
    # --- Constrain main link parameters to within 70%-130% of original (URDF) values ---
    # Only for the 70 dynamics parameters (not friction etc.)
    origin_params = x_init[:N_PARAMS] if x_init is not None else np.ones(N_PARAMS)
    for i in range(N_PARAMS):
        # The structure is: [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] × DOF
        param_type = i % 10
        if param_type == 0:
            # Mass: ±10%
            bound = 0.1 * np.abs(origin_params[i])
        elif param_type in [1, 2, 3]:
            # COM: ±30%
            bound = 0.3 * np.abs(origin_params[i])
        else:
            # Inertia: ±50%
            bound = 0.0 * np.abs(origin_params[i])
        opti.subject_to(x[i] >= origin_params[i] - bound)
        opti.subject_to(x[i] <= origin_params[i] + bound)

    # Set initial guess
    if x_init is not None:
        opti.set_initial(x, ca.DM(x_init))
    else:
        opti.set_initial(x, ca.DM.zeros(N_TOTAL, 1))
    
    # Choose solver
    opti.solver('ipopt')
    
    print("Solving least squares problem with CasADi...")
    
    # Solve
    sol = opti.solve()
    
    # Extract solution
    x_opt = np.array(sol.value(x)).flatten()

    # Compute residual for verification
    residual_opt = A @ x_opt.reshape(-1, 1) - b
    residual_norm = np.linalg.norm(residual_opt)
    residual_norm_sq = residual_norm ** 2
    
    print(f"\nResidual norm: ||A*x - b|| = {residual_norm:.6e}")
    print(f"Residual squared norm: ||A*x - b||^2 = {residual_norm_sq:.6e}")
    
    # Verify constraints if they were applied
    # if add_physical_constraints:
    verify_physical_constraints(x_opt)
    
    return x_opt, residual_norm


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
    parser.add_argument('--no-constraints', action='store_true',
                       help='Disable physical feasibility constraints')
    
    args = parser.parse_args()
    
    # Determine URDF path
    if args.urdf is None:
        # Try default paths
        default_paths = [
            # '../sysid/matlab/vr_m2_right_arm.urdf',
            # '../../sysid/matlab/vr_m2_right_arm.urdf',
            # '../models/urdf/origin/vr_m2_right_arm_updated.urdf',
            # '../../models/urdf/origin/vr_m2_right_arm_updated.urdf'
            '../../vr_m2_description/urdf/origin/vr_m2_right_arm.urdf'
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
    q, dq, tau, time_data, dt = read_json_data_new(args.json_file)
    
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
    
    # Get initial condition from model
    X_init = pinocchio_standard_params(model)
    x_init = np.zeros(N_TOTAL)
    x_init[:N_PARAMS] = X_init
    # Set the armature (motor inertia) parameters to 0.3 for each joint in the initial guess
    x_init[N_PARAMS + 2*DOF : N_PARAMS + 2*DOF + N_ARMATURE] = 0.5
    # Friction parameters initialized to zero
    
    # Solve least squares
    add_constraints = not args.no_constraints
    x_opt, residual_norm = solve_least_squares(A, b, x_init, add_physical_constraints=False)
    
    # Print results with comparison table
    print_results(x_opt, residual_norm, X_init=X_init)
    
    # Save results to file
    save_results_to_file(x_opt, residual_norm, X_init, args.json_file, urdf_path, 
                        args.t_start, args.t_end)
    
    # Update left arm URDF with identified parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # left_arm_urdf_path = os.path.normpath(os.path.join(script_dir, '../../models/urdf/origin/vr_m2_left_arm_updated_sysID.urdf'))
    right_arm_urdf_path = os.path.normpath(os.path.join(script_dir, '../../vr_m2_description/urdf/origin/vr_m2_right_arm.urdf'))
    
    # if os.path.exists(left_arm_urdf_path):
    #     print(f"\nUpdating left arm URDF with identified parameters...")
    #     update_urdf_with_parameters(x_opt, left_arm_urdf_path)
    # else:
    #     print(f"\nWarning: Left arm URDF not found at {left_arm_urdf_path}")
    #     print("Skipping URDF update.")

    updated_urdf_path = None
    if os.path.exists(right_arm_urdf_path):
        print(f"\nUpdating right arm URDF with identified parameters...")
        update_urdf_with_parameters(x_opt, right_arm_urdf_path)
        updated_urdf_path = right_arm_urdf_path
    else:
        print(f"\nWarning: Right arm URDF not found at {right_arm_urdf_path}")
        print("Skipping URDF update.")
    
    # Plot results
    if not args.no_plot:
        plot_results(q, dq, dq_filt, ddq_filt, tau, time_data, model, data, x_opt, DEFAULT_GRAVITY, urdf_path, X_init=X_init, updated_urdf_path=updated_urdf_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
