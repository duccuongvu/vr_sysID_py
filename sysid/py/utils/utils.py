import numpy as np
import pinocchio as pin
import casadi as ca
import json
import matplotlib.pyplot as plt
from scipy import signal
import os
import xml.etree.ElementTree as ET
import re


# Constants
DOF = 7
N_PARAMS = 10 * DOF  # 10 params per link × 7 links
N_FRICTION = 2*DOF  # Dv(7) + Dc(7)
N_ARMATURE = DOF
N_TOTAL =  N_PARAMS + N_FRICTION + N_ARMATURE
DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81])

def skew(c):
        """Return skew-symmetric matrix of a vector c."""
        return np.array([
            [0,     -c[2],  c[1]],
            [c[2],   0,    -c[0]],
            [-c[1],  c[0],   0 ]
        ])

def sign_func(x):
    """Sign function: returns 1 if x > 0, -1 if x < 0, 0 if x == 0."""
    return np.sign(x)

def read_json_data(json_file):
    """Read JSON file and extract q, dq, tau data."""
    print(f"Reading JSON file: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    q_data = []
    dq_data = []
    tau_data = []
    time_data = None
    
    for i in range(1, DOF + 1):
        q_key = f'q{i}_state'
        dq_key = f'dq{i}_state'
        tau_key = f'tau{i}_state'
        
        if q_key in data:
            q_data.append(np.array(data[q_key]))
        else:
            raise ValueError(f"Missing {q_key} in JSON file")
        
        if dq_key in data:
            dq_data.append(np.array(data[dq_key]))
        else:
            raise ValueError(f"Missing {dq_key} in JSON file")
        
        if tau_key in data:
            tau_data.append(np.array(data[tau_key]))
        else:
            raise ValueError(f"Missing {tau_key} in JSON file")
    
    # Check time data
    if 'time' in data:
        time_data = np.array(data['time'])
        if len(time_data) > 1:
            dt = time_data[1] - time_data[0]
        else:
            dt = 0.001
    else:
        dt = 0.001
        n_samples = len(q_data[0])
        time_data = np.arange(n_samples) * dt
    
    # Convert to numpy arrays and transpose: each row is a joint, each column is a sample
    q = np.array(q_data)  # Shape: (DOF, n_samples)
    dq = np.array(dq_data)  # Shape: (DOF, n_samples)
    tau = np.array(tau_data)  # Shape: (DOF, n_samples)
    
    n_samples = q.shape[1]
    print(f"Successfully loaded {n_samples} data points")
    print(f"Time step: {dt:.6f} seconds")
    
    return q, dq, tau, time_data, dt


def read_json_data_new(json_file):
    """Read new JSON file format with header and samples structure.
    
    Expected JSON format:
    {
        "header": {
            "timestamp": "...",
            "dof": 7,
            "joint_names": [...]
        },
        "samples": [
            {
                "t": float,
                "q_cmd": [...],
                "q_state": [...],
                "dq_cmd": [...],
                "dq_state": [...],
                "tau_state": [...],
                "ok": bool
            },
            ...
        ]
    }
    
    Returns:
        q: (DOF, n_samples) - joint positions
        dq: (DOF, n_samples) - joint velocities
        tau: (DOF, n_samples) - joint torques
        time_data: (n_samples,) - time stamps
        dt: float - average time step
    """
    print(f"Reading JSON file (new format): {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Validate header
    if 'header' not in data:
        raise ValueError("Missing 'header' in JSON file")
    if 'samples' not in data:
        raise ValueError("Missing 'samples' in JSON file")
    
    header = data['header']
    samples = data['samples']
    
    # Check DOF
    dof = header.get('dof', DOF)
    if dof != DOF:
        print(f"Warning: JSON DOF ({dof}) does not match expected DOF ({DOF})")
    
    print(f"Joint names: {header.get('joint_names', 'N/A')}")
    print(f"Data timestamp: {header.get('timestamp', 'N/A')}")
    
    # Extract data from samples
    n_samples = len(samples)
    q_data = np.zeros((DOF, n_samples))
    dq_data = np.zeros((DOF, n_samples))
    tau_data = np.zeros((DOF, n_samples))
    time_data = np.zeros(n_samples)
    
    for i, sample in enumerate(samples):
        time_data[i] = sample['t']
        
        # Use q_state, dq_state, tau_state
        q_state = sample['q_state']
        dq_state = sample['dq_state']
        tau_state = sample['tau_state']
        
        # Ensure correct size
        if len(q_state) != DOF or len(dq_state) != DOF or len(tau_state) != DOF:
            raise ValueError(f"Sample {i}: Expected {DOF} values, got q={len(q_state)}, dq={len(dq_state)}, tau={len(tau_state)}")
        
        q_data[:, i] = q_state
        dq_data[:, i] = dq_state
        tau_data[:, i] = tau_state
    
    # Calculate average dt
    if n_samples > 1:
        dt = np.mean(np.diff(time_data))
    else:
        dt = 0.001
    
    print(f"Successfully loaded {n_samples} data points")
    print(f"Time range: {time_data[0]:.6f} to {time_data[-1]:.6f} seconds")
    print(f"Average time step: {dt:.6f} seconds")
    
    return q_data, dq_data, tau_data, time_data, dt


def extract_params(model):
        """Extract parameters in Pinocchio's standard format per arm/link."""
       
        
        n_links = model.nbodies - 1
        X_pin = np.zeros(10 * n_links)
        
        for i in range(1, model.nbodies):
            inertia = model.inertias[i]
            m = inertia.mass
            c = inertia.lever
            I_C = inertia.inertia

            idx = (i - 1) * 10
            X_pin[idx:idx+10] = [m, c[0], c[1], c[2], I_C[0, 0], I_C[0, 1], I_C[0, 2], I_C[1, 1], I_C[1, 2], I_C[2, 2]]

        return X_pin


def pinocchio_standard_params(model):
    """
    Extract parameters in Pinocchio's standard format per arm/link.

    Pinocchio format (10 params per body): 
      [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz], where:
        - m = mass
        - mc_x, mc_y, mc_z = mass * center of mass (i.e., first moment of mass)
        - Ixx, Ixy, Iyy, Ixz, Iyz, Izz = components of spatial inertia matrix
          with respect to the body frame origin (not the CoM).
          I = I_C + m S(c)^T S(c), 
            where I_C is the inertia at the CoM, 
            c is the CoM position, 
            and S(c) is the skew-symmetric matrix of c.
    Returns:
        X_pinocchio: numpy array of shape (10 * n_links,) in Pinocchio format
    """
    def skew(c):
        """Return skew-symmetric matrix of a vector c."""
        return np.array([
            [0,     -c[2],  c[1]],
            [c[2],   0,    -c[0]],
            [-c[1],  c[0],   0 ]
        ])
    
    n_links = model.nbodies - 1  # Exclude universe
    X_pin = np.zeros(10 * n_links)

    for i in range(1, model.nbodies):  # Skip body 0 (universe)
        inertia = model.inertias[i]
        m = inertia.mass
        c = inertia.lever
        I_C = inertia.inertia  # 3x3 inertia matrix at barycenter (CoM)

        # Compute I = I_C + m * S(c)^T * S(c), where S(c) is the skew matrix
        S_c = skew(c)
        I = I_C + m * S_c.T @ S_c  # Inertia at body frame origin

        # Assemble parameter vector v
        mc_x = m * c[0]
        mc_y = m * c[1]
        mc_z = m * c[2]
        Ixx = I[0, 0]
        Ixy = I[0, 1]
        Iyy = I[1, 1]
        Ixz = I[0, 2]
        Iyz = I[1, 2]
        Izz = I[2, 2]

        idx = (i - 1) * 10
        X_pin[idx:idx+10] = [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

    return X_pin


def params_to_x_casadi(x):
    """
    Convert optimization variable x to x_full using CasADi operations.
    
    This function transforms the base parameters and concatenates with friction/armature.
    The transformation applies parallel axis theorem to convert inertia from COM frame to link frame.
    
    Args:
        x: CasADi variable of size N_TOTAL (84) containing:
           - First N_PARAMS (70): base parameters [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] x 7 links
           - Next DOF (7): viscous friction Dv
           - Next DOF (7): coulomb friction Dc  
           - Last DOF (7): armature inertia
    
    Returns:
        x_full: CasADi variable of size N_TOTAL (84) with transformed base parameters
    """
    # Extract components
    X_base = x[:N_PARAMS]  # Base parameters (70)
    Dv = x[N_PARAMS:N_PARAMS + DOF]  # Viscous friction (7)
    Dc = x[N_PARAMS + DOF:N_PARAMS + 2*DOF]  # Coulomb friction (7)
    armature = x[N_PARAMS + 2*DOF:N_TOTAL]  # Armature (7)
    
    # Transform base parameters for each link
    x_transformed = []
    
    for i in range(DOF):
        idx = i * 10
        
        # Extract parameters for this link
        m = X_base[idx]
        c_x = X_base[idx + 1]
        c_y = X_base[idx + 2]
        c_z = X_base[idx + 3]
        Ixx = X_base[idx + 4]
        Ixy = X_base[idx + 5]
        Iyy = X_base[idx + 6]
        Ixz = X_base[idx + 7]
        Iyz = X_base[idx + 8]
        Izz = X_base[idx + 9]
        
        # Build inertia tensor at COM (3x3 symmetric matrix)
        I_C = ca.vertcat(
            ca.horzcat(Ixx, Ixy, Ixz),
            ca.horzcat(Ixy, Iyy, Iyz),
            ca.horzcat(Ixz, Iyz, Izz)
        )
        
        # Build skew-symmetric matrix of COM vector
        c = ca.vertcat(c_x, c_y, c_z)
        S_c = ca.vertcat(
            ca.horzcat(0, -c_z, c_y),
            ca.horzcat(c_z, 0, -c_x),
            ca.horzcat(-c_y, c_x, 0)
        )
        
        # Parallel axis theorem: I = I_C + m * S_c^T @ S_c
        I = I_C + m * ca.mtimes(S_c.T, S_c)
        
        # Extract transformed inertia components
        Ixx_new = I[0, 0]
        Ixy_new = I[0, 1]
        Iyy_new = I[1, 1]
        Ixz_new = I[0, 2]
        Iyz_new = I[1, 2]
        Izz_new = I[2, 2]
        
        # Reconstruct parameters: [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        x_transformed.append(m)
        x_transformed.append(m*c_x)
        x_transformed.append(m*c_y)
        x_transformed.append(m*c_z)
        x_transformed.append(Ixx_new)
        x_transformed.append(Ixy_new)
        x_transformed.append(Iyy_new)
        x_transformed.append(Ixz_new)
        x_transformed.append(Iyz_new)
        x_transformed.append(Izz_new)
    
    # Concatenate transformed base parameters with friction and armature
    x_full = ca.vertcat(
        ca.vertcat(*x_transformed),  # Transformed base params (70)
        Dv,  # Viscous friction (7)
        Dc,  # Coulomb friction (7)
        armature  # Armature (7)
    )
    
    return x_full



def verify_physical_constraints(x_opt):
    """Verify that physical feasibility constraints are satisfied."""
    print("\nVerifying physical feasibility constraints...")
    X_opt = x_opt[:N_PARAMS]
    
    all_satisfied = True
    constraint_violations = []
    
    for link_idx in range(DOF):
        link_start = link_idx * 10
        
        # Extract mass
        m = X_opt[link_start + 0]
        
        # Extract inertia tensor components
        Ixx = X_opt[link_start + 4]
        Ixy = X_opt[link_start + 5]
        Iyy = X_opt[link_start + 6]
        Ixz = X_opt[link_start + 7]
        Iyz = X_opt[link_start + 8]
        Izz = X_opt[link_start + 9]
        
        # Check constraints
        violations = []
        
        # 0. Mass positivity
        if m < 1e-4:
            violations.append(f"  Link {link_idx+1}: m = {m:.6e} < 1e-4")
        
        # 0. Inertia matrix positive definiteness
        # Second leading principal minor: Ixx*Iyy - Ixy^2
        second_minor = Ixx * Iyy - Ixy * Ixy
        if second_minor < 1e-6:
            violations.append(f"  Link {link_idx+1}: Ixx*Iyy - Ixy^2 = {second_minor:.6e} < 1e-6")
        
        # Third leading principal minor (determinant)
        det_I = (Ixx * Iyy * Izz + 
                 2 * Ixy * Ixz * Iyz - 
                 Ixx * Iyz * Iyz - 
                 Iyy * Ixz * Ixz - 
                 Izz * Ixy * Ixy)
        if det_I < 1e-8:
            violations.append(f"  Link {link_idx+1}: det(I) = {det_I:.6e} < 1e-8")
        
        # 1. Triangle inequalities
        if Izz - (Ixx + Iyy) > 1e-6:
            violations.append(f"  Link {link_idx+1}: Izz - (Ixx + Iyy) = {Izz - (Ixx + Iyy):.6e} > 0")
        if Iyy - (Ixx + Izz) > 1e-6:
            violations.append(f"  Link {link_idx+1}: Iyy - (Ixx + Izz) = {Iyy - (Ixx + Izz):.6e} > 0")
        if Ixx - (Iyy + Izz) > 1e-6:
            violations.append(f"  Link {link_idx+1}: Ixx - (Iyy + Izz) = {Ixx - (Iyy + Izz):.6e} > 0")
        
        # 2. Inertia anisotropy (check all pairs)
        if Ixx - 100 * Iyy > 1e-6:
            violations.append(f"  Link {link_idx+1}: Ixx - 100*Iyy = {Ixx - 100*Iyy:.6e} > 0")
        if Ixx - 100 * Izz > 1e-6:
            violations.append(f"  Link {link_idx+1}: Ixx - 100*Izz = {Ixx - 100*Izz:.6e} > 0")
        if Iyy - 100 * Ixx > 1e-6:
            violations.append(f"  Link {link_idx+1}: Iyy - 100*Ixx = {Iyy - 100*Ixx:.6e} > 0")
        if Iyy - 100 * Izz > 1e-6:
            violations.append(f"  Link {link_idx+1}: Iyy - 100*Izz = {Iyy - 100*Izz:.6e} > 0")
        if Izz - 100 * Ixx > 1e-6:
            violations.append(f"  Link {link_idx+1}: Izz - 100*Ixx = {Izz - 100*Ixx:.6e} > 0")
        if Izz - 100 * Iyy > 1e-6:
            violations.append(f"  Link {link_idx+1}: Izz - 100*Iyy = {Izz - 100*Iyy:.6e} > 0")
        
        # 3. Cross inertia terms
        if 3 * abs(Ixy) - Ixx > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixy| - Ixx = {3*abs(Ixy) - Ixx:.6e} > 0")
        if 3 * abs(Ixy) - Iyy > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixy| - Iyy = {3*abs(Ixy) - Iyy:.6e} > 0")
        if 3 * abs(Ixy) - Izz > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixy| - Izz = {3*abs(Ixy) - Izz:.6e} > 0")
        if 3 * abs(Ixz) - Ixx > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixz| - Ixx = {3*abs(Ixz) - Ixx:.6e} > 0")
        if 3 * abs(Ixz) - Iyy > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixz| - Iyy = {3*abs(Ixz) - Iyy:.6e} > 0")
        if 3 * abs(Ixz) - Izz > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Ixz| - Izz = {3*abs(Ixz) - Izz:.6e} > 0")
        if 3 * abs(Iyz) - Ixx > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Iyz| - Ixx = {3*abs(Iyz) - Ixx:.6e} > 0")
        if 3 * abs(Iyz) - Iyy > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Iyz| - Iyy = {3*abs(Iyz) - Iyy:.6e} > 0")
        if 3 * abs(Iyz) - Izz > 1e-6:
            violations.append(f"  Link {link_idx+1}: 3*|Iyz| - Izz = {3*abs(Iyz) - Izz:.6e} > 0")
        
        # 4. Positivity
        if Ixx < 1e-4:
            violations.append(f"  Link {link_idx+1}: Ixx = {Ixx:.6e} < 1e-4")
        if Iyy < 1e-4:
            violations.append(f"  Link {link_idx+1}: Iyy = {Iyy:.6e} < 1e-4")
        if Izz < 1e-4:
            violations.append(f"  Link {link_idx+1}: Izz = {Izz:.6e} < 1e-4")
        
        if violations:
            all_satisfied = False
            constraint_violations.extend(violations)
    
    if all_satisfied:
        print("  All physical feasibility constraints are satisfied!")
    else:
        print("  Some constraint violations detected:")
        for violation in constraint_violations:
            print(violation)


def plot_results(q, dq, dq_filt, ddq_filt, tau, time_data, model, data, x_opt, g, urdf_path, X_init=None, updated_urdf_path=None):
    """Plot data and compare identified model with measured torques."""
    print("\nGenerating plots...")
    
    # Extract solution components
    X_opt = x_opt[:N_PARAMS]
    # Extract friction parameters if they exist
    if N_FRICTION > 0 and len(x_opt) >= N_PARAMS + N_FRICTION:
        Dv = x_opt[N_PARAMS:N_PARAMS + DOF]
        Dc = x_opt[N_PARAMS + DOF:N_PARAMS + DOF + DOF]
    else:
        Dv = np.zeros(DOF)
        Dc = np.zeros(DOF)
    # Extract armature if it exists (check actual array length, not just constant)
    armature_start_idx = N_PARAMS + N_FRICTION
    if len(x_opt) > armature_start_idx:
        # Armature might exist - extract what's available (up to DOF elements)
        available = min(DOF, len(x_opt) - armature_start_idx)
        armature = np.zeros(DOF)
        if available > 0:
            armature[:available] = x_opt[armature_start_idx : armature_start_idx + available]
    else:
        # No armature in array
        armature = np.zeros(DOF)
    
    # Set gravity
    model.gravity.linear = g
    
    # Compute torques from identified model
    n_samples = q.shape[1]
    tau_identified = np.zeros((DOF, n_samples))
    tau_id_no_armature = np.zeros((DOF, n_samples))
    tau_original = None
    tau_opt_no_friction = np.zeros((DOF, n_samples))
    tau_orig_with_friction = None
    tau_updated_urdf = None
    tau_updated_urdf_no_armature = None
    tau_updated_urdf_no_friction = None
    
    if X_init is not None:
        tau_original = np.zeros((DOF, n_samples))
        tau_orig_with_friction = np.zeros((DOF, n_samples))
    
    # Compute torques from updated URDF if provided
    if updated_urdf_path is not None and os.path.exists(updated_urdf_path):
        print("Computing torques from updated URDF...")
        model_updated = pin.buildModelFromUrdf(updated_urdf_path)
        data_updated = model_updated.createData()
        model_updated.gravity.linear = g
        X_updated = pinocchio_standard_params(model_updated)
        
        tau_updated_urdf = np.zeros((DOF, n_samples))
        tau_updated_urdf_no_armature = np.zeros((DOF, n_samples))
        tau_updated_urdf_no_friction = np.zeros((DOF, n_samples))
        
        for s in range(n_samples):
            q_sample = q[:, s]
            dq_sample = dq_filt[:, s]
            ddq_sample = ddq_filt[:, s]
            
            # Compute regressor with updated model
            Phi_updated = pin.computeJointTorqueRegressor(model_updated, data_updated, q_sample, dq_sample, ddq_sample)
            
            # Compute torque from updated URDF base parameters
            tau_base_updated = Phi_updated @ X_updated
            
            # Add friction
            if N_FRICTION > 0:
                tau_friction = Dv * dq_sample + Dc * sign_func(dq_sample)
            else:
                tau_friction = 0
            
            # Add armature
            if len(armature) == DOF and np.any(armature != 0):
                tau_armature = armature * ddq_sample
            else:
                tau_armature = 0
            
            # Total torque from updated URDF
            tau_updated_urdf[:, s] = tau_base_updated + tau_friction + tau_armature
            tau_updated_urdf_no_armature[:, s] = tau_base_updated + tau_friction
            tau_updated_urdf_no_friction[:, s] = tau_base_updated
    
    for s in range(n_samples):
        q_sample = q[:, s]
        dq_sample = dq_filt[:, s]
        ddq_sample = ddq_filt[:, s]
        
        # Compute regressor
        Phi = pin.computeJointTorqueRegressor(model, data, q_sample, dq_sample, ddq_sample)
        
        # Compute torque from identified base parameters
        tau_base = Phi @ X_opt
        
        # Add friction
        if N_FRICTION > 0:
            tau_friction = Dv * dq_sample + Dc * sign_func(dq_sample)
        else:
            tau_friction = 0
        
        # Add armature
        if len(armature) == DOF and np.any(armature != 0):
            tau_armature = armature * ddq_sample
        else:
            tau_armature = 0

        # Total torque (identified)
        tau_identified[:, s] = tau_base + tau_friction + tau_armature
        
        tau_id_no_armature[:, s] = tau_base + tau_friction

        # Torque from optimized parameters without friction
        tau_opt_no_friction[:, s] = tau_base
        
        # Compute torque from original parameters (no friction)
        if X_init is not None:
            tau_original[:, s] = Phi @ X_init
            # Torque from original parameters with optimized friction
            tau_orig_with_friction[:, s] = tau_original[:, s] + tau_friction
    
    # ===== FIRST FIGURE: Position, Velocity, Acceleration, Torque (All Joints) =====
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
        if tau_updated_urdf is not None:
            ax4.plot(time_data, tau_updated_urdf[i, :], 'brown', linestyle=':', linewidth=1.0, label=f'τ{i+1} (updated URDF)', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Torques: Measured vs Identified vs Original (All Joints)')
    # ax4.legend(loc='center right')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save first figure
    # output_file1 = 'sysID_results_figure1.png'
    # plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    # print(f"First figure saved to {output_file1}")
    
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
        if tau_updated_urdf is not None:
            ax.plot(time_data, tau_updated_urdf[joint_idx, :], 'brown', linestyle=':', linewidth=1.5, label='Updated URDF (full)', alpha=0.9)
            ax.plot(time_data, tau_updated_urdf_no_armature[joint_idx, :], 'sienna', linestyle=':', linewidth=1.2, label='Updated URDF (no armature)', alpha=0.8)
            ax.plot(time_data, tau_updated_urdf_no_friction[joint_idx, :], 'peru', linestyle=':', linewidth=1.2, label='Updated URDF (no friction)', alpha=0.8)
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
        
        # Error for updated URDF
        if tau_updated_urdf is not None:
            error_updated = tau[joint_idx, :] - tau_updated_urdf[joint_idx, :]
            rmse_updated = np.sqrt(np.mean(error_updated**2))
            max_error_updated = np.max(np.abs(error_updated))
            error_text += f'\nRMSE (updated URDF): {rmse_updated:.4f} Nm\nMax Error (updated URDF): {max_error_updated:.4f} Nm'
            
            # Error for updated URDF without armature
            error_updated_no_arm = tau[joint_idx, :] - tau_updated_urdf_no_armature[joint_idx, :]
            rmse_updated_no_arm = np.sqrt(np.mean(error_updated_no_arm**2))
            max_error_updated_no_arm = np.max(np.abs(error_updated_no_arm))
            error_text += f'\nRMSE (updated no arm): {rmse_updated_no_arm:.4f} Nm\nMax Error (updated no arm): {max_error_updated_no_arm:.4f} Nm'
            
            # Error for updated URDF without friction
            error_updated_no_fric = tau[joint_idx, :] - tau_updated_urdf_no_friction[joint_idx, :]
            rmse_updated_no_fric = np.sqrt(np.mean(error_updated_no_fric**2))
            max_error_updated_no_fric = np.max(np.abs(error_updated_no_fric))
            error_text += f'\nRMSE (updated no fric): {rmse_updated_no_fric:.4f} Nm\nMax Error (updated no fric): {max_error_updated_no_fric:.4f} Nm'
        
        ax.text(0.02, 0.98, error_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=7)
    
    plt.tight_layout()
    
    # Save second figure
    # output_file2 = 'sysID_results_figure2.png'
    # plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    # print(f"Second figure saved to {output_file2}")
    
    # Show plots
    plt.show()


def format_parameter_comparison_table(X_urdf, X_opt):
    """Format a table comparing URDF and identified parameters as a string."""
    param_names = ['m', 'mc_x', 'mc_y', 'mc_z', 'Ixx', 'Ixy', 'Iyy', 'Ixz', 'Iyz', 'Izz']
    lines = []
    
    lines.append("\n" + "="*130)
    lines.append("Parameter Comparison: URDF vs Identified")
    lines.append("="*130)
    lines.append(f"{'Link':<6} {'Parameter':<10} {'URDF Value':<20} {'Identified':<20} {'Difference':<20} {'Rel Diff %':<12}")
    lines.append("-"*130)
    
    for link_idx in range(DOF):
        link_start = link_idx * 10
        for param_idx, param_name in enumerate(param_names):
            idx = link_start + param_idx
            urdf_val = X_urdf[idx]
            opt_val = X_opt[idx]
            diff = opt_val - urdf_val
            
            # Calculate relative difference, handling edge cases
            if abs(urdf_val) > 1e-10:
                rel_diff = (diff / abs(urdf_val) * 100)
            elif abs(opt_val) > 1e-10:
                rel_diff = float('inf') if diff != 0 else 0.0
            else:
                rel_diff = 0.0
            
            # Format all values in scientific notation for consistency
            urdf_str = f"{urdf_val:>19.6e}"
            opt_str = f"{opt_val:>19.6e}"
            diff_str = f"{diff:>19.6e}"
            
            if rel_diff == float('inf'):
                rel_str = "      inf%"
            else:
                rel_str = f"{rel_diff:>11.2f}%"
            
            lines.append(f"Link{link_idx+1:<2} {param_name:<10} {urdf_str} {opt_str} {diff_str} {rel_str}")
    
    lines.append("="*130)
    return "\n".join(lines)


def print_parameter_comparison_table(X_urdf, X_opt):
    """Print a formatted table comparing URDF and identified parameters."""
    print(format_parameter_comparison_table(X_urdf, X_opt))


def format_results(x_opt, residual_norm, X_init=None):
    """Format identification results as a string."""
    lines = []
    
    # Extract solution components
    X_opt = x_opt[:N_PARAMS]
    Dv = x_opt[N_PARAMS:N_PARAMS + DOF]
    Dc = x_opt[N_PARAMS + DOF:N_PARAMS + DOF + DOF]
    
    lines.append("\n" + "="*70)
    lines.append("System Identification Results")
    lines.append("="*70)
    lines.append(f"\nResidual norm: ||A*x - b|| = {residual_norm:.6e}")
    
    # Add comparison table if X_init is provided
    if X_init is not None:
        lines.append(format_parameter_comparison_table(X_init, X_opt))
    
    lines.append("\nIdentified Parameters:")
    lines.append("\nBase Parameters X (all 70 parameters):")
    for i in range(N_PARAMS):
        lines.append(f"  X[{i:2d}] = {X_opt[i]:.6e}")
    
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
            lines.append(f"  Armature[{i}] = {x_opt[N_PARAMS + 2*DOF + i]:.6e}")
    
    lines.append("\n" + "="*70)
    return "\n".join(lines)


def print_results(x_opt, residual_norm, X_init=None):
    """Print identification results."""
    print(format_results(x_opt, residual_norm, X_init))


def save_results_to_file(x_opt, residual_norm, X_init, json_file, urdf_path, 
                         t_start, t_end, output_dir=None):
    """Save all results to a text file."""
    from datetime import datetime
    
    # Determine output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"sysID_results_{timestamp}.txt")
    
    # Build the complete report
    lines = []
    lines.append("="*80)
    lines.append("System Identification Results - Right Arm with Friction")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"JSON file: {json_file}")
    lines.append(f"URDF file: {urdf_path}")
    lines.append(f"Time range: [{t_start}, {t_end}] seconds")
    lines.append("\n" + "="*80)
    
    # Add main results
    lines.append(format_results(x_opt, residual_norm, X_init))
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def update_urdf_with_parameters(x_opt, urdf_path, output_urdf_path=None):
    """
    Update URDF file with identified parameters.
    
    Parameters:
    - x_opt: Optimized parameters array (first N_PARAMS are the base parameters)
    - urdf_path: Path to input URDF file
    - output_urdf_path: Path to output URDF file (if None, updates in place)
    """
    # Extract base parameters (first N_PARAMS)
    X_opt = x_opt[:N_PARAMS]
    
    # Determine if this is left or right arm based on URDF path
    is_left_arm = 'left' in urdf_path.lower()
    
    # Link names mapping (in order: Link1 to Link7)
    # These correspond to: shoulder_pitch, shoulder_roll, shoulder_yaw, 
    #                       elbow_pitch, wrist_yaw, wrist_roll, wrist_pitch
    if is_left_arm:
        link_names = [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link", 
            "left_shoulder_yaw_link",
            "left_elbow_pitch_link",
            "left_wrist_yaw_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link"
        ]
    else:
        link_names = [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link", 
            "right_shoulder_yaw_link",
            "right_elbow_pitch_link",
            "right_wrist_yaw_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link"
        ]
    
    # Determine output path
    if output_urdf_path is None:
        output_urdf_path = urdf_path
    
    # Read the URDF file line by line to preserve formatting
    with open(urdf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Track which link we're in and what we need to update
    current_link_idx = None
    in_inertial = False
    updated_lines = []
    link_params = {}  # Cache parameters for each link
    
    # Pre-compute parameters for all links
    for link_idx in range(DOF):
        idx = link_idx * 10
        m = X_opt[idx]
        mc_x = X_opt[idx + 1]
        mc_y = X_opt[idx + 2]
        mc_z = X_opt[idx + 3]
        
        # Compute center of mass
        if abs(m) > 1e-10:
            c_x = mc_x / m
            c_y = mc_y / m
            c_z = mc_z / m
        else:
            c_x = c_y = c_z = 0.0
        
        link_params[link_idx] = {
            'm': m,
            'c_x': c_x, 'c_y': c_y, 'c_z': c_z,
            'Ixx': X_opt[idx + 4],
            'Ixy': X_opt[idx + 5],
            'Iyy': X_opt[idx + 6],
            'Ixz': X_opt[idx + 7],
            'Iyz': X_opt[idx + 8],
            'Izz': X_opt[idx + 9]
        }
    
    for i, line in enumerate(lines):
        # Check if we're entering a link we care about
        for link_idx, link_name in enumerate(link_names):
            if f'<link name="{link_name}">' in line:
                current_link_idx = link_idx
                in_inertial = False
                break
        
        # Check if we're entering inertial block
        if current_link_idx is not None and '<inertial>' in line:
            in_inertial = True
        
        # Check if we're leaving the link
        if current_link_idx is not None and '</link>' in line:
            current_link_idx = None
            in_inertial = False
        
        # Update values if we're in the inertial block of a link we care about
        if current_link_idx is not None and in_inertial:
            params = link_params[current_link_idx]
            
            # Update origin xyz
            if 'origin' in line and 'xyz=' in line:
                line = re.sub(r'xyz="[^"]*"', 
                             f'xyz="{params["c_x"]:.9e} {params["c_y"]:.9e} {params["c_z"]:.9e}"', 
                             line)
            
            # Update mass value
            if '<mass' in line and 'value=' in line:
                line = re.sub(r'value="[^"]*"', f'value="{params["m"]:.9e}"', line)
            
            # Update inertia values
            if '<inertia' in line:
                line = re.sub(r'ixx="[^"]*"', f'ixx="{params["Ixx"]:.9e}"', line)
                line = re.sub(r'ixy="[^"]*"', f'ixy="{params["Ixy"]:.9e}"', line)
                line = re.sub(r'ixz="[^"]*"', f'ixz="{params["Ixz"]:.9e}"', line)
                line = re.sub(r'iyy="[^"]*"', f'iyy="{params["Iyy"]:.9e}"', line)
                line = re.sub(r'iyz="[^"]*"', f'iyz="{params["Iyz"]:.9e}"', line)
                line = re.sub(r'izz="[^"]*"', f'izz="{params["Izz"]:.9e}"', line)
        
        updated_lines.append(line)
    
    # Write the updated content
    with open(output_urdf_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"\nURDF updated: {output_urdf_path}")
    return output_urdf_path
