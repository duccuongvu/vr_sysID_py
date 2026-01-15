"""
Generate dynamic regressor matrix Phi(q, v, a, g) and parameter vector X using Pinocchio and CasADi.

This script generates C code for computing the dynamic regressor matrix, which relates
joint torques to inertial parameters via: tau = Phi(q, v, a, g) * X

IMPORTANT: Parameter Format
---------------------------
Pinocchio uses a different parameter format than MATLAB:

Pinocchio format (10 params per link):
    [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    where:
        - m = mass
        - mx, my, mz = mass * com (first moment of mass)
        - Ixx, Ixy, Ixz, Iyy, Iyz, Izz = inertia tensor components

MATLAB format (10 params per link):
    [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, px, py, pz, m]
    where:
        - Ixx, Ixy, Ixz, Iyy, Iyz, Izz = inertia tensor components
        - px, py, pz = mass * com (first moment of mass)
        - m = mass

The generated functions use Pinocchio's format. Use convert_pinocchio_to_matlab_params()
to convert between formats if needed.

Generated Functions:
--------------------
- {arm_name}_regressor(q, v, a, g): Computes regressor matrix Phi
- {arm_name}_params_to_X(...): Builds parameter vector X from individual parameters
- {arm_name}_tau_from_regressor(X, q, v, a, g): Computes tau = Phi * X

Usage:
------
    python gen_vr_m2_regressor.py              # Generate code for both arms
    python gen_vr_m2_regressor.py --test        # Run validation tests
"""

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi as ca
import numpy as np
import os
import time
from scipy.linalg import qr
import sys
import argparse


def load_model(urdf_path: str):
    """Load URDF model and return numeric and CasADi versions."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    cmodel = cpin.Model(model)
    cdata = cpin.Data(cmodel)
    return model, data, cmodel, cdata


def extract_inertial_parameters(model):
    """Extract inertial parameters from URDF model.
    
    Returns:
        params: Dictionary with keys 'mass', 'com', 'inertia' for each body
    """
    params = {}
    for i in range(1, model.nbodies):  # Skip body 0 (universe)
        inertia = model.inertias[i]
        params[i] = {
            'mass': inertia.mass,
            'com': inertia.lever,  # Center of mass in body frame
            'inertia': inertia.inertia  # 3x3 inertia matrix
        }
    return params


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



def casadi_regressor_functions(cmodel, cdata, arm_name: str):
    """Generate CasADi functions for regressor Phi(q, v, a, g) and parameter vector X.
    
    The regressor satisfies: tau = Phi(q, v, a, g) * X
    
    Args:
        cmodel: CasADi model
        cdata: CasADi data
        arm_name: 'left_arm' or 'right_arm'
    
    Returns:
        Dictionary of CasADi functions:
            - Phi: Regressor matrix Phi(q, v, a, g)
            - X_from_params: Function to build X from individual parameters
            - tau_from_regressor: Function tau = Phi * X
    """
    nq, nv = cmodel.nq, cmodel.nv
    n_links = cmodel.nbodies - 1
    
    # Symbolic variables
    q = ca.SX.sym('q', nq)
    v = ca.SX.sym('v', nv)
    a = ca.SX.sym('a', nv)
    g = ca.SX.sym('g', 3)  # Gravity vector
    
    # Set gravity in model (this affects the regressor computation)
    # Note: For CasADi models, gravity is already symbolic, so we can set it directly
    # Store original gravity value (it's a CasADi SX, so we can't easily copy it)
    # Instead, we'll set it and restore it after if needed
    cmodel.gravity.linear = g
    
    # Compute regressor using Pinocchio's CasADi interface
    # Pinocchio's computeJointTorqueRegressor returns regressor for standard 10-parameter format
    print(f"  Computing regressor matrix using Pinocchio CasADi...")
    Phi_sym = cpin.computeJointTorqueRegressor(cmodel, cdata, q, v, a)

    # For the parameter vector X, we'll create a function that maps from
    # individual parameters to the stacked vector in Pinocchio's format
    # Pinocchio format: [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] per link
    params_list = []
    # Build symbolic per-link physical parameters
    m_syms = []
    com_syms = []
    inertia_syms = []
    params_list = []
    X_syms = []
    for i in range(n_links):
        # Individual symbolic parameters per link
        m = ca.SX.sym(f'm_{i}', 1)                # mass
        cx = ca.SX.sym(f'com_x_{i}', 1)           # center of mass x
        cy = ca.SX.sym(f'com_y_{i}', 1)           # center of mass y
        cz = ca.SX.sym(f'com_z_{i}', 1)           # center of mass z
        Ixx = ca.SX.sym(f'Ixx_{i}', 1)
        Ixy = ca.SX.sym(f'Ixy_{i}', 1)
        Ixz = ca.SX.sym(f'Ixz_{i}', 1)
        Iyy = ca.SX.sym(f'Iyy_{i}', 1)
        Iyz = ca.SX.sym(f'Iyz_{i}', 1)
        Izz = ca.SX.sym(f'Izz_{i}', 1)

        m_syms.append(m)
        com_syms.append([cx, cy, cz])
        inertia_syms.append([Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

        # Convert to Pinocchio's standard (origin) parameters, see pinocchio_standard_params:
        # S(c) = skew-symmetric matrix of c
        # I (about body origin) = I_C + m * S(c).T * S(c)
        # First moment [m*cx, m*cy, m*cz]
        mcx = m * cx
        mcy = m * cy
        mcz = m * cz

        # Build inertia matrix at CoM
        I_C = ca.vertcat(
            ca.horzcat(Ixx, Ixy, Ixz),
            ca.horzcat(Ixy, Iyy, Iyz),
            ca.horzcat(Ixz, Iyz, Izz)
        )
        c_vec = ca.vertcat(cx, cy, cz)
        S_c = ca.vertcat(
            ca.horzcat(0,     -cz,  cy),
            ca.horzcat(cz,     0,  -cx),
            ca.horzcat(-cy,   cx,   0 )
        )

        I = I_C + m * ca.mtimes(S_c.T, S_c)

        # Extract upper-triangular elements in Pinocchio order
        # [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        # (follows the same order as pinocchio_standard_params)
        out_params = [
            m,
            mcx, mcy, mcz,
            I[0,0], I[0,1], I[1,1], I[0,2], I[1,2], I[2,2]
        ]
        params_list.extend([m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])  # For X_from_params function (physical params)
        X_syms.extend(out_params)  # For regressors (Pinocchio format)

    # Stack parameters into X vector (Pinocchio format)
    X_sym = ca.vertcat(*X_syms)
    
    # Phi_sym from Pinocchio uses Pinocchio's parameter format, so X_sym must match
    
    # For tau_from_regressor function, we need a pure symbolic variable for X
    # (not an expression, as CasADi requires purely symbolic inputs)
    n_params = len(X_syms)
    X_input = ca.SX.sym('X', n_params)
    
    # Compute torque: tau = Phi * X
    tau_sym = ca.mtimes(Phi_sym, X_input)
    
    # Construct function names
    regressor_name = f'{arm_name}_regressor'
    X_from_params_name = f'{arm_name}_params_to_X'
    tau_from_regressor_name = f'{arm_name}_tau_from_regressor'
    
    # Create functions
    functions = {
        regressor_name: ca.Function(regressor_name, [q, v, a, g], [Phi_sym]),
        X_from_params_name: ca.Function(X_from_params_name, params_list, [X_sym]),
        tau_from_regressor_name: ca.Function(tau_from_regressor_name, [X_input, q, v, a, g], [tau_sym])
    }
    
    return functions, Phi_sym, X_sym, params_list  # Return functions, Phi_sym, X_sym expression, and params_list


def calculate_base_parameters(model, data, Phi_func, X_sym, n_links, zero_q=False, zero_qd=False, zero_qdd=False, zero_g=False, n_samples=25):
    """
    Calculate base parameters using QR decomposition.
    
    This function finds the minimal set of identifiable parameters by:
    1. Building an observation matrix W by evaluating Phi at multiple configurations
    2. Performing QR decomposition with pivoting to identify independent/dependent columns
    3. Computing the relationship between independent and dependent parameters
    4. Constructing Phi_base and X_base
    
    Args:
        model: Pinocchio model (for numeric evaluation)
        data: Pinocchio data
        Phi_func: CasADi function for Phi(q, v, a, g)
        X_sym: CasADi symbolic vector of full parameters
        n_links: Number of links
        zero_q: If True, set q=0 when building observation matrix
        zero_qd: If True, set v=0 when building observation matrix
        zero_qdd: If True, set a=0 when building observation matrix
        zero_g: If True, set g=0 when building observation matrix
        n_samples: Number of random samples for observation matrix
    
    Returns:
        Tuple (Phi_base_sym, X_base_sym, E, beta, n_base_params):
            - Phi_base_sym: CasADi symbolic base regressor matrix
            - X_base_sym: CasADi symbolic base parameter vector
            - E: Permutation matrix from QR decomposition (numpy array)
            - beta: Relationship matrix between independent and dependent params
            - n_base_params: Number of base parameters
    """
    nq, nv = model.nq, model.nv
    n_params = 10 * n_links
    
    # Build observation matrix W
    print(f"  Building observation matrix with {n_samples} random configurations...")
    W_list = []
    
    q_min = -np.pi * np.ones(nq)
    q_max = np.pi * np.ones(nq)
    v_max = 3 * np.pi * np.ones(nv)
    a_max = 6 * np.pi * np.ones(nv)
    
    for i in range(n_samples):
        if zero_q:
            q_rnd = np.zeros(nq)
        else:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(nq)
        
        if zero_qd:
            v_rnd = np.zeros(nv)
        else:
            v_rnd = -v_max + 2 * v_max * np.random.rand(nv)
        
        if zero_qdd:
            a_rnd = np.zeros(nv)
        else:
            a_rnd = -a_max + 2 * a_max * np.random.rand(nv)
        
        if zero_g:
            g_rnd = np.zeros(3)
        else:
            g_rnd = np.random.rand(3)
            g_rnd = 9.81 * g_rnd / np.linalg.norm(g_rnd)
        
        # Evaluate Phi at this configuration
        Phi_val = Phi_func(q_rnd, v_rnd, a_rnd, g_rnd)
        W_list.append(Phi_val)
    
    W = np.vstack(W_list)
    print(f"  Observation matrix shape: {W.shape}")
    
    # QR decomposition with pivoting
    print(f"  Performing QR decomposition with pivoting...")
    Q, R, pivots = qr(W, pivoting=True, mode='economic')
    
    # Convert pivots (1D array of column indices) to permutation matrix E
    # pivots[i] = j means column j of original matrix is now column i in permuted matrix
    # To get permutation matrix E such that W * E gives permuted columns:
    # E[j, i] = 1 if pivots[i] == j, else 0
    E = np.eye(n_params, dtype=int)[:, pivots]
    
    # Find rank (number of base parameters)
    # Use a tolerance based on machine epsilon
    tol = np.sqrt(np.finfo(float).eps) * max(W.shape) * np.max(np.abs(np.diag(R)))
    rank_mask = np.abs(np.diag(R)) > tol
    n_base_params = np.sum(rank_mask)
    
    print(f"  Full parameter count: {n_params}")
    print(f"  Base parameter count: {n_base_params}")
    print(f"  Reduction: {n_params - n_base_params} parameters removed")
    
    # Extract independent and dependent columns
    R1 = R[:n_base_params, :n_base_params]
    R2 = R[:n_base_params, n_base_params:]
    
    # Compute relationship: R2 = R1 * beta, so beta = R1^(-1) * R2
    beta = np.linalg.solve(R1, R2)
    beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0.0
    
    # Verify relationship
    W1 = W @ E[:, :n_base_params]
    W2 = W @ E[:, n_base_params:]
    error = np.linalg.norm(W2 - W1 @ beta)
    if error > 1e-4:
        print(f"  Warning: Relationship verification error: {error:.2e} (expected < 1e-4)")
    else:
        print(f"  Relationship verified (error: {error:.2e})")
    
    # Keep numpy version for indexing
    E_np = E
    
    return E_np, beta, n_base_params


def print_parameter_relationship(E, beta, n_base_params, n_links, n_params):
    """
    Print the relationship between full and base parameters.
    
    Args:
        E: Permutation matrix from QR decomposition
        beta: Relationship matrix (X_base = X1 + beta * X2)
        n_base_params: Number of base parameters
        n_links: Number of links
        n_params: Total number of parameters (10 * n_links)
    """ 
    # Parameter names for each link (Pinocchio format)
    param_names = ['m', 'mc_x', 'mc_y', 'mc_z', 'Ixx', 'Ixy', 'Iyy', 'Ixz', 'Iyz', 'Izz']
    
    # Find which original parameter indices correspond to base parameters
    # E is a permutation matrix: E[:, i] selects column i from original matrix
    # The first n_base_params columns of E correspond to independent (base) parameters
    base_param_indices = []
    dependent_param_indices = []
    
    print(f"\n{'='*80}")
    print(f"  Given full parameter vector X_full (length {n_params}):")
    print()
    print("  Permute using permutation matrix E:")
    print("     X_permuted = E^T @ X_full")
    print()
    print("  Split into independent (X1) and dependent (X2) parts:")
    print(f"     X1 = X_permuted[:{n_base_params}]        (independent parameters)")
    print(f"     X2 = X_permuted[{n_base_params}:]        (dependent parameters)")
    print()
    
    # Show which parameters are in X1
    print(f"  X1 contains the following parameters (independent, {n_base_params} total):")
    for i, orig_idx in enumerate(base_param_indices):
        link_idx = orig_idx // 10
        param_idx = orig_idx % 10
        param_name = param_names[param_idx]
        print(f"    X1[{i:2d}] = X_full[{orig_idx:2d}] = Link{link_idx}.{param_name}")
    print()
    
    # Show which parameters are in X2
    print(f"  X2 contains the following parameters (dependent, {n_params - n_base_params} total):")
    for i, orig_idx in enumerate(dependent_param_indices):
        link_idx = orig_idx // 10
        param_idx = orig_idx % 10
        param_name = param_names[param_idx]
        print(f"    X2[{i:2d}] = X_full[{orig_idx:2d}] = Link{link_idx}.{param_name}")
    print()
    
    print("  Calculate base parameters:")
    print("     X_base = X1 + beta @ X2")
    print()
    print("  Where:")
    print(f"    - E: permutation matrix from QR decomposition ({n_params} × {n_params})")
    print(f"    - beta: relationship matrix ({n_base_params} × {n_params - n_base_params})")
    print(f"    - X_base: reduced parameter vector (length {n_base_params})")
    print()
    print("  In Python/NumPy:")
    print("    X_permuted = E.T @ X_full")
    print(f"    X1 = X_permuted[:{n_base_params}]")
    print(f"    X2 = X_permuted[{n_base_params}:]")
    print("    X_base = X1 + beta @ X2")
    print(f"\n{'='*80}")


def casadi_base_regressor_functions(cmodel, cdata, X_sym_expr, params_list, E, beta, n_base_params, arm_name: str):
    """
    Generate CasADi functions for base regressor Phi_base and base parameter vector X_base.
    
    Args:
        cmodel: CasADi model
        cdata: CasADi data
        X_sym_expr: CasADi expression that builds X from params_list
        params_list: List of individual parameter symbols (for X_base_from_params)
        E: Permutation matrix from QR (numpy array)
        beta: Relationship matrix (numpy array)
        n_base_params: Number of base parameters
        arm_name: 'left_arm' or 'right_arm'
    
    Returns:
        Dictionary of CasADi functions for base parameters
    """
    nq, nv = cmodel.nq, cmodel.nv
    
    # Symbolic variables
    q = ca.SX.sym('q', nq)
    v = ca.SX.sym('v', nv)
    a = ca.SX.sym('a', nv)
    g = ca.SX.sym('g', 3)
    
    # Set gravity
    cmodel.gravity.linear = g
    
    # Recompute Phi_sym with the new symbolic variables to ensure consistency
    # (Phi_sym from the calling function may have different symbolic variables)
    Phi_sym_new = cpin.computeJointTorqueRegressor(cmodel, cdata, q, v, a)
    
    # Construct Phi_base by selecting independent columns: Phi_base = Phi * E(:, 1:n_base_params)
    # Convert E to CasADi constant
    E_ca = ca.DM(E[:, :n_base_params])  # Select first n_base_params columns
    Phi_base_sym = ca.mtimes(Phi_sym_new, E_ca)
    
    # Construct X_base: X_base = X1 + beta * X2 where:
    # X_permuted = E' * X
    # X1 = X_permuted[:n_base_params] (independent)
    # X2 = X_permuted[n_base_params:] (dependent)
    # X_base = X1 + beta * X2
    # First, we need a symbolic variable for X to build the expression
    # Get size from X_sym_expr (CasADi SX expression)
    n_params_full = X_sym_expr.size1()  # size1() gives number of rows (length for a vector)
    X_sym_var = ca.SX.sym('X', n_params_full)
    E_full_ca = ca.DM(E)  # Full permutation matrix
    X_permuted = ca.mtimes(E_full_ca.T, X_sym_var)
    
    # Split into independent (X1) and dependent (X2) parts
    X1 = X_permuted[:n_base_params]
    X2 = X_permuted[n_base_params:]
    
    # Convert beta to CasADi constant
    beta_ca = ca.DM(beta)
    
    # Compute X_base = X1 + beta * X2
    X_base_sym = X1 + ca.mtimes(beta_ca, X2)
    
    # Create X_base_from_params by substituting X_sym_var with X_sym_expr
    X_base_from_params_expr = ca.substitute(X_base_sym, X_sym_var, X_sym_expr)
    
    # For tau_from_regressor_base function
    X_base_input = ca.SX.sym('X_base', n_base_params)
    tau_base_sym = ca.mtimes(Phi_base_sym, X_base_input)
    
    # Construct function names
    regressor_base_name = f'{arm_name}_regressor_base'
    X_base_from_params_name = f'{arm_name}_params_to_X_base'
    tau_from_regressor_base_name = f'{arm_name}_tau_from_regressor_base'
    
    # Create functions
    functions = {
        regressor_base_name: ca.Function(regressor_base_name, [q, v, a, g], [Phi_base_sym]),
        X_base_from_params_name: ca.Function(X_base_from_params_name, params_list, [X_base_from_params_expr]),
        tau_from_regressor_base_name: ca.Function(tau_from_regressor_base_name, [X_base_input, q, v, a, g], [tau_base_sym])
    }
    
    return functions


def verify_base_parameters(Phi_func, X_full, Phi_base_func, X_base, q, v, a, g):
    """
    Verify that Phi_base * X_base produces the same torque as Phi * X.
    
    Args:
        Phi_func: CasADi function to compute full regressor Phi(q, v, a, g)
        X_full: Full parameter vector (numpy array)
        Phi_base_func: CasADi function to compute base regressor Phi_base(q, v, a, g)
        X_base: Base parameter vector (numpy array)
        q: Joint positions (numpy array)
        v: Joint velocities (numpy array)
        a: Joint accelerations (numpy array)
        g: Gravity vector (numpy array)
    
    Returns:
        error: Maximum absolute error between tau_full and tau_base
        tau_full: Torque from full regressor
        tau_base: Torque from base regressor
    """
    # Compute torque using full regressor
    Phi_full = np.array(Phi_func(q, v, a, g))
    tau_full = Phi_full @ X_full
    
    # Compute torque using base regressor
    Phi_base = np.array(Phi_base_func(q, v, a, g))
    tau_base = Phi_base @ X_base
    
    # Compare
    error = np.max(np.abs(tau_full - tau_base))
    
    return error, tau_full, tau_base

def fourier_trajectory(t, A, B, omega, n_dof, n_fourier):
    """
    Create Fourier series trajectory and its derivatives using CasADi symbolic expressions.

    Args:
        t: Time (scalar)
        A: CasADi SX or MX matrix (n_dof, n_fourier)
        B: CasADi SX or MX matrix (n_dof, n_fourier)
        omega: Frequencies (n_fourier,)
        n_dof: Number of DOF (int)
        n_fourier: Number of Fourier terms (int)
    Returns:
        q, dq, ddq: CasADi symbolic vectors (n_dof,)
    """

    # Make sure shapes are consistent

    sin_mat = ca.sin(t * omega)   # shape (n_fourier,)
    cos_mat = ca.cos(t * omega)   # shape (n_fourier,)

    # Reshape for broadcasting
    sin_mat = ca.reshape(sin_mat, (1, n_fourier))      # (1, n_fourier)
    cos_mat = ca.reshape(cos_mat, (1, n_fourier))      # (1, n_fourier)
    omega_row = ca.reshape(omega, (1, n_fourier))      # (1, n_fourier)

    q = ca.SX.zeros(n_dof)
    dq = ca.SX.zeros(n_dof)
    ddq = ca.SX.zeros(n_dof)

    for i in range(n_dof):
        for j in range(n_fourier):
            q[i] += (A[i, j] / omega[j]) * sin_mat[j] - (B[i, j] / omega[j]) * cos_mat[j]
            dq[i] += A[i, j] * cos_mat[j] + B[i, j] * sin_mat[j]
            ddq[i] += -(A[i, j] / omega[j]) * omega[j] * omega[j] * sin_mat[j] + (B[i, j] / omega[j]) * omega[j] * omega[j] * cos_mat[j]

    return q, dq, ddq

def condition_number(Phi_func, q, dq, ddq, g):
    """
    Compute CasADi symbolic condition number of the regressor matrix (for a single sample).
    Note: This uses CasADi operations, not numpy.
    """
    Phi = Phi_func(q, dq, ddq, g)
    s = ca.svd(Phi)[1]  # s is the vector of singular values
    cond = s[0] / s[-1]  # largest / smallest singular value
    return cond

if __name__ == '__main__':
    urdf_path = '../models/urdf/origin/vr_m2_right_arm_updated.urdf'
    model, data, cmodel, cdata = load_model(urdf_path)
    nq, nv = cmodel.nq, cmodel.nv
    n_links = model.nbodies - 1
    
    q = ca.SX.sym('q', nq)
    v = ca.SX.sym('v', nv)
    a = ca.SX.sym('a', nv)
    g = ca.SX.sym('g', 3)
    # Set gravity in model (needed for regressor computation)
    cmodel.gravity.linear = g
    Phi_sym = cpin.computeJointTorqueRegressor(cmodel, cdata, q, v, a)
    # Create a callable function from Phi_sym
    Phi_func = ca.Function('Phi', [q, v, a, g], [Phi_sym])
    
    # Build params_list and X_sym_expr (same structure as in casadi_regressor_functions)
    params_list = []
    X_syms = []
    for i in range(n_links):
        # Individual symbolic parameters per link
        m = ca.SX.sym(f'm_{i}', 1)                # mass
        cx = ca.SX.sym(f'com_x_{i}', 1)           # center of mass x
        cy = ca.SX.sym(f'com_y_{i}', 1)           # center of mass y
        cz = ca.SX.sym(f'com_z_{i}', 1)           # center of mass z
        Ixx = ca.SX.sym(f'Ixx_{i}', 1)
        Ixy = ca.SX.sym(f'Ixy_{i}', 1)
        Ixz = ca.SX.sym(f'Ixz_{i}', 1)
        Iyy = ca.SX.sym(f'Iyy_{i}', 1)
        Iyz = ca.SX.sym(f'Iyz_{i}', 1)
        Izz = ca.SX.sym(f'Izz_{i}', 1)

        # Convert to Pinocchio's standard (origin) parameters
        mcx = m * cx
        mcy = m * cy
        mcz = m * cz

        # Build inertia matrix at CoM
        I_C = ca.vertcat(
            ca.horzcat(Ixx, Ixy, Ixz),
            ca.horzcat(Ixy, Iyy, Iyz),
            ca.horzcat(Ixz, Iyz, Izz)
        )
        S_c = ca.vertcat(
            ca.horzcat(0,     -cz,  cy),
            ca.horzcat(cz,     0,  -cx),
            ca.horzcat(-cy,   cx,   0 )
        )

        I = I_C + m * ca.mtimes(S_c.T, S_c)

        # Extract upper-triangular elements in Pinocchio order
        # [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        out_params = [
            m,
            mcx, mcy, mcz,
            I[0,0], I[0,1], I[1,1], I[0,2], I[1,2], I[2,2]
        ]
        params_list.extend([m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])  # For X_from_params function (physical params)
        X_syms.extend(out_params)  # For regressors (Pinocchio format)

    # Stack parameters into X vector (Pinocchio format)
    X_sym_expr = ca.vertcat(*X_syms)
    
    # Placeholder for calculate_base_parameters (it only needs the size)
    X_sym_placeholder = ca.SX.sym('X', 10 * n_links)
    
    E, beta, n_base_params = calculate_base_parameters(model, data, Phi_func, X_sym_placeholder, n_links, zero_q=False, zero_qd=False, zero_qdd=False, zero_g=False, n_samples=25)
    regressorBasFunction = casadi_base_regressor_functions(cmodel, cdata, X_sym_expr, params_list, E, beta, n_base_params, 'right_arm')

    # Prepare zero configuration for comparison
    nq, nv = model.nq, model.nv
    q_zero = np.random.randn(nq)
    v_zero = np.random.randn(nv)
    a_zero = np.random.randn(nv)
    g_zero = np.array([0., 0., -9.81])

    # Prepare full parameter vector in Pinocchio format
    X_full = pinocchio_standard_params(model)
    # Permute and split as in the printout
    X_permuted = E.T @ X_full
    X1 = X_permuted[:n_base_params]
    X2 = X_permuted[n_base_params:]
    X_base = X1 + beta @ X2

    # Compute Phi_base symbolically at zero config
    Phi_base_num = np.array(regressorBasFunction['right_arm_regressor_base'](q_zero, v_zero, a_zero, g_zero)).squeeze()
    tau_base_num = Phi_base_num @ X_base

    # Compute dynamics using RNEA as ground truth (use numeric model, not CasADi model)
    model.gravity.linear = g_zero
    tau_rnea = pin.rnea(model, data, q_zero, v_zero, a_zero)

    print("Torque via Phi_base * X_base at zero config:", tau_base_num)
    print("Torque via RNEA at zero config:", tau_rnea)
    print("Max abs diff:", np.max(np.abs(tau_base_num - tau_rnea)))


    # Generate excitation trajectory
    n_dof = 7
    n_fourier = 5
    g = np.array([0.0, 0.0, -9.81])
    t = 0
    A = ca.SX.sym('A', n_dof, n_fourier)
    B = ca.SX.sym('B', n_dof, n_fourier)
    omega = ca.SX.sym('omega', n_fourier)
    q, dq, ddq = fourier_trajectory(t, A, B, omega, n_dof, n_fourier)
    cond = condition_number(Phi_func, q, dq, ddq, g)
    print(f"Condition number: {cond}")

    


