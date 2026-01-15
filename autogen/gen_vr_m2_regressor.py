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


def casadi_base_regressor_functions(cmodel, cdata, Phi_sym, X_sym_expr, params_list, E, beta, n_base_params, arm_name: str):
    """
    Generate CasADi functions for base regressor Phi_base and base parameter vector X_base.
    
    Args:
        cmodel: CasADi model
        cdata: CasADi data
        Phi_sym: CasADi symbolic full regressor matrix (expression)
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


def generate_regressor_code(functions: dict, output_dir: str):
    """Generate C sources + headers for regressor functions.
    
    Args:
        functions: Dictionary of CasADi functions to generate code for
        output_dir: Directory where generated files should be saved
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(output_dir)
        opts = dict(with_header=True)
        print(f'Generating C code for regressor functions in {output_dir}...')
        for key, fn in functions.items():
            output_path = f"{key}.c"
            print(f'- {key} -> {os.path.join(output_dir, output_path)}')
            fn.generate(output_path, opts)
    finally:
        os.chdir(original_cwd)


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


def validate_regressor(model, data, q, v, a, g):
    """Validate regressor by comparing with inverse dynamics.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        q: Joint positionsurdf
        v: Joint velocities
        a: Joint accelerations
        g: Gravity vector
    
    Returns:
        error: Maximum absolute error between regressor-based and RNEA torques
    """
    # Set gravity
    model.gravity.linear = g
    
    # Compute regressor
    Phi = pin.computeJointTorqueRegressor(model, data, q, v, a)
    
    # Extract parameters from model in Pinocchio's standard format
    # (Pinocchio's regressor uses its own parameter format, not MATLAB format)
    X = pinocchio_standard_params(model)
    # other calculation of X
    # pi_i = model.inertias[i].toDyanamicsParameters
    # X = np.zeros(10 * (model.nbodies - 1))
    # for i in range(1, model.nbodies):
    #     inertia = model.inertias[i]
    #     print(inertia)
    #     idx = (i - 1) * 10
    #     X[idx:idx+10] = inertia.toDynamicParameters()
    
    # print("X_custom---")
    # print(X_custom)
    # print(X_custom.shape)
    # print("X---")
    # print(X)
    # print(X.shape)

    # Compute torque using regressor
    tau_regressor = Phi @ X
    
    # Compute torque using RNEA (inverse dynamics)
    tau_rnea = pin.rnea(model, data, q, v, a)
    
    # Compare
    error = np.max(np.abs(tau_regressor - tau_rnea))
    
    return error, tau_regressor, tau_rnea, Phi


def test_regressor_random_configs(model, data, n_tests=100, q_bounds=None, v_bounds=None, a_bounds=None, g=None, verbose=False):
    """Test regressor with N random configurations.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        n_tests: Number of random configurations to test
        q_bounds: Tuple (q_min, q_max) for joint position bounds. Default: (-pi, pi)
        v_bounds: Tuple (v_min, v_max) for joint velocity bounds. Default: (-5, 5) rad/s
        a_bounds: Tuple (a_min, a_max) for joint acceleration bounds. Default: (-10, 10) rad/s²
        g: Gravity vector. Default: [0, 0, -9.81]
        verbose: If True, print detailed results for each test
    
    Returns:
        Dictionary with test statistics:
            - errors: List of errors for each configuration
            - max_error: Maximum error across all tests
            - mean_error: Mean error across all tests
            - std_error: Standard deviation of errors
            - max_rel_error: Maximum relative error (error / max(|tau_rnea|))
            - failed_tests: Number of tests with error > 1e-3
    """
    nq = model.nq
    nv = model.nv
    
    # Default bounds
    if q_bounds is None:
        q_bounds = (-np.pi, np.pi)
    if v_bounds is None:
        v_bounds = (-5.0, 5.0)
    if a_bounds is None:
        a_bounds = (-10.0, 10.0)
    if g is None:
        g = np.array([0.0, 0.0, -9.81])
    
    errors = []
    rel_errors = []
    
    print(f'Testing {n_tests} random configurations...')
    print(f'  Joint position bounds: [{q_bounds[0]:.2f}, {q_bounds[1]:.2f}] rad')
    print(f'  Joint velocity bounds: [{v_bounds[0]:.2f}, {v_bounds[1]:.2f}] rad/s')
    print(f'  Joint acceleration bounds: [{a_bounds[0]:.2f}, {a_bounds[1]:.2f}] rad/s²')
    
    for i in range(n_tests):
        # Generate random configuration
        q = np.random.uniform(q_bounds[0], q_bounds[1], nq)
        v = np.random.uniform(v_bounds[0], v_bounds[1], nv)
        a = np.random.uniform(a_bounds[0], a_bounds[1], nv)
        
        try:
            error, tau_reg, tau_rnea, Phi = validate_regressor(model, data, q, v, a, g)
            errors.append(error)
            
            # Compute relative error
            max_tau = np.max(np.abs(tau_rnea))
            if max_tau > 1e-10:  # Avoid division by zero
                rel_error = error / max_tau
            else:
                rel_error = 0.0
            rel_errors.append(rel_error)
            
            if verbose and (i < 10 or i % (n_tests // 10) == 0):
                print(f'  Test {i+1:4d}/{n_tests}: error = {error:.2e} Nm, rel_error = {rel_error:.2e}')
        except Exception as e:
            print(f'  Test {i+1:4d}/{n_tests}: FAILED - {e}')
            errors.append(np.inf)
            rel_errors.append(np.inf)
    
    errors = np.array(errors)
    rel_errors = np.array(rel_errors)
    
    # Filter out infinite errors for statistics
    valid_errors = errors[np.isfinite(errors)]
    valid_rel_errors = rel_errors[np.isfinite(rel_errors)]
    
    stats = {
        'errors': errors.tolist(),
        'max_error': np.max(valid_errors) if len(valid_errors) > 0 else np.inf,
        'mean_error': np.mean(valid_errors) if len(valid_errors) > 0 else np.inf,
        'std_error': np.std(valid_errors) if len(valid_errors) > 0 else np.inf,
        'max_rel_error': np.max(valid_rel_errors) if len(valid_rel_errors) > 0 else np.inf,
        'failed_tests': np.sum(errors > 1e-3) if len(errors) > 0 else n_tests,
        'n_valid': len(valid_errors)
    }
    
    return stats


def generateLeftArmRegressor(urdf_path: str = None, output_dir: str = None):
    """Generate regressor code for the left arm.
    
    Args:
        urdf_path: Path to left arm URDF file. If None, uses default path.
        output_dir: Output directory for generated files. If None, uses default path.
    """
    if urdf_path is None:
        urdf_path = '../vr_m2_description/urdf/origin/vr_m2_left_arm.urdf'
    if output_dir is None:
        output_dir = '../sysid/c/autogen/VRM2LeftArm'
    
    start_time = time.time()
    
    print('='*60)
    print('Generating Left Arm Regressor Code')
    print('='*60)
    print(f'URDF: {urdf_path}')
    print(f'Output: {output_dir}')
    
    # Load model
    model, data, cmodel, cdata = load_model(urdf_path)
    print(f'nq: {model.nq}, nv: {model.nv}, nbodies: {model.nbodies}')
    
    # Validate regressor at home configuration
    print('\n=== Validating Regressor ===')
    # q_home = np.zeros(model.nq)
    # v_home = np.zeros(model.nv)
    # a_home = np.zeros(model.nv)
    # Choose random values for home configuration for validation
    q_home = np.random.uniform(-np.pi, np.pi, model.nq)
    v_home = np.random.uniform(-3*np.pi, 3*np.pi, model.nv)
    a_home = np.random.uniform(-6*np.pi, 6*np.pi, model.nv)
    g_vec = np.random.uniform(-1, 1, 3)
    g_vec = 9.81 * g_vec / np.linalg.norm(g_vec)
    
    error, tau_reg, tau_rnea, Phi_home = validate_regressor(model, data, q_home, v_home, a_home, g_vec)
    print(f'Max error at home config: {error:.2e} Nm')
    print(f'Regressor shape: {Phi_home.shape}')
    
    if error > 1e-3:
        print(f'Warning: Large validation error: {error:.2e} Nm')
    else:
        print('Regressor validation passed!')
    
    # Build & export regressor C code
    print('\n=== Generating CasADi Functions ===')
    reg_fns, Phi_sym, X_sym_expr, params_list = casadi_regressor_functions(cmodel, cdata, 'left_arm')
    generate_regressor_code(reg_fns, output_dir)
    
    # Compute base parameters
    print('\n=== Computing Base Parameters ===')
    Phi_func = reg_fns['left_arm_regressor']
    # Create a dummy X_sym for calculate_base_parameters (it's not used in the function)
    X_sym_dummy = ca.SX.sym('X_dummy', 10 * (model.nbodies - 1))
    E, beta, n_base_params = calculate_base_parameters(
        model, data, Phi_func, X_sym_dummy, model.nbodies - 1,
        zero_q=False, zero_qd=False, zero_qdd=False, zero_g=False, n_samples=25
    )
    
    # Print parameter relationship
    print_parameter_relationship(E, beta, n_base_params, model.nbodies - 1, 10 * (model.nbodies - 1))
    
    # Generate base parameter functions
    print('\n=== Generating Base Parameter Functions ===')
    base_fns = casadi_base_regressor_functions(
        cmodel, cdata, Phi_sym, X_sym_expr,
        params_list, E, beta, n_base_params, 'left_arm'
    )
    generate_regressor_code(base_fns, output_dir)
    
    # Verify base parameters
    print('\n=== Verifying Base Parameters ===')
    X_full = pinocchio_standard_params(model)
    # Build X_base from X_full
    E_full = E
    X_permuted = E_full.T @ X_full
    X1 = X_permuted[:n_base_params]
    X2 = X_permuted[n_base_params:]
    X_base = X1 + beta @ X2
    
    error_base, tau_full, tau_base = verify_base_parameters(
        Phi_func, X_full, base_fns['left_arm_regressor_base'],
        X_base, q_home, v_home, a_home, g_vec
    )
    print(f'Max error between Phi*X and Phi_base*X_base: {error_base:.2e} Nm')
    if error_base > 1e-6:
        print(f'Warning: Large verification error: {error_base:.2e} Nm')
    else:
        print('Base parameter verification passed!')
    
    elapsed_time = time.time() - start_time
    print('Left arm regressor code generation complete!')
    print(f'Execution time: {elapsed_time:.2f} seconds')
    print()


def generateRightArmRegressor(urdf_path: str = None, output_dir: str = None):
    """Generate regressor code for the right arm.
    
    Args:
        urdf_path: Path to right arm URDF file. If None, uses default path.
        output_dir: Output directory for generated files. If None, uses default path.
    """
    if urdf_path is None:
        urdf_path = '../vr_m2_description/urdf/origin/vr_m2_right_arm.urdf'
    if output_dir is None:
        output_dir = '../sysid/c/autogen/VRM2RightArm'
    
    start_time = time.time()
    
    print('='*60)
    print('Generating Right Arm Regressor Code')
    print('='*60)
    print(f'URDF: {urdf_path}')
    print(f'Output: {output_dir}')
    
    # Load model
    model, data, cmodel, cdata = load_model(urdf_path)
    print(f'nq: {model.nq}, nv: {model.nv}, nbodies: {model.nbodies}')
    
    # Validate regressor at home configuration
    print('\n=== Validating Regressor ===')
    q_home = np.random.uniform(-np.pi, np.pi, model.nq)
    v_home = np.random.uniform(-3*np.pi, 3*np.pi, model.nv)
    a_home = np.random.uniform(-6*np.pi, 6*np.pi, model.nv)
    g_vec = np.random.uniform(-1, 1, 3)
    g_vec = 9.81 * g_vec / np.linalg.norm(g_vec)
    
    error, tau_reg, tau_rnea, Phi_home = validate_regressor(model, data, q_home, v_home, a_home, g_vec)
    print(f'Max error at home config: {error:.2e} Nm')
    print(f'Regressor shape: {Phi_home.shape}')
    
    if error > 1e-3:
        print(f'Warning: Large validation error: {error:.2e} Nm')
    else:
        print('Regressor validation passed!')
    
    # Build & export regressor C code
    print('\n=== Generating CasADi Functions ===')
    reg_fns, Phi_sym, X_sym_expr, params_list = casadi_regressor_functions(cmodel, cdata, 'right_arm')
    generate_regressor_code(reg_fns, output_dir)
    
    # Compute base parameters
    print('\n=== Computing Base Parameters ===')
    Phi_func = reg_fns['right_arm_regressor']
    # Create a dummy X_sym for calculate_base_parameters (it's not used in the function)
    X_sym_dummy = ca.SX.sym('X_dummy', 10 * (model.nbodies - 1))
    E, beta, n_base_params = calculate_base_parameters(
        model, data, Phi_func, X_sym_dummy, model.nbodies - 1,
        zero_q=False, zero_qd=False, zero_qdd=False, zero_g=False, n_samples=25
    )
    
    # Print parameter relationship
    print_parameter_relationship(E, beta, n_base_params, model.nbodies - 1, 10 * (model.nbodies - 1))
    
    # Generate base parameter functions
    print('\n=== Generating Base Parameter Functions ===')
    base_fns = casadi_base_regressor_functions(
        cmodel, cdata, Phi_sym, X_sym_expr,
        params_list, E, beta, n_base_params, 'right_arm'
    )
    generate_regressor_code(base_fns, output_dir)
    
    # Verify base parameters
    print('\n=== Verifying Base Parameters ===')
    X_full = pinocchio_standard_params(model)
    # Build X_base from X_full
    E_full = E
    X_permuted = E_full.T @ X_full
    X1 = X_permuted[:n_base_params]
    X2 = X_permuted[n_base_params:]
    X_base = X1 + beta @ X2
    
    error_base, tau_full, tau_base = verify_base_parameters(
        Phi_func, X_full, base_fns['right_arm_regressor_base'],
        X_base, q_home, v_home, a_home, g_vec
    )
    print(f'Max error between Phi*X and Phi_base*X_base: {error_base:.2e} Nm')
    if error_base > 1e-6:
        print(f'Warning: Large verification error: {error_base:.2e} Nm')
    else:
        print('Base parameter verification passed!')
    
    elapsed_time = time.time() - start_time
    print('Right arm regressor code generation complete!')
    print(f'Execution time: {elapsed_time:.2f} second')
    print()


def generateBothArmsRegressor():
    """Generate regressor code for both left and right arms."""
    generateLeftArmRegressor()
    generateRightArmRegressor()


if __name__ == '__main__':
    import sys
    import argparse
    
    program_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Generate or test dynamic regressor')
    parser.add_argument('--test', type=int, nargs='?', const=100, metavar='N',
                       help='Run validation test with N random configurations (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed results for each test configuration')
    args = parser.parse_args()
    
    # Check if user wants to run test
    if args.test is not None:
        test_start_time = time.time()
        n_tests = args.test
        print('\n' + '='*60)
        print(f'Running Regressor Validation Test ({n_tests} random configurations)')
        print('='*60 + '\n')
        
        # Test left arm
        left_urdf = '../vr_m2_description/urdf/origin/vr_m2_left_arm.urdf'

        if os.path.exists(left_urdf):
            print('='*60)
            print('Testing Left Arm')
            print('='*60)
            model, data, _, _ = load_model(left_urdf)
            g = np.array([0.0, 0.0, -9.81])
            
            stats = test_regressor_random_configs(model, data, n_tests=n_tests, g=g, verbose=args.verbose)
            
            print(f'\nResults ({stats["n_valid"]}/{n_tests} valid tests):')
            print(f'  Max absolute error:     {stats["max_error"]:.2e} Nm')
            print(f'  Mean absolute error:    {stats["mean_error"]:.2e} Nm')
            print(f'  Std dev of errors:      {stats["std_error"]:.2e} Nm')
            print(f'  Max relative error:     {stats["max_rel_error"]:.2e}')
            print(f'  Tests with error > 1e-3: {stats["failed_tests"]}/{n_tests}')
            
            if stats["max_error"] > 1e-3:
                print(f'  Warning: Large validation error detected!')
            else:
                print(f'  All tests passed (max error < 1e-3 Nm)')
        else:
            print(f'Left arm URDF not found: {left_urdf}')
        
        print('\n')
        
        # Test right arm
        right_urdf = '../vr_m2_description/urdf/origin/vr_m2_right_arm.urdf'
        if not os.path.exists(right_urdf):
            right_urdf = '../vr_m2_description/urdf/origin/vr_m2_right_arm.urdf'
        
        if os.path.exists(right_urdf):
            print('='*60)
            print('Testing Right Arm')
            print('='*60)
            model, data, _, _ = load_model(right_urdf)
            g = np.array([0.0, 0.0, -9.81])
            
            stats = test_regressor_random_configs(model, data, n_tests=n_tests, g=g, verbose=args.verbose)
            
            print(f'\nResults ({stats["n_valid"]}/{n_tests} valid tests):')
            print(f'  Max absolute error:     {stats["max_error"]:.2e} Nm')
            print(f'  Mean absolute error:    {stats["mean_error"]:.2e} Nm')
            print(f'  Std dev of errors:      {stats["std_error"]:.2e} Nm')
            print(f'  Max relative error:     {stats["max_rel_error"]:.2e}')
            print(f'  Tests with error > 1e-3: {stats["failed_tests"]}/{n_tests}')
            
            if stats["max_error"] > 1e-3:
                print(f'   Warning: Large validation error detected!')
            else:
                print(f'  All tests passed (max error < 1e-3 Nm)')
        else:
            print(f'Right arm URDF not found')
        
        test_time = time.time() - test_start_time
        print(f'\nTest execution time: {test_time:.2f} seconds')
    else:
        # Generate code for both arms
        generateBothArmsRegressor()
    
    total_time = time.time() - program_start_time
    print('\n' + '='*60)
    print(f'Total execution time: {total_time:.2f} seconds')
    print('='*60)

