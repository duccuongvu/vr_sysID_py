#!/usr/bin/env python3
"""
Benchmark script comparing C regressor functions vs Pinocchio.

This script benchmarks the performance of:
1. C-generated regressor functions (via Python wrapper)
2. Pinocchio's regressor computation
3. C-generated tau computation vs Pinocchio's RNEA

Usage:
    python benchmark_regressor.py [--n <N>] [--urdf <path>] [--verbose]
"""

import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

try:
    from right_arm_regressor_wrapper import RightArmRegressor
    C_REGressor_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RightArmRegressor: {e}")
    print("C regressor functions will not be benchmarked.")
    C_REGressor_AVAILABLE = False

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    print("Error: Pinocchio not available. Please install pinocchio.")
    sys.exit(1)

# Add autogen directory to path for pinocchio_standard_params
autogen_dir = script_dir.parent.parent.parent / "autogen"
if autogen_dir.exists():
    sys.path.insert(0, str(autogen_dir))
    try:
        from gen_vr_m2_regressor import pinocchio_standard_params
    except ImportError:
        # Define locally if import fails
        def pinocchio_standard_params(model):
            """Extract parameters in Pinocchio's standard format."""
            def skew(c):
                return np.array([
                    [0, -c[2], c[1]],
                    [c[2], 0, -c[0]],
                    [-c[1], c[0], 0]
                ])
            
            n_links = model.nbodies - 1
            X_pin = np.zeros(10 * n_links)
            
            for i in range(1, model.nbodies):
                inertia = model.inertias[i]
                m = inertia.mass
                c = inertia.lever
                I_C = inertia.inertia
                
                S_c = skew(c)
                I = I_C + m * S_c.T @ S_c
                
                mc_x = m * c[0]
                mc_y = m * c[1]
                mc_z = m * c[2]
                
                idx = (i - 1) * 10
                X_pin[idx:idx+10] = [m, mc_x, mc_y, mc_z, I[0,0], I[0,1], I[1,1], I[0,2], I[1,2], I[2,2]]
            
            return X_pin
else:
    # Fallback definition
    def pinocchio_standard_params(model):
        """Extract parameters in Pinocchio's standard format."""
        def skew(c):
            return np.array([
                [0, -c[2], c[1]],
                [c[2], 0, -c[0]],
                [-c[1], c[0], 0]
            ])
        
        n_links = model.nbodies - 1
        X_pin = np.zeros(10 * n_links)
        
        for i in range(1, model.nbodies):
            inertia = model.inertias[i]
            m = inertia.mass
            c = inertia.lever
            I_C = inertia.inertia
            
            S_c = skew(c)
            I = I_C + m * S_c.T @ S_c
            
            mc_x = m * c[0]
            mc_y = m * c[1]
            mc_z = m * c[2]
            
            idx = (i - 1) * 10
            X_pin[idx:idx+10] = [m, mc_x, mc_y, mc_z, I[0,0], I[0,1], I[1,1], I[0,2], I[1,2], I[2,2]]
        
        return X_pin


def load_model(urdf_path):
    """Load URDF model."""
    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def benchmark_regressor_computation(model, data, c_regressor, n_iterations=1000):
    """Benchmark regressor matrix computation."""
    nq = model.nq
    nv = model.nv
    
    # Generate random configurations
    np.random.seed(42)
    qs = np.random.uniform(-np.pi, np.pi, (n_iterations, nq))
    vs = np.random.uniform(-5.0, 5.0, (n_iterations, nv))
    as_ = np.random.uniform(-10.0, 10.0, (n_iterations, nv))
    g = np.array([0.0, 0.0, -9.81])
    
    results = {
        'c_times': [],
        'pinocchio_times': [],
        'errors': []
    }
    
    print(f"\n{'='*70}")
    print(f"Benchmarking Regressor Computation ({n_iterations} iterations)")
    print(f"{'='*70}")
    
    # Benchmark C implementation
    if C_REGressor_AVAILABLE:
        print("\nBenchmarking C implementation...")
        c_times = []
        for i in range(n_iterations):
            start = time.perf_counter()
            Phi_c = c_regressor.compute_regressor(qs[i], vs[i], as_[i], g)
            end = time.perf_counter()
            c_times.append((end - start) * 1e6)  # Convert to microseconds
        
        results['c_times'] = c_times
        print(f"  Mean time: {np.mean(c_times):.2f} μs")
        print(f"  Std dev:  {np.std(c_times):.2f} μs")
        print(f"  Min time: {np.min(c_times):.2f} μs")
        print(f"  Max time: {np.max(c_times):.2f} μs")
    
    # Benchmark Pinocchio implementation
    print("\nBenchmarking Pinocchio implementation...")
    model.gravity.linear = g
    pinocchio_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        Phi_pin = pin.computeJointTorqueRegressor(model, data, qs[i], vs[i], as_[i])
        end = time.perf_counter()
        pinocchio_times.append((end - start) * 1e6)  # Convert to microseconds
    
    results['pinocchio_times'] = pinocchio_times
    print(f"  Mean time: {np.mean(pinocchio_times):.2f} μs")
    print(f"  Std dev:  {np.std(pinocchio_times):.2f} μs")
    print(f"  Min time: {np.min(pinocchio_times):.2f} μs")
    print(f"  Max time: {np.max(pinocchio_times):.2f} μs")
    
    # Compare results if both available
    if C_REGressor_AVAILABLE:
        print("\n" + "="*70)
        print("Accuracy Comparison (Regressor Matrix)")
        print("="*70)
        
        # Compare all iterations for accuracy
        print(f"\nComparing regressor matrices for all {n_iterations} iterations...")
        errors = []
        rel_errors = []
        max_abs_errors = []
        mean_abs_errors = []
        
        for i in range(n_iterations):
            Phi_c = c_regressor.compute_regressor(qs[i], vs[i], as_[i], g)
            Phi_pin = pin.computeJointTorqueRegressor(model, data, qs[i], vs[i], as_[i])
            
            # Absolute error
            abs_error = np.abs(Phi_c - Phi_pin)
            max_abs_error = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            
            errors.append(max_abs_error)
            max_abs_errors.append(max_abs_error)
            mean_abs_errors.append(mean_abs_error)
            
            # Relative error (avoid division by zero)
            max_Phi_pin = np.max(np.abs(Phi_pin))
            if max_Phi_pin > 1e-10:
                rel_error = max_abs_error / max_Phi_pin
                rel_errors.append(rel_error)
        
        results['errors'] = errors
        results['rel_errors'] = rel_errors
        results['max_abs_errors'] = max_abs_errors
        results['mean_abs_errors'] = mean_abs_errors
        
        print(f"\nAbsolute Error Statistics:")
        print(f"  Max error:     {np.max(errors):.2e}")
        print(f"  Mean error:    {np.mean(errors):.2e}")
        print(f"  Min error:     {np.min(errors):.2e}")
        print(f"  Std dev:       {np.std(errors):.2e}")
        print(f"  Median error:  {np.median(errors):.2e}")
        
        print(f"\nMean Absolute Error (per matrix):")
        print(f"  Max:           {np.max(mean_abs_errors):.2e}")
        print(f"  Mean:          {np.mean(mean_abs_errors):.2e}")
        print(f"  Min:           {np.min(mean_abs_errors):.2e}")
        
        if rel_errors:
            print(f"\nRelative Error Statistics:")
            print(f"  Max rel error: {np.max(rel_errors):.2e}")
            print(f"  Mean rel error: {np.mean(rel_errors):.2e}")
            print(f"  Min rel error: {np.min(rel_errors):.2e}")
        
        # Error distribution
        error_thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]
        print(f"\nError Distribution:")
        for threshold in error_thresholds:
            count = np.sum(np.array(errors) < threshold)
            pct = 100.0 * count / len(errors)
            print(f"  Error < {threshold:.0e}: {count:5d} ({pct:5.1f}%)")
        
        speedup = np.mean(pinocchio_times) / np.mean(c_times)
        print(f"\nPerformance:")
        print(f"  Speedup (Pinocchio/C): {speedup:.2f}x")
    
    return results


def benchmark_tau_computation(model, data, c_regressor, n_iterations=1000):
    """Benchmark torque computation."""
    nq = model.nq
    nv = model.nv
    
    # Generate random configurations
    np.random.seed(42)
    qs = np.random.uniform(-np.pi, np.pi, (n_iterations, nq))
    vs = np.random.uniform(-5.0, 5.0, (n_iterations, nv))
    as_ = np.random.uniform(-10.0, 10.0, (n_iterations, nv))
    g = np.array([0.0, 0.0, -9.81])
    
    # Get parameter vector X
    X = pinocchio_standard_params(model)
    
    results = {
        'c_times': [],
        'pinocchio_times': [],
        'errors': []
    }
    
    print(f"\n{'='*70}")
    print(f"Benchmarking Torque Computation ({n_iterations} iterations)")
    print(f"{'='*70}")
    
    # Benchmark C implementation
    if C_REGressor_AVAILABLE:
        print("\nBenchmarking C implementation (tau = Phi * X)...")
        c_times = []
        for i in range(n_iterations):
            start = time.perf_counter()
            tau_c = c_regressor.compute_tau_from_regressor(X, qs[i], vs[i], as_[i], g)
            end = time.perf_counter()
            c_times.append((end - start) * 1e6)  # Convert to microseconds
        
        results['c_times'] = c_times
        print(f"  Mean time: {np.mean(c_times):.2f} μs")
        print(f"  Std dev:  {np.std(c_times):.2f} μs")
        print(f"  Min time: {np.min(c_times):.2f} μs")
        print(f"  Max time: {np.max(c_times):.2f} μs")
    
    # Benchmark Pinocchio RNEA
    print("\nBenchmarking Pinocchio RNEA...")
    model.gravity.linear = g
    pinocchio_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        tau_pin = pin.rnea(model, data, qs[i], vs[i], as_[i])
        end = time.perf_counter()
        pinocchio_times.append((end - start) * 1e6)  # Convert to microseconds
    
    results['pinocchio_times'] = pinocchio_times
    print(f"  Mean time: {np.mean(pinocchio_times):.2f} μs")
    print(f"  Std dev:  {np.std(pinocchio_times):.2f} μs")
    print(f"  Min time: {np.min(pinocchio_times):.2f} μs")
    print(f"  Max time: {np.max(pinocchio_times):.2f} μs")
    
    # Compare results if both available
    if C_REGressor_AVAILABLE:
        print("\n" + "="*70)
        print("Accuracy Comparison (Torque)")
        print("="*70)
        
        # Compare all iterations for accuracy
        print(f"\nComparing torques for all {n_iterations} iterations...")
        errors = []
        rel_errors = []
        per_joint_errors = np.zeros((n_iterations, 7))
        per_joint_max_errors = np.zeros(7)
        per_joint_mean_errors = np.zeros(7)
        
        for i in range(n_iterations):
            tau_c = c_regressor.compute_tau_from_regressor(X, qs[i], vs[i], as_[i], g)
            tau_pin = pin.rnea(model, data, qs[i], vs[i], as_[i])
            
            # Absolute error per joint
            abs_error = np.abs(tau_c - tau_pin)
            per_joint_errors[i, :] = abs_error
            
            # Max error across all joints
            max_error = np.max(abs_error)
            errors.append(max_error)
            
            # Relative error (avoid division by zero)
            max_tau_pin = np.max(np.abs(tau_pin))
            if max_tau_pin > 1e-10:
                rel_error = max_error / max_tau_pin
                rel_errors.append(rel_error)
        
        # Compute per-joint statistics
        for j in range(7):
            per_joint_max_errors[j] = np.max(per_joint_errors[:, j])
            per_joint_mean_errors[j] = np.mean(per_joint_errors[:, j])
        
        results['errors'] = errors
        results['rel_errors'] = rel_errors
        results['per_joint_errors'] = per_joint_errors
        results['per_joint_max_errors'] = per_joint_max_errors
        results['per_joint_mean_errors'] = per_joint_mean_errors
        
        print(f"\nAbsolute Error Statistics (max across joints):")
        print(f"  Max error:     {np.max(errors):.2e} Nm")
        print(f"  Mean error:    {np.mean(errors):.2e} Nm")
        print(f"  Min error:     {np.min(errors):.2e} Nm")
        print(f"  Std dev:       {np.std(errors):.2e} Nm")
        print(f"  Median error:  {np.median(errors):.2e} Nm")
        
        print(f"\nPer-Joint Error Statistics:")
        print(f"{'Joint':<8} {'Max Error (Nm)':<18} {'Mean Error (Nm)':<18}")
        print("-" * 50)
        for j in range(7):
            print(f"{j+1:<8} {per_joint_max_errors[j]:<18.2e} {per_joint_mean_errors[j]:<18.2e}")
        
        if rel_errors:
            print(f"\nRelative Error Statistics:")
            print(f"  Max rel error: {np.max(rel_errors):.2e}")
            print(f"  Mean rel error: {np.mean(rel_errors):.2e}")
            print(f"  Min rel error: {np.min(rel_errors):.2e}")
        
        # Error distribution
        error_thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
        print(f"\nError Distribution (max error across joints):")
        for threshold in error_thresholds:
            count = np.sum(np.array(errors) < threshold)
            pct = 100.0 * count / len(errors)
            print(f"  Error < {threshold:.0e} Nm: {count:5d} ({pct:5.1f}%)")
        
        speedup = np.mean(pinocchio_times) / np.mean(c_times)
        print(f"\nPerformance:")
        print(f"  Speedup (Pinocchio/C): {speedup:.2f}x")
    
    return results


def benchmark_params_to_X(c_regressor, n_iterations=1000):
    """Benchmark parameter vector construction."""
    if not C_REGressor_AVAILABLE:
        print("C regressor not available, skipping params_to_X benchmark")
        return None
    
    print(f"\n{'='*70}")
    print(f"Benchmarking params_to_X ({n_iterations} iterations)")
    print(f"{'='*70}")
    
    # Generate random parameters
    np.random.seed(42)
    params_list = []
    for _ in range(n_iterations):
        params = np.random.uniform(0.1, 10.0, 70).tolist()
        params_list.append(params)
    
    times = []
    for params in params_list:
        start = time.perf_counter()
        X = c_regressor.params_to_X(*params)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    
    print(f"\n  Mean time: {np.mean(times):.2f} μs")
    print(f"  Std dev:  {np.std(times):.2f} μs")
    print(f"  Min time: {np.min(times):.2f} μs")
    print(f"  Max time: {np.max(times):.2f} μs")
    
    return {'times': times}


def benchmark_params_to_X_base(c_regressor, n_iterations=1000):
    """Benchmark base parameter vector construction."""
    if not C_REGressor_AVAILABLE:
        print("C regressor not available, skipping params_to_X_base benchmark")
        return None
    
    print(f"\n{'='*70}")
    print(f"Benchmarking params_to_X_base ({n_iterations} iterations)")
    print(f"{'='*70}")
    
    # Generate random parameters
    np.random.seed(42)
    params_list = []
    for _ in range(n_iterations):
        params = np.random.uniform(0.1, 10.0, 70).tolist()
        params_list.append(params)
    
    times = []
    for params in params_list:
        start = time.perf_counter()
        X_base = c_regressor.params_to_X_base(*params)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds
    
    print(f"\n  Mean time: {np.mean(times):.2f} μs")
    print(f"  Std dev:  {np.std(times):.2f} μs")
    print(f"  Min time: {np.min(times):.2f} μs")
    print(f"  Max time: {np.max(times):.2f} μs")
    
    return {'times': times}



def main():
    parser = argparse.ArgumentParser(description="Benchmark C regressor functions vs Pinocchio")
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of iterations (default: 1000)"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to URDF file (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    args = parser.parse_args()
    
    # Find URDF file
    if args.urdf is None:
        # Try to find URDF file
        possible_paths = [
            script_dir.parent.parent.parent / "models" / "urdf" / "origin" / "vr_m2_right_arm_updated.urdf",
            # script_dir.parent.parent.parent / "sysid" / "matlab" / "vr_m2_right_arm.urdf",
        ]
        
        urdf_path = None
        for path in possible_paths:
            if path.exists():
                urdf_path = str(path)
                break
        
        if urdf_path is None:
            print("Error: Could not find URDF file. Please specify with --urdf")
            sys.exit(1)
    else:
        urdf_path = args.urdf
    
    print(f"Loading model from: {urdf_path}")
    model, data = load_model(urdf_path)
    print(f"Model loaded: nq={model.nq}, nv={model.nv}, nbodies={model.nbodies}")
    
    # Initialize C regressor if available
    c_regressor = None
    if C_REGressor_AVAILABLE:
        try:
            c_regressor = RightArmRegressor()
            print("C regressor wrapper loaded")
        except Exception as e:
            print(f"Failed to load C regressor: {e}")
            print("  Continuing with Pinocchio-only benchmarks...")
    
    # Run benchmarks
    print(f"\n{'='*70}")
    print("Starting Benchmarks")
    print(f"{'='*70}")
    
    # Benchmark regressor computation
    regressor_results = benchmark_regressor_computation(model, data, c_regressor, args.n)
    
    # Benchmark torque computation
    tau_results = benchmark_tau_computation(model, data, c_regressor, args.n)
    
    # Benchmark params_to_X
    params_results = benchmark_params_to_X(c_regressor, args.n)
    # Benchmark params_to_X_base
    params_base_results = benchmark_params_to_X_base(c_regressor, args.n)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    if C_REGressor_AVAILABLE and regressor_results.get('c_times'):
        print(f"\nRegressor Matrix Computation:")
        print(f"  Performance:")
        print(f"    C:        {np.mean(regressor_results['c_times']):.2f} μs")
        print(f"    Pinocchio: {np.mean(regressor_results['pinocchio_times']):.2f} μs")
        print(f"    Speedup:   {np.mean(regressor_results['pinocchio_times']) / np.mean(regressor_results['c_times']):.2f}x")
        if regressor_results.get('errors'):
            print(f"  Accuracy:")
            print(f"    Max error:  {np.max(regressor_results['errors']):.2e}")
            print(f"    Mean error: {np.mean(regressor_results['errors']):.2e}")
            if regressor_results.get('rel_errors'):
                print(f"    Mean rel error: {np.mean(regressor_results['rel_errors']):.2e}")
    
    if C_REGressor_AVAILABLE and tau_results.get('c_times'):
        print(f"\nTorque Computation:")
        print(f"  Performance:")
        print(f"    C:        {np.mean(tau_results['c_times']):.2f} μs")
        print(f"    Pinocchio: {np.mean(tau_results['pinocchio_times']):.2f} μs")
        print(f"    Speedup:   {np.mean(tau_results['pinocchio_times']) / np.mean(tau_results['c_times']):.2f}x")
        if tau_results.get('errors'):
            print(f"  Accuracy:")
            print(f"    Max error:  {np.max(tau_results['errors']):.2e} Nm")
            print(f"    Mean error: {np.mean(tau_results['errors']):.2e} Nm")
            if tau_results.get('rel_errors'):
                print(f"    Mean rel error: {np.mean(tau_results['rel_errors']):.2e}")
            if tau_results.get('per_joint_max_errors') is not None:
                print(f"    Max per-joint errors: {tau_results['per_joint_max_errors']}")
    
    if params_results:
        print(f"\nparams_to_X:")
        print(f"  Mean time: {np.mean(params_results['times']):.2f} μs")
    
    if params_base_results:
        print(f"\nparams_to_X_base:")
        print(f"  Mean time: {np.mean(params_base_results['times']):.2f} μs")

    # Overall accuracy assessment
    if C_REGressor_AVAILABLE:
        print(f"\n{'='*70}")
        print("Overall Accuracy Assessment")
        print(f"{'='*70}")
        
        regressor_passed = False
        tau_passed = False
        
        if regressor_results.get('errors'):
            max_reg_error = np.max(regressor_results['errors'])
            if max_reg_error < 1e-6:
                regressor_passed = True
                print(f"Regressor Matrix: PASSED (max error < 1e-6)")
            elif max_reg_error < 1e-3:
                print(f"Regressor Matrix: WARNING (max error < 1e-3 but >= 1e-6)")
            else:
                print(f"Regressor Matrix: FAILED (max error >= 1e-3)")
            print(f"  Max error: {max_reg_error:.2e}")
        
        if tau_results.get('errors'):
            max_tau_error = np.max(tau_results['errors'])
            if max_tau_error < 1e-6:
                tau_passed = True
                print(f"Torque Computation: PASSED (max error < 1e-6 Nm)")
            elif max_tau_error < 1e-3:
                print(f"Torque Computation: WARNING (max error < 1e-3 Nm but >= 1e-6 Nm)")
            else:
                print(f"Torque Computation: FAILED (max error >= 1e-3 Nm)")
            print(f"  Max error: {max_tau_error:.2e} Nm")
        
        if regressor_passed and tau_passed:
            print(f"\nOverall: All accuracy tests PASSED")
        elif regressor_passed or tau_passed:
            print(f"\nOverall: Some accuracy tests have warnings")
        else:
            print(f"\nOverall: Accuracy tests FAILED")


if __name__ == "__main__":
    main()

