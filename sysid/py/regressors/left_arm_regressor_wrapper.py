#!/usr/bin/env python3
"""
Python wrapper for VRM2 Left Arm regressor C functions.

This module provides a Python interface to the CasADi-generated C functions
for computing the dynamic regressor matrix and parameter vector.

Functions:
    - compute_regressor(q, v, a, g): Compute regressor matrix Phi (7x70)
    - params_to_X(...): Build parameter vector X from individual parameters
    - compute_tau_from_regressor(X, q, v, a, g): Compute tau = Phi * X

Usage:
    from left_arm_regressor_wrapper import LeftArmRegressor
    
    regressor = LeftArmRegressor()
    Phi = regressor.compute_regressor(q, v, a, g)
    X = regressor.params_to_X(m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz, ...)
    tau = regressor.compute_tau_from_regressor(X, q, v, a, g)
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path

# Find the shared library
def find_library():
    """Find the compiled shared library."""
    script_dir = Path(__file__).parent.absolute()
    
    # Possible locations for the shared library
    possible_paths = [
        script_dir / "libleft_arm_regressor.so",
        script_dir / "build" / "libleft_arm_regressor.so",
        script_dir.parent.parent / "c" / "build" / "libleft_arm_regressor.so",
        script_dir.parent.parent / "py" / "build" / "libleft_arm_regressor.so",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Try to find in system paths
    try:
        import ctypes.util
        lib_path = ctypes.util.find_library("left_arm_regressor")
        if lib_path:
            return lib_path
    except:
        pass
    
    raise FileNotFoundError(
        f"Could not find libleft_arm_regressor.so. "
        f"Searched in: {[str(p) for p in possible_paths]}. "
        f"Please compile the C files first using build_regressor_lib.py"
    )


class LeftArmRegressor:
    """Python wrapper for VRM2 Left Arm regressor C functions."""
    
    def __init__(self, lib_path=None):
        """Initialize the wrapper and load the shared library.
        
        Args:
            lib_path: Path to the shared library. If None, attempts to find it automatically.
        """
        if lib_path is None:
            lib_path = find_library()
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._setup_function_signatures()
        
        # Initialize memory management
        self._init_memory()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for all C functions."""
        # Define types
        casadi_real = ctypes.c_double
        casadi_int = ctypes.c_longlong
        casadi_real_p = ctypes.POINTER(casadi_real)
        casadi_int_p = ctypes.POINTER(casadi_int)
        
        # left_arm_regressor: (q[7], v[7], a[7], g[3]) -> Phi[7x70]
        self.lib.left_arm_regressor.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_regressor.restype = ctypes.c_int
        
        # left_arm_regressor_work
        self.lib.left_arm_regressor_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_regressor_work.restype = ctypes.c_int
        
        # left_arm_params_to_X: (70 params) -> X[70]
        self.lib.left_arm_params_to_X.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_params_to_X.restype = ctypes.c_int
        
        # left_arm_params_to_X_work
        self.lib.left_arm_params_to_X_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_params_to_X_work.restype = ctypes.c_int
        
        # left_arm_tau_from_regressor: (X[70], q[7], v[7], a[7], g[3]) -> tau[7]
        self.lib.left_arm_tau_from_regressor.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_tau_from_regressor.restype = ctypes.c_int
        
        # left_arm_tau_from_regressor_work
        self.lib.left_arm_tau_from_regressor_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_tau_from_regressor_work.restype = ctypes.c_int

        # left_arm_regressor_base: (q[7], v[7], a[7], g[3]) -> Phi_base[7x45]
        self.lib.left_arm_regressor_base.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_regressor_base.restype = ctypes.c_int
        
        # left_arm_regressor_base_work
        self.lib.left_arm_regressor_base_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_regressor_base_work.restype = ctypes.c_int
        
        # left_arm_params_to_X_base: (70 params) -> X_base[45]
        self.lib.left_arm_params_to_X_base.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_params_to_X_base.restype = ctypes.c_int
        
        # left_arm_params_to_X_base_work
        self.lib.left_arm_params_to_X_base_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_params_to_X_base_work.restype = ctypes.c_int
        
        # left_arm_tau_from_regressor_base: (X_base[45], q[7], v[7], a[7], g[3]) -> tau[7]
        self.lib.left_arm_tau_from_regressor_base.argtypes = [
            ctypes.POINTER(casadi_real_p),  # arg
            ctypes.POINTER(casadi_real_p),  # res
            casadi_int_p,                   # iw
            casadi_real_p,                  # w
            ctypes.c_int                    # mem
        ]
        self.lib.left_arm_tau_from_regressor_base.restype = ctypes.c_int
        
        # left_arm_tau_from_regressor_base_work
        self.lib.left_arm_tau_from_regressor_base_work.argtypes = [
            casadi_int_p,  # sz_arg
            casadi_int_p,  # sz_res
            casadi_int_p,  # sz_iw
            casadi_int_p   # sz_w
        ]
        self.lib.left_arm_tau_from_regressor_base_work.restype = ctypes.c_int
    
    def _init_memory(self):
        """Initialize memory for CasADi functions."""
        # Get work sizes
        sz_arg = ctypes.c_longlong()
        sz_res = ctypes.c_longlong()
        sz_iw = ctypes.c_longlong()
        sz_w = ctypes.c_longlong()
        
        # Get work sizes for each function
        self.lib.left_arm_regressor_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.regressor_sz_iw = sz_iw.value
        self.regressor_sz_w = sz_w.value
        
        self.lib.left_arm_params_to_X_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.params_sz_iw = sz_iw.value
        self.params_sz_w = sz_w.value
        
        self.lib.left_arm_tau_from_regressor_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.tau_sz_iw = sz_iw.value
        self.tau_sz_w = sz_w.value

        self.lib.left_arm_regressor_base_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.regressor_base_sz_iw = sz_iw.value
        self.regressor_base_sz_w = sz_w.value
        
        self.lib.left_arm_params_to_X_base_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.params_base_sz_iw = sz_iw.value
        self.params_base_sz_w = sz_w.value
        
        self.lib.left_arm_tau_from_regressor_base_work(
            ctypes.byref(sz_arg), ctypes.byref(sz_res),
            ctypes.byref(sz_iw), ctypes.byref(sz_w)
        )
        self.tau_base_sz_iw = sz_iw.value
        self.tau_base_sz_w = sz_w.value
    
    def compute_regressor(self, q, v, a, g):
        """Compute regressor matrix Phi(q, v, a, g).

        Args:
            q: Joint positions (7,) array
            v: Joint velocities (7,) array
            a: Joint accelerations (7,) array
            g: Gravity vector (3,) array

        Returns:
            Phi: Regressor matrix (7, 70) array (row-major, but CasADi returns column-major)
        """
        # Convert inputs to numpy arrays and ensure contiguous
        q = np.ascontiguousarray(q, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        a = np.ascontiguousarray(a, dtype=np.float64)
        g = np.ascontiguousarray(g, dtype=np.float64)

        if q.shape != (7,) or v.shape != (7,) or a.shape != (7,) or g.shape != (3,):
            raise ValueError(f"Invalid input shapes: q={q.shape}, v={v.shape}, a={a.shape}, g={g.shape}")

        # Allocate output (column-major, according to CasADi's convention)
        Phi_raw = np.zeros((7, 70), dtype=np.float64, order='F')  # Fortran order for column-major

        # Prepare arguments
        arg = (ctypes.POINTER(ctypes.c_double) * 4)()
        arg[0] = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[1] = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[2] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[3] = g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = Phi_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Allocate work arrays
        iw = None
        w = None
        if self.regressor_sz_iw > 0:
            iw = (ctypes.c_longlong * self.regressor_sz_iw)()
        if self.regressor_sz_w > 0:
            w = (ctypes.c_double * self.regressor_sz_w)()

        # Call function
        ret = self.lib.left_arm_regressor(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )

        if ret != 0:
            raise RuntimeError(f"left_arm_regressor returned error code: {ret}")

        # CasADi outputs Phi in column-major order (Fortran). Return Phi as (7, 70) C-order.
        Phi = np.array(Phi_raw, order='C')  # Convert to C-order for user (row-major)
        return Phi
    
    def params_to_X(self, *params):
        """Build parameter vector X from individual parameters.
        
        Args:
            *params: 70 individual parameters in order:
                For each of 7 links: [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
                Total: 7 * 10 = 70 parameters
        
        Returns:
            X: Parameter vector (70,) array in Pinocchio format
        """
        if len(params) != 70:
            raise ValueError(f"Expected 70 parameters, got {len(params)}")
        
        # Convert to numpy array (contiguous)
        params_arr = np.ascontiguousarray(params, dtype=np.float64)
        
        # Allocate output
        X = np.zeros(70, dtype=np.float64)
        
        # Prepare arguments (70 individual pointers)
        # Create array of pointers, each pointing to an element of params_arr
        arg = (ctypes.POINTER(ctypes.c_double) * 70)()
        for i in range(70):
            # Get pointer to the i-th element of the array
            arg[i] = ctypes.cast(
                params_arr.ctypes.data + i * ctypes.sizeof(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            )
        
        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Allocate work arrays
        iw = None
        w = None
        if self.params_sz_iw > 0:
            iw = (ctypes.c_longlong * self.params_sz_iw)()
        if self.params_sz_w > 0:
            w = (ctypes.c_double * self.params_sz_w)()
        
        # Call function
        ret = self.lib.left_arm_params_to_X(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )
        
        if ret != 0:
            raise RuntimeError(f"left_arm_params_to_X returned error code: {ret}")
        
        return X
    
    def compute_tau_from_regressor(self, X, q, v, a, g):
        """Compute torque from regressor: tau = Phi(q, v, a, g) * X.
        
        Args:
            X: Parameter vector (70,) array
            q: Joint positions (7,) array
            v: Joint velocities (7,) array
            a: Joint accelerations (7,) array
            g: Gravity vector (3,) array
        
        Returns:
            tau: Joint torques (7,) array
        """
        # Convert inputs to numpy arrays and ensure contiguous
        X = np.ascontiguousarray(X, dtype=np.float64)
        q = np.ascontiguousarray(q, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        a = np.ascontiguousarray(a, dtype=np.float64)
        g = np.ascontiguousarray(g, dtype=np.float64)
        
        if X.shape != (70,):
            raise ValueError(f"Invalid X shape: {X.shape}, expected (70,)")
        if q.shape != (7,) or v.shape != (7,) or a.shape != (7,) or g.shape != (3,):
            raise ValueError(f"Invalid input shapes: q={q.shape}, v={v.shape}, a={a.shape}, g={g.shape}")
        
        # Allocate output
        tau = np.zeros(7, dtype=np.float64)
        
        # Prepare arguments
        arg = (ctypes.POINTER(ctypes.c_double) * 5)()
        arg[0] = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[1] = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[2] = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[3] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[4] = g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Allocate work arrays
        iw = None
        w = None
        if self.tau_sz_iw > 0:
            iw = (ctypes.c_longlong * self.tau_sz_iw)()
        if self.tau_sz_w > 0:
            w = (ctypes.c_double * self.tau_sz_w)()
        
        # Call function
        ret = self.lib.left_arm_tau_from_regressor(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )
        
        if ret != 0:
            raise RuntimeError(f"left_arm_tau_from_regressor returned error code: {ret}")
        
        return tau

    def params_to_X_base(self, *params):
        """Build base parameter vector X_base from individual parameters.
        
        Args:
            *params: 70 individual parameters in order:
                For each of 7 links: [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
                Total: 7 * 10 = 70 parameters
        
        Returns:
            X_base: Base parameter vector (45,) array
        """
        if len(params) != 70:
            raise ValueError(f"Expected 70 parameters, got {len(params)}")
        
        # Convert to numpy array (contiguous)
        params_arr = np.ascontiguousarray(params, dtype=np.float64)
        
        # Allocate output (45 base parameters)
        X_base = np.zeros(45, dtype=np.float64)
        
        # Prepare arguments (70 individual pointers)
        # Create array of pointers, each pointing to an element of params_arr
        arg = (ctypes.POINTER(ctypes.c_double) * 70)()
        for i in range(70):
            # Get pointer to the i-th element of the array
            arg[i] = ctypes.cast(
                params_arr.ctypes.data + i * ctypes.sizeof(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            )
        
        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = X_base.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Allocate work arrays
        iw = None
        w = None
        if self.params_base_sz_iw > 0:
            iw = (ctypes.c_longlong * self.params_base_sz_iw)()
        if self.params_base_sz_w > 0:
            w = (ctypes.c_double * self.params_base_sz_w)()
        
        # Call function
        ret = self.lib.left_arm_params_to_X_base(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )
        
        if ret != 0:
            raise RuntimeError(f"left_arm_params_to_X_base returned error code: {ret}")
        
        return X_base
    
    def compute_regressor_base(self, q, v, a, g):
        """Compute base parameter regressor matrix Phi_base(q, v, a, g).

        Args:
            q: Joint positions (7,) array
            v: Joint velocities (7,) array
            a: Joint accelerations (7,) array
            g: Gravity vector (3,) array

        Returns:
            Phi_base: Base parameter regressor matrix (7, 45) array
        """
        # Convert inputs to numpy arrays and ensure contiguous
        q = np.ascontiguousarray(q, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        a = np.ascontiguousarray(a, dtype=np.float64)
        g = np.ascontiguousarray(g, dtype=np.float64)

        if q.shape != (7,) or v.shape != (7,) or a.shape != (7,) or g.shape != (3,):
            raise ValueError(f"Invalid input shapes: q={q.shape}, v={v.shape}, a={a.shape}, g={g.shape}")

        # Allocate output (column-major, according to CasADi's convention)
        Phi_base_raw = np.zeros((7, 45), dtype=np.float64, order='F')  # Fortran order for column-major

        # Prepare arguments
        arg = (ctypes.POINTER(ctypes.c_double) * 4)()
        arg[0] = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[1] = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[2] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[3] = g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = Phi_base_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Allocate work arrays
        iw = None
        w = None
        if self.regressor_base_sz_iw > 0:
            iw = (ctypes.c_longlong * self.regressor_base_sz_iw)()
        if self.regressor_base_sz_w > 0:
            w = (ctypes.c_double * self.regressor_base_sz_w)()

        # Call function
        ret = self.lib.left_arm_regressor_base(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )

        if ret != 0:
            raise RuntimeError(f"left_arm_regressor_base returned error code: {ret}")

        # CasADi outputs Phi_base in column-major order (Fortran). Return as (7, 45) C-order.
        Phi_base = np.array(Phi_base_raw, order='C')  # Convert to C-order for user (row-major)
        return Phi_base
    
    def compute_tau_from_regressor_base(self, X_base, q, v, a, g):
        """Compute torque from base parameter regressor: tau = Phi_base(q, v, a, g) * X_base.
        
        Args:
            X_base: Base parameter vector (45,) array
            q: Joint positions (7,) array
            v: Joint velocities (7,) array
            a: Joint accelerations (7,) array
            g: Gravity vector (3,) array
        
        Returns:
            tau: Joint torques (7,) array
        """
        # Convert inputs to numpy arrays and ensure contiguous
        X_base = np.ascontiguousarray(X_base, dtype=np.float64)
        q = np.ascontiguousarray(q, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        a = np.ascontiguousarray(a, dtype=np.float64)
        g = np.ascontiguousarray(g, dtype=np.float64)
        
        if X_base.shape != (45,):
            raise ValueError(f"Invalid X_base shape: {X_base.shape}, expected (45,)")
        if q.shape != (7,) or v.shape != (7,) or a.shape != (7,) or g.shape != (3,):
            raise ValueError(f"Invalid input shapes: q={q.shape}, v={v.shape}, a={a.shape}, g={g.shape}")
        
        # Allocate output
        tau = np.zeros(7, dtype=np.float64)
        
        # Prepare arguments
        arg = (ctypes.POINTER(ctypes.c_double) * 5)()
        arg[0] = X_base.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[1] = q.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[2] = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[3] = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        arg[4] = g.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        res = (ctypes.POINTER(ctypes.c_double) * 1)()
        res[0] = tau.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Allocate work arrays
        iw = None
        w = None
        if self.tau_base_sz_iw > 0:
            iw = (ctypes.c_longlong * self.tau_base_sz_iw)()
        if self.tau_base_sz_w > 0:
            w = (ctypes.c_double * self.tau_base_sz_w)()
        
        # Call function
        ret = self.lib.left_arm_tau_from_regressor_base(
            arg, res,
            iw if iw else None,
            w if w else None,
            0  # mem
        )
        
        if ret != 0:
            raise RuntimeError(f"left_arm_tau_from_regressor_base returned error code: {ret}")
        
        return tau

if __name__ == "__main__":
    # Simple test
    print("Testing LeftArmRegressor wrapper...")
    
    try:
        regressor = LeftArmRegressor()
        print("Successfully loaded shared library")
        
        # Test with simple inputs
        q = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        v = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        a = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        g = np.array([0.0, 0.0, -9.81])
        
        # Test regressor computation
        Phi = regressor.compute_regressor(q, v, a, g)
        print(f"Regressor shape: {Phi.shape}")
        
        # Test params_to_X (using dummy parameters)
        params = [1.0] * 70  # Dummy parameters
        X = regressor.params_to_X(*params)
        print(f"Parameter vector shape: {X.shape}")
        
        # Test tau computation
        tau = regressor.compute_tau_from_regressor(X, q, v, a, g)
        print(f"Torque shape: {tau.shape}")
        print(f"Torque values: {tau}")
        
        print("\nAll tests passed!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease compile the C files first:")
        print("  python build_regressor_lib.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

