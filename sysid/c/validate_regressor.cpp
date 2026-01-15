/**
 * Validate Regressor Matrix Phi*X against Inverse Dynamics (RNEA)
 * 
 * This program validates that the regressor-based torque computation
 * (tau = Phi(q, v, a, g) * X) matches the inverse dynamics computation
 * (tau = M(q)*qdd + C(q, qd) + G(q))
 * 
 * Similar to the Python validation in gen_vr_m2_regressor.py
 * 
 * Usage:
 *   ./validate_regressor [--arm left|right] [--config q v a]
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

// CasADi generated function includes
extern "C" {
// Regressor functions (in sysid/py/autogen)
#include "autogen/VRM2RightArm/right_arm_regressor.h"
#include "autogen/VRM2RightArm/right_arm_tau_from_regressor.h"
#include "autogen/VRM2RightArm/right_arm_params_to_X.h"
#include "autogen/VRM2LeftArm/left_arm_regressor.h"
#include "autogen/VRM2LeftArm/left_arm_tau_from_regressor.h"
#include "autogen/VRM2LeftArm/left_arm_params_to_X.h"
// Dynamics functions (in lib/include)
#include "VRM2RightArm/right_arm_massMatrix.h"
#include "VRM2RightArm/right_arm_coriolisVector.h"
#include "VRM2RightArm/right_arm_gravityVector.h"
#include "VRM2LeftArm/left_arm_massMatrix.h"
#include "VRM2LeftArm/left_arm_coriolisVector.h"
#include "VRM2LeftArm/left_arm_gravityVector.h"
}

// Constants
const int DOF = 7;
const int N_PARAMS = 70;  // 10 params per link × 7 links
const double DEFAULT_GRAVITY[3] = {0.0, 0.0, -9.81};

// Joint limits (from IK files, in radians)
// Right arm limits (from right_arm_IK.c)
const double Q_MIN_RIGHT[DOF] = {-3.14, -2.36, -2.97, -0.52, -2.97, -0.34, -0.96};
const double Q_MAX_RIGHT[DOF] = {1.05, 0.00, 2.97, 1.57, 2.97, 0.34, 0.96};
// Left arm limits (from left_arm_IK.c)
const double Q_MIN_LEFT[DOF] = {-3.14, 0.0, -2.97, -0.52, -2.97, -0.34, -0.96};
const double Q_MAX_LEFT[DOF] = {1.05, 2.36, 2.97, 1.57, 2.97, 0.34, 0.96};
const double V_MAX = 0.0;  // Maximum velocity in rad/s
const double A_MAX = 0.0;  // Maximum acceleration in rad/s²

// Helper function to call CasADi functions
template<typename Func, typename WorkFunc>
int call_casadi_function(Func func, WorkFunc work_func, 
                        const double** arg, double** res,
                        casadi_int* iw, double* w) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    work_func(&sz_arg, &sz_res, &sz_iw, &sz_w);
    
    if (iw == nullptr && sz_iw > 0) {
        iw = (casadi_int*)malloc((size_t)sz_iw * sizeof(casadi_int));
    }
    if (w == nullptr && sz_w > 0) {
        w = (double*)malloc((size_t)sz_w * sizeof(double));
    }
    
    int ret = func(arg, res, iw, w, 0);
    
    if (iw && sz_iw > 0) free(iw);
    if (w && sz_w > 0) free(w);
    
    return ret;
}

// Compute regressor matrix Phi(q, v, a, g)
int compute_regressor_matrix_right_arm(const double* q, const double* v, const double* a,
                                       const double* g, double* Phi) {
    const double* arg[4] = {q, v, a, g};
    double* res[1] = {Phi};
    return call_casadi_function(right_arm_regressor,
                                right_arm_regressor_work,
                                arg, res, nullptr, nullptr);
}

// Compute regressor-based torque: tau = Phi * X
int compute_tau_regressor_right_arm(const double* q, const double* v, const double* a, 
                                    const double* g, const double* X, double* tau) {
    const double* arg[5] = {X, q, v, a, g};
    double* res[1] = {tau};
    return call_casadi_function(right_arm_tau_from_regressor, 
                                right_arm_tau_from_regressor_work,
                                arg, res, nullptr, nullptr);
}

// Compute regressor-based torque by computing Phi and then Phi * X
// This is used to verify that Phi @ X matches the direct tau computation
int compute_tau_from_Phi_X_right_arm(const double* q, const double* v, const double* a,
                                     const double* g, const double* X, double* tau) {
    // Compute regressor matrix Phi (7x70)
    double Phi[7 * 70];
    int ret = compute_regressor_matrix_right_arm(q, v, a, g, Phi);
    if (ret != 0) return ret;
    
    // Compute tau = Phi * X
    // IMPORTANT: CasADi outputs matrices in column-major (Fortran) order.
    // For a matrix Phi with shape (nrows=DOF, ncols=N_PARAMS), the element
    // at row i, column j is stored at index: Phi[j * nrows + i] = Phi[j * DOF + i]
    // This is different from row-major (C-order) where it would be: Phi[i * ncols + j]
    for (int i = 0; i < DOF; i++) {
        tau[i] = 0.0;
        for (int j = 0; j < N_PARAMS; j++) {
            tau[i] += Phi[j * DOF + i] * X[j];  // Column-major indexing
        }
    }
    
    return 0;
}

// Compute inverse dynamics torque: tau = M*qdd + C + G
int compute_tau_rnea_right_arm(const double* q, const double* v, const double* a,
                               double* tau) {
    // Compute mass matrix M
    double M[DOF * DOF];
    const double* arg_M[1] = {q};
    double* res_M[1] = {M};
    int ret = call_casadi_function(right_arm_massMatrix, 
                                   right_arm_massMatrix_work,
                                   arg_M, res_M, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute Coriolis vector C
    double C[DOF];
    const double* arg_C[2] = {q, v};
    double* res_C[1] = {C};
    ret = call_casadi_function(right_arm_coriolisVector,
                               right_arm_coriolisVector_work,
                               arg_C, res_C, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute gravity vector G
    double G[DOF];
    const double* arg_G[1] = {q};
    double* res_G[1] = {G};
    ret = call_casadi_function(right_arm_gravityVector,
                               right_arm_gravityVector_work,
                               arg_G, res_G, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute tau = M*qdd + C + G
    // Note: CasADi outputs M in column-major order: M(i,j) = M[j*DOF + i]
    for (int i = 0; i < DOF; i++) {
        tau[i] = G[i] + C[i];
        for (int j = 0; j < DOF; j++) {
            tau[i] += M[j*DOF + i] * a[j];  // Column-major: M(i,j) = M[j*DOF + i]
        }
    }
    
    return 0;
}

// Compute regressor matrix Phi(q, v, a, g) (left arm)
int compute_regressor_matrix_left_arm(const double* q, const double* v, const double* a,
                                      const double* g, double* Phi) {
    const double* arg[4] = {q, v, a, g};
    double* res[1] = {Phi};
    return call_casadi_function(left_arm_regressor,
                                left_arm_regressor_work,
                                arg, res, nullptr, nullptr);
}

// Compute regressor-based torque: tau = Phi * X (left arm)
int compute_tau_regressor_left_arm(const double* q, const double* v, const double* a,
                                   const double* g, const double* X, double* tau) {
    const double* arg[5] = {X, q, v, a, g};
    double* res[1] = {tau};
    return call_casadi_function(left_arm_tau_from_regressor,
                                left_arm_tau_from_regressor_work,
                                arg, res, nullptr, nullptr);
}

// Compute inverse dynamics torque: tau = M*qdd + C + G (left arm)
int compute_tau_rnea_left_arm(const double* q, const double* v, const double* a,
                              double* tau) {
    // Compute mass matrix M
    double M[DOF * DOF];
    const double* arg_M[1] = {q};
    double* res_M[1] = {M};
    int ret = call_casadi_function(left_arm_massMatrix,
                                   left_arm_massMatrix_work,
                                   arg_M, res_M, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute Coriolis vector C
    double C[DOF];
    const double* arg_C[2] = {q, v};
    double* res_C[1] = {C};
    ret = call_casadi_function(left_arm_coriolisVector,
                               left_arm_coriolisVector_work,
                               arg_C, res_C, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute gravity vector G
    double G[DOF];
    const double* arg_G[1] = {q};
    double* res_G[1] = {G};
    ret = call_casadi_function(left_arm_gravityVector,
                               left_arm_gravityVector_work,
                               arg_G, res_G, nullptr, nullptr);
    if (ret != 0) return ret;
    
    // Compute tau = M*qdd + C + G
    // Note: CasADi outputs M in column-major order: M(i,j) = M[j*DOF + i]
    for (int i = 0; i < DOF; i++) {
        tau[i] = G[i] + C[i];
        for (int j = 0; j < DOF; j++) {
            tau[i] += M[j*DOF + i] * a[j];  // Column-major: M(i,j) = M[j*DOF + i]
        }
    }
    
    return 0;
}

// Build parameter vector X from individual parameters (Pinocchio format)
// Format: [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] per link
int build_X_right_arm(const double params[70], double* X) {
    const double* arg[70];
    for (int i = 0; i < 70; i++) {
        arg[i] = &params[i];
    }
    double* res[1] = {X};
    return call_casadi_function(right_arm_params_to_X,
                                right_arm_params_to_X_work,
                                arg, res, nullptr, nullptr);
}

int build_X_left_arm(const double params[70], double* X) {
    const double* arg[70];
    for (int i = 0; i < 70; i++) {
        arg[i] = &params[i];
    }
    double* res[1] = {X};
    return call_casadi_function(left_arm_params_to_X,
                                left_arm_params_to_X_work,
                                arg, res, nullptr, nullptr);
}

// Default parameters (approximate values - should match URDF)
// These are placeholder values. In practice, these should be extracted from URDF
// or loaded from a config file. For validation, we use reasonable defaults.
// void get_default_params_right_arm(double params[70]) {
//     // Initialize to zero (will need actual values from URDF for accurate validation)
//     // This is a placeholder - actual validation requires correct parameters
//     for (int i = 0; i < 70; i++) {
//         params[i] = 0.0;
//     }
//     // Note: For proper validation, parameters should be extracted from URDF
//     // or provided via command line/config file
// }

// Get default parameters in Pinocchio format (original format)
// Pinocchio format: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] per link
// where mx, my, mz are first moments (m * com), and I is inertia at body origin
void get_default_params_right_arm_pinocchio(double params[70]) {
    // Pinocchio format: [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] per link

    // Link 1
    params[ 0] = 1.039499998093e+00;  // m
    params[ 1] = -3.785703068054e-03;  // mx
    params[ 2] = -5.709297814524e-05;  // my
    params[ 3] = 1.796391131704e-03;  // mz
    params[ 4] = 1.039200000000e-03;  // Ixx
    params[ 5] = -9.235950000000e-08;  // Ixy
    params[ 6] = 3.918970000000e-07;  // Ixz
    params[ 7] = 1.008190000000e-03;  // Iyy
    params[ 8] = 2.309200000000e-05;  // Iyz
    params[ 9] = 9.519280000000e-04;  // Izz

    // Link 2
    params[10] = 1.132040023804e+00;  // m
    params[11] = 9.064527480602e-02;  // mx
    params[12] = -1.214599702740e-04;  // my
    params[13] = -8.197022569561e-04;  // mz
    params[14] = 1.190160000000e-03;  // Ixx
    params[15] = 5.335010000000e-06;  // Ixy
    params[16] = -6.340060000000e-07;  // Ixz
    params[17] = 1.972470000000e-03;  // Iyy
    params[18] = -1.660840000000e-06;  // Iyz
    params[19] = 1.848450000000e-03;  // Izz

    // Link 3
    params[20] = 2.892319858074e-01;  // m
    params[21] = 3.817862212658e-03;  // mx
    params[22] = -1.561849831040e-03;  // my
    params[23] = 1.709855622818e-02;  // mz
    params[24] = 5.041660000000e-04;  // Ixx
    params[25] = 4.699980000000e-06;  // Ixy
    params[26] = -1.250680000000e-04;  // Ixz
    params[27] = 5.511090000000e-04;  // Iyy
    params[28] = 8.025970000000e-06;  // Iyz
    params[29] = 3.177650000000e-04;  // Izz

    // Link 4
    params[30] = 1.340999960899e+00;  // m
    params[31] = 5.695213423940e-02;  // mx
    params[32] = 2.599850264194e-04;  // my
    params[33] = -3.611286074703e-03;  // mz
    params[34] = 7.061820000000e-04;  // Ixx
    params[35] = -3.861300000000e-06;  // Ixy
    params[36] = 1.497640000000e-04;  // Ixz
    params[37] = 3.111930000000e-03;  // Iyy
    params[38] = 1.294140000000e-06;  // Iyz
    params[39] = 3.223790000000e-03;  // Izz

    // Link 5
    params[40] = 7.200000286102e-01;  // m
    params[41] = 1.166356846347e-03;  // mx
    params[42] = 5.132246603937e-06;  // my
    params[43] = 4.225809767919e-02;  // mz
    params[44] = 7.604860000000e-04;  // Ixx
    params[45] = -2.255120000000e-07;  // Ixy
    params[46] = -1.076320000000e-05;  // Ixz
    params[47] = 9.321310000000e-04;  // Iyy
    params[48] = 1.300310000000e-07;  // Iyz
    params[49] = 6.077820000000e-04;  // Izz

    // Link 6
    params[50] = 4.995679855347e-01;  // m
    params[51] = -1.711754715395e-03;  // mx
    params[52] = -4.935886563158e-03;  // my
    params[53] = -1.123598338985e-06;  // mz
    params[54] = 2.212900000000e-04;  // Ixx
    params[55] = -1.820100000000e-07;  // Ixy
    params[56] = 4.540300000000e-07;  // Ixz
    params[57] = 2.917360000000e-04;  // Iyy
    params[58] = 1.779890000000e-07;  // Iyz
    params[59] = 2.862090000000e-04;  // Izz

    // Link 7
    params[60] = 9.673000335693e-02;  // m
    params[61] = 3.430727450708e-03;  // mx
    params[62] = -9.224498564440e-07;  // my
    params[63] = -1.784036069589e-03;  // mz
    params[64] = 6.132550116600e-05;  // Ixx
    params[65] = 3.639493876794e-08;  // Ixy
    params[66] = 1.753894428823e-05;  // Ixz
    params[67] = 9.626207575310e-05;  // Iyy
    params[68] = 1.437646971841e-08;  // Iyz
    params[69] = 7.484092919592e-05;  // Izz

}
// Convert parameters from Pinocchio format to physical format expected by params_to_X
// Pinocchio format: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz] (inertia at origin)
// Physical format: [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] (inertia at CoM)
// where cx = mx/m, cy = my/m, cz = mz/m, and I_com = I_origin - m * S(c)^T * S(c)
void convert_pinocchio_to_physical_params(const double params_pin[70], double params_phys[70]) {
    for (int link = 0; link < 7; link++) {
        int idx = link * 10;
        
        // Extract Pinocchio format parameters
        double m = params_pin[idx + 0];
        double mx = params_pin[idx + 1];
        double my = params_pin[idx + 2];
        double mz = params_pin[idx + 3];
        double Ixx_orig = params_pin[idx + 4];
        double Ixy_orig = params_pin[idx + 5];
        double Iyy_orig = params_pin[idx + 6];
        double Ixz_orig = params_pin[idx + 7];
        double Iyz_orig = params_pin[idx + 8];
        double Izz_orig = params_pin[idx + 9];
        
        // Convert first moments to CoM coordinates
        double cx = mx / m;
        double cy = my / m;
        double cz = mz / m;
        
        // Convert inertia from origin to CoM: I_com = I_origin - m * S(c)^T * S(c)
        // S(c)^T * S(c) = [cy^2+cz^2, -cx*cy, -cx*cz]
        //                  [-cx*cy, cx^2+cz^2, -cy*cz]
        //                  [-cx*cz, -cy*cz, cx^2+cy^2]
        double cx2 = cx * cx;
        double cy2 = cy * cy;
        double cz2 = cz * cz;
        double m_term_xx = m * (cy2 + cz2);
        double m_term_yy = m * (cx2 + cz2);
        double m_term_zz = m * (cx2 + cy2);
        double m_term_xy = -m * cx * cy;
        double m_term_xz = -m * cx * cz;
        double m_term_yz = -m * cy * cz;
        
        // Physical format: [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        params_phys[idx + 0] = m;
        params_phys[idx + 1] = cx;
        params_phys[idx + 2] = cy;
        params_phys[idx + 3] = cz;
        params_phys[idx + 4] = Ixx_orig - m_term_xx;  // Ixx_com
        params_phys[idx + 5] = Ixy_orig - m_term_xy;  // Ixy_com
        params_phys[idx + 6] = Ixz_orig - m_term_xz;  // Ixz_com
        params_phys[idx + 7] = Iyy_orig - m_term_yy;  // Iyy_com
        params_phys[idx + 8] = Iyz_orig - m_term_yz;  // Iyz_com
        params_phys[idx + 9] = Izz_orig - m_term_zz;  // Izz_com
    }
}

// Get default parameters in physical format (for params_to_X function)
void get_default_params_right_arm(double params[70]) {
    double params_pin[70];
    get_default_params_right_arm_pinocchio(params_pin);
    convert_pinocchio_to_physical_params(params_pin, params);
}


void get_default_params_left_arm(double params[70]) {
    // Physical format expected by params_to_X: [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz] per link
    // TODO: Extract actual parameters from URDF (similar to right arm)
    // For now, all zeros (will cause incorrect validation results)
    for (int i = 0; i < 70; i++) {
        params[i] = 0.0;
    }
}

// Generate random configuration
void generate_random_config(std::mt19937& gen, const char* arm_name, double* q, double* v, double* a) {
    const double* q_min = (strcmp(arm_name, "right") == 0) ? Q_MIN_RIGHT : Q_MIN_LEFT;
    const double* q_max = (strcmp(arm_name, "right") == 0) ? Q_MAX_RIGHT : Q_MAX_LEFT;
    
    for (int i = 0; i < DOF; i++) {
        std::uniform_real_distribution<double> q_dist(q_min[i], q_max[i]);
        std::uniform_real_distribution<double> v_dist(-V_MAX, V_MAX);
        std::uniform_real_distribution<double> a_dist(-A_MAX, A_MAX);
        
        q[i] = q_dist(gen);
        v[i] = v_dist(gen);
        a[i] = a_dist(gen);
    }
}

// Validate regressor with timing
bool validate_regressor_config_timed(const char* arm_name,
                                    const double* q, const double* v, const double* a,
                                    const double* g, const double* X,
                                    double& max_error, double* tau_reg, double* tau_rnea,
                                    double& time_regressor_us, double& time_rnea_us) {
    bool is_right_arm = (strcmp(arm_name, "right") == 0);
    
    // Time regressor computation
    auto start = std::chrono::high_resolution_clock::now();
    int ret;
    if (is_right_arm) {
        ret = compute_tau_regressor_right_arm(q, v, a, g, X, tau_reg);
    } else {
        ret = compute_tau_regressor_left_arm(q, v, a, g, X, tau_reg);
    }
    auto end = std::chrono::high_resolution_clock::now();
    time_regressor_us = std::chrono::duration<double, std::micro>(end - start).count();
    
    if (ret != 0) {
        std::cerr << "Error computing regressor torque: " << ret << std::endl;
        return false;
    }
    
    // Additional validation: Check if Phi @ X matches the direct computation
    if (is_right_arm) {
        double tau_from_Phi[DOF];
        ret = compute_tau_from_Phi_X_right_arm(q, v, a, g, X, tau_from_Phi);
        if (ret == 0) {
            double phi_error = 0.0;
            for (int i = 0; i < DOF; i++) {
                double err = std::abs(tau_reg[i] - tau_from_Phi[i]);
                if (err > phi_error) phi_error = err;
            }
            if (phi_error > 1e-6) {
                std::cerr << "WARNING: Phi @ X does not match direct tau computation!" << std::endl;
                std::cerr << "  Error: " << std::scientific << phi_error << " Nm" << std::endl;
                std::cerr << "  This indicates the regressor matrix Phi is incorrect." << std::endl;
            }
        }
    }
    
    // Time RNEA computation
    start = std::chrono::high_resolution_clock::now();
    if (is_right_arm) {
        ret = compute_tau_rnea_right_arm(q, v, a, tau_rnea);
    } else {
        ret = compute_tau_rnea_left_arm(q, v, a, tau_rnea);
    }
    end = std::chrono::high_resolution_clock::now();
    time_rnea_us = std::chrono::duration<double, std::micro>(end - start).count();
    
    if (ret != 0) {
        std::cerr << "Error computing RNEA torque: " << ret << std::endl;
        return false;
    }
    
    // Compute maximum error
    max_error = 0.0;
    for (int i = 0; i < DOF; i++) {
        double error = std::abs(tau_reg[i] - tau_rnea[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    
    return true;
}

// Validate regressor for a given configuration
bool validate_regressor_config(const char* arm_name, 
                              const double* q, const double* v, const double* a,
                              const double* g, const double* X,
                              double& max_error, double* tau_reg, double* tau_rnea) {
    double time_reg, time_rnea;
    return validate_regressor_config_timed(arm_name, q, v, a, g, X, max_error, 
                                          tau_reg, tau_rnea, time_reg, time_rnea);
}

// Statistics structure
struct TestStatistics {
    int n_tests;
    double max_error;
    double mean_error;
    double min_error;
    double std_error;
    double total_time_regressor_us;
    double total_time_rnea_us;
    double mean_time_regressor_us;
    double mean_time_rnea_us;
    int passed_count;
    int warning_count;
    int failed_count;
};

// Run N randomized test cases
TestStatistics run_randomized_tests(const char* arm_name, const double* X, 
                                    const double* g, int n_tests, int seed = 0) {
    TestStatistics stats = {};
    stats.n_tests = n_tests;
    stats.min_error = 1e10;
    stats.max_error = 0.0;
    stats.mean_error = 0.0;
    stats.passed_count = 0;
    stats.warning_count = 0;
    stats.failed_count = 0;
    
    // Initialize random number generator
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    // Storage for errors
    std::vector<double> errors;
    errors.reserve(n_tests);
    
    double q[DOF], v[DOF], a[DOF];
    double tau_reg[DOF], tau_rnea[DOF];
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Running " << n_tests << " Randomized Test Cases for " << arm_name << " Arm" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Progress: ";
    
    for (int test = 0; test < n_tests; test++) {
        // Generate random configuration
        generate_random_config(gen, arm_name, q, v, a);
        
        // Validate with timing
        double max_error;
        double time_regressor_us, time_rnea_us;
        
        if (!validate_regressor_config_timed(arm_name, q, v, a, g, X, max_error,
                                            tau_reg, tau_rnea, 
                                            time_regressor_us, time_rnea_us)) {
            std::cerr << "\nTest " << test + 1 << " failed!" << std::endl;
            stats.failed_count++;
            continue;
        }
        
        // Update statistics
        errors.push_back(max_error);
        stats.total_time_regressor_us += time_regressor_us;
        stats.total_time_rnea_us += time_rnea_us;
        
        if (max_error < stats.min_error) stats.min_error = max_error;
        if (max_error > stats.max_error) stats.max_error = max_error;
        
        if (max_error < 1e-6) {
            stats.passed_count++;
        } else if (max_error < 1e-3) {
            stats.warning_count++;
        } else {
            stats.failed_count++;
        }
        
        // Progress indicator
        if ((test + 1) % std::max(1, n_tests / 20) == 0 || test == n_tests - 1) {
            std::cout << "." << std::flush;
        }
    }
    
    // Calculate mean and standard deviation
    if (!errors.empty()) {
        stats.mean_error = 0.0;
        for (double err : errors) {
            stats.mean_error += err;
        }
        stats.mean_error /= errors.size();
        
        stats.std_error = 0.0;
        for (double err : errors) {
            double diff = err - stats.mean_error;
            stats.std_error += diff * diff;
        }
        stats.std_error = std::sqrt(stats.std_error / errors.size());
        
        stats.mean_time_regressor_us = stats.total_time_regressor_us / n_tests;
        stats.mean_time_rnea_us = stats.total_time_rnea_us / n_tests;
    }
    
    std::cout << " Done!" << std::endl;
    return stats;
}

// Print test statistics
void print_statistics(const TestStatistics& stats) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Statistics" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nNumber of Tests: " << stats.n_tests << std::endl;
    
    std::cout << "\nError Statistics:" << std::endl;
    std::cout << "  Min Error:   " << std::scientific << std::setprecision(6) << stats.min_error << " Nm" << std::endl;
    std::cout << "  Max Error:   " << std::scientific << std::setprecision(6) << stats.max_error << " Nm" << std::endl;
    std::cout << "  Mean Error:  " << std::scientific << std::setprecision(6) << stats.mean_error << " Nm" << std::endl;
    std::cout << "  Std Error:   " << std::scientific << std::setprecision(6) << stats.std_error << " Nm" << std::endl;
    
    std::cout << "\nExecution Time Statistics:" << std::endl;
    std::cout << "  Regressor Method:" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(2) << stats.mean_time_regressor_us << " μs" << std::endl;
    std::cout << "    Total: " << std::fixed << std::setprecision(2) << stats.total_time_regressor_us << " μs" << std::endl;
    std::cout << "  RNEA Method:" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(2) << stats.mean_time_rnea_us << " μs" << std::endl;
    std::cout << "    Total: " << std::fixed << std::setprecision(2) << stats.total_time_rnea_us << " μs" << std::endl;
    
    double speedup = stats.mean_time_rnea_us / stats.mean_time_regressor_us;
    std::cout << "  Speedup (RNEA/Regressor): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    std::cout << "\nTest Results:" << std::endl;
    std::cout << "  Passed (error < 1e-6):  " << stats.passed_count << " (" 
              << std::fixed << std::setprecision(1) 
              << 100.0 * stats.passed_count / stats.n_tests << "%)" << std::endl;
    std::cout << "  Warning (error < 1e-3): " << stats.warning_count << " (" 
              << std::fixed << std::setprecision(1) 
              << 100.0 * stats.warning_count / stats.n_tests << "%)" << std::endl;
    std::cout << "  Failed (error >= 1e-3): " << stats.failed_count << " (" 
              << std::fixed << std::setprecision(1) 
              << 100.0 * stats.failed_count / stats.n_tests << "%)" << std::endl;
    
    std::cout << "\nOverall Status: ";
    if (stats.failed_count == 0 && stats.warning_count == 0) {
        std::cout << "PASSED (all tests passed)" << std::endl;
    } else if (stats.failed_count == 0) {
        std::cout << "PASSED (some warnings)" << std::endl;
    } else {
        std::cout << "FAILED (some tests failed)" << std::endl;
    }
    
    std::cout << std::string(70, '=') << std::endl;
}

// Print validation results
void print_results(const char* arm_name, const double* q, const double* v, const double* a,
                   const double* tau_reg, const double* tau_rnea, double max_error) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Validation Results for " << arm_name << " Arm" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  q: [";
    for (int i = 0; i < DOF; i++) {
        std::cout << std::fixed << std::setprecision(4) << q[i];
        if (i < DOF - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  v: [";
    for (int i = 0; i < DOF; i++) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < DOF - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  a: [";
    for (int i = 0; i < DOF; i++) {
        std::cout << std::fixed << std::setprecision(4) << a[i];
        if (i < DOF - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "\nTorque Comparison:" << std::endl;
    std::cout << std::setw(5) << "Joint" 
              << std::setw(15) << "tau_regressor" 
              << std::setw(15) << "tau_rnea" 
              << std::setw(15) << "error" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (int i = 0; i < DOF; i++) {
        double error = std::abs(tau_reg[i] - tau_rnea[i]);
        std::cout << std::setw(5) << i + 1
                  << std::setw(15) << std::scientific << std::setprecision(6) << tau_reg[i]
                  << std::setw(15) << tau_rnea[i]
                  << std::setw(15) << error << std::endl;
    }
    
    std::cout << "\nMax Error: " << std::scientific << std::setprecision(6) << max_error << " Nm" << std::endl;
    
    if (max_error < 1e-6) {
        std::cout << "Status: PASSED (error < 1e-6)" << std::endl;
    } else if (max_error < 1e-3) {
        std::cout << "Status: PASSED (error < 1e-3)" << std::endl;
    } else {
        std::cout << "Status: WARNING (error >= 1e-3)" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;
}

// Print usage information
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --arm <left|right>    Arm to validate (default: right)" << std::endl;
    std::cout << "  --n <N>               Number of randomized test cases (default: 1, single test)" << std::endl;
    std::cout << "  --seed <seed>         Random seed (default: random)" << std::endl;
    std::cout << "  --q <q1 q2 ... q7>   Joint positions in radians (default: all zeros, ignored if --n > 1)" << std::endl;
    std::cout << "  --v <v1 v2 ... v7>   Joint velocities in rad/s (default: all zeros, ignored if --n > 1)" << std::endl;
    std::cout << "  --a <a1 a2 ... a7>   Joint accelerations in rad/s² (default: all zeros, ignored if --n > 1)" << std::endl;
    std::cout << "  --help                Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << prog_name << " --arm right --n 1000          # Run 1000 randomized tests" << std::endl;
    std::cout << "  " << prog_name << " --arm right --n 100 --seed 42  # Run 100 tests with seed 42" << std::endl;
    std::cout << "  " << prog_name << " --arm right --q 0.1 0.2 ...   # Single test with custom config" << std::endl;
    std::cout << "\nNote: Parameter vector X should be set correctly for accurate validation." << std::endl;
}

int main(int argc, char* argv[]) {
    // Default configuration
    const char* arm_name = "right";
    int n_tests = 1;
    int seed = 0;
    double q[DOF] = {0.0};
    double v[DOF] = {0.0};
    double a[DOF] = {0.0};
    double g[3] = {DEFAULT_GRAVITY[0], DEFAULT_GRAVITY[1], DEFAULT_GRAVITY[2]};
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--arm") == 0 && i + 1 < argc) {
            arm_name = argv[++i];
            if (strcmp(arm_name, "left") != 0 && strcmp(arm_name, "right") != 0) {
                std::cerr << "Error: --arm must be 'left' or 'right'" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n_tests = std::stoi(argv[++i]);
            if (n_tests < 1) {
                std::cerr << "Error: --n must be >= 1" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--q") == 0 && i + DOF < argc) {
            for (int j = 0; j < DOF; j++) {
                q[j] = std::stod(argv[++i]);
            }
        } else if (strcmp(argv[i], "--v") == 0 && i + DOF < argc) {
            for (int j = 0; j < DOF; j++) {
                v[j] = std::stod(argv[++i]);
            }
        } else if (strcmp(argv[i], "--a") == 0 && i + DOF < argc) {
            for (int j = 0; j < DOF; j++) {
                a[j] = std::stod(argv[++i]);
            }
        }
    }
    
    // Build parameter vector X
    double params[70];
    double X[N_PARAMS];
    
    if (strcmp(arm_name, "right") == 0) {
        get_default_params_right_arm(params);
        if (build_X_right_arm(params, X) != 0) {
            std::cerr << "Error building parameter vector X for right arm" << std::endl;
            return 1;
        }
    } else {
        get_default_params_left_arm(params);
        if (build_X_left_arm(params, X) != 0) {
            std::cerr << "Error building parameter vector X for left arm" << std::endl;
            return 1;
        }
    }
    
    // Run tests
    if (n_tests > 1) {
        // Run randomized test cases
        TestStatistics stats = run_randomized_tests(arm_name, X, g, n_tests, seed);
        print_statistics(stats);
        
        return (stats.failed_count == 0) ? 0 : 1;
    } else {
        // Single test case
        double tau_reg[DOF], tau_rnea[DOF];
        double max_error;
        double time_regressor_us, time_rnea_us;
        
        if (!validate_regressor_config_timed(arm_name, q, v, a, g, X, max_error,
                                            tau_reg, tau_rnea, 
                                            time_regressor_us, time_rnea_us)) {
            std::cerr << "Validation failed!" << std::endl;
            return 1;
        }
        
        // Print results with timing
        print_results(arm_name, q, v, a, tau_reg, tau_rnea, max_error);
        
        std::cout << "\nExecution Times:" << std::endl;
        std::cout << "  Regressor Method: " << std::fixed << std::setprecision(2) 
                  << time_regressor_us << " μs" << std::endl;
        std::cout << "  RNEA Method:      " << std::fixed << std::setprecision(2) 
                  << time_rnea_us << " μs" << std::endl;
        std::cout << "  Speedup:           " << std::fixed << std::setprecision(2) 
                  << time_rnea_us / time_regressor_us << "x" << std::endl;
        
        return (max_error < 1e-3) ? 0 : 1;
    }
}

