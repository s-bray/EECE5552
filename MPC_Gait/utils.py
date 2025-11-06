# utils.py

import numpy as np
from scipy.spatial.transform import Rotation

from config import MPCParameters

def generate_reference_trajectory(params: MPCParameters, 
                                  current_state: np.ndarray,
                                  target_velocity: np.ndarray) -> np.ndarray:
    """Generate reference trajectory for MPC"""
    N = params.num_nodes
    dt = params.dt
    
    x_ref = np.zeros((N, 24))
    
    theta_0 = current_state[0:3]
    p_0 = current_state[3:6]
    omega_ref = target_velocity[3:6]
    v_ref = target_velocity[0:3]
    q_j_nominal = current_state[12:24]

    # q_j_nominal = np.array([
    #     0.0, 0.8, -1.6,  # FL
    #     0.0, 0.8, -1.6,  # FR
    #     0.0, 0.8, -1.6,  # HL
    #     0.0, 0.8, -1.6   # HR
    # ])

    for k in range(N):
        t = k * dt
        
        theta_k = theta_0 + omega_ref * t
        
        R_WB = Rotation.from_euler('xyz', theta_0).as_matrix()
        v_world = R_WB @ v_ref
        p_k = p_0 + v_world * t
        
        p_k[2] = 0.45  # Desired height
        
        x_ref[k] = np.concatenate([
            theta_k,
            p_k,
            omega_ref,
            v_ref,
            q_j_nominal
        ])
    
    return x_ref