# dynamics.py

import numpy as np
from scipy.spatial.transform import Rotation
from typing import List

from config import MPCParameters

class SingleRigidBodyDynamics:
    """
    Reduced-order model: Single Rigid Body Dynamics (SRBD)
    
    State: x = [theta, p, omega, v, q_j]^T
    - theta: [roll, pitch, yaw] (3) - Euler angles
    - p: [x, y, z] (3) - COM position in world frame
    - omega: [wx, wy, wz] (3) - angular velocity in body frame
    - v: [vx, vy, vz] (3) - linear velocity in body frame
    - q_j: joint positions (12) - 3 per leg x 4 legs
    
    Total state dimension: 24
    
    Control: u = [lambda_e, u_j]^T
    - lambda_e: contact forces (12) - 3D force per leg x 4 legs
    - u_j: joint velocities (12)
    
    Total control dimension: 24
    """
    
    def __init__(self, params: MPCParameters):
        self.params = params
        self.m = params.robot_mass
        self.I = params.robot_inertia
        self.I_inv = np.linalg.inv(self.I)
        self.g = np.array([0., 0., -9.81])
        
        # State and control dimensions
        self.state_dim = 24
        self.control_dim = 24
        
    def euler_rate_transform(self, theta: np.ndarray) -> np.ndarray:
        """Transform matrix T(theta) for Euler angle rates"""
        phi, psi, chi = theta  # roll, pitch, yaw
        
        T = np.array([
            [1, np.sin(phi) * np.tan(psi), np.cos(phi) * np.tan(psi)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(psi), np.cos(phi) / np.cos(psi)]
        ])
        
        return T
    
    def rotation_matrix_body_to_world(self, theta: np.ndarray) -> np.ndarray:
        """Rotation matrix R_WB(theta) from body frame to world frame"""
        r = Rotation.from_euler('xyz', theta)
        return r.as_matrix()
    
    def rotation_matrix_world_to_body(self, theta: np.ndarray) -> np.ndarray:
        """Rotation matrix R_BW(theta) from world frame to body frame"""
        return self.rotation_matrix_body_to_world(theta).T
    
    def gravity_in_body_frame(self, theta: np.ndarray) -> np.ndarray:
        """Gravity vector expressed in body frame"""
        R_BW = self.rotation_matrix_world_to_body(theta)
        g_world = np.array([0., 0., -9.81])
        return R_BW @ g_world
    
    def skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix for cross product"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def compute_contact_positions(self, q_j: np.ndarray, 
                                  base_orientation: np.ndarray) -> List[np.ndarray]:
        """Compute end-effector positions relative to COM"""
        contact_positions = []
        
        # Simplified leg geometry
        hip_offset = 0.3  # m
        thigh_length = 0.25  # m
        shank_length = 0.25  # m
        
        hip_positions_body = [
            np.array([hip_offset, hip_offset, 0]),   # LF
            np.array([hip_offset, -hip_offset, 0]),  # RF
            np.array([-hip_offset, hip_offset, 0]),  # LH
            np.array([-hip_offset, -hip_offset, 0])  # RH
        ]
        
        for i in range(4):
            q_hip, q_thigh, q_shank = q_j[i*3:(i+1)*3]
            
            x = thigh_length * np.sin(q_thigh) + shank_length * np.sin(q_thigh + q_shank)
            z = -thigh_length * np.cos(q_thigh) - shank_length * np.cos(q_thigh + q_shank)
            
            foot_hip = np.array([x, 0, z])
            R_hip = Rotation.from_euler('z', q_hip).as_matrix()
            foot_hip_rotated = R_hip @ foot_hip
            r_ei = hip_positions_body[i] + foot_hip_rotated
            
            contact_positions.append(r_ei)
        
        return contact_positions
    
    def dynamics(self, x: np.ndarray, u: np.ndarray, 
                contact_states: np.ndarray) -> np.ndarray:
        """Compute state derivative: x_dot = f(x, u)"""
        theta = x[0:3]
        p = x[3:6]
        omega = x[6:9]
        v = x[9:12]
        q_j = x[12:24]
        
        lambda_e = u[0:12].reshape(4, 3)
        u_j = u[12:24]
        
        for i in range(4):
            if contact_states[i] == 0:
                lambda_e[i] = 0.0
        
        r_contacts = self.compute_contact_positions(q_j, theta)
        
        T = self.euler_rate_transform(theta)
        theta_dot = T @ omega
        
        R_WB = self.rotation_matrix_body_to_world(theta)
        p_dot = R_WB @ v
        
        omega_cross_Iomega = np.cross(omega, self.I @ omega)
        net_torque = np.zeros(3)
        for i in range(4):
            if contact_states[i] == 1:
                net_torque += np.cross(r_contacts[i], lambda_e[i])
        
        omega_dot = self.I_inv @ (-omega_cross_Iomega + net_torque)
        
        g_body = self.gravity_in_body_frame(theta)
        net_force = np.sum(lambda_e, axis=0)
        v_dot = g_body + net_force / self.m
        
        q_j_dot = u_j
        
        x_dot = np.concatenate([theta_dot, p_dot, omega_dot, v_dot, q_j_dot])
        return x_dot
    
    def integrate_euler(self, x: np.ndarray, u: np.ndarray, 
                       contact_states: np.ndarray, dt: float) -> np.ndarray:
        """Simple Euler integration"""
        x_dot = self.dynamics(x, u, contact_states)
        x_next = x + dt * x_dot
        return x_next