# mpc_controller.py

import numpy as np
from typing import Tuple

from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from gait_generator import GaitSequenceGenerator

class MPCController:
    """Model Predictive Controller for whole-body motion planning"""
    
    def __init__(self, dynamics: SingleRigidBodyDynamics, params: MPCParameters):
        self.dynamics = dynamics
        self.params = params
        self.gait_gen = GaitSequenceGenerator(params)
        self.max_iterations = 5
        self.learning_rate = 0.01
        
    def compute_cost(self, x: np.ndarray, u: np.ndarray, 
                    x_ref: np.ndarray, u_ref: np.ndarray) -> float:
        """Quadratic cost function"""
        x_error = x - x_ref
        
        theta_err = x_error[0:3]
        p_err = x_error[3:6]
        omega_err = x_error[6:9]
        v_err = x_error[9:12]
        q_j_err = x_error[12:24]
        
        cost_state = (
            self.params.weight_orientation * np.sum(theta_err**2) +
            self.params.weight_position * np.sum(p_err**2) +
            self.params.weight_angular_velocity * np.sum(omega_err**2) +
            self.params.weight_linear_velocity * np.sum(v_err**2) +
            self.params.weight_joint_position * np.sum(q_j_err**2)
        )
        
        u_error = u - u_ref
        lambda_err = u_error[0:12]
        u_j_err = u_error[12:24]
        
        cost_control = (
            self.params.weight_contact_force * np.sum(lambda_err**2) +
            self.params.weight_joint_velocity * np.sum(u_j_err**2)
        )
        
        return 0.5 * (cost_state + cost_control)
    
    def enforce_constraints(self, u: np.ndarray, contact_states: np.ndarray) -> np.ndarray:
        """Enforce physical constraints on control inputs"""
        u_constrained = u.copy()
        lambda_e = u_constrained[0:12].reshape(4, 3)
        u_j = u_constrained[12:24]
        
        for i in range(4):
            if contact_states[i] == 1:
                force = lambda_e[i]
                force[2] = max(0.0, force[2])
                
                f_tangent = np.linalg.norm(force[0:2])
                f_normal = force[2]
                
                if f_tangent > self.params.friction_coeff * f_normal:
                    scale = self.params.friction_coeff * f_normal / (f_tangent + 1e-6)
                    force[0:2] *= scale
                
                force_mag = np.linalg.norm(force)
                if force_mag > self.params.max_contact_force:
                    force *= self.params.max_contact_force / force_mag
                
                lambda_e[i] = force
            else:
                lambda_e[i] = 0.0
        
        u_j = np.clip(u_j, -self.params.max_joint_velocity, self.params.max_joint_velocity)
        
        u_constrained[0:12] = lambda_e.flatten()
        u_constrained[12:24] = u_j
        
        return u_constrained
    
    def solve_mpc(self, x0: np.ndarray, x_ref_traj: np.ndarray, 
                 contact_schedule: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve MPC optimization problem"""
        N = self.params.num_nodes
        dt = self.params.dt
        
        u_seq = np.zeros((N, 24))
        u_ref = np.zeros(24)
        
        for iteration in range(self.max_iterations):
            x_traj = np.zeros((N, 24))
            x_traj[0] = x0
            total_cost = 0.0
            
            for k in range(N - 1):
                u_seq[k] = self.enforce_constraints(u_seq[k], contact_schedule[k])
                total_cost += self.compute_cost(x_traj[k], u_seq[k], 
                                               x_ref_traj[k], u_ref)
                
                x_traj[k+1] = self.dynamics.integrate_euler(
                    x_traj[k], u_seq[k], contact_schedule[k], dt
                )
            
            for k in range(N - 1):
                state_error = x_ref_traj[k+1] - x_traj[k+1]
                lambda_update = state_error[9:12] * self.params.robot_mass * 0.1
                u_j_update = state_error[12:24] * 0.5
                
                u_seq[k][0:3] += lambda_update * self.learning_rate
                u_seq[k][12:24] += u_j_update * self.learning_rate
        
        return u_seq, x_traj