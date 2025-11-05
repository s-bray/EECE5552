# config.py

import numpy as np
from dataclasses import dataclass

@dataclass
class MPCParameters:
    """Configuration parameters for the MPC controller"""
    
    # Robot physical parameters
    robot_mass: float = 30.0  # kg
    robot_inertia: np.ndarray = None  # Will be computed from XML
    
    # MPC horizon parameters
    horizon_length: float = 0.8  # seconds
    num_nodes: int = 20  # discretization points
    control_freq: float = 50.0  # Hz
    
    # Cost function weights
    weight_position: float = 100.0
    weight_orientation: float = 50.0
    weight_linear_velocity: float = 10.0
    weight_angular_velocity: float = 5.0
    weight_joint_position: float = 1.0
    weight_contact_force: float = 0.01
    weight_joint_velocity: float = 0.1
    
    # Physical constraints
    friction_coeff: float = 0.7
    max_joint_velocity: float = 10.0  # rad/s
    max_contact_force: float = 500.0  # N
    
    # Gait parameters
    swing_height: float = 0.1  # m
    swing_duration: float = 0.3  # s
    
    # Kinematics
    num_legs: int = 4
    joints_per_leg: int = 3
    
    def __post_init__(self):
        if self.robot_inertia is None:
            # Default inertia for quadruped (rough approximation)
            self.robot_inertia = np.diag([0.5, 1.0, 1.0])
        
        self.dt = self.horizon_length / self.num_nodes
        self.total_joints = self.num_legs * self.joints_per_leg