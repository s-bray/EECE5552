"""
Whole-Body Model Predictive Control for Wheeled-Legged Robots
Based on: Bjelonic et al. "Whole-Body MPC and Online Gait Sequence Generation"

This implementation focuses on the walking/legged locomotion part using
Single Rigid Body Dynamics (SRBD) model.
"""

import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation
import time


@dataclass
class MPCParameters:
    """Configuration parameters for the MPC controller"""
    
    # Robot physical parameters
    robot_mass: float = 30.0  # kg
    robot_inertia: np.ndarray = None  # Will be computed from URDF
    
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
        """
        Transform matrix T(theta) that converts body angular velocity 
        to Euler angle rates: theta_dot = T(theta) * omega
        
        Args:
            theta: [roll, pitch, yaw]
        
        Returns:
            T: 3x3 transformation matrix
        """
        phi, psi, chi = theta  # roll, pitch, yaw
        
        T = np.array([
            [1, np.sin(phi) * np.tan(psi), np.cos(phi) * np.tan(psi)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(psi), np.cos(phi) / np.cos(psi)]
        ])
        
        return T
    
    def rotation_matrix_body_to_world(self, theta: np.ndarray) -> np.ndarray:
        """
        Rotation matrix R_WB(theta) from body frame to world frame
        
        Args:
            theta: [roll, pitch, yaw]
        
        Returns:
            R_WB: 3x3 rotation matrix
        """
        r = Rotation.from_euler('xyz', theta)
        return r.as_matrix()
    
    def rotation_matrix_world_to_body(self, theta: np.ndarray) -> np.ndarray:
        """
        Rotation matrix R_BW(theta) from world frame to body frame
        
        Args:
            theta: [roll, pitch, yaw]
        
        Returns:
            R_BW: 3x3 rotation matrix
        """
        return self.rotation_matrix_body_to_world(theta).T
    
    def gravity_in_body_frame(self, theta: np.ndarray) -> np.ndarray:
        """
        Gravity vector expressed in body frame: g(theta)
        
        Args:
            theta: [roll, pitch, yaw]
        
        Returns:
            g_body: 3D gravity vector in body frame
        """
        R_BW = self.rotation_matrix_world_to_body(theta)
        g_world = np.array([0., 0., -9.81])
        return R_BW @ g_world
    
    def skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """
        Skew-symmetric matrix for cross product: [v]_x
        Such that [v]_x * w = v x w
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def compute_contact_positions(self, q_j: np.ndarray, 
                                  base_orientation: np.ndarray) -> List[np.ndarray]:
        """
        Compute end-effector positions r_ei(q_j) relative to COM
        Using simplified forward kinematics
        
        Args:
            q_j: Joint positions (12,)
            base_orientation: [roll, pitch, yaw]
        
        Returns:
            List of 4 contact position vectors (3D each)
        """
        contact_positions = []
        
        # Simplified leg geometry (typical quadruped dimensions)
        hip_offset = 0.3  # m - distance from center to hip
        thigh_length = 0.25  # m
        shank_length = 0.25  # m
        
        # Leg ordering: LF, RF, LH, RH (Left-Front, Right-Front, Left-Hind, Right-Hind)
        hip_positions_body = [
            np.array([hip_offset, hip_offset, 0]),   # LF
            np.array([hip_offset, -hip_offset, 0]),  # RF
            np.array([-hip_offset, hip_offset, 0]),  # LH
            np.array([-hip_offset, -hip_offset, 0])  # RH
        ]
        
        for i in range(4):
            # Get joint angles for this leg
            q_hip, q_thigh, q_shank = q_j[i*3:(i+1)*3]
            
            # Simple 2D FK in sagittal plane (simplified)
            # In practice, this would be full 3D FK from URDF
            x = thigh_length * np.sin(q_thigh) + shank_length * np.sin(q_thigh + q_shank)
            z = -thigh_length * np.cos(q_thigh) - shank_length * np.cos(q_thigh + q_shank)
            
            # Foot position relative to hip (in leg plane)
            foot_hip = np.array([x, 0, z])
            
            # Rotate by hip abduction angle
            R_hip = Rotation.from_euler('z', q_hip).as_matrix()
            foot_hip_rotated = R_hip @ foot_hip
            
            # Add hip offset
            r_ei = hip_positions_body[i] + foot_hip_rotated
            
            contact_positions.append(r_ei)
        
        return contact_positions
    
    def dynamics(self, x: np.ndarray, u: np.ndarray, 
                contact_states: np.ndarray) -> np.ndarray:
        """
        Compute state derivative: x_dot = f(x, u)
        
        Implements Equation (4) from the paper:
        theta_dot = T(theta) * omega
        p_dot = R_WB(theta) * v
        omega_dot = I^-1 * (-omega x I*omega + sum(r_ei x lambda_ei))
        v_dot = g(theta) + (1/m) * sum(lambda_ei)
        q_j_dot = u_j
        
        Args:
            x: State vector (24,)
            u: Control vector (24,)
            contact_states: Binary array (4,) indicating which legs are in contact
        
        Returns:
            x_dot: State derivative (24,)
        """
        # Parse state
        theta = x[0:3]      # orientation
        p = x[3:6]          # position
        omega = x[6:9]      # angular velocity
        v = x[9:12]         # linear velocity
        q_j = x[12:24]      # joint positions
        
        # Parse control
        lambda_e = u[0:12].reshape(4, 3)  # contact forces (4 legs x 3D)
        u_j = u[12:24]                     # joint velocities
        
        # Zero out forces for legs not in contact
        for i in range(4):
            if contact_states[i] == 0:
                lambda_e[i] = 0.0
        
        # Compute contact positions
        r_contacts = self.compute_contact_positions(q_j, theta)
        
        # (a) Orientation kinematics: theta_dot = T(theta) * omega
        T = self.euler_rate_transform(theta)
        theta_dot = T @ omega
        
        # (b) Position kinematics: p_dot = R_WB(theta) * v
        R_WB = self.rotation_matrix_body_to_world(theta)
        p_dot = R_WB @ v
        
        # (c) Angular dynamics: omega_dot = I^-1 * (-omega x I*omega + sum(r_ei x lambda_ei))
        # Gyroscopic term
        omega_cross_Iomega = np.cross(omega, self.I @ omega)
        
        # Torque from contact forces
        net_torque = np.zeros(3)
        for i in range(4):
            if contact_states[i] == 1:
                net_torque += np.cross(r_contacts[i], lambda_e[i])
        
        omega_dot = self.I_inv @ (-omega_cross_Iomega + net_torque)
        
        # (d) Linear dynamics: v_dot = g(theta) + (1/m) * sum(lambda_ei)
        g_body = self.gravity_in_body_frame(theta)
        net_force = np.sum(lambda_e, axis=0)
        v_dot = g_body + net_force / self.m
        
        # (e) Joint kinematics: q_j_dot = u_j
        q_j_dot = u_j
        
        # Assemble state derivative
        x_dot = np.concatenate([theta_dot, p_dot, omega_dot, v_dot, q_j_dot])
        
        return x_dot
    
    def integrate_euler(self, x: np.ndarray, u: np.ndarray, 
                       contact_states: np.ndarray, dt: float) -> np.ndarray:
        """
        Simple Euler integration: x_next = x + dt * f(x, u)
        
        Args:
            x: Current state (24,)
            u: Control input (24,)
            contact_states: Binary contact states (4,)
            dt: Time step
        
        Returns:
            x_next: Next state (24,)
        """
        x_dot = self.dynamics(x, u, contact_states)
        x_next = x + dt * x_dot
        return x_next


class GaitSequenceGenerator:
    """
    Generates gait sequences based on kinematic leg utilities
    Implements Section III-B from the paper
    """
    
    def __init__(self, params: MPCParameters):
        self.params = params
        
        # Kinematic workspace parameters (ellipse half-axes)
        self.lambda_parallel = 0.4  # m - along forward direction
        self.lambda_perpendicular = 0.2  # m - lateral direction
        
        # Utility threshold for triggering swing
        self.utility_threshold = 0.3
        
        # Current contact states
        self.contact_states = np.ones(4)  # All legs in contact initially
        self.swing_timers = np.zeros(4)
        
    def compute_leg_utility(self, leg_idx: int, 
                           current_foot_pos: np.ndarray,
                           reference_pos: np.ndarray) -> float:
        """
        Compute kinematic leg utility based on Equation (7)
        
        u_i(t) = 1 - sqrt((r_parallel/lambda_parallel)^2 + (r_perp/lambda_perp)^2)
        
        Args:
            leg_idx: Leg index (0-3)
            current_foot_pos: Current foot position (3,)
            reference_pos: Reference foot position based on body motion (3,)
        
        Returns:
            utility: Value in [0, 1]
        """
        # Position error
        r_error = reference_pos - current_foot_pos
        
        # Project onto forward (x) and lateral (y) directions
        r_parallel = abs(r_error[0])
        r_perp = abs(r_error[1])
        
        # Compute utility (ellipse-based)
        utility = 1.0 - np.sqrt(
            (r_parallel / self.lambda_parallel)**2 + 
            (r_perp / self.lambda_perpendicular)**2
        )
        
        return np.clip(utility, 0.0, 1.0)
    
    def update_gait(self, utilities: np.ndarray, dt: float):
        """
        Update gait sequence based on utilities
        
        Rules:
        1. If utility < threshold, schedule swing
        2. Only swing if neighboring legs are in contact
        3. Prioritize legs with lowest utility
        
        Args:
            utilities: Leg utilities (4,)
            dt: Time step
        """
        # Update swing timers
        for i in range(4):
            if self.contact_states[i] == 0:  # Leg is swinging
                self.swing_timers[i] += dt
                if self.swing_timers[i] >= self.params.swing_duration:
                    # Swing complete, touch down
                    self.contact_states[i] = 1
                    self.swing_timers[i] = 0.0
        
        # Check if any leg needs to swing
        for i in np.argsort(utilities):  # Process lowest utility first
            if utilities[i] < self.utility_threshold and self.contact_states[i] == 1:
                # Check neighboring legs
                neighbors = self.get_neighbors(i)
                if all(self.contact_states[n] == 1 for n in neighbors):
                    # Safe to swing
                    self.contact_states[i] = 0
                    self.swing_timers[i] = 0.0
                    break  # Only lift one leg at a time for static stability
    
    def get_neighbors(self, leg_idx: int) -> List[int]:
        """
        Get neighboring leg indices
        Leg ordering: 0=LF, 1=RF, 2=LH, 3=RH
        
        Neighbors:
        LF (0): RF (1), LH (2)
        RF (1): LF (0), RH (3)
        LH (2): LF (0), RH (3)
        RH (3): RF (1), LH (2)
        """
        neighbor_map = {
            0: [1, 2],  # LF
            1: [0, 3],  # RF
            2: [0, 3],  # LH
            3: [1, 2]   # RH
        }
        return neighbor_map[leg_idx]


class MPCController:
    """
    Model Predictive Controller for whole-body motion planning
    Uses simplified gradient-based optimization (not full SLQ/DDP)
    """
    
    def __init__(self, dynamics: SingleRigidBodyDynamics, params: MPCParameters):
        self.dynamics = dynamics
        self.params = params
        self.gait_gen = GaitSequenceGenerator(params)
        
        # Optimization parameters
        self.max_iterations = 5  # Number of optimization iterations per MPC step
        self.learning_rate = 0.01
        
    def compute_cost(self, x: np.ndarray, u: np.ndarray, 
                    x_ref: np.ndarray, u_ref: np.ndarray) -> float:
        """
        Quadratic cost function from Equation (3)
        
        l(x, u) = 0.5 * (x - x_ref)^T Q (x - x_ref) + 0.5 * (u - u_ref)^T R (u - u_ref)
        """
        # State error
        x_error = x - x_ref
        
        # Unpack errors
        theta_err = x_error[0:3]
        p_err = x_error[3:6]
        omega_err = x_error[6:9]
        v_err = x_error[9:12]
        q_j_err = x_error[12:24]
        
        # State cost
        cost_state = (
            self.params.weight_orientation * np.sum(theta_err**2) +
            self.params.weight_position * np.sum(p_err**2) +
            self.params.weight_angular_velocity * np.sum(omega_err**2) +
            self.params.weight_linear_velocity * np.sum(v_err**2) +
            self.params.weight_joint_position * np.sum(q_j_err**2)
        )
        
        # Control error
        u_error = u - u_ref
        lambda_err = u_error[0:12]
        u_j_err = u_error[12:24]
        
        # Control cost
        cost_control = (
            self.params.weight_contact_force * np.sum(lambda_err**2) +
            self.params.weight_joint_velocity * np.sum(u_j_err**2)
        )
        
        return 0.5 * (cost_state + cost_control)
    
    def enforce_constraints(self, u: np.ndarray, contact_states: np.ndarray) -> np.ndarray:
        """
        Enforce physical constraints on control inputs
        
        1. Friction cone: Forces must be within friction cone
        2. Force limits: Maximum contact force
        3. Joint velocity limits
        4. Zero force for swing legs
        """
        u_constrained = u.copy()
        
        # Parse control
        lambda_e = u_constrained[0:12].reshape(4, 3)
        u_j = u_constrained[12:24]
        
        # For each leg
        for i in range(4):
            if contact_states[i] == 1:  # Stance leg
                force = lambda_e[i]
                
                # Normal force must be positive
                force[2] = max(0.0, force[2])
                
                # Friction cone constraint: |F_tangent| <= mu * F_normal
                f_tangent = np.linalg.norm(force[0:2])
                f_normal = force[2]
                
                if f_tangent > self.params.friction_coeff * f_normal:
                    # Scale down tangential forces
                    scale = self.params.friction_coeff * f_normal / (f_tangent + 1e-6)
                    force[0:2] *= scale
                
                # Limit total force magnitude
                force_mag = np.linalg.norm(force)
                if force_mag > self.params.max_contact_force:
                    force *= self.params.max_contact_force / force_mag
                
                lambda_e[i] = force
            else:  # Swing leg
                lambda_e[i] = 0.0
        
        # Joint velocity limits
        u_j = np.clip(u_j, -self.params.max_joint_velocity, self.params.max_joint_velocity)
        
        # Reassemble
        u_constrained[0:12] = lambda_e.flatten()
        u_constrained[12:24] = u_j
        
        return u_constrained
    
    def solve_mpc(self, x0: np.ndarray, x_ref_traj: np.ndarray, 
                 contact_schedule: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve MPC optimization problem (simplified gradient descent)
        
        In practice, this would use SLQ/DDP as in the paper, but for simplicity
        we use a basic gradient-based approach here.
        
        Args:
            x0: Initial state (24,)
            x_ref_traj: Reference trajectory (num_nodes, 24)
            contact_schedule: Contact states over horizon (num_nodes, 4)
        
        Returns:
            u_optimal: Optimal control sequence (num_nodes, 24)
            x_predicted: Predicted state trajectory (num_nodes, 24)
        """
        N = self.params.num_nodes
        dt = self.params.dt
        
        # Initialize control sequence (zero input)
        u_seq = np.zeros((N, 24))
        
        # Reference control (regularization)
        u_ref = np.zeros(24)
        
        # Optimization loop (simplified)
        for iteration in range(self.max_iterations):
            # Forward pass: simulate with current control
            x_traj = np.zeros((N, 24))
            x_traj[0] = x0
            
            total_cost = 0.0
            
            for k in range(N - 1):
                # Enforce constraints
                u_seq[k] = self.enforce_constraints(u_seq[k], contact_schedule[k])
                
                # Compute cost
                total_cost += self.compute_cost(x_traj[k], u_seq[k], 
                                               x_ref_traj[k], u_ref)
                
                # Integrate dynamics
                x_traj[k+1] = self.dynamics.integrate_euler(
                    x_traj[k], u_seq[k], contact_schedule[k], dt
                )
            
            # Simple gradient update (in practice, use SLQ/DDP)
            # This is a placeholder - real implementation needs proper gradient computation
            for k in range(N - 1):
                # Heuristic control update based on state error
                state_error = x_ref_traj[k+1] - x_traj[k+1]
                
                # Simple proportional control update
                lambda_update = state_error[9:12] * self.params.robot_mass * 0.1  # Force from velocity error
                u_j_update = state_error[12:24] * 0.5  # Joint velocity from position error
                
                # Update controls
                u_seq[k][0:3] += lambda_update * self.learning_rate  # LF leg
                u_seq[k][12:24] += u_j_update * self.learning_rate
        
        return u_seq, x_traj


class RobotSimulation:
    """
    PyBullet simulation environment for the wheeled-legged robot
    """
    
    def __init__(self, urdf_path: str, params: MPCParameters, use_gui: bool = True):
        self.params = params
        self.urdf_path = urdf_path
        
        # Connect to PyBullet
        if use_gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        try:
            self.robot_id = p.loadURDF(urdf_path, start_pos, start_orientation)
        except:
            print(f"Could not load URDF from {urdf_path}")
            print("Creating simple quadruped robot...")
            self.robot_id = self.create_simple_quadruped(start_pos, start_orientation)
        
        # Get robot info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))
        
        # Setup robot
        self.setup_robot()
        
        print(f"Robot loaded with {self.num_joints} joints")
    
    def create_simple_quadruped(self, pos, orn):
        """Create a simple quadruped robot programmatically with articulated legs"""
        # Link dimensions
        body_size = [0.3, 0.15, 0.08]   # halfExtents for body box
        hip_length = 0.06
        thigh_length = 0.22
        shank_length = 0.22
        foot_radius = 0.03

        # Masses
        body_mass = max(0.01, self.params.robot_mass * 0.6)  # guard against zero
        leg_mass = max(0.001, self.params.robot_mass * 0.1)
        link_mass = leg_mass / 3.0  # three main moving links per leg

        # Collision & visual shapes (use box for thigh/shank to avoid capsule-axis confusion)
        body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_size)
        hip_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hip_length/2, 0.03, 0.03])
        thigh_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, thigh_length/2])
        shank_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, shank_length/2])
        foot_col = p.createCollisionShape(p.GEOM_SPHERE, radius=foot_radius)

        body_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=body_size, rgbaColor=[0.2, 0.2, 0.8, 1])
        hip_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hip_length/2, 0.03, 0.03], rgbaColor=[0.8,0.2,0.2,1])
        thigh_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,thigh_length/2], rgbaColor=[0.2,0.8,0.2,1])
        shank_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,shank_length/2], rgbaColor=[0.2,0.8,0.2,1])
        foot_vis = p.createVisualShape(p.GEOM_SPHERE, radius=foot_radius, rgbaColor=[0.1,0.1,0.1,1])

        # Leg attachment positions relative to body center (in body frame)
        # order: LF, RF, LH, RH
        x_off = body_size[0] * 0.9
        y_off = body_size[1] * 0.95
        leg_positions = [
            [ x_off,  y_off, -body_size[2]],  # Left Front
            [ x_off, -y_off, -body_size[2]],  # Right Front
            [-x_off,  y_off, -body_size[2]],  # Left Hind
            [-x_off, -y_off, -body_size[2]],  # Right Hind
        ]

        # lists for createMultiBody
        linkMasses = []
        linkCollisionShapes = []
        linkVisualShapes = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []

        # We will create for each leg: hip_link (revolute around Z),
        # thigh_link (revolute around Y), shank_link (revolute around Y), foot (fixed)
        #
        # Important: linkPositions are joint frame offsets *from parent link frame to child joint frame*.
        # For hip (parent is base): position is leg attachment point in body frame.
        # For thigh (parent is hip): place along -z so thigh is below the hip joint.
        # For shank (parent is thigh): place further along -z.
        # For foot (parent is shank): small offset below shank.

        current_link_index = 0  # used for parent indices

        for leg_idx in range(4):
            attach = leg_positions[leg_idx]

            # Hip link
            linkMasses.append(link_mass)
            linkCollisionShapes.append(hip_col)
            linkVisualShapes.append(hip_vis)
            # hip joint location relative to base (body)
            linkPositions.append(attach)                    # child joint frame relative to base
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            linkParentIndices.append(0)                     # parent is base (index 0)
            linkJointTypes.append(p.JOINT_REVOLUTE)
            linkJointAxis.append([0, 0, 1])                 # abduction/adduction around body Z
            hip_index = current_link_index
            current_link_index += 1

            # Thigh link (parent = hip)
            linkMasses.append(link_mass)
            linkCollisionShapes.append(thigh_col)
            linkVisualShapes.append(thigh_vis)
            # place thigh center below hip joint by half thigh length + small offset
            thigh_offset_z = -(hip_length/2.0 + thigh_length/2.0)
            linkPositions.append([0, 0, thigh_offset_z])   # relative to hip frame
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            linkParentIndices.append(hip_index)
            linkJointTypes.append(p.JOINT_REVOLUTE)
            linkJointAxis.append([0, 1, 0])                 # hip flexion around local Y
            thigh_index = current_link_index
            current_link_index += 1

            # Shank link (parent = thigh)
            linkMasses.append(link_mass)
            linkCollisionShapes.append(shank_col)
            linkVisualShapes.append(shank_vis)
            shank_offset_z = -(thigh_length/2.0 + shank_length/2.0)
            linkPositions.append([0, 0, shank_offset_z])   # relative to thigh frame
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            linkParentIndices.append(thigh_index)
            linkJointTypes.append(p.JOINT_REVOLUTE)
            linkJointAxis.append([0, 1, 0])                 # knee around local Y
            shank_index = current_link_index
            current_link_index += 1

            # Foot (fixed) - small mass
            linkMasses.append(link_mass * 0.1)
            linkCollisionShapes.append(foot_col)
            linkVisualShapes.append(foot_vis)
            foot_offset_z = -(shank_length/2.0 + foot_radius)
            linkPositions.append([0, 0, foot_offset_z])    # relative to shank frame
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            linkParentIndices.append(shank_index)
            linkJointTypes.append(p.JOINT_FIXED)
            linkJointAxis.append([0, 0, 0])
            current_link_index += 1

        robot_id = p.createMultiBody(
            baseMass=body_mass,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=pos,
            baseOrientation=orn,
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapes,
            linkVisualShapeIndices=linkVisualShapes,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis,
            physicsClientId=p._physics_client_id if hasattr(p, "_physics_client_id") else 0
        )

        # Optional: set joint damping, friction, motors off by default
        for j in range(p.getNumJoints(robot_id)):
            p.changeDynamics(robot_id, j, lateralFriction=1.0, restitution=0.0)
            # Disable motor control so joints are free (user can enable POSITION_CONTROL later)
            p.setJointMotorControl2(robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)

        return robot_id
    
    def setup_robot(self):
        """Configure robot joints and parameters"""
        
        # Filter to get only revolute/prismatic joints (exclude fixed joints)
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            
            # Only control revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
        
        print(f"  ℹ Found {len(self.joint_indices)} controllable joints")
        
        # Enable force/torque sensors on controllable joints
        for joint_idx in self.joint_indices:
            p.enableJointForceTorqueSensor(self.robot_id, joint_idx, True)
        
        # Set joint damping
        for joint_idx in self.joint_indices:
            p.changeDynamics(self.robot_id, joint_idx, 
                           linearDamping=0.0, angularDamping=0.0,
                           jointDamping=0.5)
        
        # Set initial joint positions to nominal standing configuration
        if len(self.joint_indices) >= 12:  # Quadruped with 3 joints per leg
            # Nominal configuration: legs slightly bent
            for leg in range(4):
                hip_idx = leg * 3
                thigh_idx = leg * 3 + 1
                shank_idx = leg * 3 + 2
                
                if hip_idx < len(self.joint_indices):
                    p.resetJointState(self.robot_id, self.joint_indices[hip_idx], 0.0)
                if thigh_idx < len(self.joint_indices):
                    p.resetJointState(self.robot_id, self.joint_indices[thigh_idx], 0.6)
                if shank_idx < len(self.joint_indices):
                    p.resetJointState(self.robot_id, self.joint_indices[shank_idx], -1.2)
    
    def get_state(self) -> np.ndarray:
        """
        Get current robot state in SRBD format
        
        Returns:
            x: State vector (24,) = [theta, p, omega, v, q_j]
        """
        # Get base state
        base_pos, base_orn_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_vel_linear, base_vel_angular = p.getBaseVelocity(self.robot_id)
        
        # Convert quaternion to Euler angles
        base_orn_euler = p.getEulerFromQuaternion(base_orn_quat)
        
        # Get joint states (only from controllable joints)
        if len(self.joint_indices) > 0:
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            joint_positions = [state[0] for state in joint_states]
        else:
            joint_positions = []
        
        # Pad or truncate to 12 joints
        if len(joint_positions) < 12:
            joint_positions.extend([0.0] * (12 - len(joint_positions)))
        else:
            joint_positions = joint_positions[:12]
        
        # Assemble state (convert world frame to body frame for velocities)
        R_WB = Rotation.from_euler('xyz', base_orn_euler).as_matrix()
        R_BW = R_WB.T
        
        v_body = R_BW @ np.array(base_vel_linear)
        omega_body = R_BW @ np.array(base_vel_angular)
        
        x = np.concatenate([
            base_orn_euler,      # theta (3)
            base_pos,            # p (3)
            omega_body,          # omega (3)
            v_body,              # v (3)
            joint_positions      # q_j (12)
        ])
        
        return x
    
    def apply_control(self, u: np.ndarray):
        """
        Apply control inputs to robot
        
        Args:
            u: Control vector (24,) = [lambda_e, u_j]
        
        Note: In simulation, we apply joint velocities directly.
        Contact forces are handled by physics engine.
        """
        # Parse control
        lambda_e = u[0:12].reshape(4, 3)  # Contact forces (not directly applied)
        u_j = u[12:24]  # Joint velocities
        
        # Apply joint velocity control to available joints
        num_controllable = min(len(self.joint_indices), 12)
        for i in range(num_controllable):
            joint_idx = self.joint_indices[i]
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=u_j[i],
                force=100.0  # Maximum force
            )
    
    def step(self):
        """Step the simulation"""
        p.stepSimulation()
    
    def disconnect(self):
        """Disconnect from PyBullet"""
        p.disconnect()


def generate_reference_trajectory(params: MPCParameters, 
                                  current_state: np.ndarray,
                                  target_velocity: np.ndarray) -> np.ndarray:
    """
    Generate reference trajectory for MPC
    
    Args:
        params: MPC parameters
        current_state: Current robot state (24,)
        target_velocity: Desired [vx, vy, vz, wx, wy, wz] (6,)
    
    Returns:
        x_ref: Reference trajectory (num_nodes, 24)
    """
    N = params.num_nodes
    dt = params.dt
    
    x_ref = np.zeros((N, 24))
    
    # Current state as starting point
    theta_0 = current_state[0:3]
    p_0 = current_state[3:6]
    omega_ref = target_velocity[3:6]  # Desired angular velocity
    v_ref = target_velocity[0:3]      # Desired linear velocity
    q_j_nominal = current_state[12:24]  # Keep current joint configuration
    
    # Generate trajectory by integrating desired velocities
    for k in range(N):
        t = k * dt
        
        # Integrate orientation (simplified - assumes small angles)
        theta_k = theta_0 + omega_ref * t
        
        # Integrate position (in world frame)
        R_WB = Rotation.from_euler('xyz', theta_0).as_matrix()
        v_world = R_WB @ v_ref
        p_k = p_0 + v_world * t
        
        # Keep constant nominal height
        p_k[2] = 0.45  # Desired height above ground
        
        # Assemble reference state
        x_ref[k] = np.concatenate([
            theta_k,
            p_k,
            omega_ref,
            v_ref,
            q_j_nominal
        ])
    
    return x_ref


def main():
    """
    Main driver code for MPC-based walking controller
    """
    
    print("="*60)
    print("Whole-Body MPC for Wheeled-Legged Robots")
    print("Based on: Bjelonic et al. 2021")
    print("="*60)
    
    # ==========================================
    # CONFIGURATION - MODIFY PATHS HERE
    # ==========================================
    
    # Option 1: Provide your own URDF path
    ROBOT_URDF_PATH = "/home/poison-arrow/EECE5552/MPC_Gait/anymal_c_simple_description-master/urdf/anymal.urdf"  # <-- CHANGE THIS
    
    # Option 2: Use built-in simple quadruped (if URDF not found)
    USE_SIMPLE_ROBOT = False  # Set to False if you have a URDF
    
    # Simulation parameters
    USE_GUI = True  # Set to False for headless simulation
    SIMULATION_TIME = 30.0  # seconds
    
    # Target velocity command [vx, vy, vz, wx, wy, wz]
    # This is the desired motion you want the robot to execute
    TARGET_VELOCITY = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.5 m/s forward
    
    # ==========================================
    # SETUP
    # ==========================================
    
    print("\n[1/5] Loading parameters...")
    
    # Initialize parameters (you can modify these)
    params = MPCParameters(
        # Robot physical parameters
        robot_mass=30.0,
        robot_inertia=np.diag([0.5, 1.0, 1.0]),
        
        # MPC horizon
        horizon_length=0.8,
        num_nodes=20,
        control_freq=50.0,
        
        # Cost weights (tune these for different behaviors)
        weight_position=100.0,
        weight_orientation=50.0,
        weight_linear_velocity=10.0,
        weight_angular_velocity=5.0,
        weight_joint_position=1.0,
        weight_contact_force=0.01,
        weight_joint_velocity=0.1,
        
        # Physical constraints
        friction_coeff=0.7,
        max_joint_velocity=10.0,
        max_contact_force=500.0,
        
        # Gait parameters
        swing_height=0.1,
        swing_duration=0.3
    )
    
    print(f"  ✓ MPC horizon: {params.horizon_length}s with {params.num_nodes} nodes")
    print(f"  ✓ Control frequency: {params.control_freq} Hz")
    print(f"  ✓ Robot mass: {params.robot_mass} kg")
    
    # ==========================================
    # INITIALIZE DYNAMICS MODEL
    # ==========================================
    
    print("\n[2/5] Initializing reduced-order dynamics model...")
    dynamics = SingleRigidBodyDynamics(params)
    print(f"  ✓ State dimension: {dynamics.state_dim}")
    print(f"  ✓ Control dimension: {dynamics.control_dim}")
    print(f"  ✓ Model: Single Rigid Body Dynamics (SRBD)")
    
    # ==========================================
    # INITIALIZE MPC CONTROLLER
    # ==========================================
    
    print("\n[3/5] Initializing MPC controller...")
    controller = MPCController(dynamics, params)
    print(f"  ✓ MPC iterations: {controller.max_iterations}")
    print(f"  ✓ Gait generator: Kinematic utility-based")
    
    # ==========================================
    # SETUP SIMULATION
    # ==========================================
    
    print("\n[4/5] Setting up PyBullet simulation...")
    
    if USE_SIMPLE_ROBOT:
        print("  ℹ Using simple programmatic quadruped")
        ROBOT_URDF_PATH = "simple_quadruped"  # Placeholder
    else:
        print(f"  ℹ Loading URDF from: {ROBOT_URDF_PATH}")
    
    try:
        sim = RobotSimulation(ROBOT_URDF_PATH, params, use_gui=USE_GUI)
        print(f"  ✓ Robot loaded successfully")
        print(f"  ✓ Number of joints: {sim.num_joints}")
    except Exception as e:
        print(f"  ✗ Error loading robot: {e}")
        print("  ℹ Falling back to simple robot...")
        sim = RobotSimulation("simple_quadruped", params, use_gui=USE_GUI)
    
    # ==========================================
    # WAIT FOR USER CONFIRMATION
    # ==========================================
    
    print("\n" + "="*60)
    print("ROBOT LOADED - READY TO START")
    print("="*60)
    print(f"\n📋 Configuration Summary:")
    print(f"  • Robot mass: {params.robot_mass} kg")
    print(f"  • Controllable joints: {len(sim.joint_indices)}")
    print(f"  • MPC horizon: {params.horizon_length}s")
    print(f"  • Control frequency: {params.control_freq} Hz")
    print(f"  • Target velocity: vx={TARGET_VELOCITY[0]:.2f} m/s, "
          f"vy={TARGET_VELOCITY[1]:.2f} m/s, wz={TARGET_VELOCITY[5]:.2f} rad/s")
    
    print("\n🤖 The quadruped robot is now visible in the PyBullet GUI.")
    print("   Check that the robot is:")
    print("   ✓ Standing in a reasonable configuration")
    print("   ✓ Not penetrating the ground")
    print("   ✓ Properly balanced")
    
    print("\n" + "="*60)
    input("Press ENTER to start the MPC controller... ")
    print("="*60)
    
    # ==========================================
    # MAIN CONTROL LOOP
    # ==========================================
    
    print("\n[5/5] Starting control loop...")
    print("\n" + "="*60)
    print("SIMULATION RUNNING")
    print("="*60)
    
    dt_control = 1.0 / params.control_freq
    num_steps = int(SIMULATION_TIME / dt_control)
    
    # Data logging
    state_history = []
    control_history = []
    cost_history = []
    
    # Initial contact states (all legs in contact)
    current_contact_states = np.ones(4)
    
    try:
        for step in range(num_steps):
            t = step * dt_control
            
            # Get current state from simulation
            x_current = sim.get_state()
            state_history.append(x_current.copy())
            
            # Generate reference trajectory
            x_ref_traj = generate_reference_trajectory(
                params, x_current, TARGET_VELOCITY
            )
            
            # Update gait sequence (simplified - using fixed trotting gait for now)
            if step % 50 == 0:  # Update gait every 1 second
                # Compute leg utilities (simplified)
                utilities = np.random.rand(4) * 0.5 + 0.5  # Placeholder
                controller.gait_gen.update_gait(utilities, dt_control * 50)
                current_contact_states = controller.gait_gen.contact_states.copy()
            
            # Create contact schedule for MPC horizon
            contact_schedule = np.tile(current_contact_states, (params.num_nodes, 1))
            
            # Solve MPC optimization
            u_optimal, x_predicted = controller.solve_mpc(
                x_current, x_ref_traj, contact_schedule
            )
            
            # Apply first control input
            u_apply = u_optimal[0]
            control_history.append(u_apply.copy())
            sim.apply_control(u_apply)
            
            # Compute cost for logging
            cost = controller.compute_cost(x_current, u_apply, 
                                          x_ref_traj[0], np.zeros(24))
            cost_history.append(cost)
            
            # Step simulation
            sim.step()
            
            # Print status every second
            if step % int(params.control_freq) == 0:
                pos = x_current[3:6]
                vel = x_current[9:12]
                contact_str = ''.join(['█' if c else '░' for c in current_contact_states])
                
                print(f"t={t:6.2f}s | "
                      f"pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                      f"vel=[{vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}] | "
                      f"contacts={contact_str} | "
                      f"cost={cost:8.2f}")
            
            # Sleep to maintain real-time (if GUI)
            if USE_GUI:
                time.sleep(dt_control)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("SIMULATION INTERRUPTED BY USER")
        print("="*60)
    
    finally:
        # ==========================================
        # CLEANUP AND RESULTS
        # ==========================================
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
        # Statistics
        if len(state_history) > 0:
            state_history = np.array(state_history)
            control_history = np.array(control_history)
            cost_history = np.array(cost_history)
            
            print(f"\nSimulation Statistics:")
            print(f"  Total steps: {len(state_history)}")
            print(f"  Average cost: {np.mean(cost_history):.2f}")
            print(f"  Final position: [{state_history[-1][3]:.2f}, "
                  f"{state_history[-1][4]:.2f}, {state_history[-1][5]:.2f}]")
            print(f"  Distance traveled: {np.linalg.norm(state_history[-1][3:5] - state_history[0][3:5]):.2f} m")
        else:
            print(f"\nSimulation Statistics:")
            print(f"  No data collected (simulation ended immediately)")
        
        # Disconnect simulation
        sim.disconnect()
        print("\n✓ Simulation environment closed")
        print("="*60)


if __name__ == "__main__":
    """
    Entry point for the MPC walking controller
    
    To use with your own robot:
    1. Set ROBOT_URDF_PATH to your URDF file path
    2. Set USE_SIMPLE_ROBOT = False
    3. Adjust MPCParameters if needed
    4. Run: python mpc_walking_controller.py
    """
    
    # Check dependencies
    try:
        import pybullet
        import scipy
        print("✓ All dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install pybullet numpy scipy")
        exit(1)
    
    # Run main controller
    main()