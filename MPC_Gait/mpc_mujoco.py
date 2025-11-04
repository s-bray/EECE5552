"""
Whole-Body Model Predictive Control for Wheeled-Legged Robots
Based on: Bjelonic et al. "Whole-Body MPC and Online Gait Sequence Generation"

MuJoCo Implementation
"""

import numpy as np
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation
import time
import os


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


class GaitSequenceGenerator:
    """Generates gait sequences based on kinematic leg utilities"""
    
    def __init__(self, params: MPCParameters):
        self.params = params
        self.lambda_parallel = 0.4
        self.lambda_perpendicular = 0.2
        self.utility_threshold = 0.3
        self.contact_states = np.ones(4)
        self.swing_timers = np.zeros(4)
        
    def compute_leg_utility(self, leg_idx: int, 
                           current_foot_pos: np.ndarray,
                           reference_pos: np.ndarray) -> float:
        """Compute kinematic leg utility"""
        r_error = reference_pos - current_foot_pos
        r_parallel = abs(r_error[0])
        r_perp = abs(r_error[1])
        
        utility = 1.0 - np.sqrt(
            (r_parallel / self.lambda_parallel)**2 + 
            (r_perp / self.lambda_perpendicular)**2
        )
        
        return np.clip(utility, 0.0, 1.0)
    
    def update_gait(self, utilities: np.ndarray, dt: float):
        """Update gait sequence based on utilities"""
        for i in range(4):
            if self.contact_states[i] == 0:
                self.swing_timers[i] += dt
                if self.swing_timers[i] >= self.params.swing_duration:
                    self.contact_states[i] = 1
                    self.swing_timers[i] = 0.0
        
        for i in np.argsort(utilities):
            if utilities[i] < self.utility_threshold and self.contact_states[i] == 1:
                neighbors = self.get_neighbors(i)
                if all(self.contact_states[n] == 1 for n in neighbors):
                    self.contact_states[i] = 0
                    self.swing_timers[i] = 0.0
                    break
    
    def get_neighbors(self, leg_idx: int) -> List[int]:
        """Get neighboring leg indices"""
        neighbor_map = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        return neighbor_map[leg_idx]


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


class RobotSimulation:
    """MuJoCo simulation environment for the wheeled-legged robot"""
    
    def __init__(self, xml_path: str, params: MPCParameters, use_gui: bool = True):
        self.params = params
        self.xml_path = xml_path
        self.use_gui = use_gui
        
        # Load model
        if os.path.exists(xml_path):
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            print(f"XML file not found: {xml_path}")
            print("Creating simple quadruped model...")
            xml_content = self.create_simple_quadruped_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_content)
        
        self.data = mujoco.MjData(self.model)
        
        # Find actuated joints
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.model.njnt):
            joint_type = self.model.jnt_type[i]
            if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                self.joint_indices.append(i)
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                self.joint_names.append(joint_name if joint_name else f"joint_{i}")
        
        print(f"  ℹ Found {len(self.joint_indices)} controllable joints")
        
        # Initialize viewer if GUI enabled
        self.viewer = None
        if use_gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Set initial configuration
        self.set_initial_configuration()
        mujoco.mj_forward(self.model, self.data)
    
    def create_simple_quadruped_xml(self) -> str:
        """Create a simple quadruped robot XML"""
        xml = """
        <mujoco model="simple_quadruped">
            <compiler angle="radian" coordinate="local"/>
            
            <option timestep="0.002" gravity="0 0 -9.81"/>
            
            <default>
                <geom rgba="0.8 0.6 0.4 1" friction="1 0.005 0.0001"/>
                <joint damping="0.5" armature="0.01"/>
            </default>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                         rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                          reflectance=".2"/>
            </asset>
            
            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom name="floor" size="5 5 .05" type="plane" material="grid"/>
                
                <body name="base" pos="0 0 0.5">
                    <freejoint/>
                    <geom name="torso" type="box" size="0.3 0.15 0.08" mass="18.0" rgba="0.2 0.2 0.8 1"/>
                    <geom name="torso_imu" type="sphere" size="0.02" pos="0 0 0" mass="0.1" rgba="1 0 0 1"/>
                    
                    <!-- Front Left Leg -->
                    <body name="fl_hip" pos="0.27 0.17 -0.08">
                        <joint name="fl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                        
                        <body name="fl_thigh" pos="0 0.05 0">
                            <joint name="fl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                            
                            <body name="fl_shank" pos="0 0 -0.22">
                                <joint name="fl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                                <geom name="fl_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Front Right Leg -->
                    <body name="fr_hip" pos="0.27 -0.17 -0.08">
                        <joint name="fr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                        
                        <body name="fr_thigh" pos="0 -0.05 0">
                            <joint name="fr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                            
                            <body name="fr_shank" pos="0 0 -0.22">
                                <joint name="fr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                                <geom name="fr_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Hind Left Leg -->
                    <body name="hl_hip" pos="-0.27 0.17 -0.08">
                        <joint name="hl_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                        
                        <body name="hl_thigh" pos="0 0.05 0">
                            <joint name="hl_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                            
                            <body name="hl_shank" pos="0 0 -0.22">
                                <joint name="hl_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                                <geom name="hl_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Hind Right Leg -->
                    <body name="hr_hip" pos="-0.27 -0.17 -0.08">
                        <joint name="hr_hip_joint" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="1.0"/>
                        <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.03" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                        
                        <body name="hr_thigh" pos="0 -0.05 0">
                            <joint name="hr_thigh_joint" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.0"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.025" mass="1.5" rgba="0.2 0.8 0.2 1"/>
                            
                            <body name="hr_shank" pos="0 0 -0.22">
                                <joint name="hr_shank_joint" type="hinge" axis="0 1 0" range="-2.5 -0.2" damping="1.0"/>
                                <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.02" mass="1.0" rgba="0.2 0.8 0.2 1"/>
                                <geom name="hr_foot" type="sphere" pos="0 0 -0.22" size="0.03" rgba="0.1 0.1 0.1 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            
            <actuator>
                <motor name="fl_hip_motor" joint="fl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="fl_thigh_motor" joint="fl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="fl_shank_motor" joint="fl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                
                <motor name="fr_hip_motor" joint="fr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="fr_thigh_motor" joint="fr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="fr_shank_motor" joint="fr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                
                <motor name="hl_hip_motor" joint="hl_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="hl_thigh_motor" joint="hl_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="hl_shank_motor" joint="hl_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                
                <motor name="hr_hip_motor" joint="hr_hip_joint" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
                <motor name="hr_thigh_motor" joint="hr_thigh_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
                <motor name="hr_shank_motor" joint="hr_shank_joint" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
            </actuator>
        </mujoco>
        """
        return xml
    
    def set_initial_configuration(self):
        """Set initial joint positions to nominal standing configuration"""
        # Standing pose: legs slightly bent
        nominal_config = [
            0.0, 0.6, -1.2,  # FL: hip, thigh, shank
            0.0, 0.6, -1.2,  # FR
            0.0, 0.6, -1.2,  # HL
            0.0, 0.6, -1.2,  # HR
        ]
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            if i < len(nominal_config):
                qpos_addr = self.model.jnt_qposadr[joint_idx]
                self.data.qpos[qpos_addr] = nominal_config[i]
    
    def get_state(self) -> np.ndarray:
        """Get current robot state in SRBD format"""
        # Base position and quaternion (MuJoCo qpos quaternion ordering: w, x, y, z)
        base_pos = self.data.qpos[0:3].copy()
        base_quat_mj = self.data.qpos[3:7].copy()  # [w, x, y, z]

        # Reorder to [x, y, z, w] for scipy
        quat_scipy = np.array([base_quat_mj[1], base_quat_mj[2], base_quat_mj[3], base_quat_mj[0]])
        r = Rotation.from_quat(quat_scipy)
        base_orn_euler = r.as_euler('xyz', degrees=False)

        # Base velocities -- note: MuJoCo qvel ordering for a freejoint is linear then angular often in world frame.
        # We'll extract world linear/ang vel from qvel (MuJoCo: qvel[0:3]=linear, qvel[3:6]=angular)
        base_vel_linear_world = self.data.qvel[0:3].copy()
        base_vel_angular_world = self.data.qvel[3:6].copy()

        # Convert velocities to body frame
        R_WB = r.as_matrix()
        R_BW = R_WB.T

        v_body = R_BW @ base_vel_linear_world
        omega_body = R_BW @ base_vel_angular_world

        # Joint positions (skip free joint, get actuated joints)
        joint_positions = []
        for joint_idx in self.joint_indices[:12]:
            qpos_addr = self.model.jnt_qposadr[joint_idx]
            joint_positions.append(self.data.qpos[qpos_addr])

        while len(joint_positions) < 12:
            joint_positions.append(0.0)
        joint_positions = joint_positions[:12]

        x = np.concatenate([
            base_orn_euler,      # theta (3)
            base_pos,            # p (3)
            omega_body,          # omega (3)
            v_body,              # v (3)
            joint_positions      # q_j (12)
        ])

        return x

    
    def apply_control(self, u: np.ndarray):
        """Apply control inputs to robot (safer mapping + clamping)"""
        # Only use joint velocity part for now; ignore SRBD lambda_e until it's hooked into MuJoCo
        u_j = u[12:24]

        # Conservative PD gains to avoid violent torques
        kp = 5.0
        kd = 0.5

        # Build mapping from actuator index -> joint index if possible.
        # Many simple XMLs have actuator order matching joint order; still we'll be defensive.
        num_actuators = min(self.model.nu, 12)
        for act_idx in range(num_actuators):
            # try to get joint index for this actuator; fallback to joint_indices[act_idx]
            try:
                # actuator_trnid gives tuple (obj_type, obj_id)
                trnid = self.model.actuator_trnid[act_idx]  # shape (3,) in mujoco python binding, type/id pair
                # actuator_trnid layout can be (0, joint_id, dof_index) — we'll try second element
                joint_obj_id = int(trnid[1])
                joint_idx = joint_obj_id
            except Exception:
                joint_idx = self.joint_indices[act_idx] if act_idx < len(self.joint_indices) else None

            if joint_idx is None:
                continue

            # current joint velocity address
            qvel_addr = int(self.model.jnt_dofadr[joint_idx])
            current_vel = float(self.data.qvel[qvel_addr])

            desired_vel = float(u_j[act_idx]) if act_idx < len(u_j) else 0.0
            vel_error = desired_vel - current_vel

            # raw control value
            ctrl_val = kp * vel_error - kd * current_vel

            # clamp to actuator ctrlrange if available
            try:
                lo, hi = self.model.actuator_ctrlrange[act_idx]
                ctrl_clamped = float(np.clip(ctrl_val, lo - 1e-6, hi + 1e-6))
            except Exception:
                ctrl_clamped = float(np.clip(ctrl_val, -100.0, 100.0))

            self.data.ctrl[act_idx] = ctrl_clamped

    
    def step(self):
        """Step the simulation"""
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
    
    def close(self):
        """Close the simulation"""
        if self.viewer is not None:
            self.viewer.close()


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


def main():
    """Main driver code for MPC-based walking controller"""
    
    print("="*60)
    print("Whole-Body MPC for Wheeled-Legged Robots - MuJoCo")
    print("Based on: Bjelonic et al. 2021")
    print("="*60)
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    # Option 1: Provide your own MuJoCo XML path
    ROBOT_XML_PATH = "/home/poison-arrow/MPC_Gait/anymal_simplified.xml"
    
    # Option 2: Use built-in simple quadruped
    USE_SIMPLE_ROBOT = False  # Set to False if you have an XML
    
    # Simulation parameters
    USE_GUI = True
    SIMULATION_TIME = 30.0  # seconds
    
    # Target velocity [vx, vy, vz, wx, wy, wz]
    TARGET_VELOCITY = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.5 m/s forward
    
    # ==========================================
    # SETUP
    # ==========================================
    
    print("\n[1/5] Loading parameters...")
    
    params = MPCParameters(
        robot_mass=30.0,
        robot_inertia=np.diag([0.5, 1.0, 1.0]),
        horizon_length=0.8,
        num_nodes=20,
        control_freq=50.0,
        weight_position=100.0,
        weight_orientation=50.0,
        weight_linear_velocity=10.0,
        weight_angular_velocity=5.0,
        weight_joint_position=1.0,
        weight_contact_force=0.01,
        weight_joint_velocity=0.1,
        friction_coeff=0.7,
        max_joint_velocity=10.0,
        max_contact_force=500.0,
        swing_height=0.1,
        swing_duration=0.3
    )
    
    print(f"  ✓ MPC horizon: {params.horizon_length}s with {params.num_nodes} nodes")
    print(f"  ✓ Control frequency: {params.control_freq} Hz")
    print(f"  ✓ Robot mass: {params.robot_mass} kg")
    
    # ==========================================
    # INITIALIZE DYNAMICS
    # ==========================================
    
    print("\n[2/5] Initializing reduced-order dynamics model...")
    dynamics = SingleRigidBodyDynamics(params)
    print(f"  ✓ State dimension: {dynamics.state_dim}")
    print(f"  ✓ Control dimension: {dynamics.control_dim}")
    print(f"  ✓ Model: Single Rigid Body Dynamics (SRBD)")
    
    # ==========================================
    # INITIALIZE MPC
    # ==========================================
    
    print("\n[3/5] Initializing MPC controller...")
    controller = MPCController(dynamics, params)
    print(f"  ✓ MPC iterations: {controller.max_iterations}")
    print(f"  ✓ Gait generator: Kinematic utility-based")
    
    # ==========================================
    # SETUP SIMULATION
    # ==========================================
    
    print("\n[4/5] Setting up MuJoCo simulation...")
    
    if USE_SIMPLE_ROBOT:
        print("  ℹ Using simple programmatic quadruped")
        ROBOT_XML_PATH = "simple_quadruped.xml"
    else:
        print(f"  ℹ Loading XML from: {ROBOT_XML_PATH}")
    
    try:
        sim = RobotSimulation(ROBOT_XML_PATH, params, use_gui=USE_GUI)
        print(f"  ✓ Robot loaded successfully")
        print(f"  ✓ Number of joints: {len(sim.joint_indices)}")
    except Exception as e:
        print(f"  ✗ Error loading robot: {e}")
        print("  ℹ Falling back to simple robot...")
        sim = RobotSimulation("fallback.xml", params, use_gui=USE_GUI)
    
    # ==========================================
    # USER CONFIRMATION
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
    
    print("\n🤖 The quadruped robot is now visible in the MuJoCo viewer.")
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
    
    # Initial contact states
    current_contact_states = np.ones(4)
    
    try:
        for step in range(num_steps):
            t = step * dt_control
            
            # Get current state
            x_current = sim.get_state()
            state_history.append(x_current.copy())
            
            # Generate reference trajectory
            x_ref_traj = generate_reference_trajectory(
                params, x_current, TARGET_VELOCITY
            )
            
            # Update gait sequence
            if step % 50 == 0:
                utilities = np.random.rand(4) * 0.5 + 0.5
                controller.gait_gen.update_gait(utilities, dt_control * 50)
                current_contact_states = controller.gait_gen.contact_states.copy()
                print("qpos (base+quat):", sim.data.qpos[0:7])
                print("qvel (first 6):", sim.data.qvel[0:6])
                print("data.ctrl[:12]:", sim.data.ctrl[:12])
                # contact info
                print("ncon:", sim.data.ncon)
                # optionally show contact forces (if available)
                if hasattr(sim.data, "contact"):
                    print("contact list length:", len(sim.data.contact))

            
            # Contact schedule for MPC
            contact_schedule = np.tile(current_contact_states, (params.num_nodes, 1))
            
            # Solve MPC
            u_optimal, x_predicted = controller.solve_mpc(
                x_current, x_ref_traj, contact_schedule
            )
            
            # Apply control
            u_apply = u_optimal[0]
            control_history.append(u_apply.copy())
            sim.apply_control(u_apply)
            
            # Compute cost
            cost = controller.compute_cost(x_current, u_apply, 
                                          x_ref_traj[0], np.zeros(24))
            cost_history.append(cost)
            
            # Step simulation
            sim.step()
            
            # Print status
            if step % int(params.control_freq) == 0:
                pos = x_current[3:6]
                vel = x_current[9:12]
                contact_str = ''.join(['█' if c else '░' for c in current_contact_states])
                
                print(f"t={t:6.2f}s | "
                      f"pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                      f"vel=[{vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}] | "
                      f"contacts={contact_str} | "
                      f"cost={cost:8.2f}")
            
            # Real-time control (no sleep needed with MuJoCo viewer)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("SIMULATION INTERRUPTED BY USER")
        print("="*60)
    
    finally:
        # ==========================================
        # CLEANUP
        # ==========================================
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
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
            print(f"  No data collected")
        
        sim.close()
        print("\n✓ Simulation environment closed")
        print("="*60)


if __name__ == "__main__":
    """
    Entry point for the MPC walking controller with MuJoCo
    
    To use with your own robot:
    1. Set ROBOT_XML_PATH to your MuJoCo XML file path
    2. Set USE_SIMPLE_ROBOT = False
    3. Adjust MPCParameters if needed
    4. Run: python mpc_mujoco_controller.py
    
    Requirements:
    - mujoco >= 3.0.0
    - numpy
    - scipy
    
    Install with: pip install mujoco numpy scipy
    """
    
    # Check dependencies
    try:
        import mujoco
        import scipy
        print("✓ All dependencies found")
        print(f"  MuJoCo version: {mujoco.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install mujoco numpy scipy")
        exit(1)
    
    # Run main controller
    main()