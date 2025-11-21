"""
MuJoCo Simulation Environment for Wheeled-Legged Robot

This module provides a complete simulation interface for a quadrupedal robot
using MuJoCo physics engine. It handles robot initialization, state extraction,
control application, and physics stepping with proper contact handling.

Classes:
    RobotSimulation: Main simulation environment wrapper

Dependencies:
    - numpy: Numerical computations
    - mujoco: Physics simulation engine
    - scipy.spatial.transform: Rotation representations
    - config: MPC parameters
    - create_simple_quadruped: XML model generation

Author: Based on Bjelonic et al. 2021 whole-body MPC approach
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation
from create_simple_quadruped import create_simple_quadruped_xml, create_simple_quadruped_xml_wheels

from config import MPCParameters

class RobotSimulation:
    """
    MuJoCo simulation environment for wheeled-legged quadruped robot.
    
    This class manages the entire simulation lifecycle including:
    - Robot model loading (from XML or programmatic generation)
    - State extraction in Single Rigid Body Dynamics (SRBD) format
    - Control application via PD control + contact forces
    - Physics stepping and rendering
    
    The simulation uses a 24-DOF control interface:
    - 12 contact forces (4 legs × 3D forces)
    - 12 joint velocities (4 legs × 3 joints)
    
    Attributes:
        params (MPCParameters): MPC configuration parameters
        xml_path (str): Path to robot XML model file
        use_gui (bool): Whether to enable visual rendering
        verify (bool): Verification mode flag (disables control)
        wheels (bool): Whether robot has wheels on legs
        model (mujoco.MjModel): MuJoCo physics model
        data (mujoco.MjData): MuJoCo simulation data
        joint_indices (list): Indices of controllable joints
        joint_names (list): Names of controllable joints
        joint_to_actuator (dict): Mapping from joint IDs to actuator IDs
        viewer (mujoco.viewer): Optional GUI viewer
    
    Example:
        >>> params = MPCParameters(robot_mass=15.0)
        >>> sim = RobotSimulation("robot.xml", params, use_gui=True)
        >>> state = sim.get_state()  # Get current state
        >>> control = np.zeros(24)  # Create control input
        >>> sim.apply_control(control)
        >>> sim.step_physics()
        >>> sim.render()
    """
    
    def __init__(self, xml_path: str, params: MPCParameters, verify=False, 
                 use_gui: bool = True, wheels=False):
        """
        Initialize the MuJoCo simulation environment.
        
        Args:
            xml_path (str): Path to robot XML model file. If file doesn't exist,
                          a simple quadruped model will be generated programmatically.
            params (MPCParameters): MPC configuration parameters including robot
                                  mass, inertia, and control settings.
            verify (bool, optional): If True, enables verification mode which
                                   disables control for debugging. Defaults to False.
            use_gui (bool, optional): If True, launches interactive 3D viewer.
                                    Defaults to True.
            wheels (bool, optional): If True, creates robot with wheels on legs.
                                   Defaults to False.
        
        Raises:
            FileNotFoundError: If XML file doesn't exist and programmatic
                             generation fails.
        
        Notes:
            - Automatically detects and maps all controllable joints
            - Skips freejoint (floating base) from control interface
            - Initializes robot in nominal standing configuration
            - Verifies ground contact after initialization
        """
        self.params = params
        self.xml_path = xml_path
        self.use_gui = use_gui
        self.verify = verify
        self.wheels = wheels
        
        # Load model
        if os.path.exists(xml_path):
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            print(f"XML file not found: {xml_path}")
            print("Creating simple quadruped model...")
            if self.wheels:
                xml_content = create_simple_quadruped_xml_wheels()
            else:
                xml_content = create_simple_quadruped_xml()
            self.model = mujoco.MjModel.from_xml_string(xml_content)
        
        self.data = mujoco.MjData(self.model)
        
        # Find actuated joints - FIXED for programmatic XML
        self.joint_indices = []
        self.joint_names = []

        print("\n[DEBUG] Scanning all joints in model:")
        for joint_id in range(self.model.njnt):
            joint_type = self.model.jnt_type[joint_id]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            
            if joint_name is None:
                joint_name = f"joint_{joint_id}"
            
            print(f"  Joint {joint_id}: {joint_name:20s} type={joint_type} "
                f"qpos[{qpos_addr}] dof[{dof_addr}]")
            
            # Skip freejoint (type 0) and wheel joints
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                if "wheel" not in joint_name:
                    self.joint_indices.append(joint_id)
                    self.joint_names.append(joint_name)

        print(f"\n  ℹ Found {len(self.joint_indices)} controllable joints")
        print(f"  Joint order: {self.joint_names[:12]}")
        
        print("\n[DEBUG] Joint ID mapping:")
        for i, (name, jid) in enumerate(zip(self.joint_names[:12], self.joint_indices[:12])):
            qpos_addr = self.model.jnt_qposadr[jid]
            print(f"  {i:2d}. {name:15s} -> joint_id={jid:2d} qpos_addr={qpos_addr:3d}")

        # Check if all IDs are the same
        unique_ids = set(self.joint_indices[:12])
        if len(unique_ids) == 1:
            print(f"\n⚠️⚠️⚠️ CRITICAL BUG: All joints mapped to same ID ({unique_ids.pop()})!")
            print("  This means joint names don't match XML or lookup is failing!")

        # Build actuator mapping
        self._build_actuator_mapping()
        
        # Detect and map thruster actuators (for wheeled mode)
        self.thruster_indices = []
        thruster_names = ['fl_thruster', 'fr_thruster', 'hl_thruster', 'hr_thruster']
        for thruster_name in thruster_names:
            try:
                thruster_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, thruster_name)
                self.thruster_indices.append(thruster_id)
            except:
                pass  # Thrusters don't exist in this model
        
        if len(self.thruster_indices) > 0:
            print(f"\n  ℹ Found {len(self.thruster_indices)} thruster actuators")
        
        # Initialize viewer if GUI enabled
        self.viewer = None
        if use_gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Set initial configuration
        self.set_initial_configuration()
        mujoco.mj_forward(self.model, self.data)
    
    def _build_actuator_mapping(self):
        """
        Build mapping from joint IDs to actuator IDs.
        
        This is necessary because MuJoCo's actuator indices may not directly
        correspond to joint indices. The mapping is stored in self.joint_to_actuator.
        
        Side Effects:
            Populates self.joint_to_actuator dictionary with {joint_id: actuator_id} pairs.
        
        Notes:
            - Uses model.actuator_trnid to determine which joint each actuator controls
            - Prints warning if any actuator cannot be mapped
        """
        self.joint_to_actuator = {}
        
        for act_idx in range(self.model.nu):
            try:
                # Get the joint this actuator controls
                joint_id = int(self.model.actuator_trnid[act_idx][0])
                self.joint_to_actuator[joint_id] = act_idx
            except Exception as e:
                print(f"  ⚠️ Warning: Could not map actuator {act_idx}: {e}")
        
        print(f"  ℹ Built actuator mapping for {len(self.joint_to_actuator)} actuators")
    
    def set_initial_configuration(self):
        """
        Set robot to nominal standing configuration.
        
        Initializes the robot in a stable standing pose with:
        - Hip joints at 0° (neutral)
        - Thigh joints at ~34° (0.6 rad)
        - Shank joints at ~-74° (-1.29 rad)
        - Base height at 0.40m above ground
        
        Side Effects:
            - Modifies self.data.qpos for joint positions
            - Runs forward kinematics via mujoco.mj_forward()
            - Adjusts base height if no ground contact detected
        
        Notes:
            - Configuration designed to center CoM over support polygon
            - Automatically verifies ground contact after initialization
            - Will lower robot by 5cm if initially floating
        """
        # Standing pose: legs slightly bent
        nominal_config = [
            0.0, 0.6, -1.2,  # FL
            0.0, 0.6, -1.2,  # FR
            0.0, 0.6, -1.2,  # HL
            0.0, 0.6, -1.2,  # HR
        ]
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            if i < len(nominal_config):
                qpos_addr = self.model.jnt_qposadr[joint_idx]
                self.data.qpos[qpos_addr] = nominal_config[i]

        # Set base height
        self.data.qpos[2] = 0.40  # Z position
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Check contacts
        print(f"  ℹ Initial contacts = {self.data.ncon}")
        if self.data.ncon == 0:
            print("  ⚠️ WARNING: No ground contact! Adjusting height...")
            self.data.qpos[2] = 0.35
            mujoco.mj_forward(self.model, self.data)
            print(f"  ℹ After adjustment: contacts = {self.data.ncon}")
        
    def get_state(self) -> np.ndarray:
        """
        Extract current robot state in Single Rigid Body Dynamics (SRBD) format.
        
        The state vector follows the convention used in Bjelonic et al. 2021:
        x = [θ, p, ω, v, q_j]
        
        Returns:
            np.ndarray: State vector of shape (24,) with layout:
                [0:3]   - θ (roll, pitch, yaw) in radians (Euler XYZ)
                [3:6]   - p (base position x, y, z) in world frame [m]
                [6:9]   - ω (angular velocity) in body frame [rad/s]
                [9:12]  - v (linear velocity) in body frame [m/s]
                [12:24] - q_j (joint positions) for 12 joints [rad]
                          Ordering: FL_hip, FL_thigh, FL_shank, FR_..., HL_..., HR_...
        
        Notes:
            - Quaternions from MuJoCo [w,x,y,z] are converted to Euler angles
            - Velocities are transformed from world frame to body frame
            - Joint angles are read directly from qpos using proper addressing
            - All angles are in radians
            - Performs sanity checking to detect incorrect joint reading
        
        Example:
            >>> state = sim.get_state()
            >>> roll, pitch, yaw = state[0:3]
            >>> base_height = state[5]
            >>> front_left_hip_angle = state[12]
        """
        # Base position and quaternion
        base_pos = self.data.qpos[0:3].copy()
        base_quat_mj = self.data.qpos[3:7].copy()  # [w, x, y, z]

        # Reorder to [x, y, z, w] for scipy
        quat_scipy = np.array([base_quat_mj[1], base_quat_mj[2], base_quat_mj[3], base_quat_mj[0]])
        r = Rotation.from_quat(quat_scipy)
        base_orn_euler = r.as_euler('xyz', degrees=False)

        # Base velocities
        base_vel_linear_world = self.data.qvel[0:3].copy()
        base_vel_angular_world = self.data.qvel[3:6].copy()

        # Convert velocities to body frame
        R_WB = r.as_matrix()
        R_BW = R_WB.T
        v_body = R_BW @ base_vel_linear_world
        omega_body = R_BW @ base_vel_angular_world

        # Joint positions with explicit checking
        joint_positions = []
        
        # Debug first time
        if not hasattr(self, '_qpos_debug_done'):
            print("\n[DEBUG] Joint reading check:")
            self._qpos_debug_done = True
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            qpos_addr = int(self.model.jnt_qposadr[joint_idx])
            joint_val = float(self.data.qpos[qpos_addr])
            joint_positions.append(joint_val)
            
            # Debug first time
            if not hasattr(self, f'_joint_debug_{i}_done'):
                setattr(self, f'_joint_debug_{i}_done', True)
                print(f"  Joint {i:2d} ({self.joint_names[i]:15s}): "
                    f"qpos[{qpos_addr:3d}] = {np.rad2deg(joint_val):7.2f}°")
        
        # Sanity check: joints should NOT all be the same!
        if len(set([round(q, 3) for q in joint_positions])) == 1:
            print(f"⚠️ WARNING: All joints have same value! ({joint_positions[0]:.3f})")
            print(f"   This indicates incorrect qpos addressing!")

        # Pad if necessary
        while len(joint_positions) < 12:
            joint_positions.append(0.0)
        joint_positions = np.array(joint_positions[:12])

        x = np.concatenate([
            base_orn_euler,      # theta (3)
            base_pos,            # p (3)
            omega_body,          # omega (3)
            v_body,              # v (3)
            joint_positions      # q_j (12)
        ])

        return x

    def apply_control(self, u: np.ndarray, contact_states: np.ndarray = None):
        """
        Apply control input using robust PD control with contact force integration.
        
        This is the LEGACY control method. Consider using apply_control_new() instead.
        
        Args:
            u (np.ndarray): Control vector of shape (24,) with layout:
                [0:12]  - λ_e (contact forces) in body frame [N]
                          Format: [FL_x, FL_y, FL_z, FR_x, FR_y, FR_z, HL_x, HL_y, HL_z, HR_x, HR_y, HR_z]
                [12:24] - u_j (desired joint velocities) [rad/s]
            contact_states (np.ndarray, optional): Binary array [4] indicating contact
                                                  (1=stance, 0=swing) for each leg.
                                                  If None, will auto-detect from simulation.
        
        Side Effects:
            - Modifies self.data.ctrl actuator commands
            - Updates internal tracking variables for debugging
        
        Control Law:
            τ = K_p(q_d - q) + K_d(q̇_d - q̇) + J^T λ + τ_gravity
            
            Where:
            - K_p: Position gain (100.0 N⋅m/rad, softer for hip joints)
            - K_d: Damping gain (10.0 N⋅m⋅s/rad)
            - J: Foot Jacobian (computed via mujoco.mj_jacSite)
            - λ: Contact forces (only applied to stance legs)
            - τ_gravity: Gravity compensation from qfrc_bias
        
        Notes:
            - Hip joints use 30% of standard K_p for stability
            - Contact forces only applied when contact_states[leg] == 1
            - Torques clamped to ±80 N⋅m per joint
            - Forces transformed from body frame to world frame
            - Unilateral contact constraint: only pushing forces (F_z ≥ 0)
        """
        
        # UNPACK CONTROL
        lambda_e = np.asarray(u[0:12]).reshape(4, 3)  # 4 legs × (Fx, Fy, Fz)
        u_j_desired = np.asarray(u[12:24]).flatten()
        
        dt = float(self.model.opt.timestep)
        
        # PD GAINS (TUNED FOR STABILITY)
        kp = 100.0  # Position gain
        kd = 50.0   # Damping gain
        tau_limit = 80.0  # Torque limit per joint
        
        # READ CURRENT JOINT STATES
        q_current = np.zeros(12)
        qd_current = np.zeros(12)
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            qpos_addr = int(self.model.jnt_qposadr[joint_idx])
            dof_addr = int(self.model.jnt_dofadr[joint_idx])
            q_current[i] = float(self.data.qpos[qpos_addr])
            qd_current[i] = float(self.data.qvel[dof_addr])
        
        # POSTURE CONTROL
        # Maintain nominal standing pose
        if not hasattr(self, '_q_nominal'):
            self._q_nominal = np.array([
                0.0, 0.6, -1.29,  # FL
                0.0, 0.6, -1.29,  # FR
                0.0, 0.6, -1.29,  # HL
                0.0, 0.6, -1.29   # HR
            ])
        
        # If no velocity command, hold nominal pose
        if np.linalg.norm(u_j_desired) < 1e-3:
            q_desired = self._q_nominal.copy()
            qd_desired = np.zeros(12)
        else:
            # Track velocity command
            q_desired = q_current + u_j_desired * dt
            qd_desired = u_j_desired
        
        # COMPUTE PD TORQUES
        def wrap_to_pi(angle):
            """Wrap angle to [-pi, pi]"""
            return (angle + np.pi) % (2.0 * np.pi) - np.pi
        
        # Position error with angle wrapping
        q_error = wrap_to_pi(q_desired - q_current)
        qd_error = qd_desired - qd_current
        
        # Per-joint gains (hip joints softer)
        kp_vec = np.array([0.3, 1.0, 1.0] * 4) * kp
        kd_vec = np.array([0.4, 1.0, 1.0] * 4) * kd
        
        tau_pd = kp_vec * q_error + kd_vec * qd_error
        
        # CONTACT FORCE CONTRIBUTION
        tau_contact = np.zeros(12)
        
        # Detect actual contacts if not provided
        if contact_states is None:
            contact_states = self._detect_contacts()
        
        # Convert forces to joint torques via Jacobian transpose
        for leg_idx in range(4):
            if contact_states[leg_idx] == 0:
                continue  # Leg in swing, no force
            
            # Get foot force in world frame
            f_body = lambda_e[leg_idx]
            
            # Convert to world frame
            base_quat = self.data.qpos[3:7]
            quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
            R_WB = Rotation.from_quat(quat_scipy).as_matrix()
            f_world = R_WB @ f_body
            
            # Only push forces (no pulling)
            f_world[2] = max(0.0, f_world[2])
            
            # Get Jacobian for this foot
            foot_name = ['fl_foot_site', 'fr_foot_site', 'hl_foot_site', 'hr_foot_site'][leg_idx]
            jacp = self._compute_foot_jacobian(foot_name)
            
            if jacp is not None:
                # τ = J^T f
                tau_from_force = jacp.T @ f_world
                
                # Extract joint contributions for this leg
                leg_start = leg_idx * 3
                for j in range(3):
                    joint_idx = self.joint_indices[leg_start + j]
                    dof_addr = int(self.model.jnt_dofadr[joint_idx])
                    if dof_addr < len(tau_from_force):
                        tau_contact[leg_start + j] += tau_from_force[dof_addr]
        
        # GRAVITY COMPENSATION
        mujoco.mj_forward(self.model, self.data)
        qfrc_bias = np.asarray(self.data.qfrc_bias)
        
        tau_gravity = np.zeros(12)
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            dof_addr = int(self.model.jnt_dofadr[joint_idx])
            if dof_addr < len(qfrc_bias):
                tau_gravity[i] = qfrc_bias[dof_addr]
        
        # TOTAL TORQUE
        # Weight contact forces moderately
        w_contact = 1.0
        tau_total = tau_pd + w_contact * tau_contact + tau_gravity
        
        # Clamp to limits
        tau_total = np.clip(tau_total, -tau_limit, tau_limit)
        
        # APPLY TO ACTUATORS
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            act_idx = self.joint_to_actuator.get(joint_idx, i)
            
            if act_idx >= self.model.nu:
                continue
            
            # Get actuator gear ratio
            try:
                gear = float(self.model.actuator_gear[act_idx, 0])
                if not np.isfinite(gear) or abs(gear) < 1e-9:
                    gear = 1.0
            except:
                gear = 1.0
            
            # Convert torque to control signal
            ctrl_signal = tau_total[i] / gear
            
            # Clamp to actuator limits
            try:
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_idx]
                ctrl_signal = np.clip(ctrl_signal, ctrl_min, ctrl_max)
            except:
                ctrl_signal = np.clip(ctrl_signal, -1.0, 1.0)
            
            self.data.ctrl[act_idx] = float(ctrl_signal)
    
    def _detect_contacts(self) -> np.ndarray:
        """
        Detect which feet are currently in contact with ground.
        
        Returns:
            np.ndarray: Binary contact state array of shape (4,) with layout:
                       [FL, FR, HL, HR] where 1=contact, 0=no contact
        
        Detection Method:
            Iterates through all active contacts in self.data.contact and checks
            if geometry names match foot or wheel geometry patterns.
        
        Notes:
            - Checks both foot geometries ('fl_foot', etc.) and wheel geometries
            - Uses substring matching on geometry names
            - Returns all zeros if no contacts detected
            - Does not consider contact normal or force magnitude
        
        Example:
            >>> contacts = sim._detect_contacts()
            >>> print(contacts)  # [1, 1, 0, 0] means front legs in contact
        """
        contact_states = np.zeros(4, dtype=int)
        
        # Foot geometry names (adjust if your XML uses different names)
        foot_geoms = ['fl_foot', 'fr_foot', 'hl_foot', 'hr_foot']
        
        # Also check for wheel contact if wheels are present
        wheel_geoms = ['fl_wheel_geom', 'fr_wheel_geom', 'hl_wheel_geom', 'hr_wheel_geom']
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name or ""
            geom2_name = self.model.geom(contact.geom2).name or ""
            
            for leg_idx, (foot_name, wheel_name) in enumerate(zip(foot_geoms, wheel_geoms)):
                if (foot_name in geom1_name or foot_name in geom2_name or
                    wheel_name in geom1_name or wheel_name in geom2_name):
                    contact_states[leg_idx] = 1
        
        return contact_states
    
    def _compute_foot_jacobian(self, foot_site_name: str):
        """
        Compute translational Jacobian for a foot site.
        
        Args:
            foot_site_name (str): Name of the foot site in the MuJoCo model
                                 (e.g., 'fl_foot_site')
        
        Returns:
            np.ndarray: Translational Jacobian matrix of shape (3, nv) where nv is
                       the number of degrees of freedom. Each row corresponds to
                       [x, y, z] and columns to generalized velocities.
            None: If site not found or computation fails
        
        Notes:
            - Uses mujoco.mj_jacSite for computation
            - Returns positional Jacobian (jacp), not rotational (jacr)
            - Useful for computing joint torques from Cartesian forces: τ = J^T F
        
        Example:
            >>> J = sim._compute_foot_jacobian('fl_foot_site')
            >>> foot_force = np.array([0, 0, 100])  # 100N vertical
            >>> joint_torques = J.T @ foot_force
        """
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            return jacp
        except:
            return None
    
    def apply_control_new(self, u: np.ndarray, contact_states: np.ndarray = None):
        """
        Apply control with improved Jacobian-based force mapping (RECOMMENDED).
        
        This is the IMPROVED control method with better stability and performance.
        Uses proper τ = J^T λ + τ_PD formulation with enhanced tuning.
        
        Args:
            u (np.ndarray): Control vector of shape (24,) with layout:
                [0:12]  - λ_e (contact forces) in body frame [N]
                          Format: [FL_x, FL_y, FL_z, FR_x, ...] (4 legs × 3)
                [12:24] - u_j (desired joint velocities) [rad/s]
            contact_states (np.ndarray, optional): Binary contact indicators [4].
                                                  If None, auto-detects from physics.
        
        Side Effects:
            - Sets self.data.ctrl actuator commands
            - Prints verification info on first call if verify=True
            - Tracks saturation warnings
        
        Control Improvements over apply_control():
            1. Better PD gain tuning (K_p=100, K_d=50)
            2. Proper velocity integration: q_d = q + u_j * dt
            3. Robust contact force transformation
            4. Fallback to body Jacobian if site unavailable
            5. Saturation monitoring
        
        Control Law:
            τ_total = K_p⋅(q_des - q) + K_d⋅(q̇_des - q̇) + w_contact⋅J^T⋅λ + τ_gravity
            
        Gains:
            - Hip yaw: K_p = 25 N⋅m/rad, K_d = 15 N⋅m⋅s/rad
            - Thigh/shank: K_p = 100 N⋅m/rad, K_d = 50 N⋅m⋅s/rad
            - Contact weight: w_contact = 1.0
            - Torque limit: ±80 N⋅m
        
        Notes:
            - In verification mode, sets all controls to zero
            - Enforces unilateral contact (F_z ≥ 0)
            - Wraps angle errors to [-π, π]
            - Clips final torques to actuator limits
        
        Example:
            >>> u = np.zeros(24)
            >>> u[2] = 100.0  # FL vertical force
            >>> u[14] = 0.5   # FL thigh velocity
            >>> sim.apply_control_new(u, contact_states=[1,1,1,1])
        """

        # VERIFICATION MODE
        if getattr(self, "verify", False):
            if not hasattr(self, "_verified_once"):
                self.debug_mujoco_mapping()
                print("\n=== ACTUATOR PROPERTIES (ctrlrange, gear) ===")
                for a in range(self.model.nu):
                    cmin, cmax = self.model.actuator_ctrlrange[a]
                    try:
                        gear = float(self.model.actuator_gear[a, 0])
                    except Exception:
                        gear = 1.0
                    print(f"  act {a:2d}  ctrlrange=[{cmin:.1f} {cmax:.1f}]  gear={gear:.1f}")
                self._verified_once = True
            self.data.ctrl[:] = 0.0
            return

        # UNPACK CONTROL INPUTS
        lambda_e = np.asarray(u[0:12]).reshape(4, 3)     # Contact forces [FL, FR, HL, HR] × (Fx,Fy,Fz)
        u_j_desired = np.asarray(u[12:24]).reshape(-1)   # Desired joint velocities (12)
        dt = float(self.model.opt.timestep)

        # READ CURRENT JOINT STATES
        joint_ids = self.joint_indices[:12]
        q = np.zeros(12)
        qd = np.zeros(12)
        dofaddrs = []
        
        for i, jid in enumerate(joint_ids):
            qpos_addr = int(self.model.jnt_qposadr[jid])
            dofadr = int(self.model.jnt_dofadr[jid])
            q[i] = float(self.data.qpos[qpos_addr])
            qd[i] = float(self.data.qvel[dofadr])
            dofaddrs.append(dofadr)

        # NOMINAL POSTURE
        if not hasattr(self, '_q_nominal') or self._q_nominal is None:
            self._q_nominal = np.array([0.0, 0.7, -1.4] * 4)

        # COMPUTE DESIRED JOINT POSITIONS
        if np.linalg.norm(u_j_desired) < 1e-6:
            q_des = self._q_nominal.copy()
            qd_des = np.zeros_like(qd)
        else:
            q_des = q + u_j_desired * dt
            qd_des = u_j_desired

        # PD CONTROL WITH ANGLE WRAPPING
        def wrap_pi(e):
            """Wrap angle error to [-pi, pi]"""
            return (e + np.pi) % (2.0 * np.pi) - np.pi

        q_err = wrap_pi(q_des - q)
        qd_err = qd_des - qd

        # Per-joint PD gains (hip yaw joints softer for stability)
        if self.wheels:
            # HEAVIER ROBOT (~32kg)
            kp_base = 400.0
            kd_base = 20.0
        else:
            # LIGHTER ROBOT (~15kg)
            kp_base = 100.0
            kd_base = 50.0
        
        kp_vec = np.array([0.25, 1.0, 1.0] * 4) * kp_base  # Hip softer
        kd_vec = np.array([0.30, 1.0, 1.0] * 4) * kd_base
        
        tau_pd = kp_vec * q_err + kd_vec * qd_err

        # CONTACT FORCE CONTRIBUTION VIA JACOBIAN TRANSPOSE
        
        # Detect contact states if not provided
        if contact_states is None:
            contact_states = self._detect_contacts()
        else:
            contact_states = np.asarray(contact_states).astype(int).reshape(-1)[:4]

        # Get base rotation for force transformation
        base_quat = self.data.qpos[3:7]  # [w, x, y, z]
        quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        R_WB = Rotation.from_quat(quat_scipy).as_matrix()

        # Foot site names (preferred) with shank body fallback
        site_names = ["fl_foot_site", "fr_foot_site", "hl_foot_site", "hr_foot_site"]
        body_fallback = ["fl_shank", "fr_shank", "hl_shank", "hr_shank"]

        nv = self.model.nv
        tau_contact_nv = np.zeros(nv)  # Torques in full generalized coordinate space

        for leg in range(4):
            if contact_states[leg] == 0:
                continue  # Leg in swing, skip
            
            # Get foot Jacobian
            try:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_names[leg])
                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            except Exception:
                jacp = self._compute_foot_jacobian(body_fallback[leg])
                if jacp is None:
                    continue

            # Transform force from body frame to world frame
            f_body = lambda_e[leg].copy()
            f_world = R_WB @ f_body
            
            # Enforce unilateral contact (no pulling)
            f_world[2] = max(0.0, f_world[2])
            
            # Compute joint torques: τ = J^T f
            tau_contact_nv += jacp.T @ f_world

        # Extract joint torques from full generalized torque vector
        tau_contact = np.zeros(12)
        for i in range(12):
            dofadr = dofaddrs[i]
            if 0 <= dofadr < nv:
                tau_contact[i] = tau_contact_nv[dofadr]

        # GRAVITY COMPENSATION
        mujoco.mj_forward(self.model, self.data)
        qfrc_bias = np.asarray(self.data.qfrc_bias)
        
        tau_gravity = np.zeros(12)
        for i in range(12):
            dofadr = dofaddrs[i]
            if 0 <= dofadr < len(qfrc_bias):
                tau_gravity[i] = qfrc_bias[dofadr]

        # TOTAL TORQUE CALCULATION
        w_contact = 1.0  # Contact force weight
        
        tau_total_joint = tau_pd + w_contact * tau_contact + tau_gravity
        
        # Clamp to safe limits
        tau_limit = 80.0
        tau_total_joint = np.clip(tau_total_joint, -tau_limit, tau_limit)

        # APPLY TO ACTUATORS
        if not hasattr(self, '_joint_to_actuator') or self._joint_to_actuator is None:
            jtact = {}
            for a in range(self.model.nu):
                try:
                    jid = int(self.model.actuator_trnid[a][0])
                    jtact[jid] = a
                except Exception:
                    pass
            self._joint_to_actuator = jtact

        max_ctrl_mag = 0.0
        
        for i, jid in enumerate(joint_ids):
            act_idx = self._joint_to_actuator.get(jid, i if i < self.model.nu else None)
            if act_idx is None:
                continue
            
            # Get actuator gear ratio
            try:
                gear = float(self.model.actuator_gear[act_idx, 0])
                if not np.isfinite(gear) or abs(gear) < 1e-9:
                    gear = 1.0
            except Exception:
                gear = 1.0

            # Convert torque to control signal
            ctrl = float(tau_total_joint[i] / gear)
            
            # Clamp to actuator limits
            try:
                cmin, cmax = self.model.actuator_ctrlrange[act_idx]
                ctrl = float(np.clip(ctrl, cmin, cmax))
            except Exception:
                ctrl = float(np.clip(ctrl, -1.0, 1.0))

            self.data.ctrl[act_idx] = ctrl
            max_ctrl_mag = max(max_ctrl_mag, abs(ctrl))

        # SATURATION WARNING
        try:
            overall_max = float(np.max(self.model.actuator_ctrlrange[:, 1]))
            if max_ctrl_mag > 0.9 * overall_max:
                if not hasattr(self, '_saturation_warning_count'):
                    self._saturation_warning_count = 0
                self._saturation_warning_count += 1
                
                if self._saturation_warning_count % 100 == 0:
                    print(f"[WARN] Actuator near saturation: max|ctrl|={max_ctrl_mag:.2f}/{overall_max:.2f}")
        except Exception:
            pass

    def apply_stabilized_thruster_control(self, contact_states: np.ndarray, 
                                        base_thrust_ratio: float = 0.6):
        """
        Apply stabilized thruster control with PID for Roll/Pitch/Yaw.
        
        Args:
            contact_states (np.ndarray): Contact flags [4] (1=stance, 0=swing)
            base_thrust_ratio (float): Base thrust as fraction of weight (default 0.6)
        """
        if len(self.thruster_indices) == 0:
            return

        # 1. Get Orientation State
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
        cosr_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (quat[0] * quat[2] - quat[3] * quat[1])
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Angular velocities
        w_x = self.data.qvel[3]
        w_y = self.data.qvel[4]
        w_z = self.data.qvel[5]

        # 2. PID Gains (Tuned for 32kg robot)
        kp_roll = 800.0
        kd_roll = 50.0
        
        kp_pitch = 800.0
        kd_pitch = 50.0
        
        kp_yaw = 800.0
        kd_yaw = 50.0

        # 3. Compute Control Efforts
        # Targets are 0.0 (stabilize to flat)
        t_roll = (kp_roll * (0.0 - roll)) - (kd_roll * w_x)
        t_pitch = (kp_pitch * (0.0 - pitch)) - (kd_pitch * w_y)
        t_yaw = (kp_yaw * (0.0 - yaw)) - (kd_yaw * w_z) # Yaw target 0 might be bad if turning? 
        # For now, stabilize Yaw to 0 (straight line walking). 
        # Ideally, target yaw should come from planner.
        
        # 4. Compute Base Thrust
        robot_weight = self.params.robot_mass * 9.81
        
        # Adaptive boost for swing legs
        num_swing = np.sum(contact_states == 0)
        swing_boost = 0.05 * num_swing # +5% per swing leg
        
        total_base_thrust = robot_weight * (base_thrust_ratio + swing_boost)
        base = total_base_thrust / 4.0

        # 5. Mixing Logic
        # FL (Front Left): +T_roll - T_pitch + T_yaw
        # FR (Front Right): -T_roll - T_pitch - T_yaw
        # HL (Rear Left): +T_roll + T_pitch - T_yaw
        # HR (Rear Right): -T_roll + T_pitch + T_yaw
        
        f_fl = base + t_roll - t_pitch + t_yaw
        f_fr = base - t_roll - t_pitch - t_yaw
        f_hl = base + t_roll + t_pitch - t_yaw
        f_hr = base - t_roll + t_pitch + t_yaw
        
        forces = [f_fl, f_fr, f_hl, f_hr]
        
        # 6. Apply
        for i, idx in enumerate(self.thruster_indices):
            force = np.clip(forces[i], 0, 500)
            self.data.ctrl[idx] = force

    def apply_thruster_forces(self, total_thrust: float = None):
        """
        Apply thruster forces to support the robot (for wheeled mode).
        
        Args:
            total_thrust (float, optional): Total upward thrust in Newtons.
                                           If None, uses robot weight (mg).
                                           Default distributes thrust equally across 4 thrusters.
        
        Notes:
            - Only works if thrusters exist in the model
            - Thrust is applied vertically upward at each hip
            - Helps compensate for reduced ground friction with wheels

        """
        if len(self.thruster_indices) == 0:
            return  # No thrusters in this model
        
        # Default: counteract gravity
        if total_thrust is None:
            total_thrust = self.params.robot_mass * 9.81  # mg
        
        # Distribute equally across thrusters
        thrust_per_thruster = total_thrust / len(self.thruster_indices)
        
        # Apply to each thruster actuator
        for thruster_idx in self.thruster_indices:
            self.data.ctrl[thruster_idx] = thrust_per_thruster

    def apply_adaptive_thruster_forces(self, contact_states: np.ndarray, 
                                       base_thrust_ratio: float = 0.5,
                                       swing_boost: float = 0.3):
        """
        Apply adaptive thruster forces based on gait state (SMART CONTROLLER).
        
        This controller adjusts thrust dynamically based on which legs are in 
        stance vs. swing, providing more support when legs are in the air.
        
        Args:
            contact_states (np.ndarray): Per-leg contact flags [4], where 1=stance, 0=swing
            base_thrust_ratio (float): Base thrust as fraction of robot weight (0.0-1.0)
                                      Default 0.5 = 50% of weight always supported
            swing_boost (float): Additional thrust per swinging leg as fraction of weight
                                Default 0.3 = 30% extra per swing leg
        
        Algorithm:
            total_thrust = (base_ratio + swing_boost × num_swing_legs) × mg
            
        Examples:
            Pure stance (4 legs down):
                thrust = (0.5 + 0.3×0) × mg = 0.5mg (50% support)
            
            Trot gait (2 legs down, 2 swinging):
                thrust = (0.5 + 0.3×2) × mg = 1.1mg (110% support)
            
            Single leg stance (1 down, 3 swinging):
                thrust = (0.5 + 0.3×3) × mg = 1.4mg (140% support)
        
        Notes:
            - Automatically adapts to any gait pattern
            - More swing legs → more thrust
            - Prevents robot from collapsing during swing phase
            - Works with hybrid_trot, hybrid_walk, pure_driving
        """
        if len(self.thruster_indices) == 0:
            return  # No thrusters in this model
        
        # Count swinging legs
        num_swing_legs = int(np.sum(contact_states == 0))
        
        # Calculate adaptive thrust
        robot_weight = self.params.robot_mass * 9.81  # mg in Newtons
        thrust_ratio = base_thrust_ratio + (swing_boost * num_swing_legs)
        total_thrust = thrust_ratio 
        
        # Distribute equally across all thrusters
        thrust_per_thruster = total_thrust / len(self.thruster_indices)
        
        # Apply to each thruster actuator
        for thruster_idx in self.thruster_indices:
            self.data.ctrl[thruster_idx] = thrust_per_thruster
        
        # Visualize the forces (optional, call separately if needed)
        self.visualize_thruster_forces(thrust_per_thruster)

    def visualize_thruster_forces(self, thrust_per_thruster: float):
        """
        Visualize thruster forces as arrows at thruster sites in the MuJoCo viewer.
        
        Draws upward-pointing arrows at each of the four thruster corner locations
        with length proportional to the thrust force being applied. Also applies
        the actual physics forces.
        
        Args:
            thrust_per_thruster (float): Force per thruster in Newtons
        
        Notes:
            - Arrows appear at thruster sites (four corners of torso)
            - Arrow length scales with force magnitude
            - Visible when force visualization is enabled (press 'F' key)
            - Uses MuJoCo's scene connector rendering
            - Arrows are yellow and point upward
            - Also applies actual physics forces at thruster locations
        
        Visualization:
            Press 'F' in the viewer to toggle force visualization on/off
        """
        if len(self.thruster_indices) == 0:
            return  # No thrusters
        
        # Get thruster site names
        thruster_site_names = ['fl_thruster_site', 'fr_thruster_site', 
                               'hl_thruster_site', 'hr_thruster_site']
        
        # Scale factor for arrow length (adjust for visibility)
        arrow_scale = 0.005  # 1N = 0.005m arrow length
        
        # Access the viewer's scene for custom rendering (if available)
        scene = self.viewer.user_scn if self.viewer is not None else None
        
        # Draw arrows and apply forces at each thruster location
        for site_name in thruster_site_names:
            try:
                # Get site ID
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                
                # Get site position in world frame
                site_pos = self.data.site_xpos[site_id].copy()
                
                # Get the body that this site is attached to
                body_id = self.model.site_bodyid[site_id]
                
                # === PHYSICS: Apply forces ===
                # Get body center of mass position
                body_com = self.data.xpos[body_id].copy()
                
                # Calculate offset from body COM to thruster site
                offset = site_pos - body_com
                
                # Create upward force vector
                force_vector = np.array([0.0, 0.0, thrust_per_thruster])
                
                # Calculate torque due to force at offset location
                torque_vector = np.cross(offset, force_vector)
                
                # Apply force and torque to physics
                self.data.xfrc_applied[body_id, 0:3] += force_vector
                self.data.xfrc_applied[body_id, 3:6] += torque_vector
                
                # === VISUALIZATION: Draw arrows ===
                if scene is not None:
                    # Calculate arrow endpoint (upward from site)
                    arrow_length = thrust_per_thruster * arrow_scale
                    arrow_end = site_pos + np.array([0.0, 0.0, arrow_length])
                    
                    # Add connector (arrow) to the scene
                    if scene.nconnector < scene.maxconnector:
                        connector_id = scene.nconnector
                        scene.nconnector += 1
                        
                        # Set connector properties
                        scene.connector_type[connector_id] = mujoco.mjtConnector.mjCNSTR_FORCE
                        scene.connector_width[connector_id] = 0.01  # Arrow thickness
                        scene.connector_rgba[connector_id] = [1.0, 1.0, 0.0, 1.0]  # Yellow
                        
                        # Set arrow start and end points
                        scene.connector_pos[connector_id, 0:3] = site_pos
                        scene.connector_pos[connector_id, 3:6] = arrow_end
                    
            except Exception:
                pass  # Site doesn't exist or rendering failed

    def step_physics(self):
        """
        Advance physics simulation by one timestep.
        
        Integrates equations of motion forward by model.opt.timestep seconds
        using the currently set actuator commands in self.data.ctrl.
        
        Side Effects:
            - Updates all state variables in self.data (qpos, qvel, qacc, etc.)
            - Processes contacts and constraints
            - Applies actuator forces
        
        Notes:
            - Timestep is defined in XML or defaults to 0.001s
            - Uses semi-implicit Euler integration
            - Automatically handles contact dynamics
        
        Example:
            >>> sim.apply_control(control_vector)
            >>> sim.step_physics()  # Physics advances by dt
            >>> new_state = sim.get_state()
        """
        mujoco.mj_step(self.model, self.data)
    
    def render(self):
        """
        Update the GUI viewer to display current simulation state.
        
        Synchronizes the passive viewer to match real-time playback speed.
        Should be called once per control loop iteration.
        
        Raises:
            KeyboardInterrupt: If viewer window is closed by user.
        
        Side Effects:
            - Updates viewer display
            - Processes viewer events
        
        Notes:
            - Does nothing if use_gui=False
            - Blocks briefly to maintain real-time rendering
            - Automatically handles viewer framerate
        
        Example:
            >>> while running:
            ...     sim.step_physics()
            ...     sim.render()  # Updates 3D visualization
        """
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
    
    def close(self):
        """
        Clean up simulation resources.
        
        Closes the viewer window and releases associated resources.
        Should be called when simulation is complete.
        
        Side Effects:
            - Closes GUI window if open
            - Releases viewer resources
        
        Example:
            >>> try:
            ...     sim.run_simulation()
            ... finally:
            ...     sim.close()  # Always clean up
        """
        if self.viewer is not None:
            self.viewer.close()