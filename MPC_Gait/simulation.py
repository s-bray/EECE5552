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
import controllers

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
    
    def __init__(self, params: MPCParameters, verify=False, 
                 use_gui: bool = True, wheels=False):
        """
        Initialize the MuJoCo simulation environment.
        
        Args:
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
        self.use_gui = use_gui
        self.verify = verify
        self.wheels = wheels
        
        # Load model
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
                # Check if this is a joint actuator
                if self.model.actuator_trntype[act_idx] == mujoco.mjtTrn.mjTRN_JOINT:
                    # Get the joint this actuator controls
                    joint_id = int(self.model.actuator_trnid[act_idx][0])
                    self.joint_to_actuator[joint_id] = act_idx
            except Exception as e:
                print(f"  ⚠️ Warning: Could not map actuator {act_idx}: {e}")
        
        print(f"  ℹ Built actuator mapping for {len(self.joint_to_actuator)} actuators")
        print(f"  Mapping: {self.joint_to_actuator}")
    
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
            
    def init_logging(self):
        """Initialize data logging storage."""
        self.logs = {
            'time': [],
            'qpos': [],
            'qvel': [],
            'ctrl': [],
            'base_pos': [],
            'base_rpy': []
        }
        print("  ℹ Data logging initialized")

    def log_state(self):
        """Record current simulation state."""
        if not hasattr(self, 'logs'):
            return
            
        self.logs['time'].append(self.data.time)
        self.logs['qpos'].append(self.data.qpos.copy())
        self.logs['qvel'].append(self.data.qvel.copy())
        self.logs['ctrl'].append(self.data.ctrl.copy())
        self.logs['base_pos'].append(self.data.qpos[0:3].copy())
        
        # Convert quaternion to RPY
        quat = self.data.qpos[3:7]
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        self.logs['base_rpy'].append(r.as_euler('xyz', degrees=True))

    def save_logs(self, filename="simulation_logs.csv"):
        """Save recorded logs to CSV file."""
        if not hasattr(self, 'logs') or not self.logs['time']:
            print("  ⚠️ No logs to save!")
            return
            
        import csv
        
        # Ensure csv directory exists
        log_dir = "csv"
        os.makedirs(log_dir, exist_ok=True)
        
        # Prepend directory to filename if not already there
        if not filename.startswith(log_dir):
            filepath = os.path.join(log_dir, os.path.basename(filename))
        else:
            filepath = filename
            
        # Prepare header
        header = ['time', 'base_x', 'base_y', 'base_z', 'roll', 'pitch', 'yaw']
        
        # Add control columns
        num_ctrl = len(self.logs['ctrl'][0])
        for i in range(num_ctrl):
            header.append(f'ctrl_{i}')
            
        # Add joint position columns
        for i, name in enumerate(self.joint_names):
            header.append(f'q_{name}')
            
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
                num_steps = len(self.logs['time'])
                for i in range(num_steps):
                    row = []
                    row.append(self.logs['time'][i])
                    
                    # Base Pos
                    row.extend(self.logs['base_pos'][i])
                    
                    # Base RPY
                    row.extend(self.logs['base_rpy'][i])
                    
                    # Controls
                    row.extend(self.logs['ctrl'][i])
                    
                    # Joint Positions
                    qpos = self.logs['qpos'][i]
                    for j, name in enumerate(self.joint_names):
                        jid = self.joint_indices[j]
                        qaddr = self.model.jnt_qposadr[jid]
                        row.append(qpos[qaddr])
                        
                    writer.writerow(row)
                    
            print(f"  ✓ Logs saved to {filepath} ({num_steps} rows)")
        except Exception as e:
            print(f"  ✗ Error saving logs: {e}")
        
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
        Delegates to controllers.apply_control_new.
        """
        controllers.apply_control_new(self, u, contact_states)

    def apply_stabilized_thruster_control(self, contact_states: np.ndarray, 
                                          base_thrust_ratio: float = 0.4):
        """
        Apply stabilized thruster control (Delegates to controllers).
        """
        controllers.apply_stabilized_thruster_control(self, contact_states, base_thrust_ratio)

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