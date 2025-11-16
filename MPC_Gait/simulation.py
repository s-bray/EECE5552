# simulation.py - FIXED VERSION

import numpy as np
import mujoco
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation
from create_simple_quadruped import create_simple_quadruped_xml, create_simple_quadruped_xml_wheels

from config import MPCParameters

class RobotSimulation:
    """MuJoCo simulation environment for the wheeled-legged robot"""
    
    def __init__(self, xml_path: str, params: MPCParameters, verify=False, use_gui: bool = True, wheels=False):
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
        
        # Find actuated joints (CRITICAL: proper ordering)
        self.joint_indices = []
        self.joint_names = []
        
        # Expected order: FL, FR, HL, HR (each has hip, thigh, shank)
        leg_order = ['fl', 'fr', 'hl', 'hr']
        joint_types = ['hip', 'thigh', 'shank']
        
        for leg in leg_order:
            for jtype in joint_types:
                joint_name = f"{leg}_{jtype}"
                try:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    self.joint_indices.append(joint_id)
                    self.joint_names.append(joint_name)
                except:
                    print(f"  ⚠️ Warning: Joint '{joint_name}' not found")
        
        print(f"  ℹ Found {len(self.joint_indices)} controllable joints")
        print(f"  Joint order: {self.joint_names[:12]}")
        
        # Build actuator mapping
        self._build_actuator_mapping()
        
        # Initialize viewer if GUI enabled
        self.viewer = None
        if use_gui:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Set initial configuration
        self.set_initial_configuration()
        mujoco.mj_forward(self.model, self.data)
    
    def _build_actuator_mapping(self):
        """Build mapping from joints to actuators"""
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

        # Set base height
        self.data.qpos[2] = 0.38  # Z position
        
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
        """Get current robot state in SRBD format"""
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

        # Joint positions - proper mapping
        joint_positions = []
        for joint_idx in self.joint_indices[:12]:
            qpos_addr = int(self.model.jnt_qposadr[joint_idx])
            joint_positions.append(self.data.qpos[qpos_addr])

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
        FIXED: Robust PD control with proper force integration
        
        u layout:
        - u[0:12]   -> lambda_e (contact forces in body frame)
        - u[12:24]  -> desired joint velocities
        """
        
        # ===== UNPACK CONTROL =====
        lambda_e = np.asarray(u[0:12]).reshape(4, 3)  # 4 legs × (Fx, Fy, Fz)
        u_j_desired = np.asarray(u[12:24]).flatten()
        
        dt = float(self.model.opt.timestep)
        
        # ===== PD GAINS (TUNED FOR STABILITY) =====
        kp = 2.75   # Position gain
        kd = 0.01    # Damping gain
        tau_limit = 40.0  # Torque limit per joint
        
        # ===== READ CURRENT JOINT STATES =====
        q_current = np.zeros(12)
        qd_current = np.zeros(12)
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            qpos_addr = int(self.model.jnt_qposadr[joint_idx])
            dof_addr = int(self.model.jnt_dofadr[joint_idx])
            q_current[i] = float(self.data.qpos[qpos_addr])
            qd_current[i] = float(self.data.qvel[dof_addr])
        
        # ===== POSTURE CONTROL =====
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
        
        # ===== COMPUTE PD TORQUES =====
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
        
        # ===== CONTACT FORCE CONTRIBUTION =====
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
        
        # ===== GRAVITY COMPENSATION =====
        mujoco.mj_forward(self.model, self.data)
        qfrc_bias = np.asarray(self.data.qfrc_bias)
        
        tau_gravity = np.zeros(12)
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            dof_addr = int(self.model.jnt_dofadr[joint_idx])
            if dof_addr < len(qfrc_bias):
                tau_gravity[i] = qfrc_bias[dof_addr]
        
        # ===== TOTAL TORQUE =====
        # Weight contact forces moderately
        w_contact = 0.3
        tau_total = tau_pd + w_contact * tau_contact + tau_gravity
        
        # Clamp to limits
        tau_total = np.clip(tau_total, -tau_limit, tau_limit)
        
        # ===== APPLY TO ACTUATORS =====
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
    
    # def _detect_contacts(self) -> np.ndarray:
    #     """Detect which feet are in contact with ground"""
    #     contact_states = np.zeros(4, dtype=int)
        
    #     foot_geoms = ['fl_foot', 'fr_foot', 'hl_foot', 'hr_foot']
        
    #     for i in range(self.data.ncon):
    #         contact = self.data.contact[i]
    #         geom1_name = self.model.geom(contact.geom1).name or ""
    #         geom2_name = self.model.geom(contact.geom2).name or ""
            
    #         for leg_idx, foot_name in enumerate(foot_geoms):
    #             if foot_name in geom1_name or foot_name in geom2_name:
    #                 contact_states[leg_idx] = 1
        
    #     return contact_states
    
    # def _compute_foot_jacobian(self, foot_site_name: str):
    #     """Compute translational Jacobian for foot site"""
    #     try:
    #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
    #         jacp = np.zeros((3, self.model.nv), dtype=np.float64)
    #         jacr = np.zeros((3, self.model.nv), dtype=np.float64)
    #         mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
    #         return jacp
    #     except:
    #         return None
    
    def apply_control_new(self, u: np.ndarray, contact_states: np.ndarray = None):
        """
        τ_total = J^T λ + τ_PD with proper implementation
        
        This uses the original approach from your code but with critical fixes:
        1. Proper contact force transformation
        2. Better PD gains and tuning
        3. Correct Jacobian usage
        4. Stability improvements
        """
        
        # ===== VERIFICATION MODE =====
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

        # ===== UNPACK CONTROL INPUTS =====
        lambda_e = np.asarray(u[0:12]).reshape(4, 3)     # Contact forces [FL, FR, HL, HR] × (Fx,Fy,Fz)
        u_j_desired = np.asarray(u[12:24]).reshape(-1)   # Desired joint velocities (12)
        dt = float(self.model.opt.timestep)

        # ===== READ CURRENT JOINT STATES =====
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

        # ===== NOMINAL POSTURE =====
        if not hasattr(self, '_q_nominal') or self._q_nominal is None:
            self._q_nominal = np.array([0.0, 0.7, -1.4] * 4)

        # ===== COMPUTE DESIRED JOINT POSITIONS =====
        # If no velocity command, hold nominal pose
        if np.linalg.norm(u_j_desired) < 1e-6:
            q_des = self._q_nominal.copy()
            qd_des = np.zeros_like(qd)
        else:
            # Track velocity command via simple integration
            q_des = q + u_j_desired * dt
            qd_des = u_j_desired

        # ===== PD CONTROL WITH ANGLE WRAPPING =====
        def wrap_pi(e):
            """Wrap angle error to [-pi, pi]"""
            return (e + np.pi) % (2.0 * np.pi) - np.pi

        q_err = wrap_pi(q_des - q)
        qd_err = qd_des - qd

        # Per-joint PD gains (hip yaw joints softer for stability)
        kp_base = 2.5
        kd_base = 0.0
        
        kp_vec = np.array([0.25, 1.0, 1.0] * 4) * kp_base  # Hip softer
        kd_vec = np.array([0.30, 1.0, 1.0] * 4) * kd_base
        
        tau_pd = kp_vec * q_err + kd_vec * qd_err

        # ===== CONTACT FORCE CONTRIBUTION VIA JACOBIAN TRANSPOSE =====
        
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
                # Try to use foot site first (more accurate)
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_names[leg])
                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            except Exception:
                # Fallback to shank body Jacobian
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

        # ===== GRAVITY COMPENSATION =====
        mujoco.mj_forward(self.model, self.data)
        qfrc_bias = np.asarray(self.data.qfrc_bias)
        
        tau_gravity = np.zeros(12)
        for i in range(12):
            dofadr = dofaddrs[i]
            if 0 <= dofadr < len(qfrc_bias):
                tau_gravity[i] = qfrc_bias[dofadr]

        # ===== TOTAL TORQUE CALCULATION =====
        # Weight contact forces appropriately
        w_contact = 0.5  # Moderate weight (0.0 = ignore forces, 1.0 = full weight)
        
        tau_total_joint = tau_pd + w_contact * tau_contact + tau_gravity
        
        # Clamp to safe limits
        tau_limit = 80.0
        tau_total_joint = np.clip(tau_total_joint, -tau_limit, tau_limit)

        # ===== APPLY TO ACTUATORS =====
        # Build joint-to-actuator mapping if needed
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

        # ===== SATURATION WARNING =====
        try:
            overall_max = float(np.max(self.model.actuator_ctrlrange[:, 1]))
            if max_ctrl_mag > 0.9 * overall_max:
                if not hasattr(self, '_saturation_warning_count'):
                    self._saturation_warning_count = 0
                self._saturation_warning_count += 1
                
                # Only print warning occasionally (not every step)
                if self._saturation_warning_count % 100 == 0:
                    print(f"[WARN] Actuator near saturation: max|ctrl|={max_ctrl_mag:.2f}/{overall_max:.2f}")
        except Exception:
            pass


    def _compute_foot_jacobian(self, body_name: str):
        """
        Compute translational Jacobian for a body (fallback method)
        Returns shape (3, nv) or None on failure
        """
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            nv = self.model.nv
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, body_id)
            return jacp
        except Exception:
            return None


    def _detect_contacts(self) -> np.ndarray:
        """
        Detect which feet are in contact with ground
        Returns array [FL, FR, HL, HR] with 1=contact, 0=swing
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

    def step_physics(self):
        """Step the simulation's physics by one timestep"""
        mujoco.mj_step(self.model, self.data)
    
    def render(self):
        """Sync the viewer (if it exists) to match real-time"""
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
    
    def close(self):
        """Close the simulation"""
        if self.viewer is not None:
            self.viewer.close()