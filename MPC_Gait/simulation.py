# simulation.py

import numpy as np
import mujoco
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation

from config import MPCParameters

class RobotSimulation:
    """MuJoCo simulation environment for the wheeled-legged robot"""
    
    def __init__(self, xml_path: str, params: MPCParameters, verify=False, use_gui: bool = True):
        self.params = params
        self.xml_path = xml_path
        self.use_gui = use_gui
        self.verify = verify
        
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
            
            <option timestep="0.001" gravity="0 0 -9.81"/>
            
            <default>
                <geom rgba="0.8 0.6 0.4 1" friction="1 0.005 0.0001"/>
                <joint damping="5.0" armature="0.1"/>  </default>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                         rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
                <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                          reflectance=".2"/>
            </asset>
            
            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
                <geom name="floor" pos="0 0 -0.1" size="5 5 .05" type="plane" material="grid"/>
                
                <body name="base" pos="0 0 0.5">
                    <freejoint/>
                    <geom name="torso" type="box" size="0.3 0.15 0.08" mass="18.0" rgba="0.2 0.2 0.8 1"/>
                    <geom name="torso_imu" type="sphere" size="0.02" pos="0 0 0" mass="0.1" rgba="1 0 0 1"/>
                    
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
            0.0, 0.8, -1.6,  # FL: hip, thigh, shank
            0.0, 0.8, -1.6,  # FR
            0.0, 0.8, -1.6,  # HL
            0.0, 0.8, -1.6,  # HR
        ]
        
        for i, joint_idx in enumerate(self.joint_indices[:12]):
            if i < len(nominal_config):
                qpos_addr = self.model.jnt_qposadr[joint_idx]
                self.data.qpos[qpos_addr] = nominal_config[i]

        # Lower base height so feet touch ground
        self.data.qpos[2] = 0.35  # Z position
        
        # Forward kinematics to update everything
        mujoco.mj_forward(self.model, self.data)
        
        # DEBUG: Check if feet are in contact
        print(f"  DEBUG: Initial contacts = {self.data.ncon}")
        if self.data.ncon == 0:
            print("  ⚠️ WARNING: No ground contact! Legs not touching floor!")
        
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

        # Joint positions - USE PROPER MAPPING!
        joint_positions = []
        for joint_idx in self.joint_indices[:12]:  # ✓ Correct
            qpos_addr = self.model.jnt_qposadr[joint_idx]
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

    def simple_actuator_step(self, act_idx=0, torque_cmd=1.0, steps=50):
        print("=== ACTUATORS (index -> joint) ===")
        for i in range(self.model.nu):
            name = self.model.actuator(i).name or "<unnamed actuator>"
            joint_id = int(self.model.actuator_trnid[i][0])
            jname = self.model.joint(joint_id).name or "<no joint>"
            print(f"  actuator {i:2d}  {name:20s} -> joint {joint_id:2d} ({jname})")

    def debug_mujoco_mapping(self):
        print("\n=== MODEL / ACTUATOR MAPPING ===")
        print(f"model.nu (actuators) = {self.model.nu}")
        print(f"model.nv (dofs)      = {self.model.nv}")

        print("\n=== JOINTS (index / qposadr / dofadr) ===")
        for i in range(self.model.njnt):
            name = self.model.joint(i).name or "<unnamed joint>"
            qposadr = int(self.model.jnt_qposadr[i])
            dofadr  = int(self.model.jnt_dofadr[i])
            print(f"  joint {i:2d}  name={name:20s}  qposadr={qposadr:3d}  dofadr={dofadr:3d}")

        print("\n=== ACTUATORS (index -> joint) ===")
        for i in range(self.model.nu):
            name = self.model.actuator(i).name or "<unnamed actuator>"
            joint_id = int(self.model.actuator_trnid[i][0])
            jname = self.model.joint(joint_id).name or "<no joint>"
            print(f"  actuator {i:2d}  name={name:20s} -> joint {joint_id:2d} ({jname})")

        print("\n=== CURRENT STATE SAMPLE ===")
        print("qpos[:20] =", np.round(self.data.qpos[:20], 4))
        print("qvel[:20] =", np.round(self.data.qvel[:20], 4))


    def compute_leg_jacobian(self, body_name: str):
        """
        Return the translational Jacobian (3 x nv) of the requested body (e.g. 'fl_shank').
        Uses mujoco.mj_jac if available. If not available or fails, returns zeros.
        """
        try:
            # get body id
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            nv = self.model.nv
            # allocate arrays for jacobians (MuJoCo expects C-contiguous arrays)
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            # compute jacobian at current state
            mujoco.mj_jac(self.model, self.data, jacp, jacr, body_id)
            # jacp is translation jacobian in world (or model) frame for that body
            return jacp  # shape (3, nv)
        except Exception:
            # safe fallback: return zeros so J^T f contributes nothing
            return np.zeros((3, self.model.nv), dtype=np.float64)

    def apply_control(self, u: np.ndarray):
        """
        Safer PD velocity servo with diagnostics and gravity compensation.
        u layout:
        - u[0:12]   -> lambda_e (ignored for now at low-level)
        - u[12:24]  -> desired joint velocities (len 12)
        """

        # ============= VERIFICATION MODE: print mappings and test actuator =============
        if getattr(self, "verify", False):
            if not hasattr(self, "_verified_once"):
                self.debug_mujoco_mapping()
                print("\n=== ACTUATOR PROPERTIES (ctrlrange, gear) ===")
                for a in range(self.model.nu):
                    try:
                        ctrlrange = self.model.actuator_ctrlrange[a]
                    except Exception:
                        ctrlrange = ("N/A", "N/A")
                    try:
                        gear = float(self.model.actuator_gainprm[a][0])  # occasionally used
                    except Exception:
                        # fallback: read gear from XML motor attribute if present
                        gear = getattr(self.model.actuator, "gear", None)
                    print(f"  act {a:2d}  ctrlrange={ctrlrange}  gear={gear}")
                print("\n[DEBUG] Single-actuator poke (actuator 0): small positive, then negative.")
                # small poke to see sign and response
                self.simple_actuator_step(act_idx=0, torque_cmd=0.1, steps=50)
                self.simple_actuator_step(act_idx=0, torque_cmd=-0.1, steps=50)
                self._verified_once = True

            # freeze robot while verifying
            self.data.ctrl[:] = 0.0
            return

        # ============= UNPACK & SAFETY DEFAULTS =============
        # Desired velocities
        u_j_desired = np.asarray(u[12:24]).flatten()
        dt = float(self.model.opt.timestep)

        kp = 50
        kd = 1.5
        tau_hard_limit = 33.5

        # READ CURRENT JOINT STATES PROPERLY
        q_j_current = []
        q_j_dot_current = []
        
        for joint_idx in self.joint_indices[:12]:
            qpos_addr = int(self.model.jnt_qposadr[joint_idx])
            dof_addr = int(self.model.jnt_dofadr[joint_idx])
            q_j_current.append(float(self.data.qpos[qpos_addr]))
            q_j_dot_current.append(float(self.data.qvel[dof_addr]))
        
        q_j_current = np.array(q_j_current)
        q_j_dot_current = np.array(q_j_dot_current)

        # Compute desired joint positions by simple Euler integration
        q_j_desired = q_j_current + u_j_desired * dt

        # Gravity / bias compensation (MuJoCo provides qfrc_bias in generalized coordinates)
        mujoco.mj_forward(self.model, self.data)
        try:
            qfrc_bias = np.asarray(self.data.qfrc_bias)  # length nv
        except Exception:
            qfrc_bias = np.zeros(self.model.nv, dtype=np.float64)

        # Build per-actuator torque command safely
        # Ensure mapping cache exists
        if not hasattr(self, "_joint_to_actuator"):
            jtact = {}
            for act_idx in range(self.model.nu):
                try:
                    trnid = self.model.actuator_trnid[act_idx]
                    joint_id = int(trnid[1])
                    jtact[joint_id] = act_idx
                except Exception:
                    pass
            self._joint_to_actuator = jtact

        # Track max torque we try to apply (for safety/autotune)
        max_requested = 0.0

        for i, joint_id in enumerate(self.joint_indices[:12]):
            # state
            q = float(q_j_current[i])
            qd = float(q_j_dot_current[i])
            q_des = float(q_j_desired[i])
            qd_des = float(u_j_desired[i])

            # PD law: small position term + damping on velocity error
            tau_pd = kp * (q_des - q) + kd * (qd_des - qd)

            # add gravity compensation for this DOF if available
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            tau_bias = float(qfrc_bias[dof_addr]) if 0 <= dof_addr < self.model.nv else 0.0

            tau_total = tau_pd + tau_bias

            # mapping joint -> actuator index
            act_idx = self._joint_to_actuator.get(joint_id, None)
            if act_idx is None:
                # fallback assume actuator i controls joint i
                act_idx = i if i < self.data.ctrl.shape[0] else None

            if act_idx is None:
                continue

            # Clamp using actuator ctrlrange if available, fallback to tau_hard_limit
            try:
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_idx]
                tau_clamped = np.clip(tau_total, ctrl_min, ctrl_max)
            except Exception:
                tau_clamped = np.clip(tau_total, -tau_hard_limit, tau_hard_limit)

            self.data.ctrl[act_idx] = float(tau_clamped)
            max_requested = max(max_requested, abs(tau_clamped))

        # # Safety autoscale if we keep requesting insane torque
        # if max_requested > tau_hard_limit * 0.95:
        #     # scale down gains if we saturate actuators
        #     print(f"[WARNING] Requested torque ~{max_requested:.1f}Nm near limit; scaling gains down.")
        #     # scale factor and apply to next call via stored params
        #     if not hasattr(self, "_autoscale_factor"):
        #         self._autoscale_factor = 0.5
        #     else:
        #         self._autoscale_factor = max(0.25, self._autoscale_factor * 0.7)
        #     # reduce kp/kd for subsequent steps (affects next calls)
        #     # store them for observation (we don't mutate kp in this call; next loop will pick the reduced values if you implement it)
        #     print(f"[INFO] Suggested autoscale factor now {self._autoscale_factor:.3f}")

    def apply_control_old(self, u: np.ndarray):
        # unpack
        lambda_e = np.array(u[0:12]).reshape(4,3)
        u_j_desired = np.array(u[12:24]).flatten()
        dt = float(self.model.opt.timestep)

        # conservative gains — start small
        kp_pos = 50.0      # position gain (start smaller if robot too twitchy)
        kd_vel = 2.0       # velocity gain (damping)

        # read joint states with verified addresses
        q_current = np.zeros(12)
        qd_current = np.zeros(12)
        joint_ids = self.joint_indices[:12]
        for i, jid in enumerate(joint_ids):
            qposadr = int(self.model.jnt_qposadr[jid])
            dofadr = int(self.model.jnt_dofadr[jid])
            q_current[i] = float(self.data.qpos[qposadr])
            qd_current[i] = float(self.data.qvel[dofadr])

        # integrate to get desired joint positions
        q_des = q_current + u_j_desired * dt

        # Estimate gravity/bias torques (MuJoCo's qfrc_bias)
        # Make sure data is up to date
        mujoco.mj_forward(self.model, self.data)
        try:
            qfrc_bias = np.array(self.data.qfrc_bias)  # size nv
        except Exception:
            qfrc_bias = np.zeros(self.model.nv)

        # Build torque per actuator/joint
        tau_hard_limit = 200.0
        for i, jid in enumerate(joint_ids):
            dofadr = int(self.model.jnt_dofadr[jid])
            bias = float(qfrc_bias[dofadr]) if 0 <= dofadr < self.model.nv else 0.0

            # PD on position + gravity feedforward
            tau_pd = kp_pos * (q_des[i] - q_current[i]) + kd_vel * (u_j_desired[i] - qd_current[i])
            tau = tau_pd + bias

            # clamp to actuator ctrlrange if available, else hard clip
            # map joint -> actuator index
            act_idx = self._joint_to_actuator.get(jid, i if i < self.data.ctrl.shape[0] else None)
            if act_idx is not None and 0 <= act_idx < self.data.ctrl.shape[0]:
                try:
                    ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_idx]
                    tau = np.clip(tau, ctrl_min, ctrl_max)
                except Exception:
                    tau = np.clip(tau, -tau_hard_limit, tau_hard_limit)
                self.data.ctrl[act_idx] = float(tau)


    def step_physics(self):
        """Step the simulation's physics by one timestep"""
        mujoco.mj_step(self.model, self.data)
    
    def render(self):
        """Sync the viewer (if it exists) to match real-time"""
        if self.viewer is not None:
            if not self.viewer.is_running():
                # This allows user to close the window
                raise KeyboardInterrupt
            self.viewer.sync()
    
    def close(self):
        """Close the simulation"""
        if self.viewer is not None:
            self.viewer.close()