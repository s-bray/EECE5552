# simulation.py

import numpy as np
import mujoco
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation

from config import MPCParameters

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

    
    def apply_control_optimal(self, u: np.ndarray):
            """Apply PD control to track nominal standing configuration"""
            
            # Nominal standing pose - legs bent to support body
            q_nominal = np.array([
                0.0, 0.8, -1.6,  # FL: hip, thigh, shank
                0.0, 0.8, -1.6,  # FR
                0.0, 0.8, -1.6,  # HL
                0.0, 0.8, -1.6,  # HR
            ])
            
            # PD gains - tune these if robot is too stiff or too loose
            kp = 200.0   # Position gain
            kd = 50.0    # Velocity gain (damping)
            
            # Apply control to each actuator
            num_actuators = min(self.model.nu, 12)
            
            for act_idx in range(num_actuators):
                # Get the joint this actuator controls
                try:
                    trnid = self.model.actuator_trnid[act_idx]
                    joint_idx = int(trnid[1])
                except:
                    if act_idx < len(self.joint_indices):
                        joint_idx = self.joint_indices[act_idx]
                    else:
                        continue
                
                # Get current joint state
                qpos_addr = self.model.jnt_qposadr[joint_idx]
                qvel_addr = self.model.jnt_dofadr[joint_idx]
                
                q_current = float(self.data.qpos[qpos_addr])
                qd_current = float(self.data.qvel[qvel_addr])
                
                # PD control law: τ = Kp*(q_desired - q) - Kd*qd
                if act_idx < len(q_nominal):
                    q_desired = q_nominal[act_idx]
                else:
                    q_desired = 0.0
                
                q_error = q_desired - q_current
                torque = kp * q_error - kd * qd_current
                
                # Clamp to actuator limits
                try:
                    ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_idx]
                    torque = np.clip(torque, ctrl_min, ctrl_max)
                except:
                    torque = np.clip(torque, -150.0, 150.0)
                
                # Apply torque
                self.data.ctrl[act_idx] = torque

    
    def apply_control(self, u: np.ndarray):
        """Apply control inputs to robot (safer mapping + clamping)"""
        # Only use joint velocity part for now; ignore SRBD lambda_e until it's hooked into MuJoCo
        u_j = u[12:24]

        # Conservative PD gains to avoid violent torques
        kp = 200.0
        kd = 150

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