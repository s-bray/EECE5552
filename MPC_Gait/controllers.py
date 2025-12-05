import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def apply_stabilized_thruster_control(sim, contact_states, base_thrust_ratio=0.4):
    """
    Apply stabilized thruster control for walking gait.
    
    Args:
        sim: RobotSimulation instance (provides access to model, data, params)
        contact_states (np.ndarray): Contact flags [4]
        base_thrust_ratio (float): Base thrust as fraction of weight
    """
    if len(sim.thruster_indices) == 0:
        return

    # 1. Get Orientation State
    quat = sim.data.qpos[3:7]  # [w, x, y, z]
    
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

    # Angular velocities
    w_x = sim.data.qvel[3]
    w_y = sim.data.qvel[4]
    
    # 2. PID Gains (Tuned for 32kg robot)
    kp_roll = 100.0
    kd_roll = 10.0
    
    kp_pitch = 100.0
    kd_pitch = 10.0
    
    # 3. Compute Control Efforts
    # Targets are 0.0 (stabilize to flat)
    t_roll = (kp_roll * (0.0 - roll)) - (kd_roll * w_x)
    t_pitch = (kp_pitch * (0.0 - pitch)) - (kd_pitch * w_y)
    
    # 4. Compute Base Thrust
    robot_weight = sim.params.robot_mass * 9.81
    
    # Adaptive boost for swing legs
    num_swing = np.sum(contact_states == 0)
    swing_boost = 0.05 * num_swing # +5% per swing leg
    
    total_base_thrust = robot_weight * (base_thrust_ratio + swing_boost)
    base = total_base_thrust / 4.0

    # 5. Mixing Logic
    # FL (Front Left): +T_roll - T_pitch
    # FR (Front Right): -T_roll - T_pitch
    # HL (Rear Left): +T_roll + T_pitch
    # HR (Rear Right): -T_roll + T_pitch
    
    # NOTE: Vertical thrusters cannot control Yaw! 
    # Removed t_yaw to prevent warping forces.
    
    f_fl = base + t_roll - t_pitch
    f_fr = base - t_roll - t_pitch
    f_hl = base + t_roll + t_pitch
    f_hr = base - t_roll + t_pitch
    
    forces = [f_fl, f_fr, f_hl, f_hr]
    
    # 6. Apply
    for i, idx in enumerate(sim.thruster_indices):
        force = np.clip(forces[i], 0, 500)
        sim.data.ctrl[idx] = force

def apply_trot_thruster_control(sim, contact_states, base_thrust_ratio=0.5):
    """
    Apply trot-specific thruster control with targeted support for swinging legs.
    
    Args:
        sim: RobotSimulation instance
        contact_states (np.ndarray): Contact flags [4]
        base_thrust_ratio (float): Base thrust as fraction of weight
    """
    if len(sim.thruster_indices) == 0:
        return

    # 1. Get Orientation State (same as walk mode)
    quat = sim.data.qpos[3:7]  # [w, x, y, z]
    
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
        
    # Angular velocities
    w_x = sim.data.qvel[3]
    w_y = sim.data.qvel[4]
    v_z = sim.data.qvel[2]  # Vertical velocity

    # 2. PID Gains (TROT-SPECIFIC: Stronger damping + Integral for roll)
    kp_roll = 600.0
    ki_roll = 100.0  # Integral to eliminate steady-state roll bias
    kd_roll = 50.0
    
    kp_pitch = 800.0
    kd_pitch = 50.0
    
    # Roll integral accumulator (stored in sim object)
    if not hasattr(sim, '_roll_integral'):
        sim._roll_integral = 0.0
    
    dt = sim.model.opt.timestep
    sim._roll_integral += (0.0 - roll) * dt
    sim._roll_integral = np.clip(sim._roll_integral, -0.2, 0.2)  # Anti-windup

    # 3. Compute Control Efforts
    t_roll = (kp_roll * (0.0 - roll)) + (ki_roll * sim._roll_integral) - (kd_roll * w_x)
    t_pitch = (kp_pitch * (0.0 - pitch)) - (kd_pitch * w_y)
    
    # 4. Compute Base Thrust (TROT-SPECIFIC)
    robot_weight = sim.params.robot_mass * 9.81
    
    # Higher base thrust for trot (50% vs 40% for walk)
    total_base_thrust = robot_weight * base_thrust_ratio
    
    # VERTICAL VELOCITY DAMPING: Reduce thrust if rising too fast
    kd_vertical = 20.0  # Vertical damping coefficient
    vertical_damping = -kd_vertical * v_z  # Negative feedback on upward velocity
    
    # Distribute base thrust + vertical damping
    base = (total_base_thrust + vertical_damping) / 4.0
    
    # 5. TROT-SPECIFIC: Targeted boost for swinging legs
    # Add extra thrust on thrusters corresponding to legs in the air
    swing_boost_per_leg = robot_weight * 0.15  # 15% extra per swing leg
    
    # Create per-thruster boost array
    # Thruster order: [FL, FR, HL, HR]
    leg_boost = np.zeros(4)
    for i in range(4):
        if contact_states[i] == 0:  # Leg is swinging
            leg_boost[i] = swing_boost_per_leg / 4.0  # Distribute to this thruster

    # 6. Mixing Logic (same as walk, but with per-leg boost)
    # NOTE: Removed t_yaw as vertical thrusters cannot control yaw
    f_fl = base + leg_boost[0] + t_roll - t_pitch
    f_fr = base + leg_boost[1] - t_roll - t_pitch
    f_hl = base + leg_boost[2] + t_roll + t_pitch
    f_hr = base + leg_boost[3] - t_roll + t_pitch
    
    forces = [f_fl, f_fr, f_hl, f_hr]
    
    # 7. Apply
    for i, idx in enumerate(sim.thruster_indices):
        force = np.clip(forces[i], 0, 500)
        sim.data.ctrl[idx] = force

def apply_control(sim, u, contact_states=None):
    """
    Apply PD control + Feedforward contact forces (Legacy method).
    
    Args:
        sim: RobotSimulation instance
        u (np.ndarray): Control vector [0:12]=forces, [12:24]=q_des (position control)
        contact_states (np.ndarray): Contact flags [4]
    """
    # UNPACK CONTROL
    lambda_e = np.asarray(u[0:12]).reshape(4, 3)  # 4 legs × (Fx, Fy, Fz)
    u_j_desired = np.asarray(u[12:24]).flatten()
    
    dt = float(sim.model.opt.timestep)
    
    # PD GAINS (TUNED FOR STABILITY)
    kp = 100.0  # Position gain
    kd = 50.0   # Damping gain
    tau_limit = 80.0  # Torque limit per joint
    
    # READ CURRENT JOINT STATES
    q_current = np.zeros(12)
    qd_current = np.zeros(12)
    
    for i, joint_idx in enumerate(sim.joint_indices[:12]):
        qpos_addr = int(sim.model.jnt_qposadr[joint_idx])
        dof_addr = int(sim.model.jnt_dofadr[joint_idx])
        q_current[i] = float(sim.data.qpos[qpos_addr])
        qd_current[i] = float(sim.data.qvel[dof_addr])
    
    # POSTURE CONTROL
    # Maintain nominal standing pose
    if not hasattr(sim, '_q_nominal'):
        sim._q_nominal = np.array([0.0, 0.7, -1.4] * 4)
        
    # If u_j_desired is small (zero), use nominal pose
    if np.linalg.norm(u_j_desired) < 1e-6:
        q_desired = sim._q_nominal.copy()
        qd_desired = np.zeros(12)
    else:
        # Interpret u_j as desired POSITION directly (legacy behavior)
        # OR interpret as velocity?
        # In apply_control_new, it's velocity. In original apply_control, it seemed to be position?
        # Let's check the original code.
        # "q_desired = u_j_desired" implies it was position.
        # But MPC usually outputs velocity or position.
        # Let's assume it matches the original logic: u[12:24] is q_des.
        q_desired = u_j_desired
        qd_desired = np.zeros(12)
    
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
        contact_states = sim._detect_contacts()
    
    # Convert forces to joint torques via Jacobian transpose
    for leg_idx in range(4):
        if contact_states[leg_idx] == 0:
            continue  # Leg in swing, no force
        
        # Get foot force in world frame
        f_body = lambda_e[leg_idx]
        
        # Convert to world frame
        base_quat = sim.data.qpos[3:7]
        quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        R_WB = Rotation.from_quat(quat_scipy).as_matrix()
        f_world = R_WB @ f_body
        
        # Only push forces (no pulling)
        f_world[2] = max(0.0, f_world[2])
        
        # Get Jacobian for this foot
        foot_name = ['fl_foot_site', 'fr_foot_site', 'hl_foot_site', 'hr_foot_site'][leg_idx]
        jacp = sim._compute_foot_jacobian(foot_name)
        
        if jacp is not None:
            # τ = J^T f
            tau_from_force = jacp.T @ f_world
            
            # Extract joint contributions for this leg
            leg_start = leg_idx * 3
            for j in range(3):
                joint_idx = sim.joint_indices[leg_start + j]
                dof_addr = int(sim.model.jnt_dofadr[joint_idx])
                if dof_addr < len(tau_from_force):
                    tau_contact[leg_start + j] += tau_from_force[dof_addr]
    
    # GRAVITY COMPENSATION
    mujoco.mj_forward(sim.model, sim.data)
    qfrc_bias = np.asarray(sim.data.qfrc_bias)
    
    tau_gravity = np.zeros(12)
    for i, joint_idx in enumerate(sim.joint_indices[:12]):
        dof_addr = int(sim.model.jnt_dofadr[joint_idx])
        if dof_addr < len(qfrc_bias):
            tau_gravity[i] = qfrc_bias[dof_addr]
    
    # TOTAL TORQUE
    # Weight contact forces moderately
    w_contact = 1.0
    tau_total = tau_pd + w_contact * tau_contact + tau_gravity
    
    # Clamp to limits
    tau_total = np.clip(tau_total, -tau_limit, tau_limit)
    
    # APPLY TO ACTUATORS
    for i, joint_idx in enumerate(sim.joint_indices[:12]):
        act_idx = sim.joint_to_actuator.get(joint_idx, i)
        
        if act_idx >= sim.model.nu:
            continue
        
        # Get actuator gear ratio
        try:
            gear = float(sim.model.actuator_gear[act_idx, 0])
            if not np.isfinite(gear) or abs(gear) < 1e-9:
                gear = 1.0
        except:
            gear = 1.0
        
        # Convert torque to control signal
        ctrl_signal = tau_total[i] / gear
        
        # Clamp to actuator limits
        try:
            ctrl_min, ctrl_max = sim.model.actuator_ctrlrange[act_idx]
            ctrl_signal = np.clip(ctrl_signal, ctrl_min, ctrl_max)
        except:
            ctrl_signal = np.clip(ctrl_signal, -1.0, 1.0)
        
        sim.data.ctrl[act_idx] = float(ctrl_signal)

def apply_control_new(sim, u, contact_states=None):
    """
    Apply control with improved Jacobian-based force mapping (RECOMMENDED).
    
    Args:
        sim: RobotSimulation instance
        u (np.ndarray): Control vector [0:12]=forces, [12:24]=u_j (velocity)
        contact_states (np.ndarray): Contact flags [4]
    """
    # VERIFICATION MODE
    if getattr(sim, "verify", False):
        if not hasattr(sim, "_verified_once"):
            sim.debug_mujoco_mapping()
            print("\n=== ACTUATOR PROPERTIES (ctrlrange, gear) ===")
            for a in range(sim.model.nu):
                cmin, cmax = sim.model.actuator_ctrlrange[a]
                try:
                    gear = float(sim.model.actuator_gear[a, 0])
                except Exception:
                    gear = 1.0
                print(f"  act {a:2d}  ctrlrange=[{cmin:.1f} {cmax:.1f}]  gear={gear:.1f}")
            sim._verified_once = True
        sim.data.ctrl[:] = 0.0
        return

    # UNPACK CONTROL INPUTS
    lambda_e = np.asarray(u[0:12]).reshape(4, 3)     # Contact forces [FL, FR, HL, HR] × (Fx,Fy,Fz)
    u_j_desired = np.asarray(u[12:24]).reshape(-1)   # Desired joint velocities (12)
    dt = float(sim.model.opt.timestep)

    # READ CURRENT JOINT STATES
    joint_ids = sim.joint_indices[:12]
    q = np.zeros(12)
    qd = np.zeros(12)
    dofaddrs = []
    
    for i, jid in enumerate(joint_ids):
        qpos_addr = int(sim.model.jnt_qposadr[jid])
        dofadr = int(sim.model.jnt_dofadr[jid])
        q[i] = float(sim.data.qpos[qpos_addr])
        qd[i] = float(sim.data.qvel[dofadr])
        dofaddrs.append(dofadr)

    # NOMINAL POSTURE
    if not hasattr(sim, '_q_nominal') or sim._q_nominal is None:
        sim._q_nominal = np.array([0.0, 0.7, -1.4] * 4)

    # COMPUTE DESIRED JOINT POSITIONS
    if np.linalg.norm(u_j_desired) < 1e-6:
        q_des = sim._q_nominal.copy()
        qd_des = np.zeros_like(qd)
    else:
        q_des = q + u_j_desired * dt
        qd_des = u_j_desired

    # PD CONTROL
    q_err = wrap_to_pi(q_des - q)
    qd_err = qd_des - qd

    # Per-joint PD gains
    if sim.wheels:
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
        contact_states = sim._detect_contacts()
    else:
        contact_states = np.asarray(contact_states).astype(int).reshape(-1)[:4]

    # Get base rotation for force transformation
    base_quat = sim.data.qpos[3:7]  # [w, x, y, z]
    quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
    R_WB = Rotation.from_quat(quat_scipy).as_matrix()

    # Foot site names (preferred) with shank body fallback
    site_names = ["fl_foot_site", "fr_foot_site", "hl_foot_site", "hr_foot_site"]
    body_fallback = ["fl_shank", "fr_shank", "hl_shank", "hr_shank"]

    nv = sim.model.nv
    tau_contact_nv = np.zeros(nv)  # Torques in full generalized coordinate space

    for leg in range(4):
        if contact_states[leg] == 0:
            continue  # Leg in swing, skip
        
        # Get foot Jacobian
        try:
            sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, site_names[leg])
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, sid)
        except Exception:
            jacp = sim._compute_foot_jacobian(body_fallback[leg])
            if jacp is None:
                continue

        # Transform force from body frame to world frame
        f_body = lambda_e[leg].copy()
        f_world = R_WB @ f_body
        
        # Enforce unilateral contact (no pulling)
        f_world[2] = max(0.0, f_world[2])
        
        # τ = J^T F
        tau_contact_nv += jacp.T @ f_world

    # Extract joint torques from generalized torque vector
    tau_contact = np.zeros(12)
    for i, jid in enumerate(joint_ids):
        dofadr = int(sim.model.jnt_dofadr[jid])
        if dofadr < nv:
            tau_contact[i] = tau_contact_nv[dofadr]

    # GRAVITY COMPENSATION
    mujoco.mj_forward(sim.model, sim.data)
    qfrc_bias = np.asarray(sim.data.qfrc_bias)
    
    tau_gravity = np.zeros(12)
    for i, jid in enumerate(joint_ids):
        dofadr = int(sim.model.jnt_dofadr[jid])
        if dofadr < len(qfrc_bias):
            tau_gravity[i] = qfrc_bias[dofadr]

    # TOTAL TORQUE
    w_contact = 1.0
    tau_total = tau_pd + w_contact * tau_contact + tau_gravity
    
    # Clamp to limits
    tau_limit = 80.0
    tau_total = np.clip(tau_total, -tau_limit, tau_limit)

    # APPLY TO ACTUATORS
    for i, jid in enumerate(joint_ids):
        act_idx = sim.joint_to_actuator.get(jid, i)
        
        if act_idx >= sim.model.nu:
            continue
            
        # Get actuator gear ratio
        try:
            gear = float(sim.model.actuator_gear[act_idx, 0])
            if not np.isfinite(gear) or abs(gear) < 1e-9:
                gear = 1.0
        except:
            gear = 1.0
        
        # Convert torque to control signal
        ctrl_signal = tau_total[i] / gear
        
        # Clamp to actuator limits
        try:
            ctrl_min, ctrl_max = sim.model.actuator_ctrlrange[act_idx]
            ctrl_signal = np.clip(ctrl_signal, ctrl_min, ctrl_max)
        except:
            ctrl_signal = np.clip(ctrl_signal, -1.0, 1.0)
        
        sim.data.ctrl[act_idx] = float(ctrl_signal)
