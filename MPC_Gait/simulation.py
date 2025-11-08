# simulation.py

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
        self.data.qpos[2] = 0.320  # Z position
        
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

        # Safety autoscale if we keep requesting insane torque
        if max_requested > tau_hard_limit * 0.95:
            # scale down gains if we saturate actuators
            print(f"[WARNING] Requested torque ~{max_requested:.1f}Nm near limit; scaling gains down.")
            # scale factor and apply to next call via stored params
            if not hasattr(self, "_autoscale_factor"):
                self._autoscale_factor = 0.5
            else:
                self._autoscale_factor = max(0.25, self._autoscale_factor * 0.7)
            # reduce kp/kd for subsequent steps (affects next calls)
            # store them for observation (we don't mutate kp in this call; next loop will pick the reduced values if you implement it)
            print(f"[INFO] Suggested autoscale factor now {self._autoscale_factor:.3f}")

    def apply_control_new(self, u: np.ndarray, contact_states: np.ndarray | None = None):
        """
        τ_total = Jᵀ λ + τ_PD. PD uses shortest-angle errors for stability.
        """
        # --- verify mode ---
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

        # ---- unpack ----
        lam = np.asarray(u[0:12]).reshape(4, 3)     # [FL, FR, HL, RH] × (Fx,Fy,Fz)
        u_j_desired = np.asarray(u[12:24]).reshape(-1)
        dt = float(self.model.opt.timestep)

        # ---- joint states (12) ----
        joint_ids = self.joint_indices[:12]
        q  = np.zeros(12)
        qd = np.zeros(12)
        dofaddrs = []
        for i, jid in enumerate(joint_ids):
            qposadr = int(self.model.jnt_qposadr[jid])
            dofadr  = int(self.model.jnt_dofadr[jid])
            q[i]  = float(self.data.qpos[qposadr])
            qd[i] = float(self.data.qvel[dofadr])
            dofaddrs.append(dofadr)

        # ---- posture hold if no vel ref ----
        if not hasattr(self, "_q_nominal") or self._q_nominal is None:
            self._q_nominal = np.array([0.0, 0.6, -1.2] * 4)

        if np.linalg.norm(u_j_desired) < 1e-6:
            q_des  = self._q_nominal.copy()
            qd_des = np.zeros_like(qd)
        else:
            q_des  = q + u_j_desired * dt
            qd_des = u_j_desired

        # ---- shortest-angle PD (wrap to (-pi,pi]) ----
        def wrap_pi(e):
            return (e + np.pi) % (2.0*np.pi) - np.pi

        q_err  = wrap_pi(q_des - q)
        qd_err = qd_des - qd

        # per-joint gains (hip yaw softer)
        kp_all, kd_all = 100.0, 10
        kp_vec = np.array([0.25, 1.0, 1.0] * 4) * kp_all
        kd_vec = np.array([0.30, 1.0, 1.0] * 4) * kd_all
        tau_pd = kp_vec * q_err + kd_vec * qd_err

        # ---- contact gating ----
        if contact_states is None:
            contact_states = np.zeros(4, dtype=int)
            touching = set()
            for k in range(self.data.ncon):
                con = self.data.contact[k]
                g1 = self.model.geom(con.geom1).name or ""
                g2 = self.model.geom(con.geom2).name or ""
                touching.add(g1); touching.add(g2)
            foot_names = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
            for i, nm in enumerate(foot_names):
                if nm in touching:
                    contact_states[i] = 1
        else:
            contact_states = np.asarray(contact_states).astype(int).reshape(-1)[:4]

        # ---- J^T λ using foot sites (fallback to shank) ----
        site_names   = ["fl_foot_site", "fr_foot_site", "hl_foot_site", "hr_foot_site"]
        body_fallback = ["fl_shank", "fr_shank", "hl_shank", "hr_shank"]

        nv = self.model.nv
        tau_lambda_nv = np.zeros(nv)
        for leg in range(4):
            if contact_states[leg] == 0:
                continue
            try:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_names[leg])
                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            except Exception:
                jacp = self.compute_leg_jacobian(body_fallback[leg])

            f = lam[leg].copy()
            f[2] = max(0.0, f[2])  # push, not pull
            tau_lambda_nv += jacp.T @ f

        tau_lambda = np.zeros(12)
        for i in range(12):
            dofadr = dofaddrs[i]
            if 0 <= dofadr < nv:
                tau_lambda[i] = tau_lambda_nv[dofadr]

        # ---- blend & clamp ----
        w_lambda = 0.8
        tau_total_joint = np.clip(tau_pd + w_lambda * tau_lambda, -40.0, 40.0)

        # ---- map torque -> ctrl via gear and clamp ----
        if not hasattr(self, "_joint_to_actuator") or self._joint_to_actuator is None:
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
            try:
                gear = float(self.model.actuator_gear[act_idx, 0])
                if not np.isfinite(gear) or abs(gear) < 1e-9:
                    gear = 1.0
            except Exception:
                gear = 1.0

            ctrl = float(tau_total_joint[i] / gear)
            try:
                cmin, cmax = self.model.actuator_ctrlrange[act_idx]
                ctrl = float(np.clip(ctrl, cmin, cmax))
            except Exception:
                ctrl = float(np.clip(ctrl, -1.0, 1.0))

            self.data.ctrl[act_idx] = ctrl
            max_ctrl_mag = max(max_ctrl_mag, abs(ctrl))

        try:
            overall_max = float(np.max(self.model.actuator_ctrlrange[:, 1]))
            if max_ctrl_mag > 0.9 * overall_max:
                print(f"[WARN] |ctrl| near limit (max |ctrl|={max_ctrl_mag:.2f}). "
                    f"Consider lowering kp/kd or w_lambda.")
        except Exception:
            pass


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