# main.py 

import numpy as np
import sys
from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from mpc_controller import MPCController
from simulation import RobotSimulation
from scipy.spatial.transform import Rotation
from kinematics import ik_sagittal
from utils import generate_reference_trajectory
from debug_swing import debug_swing_detailed, check_ik_reachability, test_ik_at_swing_phases

def compute_stance_control(leg_idx, q_now):
    """
    Gentle position hold for stance legs
    SIMPLIFIED: No velocity feedback to avoid state vector issues
    
    Args:
        leg_idx: Leg index (0-3)
        q_now: Current joint positions [3]
    
    Returns:
        u_joint: Joint velocity command [3]
    """
    # Nominal stance configuration
    q_nom = np.array([0.0, 0.6, -1.29])
    
    # Simple proportional control (no derivative term)
    kp_stance = 3.0
    
    u_joint = kp_stance * (q_nom - q_now)
    
    return u_joint

def compute_swing_ik_simple(leg_idx, swing_phase, q_now):
    """
    TUNED VERSION: Bigger steps, more clearance
    """
    
    s = swing_phase
    ss = 6*s**5 - 15*s**4 + 10*s**3
    
    L1, L2 = 0.22, 0.22
    
    # TUNED FOR FORWARD LOCOMOTION
    step_length = 0.12   # 12cm steps (was 6cm)
    clearance = 0.05     # 5cm lift (was 3cm)
    
    # Start position (lift-off)
    x0 = 0.05            # Start further back (was 0.08)
    z0 = -(L1 + L2 - 0.10)  # Less extended initially
    
    # End position (touch-down)
    x1 = x0 + step_length
    z1 = z0
    
    # Interpolate
    x = x0 + (x1 - x0) * ss
    z = z0 + (z1 - z0) * ss + clearance * (1 - (2*s - 1)**2)
    z = -abs(z)
    
    # SAFETY: Limit to 90% of max reach
    dist = np.sqrt(x**2 + z**2)
    max_reach = L1 + L2
    
    if dist > max_reach * 0.90:
        scale = (max_reach * 0.88) / dist
        x *= scale
        z *= scale
    
    # IK with error handling
    try:
        q_thigh, q_shank = ik_sagittal(L1, L2, x, z)
        
        # Sanity check
        if abs(q_thigh) > 1.6 or abs(q_shank) > 2.2:
            raise ValueError("Joint angles extreme")
            
    except:
        # Conservative fallback
        q_thigh = 0.7
        q_shank = -1.3
    
    q_des = np.array([0.0, q_thigh, -abs(q_shank)])
    
    # INCREASED GAIN for better tracking
    kp_swing = 12.0  # Was 8.0
    u_joint = kp_swing * (q_des - q_now)
    
    return u_joint


def debug_gait_pattern(step, t, x_current, contact_states, dynamics):
    """
    FIXED: Properly computes foot positions in WORLD frame
    """
    if step % 50 != 0:  # 1 Hz at 50Hz control
        return

    theta = x_current[0:3]        # roll, pitch, yaw
    p_base = x_current[3:6]       # base position in world
    q_j = x_current[12:24]        # joint angles

    # Compute body-to-world rotation
    R_WB = Rotation.from_euler('xyz', theta).as_matrix()

    # Get contact positions in BODY frame (relative to COM)
    contact_points_B = dynamics.compute_contact_positions(q_j, theta)

    # Convert to WORLD frame
    foot_z_world = []
    foot_world_coords = []
    for fp_B in contact_points_B:
        # Transform: p_foot_world = p_base + R_WB @ p_foot_body
        foot_W = p_base + R_WB @ fp_B
        foot_world_coords.append(foot_W)
        foot_z_world.append(foot_W[2])

    # Format contact states
    contact_str = ''.join(['█' if c else '░' for c in contact_states])

    # Diagonal pattern checks
    fl, fr, hl, hr = contact_states
    fl_rh_sync = (fl == hr)
    fr_hl_sync = (fr == hl)
    diagonals_opposite = (fl != fr)
    pattern_ok = fl_rh_sync and fr_hl_sync and diagonals_opposite

    # Print analysis
    print(f"\n[DEBUG t={t:.2f}s] Gait Analysis:")
    print(f"  Contacts: {contact_str} | FL:{fl} FR:{fr} HL:{hl} HR:{hr}")
    print(f"  Pattern: {'✓ OK' if pattern_ok else '✗ WRONG'} | "
          f"FL+RH:{fl_rh_sync} FR+HL:{fr_hl_sync} Opp:{diagonals_opposite}")
    
    # WORLD frame heights (FIXED!)
    print(f"  Foot Z (world): FL={foot_z_world[0]:.3f} FR={foot_z_world[1]:.3f} "
          f"HL={foot_z_world[2]:.3f} HR={foot_z_world[3]:.3f}")
    
    # BODY frame heights (for reference)
    body_z = [fp[2] for fp in contact_points_B]
    print(f"  Foot Z (body):  FL={body_z[0]:.3f} FR={body_z[1]:.3f} "
          f"HL={body_z[2]:.3f} HR={body_z[3]:.3f}")

    # Check swing clearance
    ground_level = 0.0
    for leg_idx, name in enumerate(['FL', 'FR', 'HL', 'HR']):
        if contact_states[leg_idx] == 0:  # Swing leg
            height_above_ground = foot_z_world[leg_idx] - ground_level
            if height_above_ground < 0.03:  # Should be >3cm
                print(f"    ⚠️ {name} in swing but only {height_above_ground*100:.1f}cm above ground!")
            else:
                print(f"    ✓ {name} swing clearance: {height_above_ground*100:.1f}cm")

    print(f"  Base: x={p_base[0]:.3f} y={p_base[1]:.3f} z={p_base[2]:.3f}")
    print(f"  Base orientation: roll={np.rad2deg(theta[0]):.1f}° "
          f"pitch={np.rad2deg(theta[1]):.1f}° yaw={np.rad2deg(theta[2]):.1f}°")


def main():
    """Main driver code for MPC-based walking controller"""
    
    print("="*60)
    print("Whole-Body MPC for Wheeled-Legged Robots - MuJoCo")
    print("Based on: Bjelonic et al. 2021")
    print("="*60)
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    ROBOT_XML_PATH = "/home/poison-arrow/MPC_Gait/anymal_simplified.xml"
    USE_SIMPLE_ROBOT = False
    USE_GUI = True
    SIMULATION_TIME = 30.0
    IN_VERIFICATION = False
    WHEELS_ON = True
    ENABLE_GAIT_DEBUG = True
    
    # REDUCED velocity for debugging
    TARGET_VELOCITY = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.2 m/s
    
    # ==========================================
    # SETUP
    # ==========================================
    
    print("\n[1/5] Loading parameters...")
    
    params = MPCParameters(
        robot_mass=15.0,
        robot_inertia=np.diag([0.5, 1.0, 1.0]),
        horizon_length=0.8,
        num_nodes=20,
        control_freq=50.0,  # Back to 50Hz
        weight_position=100.0,
        weight_orientation=50.0,
        weight_linear_velocity=10.0,
        weight_angular_velocity=5.0,
        weight_joint_position=1.0,
        weight_contact_force=0.01,
        weight_joint_velocity=0.1,
        friction_coeff=0.7,
        max_joint_velocity=8.0,  # Increased for swing
        max_contact_force=500.0,
        swing_height=0.06,  # Reduced for safety
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
    
    # ==========================================
    # INITIALIZE MPC
    # ==========================================
    
    print("\n[3/5] Initializing MPC controller...")
    controller = MPCController(dynamics, params)
    print(f"  ✓ MPC iterations: {controller.max_iterations}")
    
    # ==========================================
    # SETUP SIMULATION
    # ==========================================
    
    print("\n[4/5] Setting up MuJoCo simulation...")
    
    try:
        sim = RobotSimulation(ROBOT_XML_PATH, params, use_gui=USE_GUI, 
                             verify=IN_VERIFICATION, wheels=WHEELS_ON)
        print(f"  ✓ Robot loaded successfully")
        print(f"  ✓ Number of joints: {len(sim.joint_indices)}")
    except Exception as e:
        print(f"  ✗ Error loading robot: {e}")
        sys.exit(1)
    
    # ==========================================
    # USER CONFIRMATION
    # ==========================================
    
    print("\n" + "="*60)
    print("ROBOT LOADED - READY TO START")
    print("="*60)
    print(f"\n📋 Configuration Summary:")
    print(f"  • Robot mass: {params.robot_mass} kg")
    print(f"  • Control frequency: {params.control_freq} Hz")
    print(f"  • Target velocity: {TARGET_VELOCITY[0]:.2f} m/s forward")
    print(f"  • Swing height: {params.swing_height:.2f}m")
    
    print("\n" + "="*60)
    input("Press ENTER to start the MPC controller... ")
    print("="*60)
    
    # ==========================================
    # CALCULATE TIMING
    # ==========================================
    
    dt_control = 1.0 / params.control_freq
    num_steps = int(SIMULATION_TIME / dt_control)
    
    sim_timestep = sim.model.opt.timestep
    if sim_timestep <= 0:
        sim_timestep = 0.001
    num_sim_steps_per_control = int(dt_control / sim_timestep)
    
    # ==========================================
    # STABILIZATION PHASE
    # ==========================================
    
    print("\n[Pre-flight] Stabilizing robot for 1 second...")
    stabilization_steps = int(1.0 * params.control_freq)
    
    for step in range(stabilization_steps):
        x_current = sim.get_state()
        u_stabilize = np.zeros(24)
        contact_states_stable = np.ones(4)
        
        for _ in range(num_sim_steps_per_control):
            sim.apply_control_new(u_stabilize, contact_states_stable)
            sim.step_physics()
        sim.render()
    
    print("  ✓ Robot stabilized!\n")
    
    # At startup, before control loop:
    print("\n[Diagnostic] Testing IK reachability...")
    test_ik_at_swing_phases()

    # ==========================================
    # MAIN CONTROL LOOP
    # ==========================================
    
    print("\n[5/5] Starting control loop...")
    print("\n" + "="*60)
    print("SIMULATION RUNNING")
    print("="*60)
    
    state_history = []
    control_history = []
    cost_history = []
    
    current_contact_states = np.ones(4)
    controller.gait_gen.set_gait_mode('hybrid_trot')

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
            
            # Update gait (every 0.1s)
            if step % 5 == 0:
                utilities = np.ones(4) * 0.8
                controller.gait_gen.update_gait(utilities, dt_control * 5)
                current_contact_states = controller.gait_gen.contact_states.copy()
            
            # ===== DEBUG GAIT =====
            if ENABLE_GAIT_DEBUG:
                debug_gait_pattern(step, t, x_current, current_contact_states, dynamics)
                debug_swing_detailed(step, t, x_current, current_contact_states, dynamics, controller)
            
            # Contact schedule for MPC
            contact_schedule = np.tile(current_contact_states, (params.num_nodes, 1))

            u_joint = np.zeros(12)
            q_now = x_current[12:24].copy()  # Joint positions

            for leg in range(4):
                i = 3 * leg
                q_leg_now = q_now[i:i+3]
                
                if current_contact_states[leg] == 0:  # SWING LEG
                    swing_phase = controller.gait_gen.swing_phase[leg]
                    u_joint[i:i+3] = compute_swing_ik_simple(leg, swing_phase, q_leg_now)
                else:  # STANCE LEG
                    u_joint[i:i+3] = compute_stance_control(leg, q_leg_now)

            # Clip to limits
            u_joint = np.clip(u_joint, -params.max_joint_velocity, params.max_joint_velocity)

            # Solve MPC
            u_optimal, x_predicted = controller.solve_mpc(
                x_current, x_ref_traj, contact_schedule
            )

            # Apply control (CREATE u_apply HERE!)
            u_apply = u_optimal[0].copy()
            u_apply[12:24] = u_joint  # Replace MPC joint commands with IK commands

            control_history.append(u_apply.copy())

            # Zero forces on swing legs
            for leg_idx in range(4):
                if current_contact_states[leg_idx] == 0:
                    u_apply[leg_idx*3:(leg_idx+1)*3] = 0.0

            # NOW you can debug u_apply
            if step % 50 == 0:  # Every second
                print(f"\n[CONTROL CHECK]")
                print(f"  u_joint computed: {u_joint[:6]}")
                print(f"  q_now: {q_now[:6]}")
                print(f"  u_apply[12:18]: {u_apply[12:18]}")  # ✓ Now it exists!

            # Compute cost
            cost = controller.compute_cost(x_current, u_apply, 
                                        x_ref_traj[0], np.zeros(24))
            cost_history.append(cost)

            # Apply control and step physics
            for _ in range(num_sim_steps_per_control):
                sim.apply_control_new(u_apply, current_contact_states)
                sim.step_physics()
            sim.render()

            # Print status
            if step % int(params.control_freq) == 0:
                pos = x_current[3:6]
                vel = x_current[9:12]
                contact_str = ''.join(['█' if c else '░' for c in current_contact_states])
                
                print(f"t={t:6.2f}s | "
                      f"pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                      f"vel=[{vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}] | "
                      f"contacts={contact_str} | cost={cost:8.2f}")
    
    except KeyboardInterrupt:
        print("\n\nSIMULATION INTERRUPTED BY USER\n")
    
    finally:
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        
        if len(state_history) > 0:
            state_history = np.array(state_history)
            
            print(f"\nStatistics:")
            print(f"  Total steps: {len(state_history)}")
            print(f"  Average cost: {np.mean(cost_history):.2f}")
            print(f"  Final position: [{state_history[-1][3]:.2f}, "
                  f"{state_history[-1][4]:.2f}, {state_history[-1][5]:.2f}]")
            print(f"  Distance traveled: "
                  f"{np.linalg.norm(state_history[-1][3:5] - state_history[0][3:5]):.2f} m")
        
        sim.close()
        print("\n✓ Simulation closed")
        print("="*60)


if __name__ == "__main__":
    try:
        import mujoco
        import scipy
        print("✓ All dependencies found")
        print(f"  MuJoCo version: {mujoco.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        sys.exit(1)
    
    main()