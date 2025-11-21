# main.py 

"""
Whole-Body Model Predictive Control for Wheeled-Legged Quadruped Robot

This is the main executable for running MPC-based locomotion control on a
quadrupedal robot in MuJoCo simulation. Implements the control architecture
from Bjelonic et al. 2021 with hybrid stance/swing control.

Key Features:
    - Single Rigid Body Dynamics (SRBD) reduced-order model
    - Nonlinear MPC with iLQR solver
    - Hybrid trot gait generation
    - IK-based swing leg control
    - Stance leg force optimization
    - Real-time debugging and visualization

Architecture:
    1. MPC Layer: Optimizes contact forces for base motion tracking
    2. IK Layer: Generates joint trajectories for swing legs
    3. Low-level PD: Tracks joint positions with gravity compensation
    
Control Flow:
    State → MPC → Forces (stance) + IK → Positions (swing) → PD → Torques

Usage:
    python main.py
    
    Adjust configuration in main() function:
    - ROBOT_XML_PATH: Path to robot model
    - TARGET_VELOCITY: Desired [vx, vy, vz, ωx, ωy, ωz]
    - SIMULATION_TIME: Duration in seconds
    - USE_GUI: Enable/disable 3D visualization

Dependencies:
    - numpy: Numerical operations
    - scipy: Rotation representations and IK
    - mujoco: Physics simulation
    - config: MPC parameters
    - dynamics: SRBD model
    - mpc_controller: iLQR optimizer
    - simulation: MuJoCo wrapper
    - kinematics: Inverse kinematics
    - utils: Trajectory generation

References:
    Bjelonic et al. "Whole-Body MPC for a Dynamically Stable Mobile Manipulator"
    IEEE RA-L 2021. https://doi.org/10.1109/LRA.2021.3068908

Author: Implementation based on paper methodology
Date: 2024
"""

import numpy as np
import sys
from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from mpc_controller import MPCController
from simulation import RobotSimulation
from scipy.spatial.transform import Rotation
from kinematics import ik_sagittal
from utils import generate_reference_trajectory
from debug_swing import debug_swing_detailed, check_ik_reachability, test_ik_at_swing_phases, debug_gait_pattern

def compute_stance_control(leg_idx, q_now):
    """
    Generate position-holding control for stance legs.
    
    Applies proportional feedback to maintain nominal stance configuration,
    providing compliance during stance phase without fighting MPC forces.
    
    Args:
        leg_idx (int): Leg index (0=FL, 1=FR, 2=HL, 3=HR)
        q_now (np.ndarray): Current joint positions [hip, thigh, shank] in radians
    
    Returns:
        np.ndarray: Joint velocity commands [3] in rad/s
    
    Control Law:
        u_joint = K_p * (q_nominal - q_current)
        
    Gains:
        K_p = 3.0 rad/s per rad (gentle compliance)
    
    Notes:
        - No derivative term to avoid coupling with MPC force optimization
        - Nominal stance: hip=0°, thigh=34.4°, shank=-73.9°
        - Provides soft position spring without velocity feedback
        - Allows MPC forces to dominate motion
    
    Example:
        >>> q_current = np.array([0.1, 0.7, -1.3])
        >>> u = compute_stance_control(0, q_current)
        >>> # Returns velocity to drive toward [0.0, 0.6, -1.29]
    """
    # Nominal stance configuration
    q_nom = np.array([0.0, 0.6, -1.29])
    
    # Simple proportional control (no derivative term)
    kp_stance = 10.0
    
    u_joint = kp_stance * (q_nom - q_now)
    
    return u_joint

def compute_swing_ik_simple(leg_idx, swing_phase, q_now):
    """
        Generate IK-based swing trajectory with bezier-like foot path.
        
        Computes desired joint positions for swing legs using inverse kinematics
        on a smooth foot trajectory with parabolic clearance profile.
        
        Args:
            leg_idx (int): Leg index (0-3), currently unused but kept for extensibility
            swing_phase (float): Normalized swing phase in [0, 1]
                                0 = lift-off, 0.5 = apex, 1 = touchdown
            q_now (np.ndarray): Current joint positions [hip, thigh, shank] in radians
        
        Returns:
            np.ndarray: Joint velocity commands [3] in rad/s
        
        Trajectory Design:
            - Horizontal: Linear interpolation from x0 to x0+step_length
            - Vertical: Parabolic arc with peak at phase=0.5
            - Smoothing: 5th-order polynomial (6s^5 - 15s^4 + 10s^3)
        
        Parameters:
            - step_length: 12cm (0.12m) forward step
            - clearance: 5cm (0.05m) maximum height
            - x0: 5cm (0.05m) starting position (behind hip)
            - Safety margin: 90% of kinematic reach limit
        
        Control:
            K_p = 12.0 rad/s per rad (stiff tracking)
        
        Notes:
            - Uses 2D sagittal plane IK (hip always 0)
            - Clips to joint limits for safety
            - Falls back to conservative pose if IK fails
            - Higher gain than stance for precise tracking
        
        Raises:
            ValueError: If joint angles exceed physical limits (caught internally)
        
        Example:
            >>> q_current = np.array([0.0, 0.5, -1.0])
            >>> u = compute_swing_ik_simple(0, 0.5, q_current)
            >>> # Returns velocity to reach apex position
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


def main():
    """
        Main control loop for MPC-based quadruped locomotion.
        
        Execution Phases:
            1. Configuration and parameter loading
            2. Dynamics model initialization (SRBD)
            3. MPC controller setup (iLQR)
            4. MuJoCo simulation initialization
            5. Stabilization phase (1 second)
            6. Main control loop
            7. Statistics and cleanup
        
        Control Architecture:
            For each control step (20ms @ 50Hz):
                1. Read state from simulation
                2. Generate reference trajectory
                3. Update gait state (every 0.1s)
                4. Compute stance control (position holding)
                5. Compute swing control (IK trajectory)
                6. Solve MPC for contact forces
                7. Combine MPC forces + IK positions
                8. Apply to simulation
                9. Step physics multiple times
                10. Render visualization
        
        Configuration Variables:
            ROBOT_XML_PATH (str): Path to MuJoCo XML model
            USE_SIMPLE_ROBOT (bool): Use programmatic model (deprecated)
            USE_GUI (bool): Enable 3D visualization
            SIMULATION_TIME (float): Total duration in seconds
            IN_VERIFICATION (bool): Debug mode (no control)
            WHEELS_ON (bool): Use wheeled robot variant
            ENABLE_GAIT_DEBUG (bool): Print gait diagnostics
            TARGET_VELOCITY (np.ndarray): Desired [vx, vy, vz, ωx, ωy, ωz]
        
        MPC Parameters:
            - Horizon: 0.8s with 20 nodes (40ms per node)
            - Control frequency: 50 Hz
            - Weights: position=100, orientation=50, velocity=10
            - Friction coefficient: 0.7
            - Max joint velocity: 8 rad/s
            - Max contact force: 500 N
        
        Data Collection:
            - state_history: All state vectors
            - control_history: All control inputs
            - cost_history: Tracking cost at each step
        
        Keyboard Interrupt:
            Press Ctrl+C to stop simulation gracefully
        
        Returns:
            None
        
        Raises:
            KeyboardInterrupt: On user interrupt or viewer close
            ImportError: If required dependencies missing
            FileNotFoundError: If robot XML not found and generation fails
        
        Example:
            >>> if __name__ == "__main__":
            ...     main()  # Runs full simulation
        
        Notes:
            - Physics substeps: Typically 20 per control step (1ms physics dt)
            - Gait: Hybrid trot with 0.3s swing duration
            - IK step length: 12cm with 5cm clearance
            - Stance gains: K_p=3.0 (gentle compliance)
            - Swing gains: K_p=12.0 (stiff tracking)
    """
    
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
    ENABLE_GAIT_DEBUG = False
    
    # REDUCED velocity for debugging
    TARGET_VELOCITY = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])  # [vx, vy, vz, wx, wy, wz]
    
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
    controller.gait_gen.set_gait_mode('hybrid_walk')

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
            if ENABLE_GAIT_DEBUG and step % 50 == 0:  # Every second
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
                # Apply adaptive thrusters if in wheeled mode
                if WHEELS_ON:
                    # Adaptive thrust based on gait state
                    sim.apply_adaptive_thruster_forces(
                        contact_states=current_contact_states,
                        base_thrust_ratio=0.5,   # 50% base support
                        swing_boost=0.3          # +30% per swinging leg
                    )
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