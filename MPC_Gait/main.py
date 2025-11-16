# main.py - FIXED VERSION

import numpy as np
import sys

# Import components from other files
from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from mpc_controller import MPCController
from simulation import RobotSimulation
from utils import generate_reference_trajectory

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
    IN_VERIFICATION = False
    WHEELS_ON = True
    
    # Target velocity [vx, vy, vz, wx, wy, wz]
    TARGET_VELOCITY = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])  # Slower for stability
    
    # ==========================================
    # SETUP
    # ==========================================
    
    print("\n[1/5] Loading parameters...")
    
    params = MPCParameters(
        robot_mass=30.0,
        robot_inertia=np.diag([0.5, 1.0, 1.0]),
        horizon_length=0.8,
        num_nodes=20,
        control_freq=15.0,
        weight_position=100.0,
        weight_orientation=50.0,
        weight_linear_velocity=10.0,
        weight_angular_velocity=5.0,
        weight_joint_position=1.0,
        weight_contact_force=0.01,
        weight_joint_velocity=0.1,
        friction_coeff=0.7,
        max_joint_velocity=5.0,  # Reduced for stability
        max_contact_force=500.0,
        swing_height=0.08,  # Lower swing
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
        sim = RobotSimulation(ROBOT_XML_PATH, params, use_gui=USE_GUI, verify=IN_VERIFICATION, wheels=WHEELS_ON)
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
    # CALCULATE TIMING PARAMETERS FIRST
    # ==========================================
    
    dt_control = 1.0 / params.control_freq
    num_steps = int(SIMULATION_TIME / dt_control)
    
    # Get simulation timestep
    try:
        sim_timestep = sim.model.opt.timestep
    except Exception:
        sim_timestep = 0.001
    
    if sim_timestep <= 0:
        sim_timestep = 0.001
        print(f"  ⚠️ Warning: Invalid sim_timestep, defaulting to {sim_timestep}s")

    num_sim_steps_per_control = int(dt_control / sim_timestep)
    
    # ==========================================
    # STABILIZATION PHASE
    # ==========================================
    
    print("\n[Pre-flight] Stabilizing robot for 2 seconds...")
    print("  ℹ Applying gravity compensation only...")
    
    stabilization_steps = int(2.0 * params.control_freq)  # 2 seconds at 50Hz
    
    for step in range(stabilization_steps):
        # Get current state
        x_current = sim.get_state()
        
        # Zero velocity command (just hold position)
        u_stabilize = np.zeros(24)
        
        # Apply control with all feet in contact
        contact_states_stable = np.ones(4)
        
        for _ in range(num_sim_steps_per_control):
            sim.apply_control(u_stabilize, contact_states_stable)
            sim.step_physics()
        
        sim.render()
    
    print("  ✓ Robot stabilized and ready!")
    print("  ✓ Starting walking controller...\n")
    
    # ==========================================
    # MAIN CONTROL LOOP
    # ==========================================
    
    print("\n[5/5] Starting control loop...")
    print("\n" + "="*60)
    print("SIMULATION RUNNING")
    print("="*60)
    print(f"  ✓ Control @ {params.control_freq}Hz, Sim @ {1.0/sim_timestep:.0f}Hz")
    print(f"  ✓ Running {num_sim_steps_per_control} sim steps per control step.")
    
    # Data logging
    state_history = []
    control_history = []
    cost_history = []
    
    # Initial contact states - start with all feet on ground
    current_contact_states = np.ones(4)
    
    # Set gait mode
    controller.gait_gen.set_gait_mode('hybrid_trot')  # Use trot for forward motion
    
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
            
            # Update gait sequence (every 5 control steps = 0.1s)
            if step % 5 == 0:
                # Use small dummy utilities (gait is pattern-based)
                utilities = np.ones(4) * 0.8
                controller.gait_gen.update_gait(utilities, dt_control * 5)
                current_contact_states = controller.gait_gen.contact_states.copy()
            
            # Contact schedule for MPC
            contact_schedule = np.tile(current_contact_states, (params.num_nodes, 1))
            
            # Solve MPC
            u_optimal, x_predicted = controller.solve_mpc(
                x_current, x_ref_traj, contact_schedule
            )
            
            # Apply control (with contact gating)
            u_apply = u_optimal[0]
            control_history.append(u_apply.copy())
            
            # Zero out forces for swing legs
            for leg_idx in range(4):
                if current_contact_states[leg_idx] == 0:
                    u_apply[leg_idx*3:(leg_idx+1)*3] = 0.0
            
            # Compute cost
            cost = controller.compute_cost(x_current, u_apply, 
                                          x_ref_traj[0], np.zeros(24))
            cost_history.append(cost)
            
            # Apply control and step physics (inner loop)
            for _ in range(num_sim_steps_per_control):
                sim.apply_control(u_apply, current_contact_states)
                sim.step_physics()
            
            # Render
            sim.render()

            # Print status (every second)
            if step % int(params.control_freq) == 0:
                pos = x_current[3:6]
                vel = x_current[9:12]
                contact_str = ''.join(['█' if c else '░' for c in current_contact_states])
                
                print(f"t={t:6.2f}s | "
                      f"pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                      f"vel=[{vel[0]:5.2f}, {vel[1]:5.2f}, {vel[2]:5.2f}] | "
                      f"contacts={contact_str} | "
                      f"cost={cost:8.2f}")
    
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
    
    To run:
    1. Save all files (config.py, dynamics.py, etc.) in the same directory.
    2. Ensure you have a valid MuJoCo XML file and update ROBOT_XML_PATH in main.py
       OR set USE_SIMPLE_ROBOT = True to use the built-in model.
    3. Run: python main.py
    
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
        import numpy
        print("✓ All dependencies found")
        print(f"  MuJoCo version: {mujoco.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install mujoco numpy scipy")
        sys.exit(1)
    
    # Run main controller
    main()