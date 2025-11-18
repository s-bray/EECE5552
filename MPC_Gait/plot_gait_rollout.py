"""
Debug script to verify gait patterns and foot trajectories
This simulates the MPC rollout and checks if diagonal legs are alternating correctly by plotting results
"""

import numpy as np
import matplotlib.pyplot as plt
from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from gait_generator import GaitSequenceGenerator

def simulate_gait_rollout(duration=5.0, dt=0.02):
    """
    Simulate gait pattern and foot positions over time
    
    Args:
        duration: Simulation time in seconds
        dt: Time step (20ms = 50Hz control rate)
    """
    
    print("="*70)
    print("GAIT PATTERN ROLLOUT VERIFICATION")
    print("="*70)
    
    # Initialize
    params = MPCParameters()
    dynamics = SingleRigidBodyDynamics(params)
    gait_gen = GaitSequenceGenerator(params)
    
    # Set gait mode
    gait_mode = 'hybrid_trot'  # Test trot pattern
    gait_gen.set_gait_mode(gait_mode)
    print(f"\n[Config] Gait mode: {gait_mode}")
    print(f"[Config] Duration: {duration}s")
    print(f"[Config] Time step: {dt}s ({1.0/dt:.0f}Hz)")
    print(f"[Config] Expected pattern: Diagonal pairs (FL+RH, FR+HL)")
    
    # Initial state (standing pose)
    x_initial = np.array([
        0.0, 0.0, 0.0,           # theta (roll, pitch, yaw)
        0.0, 0.0, 0.35,          # p (x, y, z)
        0.0, 0.0, 0.0,           # omega (angular velocity)
        0.3, 0.0, 0.0,           # v (linear velocity) - 0.3 m/s forward
        0.0, 0.7, -1.4,          # FL joints
        0.0, 0.7, -1.4,          # FR joints
        0.0, 0.7, -1.4,          # HL joints
        0.0, 0.7, -1.4           # HR joints
    ])
    
    # Storage for trajectory
    num_steps = int(duration / dt)
    time_array = np.zeros(num_steps)
    contact_history = np.zeros((num_steps, 4))  # 4 legs
    foot_positions = np.zeros((num_steps, 4, 3))  # 4 legs × (x,y,z)
    swing_phases = np.zeros((num_steps, 4))
    base_positions = np.zeros((num_steps, 3))
    
    # Current state
    x_current = x_initial.copy()
    
    print(f"\n[Rollout] Simulating {num_steps} steps...")
    print("\nTime | Contacts | FL_z | FR_z | HL_z | HR_z | Pattern Check")
    print("-" * 70)
    
    for step in range(num_steps):
        t = step * dt
        time_array[step] = t
        
        # Update gait (every 5 steps = 0.1s like in main.py)
        if step % 5 == 0:
            utilities = np.ones(4) * 0.8  # Dummy utilities for fixed pattern
            gait_gen.update_gait(utilities, dt * 5)
        
        # Store contact states
        contact_history[step] = gait_gen.contact_states.copy()
        swing_phases[step] = gait_gen.swing_phase.copy()
        
        # Compute foot positions using FK
        theta = x_current[0:3]
        q_j = x_current[12:24]
        foot_pos = dynamics.compute_contact_positions(q_j, theta)
        
        for leg_idx in range(4):
            foot_positions[step, leg_idx] = foot_pos[leg_idx]
        
        # Store base position
        base_positions[step] = x_current[3:6].copy()
        
        # Simple dynamics update (just move forward)
        x_current[3] += x_current[9] * dt  # x position
        
        # Print status every 0.5 seconds
        if step % int(0.5 / dt) == 0:
            contacts = contact_history[step]
            contact_str = ''.join(['█' if c else '░' for c in contacts])
            
            # Check if diagonal pattern is correct
            diagonal1 = (contacts[0] == contacts[3])  # FL and RH same
            diagonal2 = (contacts[1] == contacts[2])  # FR and HL same
            diagonal_opposite = (contacts[0] != contacts[1])  # FL and FR opposite
            
            pattern_ok = diagonal1 and diagonal2 and diagonal_opposite
            status = "✓ OK" if pattern_ok else "✗ WRONG"
            
            print(f"{t:4.2f} | {contact_str} | "
                  f"{foot_pos[0][2]:5.3f} | {foot_pos[1][2]:5.3f} | "
                  f"{foot_pos[2][2]:5.3f} | {foot_pos[3][2]:5.3f} | {status}")
    
    print("-" * 70)
    
    return {
        'time': time_array,
        'contacts': contact_history,
        'foot_positions': foot_positions,
        'swing_phases': swing_phases,
        'base_positions': base_positions,
        'gait_gen': gait_gen
    }


def analyze_results(results):
    """Analyze and visualize the rollout results"""
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    time = results['time']
    contacts = results['contacts']
    foot_pos = results['foot_positions']
    
    # Check contact pattern
    print("\n[1] CONTACT PATTERN ANALYSIS")
    
    leg_names = ['FL', 'FR', 'HL', 'HR']
    
    # Count contact vs swing time for each leg
    for leg_idx, name in enumerate(leg_names):
        contact_time = np.sum(contacts[:, leg_idx]) * (time[1] - time[0])
        total_time = time[-1]
        duty_factor = contact_time / total_time
        print(f"  {name}: {duty_factor*100:.1f}% stance time (expected: ~60% for trot)")
    
    # Check diagonal coordination
    print("\n[2] DIAGONAL PAIR COORDINATION")
    
    # FL + RH should be synchronized
    fl_rh_sync = np.sum(contacts[:, 0] == contacts[:, 3]) / len(contacts)
    print(f"  FL + RH synchronized: {fl_rh_sync*100:.1f}% (should be ~100%)")
    
    # FR + HL should be synchronized
    fr_hl_sync = np.sum(contacts[:, 1] == contacts[:, 2]) / len(contacts)
    print(f"  FR + HL synchronized: {fr_hl_sync*100:.1f}% (should be ~100%)")
    
    # Diagonal pairs should be opposite
    diagonal_opposite = np.sum(contacts[:, 0] != contacts[:, 1]) / len(contacts)
    print(f"  Diagonal pairs opposite: {diagonal_opposite*100:.1f}% (should be ~100%)")
    
    # Overall pattern correctness
    if fl_rh_sync > 0.95 and fr_hl_sync > 0.95 and diagonal_opposite > 0.95:
        print("\n  ✓ GAIT PATTERN IS CORRECT!")
    else:
        print("\n  ✗ GAIT PATTERN HAS ERRORS!")
        print("  → This means FK or gait generator has issues")
    
    # Check foot height variations
    print("\n[3] FOOT HEIGHT ANALYSIS (Z-axis)")
    
    for leg_idx, name in enumerate(leg_names):
        z_positions = foot_pos[:, leg_idx, 2]
        z_min = np.min(z_positions)
        z_max = np.max(z_positions)
        z_range = z_max - z_min
        
        swing_steps = np.sum(contacts[:, leg_idx] == 0)
        
        print(f"  {name}: z_min={z_min:.3f}m, z_max={z_max:.3f}m, "
              f"range={z_range:.3f}m, swings={swing_steps}")
        
        if z_range < 0.01 and swing_steps > 0:
            print(f"    ⚠️ Warning: {name} barely lifts during swing! "
                  f"FK might be incorrect.")
    
    # Check forward motion
    print("\n[4] BASE MOTION ANALYSIS")
    base_pos = results['base_positions']
    distance_traveled = base_pos[-1, 0] - base_pos[0, 0]
    avg_velocity = distance_traveled / time[-1]
    
    print(f"  Distance traveled: {distance_traveled:.3f}m in {time[-1]:.1f}s")
    print(f"  Average velocity: {avg_velocity:.3f}m/s (commanded: 0.30m/s)")
    
    if abs(avg_velocity - 0.30) < 0.05:
        print("  ✓ Velocity tracking is reasonable")
    else:
        print("  ✗ Velocity tracking is poor - check dynamics")


def plot_results(results):
    """Create visualization plots"""
    
    time = results['time']
    contacts = results['contacts']
    foot_pos = results['foot_positions']
    swing_phases = results['swing_phases']
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    leg_names = ['FL (Front Left)', 'FR (Front Right)', 
                 'HL (Hind Left)', 'HR (Hind Right)']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Plot 1: Contact states
    ax = axes[0]
    for leg_idx, (name, color) in enumerate(zip(leg_names, colors)):
        ax.plot(time, contacts[:, leg_idx] + leg_idx * 1.2, 
                label=name, color=color, linewidth=2)
    ax.set_ylabel('Contact State')
    ax.set_title('Contact Pattern Over Time (Should show diagonal alternation)')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 5)
    
    # Plot 2: Foot heights (Z position)
    ax = axes[1]
    for leg_idx, (name, color) in enumerate(zip(leg_names, colors)):
        z_positions = foot_pos[:, leg_idx, 2]
        ax.plot(time, z_positions, label=name, color=color, linewidth=2)
    ax.set_ylabel('Foot Height (m)')
    ax.set_title('Foot Height Trajectories (Swing legs should lift)')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='Ground')
    
    # Plot 3: Swing phases
    ax = axes[2]
    for leg_idx, (name, color) in enumerate(zip(leg_names, colors)):
        ax.plot(time, swing_phases[:, leg_idx], label=name, color=color, linewidth=2)
    ax.set_ylabel('Swing Phase (0-1)')
    ax.set_title('Swing Phase Progress')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Forward motion
    ax = axes[3]
    for leg_idx, (name, color) in enumerate(zip(leg_names, colors)):
        x_positions = foot_pos[:, leg_idx, 0]
        ax.plot(time, x_positions, label=name, color=color, linewidth=2)
    ax.set_ylabel('X Position (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Foot Forward Motion')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gait_rollout_analysis.png', dpi=150)
    print("\n[Plot] Saved to: gait_rollout_analysis.png")
    plt.show()


def verify_forward_kinematics():
    """
    Verify FK is computing correct foot positions
    """
    print("\n" + "="*70)
    print("FORWARD KINEMATICS VERIFICATION")
    print("="*70)
    
    params = MPCParameters()
    dynamics = SingleRigidBodyDynamics(params)
    
    # Test case: Standing pose
    q_j_standing = np.array([
        0.0, 0.7, -1.4,  # FL
        0.0, 0.7, -1.4,  # FR
        0.0, 0.7, -1.4,  # HL
        0.0, 0.7, -1.4   # HR
    ])
    
    theta = np.array([0.0, 0.0, 0.0])  # No base rotation
    
    print("\n[Test] Standing pose: q_thigh=0.7rad, q_shank=-1.4rad")
    print("\nExpected foot positions (approximate):")
    print("  FL: x≈0.30, y≈0.17, z≈-0.28")
    print("  FR: x≈0.30, y≈-0.17, z≈-0.28")
    print("  HL: x≈-0.30, y≈0.17, z≈-0.28")
    print("  HR: x≈-0.30, y≈-0.17, z≈-0.28")
    
    foot_positions = dynamics.compute_contact_positions(q_j_standing, theta)
    
    print("\nActual FK output:")
    leg_names = ['FL', 'FR', 'HL', 'HR']
    for i, (name, pos) in enumerate(zip(leg_names, foot_positions)):
        print(f"  {name}: x={pos[0]:6.3f}, y={pos[1]:6.3f}, z={pos[2]:6.3f}")
    
    # Check if all feet are at similar height
    z_positions = [pos[2] for pos in foot_positions]
    z_std = np.std(z_positions)
    
    if z_std < 0.01:
        print(f"\n✓ All feet at similar height (std={z_std:.4f}m) - FK looks correct")
    else:
        print(f"\n✗ Feet at different heights (std={z_std:.4f}m) - FK might be wrong")
    
    # Check if feet are outside body (reasonable positions)
    all_reasonable = True
    for i, pos in enumerate(foot_positions):
        if abs(pos[0]) > 0.6 or abs(pos[1]) > 0.4 or pos[2] > 0:
            print(f"✗ {leg_names[i]} position unreasonable: {pos}")
            all_reasonable = False
    
    if all_reasonable:
        print("✓ All foot positions are physically reasonable")
    
    return foot_positions


if __name__ == "__main__":
    """
    Run this debug script standalone to verify gait patterns
    
    Usage:
        python debug_gait_rollout.py
    
    This will:
    1. Simulate 5 seconds of gait pattern
    2. Check if diagonal legs alternate correctly
    3. Verify FK is computing reasonable foot positions
    4. Generate plots showing contact pattern and foot trajectories
    """
    
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*20 + "GAIT ROLLOUT DEBUG TOOL" + " "*25 + "║")
    print("╚"+"═"*68+"╝\n")
    
    # Step 1: Verify FK works correctly
    print("\n[STEP 1/3] Testing Forward Kinematics...")
    verify_forward_kinematics()
    
    input("\nPress ENTER to continue with gait rollout simulation...")
    
    # Step 2: Simulate gait pattern
    print("\n[STEP 2/3] Simulating Gait Pattern...")
    results = simulate_gait_rollout(duration=5.0, dt=0.02)
    
    # Step 3: Analyze results
    print("\n[STEP 3/3] Analyzing Results...")
    analyze_results(results)
    
    # Step 4: Generate plots
    print("\n[VISUALIZATION] Generating plots...")
    try:
        plot_results(results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("(matplotlib might not be available)")
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\nKey things to check:")
    print("  1. Are diagonal pairs (FL+RH, FR+HL) synchronized? (should be ~100%)")
    print("  2. Do swing legs lift off ground? (z should increase during swing)")
    print("  3. Are foot positions reasonable? (within ±0.5m of body)")
    print("\nIf any of these fail, the FK or gait generator needs fixing!")
    print("="*70)