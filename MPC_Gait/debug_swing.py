"""
Debug tool to visualize swing trajectories and diagnose why legs don't touch ground
Add this to your main.py
"""

import numpy as np
from scipy.spatial.transform import Rotation

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
    
def debug_swing_detailed(step, t, x_current, contact_states, dynamics, controller):
    """
    Detailed swing trajectory analysis
    Call this every 0.5s during simulation
    """
    if step % 25 != 0:  # Every 0.5s at 50Hz
        return
    
    theta = x_current[0:3]
    p_base = x_current[3:6]
    q_j = x_current[12:24]
    
    R_WB = Rotation.from_euler('xyz', theta).as_matrix()
    
    # Get foot positions in world frame
    contact_points_B = dynamics.compute_contact_positions(q_j, theta)
    
    print(f"\n{'='*70}")
    print(f"[SWING DEBUG t={t:.2f}s]")
    print(f"{'='*70}")
    
    leg_names = ['FL', 'FR', 'HL', 'HR']
    
    for leg in range(4):
        name = leg_names[leg]
        contact = contact_states[leg]
        phase = controller.gait_gen.swing_phase[leg]
        
        # Foot position in world
        foot_B = contact_points_B[leg]
        foot_W = p_base + R_WB @ foot_B
        
        # Joint angles
        q_leg = q_j[leg*3:(leg+1)*3]
        
        status = "STANCE" if contact == 1 else f"SWING(s={phase:.2f})"
        
        print(f"\n{name} [{status}]:")
        print(f"  Joints: hip={np.rad2deg(q_leg[0]):6.1f}° "
              f"thigh={np.rad2deg(q_leg[1]):6.1f}° "
              f"shank={np.rad2deg(q_leg[2]):6.1f}°")
        print(f"  Foot (body):  [{foot_B[0]:6.3f}, {foot_B[1]:6.3f}, {foot_B[2]:6.3f}]")
        print(f"  Foot (world): [{foot_W[0]:6.3f}, {foot_W[1]:6.3f}, {foot_W[2]:6.3f}]")
        print(f"  Height above ground: {foot_W[2]:.3f}m")
        
        # Check for problems
        if contact == 0:  # Should be in swing
            if foot_W[2] < 0.02:
                print(f"  ⚠️ PROBLEM: Swing leg too low! (z={foot_W[2]:.3f}m)")
            if foot_W[2] > 0.15:
                print(f"  ⚠️ PROBLEM: Swing leg too high! (z={foot_W[2]:.3f}m)")
            
            # Check if leg is stretching out vs tucking in
            leg_extension = np.sqrt(foot_B[0]**2 + foot_B[2]**2)
            max_reach = 0.22 + 0.22  # L1 + L2
            extension_percent = (leg_extension / max_reach) * 100
            print(f"  Leg extension: {extension_percent:.1f}% of max reach")
            
            if extension_percent < 50:
                print(f"  ⚠️ PROBLEM: Leg too tucked! Not extending enough")
            if extension_percent > 95:
                print(f"  ⚠️ PROBLEM: Leg fully extended! Might be unreachable")
        
        else:  # Should be in stance
            if foot_W[2] > 0.05:
                print(f"  ⚠️ PROBLEM: Stance leg off ground! (z={foot_W[2]:.3f}m)")
    
    print(f"\n{'='*70}")
    print(f"Base: x={p_base[0]:.3f} y={p_base[1]:.3f} z={p_base[2]:.3f}")
    print(f"Orientation: roll={np.rad2deg(theta[0]):.1f}° "
          f"pitch={np.rad2deg(theta[1]):.1f}° yaw={np.rad2deg(theta[2]):.1f}°")


def check_ik_reachability(x_target, z_target, L1=0.22, L2=0.22):
    """
    Check if a target (x,z) position is reachable by 2-link leg
    
    Returns:
        (reachable, distance, max_reach)
    """
    distance = np.sqrt(x_target**2 + z_target**2)
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    
    reachable = (min_reach <= distance <= max_reach)
    
    return reachable, distance, max_reach


def test_ik_at_swing_phases():
    """
    Test IK computation at different swing phases
    Run this once at startup to verify IK is working
    """
    from kinematics import ik_sagittal
    
    L1, L2 = 0.22, 0.22
    step_length = 0.15  # 15cm step
    clearance = 0.06    # 6cm lift
    
    print("\n" + "="*70)
    print("IK REACHABILITY TEST")
    print("="*70)
    
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ss = 6*s**5 - 15*s**4 + 10*s**3
        
        x0, z0 = 0.08, -(L1 + L2 - 0.04)
        x1, z1 = x0 + step_length, z0
        
        x = x0 + (x1 - x0) * ss
        z = z0 + (z1 - z0) * ss + clearance * (1 - (2*s - 1)**2)
        z = -abs(z)
        
        reachable, dist, max_reach = check_ik_reachability(x, z, L1, L2)
        
        try:
            q_thigh, q_shank = ik_sagittal(L1, L2, x, z)
            ik_success = True
            ik_error = ""
        except Exception as e:
            ik_success = False
            ik_error = str(e)
            q_thigh, q_shank = 0, 0
        
        status = "✓" if (reachable and ik_success) else "✗"
        
        print(f"\nPhase s={s:.2f} (ss={ss:.2f}):")
        print(f"  Target: x={x:.3f}, z={z:.3f}")
        print(f"  Distance: {dist:.3f}m (max: {max_reach:.3f}m) {status}")
        print(f"  Reachable: {reachable}, IK Success: {ik_success}")
        if ik_success:
            print(f"  Joints: thigh={np.rad2deg(q_thigh):.1f}°, "
                  f"shank={np.rad2deg(q_shank):.1f}°")
        else:
            print(f"  IK Error: {ik_error}")
    
    print("="*70)