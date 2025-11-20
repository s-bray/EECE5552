"""
Ratchet Wheel Test Suite - Complete Version

Tests one-way wheel ratchet mechanism with proper ramp geometry.
User controls progression through tests via ENTER key.

Test Sequence:
    1. Ramp World → Spawn robot → Check backward sliding → Kill
    2. Flat World → Spawn robot → Check forward rolling → Kill
    3. Diagnostics → Monitor wheel velocities

Author: Ratchet validation
Date: 2024
"""

import numpy as np
import mujoco
import mujoco.viewer
import sys

from config import MPCParameters
from create_simple_quadruped import create_simple_quadruped_xml_wheels


# ============================================================
#  XML WORLD BUILDERS
# ============================================================

def create_ramp_world_xml():
    """
    Build a world XML that contains the standard wheeled quadruped
    plus an extra test ramp geometry to stand on.

    We take the XML from create_simple_quadruped_xml_wheels() and
    inject a 'test_ramp' body inside <worldbody> before its closing tag.

    Ramp details:
        - Type: box
        - 45° pitched about Y axis
        - Size: 1.0m × 0.5m × 0.25m (x, y, z)
        - Positioned at origin
        - Colored red for visibility

    Returns:
        str: Complete, valid MuJoCo XML string.
    """
    base_xml = create_simple_quadruped_xml_wheels()

    # Where to inject: just before </worldbody>
    insert_pos = base_xml.rfind("</worldbody>")
    if insert_pos == -1:
        raise ValueError("Could not find </worldbody> in base robot XML")

    # 45° slope about Y axis → euler="0 0.785398 0"
    ramp_body = """
        <!-- Test ramp body for ratchet validation -->
        <body name="test_ramp" pos="0 0 0">
            <geom name="test_ramp_geom" type="box"
                  size="0.5 0.25 0.25"
                  pos="0 0 0.25"
                  euler="0 0.785398 0"
                  rgba="0.8 0.2 0.2 1"
                  friction="0.9 0.005 0.0001"/>
        </body>
"""

    # Inject ramp body inside worldbody
    new_xml = base_xml[:insert_pos] + ramp_body + base_xml[insert_pos:]

    return new_xml


def create_flat_world_xml():
    """
    Build world XML with flat ground only.
    
    Returns:
        str: Complete MuJoCo XML string
    """
    return create_simple_quadruped_xml_wheels()


# ============================================================
#  SIMULATION BUILDER
# ============================================================

def build_simulation(xml_string: str, with_gui: bool = True):
    """
    Create MuJoCo simulation from XML string.
    
    Args:
        xml_string (str): Complete MuJoCo XML
        with_gui (bool): Enable viewer
    
    Returns:
        tuple: (model, data, viewer)
    """
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    viewer = None
    if with_gui:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    return model, data, viewer


# ============================================================
#  CONTROL HELPERS
# ============================================================

def apply_stance_control(model, data):
    """
    Apply PD control to hold nominal standing pose.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
    """
    # Nominal joint positions: [hip, thigh, shank] for all 4 legs
    q_nominal = np.array([0.0, 0.6, -1.29] * 4)
    
    # Read current joint positions
    # Skip freejoint (first 7 qpos values), then 12 leg joints
    q_current = data.qpos[7:19].copy()
    
    # PD gains
    kp = 100.0
    kd = 10.0
    
    # Compute torques
    q_error = q_nominal - q_current
    qd_current = data.qvel[6:18].copy()  # Skip base velocities
    
    tau = kp * q_error - kd * qd_current
    tau = np.clip(tau, -80.0, 80.0)
    
    # Apply to first 12 actuators (legs)
    data.ctrl[:12] = tau / 150.0  # Divide by gear ratio


def apply_ratchet_control(model, data):
    """
    Apply ratchet logic to wheel joints.
    
    Prevents backward rotation by applying strong braking
    when wheels spin backward.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
    """
    wheel_joint_names = [
        'fl_wheel_joint', 'fr_wheel_joint',
        'hl_wheel_joint', 'hr_wheel_joint'
    ]
    
    for i, wheel_name in enumerate(wheel_joint_names):
        try:
            # Get wheel joint velocity
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
            dof_addr = model.jnt_dofadr[jid]
            wheel_vel = data.qvel[dof_addr]
            
            # Ratchet actuator index (after 12 leg motors)
            ratchet_idx = 12 + i
            
            if wheel_vel < -0.01:  # Backward rotation detected
                data.ctrl[ratchet_idx] = -500.0  # LOCK!
            else:  # Forward or stopped
                data.ctrl[ratchet_idx] = 0.0  # FREE
                
        except Exception:
            pass  # Wheel joint not found


def control_step(model, data):
    """Execute one control cycle: stance + ratchet."""
    apply_stance_control(model, data)
    apply_ratchet_control(model, data)
    mujoco.mj_step(model, data)


# ============================================================
#  TEST 1: RAMP BACKWARD SLIDE
# ============================================================

def test_ramp_slide():
    """
    Test 1: Spawn robot on 45° ramp and check for backward sliding.
    
    Returns:
        bool: True if ratchet prevented sliding
    """
    print("\n" + "="*70)
    print(" TEST 1: RAMP BACKWARD SLIDE")
    print("="*70)
    
    print("\n[Step 1/6] Creating ramp world...")
    xml = create_ramp_world_xml()
    model, data, viewer = build_simulation(xml, with_gui=True)
    print("  ✓ Ramp world created (45° incline, red color)")
    
    input("\n[Step 2/6] Press ENTER to spawn robot on ramp...\n")
    
    # Calculate ramp angle and position
    ramp_angle = 0.7854  # 45° in radians
    
    # Create quaternion for robot orientation (parallel to ramp)
    # Rotation about Y-axis by ramp_angle
    qw = np.cos(ramp_angle / 2)
    qy = np.sin(ramp_angle / 2)
    base_quat = np.array([qw, 0.0, qy, 0.0])  # [w, x, y, z]
    
    # Position robot on ramp surface
    # Ramp center at (0, 0, 0.25), robot slightly above
    ramp_height = 0.25 + 0.45
    base_pos = np.array([0.0, 0.0, ramp_height])
    
    # Set initial pose
    data.qpos[0:3] = base_pos
    data.qpos[3:7] = base_quat
    
    # Set initial joint positions to nominal stance
    q_nominal = np.array([0.0, 0.6, -1.29] * 4)
    data.qpos[7:19] = q_nominal
    
    mujoco.mj_forward(model, data)
    
    print(f"  ✓ Robot positioned at X={base_pos[0]:.3f}, Z={base_pos[2]:.3f}")
    print(f"  ✓ Robot tilted {np.rad2deg(ramp_angle):.1f}° to match ramp")
    
    # Let robot settle
    print("\n[Step 3/6] Settling robot (50 steps)...")
    for _ in range(50):
        control_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()
    
    x_initial = data.qpos[0]
    print(f"  ✓ Robot settled at X={x_initial:.3f}m")
    
    input("\n[Step 4/6] Press ENTER to start ramp hold test (5 seconds)...\n")
    
    print("  → Holding stance on ramp...")
    print("  → If ratchet works: robot stays in place")
    print("  → If ratchet fails: robot slides downhill (negative X)\n")
    
    # Hold for 5 seconds (250 steps @ 200Hz physics, 50Hz control)
    for step in range(250):
        control_step(model, data)
        
        if viewer and viewer.is_running():
            viewer.sync()
        
        # Status every second
        if step % 50 == 0 and step > 0:
            x_current = data.qpos[0]
            displacement = x_current - x_initial
            print(f"  t={step/50:.0f}s: X={x_current:.3f}m, ΔX={displacement:+.4f}m")
    
    # Final measurement
    x_final = data.qpos[0]
    displacement = x_final - x_initial
    
    print(f"\n[Step 5/6] RESULTS:")
    print(f"  Initial X:     {x_initial:+.4f}m")
    print(f"  Final X:       {x_final:+.4f}m")
    print(f"  Displacement:  {displacement:+.4f}m")
    
    # Evaluate
    SLIDE_THRESHOLD = -0.05  # 5cm backward = failure
    
    if displacement < SLIDE_THRESHOLD:
        print(f"\n  ❌ FAILED: Robot slid {abs(displacement)*100:.1f}cm DOWN the ramp!")
        print("     Ratchet is NOT preventing backward roll")
        result = False
    else:
        print(f"\n  ✅ PASSED: Robot stayed on ramp (drift {displacement*100:+.1f}cm)")
        print("     Ratchet successfully prevented backward sliding!")
        result = True
    
    input("\n[Step 6/6] Press ENTER to KILL simulation and proceed to Test 2...\n")
    
    # Kill simulation
    if viewer:
        viewer.close()
    del model, data, viewer
    
    print("  ✓ Ramp simulation terminated\n")
    
    return result


# ============================================================
#  TEST 2: FLAT GROUND FORWARD PUSH
# ============================================================

def test_forward_push():
    """
    Test 2: Apply forward force and check if wheels roll freely.
    
    Returns:
        bool: True if wheels rolled forward
    """
    print("\n" + "="*70)
    print(" TEST 2: FLAT GROUND FORWARD PUSH")
    print("="*70)
    
    print("\n[Step 1/6] Creating flat world...")
    xml = create_flat_world_xml()
    model, data, viewer = build_simulation(xml, with_gui=True)
    print("  ✓ Flat world created")
    
    input("\n[Step 2/6] Press ENTER to spawn robot...\n")
    
    # Set initial configuration
    q_nominal = np.array([0.0, 0.6, -1.29] * 4)
    data.qpos[7:19] = q_nominal
    data.qpos[2] = 0.40  # Base height
    mujoco.mj_forward(model, data)
    
    print("  ✓ Robot spawned on flat ground")
    
    print("\n[Step 3/6] Stabilizing robot (100 steps)...")
    for _ in range(100):
        control_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()
    
    x_initial = data.qpos[0]
    print(f"  ✓ Robot stable at X={x_initial:.3f}m")
    
    input("\n[Step 4/6] Press ENTER to apply 50N forward push (3 seconds)...\n")
    
    print("  → Applying forward force...")
    print("  → If wheels work: robot rolls forward (positive X)")
    print("  → If wheels locked: robot barely moves\n")
    
    # Apply push for 3 seconds (150 steps)
    for step in range(150):
        # Apply external force
        data.qfrc_applied[0] = 50.0  # 50N along +X
        
        control_step(model, data)
        
        if viewer and viewer.is_running():
            viewer.sync()
        
        # Status every second
        if step % 50 == 0 and step > 0:
            x_current = data.qpos[0]
            displacement = x_current - x_initial
            print(f"  t={step/50:.0f}s: X={x_current:.3f}m, ΔX={displacement:+.4f}m")
    
    # Final measurement
    x_final = data.qpos[0]
    displacement = x_final - x_initial
    
    print(f"\n[Step 5/6] RESULTS:")
    print(f"  Before push:   {x_initial:+.4f}m")
    print(f"  After push:    {x_final:+.4f}m")
    print(f"  Displacement:  {displacement:+.4f}m ({displacement*100:+.1f}cm)")
    
    # Evaluate
    MOTION_THRESHOLD = 0.05  # 5cm forward = success
    
    if displacement > MOTION_THRESHOLD:
        print(f"\n  ✅ PASSED: Robot rolled {displacement*100:.1f}cm FORWARD")
        print("     Wheels are rolling freely in forward direction!")
        result = True
    else:
        print(f"\n  ⚠️ WARNING: Robot only moved {displacement*100:.1f}cm")
        print("     Wheels may have excessive friction or be locked")
        result = False
    
    input("\n[Step 6/6] Press ENTER to KILL simulation and proceed to diagnostics...\n")
    
    # Kill simulation
    if viewer:
        viewer.close()
    del model, data, viewer
    
    print("  ✓ Flat simulation terminated\n")
    
    return result


# ============================================================
#  TEST 3: DIAGNOSTICS
# ============================================================

def test_diagnostics():
    """
    Test 3: Real-time wheel velocity monitoring.
    """
    print("\n" + "="*70)
    print(" TEST 3: WHEEL VELOCITY DIAGNOSTICS")
    print("="*70)
    
    print("\n[Step 1/4] Creating diagnostic world...")
    xml = create_flat_world_xml()
    model, data, viewer = build_simulation(xml, with_gui=True)
    print("  ✓ Diagnostic world created")
    
    input("\n[Step 2/4] Press ENTER to stabilize robot...\n")
    
    # Stabilize
    q_nominal = np.array([0.0, 0.6, -1.29] * 4)
    data.qpos[7:19] = q_nominal
    data.qpos[2] = 0.40
    mujoco.mj_forward(model, data)
    
    for _ in range(50):
        control_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()
    
    print("  ✓ Robot stabilized")
    
    input("\n[Step 3/4] Press ENTER to start velocity monitoring...\n")
    
    print("\n  Monitoring wheel velocities during forward push:")
    print("  " + "-"*66)
    print("  | Time  | FL_wheel | FR_wheel | HL_wheel | HR_wheel | Base_vel |")
    print("  " + "-"*66)
    
    wheel_joint_names = [
        'fl_wheel_joint', 'fr_wheel_joint',
        'hl_wheel_joint', 'hr_wheel_joint'
    ]
    
    for step in range(200):
        # Apply push
        data.qfrc_applied[0] = 50.0
        
        control_step(model, data)
        
        if viewer and viewer.is_running():
            viewer.sync()
        
        # Print every 0.5 seconds
        if step % 25 == 0:
            velocities = []
            for wheel_name in wheel_joint_names:
                try:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
                    dof_addr = model.jnt_dofadr[jid]
                    vel = data.qvel[dof_addr]
                    velocities.append(vel)
                except:
                    velocities.append(0.0)
            
            base_vel = data.qvel[0]  # Base X velocity
            
            print(f"  | {step*0.002:5.2f}s | {velocities[0]:8.2f} | {velocities[1]:8.2f} | "
                  f"{velocities[2]:8.2f} | {velocities[3]:8.2f} | {base_vel:8.3f} |")
    
    print("  " + "-"*66)
    print("\n  ℹ Wheel velocity interpretation:")
    print("    • Positive values = Forward rotation (GOOD)")
    print("    • Negative values = Backward rotation (Ratchet should prevent!)")
    print("    • ~0 rad/s = Locked or not in contact")
    
    input("\n[Step 4/4] Press ENTER to KILL diagnostic simulation...\n")
    
    # Kill simulation
    if viewer:
        viewer.close()
    del model, data, viewer
    
    print("  ✓ Diagnostic simulation terminated\n")


# ============================================================
#  MAIN TEST SEQUENCE
# ============================================================

def main():
    """
    Execute complete ratchet test sequence with user control.
    """
    print("="*70)
    print(" RATCHET WHEEL TEST SUITE")
    print("="*70)
    print("\n This test suite validates one-way wheel behavior:")
    print("\n   1. RAMP TEST    - Robot on 45° incline (no backward sliding)")
    print("   2. PUSH TEST    - Robot on flat ground (forward rolling)")
    print("   3. DIAGNOSTICS  - Real-time wheel velocity monitoring")
    print("\n Each test creates a new simulation world.")
    print(" Progress controlled by pressing ENTER.")
    print("\n" + "="*70)
    
    input("\nPress ENTER to begin Test 1 (Ramp)...\n")
    
    results = {}
    
    # Test 1: Ramp
    try:
        results['ramp'] = test_ramp_slide()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ramp test crashed: {e}")
        import traceback
        traceback.print_exc()
        results['ramp'] = False
    
    # Test 2: Push
    try:
        results['push'] = test_forward_push()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Push test crashed: {e}")
        import traceback
        traceback.print_exc()
        results['push'] = False
    
    # Test 3: Diagnostics
    try:
        test_diagnostics()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostics crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)
    
    print("\n  Test 1 - Ramp (no backward slide): ", end="")
    print("✅ PASS" if results.get('ramp', False) else "❌ FAIL")
    
    print("  Test 2 - Push (forward rolling):   ", end="")
    print("✅ PASS" if results.get('push', False) else "❌ FAIL")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print(" 🎉 ALL TESTS PASSED")
        print(" Ratchet mechanism is working correctly!")
    else:
        print(" ⚠️  SOME TESTS FAILED")
        print(" Check ratchet implementation:")
        print("   • Add ratchet actuators to XML (ctrlrange=-500 0)")
        print("   • Implement ratchet control in apply_control_new()")
        print("   • Verify wheel joint range limits (0 to 1000)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)