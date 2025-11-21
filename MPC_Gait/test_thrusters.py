"""
Test script for quadruped with thrusters.

This script demonstrates thruster control on a quadruped with normal feet.
Perfect for testing thruster logic without the complexity of wheels.

Usage:
    python3 test_thrusters.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

def test_thrusters():
    """
    Test thruster control on quadruped with normal feet.
    
    Demonstrates:
    - Hovering with thrusters
    - Roll/pitch/yaw control
    - Adaptive thrust based on orientation
    """
    
    print("=" * 60)
    print("THRUSTER TEST - Quadruped with Normal Feet")
    print("=" * 60)
    
    # Load model
    model = mujoco.MjModel.from_xml_path("/home/poison-arrow/EECE5552/MPC_Gait/quadruped_with_thrusters.xml")
    data = mujoco.MjData(model)
    
    print("\n✓ Model loaded successfully")
    print(f"  Total mass: {np.sum(model.body_mass):.1f} kg")
    print(f"  Number of actuators: {model.nu}")
    
    # Find thruster actuator indices
    thruster_names = ['fl_thruster', 'fr_thruster', 'hl_thruster', 'hr_thruster']
    thruster_indices = []
    
    for name in thruster_names:
        try:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            thruster_indices.append(idx)
        except:
            print(f"⚠️  Warning: Thruster '{name}' not found")
    
    print(f"\n✓ Found {len(thruster_indices)} thrusters")
    
    # Set initial configuration (standing pose)
    # Position and orientation
    data.qpos[0:3] = [0, 0, 0.5]  # x, y, z position
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z) - upright
    
    # Leg joint angles (hip, thigh, shank for each leg)
    # Bent leg configuration (Lower stance)
    # Thigh: 0.9 rad (~51 deg), Shank: -1.8 rad (~-103 deg)
    q_bent = [0.0, 0.9, -1.8]
    
    data.qpos[7:10] = q_bent   # FL
    data.qpos[10:13] = q_bent  # FR
    data.qpos[13:16] = q_bent  # HL
    data.qpos[16:19] = q_bent  # HR
    
    # Zero velocities
    data.qvel[:] = 0.0
    
    mujoco.mj_forward(model, data)
    
    print("\n✓ Initial configuration set (Bent Stance)")
    print(f"  Height: {data.qpos[2]:.2f} m")
    
    # Launch viewer
    print("\n✓ Launching viewer...")
    print("\nControls:")
    print("  Press 'F' to toggle force visualization")
    print("  Press 'Ctrl+C' to exit")
    print("\nThruster Test Sequence:")
    print("  0-5s:   Stabilization (equal thrust)")
    print("  5-15s:  Pitch Verify (nose up/down)")
    print("  15-25s: Yaw Verify (rotation)")
    print("  25s+:   Relax")
    print("\nWatch the orientation angles (r/p/y) change!")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # Target joint angles for holding stance
        target_joints = np.array(q_bent * 4)
        
        while viewer.is_running():
            step_start = time.time()
            current_time = time.time() - start_time
            
            # === LEG CONTROL (HOLD STANCE) ===
            # Apply position commands to leg actuators (indices 0-11)
            # Assuming the first 12 actuators are the leg joints
            for i in range(12):
                data.ctrl[i] = target_joints[i]

            # === THRUSTER CONTROL PARAMETERS ===
            # PID Gains
            kp_roll = 200.0
            kd_roll = 50.0
            
            kp_pitch = 200.0
            kd_pitch = 50.0
            
            kp_yaw = 100.0
            kd_yaw = 20.0

            # Get robot state
            # Height
            z = data.qpos[2]
            vz = data.qvel[2]
            
            # Orientation (quaternion -> euler)
            quat = data.qpos[3:7]
            # Simple roll/pitch/yaw extraction
            # MuJoCo quat is [w, x, y, z]
            
            # Manual conversion for safety if helper not available or for consistency
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

            # Yaw (z-axis rotation)
            siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
            cosy_cosp = 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3])
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Angular velocities
            w_x = data.qvel[3]
            w_y = data.qvel[4]
            w_z = data.qvel[5]

            # === TARGETS ===
            target_roll = 0.0
            target_pitch = 0.0
            target_yaw = 0.0 
            
            # Weight Support Ratio (0.0 to 1.0)
            # 0.6 = Thrusters carry 60% of robot weight, Legs carry 40%
            support_ratio = 0.6 
            
            phase_name = "Assisted Stand"

            # Test Sequence
            if current_time < 5.0:
                phase_name = "Stand Stabilize"
                target_pitch = 0.0
                target_yaw = 0.0
                
            elif current_time < 10.0:
                phase_name = "Pitch Verify"
                # Oscillate pitch +/- 10 degrees to verify authority
                target_pitch = 0.17 * np.sin((current_time - 5.0) * 2.0)
                
            elif current_time < 15.0:
                phase_name = "Yaw Verify"
                # Oscillate yaw +/- 10 degrees to verify authority
                target_yaw = 0.17 * np.sin((current_time - 10.0) * 2.0)
                
            else:
                phase_name = "Relax"
                support_ratio = 0.0 # Slowly turn off thrusters? Or just stop.
                if current_time > 16.0:
                    support_ratio = 0.0

            # === CONTROL COMPUTATION ===
            
            # Height/Weight Support (Feedforward only)
            total_mass = np.sum(model.body_mass)
            gravity = 9.81
            f_gravity = total_mass * gravity
            
            f_height = f_gravity * support_ratio
            
            # Orientation Control (PID -> Differential Thrust)
            
            # Roll
            error_roll = target_roll - roll
            t_roll = (kp_roll * error_roll) - (kd_roll * w_x)
            
            # Pitch
            error_pitch = target_pitch - pitch
            t_pitch = (kp_pitch * error_pitch) - (kd_pitch * w_y)
            
            # Yaw
            error_yaw = target_yaw - yaw
            t_yaw = (kp_yaw * error_yaw) - (kd_yaw * w_z)
            
            # === MIXING LOGIC ===
            # FL (0), FR (1), HL (2), HR (3)
            # Layout:
            # FL: +Roll (Right is neg?), +Pitch (Nose up?), +Yaw (CCW?)
            # Let's derive signs:
            # Roll: Positive roll is Right side down. To correct positive roll, we need Right thrusters (FR, HR) to push UP more.
            #       So +Roll Error (Target > Current) means we want to roll right? No, standard frame:
            #       Roll is rotation about X. Y is left. Z is up.
            #       Let's stick to: To roll RIGHT (positive?), Left thrusters UP, Right thrusters DOWN.
            #       Actually, let's just use the torque direction.
            #       Torque X (Roll): Positive = Roll Right. Needs Left thrusters (+), Right thrusters (-).
            #       Torque Y (Pitch): Positive = Pitch Down (Nose down). Needs Rear thrusters (+), Front thrusters (-).
            #       Torque Z (Yaw): Positive = Turn Left (CCW). Needs FL+HR (+), FR+HL (-).
            
            # Base thrust per motor
            base = f_height / 4.0
            
            # Mixing
            # FL (Front Left): +T_roll - T_pitch + T_yaw
            # FR (Front Right): -T_roll - T_pitch - T_yaw
            # HL (Rear Left): +T_roll + T_pitch - T_yaw
            # HR (Rear Right): -T_roll + T_pitch + T_yaw
            
            # Note: Signs depend heavily on definition. We'll tune if needed.
            # Assuming:
            # t_roll > 0 => Want to roll right (Lift Left)
            # t_pitch > 0 => Want to pitch down (Lift Rear)
            # t_yaw > 0 => Want to yaw CCW
            
            f_fl = base + t_roll - t_pitch + t_yaw
            f_fr = base - t_roll - t_pitch - t_yaw
            f_hl = base + t_roll + t_pitch - t_yaw
            f_hr = base - t_roll + t_pitch + t_yaw
            
            # Apply to actuators
            thruster_forces = [f_fl, f_fr, f_hl, f_hr]
            
            for i, idx in enumerate(thruster_indices):
                # Clamp to valid range [0, 500]
                force = np.clip(thruster_forces[i], 0, 500)
                data.ctrl[idx] = force

            # Print status
            if int(current_time * 10) % 5 == 0: # 2Hz
                 print(f"t={current_time:5.1f}s | {phase_name:15s} | "
                      f"Support={support_ratio*100:.0f}% | "
                      f"R={np.degrees(roll):5.1f}° | "
                      f"P={np.degrees(pitch):5.1f}° (T={np.degrees(target_pitch):.1f}) | "
                      f"Y={np.degrees(yaw):5.1f}° (T={np.degrees(target_yaw):.1f})")

            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            

            
            # Maintain real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    try:
        test_thrusters()
    except KeyboardInterrupt:
        print("\n\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
