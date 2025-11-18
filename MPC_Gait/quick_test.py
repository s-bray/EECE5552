#!/usr/bin/env python3
"""
MINIMAL TEST: Check if joint control actually works
Run this FIRST before trying full walking
"""

import numpy as np
from simulation import RobotSimulation
from config import MPCParameters

print("="*70)
print("MINIMAL JOINT CONTROL TEST")
print("="*70)

# Setup
params = MPCParameters(robot_mass=15.0, control_freq=50.0)
sim = RobotSimulation(
    "/home/poison-arrow/MPC_Gait/anymal_simplified.xml",
    params, use_gui=True, verify=False, wheels=False
)

print("\n[1] Initial Configuration")
print("-" * 70)

# Get initial state
q_initial = np.zeros(12)
for i, joint_idx in enumerate(sim.joint_indices[:12]):
    qpos_addr = int(sim.model.jnt_qposadr[joint_idx])
    q_initial[i] = sim.data.qpos[qpos_addr]

print(f"Initial joint angles (deg):")
for i in range(4):
    leg_name = ['FL', 'FR', 'HL', 'HR'][i]
    joints = q_initial[i*3:(i+1)*3]
    print(f"  {leg_name}: hip={np.rad2deg(joints[0]):6.1f}° "
          f"thigh={np.rad2deg(joints[1]):6.1f}° "
          f"shank={np.rad2deg(joints[2]):6.1f}°")

print("\n[2] Test: Command FL hip to move")
print("-" * 70)

# Create command: move FL hip joint at 1 rad/s
u_test = np.zeros(24)
u_test[12] = 1.0  # FL hip velocity = 1 rad/s

contact_states = np.ones(4)  # All legs in stance

# Apply for 1 second (50 steps at 50Hz)
print("Applying command for 1 second...")
for step in range(50):
    sim.apply_control_new(u_test, contact_states)
    for _ in range(20):  # 20 physics steps per control step
        sim.step_physics()
    sim.render()

# Check result
q_final = np.zeros(12)
for i, joint_idx in enumerate(sim.joint_indices[:12]):
    qpos_addr = int(sim.model.jnt_qposadr[joint_idx])
    q_final[i] = sim.data.qpos[qpos_addr]

print(f"\nFinal joint angles (deg):")
for i in range(4):
    leg_name = ['FL', 'FR', 'HL', 'HR'][i]
    joints = q_final[i*3:(i+1)*3]
    print(f"  {leg_name}: hip={np.rad2deg(joints[0]):6.1f}° "
          f"thigh={np.rad2deg(joints[1]):6.1f}° "
          f"shank={np.rad2deg(joints[2]):6.1f}°")

print(f"\nChange in FL hip: {np.rad2deg(q_final[0] - q_initial[0]):.1f}°")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

expected_change = np.rad2deg(1.0 * 1.0)  # 1 rad/s × 1 second
actual_change = np.rad2deg(q_final[0] - q_initial[0])

if abs(actual_change) < 5.0:
    print("❌ JOINT DID NOT MOVE!")
    print(f"   Expected: ~{expected_change:.1f}°, Got: {actual_change:.1f}°")
    print("\nProblem: apply_control_new() is not using your commands!")
    print("Solution: Check that u_j_desired is actually being used in PD control")
elif abs(actual_change) > 80.0:
    print("⚠️ JOINT MOVED TOO MUCH!")
    print(f"   Expected: ~{expected_change:.1f}°, Got: {actual_change:.1f}°")
    print("\nProblem: Gains too high or no clamping")
else:
    print("✓ JOINT CONTROL WORKS!")
    print(f"   Expected: ~{expected_change:.1f}°, Got: {actual_change:.1f}°")
    print("\n→ Control system is working. Issue must be in swing IK.")

sim.close()
print("="*70)