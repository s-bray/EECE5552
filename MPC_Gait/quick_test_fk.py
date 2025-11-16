#!/usr/bin/env python3
"""
QUICK 30-SECOND TEST: Verify FK and Gait Pattern
Run this standalone to check if your system is working correctly
"""

import numpy as np
import sys

# Import your modules
try:
    from config import MPCParameters
    from dynamics import SingleRigidBodyDynamics
    from gait_generator import GaitSequenceGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the correct directory!")
    sys.exit(1)

print("="*70)
print("QUICK FK & GAIT TEST (30 seconds)")
print("="*70)

# Initialize
params = MPCParameters()
dynamics = SingleRigidBodyDynamics(params)
gait_gen = GaitSequenceGenerator(params)

# Set trot gait
gait_gen.set_gait_mode('hybrid_trot')

print("\n[1/3] Testing Forward Kinematics...")
print("-" * 70)

# Standing pose
q_standing = np.array([0.0, 0.7, -1.4] * 4)
theta = np.array([0.0, 0.0, 0.0])

foot_pos = dynamics.compute_contact_positions(q_standing, theta)

print("Standing pose joint angles: [0.0, 0.7, -1.4] per leg")
print("\nComputed foot positions:")
for i, name in enumerate(['FL', 'FR', 'HL', 'HR']):
    pos = foot_pos[i]
    print(f"  {name}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")

# Check if reasonable
all_below_ground = all(pos[2] < 0 for pos in foot_pos)
z_values = [pos[2] for pos in foot_pos]
z_range = max(z_values) - min(z_values)

print(f"\nFK Sanity Check:")
print(f"  All feet below base? {all_below_ground} {'✓' if all_below_ground else '✗'}")
print(f"  Height variation: {z_range:.4f}m {'✓ OK' if z_range < 0.01 else '✗ TOO LARGE'}")

if all_below_ground and z_range < 0.01:
    print("  → FK appears CORRECT ✓")
else:
    print("  → FK might be WRONG ✗")

print("\n[2/3] Testing Gait Pattern (5 seconds of trot)...")
print("-" * 70)

dt = 0.02  # 50Hz
duration = 5.0
num_steps = int(duration / dt)

contact_log = []
pattern_errors = 0

print("\nTime | Contacts | FL+RH | FR+HL | Opposite | Status")
print("-" * 70)

for step in range(num_steps):
    t = step * dt
    
    # Update gait every 0.1s
    if step % 5 == 0:
        utilities = np.ones(4) * 0.8
        gait_gen.update_gait(utilities, dt * 5)
    
    contacts = gait_gen.contact_states.copy()
    contact_log.append(contacts)
    
    # Check pattern every 0.5s
    if step % 25 == 0:
        fl_rh = contacts[0] == contacts[3]
        fr_hl = contacts[1] == contacts[2]
        opposite = contacts[0] != contacts[1]
        
        pattern_ok = fl_rh and fr_hl and opposite
        if not pattern_ok:
            pattern_errors += 1
        
        contact_str = ''.join(['█' if c else '░' for c in contacts])
        status = '✓' if pattern_ok else '✗'
        
        print(f"{t:4.2f} | {contact_str} | "
              f"{'Yes' if fl_rh else 'NO '} | "
              f"{'Yes' if fr_hl else 'NO '} | "
              f"{'Yes' if opposite else 'NO '} | {status}")

print("-" * 70)

# Calculate statistics
contact_log = np.array(contact_log)
fl_rh_sync = np.mean(contact_log[:, 0] == contact_log[:, 3]) * 100
fr_hl_sync = np.mean(contact_log[:, 1] == contact_log[:, 2]) * 100
opposite_sync = np.mean(contact_log[:, 0] != contact_log[:, 1]) * 100

print("\n[3/3] Gait Pattern Statistics:")
print("-" * 70)
print(f"  FL+RH synchronized: {fl_rh_sync:.1f}% (target: >95%)")
print(f"  FR+HL synchronized: {fr_hl_sync:.1f}% (target: >95%)")
print(f"  Diagonal opposite:  {opposite_sync:.1f}% (target: >95%)")
print(f"  Pattern errors:     {pattern_errors}/{int(duration/0.5)} checks")

# Calculate duty factors
for i, name in enumerate(['FL', 'FR', 'HL', 'HR']):
    duty = np.mean(contact_log[:, i]) * 100
    print(f"  {name} duty factor: {duty:.1f}% (target: ~60% for trot)")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

fk_ok = all_below_ground and z_range < 0.01
gait_ok = fl_rh_sync > 95 and fr_hl_sync > 95 and opposite_sync > 95

if fk_ok and gait_ok:
    print("✓✓✓ BOTH FK AND GAIT PATTERN ARE CORRECT!")
    print("\nYour system should work. If robot still falls:")
    print("  → Check PD gains (kp, kd)")
    print("  → Check torque limits")
    print("  → Check initial pose in simulation.py")
elif fk_ok and not gait_ok:
    print("✓ FK is correct")
    print("✗ GAIT PATTERN IS WRONG!")
    print("\nProblem: Diagonal legs not alternating properly")
    print("Fix: Check gait_generator.py update logic")
elif not fk_ok and gait_ok:
    print("✗ FK IS WRONG!")
    print("✓ Gait pattern is correct")
    print("\nProblem: Forward kinematics computing bad foot positions")
    print("Fix: Check dynamics.py compute_contact_positions()")
else:
    print("✗✗ BOTH FK AND GAIT HAVE ISSUES!")
    print("\nStart by fixing FK first (dynamics.py)")
    print("Then fix gait pattern (gait_generator.py)")

print("="*70)