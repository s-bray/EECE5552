# MPC-Based Quadruped Locomotion Controller

**Branch:** `mpc/shayak-` (Stable Version)

A Model Predictive Control (MPC) implementation for quadruped robot locomotion using MuJoCo physics simulation. This implementation features a hierarchical control architecture with MPC for contact force optimization, inverse kinematics for swing leg control, and low-level PD tracking.

---

## 🚀 Quick Start

### Prerequisites
```bash
# Required Python packages
pip install numpy scipy mujoco
```

### Running the Simulation
```bash
cd MPC_Gait
python main.py
```

**Controls:**
- Press `ENTER` to start the simulation
- Press `Ctrl+C` to stop
- Press `Space` in viewer to pause/resume
- Press `F` to toggle force visualization

---

## 📋 Project Structure

```
MPC_Gait/
├── main.py                    # Main execution script
├── config.py                  # MPC parameters configuration
├── mpc_controller.py          # MPC optimizer implementation
├── dynamics.py                # Single Rigid Body Dynamics model
├── simulation.py              # MuJoCo simulation wrapper
├── gait_generator.py          # Gait pattern generator
├── kinematics.py              # Forward/Inverse kinematics
├── utils.py                   # Trajectory generation utilities
├── create_simple_quadruped.py # Robot model generation
├── quick_test_fk.py           # FK debugging tool
├── quick_test_plot_gait_rollout.py  # Gait visualization
└── debug_swing.py             # Swing trajectory debugging
```

---

## 🎯 Main Workflow (`main.py`)

### Execution Flow

```
1. Load Parameters (config.py)
   ↓
2. Initialize Dynamics Model (SRBD)
   ↓
3. Initialize MPC Controller
   ↓
4. Setup MuJoCo Simulation
   ↓
5. Stabilization Phase (1 second)
   ↓
6. Main Control Loop:
   ├─ Get robot state
   ├─ Generate reference trajectory
   ├─ Update gait pattern
   ├─ Compute joint commands (stance/swing)
   ├─ Solve MPC (contact forces)
   ├─ Apply control to simulation
   └─ Step physics & render
```

### Control Architecture

```
┌─────────────────────────────────────────┐
│         MPC Layer (High-level)          │
│  Optimizes contact forces for base      │
│  motion tracking using SRBD model        │
└──────────────┬──────────────────────────┘
               │ Contact Forces (λ_e)
               ↓
┌─────────────────────────────────────────┐
│         IK Layer (Mid-level)            │
│  Generates swing leg trajectories       │
│  using inverse kinematics               │
└──────────────┬──────────────────────────┘
               │ Joint Positions/Velocities
               ↓
┌─────────────────────────────────────────┐
│      Low-level PD Controller            │
│  Tracks joint commands with gravity     │
│  compensation and contact forces        │
└─────────────────────────────────────────┘
```

### Key Configuration (`main.py` lines 287-295)

```python
USE_GUI = True              # Enable 3D visualization
SIMULATION_TIME = 30.0      # Duration in seconds
WHEELS_ON = False           # Enable wheeled mode
ENABLE_GAIT_DEBUG = False   # Print gait state info

# Target velocity [vx, vy, vz, ωx, ωy, ωz]
TARGET_VELOCITY = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
```

---

## ⚙️ MPC Controller Parameters

### Primary Parameters (`config.py`)

```python
@dataclass
class MPCParameters:
    # === Robot Physical Parameters ===
    robot_mass: float = 15.0              # kg
    robot_inertia: np.ndarray = np.diag([0.5, 1.0, 1.0])  # kg⋅m²
    
    # === MPC Horizon Settings ===
    horizon_length: float = 0.8           # seconds (prediction horizon)
    num_nodes: int = 20                   # discretization points
    control_freq: float = 50.0            # Hz (control loop frequency)
    
    # === Cost Function Weights ===
    weight_tracking: float = 1.0          # Base position/velocity tracking
    weight_orientation: float = 5.0       # Orientation tracking
    weight_angular_vel: float = 0.5       # Angular velocity tracking
    weight_force: float = 0.001           # Contact force regularization
    weight_joint_vel: float = 0.01        # Joint velocity regularization
    
    # === Physical Constraints ===
    friction_coeff: float = 0.7           # Ground friction coefficient
    max_joint_velocity: float = 8.0       # rad/s
    max_contact_force: float = 500.0      # N per leg
    
    # === Gait Parameters ===
    swing_height: float = 0.06            # m (foot clearance)
    swing_duration: float = 0.3           # seconds
```

### Where to Change Parameters

**Option 1: Modify defaults in `config.py`**
```python
# config.py, line 15-45
@dataclass
class MPCParameters:
    horizon_length: float = 1.0  # Increase prediction horizon
    weight_tracking: float = 2.0  # Increase tracking weight
```

**Option 2: Override in `main.py`**
```python
# main.py, line 300-325
params = MPCParameters(
    robot_mass=15.0,
    horizon_length=0.8,        # ← Change here
    num_nodes=20,
    weight_tracking=1.0,       # ← Change here
    weight_orientation=5.0,
    # ... other parameters
)
```

### Parameter Tuning Guide

| Parameter | Effect | Increase to... | Decrease to... |
|-----------|--------|----------------|----------------|
| `horizon_length` | Prediction time | Better planning, slower | Faster computation |
| `num_nodes` | Discretization | Higher accuracy, slower | Faster computation |
| `weight_tracking` | Position tracking | Tighter tracking | More compliant |
| `weight_orientation` | Orientation tracking | More stable orientation | Allow more tilt |
| `weight_force` | Force smoothness | Smoother forces | More aggressive |
| `friction_coeff` | Ground friction | More grip | Allow slipping |
| `max_contact_force` | Force limits | Allow higher forces | Gentler contact |
| `swing_height` | Foot clearance | Higher steps | Lower energy |

---

## 🛠️ Debugging Tools

### 1. Forward Kinematics Test (`quick_test_fk.py`)

**Purpose:** Verify FK calculations and joint limits

```bash
python quick_test_fk.py
```

**What it does:**
- Tests FK for all 4 legs
- Checks joint angle limits
- Validates foot positions
- Prints detailed diagnostics

**Use when:**
- FK calculations seem incorrect
- Foot positions are wrong
- Joint limits are violated

---

### 2. Gait Rollout Visualization (`quick_test_plot_gait_rollout.py`)

**Purpose:** Visualize gait patterns and contact schedules

```bash
python quick_test_plot_gait_rollout.py
```

**What it does:**
- Plots contact states over time
- Shows swing/stance transitions
- Visualizes gait patterns (trot, walk, etc.)
- Generates matplotlib plots

**Use when:**
- Gait timing seems off
- Contact transitions are abrupt
- Need to verify gait pattern

**Output:**
- Contact schedule plot
- Gait phase diagram
- Timing analysis

---

### 3. Swing Trajectory Debug (`debug_swing.py`)

**Purpose:** Analyze and visualize swing leg trajectories

```bash
python debug_swing.py
```

**What it does:**
- Plots swing foot trajectories
- Shows clearance height
- Validates IK solutions
- Checks reachability

**Use when:**
- Swing legs hit the ground
- Foot trajectories look wrong
- IK fails during swing
- Clearance is insufficient

**Output:**
- 3D trajectory plot
- Height profile
- IK solution validity
- Reachability analysis

---

## 🎮 Gait Modes

Available gait patterns (set in `main.py` line 424):

```python
controller.gait_gen.set_gait_mode('hybrid_walk')  # Default
```

**Available modes:**
- `'hybrid_trot'` - Diagonal leg pairs (faster, less stable)
- `'hybrid_walk'` - One leg at a time (slower, more stable)
- `'pure_driving'` - All legs in stance (no swing phase)

---

## 📊 Performance Metrics

The simulation prints real-time metrics:

```
t=  1.00s | pos=[ 0.15, -0.29,  0.17] | vel=[ 0.00, -0.00,  0.00] | contacts=█░██ | cost=   40.18
```

**Metrics:**
- `t` - Simulation time (seconds)
- `pos` - Base position [x, y, z] (meters)
- `vel` - Base velocity [vx, vy, vz] (m/s)
- `contacts` - Contact state (█=stance, ░=swing)
- `cost` - MPC cost function value

---

## 🔧 Common Issues & Solutions

### Issue: Robot falls immediately
**Solution:**
1. Check `robot_mass` matches model
2. Increase `weight_orientation`
3. Verify initial configuration in `simulation.py`

### Issue: Jerky motion
**Solution:**
1. Increase `horizon_length`
2. Decrease `weight_force`
3. Increase `num_nodes` for smoother optimization

### Issue: Legs slip
**Solution:**
1. Increase `friction_coeff`
2. Decrease `max_contact_force`
3. Check ground contact in MuJoCo viewer

### Issue: Swing legs hit ground
**Solution:**
1. Increase `swing_height`
2. Run `debug_swing.py` to visualize
3. Check IK reachability test output

### Issue: MPC solver fails
**Solution:**
1. Decrease `horizon_length`
2. Reduce `num_nodes`
3. Check constraint violations in console

---

## 📈 Advanced Features

### Wheeled Mode (Experimental)
```python
# main.py, line 290
WHEELS_ON = True
```

Enables wheeled quadruped with ratchet mechanism (one-way wheels).

### Custom Robot Models
Modify `create_simple_quadruped.py` to change:
- Link lengths
- Mass distribution
- Joint limits
- Wheel properties

---

## 📝 Code Organization

### Control Flow in `main.py`

**Lines 97-110:** Stance leg control (proportional feedback)
```python
def compute_stance_control(leg_idx, q_now):
    # Soft proportional control (Kp=10.0)
    # Allows MPC forces to dominate
```

**Lines 113-180:** Swing leg control (IK-based)
```python
def compute_swing_ik_simple(leg_idx, swing_phase, q_now):
    # Generates smooth foot trajectory
    # Uses IK to find joint angles
    # Stiff tracking (Kp=12.0)
```

**Lines 300-325:** MPC parameter initialization
```python
params = MPCParameters(
    # Override default parameters here
)
```

**Lines 420-550:** Main control loop
```python
for step in range(num_steps):
    # 1. Get state
    # 2. Generate reference
    # 3. Update gait
    # 4. Compute commands
    # 5. Solve MPC
    # 6. Apply control
    # 7. Step physics
```

---

## 🎓 Theory Background

### Single Rigid Body Dynamics (SRBD)

The MPC uses a reduced-order model treating the robot as a single rigid body:

**State (24D):**
```
x = [θ, p, ω, v, q_j]
  θ: Euler angles (roll, pitch, yaw)
  p: Position (x, y, z)
  ω: Angular velocity (body frame)
  v: Linear velocity (body frame)
  q_j: Joint angles (12D)
```

**Control (24D):**
```
u = [F_contact, q̇_j]
  F_contact: Contact forces (12D, 3 per leg)
  q̇_j: Joint velocities (12D)
```

### MPC Optimization

**Objective:**
```
min Σ [||x - x_ref||²_Q + ||u||²_R]
 u

Subject to:
  - Dynamics: ẋ = f(x, u)
  - Friction cone: √(Fx² + Fy²) ≤ μ⋅Fz
  - Unilateral contact: Fz ≥ 0
  - Force limits: ||F|| ≤ F_max
  - Joint velocity limits: |q̇| ≤ q̇_max
```

**Solver:** Iterative refinement (5 iterations, learning rate 0.01)

---

## 📚 References

Based on:
- Bjelonic et al. (2021) - "Whole-Body MPC for Wheeled-Legged Robots"
- Di Carlo et al. (2018) - "Dynamic Locomotion in the MIT Cheetah 3"

---

## ✅ Verification Checklist

Before running:
- [ ] MuJoCo installed (`pip install mujoco`)
- [ ] All dependencies installed
- [ ] `main.py` parameters configured
- [ ] Gait mode selected

After running:
- [ ] Robot stabilizes in first second
- [ ] IK reachability test passes (4/5 phases)
- [ ] Contact states alternate correctly
- [ ] Average cost < 100
- [ ] Robot moves forward

---

## 🐛 Debug Workflow

1. **Run main simulation**
   ```bash
   python main.py
   ```

2. **If FK issues:** Run FK test
   ```bash
   python quick_test_fk.py
   ```

3. **If gait issues:** Visualize gait
   ```bash
   python quick_test_plot_gait_rollout.py
   ```

4. **If swing issues:** Debug swing
   ```bash
   python debug_swing.py
   ```

5. **Adjust parameters** in `config.py` or `main.py`

6. **Repeat** until stable

---

**Version:** Stable (Branch: `mpc/shayak-`)  
**Last Updated:** 2025-11-20  
**Status:** ✅ Working and Tested
