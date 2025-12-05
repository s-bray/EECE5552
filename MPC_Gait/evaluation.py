import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import sys

def load_log_file(filename):
    """Load a single log file and return data dict + header."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None, None

    print(f"Loading {filename}...")
    
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        
        if data.size == 0:
            print(f"Warning: {filename} is empty.")
            return None, None
            
        return data, header
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def plot_evaluation():
    # 1. Identify files to plot
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        # Auto-detect logs in csv/ directory
        filenames = glob.glob("csv/logs_*.csv")
        if not filenames:
            # Fallback to current directory (legacy)
            filenames = glob.glob("logs_*.csv")
            if not filenames and os.path.exists("simulation_logs.csv"):
                filenames = ["simulation_logs.csv"]
    
    if not filenames:
        print("No log files found. Run simulation first.")
        return

    print(f"Found {len(filenames)} log files: {filenames}")
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Prepare figures
    fig_ctrl, ax_ctrl = plt.subplots(figsize=(15, 10))
    # Legs: Create separate figures for each leg
    legs = ['FL', 'FR', 'HL', 'HR']
    fig_legs = {}
    axes_legs = {}
    for i in range(4):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig_legs[i] = fig
        axes_legs[i] = ax
    
    fig_thrust, ax_thrust = plt.subplots(figsize=(10, 6))
    fig_orient, ax_orient = plt.subplots(figsize=(10, 6))
    fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(filenames)))
    
    has_thruster_data = False
    
    for idx, filename in enumerate(filenames):
        data, header = load_log_file(filename)
        if data is None:
            continue
            
        # Parse label from filename (e.g. logs_wheels_hybrid_walk.csv -> wheels_hybrid_walk)
        label = os.path.basename(filename).replace("logs_", "").replace(".csv", "")
        if label == "simulation_logs": label = "Legacy"
        
        color = colors[idx]
        
        # Map columns
        col_map = {name: i for i, name in enumerate(header)}
        def get_col(name):
            i = col_map.get(name)
            return data[:, i] if i is not None else None
            
        time = get_col('time')
        if time is None: continue
        
        # Normalize time to start at 0
        if len(time) > 0:
            time = time - time[0]
        
        # 1. All Controls (Summary)
        # Just plot the first few to avoid clutter, or maybe norm?
        # Let's plot total control effort (norm)
        ctrl_cols = [c for c in header if c.startswith('ctrl_')]
        ctrl_data = []
        for c in ctrl_cols:
            val = get_col(c)
            if val is not None: ctrl_data.append(val)
        if ctrl_data:
            ctrl_norm = np.linalg.norm(np.array(ctrl_data), axis=0)
            ax_ctrl.plot(time, ctrl_norm, label=f"{label} (Norm)", color=color)
            
        # 2. Leg Torques
        joints = ['Hip', 'Thigh', 'Shank']
        linestyles = ['-', '--', ':']
        
        for i in range(4):
            ax = axes_legs[i]
            # Plot Hip torque only for clarity in comparison? 
            # Or plot all with different styles?
            # Let's plot Thigh torque (main load bearing)
            
            # Thigh is index 1 in the leg triplet (Hip, Thigh, Shank)
            # Leg i starts at ctrl index i*3
            thigh_idx = i*3 + 1
            col_name = f'ctrl_{thigh_idx}'
            val = get_col(col_name)
            if val is not None:
                ax.plot(time, val, label=f"{label}", color=color)
            
            ax.set_title(f"{legs[i]} Thigh Torque Comparison")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Torque (Nm)")
            ax.grid(True)
            
        # 3. Thrusters
        thruster_indices = [12, 13, 14, 15]
        # Sum of thruster forces
        thrust_sum = np.zeros_like(time)
        found_thrust = False
        for i in thruster_indices:
            val = get_col(f'ctrl_{i}')
            if val is not None:
                thrust_sum += val
                found_thrust = True
        
        if found_thrust:
            has_thruster_data = True
            ax_thrust.plot(time, thrust_sum, label=f"{label} (Total)", color=color)
            
        # 4. Orientation (Pitch)
        pitch = get_col('pitch')
        if pitch is not None:
            ax_orient.plot(time, pitch, label=f"{label} Pitch", color=color)
            
        # 5. Position (Z)
        z = get_col('base_z')
        if z is not None:
            ax_pos.plot(time, z, label=f"{label} Z", color=color)

    # Finalize Plots
    
    # Controls
    ax_ctrl.set_title("Control Effort Norm")
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_ylabel("Norm")
    ax_ctrl.legend()
    fig_ctrl.savefig("plots/control_effort.png")
    plt.close(fig_ctrl)
    
    # Legs (Save separate files)
    for i in range(4):
        ax = axes_legs[i]
        ax.legend()
        fig = fig_legs[i]
        fig.savefig(f"plots/leg_torque_{legs[i]}.png")
        plt.close(fig)
    
    # Thrusters
    if has_thruster_data:
        ax_thrust.set_title("Total Thruster Force")
        ax_thrust.set_xlabel("Time (s)")
        ax_thrust.set_ylabel("Force (N)")
        ax_thrust.legend()
        ax_thrust.grid(True)
        fig_thrust.savefig("plots/thruster_comparison.png")
    plt.close(fig_thrust)
    
    # Orientation
    ax_orient.set_title("Base Pitch Angle")
    ax_orient.set_xlabel("Time (s)")
    ax_orient.set_ylabel("Angle (deg)")
    ax_orient.legend()
    ax_orient.grid(True)
    fig_orient.savefig("plots/pitch_comparison.png")
    plt.close(fig_orient)
    
    # Position
    ax_pos.set_title("Base Height (Z)")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Height (m)")
    ax_pos.legend()
    ax_pos.grid(True)
    fig_pos.savefig("plots/height_comparison.png")
    plt.close(fig_pos)

    print("✓ Comparison plots saved to 'plots/' directory")

if __name__ == "__main__":
    plot_evaluation()
