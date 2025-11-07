# gait_generator.py

import numpy as np
from typing import List

from config import MPCParameters

class GaitSequenceGenerator:
    """
    Generates gait sequences based on kinematic leg utilities
    Enhanced with insights from both papers, simplified for existing code structure
    """
    
    def __init__(self, params: MPCParameters):
        self.params = params
        
        # Kinematic utility parameters
        self.lambda_parallel = 0.4  # Half-axis along rolling direction
        self.lambda_perpendicular = 0.2  # Half-axis perpendicular to rolling
        self.utility_threshold = 0.3  # Threshold for triggering swing
        
        # Contact states and timers
        self.contact_states = np.ones(4)  # 1 = contact, 0 = swing
        self.swing_timers = np.zeros(4)
        self.swing_phase = np.zeros(4)  # 0-1 progress through swing
        
        # Gait mode selection
        self.gait_mode = 'hybrid_trot'  # Options: 'pure_driving', 'hybrid_trot', 'hybrid_walk'
        self.gait_cycle_time = 0.0
        self.stride_duration = 0.8  # One complete gait cycle
        
        # Fixed gait patterns (from "Rolling in the Deep" paper)
        self.gait_patterns = {
            'pure_driving': {
                'duty_factor': 1.0,  # All legs always in contact
                'phase_offsets': [0.0, 0.0, 0.0, 0.0],
            },
            'hybrid_trot': {
                'duty_factor': 0.6,  # 60% stance, 40% swing
                'phase_offsets': [0.0, 0.5, 0.5, 0.0],  # Diagonal pairs: LF+RH, RF+LH
            },
            'hybrid_walk': {
                'duty_factor': 0.75,  # 75% stance (3 legs always in contact)
                'phase_offsets': [0.0, 0.5, 0.75, 0.25],
            },
        }
        
    def set_gait_mode(self, mode: str):
        """Change gait pattern"""
        if mode in self.gait_patterns:
            self.gait_mode = mode
            print(f"[Gait] Switched to: {mode}")
        else:
            print(f"[Gait] Unknown mode '{mode}'. Available: {list(self.gait_patterns.keys())}")
    
    def compute_leg_utility(self, leg_idx: int, 
                           current_foot_pos: np.ndarray,
                           reference_pos: np.ndarray) -> float:
        """
        Compute kinematic leg utility (equation 7 from whole-body MPC paper)
        
        Utility = 1 - sqrt((r_parallel/λ_parallel)² + (r_perp/λ_perp)²)
        
        Returns value in [0, 1] where:
        - 1.0 = foot at ideal position
        - 0.0 = foot at kinematic limit
        """
        r_error = reference_pos - current_foot_pos
        r_parallel = abs(r_error[0])  # Along rolling direction (x)
        r_perp = np.linalg.norm(r_error[1:3])  # Perpendicular (y, z)
        
        utility = 1.0 - np.sqrt(
            (r_parallel / self.lambda_parallel)**2 + 
            (r_perp / self.lambda_perpendicular)**2
        )
        
        return np.clip(utility, 0.0, 1.0)
    
    def update_gait(self, utilities: np.ndarray, dt: float):
        """
        Main update function - compatible with your existing main.py code
        
        This function updates contact states based on EITHER:
        1. Fixed gait patterns (stable, predictable)
        2. Utility-based adaptive gait (flexible, reactive)
        
        Args:
            utilities: Leg utilities (4,) - can be dummy values if using fixed pattern
            dt: Time step since last update
        """
        # Update gait cycle time
        self.gait_cycle_time += dt
        if self.gait_cycle_time >= self.stride_duration:
            self.gait_cycle_time -= self.stride_duration
        
        # Get gait parameters
        gait = self.gait_patterns[self.gait_mode]
        duty_factor = gait['duty_factor']
        phase_offsets = gait['phase_offsets']
        
        # === UPDATE METHOD 1: FIXED PATTERN (More stable) ===
        if self.gait_mode in ['pure_driving', 'hybrid_trot', 'hybrid_walk']:
            self._update_fixed_pattern(duty_factor, phase_offsets)
        
        # === UPDATE METHOD 2: UTILITY-BASED (More adaptive) ===
        # Uncomment below and comment above if you want utility-based gait
        # self._update_utility_based(utilities, dt)
    
    def _update_fixed_pattern(self, duty_factor: float, phase_offsets: List[float]):
        """
        Update using fixed gait pattern (from "Rolling in the Deep" paper)
        This is MORE STABLE than utility-based
        """
        # Calculate phase for each leg (0 to 1)
        phases = [(self.gait_cycle_time / self.stride_duration + offset) % 1.0 
                  for offset in phase_offsets]
        
        # Update contact states based on phase
        for i in range(4):
            if phases[i] < duty_factor:
                # Leg in stance phase
                if self.contact_states[i] == 0:
                    # Just touched down
                    self.swing_timers[i] = 0.0
                    self.swing_phase[i] = 0.0
                self.contact_states[i] = 1
            else:
                # Leg in swing phase
                self.contact_states[i] = 0
                # Update swing progress (0 to 1)
                swing_progress = (phases[i] - duty_factor) / (1.0 - duty_factor)
                self.swing_phase[i] = swing_progress
    
    def _update_utility_based(self, utilities: np.ndarray, dt: float):
        """
        Update using utility-based adaptive gait (from whole-body MPC paper)
        This is MORE FLEXIBLE but less stable
        """
        # Update swing timers for legs in the air
        for i in range(4):
            if self.contact_states[i] == 0:
                self.swing_timers[i] += dt
                self.swing_phase[i] = min(1.0, self.swing_timers[i] / self.params.swing_duration)
                
                # Touch down when swing duration complete
                if self.swing_timers[i] >= self.params.swing_duration:
                    self.contact_states[i] = 1
                    self.swing_timers[i] = 0.0
                    self.swing_phase[i] = 0.0
        
        # Find legs that need to swing (utility below threshold)
        legs_needing_swing = []
        for i in range(4):
            if utilities[i] < self.utility_threshold and self.contact_states[i] == 1:
                legs_needing_swing.append((i, utilities[i]))
        
        # Sort by utility (lowest first = most urgent)
        legs_needing_swing.sort(key=lambda x: x[1])
        
        # Try to lift legs (with neighbor check for stability)
        for leg_idx, utility in legs_needing_swing:
            neighbors = self.get_neighbors(leg_idx)
            
            # Only lift if neighbors are in contact (maintain stability)
            if all(self.contact_states[n] == 1 for n in neighbors):
                self.contact_states[leg_idx] = 0
                self.swing_timers[leg_idx] = 0.0
                self.swing_phase[leg_idx] = 0.0
                break  # Only lift one leg at a time
    
    def get_neighbors(self, leg_idx: int) -> List[int]:
        """
        Get neighboring leg indices for stability checking
        
        Leg layout:
        0 (LF) --- 1 (RF)
           |         |
        2 (LH) --- 3 (RH)
        """
        neighbor_map = {
            0: [1, 2],  # LF neighbors: RF, LH
            1: [0, 3],  # RF neighbors: LF, RH
            2: [0, 3],  # LH neighbors: LF, RH
            3: [1, 2]   # RH neighbors: RF, LH
        }
        return neighbor_map[leg_idx]
    
    def get_swing_trajectory_point(self, leg_idx: int, 
                                   start_pos: np.ndarray,
                                   end_pos: np.ndarray) -> np.ndarray:
        """
        Get current point on swing trajectory
        Uses smooth quintic spline (from "Rolling in the Deep")
        
        Args:
            leg_idx: Which leg
            start_pos: Lift-off position [x, y, z]
            end_pos: Touch-down position [x, y, z]
        
        Returns:
            Current position on swing arc [x, y, z]
        """
        # Get swing progress (0 to 1)
        s = self.swing_phase[leg_idx]
        
        # Quintic polynomial for smooth acceleration/deceleration
        # Position goes smoothly from 0 to 1
        s_smooth = 6*s**5 - 15*s**4 + 10*s**3
        
        # Horizontal interpolation (x, y)
        pos_xy = start_pos[0:2] + (end_pos[0:2] - start_pos[0:2]) * s_smooth
        
        # Vertical trajectory: parabolic arc with peak at s=0.5
        swing_height = self.params.swing_height
        z_arc = swing_height * (1 - (2*s - 1)**2)  # Parabola
        pos_z = start_pos[2] + (end_pos[2] - start_pos[2]) * s_smooth + z_arc
        
        return np.array([pos_xy[0], pos_xy[1], pos_z])