"""
Gait pattern generator for the quadruped.

This module implements the `GaitSequenceGenerator`, which is responsible for
deciding which legs are in stance or swing at each control step and how far
each swing has progressed. It supports both simple, fixed-phase patterns
(e.g., trot, walk, pure driving) and an optional utility-based adaptive gait
that decides when to lift a leg based on kinematic leg utility. The outputs
`contact_states` (stance/swing flags) and `swing_phase` (0–1 progress) are
consumed by the MPC controller and the swing-leg IK to produce coordinated
whole-body locomotion.
"""

import numpy as np
from typing import List

from config import MPCParameters

class GaitSequenceGenerator:
    """
    Central gait scheduler for the quadruped.

    This class maintains per-leg contact states (stance vs swing), swing
    timers, and normalized swing phases, and updates them over time according
    to a chosen gait pattern. It can operate in two modes:

      * fixed-pattern mode (`pure_driving`, `hybrid_trot`, `hybrid_walk`),
        where each leg follows a pre-defined phase offset and duty factor, and
        the resulting pattern is stable and predictable;

      * utility-based mode, where legs are lifted when their kinematic
        "utility" drops below a threshold, allowing more adaptive stepping
        while enforcing simple neighbor-based stability constraints.

    The resulting `contact_states` and `swing_phase` arrays are used by the
    higher-level controller to decide which legs should apply contact forces
    and which legs should execute a swing trajectory.
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
        """
        Switch the active gait pattern.

        Args:
            mode (str): Name of the gait mode to use. Must be one of
                        'pure_driving', 'hybrid_trot', or 'hybrid_walk'.
                        If an unknown mode is passed, the current mode
                        is left unchanged and a warning is printed.
        """
        if mode in self.gait_patterns:
            self.gait_mode = mode
            print(f"[Gait] Switched to: {mode}")
        else:
            print(f"[Gait] Unknown mode '{mode}'. Available: {list(self.gait_patterns.keys())}")
    
    def compute_leg_utility(self, leg_idx: int, 
                            current_foot_pos: np.ndarray,
                            reference_pos: np.ndarray) -> float:
        """
        Compute kinematic leg utility for a single leg.

        Utility is a scalar in [0, 1] that measures how close the current
        foot position is to a desired reference position, scaled by an
        elliptical workspace. It implements equation (7) from the whole-body
        MPC paper:

            utility = 1 - sqrt((r_parallel/λ_parallel)^2 + (r_perp/λ_perp)^2)

        where `r_parallel` is the error along the rolling (x) direction and
        `r_perp` is the error in the y–z plane.

        Args:
            leg_idx (int): Leg index (0–3). Currently unused but kept for
                           debugging/extension.
            current_foot_pos (np.ndarray): Current foot position [x, y, z].
            reference_pos (np.ndarray): Desired reference foot position [x, y, z].

        Returns:
            float: Utility value in [0, 1], where 1.0 means the foot is at
                   the ideal location and 0.0 means it is at the edge of the
                   kinematic workspace.
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
        Advance the gait state by one time step.

        This is the main entry point called from the control loop. It updates
        the internal notion of gait cycle time, then recomputes `contact_states`
        and `swing_phase` for all four legs based on the selected update rule.

        By default, it uses the fixed-pattern update (`_update_fixed_pattern`)
        for the currently selected `gait_mode`, which yields stable and
        predictable coordination. If you want more adaptive behavior, you can
        switch to the utility-based update by enabling `_update_utility_based`.

        Args:
            utilities (np.ndarray): Per-leg utility values, shape (4,). These can
                                    be dummy (e.g., all 0.8) when using the fixed
                                    pattern, but are required for utility-based
                                    gait.
            dt (float): Time step since the last update (seconds).
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
        Update contact and swing phases using a pre-defined fixed gait pattern.

        For each leg, a phase in [0, 1) is computed from the global gait cycle
        time and that leg's `phase_offset`. If the phase is below the
        `duty_factor`, the leg is in stance; otherwise it is in swing. For swing
        legs, a normalized `swing_phase` in [0, 1] is also computed to drive
        swing trajectories.

        This implementation follows the timing structure of the "Rolling in the
        Deep" paper and is intentionally simple and stable.

        Args:
            duty_factor (float): Portion of the gait cycle spent in stance
                                 (e.g., 0.6 for 60% stance, 40% swing).
            phase_offsets (List[float]): Per-leg phase offsets in [0, 1),
                                         ordered [FL, FR, HL, HR].
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
        Update contact and swing phases using a utility-based adaptive gait.

        In this mode, stance legs are lifted into swing when their kinematic
        utility drops below a threshold, subject to a simple neighbor check to
        preserve stability (never lift a leg if both of its neighbors are also
        in swing). Swing legs stay in the air for `params.swing_duration` and
        then touch down again.

        This mode is more flexible and terrain-aware than the fixed patterns,
        but typically less predictable and a bit harder to tune.

        Args:
            utilities (np.ndarray): Per-leg utility values, shape (4,).
            dt (float): Time step since the last update (seconds).
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
        Return the indices of neighboring legs used for simple stability checks.

        The leg indexing convention is:
            0: FL (front left)
            1: FR (front right)
            2: HL (hind left)
            3: HR (hind right)

        For each leg, its "neighbors" are the two legs that share either the
        same side or the same end (front/hind). This is used by the
        utility-based gait to ensure we never lift a leg when both of its
        neighbors are already in swing.

        Args:
            leg_idx (int): Index of the leg (0–3).

        Returns:
            List[int]: List of neighbor leg indices.
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
        Compute the current desired swing foot position for a given leg.

        Given a lift-off position `start_pos` and a touch-down position
        `end_pos`, this method uses the stored `swing_phase` for the leg and
        a smooth quintic spline to interpolate a 3D swing trajectory. The x–y
        coordinates are linearly interpolated, while the z coordinate follows
        a parabolic arc with peak at mid-swing, using `params.swing_height` as
        the maximum additional lift.

        Args:
            leg_idx (int): Leg index (0–3) whose swing phase should be used.
            start_pos (np.ndarray): Lift-off position [x, y, z] in some frame.
            end_pos (np.ndarray): Touch-down position [x, y, z] in the same frame.

        Returns:
            np.ndarray: Current target position on the swing arc, shape (3,).
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
