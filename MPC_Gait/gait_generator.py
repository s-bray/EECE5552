# gait_generator.py

import numpy as np
from typing import List

from config import MPCParameters

class GaitSequenceGenerator:
    """Generates gait sequences based on kinematic leg utilities"""
    
    def __init__(self, params: MPCParameters):
        self.params = params
        self.lambda_parallel = 0.4
        self.lambda_perpendicular = 0.2
        self.utility_threshold = 0.3
        self.contact_states = np.ones(4)
        self.swing_timers = np.zeros(4)
        
    def compute_leg_utility(self, leg_idx: int, 
                           current_foot_pos: np.ndarray,
                           reference_pos: np.ndarray) -> float:
        """Compute kinematic leg utility"""
        r_error = reference_pos - current_foot_pos
        r_parallel = abs(r_error[0])
        r_perp = abs(r_error[1])
        
        utility = 1.0 - np.sqrt(
            (r_parallel / self.lambda_parallel)**2 + 
            (r_perp / self.lambda_perpendicular)**2
        )
        
        return np.clip(utility, 0.0, 1.0)
    
    def update_gait(self, utilities: np.ndarray, dt: float):
        """Update gait sequence based on utilities"""
        for i in range(4):
            if self.contact_states[i] == 0:
                self.swing_timers[i] += dt
                if self.swing_timers[i] >= self.params.swing_duration:
                    self.contact_states[i] = 1
                    self.swing_timers[i] = 0.0
        
        for i in np.argsort(utilities):
            if utilities[i] < self.utility_threshold and self.contact_states[i] == 1:
                neighbors = self.get_neighbors(i)
                if all(self.contact_states[n] == 1 for n in neighbors):
                    self.contact_states[i] = 0
                    self.swing_timers[i] = 0.0
                    break
    
    def get_neighbors(self, leg_idx: int) -> List[int]:
        """Get neighboring leg indices"""
        neighbor_map = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        return neighbor_map[leg_idx]