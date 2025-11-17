import numpy as np

def ik_sagittal(L1, L2, x, z):
    """
    2-link IK in the x-z plane for joints rotating about +y.
    Hip yaw is handled outside, this returns (q_thigh, q_shank).
    x>0 forward, z<0 down from hip.
    """
    # clamp to reachable workspace
    r = np.hypot(x, z)
    r = np.clip(r, 1e-6, L1 + L2 - 1e-6)

    cos_knee = (L1**2 + L2**2 - r**2) / (2*L1*L2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    q_shank = np.pi - np.arccos(cos_knee)          # knee flexion (about +y)

    # hip pitch
    phi = np.arctan2(-z, x)                         # angle to target
    beta = np.arctan2(L2*np.sin(q_shank), L1 + L2*np.cos(q_shank))
    q_thigh = phi - beta
    return q_thigh, q_shank
