"""
Model Predictive Controller (MPC) for whole-body motion planning.

This module defines the `MPCController`, which operates on a reduced-order
Single Rigid Body Dynamics (SRBD) model of the quadruped. Given a reference
state trajectory and a contact schedule from the gait generator, it produces
a sequence of control inputs consisting of:

  * contact forces at the feet (lambda_e), and
  * joint-level "regularization" velocities (u_j),

which are later combined with swing/stance leg controllers in `main.py` and
applied to the full MuJoCo model. The MPC here is relatively simple: it uses
a handcrafted policy inside a rollout loop, rather than solving a full QP,
but it still enforces costs and physical constraints in an MPC-like fashion.
"""

import numpy as np
from typing import Tuple

from config import MPCParameters
from dynamics import SingleRigidBodyDynamics
from gait_generator import GaitSequenceGenerator

class MPCController:
    """
    Lightweight whole-body MPC controller built on top of SRBD dynamics.

    This controller:
      * Holds a reference to the reduced-order SRBD model (`SingleRigidBodyDynamics`)
        and MPC weights/parameters (`MPCParameters`).
      * Owns a `GaitSequenceGenerator` instance that produces contact schedules
        and swing phases (used elsewhere by IK and control).
      * Provides:
          - `compute_cost`: quadratic cost function for states and controls,
          - `enforce_constraints`: projection of controls into physically valid
            regions (friction cone, force limits, joint velocity limits),
          - `solve_mpc`: a rollout-based MPC routine that generates a sequence of
            contact forces and joint regularization commands over the prediction
            horizon.

    In this implementation, `solve_mpc` does not solve a full optimization
    problem; instead, it uses a structured policy (mg split + horizontal push)
    and iteratively refines forces based on velocity tracking error, while
    always enforcing constraints.
    """
    
    def __init__(self, dynamics: SingleRigidBodyDynamics, params: MPCParameters):
        """
        Initialize the MPC controller with a dynamics model and parameters.

        Args:
            dynamics (SingleRigidBodyDynamics): Reduced-order SRBD model used
                                                for forward simulation and
                                                gravity computations.
            params (MPCParameters): Container of MPC weights, limits, and
                                    horizon settings.
        """
        self.dynamics = dynamics
        self.params = params
        self.gait_gen = GaitSequenceGenerator(params)
        self.max_iterations = 5
        self.learning_rate = 0.01
        self.control_dim = 24   # 12 contact forces + 12 joint velocities
        
    def compute_cost(self, x: np.ndarray, u: np.ndarray, 
                     x_ref: np.ndarray, u_ref: np.ndarray) -> float:
        """
        Compute a quadratic cost for a single state-control pair.

        The cost penalizes deviations from a reference state and reference
        control, weighted by the fields in `MPCParameters`. It separates
        errors into orientation, position, angular velocity, linear velocity,
        and joint positions for the state, and into contact forces and joint
        velocities for the control.

        Args:
            x (np.ndarray): Current state vector, shape (24,).
            u (np.ndarray): Current control vector, shape (24,).
            x_ref (np.ndarray): Reference state vector, shape (24,).
            u_ref (np.ndarray): Reference control vector, shape (24,).

        Returns:
            float: Scalar cost value for this (x, u) pair.
        """
        x_error = x - x_ref
        
        theta_err = x_error[0:3]
        p_err = x_error[3:6]
        omega_err = x_error[6:9]
        v_err = x_error[9:12]
        q_j_err = x_error[12:24]
        
        cost_state = (
            self.params.weight_orientation * np.sum(theta_err**2) +
            self.params.weight_position * np.sum(p_err**2) +
            self.params.weight_angular_velocity * np.sum(omega_err**2) +
            self.params.weight_linear_velocity * np.sum(v_err**2) +
            self.params.weight_joint_position * np.sum(q_j_err**2)
        )
        
        u_error = u - u_ref
        lambda_err = u_error[0:12]
        u_j_err = u_error[12:24]
        
        cost_control = (
            self.params.weight_contact_force * np.sum(lambda_err**2) +
            self.params.weight_joint_velocity * np.sum(u_j_err**2)
        )
        
        return 0.5 * (cost_state + cost_control)
    
    def enforce_constraints(self, u: np.ndarray, contact_states: np.ndarray) -> np.ndarray:
        """
        Project the control input into the set of physically valid inputs.

        This method applies the following constraints:
          * For stance legs: enforce unilateral contact in the normal direction
            (no negative Fz), and restrict tangential forces to lie inside a
            Coulomb friction cone (|Ft| <= μ * Fn).
          * Limit the total magnitude of each leg's contact force.
          * For swing legs: zero out contact forces entirely.
          * Clip joint velocity commands to their maximum allowed magnitude.

        Args:
            u (np.ndarray): Raw control vector, shape (24,), containing
                            [lambda_e(0:12), u_j(12:24)].
            contact_states (np.ndarray): Per-leg stance/swing flags, shape (4,),
                                         where 1 = stance, 0 = swing.

        Returns:
            np.ndarray: Constrained control vector of the same shape, guaranteed
                        to satisfy friction, normal-force, and joint-limit
                        constraints.
        """
        u_constrained = u.copy()
        lambda_e = u_constrained[0:12].reshape(4, 3)
        u_j = u_constrained[12:24]
        
        for i in range(4):
            if contact_states[i] == 1:
                force = lambda_e[i]
                # Enforce non-negative normal force (no pulling on the ground)
                force[2] = max(0.0, force[2])
                
                # Enforce friction cone on tangential forces
                f_tangent = np.linalg.norm(force[0:2])
                f_normal = force[2]
                
                if f_tangent > self.params.friction_coeff * f_normal:
                    scale = self.params.friction_coeff * f_normal / (f_tangent + 1e-6)
                    force[0:2] *= scale
                
                # Enforce max contact force magnitude
                force_mag = np.linalg.norm(force)
                if force_mag > self.params.max_contact_force:
                    force *= self.params.max_contact_force / force_mag
                
                lambda_e[i] = force
            else:
                # No forces for swing legs
                lambda_e[i] = 0.0
        
        # Joint velocity limits
        u_j = np.clip(u_j, -self.params.max_joint_velocity, self.params.max_joint_velocity)
        
        u_constrained[0:12] = lambda_e.flatten()
        u_constrained[12:24] = u_j
        
        return u_constrained
    
    def solve_mpc(self, x0: np.ndarray, x_ref_traj: np.ndarray,
                  contact_schedule: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a stance-aware control sequence over the MPC horizon.

        This routine performs a forward rollout over the SRBD dynamics for
        `num_nodes` steps, constructing a control policy that:

          1. Supports the body weight by splitting the required normal force
             (mg in the body frame) across the stance legs at each node.

          2. Adds a horizontal push term (Fx, Fy) per stance leg that attempts
             to reduce the tracking error between the current and reference
             body-frame linear velocity (v_ref - v).

          3. Applies a small joint-position regularization term to keep the
             joint angles close to their reference values, unless they are
             overridden downstream by swing/stance IK in `main.py`.

          4. Enforces physical constraints (friction cone, force limits, joint
             velocity limits) at each step via `enforce_constraints`.

        The update is iterated a few times (`max_iterations`) to refine the
        horizontal force components using the predicted velocity error at the
        next time step.

        Args:
            x0 (np.ndarray): Current state vector, shape (24,).
            x_ref_traj (np.ndarray): Reference state trajectory over the horizon,
                                     shape (N, 24), where N = `params.num_nodes`.
            contact_schedule (np.ndarray): Contact flags over the horizon,
                                           shape (N, 4), where each row is
                                           [FL, FR, HL, HR] with 1=stance, 0=swing.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - u_seq: Control sequence of shape (N, 24), where each row is
                         [lambda_e(12), u_j(12)].
                - x_traj: Predicted state trajectory of shape (N, 24) obtained
                          by integrating the SRBD dynamics with u_seq.
        """
        N = self.params.num_nodes
        dt = self.params.dt

        # Allocate
        u_seq = np.zeros((N, self.control_dim))
        x_traj = np.zeros((N, self.dynamics.state_dim))
        x_traj[0] = x0

        # Gains (you can tune these 2–4 and 0.2–1.0)
        kv = 2.0        # horizontal velocity tracking gain (-> desired force m*kv*(v_ref - v))
        kq = 0.1        # small joint regularization toward x_ref q_j

        # We'll iterate a few times to refine the horizontal push
        for _ in range(self.max_iterations):
            total_cost = 0.0
            x_traj[0] = x0

            for k in range(N - 1):
                xk = x_traj[k]
                xref_k = x_ref_traj[k]

                theta_k = xk[0:3]
                v_k     = xk[9:12]
                v_ref   = xref_k[9:12]
                qk      = xk[12:24]
                qref_k  = xref_k[12:24]

                stance = np.asarray(contact_schedule[k]).astype(int)
                n_stance = int(np.sum(stance))

                # --------------------------
                # Construct lambda_e (body frame)
                # --------------------------
                lam = np.zeros((4, 3))

                if n_stance > 0:
                    # Gravity in body frame -> required total normal force
                    g_b = self.dynamics.gravity_in_body_frame(theta_k)
                    Fz_total = -self.params.robot_mass * g_b[2]
                    Fz_each = max(0.0, Fz_total / n_stance)

                    # Horizontal push toward velocity reference
                    Fxy_des = self.params.robot_mass * kv * (v_ref - v_k)

                    for leg in range(4):
                        if stance[leg] == 1:
                            lam[leg, 0] = Fxy_des[0] / n_stance   # Fx per stance leg
                            lam[leg, 1] = Fxy_des[1] / n_stance   # Fy per stance leg
                            lam[leg, 2] = Fz_each                 # Fz support
                        else:
                            lam[leg, :] = 0.0
                else:
                    # No stance foot (shouldn't happen in your patterns); safe fallback
                    lam[:, :] = 0.0

                # --------------------------
                # Joint-velocity regularization (gentle hold)
                # (If you generate swing IK in main.py, that will overwrite this.)
                # --------------------------
                u_j = kq * (qref_k - qk)
                u_j = np.clip(u_j, -self.params.max_joint_velocity, self.params.max_joint_velocity)

                # Pack control and enforce physical constraints
                uk = np.zeros(self.control_dim)
                uk[0:12] = lam.flatten()
                uk[12:24] = u_j
                uk = self.enforce_constraints(uk, stance)

                u_seq[k] = uk

                # Cost (purely diagnostic here)
                total_cost += self.compute_cost(xk, uk, xref_k, np.zeros(self.control_dim))

                # Roll dynamics one step with these controls
                x_traj[k + 1] = self.dynamics.integrate_euler(xk, uk, stance, dt)

            # Small refinement: if predicted v is still off, nudge horizontal forces
            for k in range(N - 1):
                stance = np.asarray(contact_schedule[k]).astype(int)
                n_stance = int(np.sum(stance))
                if n_stance == 0:
                    continue

                v_err_next = x_ref_traj[k + 1, 9:12] - x_traj[k + 1, 9:12]
                dFxy = self.params.robot_mass * 0.5 * v_err_next  # secondary gain

                for leg in range(4):
                    if stance[leg] == 1:
                        base = 3 * leg
                        u_seq[k, base + 0] += (dFxy[0] / n_stance) * self.learning_rate
                        u_seq[k, base + 1] += (dFxy[1] / n_stance) * self.learning_rate

                # Re-enforce constraints after tweaking
                u_seq[k] = self.enforce_constraints(u_seq[k], stance)

        # Last control = previous (simple hold)
        if N >= 2:
            u_seq[-1] = u_seq[-2]

        return u_seq, x_traj