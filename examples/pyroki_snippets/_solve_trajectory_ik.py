"""
Tracking IK solver for real robot control.

This version expands on the basic IK solver by adding:

- SE(3) pose tracking
    - translational tracking
    - rotational tracking

- Temporal continuity
    - penalizes large jumps between timesteps

- Velocity-aware regularization
    - discourages commands exceeding actuator capability

- Joint limit constraints

"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

"""
Solves a tracking IK problem for real-time control.

Args:
    robot:
        PyRoKi robot.

    target_link_name:
        End-effector link name.

    target_wxyz:
        Target orientation quaternion.
        Shape: (4,)

    target_position:
        Target position.
        Shape: (3,)

    prev_q:
        Previous joint configuration.
        Shape: (num_joints,)

    dt:
        Control timestep in seconds.

    joint_velocity_limits:
        Joint velocity limits in rad/s.
        Shape: (num_joints,)

    pos_weight:
        Position tracking weight.

    ori_weight:
        Orientation tracking weight.

    dq_weight:
        Smoothness / velocity regularization weight.

Returns:
    q_next:
        Next joint configuration.
        Shape: (num_joints,)
"""

def solve_trajectory_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_position: np.ndarray,
    target_wxyz: np.ndarray,
    prev_q: np.ndarray,
    dt: float,
    joint_velocity_limits: np.ndarray,
    left_arm_indices: np.ndarray,
    pos_weight: float = 50.0,
    ori_weight: float = 0.0,
    dq_weight: float = 0.5,
) -> np.ndarray:

    assert target_position.shape == (3,)
    assert target_wxyz.shape == (4,)

    assert prev_q.shape == (
        robot.joints.num_actuated_joints,
    )

    assert joint_velocity_limits.shape == (
        robot.joints.num_actuated_joints,
    )

    target_link_index = (
        robot.links.names.index(target_link_name)
    )

    q_next = _solve_trajectory_ik_jax(
        robot=robot,
        target_link_index=jnp.array(target_link_index),
        target_wxyz=jnp.array(target_wxyz),
        target_position=jnp.array(target_position),
        prev_q=jnp.array(prev_q),
        dt=jnp.array(dt),
        joint_velocity_limits=jnp.array(
            joint_velocity_limits
        ),
        left_arm_indices=jnp.array(left_arm_indices),
        pos_weight=jnp.array(pos_weight),
        ori_weight=jnp.array(ori_weight),
        dq_weight=jnp.array(dq_weight),
    )

    q_next = np.array(q_next)

    # Clamp velocity to match hardware and safety constraints

    dq = q_next - prev_q

    max_dq = (
        joint_velocity_limits * dt
    )

    dq = np.clip(
        dq,
        -max_dq,
        max_dq,
    )

    q_next = prev_q + dq

    return q_next


@jaxls.Cost.create_factory
def previous_configuration_residual(
    vals,
    joint_var,
    prev_q,
    weight,
):
    q = vals[joint_var]
    return weight * (q - prev_q)



@jdc.jit
def _solve_trajectory_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_position: jax.Array,
    target_wxyz: jax.Array,
    prev_q: jax.Array,
    dt: jax.Array,
    joint_velocity_limits: jax.Array,
    left_arm_indices: np.ndarray,
    pos_weight: jax.Array,
    ori_weight: jax.Array,
    dq_weight: jax.Array,
) -> jax.Array:
    
    left_arm_indices=jnp.array(
        left_arm_indices
    ),

    joint_var = robot.joint_var_cls(0)

    variables = [joint_var]

    # TARGET TRANSFORM

    T_world_target = (
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz),
            target_position,
        )
    )

    # ========================================================
    # JOINT MASK
    # ========================================================
    #
    # Optimize ONLY left arm joints.
    # All other joints are frozen.
    #
    # ========================================================

    # joint_mask = jnp.zeros(
    #     robot.joints.num_actuated_joints
    # )

    # joint_mask = joint_mask.at[
    #     left_arm_indices
    # ].set(1.0)

    # COSTS

    costs = [

        # ----------------------------------------------------
        # POSE TRACKING COST
        # ----------------------------------------------------
        #
        # Main SE(3) tracking objective.
        #
        # Penalizes:
        #   - position error
        #   - orientation error
        #
        # ----------------------------------------------------

        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            T_world_target,
            target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            #joint_mask=joint_mask,
        ),

        # ----------------------------------------------------
        # JOINT LIMIT CONSTRAINT
        # ----------------------------------------------------

        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),

        # ----------------------------------------------------
        # DQ / VELOCITY REGULARIZATION
        # ----------------------------------------------------
        #
        # Encourages:
        #
        #   q_next ~= prev_q
        #
        # while scaling by velocity limits and dt.
        #
        # This behaves like:
        #
        #   || dq / (vmax * dt) ||^2
        #
        # ----------------------------------------------------

        previous_configuration_residual(
            joint_var=joint_var,
            prev_q=prev_q,
            weight=dq_weight,
        ),
        
    ]

    # ========================================================
    # SOLVE
    # ========================================================

    problem = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=variables,
        )
    )

    sol = (
        problem
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",

            # ------------------------------------------------
            # Trust region damping.
            #
            # Helps stabilize nonlinear solves.
            # ------------------------------------------------

            trust_region=jaxls.TrustRegionConfig(
                lambda_initial=1.0,
            ),
        )
    )

    q_next = sol[joint_var]

    return q_next