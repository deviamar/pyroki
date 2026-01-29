"""Core residual functions for pyroki costs and constraints.

These functions have the same signature as cost/constraint functions
(taking `vals: VarValues` as first arg), so they can be directly wrapped:

    limit_cost = Cost.create_factory(limit_residual)
    limit_constraint = Constraint.create_factory(limit_residual, constraint_type="leq_zero")
"""

import jax
import jax.numpy as jnp
import jaxlie
from jax import Array
from jaxls import Var, VarValues

from .._robot import Robot
from ..collision import CollGeom, RobotCollision, colldist_from_sdf


# --- Pose Residuals ---


def pose_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_pose: jaxlie.SE3,
    target_link_index: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses to target poses."""
    assert target_link_index.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    Ts_link_world = robot.forward_kinematics(joint_cfg)
    pose_actual = jaxlie.SE3(Ts_link_world[..., target_link_index, :])
    residual = (pose_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


def pose_with_base_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    target_pose: jaxlie.SE3,
    target_link_indices: Array,
    pos_weight: Array | float,
    ori_weight: Array | float,
) -> Array:
    """Computes the residual for matching link poses relative to a mobile base."""
    assert target_link_indices.dtype == jnp.int32
    joint_cfg = vals[joint_var]
    T_world_base = vals[T_world_base_var]
    Ts_base_link = robot.forward_kinematics(joint_cfg)  # FK is T_base_link
    T_base_target_link = jaxlie.SE3(Ts_base_link[..., target_link_indices, :])
    T_world_target_link_actual = T_world_base @ T_base_target_link

    residual = (T_world_target_link_actual.inverse() @ target_pose).log()
    pos_residual = residual[..., :3] * pos_weight
    ori_residual = residual[..., 3:] * ori_weight
    return jnp.concatenate([pos_residual, ori_residual]).flatten()


# --- Limit Residuals ---


def limit_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint limit violation residual.

    Returns values that are:
    - Positive when violated (joint outside limits)
    - Negative when satisfied (joint within limits)

    For inequality constraints: residual <= 0 means satisfied.
    """
    joint_cfg = vals[joint_var]
    joint_cfg_eff = robot.joints.get_full_config(joint_cfg)
    upper_violation = jnp.maximum(0.0, joint_cfg_eff - robot.joints.upper_limits_all)
    lower_violation = jnp.maximum(0.0, robot.joints.lower_limits_all - joint_cfg_eff)
    return (jnp.concatenate([upper_violation, lower_violation]) * weight).flatten()


def limit_velocity_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    prev_joint_var: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint velocity limit violation residual.

    Returns values that are:
    - Positive when violated (|velocity| > limit)
    - Zero when satisfied (|velocity| <= limit)
    """
    joint_vel = (vals[joint_var] - vals[prev_joint_var]) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


# --- Regularization Residuals ---


def rest_residual(
    vals: VarValues,
    joint_var: Var[Array],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints towards a rest pose."""
    return ((vals[joint_var] - rest_pose) * weight).flatten()


def rest_with_base_residual(
    vals: VarValues,
    joint_var: Var[Array],
    T_world_base_var: Var[jaxlie.SE3],
    rest_pose: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual biasing joints and base towards its default pose."""
    residual_joints = vals[joint_var] - rest_pose
    residual_base = (
        vals[T_world_base_var].inverse() @ T_world_base_var.default_factory()
    ).log()
    return (jnp.concatenate([residual_joints, residual_base]) * weight).flatten()


def smoothness_residual(
    vals: VarValues,
    curr_joint_var: Var[Array],
    past_joint_var: Var[Array],
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing joint configuration differences (velocity)."""
    return ((vals[curr_joint_var] - vals[past_joint_var]) * weight).flatten()


# --- Manipulability Residual ---


def _compute_manip_yoshikawa(
    cfg: Array,
    robot: Robot,
    target_link_index: jax.Array,
) -> Array:
    """Helper: Computes manipulability measure for a single link."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[target_link_index]
    JJT = jacobian @ jacobian.T
    assert JJT.shape == (3, 3)
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(JJT)))


def manipulability_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    target_link_indices: Array,
    weight: Array | float = 1.0,
) -> Array:
    """Computes the residual penalizing low manipulability (translation)."""
    cfg = vals[joint_var]
    if target_link_indices.ndim == 0:
        vmapped_manip = _compute_manip_yoshikawa(cfg, robot, target_link_indices)
    else:
        vmapped_manip = jax.vmap(_compute_manip_yoshikawa, in_axes=(None, None, 0))(
            cfg, robot, target_link_indices
        )
    residual = 1.0 / (vmapped_manip + 1e-6)
    return (residual * weight).flatten()


# --- Collision Residuals ---


def self_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes self-collision violation residual. Residual is >0 if collision is detected."""
    cfg = vals[joint_var]
    active_distances = robot_coll.compute_self_collision_distance(robot, cfg)
    return -(colldist_from_sdf(active_distances, margin) * weight).flatten()


def world_collision_residual(
    vals: VarValues,
    robot: Robot,
    robot_coll: RobotCollision,
    joint_var: Var[Array],
    world_geom: CollGeom,
    margin: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes world collision violation residual. Residual is >0 if collision is detected."""
    cfg = vals[joint_var]
    dist_matrix = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
    return -(colldist_from_sdf(dist_matrix, margin) * weight).flatten()


# --- Finite Difference Residuals ---


def five_point_velocity_residual(
    vals: VarValues,
    robot: Robot,
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes velocity limit violation using 5-point stencil.

    Returns values that are:
    - Positive when violated (|velocity| > limit)
    - Negative when satisfied (|velocity| <= limit)
    """
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
    vel_limits = robot.joints.velocity_limits
    residual = jnp.maximum(0.0, jnp.abs(velocity) - vel_limits)
    return (residual * weight).flatten()


def five_point_acceleration_residual(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint acceleration using 5-point stencil."""
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    residual = jnp.abs(acceleration)
    return (residual * weight).flatten()


def limit_acceleration_residual(
    vals: VarValues,
    var_t: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    dt: float,
    acceleration_limit: Array | float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint acceleration limit violation residual using 5-point stencil.

    Returns values that are:
    - Positive when violated (|acceleration| > limit)
    - Negative when satisfied (|acceleration| <= limit)
    """
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_t = vals[var_t]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]

    acceleration = (-q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2) / (12 * dt**2)
    residual = jnp.maximum(0.0, jnp.abs(acceleration) - acceleration_limit)
    return (residual * weight).flatten()


def five_point_jerk_residual(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint jerk using 7-point stencil."""
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]

    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    return (jnp.abs(jerk) * weight).flatten()


def limit_jerk_residual(
    vals: VarValues,
    var_t_plus_3: Var[Array],
    var_t_plus_2: Var[Array],
    var_t_plus_1: Var[Array],
    var_t_minus_1: Var[Array],
    var_t_minus_2: Var[Array],
    var_t_minus_3: Var[Array],
    dt: float,
    jerk_limit: Array | float,
    weight: Array | float = 1.0,
) -> Array:
    """Computes joint jerk limit violation residual using 7-point stencil.

    Returns values that are:
    - Positive when violated (|jerk| > limit)
    - Negative when satisfied (|jerk| <= limit)
    """
    q_tm3 = vals[var_t_minus_3]
    q_tm2 = vals[var_t_minus_2]
    q_tm1 = vals[var_t_minus_1]
    q_tp1 = vals[var_t_plus_1]
    q_tp2 = vals[var_t_plus_2]
    q_tp3 = vals[var_t_plus_3]

    jerk = (-q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3) / (
        8 * dt**3
    )
    residual = jnp.maximum(0.0, jnp.abs(jerk) - jerk_limit)
    return (residual * weight).flatten()


# --- Loop Closure Residuals ---


def loop_closure_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    link_a_index: Array,
    link_b_index: Array,
    T_a_b: jaxlie.SE3,
    pos_weight: Array | float = 1.0,
    ori_weight: Array | float = 1.0,
) -> Array:
    """Computes the residual for enforcing loop closure between two links.

    This residual function penalizes deviations from the expected relative transform
    between two links in the kinematic chain. It can be used to model closed-loop
    mechanisms like 4-bar linkages, parallel robots, or Stewart platforms.

    The residual is computed as the SE(3) logarithm of the error between the
    actual and expected relative transforms:
        error = (T_world_a.inverse() @ T_world_b @ T_a_b.inverse()).log()

    Args:
        vals: The optimization variable values.
        robot: The robot model.
        joint_var: The joint configuration variable.
        link_a_index: Index of the first link (int32 array).
        link_b_index: Index of the second link (int32 array).
        T_a_b: The expected SE(3) transform from link A to link B.
        pos_weight: Weight for positional error (translation components).
        ori_weight: Weight for orientation error (rotation components).

    Returns:
        Flattened residual array (6 elements: 3 position + 3 orientation).
    """
    assert link_a_index.dtype == jnp.int32
    assert link_b_index.dtype == jnp.int32

    joint_cfg = vals[joint_var]
    Ts_world_link = robot.forward_kinematics(joint_cfg)

    T_world_a = jaxlie.SE3(Ts_world_link[..., link_a_index, :])
    T_world_b = jaxlie.SE3(Ts_world_link[..., link_b_index, :])

    # Compute actual relative transform from A to B
    T_a_b_actual = T_world_a.inverse() @ T_world_b

    # Compute error: how far is actual from expected
    error = (T_a_b_actual @ T_a_b.inverse()).log()

    pos_residual = error[..., :3] * pos_weight
    ori_residual = error[..., 3:] * ori_weight

    return jnp.concatenate([pos_residual, ori_residual]).flatten()


def loop_closure_residual_unweighted(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    link_a_index: Array,
    link_b_index: Array,
    T_a_b: jaxlie.SE3,
) -> Array:
    """Computes the residual for enforcing exact loop closure between two links.

    This is an unweighted version of loop_closure_residual, suitable for use
    as a hard equality constraint (Augmented Lagrangian). Use this when the
    loop closure must be satisfied exactly rather than just minimized.

    Args:
        vals: The optimization variable values.
        robot: The robot model.
        joint_var: The joint configuration variable.
        link_a_index: Index of the first link (int32 array).
        link_b_index: Index of the second link (int32 array).
        T_a_b: The expected SE(3) transform from link A to link B.

    Returns:
        Flattened residual array (6 elements: 3 position + 3 orientation).
    """
    assert link_a_index.dtype == jnp.int32
    assert link_b_index.dtype == jnp.int32

    joint_cfg = vals[joint_var]
    Ts_world_link = robot.forward_kinematics(joint_cfg)

    T_world_a = jaxlie.SE3(Ts_world_link[..., link_a_index, :])
    T_world_b = jaxlie.SE3(Ts_world_link[..., link_b_index, :])

    # Compute actual relative transform from A to B
    T_a_b_actual = T_world_a.inverse() @ T_world_b

    # Compute error: how far is actual from expected
    error = (T_a_b_actual @ T_a_b.inverse()).log()

    return error.flatten()


def stewart_closure_residual(
    vals: VarValues,
    robot: Robot,
    joint_var: Var[Array],
    platform_var: Var[jaxlie.SE3],
    tip_link_indices: Array,
    T_platform_tips: jaxlie.SE3,
    weight: Array | float = 1.0,
) -> Array:
    """Computes position-only closure residual for parallel manipulator legs.

    This residual function enforces that each leg tip (from forward kinematics)
    matches its expected attachment point on the moving platform. It is designed
    for Stewart platforms and similar parallel manipulators where multiple legs
    connect a fixed base to a moving platform.

    The residual only considers position (not orientation), as the spherical
    joints at the leg tips allow arbitrary orientation.

    Args:
        vals: The optimization variable values.
        robot: The robot model.
        joint_var: The joint configuration variable.
        platform_var: The SE3 variable representing the platform pose in world frame.
        tip_link_indices: Indices of the leg tip links (int32 array, shape (n_legs,)).
        T_platform_tips: SE3 transforms from platform frame to each tip attachment
            point (batched, shape (n_legs,)).
        weight: Weight for position error. Can be scalar or per-leg array.

    Returns:
        Flattened residual array (3 * n_legs elements for position errors).
    """
    assert tip_link_indices.dtype == jnp.int32

    joint_cfg = vals[joint_var]
    T_world_platform = vals[platform_var]
    Ts_world_link = robot.forward_kinematics(joint_cfg)

    # Compute position errors for each leg
    def compute_leg_error(tip_idx: Array, T_platform_tip: jaxlie.SE3) -> Array:
        # Actual tip position from FK
        T_world_tip = jaxlie.SE3(Ts_world_link[..., tip_idx, :])
        tip_pos_actual = T_world_tip.translation()

        # Expected tip position from platform pose
        T_world_target = T_world_platform @ T_platform_tip
        tip_pos_expected = T_world_target.translation()

        return (tip_pos_actual - tip_pos_expected) * weight

    # Vmap over all legs
    pos_errors = jax.vmap(compute_leg_error)(tip_link_indices, T_platform_tips)

    return pos_errors.flatten()
