"""Sphere self-collision cost with analytic Jacobian computation.

Optimized implementation using flat geometry-pair indexing.
Key optimizations:
- Precompute ancestor relationships once outside the cost function
- Use flat geometry-pair indices instead of (P, S, S) expansion
- Compute per-link Jacobians, then index directly for each geometry pair
- No validity masking needed - only valid pairs are in the flat index list
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxlie
import jaxls


if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision import RobotCollision

# Cache with flat geometry-pair structure
_SelfCollisionJacCache = tuple[
    jax.Array,  # Ts_world_joint: (num_joints, 7)
    jax.Array,  # geom_positions: (num_links, max_geoms, 3)
    jax.Array,  # directions: (num_geom_pairs, 3) - flat!
    jax.Array,  # distances: (num_geom_pairs,) - flat!
    jax.Array,  # joints_applied_to_links: (num_unique_links, num_joints)
    jax.Array,  # unique_links: (num_unique_links,)
    jax.Array,  # link_to_sparse_idx: (num_links,)
    float,  # margin: collision margin
]


def _get_actuated_joints_applied_to_target(
    robot: "Robot",
    target_joint_idx: jax.Array,
    actuated_indices: jax.Array,
    mimic_act_indices: jax.Array,
    parent_indices: jax.Array,
) -> jax.Array:
    """For each joint in the robot, return actuated joint index if it affects target.

    Returns (num_joints,) array where value is actuated index if joint affects
    target, else -1.
    """

    def body_fun(carry):
        joint_idx, indices = carry
        active_act_joint = jnp.where(
            actuated_indices[joint_idx] != -1,
            actuated_indices[joint_idx],
            mimic_act_indices[joint_idx],
        )
        parent_joint = parent_indices[joint_idx]
        next_indices = indices.at[joint_idx].set(active_act_joint)
        return (parent_joint, next_indices)

    def cond_fun(carry):
        joint_idx, _ = carry
        return joint_idx >= 0

    idx_applied_to_target = jnp.full(
        (robot.joints.num_joints,),
        fill_value=-1,
        dtype=jnp.int32,
    )
    _, idx_applied_to_target = jax.lax.while_loop(
        cond_fun, body_fun, (target_joint_idx, idx_applied_to_target)
    )
    return idx_applied_to_target


def _create_joint_to_actuated_matrix(
    joints_applied: jax.Array,
    num_actuated: int,
) -> jax.Array:
    """Create one-hot matrix mapping joints to actuated indices."""
    valid_mask = joints_applied >= 0
    safe_indices = jnp.maximum(joints_applied, 0)
    one_hot = jax.nn.one_hot(safe_indices, num_actuated)
    return one_hot * valid_mask[:, None]


def _get_joints_applied_to_all_links(robot: "Robot") -> jax.Array:
    """Compute actuated joint indices affecting each link.

    Returns (num_links, num_joints) array where value [l, j] is the actuated
    joint index if joint j affects link l, else -1.
    """
    num_links = robot.links.num_links

    actuated_indices = jnp.array(robot.joints.actuated_indices, dtype=jnp.int32)
    mimic_act_indices = jnp.array(robot.joints.mimic_act_indices, dtype=jnp.int32)
    parent_indices = jnp.array(robot.joints.parent_indices, dtype=jnp.int32)
    parent_joint_indices = jnp.array(robot.links.parent_joint_indices, dtype=jnp.int32)

    target_joint_for_link = jnp.where(
        parent_joint_indices == -1, 0, parent_joint_indices
    )

    def get_joints_for_link(link_idx: jax.Array) -> jax.Array:
        target_joint = target_joint_for_link[link_idx]
        result = _get_actuated_joints_applied_to_target(
            robot, target_joint, actuated_indices, mimic_act_indices, parent_indices
        )
        return jnp.where(parent_joint_indices[link_idx] == -1, -1, result)

    return jax.vmap(get_joints_for_link)(jnp.arange(num_links))


def _compute_all_link_position_jacobians(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    sphere_positions: jax.Array,
    joints_applied_to_links: jax.Array,
) -> jax.Array:
    """Compute position Jacobians for all spheres on all links.

    Uses dense matmul instead of scatter-add for better XLA fusion.

    Args:
        robot: Robot model.
        Ts_world_joint: Joint poses, shape (num_joints, 7).
        sphere_positions: Sphere positions in world frame, (num_links, S, 3).
        joints_applied_to_links: Precomputed (num_links, num_joints) matrix.

    Returns:
        Jacobian of shape (num_links, S, 3, num_actuated).
    """
    num_actuated = robot.joints.num_actuated_joints

    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_world = Ts_world_joint_se3.rotation() @ joint_twists[:, 3:]
    vel_world = Ts_world_joint_se3.rotation() @ joint_twists[:, :3]
    joint_positions = Ts_world_joint_se3.translation()

    def compute_link_jacobian(link_idx: jax.Array) -> jax.Array:
        link_sphere_pos = sphere_positions[link_idx]
        joints_applied = joints_applied_to_links[link_idx]
        joint_to_act = _create_joint_to_actuated_matrix(joints_applied, num_actuated)

        r = link_sphere_pos[:, None, :] - joint_positions[None, :, :]
        vel_per_joint = jnp.cross(omega_world[None, :, :], r) + vel_world[None, :, :]
        jac = jnp.einsum("sjd,ja->sda", vel_per_joint, joint_to_act)
        return jac

    return jax.vmap(compute_link_jacobian)(jnp.arange(sphere_positions.shape[0]))


def _compute_link_position_jacobians_sparse(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    sphere_positions: jax.Array,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
) -> jax.Array:
    """Compute position Jacobians for spheres on specified unique links."""
    del unique_links  # Unused in computation, implicitly defined by inputs
    num_actuated = robot.joints.num_actuated_joints

    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_world = Ts_world_joint_se3.rotation() @ joint_twists[:, 3:]
    vel_world = Ts_world_joint_se3.rotation() @ joint_twists[:, :3]
    joint_positions = Ts_world_joint_se3.translation()

    def compute_link_jacobian(idx: jax.Array) -> jax.Array:
        # idx is index into the passed sphere_positions array
        link_sphere_pos = sphere_positions[idx]
        joints_applied = joints_applied_to_links[idx]
        joint_to_act = _create_joint_to_actuated_matrix(joints_applied, num_actuated)

        r = link_sphere_pos[:, None, :] - joint_positions[None, :, :]
        vel_per_joint = jnp.cross(omega_world[None, :, :], r) + vel_world[None, :, :]
        jac = jnp.einsum("sjd,ja->sda", vel_per_joint, joint_to_act)
        return jac

    return jax.vmap(compute_link_jacobian)(jnp.arange(sphere_positions.shape[0]))


def _sphere_self_collision_jac(
    vals: jaxls.VarValues,
    jac_cache: _SelfCollisionJacCache,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
    link_to_sparse_idx: jax.Array,
) -> jax.Array:
    """Analytic Jacobian for sphere self-collision cost.

    Computes Jacobian by gathering per-link Jacobians.
    Handles mimic joints correctly.
    """
    del vals, joint_var, margin

    (
        Ts_world_joint,
        geom_positions,
        directions,
        distances,
        _,
        _,
        _,
        cached_margin,
    ) = jac_cache

    # 1. Compute Jacobians for geometries on unique links involved in collision
    # Cache contains all links; slice for unique ones
    unique_geom_pos = geom_positions[unique_links]

    # Shape: (NumUniqueLinks, MaxGeoms, 3, NumActuated)
    jac_unique = _compute_link_position_jacobians_sparse(
        robot, Ts_world_joint, unique_geom_pos, joints_applied_to_links, unique_links
    )

    # 2. Gather Jacobians for active pairs
    link_i = jnp.array(robot_coll.geom_pair_link_i)
    idx_i = jnp.array(robot_coll.geom_pair_idx_i)
    link_j = jnp.array(robot_coll.geom_pair_link_j)
    idx_j = jnp.array(robot_coll.geom_pair_idx_j)

    sparse_link_i = link_to_sparse_idx[link_i]
    sparse_link_j = link_to_sparse_idx[link_j]

    # Gather: (NumPairs, 3, NumActuated)
    jac_i = jac_unique[sparse_link_i, idx_i]
    jac_j = jac_unique[sparse_link_j, idx_j]

    # 3. Project onto normal direction
    # d_dist/dq = n . (v_j - v_i) = n . (J_j - J_i) q_dot
    jac_diff = jac_j - jac_i

    # Contract with direction n (which is unit vector from i to j)
    # result: (NumPairs, NumActuated)
    d_dist_d_q = jnp.einsum("pd,pda->pa", directions, jac_diff)

    # 4. Chain Rule with Residual
    d_res_d_dist = jnp.where(distances < cached_margin, -1.0, 0.0)
    d_res_d_q = d_res_d_dist[:, None] * d_dist_d_q

    return d_res_d_q * weight


def _sphere_self_collision_cost_impl(
    vals: jaxls.VarValues,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
    link_to_sparse_idx: jax.Array,
) -> tuple[jax.Array, _SelfCollisionJacCache]:
    """Compute residual and cache for sphere self-collision using flat indexing."""
    cfg = vals[joint_var]

    Ts_world_joint = robot._forward_kinematics_joints(cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    local_centers = robot_coll.coll.pose.translation()
    Ts_link_broadcast = jaxlie.SE3(Ts_world_link[:, None, :])
    geom_positions = Ts_link_broadcast.apply(local_centers)

    # Use flat distance computation
    num_geom_pairs = len(robot_coll.geom_pair_link_i)

    if num_geom_pairs == 0:
        empty_residuals = jnp.zeros((0,))
        cache: _SelfCollisionJacCache = (
            Ts_world_joint,
            geom_positions,
            jnp.zeros((0, 3)),  # directions
            jnp.zeros((0,)),  # distances
            joints_applied_to_links,
            unique_links,
            link_to_sparse_idx,
            margin,
        )
        return empty_residuals, cache

    # Convert flat geometry-pair indices to arrays
    link_i = jnp.array(robot_coll.geom_pair_link_i)
    idx_i = jnp.array(robot_coll.geom_pair_idx_i)
    link_j = jnp.array(robot_coll.geom_pair_link_j)
    idx_j = jnp.array(robot_coll.geom_pair_idx_j)

    # Direct flat indexing - no S×S expansion
    pos_i = geom_positions[link_i, idx_i, :]  # (num_geom_pairs, 3)
    pos_j = geom_positions[link_j, idx_j, :]  # (num_geom_pairs, 3)
    rad_i = robot_coll.coll.radius[link_i, idx_i]  # (num_geom_pairs,)
    rad_j = robot_coll.coll.radius[link_j, idx_j]  # (num_geom_pairs,)

    # Compute distances and directions
    diff = pos_j - pos_i  # (num_geom_pairs, 3)
    center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)  # (num_geom_pairs,)
    directions = diff / (center_dist[:, None] + 1e-8)  # (num_geom_pairs, 3)
    distances = center_dist - rad_i - rad_j  # (num_geom_pairs,)

    # Compute residuals: max(0, margin - d)
    residuals = jnp.maximum(0.0, margin - distances)

    cache = (
        Ts_world_joint,
        sphere_positions,
        directions,
        distances,
        joints_applied_to_links,
        unique_links,
        link_to_sparse_idx,
        margin,
    )

    return residuals * weight, cache


# Create cost and constraint versions using the factory pattern
_sphere_self_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_self_collision_jac
)(_sphere_self_collision_cost_impl)

_sphere_self_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_self_collision_jac,
    kind="constraint_leq_zero",
)(_sphere_self_collision_cost_impl)


def sphere_self_collision_cost_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere self-collision cost with analytic Jacobian.

    Uses flat geometry-pair indexing for efficiency - no (P, S, S) expansion.
    """
    all_joints_applied = _get_joints_applied_to_all_links(robot)

    unique_links_set = sorted(
        set(robot_coll.geom_pair_link_i) | set(robot_coll.geom_pair_link_j)
    )
    unique_links = jnp.array(unique_links_set, dtype=jnp.int32)

    joints_applied_to_links = all_joints_applied[unique_links]

    link_to_sparse_idx = jnp.full(robot.links.num_links, -1, dtype=jnp.int32)
    link_to_sparse_idx = link_to_sparse_idx.at[unique_links].set(
        jnp.arange(len(unique_links), dtype=jnp.int32)
    )

    return _sphere_self_collision_cost(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied_to_links,
        unique_links,
        link_to_sparse_idx,
    )


def sphere_self_collision_constraint_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere self-collision constraint with analytic Jacobian.

    Uses flat geometry-pair indexing for efficiency - no (P, S, S) expansion.
    Constraint version uses augmented Lagrangian for enforcement.
    """
    all_joints_applied = _get_joints_applied_to_all_links(robot)

    unique_links_set = sorted(
        set(robot_coll.geom_pair_link_i) | set(robot_coll.geom_pair_link_j)
    )
    unique_links = jnp.array(unique_links_set, dtype=jnp.int32)

    joints_applied_to_links = all_joints_applied[unique_links]

    link_to_sparse_idx = jnp.full(robot.links.num_links, -1, dtype=jnp.int32)
    link_to_sparse_idx = link_to_sparse_idx.at[unique_links].set(
        jnp.arange(len(unique_links), dtype=jnp.int32)
    )

    return _sphere_self_collision_constraint(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied_to_links,
        unique_links,
        link_to_sparse_idx,
    )
