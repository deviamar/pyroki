"""Sphere collision costs with analytic Jacobian computation.

Provides sphere self-collision and world-collision costs with analytical Jacobians
for faster optimization compared to autodiff.

Key optimizations:
- Precompute ancestor relationships once outside the cost function
- Use flat geometry-pair indices instead of (P, S, S) expansion for self-collision
- Compute per-link Jacobians, then index directly for each geometry pair
- No validity masking needed for self-collision - only valid pairs are in the flat index list
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from ._collision_jac_helpers import (
    compute_all_link_position_jacobians,
    compute_link_position_jacobians_sparse,
    get_joints_applied_to_all_links,
    prepare_sparse_link_indices,
)

if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision import RobotCollision, Sphere


# =============================================================================
# Self-Collision
# =============================================================================

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
    joints_applied, unique_links, link_to_sparse = prepare_sparse_link_indices(
        robot, robot_coll.geom_pair_link_i, robot_coll.geom_pair_link_j
    )

    return _sphere_self_collision_cost(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied,
        unique_links,
        link_to_sparse,
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
    joints_applied, unique_links, link_to_sparse = prepare_sparse_link_indices(
        robot, robot_coll.geom_pair_link_i, robot_coll.geom_pair_link_j
    )

    return _sphere_self_collision_constraint(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied,
        unique_links,
        link_to_sparse,
    )


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
    """Analytic Jacobian for sphere self-collision cost."""
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

    # Compute Jacobians for geometries on unique links involved in collision
    unique_geom_pos = geom_positions[unique_links]

    # Shape: (NumUniqueLinks, MaxGeoms, 3, NumActuated)
    jac_unique = compute_link_position_jacobians_sparse(
        robot, Ts_world_joint, unique_geom_pos, joints_applied_to_links, unique_links
    )

    # Gather Jacobians for active pairs
    link_i = jnp.array(robot_coll.geom_pair_link_i)
    idx_i = jnp.array(robot_coll.geom_pair_idx_i)
    link_j = jnp.array(robot_coll.geom_pair_link_j)
    idx_j = jnp.array(robot_coll.geom_pair_idx_j)

    sparse_link_i = link_to_sparse_idx[link_i]
    sparse_link_j = link_to_sparse_idx[link_j]

    # Gather: (NumPairs, 3, NumActuated)
    jac_i = jac_unique[sparse_link_i, idx_i]
    jac_j = jac_unique[sparse_link_j, idx_j]

    # Project onto normal direction
    # d_dist/dq = n . (v_j - v_i) = n . (J_j - J_i) q_dot
    jac_diff = jac_j - jac_i

    # Contract with direction n (which is unit vector from i to j)
    d_dist_d_q = jnp.einsum("pd,pda->pa", directions, jac_diff)

    # Chain Rule with Residual
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
        geom_positions,
        directions,
        distances,
        joints_applied_to_links,
        unique_links,
        link_to_sparse_idx,
        margin,
    )

    return residuals * weight, cache


_sphere_self_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_self_collision_jac
)(_sphere_self_collision_cost_impl)

_sphere_self_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_self_collision_jac,
    kind="constraint_leq_zero",
)(_sphere_self_collision_cost_impl)


# =============================================================================
# World-Collision
# =============================================================================

_WorldCollisionJacCache = tuple[
    jax.Array,  # Ts_world_joint: (num_joints, 7)
    jax.Array,  # robot_positions: (num_links, max_spheres, 3)
    jax.Array,  # directions: (num_links, max_spheres, num_world, 3)
    jax.Array,  # distances: (num_links, max_spheres, num_world) - signed distances
    jax.Array,  # valid_mask: (num_links, max_spheres) - validity mask
    jax.Array,  # joints_applied_to_links: (num_links, num_joints)
    float,  # margin: collision margin
]


def sphere_world_collision_cost_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere world-collision cost with analytic Jacobian.

    Uses max(0, margin - distance) residual for all valid sphere pairs.
    """
    joints_applied_to_links = get_joints_applied_to_all_links(robot)

    return _sphere_world_collision_cost(
        robot,
        robot_coll,
        joint_var,
        world_spheres,
        margin,
        weight,
        joints_applied_to_links,
    )


def sphere_world_collision_constraint_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere world-collision constraint with analytic Jacobian.

    Uses max(0, margin - distance) residual for all valid sphere pairs.
    Constraint version uses augmented Lagrangian for enforcement.
    """
    joints_applied_to_links = get_joints_applied_to_all_links(robot)

    return _sphere_world_collision_constraint(
        robot,
        robot_coll,
        joint_var,
        world_spheres,
        margin,
        weight,
        joints_applied_to_links,
    )


def _sphere_world_collision_jac(
    vals: jaxls.VarValues,
    jac_cache: _WorldCollisionJacCache,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
) -> jax.Array:
    """Analytic Jacobian for sphere world-collision cost."""
    del vals, joint_var, margin, world_spheres

    (
        Ts_world_joint,
        robot_positions,
        directions,
        distances,
        valid_mask,
        _,
        cached_margin,
    ) = jac_cache

    num_actuated = robot.joints.num_actuated_joints

    # Compute per-link position Jacobians (once for all links)
    # Shape: (num_links, S, 3, num_actuated)
    link_jacs = compute_all_link_position_jacobians(
        robot, Ts_world_joint, robot_positions, joints_applied_to_links
    )

    # Expand link_jacs for world dimension
    link_jacs_exp = link_jacs[:, :, None, :, :]  # (L, S, 1, 3, num_actuated)

    # d_distance/d_q = direction . d_pos_robot/d_q
    d_dist_d_q = jnp.einsum("lswd,lswda->lswa", directions, link_jacs_exp)

    # Gradient of max(0, margin - d) with respect to distance
    d_res_d_dist = jnp.where(distances < cached_margin, -1.0, 0.0)

    # Apply validity mask and compute final gradient
    active_mask = valid_mask[:, :, None]
    d_res_d_q = jnp.where(
        active_mask[..., None], d_res_d_dist[..., None] * d_dist_d_q, 0.0
    )

    return (d_res_d_q * weight).reshape(-1, num_actuated)


def _sphere_world_collision_cost_impl(
    vals: jaxls.VarValues,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
) -> tuple[jax.Array, _WorldCollisionJacCache]:
    """Compute residual and cache for sphere world-collision."""
    cfg = vals[joint_var]

    Ts_world_joint = robot._forward_kinematics_joints(cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    # Get robot sphere positions in world frame
    local_centers = robot_coll.coll.pose.translation()
    Ts_link_broadcast = jaxlie.SE3(Ts_world_link[:, None, :])
    robot_positions = Ts_link_broadcast.apply(local_centers)

    # Get world sphere positions
    world_axes = world_spheres.get_batch_axes()
    if len(world_axes) == 0:
        _world_spheres = world_spheres.broadcast_to((1,))
        num_world = 1
    else:
        _world_spheres = world_spheres
        num_world = world_axes[-1]

    world_positions = _world_spheres.pose.translation()
    world_radii = _world_spheres.radius

    robot_radii = robot_coll.coll.radius
    valid_mask = robot_coll._get_geom_valid_mask()

    # Compute distances: (num_links, S, num_world)
    robot_pos_exp = robot_positions[:, :, None, :]  # (L, S, 1, 3)
    world_pos_exp = world_positions[None, None, :, :]  # (1, 1, W, 3)
    robot_rad_exp = robot_radii[:, :, None]  # (L, S, 1)
    world_rad_exp = world_radii[None, None, :]  # (1, 1, W)

    diff = robot_pos_exp - world_pos_exp  # (L, S, W, 3)
    center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)  # (L, S, W)
    directions = diff / (center_dist[..., None] + 1e-8)  # (L, S, W, 3)
    distances = center_dist - robot_rad_exp - world_rad_exp  # (L, S, W)

    # Compute residuals using max(0, margin - d)
    residuals = jnp.maximum(0.0, margin - distances)
    residuals = jnp.where(valid_mask[:, :, None], residuals, 0.0)

    cache: _WorldCollisionJacCache = (
        Ts_world_joint,
        robot_positions,
        directions,
        distances,
        valid_mask,
        joints_applied_to_links,
        margin,
    )

    return (residuals * weight).flatten(), cache


_sphere_world_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_world_collision_jac
)(_sphere_world_collision_cost_impl)

_sphere_world_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_world_collision_jac,
    kind="constraint_leq_zero",
)(_sphere_world_collision_cost_impl)
