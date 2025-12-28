"""Sphere world-collision cost with analytic Jacobian computation.

Optimized implementation following the efficient pattern from pose_cost_analytic_jac.
Key optimizations:
- Precompute ancestor relationships once outside the cost function
- Compute per-link Jacobians in a single vectorized pass
- Use chain rule with simple indexing for collision pairs
- No Python for-loops or nested while_loops in traced code
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxlie
import jaxls


# Reuse helper functions from self-collision module
from ._sphere_self_collision_analytic_jac import (
    _compute_all_link_position_jacobians,
    _get_joints_applied_to_all_links,
)

if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision import RobotSphereCollision, Sphere

# Cache now includes precomputed joints_applied_to_links
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
    robot_coll: "RobotSphereCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere world-collision cost with analytic Jacobian.

    Uses max(0, margin - distance) residual for all valid sphere pairs.

    This implementation precomputes the kinematic tree relationships once
    for efficiency, avoiding repeated while_loop calls during optimization.
    """
    # Precompute which joints affect each link (done once, outside cost function)
    joints_applied_to_links = _get_joints_applied_to_all_links(robot)

    return _sphere_world_collision_cost(
        robot, robot_coll, joint_var, world_spheres, margin, weight,
        joints_applied_to_links
    )


def _sphere_world_collision_jac(
    vals: jaxls.VarValues,
    jac_cache: _WorldCollisionJacCache,
    robot: "Robot",
    robot_coll: "RobotSphereCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
) -> jax.Array:
    """Analytic Jacobian for sphere world-collision cost.

    Uses chain rule: ∂residual/∂q = ∂residual/∂dist × ∂dist/∂pos × ∂pos/∂q

    Note: World spheres are static, so only robot sphere positions contribute.

    The gradient of max(0, margin - d) with respect to d is:
        - For d >= margin: 0 (residual is 0)
        - For d < margin: -1 (residual is margin - d)
    """
    del vals, joint_var, margin, world_spheres

    (Ts_world_joint, robot_positions, directions, distances, valid_mask, _, cached_margin) = jac_cache

    num_links = robot_coll.num_links
    S = robot_coll.max_spheres_per_link
    num_world = directions.shape[2]
    num_actuated = robot.joints.num_actuated_joints

    # Compute per-link position Jacobians (once for all links)
    # Shape: (num_links, S, 3, num_actuated)
    link_jacs = _compute_all_link_position_jacobians(
        robot, Ts_world_joint, robot_positions, joints_applied_to_links
    )

    # directions: (num_links, S, num_world, 3)
    # link_jacs: (num_links, S, 3, num_actuated)
    # Need to compute: direction · jac for each (link, sphere, world) pair

    # Expand link_jacs for world dimension
    link_jacs_exp = link_jacs[:, :, None, :, :]  # (L, S, 1, 3, num_actuated)

    # ∂distance/∂q = direction · ∂pos_robot/∂q
    # (robot moves toward world sphere => distance decreases)
    d_dist_d_q = jnp.einsum("lswd,lswda->lswa", directions, link_jacs_exp)

    # Gradient of max(0, margin - d) with respect to distance
    # When d < margin, residual = margin - d, so gradient = -1
    # When d >= margin, residual = 0, so gradient = 0
    d_res_d_dist = jnp.where(distances < cached_margin, -1.0, 0.0)  # (num_links, S, num_world)

    # Apply validity mask and compute final gradient
    active_mask = valid_mask[:, :, None]
    d_res_d_q = jnp.where(active_mask[..., None], d_res_d_dist[..., None] * d_dist_d_q, 0.0)

    return (d_res_d_q * weight).reshape(-1, num_actuated)


def _sphere_world_collision_cost_impl(
    vals: jaxls.VarValues,
    robot: "Robot",
    robot_coll: "RobotSphereCollision",
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
    local_centers = robot_coll.spheres.pose.translation()
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

    robot_radii = robot_coll.spheres.radius
    valid_mask = robot_coll._get_sphere_valid_mask()

    # Compute distances: (num_links, S, num_world)
    robot_pos_exp = robot_positions[:, :, None, :]  # (L, S, 1, 3)
    world_pos_exp = world_positions[None, None, :, :]  # (1, 1, W, 3)
    robot_rad_exp = robot_radii[:, :, None]  # (L, S, 1)
    world_rad_exp = world_radii[None, None, :]  # (1, 1, W)

    diff = robot_pos_exp - world_pos_exp  # (L, S, W, 3)
    center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)  # (L, S, W)
    directions = diff / (center_dist[..., None] + 1e-8)  # (L, S, W, 3)
    distances = center_dist - robot_rad_exp - world_rad_exp  # (L, S, W)

    # Compute residuals using max(0, margin - d) to match autodiff sphere_world_collision_residual
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


# Create cost and constraint versions using the factory pattern
_sphere_world_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_world_collision_jac
)(_sphere_world_collision_cost_impl)

_sphere_world_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_sphere_world_collision_jac,
    kind="constraint_leq_zero",
)(_sphere_world_collision_cost_impl)


def sphere_world_collision_constraint_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotSphereCollision",
    joint_var: jaxls.Var[jax.Array],
    world_spheres: "Sphere",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create sphere world-collision constraint with analytic Jacobian.

    Uses max(0, margin - distance) residual for all valid sphere pairs.
    Constraint version uses augmented Lagrangian for enforcement.

    This implementation precomputes the kinematic tree relationships once
    for efficiency, avoiding repeated while_loop calls during optimization.
    """
    # Precompute which joints affect each link (done once, outside cost function)
    joints_applied_to_links = _get_joints_applied_to_all_links(robot)

    return _sphere_world_collision_constraint(
        robot, robot_coll, joint_var, world_spheres, margin, weight,
        joints_applied_to_links
    )
