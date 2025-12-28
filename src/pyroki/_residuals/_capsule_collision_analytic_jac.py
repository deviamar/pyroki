"""Capsule collision cost with analytic Jacobian computation.

Provides capsule self-collision and world-collision costs with analytical Jacobians
for faster optimization compared to autodiff.

Key features:
- Uses straight-through estimator with soft clamping for stable gradients
- Precomputes kinematic tree relationships once outside the cost function
- Computes both position and orientation Jacobians for capsule endpoints
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
from ..collision._utils import closest_segment_to_segment_with_jac

if TYPE_CHECKING:
    from .._robot import Robot
    from ..collision import RobotCollision, Capsule


# Cache for self-collision (with sparse link indices)
_CapsuleSelfCollisionJacCache = tuple[
    jax.Array,  # Ts_world_joint: (num_joints, 7)
    jax.Array,  # capsule_centers: (num_links, 3)
    jax.Array,  # capsule_endpoints_a: (num_links, 3)
    jax.Array,  # capsule_endpoints_b: (num_links, 3)
    jax.Array,  # directions: (num_pairs, 3) - normalized (c2 - c1)
    jax.Array,  # distances: (num_pairs,) - signed distances
    jax.Array,  # s_params: (num_pairs,) - parametric position on segment 1
    jax.Array,  # t_params: (num_pairs,) - parametric position on segment 2
    jax.Array,  # d_dist_d_a1: (num_pairs, 3)
    jax.Array,  # d_dist_d_b1: (num_pairs, 3)
    jax.Array,  # d_dist_d_a2: (num_pairs, 3)
    jax.Array,  # d_dist_d_b2: (num_pairs, 3)
    jax.Array,  # joints_applied_to_links: (num_unique_links, num_joints) - sparse
    jax.Array,  # unique_links: (num_unique_links,) - which links we computed Jacobians for
    jax.Array,  # link_to_sparse_idx: (num_links,) - maps original index to sparse index
    float,  # margin: collision margin
]

# Cache for world-collision
_CapsuleWorldCollisionJacCache = tuple[
    jax.Array,  # Ts_world_joint: (num_joints, 7)
    jax.Array,  # robot_endpoints_a: (num_links, 3)
    jax.Array,  # robot_endpoints_b: (num_links, 3)
    jax.Array,  # directions: (num_links, num_world, 3)
    jax.Array,  # distances: (num_links, num_world) - signed distances
    jax.Array,  # s_params: (num_links, num_world)
    jax.Array,  # t_params: (num_links, num_world)
    jax.Array,  # d_dist_d_a1: (num_links, num_world, 3)
    jax.Array,  # d_dist_d_b1: (num_links, num_world, 3)
    jax.Array,  # joints_applied_to_links: (num_links, num_joints)
    float,  # margin: collision margin
]


def _compute_all_link_endpoint_jacobians(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    capsule_endpoints_a: jax.Array,
    capsule_endpoints_b: jax.Array,
    joints_applied_to_links: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute position Jacobians for capsule endpoints on all links.

    Batches both endpoints in a single call for better efficiency.

    Args:
        robot: Robot model.
        Ts_world_joint: Joint poses, shape (num_joints, 7).
        capsule_endpoints_a: Endpoint A positions in world frame, (num_links, 3).
        capsule_endpoints_b: Endpoint B positions in world frame, (num_links, 3).
        joints_applied_to_links: Precomputed (num_links, num_joints) matrix.

    Returns:
        Tuple of (jac_a, jac_b), each shape (num_links, 3, num_actuated).
    """
    # Stack both endpoints: (num_links, 2, 3) - batched computation
    endpoints_stacked = jnp.stack([capsule_endpoints_a, capsule_endpoints_b], axis=1)

    # Compute Jacobians for both endpoints in a single call
    jac_both = compute_all_link_position_jacobians(
        robot, Ts_world_joint, endpoints_stacked, joints_applied_to_links
    )  # (num_links, 2, 3, num_actuated)

    return jac_both[:, 0, :, :], jac_both[:, 1, :, :]


def _compute_link_endpoint_jacobians_sparse(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    capsule_endpoints_a: jax.Array,
    capsule_endpoints_b: jax.Array,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute position Jacobians for capsule endpoints on specified links (sparse).

    Batches both endpoints in a single call for better efficiency.

    Args:
        robot: Robot model.
        Ts_world_joint: Joint poses, shape (num_joints, 7).
        capsule_endpoints_a: Endpoint A positions in world frame, (num_unique_links, 3).
        capsule_endpoints_b: Endpoint B positions in world frame, (num_unique_links, 3).
        joints_applied_to_links: Precomputed (num_unique_links, num_joints) matrix
            for only the unique links.
        unique_links: Array of link indices to compute Jacobians for.

    Returns:
        Tuple of (jac_a, jac_b), each shape (num_unique_links, 3, num_actuated).
    """
    # Stack both endpoints: (num_unique_links, 2, 3) - batched computation
    endpoints_stacked = jnp.stack([capsule_endpoints_a, capsule_endpoints_b], axis=1)

    # Compute Jacobians for both endpoints in a single call
    jac_both = compute_link_position_jacobians_sparse(
        robot, Ts_world_joint, endpoints_stacked, joints_applied_to_links, unique_links
    )  # (num_unique_links, 2, 3, num_actuated)

    return jac_both[:, 0, :, :], jac_both[:, 1, :, :]


# =============================================================================
# Self-Collision
# =============================================================================


def capsule_self_collision_cost_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create capsule self-collision cost with analytic Jacobian.

    Uses max(0, margin - distance) residual for all active collision pairs.
    Uses sparse Jacobian computation - only computes for unique links in active pairs.

    Args:
        robot: Robot kinematic model.
        robot_coll: Robot collision model with capsules (max_geoms_per_link == 1).
        joint_var: Variable for joint configuration.
        margin: Safety margin for collision detection.
        weight: Weight for the cost.

    Returns:
        jaxls.Cost with analytic Jacobian.

    Raises:
        AssertionError: If robot_coll uses sphere decomposition (max_geoms_per_link > 1).
    """
    assert robot_coll.max_geoms_per_link == 1, (
        f"capsule_self_collision_cost_analytic_jac requires capsule-based RobotCollision "
        f"(max_geoms_per_link == 1), got {robot_coll.max_geoms_per_link}. "
        f"Use RobotCollision.from_urdf() without sphere_decomposition, "
        f"or use sphere_self_collision_cost_analytic_jac for sphere-based collision."
    )

    joints_applied, unique_links, link_to_sparse = prepare_sparse_link_indices(
        robot, robot_coll.geom_pair_link_i, robot_coll.geom_pair_link_j
    )

    return _capsule_self_collision_cost(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied,
        unique_links,
        link_to_sparse,
    )


def capsule_self_collision_constraint_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create capsule self-collision constraint with analytic Jacobian.

    Uses max(0, margin - distance) residual for all active collision pairs.
    Constraint version uses augmented Lagrangian for enforcement.
    Uses sparse Jacobian computation - only computes for unique links in active pairs.

    Args:
        robot: Robot kinematic model.
        robot_coll: Robot collision model with capsules (max_geoms_per_link == 1).
        joint_var: Variable for joint configuration.
        margin: Safety margin for collision detection.
        weight: Weight for the constraint.

    Returns:
        jaxls.Cost (constraint) with analytic Jacobian.

    Raises:
        AssertionError: If robot_coll uses sphere decomposition (max_geoms_per_link > 1).
    """
    assert robot_coll.max_geoms_per_link == 1, (
        f"capsule_self_collision_constraint_analytic_jac requires capsule-based RobotCollision "
        f"(max_geoms_per_link == 1), got {robot_coll.max_geoms_per_link}. "
        f"Use RobotCollision.from_urdf() without sphere_decomposition, "
        f"or use sphere_self_collision_constraint_analytic_jac for sphere-based collision."
    )

    joints_applied, unique_links, link_to_sparse = prepare_sparse_link_indices(
        robot, robot_coll.geom_pair_link_i, robot_coll.geom_pair_link_j
    )

    return _capsule_self_collision_constraint(
        robot,
        robot_coll,
        joint_var,
        margin,
        weight,
        joints_applied,
        unique_links,
        link_to_sparse,
    )


def _capsule_self_collision_jac(
    vals: jaxls.VarValues,
    jac_cache: _CapsuleSelfCollisionJacCache,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
    link_to_sparse_idx: jax.Array,
) -> jax.Array:
    """Analytic Jacobian for capsule self-collision cost.

    Uses chain rule:
        ∂residual/∂q = ∂residual/∂dist × (∂dist/∂a1 × ∂a1/∂q + ∂dist/∂b1 × ∂b1/∂q
                                          + ∂dist/∂a2 × ∂a2/∂q + ∂dist/∂b2 × ∂b2/∂q)

    The gradient of max(0, margin - d) with respect to d is:
        - For d >= margin: 0 (residual is 0)
        - For d < margin: -1 (residual is margin - d)

    Uses sparse Jacobian computation - only computes Jacobians for unique links
    involved in active collision pairs.
    """
    del vals, joint_var, margin

    (
        Ts_world_joint,
        capsule_centers,
        capsule_endpoints_a,
        capsule_endpoints_b,
        directions,
        distances,
        s_params,
        t_params,
        d_dist_d_a1,
        d_dist_d_b1,
        d_dist_d_a2,
        d_dist_d_b2,
        _,
        _,
        _,
        cached_margin,
    ) = jac_cache

    num_pairs = len(robot_coll.geom_pair_link_i)
    num_actuated = robot.joints.num_actuated_joints
    active_i = jnp.array(robot_coll.geom_pair_link_i)
    active_j = jnp.array(robot_coll.geom_pair_link_j)

    # Compute endpoint Jacobians only for unique links (sparse)
    # Slice endpoints to only include unique links (matching sphere collision pattern)
    unique_endpoints_a = capsule_endpoints_a[unique_links]
    unique_endpoints_b = capsule_endpoints_b[unique_links]
    sparse_jac_a, sparse_jac_b = _compute_link_endpoint_jacobians_sparse(
        robot,
        Ts_world_joint,
        unique_endpoints_a,
        unique_endpoints_b,
        joints_applied_to_links,
        unique_links,
    )  # Each: (num_unique_links, 3, num_actuated)

    # Map active pair indices to sparse indices and get Jacobians
    sparse_i = link_to_sparse_idx[active_i]  # (num_pairs,)
    sparse_j = link_to_sparse_idx[active_j]  # (num_pairs,)

    jac_a1 = sparse_jac_a[sparse_i]  # (num_pairs, 3, num_actuated)
    jac_b1 = sparse_jac_b[sparse_i]  # (num_pairs, 3, num_actuated)
    jac_a2 = sparse_jac_a[sparse_j]  # (num_pairs, 3, num_actuated)
    jac_b2 = sparse_jac_b[sparse_j]  # (num_pairs, 3, num_actuated)

    # Chain rule: ∂dist/∂q = ∂dist/∂a1 · ∂a1/∂q + ∂dist/∂b1 · ∂b1/∂q + ...
    # d_dist_d_a1: (num_pairs, 3), jac_a1: (num_pairs, 3, num_actuated)
    # einsum: "pd,pda->pa" contracts over the 3D direction
    d_dist_d_q = (
        jnp.einsum("pd,pda->pa", d_dist_d_a1, jac_a1)
        + jnp.einsum("pd,pda->pa", d_dist_d_b1, jac_b1)
        + jnp.einsum("pd,pda->pa", d_dist_d_a2, jac_a2)
        + jnp.einsum("pd,pda->pa", d_dist_d_b2, jac_b2)
    )  # (num_pairs, num_actuated)

    # Gradient of max(0, margin - d) with respect to distance
    # When d < margin, residual = margin - d, so gradient = -1
    # When d >= margin, residual = 0, so gradient = 0
    d_res_d_dist = jnp.where(distances < cached_margin, -1.0, 0.0)  # (num_pairs,)

    # ∂residual/∂q = ∂residual/∂dist × ∂dist/∂q
    d_res_d_q = d_res_d_dist[:, None] * d_dist_d_q

    return (d_res_d_q * weight).reshape(-1, num_actuated)


def _capsule_self_collision_cost_impl(
    vals: jaxls.VarValues,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
    link_to_sparse_idx: jax.Array,
) -> tuple[jax.Array, _CapsuleSelfCollisionJacCache]:
    """Compute residual and cache for capsule self-collision."""
    cfg = vals[joint_var]

    # Forward kinematics
    Ts_world_joint = robot._forward_kinematics_joints(cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    # Get capsule properties in world frame
    # robot_coll.coll has shape (num_links, 1), squeeze to (num_links,)
    # Note: max_geoms_per_link == 1 is validated in the public API functions
    capsule_local = jax.tree.map(lambda x: x[:, 0, ...], robot_coll.coll)
    Ts_link = jaxlie.SE3(Ts_world_link)

    # Transform capsule to world frame
    capsule_world = capsule_local.transform(Ts_link)

    # Get capsule endpoints in world frame
    capsule_centers = capsule_world.pose.translation()  # (num_links, 3)
    capsule_axes = capsule_world.axis  # (num_links, 3)
    capsule_heights = capsule_world.height  # (num_links,)
    capsule_radii = capsule_world.radius  # (num_links,)

    half_heights = capsule_heights / 2.0
    capsule_endpoints_a = capsule_centers - capsule_axes * half_heights[:, None]
    capsule_endpoints_b = capsule_centers + capsule_axes * half_heights[:, None]

    # Get active pairs
    active_i = jnp.array(robot_coll.geom_pair_link_i)
    active_j = jnp.array(robot_coll.geom_pair_link_j)
    num_pairs = len(robot_coll.geom_pair_link_i)

    # Get endpoints for each pair
    a1 = capsule_endpoints_a[active_i]  # (num_pairs, 3)
    b1 = capsule_endpoints_b[active_i]  # (num_pairs, 3)
    a2 = capsule_endpoints_a[active_j]  # (num_pairs, 3)
    b2 = capsule_endpoints_b[active_j]  # (num_pairs, 3)

    radius_i = capsule_radii[active_i]  # (num_pairs,)
    radius_j = capsule_radii[active_j]  # (num_pairs,)

    # Compute closest points and gradients using vectorized function
    (
        c1,
        c2,
        s_params,
        t_params,
        center_dist,
        directions,
        d_dist_d_a1,
        d_dist_d_b1,
        d_dist_d_a2,
        d_dist_d_b2,
    ) = closest_segment_to_segment_with_jac(a1, b1, a2, b2)

    # Signed distance (subtract radii)
    distances = center_dist - radius_i - radius_j

    # Compute residuals using max(0, margin - d) to match autodiff collision residual
    residuals = jnp.maximum(0.0, margin - distances)

    cache: _CapsuleSelfCollisionJacCache = (
        Ts_world_joint,
        capsule_centers,
        capsule_endpoints_a,
        capsule_endpoints_b,
        directions,
        distances,
        s_params,
        t_params,
        d_dist_d_a1,
        d_dist_d_b1,
        d_dist_d_a2,
        d_dist_d_b2,
        joints_applied_to_links,
        unique_links,
        link_to_sparse_idx,
        margin,
    )

    return (residuals * weight).flatten(), cache


_capsule_self_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_capsule_self_collision_jac
)(_capsule_self_collision_cost_impl)

_capsule_self_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_capsule_self_collision_jac,
    kind="constraint_leq_zero",
)(_capsule_self_collision_cost_impl)


# =============================================================================
# World-Collision
# =============================================================================


def capsule_world_collision_cost_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_capsules: "Capsule",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create capsule world-collision cost with analytic Jacobian.

    Uses max(0, margin - distance) residual for all link-world capsule pairs.

    Args:
        robot: Robot kinematic model.
        robot_coll: Robot collision model with capsules (max_geoms_per_link == 1).
        joint_var: Variable for joint configuration.
        world_capsules: Static world capsule obstacles.
        margin: Safety margin for collision detection.
        weight: Weight for the cost.

    Returns:
        jaxls.Cost with analytic Jacobian.

    Raises:
        AssertionError: If robot_coll uses sphere decomposition (max_geoms_per_link > 1).
    """
    assert robot_coll.max_geoms_per_link == 1, (
        f"capsule_world_collision_cost_analytic_jac requires capsule-based RobotCollision "
        f"(max_geoms_per_link == 1), got {robot_coll.max_geoms_per_link}. "
        f"Use RobotCollision.from_urdf() without sphere_decomposition, "
        f"or use sphere_world_collision_cost_analytic_jac for sphere-based collision."
    )

    joints_applied_to_links = get_joints_applied_to_all_links(robot)

    return _capsule_world_collision_cost(
        robot,
        robot_coll,
        joint_var,
        world_capsules,
        margin,
        weight,
        joints_applied_to_links,
    )


def capsule_world_collision_constraint_analytic_jac(
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_capsules: "Capsule",
    margin: float,
    weight: jax.Array | float = 1.0,
) -> jaxls.Cost:
    """Create capsule world-collision constraint with analytic Jacobian.

    Uses max(0, margin - distance) residual for all link-world capsule pairs.
    Constraint version uses augmented Lagrangian for enforcement.

    Args:
        robot: Robot kinematic model.
        robot_coll: Robot collision model with capsules (max_geoms_per_link == 1).
        joint_var: Variable for joint configuration.
        world_capsules: Static world capsule obstacles.
        margin: Safety margin for collision detection.
        weight: Weight for the constraint.

    Returns:
        jaxls.Cost (constraint) with analytic Jacobian.

    Raises:
        AssertionError: If robot_coll uses sphere decomposition (max_geoms_per_link > 1).
    """
    assert robot_coll.max_geoms_per_link == 1, (
        f"capsule_world_collision_constraint_analytic_jac requires capsule-based RobotCollision "
        f"(max_geoms_per_link == 1), got {robot_coll.max_geoms_per_link}. "
        f"Use RobotCollision.from_urdf() without sphere_decomposition, "
        f"or use sphere_world_collision_constraint_analytic_jac for sphere-based collision."
    )

    joints_applied_to_links = get_joints_applied_to_all_links(robot)

    return _capsule_world_collision_constraint(
        robot,
        robot_coll,
        joint_var,
        world_capsules,
        margin,
        weight,
        joints_applied_to_links,
    )


def _capsule_world_collision_jac(
    vals: jaxls.VarValues,
    jac_cache: _CapsuleWorldCollisionJacCache,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_capsules: "Capsule",
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
) -> jax.Array:
    """Analytic Jacobian for capsule world-collision cost.

    World capsules are static, so only robot endpoint gradients contribute.

    The gradient of max(0, margin - d) with respect to d is:
        - For d >= margin: 0 (residual is 0)
        - For d < margin: -1 (residual is margin - d)
    """
    del vals, joint_var, margin, world_capsules

    (
        Ts_world_joint,
        robot_endpoints_a,
        robot_endpoints_b,
        directions,
        distances,
        s_params,
        t_params,
        d_dist_d_a1,
        d_dist_d_b1,
        _,
        cached_margin,
    ) = jac_cache

    num_links = robot_coll.num_links
    num_world = directions.shape[1]
    num_actuated = robot.joints.num_actuated_joints

    # Compute endpoint Jacobians for all links
    jac_a, jac_b = _compute_all_link_endpoint_jacobians(
        robot,
        Ts_world_joint,
        robot_endpoints_a,
        robot_endpoints_b,
        joints_applied_to_links,
    )  # Each: (num_links, 3, num_actuated)

    # Expand for world dimension
    jac_a_exp = jac_a[:, None, :, :]  # (num_links, 1, 3, num_actuated)
    jac_b_exp = jac_b[:, None, :, :]  # (num_links, 1, 3, num_actuated)

    # Chain rule: ∂dist/∂q = ∂dist/∂a1 · ∂a1/∂q + ∂dist/∂b1 · ∂b1/∂q
    # d_dist_d_a1: (num_links, num_world, 3)
    # jac_a_exp: (num_links, 1, 3, num_actuated)
    d_dist_d_q = jnp.einsum("lwd,lwda->lwa", d_dist_d_a1, jac_a_exp) + jnp.einsum(
        "lwd,lwda->lwa", d_dist_d_b1, jac_b_exp
    )  # (num_links, num_world, num_actuated)

    # Gradient of max(0, margin - d) with respect to distance
    # When d < margin, residual = margin - d, so gradient = -1
    # When d >= margin, residual = 0, so gradient = 0
    d_res_d_dist = jnp.where(
        distances < cached_margin, -1.0, 0.0
    )  # (num_links, num_world)

    # ∂residual/∂q = ∂residual/∂dist × ∂dist/∂q
    d_res_d_q = d_res_d_dist[..., None] * d_dist_d_q

    return (d_res_d_q * weight).reshape(-1, num_actuated)


def _capsule_world_collision_cost_impl(
    vals: jaxls.VarValues,
    robot: "Robot",
    robot_coll: "RobotCollision",
    joint_var: jaxls.Var[jax.Array],
    world_capsules: "Capsule",
    margin: float,
    weight: jax.Array | float,
    joints_applied_to_links: jax.Array,
) -> tuple[jax.Array, _CapsuleWorldCollisionJacCache]:
    """Compute residual and cache for capsule world-collision."""
    cfg = vals[joint_var]

    # Forward kinematics
    Ts_world_joint = robot._forward_kinematics_joints(cfg)
    Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

    # Get robot capsule properties in world frame
    # robot_coll.coll has shape (num_links, 1), squeeze to (num_links,)
    # Note: max_geoms_per_link == 1 is validated in the public API functions
    capsule_local = jax.tree.map(lambda x: x[:, 0, ...], robot_coll.coll)
    Ts_link = jaxlie.SE3(Ts_world_link)
    capsule_world = capsule_local.transform(Ts_link)

    robot_centers = capsule_world.pose.translation()  # (num_links, 3)
    robot_axes = capsule_world.axis  # (num_links, 3)
    robot_heights = capsule_world.height  # (num_links,)
    robot_radii = capsule_world.radius  # (num_links,)

    robot_half_heights = robot_heights / 2.0
    robot_endpoints_a = robot_centers - robot_axes * robot_half_heights[:, None]
    robot_endpoints_b = robot_centers + robot_axes * robot_half_heights[:, None]

    # Get world capsule properties (static)
    world_axes = world_capsules.get_batch_axes()
    if len(world_axes) == 0:
        _world_capsules = world_capsules.broadcast_to((1,))
        num_world = 1
    else:
        _world_capsules = world_capsules
        num_world = world_axes[-1]

    world_centers = _world_capsules.pose.translation()  # (num_world, 3)
    world_axes_dir = _world_capsules.axis  # (num_world, 3)
    world_heights = _world_capsules.height  # (num_world,)
    world_radii = _world_capsules.radius  # (num_world,)

    world_half_heights = world_heights / 2.0
    world_endpoints_a = world_centers - world_axes_dir * world_half_heights[:, None]
    world_endpoints_b = world_centers + world_axes_dir * world_half_heights[:, None]

    num_links = robot_coll.num_links

    # Expand dimensions for broadcasting: (num_links, num_world, 3)
    robot_a_exp = robot_endpoints_a[:, None, :]  # (L, 1, 3)
    robot_b_exp = robot_endpoints_b[:, None, :]  # (L, 1, 3)
    world_a_exp = world_endpoints_a[None, :, :]  # (1, W, 3)
    world_b_exp = world_endpoints_b[None, :, :]  # (1, W, 3)

    robot_rad_exp = robot_radii[:, None]  # (L, 1)
    world_rad_exp = world_radii[None, :]  # (1, W)

    # Compute closest points for all (link, world) pairs
    (
        c1,
        c2,
        s_params,
        t_params,
        center_dist,
        directions,
        d_dist_d_a1,
        d_dist_d_b1,
        d_dist_d_a2,
        d_dist_d_b2,
    ) = closest_segment_to_segment_with_jac(
        robot_a_exp, robot_b_exp, world_a_exp, world_b_exp
    )
    # Outputs have shape (num_links, num_world, ...) due to broadcasting

    # Signed distance (subtract radii)
    distances = center_dist - robot_rad_exp - world_rad_exp  # (L, W)

    # Compute residuals using max(0, margin - d) to match autodiff collision residual
    residuals = jnp.maximum(0.0, margin - distances)

    cache: _CapsuleWorldCollisionJacCache = (
        Ts_world_joint,
        robot_endpoints_a,
        robot_endpoints_b,
        directions,
        distances,
        s_params,
        t_params,
        d_dist_d_a1,
        d_dist_d_b1,
        joints_applied_to_links,
        margin,
    )

    return (residuals * weight).flatten(), cache


_capsule_world_collision_cost = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_capsule_world_collision_jac
)(_capsule_world_collision_cost_impl)

_capsule_world_collision_constraint = jaxls.Cost.factory(
    jac_custom_with_cache_fn=_capsule_world_collision_jac,
    kind="constraint_leq_zero",
)(_capsule_world_collision_cost_impl)
