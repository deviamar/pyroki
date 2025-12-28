"""Shared helper functions for analytic Jacobian computation.

Provides common utilities for computing kinematic Jacobians used by
both sphere and capsule collision costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jaxlie

if TYPE_CHECKING:
    from .._robot import Robot


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


def get_joints_applied_to_all_links(robot: "Robot") -> jax.Array:
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


def compute_all_link_position_jacobians(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    point_positions: jax.Array,
    joints_applied_to_links: jax.Array,
) -> jax.Array:
    """Compute position Jacobians for points on all links.

    Uses dense matmul instead of scatter-add for better XLA fusion.

    Args:
        robot: Robot model.
        Ts_world_joint: Joint poses, shape (num_joints, 7).
        point_positions: Point positions in world frame, (num_links, S, 3).
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
        link_point_pos = point_positions[link_idx]
        joints_applied = joints_applied_to_links[link_idx]
        joint_to_act = _create_joint_to_actuated_matrix(joints_applied, num_actuated)

        r = link_point_pos[:, None, :] - joint_positions[None, :, :]
        vel_per_joint = jnp.cross(omega_world[None, :, :], r) + vel_world[None, :, :]
        jac = jnp.einsum("sjd,ja->sda", vel_per_joint, joint_to_act)
        return jac

    return jax.vmap(compute_link_jacobian)(jnp.arange(point_positions.shape[0]))


def compute_link_position_jacobians_sparse(
    robot: "Robot",
    Ts_world_joint: jax.Array,
    point_positions: jax.Array,
    joints_applied_to_links: jax.Array,
    unique_links: jax.Array,
) -> jax.Array:
    """Compute position Jacobians for points on specified unique links.

    Args:
        robot: Robot model.
        Ts_world_joint: Joint poses, shape (num_joints, 7).
        point_positions: Point positions in world frame, (num_unique_links, S, 3).
        joints_applied_to_links: Precomputed (num_unique_links, num_joints) matrix.
        unique_links: Array of link indices (unused in computation but documents intent).

    Returns:
        Jacobian of shape (num_unique_links, S, 3, num_actuated).
    """
    del unique_links  # Unused in computation, implicitly defined by inputs
    num_actuated = robot.joints.num_actuated_joints

    Ts_world_joint_se3 = jaxlie.SE3(Ts_world_joint)
    joint_twists = robot.joints.twists * robot.joints.mimic_multiplier[..., None]
    omega_world = Ts_world_joint_se3.rotation() @ joint_twists[:, 3:]
    vel_world = Ts_world_joint_se3.rotation() @ joint_twists[:, :3]
    joint_positions = Ts_world_joint_se3.translation()

    def compute_link_jacobian(idx: jax.Array) -> jax.Array:
        link_point_pos = point_positions[idx]
        joints_applied = joints_applied_to_links[idx]
        joint_to_act = _create_joint_to_actuated_matrix(joints_applied, num_actuated)

        r = link_point_pos[:, None, :] - joint_positions[None, :, :]
        vel_per_joint = jnp.cross(omega_world[None, :, :], r) + vel_world[None, :, :]
        jac = jnp.einsum("sjd,ja->sda", vel_per_joint, joint_to_act)
        return jac

    return jax.vmap(compute_link_jacobian)(jnp.arange(point_positions.shape[0]))


def prepare_sparse_link_indices(
    robot: "Robot",
    geom_pair_link_i: list[int],
    geom_pair_link_j: list[int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Prepare sparse link indexing for self-collision.

    Computes which joints affect the unique links involved in collision pairs,
    and creates a mapping from original link indices to sparse indices.

    Args:
        robot: Robot model.
        geom_pair_link_i: List of link indices for first geometry in each pair.
        geom_pair_link_j: List of link indices for second geometry in each pair.

    Returns:
        Tuple of:
        - joints_applied_to_links: (num_unique_links, num_joints) array
        - unique_links: (num_unique_links,) array of link indices
        - link_to_sparse_idx: (num_links,) array mapping original to sparse indices
    """
    all_joints_applied = get_joints_applied_to_all_links(robot)

    unique_links_set = sorted(set(geom_pair_link_i) | set(geom_pair_link_j))
    unique_links = jnp.array(unique_links_set, dtype=jnp.int32)

    joints_applied_to_links = all_joints_applied[unique_links]

    link_to_sparse_idx = jnp.full(robot.links.num_links, -1, dtype=jnp.int32)
    link_to_sparse_idx = link_to_sparse_idx.at[unique_links].set(
        jnp.arange(len(unique_links), dtype=jnp.int32)
    )

    return joints_applied_to_links, unique_links, link_to_sparse_idx
