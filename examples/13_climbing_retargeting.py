"""Climbing Retargeting (DiffIK)

Retarget motion to G1 humanoid using Laplacian mesh deformation with diffIK.
Based on the holosoma interaction mesh retargeting approach.
"""

import json
import time
from pathlib import Path
from typing import Tuple, TypedDict

import trimesh
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
import tqdm
from viser.extras import ViserUrdf

from retarget_helpers._climb_utils import (
    adjacency_list_to_matrix,
    calculate_laplacian_coordinates_vectorized,
    create_interaction_mesh,
    get_adjacency_list,
    get_climb_retarget_indices,
    load_climb_motion,
    load_object_mesh,
    load_object_points,
)


class ClimbingRetargetingWeights(TypedDict):
    laplacian: float
    """Laplacian mesh deformation weight (default: 10.0 from holosoma)."""
    smoothness: float
    """Smoothness weight towards previous frame (default: 0.2 from holosoma)."""
    rest: float
    """Rest pose regularization weight (very small, default: 0.01)."""


def main():
    """Main function for climbing retargeting."""

    # Load robot.
    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)

    sphere_json_path = Path(__file__).parent / "assets" / "g1_spheres.json"
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Load motion data and object
    asset_dir = Path(__file__).parent / "retarget_helpers" / "omniretarget_climb_data"
    motion = load_climb_motion(
        asset_dir / "mocap_climb_seq_0_joint_positions_f900-3700.npy"
    )
    object_points = load_object_points(asset_dir / "multi_boxes.obj", sample_count=30)
    object_mesh = load_object_mesh(asset_dir / "multi_boxes.obj")

    num_timesteps = motion.shape[0]
    print(f"Loaded motion with {num_timesteps} timesteps, {motion.shape[1]} joints")
    print(f"Object points shape: {object_points.shape}")

    # Get retarget indices and local offsets.
    mocap_indices, g1_link_indices, local_offsets = get_climb_retarget_indices(robot)
    print(f"Retargeting {len(mocap_indices)} joints")

    # Setup visualization
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", True, disabled=True)
    timestep_slider = server.gui.add_slider(
        "timestep", 0, num_timesteps - 1, 1, 0, disabled=True
    )

    # Add object mesh to scene
    server.scene.add_mesh_trimesh("/object", object_mesh)

    # Split up the object mesh into individual boxes.
    meshes = object_mesh.split()
    assert len(meshes) == 3
    box_1_transform, box_1_extents = trimesh.bounds.oriented_bounds(meshes[0])
    box_2_transform, box_2_extents = trimesh.bounds.oriented_bounds(meshes[1])
    box_3_transform, box_3_extents = trimesh.bounds.oriented_bounds(meshes[2])
    box_1_jaxlie_transform = jaxlie.SE3.from_matrix(box_1_transform).inverse()
    box_2_jaxlie_transform = jaxlie.SE3.from_matrix(box_2_transform).inverse()
    box_3_jaxlie_transform = jaxlie.SE3.from_matrix(box_3_transform).inverse()

    coll_box = pk.collision.Box.from_extent(
        extent=onp.array([box_1_extents, box_2_extents, box_3_extents]),
        position=onp.array(
            [
                box_1_jaxlie_transform.translation(),
                box_2_jaxlie_transform.translation(),
                box_3_jaxlie_transform.translation(),
            ]
        ),
        wxyz=onp.array(
            [
                box_1_jaxlie_transform.rotation().wxyz,
                box_2_jaxlie_transform.rotation().wxyz,
                box_3_jaxlie_transform.rotation().wxyz,
            ]
        ),
    )
    server.scene.add_mesh_trimesh("/box_coll_pk", coll_box.to_trimesh())

    # Add ground plane
    server.scene.add_grid("/grid", width=4, height=4, position=(0.0, 0.0, 0.0))

    weights = pk.viewer.WeightTuner(
        server,
        ClimbingRetargetingWeights(
            laplacian=10.0,
            smoothness=0.2,
            rest=0.01,
        ),  # type: ignore
    )
    rerun_button = server.gui.add_button("Retarget Trajectory")

    # Precompute interaction mesh (must be done outside JIT)
    n_robot_pts = len(mocap_indices)
    n_obj_pts = object_points.shape[0]
    n_total_pts = n_robot_pts + n_obj_pts

    # Build interaction mesh from first frame
    demo_robot_pts = motion[0, mocap_indices]
    all_vertices_np = onp.concatenate(
        [onp.array(demo_robot_pts), onp.array(object_points)], axis=0
    )
    _, tetrahedra = create_interaction_mesh(all_vertices_np)
    adj_list = get_adjacency_list(tetrahedra, n_total_pts)
    adj_matrix = adjacency_list_to_matrix(adj_list, n_total_pts)

    # Precompute target Laplacian coordinates for each frame
    def compute_target_lap(keypoints_frame):
        robot_pts = keypoints_frame[mocap_indices]
        all_pts = jnp.concatenate([robot_pts, object_points], axis=0)
        return calculate_laplacian_coordinates_vectorized(all_pts, adj_matrix)

    target_laplacians = jax.vmap(compute_target_lap)(motion)
    assert target_laplacians.shape == (num_timesteps, n_total_pts, 3)

    # Get the current robot status, step-by-step.
    list_T_world_root, list_joints = [], []

    def retarget_trajectory():
        nonlocal list_T_world_root, list_joints
        list_T_world_root, list_joints = [], []

        with server.atomic():
            # Temporarily disable weight tuning during solve.
            for handle in weights._weight_handles.values():
                handle.disabled = True
            rerun_button.disabled = True

            tstep = timestep_slider.value = 0

            init_T_world_root = jaxlie.SE3.identity()
            init_joints = jnp.array(robot.joint_var_cls.default_factory())

            for t in tqdm.trange(num_timesteps, desc="Solving frames"):
                T_world_root_t, joints_t = solve_single_frame(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_coll_list=[coll_box],
                    target_lap=target_laplacians[t],
                    object_points=object_points,
                    g1_link_indices=g1_link_indices,
                    local_offsets=local_offsets,
                    adj_matrix=adj_matrix,
                    weights=weights.get_weights(),  # type: ignore
                    init_joints=init_joints,
                    init_T_world_root=init_T_world_root,
                    prev_joints=init_joints,
                    prev_T_world_root=init_T_world_root,
                )
                jax.block_until_ready((T_world_root_t, joints_t))

                # Store results.
                list_T_world_root.append(T_world_root_t)
                list_joints.append(joints_t)
                init_joints = joints_t
                init_T_world_root = T_world_root_t

                # Update the visualization online.
                timestep_slider.value = t
                base_frame.wxyz = onp.array(T_world_root_t.wxyz_xyz[:4])
                base_frame.position = onp.array(T_world_root_t.wxyz_xyz[4:])
                urdf_vis.update_cfg(onp.array(joints_t))

            # All results are ready; we should be able to do playback.
            playing.disabled = False
            timestep_slider.disabled = False
            # We should also be able to tune weights now.
            for handle in weights._weight_handles.values():
                handle.disabled = False
            rerun_button.disabled = False

    retarget_trajectory()
    rerun_button.on_click(lambda _: retarget_trajectory())

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
            tstep = timestep_slider.value
            base_frame.wxyz = onp.array(list_T_world_root[tstep].wxyz_xyz[:4])
            base_frame.position = onp.array(list_T_world_root[tstep].wxyz_xyz[4:])
            urdf_vis.update_cfg(onp.array(list_joints[tstep]))

            # Show target keypoints
            keypoints_to_show = motion[tstep, onp.array(mocap_indices)]
            server.scene.add_point_cloud(
                "/target_keypoints",
                onp.array(keypoints_to_show),
                onp.array((0, 0, 255))[None].repeat(len(mocap_indices), axis=0),
                point_size=0.02,
            )

            # Show object sample points
            server.scene.add_point_cloud(
                "/object_points",
                onp.array(object_points),
                onp.array((255, 0, 0))[None].repeat(len(object_points), axis=0),
                point_size=0.015,
            )

        time.sleep(0.05)


@jdc.jit
def solve_single_frame(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll_list: list[pk.collision.CollGeom],
    target_lap: jnp.ndarray,
    object_points: jnp.ndarray,
    g1_link_indices: jnp.ndarray,
    local_offsets: jnp.ndarray,
    adj_matrix: jnp.ndarray,
    weights: ClimbingRetargetingWeights,
    init_joints: jnp.ndarray,
    init_T_world_root: jaxlie.SE3,
    prev_joints: jnp.ndarray,
    prev_T_world_root: jaxlie.SE3,
) -> tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve a single frame of the climbing retargeting problem.

    Args:
        robot: PyRoki robot model.
        target_lap: (n_total, 3) target Laplacian coordinates for this frame.
        object_points: (num_obj_points, 3) sampled object surface points.
        g1_link_indices: Corresponding G1 robot link indices.
        local_offsets: (N, 3) local-frame offsets for each keypoint.
        adj_matrix: (n_total, n_total) precomputed adjacency matrix.
        weights: Retargeting weights.
        init_joints: Initial joint configuration (from previous frame).
        init_T_world_root: Initial root transform (from previous frame).
        prev_joints: Previous frame's joint configuration (for smoothness).
        prev_T_world_root: Previous frame's root transform (for smoothness).

    Returns:
        Tuple of (T_world_root, joints) for this frame.
    """
    # Variables for single frame (use scalar index 0)
    var_joints = robot.joint_var_cls(0)
    var_T_world_root = jaxls.SE3Var(0)

    # Costs
    @jaxls.Cost.factory
    def laplacian_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        """Laplacian mesh deformation cost."""
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_T_world_root]
        T_world_link = T_world_root @ T_root_link

        # Get link transforms for keypoint links
        T_keypoint_links = jaxlie.SE3(T_world_link.wxyz_xyz[g1_link_indices])

        # Apply local offsets: rotate by link orientation and add to position
        # local_offsets: (N, 3), need to rotate each by corresponding link rotation
        rotated_offsets = jax.vmap(lambda T, off: T.rotation() @ off)(
            T_keypoint_links, local_offsets
        )
        robot_keypoints = T_keypoint_links.translation() + rotated_offsets

        # Combine with fixed object points
        vertices = jnp.concatenate([robot_keypoints, object_points], axis=0)

        # Compute current Laplacian
        current_lap = calculate_laplacian_coordinates_vectorized(vertices, adj_matrix)

        # Residual
        return (current_lap - target_lap).flatten() * weights["laplacian"]

    @jaxls.Cost.factory
    def smoothness_to_prev_joints(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        """Smoothness cost towards previous frame's joints."""
        robot_cfg = var_values[var_robot_cfg]
        return (robot_cfg - prev_joints) * weights["smoothness"]

    prev_T_world_joints = prev_T_world_root @ jaxlie.SE3(
        robot.forward_kinematics(prev_joints)
    )

    @jaxls.Cost.factory
    def smoothness_to_prev_pose(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_T_world_root: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost towards previous frame's pose."""
        T_world_root = var_values[var_T_world_root]
        T_root_robot = jaxlie.SE3(robot.forward_kinematics(var_values[var_robot_cfg]))
        T_world_robot = T_world_root @ T_root_robot
        return (prev_T_world_joints.inverse() @ T_world_robot).log() * weights[
            "smoothness"
        ]

    @jaxls.Cost.factory(kind="constraint_geq_zero")
    def world_collision_constraint(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        joint_var: jaxls.Var[jax.Array],
        var_T_world_root: jaxls.SE3Var,
        world_geom: pk.collision.CollGeom,
    ) -> jax.Array:
        """Computes world collision violation residual. Residual is >0 if collision is detected."""
        cfg = vals[joint_var]
        world_geom_in_root = world_geom.transform(vals[var_T_world_root].inverse())
        dist_matrix = robot_coll.compute_world_collision_distance(
            robot, cfg, world_geom_in_root
        )
        return dist_matrix.flatten()

    costs: list[jaxls.Cost] = [
        # Laplacian cost
        laplacian_cost(var_T_world_root, var_joints),
        # Smoothness to previous frame
        smoothness_to_prev_joints(var_joints),
        smoothness_to_prev_pose(var_joints, var_T_world_root),
        # Rest pose regularization (very small)
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory(),
            jnp.full(var_joints.default_factory().shape, weights["rest"]),
        ),
        # Joint limits as constraint
        pk.costs.limit_constraint(
            robot,
            var_joints,
        ),
        # Collision avoidance constraint
        pk.costs.self_collision_cost(
            robot, robot_coll, var_joints, margin=0.01, weight=10.0
        ),
    ]

    # Collision avoidance with world objects
    for world_coll in world_coll_list:
        costs.append(
            world_collision_constraint(
                robot, robot_coll, var_joints, var_T_world_root, world_coll
            ),
        )

    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[var_joints, var_T_world_root],
        )
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(init_joints),
                    var_T_world_root.with_value(init_T_world_root),
                ]
            ),
        )
    )

    return solution[var_T_world_root], solution[var_joints]


if __name__ == "__main__":
    main()
