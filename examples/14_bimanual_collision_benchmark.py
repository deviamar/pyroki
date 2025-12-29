"""Bimanual IK Collision Avoidance Benchmark

Compares autodiff vs analytical Jacobian performance for bimanual IK with sphere
collision avoidance. Uses YuMi dual-arm robot with ballpark library for automatic
sphere decomposition.

Usage:
    # Default (64 spheres)
    python 14_bimanual_collision_benchmark.py

    # Custom sphere count
    python 14_bimanual_collision_benchmark.py --total-spheres 128

    # Different ballpark preset
    python 14_bimanual_collision_benchmark.py --preset conservative

Requires: pip install 'pyroki[ballpark]'
"""

import argparse
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def create_sphere_decomposition_from_ballpark(
    urdf,
    total_spheres: int = 64,
    preset: str = "balanced",
) -> dict[str, list[dict]]:
    """Create sphere decomposition using ballpark library.

    Args:
        urdf: The URDF object (with collision meshes loaded).
        total_spheres: Total number of spheres to distribute across links.
        preset: Ballpark preset ("balanced", "conservative", "surface").

    Returns:
        Dictionary mapping link names to lists of sphere definitions.
        Each sphere is a dict with 'center' (list of 3 floats) and 'radius' (float).
    """
    try:
        import ballpark
    except ImportError:
        raise ImportError(
            "ballpark is required for this benchmark. "
            "Install with: pip install 'pyroki[ballpark]'"
        )

    # Use ballpark to compute spheres for the entire robot
    result = ballpark.compute_spheres_for_robot(
        urdf,
        target_spheres=total_spheres,
        preset=preset,
    )

    # Convert ballpark result to the format expected by RobotCollision
    decomposition = {}
    for link_name, spheres in result.link_spheres.items():
        link_spheres = []
        for sphere in spheres:
            link_spheres.append(
                {
                    "center": sphere.center.tolist(),
                    "radius": float(sphere.radius),
                }
            )
        if link_spheres:
            decomposition[link_name] = link_spheres

    return decomposition


@jdc.jit
def solve_bimanual_ik_analytic(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_spheres: Sphere,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target_left: jaxlie.SE3,
    T_world_target_right: jaxlie.SE3,
    target_link_index_left: jax.Array,
    target_link_index_right: jax.Array,
) -> jax.Array:
    """Solves bimanual IK with sphere collision using analytical Jacobians."""
    JointVar = robot.joint_var_cls
    joint_var = JointVar(0)
    variables = [joint_var]
    batch_axes = (2,)

    costs = [
        # Left arm pose
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            # stack left and right target poses along batch axis
            jax.tree.map(
                lambda x, y: jnp.stack([x, y], axis=0),
                T_world_target_left,
                T_world_target_right,
            ),
            jnp.stack([target_link_index_left, target_link_index_right], axis=0),
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.sphere_self_collision_constraint_analytic_jac(
            robot=robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            margin=0.0,
            # weight=5.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
        ),
        pk.costs.sphere_world_collision_constraint_analytic_jac(
            robot=robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            world_spheres=world_spheres,
            margin=0.05,
        ),
        *[
            pk.costs.world_collision_constraint(
                robot, robot_coll, joint_var, halfspace, margin=0.05
            )
            for halfspace in world_halfspaces
        ],
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            # augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]


@jdc.jit
def solve_bimanual_ik_autodiff(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_spheres: Sphere,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target_left: jaxlie.SE3,
    T_world_target_right: jaxlie.SE3,
    target_link_index_left: jax.Array,
    target_link_index_right: jax.Array,
) -> jax.Array:
    """Solves bimanual IK with sphere collision using autodiff Jacobians."""
    JointVar = robot.joint_var_cls
    joint_var = JointVar(0)
    variables = [joint_var]
    batch_axes = (2,)

    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            # stack left and right target poses along batch axis
            jax.tree.map(
                lambda x, y: jnp.stack([x, y], axis=0),
                T_world_target_left,
                T_world_target_right,
            ),
            jnp.stack([target_link_index_left, target_link_index_right], axis=0),
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.sphere_self_collision_constraint(
            robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            margin=0.0,
            # weight=5.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.sphere_world_collision_constraint(
            robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            world_spheres=world_spheres,
            margin=0.05,
        ),
        *[
            pk.costs.world_collision_constraint(
                robot, robot_coll, joint_var, halfspace, margin=0.05
            )
            for halfspace in world_halfspaces
        ],
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            # augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]


def main():
    parser = argparse.ArgumentParser(
        description="Bimanual IK Collision Avoidance Benchmark (Sphere)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--total-spheres",
        type=int,
        default=64,
        help="Total spheres for sphere decomposition (default: 64)",
    )
    parser.add_argument(
        "--preset",
        choices=["balanced", "conservative", "surface"],
        default="balanced",
        help="Ballpark preset for sphere decomposition (default: balanced)",
    )
    args = parser.parse_args()

    # Load YuMi robot (bimanual)
    urdf = load_robot_description("yumi_description")
    target_link_name_left = "yumi_link_7_l"
    target_link_name_right = "yumi_link_7_r"
    robot = pk.Robot.from_urdf(urdf)
    target_link_idx_left = robot.links.names.index(target_link_name_left)
    target_link_idx_right = robot.links.names.index(target_link_name_right)

    # Create sphere-based robot collision model
    print(f"Creating sphere-based collision model ({args.total_spheres} spheres)...")
    sphere_decomposition = create_sphere_decomposition_from_ballpark(
        urdf=urdf,
        total_spheres=args.total_spheres,
        preset=args.preset,
    )
    robot_coll = RobotCollision.from_urdf(
        urdf,
        sphere_decomposition=sphere_decomposition,
    )

    # Ground plane
    ground_plane = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controllers for IK targets (left and right arms)
    ik_target_left = server.scene.add_transform_controls(
        "/ik_target_left", scale=0.15, position=(0.41, 0.3, 0.56), wxyz=(0, 0, 1, 0)
    )
    ik_target_right = server.scene.add_transform_controls(
        "/ik_target_right", scale=0.15, position=(0.41, -0.3, 0.56), wxyz=(0, 0, 1, 0)
    )

    # Create sphere obstacles between the two arms
    world_obstacles = Sphere.from_center_and_radius(
        center=np.zeros((2, 3), dtype=np.float32),
        radius=np.array([0.08, 0.06]),
    )
    obstacle_handles = []
    for i in range(2):
        pos = world_obstacles.pose.translation()[i]
        handle = server.scene.add_transform_controls(
            f"/obstacle_{i}",
            scale=0.12,
            position=tuple(float(x) for x in pos),
        )
        obstacle_i = jax.tree.map(lambda x: x[i], world_obstacles)
        server.scene.add_mesh_trimesh(
            f"/obstacle_{i}/mesh",
            mesh=obstacle_i.to_trimesh(),
        )
        obstacle_handles.append(handle)

    # GUI controls
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    use_analytic_jac = server.gui.add_checkbox(
        "Use Analytic Jacobian", initial_value=True
    )

    print("\nBimanual collision benchmark started!")
    print("- Drag left target (near y+) to control left arm")
    print("- Drag right target (near y-) to control right arm")
    print("- Drag the obstacles to test collision avoidance")
    print("- Toggle 'Use Analytic Jacobian' to compare performance")

    while True:
        start_time = time.time()

        # Update world obstacles from interactive handles
        positions = []
        for handle in obstacle_handles:
            positions.append(np.array(handle.position))
        world_obstacles_current = Sphere.from_center_and_radius(
            center=np.array(positions),
            radius=world_obstacles.radius,
        )

        # Build target poses
        T_world_target_left = jaxlie.SE3(
            jnp.concatenate(
                [
                    jnp.array(ik_target_left.wxyz),
                    jnp.array(ik_target_left.position),
                ],
                axis=-1,
            )
        )
        T_world_target_right = jaxlie.SE3(
            jnp.concatenate(
                [
                    jnp.array(ik_target_right.wxyz),
                    jnp.array(ik_target_right.position),
                ],
                axis=-1,
            )
        )

        # Solve IK
        if use_analytic_jac.value:
            solution = solve_bimanual_ik_analytic(
                robot=robot,
                robot_coll=robot_coll,
                world_spheres=world_obstacles_current,
                world_halfspaces=[ground_plane],
                T_world_target_left=T_world_target_left,
                T_world_target_right=T_world_target_right,
                target_link_index_left=jnp.array(target_link_idx_left),
                target_link_index_right=jnp.array(target_link_idx_right),
            )
        else:
            solution = solve_bimanual_ik_autodiff(
                robot=robot,
                robot_coll=robot_coll,
                world_spheres=world_obstacles_current,
                world_halfspaces=[ground_plane],
                T_world_target_left=T_world_target_left,
                T_world_target_right=T_world_target_right,
                target_link_index_left=jnp.array(target_link_idx_left),
                target_link_index_right=jnp.array(target_link_idx_right),
            )

        # Update timing
        elapsed_ms = (time.time() - start_time) * 1000
        timing_handle.value = elapsed_ms

        # Update visualizer
        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
