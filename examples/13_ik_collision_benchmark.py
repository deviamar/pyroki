"""IK Collision Avoidance Benchmark

Compares autodiff vs analytical Jacobian performance for IK with collision avoidance.
Supports both capsule-based (from URDF) and sphere-based (from ballpark) collision models.

Usage:
    # Capsule collision (default, no extra dependencies)
    python 13_ik_collision_benchmark.py

    # Sphere collision (requires: pip install 'pyroki[ballpark]')
    python 13_ik_collision_benchmark.py --collision-type sphere

    # Sphere collision with custom sphere count
    python 13_ik_collision_benchmark.py --collision-type sphere --total-spheres 64
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
from pyroki.collision import Capsule, HalfSpace, RobotCollision, Sphere
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
            "ballpark is required for sphere collision. "
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
def solve_ik_capsule_analytic(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_capsules: Capsule,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves IK with capsule collision using analytical Jacobians."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        # pk.costs.capsule_self_collision_cost_analytic_jac(
        #     robot=robot,
        #     robot_coll=robot_coll,
        #     joint_var=joint_var,
        #     margin=0.02,
        #     weight=5.0,
        # ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.capsule_world_collision_constraint_analytic_jac(
            robot=robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            world_capsules=world_capsules,
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
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]


@jdc.jit
def solve_ik_capsule_autodiff(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_capsules: Capsule,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves IK with capsule collision using autodiff Jacobians."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        # pk.costs.self_collision_cost(
        #     robot,
        #     robot_coll=robot_coll,
        #     joint_var=joint_var,
        #     margin=0.02,
        #     weight=5.0,
        # ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.world_collision_constraint(
            robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            world_geom=world_capsules,
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
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]


@jdc.jit
def solve_ik_sphere_analytic(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_spheres: Sphere,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves IK with sphere collision using analytical Jacobians."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.sphere_self_collision_cost_analytic_jac(
            robot=robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.sphere_world_collision_constraint_analytic_jac(
            robot=robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            world_spheres=world_spheres,
            margin=0.05,
            weight=1.0,
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
            termination=jaxls.TerminationConfig(max_iterations=20),
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=1),
        )
    )
    return sol[joint_var]


@jdc.jit
def solve_ik_sphere_autodiff(
    robot: pk.Robot,
    robot_coll: RobotCollision,
    world_spheres: Sphere,
    world_halfspaces: Sequence[HalfSpace],
    T_world_target: jaxlie.SE3,
    target_link_index: jax.Array,
) -> jax.Array:
    """Solves IK with sphere collision using autodiff Jacobians."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_world_target,
            target_link_index=target_link_index,
            pos_weight=5.0,
            ori_weight=1.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.sphere_self_collision_cost(
            robot,
            robot_coll=robot_coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
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
            augmented_lagrangian=jaxls.AugmentedLagrangianConfig(max_iterations=5),
        )
    )
    return sol[joint_var]


def main():
    parser = argparse.ArgumentParser(
        description="IK Collision Avoidance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collision-type",
        choices=["capsule", "sphere"],
        default="capsule",
        help="Type of collision geometry (default: capsule)",
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

    # Load robot
    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"
    robot = pk.Robot.from_urdf(urdf)
    target_link_idx = robot.links.names.index(target_link_name)

    # Create robot collision model based on type
    if args.collision_type == "sphere":
        print(
            f"Creating sphere-based collision model ({args.total_spheres} spheres)..."
        )
        sphere_decomposition = create_sphere_decomposition_from_ballpark(
            urdf=urdf,
            total_spheres=args.total_spheres,
            preset=args.preset,
        )
        robot_coll = RobotCollision.from_urdf(
            urdf,
            sphere_decomposition=sphere_decomposition,
        )
    else:
        print("Creating capsule-based collision model...")
        robot_coll = RobotCollision.from_urdf(urdf)

    # Ground plane
    ground_plane = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
    )

    # Create world obstacles based on collision type
    if args.collision_type == "sphere":
        # Sphere obstacles
        world_obstacles = Sphere.from_center_and_radius(
            center=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            radius=np.array([0.08, 0.08]),
        )
        obstacle_handles = []
        for i in range(2):
            pos = world_obstacles.pose.translation()[i]
            handle = server.scene.add_transform_controls(
                f"/obstacle_{i}",
                scale=0.15,
                position=tuple(float(x) for x in pos),
            )
            obstacle_i = jax.tree.map(lambda x: x[i], world_obstacles)
            server.scene.add_mesh_trimesh(
                f"/obstacle_{i}/mesh",
                mesh=obstacle_i.to_trimesh(),
            )
            obstacle_handles.append(handle)
    else:
        # Capsule obstacles (pillars)
        world_obstacles = Capsule.from_radius_height(
            radius=np.array([0.05, 0.05]),
            height=np.array([0.4, 0.4]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        obstacle_handles = []
        for i in range(2):
            pos = world_obstacles.pose.translation()[i]
            handle = server.scene.add_transform_controls(
                f"/obstacle_{i}",
                scale=0.15,
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

    print(f"\nCollision benchmark ({args.collision_type}) started!")
    print("- Drag the target (blue axes) to move the end-effector")
    print("- Drag the obstacles to test collision avoidance")
    print("- Toggle 'Use Analytic Jacobian' to compare performance")

    while True:
        start_time = time.time()

        # Update world obstacles from interactive handles
        if args.collision_type == "sphere":
            positions = []
            for handle in obstacle_handles:
                positions.append(np.array(handle.position))
            world_obstacles_current = Sphere.from_center_and_radius(
                center=np.array(positions),
                radius=world_obstacles.radius,
            )
        else:
            positions = []
            rotations = []
            for handle in obstacle_handles:
                positions.append(np.array(handle.position))
                rotations.append(np.array(handle.wxyz))
            world_obstacles_current = Capsule.from_radius_height(
                radius=world_obstacles.radius,
                height=world_obstacles.height,
                position=np.array(positions),
                wxyz=np.array(rotations),
            )

        # Build target pose
        T_world_target = jaxlie.SE3(
            jnp.concatenate(
                [
                    jnp.array(ik_target_handle.wxyz),
                    jnp.array(ik_target_handle.position),
                ],
                axis=-1,
            )
        )

        # Solve IK
        if args.collision_type == "sphere":
            if use_analytic_jac.value:
                solution = solve_ik_sphere_analytic(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_spheres=world_obstacles_current,
                    world_halfspaces=[ground_plane],
                    T_world_target=T_world_target,
                    target_link_index=jnp.array(target_link_idx),
                )
            else:
                solution = solve_ik_sphere_autodiff(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_spheres=world_obstacles_current,
                    world_halfspaces=[ground_plane],
                    T_world_target=T_world_target,
                    target_link_index=jnp.array(target_link_idx),
                )
        else:
            if use_analytic_jac.value:
                solution = solve_ik_capsule_analytic(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_capsules=world_obstacles_current,
                    world_halfspaces=[ground_plane],
                    T_world_target=T_world_target,
                    target_link_index=jnp.array(target_link_idx),
                )
            else:
                solution = solve_ik_capsule_autodiff(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_capsules=world_obstacles_current,
                    world_halfspaces=[ground_plane],
                    T_world_target=T_world_target,
                    target_link_index=jnp.array(target_link_idx),
                )

        # Update timing
        elapsed_ms = (time.time() - start_time) * 1000
        timing_handle.value = elapsed_ms

        # Update visualizer
        urdf_vis.update_cfg(np.array(solution))


if __name__ == "__main__":
    main()
