"""4-Bar Linkage with Loop Closure Constraint

This example demonstrates how to model a closed-loop 4-bar linkage mechanism
using PyRoki's loop closure constraints.

A 4-bar linkage has 4 rigid links connected by 4 revolute joints forming a closed
loop. Since URDF only supports tree-based kinematics, we model it as an open chain
with 4 joints and add a loop closure constraint to enforce that the end of the
chain connects back to the base.

Geometry:
             link_1 (horizontal bar)
    base ---[j0]-------------------[j1]--- link_2 (coupler)
      |                              |
      |                              |
      | link_0 (ground)              | link_3 (output)
      |                              |
      |                              |
    fixed                          [j3]--- (connects back to base via constraint)

The mechanism has 1 DOF - all joint angles are coupled through the closure constraint.
"""

import tempfile
import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf


def create_4bar_urdf(
    link_lengths: tuple[float, float, float, float] = (0.3, 0.4, 0.3, 0.4),
) -> str:
    """Create a URDF for an open-chain 4-bar linkage.

    The linkage is modeled as a chain: base -> link_0 -> link_1 -> link_2 -> link_3
    A loop closure constraint will be added to connect link_3 back to base.

    Args:
        link_lengths: (L0, L1, L2, L3) lengths of the 4 links
    """
    L0, L1, L2, L3 = link_lengths

    xml = f"""<?xml version="1.0"?>
    <robot name="four_bar_linkage">
        <!-- Base/ground link (fixed) -->
        <link name="base_link">
            <visual>
                <geometry>
                    <box size="0.05 0.05 0.05"/>
                </geometry>
                <material name="gray">
                    <color rgba="0.5 0.5 0.5 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="1.0"/>
                <inertia ixx="0.001" iyy="0.001" izz="0.001"/>
            </inertial>
        </link>

        <!-- Link 0: Input crank (vertical) -->
        <link name="link_0">
            <visual>
                <origin xyz="0 0 {L0 / 2}"/>
                <geometry>
                    <cylinder length="{L0}" radius="0.015"/>
                </geometry>
                <material name="red">
                    <color rgba="1 0 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
            </inertial>
        </link>

        <!-- Joint 0: Base to input crank -->
        <joint name="joint_0" type="revolute">
            <parent link="base_link"/>
            <child link="link_0"/>
            <origin xyz="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" velocity="1.0"/>
        </joint>

        <!-- Link 1: Coupler (horizontal) -->
        <link name="link_1">
            <visual>
                <origin xyz="{L1 / 2} 0 0"/>
                <geometry>
                    <box size="{L1} 0.03 0.03"/>
                </geometry>
                <material name="green">
                    <color rgba="0 1 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
            </inertial>
        </link>

        <!-- Joint 1: Input crank to coupler -->
        <joint name="joint_1" type="revolute">
            <parent link="link_0"/>
            <child link="link_1"/>
            <origin xyz="0 0 {L0}"/>
            <axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" velocity="1.0"/>
        </joint>

        <!-- Link 2: Output rocker (vertical) -->
        <link name="link_2">
            <visual>
                <origin xyz="0 0 {-L2 / 2}"/>
                <geometry>
                    <cylinder length="{L2}" radius="0.015"/>
                </geometry>
                <material name="blue">
                    <color rgba="0 0 1 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
            </inertial>
        </link>

        <!-- Joint 2: Coupler to output rocker -->
        <joint name="joint_2" type="revolute">
            <parent link="link_1"/>
            <child link="link_2"/>
            <origin xyz="{L1} 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" velocity="1.0"/>
        </joint>

        <!-- Link 3: Closure link (this should connect back to base) -->
        <link name="link_3">
            <visual>
                <geometry>
                    <sphere radius="0.025"/>
                </geometry>
                <material name="yellow">
                    <color rgba="1 1 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
            </inertial>
        </link>

        <!-- Joint 3: Output rocker to closure point -->
        <joint name="joint_3" type="revolute">
            <parent link="link_2"/>
            <child link="link_3"/>
            <origin xyz="0 0 {-L2}"/>
            <axis xyz="0 1 0"/>
            <limit lower="-3.14" upper="3.14" velocity="1.0"/>
        </joint>

        <!-- Target link: This is where link_3 should be positioned (at base level, offset by L3) -->
        <link name="closure_target">
            <visual>
                <geometry>
                    <sphere radius="0.02"/>
                </geometry>
                <material name="purple">
                    <color rgba="0.5 0 0.5 0.5"/>
                </material>
            </visual>
        </link>

        <!-- Fixed joint showing where closure should happen -->
        <joint name="closure_target_joint" type="fixed">
            <parent link="base_link"/>
            <child link="closure_target"/>
            <origin xyz="{L3} 0 0"/>
        </joint>
    </robot>
    """
    return xml


@jdc.jit
def _solve_4bar_jax(
    robot: pk.Robot,
    input_angle_rad: jax.Array,
    link_3_idx: jax.Array,
    base_link_idx: jax.Array,
    T_base_closure_target: jaxlie.SE3,
) -> jax.Array:
    """JIT-compiled solver for 4-bar linkage with loop closure."""
    joint_var = robot.joint_var_cls(0)

    factors = [
        # Input angle constraint: joint_0 should be at the specified angle
        jaxls.Cost(
            lambda vals, var, target=input_angle_rad: (
                (vals[var][0] - target) * 100.0
            ).reshape(-1),
            (joint_var,),
            name="input_angle_cost",
        ),
        # Loop closure constraint: link_3 position should match closure target
        pk.costs.loop_closure_cost(
            robot=robot,
            joint_var=joint_var,
            link_a_index=base_link_idx,
            link_b_index=link_3_idx,
            T_a_b=T_base_closure_target,
            pos_weight=500.0,
            ori_weight=50.0,
        ),
        # Joint limits
        pk.costs.limit_cost(
            robot=robot,
            joint_var=joint_var,
            weight=100.0,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        )
    )
    return sol[joint_var]


def main():
    """Main function demonstrating 4-bar linkage with loop closure."""

    # Create the 4-bar linkage
    # Link lengths: L0=0.3 (input), L1=0.4 (coupler), L2=0.3 (output), L3=0.4 (ground)
    link_lengths = (0.3, 0.4, 0.3, 0.4)
    L0, L1, L2, L3 = link_lengths

    xml = create_4bar_urdf(link_lengths)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)

    robot = pk.Robot.from_urdf(urdf)

    print("Robot links:", robot.links.names)
    print("Robot joints:", robot.joints.names)
    print("Num actuated joints:", robot.joints.num_actuated_joints)

    # Get link indices
    link_3_idx = jnp.array(robot.links.names.index("link_3"), dtype=jnp.int32)
    base_link_idx = jnp.array(robot.links.names.index("base_link"), dtype=jnp.int32)

    # The loop closure constraint: link_3 should be at (L3, 0, 0) relative to base
    # This is where the ground link would connect in a true 4-bar
    T_base_closure_target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([L3, 0.0, 0.0])
    )

    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Add slider for input angle
    input_angle_slider = server.gui.add_slider(
        "Input Angle (deg)", min=-180, max=180, initial_value=45, step=1
    )
    timing_handle = server.gui.add_number("Solve time (ms)", 0.001, disabled=True)
    closure_error_handle = server.gui.add_number(
        "Closure error (mm)", 0.0, disabled=True
    )

    # Visualize the closure constraint
    server.scene.add_frame("/closure_target", axes_length=0.05, axes_radius=0.005)

    print("\nUse the slider to set the input angle. The mechanism will solve for")
    print("joint angles that satisfy the loop closure constraint.")
    print(f"Loop closure: link_3 should be at ({L3}, 0, 0) relative to base_link")

    while True:
        input_angle_rad = jnp.array(np.radians(input_angle_slider.value))

        start_time = time.time()
        cfg_sol = _solve_4bar_jax(
            robot,
            input_angle_rad,
            link_3_idx,
            base_link_idx,
            T_base_closure_target,
        )
        jax.block_until_ready(cfg_sol)
        elapsed = time.time() - start_time

        # Compute closure error
        Ts_world_link = robot.forward_kinematics(cfg_sol)
        T_world_base = jaxlie.SE3(Ts_world_link[base_link_idx])
        T_world_link3 = jaxlie.SE3(Ts_world_link[link_3_idx])
        T_base_link3 = T_world_base.inverse() @ T_world_link3

        closure_error = (T_base_link3 @ T_base_closure_target.inverse()).log()
        pos_error_mm = float(jnp.linalg.norm(closure_error[:3])) * 1000

        # Update UI
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * (elapsed * 1000)
        closure_error_handle.value = pos_error_mm

        # Update robot visualization
        urdf_vis.update_cfg(np.array(cfg_sol))

        # Update closure target visualization
        T_world_target = T_world_base @ T_base_closure_target
        server.scene.add_frame(
            "/closure_target",
            position=np.array(T_world_target.translation()),
            wxyz=np.array(T_world_target.rotation().wxyz),
            axes_length=0.05,
            axes_radius=0.005,
        )


if __name__ == "__main__":
    main()
