r"""Stewart Platform (Hexapod) with Loop Closure Constraints

This example demonstrates a 6-DOF Stewart platform (hexapod) parallel manipulator
using PyRoki's loop closure constraints.

A Stewart platform has 6 legs connecting a fixed base to a moving platform.
Each leg consists of:
- Universal joint at base (2 DOF: pitch + roll)
- Prismatic actuator (1 DOF: extension)
- Spherical joint at platform (3 DOF: modeled as 3 revolute joints)

Since URDF only supports tree-based kinematics, we model each leg as an
independent kinematic chain and use loop closure constraints to connect
the leg tips to the moving platform.

The platform pose (SE3) is an optimization variable, and the solver finds
joint configurations that satisfy the closure constraints.

Geometry:
                Platform (movable)
           o-------o-------o
          /|       |       |\
         / |       |       | \
        /  |       |       |  \
       /   |       |       |   \
      o    o       o       o    o   (6 legs with prismatic actuators)
       \   |       |       |   /
        \  |       |       |  /
         \ |       |       | /
          \|       |       |/
           o-------o-------o
                Base (fixed)
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


def create_stewart_urdf(
    base_radius: float = 0.25,
    platform_radius: float = 0.20,
    nominal_leg_length: float = 0.4,
    stroke: float = 0.1,
) -> tuple[str, list[float], list[float]]:
    """Create a URDF for a Stewart platform with 6 legs.

    Each leg is modeled as an independent kinematic chain:
    base_link -> leg_i_base -> leg_i_pitch -> leg_i_roll -> leg_i_prismatic
              -> leg_i_sphere_1 -> leg_i_sphere_2 -> leg_i_sphere_3 -> leg_i_tip

    Args:
        base_radius: Radius of the hexagonal base (distance from center to vertices).
        platform_radius: Radius of the hexagonal platform.
        nominal_leg_length: Nominal length of the prismatic actuator.
        stroke: Maximum extension/retraction from nominal (±stroke).

    Returns:
        Tuple of (urdf_xml, base_angles, platform_angles) where angles are in radians.
    """
    # Base attachment points at 60 degree intervals
    base_angles = [i * np.pi / 3 for i in range(6)]
    # Platform attachment points offset by 30 degrees from base
    platform_angles = [i * np.pi / 3 + np.pi / 6 for i in range(6)]

    xml = f"""<?xml version="1.0"?>
    <robot name="stewart_platform">
        <!-- Base link (fixed) -->
        <link name="base_link">
            <visual>
                <geometry>
                    <cylinder length="0.02" radius="{base_radius}"/>
                </geometry>
                <material name="gray">
                    <color rgba="0.5 0.5 0.5 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="10.0"/>
                <inertia ixx="0.1" iyy="0.1" izz="0.1"/>
            </inertial>
        </link>
"""

    for i in range(6):
        # Base attachment point position
        bx = base_radius * np.cos(base_angles[i])
        by = base_radius * np.sin(base_angles[i])

        xml += f"""
        <!-- Leg {i}: Base attachment point -->
        <link name="leg_{i}_base">
            <visual>
                <geometry>
                    <sphere radius="0.015"/>
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

        <joint name="leg_{i}_base_joint" type="fixed">
            <parent link="base_link"/>
            <child link="leg_{i}_base"/>
            <origin xyz="{bx:.6f} {by:.6f} 0.01"/>
        </joint>

        <!-- Leg {i}: Universal joint - pitch -->
        <link name="leg_{i}_pitch">
            <visual>
                <geometry>
                    <cylinder length="0.02" radius="0.01"/>
                </geometry>
                <material name="green">
                    <color rgba="0 0.8 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.05"/>
                <inertia ixx="0.00001" iyy="0.00001" izz="0.00001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_pitch_joint" type="revolute">
            <parent link="leg_{i}_base"/>
            <child link="leg_{i}_pitch"/>
            <origin xyz="0 0 0"/>
            <axis xyz="1 0 0"/>
            <limit lower="-1.57" upper="1.57" velocity="2.0"/>
        </joint>

        <!-- Leg {i}: Universal joint - roll -->
        <link name="leg_{i}_roll">
            <visual>
                <geometry>
                    <cylinder length="0.02" radius="0.01"/>
                </geometry>
                <material name="green">
                    <color rgba="0 0.8 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.05"/>
                <inertia ixx="0.00001" iyy="0.00001" izz="0.00001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_roll_joint" type="revolute">
            <parent link="leg_{i}_pitch"/>
            <child link="leg_{i}_roll"/>
            <origin xyz="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-1.57" upper="1.57" velocity="2.0"/>
        </joint>

        <!-- Leg {i}: Prismatic actuator (no visual - drawn with Viser lines) -->
        <link name="leg_{i}_prismatic">
            <inertial>
                <mass value="0.2"/>
                <inertia ixx="0.001" iyy="0.001" izz="0.001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_prismatic_joint" type="prismatic">
            <parent link="leg_{i}_roll"/>
            <child link="leg_{i}_prismatic"/>
            <origin xyz="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="{-stroke}" upper="{stroke}" velocity="0.5"/>
        </joint>

        <!-- Leg {i}: Spherical joint - first axis -->
        <link name="leg_{i}_sphere_1">
            <visual>
                <geometry>
                    <sphere radius="0.01"/>
                </geometry>
                <material name="yellow">
                    <color rgba="1 1 0 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.02"/>
                <inertia ixx="0.000001" iyy="0.000001" izz="0.000001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_sphere_1_joint" type="revolute">
            <parent link="leg_{i}_prismatic"/>
            <child link="leg_{i}_sphere_1"/>
            <origin xyz="0 0 {nominal_leg_length}"/>
            <axis xyz="1 0 0"/>
            <limit lower="-1.57" upper="1.57" velocity="2.0"/>
        </joint>

        <!-- Leg {i}: Spherical joint - second axis -->
        <link name="leg_{i}_sphere_2">
            <inertial>
                <mass value="0.01"/>
                <inertia ixx="0.000001" iyy="0.000001" izz="0.000001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_sphere_2_joint" type="revolute">
            <parent link="leg_{i}_sphere_1"/>
            <child link="leg_{i}_sphere_2"/>
            <origin xyz="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-1.57" upper="1.57" velocity="2.0"/>
        </joint>

        <!-- Leg {i}: Spherical joint - third axis -->
        <link name="leg_{i}_sphere_3">
            <inertial>
                <mass value="0.01"/>
                <inertia ixx="0.000001" iyy="0.000001" izz="0.000001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_sphere_3_joint" type="revolute">
            <parent link="leg_{i}_sphere_2"/>
            <child link="leg_{i}_sphere_3"/>
            <origin xyz="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" velocity="2.0"/>
        </joint>

        <!-- Leg {i}: Tip link (closure point) -->
        <link name="leg_{i}_tip">
            <visual>
                <geometry>
                    <sphere radius="0.015"/>
                </geometry>
                <material name="purple">
                    <color rgba="0.8 0 0.8 1"/>
                </material>
            </visual>
            <inertial>
                <mass value="0.01"/>
                <inertia ixx="0.000001" iyy="0.000001" izz="0.000001"/>
            </inertial>
        </link>

        <joint name="leg_{i}_tip_joint" type="fixed">
            <parent link="leg_{i}_sphere_3"/>
            <child link="leg_{i}_tip"/>
            <origin xyz="0 0 0"/>
        </joint>
"""

    xml += """
    </robot>
    """
    return xml, base_angles, platform_angles


def get_platform_attachment_transforms(
    platform_radius: float,
    platform_angles: list[float],
) -> jaxlie.SE3:
    """Compute SE3 transforms from platform center to each leg attachment point.

    Returns a batched SE3 with shape (6,).
    """
    wxyz_xyz_list = []
    for angle in platform_angles:
        x = platform_radius * np.cos(angle)
        y = platform_radius * np.sin(angle)
        z = -0.01  # Slightly below platform center
        wxyz_xyz_list.append(
            jnp.concatenate([jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([x, y, z])])
        )
    return jaxlie.SE3(jnp.stack(wxyz_xyz_list))


@jdc.jit
def _solve_stewart_ik_jax(
    robot: pk.Robot,
    platform_pose_init: jaxlie.SE3,
    target_platform_pose: jaxlie.SE3,
    tip_link_indices: jax.Array,
    T_platform_tips: jaxlie.SE3,
) -> tuple[jax.Array, jaxlie.SE3]:
    """JIT-compiled solver for Stewart platform IK.

    Args:
        robot: The robot model.
        platform_pose_init: Initial guess for platform pose.
        target_platform_pose: Target pose for the platform.
        tip_link_indices: Indices of tip links for each leg.
        T_platform_tips: Transforms from platform to tip attachment points.

    Returns:
        Tuple of (joint_configuration, solved_platform_pose).
    """
    joint_var = robot.joint_var_cls(0)

    # Create a custom SE3Var with the initial pose as default
    class PlatformVar(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: platform_pose_init,
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=jaxls.SE3Var.retract_fn,
    ): ...

    platform_var = PlatformVar(0)

    factors = [
        # Target platform pose cost
        jaxls.Cost(
            lambda vals, p_var, target=target_platform_pose: (
                (vals[p_var].inverse() @ target).log()
                * jnp.array([50.0] * 3 + [10.0] * 3)
            ).flatten(),
            (platform_var,),
            name="platform_pose_cost",
        ),
        # Loop closure: leg tips match platform attachment points
        pk.costs.stewart_closure_cost(
            robot=robot,
            joint_var=joint_var,
            platform_var=platform_var,
            tip_link_indices=tip_link_indices,
            T_platform_tips=T_platform_tips,
            weight=1000.0,
        ),
        # Joint limits
        pk.costs.limit_cost(
            robot=robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        # Rest pose regularization (keep joints near zero)
        pk.costs.rest_cost(
            joint_var=joint_var,
            rest_pose=jnp.zeros(robot.joints.num_actuated_joints),
            weight=1.0,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var, platform_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=0.1),
        )
    )

    return sol[joint_var], sol[platform_var]


def compute_closure_error(
    robot: pk.Robot,
    cfg: jax.Array,
    platform_pose: jaxlie.SE3,
    tip_link_indices: jax.Array,
    T_platform_tips: jaxlie.SE3,
) -> float:
    """Compute the maximum closure error across all legs."""
    Ts_world_link = robot.forward_kinematics(cfg)

    max_error = 0.0
    for i in range(6):
        T_world_tip = jaxlie.SE3(Ts_world_link[tip_link_indices[i]])
        T_world_target = platform_pose @ jaxlie.SE3(T_platform_tips.wxyz_xyz[i])
        pos_error = jnp.linalg.norm(
            T_world_tip.translation() - T_world_target.translation()
        )
        max_error = max(max_error, float(pos_error))

    return max_error


def compute_leg_lengths(
    robot: pk.Robot,
    cfg: jax.Array,
    nominal_length: float,
) -> list[float]:
    """Compute the current length of each leg's prismatic actuator."""
    # The prismatic joint values directly give the extension from nominal
    # Joint order: for each leg (pitch, roll, prismatic, sphere_1, sphere_2, sphere_3)
    lengths = []
    joints_per_leg = 6
    for i in range(6):
        prismatic_idx = i * joints_per_leg + 2  # pitch, roll, prismatic
        extension = float(cfg[prismatic_idx])
        lengths.append(nominal_length + extension)
    return lengths


def update_leg_visualization(
    server: viser.ViserServer,
    robot: pk.Robot,
    cfg: jax.Array,
    base_link_indices: jax.Array,
    tip_link_indices: jax.Array,
) -> None:
    """Draw lines from base attachment points to tip positions."""
    Ts_world_link = robot.forward_kinematics(cfg)
    for i in range(6):
        base_pos = jaxlie.SE3(Ts_world_link[base_link_indices[i]]).translation()
        tip_pos = jaxlie.SE3(Ts_world_link[tip_link_indices[i]]).translation()
        server.scene.add_spline_catmull_rom(
            f"/leg_{i}_line",
            positions=np.array([base_pos, tip_pos]),
            color=(0.2, 0.2, 0.8),
            line_width=3.0,
        )


def update_platform_visualization(
    server: viser.ViserServer,
    platform_pose: jaxlie.SE3,
    platform_radius: float,
    thickness: float = 0.015,
) -> None:
    """Draw hexagonal platform plate at solved pose."""
    # Create hexagon vertices in platform local frame
    angles = [i * np.pi / 3 + np.pi / 6 for i in range(6)]
    half_t = thickness / 2

    # Vertices: bottom center, bottom hex (6), top center, top hex (6) = 14 vertices
    vertices = []
    # Bottom center (index 0)
    vertices.append([0, 0, -half_t])
    # Bottom hexagon (indices 1-6)
    for a in angles:
        vertices.append(
            [platform_radius * np.cos(a), platform_radius * np.sin(a), -half_t]
        )
    # Top center (index 7)
    vertices.append([0, 0, half_t])
    # Top hexagon (indices 8-13)
    for a in angles:
        vertices.append(
            [platform_radius * np.cos(a), platform_radius * np.sin(a), half_t]
        )

    vertices = np.array(vertices)

    faces = []
    # Bottom face (fan from center, winding for downward normal)
    for i in range(6):
        faces.append([0, ((i + 1) % 6) + 1, i + 1])
    # Top face (fan from center, winding for upward normal)
    for i in range(6):
        faces.append([7, i + 8, ((i + 1) % 6) + 8])
    # Side faces (6 quads = 12 triangles)
    for i in range(6):
        b1 = i + 1  # bottom vertex
        b2 = ((i + 1) % 6) + 1  # next bottom vertex
        t1 = i + 8  # top vertex
        t2 = ((i + 1) % 6) + 8  # next top vertex
        faces.append([b1, b2, t2])
        faces.append([b1, t2, t1])

    faces = np.array(faces)

    server.scene.add_mesh_simple(
        "/platform_mesh",
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.uint32),
        position=np.array(platform_pose.translation()),
        wxyz=np.array(platform_pose.rotation().wxyz),
        color=(0.6, 0.6, 0.7),
        opacity=0.9,
    )


def main():
    """Main function demonstrating Stewart platform with loop closure."""

    # Platform geometry parameters
    base_radius = 0.25
    platform_radius = 0.20
    nominal_leg_length = 0.4
    stroke = 0.1

    # Create URDF
    xml, base_angles, platform_angles = create_stewart_urdf(
        base_radius=base_radius,
        platform_radius=platform_radius,
        nominal_leg_length=nominal_leg_length,
        stroke=stroke,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)

    robot = pk.Robot.from_urdf(urdf)

    print("Robot links:", robot.links.names)
    print("Robot joints:", robot.joints.names)
    print("Num actuated joints:", robot.joints.num_actuated_joints)

    # Get tip link indices
    tip_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_tip") for i in range(6)],
        dtype=jnp.int32,
    )

    # Get base link indices (for leg line visualization)
    base_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_base") for i in range(6)],
        dtype=jnp.int32,
    )

    # Compute platform attachment transforms
    T_platform_tips = get_platform_attachment_transforms(
        platform_radius=platform_radius,
        platform_angles=platform_angles,
    )

    # Initial platform pose: centered above the base
    initial_height = (
        nominal_leg_length * 0.8
    )  # Slightly less than nominal for stability
    initial_platform_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(),
        jnp.array([0.0, 0.0, initial_height]),
    )

    # Set up visualizer
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=1.0, height=1.0)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Add transform controls for target platform pose
    target_handle = server.scene.add_transform_controls(
        "/platform_target",
        scale=0.15,
        position=(0.0, 0.0, initial_height),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    # GUI elements
    timing_handle = server.gui.add_number("Solve time (ms)", 0.001, disabled=True)
    closure_error_handle = server.gui.add_number(
        "Max closure error (mm)", 0.0, disabled=True
    )

    # Leg length displays
    leg_length_folder = server.gui.add_folder("Leg Lengths (m)")
    leg_length_handles = []
    with leg_length_folder:
        for i in range(6):
            handle = server.gui.add_number(
                f"Leg {i}", nominal_leg_length, disabled=True
            )
            leg_length_handles.append(handle)

    # Add platform visualization
    server.scene.add_frame(
        "/platform_actual",
        axes_length=0.1,
        axes_radius=0.005,
    )

    print("\nStewart Platform Demo")
    print("=" * 40)
    print(f"Base radius: {base_radius}m")
    print(f"Platform radius: {platform_radius}m")
    print(f"Nominal leg length: {nominal_leg_length}m")
    print(
        f"Stroke range: {nominal_leg_length - stroke}m to {nominal_leg_length + stroke}m"
    )
    print("\nUse the transform controls to move the target platform pose.")
    print("The solver will find joint configurations satisfying loop closure.\n")

    current_platform_pose = initial_platform_pose

    while True:
        # Get target from transform controls
        target_platform_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.array(target_handle.wxyz)),
            jnp.array(target_handle.position),
        )

        # Solve IK
        start_time = time.time()
        cfg_sol, platform_sol = _solve_stewart_ik_jax(
            robot,
            current_platform_pose,
            target_platform_pose,
            tip_link_indices,
            T_platform_tips,
        )
        jax.block_until_ready(cfg_sol)
        elapsed = time.time() - start_time

        # Update current platform pose for warm-starting next solve
        current_platform_pose = platform_sol

        # Compute closure error
        closure_error_m = compute_closure_error(
            robot, cfg_sol, platform_sol, tip_link_indices, T_platform_tips
        )
        closure_error_mm = closure_error_m * 1000

        # Compute leg lengths
        leg_lengths = compute_leg_lengths(robot, cfg_sol, nominal_leg_length)

        # Update UI
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * (elapsed * 1000)
        closure_error_handle.value = (
            0.9 * closure_error_handle.value + 0.1 * closure_error_mm
        )
        for i, length in enumerate(leg_lengths):
            leg_length_handles[i].value = round(length, 4)

        # Update robot visualization
        urdf_vis.update_cfg(np.array(cfg_sol))

        # Update platform pose visualization
        server.scene.add_frame(
            "/platform_actual",
            position=np.array(platform_sol.translation()),
            wxyz=np.array(platform_sol.rotation().wxyz),
            axes_length=0.1,
            axes_radius=0.005,
        )

        # Update leg lines (from base to tip)
        update_leg_visualization(
            server, robot, cfg_sol, base_link_indices, tip_link_indices
        )

        # Update platform hexagon mesh
        update_platform_visualization(server, platform_sol, platform_radius)


if __name__ == "__main__":
    main()
