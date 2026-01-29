"""Tests for Stewart platform loop closure."""

import tempfile

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy


def create_stewart_urdf(
    base_radius: float = 0.25,
    platform_radius: float = 0.20,
    nominal_leg_length: float = 0.4,
    stroke: float = 0.1,
) -> tuple[str, list[float], list[float]]:
    """Create a URDF for a Stewart platform with 6 legs."""
    base_angles = [i * np.pi / 3 for i in range(6)]
    platform_angles = [i * np.pi / 3 + np.pi / 6 for i in range(6)]

    xml = f"""<?xml version="1.0"?>
    <robot name="stewart_platform">
        <link name="base_link">
            <visual>
                <geometry>
                    <cylinder length="0.02" radius="{base_radius}"/>
                </geometry>
            </visual>
            <inertial>
                <mass value="10.0"/>
                <inertia ixx="0.1" iyy="0.1" izz="0.1"/>
            </inertial>
        </link>
"""

    for i in range(6):
        bx = base_radius * np.cos(base_angles[i])
        by = base_radius * np.sin(base_angles[i])

        xml += f"""
        <link name="leg_{i}_base">
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

        <link name="leg_{i}_pitch">
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

        <link name="leg_{i}_roll">
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

        <link name="leg_{i}_sphere_1">
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

        <link name="leg_{i}_tip">
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


def _load_stewart_robot() -> tuple[pk.Robot, list[float]]:
    """Load a Stewart platform robot for testing."""
    xml, _, platform_angles = create_stewart_urdf()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)
    return pk.Robot.from_urdf(urdf), platform_angles


def _get_platform_attachment_transforms(
    platform_radius: float,
    platform_angles: list[float],
) -> jaxlie.SE3:
    """Compute SE3 transforms from platform center to each leg attachment point."""
    wxyz_xyz_list = []
    for angle in platform_angles:
        x = platform_radius * np.cos(angle)
        y = platform_radius * np.sin(angle)
        z = -0.01
        wxyz_xyz_list.append(
            jnp.concatenate([jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([x, y, z])])
        )
    return jaxlie.SE3(jnp.stack(wxyz_xyz_list))


def test_stewart_urdf_loads():
    """Test that the Stewart platform URDF loads correctly."""
    robot, _ = _load_stewart_robot()

    # Should have 6 legs x 6 actuated joints per leg = 36 actuated joints
    assert robot.joints.num_actuated_joints == 36

    # Check that all tip links exist
    for i in range(6):
        assert f"leg_{i}_tip" in robot.links.names
        assert f"leg_{i}_prismatic_joint" in robot.joints.names


def test_stewart_closure_residual_shape():
    """Test that stewart_closure_residual returns correct shape."""
    robot, platform_angles = _load_stewart_robot()

    tip_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_tip") for i in range(6)],
        dtype=jnp.int32,
    )

    T_platform_tips = _get_platform_attachment_transforms(0.20, platform_angles)

    platform_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(),
        jnp.array([0.0, 0.0, 0.32]),
    )

    cfg = jnp.zeros(robot.joints.num_actuated_joints)

    joint_var = robot.joint_var_cls(0)
    platform_var = jaxls.SE3Var(0)

    vals = jaxls.VarValues.make(
        [joint_var.with_value(cfg), platform_var.with_value(platform_pose)]
    )

    from pyroki._residuals import stewart_closure_residual

    residual = stewart_closure_residual(
        vals=vals,
        robot=robot,
        joint_var=joint_var,
        platform_var=platform_var,
        tip_link_indices=tip_link_indices,
        T_platform_tips=T_platform_tips,
        weight=1.0,
    )

    # Should be 3 position components per leg x 6 legs = 18
    assert residual.shape == (18,)


def test_stewart_ik_closure_error():
    """Test that Stewart IK achieves low closure error."""
    robot, platform_angles = _load_stewart_robot()

    tip_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_tip") for i in range(6)],
        dtype=jnp.int32,
    )

    T_platform_tips = _get_platform_attachment_transforms(0.20, platform_angles)

    # Target platform pose
    target_platform_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(),
        jnp.array([0.0, 0.0, 0.32]),
    )

    joint_var = robot.joint_var_cls(0)
    platform_var = jaxls.SE3Var(0)

    factors = [
        # Target platform pose
        jaxls.Cost(
            lambda vals, p_var, target=target_platform_pose: (
                (vals[p_var].inverse() @ target).log() * 50.0
            ).flatten(),
            (platform_var,),
            name="platform_pose_cost",
        ),
        # Loop closure
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

    cfg_sol = sol[joint_var]
    platform_sol = sol[platform_var]

    # Compute closure error
    Ts_world_link = robot.forward_kinematics(cfg_sol)

    max_error = 0.0
    for i in range(6):
        T_world_tip = jaxlie.SE3(Ts_world_link[tip_link_indices[i]])
        T_world_target = platform_sol @ jaxlie.SE3(T_platform_tips.wxyz_xyz[i])
        pos_error = float(
            jnp.linalg.norm(T_world_tip.translation() - T_world_target.translation())
        )
        max_error = max(max_error, pos_error)

    # Closure error should be less than 1mm
    assert max_error < 0.001, f"Max closure error: {max_error * 1000:.3f}mm"


def test_stewart_workspace_reachability():
    """Test that various platform poses within workspace are reachable."""
    robot, platform_angles = _load_stewart_robot()

    tip_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_tip") for i in range(6)],
        dtype=jnp.int32,
    )

    T_platform_tips = _get_platform_attachment_transforms(0.20, platform_angles)

    # Test various target poses
    test_poses = [
        # Centered at different heights
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.30])
        ),
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.35])
        ),
        # Small translations
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.02, 0.0, 0.32])
        ),
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.0, 0.02, 0.32])
        ),
        # Small rotations
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_x_radians(0.1), jnp.array([0.0, 0.0, 0.32])
        ),
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_y_radians(0.1), jnp.array([0.0, 0.0, 0.32])
        ),
    ]

    for i, target_pose in enumerate(test_poses):
        joint_var = robot.joint_var_cls(0)
        platform_var = jaxls.SE3Var(0)

        factors = [
            jaxls.Cost(
                lambda vals, p_var, target=target_pose: (
                    (vals[p_var].inverse() @ target).log() * 50.0
                ).flatten(),
                (platform_var,),
                name="platform_pose_cost",
            ),
            pk.costs.stewart_closure_cost(
                robot=robot,
                joint_var=joint_var,
                platform_var=platform_var,
                tip_link_indices=tip_link_indices,
                T_platform_tips=T_platform_tips,
                weight=1000.0,
            ),
            pk.costs.limit_cost(
                robot=robot,
                joint_var=joint_var,
                weight=100.0,
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

        cfg_sol = sol[joint_var]
        platform_sol = sol[platform_var]

        # Compute closure error
        Ts_world_link = robot.forward_kinematics(cfg_sol)
        max_error = 0.0
        for leg_i in range(6):
            T_world_tip = jaxlie.SE3(Ts_world_link[tip_link_indices[leg_i]])
            T_world_target = platform_sol @ jaxlie.SE3(T_platform_tips.wxyz_xyz[leg_i])
            pos_error = float(
                jnp.linalg.norm(
                    T_world_tip.translation() - T_world_target.translation()
                )
            )
            max_error = max(max_error, pos_error)

        assert max_error < 0.001, (
            f"Pose {i}: max closure error {max_error * 1000:.3f}mm"
        )


def test_stewart_gradients():
    """Test that gradients flow through stewart_closure_residual."""
    robot, platform_angles = _load_stewart_robot()

    tip_link_indices = jnp.array(
        [robot.links.names.index(f"leg_{i}_tip") for i in range(6)],
        dtype=jnp.int32,
    )

    T_platform_tips = _get_platform_attachment_transforms(0.20, platform_angles)

    platform_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(),
        jnp.array([0.0, 0.0, 0.32]),
    )

    cfg = jnp.zeros(robot.joints.num_actuated_joints) + 0.1

    joint_var = robot.joint_var_cls(0)
    platform_var = jaxls.SE3Var(0)

    from pyroki._residuals import stewart_closure_residual

    def loss_fn(cfg_val):
        vals = jaxls.VarValues.make(
            [joint_var.with_value(cfg_val), platform_var.with_value(platform_pose)]
        )
        residual = stewart_closure_residual(
            vals=vals,
            robot=robot,
            joint_var=joint_var,
            platform_var=platform_var,
            tip_link_indices=tip_link_indices,
            T_platform_tips=T_platform_tips,
            weight=1.0,
        )
        return jnp.sum(residual**2)

    grad = jax.grad(loss_fn)(cfg)

    # Gradients should be finite and non-zero
    assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradients: {grad}"
    assert jnp.linalg.norm(grad) > 1e-6, (
        f"Gradient norm too small: {jnp.linalg.norm(grad)}"
    )
