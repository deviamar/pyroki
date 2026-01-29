"""Tests for loop closure cost functions."""

import tempfile

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy


def _create_simple_chain_urdf(num_links: int = 5, link_length: float = 0.2) -> str:
    """Create a simple chain robot URDF for testing."""
    xml = """<?xml version="1.0"?>
    <robot name="test_chain">
        <link name="base_link"/>
    """

    for i in range(num_links):
        xml += f"""
        <link name="link_{i}">
            <visual>
                <geometry>
                    <cylinder length="{link_length}" radius="0.02"/>
                </geometry>
            </visual>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"/>
            </inertial>
        </link>
        """

        parent = "base_link" if i == 0 else f"link_{i-1}"
        z_offset = link_length / 2 if i == 0 else link_length
        xml += f"""
        <joint name="joint_{i}" type="revolute">
            <parent link="{parent}"/>
            <child link="link_{i}"/>
            <axis xyz="1 0 0"/>
            <origin xyz="0 0 {z_offset}" rpy="0 0 0"/>
            <limit lower="-3.14" upper="3.14" velocity="1.0"/>
        </joint>
        """

    xml += "</robot>"
    return xml


def _load_test_robot(num_links: int = 5) -> pk.Robot:
    """Load a simple test robot."""
    xml = _create_simple_chain_urdf(num_links=num_links)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(xml)
        f.flush()
        urdf = yourdfpy.URDF.load(f.name)
    return pk.Robot.from_urdf(urdf)


def _compute_loop_closure_error(robot, cfg, link_a_idx, link_b_idx, T_a_b_expected):
    """Helper to compute loop closure error directly."""
    Ts_world_link = robot.forward_kinematics(cfg)
    T_world_a = jaxlie.SE3(Ts_world_link[link_a_idx])
    T_world_b = jaxlie.SE3(Ts_world_link[link_b_idx])
    T_a_b_actual = T_world_a.inverse() @ T_world_b
    error = (T_a_b_actual @ T_a_b_expected.inverse()).log()
    return error


def test_loop_closure_error_zero_at_expected_pose():
    """Test that loop closure error is zero when links are at expected relative pose."""
    robot = _load_test_robot(num_links=5)

    # Get initial config
    cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # Compute FK to get actual link poses
    Ts_world_link = robot.forward_kinematics(cfg)

    # Pick two links
    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)

    T_world_a = jaxlie.SE3(Ts_world_link[link_a_idx])
    T_world_b = jaxlie.SE3(Ts_world_link[link_b_idx])

    # Compute the actual relative transform (this should be the "expected" one)
    T_a_b_expected = T_world_a.inverse() @ T_world_b

    # Compute error
    error = _compute_loop_closure_error(robot, cfg, link_a_idx, link_b_idx, T_a_b_expected)

    # Should be essentially zero
    assert jnp.allclose(error, 0.0, atol=1e-6), f"Error: {error}"


def test_loop_closure_error_nonzero_for_wrong_pose():
    """Test that loop closure error is non-zero when links are NOT at expected pose."""
    robot = _load_test_robot(num_links=5)

    cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)

    # Use identity as expected transform (definitely wrong for non-zero config)
    T_a_b_wrong = jaxlie.SE3.identity()

    error = _compute_loop_closure_error(robot, cfg, link_a_idx, link_b_idx, T_a_b_wrong)

    # Should be non-zero since we gave a wrong expected pose
    assert jnp.linalg.norm(error) > 1e-3, f"Error norm: {jnp.linalg.norm(error)}"


def test_loop_closure_gradients():
    """Test that gradients flow correctly through loop closure computation."""
    robot = _load_test_robot(num_links=5)

    # Use non-zero config to ensure gradients are non-zero
    cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2 + 0.5

    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)

    # Use identity as expected (will produce non-zero error for non-zero config)
    T_a_b = jaxlie.SE3.identity()

    def loss_fn(cfg_val):
        error = _compute_loop_closure_error(robot, cfg_val, link_a_idx, link_b_idx, T_a_b)
        return jnp.sum(error**2)

    # Compute gradients
    grad = jax.grad(loss_fn)(cfg)

    # Gradients should be finite and non-zero
    assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradients: {grad}"
    assert jnp.linalg.norm(grad) > 1e-6, f"Gradient norm too small: {jnp.linalg.norm(grad)}"


def test_loop_closure_in_optimization():
    """Test that loop closure can be used in actual optimization."""
    robot = _load_test_robot(num_links=5)

    # Start from mid-range config
    cfg_init = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # Get a target pose for the end effector
    target_link_idx = jnp.array(robot.links.num_links - 1, dtype=jnp.int32)

    Ts_init = robot.forward_kinematics(cfg_init)
    target_pose = jaxlie.SE3(Ts_init[target_link_idx])

    # Also capture initial relative transform between two links (for loop closure)
    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)

    T_world_a = jaxlie.SE3(Ts_init[link_a_idx])
    T_world_b = jaxlie.SE3(Ts_init[link_b_idx])
    T_a_b = T_world_a.inverse() @ T_world_b

    # Perturb the target slightly
    target_pose_perturbed = target_pose @ jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.05, 0.0, 0.0])
    )

    joint_var = robot.joint_var_cls(0)

    # Set up optimization with pose cost + loop closure cost
    factors = [
        pk.costs.pose_cost(
            robot=robot,
            joint_var=joint_var,
            target_pose=target_pose_perturbed,
            target_link_index=target_link_idx,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.loop_closure_cost(
            robot=robot,
            joint_var=joint_var,
            link_a_index=link_a_idx,
            link_b_index=link_b_idx,
            T_a_b=T_a_b,
            pos_weight=100.0,  # Higher weight to enforce closure
            ori_weight=20.0,
        ),
        pk.costs.limit_cost(
            robot=robot,
            joint_var=joint_var,
            weight=100.0,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )

    cfg_sol = sol[joint_var]

    # Check that the solution satisfies the loop closure reasonably well
    Ts_sol = robot.forward_kinematics(cfg_sol)
    T_world_a_sol = jaxlie.SE3(Ts_sol[link_a_idx])
    T_world_b_sol = jaxlie.SE3(Ts_sol[link_b_idx])
    T_a_b_sol = T_world_a_sol.inverse() @ T_world_b_sol

    # Compute error
    error = (T_a_b_sol @ T_a_b.inverse()).log()
    error_norm = jnp.linalg.norm(error)

    # Should have small closure error
    assert error_norm < 0.1, f"Loop closure error: {error_norm}"


def test_loop_closure_constraint_in_optimization():
    """Test the equality constraint version of loop closure."""
    robot = _load_test_robot(num_links=5)

    cfg_init = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # Get poses
    Ts_init = robot.forward_kinematics(cfg_init)
    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)
    target_link_idx = jnp.array(robot.links.num_links - 1, dtype=jnp.int32)

    T_world_a = jaxlie.SE3(Ts_init[link_a_idx])
    T_world_b = jaxlie.SE3(Ts_init[link_b_idx])
    T_a_b_expected = T_world_a.inverse() @ T_world_b

    # Perturb target
    target_pose = jaxlie.SE3(Ts_init[target_link_idx]) @ jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.05, 0.0, 0.0])
    )

    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost(
            robot=robot,
            joint_var=joint_var,
            target_pose=target_pose,
            target_link_index=target_link_idx,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.loop_closure_constraint(
            robot=robot,
            joint_var=joint_var,
            link_a_index=link_a_idx,
            link_b_index=link_b_idx,
            T_a_b=T_a_b_expected,
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )

    cfg_sol = sol[joint_var]

    # Check loop closure
    Ts_sol = robot.forward_kinematics(cfg_sol)
    T_a_sol = jaxlie.SE3(Ts_sol[link_a_idx])
    T_b_sol = jaxlie.SE3(Ts_sol[link_b_idx])
    T_a_b_sol = T_a_sol.inverse() @ T_b_sol
    error = jnp.linalg.norm((T_a_b_sol @ T_a_b_expected.inverse()).log())

    # With constraint, should have very small error
    assert error < 0.01, f"Loop closure constraint error: {error}"


def test_loop_closure_weight_effect():
    """Test that increasing weight reduces closure error."""
    robot = _load_test_robot(num_links=5)

    cfg_init = (robot.joints.lower_limits + robot.joints.upper_limits) / 2 + 0.5

    # Set up a conflicting objective: perturbed target pose
    target_link_idx = jnp.array(robot.links.num_links - 1, dtype=jnp.int32)
    Ts_init = robot.forward_kinematics(cfg_init)
    target_pose = jaxlie.SE3(Ts_init[target_link_idx]) @ jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.identity(), jnp.array([0.1, 0.1, 0.0])
    )

    # Loop closure between links
    link_a_idx = jnp.array(1, dtype=jnp.int32)
    link_b_idx = jnp.array(3, dtype=jnp.int32)
    T_world_a_init = jaxlie.SE3(Ts_init[link_a_idx])
    T_world_b_init = jaxlie.SE3(Ts_init[link_b_idx])
    T_a_b = T_world_a_init.inverse() @ T_world_b_init

    errors = []
    weights = [1.0, 10.0, 100.0, 1000.0]

    for w in weights:
        joint_var = robot.joint_var_cls(0)
        factors = [
            pk.costs.pose_cost(
                robot=robot,
                joint_var=joint_var,
                target_pose=target_pose,
                target_link_index=target_link_idx,
                pos_weight=10.0,
                ori_weight=5.0,
            ),
            pk.costs.loop_closure_cost(
                robot=robot,
                joint_var=joint_var,
                link_a_index=link_a_idx,
                link_b_index=link_b_idx,
                T_a_b=T_a_b,
                pos_weight=w,
                ori_weight=w / 5,
            ),
        ]

        sol = (
            jaxls.LeastSquaresProblem(factors, [joint_var])
            .analyze()
            .solve(verbose=False, linear_solver="dense_cholesky")
        )

        Ts_sol = robot.forward_kinematics(sol[joint_var])
        T_a_sol = jaxlie.SE3(Ts_sol[link_a_idx])
        T_b_sol = jaxlie.SE3(Ts_sol[link_b_idx])
        T_a_b_sol = T_a_sol.inverse() @ T_b_sol
        error = jnp.linalg.norm((T_a_b_sol @ T_a_b.inverse()).log())
        errors.append(float(error))

    # Higher weights should generally result in lower errors
    # (not strictly monotonic due to optimization landscape, but trend should be clear)
    assert errors[-1] < errors[0], f"Higher weight should reduce error: {errors}"
