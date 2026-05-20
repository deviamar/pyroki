import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


# ============================================================
# DQ REGULARIZATION
# ============================================================

@jaxls.Cost.create_factory
def previous_configuration_residual(
    vals,
    joint_var,
    prev_q,
    weight,
):
    q = vals[joint_var]
    return weight * (q - prev_q)


# ============================================================
# PUBLIC SOLVER
# ============================================================

def solve_trajectories_ik(

    robot: pk.Robot,

    target_link_names: list[str],

    target_positions: list[np.ndarray],

    target_wxyzs: list[np.ndarray],

    prev_q: np.ndarray,

    dt: float,

    joint_velocity_limits: np.ndarray,

    pos_weight: float = 50.0,

    ori_weight: float = 0.0,

    dq_weight: float = 0.5,

) -> np.ndarray:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    n_targets = len(target_link_names)

    assert len(target_positions) == n_targets
    assert len(target_wxyzs) == n_targets

    for p in target_positions:
        assert p.shape == (3,)

    for q in target_wxyzs:
        assert q.shape == (4,)

    assert prev_q.shape == (
        robot.joints.num_actuated_joints,
    )

    # --------------------------------------------------------
    # LINK INDICES
    # --------------------------------------------------------

    target_link_indices = np.array([

        robot.links.names.index(name)

        for name in target_link_names
    ])

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------

    q_next = _solve_trajectories_ik_jax(

        robot=robot,

        target_link_indices=jnp.array(
            target_link_indices
        ),

        target_positions=jnp.array(
            target_positions
        ),

        target_wxyzs=jnp.array(
            target_wxyzs
        ),

        prev_q=jnp.array(prev_q),

        dt=jnp.array(dt),

        joint_velocity_limits=jnp.array(
            joint_velocity_limits
        ),

        pos_weight=jnp.array(pos_weight),

        ori_weight=jnp.array(ori_weight),

        dq_weight=jnp.array(dq_weight),
    )

    q_next = np.array(q_next)

    # --------------------------------------------------------
    # VELOCITY CLAMP
    # --------------------------------------------------------

    dq = q_next - prev_q

    max_dq = (
        joint_velocity_limits * dt
    )

    dq = np.clip(
        dq,
        -max_dq,
        max_dq,
    )

    q_next = prev_q + dq

    return q_next


# ============================================================
# JAX SOLVER
# ============================================================

@jdc.jit
def _solve_trajectories_ik_jax(

    robot: pk.Robot,

    target_link_indices: jax.Array,

    target_positions: jax.Array,

    target_wxyzs: jax.Array,

    prev_q: jax.Array,

    dt: jax.Array,

    joint_velocity_limits: jax.Array,

    pos_weight: jax.Array,

    ori_weight: jax.Array,

    dq_weight: jax.Array,

) -> jax.Array:

    # --------------------------------------------------------
    # JOINT VARIABLE
    # --------------------------------------------------------

    joint_var = robot.joint_var_cls(0)

    variables = [joint_var]

    # --------------------------------------------------------
    # COSTS
    # --------------------------------------------------------

    costs = []

    # ========================================================
    # MULTI-END-EFFECTOR TASKS
    # ========================================================

    for i in range(target_link_indices.shape[0]):

        T_world_target = (
            jaxlie.SE3.from_rotation_and_translation(

                jaxlie.SO3(
                    target_wxyzs[i]
                ),

                target_positions[i],
            )
        )

        costs.append(

            pk.costs.pose_cost_analytic_jac(

                robot,

                joint_var,

                T_world_target,

                target_link_indices[i],

                pos_weight=pos_weight,

                ori_weight=ori_weight,
            )
        )

    # ========================================================
    # JOINT LIMITS
    # ========================================================

    costs.append(

        pk.costs.limit_constraint(

            robot,

            joint_var,
        )
    )

    # ========================================================
    # TEMPORAL SMOOTHNESS
    # ========================================================

    costs.append(

        previous_configuration_residual(

            joint_var=joint_var,

            prev_q=prev_q,

            weight=dq_weight,
        )
    )

    # ========================================================
    # SOLVE
    # ========================================================

    problem = (

        jaxls.LeastSquaresProblem(

            costs=costs,

            variables=variables,
        )
    )

    sol = (

        problem

        .analyze()

        .solve(

            verbose=False,

            linear_solver="dense_cholesky",

            trust_region=jaxls.TrustRegionConfig(

                lambda_initial=1.0,
            ),
        )
    )

    q_next = sol[joint_var]

    return q_next