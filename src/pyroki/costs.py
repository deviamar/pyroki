"""Cost and constraint factories for pyroki optimization.

Costs are soft penalties (minimized via least squares).
Constraints use augmented Lagrangian method (iteratively tightened penalties).

All factories are created by wrapping residual functions with Cost.factory.
"""

from jaxls import Cost

from ._residuals import (
    five_point_acceleration_residual,
    five_point_jerk_residual,
    five_point_velocity_residual,
    limit_residual,
    limit_velocity_residual,
    manipulability_residual,
    pose_residual,
    pose_with_base_residual,
    rest_residual,
    rest_with_base_residual,
    self_collision_residual,
    smoothness_residual,
    sphere_self_collision_residual,
    sphere_world_collision_residual,
    world_collision_residual,
)
from ._residuals._pose_residual_analytic_jac import (
    pose_cost_analytic_jac as pose_cost_analytic_jac,
)
from ._residuals._pose_residual_numerical_jac import (
    pose_cost_numerical_jac as pose_cost_numerical_jac,
)
from ._residuals._sphere_collision_analytic_jac import (
    sphere_self_collision_cost_analytic_jac as sphere_self_collision_cost_analytic_jac,
    sphere_self_collision_constraint_analytic_jac as sphere_self_collision_constraint_analytic_jac,
    sphere_world_collision_cost_analytic_jac as sphere_world_collision_cost_analytic_jac,
    sphere_world_collision_constraint_analytic_jac as sphere_world_collision_constraint_analytic_jac,
)
from ._residuals._capsule_collision_analytic_jac import (
    capsule_self_collision_cost_analytic_jac as capsule_self_collision_cost_analytic_jac,
    capsule_self_collision_constraint_analytic_jac as capsule_self_collision_constraint_analytic_jac,
    capsule_world_collision_cost_analytic_jac as capsule_world_collision_cost_analytic_jac,
    capsule_world_collision_constraint_analytic_jac as capsule_world_collision_constraint_analytic_jac,
)

# Pose costs
pose_cost = Cost.factory(pose_residual)
pose_cost_with_base = Cost.factory(pose_with_base_residual)

# Limit costs
limit_cost = Cost.factory(limit_residual)
limit_velocity_cost = Cost.factory(limit_velocity_residual)

# Regularization costs
rest_cost = Cost.factory(rest_residual)
rest_with_base_cost = Cost.factory(rest_with_base_residual)
smoothness_cost = Cost.factory(smoothness_residual)

# Manipulability cost
manipulability_cost = Cost.factory(manipulability_residual)

# Collision costs
self_collision_cost = Cost.factory(self_collision_residual)
world_collision_cost = Cost.factory(world_collision_residual)

# Sphere collision costs (using RobotCollision with sphere decomposition)
sphere_self_collision_cost = Cost.factory(sphere_self_collision_residual)
sphere_world_collision_cost = Cost.factory(sphere_world_collision_residual)

# Finite difference costs
five_point_velocity_cost = Cost.factory(five_point_velocity_residual)
five_point_acceleration_cost = Cost.factory(five_point_acceleration_residual)
five_point_jerk_cost = Cost.factory(five_point_jerk_residual)

# Constraint factories (augmented Lagrangian penalties)
# These use kind="constraint_leq_zero" which enforces residual <= 0
limit_constraint = Cost.factory(kind="constraint_leq_zero")(limit_residual)
limit_velocity_constraint = Cost.factory(kind="constraint_leq_zero")(
    limit_velocity_residual
)
world_collision_constraint = Cost.factory(kind="constraint_leq_zero")(
    world_collision_residual
)
sphere_self_collision_constraint = Cost.factory(kind="constraint_leq_zero")(
    sphere_self_collision_residual
)
sphere_world_collision_constraint = Cost.factory(kind="constraint_leq_zero")(
    sphere_world_collision_residual
)
