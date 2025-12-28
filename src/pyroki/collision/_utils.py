from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Float, Array

_SAFE_EPS = 1e-6


def make_frame(direction: jax.Array) -> jax.Array:
    """Make a frame from a direction vector, aligning the z-axis with the direction."""
    # Based on `mujoco.mjx._src.math.make_frame`.

    is_zero = jnp.isclose(direction, 0.0).all(axis=-1, keepdims=True)
    direction = jnp.where(
        is_zero,
        jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), direction.shape),
        direction,
    )
    direction /= jnp.linalg.norm(direction, axis=-1, keepdims=True) + _SAFE_EPS

    y = jnp.broadcast_to(jnp.array([0, 1, 0]), (*direction.shape[:-1], 3))
    z = jnp.broadcast_to(jnp.array([0, 0, 1]), (*direction.shape[:-1], 3))

    normal = jnp.where((-0.5 < direction[..., 1:2]) & (direction[..., 1:2] < 0.5), y, z)
    normal -= direction * jnp.einsum("...i,...i->...", normal, direction)[..., None]
    normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True) + _SAFE_EPS

    return jnp.stack([jnp.cross(normal, direction), normal, direction], axis=-1)


def normalize(x: Float[Array, "*batch N"]) -> Float[Array, "*batch N"]:
    """Normalizes a vector, handling the zero vector."""
    norm: jax.Array = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    return jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)


def normalize_with_norm(
    x: Float[Array, "*batch N"],
) -> Tuple[Float[Array, "*batch N"], Float[Array, "*batch"]]:
    """Normalizes a vector and returns the norm, handling the zero vector."""
    norm: jax.Array = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(norm, 1e-8)
    normalized_x = x / safe_norm
    result_vec = jnp.where(norm < 1e-8, jnp.zeros_like(x), normalized_x)
    result_norm = norm[..., 0]
    return result_vec, result_norm


def closest_segment_point(
    a: Float[Array, "*batch 3"],
    b: Float[Array, "*batch 3"],
    pt: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3"]:
    """Finds the closest point on the line segment [a, b] to point pt."""
    ab = b - a
    t = jnp.einsum("...i,...i->...", pt - a, ab) / (
        jnp.einsum("...i,...i->...", ab, ab) + _SAFE_EPS
    )
    t_clamped = jnp.clip(t, 0.0, 1.0)
    return a + ab * t_clamped[..., None]


def closest_segment_to_segment_points(
    a1: Float[Array, "*batch 3"],
    b1: Float[Array, "*batch 3"],
    a2: Float[Array, "*batch 3"],
    b2: Float[Array, "*batch 3"],
) -> Tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
    """Finds the closest points between two line segments [a1, b1] and [a2, b2]."""
    d1 = b1 - a1  # Direction vector of segment S1
    d2 = b2 - a2  # Direction vector of segment S2
    r = a1 - a2

    a = jnp.einsum("...i,...i->...", d1, d1)  # Squared length of segment S1
    e = jnp.einsum("...i,...i->...", d2, d2)  # Squared length of segment S2
    f = jnp.einsum("...i,...i->...", d2, r)
    c = jnp.einsum("...i,...i->...", d1, r)
    b = jnp.einsum("...i,...i->...", d1, d2)
    denom = a * e - b * b  # Squared area of the parallelogram defined by d1, d2

    s_num = b * f - c * e
    t_num = a * f - b * c

    s_parallel = -c / (a + _SAFE_EPS)
    t_parallel = f / (e + _SAFE_EPS)

    s = jnp.where(denom < _SAFE_EPS, s_parallel, s_num / (denom + _SAFE_EPS))
    t = jnp.where(denom < _SAFE_EPS, t_parallel, t_num / (denom + _SAFE_EPS))

    s_clamped = jnp.clip(s, 0.0, 1.0)
    t_clamped = jnp.clip(t, 0.0, 1.0)
    s_was_clamped = jnp.abs(s - s_clamped) > _SAFE_EPS
    t_was_clamped = jnp.abs(t - t_clamped) > _SAFE_EPS

    t_recomp = jnp.einsum(
        "...i,...i->...", d2, (a1 + d1 * s_clamped[..., None]) - a2
    ) / (e + _SAFE_EPS)
    t_recomp_clamped = jnp.clip(t_recomp, 0.0, 1.0)
    t_recomp_was_clamped = jnp.abs(t_recomp - t_recomp_clamped) > _SAFE_EPS
    t_final = jnp.where(s_was_clamped, t_recomp_clamped, t_clamped)

    s_recomp = jnp.einsum("...i,...i->...", d1, (a2 + d2 * t_final[..., None]) - a1) / (
        a + _SAFE_EPS
    )
    # Only recompute s if: (s was clamped AND t_recomp was clamped) OR (s wasn't clamped AND t was clamped)
    need_s_recomp = jnp.where(s_was_clamped, t_recomp_was_clamped, t_was_clamped)
    s_final = jnp.where(need_s_recomp, jnp.clip(s_recomp, 0.0, 1.0), s_clamped)

    c1 = a1 + d1 * s_final[..., None]
    c2 = a2 + d2 * t_final[..., None]
    return c1, c2


def soft_clamp(
    x: Float[Array, "*batch"],
    low: float,
    high: float,
    sharpness: float = 20.0,
) -> Float[Array, "*batch"]:
    """Smooth clamp with continuous gradients everywhere.

    Uses logaddexp for numerical stability. As sharpness → ∞, approaches hard clamp.
    With sharpness=20, geometry error is ~2% at boundaries.

    Args:
        x: Input values to clamp.
        low: Lower bound.
        high: Upper bound.
        sharpness: Controls smoothness. Higher = closer to hard clamp.

    Returns:
        Smoothly clamped values.
    """

    def soft_max(a: jax.Array, b: float) -> jax.Array:
        return jnp.logaddexp(a * sharpness, b * sharpness) / sharpness

    def soft_min(a: jax.Array, b: float) -> jax.Array:
        return -soft_max(-a, -b)

    return soft_min(soft_max(x, low), high)


def straight_through_clamp(
    x: Float[Array, "*batch"],
    low: float = 0.0,
    high: float = 1.0,
    sharpness: float = 20.0,
) -> Float[Array, "*batch"]:
    """Clamp with exact forward pass but smooth backward pass.

    Uses straight-through estimator: exact geometry in forward pass,
    smooth gradients from soft_clamp in backward pass.

    Args:
        x: Input values to clamp.
        low: Lower bound.
        high: Upper bound.
        sharpness: Controls gradient smoothness.

    Returns:
        Hard-clamped values with smooth gradients.
    """
    x_exact = jnp.clip(x, low, high)
    x_smooth = soft_clamp(x, low, high, sharpness)
    # Straight-through: use exact value, but gradient flows through smooth version
    return x_exact + jax.lax.stop_gradient(x_exact - x_smooth)


def closest_segment_to_segment_with_jac(
    a1: Float[Array, "*batch 3"],
    b1: Float[Array, "*batch 3"],
    a2: Float[Array, "*batch 3"],
    b2: Float[Array, "*batch 3"],
    sharpness: float = 20.0,
) -> Tuple[
    Float[Array, "*batch 3"],  # c1: closest point on segment 1
    Float[Array, "*batch 3"],  # c2: closest point on segment 2
    Float[Array, "*batch"],  # s: parametric position on segment 1
    Float[Array, "*batch"],  # t: parametric position on segment 2
    Float[Array, "*batch"],  # distance: ||c2 - c1||
    Float[Array, "*batch 3"],  # direction: normalized (c2 - c1)
    Float[Array, "*batch 3"],  # d_dist_d_a1: gradient of distance w.r.t. a1
    Float[Array, "*batch 3"],  # d_dist_d_b1: gradient of distance w.r.t. b1
    Float[Array, "*batch 3"],  # d_dist_d_a2: gradient of distance w.r.t. a2
    Float[Array, "*batch 3"],  # d_dist_d_b2: gradient of distance w.r.t. b2
]:
    """Finds closest points between two segments with analytical Jacobians.

    Uses straight-through estimator for stable gradients: exact geometry in
    forward pass, smooth gradients in backward pass.

    The distance gradient is computed via chain rule:
        d_dist/d_endpoint = d_dist/d_c × d_c/d_endpoint

    Where d_c/d_endpoint depends on whether the parametric value (s or t)
    is clamped (at endpoint) or free (in interior).

    Args:
        a1, b1: Endpoints of segment 1.
        a2, b2: Endpoints of segment 2.
        sharpness: Controls gradient smoothness (higher = sharper transitions).

    Returns:
        Tuple of (c1, c2, s, t, distance, direction, d_dist_d_a1, d_dist_d_b1,
                  d_dist_d_a2, d_dist_d_b2).
    """
    d1 = b1 - a1  # Direction vector of segment S1
    d2 = b2 - a2  # Direction vector of segment S2
    r = a1 - a2

    a = jnp.einsum("...i,...i->...", d1, d1)  # Squared length of segment S1
    e = jnp.einsum("...i,...i->...", d2, d2)  # Squared length of segment S2
    f = jnp.einsum("...i,...i->...", d2, r)
    c = jnp.einsum("...i,...i->...", d1, r)
    b_coef = jnp.einsum("...i,...i->...", d1, d2)
    denom = a * e - b_coef * b_coef

    # Unclamped parametric values
    s_num = b_coef * f - c * e
    t_num = a * f - b_coef * c

    s_parallel = -c / (a + _SAFE_EPS)
    t_parallel = f / (e + _SAFE_EPS)

    s_unclamped = jnp.where(denom < _SAFE_EPS, s_parallel, s_num / (denom + _SAFE_EPS))
    t_unclamped = jnp.where(denom < _SAFE_EPS, t_parallel, t_num / (denom + _SAFE_EPS))

    # Use straight-through clamp for stable gradients
    s_clamped = straight_through_clamp(s_unclamped, 0.0, 1.0, sharpness)
    t_clamped = straight_through_clamp(t_unclamped, 0.0, 1.0, sharpness)
    s_was_clamped = jnp.abs(s_unclamped - s_clamped) > _SAFE_EPS
    t_was_clamped = jnp.abs(t_unclamped - t_clamped) > _SAFE_EPS

    # Recomputation when s is clamped
    t_recomp = jnp.einsum(
        "...i,...i->...", d2, (a1 + d1 * s_clamped[..., None]) - a2
    ) / (e + _SAFE_EPS)
    t_recomp_clamped = straight_through_clamp(t_recomp, 0.0, 1.0, sharpness)
    t_recomp_was_clamped = jnp.abs(t_recomp - t_recomp_clamped) > _SAFE_EPS
    t_final = jnp.where(s_was_clamped, t_recomp_clamped, t_clamped)

    # Recomputation when t is clamped
    s_recomp = jnp.einsum(
        "...i,...i->...", d1, (a2 + d2 * t_final[..., None]) - a1
    ) / (a + _SAFE_EPS)
    # Only recompute s if: (s was clamped AND t_recomp was clamped) OR (s wasn't clamped AND t was clamped)
    need_s_recomp = jnp.where(s_was_clamped, t_recomp_was_clamped, t_was_clamped)
    s_final = jnp.where(
        need_s_recomp,
        straight_through_clamp(s_recomp, 0.0, 1.0, sharpness),
        s_clamped,
    )

    # Closest points
    c1 = a1 + d1 * s_final[..., None]
    c2 = a2 + d2 * t_final[..., None]

    # Distance and direction
    diff = c2 - c1
    direction, distance = normalize_with_norm(diff)

    # --- Compute analytical Jacobians ---
    # We need: d_dist/d_a1, d_dist/d_b1, d_dist/d_a2, d_dist/d_b2
    #
    # Key insight: distance = ||c2 - c1||
    # d_dist/d_endpoint = direction · d(c2-c1)/d_endpoint
    #
    # c1 = a1 + s * (b1 - a1) = a1 * (1 - s) + b1 * s
    # c2 = a2 + t * (b2 - a2) = a2 * (1 - t) + b2 * t
    #
    # For the gradient, we use the smooth s/t values from soft_clamp.
    # The straight-through estimator ensures gradients flow through soft_clamp.

    # Get the "soft" parametric values for gradient computation
    # These have smooth gradients unlike the hard-clamped values
    s_soft = soft_clamp(s_unclamped, 0.0, 1.0, sharpness)
    t_soft_initial = soft_clamp(t_unclamped, 0.0, 1.0, sharpness)

    # Handle recomputation cases with soft clamp
    t_recomp_soft = jnp.einsum(
        "...i,...i->...", d2, (a1 + d1 * s_soft[..., None]) - a2
    ) / (e + _SAFE_EPS)
    t_soft = jnp.where(
        s_was_clamped,
        soft_clamp(t_recomp_soft, 0.0, 1.0, sharpness),
        t_soft_initial,
    )

    s_recomp_soft = jnp.einsum(
        "...i,...i->...", d1, (a2 + d2 * t_soft[..., None]) - a1
    ) / (a + _SAFE_EPS)
    s_soft_final = jnp.where(
        t_was_clamped,
        soft_clamp(s_recomp_soft, 0.0, 1.0, sharpness),
        s_soft,
    )
    t_soft_final = t_soft

    # Gradient of distance w.r.t. closest points:
    # d_dist/d_c1 = -direction
    # d_dist/d_c2 = +direction

    # Gradient of c1 w.r.t. endpoints (treating s as constant for now):
    # d_c1/d_a1 = (1 - s) * I
    # d_c1/d_b1 = s * I
    # d_c1/d_a2 = 0, d_c1/d_b2 = 0

    # Gradient of c2 w.r.t. endpoints (treating t as constant):
    # d_c2/d_a2 = (1 - t) * I
    # d_c2/d_b2 = t * I
    # d_c2/d_a1 = 0, d_c2/d_b1 = 0

    # Combined: d_dist/d_a1 = direction · d(c2-c1)/d_a1 = direction · (-d_c1/d_a1)
    #         = -direction · (1 - s) * I = -(1 - s) * direction
    # Similarly for others.

    # Use soft parametric values for smooth gradients
    one_minus_s = 1.0 - s_soft_final
    one_minus_t = 1.0 - t_soft_final

    # d_dist/d_a1 = -direction * (1 - s)  (c1 moves with a1, decreases distance if moving toward c2)
    d_dist_d_a1 = -direction * one_minus_s[..., None]

    # d_dist/d_b1 = -direction * s
    d_dist_d_b1 = -direction * s_soft_final[..., None]

    # d_dist/d_a2 = +direction * (1 - t)  (c2 moves with a2, increases distance if moving away from c1)
    d_dist_d_a2 = direction * one_minus_t[..., None]

    # d_dist/d_b2 = +direction * t
    d_dist_d_b2 = direction * t_soft_final[..., None]

    return (
        c1,
        c2,
        s_final,
        t_final,
        distance,
        direction,
        d_dist_d_a1,
        d_dist_d_b1,
        d_dist_d_a2,
        d_dist_d_b2,
    )
