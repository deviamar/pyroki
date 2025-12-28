from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxtyping import Array, Bool, Float, Int
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot

from ._collision import collide
from ._geometry import Sphere


def _import_ballpark():
    try:
        import ballpark  # noqa: PLC0415

        return ballpark
    except ImportError as e:
        raise ImportError(
            "ballpark is required for sphere-based collision. "
            "Install with: pip install 'pyroki[ballpark]' or "
            "pip install git+https://github.com/chungmin99/ballpark.git"
        ) from e


@jdc.pytree_dataclass
class RobotSphereCollision:
    """Sphere-based collision model for a robot, using ballpark decomposition."""

    num_links: jdc.Static[int]
    """Number of links in the model."""

    max_spheres_per_link: jdc.Static[int]
    """Maximum number of spheres for any single link (used for padding)."""

    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""

    spheres: Sphere
    """Sphere geometries in link-local frames. Shape: (num_links, max_spheres_per_link)."""

    sphere_counts: Int[Array, " num_links"]
    """Actual number of valid spheres per link (before padding)."""

    active_idx_i: Int[Array, " P"]
    """Row indices (first link) of active self-collision pairs."""

    active_idx_j: Int[Array, " P"]
    """Column indices (second link) of active self-collision pairs."""

    # Flat sphere-pair indices for efficient self-collision (no S×S expansion)
    sphere_pair_link_i: jdc.Static[tuple[int, ...]]
    """Link index for the first sphere in each sphere-sphere pair."""

    sphere_pair_idx_i: jdc.Static[tuple[int, ...]]
    """Sphere index within link for the first sphere in each pair."""

    sphere_pair_link_j: jdc.Static[tuple[int, ...]]
    """Link index for the second sphere in each sphere-sphere pair."""

    sphere_pair_idx_j: jdc.Static[tuple[int, ...]]
    """Sphere index within link for the second sphere in each pair."""

    @staticmethod
    def from_ballpark_result(
        robot: "Robot",
        link_spheres: dict,
        ignore_pairs: Sequence[tuple[str, str]] | None = None,
        ignore_adjacent: bool = True,
    ) -> "RobotSphereCollision":
        """Create a RobotSphereCollision from ballpark output.

        Args:
            robot: The Robot instance (used for link names and adjacency).
            link_spheres: Dictionary mapping link names to lists of ballpark.Sphere objects.
            ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_adjacent: If True, also ignore adjacent (parent-child) links.

        Returns:
            A RobotSphereCollision instance.
        """
        link_names = robot.links.names
        num_links = len(link_names)

        max_spheres = max(
            (len(link_spheres.get(name, [])) for name in link_names),
            default=1,
        )
        max_spheres = max(max_spheres, 1)

        all_centers = []
        all_radii = []
        sphere_counts = []

        for name in link_names:
            spheres_for_link = link_spheres.get(name, [])
            n = len(spheres_for_link)
            sphere_counts.append(n)

            link_centers = [jnp.array(s.center) for s in spheres_for_link]
            link_radii = [float(s.radius) for s in spheres_for_link]

            while len(link_centers) < max_spheres:
                link_centers.append(jnp.zeros(3))
                link_radii.append(0.0)

            all_centers.append(jnp.stack(link_centers[:max_spheres]))
            all_radii.append(jnp.array(link_radii[:max_spheres]))

        centers_array = jnp.stack(all_centers)
        radii_array = jnp.stack(all_radii)
        counts_array = jnp.array(sphere_counts)

        spheres = Sphere.from_center_and_radius(centers_array, radii_array)

        active_idx_i, active_idx_j = RobotSphereCollision._compute_active_pairs(
            link_names=link_names,
            robot=robot,
            user_ignore_pairs=tuple(ignore_pairs) if ignore_pairs else (),
            ignore_adjacent=ignore_adjacent,
        )

        # Compute flat sphere-pair indices from link pairs and sphere counts
        sphere_pair_link_i = []
        sphere_pair_idx_i = []
        sphere_pair_link_j = []
        sphere_pair_idx_j = []

        for link_i, link_j in zip(active_idx_i.tolist(), active_idx_j.tolist()):
            count_i = sphere_counts[link_i]
            count_j = sphere_counts[link_j]
            for si in range(count_i):
                for sj in range(count_j):
                    sphere_pair_link_i.append(link_i)
                    sphere_pair_idx_i.append(si)
                    sphere_pair_link_j.append(link_j)
                    sphere_pair_idx_j.append(sj)

        total_spheres = sum(sphere_counts)
        num_link_pairs = len(active_idx_i)
        num_sphere_pairs = len(sphere_pair_link_i)
        logger.info(
            f"Created RobotSphereCollision with {num_links} links, "
            f"{total_spheres} spheres (max {max_spheres}/link), "
            f"{num_link_pairs} active link pairs, and {num_sphere_pairs} sphere pairs."
        )

        return RobotSphereCollision(
            num_links=num_links,
            max_spheres_per_link=max_spheres,
            link_names=link_names,
            spheres=spheres,
            sphere_counts=counts_array,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            sphere_pair_link_i=tuple(sphere_pair_link_i),
            sphere_pair_idx_i=tuple(sphere_pair_idx_i),
            sphere_pair_link_j=tuple(sphere_pair_link_j),
            sphere_pair_idx_j=tuple(sphere_pair_idx_j),
        )

    @staticmethod
    def from_urdf_with_ballpark(
        urdf,
        robot: "Robot",
        total_spheres: int = 100,
        preset: str = "balanced",
        refine_self_collision: bool = True,
        ignore_adjacent: bool = True,
        **ballpark_kwargs,
    ) -> "RobotSphereCollision":
        """Create a RobotSphereCollision by running ballpark on a URDF.

        Args:
            urdf: The URDF object (yourdfpy.URDF).
            robot: The pyroki Robot instance.
            total_spheres: Total sphere budget across all links.
            preset: Ballpark preset ("conservative", "balanced", "surface").
            refine_self_collision: Whether to run self-collision refinement in ballpark.
            ignore_adjacent: Whether to ignore adjacent links in self-collision.
            **ballpark_kwargs: Additional arguments passed to ballpark.

        Returns:
            A RobotSphereCollision instance.
        """
        bp = _import_ballpark()

        result = bp.compute_spheres_for_robot(
            urdf=urdf,
            total_spheres=total_spheres,
            preset=preset,
            robot=robot if refine_self_collision else None,
            refine_self_collision=refine_self_collision,
            **ballpark_kwargs,
        )

        return RobotSphereCollision.from_ballpark_result(
            robot=robot,
            link_spheres=result.link_spheres,
            ignore_pairs=result.ignore_pairs,
            ignore_adjacent=ignore_adjacent,
        )

    @staticmethod
    def _compute_active_pairs(
        link_names: tuple[str, ...],
        robot: "Robot",
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_adjacent: bool,
    ) -> tuple[Int[Array, " P"], Int[Array, " P"]]:
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}

        ignore_matrix = jnp.eye(num_links, dtype=bool)

        if ignore_adjacent:
            parent_indices = robot.links.parent_joint_indices
            for child_idx in range(num_links):
                parent_joint_idx = int(parent_indices[child_idx])
                if parent_joint_idx >= 0 and parent_joint_idx < num_links:
                    ignore_matrix = ignore_matrix.at[child_idx, parent_joint_idx].set(
                        True
                    )
                    ignore_matrix = ignore_matrix.at[parent_joint_idx, child_idx].set(
                        True
                    )

        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return active_i, active_j

    @jdc.jit
    def at_config(
        self, robot: "Robot", cfg: Float[Array, "*batch actuated_count"]
    ) -> Sphere:
        """Transform spheres to world frame at the given configuration."""
        assert self.link_names == robot.links.names, (
            "Link name mismatch between RobotSphereCollision and Robot."
        )

        batch_axes = cfg.shape[:-1]
        Ts_world_link_wxyz_xyz = robot.forward_kinematics(cfg)
        Ts_world_link = jaxlie.SE3(Ts_world_link_wxyz_xyz)

        Ts_broadcast = jaxlie.SE3(
            jnp.broadcast_to(
                Ts_world_link.wxyz_xyz[..., None, :],
                (*batch_axes, self.num_links, self.max_spheres_per_link, 7),
            )
        )

        return self.spheres.transform(Ts_broadcast)

    def _get_sphere_valid_mask(self) -> Bool[Array, "num_links max_spheres_per_link"]:
        sphere_indices = jnp.arange(self.max_spheres_per_link)
        return sphere_indices[None, :] < self.sphere_counts[:, None]

    @jdc.jit
    def compute_self_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """Compute minimum sphere-to-sphere distance for each active link pair."""
        batch_axes = cfg.shape[:-1]
        num_pairs = len(self.active_idx_i)

        if num_pairs == 0:
            return jnp.zeros((*batch_axes, 0))

        spheres_world = self.at_config(robot, cfg)
        valid_mask = self._get_sphere_valid_mask()

        positions = spheres_world.pose.translation()
        radii = spheres_world.radius

        active_i = self.active_idx_i
        active_j = self.active_idx_j

        def compute_pair_distance(pair_idx: Array) -> Float[Array, "*batch"]:
            link_i = active_i[pair_idx]
            link_j = active_j[pair_idx]

            pos_i = positions[..., link_i, :, :]
            pos_j = positions[..., link_j, :, :]
            rad_i = radii[..., link_i, :]
            rad_j = radii[..., link_j, :]

            pos_i_exp = pos_i[..., :, None, :]
            pos_j_exp = pos_j[..., None, :, :]
            rad_i_exp = rad_i[..., :, None]
            rad_j_exp = rad_j[..., None, :]

            diff = pos_i_exp - pos_j_exp
            center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)
            distances = center_dist - rad_i_exp - rad_j_exp

            valid_i = valid_mask[link_i]
            valid_j = valid_mask[link_j]
            pair_valid = valid_i[:, None] & valid_j[None, :]

            masked_distances = jnp.asarray(jnp.where(pair_valid, distances, jnp.inf))
            return masked_distances.min()

        all_pair_distances = jax.vmap(compute_pair_distance)(jnp.arange(num_pairs))
        return jnp.moveaxis(all_pair_distances, 0, -1)

    @jdc.jit
    def compute_all_self_collision_distances(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> tuple[
        Float[Array, "*batch num_active_pairs max_spheres max_spheres"],
        Bool[Array, "num_active_pairs max_spheres max_spheres"],
    ]:
        """Compute all sphere-to-sphere distances for active link pairs.

        Returns all pairwise distances (not just minimum), with validity mask.
        Invalid pairs (padding spheres) are set to inf in distances.
        """
        batch_axes = cfg.shape[:-1]
        num_pairs = len(self.active_idx_i)
        S = self.max_spheres_per_link

        if num_pairs == 0:
            empty_dist: Float[Array, "*batch 0 S S"] = jnp.zeros((*batch_axes, 0, S, S))
            empty_mask: Bool[Array, "0 S S"] = jnp.zeros((0, S, S), dtype=bool)
            return empty_dist, empty_mask

        spheres_world = self.at_config(robot, cfg)
        sphere_valid_mask = self._get_sphere_valid_mask()

        positions = spheres_world.pose.translation()
        radii = spheres_world.radius

        pos_i = positions[..., self.active_idx_i, :, :]
        pos_j = positions[..., self.active_idx_j, :, :]
        rad_i = radii[..., self.active_idx_i, :]
        rad_j = radii[..., self.active_idx_j, :]

        pos_i_exp = pos_i[..., :, None, :]
        pos_j_exp = pos_j[..., None, :, :]
        rad_i_exp = rad_i[..., :, None]
        rad_j_exp = rad_j[..., None, :]

        diff = pos_i_exp - pos_j_exp
        center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)
        distances = center_dist - rad_i_exp - rad_j_exp

        valid_i = sphere_valid_mask[self.active_idx_i]
        valid_j = sphere_valid_mask[self.active_idx_j]
        pair_valid_mask = valid_i[:, :, None] & valid_j[:, None, :]

        distances_masked = jnp.asarray(jnp.where(pair_valid_mask, distances, jnp.inf))

        return distances_masked, pair_valid_mask

    @jdc.jit
    def compute_self_collision_distances_flat(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> tuple[
        Float[Array, "*batch num_sphere_pairs"],
        Float[Array, "*batch num_sphere_pairs 3"],
    ]:
        """Compute sphere-to-sphere distances for all valid sphere pairs (flat output).

        Uses precomputed flat sphere-pair indices instead of (P, S, S) expansion.
        Only computes distances for valid sphere pairs (no padding).

        Returns:
            distances: Signed distances for each sphere pair, shape (*batch, num_sphere_pairs).
            directions: Unit direction from sphere i to sphere j, shape (*batch, num_sphere_pairs, 3).
        """
        num_sphere_pairs = len(self.sphere_pair_link_i)

        if num_sphere_pairs == 0:
            batch_axes = cfg.shape[:-1]
            return jnp.zeros((*batch_axes, 0)), jnp.zeros((*batch_axes, 0, 3))

        spheres_world = self.at_config(robot, cfg)
        positions = spheres_world.pose.translation()  # (*batch, num_links, S, 3)
        radii = spheres_world.radius  # (*batch, num_links, S)

        # Convert to arrays for indexing
        link_i = jnp.array(self.sphere_pair_link_i)
        idx_i = jnp.array(self.sphere_pair_idx_i)
        link_j = jnp.array(self.sphere_pair_link_j)
        idx_j = jnp.array(self.sphere_pair_idx_j)

        # Direct flat indexing - no S×S expansion
        pos_i = positions[..., link_i, idx_i, :]  # (*batch, num_sphere_pairs, 3)
        pos_j = positions[..., link_j, idx_j, :]  # (*batch, num_sphere_pairs, 3)
        rad_i = radii[..., link_i, idx_i]  # (*batch, num_sphere_pairs)
        rad_j = radii[..., link_j, idx_j]  # (*batch, num_sphere_pairs)

        # Compute distances and directions
        diff = pos_j - pos_i  # (*batch, num_sphere_pairs, 3)
        center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)  # (*batch, num_sphere_pairs)
        directions = diff / (center_dist[..., None] + 1e-8)  # (*batch, num_sphere_pairs, 3)
        distances = center_dist - rad_i - rad_j  # (*batch, num_sphere_pairs)

        return distances, directions

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom,
    ) -> Float[Array, "*batch_combined num_links M"]:
        """Compute minimum distance between robot spheres and world obstacles."""
        batch_cfg = cfg.shape[:-1]

        spheres_world = self.at_config(robot, cfg)
        valid_mask = self._get_sphere_valid_mask()

        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:
            _world_geom = world_geom.broadcast_to((1,))
            num_world_geoms = 1
            batch_world: tuple[int, ...] = ()
        else:
            _world_geom = world_geom
            num_world_geoms = world_axes[-1]
            batch_world = world_axes[:-1]

        batch_combined = jnp.broadcast_shapes(batch_cfg, batch_world)

        total_spheres = self.num_links * self.max_spheres_per_link
        flat_spheres = spheres_world.reshape((*batch_cfg, total_spheres))

        _collide_vmap = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)
        dist_flat = _collide_vmap(
            flat_spheres.broadcast_to((*batch_combined, total_spheres)),
            _world_geom.broadcast_to((*batch_combined, num_world_geoms)),
        )

        dist_reshaped = dist_flat.reshape(
            *batch_combined, self.num_links, self.max_spheres_per_link, num_world_geoms
        )

        valid_expanded = valid_mask[..., None]
        dist_masked = jnp.where(valid_expanded, dist_reshaped, jnp.inf)

        return dist_masked.min(axis=-2)

    @jdc.jit
    def compute_all_world_collision_distances(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_spheres: Sphere,
    ) -> tuple[
        Float[Array, "*batch_combined num_links max_spheres M"],
        Bool[Array, "num_links max_spheres"],
    ]:
        """Compute all sphere-to-sphere distances between robot and world spheres.

        Returns all pairwise distances (not just minimum per link), with validity mask.
        Invalid robot spheres (padding) are set to inf in distances.
        """
        batch_cfg = cfg.shape[:-1]

        spheres_world = self.at_config(robot, cfg)
        sphere_valid_mask = self._get_sphere_valid_mask()

        world_axes = world_spheres.get_batch_axes()
        if len(world_axes) == 0:
            _world_spheres = world_spheres.broadcast_to((1,))
            num_world = 1
            batch_world: tuple[int, ...] = ()
        else:
            _world_spheres = world_spheres
            num_world = world_axes[-1]
            batch_world = world_axes[:-1]

        batch_combined = jnp.broadcast_shapes(batch_cfg, batch_world)

        robot_pos = spheres_world.pose.translation()
        robot_rad = spheres_world.radius
        world_pos = _world_spheres.pose.translation()
        world_rad = _world_spheres.radius

        robot_pos = jnp.broadcast_to(
            robot_pos, (*batch_combined, self.num_links, self.max_spheres_per_link, 3)
        )
        robot_rad = jnp.broadcast_to(
            robot_rad, (*batch_combined, self.num_links, self.max_spheres_per_link)
        )
        world_pos = jnp.broadcast_to(world_pos, (*batch_combined, num_world, 3))
        world_rad = jnp.broadcast_to(world_rad, (*batch_combined, num_world))

        robot_pos_exp = robot_pos[..., :, :, None, :]
        world_pos_exp = world_pos[..., None, None, :, :]
        robot_rad_exp = robot_rad[..., :, :, None]
        world_rad_exp = world_rad[..., None, None, :]

        diff = robot_pos_exp - world_pos_exp
        center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)
        distances = center_dist - robot_rad_exp - world_rad_exp

        distances_masked = jnp.asarray(
            jnp.where(sphere_valid_mask[:, :, None], distances, jnp.inf)
        )

        return distances_masked, sphere_valid_mask
