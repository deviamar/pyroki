from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import trimesh
import yourdfpy
from jaxtyping import Array, Bool, Float, Int

from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot

from .._robot_urdf_parser import LinkInfo, RobotURDFParser
from ._collision import collide
from ._geometry import Capsule, CollGeom, Sphere


@jdc.pytree_dataclass
class RobotCollision:
    """Unified collision model for a robot, supporting both capsule and sphere geometries.

    This class handles collision detection for robots using either:
    - Single geometry per link (e.g., capsules from URDF collision meshes)
    - Multiple geometries per link (e.g., sphere decomposition from ballpark)

    The geometry-pair indices (`geom_pair_*`) provide efficient flat indexing for
    both self-collision computation and analytical Jacobian calculations.
    """

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""

    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""

    coll: CollGeom
    """Collision geometries for the robot. Shape: (num_links, max_geoms_per_link)."""

    max_geoms_per_link: jdc.Static[int]
    """Maximum number of geometries per link. 1 for capsules, N for spheres."""

    geom_counts: Int[Array, " num_links"]
    """Actual number of valid geometries per link (before padding)."""

    # Flat geometry-pair indices for efficient self-collision computation
    geom_pair_link_i: jdc.Static[tuple[int, ...]]
    """Link index for the first geometry in each geometry-geometry pair."""

    geom_pair_idx_i: jdc.Static[tuple[int, ...]]
    """Geometry index within link for the first geometry in each pair."""

    geom_pair_link_j: jdc.Static[tuple[int, ...]]
    """Link index for the second geometry in each geometry-geometry pair."""

    geom_pair_idx_j: jdc.Static[tuple[int, ...]]
    """Geometry index within link for the second geometry in each pair."""

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        sphere_decomposition: Optional[dict[str, list[dict]]] = None,
        ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_adjacent: bool = True,
    ) -> "RobotCollision":
        """Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes and link structure).
            sphere_decomposition: Optional dictionary mapping link names to lists of
                sphere definitions. Each sphere is a dict with 'center' (list of 3 floats)
                and 'radius' (float). If provided, uses sphere-based collision instead
                of capsule approximation from URDF meshes.
            ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_adjacent: If True, automatically ignore collisions between adjacent
                (parent/child) links based on the URDF structure.

        Returns:
            A RobotCollision instance.
        """
        # Parse link info from URDF
        _, link_info = RobotURDFParser.parse(urdf)

        if sphere_decomposition is not None:
            return RobotCollision._from_urdf_with_spheres(
                urdf=urdf,
                link_info=link_info,
                sphere_decomposition=sphere_decomposition,
                ignore_pairs=ignore_pairs,
                ignore_adjacent=ignore_adjacent,
            )
        else:
            return RobotCollision._from_urdf_with_capsules(
                urdf=urdf,
                link_info=link_info,
                ignore_pairs=ignore_pairs,
                ignore_adjacent=ignore_adjacent,
            )

    @staticmethod
    def _from_urdf_with_capsules(
        urdf: yourdfpy.URDF,
        link_info: LinkInfo,
        ignore_pairs: tuple[tuple[str, str], ...],
        ignore_adjacent: bool,
    ) -> "RobotCollision":
        """Build collision model using capsule approximation from URDF meshes."""
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        link_names = link_info.names

        # Gather all collision meshes and create capsules.
        cap_list = list[Capsule]()
        for link_name in link_names:
            cap_list.append(
                Capsule.from_trimesh(
                    RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
                )
            )

        # Stack capsules: shape (num_links,)
        capsules_flat = cast(
            Capsule, jax.tree.map(lambda *args: jnp.stack(args), *cap_list)
        )
        assert capsules_flat.get_batch_axes() == (link_info.num_links,)

        # Add the max_geoms_per_link=1 dimension
        # For pose.wxyz_xyz: (num_links, 7) -> (num_links, 1, 7)
        # For size: (num_links, 2) -> (num_links, 1, 2)
        num_batch_dims = len(capsules_flat.get_batch_axes())
        capsules = cast(
            Capsule,
            jax.tree.map(
                lambda x: jnp.expand_dims(x, axis=num_batch_dims), capsules_flat
            ),
        )
        assert capsules.get_batch_axes() == (link_info.num_links, 1)

        # All links have exactly 1 geometry
        geom_counts = jnp.ones(link_info.num_links, dtype=jnp.int32)

        # Compute active link pairs
        active_link_pairs = RobotCollision._compute_active_link_pairs(
            link_info=link_info,
            ignore_pairs=ignore_pairs,
            ignore_adjacent=ignore_adjacent,
        )

        # For single-geom-per-link, geometry pairs = link pairs with idx=0
        geom_pair_link_i = tuple(p[0] for p in active_link_pairs)
        geom_pair_idx_i = tuple(0 for _ in active_link_pairs)
        geom_pair_link_j = tuple(p[1] for p in active_link_pairs)
        geom_pair_idx_j = tuple(0 for _ in active_link_pairs)

        logger.info(
            f"Created RobotCollision (capsules) with {link_info.num_links} links and "
            f"{len(active_link_pairs)} active self-collision pairs."
        )

        return RobotCollision(
            num_links=link_info.num_links,
            link_names=link_names,
            coll=capsules,
            max_geoms_per_link=1,
            geom_counts=geom_counts,
            geom_pair_link_i=geom_pair_link_i,
            geom_pair_idx_i=geom_pair_idx_i,
            geom_pair_link_j=geom_pair_link_j,
            geom_pair_idx_j=geom_pair_idx_j,
        )

    @staticmethod
    def _from_urdf_with_spheres(
        urdf: yourdfpy.URDF,
        link_info: LinkInfo,
        sphere_decomposition: dict[str, list[dict]],
        ignore_pairs: tuple[tuple[str, str], ...],
        ignore_adjacent: bool,
    ) -> "RobotCollision":
        """Build collision model using sphere decomposition."""
        link_names = link_info.names
        num_links = link_info.num_links

        # Determine max spheres per link
        max_spheres = max(
            (len(sphere_decomposition.get(name, [])) for name in link_names),
            default=1,
        )
        max_spheres = max(max_spheres, 1)

        # Build sphere arrays with padding
        all_centers = []
        all_radii = []
        sphere_counts = []

        for name in link_names:
            spheres_for_link = sphere_decomposition.get(name, [])
            n = len(spheres_for_link)
            sphere_counts.append(n)

            link_centers = [jnp.array(s["center"]) for s in spheres_for_link]
            link_radii = [float(s["radius"]) for s in spheres_for_link]

            # Pad to max_spheres
            while len(link_centers) < max_spheres:
                link_centers.append(jnp.zeros(3))
                link_radii.append(0.0)

            all_centers.append(jnp.stack(link_centers[:max_spheres]))
            all_radii.append(jnp.array(link_radii[:max_spheres]))

        centers_array = jnp.stack(all_centers)  # (num_links, max_spheres, 3)
        radii_array = jnp.stack(all_radii)  # (num_links, max_spheres)
        counts_array = jnp.array(sphere_counts, dtype=jnp.int32)

        spheres = Sphere.from_center_and_radius(centers_array, radii_array)
        assert spheres.get_batch_axes() == (num_links, max_spheres)

        # Compute active link pairs
        active_link_pairs = RobotCollision._compute_active_link_pairs(
            link_info=link_info,
            ignore_pairs=ignore_pairs,
            ignore_adjacent=ignore_adjacent,
        )

        # Expand link pairs to geometry pairs
        geom_pair_link_i = []
        geom_pair_idx_i = []
        geom_pair_link_j = []
        geom_pair_idx_j = []

        for link_i, link_j in active_link_pairs:
            count_i = sphere_counts[link_i]
            count_j = sphere_counts[link_j]
            for si in range(count_i):
                for sj in range(count_j):
                    geom_pair_link_i.append(link_i)
                    geom_pair_idx_i.append(si)
                    geom_pair_link_j.append(link_j)
                    geom_pair_idx_j.append(sj)

        total_spheres = sum(sphere_counts)
        num_link_pairs = len(active_link_pairs)
        num_geom_pairs = len(geom_pair_link_i)

        logger.info(
            f"Created RobotCollision (spheres) with {num_links} links, "
            f"{total_spheres} spheres (max {max_spheres}/link), "
            f"{num_link_pairs} active link pairs, and {num_geom_pairs} geometry pairs."
        )

        return RobotCollision(
            num_links=num_links,
            link_names=link_names,
            coll=spheres,
            max_geoms_per_link=max_spheres,
            geom_counts=counts_array,
            geom_pair_link_i=tuple(geom_pair_link_i),
            geom_pair_idx_i=tuple(geom_pair_idx_i),
            geom_pair_link_j=tuple(geom_pair_link_j),
            geom_pair_idx_j=tuple(geom_pair_idx_j),
        )

    @staticmethod
    def _compute_active_link_pairs(
        link_info: LinkInfo,
        ignore_pairs: tuple[tuple[str, str], ...],
        ignore_adjacent: bool,
    ) -> list[tuple[int, int]]:
        """Compute list of (link_i, link_j) pairs to check for self-collision.

        Returns pairs where link_i < link_j.
        """
        link_names = link_info.names
        num_links = link_info.num_links
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}

        # Build ignore matrix
        ignore_matrix = jnp.eye(num_links, dtype=bool)

        if ignore_adjacent:
            parent_joint_indices = link_info.parent_joint_indices
            for child_idx in range(num_links):
                parent_joint_idx = int(parent_joint_indices[child_idx])
                if 0 <= parent_joint_idx < num_links:
                    ignore_matrix = ignore_matrix.at[child_idx, parent_joint_idx].set(
                        True
                    )
                    ignore_matrix = ignore_matrix.at[parent_joint_idx, child_idx].set(
                        True
                    )

        for name1, name2 in ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        # Get lower triangular indices (i > j, so we'll swap to i < j)
        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]

        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        # Return as list of tuples, ensuring i < j
        pairs = []
        for i, j in zip(onp.array(active_i).tolist(), onp.array(active_j).tolist()):
            if i < j:
                pairs.append((i, j))
            else:
                pairs.append((j, i))
        return pairs

    @staticmethod
    def _get_trimesh_collision_geometries(
        urdf: yourdfpy.URDF, link_name: str
    ) -> trimesh.Trimesh:
        """Extracts trimesh collision geometries for a given link name."""
        if link_name not in urdf.link_map:
            return trimesh.Trimesh()

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional[trimesh.Trimesh] = None

            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = jaxlie.SE3.identity().as_matrix()

            if geom.box is not None:
                mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                try:
                    mesh_path = geom.mesh.filename
                    loaded_obj = trimesh.load(
                        file_obj=filename_handler(mesh_path), force="mesh"
                    )

                    scale = (
                        geom.mesh.scale
                        if geom.mesh.scale is not None
                        else [1.0, 1.0, 1.0]
                    )

                    if isinstance(loaded_obj, trimesh.Trimesh):
                        mesh = loaded_obj.copy()
                        mesh.apply_scale(scale)
                    elif isinstance(loaded_obj, trimesh.Scene):
                        if len(loaded_obj.geometry) > 0:
                            geom_candidate = list(loaded_obj.geometry.values())[0]
                            if isinstance(geom_candidate, trimesh.Trimesh):
                                mesh = geom_candidate.copy()
                                mesh.apply_scale(scale)
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue

                    if mesh:
                        mesh.fix_normals()

                except Exception as e:
                    logger.error(
                        f"Failed processing mesh '{geom.mesh.filename}' "
                        f"for link '{link_name}': {e}"
                    )
                    continue
            else:
                logger.warning(
                    f"Unsupported collision geometry type for link '{link_name}'."
                )
                continue

            if mesh is not None:
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        coll_mesh = sum(coll_meshes, trimesh.Trimesh())
        return coll_mesh

    def _get_geom_valid_mask(self) -> Bool[Array, "num_links max_geoms_per_link"]:
        """Get mask indicating which geometries are valid (not padding)."""
        geom_indices = jnp.arange(self.max_geoms_per_link)
        return geom_indices[None, :] < self.geom_counts[:, None]

    @jdc.jit
    def at_config(
        self, robot: "Robot", cfg: Float[Array, "*batch actuated_count"]
    ) -> CollGeom:
        """Returns the collision geometry transformed to the given robot configuration.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame.
            Shape: (*batch, num_links, max_geoms_per_link).
        """
        assert self.link_names == robot.links.names, (
            "Link name mismatch between RobotCollision and Robot kinematics."
        )

        batch_axes = cfg.shape[:-1]
        Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg)
        Ts_link_world = jaxlie.SE3(Ts_link_world_wxyz_xyz)

        # Broadcast transforms to match geometry shape
        # Ts_link_world: (*batch, num_links)
        # Need: (*batch, num_links, max_geoms_per_link)
        Ts_broadcast = jaxlie.SE3(
            jnp.broadcast_to(
                Ts_link_world.wxyz_xyz[..., None, :],
                (*batch_axes, self.num_links, self.max_geoms_per_link, 7),
            )
        )

        return self.coll.transform(Ts_broadcast)

    @jdc.jit
    def compute_self_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_geom_pairs"]:
        """Computes the signed distances for active self-collision geometry pairs.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each geometry pair.
            Shape: (*batch, num_geom_pairs).
            Positive distance means separation, negative means penetration.
        """
        num_geom_pairs = len(self.geom_pair_link_i)

        if num_geom_pairs == 0:
            batch_axes = cfg.shape[:-1]
            return jnp.zeros((*batch_axes, 0))

        # Get collision geometry at the current config
        # Shape: (*batch, num_links, max_geoms_per_link)
        coll_world = self.at_config(robot, cfg)

        # Get positions and compute distances using flat indexing
        link_i = jnp.array(self.geom_pair_link_i)
        idx_i = jnp.array(self.geom_pair_idx_i)
        link_j = jnp.array(self.geom_pair_link_j)
        idx_j = jnp.array(self.geom_pair_idx_j)

        # Extract geometry pairs
        # For Sphere: use center distance - radii
        # For Capsule: use segment-to-segment distance - radii
        # The collide function handles this generically

        # Slice geometries for each pair
        def get_geom_at_indices(
            geom: CollGeom, link_indices: Array, geom_indices: Array
        ) -> CollGeom:
            """Extract geometries at specified (link, geom) indices."""
            return jax.tree.map(
                lambda x: x[..., link_indices, geom_indices, :],
                geom,
            )

        geom_i = get_geom_at_indices(coll_world, link_i, idx_i)
        geom_j = get_geom_at_indices(coll_world, link_j, idx_j)

        # Compute distances for all pairs
        # collide expects the last axis to be the batch of geometries
        distances = collide(geom_i, geom_j)

        return distances

    @jdc.jit
    def compute_self_collision_distances_with_directions(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch actuated_count"],
    ) -> tuple[Float[Array, "*batch num_geom_pairs"], Float[Array, "*batch num_geom_pairs 3"]]:
        """Computes distances and directions for active self-collision geometry pairs.

        This is useful for analytical Jacobian computation where the direction
        from geometry i to geometry j is needed.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Tuple of:
            - distances: Signed distances for each geometry pair. Shape: (*batch, num_geom_pairs).
            - directions: Unit direction from geometry i to j. Shape: (*batch, num_geom_pairs, 3).
        """
        num_geom_pairs = len(self.geom_pair_link_i)

        if num_geom_pairs == 0:
            batch_axes = cfg.shape[:-1]
            return jnp.zeros((*batch_axes, 0)), jnp.zeros((*batch_axes, 0, 3))

        # Get collision geometry at the current config
        coll_world = self.at_config(robot, cfg)

        link_i = jnp.array(self.geom_pair_link_i)
        idx_i = jnp.array(self.geom_pair_idx_i)
        link_j = jnp.array(self.geom_pair_link_j)
        idx_j = jnp.array(self.geom_pair_idx_j)

        # For spheres, compute directly
        if isinstance(self.coll, Sphere):
            positions = coll_world.pose.translation()  # (*batch, num_links, max_geoms, 3)
            radii = coll_world.radius  # (*batch, num_links, max_geoms)

            pos_i = positions[..., link_i, idx_i, :]  # (*batch, num_geom_pairs, 3)
            pos_j = positions[..., link_j, idx_j, :]
            rad_i = radii[..., link_i, idx_i]  # (*batch, num_geom_pairs)
            rad_j = radii[..., link_j, idx_j]

            diff = pos_j - pos_i
            center_dist = jnp.linalg.norm(diff + 1e-8, axis=-1)
            directions = diff / (center_dist[..., None] + 1e-8)
            distances = center_dist - rad_i - rad_j

            return distances, directions
        else:
            # For capsules, use collide and compute direction separately
            # This is a simplified version; full capsule support would need
            # closest_segment_to_segment_with_jac
            distances = self.compute_self_collision_distance(robot, cfg)
            # Direction computation for capsules is more complex
            # For now, return zeros (capsule Jacobians use a different path)
            batch_axes = cfg.shape[:-1]
            directions = jnp.zeros((*batch_axes, num_geom_pairs, 3))
            return distances, directions

    @jdc.jit
    def compute_world_collision_distance(
        self,
        robot: "Robot",
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,
    ) -> Float[Array, "*batch_combined num_links M"]:
        """Computes signed distances between robot links and world obstacles.

        For multi-geometry-per-link, returns the minimum distance across all
        geometries in each link.

        Args:
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles.

        Returns:
            Matrix of signed distances. Shape: (*batch_combined, num_links, M).
            Positive distance means separation, negative means penetration.
        """
        batch_cfg = cfg.shape[:-1]

        # Get robot collision geometry at the current config
        # Shape: (*batch_cfg, num_links, max_geoms_per_link)
        coll_robot_world = self.at_config(robot, cfg)

        # Normalize world_geom shape
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

        # Flatten robot geometries for collision computation
        total_robot_geoms = self.num_links * self.max_geoms_per_link
        flat_robot = coll_robot_world.reshape((*batch_cfg, total_robot_geoms))

        # Compute distances: (total_robot_geoms,) vs (num_world_geoms,)
        _collide_vmap = jax.vmap(collide, in_axes=(-2, None), out_axes=-2)
        dist_flat = _collide_vmap(
            flat_robot.broadcast_to((*batch_combined, total_robot_geoms)),
            _world_geom.broadcast_to((*batch_combined, num_world_geoms)),
        )

        # Reshape to (num_links, max_geoms_per_link, num_world_geoms)
        dist_reshaped = dist_flat.reshape(
            *batch_combined, self.num_links, self.max_geoms_per_link, num_world_geoms
        )

        # Mask invalid geometries and take min per link
        valid_mask = self._get_geom_valid_mask()  # (num_links, max_geoms_per_link)
        valid_expanded = valid_mask[..., None]  # (num_links, max_geoms_per_link, 1)
        dist_masked = jnp.where(valid_expanded, dist_reshaped, jnp.inf)

        # Min over max_geoms_per_link axis
        return dist_masked.min(axis=-2)

    # --- Capsule-specific methods ---

    def get_swept_capsules(
        self,
        robot: "Robot",
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ) -> Capsule:
        """Computes swept-volume capsules between two configurations.

        Only valid when max_geoms_per_link == 1 and coll is Capsule.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
            The batch axes will be (*batch, 5, num_links).
        """
        assert self.max_geoms_per_link == 1, (
            "get_swept_capsules only works with single-geometry-per-link"
        )
        assert isinstance(self.coll, Capsule), (
            "get_swept_capsules requires Capsule geometry"
        )

        n_segments = 5

        # Get collision geometries at start and end configurations
        # Shape: (*batch, num_links, 1)
        coll_prev_world = cast(Capsule, self.at_config(robot, cfg_prev))
        coll_next_world = cast(Capsule, self.at_config(robot, cfg_next))

        # Squeeze the max_geoms_per_link dimension
        coll_prev = cast(
            Capsule, jax.tree.map(lambda x: x[..., 0, :], coll_prev_world)
        )
        coll_next = cast(
            Capsule, jax.tree.map(lambda x: x[..., 0, :], coll_next_world)
        )

        # Decompose capsules into spheres
        spheres_prev = coll_prev.decompose_to_spheres(n_segments)
        spheres_next = coll_next.decompose_to_spheres(n_segments)

        # Create swept capsules by connecting corresponding sphere pairs
        swept_capsules = Capsule.from_sphere_pairs(spheres_prev, spheres_next)

        return swept_capsules
