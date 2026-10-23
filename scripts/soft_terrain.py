"""IsaacLab/PhysX adapter for newton-terrain's fast sand force model.

Import only after AppLauncher. No Newton solver, policy or reward code is used.
Mixed soil uses a shared triangulated height field and local tangent-plane forces.
"""

import hashlib
import importlib.util
import json
import linecache
import math
import os
from pathlib import Path
import re
import sys

import torch
import warp as wp
import omni.usd
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_dreamwaq.rough_env_cfg import (
    Go2DreamWaQRoughEnvCfg, Go2DreamWaQRoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_dreamwaq.rsl_rl_ppo_cfg import (
    Go2DreamWaQRoughPPORunnerCfg,
)


@wp.func
def soil_surface_sample(point: wp.vec3, heights: wp.array2d(dtype=float),
                        material: wp.array2d(dtype=int), lower: wp.vec3, spacing: float):
    """Exact height/normal of the same two triangles used by the rendered grid."""
    u = (point[0] - lower[0]) / spacing
    v = (point[1] - lower[1]) / spacing
    height = float(0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)
    active = int(0)
    i, j = int(wp.floor(u)), int(wp.floor(v))
    if i >= 0 and j >= 0 and i < material.shape[0] and j < material.shape[1]:
        x, y = u - float(i), v - float(j)
        h00, h10 = heights[i, j], heights[i + 1, j]
        h01, h11 = heights[i, j + 1], heights[i + 1, j + 1]
        dx, dy = float(0.0), float(0.0)
        if x + y <= 1.0:
            height = h00 + (h10 - h00) * x + (h01 - h00) * y
            dx, dy = (h10 - h00) / spacing, (h01 - h00) / spacing
        else:
            height = h11 + (h01 - h11) * (1.0 - x) + (h10 - h11) * (1.0 - y)
            dx, dy = (h11 - h01) / spacing, (h11 - h10) / spacing
        normal = wp.normalize(wp.vec3(-dx, -dy, 1.0))
        active = material[i, j]
    return height, normal, active


@wp.kernel
def sample_soil_points(points: wp.array(dtype=wp.vec3), heights: wp.array2d(dtype=float),
                       material: wp.array2d(dtype=int), lower: wp.vec3, spacing: float,
                       height_out: wp.array(dtype=float), normal_out: wp.array(dtype=wp.vec3),
                       active_out: wp.array(dtype=int)):
    i = wp.tid()
    h, n, active = soil_surface_sample(points[i], heights, material, lower, spacing)
    height_out[i] = h
    normal_out[i] = n
    active_out[i] = active


@wp.func
def soil_rotate_velocity(value: wp.spatial_vector, rotation: wp.quat):
    return wp.spatial_vector(wp.quat_rotate(rotation, wp.spatial_top(value)),
                             wp.quat_rotate(rotation, wp.spatial_bottom(value)))


@wp.func
def soil_normal_rotation(normal: wp.vec3):
    # Shortest rotation from an upward surface normal onto +Z (no antipodes).
    length = wp.sqrt(2.0 * (1.0 + normal[2]))
    return wp.quat(normal[1] / length, -normal[0] / length, 0.0, (1.0 + normal[2]) / length)


def surface_proxy_code(code):
    """Reuse upstream constitutive equations in a per-leg local tangent frame.

    This is a gently curved surface approximation, not volumetric MPM or exact
    integration of a rod crossing several triangles. Keep the flat kernel intact.
    """
    kernel = code[code.index("@wp.kernel\n"):code.index("\n\nSAND_TYPES")]
    header, prefix = kernel.split("    tid = wp.tid()", 1)
    header = header.replace("def sand_forces(", "def terrain_sand_forces(")
    header = header.replace("body_f: wp.array(dtype=wp.spatial_vector)):", """body_f: wp.array(dtype=wp.spatial_vector),
                origins: wp.array(dtype=wp.vec3), heights: wp.array2d(dtype=float),
                material: wp.array2d(dtype=int), lower: wp.vec3, spacing: float,
                normals: wp.array(dtype=wp.vec3), enabled: wp.array(dtype=int)):
""")
    prefix = "    tid = wp.tid()" + prefix
    leg_start = prefix.index("    zs = surf[w]")
    belly_start = prefix.index("    # belly")
    leg = prefix[leg_start:belly_start]
    prefix = prefix[:leg_start]

    def replace_once(old, new, count=1):
        nonlocal leg
        if leg.count(old) != count:
            raise RuntimeError(f"Upstream sand kernel changed at: {old}")
        leg = leg.replace(old, new)

    leg = re.sub(r"    if typ < 0:.*?    sb =", "    sb =", leg, flags=re.S)
    replace_once("zs = surf[w]", "zs = 0.0")
    replace_once("(pf[0] - hx)", "(foot_x - hx)", 2)
    replace_once("(pf[1] - hy)", "(foot_y - hy)", 2)
    replace_once("hole_x[w * nh + slot] = pf[0]", "hole_x[w * nh + slot] = foot_x")
    replace_once("hole_y[w * nh + slot] = pf[1]", "hole_y[w * nh + slot] = foot_y")
    replace_once("pk = wp.transform_get_translation(body_q[calf])", "pk = wp.quat_rotate(rotation, wp.transform_get_translation(body_q[calf]) - anchor)")
    replace_once("com = wp.transform_point(X, body_com[calf])", "com = wp.quat_rotate(rotation, wp.transform_point(X, body_com[calf]) - anchor)")
    replace_once("body_qd[calf]", "soil_rotate_velocity(body_qd[calf], rotation)")
    replace_once("wp.spatial_vector(F + Fs, tau)", "wp.spatial_vector(wp.quat_rotate_inv(rotation, F + Fs), wp.quat_rotate_inv(rotation, tau))")
    replace_once("ph = wp.transform_get_translation(body_q[thigh])", "ph = wp.quat_rotate(rotation, wp.transform_get_translation(body_q[thigh]) - anchor)")
    replace_once("comt = wp.transform_point(body_q[thigh], body_com[thigh])", "comt = wp.quat_rotate(rotation, wp.transform_point(body_q[thigh], body_com[thigh]) - anchor)")
    replace_once("body_qd[thigh]", "soil_rotate_velocity(body_qd[thigh], rotation)")
    replace_once("wp.spatial_vector(Ft, wp.cross(pt - comt, Ft))", "wp.spatial_vector(wp.quat_rotate_inv(rotation, Ft), wp.quat_rotate_inv(rotation, wp.cross(pt - comt, Ft)))")
    frame = """    foot_x, foot_y = pf[0], pf[1]
    height, normal, active = soil_surface_sample(pf + origins[w], heights, material, lower, spacing)
    normals[w * 5 + leg] = normal
    enabled[w * 5 + leg] = active
    anchor = wp.vec3(pf[0], pf[1], height - origins[w][2])
    rotation = soil_normal_rotation(normal)
    pf = wp.quat_rotate(rotation, pf - anchor)
    vf = wp.quat_rotate(rotation, vf)
    if active != 0:
"""
    leg = "".join("    " + line if line.strip() else line for line in leg.splitlines(True))
    inactive = """    else:
        mem[tid] = -1.0
        depth_out[tid] = 0.0
        contact[tid] = 0.0
"""
    belly = """    if leg == 0:
        base = w * nb + base_local
        pb = wp.transform_get_translation(body_q[base])
        height_b, normal_b, active_b = soil_surface_sample(pb + origins[w], heights, material, lower, spacing)
        normals[w * 5 + 4] = normal_b
        enabled[w * 5 + 4] = active_b
        if active_b != 0:
            rb = soil_normal_rotation(normal_b)
            anchor_b = wp.vec3(pb[0], pb[1], height_b - origins[w][2])
            pb = wp.quat_rotate(rb, pb - anchor_b)
            vb = wp.quat_rotate(rb, wp.spatial_top(body_qd[base]))
            db = 0.06 - pb[2]
            if db > 0.0:
                fzb = wp.max(3.0e4 * db + 300.0 * wp.max(-vb[2], 0.0), 0.0)
                vhb = wp.vec3(vb[0], vb[1], 0.0)
                Fb = -0.6 * fzb * vhb / wp.sqrt(wp.dot(vhb, vhb) + 0.01)
                body_f[base] = body_f[base] + wp.spatial_vector(wp.quat_rotate_inv(rb, wp.vec3(Fb[0], Fb[1], fzb)), wp.vec3(0.0))
"""
    return "\n\n" + header + prefix + frame + leg + inactive + belly


class SoilTerrainImporter(TerrainImporter):
    def import_ground_plane(self, key, size=(2.0e6, 2.0e6)):
        super().import_ground_plane(key, size)
        if not self.cfg.soft_enabled:
            return
        if not math.isfinite(self.cfg.depth) or self.cfg.depth <= 0:
            raise ValueError("Soil depth must be positive")
        stage = omni.usd.get_context().get_stage()
        ground = UsdGeom.Xformable(stage.GetPrimAtPath(self.cfg.prim_path))
        ground.AddTranslateOp(opSuffix="soil_depth").Set(Gf.Vec3d(0, 0, -self.cfg.depth))
        # Ray casting sees the nominal top; this visual mesh has NO collider.
        visual = sim_utils.MeshCuboidCfg(
            size=(2000.0, 2000.0, 0.0002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.54, 0.40, 0.22)),
        )
        visual.func("/World/soil_surface", visual, translation=(0.0, 0.0, -0.0001))


@configclass
class SoilTerrainCfg(TerrainImporterCfg):
    class_type: type = SoilTerrainImporter
    soft_enabled: bool = True
    depth: float = 0.15
    material: str = "f0.75"
    bearing_scale: float = 1.0
    drag_scale: float = 1.0
    footprint_keep: float = 0.7
    floor_stiffness: float = 1.0e4
    normal_damping: float = 60.0
    drag_damping_cap: float = 400.0
    optimized: bool = True
    proxy_path: str = os.environ.get(
        "NEWTON_TERRAIN_PATH", str(Path(__file__).resolve().parents[2] / "newton-terrain")
    )
    proxy_sha256: str = ""
    table_sha256: str = ""
    adapter_sha256: str = ""
    warp_version: str = ""
    proxy_coefficients: dict = {}
    surface_model: str = "flat nominal top; per-foot plastic memory; 16 footprint holes per environment"


def soil_cell_grid(cfg, scenario):
    """Fixed profiles; all rows repeat the same geometry without a soil curriculum."""
    import numpy as np
    sx, sy = cfg.terrain_generator.size
    nx, ny = round(sx / cfg.soft_resolution), round(sy / cfg.soft_resolution)
    x, y = np.linspace(-sx / 2, sx / 2, nx + 1), np.linspace(-sy / 2, sy / 2, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ax, ay = sx / 2 - cfg.soft_border, sy / 2 - cfg.soft_border
    window = np.cos(np.pi * np.clip(xx / ax, -1, 1) / 2) ** 2 * np.cos(np.pi * np.clip(yy / ay, -1, 1) / 2) ** 2
    inside_vertices = (np.abs(xx) < ax) & (np.abs(yy) < ay)
    window *= inside_vertices
    z = np.zeros_like(xx)
    if scenario == "mound":
        z = cfg.soft_mound_height * window
    elif scenario == "basin":
        z = -cfg.soft_basin_depth * window
    elif scenario == "undulating":
        z = cfg.soft_wave_height * window * np.cos(2 * np.pi * xx / cfg.soft_wave_length) * np.cos(2 * np.pi * yy / cfg.soft_wave_length)
    xc, yc = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, indexing="ij")
    inner = (np.abs(xc) < ax) & (np.abs(yc) < ay)
    active = inner.copy()
    if scenario == "mixed":
        active &= np.floor((xc + cfg.soft_stripe_width / 2) / cfg.soft_stripe_width).astype(int) % 2 == 0
    return xx, yy, z, active, inner


class MixedSoilTerrainImporter(TerrainImporter):
    """Original rough cells plus sand cells in the same grid and shared outer border."""

    def import_mesh(self, key, mesh):
        if key == "terrain":
            import numpy as np
            # Keep the original generator/RNG and every active-cell triangle.
            # Its old perimeter must be removed before extending the grid, or
            # that rigid surface would support feet inside the new sand cells.
            self.cfg.source_rough_mesh_sha256 = hashlib.sha256(
                mesh.vertices.tobytes() + mesh.faces.tobytes()
            ).hexdigest()
            generator = self.cfg.terrain_generator
            half = np.array(generator.size) * [generator.num_rows, generator.num_cols] / 2
            inside = (np.abs(mesh.vertices[:, :2]) <= half + 1e-6).all(axis=1)
            active = inside[mesh.faces].all(axis=1)
            if not np.allclose(mesh.triangles[~active, :, 2], 0.0, atol=1e-6):
                raise ValueError("Expected only the flat outer border outside original terrain cells")
            mesh = mesh.copy()
            mesh.update_faces(active)
            mesh.remove_unreferenced_vertices()
        super().import_mesh(key, mesh)

    def __init__(self, cfg):
        import numpy as np
        import trimesh
        from isaaclab.terrains.trimesh.utils import make_plane

        if cfg.terrain_type != "generator" or not cfg.soft_enabled:
            raise ValueError("Mixed terrain requires the original generator and enabled soil")
        if not 0 < cfg.soft_fraction < 1 or not math.isfinite(cfg.depth) or cfg.depth <= 0:
            raise ValueError("Mixed soil requires 0 < soft_fraction < 1 and positive depth")
        if type(cfg.soft_columns) is not int or cfg.soft_columns < 1:
            raise ValueError("soft_columns must be a positive integer")
        if not math.isfinite(cfg.soft_border) or not 0 < cfg.soft_border < min(cfg.terrain_generator.size) / 2:
            raise ValueError("soft_border must be positive and less than half a tile's size")
        valid_scenarios = {"flat", "mound", "basin", "undulating", "mixed"}
        if not cfg.soft_scenarios or not set(cfg.soft_scenarios) <= valid_scenarios:
            raise ValueError(f"soft_scenarios must contain names from {sorted(valid_scenarios)}")
        if not 0 < cfg.soft_resolution <= 0.2 or not cfg.soft_stripe_width > 0:
            raise ValueError("Use positive stripe width and 0 < soft_resolution <= 0.2 m")
        for length in (*cfg.terrain_generator.size, cfg.soft_border, cfg.soft_stripe_width / 2):
            if abs(length / cfg.soft_resolution - round(length / cfg.soft_resolution)) > 1e-6:
                raise ValueError("Tile size, border and half stripe width must align with soft_resolution")
        if not all(math.isfinite(value) and value > 0 for value in
                   (cfg.soft_mound_height, cfg.soft_basin_depth, cfg.soft_wave_height, cfg.soft_wave_length)):
            raise ValueError("Soil geometry amplitudes and wavelength must be finite and positive")
        self.num_soft = round(cfg.num_envs * cfg.soft_fraction)
        self.num_rough = cfg.num_envs - self.num_soft
        if min(self.num_soft, self.num_rough) < 1:
            raise ValueError("Mixed terrain needs at least one environment in each group")
        # IsaacLab 2.0 terrain functions also use global NumPy/Torch RNGs.
        # Retain their inputs so geometry can be regenerated in parity audits.
        self._rough_numpy_state = np.random.get_state()
        self._rough_torch_state = torch.get_rng_state()
        self._rough_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        super().__init__(cfg)
        rough = self.meshes["terrain"]
        cfg.rough_mesh_sha256 = hashlib.sha256(
            rough.vertices.tobytes() + rough.faces.tobytes()
        ).hexdigest()
        self.soft_mask = torch.arange(cfg.num_envs, device=self.device) >= self.num_rough
        # Distribute the rough group over ALL original terrain columns, rather
        # than retaining only the first half of the original assignments.
        self.terrain_types[:self.num_rough] = torch.div(
            torch.arange(self.num_rough, device=self.device),
            self.num_rough / self.terrain_origins.shape[1], rounding_mode="floor",
        ).long()
        self.env_origins[:self.num_rough] = self.terrain_origins[
            self.terrain_levels[:self.num_rough], self.terrain_types[:self.num_rough]
        ]
        # Append directly to the next original grid column, with an inset sand
        # patch and a flat rigid frame inside EACH same-sized terrain cell.
        generator = cfg.terrain_generator
        sx, sy = generator.size
        lo = np.array([-generator.num_rows * sx / 2, generator.num_cols * sy / 2])
        hi = lo + np.array([generator.num_rows * sx, cfg.soft_columns * sy])
        tops, centres, rim = [], [], []
        nx, ny = round(sx / cfg.soft_resolution), round(sy / cfg.soft_resolution)
        height_grid = np.zeros((generator.num_rows * nx + 1, cfg.soft_columns * ny + 1), dtype=np.float32)
        material_grid = np.zeros((generator.num_rows * nx, cfg.soft_columns * ny), dtype=np.int32)
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        a = (i * (ny + 1) + j).flatten()
        faces = np.stack((np.stack((a, a + ny + 1, a + 1), axis=1),
                          np.stack((a + ny + 2, a + 1, a + ny + 1), axis=1)), axis=1).reshape(-1, 3)
        self.soft_scenario_names = [cfg.soft_scenarios[col % len(cfg.soft_scenarios)] for col in range(cfg.soft_columns)]

        def rectangle(lower, upper):
            lower, upper = np.asarray(lower), np.asarray(upper)
            if np.all(upper > lower):
                plane = make_plane(tuple(upper - lower), height=0.0)
                plane.apply_translation([*((lower + upper) / 2), 0.0])
                rim.append(plane)

        for row in range(generator.num_rows):
            for col in range(cfg.soft_columns):
                centre = [lo[0] + (row + 0.5) * sx, lo[1] + (col + 0.5) * sy, 0.0]
                xx, yy, zz, active, inner = soil_cell_grid(cfg, self.soft_scenario_names[col])
                vertices = np.stack((xx + centre[0], yy + centre[1], zz), axis=-1).reshape(-1, 3)
                tile = trimesh.Trimesh(vertices=vertices, faces=faces[np.repeat(active.flatten(), 2)], process=False)
                tile.remove_unreferenced_vertices()
                hard_faces = faces[np.repeat((inner & ~active).flatten(), 2)]
                if len(hard_faces):
                    hard = trimesh.Trimesh(vertices=vertices, faces=hard_faces, process=False)
                    hard.remove_unreferenced_vertices()
                    rim.append(hard)
                height_grid[row * nx:(row + 1) * nx + 1, col * ny:(col + 1) * ny + 1] = zz
                material_grid[row * nx:(row + 1) * nx, col * ny:(col + 1) * ny] = active
                tops.append(tile)
                centre[2] = float(zz[nx // 2, ny // 2])
                centres.append(centre)
                lower = np.asarray(centre[:2]) - [sx / 2, sy / 2]
                upper = lower + [sx, sy]
                inset_lo, inset_hi = lower + cfg.soft_border, upper - cfg.soft_border
                rectangle(lower, [inset_lo[0], upper[1]])
                rectangle([inset_hi[0], lower[1]], upper)
                rectangle([inset_lo[0], lower[1]], [inset_hi[0], inset_lo[1]])
                rectangle([inset_lo[0], inset_hi[1]], [inset_hi[0], upper[1]])
        self.soft_tile_origins = torch.tensor(centres, dtype=torch.float, device=self.device)
        sample = torch.arange(self.num_soft, device=self.device)
        self.soft_tile_ids = ((sample // cfg.soft_columns) % generator.num_rows) * cfg.soft_columns + sample % cfg.soft_columns
        self.scenario_masks = {}
        for scenario in dict.fromkeys(self.soft_scenario_names):
            selected = torch.zeros(self.num_soft, dtype=torch.bool, device=self.device)
            for col, name in enumerate(self.soft_scenario_names):
                if name == scenario:
                    selected |= self.soft_tile_ids % cfg.soft_columns == col
            if selected.any():
                mask = torch.zeros(cfg.num_envs, dtype=torch.bool, device=self.device)
                mask[self.soft_mask] = selected
                self.scenario_masks[scenario] = mask
        soil_origins = self.soft_tile_origins[self.soft_tile_ids]
        self.env_origins[self.soft_mask] = soil_origins
        self.terrain_levels[self.soft_mask] = 0
        self.soft_origins = soil_origins.clone()
        self.soft_bounds = torch.tensor(np.stack((lo, hi)), dtype=torch.float, device=self.device)
        self.surface_heights = wp.array(height_grid, dtype=float, device=self.device)
        self.surface_material = wp.array(material_grid, dtype=int, device=self.device)
        self.surface_lower = wp.vec3(float(lo[0]), float(lo[1]), 0.0)
        top = trimesh.util.concatenate(tops)
        bottom = top.copy()
        bottom.apply_translation([0, 0, -cfg.depth])
        self.import_mesh("soil_bottom", bottom)

        # One outer border around the COMPLETE grid, including the new columns.
        margin = generator.border_width
        grid_lo = np.array([lo[0], -generator.num_cols * sy / 2])
        rectangle(grid_lo - margin, [grid_lo[0], hi[1] + margin])
        rectangle([hi[0], grid_lo[1] - margin], hi + margin)
        rectangle([grid_lo[0], grid_lo[1] - margin], [hi[0], grid_lo[1]])
        rectangle([grid_lo[0], hi[1]], [hi[0], hi[1] + margin])
        rim_mesh = trimesh.util.concatenate(rim)
        self.import_mesh("soil_rim", rim_mesh)
        stage = omni.usd.get_context().get_stage()

        def sensor_mesh(path, mesh, color=None):
            # Define directly: create_prim_from_mesh always adds a collider.
            prim = UsdGeom.Mesh.Define(stage, path)
            prim.CreatePointsAttr(np.asarray(mesh.vertices, dtype=np.float32))
            prim.CreateFaceVertexIndicesAttr(mesh.faces.flatten())
            prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
            prim.CreateSubdivisionSchemeAttr("none")
            prim.CreateDoubleSidedAttr(True)
            if color is not None:
                prim.CreateDisplayColorAttr([color])
            else:
                UsdGeom.Imageable(prim).MakeInvisible()

        UsdGeom.Xform.Define(stage, "/World/soil_surface")
        for index, tile in enumerate(tops):
            row, col = divmod(index, cfg.soft_columns)
            color = (0.62, 0.47, 0.27) if (row + col) % 2 == 0 else (0.42, 0.28, 0.12)
            sensor_mesh(f"/World/soil_surface/tile_{row}_{col}_{self.soft_scenario_names[col]}", tile, color)
        # IsaacLab 2.0 RayCaster accepts one mesh: combine only the nominal
        # surfaces, never the soft bottom. This duplicate has no collision API.
        sensor_mesh("/World/mixed_height_scan", trimesh.util.concatenate([rough, top, rim_mesh]))
        cfg.resolved_num_rough, cfg.resolved_num_soft = self.num_rough, self.num_soft
        cfg.resolved_soft_tiles = len(tops)
        cfg.resolved_soft_scenarios = self.soft_scenario_names
        cfg.soft_surface_sha256 = hashlib.sha256(height_grid.tobytes() + material_grid.tobytes()).hexdigest()
        print(f"[Mixed soil] rough={self.num_rough}, soft={self.num_soft}; "
              f"sand tiles={generator.num_rows}x{cfg.soft_columns}, size={sx}x{sy} m; "
              f"flat frame={cfg.soft_border} m per side; "
              f"scenarios={','.join(self.soft_scenario_names)}; "
              f"rough mesh={cfg.rough_mesh_sha256[:12]}", flush=True)

    def update_env_origins(self, env_ids, move_up, move_down):
        ids = (torch.arange(self.cfg.num_envs, device=self.device)[env_ids]
               if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        rough = ~self.soft_mask[ids]
        if rough.any():
            super().update_env_origins(ids[rough], move_up[rough], move_down[rough])

    def sample_surface(self, points):
        shape = points.shape[:-1]
        points = points.contiguous().view(-1, 3)
        height = torch.empty(len(points), device=self.device)
        normal = torch.empty((len(points), 3), device=self.device)
        active = torch.empty(len(points), dtype=torch.int32, device=self.device)
        wp.launch(sample_soil_points, dim=len(points), inputs=[
            wp.from_torch(points, wp.vec3), self.surface_heights, self.surface_material,
            self.surface_lower, self.cfg.soft_resolution, wp.from_torch(height),
            wp.from_torch(normal, wp.vec3), wp.from_torch(active)], device=self.device,
            stream=(wp.stream_from_torch(torch.cuda.current_stream(self.device))
                    if str(self.device).startswith("cuda") else None))
        return height.view(shape), normal.view(*shape, 3), active.view(shape)


@configclass
class MixedSoilTerrainCfg(SoilTerrainCfg):
    class_type: type = MixedSoilTerrainImporter
    physics_substeps: int = 32
    soft_fraction: float = 0.5
    soft_columns: int = 5
    soft_border: float = 1.0
    soft_scenarios: list[str] = ["flat", "mound", "basin", "undulating", "mixed"]
    soft_resolution: float = 0.1
    soft_mound_height: float = 0.30
    soft_basin_depth: float = 0.20
    soft_wave_height: float = 0.10
    soft_wave_length: float = 2.0
    soft_stripe_width: float = 1.0
    resolved_soft_scenarios: list[str] = []
    soft_surface_sha256: str = ""
    resolved_soft_tiles: int = 0
    resolved_num_rough: int = 0
    resolved_num_soft: int = 0
    rough_mesh_sha256: str = ""
    source_rough_mesh_sha256: str = ""
    surface_model: str = "five fixed soil scenarios; triangulated surface and local tangent-plane proxy; shared outer border"


def mixed_terrain_levels(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel
    terrain = env.scene.terrain
    ids = (torch.arange(env.num_envs, device=env.device)[env_ids]
           if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    ids = ids[~terrain.soft_mask[ids]]
    if len(ids):
        terrain_levels_vel(env, ids, asset_cfg=asset_cfg)
    return terrain.terrain_levels[~terrain.soft_mask].float().mean()


class MixedSoilEnv(ManagerBasedRLEnv):
    def step(self, action):
        obs, reward, terminated, timeout, extras = super().step(action)
        extras["log"] = dict(extras["log"])
        soft = self.scene.terrain.soft_mask
        for name, mask in (("rough", ~soft), ("soft", soft)):
            extras["log"][f"Mixed/{name}_reward_per_second"] = reward[mask].mean() / self.step_dt
            extras["log"][f"Mixed/{name}_termination_fraction"] = terminated[mask].float().mean()
            extras["log"][f"Mixed/{name}_timeout_fraction"] = timeout[mask].float().mean()
        extras["log"]["Mixed/soft_sample_fraction"] = soft.float().mean()
        for name, mask in self.scene.terrain.scenario_masks.items():
            extras["log"][f"SoilScenario/{name}_reward_per_second"] = reward[mask].mean() / self.step_dt
            extras["log"][f"SoilScenario/{name}_termination_fraction"] = terminated[mask].float().mean()
            extras["log"][f"SoilScenario/{name}_timeout_fraction"] = timeout[mask].float().mean()
        return obs, reward, terminated, timeout, extras


class _SoilContactView:
    """Merge normal soil forces before the stock sensor updates history/timers."""

    def __init__(self, physical, sensor):
        self.physical, self.sensor = physical, sensor

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def get_net_contact_forces(self, dt):
        physical = self.physical.get_net_contact_forces(dt=dt)
        soil = self.sensor.soil
        if soil is None:
            return physical
        virtual = soil.normal_w[:, self.sensor.robot_body_ids]
        return physical + virtual.reshape_as(physical)


class SoilContactSensor(ContactSensor):
    soil = None

    def _initialize_impl(self):
        super()._initialize_impl()
        if self.cfg.filter_prim_paths_expr:
            raise ValueError("Soil adapter supports net forces, not filtered contact matrices")
        self._contact_physx_view = _SoilContactView(self._contact_physx_view, self)


class SoilJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.soil = SoilModel(env, self._asset)

    def apply_actions(self):
        super().apply_actions()
        self.soil.apply()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "soil"):
            self.soil.reset(env_ids)


class SoilModel:
    def __init__(self, env, asset):
        self.env, self.asset = env, asset
        cfg = env.cfg.scene.terrain
        self.cfg = cfg
        root = Path(cfg.proxy_path).expanduser().resolve()
        source, table_path = root / "newton_terrain/sand_proxy.py", root / "runs/proxy_table.json"
        # Reuse the original force kernel without importing Newton/MuJoCo.
        cfg.proxy_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        cfg.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        cfg.warp_version = wp.__version__
        coefficients = (cfg.floor_stiffness, cfg.normal_damping, cfg.drag_damping_cap)
        if not all(math.isfinite(value) and value > 0 for value in coefficients):
            raise ValueError("Soil spring/damping coefficients must be positive")
        kernel_key = hashlib.sha256(f"{cfg.proxy_sha256}:{coefficients}:{cfg.adapter_sha256}".encode()).hexdigest()[:12]
        module_name = f"dreamwaq_soil_proxy_{kernel_key}"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Isaac Sim 4.5 bundles Warp 1.5; adapt annotations and the old
            # conditional spelling, preserving numerical formulas and Sim's Warp.
            code = re.sub(r"wp\.array\[([^\]]+)\]", r"wp.array(dtype=\1)", source.read_text())
            for name, value in zip(("K_FLOOR", "C_FLOOR", "C_MAX"), coefficients):
                code = re.sub(rf"(?m)^{name} = [^#\n]+", f"{name} = {float(value)!r}   ", code)
            code = re.sub(
                r"(?m)^([ \t]*)contact\[tid\] = wp.where\((.*?), 1.0, 0.0\)",
                r"\1contact[tid] = 0.0\n\1if \2:\n\1    contact[tid] = 1.0", code,
            )
            code += surface_proxy_code(code)
            module.soil_surface_sample = soil_surface_sample
            module.soil_rotate_velocity = soil_rotate_velocity
            module.soil_normal_rotation = soil_normal_rotation
            linecache.cache[str(source)] = (len(code), None, code.splitlines(True), str(source))
            exec(compile(code, str(source), "exec"), module.__dict__)
        self.proxy = sys.modules[module_name]
        cfg.table_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
        self.row = json.loads(table_path.read_text())[cfg.material]
        cfg.proxy_coefficients = dict(self.row)
        if cfg.bearing_scale <= 0 or cfg.drag_scale < 0 or not 0 <= cfg.footprint_keep <= 1:
            raise ValueError("Invalid soil bearing/drag/memory coefficients")
        self.n, self.nb, self.device = env.num_envs, asset.num_bodies, env.device
        self.soft_mask = getattr(env.scene.terrain, "soft_mask", torch.ones(
            self.n, dtype=torch.bool, device=self.device,
        ))
        self.rough_ids = (~self.soft_mask).nonzero(as_tuple=False).flatten()
        bounds = getattr(env.scene.terrain, "soft_bounds", None)
        self.global_soft_bounds = bounds
        self.local_soft_bounds = (None if bounds is None else
                                  bounds[None] - env.scene.env_origins[:, None, :2])
        legs = ("FL", "FR", "RL", "RR")
        self.feet = [asset.body_names.index(f"{leg}_foot") for leg in legs]
        self.calves = [asset.body_names.index(f"{leg}_calf") for leg in legs]
        self.thighs = [asset.body_names.index(f"{leg}_thigh") for leg in legs]
        self.base = asset.body_names.index("base")
        self.com = asset.root_physx_view.get_coms()[..., :3].to(self.device).contiguous()
        self.application_com = self.com.clone()
        self.world_com = torch.zeros_like(self.com)
        self._refresh_application_com = True
        self._com_control_step = -1
        self.q = torch.zeros((self.n, self.nb, 7), device=self.device)
        self.qd = torch.zeros((self.n, self.nb, 6), device=self.device)
        self.wrench = torch.zeros_like(self.qd)
        self.normal_w = torch.zeros((self.n, self.nb, 3), device=self.device)
        self.mem = torch.full((self.n, 4), -1.0, device=self.device)
        self.contact = torch.zeros_like(self.mem)
        self.depth = torch.zeros_like(self.mem)
        self.hole_x = torch.zeros((self.n, 16), device=self.device)
        self.hole_y = torch.zeros_like(self.hole_x)
        self.hole_d = torch.zeros_like(self.hole_x)
        self.hole_n = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.steps = 0
        # Sampling groups choose spawn cells/statistics; actual material follows
        # each foot's location, including robots that walk in from rough cells.
        self.material_ids = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self.surface_heights = (-env.scene.env_origins[:, 2]).contiguous()
        self.surface_terrain = env.scene.terrain if bounds is not None else None
        self.surface_normals = torch.zeros((self.n, 5, 3), device=self.device)
        self.surface_normals[..., 2] = 1
        self.surface_enabled = torch.zeros((self.n, 5), dtype=torch.int32, device=self.device)
        self.origins = env.scene.env_origins.contiguous()
        wp.init()
        self.table = self.proxy.make_table([self.row], device=self.device)
        def a(value, dtype=float):
            return wp.array(value, dtype=dtype, device=self.device)
        self.inputs = [
            wp.from_torch(self.q.reshape(-1, 7), wp.transform),
            wp.from_torch(self.qd.reshape(-1, 6), wp.spatial_vector),
            wp.from_torch(self.com.reshape(-1, 3), wp.vec3), self.nb,
            a(self.calves, int), a(self.feet, int), a(self.thighs, int), self.base,
            wp.from_torch(self.surface_heights), wp.from_torch(self.material_ids),
            a([cfg.bearing_scale] * self.n), a([cfg.drag_scale] * self.n), self.table,
            a([cfg.footprint_keep] * self.n), 16,
            *[wp.from_torch(t.flatten()) for t in
              (self.hole_x, self.hole_y, self.hole_d, self.hole_n, self.mem, self.contact, self.depth)],
            wp.from_torch(self.wrench.reshape(-1, 6), wp.spatial_vector),
        ]
        self.surface_inputs = [] if self.surface_terrain is None else [
            wp.from_torch(self.origins, wp.vec3), self.surface_terrain.surface_heights,
            self.surface_terrain.surface_material, self.surface_terrain.surface_lower, cfg.soft_resolution,
            wp.from_torch(self.surface_normals.reshape(-1, 3), wp.vec3),
            wp.from_torch(self.surface_enabled.flatten()),
        ]
        sensor = env.scene["contact_forces"]
        sensor.robot_body_ids = [asset.body_names.index(name) for name in sensor.body_names]
        sensor.soil = self
        print(f"[Soil] PhysX + {cfg.material}, depth={cfg.depth} m; proxy={cfg.proxy_sha256[:12]}")

    def reset(self, env_ids=None):
        # Reset events run before action reset. Refresh after any reset-time
        # randomization; no COM changes occur inside a physics substep.
        self._refresh_application_com = True
        ids = slice(None) if env_ids is None else env_ids
        self.mem[ids] = -1.0
        for tensor in (self.contact, self.depth, self.hole_x, self.hole_y, self.hole_d,
                       self.hole_n, self.wrench, self.normal_w):
            tensor[ids] = 0

    def apply(self):
        self.prepare_state()
        self.evaluate()
        self.apply_wrench(position=self.world_com)
        self.steps += 1

    def prepare_state(self):
        data = self.asset.data
        if self.local_soft_bounds is not None:
            # Rough curriculum can change both XY and Z of an environment's
            # local origin. Sand remains at world Z=0 in the shared scene.
            self.local_soft_bounds.copy_(self.global_soft_bounds[None] - self.env.scene.env_origins[:, None, :2])
            self.surface_heights.copy_(-self.env.scene.env_origins[:, 2])
            self.origins.copy_(self.env.scene.env_origins)
        if self.cfg.optimized:
            if self._refresh_application_com or self._com_control_step != self.env.common_step_counter:
                self.application_com.copy_(self.asset.root_physx_view.get_coms()[..., :3].to(self.device))
                self._refresh_application_com = False
                self._com_control_step = self.env.common_step_counter
            # Same quat_apply expression as ArticulationData.body_com_state_w.
            # Reuse body_state_w instead of fetching poses/velocities twice and
            # downloading get_coms() to the CPU twice on EVERY physics substep.
            self.world_com = data.body_pos_w + quat_apply(data.body_quat_w, self.application_com)
        else:
            self.world_com = data.body_com_pos_w
        self.q[..., :3] = data.body_pos_w - self.env.scene.env_origins[:, None, :]
        self.q[..., 3:6] = data.body_quat_w[..., 1:4]
        self.q[..., 6] = data.body_quat_w[..., 0]
        self.qd.copy_(data.body_vel_w)
        # Upstream foot drag uses velocity at the foot origin; rod wrenches use COM velocity.
        self.qd[:, self.feet, :3] += torch.cross(
            data.body_ang_vel_w[:, self.feet],
            data.body_pos_w[:, self.feet] - self.world_com[:, self.feet], dim=-1,
        )

    def evaluate(self):
        """Evaluate the original force law on q/qd; also used by physics acceptance fixtures."""
        self.wrench.zero_()
        if self.surface_terrain is not None:
            wp.launch(self.proxy.terrain_sand_forces, dim=self.n * 4,
                      inputs=self.inputs + self.surface_inputs, device=self.device,
                      stream=(wp.stream_from_torch(torch.cuda.current_stream(self.device))
                              if str(self.device).startswith("cuda") else None))
            self._update_normal_contacts()
            return
        wp.launch(
            self.proxy.sand_forces, dim=self.n * 4, inputs=self.inputs, device=self.device,
            stream=(wp.stream_from_torch(torch.cuda.current_stream(self.device))
                    if str(self.device).startswith("cuda") else None),
        )
        self._update_normal_contacts()
        if self.local_soft_bounds is None and len(self.rough_ids):
            # The upstream rigid branch emits geometric depth/contact flags;
            # those are not soil contacts and must never reach our sensor view.
            self.depth[self.rough_ids] = 0
            self.contact[self.rough_ids] = 0
            self.mem[self.rough_ids] = -1
            self.normal_w[self.rough_ids] = 0

    def apply_wrench(self, position=None):
        """Apply the evaluated world-frame wrench explicitly at each body's COM."""
        # Explicit WORLD position is essential: PhysX's default application point
        # is the link origin, whereas the upstream torque is about the COM.
        # Add these forces before scene.write_data_to_sim; the original event
        # wrench buffers remain independent and are applied by IsaacLab as usual.
        self.force_w = self.wrench[..., :3].contiguous()
        self.torque_w = self.wrench[..., 3:].contiguous()
        if position is None:
            position = self.asset.data.body_com_pos_w
        self.asset.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.force_w, torque_data=self.torque_w,
            position_data=position.contiguous(),
            indices=self.asset._ALL_INDICES, is_global=True,
        )

    def _update_normal_contacts(self):
        # Normal bearing belongs to the FOOT sensor although its wrench is applied to the calf.
        # Tangential drag is not added to PhysX's normal-contact reporter.
        p, row = self.proxy, self.row
        d = self.depth.clamp_min(0)
        v = self.qd[:, self.feet, :3]
        vn = v[..., 2]
        speed = torch.linalg.vector_norm(v[..., :2], dim=-1)
        if self.surface_terrain is not None:
            vn = (v * self.surface_normals[:, :4]).sum(-1)
            speed = torch.linalg.vector_norm(v - vn[..., None] * self.surface_normals[:, :4], dim=-1)
        e = d - self.mem.clamp_min(0)
        bearing = torch.where(e > 0, p.K_FLOOR * e + p.C_FLOOR * (-vn).clamp_min(0), 0)
        pull = row["pull_p"] * d * (vn / 0.05).clamp(0, 1)
        fz = torch.where(self.depth > 0, bearing - pull, 0)
        fh = self.cfg.drag_scale * row["drag_a"] * d.pow(row["drag_n"])
        eps = (fh / p.C_MAX).clamp_min(0.1)
        lift = row["lift_r"] * fh * speed / torch.sqrt(speed.square() + eps.square())
        self.normal_w.zero_()
        if self.surface_terrain is not None:
            n = self.surface_normals[:, :4]
            active = self.surface_enabled[:, :4].float()
            self.normal_w[:, self.feet] = n * (fz.clamp_min(0) * active)[..., None]
            rod = ((self.wrench[:, self.calves, :3] * n).sum(-1) - fz - lift).clamp_min(0) * active
            self.normal_w[:, self.calves] = n * rod[..., None]
            thigh = (self.wrench[:, self.thighs, :3] * n).sum(-1).clamp_min(0) * active
            self.normal_w[:, self.thighs] = n * thigh[..., None]
            nb = self.surface_normals[:, 4]
            base = (self.wrench[:, self.base, :3] * nb).sum(-1).clamp_min(0) * self.surface_enabled[:, 4]
            self.normal_w[:, self.base] = nb * base[:, None]
            return
        self.normal_w[:, self.feet, 2] = fz.clamp_min(0)
        self.normal_w[:, self.calves, 2] = (self.wrench[:, self.calves, 2] - fz - lift).clamp_min(0)
        self.normal_w[:, self.thighs, 2] = self.wrench[:, self.thighs, 2].clamp_min(0)
        self.normal_w[:, self.base, 2] = self.wrench[:, self.base, 2].clamp_min(0)


def configure_flat(cfg):
    """Shared geometry for paired rigid/soft experiments; keep original task terms."""
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None


def configure_soil_timing(cfg, substeps=64):
    # 64 is the converged reference; 32 is the mixed-task speed experiment.
    # Both retain the original 20 ms policy period and 15 ms contact window.
    if type(substeps) is not int or substeps not in (32, 64):
        raise ValueError("Soil physics_substeps must be 32 or 64")
    cfg.sim.dt = 0.02 / substeps
    cfg.decimation = cfg.sim.render_interval = substeps
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.scene.contact_forces.history_length = 3 * substeps // 4
    cfg.scene.height_scanner.update_period = 0.02


def configure_soil(cfg):
    configure_flat(cfg)
    configure_soil_timing(cfg)
    terrain = SoilTerrainCfg()
    for key, value in vars(cfg.scene.terrain).items():
        setattr(terrain, key, value)
    terrain.class_type = SoilTerrainImporter
    cfg.scene.terrain = terrain


def enable_soil(cfg):
    """Resolve after Hydra overrides, before constructing the environment."""
    if cfg.scene.terrain.soft_enabled:
        mixed = isinstance(cfg.scene.terrain, MixedSoilTerrainCfg)
        if mixed:
            # Resolve the single timing selector after Hydra overrides, so dt,
            # decimation, rendering and contact sampling cannot drift apart.
            configure_soil_timing(cfg, substeps=cfg.scene.terrain.physics_substeps)
        if cfg.scene.terrain.terrain_type != "plane" and not mixed:
            raise ValueError("The first-stage soil proxy supports flat terrain only")
        cfg.actions.joint_pos.class_type = SoilJointPositionAction
        cfg.scene.contact_forces.class_type = SoilContactSensor
        cfg.scene.height_scanner.mesh_prim_paths = [
            "/World/mixed_height_scan" if mixed else "/World/soil_surface"
        ]


@configclass
class Go2SoftEnvCfg(Go2DreamWaQRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_soil(self)


@configclass
class Go2SoftPlayEnvCfg(Go2DreamWaQRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_soil(self)


def configure_mixed(cfg):
    terrain = MixedSoilTerrainCfg()
    for key, value in vars(cfg.scene.terrain).items():
        setattr(terrain, key, value)
    terrain.class_type = MixedSoilTerrainImporter
    cfg.scene.terrain = terrain
    configure_soil_timing(cfg, substeps=terrain.physics_substeps)
    if cfg.curriculum.terrain_levels is not None:
        cfg.curriculum.terrain_levels.func = mixed_terrain_levels


@configclass
class Go2MixedEnvCfg(Go2DreamWaQRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_mixed(self)


@configclass
class Go2MixedPlayEnvCfg(Go2DreamWaQRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_mixed(self)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class Go2SoftRunnerCfg(Go2DreamWaQRoughPPORunnerCfg):
    # Warm start into a new run; distinct from resuming optimizer/iteration state.
    # IsaacLab validates from_dict values against the default's runtime type.
    init_checkpoint: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isaaclab/go2_dreamwaq_soft_flat"


@configclass
class Go2MixedRunnerCfg(Go2SoftRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isaaclab/go2_dreamwaq_mixed"


def make_runner(env, train_cfg, log_dir=None, device="cpu"):
    """Original runner, optionally initialized from another experiment's weights."""
    from rsl_rl.runners import OnPolicyRunner

    checkpoint = train_cfg.get("init_checkpoint")
    if checkpoint and train_cfg.get("resume"):
        raise ValueError("Use either init_checkpoint for a new fine-tuning run or resume for continuation")
    if checkpoint:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    if checkpoint:
        runner.load(str(checkpoint), load_optimizer=False)
        runner.current_learning_iteration = 0
        print(f"[Soil fine-tune] Loaded network weights: {checkpoint}", flush=True)
        print("[Soil fine-tune] Fresh optimizer; new iteration counter starts at 0", flush=True)
    return runner


def make_env(cfg, overview=False, **kwargs):
    enable_soil(cfg)
    if overview and isinstance(cfg.scene.terrain, MixedSoilTerrainCfg):
        terrain = cfg.scene.terrain
        generator = terrain.terrain_generator
        sx, sy = generator.size
        lower_y = -generator.num_cols * sy / 2 - generator.border_width
        upper_y = generator.num_cols * sy / 2 + generator.border_width + terrain.soft_columns * sy
        centre_y = (lower_y + upper_y) / 2
        span = max(generator.num_rows * sx + 2 * generator.border_width, upper_y - lower_y)
        cfg.viewer.origin_type = "world"
        cfg.viewer.lookat = (0.0, centre_y, 0.0)
        cfg.viewer.eye = (0.9 * span, centre_y + 0.9 * span, 1.1 * span)
    env_type = MixedSoilEnv if isinstance(cfg.scene.terrain, MixedSoilTerrainCfg) else ManagerBasedRLEnv
    return env_type(cfg=cfg, **{k: v for k, v in kwargs.items() if not k.endswith("_entry_point")})
