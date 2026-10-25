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
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg, CurriculumTermCfg
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
    active_out[i] = int(active != 0)


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
    enabled[w * 5 + leg] = int(active != 0)
    typ = wp.max(active - 1, 0)
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
        enabled[w * 5 + 4] = int(active_b != 0)
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


def soil_level_parameters(cfg, row, num_rows):
    """Logical curriculum level; independent of the number of physical tiles."""
    difficulty = row / max(num_rows - 1, 1) if cfg.soft_curriculum else 1.0
    geometry = cfg.soft_min_geometry_scale + (1.0 - cfg.soft_min_geometry_scale) * difficulty
    bearing = cfg.soft_easy_bearing_scale ** (1.0 - difficulty)
    # Keep material transitions on grid edges, including both stripe boundaries.
    width = cfg.soft_easy_stripe_width + (cfg.soft_stripe_width - cfg.soft_easy_stripe_width) * difficulty
    width = max(2 * cfg.soft_resolution, round(width / (2 * cfg.soft_resolution)) * 2 * cfg.soft_resolution)
    return geometry, bearing, width


def soil_curriculum_spec(cfg):
    keys = ("soft_columns", "soft_scenarios", "soft_border", "soft_resolution", "soft_mound_height",
            "soft_basin_depth", "soft_wave_height", "soft_wave_length", "soft_stripe_width",
            "soft_min_geometry_scale", "soft_easy_bearing_scale", "soft_easy_stripe_width",
            "depth", "material", "bearing_scale", "drag_scale", "footprint_keep",
            "floor_stiffness", "normal_damping", "drag_damping_cap")
    return {**{key: getattr(cfg, key) for key in keys},
            "rows": cfg.soft_num_levels if cfg.soft_spatial_curriculum else 1, "levels": cfg.soft_num_levels,
            "layout": "spatial_soil_curriculum_v1" if cfg.soft_spatial_curriculum else "five_tiles_fixed_level_v1",
            "tile_size": list(cfg.terrain_generator.size)}


def soil_stage_state(state, cfg):
    """Migrate the former spatial-row metadata without changing model weights."""
    from copy import deepcopy
    if state is None:
        return None
    state = deepcopy(state)
    topology = state["topology"]
    expected = soil_curriculum_spec(cfg)
    if "layout" not in topology:
        topology = dict(topology, rows=1, levels=topology["rows"], layout=expected["layout"])
    if topology != expected:
        raise ValueError("Saved soil geometry/material/level definitions differ from the current environment")
    state["topology"] = expected
    state.pop("levels_by_column", None)
    level = state.get("soft_level_ceiling", 0)
    if type(level) is not int or not 0 <= level < cfg.soft_num_levels:
        raise ValueError("Invalid saved logical soil level")
    state["soft_level_ceiling"] = level
    return state


def soil_cell_grid(cfg, scenario, row=None):
    """Same lattice for rendering, collision, height scans and graded soil forces."""
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
    geometry, _, stripe_width = (soil_level_parameters(cfg, row, cfg.soft_num_levels)
                                if row is not None else (1.0, 1.0, cfg.soft_stripe_width))
    if scenario == "mound":
        z = geometry * cfg.soft_mound_height * window
    elif scenario == "basin":
        z = -geometry * cfg.soft_basin_depth * window
    elif scenario == "undulating":
        z = geometry * cfg.soft_wave_height * window * np.cos(2 * np.pi * xx / cfg.soft_wave_length) * np.cos(2 * np.pi * yy / cfg.soft_wave_length)
    xc, yc = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, indexing="ij")
    inner = (np.abs(xc) < ax) & (np.abs(yc) < ay)
    active = inner.copy()
    if scenario == "mixed":
        active &= np.floor((xc + stripe_width / 2) / stripe_width).astype(int) % 2 == 0
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
        if cfg.training_stage not in ("mixed", "hard", "soft"):
            raise ValueError("training_stage must be mixed, hard or soft")
        if type(cfg.soft_columns) is not int or not 1 <= cfg.soft_columns <= 5:
            raise ValueError("Use between one and five physical soil tiles")
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
        if not (0 < cfg.soft_min_geometry_scale <= 1 and math.isfinite(cfg.soft_easy_bearing_scale)
                and cfg.soft_easy_bearing_scale >= 1 and cfg.soft_easy_stripe_width >= cfg.soft_stripe_width
                and math.isfinite(cfg.soft_easy_stripe_width)
                and type(cfg.soft_max_init_level) is int and cfg.soft_max_init_level >= 0
                and type(cfg.soft_num_levels) is int and cfg.soft_num_levels >= 2
                and type(cfg.soft_level_ceiling) is int and 0 <= cfg.soft_level_ceiling < cfg.soft_num_levels
                and 0 <= cfg.soft_min_contact_fraction <= 1):
            raise ValueError("Invalid soft curriculum geometry, bearing, stripe, initial level or contact fraction")
        if abs(cfg.soft_easy_stripe_width / (2 * cfg.soft_resolution)
               - round(cfg.soft_easy_stripe_width / (2 * cfg.soft_resolution))) > 1e-6:
            raise ValueError("Easy half stripe width must align with soft_resolution")
        self.soft_level = cfg.soft_level_ceiling if cfg.soft_curriculum else cfg.soft_num_levels - 1
        rows = cfg.soft_num_levels if cfg.soft_spatial_curriculum else 1
        if cfg.soft_spatial_curriculum and not (cfg.isolated_training and cfg.training_stage == "soft"):
            raise ValueError("Spatial soil curriculum uses an isolated soft-only scene")
        self.num_soft = (0 if cfg.training_stage == "hard" else cfg.num_envs if cfg.training_stage == "soft"
                         else round(cfg.num_envs * cfg.soft_fraction))
        self.num_rough = cfg.num_envs - self.num_soft
        if cfg.training_stage == "mixed" and min(self.num_soft, self.num_rough) < 1:
            raise ValueError("Mixed terrain needs at least one environment in each group")
        # IsaacLab 2.0 terrain functions also use global NumPy/Torch RNGs.
        # Retain their inputs so geometry can be regenerated in parity audits.
        self._rough_numpy_state = np.random.get_state()
        self._rough_torch_state = torch.get_rng_state()
        self._rough_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        isolated = cfg.isolated_training
        if isolated:
            if cfg.training_stage != "soft":
                raise ValueError("Isolated hard training must use the native TerrainImporter")
            # Initialize the importer without generating/importing any rough
            # terrain. Only the soil bottom, frames and outer border collide.
            cfg.validate()
            self.cfg = cfg
            self.device = sim_utils.SimulationContext.instance().device
            self.meshes, self.warp_meshes, self._terrain_flat_patches = {}, {}, {}
            self.env_origins = torch.zeros((cfg.num_envs, 3), device=self.device)
            self.terrain_origins = None
            self.terrain_levels = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
            self.terrain_types = torch.arange(cfg.num_envs, device=self.device) % cfg.soft_columns
            self.max_terrain_level = rows
            rough = None
        else:
            super().__init__(cfg)
            rough = self.meshes["terrain"]
            cfg.rough_mesh_sha256 = hashlib.sha256(
                rough.vertices.tobytes() + rough.faces.tobytes()
            ).hexdigest()
        self.soft_mask = torch.arange(cfg.num_envs, device=self.device) >= self.num_rough
        # Distribute the rough group over ALL original terrain columns, rather
        # than retaining only the first half of the original assignments.
        if self.num_rough:
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
        lo = np.array([-rows * sx / 2,
                       -cfg.soft_columns * sy / 2 if isolated else generator.num_cols * sy / 2])
        hi = lo + np.array([rows * sx, cfg.soft_columns * sy])
        tops, centres, rim = [], [], []
        nx, ny = round(sx / cfg.soft_resolution), round(sy / cfg.soft_resolution)
        height_grid = np.zeros((rows * nx + 1, cfg.soft_columns * ny + 1), dtype=np.float32)
        material_grid = np.zeros((rows * nx, cfg.soft_columns * ny), dtype=np.int32)
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        a = (i * (ny + 1) + j).flatten()
        faces = np.stack((np.stack((a, a + ny + 1, a + 1), axis=1),
                          np.stack((a + ny + 2, a + 1, a + ny + 1), axis=1)), axis=1).reshape(-1, 3)
        self.soft_scenario_names = [cfg.soft_scenarios[col % len(cfg.soft_scenarios)] for col in range(cfg.soft_columns)]
        self.soft_curriculum_active = cfg.soft_curriculum and generator.curriculum
        self.soft_bearing_scales = [soil_level_parameters(
            cfg, row if cfg.soft_spatial_curriculum else self.soft_level, cfg.soft_num_levels)[1] for row in range(rows)]
        self.soft_contact_steps = torch.zeros(cfg.num_envs, device=self.device)
        self.soft_sample_steps = torch.zeros_like(self.soft_contact_steps)

        def rectangle(lower, upper):
            lower, upper = np.asarray(lower), np.asarray(upper)
            if np.all(upper > lower):
                plane = make_plane(tuple(upper - lower), height=0.0)
                plane.apply_translation([*((lower + upper) / 2), 0.0])
                rim.append(plane)

        for row in range(rows):
            for col in range(cfg.soft_columns):
                centre = [lo[0] + (row + 0.5) * sx, lo[1] + (col + 0.5) * sy, 0.0]
                xx, yy, zz, active, inner = soil_cell_grid(
                    cfg, self.soft_scenario_names[col], row if cfg.soft_spatial_curriculum else self.soft_level)
                vertices = np.stack((xx + centre[0], yy + centre[1], zz), axis=-1).reshape(-1, 3)
                tile = trimesh.Trimesh(vertices=vertices, faces=faces[np.repeat(active.flatten(), 2)], process=False)
                tile.remove_unreferenced_vertices()
                hard_faces = faces[np.repeat((inner & ~active).flatten(), 2)]
                if len(hard_faces):
                    hard = trimesh.Trimesh(vertices=vertices, faces=hard_faces, process=False)
                    hard.remove_unreferenced_vertices()
                    rim.append(hard)
                height_grid[row * nx:(row + 1) * nx + 1, col * ny:(col + 1) * ny + 1] = zz
                # Actual foot location selects the real row's material table.
                material_grid[row * nx:(row + 1) * nx, col * ny:(col + 1) * ny] = active * (row + 1)
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
        if isolated:
            self.terrain_origins = self.soft_tile_origins.view(rows, cfg.soft_columns, 3)
        sample = torch.arange(self.num_soft, device=self.device)
        soft_levels = torch.full((self.num_soft,), self.soft_level, dtype=torch.long, device=self.device)
        if cfg.soft_spatial_curriculum:
            soft_levels = torch.randint(min(cfg.soft_max_init_level, rows - 1) + 1,
                                        (self.num_soft,), device=self.device)
        self.soft_tile_ids = (soft_levels * cfg.soft_columns if cfg.soft_spatial_curriculum else 0) + sample % cfg.soft_columns
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
        self.terrain_levels[self.soft_mask] = soft_levels
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
        grid_lo = lo if isolated else np.array([-generator.num_rows * sx / 2, -generator.num_cols * sy / 2])
        grid_hi = hi if isolated else np.array([generator.num_rows * sx / 2, hi[1]])
        if not isolated:
            # Fill the unused ends beside the short soil strip. Keep the rough
            # perimeter flat without placing a rigid surface underneath soil.
            rectangle([grid_lo[0], lo[1]], [lo[0], hi[1]])
            rectangle([hi[0], lo[1]], grid_hi)
        rectangle(grid_lo - margin, [grid_lo[0], grid_hi[1] + margin])
        rectangle([grid_hi[0], grid_lo[1] - margin], grid_hi + margin)
        rectangle([grid_lo[0], grid_lo[1] - margin], [grid_hi[0], grid_lo[1]])
        rectangle([grid_lo[0], grid_hi[1]], [grid_hi[0], grid_hi[1] + margin])
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
        sensor_mesh("/World/mixed_height_scan", trimesh.util.concatenate(
            [top, rim_mesh] if isolated else [rough, top, rim_mesh]))
        if isolated:
            self.set_debug_vis(cfg.debug_vis)
        cfg.resolved_num_rough, cfg.resolved_num_soft = self.num_rough, self.num_soft
        cfg.resolved_soft_tiles = len(tops)
        cfg.resolved_soft_scenarios = self.soft_scenario_names
        cfg.soft_surface_sha256 = hashlib.sha256(height_grid.tobytes() + material_grid.tobytes()).hexdigest()
        cfg.adapter_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print(f"[Mixed soil] stage={cfg.training_stage}; rough={self.num_rough}, soft={self.num_soft}; "
              f"sand tiles={rows}x{cfg.soft_columns}, size={sx}x{sy} m; "
              f"flat frame={cfg.soft_border} m per side; "
              f"scenarios={','.join(self.soft_scenario_names)}; soft curriculum={self.soft_curriculum_active}; "
              f"rough mesh={cfg.rough_mesh_sha256[:12]}", flush=True)

    def update_env_origins(self, env_ids, move_up, move_down):
        ids = (torch.arange(self.cfg.num_envs, device=self.device)[env_ids]
               if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        rough = ~self.soft_mask[ids]
        if rough.any():
            super().update_env_origins(ids[rough], move_up[rough], move_down[rough])
        if self.cfg.soft_spatial_curriculum and (~rough).any():
            self.update_soft_origins(ids[~rough], move_up[~rough], move_down[~rough])

    def update_soft_origins(self, ids, move_up, move_down):
        soft_ids = ids - self.num_rough
        if self.cfg.soft_spatial_curriculum:
            levels = self.terrain_levels[ids] + move_up.long() - move_down.long()
            # Match TerrainImporter.update_env_origins, using the soil row count.
            levels = torch.where(levels >= self.cfg.soft_num_levels,
                                 torch.randint_like(levels, self.cfg.soft_num_levels), levels.clamp(min=0))
            self.terrain_levels[ids] = levels
            columns = self.soft_tile_ids[soft_ids] % self.cfg.soft_columns
            self.soft_tile_ids[soft_ids] = levels * self.cfg.soft_columns + columns
        else:
            self.terrain_levels[ids] = self.soft_level
        self.env_origins[ids] = self.soft_tile_origins[self.soft_tile_ids[soft_ids]]
        self.soft_origins[soft_ids] = self.env_origins[ids]

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
    physics_substeps: int = 64
    training_stage: str = "mixed"
    isolated_training: bool = False
    stage_boundary_margin: float = 0.6
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
    soft_curriculum: bool = True
    soft_max_init_level: int = 0
    soft_level_ceiling: int = 0
    soft_num_levels: int = 10
    soft_spatial_curriculum: bool = False
    soft_min_geometry_scale: float = 0.1
    soft_easy_bearing_scale: float = 4.0
    soft_easy_stripe_width: float = 3.0
    soft_min_contact_fraction: float = 0.2
    resolved_soft_scenarios: list[str] = []
    soft_surface_sha256: str = ""
    resolved_soft_tiles: int = 0
    resolved_num_rough: int = 0
    resolved_num_soft: int = 0
    rough_mesh_sha256: str = ""
    source_rough_mesh_sha256: str = ""
    surface_model: str = "at most five soil tiles at one fixed logical level; local tangent-plane proxy; flat frames and border"


def mixed_terrain_levels(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel
    terrain = env.scene.terrain
    ids = (torch.arange(env.num_envs, device=env.device)[env_ids]
           if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    rough_ids, soft_ids = ids[~terrain.soft_mask[ids]], ids[terrain.soft_mask[ids]]
    if len(rough_ids):
        terrain_levels_vel(env, rough_ids, asset_cfg=asset_cfg)
    if len(soft_ids) and terrain.soft_curriculum_active:
        if terrain.cfg.soft_spatial_curriculum:
            # Use the exact native distance/current-command rules. Initial or
            # explicit resets without an elapsed episode must not promote robots.
            active = soft_ids[env.episode_length_buf[soft_ids] > 0]
            if len(active):
                terrain_levels_vel(env, active, asset_cfg=asset_cfg)
        terrain.soft_contact_steps[soft_ids] = 0
        terrain.soft_sample_steps[soft_ids] = 0
    return (terrain.terrain_levels[~terrain.soft_mask].float().mean() if terrain.num_rough
            else terrain.terrain_levels.float().mean() if terrain.cfg.soft_spatial_curriculum
            else torch.zeros((), device=env.device))


def outside_training_region(env):
    """Truncate before feet reach another domain; count this as a timeout, not a fall."""
    terrain = env.scene.terrain
    cfg = terrain.cfg
    sx, sy = cfg.terrain_generator.size
    lower = terrain.soft_bounds[0].clone()
    upper = terrain.soft_bounds[1].clone()
    if cfg.training_stage == "hard":
        lower[0] = -cfg.terrain_generator.num_rows * sx / 2
        upper[0] = cfg.terrain_generator.num_rows * sx / 2
        lower[1] = -cfg.terrain_generator.num_cols * sy / 2
        upper[1] = cfg.terrain_generator.num_cols * sy / 2
    xy = env.scene["robot"].data.body_pos_w[..., :2]
    margin = cfg.stage_boundary_margin
    return ((xy < lower + margin) | (xy > upper - margin)).any(dim=-1).any(dim=-1)


class MixedSoilEnv(ManagerBasedRLEnv):
    def step(self, action):
        obs, reward, terminated, timeout, extras = super().step(action)
        extras["log"] = dict(extras["log"])
        soft = self.scene.terrain.soft_mask
        for name, mask in (("rough", ~soft), ("soft", soft)):
            if (name == "rough" and self.scene.terrain.num_rough == 0) or (name == "soft" and self.scene.terrain.num_soft == 0):
                continue
            extras["log"][f"Mixed/{name}_reward_per_second"] = reward[mask].mean() / self.step_dt
            extras["log"][f"Mixed/{name}_termination_fraction"] = terminated[mask].float().mean()
            extras["log"][f"Mixed/{name}_timeout_fraction"] = timeout[mask].float().mean()
        extras["log"]["Mixed/soft_sample_fraction"] = soft.float().mean()
        if "stage_boundary" in self.termination_manager.active_terms:
            extras["log"]["Stage/boundary_fraction"] = self.termination_manager.get_term("stage_boundary").float().mean()
        for name, mask in self.scene.terrain.scenario_masks.items():
            extras["log"][f"SoilScenario/{name}_reward_per_second"] = reward[mask].mean() / self.step_dt
            extras["log"][f"SoilScenario/{name}_termination_fraction"] = terminated[mask].float().mean()
            extras["log"][f"SoilScenario/{name}_timeout_fraction"] = timeout[mask].float().mean()
            extras["log"][f"SoilCurriculum/{name}_level"] = self.scene.terrain.terrain_levels[mask].float().mean()
            if self.scene.terrain.cfg.soft_spatial_curriculum:
                error = (self.scene["robot"].data.root_lin_vel_b[mask, :2]
                         - self.command_manager.get_command("base_velocity")[mask, :2])
                extras["log"][f"SoilScenario/{name}_tracking_rmse_xy"] = error.square().sum(-1).mean().sqrt()
                soil = self.action_manager.get_term("joint_pos").soil
                extras["log"][f"SoilScenario/{name}_soft_contact_fraction"] = soil.contact[mask].bool().any(-1).float().mean()
                for level in range(self.scene.terrain.cfg.soft_num_levels):
                    extras["log"][f"SoilCurriculum/{name}_level_{level}_fraction"] = (
                        self.scene.terrain.terrain_levels[mask] == level).float().mean()
        if self.scene.terrain.num_soft:
            extras["log"]["SoilCurriculum/mean_level"] = self.scene.terrain.terrain_levels[soft].float().mean()
            cfg = self.scene.terrain.cfg
            extras["log"]["SoilCurriculum/level_ceiling"] = float(
                cfg.soft_num_levels - 1 if cfg.soft_spatial_curriculum else cfg.soft_level_ceiling)
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
        rows = [dict(self.row, bear_a=self.row["bear_a"] * scale)
                for scale in getattr(env.scene.terrain, "soft_bearing_scales", [1.0])]
        self.table = self.proxy.make_table(rows, device=self.device)
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
        if (self.surface_terrain is not None and self.surface_terrain.soft_curriculum_active
                and self.env._sim_step_counter % self.env.cfg.decimation == 0):
            # Once per policy step; retain statistics until the pre-reset
            # curriculum callback, without extra PhysX reads or GPU/CPU syncs.
            self.surface_terrain.soft_contact_steps.add_((self.contact > 0).any(dim=1))
            self.surface_terrain.soft_sample_steps.add_(1)
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
    # 64 is the converged reference; smaller counts are explicit speed experiments.
    # All retain the original 20 ms policy period and 15 ms contact window.
    if type(substeps) is not int or substeps not in (4, 16, 32, 64):
        raise ValueError("Soil physics_substeps must be 4, 16, 32 or 64")
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
            stage = cfg.scene.terrain.training_stage
            if stage not in ("mixed", "hard", "soft"):
                raise ValueError("training_stage must be mixed, hard or soft")
            if not 0 < cfg.scene.terrain.stage_boundary_margin < min(cfg.scene.terrain.terrain_generator.size) / 4:
                raise ValueError("stage_boundary_margin must be positive and smaller than a quarter tile")
            if (stage != "mixed" and not (stage == "hard" and cfg.scene.terrain.isolated_training)
                    and not cfg.scene.terrain.soft_spatial_curriculum):
                cfg.terminations.stage_boundary = TerminationTermCfg(func=outside_training_region, time_out=True)
            if stage == "hard":
                cfg.sim.dt = 0.005
                cfg.decimation = cfg.sim.render_interval = 4
                cfg.scene.contact_forces.update_period = cfg.sim.dt
                cfg.scene.contact_forces.history_length = 3
                cfg.scene.height_scanner.update_period = 0.02
                cfg.actions.joint_pos.class_type = JointPositionAction
                cfg.scene.contact_forces.class_type = ContactSensor
                if cfg.scene.terrain.isolated_training:
                    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel
                    cfg.scene.terrain.class_type = TerrainImporter
                    if cfg.curriculum.terrain_levels is not None:
                        cfg.curriculum.terrain_levels.func = terrain_levels_vel
                    if hasattr(cfg.terminations, "stage_boundary"):
                        cfg.terminations.stage_boundary = None
                    cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
                else:
                    cfg.scene.height_scanner.mesh_prim_paths = ["/World/mixed_height_scan"]
                return
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


def configure_spatial_soil(cfg, play=False):
    terrain = cfg.scene.terrain
    terrain.training_stage, terrain.isolated_training = "soft", True
    terrain.soft_spatial_curriculum, terrain.soft_curriculum = True, True
    terrain.soft_num_levels, terrain.soft_columns, terrain.soft_max_init_level = 5, 5, 1
    terrain.terrain_generator.num_rows = terrain.terrain_generator.num_cols = 5
    terrain.terrain_generator.curriculum = not play
    terrain.surface_model = "five difficulty rows x five scenario columns; native distance curriculum"
    cfg.curriculum.terrain_levels = None if play else CurriculumTermCfg(func=mixed_terrain_levels)


@configclass
class Go2SoilCurriculumEnvCfg(Go2MixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_spatial_soil(self)


@configclass
class Go2SoilCurriculumPlayEnvCfg(Go2MixedPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_spatial_soil(self, play=True)
        self.scene.num_envs = 25


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
    stage_report: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isaaclab/go2_dreamwaq_mixed"


@configclass
class Go2SoilCurriculumRunnerCfg(Go2SoftRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isaaclab/go2_dreamwaq_soft_curriculum"


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
    mixed = isinstance(env.unwrapped.cfg.scene.terrain, MixedSoilTerrainCfg)
    runner_type = OnPolicyRunner
    spatial = mixed and env.unwrapped.cfg.scene.terrain.soft_spatial_curriculum
    if spatial:
        class SoilCurriculumRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                terrain = self.env.unwrapped.scene.terrain
                metadata = {"topology": soil_curriculum_spec(terrain.cfg),
                            "physics_dt": self.env.unwrapped.physics_dt,
                            "levels": terrain.terrain_levels.detach().cpu().tolist()}
                if not hasattr(self, "logger_type"):
                    self.logger_type = "tensorboard"
                return super().save(path, dict(infos or {}, dreamwaq_soil_curriculum=metadata))

            def load(self, path, load_optimizer=True, map_location=None):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                metadata = (saved.get("infos") or {}).get("dreamwaq_soil_curriculum")
                terrain = self.env.unwrapped.scene.terrain
                if metadata is None and self.cfg.get("resume"):
                    raise ValueError("Resume needs a checkpoint from the spatial soil curriculum task")
                if metadata is not None:
                    if (metadata["topology"] != soil_curriculum_spec(terrain.cfg)
                            or metadata["physics_dt"] != self.env.unwrapped.physics_dt):
                        raise ValueError("Saved spatial soil layout/material/timing differs from this environment")
                    old = torch.tensor(metadata["levels"], dtype=torch.long, device=terrain.device)
                    if not len(old) or (old < 0).any() or (old >= terrain.cfg.soft_num_levels).any():
                        raise ValueError("Invalid saved soil curriculum levels")
                result = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
                if metadata is not None:
                    # Same robot count restores exact levels. For visualization
                    # with fewer robots, cycle the saved levels within each column.
                    for col in range(terrain.cfg.soft_columns):
                        ids = torch.arange(col, self.env.num_envs, terrain.cfg.soft_columns, device=terrain.device)
                        pool = old[col::terrain.cfg.soft_columns]
                        if len(ids):
                            terrain.terrain_levels[ids] = (pool[torch.arange(len(ids), device=terrain.device) % len(pool)]
                                                          if len(pool) else 0)
                    terrain.soft_tile_ids[:] = (terrain.terrain_levels * terrain.cfg.soft_columns
                                               + torch.arange(self.env.num_envs, device=terrain.device) % terrain.cfg.soft_columns)
                    ids = torch.arange(self.env.num_envs, device=terrain.device)
                    zeros = torch.zeros_like(ids, dtype=torch.bool)
                    terrain.update_soft_origins(ids, zeros, zeros)
                    active = terrain.soft_curriculum_active
                    terrain.soft_curriculum_active = False
                    try:
                        with torch.inference_mode():
                            self.env.reset()
                    finally:
                        terrain.soft_curriculum_active = active
                return result

        runner_type = SoilCurriculumRunner
    elif mixed:
        from copy import deepcopy
        from run_soft import validate_stage_report

        class StageRunner(OnPolicyRunner):
            """Only checkpoint metadata and stage admission; native PPO/learn are unchanged."""
            stage_state = None

            def load(self, path, load_optimizer=True, map_location=None):
                terrain = self.env.unwrapped.scene.terrain
                report = None
                if self.log_dir is not None and terrain.cfg.training_stage != "hard":
                    report_path = self.cfg.get("stage_report", "")
                    if not report_path:
                        raise ValueError("Soft/mixed training needs --stage-report from mixed-eval for this checkpoint")
                    report = json.loads(Path(report_path).read_text())
                    validate_stage_report(report, path)
                    if (report.get("soft") is None or report["soft"]["terrain_spec"] != self._topology()
                            or report["soft"]["physics_dt"] != self.env.unwrapped.physics_dt):
                        raise ValueError("Soft evaluation geometry/material/timestep must match the training environment")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                state = soil_stage_state((saved.get("infos") or {}).get("dreamwaq_stage"), terrain.cfg)
                level = report["decision"]["next_soft_level"] if report is not None else (
                    state["soft_level_ceiling"] if state is not None else 0)
                if (state is not None or report is not None) and getattr(terrain, "num_soft", 0) and terrain.soft_level != level:
                    raise ValueError("Soil level differs from the already-built mesh; rebuild the environment at the checkpoint/report level")
                result = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
                self.stage_state = deepcopy(state) if state is not None else None
                if report is not None:
                    if self.stage_state is None:
                        self.stage_state = {"topology": self._topology(), "soft_started": False}
                    self.stage_state["hard_reference"] = report["hard_reference"]
                    self.stage_state["soft_level_ceiling"] = report["decision"]["next_soft_level"]
                if self.stage_state is not None and getattr(terrain, "num_soft", 0) and terrain.soft_curriculum_active:
                    ceiling = self.stage_state["soft_level_ceiling"]
                    terrain.cfg.soft_level_ceiling = ceiling
                    ids = terrain.soft_mask.nonzero().flatten()
                    zeros = torch.zeros_like(ids, dtype=torch.bool)
                    terrain.update_soft_origins(ids, zeros, zeros)
                    terrain.soft_contact_steps.zero_()
                    terrain.soft_sample_steps.zero_()
                    # Native spawn events use the restored origins; partial soil
                    # memory is deliberately reset rather than replayed mid-step.
                    with torch.inference_mode():
                        self.env.reset()
                    print(f"[Stage curriculum] fixed level={ceiling}; five-tile geometry and scenario assignment restored", flush=True)
                return result

            def _topology(self):
                return soil_curriculum_spec(self.env.unwrapped.scene.terrain.cfg)

            def save(self, path, infos=None):
                terrain = self.env.unwrapped.scene.terrain
                state = deepcopy(self.stage_state) if self.stage_state is not None else {
                    "topology": self._topology(), "soft_started": False, "soft_level_ceiling": 0}
                if getattr(terrain, "num_soft", 0) and terrain.soft_curriculum_active:
                    state["soft_started"] = True
                    state["soft_level_ceiling"] = terrain.soft_level
                # Metadata travels through hard rehearsal checkpoints as well.
                merged = dict(infos or {}, dreamwaq_stage=state)
                if not hasattr(self, "logger_type"):
                    self.logger_type = "tensorboard"
                return super().save(path, infos=merged)

        runner_type = StageRunner
        if log_dir is not None and env.unwrapped.cfg.scene.terrain.training_stage != "hard" and not (checkpoint or train_cfg.get("resume")):
            raise ValueError("Begin from scratch with --stage hard; evaluate it before starting a soft stage")
    runner = runner_type(env, train_cfg, log_dir=log_dir, device=device)
    if checkpoint:
        runner.load(str(checkpoint), load_optimizer=False)
        runner.current_learning_iteration = 0
        print(f"[Soil fine-tune] Loaded network weights: {checkpoint}", flush=True)
        print("[Soil fine-tune] Fresh optimizer; new iteration counter starts at 0", flush=True)
    return runner


def make_env(cfg, overview=False, stage=None, physics_substeps=None, isolated_training=False,
             stage_report=None, soft_level=None, **kwargs):
    if stage is not None:
        cfg.scene.terrain.training_stage = stage
    if physics_substeps is not None:
        cfg.scene.terrain.physics_substeps = physics_substeps
    if isinstance(cfg.scene.terrain, MixedSoilTerrainCfg):
        cfg.scene.terrain.isolated_training = isolated_training or cfg.scene.terrain.isolated_training
        if cfg.scene.terrain.isolated_training and cfg.scene.terrain.training_stage == "mixed":
            raise ValueError("Training uses separate hard/soft scenes; use mixed-play for the combined scene")
        if soft_level is not None:
            cfg.scene.terrain.soft_level_ceiling = soft_level
        if isolated_training and cfg.scene.terrain.training_stage == "soft" and not cfg.scene.terrain.soft_spatial_curriculum:
            if not stage_report:
                raise ValueError("Soft training needs --stage-report before its fixed-level terrain is built")
            from run_soft import validate_stage_report
            report = json.loads(Path(stage_report).read_text())
            validate_stage_report(report, report["checkpoint"])
            if report.get("soft") is None or report["soft"]["terrain_spec"] != soil_curriculum_spec(cfg.scene.terrain):
                raise ValueError("Re-run mixed-eval: a matching five-tile soil report is required")
            cfg.scene.terrain.soft_level_ceiling = report["decision"]["next_soft_level"]
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
    native_hard = (isinstance(cfg.scene.terrain, MixedSoilTerrainCfg)
                   and cfg.scene.terrain.isolated_training and cfg.scene.terrain.training_stage == "hard")
    env_type = MixedSoilEnv if isinstance(cfg.scene.terrain, MixedSoilTerrainCfg) and not native_hard else ManagerBasedRLEnv
    return env_type(cfg=cfg, **{k: v for k, v in kwargs.items() if not k.endswith("_entry_point")})
