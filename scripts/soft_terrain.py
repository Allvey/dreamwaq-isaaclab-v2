"""IsaacLab/PhysX adapter for newton-terrain's fast sand force model.

Import only after AppLauncher. No Newton solver, policy or reward code is used.
The first stage deliberately supports a flat, nominal soil surface only.
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
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_dreamwaq.rough_env_cfg import (
    Go2DreamWaQRoughEnvCfg, Go2DreamWaQRoughEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2_dreamwaq.rsl_rl_ppo_cfg import (
    Go2DreamWaQRoughPPORunnerCfg,
)


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
    proxy_path: str = os.environ.get(
        "NEWTON_TERRAIN_PATH", str(Path(__file__).resolve().parents[2] / "newton-terrain")
    )
    proxy_sha256: str = ""
    table_sha256: str = ""
    adapter_sha256: str = ""
    warp_version: str = ""
    proxy_coefficients: dict = {}
    surface_model: str = "flat nominal top; per-foot plastic memory; 16 footprint holes per environment"


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
        kernel_key = hashlib.sha256(f"{cfg.proxy_sha256}:{coefficients}".encode()).hexdigest()[:12]
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
            linecache.cache[str(source)] = (len(code), None, code.splitlines(True), str(source))
            exec(compile(code, str(source), "exec"), module.__dict__)
        self.proxy = sys.modules[module_name]
        cfg.table_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
        self.row = json.loads(table_path.read_text())[cfg.material]
        cfg.proxy_coefficients = dict(self.row)
        if cfg.bearing_scale <= 0 or cfg.drag_scale < 0 or not 0 <= cfg.footprint_keep <= 1:
            raise ValueError("Invalid soil bearing/drag/memory coefficients")
        self.n, self.nb, self.device = env.num_envs, asset.num_bodies, env.device
        legs = ("FL", "FR", "RL", "RR")
        self.feet = [asset.body_names.index(f"{leg}_foot") for leg in legs]
        self.calves = [asset.body_names.index(f"{leg}_calf") for leg in legs]
        self.thighs = [asset.body_names.index(f"{leg}_thigh") for leg in legs]
        self.base = asset.body_names.index("base")
        self.com = asset.root_physx_view.get_coms()[..., :3].to(self.device).contiguous()
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
        wp.init()
        self.table = self.proxy.make_table([self.row], device=self.device)
        def a(value, dtype=float):
            return wp.array(value, dtype=dtype, device=self.device)
        self.inputs = [
            wp.from_torch(self.q.reshape(-1, 7), wp.transform),
            wp.from_torch(self.qd.reshape(-1, 6), wp.spatial_vector),
            wp.from_torch(self.com.reshape(-1, 3), wp.vec3), self.nb,
            a(self.calves, int), a(self.feet, int), a(self.thighs, int), self.base,
            a([0.0] * self.n), a([0] * self.n, int),
            a([cfg.bearing_scale] * self.n), a([cfg.drag_scale] * self.n), self.table,
            a([cfg.footprint_keep] * self.n), 16,
            *[wp.from_torch(t.flatten()) for t in
              (self.hole_x, self.hole_y, self.hole_d, self.hole_n, self.mem, self.contact, self.depth)],
            wp.from_torch(self.wrench.reshape(-1, 6), wp.spatial_vector),
        ]
        sensor = env.scene["contact_forces"]
        sensor.robot_body_ids = [asset.body_names.index(name) for name in sensor.body_names]
        sensor.soil = self
        print(f"[Soil] PhysX + {cfg.material}, depth={cfg.depth} m; proxy={cfg.proxy_sha256[:12]}")

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.mem[ids] = -1.0
        for tensor in (self.contact, self.depth, self.hole_x, self.hole_y, self.hole_d,
                       self.hole_n, self.wrench, self.normal_w):
            tensor[ids] = 0

    def apply(self):
        data = self.asset.data
        self.q[..., :3] = data.body_pos_w - self.env.scene.env_origins[:, None, :]
        self.q[..., 3:6] = data.body_quat_w[..., 1:4]
        self.q[..., 6] = data.body_quat_w[..., 0]
        self.qd.copy_(data.body_vel_w)
        # Upstream foot drag uses velocity at the foot origin; rod wrenches use COM velocity.
        self.qd[:, self.feet, :3] += torch.cross(
            data.body_ang_vel_w[:, self.feet],
            data.body_pos_w[:, self.feet] - data.body_com_pos_w[:, self.feet], dim=-1,
        )
        self.evaluate()
        self.apply_wrench()
        self.steps += 1

    def evaluate(self):
        """Evaluate the original force law on q/qd; also used by physics acceptance fixtures."""
        self.wrench.zero_()
        wp.launch(
            self.proxy.sand_forces, dim=self.n * 4, inputs=self.inputs, device=self.device,
            stream=(wp.stream_from_torch(torch.cuda.current_stream(self.device))
                    if str(self.device).startswith("cuda") else None),
        )
        self._update_normal_contacts()

    def apply_wrench(self):
        """Apply the evaluated world-frame wrench explicitly at each body's COM."""
        # Explicit WORLD position is essential: PhysX's default application point
        # is the link origin, whereas the upstream torque is about the COM.
        # Add these forces before scene.write_data_to_sim; the original event
        # wrench buffers remain independent and are applied by IsaacLab as usual.
        self.force_w = self.wrench[..., :3].contiguous()
        self.torque_w = self.wrench[..., 3:].contiguous()
        self.asset.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.force_w, torque_data=self.torque_w,
            position_data=self.asset.data.body_com_pos_w.contiguous(),
            indices=self.asset._ALL_INDICES, is_global=True,
        )

    def _update_normal_contacts(self):
        # Normal bearing belongs to the FOOT sensor although its wrench is applied to the calf.
        # Tangential drag is not added to PhysX's normal-contact reporter.
        p, row = self.proxy, self.row
        d = self.depth.clamp_min(0)
        v = self.qd[:, self.feet, :3]
        e = d - self.mem.clamp_min(0)
        bearing = torch.where(e > 0, p.K_FLOOR * e + p.C_FLOOR * (-v[..., 2]).clamp_min(0), 0)
        pull = row["pull_p"] * d * (v[..., 2] / 0.05).clamp(0, 1)
        fz = torch.where(self.depth > 0, bearing - pull, 0)
        fh = self.cfg.drag_scale * row["drag_a"] * d.pow(row["drag_n"])
        speed = torch.linalg.vector_norm(v[..., :2], dim=-1)
        eps = (fh / p.C_MAX).clamp_min(0.1)
        lift = row["lift_r"] * fh * speed / torch.sqrt(speed.square() + eps.square())
        self.normal_w.zero_()
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


def configure_soil_timing(cfg):
    # Converged PhysX/proxy coupling; retain the original 20 ms policy period
    # and 15 ms contact-history window in both soft and matched rigid tasks.
    cfg.sim.dt = 0.0003125
    cfg.decimation = cfg.sim.render_interval = 64
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.scene.contact_forces.history_length = 48
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
        if cfg.scene.terrain.terrain_type != "plane":
            raise ValueError("The first-stage soil proxy supports flat terrain only")
        cfg.actions.joint_pos.class_type = SoilJointPositionAction
        cfg.scene.contact_forces.class_type = SoilContactSensor
        cfg.scene.height_scanner.mesh_prim_paths = ["/World/soil_surface"]


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


@configclass
class Go2SoftRunnerCfg(Go2DreamWaQRoughPPORunnerCfg):
    # Warm start into a new run; distinct from resuming optimizer/iteration state.
    # IsaacLab validates from_dict values against the default's runtime type.
    init_checkpoint: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "isaaclab/go2_dreamwaq_soft_flat"


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


def make_env(cfg, **kwargs):
    from isaaclab.envs import ManagerBasedRLEnv
    enable_soil(cfg)
    return ManagerBasedRLEnv(cfg=cfg, **{k: v for k, v in kwargs.items() if not k.endswith("_entry_point")})
