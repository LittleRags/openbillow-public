# SPDX-License-Identifier: GPL-3.0-or-later
"""The smoke solver: Blender's Mantaflow gas step order on top of a `Backend`.

Settings are in Blender units (metres, seconds, frames). This module does the
scaling to Mantaflow's cell / solver-time units (reference/NOTES.md) and calls
only the backend interface, never a GPU library.

Frame loop: Blender `manta_step` (fluid.cc) + `FluidSolver::adaptTimestep`.
Substep: `smoke_adaptive_step` + `smoke_step` (smoke_script.h).
Noise (upres): `NoisePass`, Blender's `smoke_step_noise` after each frame.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import estimate as _estimate
from . import noise as _noise
from .backend import Backend
from .backend.base import Flags, Vec3
from .forcefields import EffectorWeights, ForceField, interp_matrix
from .guiding import make_guiding
from .meshes import cube, uv_sphere

DT_DEFAULT = 0.1      # fluid.cc: frame length is 0.1 solver units at 25 fps
FLT_EPSILON = 1.1920929e-07
VECTOR_EPSILON = 1e-6


class SimulationDiverged(RuntimeError):
    """The state went to NaN or infinity in `frame` (the frame being computed)."""

    def __init__(self, frame: int):
        super().__init__(f"the simulation became unstable in frame {frame}")
        self.frame = frame


@dataclass
class MeshObject:
    """A triangle mesh in object space plus its world matrix: one (4, 4)
    matrix, or (n, 4, 4) samples at scene times `frames` (frames, may be
    subframes), interpolated linearly and held outside the range."""
    verts: np.ndarray               # (n, 3) object space, metres
    tris: np.ndarray                # (m, 3) vertex indices, outward winding
    matrix: np.ndarray              # (4, 4) or (k, 4, 4) object → world
    frames: np.ndarray | None = None

    @property
    def animated(self) -> bool:
        return np.ndim(self.matrix) == 3 and len(self.matrix) > 1

    def matrix_at(self, frame: float) -> np.ndarray:
        return interp_matrix(self.matrix, self.frames, frame)


@dataclass
class MeshFlow(MeshObject):
    """A smoke/fire source (Blender flow object, behaviour Inflow, absolute
    density, source Mesh)."""
    flow_type: str = "SMOKE"        # Blender flow type: SMOKE, FIRE or BOTH (fire + smoke)
    density: float = 1.0
    temperature: float = 1.0
    surface_distance: float = 1.0   # cells, as Blender's flow setting
    volume_density: float = 0.0     # emission inside the mesh (Blender "Volume Emission")
    plane_init: bool = False        # Blender "Is Planar"
    fuel_amount: float = 1.0        # Blender "Flame Rate"
    emit_start: int = 0             # emits while emit_start <= frame <= emit_end (emit_end 0: always);
    emit_end: int = 0               # a short range = burst/explosion


@dataclass
class MeshObstacle(MeshObject):
    """A collision object (Blender effector, type Collision). With matrix
    samples it moves and pushes the smoke (obstacle velocity)."""
    surface_thickness: float = 0.0  # cells
    plane_init: bool = False


@dataclass
class MeshGuide(MeshObstacle):
    """A guide object (Blender effector, type Guide): not an obstacle; its
    velocity inside the mesh becomes the guiding target.
    `surface_thickness` is the effector's Surface Thickness."""
    guide_mode: str = "OVERRIDE"    # OVERRIDE | AVERAGED | MAX | MIN
    vel_multi: float = 1.0


def _trs(center: Vec3, scale: Vec3) -> np.ndarray:
    m = np.diag([*scale, 1.0])
    m[:3, 3] = center
    return m


def keyed(center: Vec3, scale: Vec3, center_end: Vec3 | None, frame_start: float,
           frame_end: float, interpolation: str, samples_per_frame: int):
    """(matrix, frames): one matrix, or the location keyframed from center
    (frame_start) to center_end (frame_end) and sampled `samples_per_frame`
    times per frame. BEZIER is Blender's keyframe default (auto-clamped
    handles: flat ends, 1/3 handle length), so x(u) = 3u² - 2u³ with the
    frame linear in u."""
    if center_end is None:
        return _trs(center, scale), None
    frames = np.linspace(frame_start, frame_end,
                         int(round((frame_end - frame_start) * samples_per_frame)) + 1)
    u = (frames - frame_start) / (frame_end - frame_start)
    if interpolation == "BEZIER":
        u = u * u * (3.0 - 2.0 * u)
    c0, c1 = np.asarray(center, float), np.asarray(center_end, float)
    return np.array([_trs(c0 + (c1 - c0) * ui, scale) for ui in u]), frames


def SphereFlow(center: Vec3, radius: float, center_end: Vec3 | None = None,
               frame_start: float = 1.0, frame_end: float = 60.0,
               interpolation: str = "BEZIER", samples_per_frame: int = 4,
               **flow) -> MeshFlow:
    """Blender's default UV sphere (32 segments, 16 rings) as a flow,
    optionally keyframed (see `keyed`)."""
    verts, tris = uv_sphere()
    m, frames = keyed(center, (radius,) * 3, center_end, frame_start, frame_end,
                       interpolation, samples_per_frame)
    return MeshFlow(verts, tris, m, frames, **flow)


def BoxObstacle(center: Vec3, size: Vec3, center_end: Vec3 | None = None,
                frame_start: float = 1.0, frame_end: float = 60.0,
                interpolation: str = "BEZIER", samples_per_frame: int = 4,
                **obstacle) -> MeshObstacle:
    """Blender's default cube scaled to full edge lengths `size` (metres),
    optionally keyframed (see `keyed`)."""
    verts, tris = cube()
    m, frames = keyed(center, tuple(0.5 * a for a in size), center_end, frame_start,
                       frame_end, interpolation, samples_per_frame)
    return MeshObstacle(verts, tris, m, frames, **obstacle)


def rasterize_moving_obstacles(sv, phi, phi_init: float, t: float, dt: float) -> None:
    """Shared by the gas and liquid solvers: reset `phi`, `obvel_c` and `num_obs`, then
    rasterize every moving obstacle at scene time t into them. Vertex velocities are the
    position change since the previous substep over dt (0 on the first)."""
    b = sv.b
    b.fill(phi, phi_init)
    b.fill(sv.obvel_c, (0.0, 0.0, 0.0))
    b.fill(sv.num_obs, 0.0)
    for o, m in sv._moving:
        b.set_mesh_transform(m, sv.world_to_cell @ o.matrix_at(t), dt)
        b.rasterize_mesh_obstacle(m, phi, o.surface_thickness, o.plane_init,
                                  sv.obvel_c, sv.num_obs)


@dataclass
class GuideSettings:
    """Guiding fields shared by gas and liquid domains (Blender "Guides"; tau, sigma,
    theta fixed as in Blender)."""
    use_guide: bool = False
    guide_alpha: float = 2.0        # weight (higher = more lag)
    guide_beta: int = 5             # blur radius, cells
    guide_vel_factor: float = 2.0
    guide_source: str = "EFFECTOR"  # EFFECTOR (guide objects) | DOMAIN (a parent bake)
    # DOMAIN source: the parent's bake folder (relative = to this bake folder;
    # the CLI resolves it). write_guide: this bake is a parent and writes its
    # full MAC velocity to guide/####.npy each frame.
    guide_parent: str = ""
    write_guide: bool = False


@dataclass
class DomainSettings(GuideSettings):
    """Blender gas domain settings (DNA defaults unless noted)."""
    min_corner: Vec3 = (-1.0, -1.0, 0.0)  # world, metres
    size: Vec3 = (2.0, 2.0, 4.0)
    resolution_max: int = 64
    fps: float = 24.0
    render_fps: int = 0               # scene.render.fps (integer, vel_to_sec); 0 = round(fps)
    domain_mat3: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # domain object 3x3 (row-major), forces go through its inverse
    time_scale: float = 1.0
    cfl: float = 2.0
    timesteps_min: int = 1
    timesteps_max: int = 4
    gravity: Vec3 = (0.0, 0.0, -9.81)     # scene gravity, m/s^2
    alpha: float = 1.0                    # density buoyancy
    beta: float = 1.0                     # heat buoyancy
    vorticity: float = 0.0
    dissolve: bool = False
    dissolve_speed: int = 5
    dissolve_log: bool = True
    # Fire (used when a flow is FIRE or BOTH). Unscaled, as Blender passes them.
    burning_rate: float = 0.75
    flame_smoke: float = 1.0
    flame_vorticity: float = 0.5
    flame_ignition: float = 1.5
    flame_max_temp: float = 3.0
    # Fire model: MANTAFLOW (Blender's, matches it) or COMBUSTION (ours, not
    # Mantaflow): fuel burns only above flame_ignition and releases heat, heat
    # cools, and burning gas expands (a divergence source in the pressure
    # solve). COMBUSTION ignores dissolve for heat (cooling replaces it).
    fire_model: str = "MANTAFLOW"
    heat_release: float = 12.0         # heat per unit of fuel burned (capped at flame_max_temp)
    cooling: float = 12.0              # 1/s, linear part
    cooling_radiative: float = 20.0    # 1/s at flame_max_temp, grows as (T/Tmax)^3
    expansion: float = 0.1             # gas volume added per unit of fuel burned (>1 balloons)
    flame_kelvin: float = 1800.0       # VDB temperature (K) at flame_max_temp
    ambient_kelvin: float = 293.0      # VDB temperature (K) at heat 0
    disturbance: float = 60.0          # m/s^2, random push in hot gas (full at ignition)
    disturbance_size: float = 0.1      # m, size of the random blocks
    disturbance_rate: float = 4.0      # 1/s, how often the pattern changes
    shredding: float = 0.0             # 1/s, tears the flame edge along the heat gradient
    # FLAME_FRONT (ours, not Mantaflow; research A6 variant C, Nguyen, Fedkiw and Jensen
    # 2002): COMBUSTION's cooling, disturbance and output, but the fuel is a level-set core
    # that burns inward at flame_speed; heat and smoke come from the front, the gas
    # expands across it, and buoyancy and the pressure solve use the ideal-gas density.
    # 3D only; no noise or guiding.
    flame_speed: float = 0.5           # m/s, burn speed of the front (Nguyen 2002)
    fuel_kelvin: float = 600.0         # K, fuel gas; density jump flame_kelvin / fuel_kelvin
    flame_inject: float = 0.05         # m/s, fuel gas inflow through the emitter outline
    flame_glow_cells: float = 6.0      # cells of fuel-side glow inside the front
    flame_source_gap: int = 3          # cells round emitters without expansion
    flame_smoke_yield: float = 0.0022  # smoke per unit of fuel burned (times flame_smoke)
    # FLAME_FRONT viscosity (CODE-GAS-029): implicit diffusion of velocity once per step with the
    # kinematic viscosity of air at the local temperature (Sutherland's law, ideal gas). Keeps
    # small flames (candles) laminar; negligible at campfire scale.
    use_viscosity: bool = False  # off by default until CODE-GAS-030 (Andre, 2026-11-13)
    # DIFFUSION (CODE-GAS-037; mixture fraction with eddy-dissipation fast chemistry, FDS Technical
    # Reference Guide 6.11.1 sec. 5.1-5.2.4, NIST): fuel gas enters at the emitters, burns where it
    # mixes with air at the subgrid mixing rate, and the heat it releases (less the radiated part)
    # sets the temperature; the fuel and react grids hold the fuel and product mass fractions.
    # Buoyancy and the pressure solve use the ideal-gas density as FLAME_FRONT; COMBUSTION's
    # disturbance, shredding and flame vorticity do not apply. use_viscosity applies as for
    # FLAME_FRONT. 3D only; no noise or guiding. Defaults: the FDS pool3d propane pool (1950 kW
    # on 1 m^2; reaction constants from FDS 6.11.1 pool3d.out, soot yield from its input).
    diffusion_hrrpua: float = 1950.0              # kW/m^2 of emitter footprint (x-y plane)
    diffusion_heat_of_combustion: float = 46.007  # MJ/kg
    diffusion_air_fuel_ratio: float = 15.592      # kg air per kg fuel (stoichiometric)
    diffusion_radiative_fraction: float = 0.29    # share of the heat lost to radiation
    diffusion_soot_yield: float = 0.01            # kg soot (smoke) per kg fuel burned
    # Sides without border collision ("xXyYzZ", lower case = low side).
    # Blender's gas domain default is all sides open.
    outflow: str = "xXyYzZ"
    cg_accuracy: float = 1e-3
    # CLASSIC = Blender/Mantaflow's step; HIGH = advection-reflection (Zehnder et al. 2018):
    # keeps more swirl and detail for a second pressure solve and one more velocity grid.
    quality: str = "CLASSIC"
    # 3, or 2 = world X-Z plane: a 3-cell slab in Y at the domain's Y centre
    # (Y size ignored, y sides always walls).
    dimensions: int = 3
    # Noise / upres (Blender wavelet turbulence): a high-res density built
    # after each frame from the low-res state, which it does not change.
    use_noise: bool = False
    noise_upres: int = 2            # Blender `noise_scale`
    noise_strength: float = 1.0
    noise_pos_scale: float = 2.0
    noise_time_anim: float = 0.1
    flows: list[MeshFlow] = field(default_factory=list)
    obstacles: list[MeshObstacle] = field(default_factory=list)
    guides: list[MeshGuide] = field(default_factory=list)
    # Force field objects (world space) and the domain's Effector Weights.
    force_fields: list[ForceField] = field(default_factory=list)
    effector_weights: EffectorWeights = field(default_factory=EffectorWeights)

    def validate(self) -> None:
        """Raise ValueError naming the first bad field."""
        from .validate import gas
        gas(self)

    def estimate_kwargs(self) -> dict:
        """Keyword extras for estimate.* (obstacles, fire, noise upres, ...)."""
        return {"has_obstacles": bool(self.obstacles),
                "moving_obstacles": any(o.animated for o in self.obstacles),
                "fire": any(f.flow_type != "SMOKE" for f in self.flows),
                "noise_upres": int(self.noise_upres) if self.use_noise else 0,
                "dimensions": self.dimensions, "force_fields": bool(self.force_fields),
                "guiding": self.use_guide,
                "combustion": self.fire_model in ("COMBUSTION", "FLAME_FRONT", "DIFFUSION"),
                "flame_front": self.fire_model == "FLAME_FRONT",
                "mixture_fraction": self.fire_model == "DIFFUSION",
                "gas_viscosity": self.fire_model in ("FLAME_FRONT", "DIFFUSION") and self.use_viscosity,
                "high_quality": self.quality == "HIGH"}


@dataclass
class FrameStats:
    frame: int
    substeps: int
    dts: list[float]
    pressure_iters: list[int]      # CG iterations per substep (summed over PD iterations when guiding)
    seconds: float
    guide_iters: list[int] = field(default_factory=list)  # PD iterations per substep


@dataclass(frozen=True)
class Units:
    """Solver units of a domain (fluid_script.h / MANTA_main.cpp)."""
    domain_size: float
    dx: float
    shape: tuple
    frame_length: float
    res_per_m: float
    vel_scale: float   # m/s -> cells per solver time
    scale_acc: float   # m/s^2 -> solver units
    gravity: tuple
    dt_min: float
    dt_max: float

    @classmethod
    def from_settings(cls, s, round_frame_length: bool = False) -> "Units":
        """round_frame_length: Blender hands the liquid frame length to Mantaflow as std::to_string
        (6 decimals, 0.104167 at 24 fps); the gas value is not rounded."""
        domain_size = max(s.size)
        frame_length = DT_DEFAULT * (25.0 / s.fps) * s.time_scale
        if round_frame_length:
            frame_length = float(f"{frame_length:f}")
        res_per_m = s.resolution_max / domain_size
        scale_acc = res_per_m * (1.0 / (DT_DEFAULT * 25.0)) ** 2
        return cls(domain_size, domain_size / s.resolution_max, _estimate.grid_shape(s.size, s.resolution_max),
                   frame_length, res_per_m, res_per_m / (25.0 * DT_DEFAULT), scale_acc,
                   tuple(g * scale_acc for g in s.gravity),
                   frame_length / max(1, s.timesteps_max), frame_length / max(1, s.timesteps_min))


class _FrameStepper:
    """The frame loop and time stepping both solvers share (Blender `manta_step`, FluidSolver::step).
    Subclasses provide `_substep(dt)` and `_after_frame`, and may hook the other two."""

    def _before_frame(self) -> None:
        pass

    def _before_substep(self) -> None:
        pass

    def _after_substep(self) -> None:
        pass

    def step_frame(self):
        """Advance one Blender frame."""
        t0 = time.perf_counter()
        dts, iters, pd_iters = [], [], []
        t = 0.0
        self._before_frame()
        if self.guiding:
            self.guiding.update_frame(self.frame + 1)
        while t + FLT_EPSILON < self.frame_length:
            self._adapt_timestep(self.b.max_velocity(self.vel))
            self._before_substep()
            it, pd = self._substep(self.dt)
            iters.append(it)
            if pd is not None:
                pd_iters.append(pd)
            dts.append(self.dt)
            t += self.dt
            self._after_substep()
            # FluidSolver::step
            self.time_per_frame += self.dt
            if self.time_per_frame + VECTOR_EPSILON > self.frame_length:
                self.time_per_frame = 0.0
                self.lock_dt = False
        return self._after_frame(t0, dts, iters, pd_iters)

    def _adapt_timestep(self, max_vel: float) -> None:
        """Mantaflow `FluidSolver::adaptTimestep`."""
        if not math.isfinite(max_vel):
            raise SimulationDiverged(self.frame + 1)
        if self.lock_dt:
            return
        mvt = max_vel * self.dt
        dt = max(min(self.dt * (self.s.cfl / (mvt + 1e-05)), self.dt_max), self.dt_min)
        rest = self.frame_length - self.time_per_frame
        if self.time_per_frame + dt * 1.05 > self.frame_length:
            dt = rest + 1e-04
        elif (self.time_per_frame + dt + self.dt_min > self.frame_length
              or self.time_per_frame + dt * 1.25 > self.frame_length):
            dt = (rest + 1e-04) * 0.5
            self.lock_dt = True
        self.dt = dt

    def _check_finite(self, field, extra: float = 0.0) -> None:
        """Raise SimulationDiverged if `field`, the pressure or `extra` holds NaN or infinity.
        Sums, because a single reduction per grid propagates both."""
        probe = self.b.sum(field) + self.b.sum(self.pressure) + extra
        if not (math.isfinite(probe) and math.isfinite(self.dt)):
            raise SimulationDiverged(self.frame + 1)

    def _flow_time(self, dt: float) -> float:
        """Scene time (frames) at which Blender evaluates flows for a substep
        (fluid.cc `update_flowsfluids`, flow subframes = 0): the substep's end
        time, i.e. the frame being baked on its last substep and on the first
        frame."""
        frame = self.frame + 1
        if frame > self.frame_start and self.time_per_frame + dt + FLT_EPSILON < self.frame_length:
            return frame - 1 + min((self.time_per_frame + dt) / self.frame_length, 1.0)
        return float(frame)


class SmokeSolver(_FrameStepper):
    def __init__(self, backend: Backend, settings: DomainSettings, frame_start: int = 1):
        self.b = backend
        self.dim2 = settings.dimensions == 2
        if self.dim2:
            if settings.use_noise:
                raise ValueError("2D domains do not support noise / high resolution")
            mc, size = _estimate.slab(settings.min_corner, settings.size, settings.resolution_max)
            settings = replace(settings, min_corner=mc, size=size,
                               outflow=settings.outflow.replace("y", "").replace("Y", ""))
        elif settings.dimensions != 3:
            raise ValueError(f"dimensions must be 2 or 3, not {settings.dimensions}")
        self.s = s = settings

        # Units (fluid_script.h / MANTA_main.cpp).
        u = Units.from_settings(s)
        self.domain_size, self.dx, self.shape, self.frame_length = u.domain_size, u.dx, u.shape, u.frame_length
        self.frame_length_unscaled = self.frame_length / s.time_scale
        self.vel_scale, self.scale_acc, self.gravity = u.vel_scale, u.scale_acc, u.gravity
        self.buoy_dens = s.alpha / self.domain_size
        self.buoy_heat = s.beta / self.domain_size
        self.domain_closed = s.outflow == ""
        self.dt_min, self.dt_max = u.dt_min, u.dt_max

        # Grids.
        b, sh = self.b, self.shape
        self.flags = b.flag_grid(sh)
        self.vel = b.mac_grid(sh)
        self.vel_pre = b.mac_grid(sh) if s.quality == "HIGH" else None  # unprojected half step
        self.density = b.scalar_grid(sh)
        self.heat = b.scalar_grid(sh)
        self.pressure = b.scalar_grid(sh)
        self.phi_in = b.scalar_grid(sh)
        self.density_in = b.scalar_grid(sh)
        self.heat_in = b.scalar_grid(sh)
        self.emission_in = b.scalar_grid(sh)
        # Fire grids (smoke_script.h `using_fire`). Blender also turns fire on
        # for SMOKE flows with a fuel amount; that changes nothing without fuel,
        # so we only allocate them for FIRE/BOTH flows. `flame` doubles as the
        # flame-vorticity strength during a substep (as in Blender).
        self.fire = any(f.flow_type != "SMOKE" for f in s.flows)
        self.fuel = self.react = self.flame = self.fuel_in = self.react_in = None
        if self.fire:
            self.fuel = b.scalar_grid(sh)
            self.react = b.scalar_grid(sh)
            self.flame = b.scalar_grid(sh)
            self.fuel_in = b.scalar_grid(sh)
            self.react_in = b.scalar_grid(sh)
        if s.fire_model not in ("MANTAFLOW", "COMBUSTION", "FLAME_FRONT", "DIFFUSION"):
            raise ValueError("fire_model must be MANTAFLOW, COMBUSTION, FLAME_FRONT or DIFFUSION, "
                             f"not {s.fire_model!r}")
        # FLAME_FRONT shares COMBUSTION's cooling, disturbance, flame update and output; DIFFUSION
        # shares its expansion grid, flame update and Kelvin output, and FLAME_FRONT's ideal-gas
        # buoyancy, 1/rho projection and viscosity (`ideal_gas`).
        self.flame_front = self.fire and s.fire_model == "FLAME_FRONT"
        self.diffusion = self.fire and s.fire_model == "DIFFUSION"
        self.ideal_gas = self.flame_front or self.diffusion
        self.combustion = self.fire and s.fire_model in ("COMBUSTION", "FLAME_FRONT", "DIFFUSION")
        self.expansion_div = b.scalar_grid(sh) if self.combustion else None
        self.phi_flame = self.mixfrac = None
        if self.ideal_gas:
            if self.dim2:
                raise ValueError(f"fire_model {s.fire_model} is 3D only")
            self.lift = b.scalar_grid(sh)
            self.face_weights = b.mac_grid(sh)
            self.kelvin = b.scalar_grid(sh) if s.use_viscosity else None
        if self.diffusion:
            self.mixfrac = b.mixfrac_init(
                sh, dx=self.dx, vel_scale=self.vel_scale, ambient_kelvin=s.ambient_kelvin,
                kelvin_per_heat=(s.flame_kelvin - s.ambient_kelvin) / s.flame_max_temp,
                heat_of_combustion=s.diffusion_heat_of_combustion * 1e6,
                air_fuel_ratio=s.diffusion_air_fuel_ratio, radiative_fraction=s.diffusion_radiative_fraction,
                soot_yield=s.diffusion_soot_yield)
        if self.flame_front:
            self.phi_flame = b.scalar_grid(sh, 6.0)  # fuel core < 0, cells; +-6 band
            self.flame_speed_c = s.flame_speed * self.vel_scale
            self.flame_inject_c = s.flame_inject * self.vel_scale
            self.fuel_heat = ((s.fuel_kelvin - s.ambient_kelvin) / (s.flame_kelvin - s.ambient_kelvin)
                              * s.flame_max_temp)
            self.flame_jump = s.flame_kelvin / s.fuel_kelvin - 1.0  # rho_fuel / rho_hot - 1
        # Meshes. world → cell = Blender manta_pos_to_cell (cell centres at
        # index + 0.5). Static objects get their transform once.
        self.world_to_cell = np.diag([1.0 / self.dx] * 3 + [1.0])
        self.world_to_cell[:3, 3] = -np.asarray(s.min_corner, float) / self.dx
        self._flow_meshes = [b.create_mesh(f.verts, f.tris) for f in s.flows]
        for f, m in zip(s.flows, self._flow_meshes):
            if not f.animated:
                b.set_mesh_transform(m, self.world_to_cell @ f.matrix_at(1.0))
        # Obstacles (fluid.cc update_obstacles). Static ones are rasterised
        # once (Blender's phiObsSIn); moving ones every substep into
        # phi_obs_moving (phiObsIn) with their velocities, and phi_obs is
        # the join of both.
        self.phi_obs = self.phi_obs_static = self.phi_obs_moving = None
        self.obvel = None
        static = [o for o in s.obstacles if not o.animated]
        self._moving = [(o, b.create_mesh(o.verts, o.tris)) for o in s.obstacles if o.animated]
        if static:
            self.phi_obs_static = b.scalar_grid(sh, 9999.0)  # PHI_MAX
            for o in static:
                m = b.create_mesh(o.verts, o.tris)
                b.set_mesh_transform(m, self.world_to_cell @ o.matrix_at(1.0))
                b.rasterize_mesh_obstacle(m, self.phi_obs_static, o.surface_thickness, o.plane_init)
            self.phi_obs = self.phi_obs_static
        if self._moving:
            self.phi_obs_moving = b.scalar_grid(sh, 9999.0)
            self.phi_obs = b.scalar_grid(sh, 9999.0) if static else self.phi_obs_moving
            self.obvel_c = b.vector_grid(sh)
            self.num_obs = b.scalar_grid(sh)
            self.obvel = b.mac_grid(sh)

        # Time state (Mantaflow FluidSolver). `frame` = scene frame of the last
        # finished frame.
        self.frame_start = frame_start
        self.frame = frame_start - 1
        self.dt = self.frame_length  # fluid.cc: fds->dt = frame_length
        self.time_per_frame = 0.0
        self.lock_dt = False
        # Blender's fds->time_total (float32 sum of the substeps, clamped per
        # frame) and its value at the start of the last substep (noise time).
        self.time_total = np.float32(0.0)
        self.time_last_substep = np.float32(0.0)

        self.noise = NoisePass(self) if s.use_noise else None
        self.guiding = None
        if s.use_guide:
            self.guiding = make_guiding(b, sh, s, self.world_to_cell, self.frame_length,
                                        frame_start, self.dim2)

    # ---- time stepping -----------------------------------------------------

    def _before_frame(self) -> None:
        self._time_total_old = self.time_total

    def _before_substep(self) -> None:
        self.time_last_substep = self.time_total

    def _after_substep(self) -> None:
        self.time_total = np.float32(self.time_total + np.float32(self.dt))

    def _after_frame(self, t0, dts, iters, pd_iters) -> FrameStats:
        self.time_total = min(self.time_total,
                              np.float32(self._time_total_old + np.float32(self.frame_length)))
        self._check_finite(self.density)
        self.frame += 1
        if self.noise:
            self.noise.step()
        self.b.synchronize()
        return FrameStats(self.frame, len(dts), dts, iters, time.perf_counter() - t0, pd_iters)

    # ---- output ------------------------------------------------------------

    def output(self, grid) -> np.ndarray:
        """Host copy of a grid for the VDB: in 2D only the middle slice."""
        a = self.b.to_numpy(grid)
        return a[:, 1:2] if self.dim2 else a

    def output_temperature(self) -> np.ndarray:
        """VDB `temperature`: the heat grid, or with COMBUSTION Kelvin (ambient at
        heat 0, flame_kelvin at flame_max_temp) where heat > 0 and 0 elsewhere."""
        h = self.output(self.heat)
        if not self.combustion:
            return h
        s = self.s
        cold = ~(h > 1e-6)
        h *= (s.flame_kelvin - s.ambient_kelvin) / s.flame_max_temp  # h is our own download
        h += s.ambient_kelvin
        h[cold] = 0.0
        return h

    def guide_velocity(self, clip: float = 1e-6) -> np.ndarray:
        """MAC velocity for child domains (guide source DOMAIN): what Blender's
        data cache holds, i.e. clipped to cells with density > clip (its
        `clipGrid=density`; elsewhere 0), half float. 2D: the whole slab."""
        mask = self.b.to_numpy(self.density) > clip
        return np.where(mask[..., None], self.b.to_numpy(self.vel), 0.0).astype(np.float16)

    @property
    def output_min_corner(self) -> Vec3:
        mc = self.s.min_corner
        return (mc[0], mc[1] + self.dx, mc[2]) if self.dim2 else mc

    # ---- checkpoints -------------------------------------------------------

    def save_state(self) -> dict:
        """Everything a resume needs, as host arrays. Flags, inflow grids and
        phi_obs are rebuilt each substep / in __init__; pressure starts from 0.
        With fire also fuel and react (flame is rebuilt from react)."""
        b = self.b
        st = {"vel": b.to_numpy(self.vel), "density": b.to_numpy(self.density),
              "heat": b.to_numpy(self.heat), "frame": self.frame, "dt": self.dt,
              "time_per_frame": self.time_per_frame, "lock_dt": self.lock_dt,
              "time_total": float(self.time_total)}
        if self.fire:
            st["fuel"] = b.to_numpy(self.fuel)
            st["react"] = b.to_numpy(self.react)
        if self.flame_front:
            st["phi_flame"] = b.to_numpy(self.phi_flame)
        if self.diffusion:  # fuel and react hold Y_F and Y_P; sig is the transported density
            st["mixfrac_sig"] = b.to_numpy(self.mixfrac.sig)
        if self.noise:
            st.update(self.noise.save_state())
        return st

    def load_state(self, st) -> None:
        b = self.b
        for k in (("vel", "density", "heat") + (("fuel", "react") if self.fire else ())
                  + (("phi_flame",) if self.flame_front else ())
                  + (("mixfrac_sig",) if self.diffusion else ())):
            if k not in st:
                raise ValueError(f"checkpoint has no {k} (baked with another fire model?)")
            if tuple(st[k].shape[:3]) != self.shape:
                raise ValueError(f"checkpoint {k} shape {st[k].shape} != grid {self.shape}")
            grid = self.mixfrac.sig if k == "mixfrac_sig" else getattr(self, k)
            b.from_numpy(grid, np.asarray(st[k], np.float32))
        self.frame = int(st["frame"])
        self.dt = float(st["dt"])
        self.time_per_frame = float(st["time_per_frame"])
        self.lock_dt = bool(st["lock_dt"])
        self.time_total = np.float32(st["time_total"] if "time_total" in st else
                                     (self.frame - self.frame_start + 1) * self.frame_length)
        if self.fire:
            b.update_flame(self.react, self.flame)
        if self.noise:
            self.noise.load_state(st)
        # Re-prime the moving obstacles' previous positions at the last
        # substep's evaluation time (the end of the saved frame), so the first
        # substep after a resume gets the same velocity as an unbroken bake.
        # (Blender loses them when the file is reopened: velocity 0 there.)
        if self.frame >= self.frame_start:
            for o, m in self._moving:
                self.b.set_mesh_transform(m, self.world_to_cell @ o.matrix_at(float(self.frame)))
            if self.guiding:
                self.guiding.prime(self.frame)

    # ---- one substep -------------------------------------------------------

    def _rasterize_flows(self, t: float) -> None:
        """Blender `update_flowsfluids` at scene time t: phi_in, emission_in, density_in,
        heat_in (and fuel_in, react_in with fire). The *_in grids start as copies
        of the simulation grids and each flow raises them in its emission shell
        (see rasterize_mesh_flow).
        Blender re-samples gas flows every substep, static or not."""
        b = self.b
        b.fill(self.phi_in, 9999.0)    # PHI_MAX
        b.fill(self.emission_in, 0.0)
        b.copy(self.density_in, self.density)
        b.copy(self.heat_in, self.heat)
        if self.fire:
            b.copy(self.fuel_in, self.fuel)
            b.copy(self.react_in, self.react)
        for f, m in zip(self.s.flows, self._flow_meshes):
            if f.emit_end and not (f.emit_start - 1 < t <= f.emit_end):
                continue
            if f.animated:
                b.set_mesh_transform(m, self.world_to_cell @ f.matrix_at(t))
            b.rasterize_mesh_flow(m, self.phi_in, self.emission_in, self.density_in,
                                  self.heat_in, self.density, self.heat,
                                  0.0 if f.flow_type == "FIRE" else f.density,
                                  f.temperature, f.surface_distance, f.volume_density,
                                  f.plane_init, self.fuel_in, self.react_in, self.fuel,
                                  self.react, f.fuel_amount, f.flow_type != "SMOKE")

    def _rasterize_moving_obstacles(self, t: float, dt: float) -> None:
        """Blender `update_obstacles` + the obstacle part of `smoke_step` for
        the moving obstacles at scene time t (same time as the flows): phi_obs
        and the MAC obstacle velocity `obvel`."""
        b = self.b
        rasterize_moving_obstacles(self, self.phi_obs_moving, 9999.0, t, dt)
        # Extrapolated with the moving levelset before the static join, as Blender.
        b.obstacle_velocity(self.phi_obs_moving, self.obvel_c, self.num_obs, self.obvel)
        if self.phi_obs_static is not None:
            b.copy(self.phi_obs, self.phi_obs_static)
            b.join(self.phi_obs, self.phi_obs_moving)

    def _substep(self, dt: float) -> tuple[int, int | None]:
        """-> (CG iterations, PD iterations or None without guiding)."""
        b, s, flags, vel = self.b, self.s, self.flags, self.vel
        scale = dt / self.frame_length_unscaled

        t = self._flow_time(dt)
        self._rasterize_flows(t)
        if self.noise:
            self.noise.substep_start()
        if self._moving:
            self._rasterize_moving_obstacles(t, dt)
        forces = update_forces(self, t, self.density, self.fuel)
        b.init_domain(flags, boundary_width=0, outflow=s.outflow)
        b.set_inflow_flags(flags, self.phi_in)
        if self.phi_obs is not None:
            b.set_obstacle_flags(flags, self.phi_obs)
        b.apply_emission(flags, self.density, self.density_in, self.emission_in)
        b.apply_emission(flags, self.heat, self.heat_in, self.emission_in)
        if self.diffusion:  # fuel and react are Y_F and Y_P: the source enters as mass, not by emission
            b.mixfrac_source(self.mixfrac, flags, self.phi_in, s.diffusion_hrrpua)
            b.mixfrac_burn(self.mixfrac, flags, vel, self.fuel, self.react, self.heat, self.density,
                           self.expansion_div, scale / s.fps)
        elif self.fire:
            b.apply_emission(flags, self.fuel, self.fuel_in, self.emission_in)
            b.apply_emission(flags, self.react, self.react_in, self.emission_in)
            if self.flame_front:
                self._flame_front_step(dt, scale / s.fps)
            elif self.combustion:
                b.combust(self.fuel, self.density, self.heat, self.expansion_div, s.burning_rate,
                          s.flame_ignition, s.flame_max_temp, s.heat_release, s.flame_smoke,
                          s.cooling, s.cooling_radiative, s.expansion, dt,
                          scale / s.fps)
            else:
                b.process_burn(self.fuel, self.density, self.react, self.heat, s.burning_rate,
                               s.flame_smoke, s.flame_ignition, s.flame_max_temp, dt)

        if s.dissolve:
            b.dissolve(flags, self.density, s.dissolve_speed, s.dissolve_log, scale)
            if not self.combustion:
                b.dissolve(flags, self.heat, s.dissolve_speed, s.dissolve_log, scale)

        if self.dim2:
            for g in (self.density, self.heat, self.fuel, self.react, vel):
                if g is not None:
                    b.mirror_slab(g)
        if self.vel_pre is None:
            self._advect_scalars(dt)
            b.advect_velocity(flags, vel, dt)
            self._after_advect()
            self._add_forces(t, dt, scale, forces)
            return self._project(True)
        # Advection-reflection: half step and forces, project, scalars through the
        # projected midpoint velocity, reflect, second half step, project again.
        b.advect_velocity(flags, vel, 0.5 * dt)
        if not self.domain_closed:  # outflow flags as the final solve sees them, else the MG levels rebuild
            b.reset_outflow(flags, self.density)
        if self.dim2:
            b.mirror_slab(vel)
        self._add_forces(t, dt, scale, forces)
        b.set_wall_bcs(flags, vel, self.obvel)
        b.copy(self.vel_pre, vel)
        it1, _ = self._project(False)
        self._advect_scalars(dt)
        b.reflect(vel, self.vel_pre)
        b.advect_velocity(flags, vel, 0.5 * dt, field=self.vel_pre)
        b.copy(vel, self.vel_pre)
        self._after_advect()
        it, pd = self._project(True)
        return it1 + it, pd

    def _advect_scalars(self, dt: float) -> None:
        b, flags, vel = self.b, self.flags, self.vel
        if self.diffusion:  # species and density conservatively (heat follows Y_P); smoke as usual
            b.mixfrac_transport(self.mixfrac, flags, vel, self.fuel, self.react, self.heat,
                                dt / self.frame_length_unscaled / self.s.fps)
            b.advect(flags, vel, self.density, dt)
            return
        b.advect(flags, vel, self.density, dt)
        b.advect(flags, vel, self.heat, dt)
        if self.fire:
            b.advect(flags, vel, self.fuel, dt)
            b.advect(flags, vel, self.react, dt)
        if self.flame_front:
            b.advect(flags, vel, self.phi_flame, dt)

    def _flame_front_step(self, dt: float, dt_seconds: float) -> None:
        """FLAME_FRONT in place of `combust`: burn the level-set fuel core, heat, smoke and
        fuel from the front, and the expansion source (research A6 variant C)."""
        s = self.s
        self.b.flame_front_step(
            self.flags, self.phi_in, self.phi_flame, self.heat, self.density, self.fuel, self.fuel_in,
            self.expansion_div, burn_cells=self.flame_speed_c * dt,
            expansion_rate=self.flame_jump * self.flame_speed_c, fuel_heat=self.fuel_heat,
            flame_max_temp=s.flame_max_temp, cooling=s.cooling, cooling_radiative=s.cooling_radiative,
            dt_seconds=dt_seconds, smoke_yield=s.flame_smoke_yield * s.flame_smoke,
            glow_cells=s.flame_glow_cells, source_gap=int(s.flame_source_gap))

    def _after_advect(self) -> None:
        b = self.b
        if not self.domain_closed:
            b.reset_outflow(self.flags, self.density)
        if self.dim2:
            b.mirror_slab(self.vel)
            if self.fire:
                b.mirror_slab(self.fuel)

    def _add_forces(self, t: float, dt: float, scale: float, forces: bool) -> None:
        b, s, flags, vel = self.b, self.s, self.flags, self.vel
        if self.fire and not self.diffusion:
            # Flame vorticity: per-cell strength fuel * flame_vorticity * scale,
            # applied even when the domain vorticity is 0.
            b.vorticity_confinement(flags, vel, s.vorticity * scale, self.fuel,
                                    s.flame_vorticity * scale)
        elif s.vorticity:
            b.vorticity_confinement(flags, vel, s.vorticity * scale)
        if self.combustion and not self.diffusion:
            b.fire_forces(flags, vel, self.heat, s.disturbance * self.scale_acc * dt,
                          s.disturbance_size / self.dx, t / s.fps * s.disturbance_rate,
                          s.flame_ignition, s.shredding * scale / s.fps, s.flame_max_temp)
        g = self.gravity
        if self.ideal_gas:  # ideal-gas buoyancy g (rho_amb / rho - 1) in place of beta * heat
            b.density_lift(self.heat, self.lift, s.ambient_kelvin, s.flame_kelvin, s.flame_max_temp)
            b.add_buoyancy(flags, vel, self.lift, tuple(-gi * dt for gi in g))
        else:
            b.add_buoyancy(flags, vel, self.heat,
                           tuple(-gi * dt * self.buoy_heat for gi in g))
        b.add_buoyancy(flags, vel, self.density,
                       tuple(-gi * dt * self.buoy_dens for gi in g))
        if forces:
            b.add_force_field(flags, vel, self.forces, force_scale(self))
        if self.flame_front and self.flame_inject_c:
            # Fuel gas enters as a mass source (a velocity set in the emitter is undone by the projection).
            b.fuel_mass_source(flags, self.phi_in, self.expansion_div, self.flame_inject_c)

    def _project(self, final: bool) -> tuple[int, int | None]:
        """Wall BCs and the pressure solve (guided only when `final`); the flame update
        after the final one. -> (CG iterations, PD iterations or None)."""
        b, s, flags, vel = self.b, self.s, self.flags, self.vel
        if final and self.ideal_gas and self.kelvin is not None:
            # Viscosity (CODE-GAS-029): rho u' - dt_s div(mu grad u') = rho u in grid units,
            # alpha = dt_s / dx^2 = vel_scale dt / dx (dt_s = vel_scale dt dx), dx in metres.
            b.gas_kelvin(self.heat, self.kelvin, s.ambient_kelvin, s.flame_kelvin, s.flame_max_temp)
            b.solve_diffusion(flags, vel, self.vel_scale * self.dt / self.dx, kelvin=self.kelvin)
        b.set_wall_bcs(flags, vel, self.obvel)
        if self.dim2:
            b.mirror_slab(vel)  # drops y face velocities the obstacle BCs may set
        pd = None
        if self.guiding and final:
            gst = self.guiding.solve(flags, vel, self.pressure, accuracy=s.cg_accuracy,
                                     zero_pressure_fixing=self.domain_closed,
                                     parallel_coarsening=s.quality == "HIGH")
            it, pd = gst.pressure_iterations, gst.iterations
        else:
            weights = None
            if self.ideal_gas:  # variable-density projection, w = 1 / rho_face
                b.density_face_weights(self.heat, self.face_weights, s.ambient_kelvin, s.flame_kelvin,
                                       s.flame_max_temp)
                weights = self.face_weights
            it = b.solve_pressure(flags, vel, self.pressure, accuracy=s.cg_accuracy,
                                  zero_pressure_fixing=self.domain_closed,
                                  divergence_grid=self.expansion_div,
                                  parallel_coarsening=s.quality == "HIGH",
                                  face_weights=weights).iterations
        if self.dim2:
            b.mirror_slab(vel)
        if final and self.combustion:
            b.update_flame_temperature(self.heat, self.flame, s.flame_ignition, s.flame_max_temp)
        elif final and self.fire:
            b.update_flame(self.react, self.flame)
        return it, pd


def update_forces(sol, t, density=None, fuel=None) -> bool:
    """Blender's update_effectors into `sol.forces` (allocated on first use); False when
    no field acts. t is the scene time of the substep. Called at the substep start, before the flags are rebuilt."""
    s = sol.s
    if not s.force_fields:
        return False
    if getattr(sol, "forces", None) is None:
        sol.forces = sol.b.vector_grid(sol.shape)
    cell = tuple(n / m for n, m in zip(s.size, sol.shape))
    return sol.b.compute_forces(sol.flags, sol.vel, sol.forces, [f.at(t) for f in s.force_fields],
                                s.effector_weights, s.min_corner, cell, s.render_fps or round(s.fps), mat3=np.reshape(s.domain_mat3, (3, 3)), density=density, fuel=fuel,
                                phi_obs=sol.phi_obs)


def force_scale(sol) -> float:
    """scaleSpeedFrames: forces are in blength/frame, the grid in cells/frameLength."""
    return sol.s.resolution_max / sol.domain_size / (sol.frame_length / sol.s.time_scale)


def _blender_str(x: float) -> float:
    """A float passed to Mantaflow's Python through `std::to_string` (6 decimals)."""
    return float(f"{float(np.float32(x)):.6f}")


class NoisePass:
    """Blender's wavelet turbulence (`smoke_step_noise`, smoke_script.h): a
    one-way pass after each solver frame that advects a high-res density
    (noise_upres x per axis) through the interpolated low-res velocity plus
    curl noise. The low-res simulation is not changed. Step order and quirks:
    reference/NOTES.md (v3)."""

    def __init__(self, sol: SmokeSolver):
        self.sol = sol
        b, s = sol.b, sol.s
        self.upres = up = int(s.noise_upres)
        if up < 1:
            raise ValueError(f"noise_upres must be >= 1, got {up}")
        self.octaves = int(np.log(up) / np.log(2.0) + 0.5) if up > 1 else 1
        lo = sol.shape
        self.shape = hi = tuple(n * up for n in lo)
        # Solver values as Mantaflow sees them: frameLength comes through
        # to_string, the noise solver's dx is 1 / max high-res size.
        self.frame_length = np.float32(_blender_str(sol.frame_length))
        self.frame_length_unscaled = float(self.frame_length) / s.time_scale
        self.gs_inv = float(np.float32(1.0 / max(hi)))
        self.pos_scale = tuple(float(np.float32(n) * np.float32(1.0 / s.noise_pos_scale))
                               for n in lo)
        self.seed_offset = _noise.seed_offset()
        self.tile = b.noise_tile(_noise.get_tile())

        # Low-res extras: velocity at the start of the last substep, emission
        # max-joined over the frame's substeps, UV grids and energy.
        self.vel_tmp = b.mac_grid(lo)
        self.emission = b.scalar_grid(lo)
        self.uv = (b.vector_grid(lo), b.vector_grid(lo))
        for uv in self.uv:
            b.reset_uv_grid(uv)
        self.energy = b.scalar_grid(lo)
        self.tmp_flags = b.flag_grid(lo)
        # High-res state (density, fuel, react) and per-frame grids.
        self.flags = b.flag_grid(hi)
        self.vel = b.mac_grid(hi)
        self.phi = b.scalar_grid(hi)
        self.tmp_in = b.scalar_grid(hi)
        self.emission_in = b.scalar_grid(hi)
        self.density = b.scalar_grid(hi)
        self.fuel = self.react = self.flame = None
        if sol.fire:
            self.fuel = b.scalar_grid(hi)
            self.react = b.scalar_grid(hi)
            self.flame = b.scalar_grid(hi)

    def substep_start(self) -> None:
        """Bookkeeping in each low-res substep, after the flows are rasterised
        (smoke_adaptive_step / smoke_step)."""
        sol, b = self.sol, self.sol.b
        if not sol.time_per_frame:
            b.fill(self.emission, 0.0)
        b.join(self.emission, sol.emission_in, keep_max=True)
        b.copy(self.vel_tmp, sol.vel)

    def step(self) -> None:
        """One noise frame, after the low-res frame finished (sol.frame is
        the frame just baked)."""
        sol, b, s = self.sol, self.sol.b, self.sol.s
        flags, up = self.flags, self.upres
        dt_low = float(np.float32(sol.dt))          # s.timestep: last substep
        dt = float(self.frame_length)               # sn.timestep
        # s.timeTotal after FluidSolver::step = (frame + 1) * frameLength.
        t_uv = float(np.float32((sol.frame + 1) * np.float64(self.frame_length)))
        # sn.timeTotal = timeTotal_s = Blender time at the last substep's start.
        t_noise = np.float32(_blender_str(sol.time_last_substep))
        noise_time = float(np.float32(np.float32(t_noise * np.float32(self.gs_inv))
                                      * np.float32(s.noise_time_anim)))

        # High-res flags and velocity.
        b.init_domain(flags, boundary_width=0, outflow=s.outflow)
        b.interpolate_grid(self.phi, sol.phi_in)
        b.set_inflow_flags(flags, self.phi)
        if sol.phi_obs is not None:
            b.interpolate_grid(self.phi, sol.phi_obs)
            b.set_obstacle_flags(flags, self.phi)
        b.interpolate_mac(self.vel, self.vel_tmp)

        # Emission (density scaled by upres, fuel/react not).
        b.interpolate_grid(self.emission_in, self.emission)
        b.interpolate_grid(self.tmp_in, sol.density_in)
        b.mult_const(self.tmp_in, float(up))
        b.apply_emission(flags, self.density, self.tmp_in, self.emission_in)
        if sol.fire:
            b.interpolate_grid(self.tmp_in, sol.fuel_in)
            b.apply_emission(flags, self.fuel, self.tmp_in, self.emission_in)
            b.interpolate_grid(self.tmp_in, sol.react_in)
            b.apply_emission(flags, self.react, self.tmp_in, self.emission_in)
            b.process_burn(self.fuel, self.density, self.react, None, s.burning_rate,
                           s.flame_smoke, s.flame_ignition, s.flame_max_temp, dt)

        # step_noise
        if s.dissolve:
            b.dissolve(flags, self.density, s.dissolve_speed, s.dissolve_log,
                       dt_low / self.frame_length_unscaled)
        weights = []
        for i, uv in enumerate(self.uv):
            b.advect(sol.flags, sol.vel, uv, dt_low)
            weights.append(b.update_uv_weight(uv, t_uv, dt_low, dt * 10.0, i, len(self.uv)))
        if not sol.domain_closed:
            b.reset_outflow(flags, self.density)
        b.compute_energy(sol.flags, sol.vel, self.energy)
        b.copy(self.tmp_flags, sol.flags)
        b.extrapolate_simple_flags(sol.flags, self.tmp_flags, 2, Flags.OBSTACLE, Flags.FLUID)
        b.extrapolate_simple_flags(self.tmp_flags, self.energy, 6, Flags.FLUID, Flags.OBSTACLE)
        b.compute_wavelet_coeffs(self.energy)
        strength, spos = 1.0 * s.noise_strength, 2.0
        for _ in range(self.octaves):
            for uv, w in zip(self.uv, weights):
                b.apply_noise_vec3(flags, self.vel, self.tile, self.energy, uv, strength * w,
                                   spos, self.gs_inv, self.seed_offset, noise_time,
                                   self.pos_scale)
            strength *= 0.06  # Kolmogorov factor
            spos *= 2.0
        for _ in range(up):
            if sol.fire:
                b.advect(flags, self.vel, self.fuel, dt)
                b.advect(flags, self.vel, self.react, dt)
            b.advect(flags, self.vel, self.density, dt)
        if sol.fire:
            b.update_flame(self.react, self.flame)

    def save_state(self) -> dict:
        b = self.sol.b
        st = {"uv0_noise": b.to_numpy(self.uv[0]), "uv1_noise": b.to_numpy(self.uv[1]),
              "density_noise": b.to_numpy(self.density)}
        if self.fuel is not None:
            st["fuel_noise"] = b.to_numpy(self.fuel)
            st["react_noise"] = b.to_numpy(self.react)
        return st

    def load_state(self, st) -> None:
        b = self.sol.b
        grids = {"uv0_noise": self.uv[0], "uv1_noise": self.uv[1], "density_noise": self.density}
        if self.fuel is not None:
            grids.update(fuel_noise=self.fuel, react_noise=self.react)
        for k, g in grids.items():
            if k not in st:
                raise ValueError(f"checkpoint has no {k} (baked without noise?)")
            want = self.sol.shape if k.startswith("uv") else self.shape
            if tuple(st[k].shape[:3]) != want:
                raise ValueError(f"checkpoint {k} shape {st[k].shape} != grid {want}")
            b.from_numpy(g, np.asarray(st[k], np.float32))
        if self.flame is not None:
            b.update_flame(self.react, self.flame)
