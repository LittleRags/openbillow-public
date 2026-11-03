# SPDX-License-Identifier: GPL-3.0-or-later
"""The liquid solver: Blender's Mantaflow FLIP/APIC step order on top of a `Backend`.

Units, the frame loop and mesh handling are the smoke solver's (solver.py).
Substep: Blender's `update_flowsfluids` / `update_obstacles` (fluid.cc), then
`liquid_adaptive_step` + `liquid_step` (reference/blender/liquid_script.h).

Flows are meshes with behaviour GEOMETRY (initial liquid, sampled on every
substep of the first frame), INFLOW (every substep, optionally with an
initial velocity) or OUTFLOW (deletes liquid). Obstacles are static or move
by transform, as for smoke. Open domain sides delete liquid that reaches them.
Guiding: see guiding.py (PLAN.md step 40). Surface tension is not
ported because Blender's script passes no phi to solvePressure, so it has no
effect there either.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend, ParticleFlags
from .backend.base import Vec3
from .forcefields import EffectorWeights, ForceField
from .meshes import cube, uv_sphere
from .guiding import make_guiding
from .solver import (DT_DEFAULT, FLT_EPSILON, VECTOR_EPSILON, FrameStats, GuideSettings, MeshGuide, MeshObstacle,
                     SmokeSolver, _force_scale, rasterize_moving_obstacles, _keyed, _MeshObject, _update_forces)

PHI_MAX = 9999.0


@dataclass
class LiquidFlow(_MeshObject):
    """A liquid flow object (Blender flow type Liquid, source Mesh)."""
    behavior: str = "GEOMETRY"          # GEOMETRY (initial liquid), INFLOW or OUTFLOW
    surface_distance: float = 0.0       # cells (Blender's liquid default is 0)
    plane_init: bool = False            # Blender "Is Planar"
    use_initial_velocity: bool = False
    velocity_factor: float = 1.0        # Blender "Source" (times the object's velocity)
    velocity_normal: float = 0.0        # along the surface normal, cells per solver time as Blender
    velocity_coord: Vec3 = (0.0, 0.0, 0.0)  # Blender "Initial X/Y/Z", m/s
    emit_start: int = 0                 # INFLOW only, as MeshFlow (not in Blender: there
    emit_end: int = 0                   # one keyframes "Use Flow"); emit_end 0: always


def BoxLiquid(center: Vec3, size: Vec3, center_end: Vec3 | None = None,
              frame_start: float = 1.0, frame_end: float = 60.0,
              interpolation: str = "BEZIER", samples_per_frame: int = 4, **flow) -> LiquidFlow:
    """Blender's default cube scaled to edge lengths `size` (metres) as a
    liquid flow, optionally keyframed (see solver._keyed)."""
    verts, tris = cube()
    m, frames = _keyed(center, tuple(0.5 * a for a in size), center_end, frame_start,
                       frame_end, interpolation, samples_per_frame)
    return LiquidFlow(verts, tris, m, frames, **flow)


def SphereLiquid(center: Vec3, radius: float, center_end: Vec3 | None = None,
                 frame_start: float = 1.0, frame_end: float = 60.0,
                 interpolation: str = "BEZIER", samples_per_frame: int = 4, **flow) -> LiquidFlow:
    """Blender's default UV sphere as a liquid flow."""
    verts, tris = uv_sphere()
    m, frames = _keyed(center, (radius,) * 3, center_end, frame_start, frame_end,
                       interpolation, samples_per_frame)
    return LiquidFlow(verts, tris, m, frames, **flow)


@dataclass
class LiquidSettings(GuideSettings):
    """Blender liquid domain settings (DNA defaults unless noted)."""
    min_corner: Vec3 = (-1.0, -1.0, 0.0)  # world, metres
    size: Vec3 = (2.0, 2.0, 2.0)
    resolution_max: int = 64
    fps: float = 24.0
    render_fps: int = 0               # scene.render.fps (integer, vel_to_sec); 0 = round(fps)
    domain_mat3: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # domain object 3x3 (row-major), forces go through its inverse
    time_scale: float = 1.0
    cfl: float = 2.0
    timesteps_min: int = 1
    timesteps_max: int = 4
    gravity: Vec3 = (0.0, 0.0, -9.81)
    # Sides without border collision ("xXyYzZ", lower case = low side). Blender's
    # liquid domain default is a closed box.
    outflow: str = ""
    cg_accuracy: float = 1e-3
    # CLASSIC = Blender (p = 0 in EMPTY cells); HIGH = ghost-fluid free surface (sub-cell
    # surface position in the pressure solve, as Blender's guided liquid).
    quality: str = "CLASSIC"
    use_apic: bool = False            # Blender simulation method FLIP / APIC
    flip_ratio: float = 0.97
    particle_number: int = 2          # per axis and cell when sampling
    particle_min: int = 8
    particle_max: int = 16
    particle_radius: float = 1.0
    particle_band_width: float = 3.0
    particle_randomness: float = 0.1
    delete_in_obstacle: bool = False
    # Fractions (Advanced panel): second-order obstacle boundaries. Face fractions from phiObs
    # weight the pressure solve, walls are 2 cells thick, and particles are kept
    # `fractions_distance` cells off the obstacles (not the walls).
    use_fractions: bool = False
    fractions_threshold: float = 0.05
    fractions_distance: float = 0.5
    # Improvement (on by default; off = Blender): give velOld the same outer-layer copy as vel
    # before the FLIP update, so floor particles stop gaining speed (NOTES, FLIP quirk).
    flip_wall_fix: bool = True
    # Improvement (on by default; off = Blender's pushOutofObs): exact swept collisions of the
    # particles with the obstacle meshes and the domain walls, `collision_margin`
    # cells off the surface, with the velocity into the obstacle removed.
    robust_collisions: bool = True
    collision_margin: float = 0.1
    # Improvement (off by default): keep the level-set volume at the initial liquid plus net
    # inflow, by a uniform target divergence in the pressure solve. `volume_correction_rate`
    # = fraction of the volume error removed per frame.
    volume_correction: bool = True
    volume_correction_rate: float = 0.5
    # Viscosity panel: implicit solve before the pressure, `viscosity_value` (Strength) used as is.
    use_viscosity: bool = False
    viscosity_value: float = 0.05
    # Diffusion panel: implicit velocity diffusion before the viscosity. Blender's kinematic
    # viscosity is `viscosity_base * 10^-viscosity_exponent` m^2/s (divided by the domain
    # size squared for the grid).
    use_diffusion: bool = False
    viscosity_base: float = 1.0
    viscosity_exponent: int = 6
    # Surface tension in N/m with water's density (water 0.072), quality HIGH only: a
    # pressure jump at the ghost-fluid surface. Not Blender's slider, which does nothing
    # (its pressure solve gets no phi). Caps the substep at the capillary time step.
    surface_tension: float = 0.0
    output_particles: bool = True     # also write particles_####.ply per frame
    # Surface written as the VDB `phi` (Blender's liquid mesh step, DNA defaults): "IMPROVED"
    # (Final: averaged particle level set, concavity fix, smoothing) or "UNION" (Fast), on a
    # grid `mesh_scale` times finer than the simulation. "ANISOTROPIC" (not in Blender):
    # ellipsoid particle kernels (Yu & Turk), flatter surfaces and thinner sheets; it runs
    # ANISO_EXTRA_SMOOTHING positive passes on top of mesh_smoothen_pos (keeps sheets whole).
    mesh_generator: str = "IMPROVED"
    mesh_scale: int = 2
    mesh_particle_radius: float = 2.0
    mesh_smoothen_pos: int = 1
    mesh_smoothen_neg: int = 1
    mesh_concave_upper: float = 3.5
    mesh_concave_lower: float = 0.4
    # Secondary particles (Blender's liquid_step_particles, once per frame). Mantaflow simulates
    # all three types when any is on; the use_* flags only choose what is written.
    use_spray_particles: bool = False
    use_foam_particles: bool = False
    use_bubble_particles: bool = False
    particle_scale: int = 1           # secondary-particle grid, times finer than the simulation
    sndparticle_tau_min_wc: float = 2.0
    sndparticle_tau_max_wc: float = 8.0
    sndparticle_tau_min_ta: float = 5.0
    sndparticle_tau_max_ta: float = 20.0
    sndparticle_tau_min_k: float = 1.0
    sndparticle_tau_max_k: float = 5.0
    sndparticle_k_wc: int = 200
    sndparticle_k_ta: int = 40
    sndparticle_k_b: float = 0.5
    sndparticle_k_d: float = 0.6
    sndparticle_l_min: float = 10.0
    sndparticle_l_max: float = 25.0
    sndparticle_potential_radius: int = 2
    sndparticle_update_radius: int = 2  # only used by Mantaflow's 'cubic' update, which Blender skips
    sndparticle_width: float = 0.01       # point width written to the USD (m), constant
    sndparticle_boundary: str = "DELETE"  # or "PUSHOUT"
    # Guiding (Blender "Guides", as DomainSettings).
    flows: list[LiquidFlow] = field(default_factory=list)
    obstacles: list[MeshObstacle] = field(default_factory=list)
    guides: list[MeshGuide] = field(default_factory=list)
    # Force field objects (world space) and the domain's Effector Weights.
    force_fields: list[ForceField] = field(default_factory=list)
    effector_weights: EffectorWeights = field(default_factory=EffectorWeights)

    def estimate_kwargs(self) -> dict:
        """Keyword extras for estimate.*; the fill comes from the initial GEOMETRY flows."""
        from . import estimate
        vol = sum(estimate.mesh_volume(f.verts, f.tris, f.matrix_at(1.0))
                  for f in self.flows if f.behavior == "GEOMETRY")
        fill = vol / (self.size[0] * self.size[1] * self.size[2])
        return {"has_obstacles": bool(self.obstacles),
                "moving_obstacles": any(o.animated for o in self.obstacles),
                "liquid": True, "apic": self.use_apic,
                "particles_per_cell": estimate.liquid_particles_per_cell(
                    fill, self.particle_number), "mesh_scale": self.mesh_scale,
                "secondary": bool(self.use_spray_particles or self.use_foam_particles
                                  or self.use_bubble_particles),
                "particle_scale": self.particle_scale, "viscosity": self.use_viscosity,
                "diffusion": self.use_diffusion, "fractions": self.use_fractions,
                "guiding": self.use_guide}


@dataclass
class LiquidFrameStats(FrameStats):
    particles: int = 0
    secondary: int = 0


class LiquidSolver:
    NARROW_BAND = 3              # liquid_script.h narrowBandWidth
    ANISO_EXTRA_SMOOTHING = 2    # user A/B 2026-11-03: 1 pass breaks the splash crown into flakes
    COMBINE_BAND = NARROW_BAND - 1

    def __init__(self, backend: Backend, settings: LiquidSettings, frame_start: int = 1):
        self.b = b = backend
        self.s = s = settings
        if s.mesh_generator not in ("IMPROVED", "UNION", "ANISOTROPIC"):
            raise ValueError(f"liquid: unknown mesh_generator {s.mesh_generator!r}")
        for f in s.flows:
            if f.behavior not in ("GEOMETRY", "INFLOW", "OUTFLOW"):
                raise ValueError(f"liquid: unknown flow behavior {f.behavior!r}")

        # Units (as SmokeSolver).
        self.domain_size = max(s.size)
        self.dx = self.domain_size / s.resolution_max
        self.shape = sh = tuple(max(3, int(n / self.dx + 0.5)) for n in s.size)
        # Blender passes it to Mantaflow as std::to_string (6 decimals): 0.104167 at 24 fps.
        # The secondary-particle counts int(... * dt) depend on that rounding.
        self.frame_length = float(f"{DT_DEFAULT * (25.0 / s.fps) * s.time_scale:f}")
        res_per_m = s.resolution_max / self.domain_size
        scale_acc = res_per_m * (1.0 / (DT_DEFAULT * 25.0)) ** 2
        self.gravity = tuple(g * scale_acc for g in s.gravity)
        self.vel_scale = res_per_m / (25.0 * DT_DEFAULT)   # m/s -> cells per solver time
        self.dt_min = self.frame_length / max(1, s.timesteps_max)
        self.dt_max = self.frame_length / max(1, s.timesteps_min)
        # sigma / rho (m^3/s^2) in cells^3 per solver time^2; the solve takes it times dt.
        self.surface_tension = 0.0
        if s.quality == "HIGH" and s.surface_tension > 0.0:
            self.surface_tension = s.surface_tension / 1000.0 * res_per_m ** 2 * scale_acc
            # Capillary limit dt < sqrt(rho_avg dx^3 / (2 pi sigma)), rho_avg = rho / 2 (air).
            cap = math.sqrt(1.0 / (4.0 * math.pi * self.surface_tension))
            self.dt_max, self.dt_min = min(self.dt_max, cap), min(self.dt_min, cap)

        # Grids (liquid_alloc). phi starts as initFromFlags on a domain with
        # an empty interior: 0.5 everywhere.
        self.flags = b.flag_grid(sh)
        self.vel = b.mac_grid(sh)
        self.vel_old = b.mac_grid(sh)
        self.pressure = b.scalar_grid(sh)
        self.phi = b.scalar_grid(sh, 0.5)
        self.phi_parts = b.scalar_grid(sh)
        self.phi_in = b.scalar_grid(sh, PHI_MAX)
        self.phi_obs = b.scalar_grid(sh, PHI_MAX)
        self.vel_parts = self.map_weights = None
        if not s.use_apic:
            self.vel_parts = b.mac_grid(sh)
            self.map_weights = b.vector_grid(sh)
        self.parts = b.particles(1 << 16, apic=s.use_apic)
        self.mesh_scale = max(1, int(s.mesh_scale))
        self.phi_mesh = self.phi_mesh_parts = None   # allocated by surface()

        self.world_to_cell = np.diag([1.0 / self.dx] * 3 + [1.0])
        self.world_to_cell[:3, 3] = -np.asarray(s.min_corner, float) / self.dx
        # Fractions: walls one cell thicker; the fractions follow phi_obs, so they are
        # recomputed only when it changes (moving obstacles) and after a resume.
        self.boundary_width = 1 if s.use_fractions else 0
        self.fractions = b.mac_grid(sh) if s.use_fractions else None
        self._fractions_stale = True
        self._init_flows()
        self._init_obstacles()
        self.domain_closed = s.outflow == ""
        self.using_outflow = self.phi_out is not None

        self.frame_start = frame_start
        self.frame = frame_start - 1
        self.dt = self.frame_length
        self.time_per_frame = 0.0
        self.lock_dt = False
        self.volume_target = 0.0  # cells; initial liquid + inflow - outflow so far
        self.guiding = None
        if s.use_guide:
            self.guiding = make_guiding(b, sh, s, self.world_to_cell, self.frame_length, frame_start)
        self.snd = None
        if s.use_spray_particles or s.use_foam_particles or s.use_bubble_particles:
            self._init_secondary()

    def _init_secondary(self) -> None:
        """liquid_alloc_particles. phiTmp / velTmp hold phi and vel from the start of the
        frame (liquid_step). Without upres the sp grids are those grids themselves (Blender
        copies them; nothing in the step writes them)."""
        b, s, sh = self.b, self.s, self.shape
        if s.sndparticle_boundary not in ("DELETE", "PUSHOUT"):
            raise ValueError(f"liquid: unknown sndparticle_boundary {s.sndparticle_boundary!r}")
        self.snd_scale = sc = max(1, int(s.particle_scale))
        self.phi_tmp, self.vel_tmp = b.scalar_grid(sh), b.mac_grid(sh)
        hi = tuple(n * sc for n in sh)
        if sc == 1:
            self.vel_sp, self.phi_sp, self.phi_obs_sp, self.phi_out_sp = (
                self.vel_tmp, self.phi_tmp, self.phi_obs, self.phi_out)
        else:
            self.vel_sp, self.phi_sp, self.phi_obs_sp = (b.mac_grid(hi), b.scalar_grid(hi, PHI_MAX),
                                                         b.scalar_grid(hi, PHI_MAX))
            self.phi_out_sp = b.scalar_grid(hi, PHI_MAX) if self.phi_out is not None else None
        self.flags_sp = b.flag_grid(hi)
        self.pot_ta, self.pot_wc, self.pot_ke, self.neighbor_ratio = (b.scalar_grid(hi)
                                                                      for _ in range(4))
        self.snd = b.particles(1 << 16, life=True)

    def _step_secondary(self) -> None:
        """Blender liquid_step_particles (bake_particles_process: timestep = frameLength)."""
        b, s, dt = self.b, self.s, self.frame_length
        if self.snd_scale > 1:
            b.interpolate_mac(self.vel_sp, self.vel_tmp)
            b.interpolate_grid(self.phi_obs_sp, self.phi_obs)
            b.interpolate_grid(self.phi_sp, self.phi_tmp)
            if self.phi_out is not None:
                b.interpolate_grid(self.phi_out_sp, self.phi_out)
        flags, vel, nr = self.flags_sp, self.vel_sp, self.neighbor_ratio
        b.update_flags_obs(flags, self.phi_obs_sp, None, self.phi_out_sp, boundary_width=0)
        b.update_flags_from_levelset(flags, self.phi_sp)
        b.secondary_potentials(flags, vel, self.phi_sp, self.pot_ta, self.pot_wc, self.pot_ke, nr,
                               s.sndparticle_potential_radius,
                               (s.sndparticle_tau_min_ta, s.sndparticle_tau_max_ta),
                               (s.sndparticle_tau_min_wc, s.sndparticle_tau_max_wc),
                               (s.sndparticle_tau_min_k, s.sndparticle_tau_max_k), self.dx)
        b.sample_secondary_particles(self.snd, flags, vel, self.pot_ta, self.pot_wc, self.pot_ke, nr,
                                     s.sndparticle_k_ta, s.sndparticle_k_wc, s.sndparticle_l_min,
                                     s.sndparticle_l_max, dt)
        b.update_secondary_particles(self.snd, flags, vel, nr, self.gravity, s.sndparticle_k_b,
                                     s.sndparticle_k_d, dt)
        if s.sndparticle_boundary == "PUSHOUT":
            b.push_out_of_obstacles(self.snd, self.phi_obs_sp, shift=1.0)
        b.delete_particles_in_obstacle(self.snd, flags)

    def _mesh(self, o: _MeshObject):
        m = self.b.create_mesh(o.verts, o.tris)
        if not o.animated:
            self.b.set_mesh_transform(m, self.world_to_cell @ o.matrix_at(1.0))
        return m

    def _init_flows(self) -> None:
        """Blender keeps static INFLOW (without initial velocity) and static
        OUTFLOW objects in phiSIn / phiOutSIn from the first frame on; here they
        are rasterised once. Everything else is re-rasterised per substep."""
        b, sh = self.b, self.shape
        self._geometry, self._inflows, self._outflows = [], [], []
        self.phi_s_in = self.phi_out = self.phi_out_static = self.invel = None
        static_in = []
        for f in self.s.flows:
            m = self._mesh(f)
            if f.behavior == "GEOMETRY":
                self._geometry.append((f, m))
            elif f.behavior == "INFLOW" and (f.animated or f.use_initial_velocity or f.emit_end):
                self._inflows.append((f, m))
            elif f.behavior == "INFLOW":
                static_in.append((f, m))
            elif f.animated:
                self._outflows.append((f, m))
            else:
                if self.phi_out_static is None:
                    self.phi_out_static = b.scalar_grid(sh, PHI_MAX)
                b.rasterize_mesh_liquid_flow(m, self.phi_out_static, f.surface_distance,
                                             f.plane_init)
        if static_in:
            self.phi_s_in = b.scalar_grid(sh, PHI_MAX)
            for f, m in static_in:
                b.rasterize_mesh_liquid_flow(m, self.phi_s_in, f.surface_distance, f.plane_init)
        if self._outflows:
            self.phi_out = b.scalar_grid(sh, PHI_MAX)
        elif self.phi_out_static is not None:
            self.phi_out = self.phi_out_static
        if any(f.use_initial_velocity and f.behavior != "OUTFLOW" for f in self.s.flows):
            self.invel = b.vector_grid(sh)

    def _init_obstacles(self) -> None:
        """Blender's phiObsSIn (static) / phiObsIn (moving) and the obstacle
        part of liquid_adaptive_step. Without moving obstacles the result
        (phi_obs with the walls, phi_obs_in) is the same every substep, so it
        is built once here and init_domain later leaves phi_obs alone."""
        b, sh = self.b, self.shape
        static = [o for o in self.s.obstacles if not o.animated]
        self._moving = [(o, b.create_mesh(o.verts, o.tris)) for o in self.s.obstacles if o.animated]
        static_meshes = [self._mesh(o) for o in static]
        self._collision_meshes = static_meshes + [m for _, m in self._moving]
        self.phi_obs_in = self.phi_obs_static = self.obvel = None
        if static:
            self.phi_obs_static = b.scalar_grid(sh, PHI_MAX)
            for o, m in zip(static, static_meshes):
                b.rasterize_mesh_obstacle(m, self.phi_obs_static, o.surface_thickness, o.plane_init)
            if self.s.robust_collisions:
                b.thin_obstacles(static_meshes, self.phi_obs_static)
        if self._moving:
            self.phi_obs_in = b.scalar_grid(sh, PHI_MAX)
            self.obvel_c = b.vector_grid(sh)
            self.num_obs = b.scalar_grid(sh)
            self.obvel = b.mac_grid(sh)
            return
        b.init_domain(self.flags, boundary_width=self.boundary_width, outflow=self.s.outflow,
                      phi_walls=self.phi_obs)
        if static:
            self.phi_obs_in = self.phi_obs_static
            self._obstacle_levelsets()

    def _obstacle_levelsets(self) -> None:
        """liquid_adaptive_step, after the static join: close and extend
        phiObsIn, join it into phiObs (walls), close and extend that."""
        b = self.b
        for g in (self.phi_obs_in, self.phi_obs):
            if g is self.phi_obs:
                b.join(self.phi_obs, self.phi_obs_in)
            b.flood_fill(g, boundary_width=2 if g is self.phi_obs and self.s.use_fractions else 1)
            b.extrapolate_ls_simple(g, distance=6, inside=True)
            b.extrapolate_ls_simple(g, distance=3)

    @property
    def mesh_wall_layers(self) -> int:
        """Outer layers of the mesh level set set to 0.5 (Blender's phi_sm.setBound)."""
        return 2 * self.mesh_scale - 1 if self.s.use_fractions else self.mesh_scale

    def _wall_bcs(self) -> None:
        if self.fractions is not None:  # Blender passes no obvel with fractions
            self.b.set_wall_bcs(self.flags, self.vel, None, self.fractions, self.phi_obs)
        else:
            self.b.set_wall_bcs(self.flags, self.vel, self.obvel)

    @property
    def num_particles(self) -> int:
        return self.parts.n

    def guide_velocity(self) -> np.ndarray:
        """MAC velocity for child domains (guide source DOMAIN), half float.
        Blender's liquid data cache does not clip it, so the whole grid."""
        return self.b.to_numpy(self.vel).astype(np.float16)

    def particles_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Live particles as (pos in world metres, vel in m/s), float32 (n, 3)."""
        pos, vel, flag = self.b.get_particles(self.parts)
        live = (flag & ParticleFlags.DELETE) == 0
        p = pos[live] * np.float32(self.dx) + np.asarray(self.s.min_corner, np.float32)
        return p.astype(np.float32), (vel[live] / np.float32(self.vel_scale)).astype(np.float32)

    def secondary_world(self) -> dict:
        """Live secondary particles by type name ('spray', 'foam', 'bubble') as (pos in world
        metres, vel in m/s) float32 (n, 3); only the types switched on in the settings."""
        pos, vel, flag = self.b.get_particles(self.snd)
        live = (flag & ParticleFlags.DELETE) == 0
        # positions are in cells of the particle grid, `snd_scale` times finer than the simulation
        sc = np.float32(self.dx / self.snd_scale)
        lo = np.asarray(self.s.min_corner, np.float32)
        out = {}
        for name, bit, on in (("spray", ParticleFlags.SPRAY, self.s.use_spray_particles),
                              ("foam", ParticleFlags.FOAM, self.s.use_foam_particles),
                              ("bubble", ParticleFlags.BUBBLE, self.s.use_bubble_particles)):
            if on:
                m = live & ((flag & bit) != 0)
                out[name] = ((pos[m] * sc + lo).astype(np.float32),
                             (vel[m] / np.float32(self.vel_scale)).astype(np.float32))
        return out

    def surface(self) -> np.ndarray:
        """The level set to mesh (cells of the mesh grid, `mesh_scale` times finer): Blender's
        liquid_step_mesh. phi resampled (values stay in simulation cells, as Blender), + 1,
        joined with the particle level set, extrapolated, outer layer Neumann; the writer sets
        the 0.5 wall layers."""
        b, s, sc = self.b, self.s, self.mesh_scale
        if self.phi_mesh is None:
            hi = tuple(n * sc for n in self.shape)
            self.phi_mesh, self.phi_mesh_parts = b.scalar_grid(hi), b.scalar_grid(hi)
        phi, pp = self.phi_mesh, self.phi_mesh_parts
        if sc == 1:
            b.copy(phi, self.phi)
        else:
            b.interpolate_grid(phi, self.phi)
        if s.mesh_generator == "ANISOTROPIC":
            b.anisotropic_particle_levelset(self.parts, pp, self.shape, s.mesh_particle_radius,
                                            s.mesh_smoothen_pos + self.ANISO_EXTRA_SMOOTHING,
                                            s.mesh_smoothen_neg,
                                            scale=float(sc))
        elif s.mesh_generator == "IMPROVED":
            index = b.grid_particle_index(self.parts, phi.shape, scale=float(sc))
            b.improved_particle_levelset(self.parts, index, pp, s.mesh_particle_radius,
                                         s.mesh_smoothen_pos, s.mesh_smoothen_neg,
                                         s.mesh_concave_lower, s.mesh_concave_upper,
                                         scale=float(sc))
        else:
            index = b.grid_particle_index(self.parts, phi.shape, scale=float(sc))
            b.union_particle_levelset(self.parts, index, pp, s.mesh_particle_radius,
                                      scale=float(sc))
        b.add_const(phi, 1.0)
        b.join(phi, pp)
        b.extrapolate_ls_simple(phi, distance=self.NARROW_BAND + 2, inside=True)
        b.extrapolate_ls_simple(phi, distance=3)
        b.set_bound_neumann(phi, 0)
        return b.to_numpy(phi)

    # ---- time stepping (as SmokeSolver) ------------------------------------

    _adapt_timestep = SmokeSolver._adapt_timestep
    _flow_time = SmokeSolver._flow_time

    def step_frame(self) -> LiquidFrameStats:
        t0 = time.perf_counter()
        dts, iters, pd_iters = [], [], []
        t = 0.0
        if self.guiding:
            self.guiding.update_frame(self.frame + 1)
        while t + FLT_EPSILON < self.frame_length:
            self._adapt_timestep(self.b.max_velocity(self.vel))
            it, pd = self._substep(self.dt)
            iters.append(it)
            if pd is not None:
                pd_iters.append(pd)
            dts.append(self.dt)
            t += self.dt
            self.time_per_frame += self.dt
            if self.time_per_frame + VECTOR_EPSILON > self.frame_length:
                self.time_per_frame = 0.0
                self.lock_dt = False
        self.frame += 1
        if self.snd is not None:
            self._step_secondary()
        self.b.synchronize()
        return LiquidFrameStats(self.frame, len(dts), dts, iters, time.perf_counter() - t0,
                                guide_iters=pd_iters, particles=self.parts.n,
                                secondary=self.snd.n if self.snd is not None else 0)

    # ---- checkpoints -------------------------------------------------------

    def save_state(self) -> dict:
        """Grids, particles and time state. Not bit-exact on resume: the
        particle splats use float atomics."""
        b = self.b
        pos, vel, flag = b.get_particles(self.parts)
        st = {"vel": b.to_numpy(self.vel), "phi": b.to_numpy(self.phi), "ppos": pos,
              "pvel": vel, "pflag": flag, "frame": self.frame, "dt": self.dt,
              "time_per_frame": self.time_per_frame, "lock_dt": self.lock_dt,
              "volume_target": self.volume_target}
        aff = b.get_particle_affine(self.parts)
        if aff is not None:
            st["paffine"] = aff
        if self.snd is not None:
            st["snd_pos"], st["snd_vel"], st["snd_flag"] = b.get_particles(self.snd)
            st["snd_life"] = b.get_particle_life(self.snd)
        return st

    def load_state(self, st) -> None:
        b = self.b
        for k in ("vel", "phi"):
            if tuple(st[k].shape[:3]) != self.shape:
                raise ValueError(f"checkpoint {k} shape {st[k].shape} != grid {self.shape}")
            b.from_numpy(getattr(self, k), np.asarray(st[k], np.float32))
        n = len(st["ppos"])
        if n > self.parts.capacity:
            self.parts = b.particles(int(n * 1.5), apic=self.s.use_apic)
        b.set_particles(self.parts, st["ppos"], st["pvel"], st["pflag"], st.get("paffine"))
        if self.snd is not None and "snd_pos" in st:
            n = len(st["snd_pos"])
            if n > self.snd.capacity:
                self.snd = b.particles(int(n * 1.5), life=True)
            b.set_particles(self.snd, st["snd_pos"], st["snd_vel"], st["snd_flag"],
                            life=st["snd_life"])
        self.frame = int(st["frame"])
        self.dt = float(st["dt"])
        self.time_per_frame = float(st["time_per_frame"])
        self.lock_dt = bool(st["lock_dt"])
        self.volume_target = float(st.get("volume_target", 0.0))
        # Previous positions of moving meshes at the end of the saved frame,
        # so their velocities continue as in an unbroken bake (see SmokeSolver).
        if self.frame >= self.frame_start:
            for o, m in self._moving + self._inflows + self._outflows:
                if o.animated:
                    b.set_mesh_transform(m, self.world_to_cell @ o.matrix_at(float(self.frame)))
            if self.guiding:
                self.guiding.prime(self.frame)

    # ---- one substep -------------------------------------------------------

    def _rasterize_flows(self, t: float, dt: float) -> None:
        """Blender `update_flowsfluids` for liquid flows at scene time t:
        phi_in (reset every substep), invel, and the moving outflows' phiOut."""
        b = self.b
        b.fill(self.phi_in, PHI_MAX)
        if self.invel is not None:
            b.fill(self.invel, (0.0, 0.0, 0.0))
        first = self.frame < self.frame_start
        for f, m in (self._geometry if first else []) + self._inflows:
            if f.behavior == "INFLOW" and f.emit_end and not (f.emit_start - 1 <= t < f.emit_end):
                continue
            if f.animated:
                b.set_mesh_transform(m, self.world_to_cell @ f.matrix_at(t), dt)
            iv = self.invel if f.use_initial_velocity else None
            b.rasterize_mesh_liquid_flow(m, self.phi_in, f.surface_distance, f.plane_init, iv,
                                         f.velocity_normal, f.velocity_factor,
                                         tuple(v * self.vel_scale for v in f.velocity_coord))
        if self._outflows:
            if self.phi_out_static is not None:
                b.copy(self.phi_out, self.phi_out_static)
            else:
                b.fill(self.phi_out, PHI_MAX)
            for f, m in self._outflows:
                b.set_mesh_transform(m, self.world_to_cell @ f.matrix_at(t), dt)
                b.rasterize_mesh_liquid_flow(m, self.phi_out, f.surface_distance, f.plane_init)

    def _rasterize_moving_obstacles(self, t: float, dt: float) -> None:
        """Blender `update_obstacles` + the obstacle block of liquid_adaptive_step:
        phiObsIn and obvel from the moving obstacles, then the static join and
        the level-set clean-up."""
        b = self.b
        rasterize_moving_obstacles(self, self.phi_obs_in, PHI_MAX, t, dt)
        if self.s.robust_collisions:
            b.thin_obstacles([m for _, m in self._moving], self.phi_obs_in,
                             self.obvel_c, self.num_obs)
        b.obstacle_velocity(self.phi_obs_in, self.obvel_c, self.num_obs, self.obvel)
        if self.phi_obs_static is not None:
            b.join(self.phi_obs_in, self.phi_obs_static)
        self._obstacle_levelsets()

    def _collide(self, pos_old) -> None:
        self.b.collide_particles(self.parts, pos_old, self._collision_meshes, self.shape,
                                 self.s.collision_margin,
                                 closed="".join(c for c in "xXyYzZ" if c not in self.s.outflow),
                                 wall=1.0 + self.boundary_width)

    def _substep(self, dt: float) -> tuple[int, int | None]:
        """-> (CG iterations, PD iterations or None without guiding)."""
        b, s, flags, vel, parts, phi = self.b, self.s, self.flags, self.vel, self.parts, self.phi

        t = self._flow_time(dt)
        forces = _update_forces(self, t)  # moving obstacles: last substep's phi_obs
        self._rasterize_flows(t, dt)

        # liquid_adaptive_step. initDomain also resets phiObs to the walls
        # (only needed when obstacles move; see _init_obstacles).
        b.init_domain(flags, boundary_width=self.boundary_width, outflow=s.outflow,
                      phi_walls=self.phi_obs if self._moving else None)
        if self._moving:
            self._rasterize_moving_obstacles(t, dt)
        if self.phi_s_in is not None:
            b.join(self.phi_in, self.phi_s_in)
        b.extrapolate_ls_simple(self.phi_in, distance=6, inside=True)
        b.extrapolate_ls_simple(self.phi_in, distance=3)
        # Volume correction target: the first frame's fill (measured across the join), then
        # the flux leaving the inflows (below), minus what the outflow removes.
        vc = s.volume_correction
        first = vc and self.frame < self.frame_start
        inflow = vc and bool(self._inflows or self.phi_s_in is not None)
        if first:
            self.volume_target -= b.levelset_volume(phi, flags, 1 + self.boundary_width)
        b.join(phi, self.phi_in)
        if first:
            self.volume_target += b.levelset_volume(phi, flags, 1 + self.boundary_width)
        fr = self.fractions
        if fr is not None and (self._moving or self._fractions_stale):
            b.update_fractions(flags, self.phi_obs, fr, boundary_width=1,
                               threshold=s.fractions_threshold)
            self._fractions_stale = False
        b.update_flags_obs(flags, self.phi_obs, self.phi_in, self.phi_out, fractions=fr)
        if self.phi_obs_in is not None:
            b.subtract_levelset(phi, self.phi_obs_in)
        if self.invel is not None:
            b.extrapolate_vec3_simple(self.invel, self.phi_in, distance=6, inside=True)
        b.sample_levelset_with_particles(self.phi_in, flags, parts, s.particle_number,
                                         s.particle_randomness, vel_source=self.invel)
        b.update_flags_from_levelset(flags, phi)
        if inflow:
            self.volume_target += b.source_outflux(flags, self.phi_in, vel, self.invel) * dt

        # liquid_step
        if s.robust_collisions:
            old = b.copy_particle_positions(parts)
        b.advect_particles(parts, flags, vel, dt, delete_in_obstacle=s.delete_in_obstacle)
        if fr is not None and self.phi_obs_in is not None and s.fractions_distance > 0:
            b.push_out_of_obstacles(parts, self.phi_obs_in, thresh=s.fractions_distance)
        if s.robust_collisions:
            self._collide(old)
        else:
            b.push_out_of_obstacles(parts, self.phi_obs)
        if self.snd is not None and not self.time_per_frame:  # start of the frame
            b.copy(self.phi_tmp, phi)
            b.copy(self.vel_tmp, vel)
        b.advect(flags, vel, phi, dt, order=1)
        b.advect_velocity(flags, vel, dt, order=2)

        index = b.grid_particle_index(parts, self.shape)
        b.union_particle_levelset(parts, index, self.phi_parts, s.particle_radius)
        b.add_const(phi, 1.0)
        b.join(phi, self.phi_parts)
        b.extrapolate_ls_simple(phi, distance=self.NARROW_BAND + 2, inside=True)
        b.extrapolate_ls_simple(phi, distance=3)
        b.set_bound_neumann(phi, 0)
        sinks = vc and (not self.domain_closed or self.using_outflow)
        if sinks:
            self.volume_target -= b.levelset_volume(phi, flags, 1 + self.boundary_width)
        if not self.domain_closed or self.using_outflow:
            b.reset_outflow_particles(flags, phi, parts)
        b.update_flags_from_levelset(flags, phi)
        div = 0.0
        if vc:
            vol = b.levelset_volume(phi, flags, 1 + self.boundary_width)
            if sinks:
                self.volume_target += vol
            if self.volume_target > 1.0:
                # Uniform divergence c in FLUID cells changes the volume by c*dt per unit
                # volume; remove `rate` of the relative error per frame, at most 20%.
                err = min(max((vol - self.volume_target) / self.volume_target, -1.0), 1.0)
                corr = min(max(s.volume_correction_rate * err, -0.2), 0.2)
                div = -corr / self.frame_length

        # With APIC, Blender's extrapolateMACFromWeight / combineGridVel run on
        # grids APIC never fills, so they are no-ops: skipped.
        if s.use_apic:
            b.apic_map_parts_to_mac(flags, vel, parts)
        else:
            b.map_parts_to_mac(flags, self.vel_parts, self.vel_old, parts, self.map_weights)
            b.extrapolate_mac_from_weight(self.vel_parts, self.map_weights, distance=2)
            b.combine_grid_vel(self.vel_parts, self.map_weights, vel, phi,
                               narrow_band=self.COMBINE_BAND, thresh=0.0)
        b.copy(self.vel_old, vel)

        b.add_gravity(flags, vel, tuple(g * dt for g in self.gravity))
        if forces:
            b.add_force_field(flags, vel, self.forces, _force_scale(self))
        into_obs = fr is not None
        b.extrapolate_mac_simple(flags, vel, distance=2, into_obs=into_obs)
        self._wall_bcs()
        if s.use_diffusion:  # alpha = kinematic viscosity * dt * res^2, domain size in metres
            kin = s.viscosity_base * 10.0 ** -s.viscosity_exponent / self.domain_size ** 2
            self._wall_bcs()
            b.solve_diffusion(flags, vel, kin * dt * float(max(self.shape)) ** 2)
            self._wall_bcs()
        if s.use_viscosity:  # Mantaflow units: dx = 1 / maxres
            b.apply_viscosity(flags, phi, vel, s.viscosity_value, dt, 1.0 / max(self.shape))
            self._wall_bcs()
        pd = None
        if self.guiding:  # PD_fluid_guiding: phi (ghost fluid), no obvel
            gst = self.guiding.solve(flags, vel, self.pressure, accuracy=s.cg_accuracy,
                                     zero_pressure_fixing=self.domain_closed, divergence=div,
                                     fractions=fr, phi=phi)
            it, pd = gst.pressure_iterations, gst.iterations
        else:
            it = b.solve_pressure(flags, vel, self.pressure, accuracy=s.cg_accuracy,
                                  zero_pressure_fixing=self.domain_closed, divergence=div,
                                  fractions=fr, obvel=self.obvel if fr is not None else None,
                                  phi=phi if s.quality == "HIGH" else None,
                                  surface_tension=self.surface_tension * dt).iterations
        b.extrapolate_mac_simple(flags, vel, distance=4, into_obs=into_obs)
        self._wall_bcs()
        if fr is None:
            b.extrapolate_mac_simple(flags, vel)

        b.adjust_number(parts, vel, flags, s.particle_min, s.particle_max, phi,
                        radius_factor=s.particle_radius, narrow_band=s.particle_band_width,
                        exclude=self.phi_obs)
        if s.robust_collisions:  # particles just seeded inside solids
            self._collide(None)
        if s.use_apic:
            b.apic_map_mac_to_parts(parts, vel, flags)
        else:
            if s.flip_wall_fix:
                b.extrapolate_mac_simple(flags, self.vel_old, distance=0)
            b.flip_velocity_update(flags, vel, self.vel_old, parts, s.flip_ratio)
        return it, pd
