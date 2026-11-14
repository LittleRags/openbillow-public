# SPDX-License-Identifier: GPL-3.0-or-later
"""The backend interface: every GPU operation the solver needs.

The solver (step order, timestep, settings) and the I/O never touch a GPU
library directly; they hold opaque grid handles and call these methods. A new
GPU (AMD, Apple, CPU fallback) means one new subclass of `Backend`.

Conventions (copied from Mantaflow so its kernels port line for line):
- Grids are dense, shape (nx, ny, nz), indexed [i, j, k]; k is contiguous.
- Units are cells and solver time, as in Mantaflow. The solver does the
  Blender-unit scaling (see reference/NOTES.md) before calling in here.
- Scalar grids are cell-centred float32.
- MAC (velocity) grids have the same shape as the cell grid. Component x of
  cell (i, j, k) sits on the face at (i - 1/2, j, k), y at (i, j - 1/2, k),
  z at (i, j, k - 1/2). The far faces of the last cells are not stored; the
  outer cells are walls or outflow, so they are never needed.
- Vector grids (`vector_grid`) are cell-centred float32 Vec3.
- Flag grids are int32 bit masks with Mantaflow's values (`Flags`).
- Operations that Mantaflow does in place are in place here too. Backends may
  keep internal scratch grids (e.g. MacCormack's forward/backward passes).
- Methods return without waiting for the GPU unless they return a host value
  (`to_numpy`, reductions, `solve_pressure` stats).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import numpy as np

Shape = tuple[int, int, int]
Vec3 = tuple[float, float, float]

# Opaque handles. Only the backend that made a grid knows what is inside.
ScalarGrid = Any
MACGrid = Any
VectorGrid = Any
FlagGrid = Any
Grid = Any
Particles = Any      # particle arrays (pos, vel, flag) with a fixed capacity
ParticleIndex = Any  # particles sorted into cells, see `grid_particle_index`


class Flags(IntFlag):
    """Cell types, same bits as Mantaflow's FlagGrid (grid.h)."""
    NONE = 0
    FLUID = 1
    OBSTACLE = 2
    EMPTY = 4
    INFLOW = 8
    OUTFLOW = 16
    OPEN = 32
    STICK = 64


class ParticleFlags(IntFlag):
    """Particle flag bits, same as Mantaflow's ParticleBase (particle.h)."""
    NONE = 0
    NEW = 1 << 0
    SPRAY = 1 << 1    # secondary particle types
    BUBBLE = 1 << 2
    FOAM = 1 << 3
    TRACER = 1 << 4
    DELETE = 1 << 10  # inactive: skipped by every particle operation


class BackendOutOfMemory(RuntimeError):
    """A backend could not allocate device (or host) memory."""


@dataclass
class BackendMemory:
    total: int      # device memory, bytes
    free: int       # free right now, bytes
    allocated: int  # held by this backend's grids and scratch, bytes


@dataclass
class PressureStats:
    iterations: int
    residual: float  # max abs divergence left after the solve
    capped: bool = False  # viscosity only: stopped at the iteration cap without converging


@dataclass
class GuidingStats:
    iterations: int           # PD iterations (= pressure solves)
    pressure_iterations: int  # CG iterations summed over all pressure solves
    residual: float           # max abs divergence after the last pressure solve


class Backend(ABC):
    name: str = "base"

    # ---- device and memory -------------------------------------------------

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Human-readable GPU name, for logs and the add-on UI."""

    @property
    def on_cpu(self) -> bool:
        """True when grids live in system RAM instead of on a GPU."""
        return False

    @property
    def memory_mode(self) -> str:
        """Where the grids live: "gpu", "managed" (GPU paging through system RAM) or "cpu"."""
        return "cpu" if self.on_cpu else "gpu"

    @abstractmethod
    def memory(self) -> BackendMemory:
        """Device memory, for the VRAM estimate and warnings."""

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all queued GPU work (for timing)."""

    # ---- allocation and transfer -------------------------------------------

    @abstractmethod
    def scalar_grid(self, shape: Shape, value: float = 0.0) -> ScalarGrid:
        """New cell-centred float grid filled with `value`."""

    @abstractmethod
    def mac_grid(self, shape: Shape) -> MACGrid:
        """New zeroed staggered velocity grid."""

    @abstractmethod
    def vector_grid(self, shape: Shape) -> VectorGrid:
        """New zeroed cell-centred Vec3 grid (Mantaflow Grid<Vec3>)."""

    @abstractmethod
    def flag_grid(self, shape: Shape) -> FlagGrid:
        """New flag grid, all `Flags.FLUID`."""

    @abstractmethod
    def fill(self, grid: Grid, value: float | int | Vec3) -> None:
        """Set every cell of `grid` to `value`."""

    @abstractmethod
    def copy(self, dst: Grid, src: Grid) -> None:
        """dst = src (same kind and shape)."""

    @abstractmethod
    def join(self, dst: ScalarGrid, src: ScalarGrid, keep_max: bool = False) -> None:
        """dst = min(dst, src) per cell (Mantaflow `LevelsetGrid::join`), or
        max with keep_max (`Grid<Real>::join`)."""

    @abstractmethod
    def mult_const(self, grid: ScalarGrid | MACGrid | VectorGrid, factor: float) -> None:
        """grid *= factor. Mantaflow `Grid::multConst`."""

    @abstractmethod
    def add_const(self, grid: ScalarGrid, value: float) -> None:
        """grid += value. Mantaflow `Grid::addConst`."""

    @abstractmethod
    def set_bound_neumann(self, grid: ScalarGrid, boundary_width: int = 1) -> None:
        """Copy the nearest inner value into the outer `boundary_width`+1
        layers. Mantaflow `Grid::setBoundNeumann`."""

    @abstractmethod
    def mirror_slab(self, grid: ScalarGrid | MACGrid) -> None:
        """2D slab (ny == 3): copy the middle slice j = 1 into j = 0 and 2, so
        every y-neighbour read sees it (Mantaflow's 2D stride-Z 0). MAC grids
        get their y faces zeroed in all three slices."""

    @abstractmethod
    def to_numpy(self, grid: Grid) -> np.ndarray:
        """Download: float32 (nx,ny,nz), float32 (nx,ny,nz,3) for vector and
        MAC grids (MAC: face values, not centred), or int32 (nx,ny,nz) for flags.
        Always an independent copy: later grid operations must not change it."""

    @abstractmethod
    def from_numpy(self, grid: Grid, array: np.ndarray) -> None:
        """Upload `array` (layout as in `to_numpy`) into an existing grid."""

    # ---- reductions (return one host value) --------------------------------

    @abstractmethod
    def sum(self, grid: ScalarGrid) -> float:
        """Sum of all cells (total density, for metrics)."""

    @abstractmethod
    def particle_sum(self, parts) -> float:
        """Sum of every live particle's position and velocity components: NaN or infinite
        as soon as one of them is."""

    @abstractmethod
    def max_abs(self, grid: ScalarGrid) -> float:
        """Largest |value|."""

    @abstractmethod
    def max_velocity(self, vel: MACGrid) -> float:
        """Largest velocity magnitude in cells per solver time unit, used by
        the adaptive timestep (Mantaflow `vel.getMax()` in adaptTimestep)."""

    # ---- domain ------------------------------------------------------------

    @abstractmethod
    def init_domain(self, flags: FlagGrid, boundary_width: int = 0,
                    outflow: str = "", phi_walls: ScalarGrid | None = None) -> None:
        """Reset flags: interior FLUID, outer layer of `boundary_width`+1 cells
        OBSTACLE. Sides named in `outflow` ("xXyYzZ", lower case = low side)
        become OUTFLOW instead. Where sides meet, z wins over y over x.
        Mantaflow `FlagGrid.initDomain` followed by `fillGrid()`, as Blender's
        smoke script calls them each step. With `phi_walls`, that grid becomes
        the level set of the wall sides (1e9, then min with i - w - 0.5 per
        wall), as the liquid script asks for (phiWalls=phiObs)."""

    # ---- emission ----------------------------------------------------------

    # ---- meshes (flows and obstacles) --------------------------------------

    @abstractmethod
    def create_mesh(self, verts: np.ndarray, tris: np.ndarray) -> Any:
        """Upload a triangle mesh: verts (n, 3) in object space, tris (m, 3)
        vertex indices (counter-clockwise = outward normal, as Blender).
        The transform starts as the identity."""

    @abstractmethod
    def set_mesh_transform(self, mesh: Any, matrix: np.ndarray, dt: float = 0.0) -> None:
        """Set the 4x4 object → cell matrix (world matrix composed with
        Blender's manta_pos_to_cell) and rebuild the mesh's bounding box.
        Also sets the vertex velocities (cells per solver time) used by
        `rasterize_mesh_obstacle`: (new - previous cell positions) / dt, as
        Blender's obstacles_from_mesh; zero on the first call or if dt <= 0."""

    @abstractmethod
    def rasterize_mesh_flow(self, mesh: Any, phi_in: ScalarGrid, emission_in: ScalarGrid,
                            density_in: ScalarGrid, heat_in: ScalarGrid,
                            density: ScalarGrid, heat: ScalarGrid,
                            flow_density: float, temperature: float,
                            surface_distance: float = 1.0, volume_density: float = 0.0,
                            plane_init: bool = False,
                            fuel_in: ScalarGrid | None = None, react_in: ScalarGrid | None = None,
                            fuel: ScalarGrid | None = None, react: ScalarGrid | None = None,
                            fuel_amount: float = 0.0, emit_fuel: bool = False) -> None:
        """Blender `emit_from_mesh` for an absolute-mode smoke flow, over the
        mesh bbox + 6 cells. Strength s (sample_mesh) = sqrt(1 - d/sd) from
        the nearest surface point within 1.732 cells (0 if sd = 0), max with
        volume_density where a +x ray hits a face from behind and a -x ray
        hits too. phi_in = min(phi_in, update_distances) with the 26-ray
        inside test (or plane init), minus surface_distance. Then
        (apply_inflow_fields): emission_in = max(emission_in, s); where s > 0,
        heat_in = ADD_IF_LOWER(heat, temperature); where s*flow_density >
        density, density_in = max(density_in, s*flow_density). The caller
        starts density_in/heat_in as copies of density/heat.
        With fire grids (fuel_in/react_in start as copies of fuel/react), after
        density: ff = s*fuel_amount; if emit_fuel and ff != 0 and ff > fuel,
        fuel_in = max(fuel_in, ff). Then value = 1 - (1-s)^2; if fuel_in > FLT_EPSILON
        and value > react: f = ff/fuel_in, react_in = clamp(value*f + (1-f)*react,
        0, value) (for SMOKE flows too, with their fuel_amount). FIRE flows pass
        flow_density 0."""

    @abstractmethod
    def rasterize_mesh_obstacle(self, mesh: Any, phi_obs: ScalarGrid,
                                surface_thickness: float = 0.0,
                                plane_init: bool = False,
                                obvel: VectorGrid | None = None,
                                num_obs: ScalarGrid | None = None,
                                guide_mode: str | None = None, vel_multi: float = 1.0) -> None:
        """Blender `obstacles_from_mesh` + `apply_effector_fields`: phi_obs =
        min(phi_obs, update_distances) over the mesh bbox + 6 cells. With
        `obvel` and `num_obs` (moving obstacles), in cells where this mesh's
        distance is < 0: num_obs += 1 and obvel += the barycentric vertex
        velocity at the nearest surface point within 1.732 cells (0 if none).
        The caller zeroes both before the first obstacle of a substep.
        With `guide_mode` (a guide effector) the added velocity v = vel_multi
        * vertex velocity goes through Blender's mode on a zeroed map:
        OVERRIDE v, AVERAGED v/2, MAX |v| per component, MIN 0."""

    @abstractmethod
    def obstacle_velocity(self, phi_obs: ScalarGrid, obvel_c: VectorGrid,
                          num_obs: ScalarGrid, obvel: MACGrid) -> None:
        """Turn the summed cell velocities from `rasterize_mesh_obstacle`
        into the MAC obstacle velocity for `set_wall_bcs`, as Blender's smoke
        step: obvel_c /= num_obs (fluid_pre_step), then Mantaflow
        `extrapolateVec3Simple(obvel_c, phi_obs, 6, inside=True)`, again with
        (3, inside=False), and `resampleVec3ToMac(obvel_c, obvel)`. phi_obs is
        the moving obstacles' levelset (Blender's phiObsIn before the static
        join). obvel_c is changed in place; obvel's outer layer is left as is."""

    @abstractmethod
    def extrapolate_vec3_simple(self, vel: VectorGrid, phi: ScalarGrid, distance: int = 4,
                                inside: bool = False) -> None:
        """Mantaflow `extrapolateVec3Simple`: copy cell vectors from the known
        side of phi (phi < 0, or phi > 0 with inside) `distance` layers into
        the other side; cells further away get 0. Outer layer untouched."""

    @abstractmethod
    def rasterize_mesh_liquid_flow(self, mesh: Any, phi: ScalarGrid,
                                   surface_distance: float = 0.0, plane_init: bool = False,
                                   invel: VectorGrid | None = None, vel_normal: float = 0.0,
                                   vel_multi: float = 1.0, vel_coord: Vec3 = (0.0, 0.0, 0.0)
                                   ) -> None:
        """Blender `emit_from_mesh` + `apply_inflow_fields` for a liquid flow
        (or `apply_outflow_fields` for an outflow, without invel), over the mesh
        bbox + 6 cells: phi = min(phi, update_distances - surface_distance).
        With `invel` (initial velocity on): where a surface point lies within
        1.732 cells, v = vel_normal * (interpolated angle-weighted vertex
        normal) + vel_multi * (vertex velocity, see set_mesh_transform) +
        vel_coord (cells per solver time), and invel = v if |v|² > |invel|²."""

    @abstractmethod
    def set_inflow_flags(self, flags: FlagGrid, phi_in: ScalarGrid) -> None:
        """Mark fluid cells with phi_in < 0 as INFLOW."""

    @abstractmethod
    def set_obstacle_flags(self, flags: FlagGrid, phi_obs: ScalarGrid) -> None:
        """Set cells with phi_obs < 0 to plain OBSTACLE, skipping the outermost
        layer (overrides INFLOW). Obstacle part of Mantaflow `KnUpdateFlagsObs`
        (setObstacleFlags, boundaryWidth=1); call after set_inflow_flags."""

    @abstractmethod
    def apply_emission(self, flags: FlagGrid, target: ScalarGrid,
                       source: ScalarGrid, emission: ScalarGrid,
                       absolute: bool = True) -> None:
        """In INFLOW or OUTFLOW cells, and in any cell with emission != 0:
        target = source (absolute) or target += source. Mantaflow
        `applyEmission` (initplugins.cpp) with type=Inflow|Outflow and
        emissionTexture, as Blender calls it."""

    # ---- simulation step (order in reference/NOTES.md) ---------------------

    @abstractmethod
    def dissolve(self, flags: FlagGrid, grid: ScalarGrid, speed: float,
                 log_falloff: bool, scale: float) -> None:
        """Fade a smoke field. Mantaflow `dissolveSmoke`, one grid at a time."""

    @abstractmethod
    def advect(self, flags: FlagGrid, vel: MACGrid, grid: ScalarGrid,
               dt: float, order: int = 2, strength: float = 1.0,
               clamp_mode: int = 2) -> None:
        """Advect a scalar grid through `vel` in place. order 1 =
        semi-Lagrangian, 2 = MacCormack; clamp_mode 1 = hard clamp to the
        source cells' range, 2 = fall back to semi-Lagrangian when out of
        range. Mantaflow `advectSemiLagrange` (orderSpace=orderTrace=1, its
        deprecated openBounds/boundaryWidth dropped). Like Mantaflow it
        zeroes the outermost cell layer. 3D grids only. `grid` may also be a
        vector grid (the noise UV grids); clamp_mode 2 then falls back for the
        whole vector when any component is out of range."""

    @abstractmethod
    def advect_velocity(self, flags: FlagGrid, vel: MACGrid, dt: float,
                        order: int = 2, strength: float = 1.0,
                        clamp_mode: int = 2, field: MACGrid | None = None) -> None:
        """Self-advect the velocity grid in place (same algorithm, MAC
        sampling), then extrapolate velocities into OUTFLOW cells as
        Mantaflow's `applyOutflowBC` does. With `field`, advect that MAC grid
        in place through `vel` instead (advection-reflection's second half)."""

    @abstractmethod
    def reflect(self, vel: MACGrid, pre: MACGrid) -> None:
        """pre = 2 * vel - pre (advection-reflection, Zehnder et al. 2018)."""

    @abstractmethod
    def reset_outflow(self, flags: FlagGrid, grid: ScalarGrid) -> None:
        """Zero `grid` in OUTFLOW cells and set their flags to (f | EMPTY) & ~FLUID.
        Mantaflow `resetOutflow`; the EMPTY bit makes the pressure solve use p = 0 there."""

    @abstractmethod
    def vorticity_confinement(self, flags: FlagGrid, vel: MACGrid, strength: float,
                              strength_cell: ScalarGrid | None = None,
                              cell_scale: float = 1.0) -> None:
        """Mantaflow `vorticityConfinement`; `strength` already scaled by dt.
        With `strength_cell`, the per-cell strength is strength + cell_scale * strength_cell
        (fire: strength_cell = fuel, cell_scale = flame_vorticity * dt / frame_length)."""

    # ---- fire (Mantaflow plugin/fire.cpp, colours not supported) ----------

    @abstractmethod
    def process_burn(self, fuel: ScalarGrid, density: ScalarGrid, react: ScalarGrid,
                     heat: ScalarGrid | None, burning_rate: float, flame_smoke: float,
                     ignition_temp: float, max_temp: float, dt: float) -> None:
        """Burn fuel, scale react with it, add smoke to density (unclamped, as in
        Mantaflow) and set heat from the flame profile. Mantaflow `processBurn`;
        `dt` is the substep in solver units (not divided by the frame length)."""

    @abstractmethod
    def update_flame(self, react: ScalarGrid, flame: ScalarGrid) -> None:
        """flame = sqrt(react) where react > 0, else 0. Mantaflow `updateFlame`."""

    # ---- fire, COMBUSTION model (not in Mantaflow) -------------------------

    @abstractmethod
    def combust(self, fuel: ScalarGrid, density: ScalarGrid, heat: ScalarGrid,
                expansion_div: ScalarGrid, burning_rate: float, ignition_temp: float,
                max_temp: float, heat_release: float, flame_smoke: float, cooling: float,
                cooling_radiative: float, expansion: float, dt: float,
                dt_seconds: float) -> None:
        """First cool all heat: heat /= 1 + dt_seconds * (cooling + cooling_radiative
        * (|heat|/max_temp)^3). Then burn fuel * (1 - exp(-burning_rate * dt)) (dt in
        solver units), adding 0.05 * flame_smoke per unit to density. All fuel burns at
        that rate, but only a share r of it counts as burned for heat and expansion:
        with t0 = heat before cooling, r = smoothstep(clamp((t0 - 0.75 * ignition_temp)
        / (0.25 * ignition_temp), 0, 1)), i.e. 0 below 0.75 * ignition_temp and 1 at
        ignition_temp and above (a hard switch flickers and bands). Heat is raised by
        heat_release * r * burned (capped at max_temp); below the ramp fuel burns as
        smoke only. expansion_div = expansion * r * burned / dt (pass to solve_pressure
        as divergence_grid); 0 in the outer layer."""

    @abstractmethod
    def fire_forces(self, flags: FlagGrid, vel: MACGrid, heat: ScalarGrid, disturbance: float,
                    size_cells: float, phase: float, ignition_temp: float, shredding: float,
                    max_temp: float, seed: int = 0) -> None:
        """Add two cell-centred forces (averaged onto faces) to vel.
        Disturbance: a smooth random vector field in [-1, 1]^3 per component,
        lattice spacing size_cells, re-drawn at each integer phase and blended
        between them, times disturbance * clamp(heat / ignition_temp, 0, 1).
        Shredding: with n = grad(heat)/|grad(heat)| (skipped where |grad| <=
        1e-3 * max_temp or heat < 0.1 * ignition_temp), add s * (v.n) n with
        s = shredding at heat >= ignition_temp and -min(shredding, 1) below.
        Both values are per call (already times dt)."""

    @abstractmethod
    def update_flame_temperature(self, heat: ScalarGrid, flame: ScalarGrid,
                                 ignition_temp: float, max_temp: float) -> None:
        """flame = clamp((heat - ignition_temp) / (max_temp - ignition_temp), 0, 1)."""

    @abstractmethod
    def add_buoyancy(self, flags: FlagGrid, vel: MACGrid, grid: ScalarGrid,
                     strength: Vec3) -> None:
        """vel += strength * grid (grid averaged onto each face), fluid faces
        only. The solver passes strength = -gravity * coeff * dt in cell
        units. Mantaflow `addBuoyancy` with scale=False."""

    @abstractmethod
    def compute_forces(self, flags: FlagGrid, vel: MACGrid, force: VectorGrid, fields: list,
                       weights: Any, origin: Vec3, cell_size: Vec3, fps: float,
                       mat3: np.ndarray | None = None, density: ScalarGrid | None = None,
                       fuel: ScalarGrid | None = None,
                       phi_obs: ScalarGrid | None = None) -> bool:
        """Blender's `update_effectors_task_cb`: `force` = field objects' force per
        cell (x0.2, clamped to +-1), 0 where skipped. `fields` are
        `forcefields.ForceField`, `weights` an `EffectorWeights`. Returns False
        (and leaves `force` untouched) when no field is relevant."""

    @abstractmethod
    def add_force_field(self, flags: FlagGrid, vel: MACGrid, force: VectorGrid,
                        scale: float = 1.0) -> None:
        """vel += scale * force averaged onto faces. Mantaflow `addForceField`
        (additive, isMAC=False); the solver passes Blender's scaleSpeedFrames."""

    @abstractmethod
    def set_wall_bcs(self, flags: FlagGrid, vel: MACGrid,
                     obvel: MACGrid | None = None, fractions: MACGrid | None = None,
                     phi_obs: ScalarGrid | None = None) -> None:
        """Set the normal velocity on faces next to obstacles to the
        obstacle velocity `obvel` at that face (0 without it). Mantaflow
        `setWallBcs`. With `fractions` and `phi_obs` (KnSetWallBcsFrac, obvel
        ignored): on faces touching an OBSTACLE cell, remove the velocity
        component along the phi_obs gradient; the outer layer is left alone."""

    @abstractmethod
    def solve_pressure(self, flags: FlagGrid, vel: MACGrid,
                       pressure: ScalarGrid, accuracy: float = 1e-3,
                       max_iter_factor: float = 1.5,
                       zero_pressure_fixing: bool = False,
                       divergence: float = 0.0, fractions: MACGrid | None = None,
                       obvel: MACGrid | None = None,
                       phi: ScalarGrid | None = None,
                       divergence_grid: ScalarGrid | None = None,
                       surface_tension: float = 0.0,
                       parallel_coarsening: bool = False,
                       face_weights: MACGrid | None = None) -> PressureStats:
        """`parallel_coarsening` builds the multigrid hierarchy with a GPU rule instead
        of Mantaflow's serial one: other iteration counts, same converged answer (High
        Quality only). With `fractions` (from `update_fractions`), the matrix and the
        divergence are weighted by the face fractions, and `obvel` adds the
        (1 - fraction) obstacle-velocity flux (Mantaflow's second-order
        obstacle boundaries). Make `vel` divergence-free: divergence, multigrid-preconditioned CG
        for `pressure`, then subtract the pressure gradient. Mantaflow
        `solvePressure`. The multigrid hierarchy is rebuilt whenever the
        flag layout changes, so moving obstacles get what PcMGDynamic gives
        and static ones what PcMGStatic gives. As in Mantaflow, CG starts from
        pressure = 0 (no warm start), stops when the max-norm residual is below
        `accuracy`, and runs at most 100 iterations (`max_iter_factor` only
        applies to non-multigrid preconditioners, so it is unused here).
        Set `zero_pressure_fixing` for a fully closed domain (a cell is pinned
        only if there are no EMPTY cells). EMPTY cells are the liquid free
        surface: p = 0 there. That is all Blender's plain liquid script does:
        it passes no phi. Only its guided liquid (PD_fluid_guiding) passes the
        liquid `phi`: ghost-fluid weights on the matrix diagonal and the
        correction, plus knReplaceClampedGhostFluidVels (gfClamp 1e-4).
        `surface_tension` (with phi; Mantaflow surfTens, grid units, already
        times dt): pressure jump surface_tension * curvature at the surface, the
        curvature (CurvatureOp, clamped to +-1) taken from phi.
        `divergence` (not in Mantaflow): target divergence of the result in
        FLUID cells, added to the rhs; < 0 compresses (liquid volume correction).
        `divergence_grid` (not in Mantaflow): the same per cell, added on top
        (combustion gas expansion).
        `face_weights` (not in Mantaflow; not with phi): weight w of each face, laid
        out as a MAC grid (from `density_face_weights`), or None. Solves
        div(w grad p) = div u and subtracts w grad p (variable-density projection,
        w = 1 / rho_face); combined with `fractions` as w * fraction in the matrix.
        Weights may change every call, so the matrix is rebuilt on each call
        that passes them; None keeps the constant-coefficient solve unchanged."""

    @abstractmethod
    def density_face_weights(self, heat: ScalarGrid, weights: MACGrid, ambient_kelvin: float,
                             flame_kelvin: float, flame_max_temp: float) -> None:
        """weights = 1 / rho_face per lower face, rho_face = (rho_lo + rho_hi) / 2,
        rho = ambient_kelvin / T the ideal-gas density relative to ambient, T =
        ambient_kelvin + max(heat, 0) (flame_kelvin - ambient_kelvin) / flame_max_temp.
        Faces at index 0 use the cell's own density. For `solve_pressure(face_weights=)`."""

    # ---- flame-front fire model (3D only; A6 variant C, CODE-GAS-020) ----------

    @abstractmethod
    def flame_front_step(self, flags: FlagGrid, phi_in: ScalarGrid, phi: ScalarGrid,
                         heat: ScalarGrid, density: ScalarGrid, fuel: ScalarGrid,
                         fuel_in: ScalarGrid, expansion_div: ScalarGrid, *, burn_cells: float,
                         expansion_rate: float, fuel_heat: float, flame_max_temp: float,
                         cooling: float, cooling_radiative: float, dt_seconds: float,
                         smoke_yield: float, fuel_amount: float = 0.0, glow_cells: float = 6.0,
                         source_gap: int = 3, band: float = 6.0,
                         reinit_iterations: int = 2) -> None:
        """One step of the level-set fuel core `phi` (< 0 inside, cells; Nguyen, Fedkiw
        and Jensen 2002), after emission and before advection. Emitters are cells with
        phi_in < 0 that are not OBSTACLE. phi is set to `band` on the border and to at most
        -1 on emitters, burned inward by `burn_cells` (S dt in cells) in ceil(burn_cells /
        0.5) Godunov steps, re-pinned at the emitters, reinitialised (`reinit_iterations`
        steps, dtau 0.5) and clamped to +-band. With frac = clamp(0.5 - phi, 0, 1), burned =
        max(frac before - frac after, 0):
        heat: cooled as in `combust`, then at least flame_max_temp * max(burned / burn_cells,
        clamp(1 - phi, 0, 1) on non-core cells next to the core); in the core fuel_heat at
        emitters, else fuel_heat ramped to flame_max_temp over the `glow_cells` next to the
        front. fuel: fuel_amount * frac when fuel_amount > 0 (one amount for all flows);
        when 0, fuel_in at emitters and elsewhere the old fuel times min(1, frac / frac
        before) (0 outside the core). density += smoke_yield * burned * the cell's amount.
        expansion_div = expansion_rate * delta(phi - 1.5) * |grad phi| (cosine delta of
        half-width 1.5 cells, product side only), 0 on the border and within `source_gap`
        cells of an emitter."""

    @abstractmethod
    def density_lift(self, heat: ScalarGrid, lift: ScalarGrid, ambient_kelvin: float,
                     flame_kelvin: float, flame_max_temp: float) -> None:
        """lift = rho_amb / rho - 1 = T / ambient_kelvin - 1 (T as `density_face_weights`),
        for `add_buoyancy(flags, vel, lift, -gravity * dt)`."""

    @abstractmethod
    def fuel_mass_source(self, flags: FlagGrid, phi_in: ScalarGrid, expansion_div: ScalarGrid,
                         inject_cells: float) -> None:
        """Fuel gas entering as a mass source: expansion_div += inject_cells * (faces
        between emitter and other cells) / (emitter cells), on the emitter cells (emitters
        as in `flame_front_step`). inject_cells is the inflow speed in cells per time unit."""

    # ---- mixture-fraction fire model DIFFUSION (3D only; CODE-GAS-036, CODE-GAS-037) ----

    @abstractmethod
    def mixfrac_init(self, shape: Shape, *, dx: float, vel_scale: float, ambient_kelvin: float,
                     kelvin_per_heat: float, heat_of_combustion: float, air_fuel_ratio: float,
                     radiative_fraction: float, soot_yield: float, rho_gain: float = 0.3) -> Any:
        """State of the mixture-fraction model on a 3D grid (an opaque handle for the other
        mixfrac_* calls). It owns the transported density `.sig` = rho / rho_amb (a scalar grid,
        1 at start; persistent: checkpoint it), the source rate grid, k_sgs and work buffers.
        dx in m, heat_of_combustion in J/kg, air_fuel_ratio kg/kg, the fractions per unit;
        kelvin_per_heat = (flame_kelvin - ambient_kelvin) / flame_max_temp (heat <-> K as `gas_kelvin`)."""

    @abstractmethod
    def mixfrac_source(self, state: Any, flags: FlagGrid, phi_in: ScalarGrid, hrrpua_kw_m2: float) -> None:
        """Fuel mass source from the emitters (phi_in < 0, not OBSTACLE): per (i, j) column with
        n emitter cells, each gets HRRPUA / (dH_c dx) / n kg/(m^3 s), so the source releases
        HRRPUA (kW/m^2) times the emitters' footprint on the x-y plane."""

    @abstractmethod
    def mixfrac_burn(self, state: Any, flags: FlagGrid, vel: MACGrid, fuel: ScalarGrid,
                     products: ScalarGrid, heat: ScalarGrid, smoke: ScalarGrid,
                     expansion_div: ScalarGrid, dt_seconds: float) -> None:
        """Source, eddy-dissipation burn, heat (written from the products), soot into `smoke`
        and the target divergence (warp_mixfrac.MixtureFraction.burn). fuel and products are
        the mass fractions Y_F and Y_P."""

    @abstractmethod
    def mixfrac_transport(self, state: Any, flags: FlagGrid, vel: MACGrid, fuel: ScalarGrid,
                          products: ScalarGrid, heat: ScalarGrid, dt_seconds: float) -> int:
        """Conservative finite-volume transport of rho, rho Y_F, rho Y_P with `vel`; rewrites
        heat from Y_P. -> number of RK2 substeps."""

    @abstractmethod
    def guide_blur(self, flags: FlagGrid, vel: MACGrid, radius: int = 5,
                   two_d: bool = False) -> None:
        """In place: separable Gaussian blur of all three MAC components
        (kernel size 2 radius + 1, sigma = size, sum 1, zero-padded) along x,
        y, z (`two_d`: x and z only, for the slab), then faces touching an
        OBSTACLE cell (own cell or the lower neighbour per blurred axis) get
        their old value back. Mantaflow `applySeparableKernel`."""

    @abstractmethod
    def guiding_solve(self, flags: FlagGrid, vel: MACGrid, vel_target: MACGrid,
                      weight: ScalarGrid, pressure: ScalarGrid, blur_radius: int = 5,
                      tau: float = 1.0, sigma: float = 0.99, theta: float = 1.0,
                      eps_rel: float = 1e-3, eps_abs: float = 1e-3, max_iters: int = 200,
                      two_d: bool = False, accuracy: float = 1e-3,
                      zero_pressure_fixing: bool = False, divergence: float = 0.0,
                      fractions: MACGrid | None = None,
                      phi: ScalarGrid | None = None,
                      parallel_coarsening: bool = False) -> GuidingStats:
        """Replaces `solve_pressure` when guiding: makes `vel` divergence-free
        while pulling its blurred version toward `vel_target` with per-cell
        `weight` (Blender: alpha everywhere, so it also damps the flow where
        the target is 0). Mantaflow `PD_fluid_guiding`: a primal-dual loop
        with one full `solve_pressure` (the trailing arguments) per iteration,
        stopping when max|z - z0| < sqrt(dims) eps_abs + eps_rel max|z|.
        Blender passes tau 1, sigma 0.99, theta 1."""

    @abstractmethod
    def apply_viscosity(self, flags: FlagGrid, phi: ScalarGrid, vel: MACGrid,
                        viscosity: float, dt: float, dx: float, accuracy: float = 1e-9,
                        max_iter_factor: float = 1.5) -> PressureStats:
        """Implicit viscosity on `vel` (liquid), Mantaflow `applyViscosity` with a
        constant viscosity grid (Blender's Viscosity panel Strength, used as is).
        dt and dx in Mantaflow units (dx = 1 / maxres). Three coupled CGs without
        a preconditioner, L2 residual < accuracy, at most int(max_iter_factor *
        max(shape)) iterations; faces next to obstacles (and unsolved border
        faces) come out 0, so call set_wall_bcs afterwards, as Blender does.
        Stats: most iterations of the three CGs, largest final L2 residual."""

    @abstractmethod
    def apply_viscosity_coupled(self, flags: FlagGrid, phi: ScalarGrid, vel: MACGrid,
                                viscosity: float, dt: float, dx: float) -> PressureStats:
        """High Quality viscosity (CODE-LIQ-014): the same discrete system as
        `apply_viscosity`, solved by one Jacobi-preconditioned CG over all three
        components. Stats: iterations, final preconditioned relative residual,
        and whether it stopped without converging."""

    @abstractmethod
    def solve_diffusion(self, flags: FlagGrid, vel: MACGrid, alpha: float,
                        accuracy: float = 1e-4, max_iter_factor: float = 1.0,
                        kelvin: ScalarGrid | None = None) -> PressureStats:
        """Implicit diffusion of `vel` (liquid), Mantaflow `cgSolveDiffusion`:
        (I - alpha * Laplace) u' = u per component, applied on FLUID cells, with
        alpha = coefficient * dt / dx^2 in grid units. Un-preconditioned CG, L2
        residual < accuracy, at most int(max_iter_factor * max(shape)) iterations.
        With `kelvin` (gas temperature in K; gas viscosity, CODE-GAS-029) it is instead the
        variable-density viscous step of air at 1 atm, rho u' - alpha div(mu grad u') = rho u,
        mu by Sutherland's law and rho = p / (R T), alpha = dt / dx^2 (seconds, metres).
        Stats: most iterations of the three CGs, largest final residual."""

    @abstractmethod
    def gas_kelvin(self, heat: ScalarGrid, kelvin: ScalarGrid, ambient_kelvin: float,
                   flame_kelvin: float, flame_max_temp: float) -> None:
        """kelvin = the FLAME_FRONT temperature T = ambient + max(heat, 0) (flame - ambient) / flame_max_temp."""

    # ---- noise / upres (Mantaflow plugin/waveletturbulence.cpp) ------------
    # High-res flags reuse init_domain / set_*_flags on interpolated levelsets,
    # and the per-frame emission max-join is `join`.

    @abstractmethod
    def interpolate_grid(self, target: ScalarGrid, source: ScalarGrid) -> None:
        """Trilinear resample of `source` onto `target`'s size, sampled at
        target cell centres. Mantaflow `interpolateGrid` (defaults)."""

    @abstractmethod
    def interpolate_mac(self, target: MACGrid, source: MACGrid) -> None:
        """MAC resample; values are not rescaled (they stay in source cells
        per time). Mantaflow `interpolateMACGrid` (defaults)."""

    @abstractmethod
    def reset_uv_grid(self, uv: VectorGrid, offset: Vec3 = (0.0, 0.0, 0.0)) -> None:
        """uv = cell index + offset. Mantaflow `resetUvGrid`."""

    @abstractmethod
    def update_uv_weight(self, uv: VectorGrid, time: float, dt: float, reset_time: float,
                         index: int, num_uvs: int, offset: Vec3 = (0.0, 0.0, 0.0)) -> float:
        """Reset `uv` when its ramp wraps, store its weight in uv[0, 0, 0]
        (as Mantaflow does) and return it (Mantaflow `getUvWeight`). time and
        dt are the low-res solver's. Mantaflow `updateUvWeight`."""

    @abstractmethod
    def compute_energy(self, flags: FlagGrid, vel: MACGrid, energy: ScalarGrid) -> None:
        """energy = 0.5 |centred vel|^2 on FLUID cells, else 0. Mantaflow `computeEnergy`."""

    @abstractmethod
    def extrapolate_simple_flags(self, flags: FlagGrid, val: FlagGrid | ScalarGrid,
                                 distance: int = 4, flag_from: int = Flags.FLUID,
                                 flag_to: int = Flags.OBSTACLE) -> None:
        """Copy `val` layer by layer (6-neighbour mean, integer for flag
        grids) from `flag_from` cells into `flag_to` cells, `distance` layers
        deep. Mantaflow `extrapolateSimpleFlags`."""

    @abstractmethod
    def compute_wavelet_coeffs(self, grid: ScalarGrid) -> None:
        """In place: smoothed sqrt|grid - wavelet lowpass(grid)| inside, the
        outer layer unchanged. Mantaflow `computeWaveletCoeffs`."""

    @abstractmethod
    def noise_tile(self, tile: np.ndarray) -> Any:
        """Upload the wavelet noise tile, (3, 128, 128, 128) float32 as
        `noise.get_tile()` returns it."""

    @abstractmethod
    def apply_noise_vec3(self, flags: FlagGrid, target: MACGrid, tile: Any,
                         weight: ScalarGrid, uv: VectorGrid, scale: float,
                         scale_spatial: float, gs_inv: float, seed_offset: Vec3,
                         time: float, pos_scale: Vec3) -> None:
        """target += curl noise * scale * weight on FLUID cells of the
        high-res `target`; weight and uv are low-res and interpolated (uv
        divided by the size factor). Noise field: gs_inv = 1 / max high-res
        size, time = solver time * dx * timeAnim, pos_scale per axis
        (`noise.evaluate_curl`). Mantaflow `applyNoiseVec3`."""

    # ---- liquid: particles and level set (v4) ---------------------------------

    @abstractmethod
    def particles(self, capacity: int, apic: bool = False, life: bool = False) -> Particles:
        """Empty particle arrays that hold up to `capacity` particles (0 in use);
        apic=True adds the three affine rows per particle, life=True a
        lifetime per particle (secondary particles)."""

    @abstractmethod
    def particle_count(self, parts: Particles) -> int:
        """Particles in use."""

    @abstractmethod
    def particle_capacity(self, parts: Particles) -> int:
        """Particles the arrays can hold."""

    @abstractmethod
    def set_particles(self, parts: Particles, pos: np.ndarray, vel: np.ndarray | None = None,
                      flag: np.ndarray | None = None, affine: np.ndarray | None = None,
                      life: np.ndarray | None = None) -> None:
        """Replace the contents: pos (n, 3) in cell units, vel (n, 3), flag (n,)
        of `ParticleFlags` (default 0), affine (n, 3, 3) rows cx, cy, cz (APIC
        particles only, default 0), life (n,) (life particles only, default 0).
        n must not exceed the capacity."""

    @abstractmethod
    def get_particles(self, parts: Particles) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pos, vel, flag) of all n particles, including inactive ones.
        Independent copies, as for to_numpy (also get_particle_affine and _life)."""

    @abstractmethod
    def get_particle_affine(self, parts: Particles) -> np.ndarray | None:
        """APIC affine rows (n, 3, 3), or None for particles made without apic."""

    @abstractmethod
    def get_particle_life(self, parts: Particles) -> np.ndarray | None:
        """Lifetimes (n,), or None for particles made without life."""

    @abstractmethod
    def grid_particle_index(self, parts: Particles, shape: Shape,
                            scale: float = 1.0) -> ParticleIndex:
        """Sort the active particles into the cells of a grid of `shape`
        (floor of pos * scale; particles outside are left out). Mantaflow
        `gridParticleIndex`. Inside a cell, particles are in ascending index
        order, so the result does not vary between runs. `scale`
        = mesh grid / particle grid size (Mantaflow `readParticles`)."""

    @abstractmethod
    def union_particle_levelset(self, parts: Particles, index: ParticleIndex,
                                phi: ScalarGrid, radius_factor: float = 1.0,
                                scale: float = 1.0) -> None:
        """phi = distance to the nearest particle sphere (radius 0.5*sqrt(3)*
        (radius_factor+0.01)), looking one cell ring further than the radius;
        outer layer 0.5. Overwrites phi. Mantaflow `unionParticleLevelset`.
        Positions are multiplied by `scale` (same as the index)."""

    @abstractmethod
    def improved_particle_levelset(self, parts: Particles, index: ParticleIndex,
                                   phi: ScalarGrid, radius_factor: float = 1.0,
                                   smoothen: int = 1, smoothen_neg: int = 1,
                                   t_low: float = 0.4, t_high: float = 3.5,
                                   scale: float = 1.0) -> None:
        """Mantaflow `improvedParticleLevelset` (Blender's "Final" mesh):
        weighted-average particle level set (Zhu & Bridson), concavity
        correction between eigenvalues t_low and t_high, then `smoothen`
        7-point mean passes and `smoothen_neg` passes that only lower phi;
        outer layer 0.5. Overwrites phi. Positions are multiplied by `scale`."""

    @abstractmethod
    def anisotropic_particle_levelset(self, parts: Particles, phi: ScalarGrid,
                                      sim_shape: tuple, radius_factor: float = 1.0,
                                      smoothen: int = 1, smoothen_neg: int = 1,
                                      scale: float = 1.0, h: float = 2.0, lam: float = 0.5,
                                      kr: float = 4.0, nmin: int = 10,
                                      min_half: float = 0.0) -> None:
        """Anisotropic kernels (Yu & Turk 2013), OpenBillow's High Quality mesh: each
        particle's centre is pulled `lam` of the way to its neighbours' weighted mean
        (radius `h` simulation cells, particle grid `sim_shape`), and its sphere (radius as
        union_particle_levelset, so in mesh cells) is stretched into a same-volume ellipsoid along the
        neighbour covariance (axis ratio <= kr; isotropic with fewer than `nmin`
        neighbours). `min_half` > 0 (mesh cells): a sheet-like ellipsoid keeps at least
        this half-thickness, faded in from 0 where the kr limit starts to bind to full for a
        flat sheet, so the smoothing cannot erase a one-particle film (CODE-LIQ-016); bulk
        particles are unaffected. phi = union of the ellipsoids, then the smoothing passes of
        improved_particle_levelset; outer layer 0.5. Overwrites phi."""

    @abstractmethod
    def mirror_particles(self, parts: Particles, phi_obs: ScalarGrid, max_dist: float = 2.0,
                         reach: float = 1.0) -> Particles:
        """Meshing only (CODE-LIQ-015): the live particles of `parts` plus one ghost per
        particle 0 < phi_obs < `max_dist` cells from an obstacle, mirrored across it
        (p - 2 phi_obs n, n the phi_obs gradient as push_out_of_obstacles). A ghost is kept
        only inside the grid's inner cells (the outer layer is the domain walls, closed by the
        VDB writer's wall layers) and when the point `reach` cells further into the obstacle
        is still inside it, so thin walls get no ghosts on their far side. Returns a reused
        position/flag set (velocities are not valid), deterministic order: the originals,
        then the ghosts in particle order. Valid until the next call."""

    @abstractmethod
    def clip_to_obstacle(self, phi: ScalarGrid, phi_obs: ScalarGrid, eps: float) -> None:
        """phi = max(phi, -(phi_obs + eps)) with phi_obs sampled at each phi cell centre
        (phi may be finer than phi_obs, sampled as interpolate_grid): the liquid ends `eps`
        cells (phi_obs units) inside an obstacle. Meshing only (CODE-LIQ-015)."""

    @abstractmethod
    def extrapolate_ls_simple(self, phi: ScalarGrid, distance: int = 4,
                              inside: bool = False) -> None:
        """Extend a level set `distance` layers outwards (inside=False: from
        phi >= 0 into phi < 0 is skipped, i.e. fills the outside) or inwards,
        stepping +-1 per layer; cells further away get +-(distance + 2).
        Mantaflow `extrapolateLsSimple`."""

    @abstractmethod
    def update_flags_from_levelset(self, flags: FlagGrid, phi: ScalarGrid) -> None:
        """FLUID where phi <= 0, else EMPTY, except OBSTACLE and OUTFLOW cells
        and cells with phi <= -1000 (invalid). Mantaflow `updateFromLevelset`."""

    @abstractmethod
    def levelset_volume(self, phi: ScalarGrid, flags: FlagGrid, border: int = 1) -> float:
        """Liquid volume in cells: sum of clip(0.5 - phi, 0, 1) over the interior, without
        the `border` outer layers (2 for fractions' thicker walls) and OBSTACLE cells (not in Mantaflow; used by the volume correction)."""

    @abstractmethod
    def source_outflux(self, flags: FlagGrid, phi_in: ScalarGrid, vel: MACGrid, invel=None) -> float:
        """Flux (cells per unit time) leaving the FLUID source cells (phi_in < 0)
        through faces to non-source, non-obstacle cells, outward part only,
        interior only (not in Mantaflow; the inflow rate for the volume
        correction). Source cells with a nonzero `invel` (cell-centred vector
        grid) use it instead of the MAC velocity."""

    # ---- liquid: particle/grid transfer (v4) ----------------------------------

    @abstractmethod
    def map_parts_to_mac(self, flags: FlagGrid, vel: MACGrid, vel_old: MACGrid,
                         parts: Particles, weight: VectorGrid) -> None:
        """Splat particle velocities onto the MAC faces (trilinear weights,
        weights < 1e-6 dropped), normalise; weight keeps the per-face weight
        sums, vel_old = vel. Mantaflow `mapPartsToMAC`."""

    @abstractmethod
    def apic_map_parts_to_mac(self, flags: FlagGrid, vel: MACGrid, parts: Particles) -> None:
        """APIC splat (velocity plus affine term). Particles need apic=True.
        Mantaflow `apicMapPartsToMAC`."""

    @abstractmethod
    def apic_map_mac_to_parts(self, parts: Particles, vel: MACGrid, flags: FlagGrid) -> None:
        """Particle velocity and affine rows from the grid. Mantaflow `apicMapMACGridToParts`."""

    @abstractmethod
    def flip_velocity_update(self, flags: FlagGrid, vel: MACGrid, vel_old: MACGrid,
                             parts: Particles, flip_ratio: float) -> None:
        """v_p = r (v_p + v(x) - v_old(x)) + (1 - r) v(x). Mantaflow `flipVelocityUpdate`."""

    @abstractmethod
    def extrapolate_mac_from_weight(self, vel: MACGrid, weight: VectorGrid,
                                    distance: int = 2) -> None:
        """Extend face values with weight > 0 by `distance` layers (inner cells);
        weight is overwritten with layer numbers. Mantaflow `extrapolateMACFromWeight`."""

    @abstractmethod
    def extrapolate_mac_simple(self, flags: FlagGrid, vel: MACGrid, distance: int = 4,
                               into_obs: bool = False) -> None:
        """Extend velocities from faces next to fluid by `distance` layers, then
        copy into the outer layer. Mantaflow `extrapolateMACSimple` (no phiObs)."""

    @abstractmethod
    def combine_grid_vel(self, vel: MACGrid, weight: VectorGrid, combine_vel: MACGrid,
                         phi: ScalarGrid | None = None, narrow_band: float = 0.0,
                         thresh: float = 0.0) -> None:
        """combine_vel = vel on faces with weight > thresh (and, with phi, not
        deeper than narrow_band); vel becomes -1 there, else 0. Mantaflow `combineGridVel`."""

    @abstractmethod
    def adjust_number(self, parts: Particles, vel: MACGrid, flags: FlagGrid, min_particles: int,
                      max_particles: int, phi: ScalarGrid, radius_factor: float = 1.0,
                      narrow_band: float = -1.0, exclude: ScalarGrid | None = None) -> None:
        """Delete particles outside the liquid, outside the band or in full
        cells (except near the surface), seed new ones (flag NEW, velocity from
        vel) in fluid cells with too few, and compact; may grow the capacity.
        Seed positions follow Mantaflow's random stream exactly. Mantaflow `adjustNumber`."""

    # ---- liquid: substep pieces (v4) ------------------------------------------

    @abstractmethod
    def sample_levelset_with_particles(self, phi: ScalarGrid, flags: FlagGrid, parts: Particles,
                                       discretization: int, randomness: float,
                                       vel_source: VectorGrid | None = None) -> None:
        """Add discretization^3 jittered particles (flag NEW) per non-obstacle
        cell with phi < 1.733, keeping those where the interpolated phi <= 0;
        clears NEW on the old ones first. Their velocity is 0, or `vel_source`
        (cell-centred) interpolated at their position (Blender's invel via
        pVel.setSource(isMAC=False)). Positions follow Mantaflow's random
        stream (restarted each call). May grow the capacity. Mantaflow
        `sampleLevelsetWithParticles` (no reset/refill)."""

    def release_scratch(self, *names: str) -> None:
        """Free scratch grids kept under these role names (initialisation-only grids). Default: nothing."""

    @abstractmethod
    def flood_fill(self, phi: ScalarGrid, boundary_width: int = 1, value: float = -0.5) -> None:
        """Mantaflow `LevelsetGrid::floodFill(value, outside=True)`: inside the
        `boundary_width` border, cells with phi >= 0 that no 6-connected path of
        phi >= 0 cells links to the border's inner layer become `value`
        (closes holes in a level set)."""

    @abstractmethod
    def subtract_levelset(self, phi: ScalarGrid, other: ScalarGrid) -> None:
        """phi = -other where other < 0. Mantaflow `LevelsetGrid::subtract` (no flags)."""

    @abstractmethod
    def update_flags_obs(self, flags: FlagGrid, phi_obs: ScalarGrid,
                         phi_in: ScalarGrid | None = None, phi_out: ScalarGrid | None = None,
                         boundary_width: int = 1, fractions: MACGrid | None = None) -> None:
        """Rewrite every cell inside the `boundary_width` border: OBSTACLE where
        phi_obs < 0 (with `fractions`: where all six face fractions are 0), else FLUID|INFLOW where phi_in < 0, else EMPTY|OUTFLOW
        where phi_out < 0, else EMPTY. Mantaflow `setObstacleFlags` (KnUpdateFlagsObs)
        as the liquid script calls it."""

    @abstractmethod
    def update_fractions(self, flags: FlagGrid, phi_obs: ScalarGrid, fractions: MACGrid,
                         boundary_width: int = 1, threshold: float = 0.01) -> None:
        """Mantaflow `updateFractions`: per MAC face the open fraction of its area
        from phi_obs at the two cells (calcFraction; below `threshold` = 0), outer
        layer 0; faces of cells next to INFLOW/OUTFLOW/OPEN sides within
        `boundary_width` + 1 of the border = 1. Run on the flags from init_domain."""

    @abstractmethod
    def reset_outflow_particles(self, flags: FlagGrid, phi: ScalarGrid, parts: Particles) -> None:
        """Mantaflow `resetOutflow(phi, parts)`: delete particles in OUTFLOW
        cells and compact (order kept); in OUTFLOW cells phi = 0.5 and flags =
        (f | EMPTY) & ~FLUID."""

    @abstractmethod
    def advect_particles(self, parts: Particles, flags: FlagGrid, vel: MACGrid, dt: float,
                         delete_in_obstacle: bool = False, skip_new: bool = True) -> None:
        """RK4 through the MAC grid (u = vel(x) * dt), skipping deleted and,
        with skip_new, NEW particles. Then either delete particles that end
        outside the domain (1-cell border) or in an obstacle, or clamp them
        into the grid. Mantaflow `advectInGrid(IntRK4, stopInObstacle=False)`."""

    @abstractmethod
    def push_out_of_obstacles(self, parts: Particles, phi_obs: ScalarGrid,
                              shift: float = 0.0, thresh: float = 0.0) -> None:
        """Move particles with interpolated phi_obs < thresh along the
        normalised central-difference gradient by thresh - phi + shift.
        Mantaflow `pushOutofObs`."""

    @abstractmethod
    def copy_particle_positions(self, parts: Particles) -> Any:
        """Snapshot of the current positions (and count) for `collide_particles`."""

    @abstractmethod
    def thin_obstacles(self, meshes: list, phi_obs: ScalarGrid, obvel: Any = None,
                       num_obs: Any = None) -> None:
        """Improvement (not Mantaflow): give sub-cell parts of the obstacle
        meshes (already rasterised into `phi_obs`) obstacle cells. A cell
        whose centre is outside, with no obstacle cell among its 6
        neighbours, and within 0.5 cells of a mesh gets phi_obs =
        min(phi_obs, distance - 0.5). With `obvel`/`num_obs` (cell-centred
        sums, as `rasterize_mesh_obstacle`) the mesh velocity there is added."""

    @abstractmethod
    def collide_particles(self, parts: Particles, pos_old: Any, meshes: list, shape: Shape,
                          margin: float = 0.1, closed: str = "xXyYzZ", wall: float = 1.0) -> None:
        """Improvement (not Mantaflow): exact collisions with obstacle
        meshes (cell space, as set by `set_mesh_transform`). A particle whose
        path from `pos_old` (None: no path test) enters a mesh stops at the entry face; one inside
        or within `margin` cells of a mesh moves to the closest surface point
        plus `margin` along the outward normal. Either way the velocity part
        into the obstacle, relative to the mesh velocity there, is removed.
        A mesh with `two_sided` set (open, not watertight, or "Is Planar")
        has no inside: its path test stops a particle crossing either face
        and the closest-point test only enforces the margin on the side the
        particle is on. A closed mesh that is inside out (`orient` -1, from
        flipped normals or a mirroring transform) is handled the same way,
        as the grid's 26-ray rule also leaves its inside free.
        Then the domain sides in `closed` (\"xXyYzZ\", lower case = low side)
        clamp particles to [wall + margin, n - wall - margin] and zero their
        velocity into that wall (`wall` = 2 with fractions' thicker walls)."""

    @abstractmethod
    def add_gravity(self, flags: FlagGrid, vel: MACGrid, force: Vec3) -> None:
        """vel += force on faces between fluid cells or fluid and empty cells
        (outer layer skipped). The solver passes gravity * dt in cell units.
        Mantaflow `addGravity(scale=False)` (KnApplyForce)."""

    # ---- liquid: secondary particles (v5) -------------------------------------
    # Mantaflow plugin/secondaryparticles.cpp as Blender's liquid_step_particles
    # calls it. Secondary particles are `particles(life=True)`; the flag holds
    # the type (ParticleFlags SPRAY / BUBBLE / FOAM).

    @abstractmethod
    def secondary_potentials(self, flags: FlagGrid, vel: MACGrid, phi: ScalarGrid,
                             pot_ta: ScalarGrid, pot_wc: ScalarGrid, pot_ke: ScalarGrid,
                             neighbor_ratio: ScalarGrid, radius: int,
                             tau_ta: tuple[float, float], tau_wc: tuple[float, float],
                             tau_ke: tuple[float, float], scale_from_manta: float) -> None:
        """Trapped-air, wave-crest and kinetic-energy potentials (each clamped
        to its (tau_min, tau_max) and normalised to [0, 1]) and the fluid
        neighbour ratio of every FLUID cell at least `radius` cells inside;
        everything else 0. Normals are the central-difference gradient of phi.
        Mantaflow `flipComputeSecondaryParticlePotentials`."""

    @abstractmethod
    def sample_secondary_particles(self, parts: Particles, flags: FlagGrid, vel: MACGrid,
                                   pot_ta: ScalarGrid, pot_wc: ScalarGrid, pot_ke: ScalarGrid,
                                   neighbor_ratio: ScalarGrid, k_ta: float, k_wc: float,
                                   life_min: float, life_max: float, dt: float,
                                   c_s: float = 0.4, c_b: float = 0.77,
                                   sampling: str = "SINGLE") -> None:
        """`sampling` MULTIPLE (Mantaflow's 'multiple' sampler): 8 fixed points per cell at
        +-0.25, potentials interpolated at each, the same count formula per point, cylinder
        radius 0.25 (SINGLE: one random point, radius 0.5). Append int(KE * (k_ta*TA + k_wc*WC) * dt) particles per FLUID cell
        in a cylinder along the local flow, typed by the cell's neighbour ratio
        (< c_s spray, > c_b bubble, else foam), in Mantaflow's serial order and
        random stream. Mantaflow `flipSampleSecondaryParticles(mode='single')`."""

    @abstractmethod
    def update_secondary_particles(self, parts: Particles, flags: FlagGrid, vel: MACGrid,
                                   neighbor_ratio: ScalarGrid, gravity: Vec3, k_b: float,
                                   k_d: float, dt: float, c_s: float = 0.4,
                                   c_b: float = 0.77, antitunneling: str = "OFF",
                                   antitunnel_steps: int = 4) -> None:
        """Retype each particle by the neighbour ratio of its cell, then move it:
        spray ballistic (gravity), bubbles buoyant with drag towards the grid
        velocity, foam with the grid velocity. Life -= dt; particles outside
        the grid or with life <= 0 are deleted, then compacted (order kept).
        `gravity` in cell units. Mantaflow `flipUpdateSecondaryParticles(mode=
        'linear', scale=False, antitunneling=0)`; the force pdata is always 0.
        `antitunneling`: OFF (Blender), FIXED (Mantaflow's N = `antitunnel_steps`) or AUTO (per
        particle N = ceil(|step| / 0.5 cell) + 1, at most 16). The points ct/N of the step, ct =
        1..N-1, are tested; a particle out of bounds or in an obstacle there is deleted."""

    @abstractmethod
    def delete_particles_in_obstacle(self, parts: Particles, flags: FlagGrid) -> None:
        """Delete particles outside the grid or in OBSTACLE / OUTFLOW cells,
        then compact. Mantaflow `flipDeleteParticlesInObstacle`."""
