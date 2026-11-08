# SPDX-License-Identifier: GPL-3.0-or-later
"""Fluid guiding (Blender "Guides"), for the gas and liquid
solvers: the guide velocity built once per frame from guide effectors
(Blender's "Bake Guides" pass: `manta_guiding` + `bake_guiding_process`), the
per-substep target velocity (`fluid_pre_step`) and `PD_fluid_guiding` in place
of the pressure solve (liquid: with phi, so ghost-fluid weights). Only the backend interface is called.
"""
from __future__ import annotations

import os

import numpy as np

from .backend import Backend
from .backend.base import GuidingStats

TAU, SIGMA, THETA = 1.0, 0.99, 1.0   # fluid_variables_guiding
PHI_MAX = 9999.0


class Guiding:
    """Effector source: the guide grid has the domain's resolution."""

    def __init__(self, backend: Backend, shape, guides: list, world_to_cell: np.ndarray,
                 frame_length: float, alpha: float, beta: int, vel_factor: float,
                 two_d: bool = False):
        b = self.b = backend
        self.world_to_cell = world_to_cell
        self.frame_length = frame_length
        self.beta = int(beta)
        self.vel_factor = float(vel_factor)
        self.two_d = two_d
        # weight = alpha everywhere. Blender's alpha is 1..100; below ~0.5 the
        # PD loop diverges, so clamp for CLI users.
        self.weight = b.scalar_grid(shape, max(1.0, float(alpha)))
        self.vel_target = b.mac_grid(shape)
        self.guidevel = b.mac_grid(shape)
        self.phi = b.scalar_grid(shape, PHI_MAX)
        self.vel_c = b.vector_grid(shape)
        self.num = b.scalar_grid(shape)
        self._meshes = [(g, b.create_mesh(g.verts, g.tris)) for g in guides]

    def prime(self, frame: int) -> None:
        """After a resume: previous guide positions at the last finished
        frame, so the next frame's guide velocity matches an unbroken bake."""
        for g, m in self._meshes:
            self.b.set_mesh_transform(m, self.world_to_cell @ g.matrix_at(float(frame)))

    def update_frame(self, frame: int) -> None:
        """Guide grid for the step into scene frame `frame` (Blender reads
        guiding/frame f before stepping to f). Vertex velocity = cell
        displacement since the previous frame / frame length (0 on the
        first frame). Per object, cells inside get the mode-combined velocity
        and a count; then safeDivide, extrapolate 6 in / 3 out, resample to
        MAC (= obstacle_velocity)."""
        b = self.b
        b.fill(self.phi, PHI_MAX)
        b.fill(self.vel_c, (0.0, 0.0, 0.0))
        b.fill(self.num, 0.0)
        for g, m in self._meshes:
            b.set_mesh_transform(m, self.world_to_cell @ g.matrix_at(float(frame)), self.frame_length)
            b.rasterize_mesh_obstacle(m, self.phi, g.surface_thickness, g.plane_init,
                                      self.vel_c, self.num, g.guide_mode, g.vel_multi)
        b.obstacle_velocity(self.phi, self.vel_c, self.num, self.guidevel)

    def solve(self, flags, vel, pressure, **pressure_kw) -> GuidingStats:
        """fluid_pre_step's velT = interpolateMACGrid(guidevel) * vel_factor,
        then PD_fluid_guiding (pressure_kw: accuracy, zero_pressure_fixing, ...)."""
        b = self.b
        b.interpolate_mac(self.vel_target, self.guidevel)
        b.mult_const(self.vel_target, self.vel_factor)
        if self.two_d:
            b.mirror_slab(self.vel_target)  # no world-Y component in 2D
        return b.guiding_solve(flags, vel, self.vel_target, self.weight, pressure,
                               blur_radius=self.beta, tau=TAU, sigma=SIGMA, theta=THETA,
                               two_d=self.two_d, **pressure_kw)


def guide_frame_path(folder: str, frame: int) -> str:
    """A parent bake's full MAC velocity at the end of `frame` (written by the CLI)."""
    return os.path.join(folder, "guide", f"{frame:04d}.npy")


class DomainGuiding(Guiding):
    """Domain source (Blender `fluid_load_vel`): guidevel = the parent bake's
    velocity at the parent's resolution, read per frame, no extrapolation.
    Not rescaled to this grid's cells (as Blender: a coarser parent guides
    proportionally slower)."""

    def __init__(self, backend: Backend, shape, parent_dir: str, frame_start: int,
                 alpha: float, beta: int, vel_factor: float, two_d: bool = False):
        b = self.b = backend
        self.beta = int(beta)
        self.vel_factor = float(vel_factor)
        self.two_d = two_d
        self.parent_dir = parent_dir
        a = np.load(guide_frame_path(parent_dir, frame_start), mmap_mode="r")
        if a.ndim != 4 or a.shape[3] != 3:
            raise ValueError(f"guide frame has shape {a.shape}, expected (nx, ny, nz, 3)")
        self.guide_shape = tuple(a.shape[:3])
        if two_d and self.guide_shape[1] != 3:
            raise ValueError("a 2D domain needs a 2D guide parent")
        self.weight = b.scalar_grid(shape, max(1.0, float(alpha)))
        self.vel_target = b.mac_grid(shape)
        self.guidevel = b.mac_grid(self.guide_shape)

    def prime(self, frame: int) -> None:
        pass

    def update_frame(self, frame: int) -> None:
        a = np.load(guide_frame_path(self.parent_dir, frame))
        if tuple(a.shape[:3]) != self.guide_shape:
            raise ValueError(f"guide frame {frame} has shape {a.shape}, not {self.guide_shape}")
        self.b.from_numpy(self.guidevel, np.asarray(a, np.float32))


def make_guiding(backend: Backend, shape, s, world_to_cell, frame_length: float,
                 frame_start: int, two_d: bool = False) -> Guiding:
    """From DomainSettings / LiquidSettings (guide_parent already resolved)."""
    if s.guide_source == "DOMAIN":
        if not s.guide_parent:
            raise ValueError("guide source DOMAIN needs a guide parent bake")
        return DomainGuiding(backend, shape, s.guide_parent, frame_start,
                             s.guide_alpha, s.guide_beta, s.guide_vel_factor, two_d)
    if s.guide_source != "EFFECTOR":
        raise ValueError(f"unknown guide source {s.guide_source!r}")
    return Guiding(backend, shape, s.guides, world_to_cell, frame_length,
                   s.guide_alpha, s.guide_beta, s.guide_vel_factor, two_d)
