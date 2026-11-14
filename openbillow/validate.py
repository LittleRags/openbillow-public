# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks on bake settings, so bad values fail early with a ValueError that
names the field (exit 4) instead of a traceback or an empty bake."""
from __future__ import annotations

import numpy as np


def _fail(name, why):
    raise ValueError(f"{name}: {why}")


def _num(name, v, lo=None, lo_open=False, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        _fail(name, f"must be a number, got {v!r}")
    if not np.isfinite(v):
        _fail(name, f"must be finite, got {v!r}")
    if integer and int(v) != v:
        _fail(name, f"must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_open else v < lo):
        _fail(name, f"must be {'>' if lo_open else '>='} {lo}, got {v!r}")


def _enum(name, v, allowed):
    if v not in allowed:
        _fail(name, f"must be one of {', '.join(map(str, allowed))}, got {v!r}")


def _matrix(name, m, frames):
    if m is None:
        return
    m = np.asarray(m)
    if not (m.shape == (4, 4) or (m.ndim == 3 and m.shape[1:] == (4, 4) and len(m) >= 1)):
        _fail(f"{name}.matrix", f"shape must be (4, 4) or (k, 4, 4), got {m.shape}")
    if not np.isfinite(m).all():
        _fail(f"{name}.matrix", "has non-finite values")
    if m.ndim == 3 and len(m) > 1:
        if frames is None:
            _fail(f"{name}.frames", "missing for a sampled matrix")
        t = np.asarray(frames, float)
        if t.shape != (len(m),):
            _fail(f"{name}.frames", f"length {t.size} != {len(m)} matrix samples")
        if not (np.isfinite(t).all() and (np.diff(t) > 0).all()):
            _fail(f"{name}.frames", "must be finite and strictly increasing")


def _mesh(name, o):
    v, t = np.asarray(o.verts), np.asarray(o.tris)
    if v.ndim != 2 or v.shape[1] != 3:
        _fail(f"{name}.verts", f"shape must be (n, 3), got {v.shape}")
    if not np.isfinite(v).all():
        _fail(f"{name}.verts", "has non-finite values")
    if t.ndim != 2 or t.shape[1] != 3 or (t.size and not np.issubdtype(t.dtype, np.integer)):
        _fail(f"{name}.tris", f"must be integer (m, 3), got {t.dtype} {t.shape}")
    if t.size and (t.min() < 0 or t.max() >= len(v)):
        _fail(f"{name}.tris", f"index out of range 0..{len(v) - 1}")
    _matrix(name, o.matrix, o.frames)


def common(s):
    """Fields shared by gas and liquid domains (DomainSettings, LiquidSettings)."""
    _num("resolution_max", s.resolution_max, 3, integer=True)
    if len(s.size) != 3:
        _fail("size", f"needs 3 components, got {s.size!r}")
    for i, c in enumerate(s.size):
        _num(f"size[{i}]", c, 0, lo_open=True)
    _num("fps", s.fps, 0, lo_open=True)
    _num("time_scale", s.time_scale, 0, lo_open=True)
    _num("cfl", s.cfl, 0, lo_open=True)
    _num("timesteps_min", s.timesteps_min, 1, integer=True)
    _num("timesteps_max", s.timesteps_max, s.timesteps_min, integer=True)
    if not isinstance(s.outflow, str) or set(s.outflow) - set("xXyYzZ"):
        _fail("outflow", f"may contain only xXyYzZ, got {s.outflow!r}")
    _enum("quality", s.quality, ("CLASSIC", "HIGH"))
    _enum("guide_source", s.guide_source, ("EFFECTOR", "DOMAIN"))
    for i, o in enumerate(s.flows):
        _mesh(f"flows[{i}]", o)
    for i, o in enumerate(s.obstacles):
        _mesh(f"obstacles[{i}]", o)
    for i, o in enumerate(s.guides):
        _enum(f"guides[{i}].guide_mode", o.guide_mode, ("OVERRIDE", "AVERAGED", "MAX", "MIN"))
        _mesh(f"guides[{i}]", o)
    for i, f in enumerate(s.force_fields):
        _matrix(f"force_fields[{i}]", f.matrix, f.frames)


def gas(s):
    common(s)
    _enum("dimensions", s.dimensions, (2, 3))
    _enum("fire_model", s.fire_model, ("MANTAFLOW", "COMBUSTION", "FLAME_FRONT", "DIFFUSION"))
    _num("noise_upres", s.noise_upres, 1, integer=True)
    _num("noise_pos_scale", s.noise_pos_scale, 0, lo_open=True)
    if s.fire_model in ("COMBUSTION", "FLAME_FRONT", "DIFFUSION"):
        _num("flame_ignition", s.flame_ignition, 0, lo_open=True)
        if not s.flame_ignition < s.flame_max_temp:
            _fail("flame_max_temp", f"must be above flame_ignition ({s.flame_ignition}), got {s.flame_max_temp}")
    if s.fire_model == "FLAME_FRONT":
        _num("flame_speed", s.flame_speed, 0, lo_open=True)
        _num("ambient_kelvin", s.ambient_kelvin, 0, lo_open=True)
        _num("fuel_kelvin", s.fuel_kelvin, 0, lo_open=True)
        _num("flame_kelvin", s.flame_kelvin, 0, lo_open=True)
        if not s.ambient_kelvin <= s.fuel_kelvin < s.flame_kelvin:
            _fail("fuel_kelvin", f"must be from ambient_kelvin ({s.ambient_kelvin}) up to below "
                                 f"flame_kelvin ({s.flame_kelvin}), got {s.fuel_kelvin}")
        _num("flame_inject", s.flame_inject, 0)
        _num("flame_glow_cells", s.flame_glow_cells, 0, lo_open=True)
        _num("flame_source_gap", s.flame_source_gap, 0, integer=True)
        _num("flame_smoke_yield", s.flame_smoke_yield, 0)
        if not isinstance(s.use_viscosity, (bool, np.bool_)):
            _fail("use_viscosity", f"must be a bool, got {s.use_viscosity!r}")
    if s.fire_model == "DIFFUSION":
        _num("ambient_kelvin", s.ambient_kelvin, 0, lo_open=True)
        _num("flame_kelvin", s.flame_kelvin, 0, lo_open=True)
        if not s.flame_kelvin > s.ambient_kelvin:
            _fail("flame_kelvin", f"must be above ambient_kelvin ({s.ambient_kelvin}), got {s.flame_kelvin}")
        _num("diffusion_hrrpua", s.diffusion_hrrpua, 0)
        _num("diffusion_heat_of_combustion", s.diffusion_heat_of_combustion, 0, lo_open=True)
        _num("diffusion_air_fuel_ratio", s.diffusion_air_fuel_ratio, 0, lo_open=True)
        _num("diffusion_radiative_fraction", s.diffusion_radiative_fraction, 0)
        if not s.diffusion_radiative_fraction < 1.0:
            _fail("diffusion_radiative_fraction", f"must be below 1, got {s.diffusion_radiative_fraction!r}")
        _num("diffusion_soot_yield", s.diffusion_soot_yield, 0)
    if s.fire_model in ("FLAME_FRONT", "DIFFUSION") and any(f.flow_type != "SMOKE" for f in s.flows):
        if s.dimensions == 2:  # the model is in use
            _fail("fire_model", f"{s.fire_model} is 3D only (a 2D slice misleads: one-sided base, flapping)")
        if s.use_noise:
            _fail("use_noise", f"noise (upres) does not support fire_model {s.fire_model} yet")
        if s.use_guide:
            _fail("use_guide", f"guiding does not support fire_model {s.fire_model} yet")
        if not isinstance(s.use_viscosity, (bool, np.bool_)):
            _fail("use_viscosity", f"must be a bool, got {s.use_viscosity!r}")
    for i, f in enumerate(s.flows):
        _enum(f"flows[{i}].flow_type", f.flow_type, ("SMOKE", "FIRE", "BOTH"))


def liquid(s):
    common(s)
    _num("particle_number", s.particle_number, 1, integer=True)
    _num("mesh_scale", s.mesh_scale, 1, integer=True)
    _num("particle_scale", s.particle_scale, 1, integer=True)
    _enum("mesh_generator", s.mesh_generator, ("IMPROVED", "UNION", "ANISOTROPIC"))
    _enum("sndparticle_boundary", s.sndparticle_boundary, ("DELETE", "PUSHOUT"))
    _enum("sndparticle_antitunneling", s.sndparticle_antitunneling, ("OFF", "AUTO", "FIXED"))
    _enum("sndparticle_sampling", s.sndparticle_sampling, ("SINGLE", "MULTIPLE"))
    if s.use_spray_particles or s.use_foam_particles or s.use_bubble_particles:
        # Equal bounds divide 0 by 0 in the potential clamp and silently produce no particles (or an overflow on the CPU).
        for label, lo, hi in (("trapped air", s.sndparticle_tau_min_ta, s.sndparticle_tau_max_ta),
                              ("wave crest", s.sndparticle_tau_min_wc, s.sndparticle_tau_max_wc),
                              ("kinetic energy", s.sndparticle_tau_min_k, s.sndparticle_tau_max_k)):
            _num(f"sndparticle {label} potential Min", lo)
            _num(f"sndparticle {label} potential Max", hi)
            if not hi > lo:
                _fail(f"sndparticle {label} potential", f"Max ({hi}) must be greater than Min ({lo})")
    for i, f in enumerate(s.flows):
        _enum(f"flows[{i}].behavior", f.behavior, ("GEOMETRY", "INFLOW", "OUTFLOW"))
