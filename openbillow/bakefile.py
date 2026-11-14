# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake folder format: `bake.json` (settings) + `objects.npz` (mesh arrays)
<-> `DomainSettings`.

bake.json: {"schema", "openbillow_version", "domain_type" (GAS | LIQUID),
            "frame_start", "frame_end", "domain": {...}, "flows": [{"name", "key", ...}],
            "obstacles": [...], "guides": [...]}
Gas: DomainSettings / MeshFlow; liquid: LiquidSettings / LiquidFlow.
objects.npz, per object key K: K_verts (n,3 f32), K_tris (m,3 i32),
K_matrix (4,4) or (k,4,4), and K_times (k,) when animated.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np

from . import __version__
from .forcefields import EffectorWeights, ForceField
from .liquid import LiquidFlow, LiquidSettings
from .solver import DomainSettings, MeshFlow, MeshGuide, MeshObstacle

SCHEMA = 21  # 21: fire_model DIFFUSION (diffusion_hrrpua, diffusion_heat_of_combustion, diffusion_air_fuel_ratio,
# diffusion_radiative_fraction, diffusion_soot_yield; older bakes read the defaults, unused by their fire models);
# 20: FLAME_FRONT use_viscosity (older bakes read as off, their look unchanged);
# 19: fire_model FLAME_FRONT (flame_speed, fuel_kelvin, flame_inject, flame_glow_cells,
# flame_source_gap, flame_smoke_yield; older bakes read the defaults, unused by their fire models);
# 18: mesh_wall_contact (older bakes read as off, their look unchanged);
# 17: High Quality mesh honours mesh_particle_radius (older 2.0 reads as 1.1447);
# 2: fire settings (domain fire fields, flow_type, fuel_amount); 3: flow emit_start/emit_end;
# 4: noise settings (use_noise, noise_upres, noise_strength, noise_pos_scale, noise_time_anim);
# 5: domain_type and liquid domains (LiquidSettings, LiquidFlow fields);
# 6: force_fields, effector_weights, render_fps; 7: domain_mat3; 8: Turbulence fields (use_global_coords);
# 9: gas domain dimensions (2D); 10: liquid use_fractions, fractions_threshold, fractions_distance;
# 11: guiding (use_guide, guide_alpha, guide_beta, guide_vel_factor, guide_source, guide_parent, write_guide, guides);
# 12: COMBUSTION fire model (fire_model, heat_release, cooling, cooling_radiative, expansion, flame_kelvin,
# ambient_kelvin, disturbance, disturbance_size, disturbance_rate, shredding); 13: LiquidFlow emit_start/emit_end; 14: quality, surface_tension, mesh_generator ANISOTROPIC;
# 15: sndparticle_antitunneling, sndparticle_antitunnel_steps; 16: sndparticle_sampling
_OBS_FIELDS = ("surface_thickness", "plane_init")
_GUIDE_FIELDS = _OBS_FIELDS + ("guide_mode", "vel_multi")
_SKIP = ("flows", "obstacles", "guides", "force_fields", "effector_weights")
_FIELD_TUPLES = ("location", "z_axis")
_FIELD_FIELDS = tuple(f.name for f in dataclasses.fields(ForceField) if f.name not in ("matrix", "frames"))
_TUPLE_FIELDS = ("min_corner", "size", "gravity", "velocity_coord", "domain_mat3")


def _fields(cls, skip=()):
    return [f.name for f in dataclasses.fields(cls) if f.name not in skip]


# domain_type -> (settings class, domain fields, flow class, flow fields)
_TYPES = {
    "GAS": (DomainSettings, _fields(DomainSettings, _SKIP), MeshFlow,
            ("flow_type", "density", "temperature", "surface_distance", "volume_density",
             "plane_init", "fuel_amount", "emit_start", "emit_end")),
    "LIQUID": (LiquidSettings, _fields(LiquidSettings, _SKIP), LiquidFlow,
               _fields(LiquidFlow, ("verts", "tris", "matrix", "frames"))),
}


def _value(a, v):
    return list(v) if a in _TUPLE_FIELDS else v


# Left out of the hash: changing the end frame keeps a bake resumable. The start frame is hashed
# (CODE-SOL-012); checkpoints from 0.5.3 and earlier left it out, so the legacy hash keeps that list.
_HASH_SKIP = ("frame_end", "openbillow_version")
_LEGACY_HASH_SKIP = ("frame_start",) + _HASH_SKIP
STATIC_TOL = 1e-7  # transform samples closer than this count as an object that does not move


def _pack(arrays: dict, key: str, o) -> None:
    arrays[f"{key}_verts"] = np.asarray(o.verts, np.float32)
    arrays[f"{key}_tris"] = np.asarray(o.tris, np.int32)
    arrays[f"{key}_matrix"] = np.asarray(o.matrix, np.float64)
    if o.animated:
        arrays[f"{key}_times"] = np.asarray(o.frames, np.float64)


def write_bake(folder: str, settings: DomainSettings | LiquidSettings, frame_start: int,
               frame_end: int, flow_names: list[str] | None = None,
               obstacle_names: list[str] | None = None,
               guide_names: list[str] | None = None) -> None:
    os.makedirs(folder, exist_ok=True)
    dtype = "LIQUID" if isinstance(settings, LiquidSettings) else "GAS"
    _, dom_fields, _, flow_fields = _TYPES[dtype]
    flow_names = flow_names or [f"Flow{i}" for i in range(len(settings.flows))]
    obstacle_names = obstacle_names or [f"Obstacle{i}" for i in range(len(settings.obstacles))]
    arrays: dict[str, np.ndarray] = {}
    flows, obstacles = [], []
    for i, (n, f) in enumerate(zip(flow_names, settings.flows)):
        k = f"flow{i}"
        _pack(arrays, k, f)
        flows.append({"name": n, "key": k, **{a: _value(a, getattr(f, a)) for a in flow_fields}})
    for i, (n, o) in enumerate(zip(obstacle_names, settings.obstacles)):
        k = f"obstacle{i}"
        _pack(arrays, k, o)
        obstacles.append({"name": n, "key": k, **{a: getattr(o, a) for a in _OBS_FIELDS}})
    guide_list = getattr(settings, "guides", [])
    guide_names = guide_names or [f"Guide{i}" for i in range(len(guide_list))]
    guides = []
    for i, (n, o) in enumerate(zip(guide_names, guide_list)):
        k = f"guide{i}"
        _pack(arrays, k, o)
        guides.append({"name": n, "key": k, **{a: getattr(o, a) for a in _GUIDE_FIELDS}})
    fields = []
    for i, f in enumerate(settings.force_fields):
        e = {"key": f"field{i}", **{a: list(v) if a in _FIELD_TUPLES else v
                                       for a, v in ((a, getattr(f, a)) for a in _FIELD_FIELDS)}}
        if f.matrix is not None:
            arrays[f"field{i}_matrix"] = np.asarray(f.matrix, np.float64)
            if f.frames is not None:
                arrays[f"field{i}_times"] = np.asarray(f.frames, np.float64)
        fields.append(e)
    w = settings.effector_weights
    doc = {"schema": SCHEMA, "openbillow_version": __version__, "domain_type": dtype,
           "frame_start": int(frame_start), "frame_end": int(frame_end),
           "domain": {a: _value(a, getattr(settings, a)) for a in dom_fields},
           "flows": flows, "obstacles": obstacles, "force_fields": fields,
           **({"guides": guides} if guides else {}),
           "effector_weights": {"all": w.all, "types": dict(w.types)}}
    np.savez(os.path.join(folder, "objects.npz"), **arrays)
    with open(os.path.join(folder, "bake.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def _unpack(npz, entry: dict, cls, fields):
    k = entry["key"]
    m = npz[f"{k}_matrix"]
    times = npz[f"{k}_times"] if f"{k}_times" in npz.files else None
    kw = {a: (tuple(entry[a]) if a in _TUPLE_FIELDS else entry[a])
          for a in fields if a in entry}  # missing keys: dataclass defaults
    return cls(npz[f"{k}_verts"], npz[f"{k}_tris"], m, times, **kw)


def read_bake(folder: str) -> tuple[DomainSettings | LiquidSettings, int, int]:
    """-> (settings, frame_start, frame_end). Unknown keys are ignored; no
    domain_type (schema < 5) means GAS."""
    with open(os.path.join(folder, "bake.json")) as fh:
        doc = json.load(fh)
    if doc.get("schema", 0) > SCHEMA:
        raise ValueError(f"bake.json schema {doc['schema']} is newer than supported ({SCHEMA})")
    dtype = doc.get("domain_type", "GAS")
    if dtype not in _TYPES:
        raise ValueError(f"unknown domain_type {dtype!r}")
    cls, dom_fields, flow_cls, flow_fields = _TYPES[dtype]
    dom = {a: (tuple(v) if a in _TUPLE_FIELDS else v)
           for a, v in doc.get("domain", {}).items() if a in dom_fields}
    if doc.get("schema", 0) < 20:
        dom["use_viscosity"] = False  # before 20 FLAME_FRONT had no viscosity
    if (doc.get("schema", 0) < 17 and dom.get("mesh_generator") == "ANISOTROPIC"
            and abs(dom.get("mesh_particle_radius", 2.0) - 2.0) < 1e-4):
        dom["mesh_particle_radius"] = 1.1447  # before 17 High Quality ignored it and looked like this
    with np.load(os.path.join(folder, "objects.npz")) as npz:
        flows = [_unpack(npz, e, flow_cls, flow_fields) for e in doc.get("flows", [])]
        obstacles = [_unpack(npz, e, MeshObstacle, _OBS_FIELDS) for e in doc.get("obstacles", [])]
        guides = [_unpack(npz, e, MeshGuide, _GUIDE_FIELDS) for e in doc.get("guides", [])]
        fields = []
        for e in doc.get("force_fields", []):
            k = e["key"]
            kw = {a: (tuple(e[a]) if a in _FIELD_TUPLES else e[a]) for a in _FIELD_FIELDS if a in e}
            fields.append(ForceField(**kw, matrix=npz[f"{k}_matrix"] if f"{k}_matrix" in npz.files else None,
                                     frames=npz[f"{k}_times"] if f"{k}_times" in npz.files else None))
    w = doc.get("effector_weights", {})
    weights = EffectorWeights(float(w.get("all", 1.0)), dict(w.get("types", {})))
    settings = cls(**dom, flows=flows, obstacles=obstacles, guides=guides, force_fields=fields,
                   effector_weights=weights)
    settings.validate()
    return settings, int(doc["frame_start"]), int(doc["frame_end"])


def _doc_hash(folder: str, skip=_HASH_SKIP):
    """sha256 object over bake.json without the keys in `skip`."""
    with open(os.path.join(folder, "bake.json")) as fh:
        doc = json.load(fh)
    for k in skip:
        doc.pop(k, None)
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())


def _update(h, key: str, a: np.ndarray) -> None:
    h.update(f"{key}|{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())


class SettingsHasher:
    """Settings hash of a bake folder, optionally limited to what the frames up to
    `upto_frame` could read. Frame F only reads transform samples with time <= F, so
    a later end frame (which appends samples) leaves that digest unchanged. Loads
    objects.npz once: a bake takes one digest per checkpoint."""

    def __init__(self, folder: str):
        self._base = _doc_hash(folder)
        self._matrices = []  # (key, matrix, times or None), times only when sampled
        with np.load(os.path.join(folder, "objects.npz")) as z:
            for k in sorted(z.files):
                if k.endswith("_times"):
                    continue
                a = z[k]
                if not k.endswith("_matrix"):
                    _update(self._base, k, a)
                    continue
                tk = k[:-len("matrix")] + "times"
                self._matrices.append((k, a, z[tk] if a.ndim == 3 and tk in z.files else None))

    def digest(self, upto_frame: int | None = None) -> str:
        h = self._base.copy()
        for k, a, t in self._matrices:
            if t is not None:
                if upto_frame is not None:
                    n = int(np.searchsorted(t, upto_frame, side="right"))
                    a, t = a[:n], t[:n]
                if len(a) and np.ptp(a, axis=0).max() < STATIC_TOL:
                    a = a[0]  # the form the exporter writes for an object that does not move
                else:
                    h.update(t.tobytes())
            _update(h, k, a)
        return h.hexdigest()


def settings_hash(folder: str, upto_frame: int | None = None) -> str:
    """sha256 of bake.json without frame_start/frame_end/openbillow_version, plus the
    objects.npz arrays. See SettingsHasher for `upto_frame`."""
    return SettingsHasher(folder).digest(upto_frame)


def _legacy_hash(folder: str) -> str:
    """The hash stored in checkpoints written by 0.5.3 and earlier (raw objects.npz bytes)."""
    h = _doc_hash(folder, _LEGACY_HASH_SKIP)
    with open(os.path.join(folder, "objects.npz"), "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()
