# SPDX-License-Identifier: GPL-3.0-or-later
"""Force fields on the GPU: Blender's `update_effectors_task_cb` per cell.

The kernel follows `openbillow/forcefields.py` (the NumPy reference it is tested
against) in float32. Relevance and the per-type weight are resolved on the host;
each field is one `_Field` struct."""
from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

from .base import Flags
from .. import forcefields as ff

OBSTACLE = wp.constant(int(Flags.OBSTACLE))
FLT_EPS = wp.constant(1.1920928955078125e-07)

T_FORCE, T_WIND, T_VORTEX, T_MAGNET, T_HARMONIC, T_CHARGE, T_DRAG, T_TURBULENCE = range(8)
_TYPE = {t: i for i, t in enumerate(ff.TYPES)}
_SHAPE = {"POINT": 0, "PLANE": 1, "LINE": 2}
_FALLOFF = {"SPHERE": 0, "TUBE": 1, "CONE": 2}
_ZDIR = {"BOTH": 0, "POSITIVE": 1, "NEGATIVE": 2}


@wp.struct
class _Field:
    type: int
    shape: int
    falloff: int
    zdir: int
    gravity: int
    glob: int          # Turbulence: global coordinates
    loc: wp.vec3
    nor: wp.vec3
    weight: float      # weights.all * per-type weight
    strength: float
    flow: float
    damp: float
    size: float
    # distance falloff: use_min, min, use_max, max, power; same for radial
    d_umin: int
    d_min: float
    d_umax: int
    d_max: float
    d_pow: float
    r_umin: int
    r_min: float
    r_umax: int
    r_max: float
    r_pow: float


def pack_fields(fields, weights, device) -> wp.array | None:
    """Relevant fields as a device array of `_Field`, or None if there are none."""
    out = []
    for f in fields:
        if not ff.is_relevant(f, weights):
            continue
        if f.type not in _TYPE:
            raise ValueError(f"force field type {f.type!r} is not supported")
        s = _Field()
        s.type, s.shape = _TYPE[f.type], _SHAPE[f.shape]
        s.falloff, s.zdir = _FALLOFF[f.falloff], _ZDIR[f.z_direction]
        s.gravity, s.glob = int(f.use_gravity_falloff), int(f.use_global_coords)
        s.loc = wp.vec3(*map(float, f.location))
        z = np.asarray(f.z_axis, np.float64)
        n = np.linalg.norm(z)
        s.nor = wp.vec3(*map(float, z / n if n > 0 else z * 0.0))
        s.weight = float(weights.all * weights.weight(f.type))
        s.strength, s.flow, s.damp, s.size = map(float, (f.strength, f.flow, f.damp, f.size))
        s.d_umin, s.d_min = int(f.use_min_distance), float(f.distance_min)
        s.d_umax, s.d_max, s.d_pow = int(f.use_max_distance), float(f.distance_max), float(f.power)
        s.r_umin, s.r_min = int(f.use_radial_min), float(f.radial_min)
        s.r_umax, s.r_max, s.r_pow = int(f.use_radial_max), float(f.radial_max), float(f.radial_power)
        out.append(s)
    return wp.array(out, dtype=_Field, device=device) if out else None


_hash_arrays = {}


def noise_hash(device) -> wp.array:
    """Blender's 512-entry noise hash table on `device` (cached)."""
    key = str(device)
    if key not in _hash_arrays:
        _hash_arrays[key] = wp.array(ff._HASH.astype(np.int32), dtype=wp.int32, device=device)
    return _hash_arrays[key]


@wp.func
def _grad(h: int, x: float, y: float, z: float):
    h = h & 15
    u = y
    if h < 8:
        u = x
    v = z
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    if (h & 1) != 0:
        u = -u
    if (h & 2) != 0:
        v = -v
    return u + v


@wp.func
def _fade(t: float):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
def _lerp(t: float, a: float, b: float):
    return a + t * (b - a)


@wp.func
def _new_perlin(H: wp.array(dtype=wp.int32), x: float, y: float, z: float):
    # newPerlin (noise_c.cc)
    fx = wp.floor(x)
    fy = wp.floor(y)
    fz = wp.floor(z)
    X = int(fx) & 255
    Y = int(fy) & 255
    Z = int(fz) & 255
    x = x - fx
    y = y - fy
    z = z - fz
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    A = H[X] + Y
    AA = H[A] + Z
    AB = H[A + 1] + Z
    B = H[X + 1] + Y
    BA = H[B] + Z
    BB = H[B + 1] + Z
    a = _lerp(v, _lerp(u, _grad(H[AA], x, y, z), _grad(H[BA], x - 1.0, y, z)),
              _lerp(u, _grad(H[AB], x, y - 1.0, z), _grad(H[BB], x - 1.0, y - 1.0, z)))
    b = _lerp(v, _lerp(u, _grad(H[AA + 1], x, y, z - 1.0), _grad(H[BA + 1], x - 1.0, y, z - 1.0)),
              _lerp(u, _grad(H[AB + 1], x, y - 1.0, z - 1.0),
                    _grad(H[BB + 1], x - 1.0, y - 1.0, z - 1.0)))
    return _lerp(w, a, b)


@wp.func
def _turbulence(H: wp.array(dtype=wp.int32), size: float, x: float, y: float, z: float):
    # BLI_noise_generic_turbulence(size, x, y, z, 2, false, 2): newPerlinU, 3 octaves
    if size != 0.0:
        inv = 1.0 / size
        x = x * inv
        y = y * inv
        z = z * inv
    total = float(0.0)
    amp = float(1.0)
    fs = float(1.0)
    for _ in range(3):
        total = total + (0.5 + 0.5 * _new_perlin(H, fs * x, fs * y, fs * z)) * amp
        amp = amp * 0.5
        fs = fs * 2.0
    return total * (4.0 / 7.0)


@wp.func
def _norm_or_zero(v: wp.vec3):
    n = wp.length(v)
    if n > 0.0:
        return v / n
    return wp.vec3(0.0)


@wp.func
def _keep_length(v: wp.vec3, m: wp.mat33):
    return _norm_or_zero(m * v) * wp.length(v)


@wp.func
def _falloff_func(fac: float, usemin: int, mindist: float, usemax: int, maxdist: float,
                  power: float):
    md = mindist
    if usemin == 0:
        md = 0.0
    if usemin != 0 and fac < md:
        return 1.0
    if usemax != 0 and fac > maxdist:
        return 0.0
    return wp.pow(1.0 + fac - md, -power)


@wp.func
def _effector_force(f: _Field, pos: wp.vec3, vel: wp.vec3, vel_to_sec: float,
                    H: wp.array(dtype=wp.int32)):
    # get_effector_data
    eloc = f.loc
    if f.shape != 0:
        translate = wp.dot(pos - f.loc, f.nor) * f.nor
        if f.type == T_VORTEX or f.shape == 2:
            eloc = f.loc + translate
        else:
            eloc = pos - translate
    vtp = pos - eloc
    dist = wp.length(vtp)
    if f.type == T_HARMONIC and f.size != 0.0:
        if dist > 0.0:
            vtp = vtp * ((dist - f.size) / dist)
        else:
            vtp = wp.vec3(0.0)
    vtp2 = pos - f.loc

    # effector_falloff; a cone at its centre gives NaN in Blender, which counts as off
    fall = f.weight
    fac = wp.dot(vtp2, f.nor)
    if f.falloff == 0:
        fall = fall * _falloff_func(dist, f.d_umin, f.d_min, f.d_umax, f.d_max, f.d_pow)
    else:
        fall = fall * _falloff_func(wp.abs(fac), f.d_umin, f.d_min, f.d_umax, f.d_max, f.d_pow)
        if f.falloff == 1:
            r = wp.length(vtp2 - f.nor * fac)
        else:
            l2 = wp.length(vtp2)
            if l2 == 0.0:
                return wp.vec3(0.0)
            r = wp.degrees(wp.acos(wp.clamp(fac / l2, -1.0, 1.0)))
        fall = fall * _falloff_func(r, f.r_umin, f.r_min, f.r_umax, f.r_max, f.r_pow)
    if (f.zdir == 1 and fac < 0.0) or (f.zdir == 2 and fac > 0.0):
        fall = 0.0
    if not (fall > 0.0):
        return wp.vec3(0.0)

    # do_physical_effector without noise
    s = f.strength
    force = wp.vec3(0.0)
    if f.type == T_WIND:
        force = f.nor * (s * fall)
    elif f.type == T_FORCE:
        force = _norm_or_zero(vtp)
        if f.gravity != 0:
            g = 0.0
            if dist >= FLT_EPS:
                g = s / (dist * dist)
            force = force * (g * fall)
        else:
            force = force * (s * fall)
    elif f.type == T_VORTEX:
        if f.shape == 0:
            force = _norm_or_zero(wp.cross(f.nor, vtp)) * (s * dist * fall)
        else:
            temp = wp.cross(f.nor, vtp2) * (s * fall)
            force = wp.cross(f.nor, temp) * (s * fall) + temp - vel * vel_to_sec
    elif f.type == T_MAGNET:
        temp = f.nor
        if f.shape != 1:
            temp = wp.cross(f.nor, vtp)
        temp = _norm_or_zero(temp)
        force = wp.cross(vel, temp * (s * fall)) * vel_to_sec
    elif f.type == T_HARMONIC:
        force = vtp * (-s * fall) + vel * (-f.damp * 2.0 * wp.sqrt(wp.abs(s)) * vel_to_sec)
    elif f.type == T_DRAG:
        n = wp.length(vel)
        a = n * vel_to_sec
        force = _norm_or_zero(vel) * (-fall * a * (wp.min(s, 2.0) * a + wp.min(f.damp, 2.0)))
    elif f.type == T_TURBULENCE:
        tp = vtp2 + f.nor
        if f.glob != 0:
            tp = pos
        force = wp.vec3(_turbulence(H, f.size, tp[0], tp[1], tp[2]),
                        _turbulence(H, f.size, tp[1], tp[2], tp[0]),
                        _turbulence(H, f.size, tp[2], tp[0], tp[1])) * 2.0 - wp.vec3(1.0)
        force = force * (s * fall)
    total = force / vel_to_sec
    if f.type != T_HARMONIC and f.type != T_DRAG and f.flow != 0.0:
        total = total - vel * (f.flow * fall)
    return total


@wp.kernel
def compute_forces_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                          fields: wp.array(dtype=_Field), origin: wp.vec3, cell: wp.vec3,
                          m: wp.mat33, minv: wp.mat33, dx: float, vel_to_sec: float,
                          density: wp.array3d(dtype=wp.float32), fuel: wp.array3d(dtype=wp.float32),
                          phi_obs: wp.array3d(dtype=wp.float32), mode: int,
                          noise_hash: wp.array(dtype=wp.int32),
                          force: wp.array3d(dtype=wp.vec3)):
    # mode bits: 1 density test, 2 with fuel (max of both), 4 phi_obs test
    i, j, k = wp.tid()
    force[i, j, k] = wp.vec3(0.0)
    if (flags[i, j, k] & OBSTACLE) != 0:
        return
    if (mode & 1) != 0:
        d = density[i, j, k]
        if (mode & 2) != 0:
            d = wp.max(d, fuel[i, j, k])
        if d < FLT_EPS:
            return
    if (mode & 4) != 0 and phi_obs[i, j, k] < 0.0:
        return
    pos = origin + wp.cw_mul(wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5), cell)
    v = _keep_length(vel[i, j, k] * dx, m)
    total = wp.vec3(0.0)
    for n in range(fields.shape[0]):
        total = total + _effector_force(fields[n], pos, v, vel_to_sec, noise_hash)
    r = _keep_length(total, minv) * 0.2
    force[i, j, k] = wp.vec3(wp.clamp(r[0], -1.0, 1.0), wp.clamp(r[1], -1.0, 1.0),
                             wp.clamp(r[2], -1.0, 1.0))
