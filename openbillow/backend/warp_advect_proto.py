# SPDX-License-Identifier: GPL-3.0-or-later
"""Prototype advection schemes for research item A3 (CODE-GAS-015). Off by default.

Selected per backend with `WarpBackend.advect_scheme` (no setting, no add-on property):
  "MACCORMACK"  the Mantaflow step (default; Classic and High Quality, untouched by this module)
  "BFECC"       Kim et al. 2005: fwd = A(q), bwd = A^-1(fwd), q' = q + (q - bwd)/2, result A(q'),
                then Mantaflow's MacCormack limiter (clamp mode 2: revert to fwd outside the source range)
  "MC_CUBIC"    Mantaflow's MacCormack step with monotone cubic sampling (Fedkiw et al. 2001,
                Fritsch-Carlson slope limit) in both semi-Lagrangian passes instead of trilinear
Only float32 cell grids and MAC velocity; vec3 cell grids (noise UVs) keep the default path.
"""
from __future__ import annotations

import warp as wp

from .warp_backend import (FLUID, _axis, _border, _centered_vel, _clamp_cell, _clamp_face,
                           _face_vel, _mc_cell_kernel, _mc_mac_kernel, _outflow_vel_kernel,
                           _sl_cell_kernel, _sl_mac_kernel, _unit, _at_flag)

SCHEMES = ("MACCORMACK", "BFECC", "MC_CUBIC")


# ---- monotone cubic sampling --------------------------------------------------------------------
@wp.func
def _cub1(a: float, b: float, c: float, d: float, t: float):
    # Hermite between b and c; slopes zeroed on a sign change and limited to 3 * delta (monotone).
    dl = c - b
    m1 = 0.5 * (c - a)
    m2 = 0.5 * (d - b)
    m1 = wp.where(m1 * dl > 0.0, m1, 0.0)
    m2 = wp.where(m2 * dl > 0.0, m2, 0.0)
    lim = 3.0 * wp.abs(dl)
    m1 = wp.clamp(m1, -lim, lim)
    m2 = wp.clamp(m2, -lim, lim)
    t2 = t * t
    return b + m1 * t + (3.0 * dl - 2.0 * m1 - m2) * t2 + (m1 + m2 - 2.0 * dl) * t2 * t


@wp.func
def _row(a: wp.array3d(dtype=wp.float32), i: int, j: int, k: int, s: float):
    n = a.shape[0]
    return _cub1(a[wp.max(i - 1, 0), j, k], a[i, j, k], a[i + 1, j, k], a[wp.min(i + 2, n - 1), j, k], s)


@wp.func
def _plane(a: wp.array3d(dtype=wp.float32), i: int, j: int, k: int, s: float, t: float):
    n = a.shape[1]
    return _cub1(_row(a, i, wp.max(j - 1, 0), k, s), _row(a, i, j, k, s), _row(a, i, j + 1, k, s),
                 _row(a, i, wp.min(j + 2, n - 1), k, s), t)


@wp.func
def _sample_cubic(a: wp.array3d(dtype=wp.float32), p: wp.vec3):
    i, s = _axis(p[0] - 0.5, a.shape[0])
    j, t = _axis(p[1] - 0.5, a.shape[1])
    k, f = _axis(p[2] - 0.5, a.shape[2])
    n = a.shape[2]
    return _cub1(_plane(a, i, j, wp.max(k - 1, 0), s, t), _plane(a, i, j, k, s, t),
                 _plane(a, i, j, k + 1, s, t), _plane(a, i, j, wp.min(k + 2, n - 1), s, t), f)


@wp.func
def _row_c(a: wp.array3d(dtype=wp.vec3), c: int, i: int, j: int, k: int, s: float):
    n = a.shape[0]
    return _cub1(a[wp.max(i - 1, 0), j, k][c], a[i, j, k][c], a[i + 1, j, k][c],
                 a[wp.min(i + 2, n - 1), j, k][c], s)


@wp.func
def _plane_c(a: wp.array3d(dtype=wp.vec3), c: int, i: int, j: int, k: int, s: float, t: float):
    n = a.shape[1]
    return _cub1(_row_c(a, c, i, wp.max(j - 1, 0), k, s), _row_c(a, c, i, j, k, s),
                 _row_c(a, c, i, j + 1, k, s), _row_c(a, c, i, wp.min(j + 2, n - 1), k, s), t)


@wp.func
def _sample_comp_cubic(a: wp.array3d(dtype=wp.vec3), c: int, p: wp.vec3):
    i, s = _axis(p[0] - 0.5, a.shape[0])
    j, t = _axis(p[1] - 0.5, a.shape[1])
    k, f = _axis(p[2] - 0.5, a.shape[2])
    n = a.shape[2]
    return _cub1(_plane_c(a, c, i, j, wp.max(k - 1, 0), s, t), _plane_c(a, c, i, j, k, s, t),
                 _plane_c(a, c, i, j, k + 1, s, t), _plane_c(a, c, i, j, wp.min(k + 2, n - 1), s, t), f)


@wp.kernel
def _slc_cell_kernel(vel: wp.array3d(dtype=wp.vec3), src: wp.array3d(dtype=wp.float32),
                     dst: wp.array3d(dtype=wp.float32), dt: float):
    i, j, k = wp.tid()
    if _border(i, j, k, dst.shape[0], dst.shape[1], dst.shape[2]):
        dst[i, j, k] = 0.0
        return
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) - _centered_vel(vel, i, j, k) * dt
    dst[i, j, k] = _sample_cubic(src, p)


@wp.kernel
def _slc_mac_kernel(vel: wp.array3d(dtype=wp.vec3), src: wp.array3d(dtype=wp.vec3),
                    dst: wp.array3d(dtype=wp.vec3), dt: float):
    i, j, k = wp.tid()
    if _border(i, j, k, dst.shape[0], dst.shape[1], dst.shape[2]):
        dst[i, j, k] = wp.vec3()
        return
    p = wp.vec3i(i, j, k)
    p0 = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    r = wp.vec3()
    for c in range(3):
        r[c] = _sample_comp_cubic(src, c, p0 - _face_vel(vel, c, p) * dt)
    dst[i, j, k] = r


# ---- BFECC ----------------------------------------------------------------------------------------
@wp.kernel
def _bf_corr_cell(flags: wp.array3d(dtype=wp.int32), orig: wp.array3d(dtype=wp.float32),
                  bwd: wp.array3d(dtype=wp.float32)):
    # bwd <- orig + (orig - bwd) / 2 in fluid cells, orig elsewhere.
    i, j, k = wp.tid()
    o = orig[i, j, k]
    bwd[i, j, k] = wp.where((flags[i, j, k] & FLUID) != 0, o + 0.5 * (o - bwd[i, j, k]), o)


@wp.kernel
def _bf_corr_mac(flags: wp.array3d(dtype=wp.int32), orig: wp.array3d(dtype=wp.vec3),
                 bwd: wp.array3d(dtype=wp.vec3)):
    # Same fluid-face test as MacCormackCorrectMAC.
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    fluid = (flags[i, j, k] & FLUID) != 0
    o = orig[i, j, k]
    b = bwd[i, j, k]
    r = wp.vec3()
    for c in range(3):
        skip = not fluid
        if p[c] > 0:
            if (_at_flag(flags, p - _unit(c)) & FLUID) == 0:
                skip = True
        r[c] = wp.where(skip, o[c], o[c] + 0.5 * (o[c] - b[c]))
    bwd[i, j, k] = r


@wp.kernel
def _bf_clamp_cell(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                   orig: wp.array3d(dtype=wp.float32), fwd: wp.array3d(dtype=wp.float32),
                   res: wp.array3d(dtype=wp.float32), dt: float):
    # Result written over res; mode 2 limiter against the source range of orig, fallback fwd.
    i, j, k = wp.tid()
    f = fwd[i, j, k]
    d = f
    if (flags[i, j, k] & FLUID) != 0 and not _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        d = _clamp_cell(flags, orig, res[i, j, k], f, wp.vec3i(i, j, k),
                        _centered_vel(vel, i, j, k) * dt, 2)
    res[i, j, k] = d


@wp.kernel
def _bf_clamp_mac(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                  orig: wp.array3d(dtype=wp.vec3), fwd: wp.array3d(dtype=wp.vec3),
                  res: wp.array3d(dtype=wp.vec3), dt: float):
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    fluid = (flags[i, j, k] & FLUID) != 0
    border = _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2])
    f = fwd[i, j, k]
    e = res[i, j, k]
    r = wp.vec3()
    for c in range(3):
        skip = not fluid or border
        if p[c] > 0:
            if (_at_flag(flags, p - _unit(c)) & FLUID) == 0:
                skip = True
        d = f[c]
        if not skip:
            d = _clamp_face(flags, orig, c, e[c], f[c], p, _face_vel(vel, c, p) * dt, 2)
        r[c] = d
    res[i, j, k] = r


# ---- host side --------------------------------------------------------------------------------------
def _third(b, like):
    key = (tuple(like.shape), like.dtype)
    tmp = b.__dict__.setdefault("_proto_tmp", {})
    if key not in tmp:
        tmp[key] = b._new(like.shape, like.dtype)
    return tmp[key]


def advect(b, flags, vel, grid, dt, strength, clamp_mode):
    """Scalar advection with b.advect_scheme; returns False when the default path must run."""
    if grid.dtype != wp.float32:
        return False
    dev, dt = b.device, float(dt)
    fwd, bwd = b._scratch(grid)
    if b.advect_scheme == "MC_CUBIC":
        wp.launch(_slc_cell_kernel, dim=grid.shape, inputs=[vel, grid, fwd, dt], device=dev)
        wp.launch(_slc_cell_kernel, dim=grid.shape, inputs=[vel, fwd, bwd, -dt], device=dev)
        wp.launch(_mc_cell_kernel, dim=grid.shape,
                  inputs=[flags, vel, grid, fwd, bwd, dt, float(strength), int(clamp_mode)], device=dev)
        wp.copy(grid, bwd)
        return True
    if clamp_mode != 2:
        raise ValueError("BFECC prototype supports clamp_mode 2 only")
    res = _third(b, grid)
    wp.launch(_sl_cell_kernel, dim=grid.shape, inputs=[vel, grid, fwd, dt], device=dev)
    wp.launch(_sl_cell_kernel, dim=grid.shape, inputs=[vel, fwd, bwd, -dt], device=dev)
    wp.launch(_bf_corr_cell, dim=grid.shape, inputs=[flags, grid, bwd], device=dev)
    wp.launch(_sl_cell_kernel, dim=grid.shape, inputs=[vel, bwd, res, dt], device=dev)
    wp.launch(_bf_clamp_cell, dim=grid.shape, inputs=[flags, vel, grid, fwd, res, dt], device=dev)
    wp.copy(grid, res)
    return True


def advect_velocity(b, flags, vel, dt, strength, clamp_mode, field):
    dev, dt = b.device, float(dt)
    fwd, bwd = b._scratch(vel)
    if b.advect_scheme == "MC_CUBIC":
        wp.launch(_slc_mac_kernel, dim=vel.shape, inputs=[vel, field, fwd, dt], device=dev)
        wp.launch(_slc_mac_kernel, dim=vel.shape, inputs=[vel, fwd, bwd, -dt], device=dev)
        wp.launch(_mc_mac_kernel, dim=vel.shape,
                  inputs=[flags, vel, field, fwd, bwd, dt, float(strength), int(clamp_mode)], device=dev)
        out = bwd
    else:
        if clamp_mode != 2:
            raise ValueError("BFECC prototype supports clamp_mode 2 only")
        out = _third(b, vel)
        wp.launch(_sl_mac_kernel, dim=vel.shape, inputs=[vel, field, fwd, dt], device=dev)
        wp.launch(_sl_mac_kernel, dim=vel.shape, inputs=[vel, fwd, bwd, -dt], device=dev)
        wp.launch(_bf_corr_mac, dim=vel.shape, inputs=[flags, field, bwd], device=dev)
        wp.launch(_sl_mac_kernel, dim=vel.shape, inputs=[vel, bwd, out, dt], device=dev)
        wp.launch(_bf_clamp_mac, dim=vel.shape, inputs=[flags, vel, field, fwd, out, dt], device=dev)
    wp.launch(_outflow_vel_kernel, dim=vel.shape,
              inputs=[flags, out, field, max(1.0, dt * 4.0)], device=dev)
