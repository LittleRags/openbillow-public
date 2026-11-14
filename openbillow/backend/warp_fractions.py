# SPDX-License-Identifier: GPL-3.0-or-later
"""Obstacle face fractions for liquids (Blender's "Fractions", second-order
obstacle boundaries): Mantaflow `updateFractions` (KnUpdateFractions,
calcFraction), `setObstacleFlags` with fractions (KnUpdateFlagsObs) and
`setWallBcs` with fractions (KnSetWallBcsFrac).

Ported from reference/mantaflow/plugin/initplugins.cpp and
plugin/extforces.cpp (MantaFlow, Apache-2.0, Copyright 2011-2016 Tobias
Pfaff, Nils Thuerey). The pressure matrix and rhs with fractions are in
warp_pressure.py.

KnUpdateFractions writes the faces of its own cell and, at the high domain
sides, those of the next cell. Mantaflow runs it in k slabs, i and j serial:
a write to cell c+1 is overwritten by c+1's own pass unless c+1 is in the
outer layer (which the kernel skips). That serial result is what
_update_fractions_kernel computes, one thread per cell. Mantaflow's max-z
test uses `j` instead of `k` (kept).
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

from .base import Flags

FLUID = wp.constant(int(Flags.FLUID))
EMPTY = wp.constant(int(Flags.EMPTY))
OBSTACLE = wp.constant(int(Flags.OBSTACLE))
INFLOW = wp.constant(int(Flags.INFLOW))
OUTFLOW = wp.constant(int(Flags.OUTFLOW))
OPEN_SIDE = wp.constant(int(Flags.INFLOW | Flags.OUTFLOW | Flags.OPEN))
VECTOR_EPSILON = 1.0e-6


@wp.func
def _calc_fraction(a: float, b: float, thresh: float):
    if a > 0.0 and b > 0.0:
        return 1.0
    if a < 0.0 and b < 0.0:
        return 0.0
    p1 = wp.min(a, b)
    p2 = wp.max(a, b)
    denom = p1 - p2
    if denom > -1.0e-4:
        return 0.5
    frac = 1.0 - p1 / denom
    if frac < thresh:
        frac = 0.0
    return wp.min(1.0, frac)


@wp.func
def _interior(i: int, j: int, k: int, nx: int, ny: int, nz: int):
    return i > 0 and j > 0 and k > 0 and i < nx - 1 and j < ny - 1 and k < nz - 1


@wp.kernel
def _update_fractions_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32),
                             frac: wp.array3d(dtype=wp.vec3), w: int, thresh: float):
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    one = wp.vec3(1.0, 1.0, 1.0)
    if _interior(i, j, k, nx, ny, nz):
        p = phi[i, j, k]
        f = wp.vec3(_calc_fraction(p, phi[i - 1, j, k], thresh),
                    _calc_fraction(p, phi[i, j - 1, k], thresh),
                    _calc_fraction(p, phi[i, j, k - 1], thresh))
        if p >= 0.0:
            if i <= w + 1 and (flags[i - 1, j, k] & OPEN_SIDE) != 0:
                f = one
            if j <= w + 1 and (flags[i, j - 1, k] & OPEN_SIDE) != 0:
                f = one
            if k <= w + 1 and (flags[i, j, k - 1] & OPEN_SIDE) != 0:
                f = one
        frac[i, j, k] = f
        return
    # Outer layer: 0, unless the interior cell below it (high sides only) opened it.
    f = wp.vec3()
    if i == nx - 1 and _interior(i - 1, j, k, nx, ny, nz):
        if phi[i - 1, j, k] >= 0.0 and i - 1 >= nx - w - 2 and (flags[i, j, k] & OPEN_SIDE) != 0:
            f = one
    if j == ny - 1 and _interior(i, j - 1, k, nx, ny, nz):
        if phi[i, j - 1, k] >= 0.0 and j - 1 >= ny - w - 2 and (flags[i, j, k] & OPEN_SIDE) != 0:
            f = one
    if k == nz - 1 and _interior(i, j, k - 1, nx, ny, nz):
        if phi[i, j, k - 1] >= 0.0 and j >= nz - w - 2 and (flags[i, j, k] & OPEN_SIDE) != 0:
            f = one
    frac[i, j, k] = f


@wp.kernel
def _flags_obs_frac_kernel(flags: wp.array3d(dtype=wp.int32), frac: wp.array3d(dtype=wp.vec3),
                           phi_in: wp.array3d(dtype=wp.float32), phi_out: wp.array3d(dtype=wp.float32),
                           use_in: int, use_out: int, bnd: int):
    # KnUpdateFlagsObs with fractions: OBSTACLE only where all six faces are closed.
    i, j, k = wp.tid()
    if i < bnd or j < bnd or k < bnd or i >= flags.shape[0] - bnd or j >= flags.shape[1] - bnd \
            or k >= flags.shape[2] - bnd:
        return
    f = frac[i, j, k]
    s = f[0] + frac[i + 1, j, k][0] + f[1] + frac[i, j + 1, k][1] + f[2] + frac[i, j, k + 1][2]
    if s == 0.0:
        flags[i, j, k] = OBSTACLE
    elif use_in != 0 and phi_in[i, j, k] < 0.0:
        flags[i, j, k] = FLUID | INFLOW
    elif use_out != 0 and phi_out[i, j, k] < 0.0:
        flags[i, j, k] = EMPTY | OUTFLOW
    else:
        flags[i, j, k] = EMPTY


@wp.func
def _normalize(v: wp.vec3):
    # vectorbase.h normalize() (std::hypot)
    l = wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if wp.abs(l - 1.0) < VECTOR_EPSILON:
        return v
    if l > VECTOR_EPSILON:
        return v * (1.0 / l)
    return wp.vec3()


@wp.func
def _unit(c: int):
    e = wp.vec3i(0, 0, 0)
    e[c] = 1
    return e


@wp.func
def _ph(phi: wp.array3d(dtype=wp.float32), p: wp.vec3i):
    return phi[p[0], p[1], p[2]]


@wp.func
def _face_vel(v: wp.array3d(dtype=wp.vec3), c: int, p: wp.vec3i):
    # MACGrid::getAtMACX/Y/Z
    ec = _unit(c)
    r = wp.vec3()
    for d in range(3):
        if d == c:
            r[d] = v[p[0], p[1], p[2]][d]
        else:
            ed = _unit(d)
            q1 = p - ec
            q2 = p + ed
            q3 = p + ed - ec
            r[d] = 0.25 * (v[p[0], p[1], p[2]][d] + v[q1[0], q1[1], q1[2]][d]
                           + v[q2[0], q2[1], q2[2]][d] + v[q3[0], q3[1], q3[2]][d])
    return r


@wp.kernel
def _wall_bcs_frac_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                          out: wp.array3d(dtype=wp.vec3), phi: wp.array3d(dtype=wp.float32)):
    """KnSetWallBcsFrac (no obvel, boundaryWidth 0): on faces touching an OBSTACLE
    cell, remove the component of the interpolated face velocity along the
    normalised phiObs gradient at the face. Reads `vel`, writes `out`."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    v = vel[i, j, k]
    out[i, j, k] = v
    f = flags[i, j, k]
    cur_obs = (f & OBSTACLE) != 0
    if (f & FLUID) == 0 and not cur_obs:
        return
    if not _interior(i, j, k, nx, ny, nz):
        return
    p = wp.vec3i(i, j, k)
    r = v
    for c in range(3):
        ec = _unit(c)
        q = p - ec
        if cur_obs or (flags[q[0], q[1], q[2]] & OBSTACLE) != 0:
            tmp1 = (_ph(phi, p) + _ph(phi, q)) * 0.5
            dphi = wp.vec3()
            for d in range(3):
                if d == c:
                    dphi[d] = _ph(phi, p) - _ph(phi, q)
                else:
                    ed = _unit(d)
                    t2 = (_ph(phi, p + ed) + _ph(phi, q + ed)) * 0.5
                    phi1 = (tmp1 + t2) * 0.5
                    t2 = (_ph(phi, p - ed) + _ph(phi, q - ed)) * 0.5
                    phi2 = (tmp1 + t2) * 0.5
                    dphi[d] = phi1 - phi2
            n = _normalize(dphi)
            vm = _face_vel(vel, c, p)
            r[c] = vm[c] - wp.dot(n, vm) * n[c]
    out[i, j, k] = r


class FractionsMixin:
    """Backend methods for fractions; WarpBackend inherits them."""

    fractions_gen = 0  # bumped by update_fractions; the pressure matrix rebuilds on change

    def update_fractions(self, flags, phi_obs, fractions, boundary_width=1, threshold=0.01):
        wp.launch(_update_fractions_kernel, dim=flags.shape,
                  inputs=[flags, phi_obs, fractions, int(boundary_width), float(threshold)],
                  device=self.device)
        self.fractions_gen += 1

    def _update_flags_obs_frac(self, flags, fractions, phi_in, phi_out, boundary_width):
        wp.launch(_flags_obs_frac_kernel, dim=flags.shape,
                  inputs=[flags, fractions, self._dummy_f if phi_in is None else phi_in,
                          self._dummy_f if phi_out is None else phi_out,
                          int(phi_in is not None), int(phi_out is not None), int(boundary_width)],
                  device=self.device)

    def _set_wall_bcs_frac(self, flags, vel, phi_obs):
        tmp = getattr(self, "_frac_tmp", None)
        if tmp is None or tmp.shape != vel.shape:
            tmp = self._frac_tmp = self._new(vel.shape, wp.vec3)
        wp.launch(_wall_bcs_frac_kernel, dim=flags.shape, inputs=[flags, vel, tmp, phi_obs],
                  device=self.device)
        wp.copy(vel, tmp)
