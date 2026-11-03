# SPDX-License-Identifier: GPL-3.0-or-later
"""Liquid viscosity for the Warp backend: Mantaflow `applyViscosity`.

Ported from reference/mantaflow: plugin/viscosity.cpp and the
ApplyMatrixViscosityU/V/W kernels and GridCg in conjugategrad.h/.cpp
(MantaFlow, Apache-2.0, Copyright 2020 Sebastian Barschkis, Nils Thuerey;
the method of Batty and Bridson, "Accurate Viscous Free Surfaces for Buckling,
Coiling, and Rotating Liquids", SCA 2008).

Steps per call, as in Mantaflow:
1. Liquid volume fractions on a 2x fine grid (8 trilinear phi samples per fine
   cell at +-dx/4, dx = 1/maxres: a Mantaflow quirk, kept; two 5-tet splits),
   averaged 2x2x2 onto cell centres (c), faces (u, v, w) and edges (ex, ey, ez).
2. Face states: a face is solved unless it is on the low domain border or
   touches an OBSTACLE cell (air faces are solved too). Rows only for interior
   cells (bnd 1); obstacle neighbours go into the rhs.
3. Three un-preconditioned CGs (u, v, w) in lockstep, x = 0 at the start, L2
   residual (sum of squares) < accuracy, at most int(1.5 * maxres) iterations.
   Each matrix apply reads the other two components' search vectors from the
   end of the previous iteration; a converged component keeps its last one.
4. vel = solution on solved faces, 0 on every other face.

Differences from Mantaflow, none of which change the result:
- Matrix-free: the 15 coefficients per row are recomputed from the volume
  grids, the flags and the (constant) viscosity in every apply, in the same
  float order, instead of being stored in 45 grids.
- The fine grid is not stored: each cell evaluates the 27 fine cells it needs.
- The three CGs share one apply kernel; their scalars live on the device.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .base import Flags, PressureStats
from .warp_liquid import _sample
from .warp_pressure import BLOCK, _border, _unravel

OBSTACLE = wp.constant(int(Flags.OBSTACLE))
CHECK_EVERY = 8  # CG iterations between host-side convergence checks


# ---- volume fractions (KnEstimateVolumeFraction, KnUpdateVolumeGrid) ---------------

@wp.func
def _tet(p0: float, p1: float, p2: float, p3: float):
    # sortedTetFraction
    return p0 * p0 * p0 / ((p0 - p1) * (p0 - p2) * (p0 - p3))


@wp.func
def _prism(p0: float, p1: float, p2: float, p3: float):
    # sortedPrismFraction
    a = p0 / (p0 - p2)
    b = p0 / (p0 - p3)
    c = p1 / (p1 - p3)
    d = p1 / (p1 - p2)
    return a * b * (1.0 - d) + b * (1.0 - c) * d + c * d


@wp.func
def _frac4(a: float, b: float, c: float, d: float):
    # volumeFraction of one tetrahedron: sort ascending, then by sign pattern
    lo1 = wp.min(a, b)
    hi1 = wp.max(a, b)
    lo2 = wp.min(c, d)
    hi2 = wp.max(c, d)
    m1 = wp.max(lo1, lo2)
    m2 = wp.min(hi1, hi2)
    p0 = wp.min(lo1, lo2)
    p1 = wp.min(m1, m2)
    p2 = wp.max(m1, m2)
    p3 = wp.max(hi1, hi2)
    if p3 <= 0.0:
        return 1.0
    if p2 <= 0.0:
        return 1.0 - _tet(p3, p2, p1, p0)
    if p1 <= 0.0:
        return _prism(p0, p1, p2, p3)
    if p0 <= 0.0:
        return _tet(p0, p1, p2, p3)
    return 0.0


@wp.func
def _fine_fraction(phi: wp.array3d(dtype=wp.float32), fi: int, fj: int, fk: int,
                   start: float, off: float):
    # KnEstimateVolumeFraction for fine cell (fi, fj, fk)
    cx = start + float(fi) * 0.5
    cy = start + float(fj) * 0.5
    cz = start + float(fk) * 0.5
    p000 = _sample(phi, wp.vec3(cx - off, cy - off, cz - off))
    p001 = _sample(phi, wp.vec3(cx - off, cy - off, cz + off))
    p010 = _sample(phi, wp.vec3(cx - off, cy + off, cz - off))
    p011 = _sample(phi, wp.vec3(cx - off, cy + off, cz + off))
    p100 = _sample(phi, wp.vec3(cx + off, cy - off, cz - off))
    p101 = _sample(phi, wp.vec3(cx + off, cy - off, cz + off))
    p110 = _sample(phi, wp.vec3(cx + off, cy + off, cz - off))
    p111 = _sample(phi, wp.vec3(cx + off, cy + off, cz + off))
    return (_frac4(p000, p001, p101, p011)
            + _frac4(p000, p101, p100, p110)
            + _frac4(p000, p010, p011, p110)
            + _frac4(p101, p011, p111, p110)
            + 2.0 * _frac4(p000, p011, p101, p110)
            + _frac4(p100, p101, p001, p111)
            + _frac4(p100, p001, p000, p010)
            + _frac4(p100, p110, p111, p010)
            + _frac4(p001, p111, p011, p010)
            + 2.0 * _frac4(p100, p111, p001, p010)) / 12.0


@wp.kernel
def _volumes_kernel(phi: wp.array3d(dtype=wp.float32), vol: wp.array4d(dtype=wp.float32),
                    start: float, off: float):
    """vol[0..6] = c, u, v, w, ex, ey, ez: means of 2x2x2 fine cells, summed in
    FOR_INT_IJK order (k outer). Cell axes use fine offsets 1-2, face axes 0-1
    (and stay 0 at index 0)."""
    i, j, k = wp.tid()
    c = float(0.0)
    u = float(0.0)
    v = float(0.0)
    w = float(0.0)
    ex = float(0.0)
    ey = float(0.0)
    ez = float(0.0)
    for kk in range(3):
        for jj in range(3):
            for ii in range(3):
                f = float(0.0)
                if (ii > 0 or i > 0) and (jj > 0 or j > 0) and (kk > 0 or k > 0):
                    f = _fine_fraction(phi, 2 * i - 1 + ii, 2 * j - 1 + jj, 2 * k - 1 + kk, start, off)
                cxa = ii >= 1
                cya = jj >= 1
                cza = kk >= 1
                fxa = ii <= 1 and i >= 1
                fya = jj <= 1 and j >= 1
                fza = kk <= 1 and k >= 1
                c += wp.where(cxa and cya and cza, f, 0.0)
                u += wp.where(fxa and cya and cza, f, 0.0)
                v += wp.where(cxa and fya and cza, f, 0.0)
                w += wp.where(cxa and cya and fza, f, 0.0)
                ex += wp.where(cxa and fya and fza, f, 0.0)
                ey += wp.where(fxa and cya and fza, f, 0.0)
                ez += wp.where(fxa and fya and cza, f, 0.0)
    vol[0, i, j, k] = c / 8.0
    vol[1, i, j, k] = u / 8.0
    vol[2, i, j, k] = v / 8.0
    vol[3, i, j, k] = w / 8.0
    vol[4, i, j, k] = ex / 8.0
    vol[5, i, j, k] = ey / 8.0
    vol[6, i, j, k] = ez / 8.0


# ---- matrix rows (solveViscosity, ApplyMatrixViscosityU/V/W) -----------------------
# Row of component a at face p. Directions are listed +x, -x, +y, -y, +z, -z.
# Own axis: factor 2 * factor * visc * c-volume; other axes b: factor * edge visc *
# volume of the edge along the third axis.

@wp.func
def _e(a: int):
    return wp.vec3i(wp.where(a == 0, 1, 0), wp.where(a == 1, 1, 0), wp.where(a == 2, 1, 0))


@wp.func
def _solved(flags: wp.array3d(dtype=wp.int32), a: int, p: wp.vec3i):
    # KnUpdateFaceStates: TypeFluid unless on the low border or next to an obstacle cell
    if p[a] == 0:
        return False
    q = p - _e(a)
    return (flags[p[0], p[1], p[2]] & OBSTACLE) == 0 and (flags[q[0], q[1], q[2]] & OBSTACLE) == 0


@wp.func
def _fplus(vol: wp.array4d(dtype=wp.float32), a: int, b: int, p: wp.vec3i, c2: float, ce: float):
    if b == a:
        return c2 * vol[0, p[0], p[1], p[2]]
    q = p + _e(b)
    return ce * vol[7 - a - b, q[0], q[1], q[2]]


@wp.func
def _fminus(vol: wp.array4d(dtype=wp.float32), a: int, b: int, p: wp.vec3i, c2: float, ce: float):
    if b == a:
        q = p - _e(a)
        return c2 * vol[0, q[0], q[1], q[2]]
    return ce * vol[7 - a - b, p[0], p[1], p[2]]


@wp.func
def _own(flags: wp.array3d(dtype=wp.int32), s: wp.array3d(dtype=wp.vec3), a: int, q: wp.vec3i, f: float):
    # src(nb) * A(nb) with A = -factor on a solved face, 0 otherwise
    return wp.where(_solved(flags, a, q), s[q[0], q[1], q[2]][a] * (-f), 0.0)


@wp.func
def _cpl(flags: wp.array3d(dtype=wp.int32), s: wp.array3d(dtype=wp.vec3), b: int, q: wp.vec3i, f: float):
    # srcV(nb) * helper(nb), helper = +-factor (sign carried by f) on a solved face
    return wp.where(_solved(flags, b, q), s[q[0], q[1], q[2]][b] * f, 0.0)


@wp.func
def _coupling(flags: wp.array3d(dtype=wp.int32), s: wp.array3d(dtype=wp.vec3), a: int, b: int,
              p: wp.vec3i, fp: float, fm: float):
    ea = _e(a)
    eb = _e(b)
    return (_cpl(flags, s, b, p + eb, -fp) + _cpl(flags, s, b, p - ea + eb, fp)
            + _cpl(flags, s, b, p, fm) + _cpl(flags, s, b, p - ea, -fm))


@wp.func
def _apply_row(flags: wp.array3d(dtype=wp.int32), vol: wp.array4d(dtype=wp.float32),
               s: wp.array3d(dtype=wp.vec3), a: int, p: wp.vec3i, c2: float, ce: float):
    fx0 = _fplus(vol, a, 0, p, c2, ce)
    fx1 = _fminus(vol, a, 0, p, c2, ce)
    fy0 = _fplus(vol, a, 1, p, c2, ce)
    fy1 = _fminus(vol, a, 1, p, c2, ce)
    fz0 = _fplus(vol, a, 2, p, c2, ce)
    fz1 = _fminus(vol, a, 2, p, c2, ce)
    a0 = vol[a + 1, p[0], p[1], p[2]] + fx0 + fx1 + fy0 + fy1 + fz0 + fz1
    ex = _e(0)
    ey = _e(1)
    ez = _e(2)
    v = (s[p[0], p[1], p[2]][a] * a0 + _own(flags, s, a, p + ex, fx0) + _own(flags, s, a, p + ey, fy0)
         + _own(flags, s, a, p + ez, fz0) + _own(flags, s, a, p - ex, fx1)
         + _own(flags, s, a, p - ey, fy1) + _own(flags, s, a, p - ez, fz1))
    # the other two components in axis order: U row reads V then W, V reads U then W, W reads U then V
    b1 = wp.where(a == 0, 1, 0)
    b2 = wp.where(a == 2, 1, 2)
    f1p = wp.where(b1 == 0, fx0, fy0)
    f1m = wp.where(b1 == 0, fx1, fy1)
    f2p = wp.where(b2 == 1, fy0, fz0)
    f2m = wp.where(b2 == 1, fy1, fz1)
    return v + (_coupling(flags, s, a, b1, p, f1p, f1m) + _coupling(flags, s, a, b2, p, f2p, f2m))


@wp.func
def _rhs_own(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3), a: int, q: wp.vec3i,
             f: float, rhs: float):
    if _solved(flags, a, q):
        return rhs
    return rhs - (-vel[q[0], q[1], q[2]][a]) * f


@wp.func
def _rhs_cpl(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3), b: int, q: wp.vec3i,
             neg: bool, f: float, rhs: float):
    if _solved(flags, b, q):
        return rhs
    x = vel[q[0], q[1], q[2]][b]
    return rhs - wp.where(neg, -x, x) * f


@wp.func
def _rhs_coupling(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3), a: int, b: int,
                  p: wp.vec3i, fp: float, fm: float, rhs: float):
    ea = _e(a)
    eb = _e(b)
    r = _rhs_cpl(flags, vel, b, p + eb, True, fp, rhs)
    r = _rhs_cpl(flags, vel, b, p - ea + eb, False, fp, r)
    r = _rhs_cpl(flags, vel, b, p, False, fm, r)
    return _rhs_cpl(flags, vel, b, p - ea, True, fm, r)


@wp.func
def _rhs_row(flags: wp.array3d(dtype=wp.int32), vol: wp.array4d(dtype=wp.float32),
             vel: wp.array3d(dtype=wp.vec3), a: int, p: wp.vec3i, c2: float, ce: float):
    fx0 = _fplus(vol, a, 0, p, c2, ce)
    fx1 = _fminus(vol, a, 0, p, c2, ce)
    fy0 = _fplus(vol, a, 1, p, c2, ce)
    fy1 = _fminus(vol, a, 1, p, c2, ce)
    fz0 = _fplus(vol, a, 2, p, c2, ce)
    fz1 = _fminus(vol, a, 2, p, c2, ce)
    ex = _e(0)
    ey = _e(1)
    ez = _e(2)
    r = vol[a + 1, p[0], p[1], p[2]] * vel[p[0], p[1], p[2]][a]
    r = _rhs_own(flags, vel, a, p + ex, fx0, r)
    r = _rhs_own(flags, vel, a, p - ex, fx1, r)
    r = _rhs_own(flags, vel, a, p + ey, fy0, r)
    r = _rhs_own(flags, vel, a, p - ey, fy1, r)
    r = _rhs_own(flags, vel, a, p + ez, fz0, r)
    r = _rhs_own(flags, vel, a, p - ez, fz1, r)
    b1 = wp.where(a == 0, 1, 0)
    b2 = wp.where(a == 2, 1, 2)
    r = _rhs_coupling(flags, vel, a, b1, p, wp.where(b1 == 0, fx0, fy0), wp.where(b1 == 0, fx1, fy1), r)
    return _rhs_coupling(flags, vel, a, b2, p, wp.where(b2 == 1, fy0, fz0), wp.where(b2 == 1, fy1, fz1), r)


@wp.func
def _row(flags: wp.array3d(dtype=wp.int32), a: int, p: wp.vec3i):
    # FOR_IJK_BND(flags, 1) and a solved face
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    return (not _border(p[0], p[1], p[2], nx, ny, nz)) and _solved(flags, a, p)


# ---- CG (GridCg::doInit / iterate, three components in lockstep) ---------------------
# sc (float64): [0:3] dot(tmp, search), [3:6] sum of squares of the residual (GridSumSqr,
# squared in double), [6:9] dot(residual, residual) (GridDotProduct: float products).
# sig = mSigma, beta, res = mResNorm (float32); act = still iterating, upd = update search.

@wp.kernel
def _rhs_kernel(flags: wp.array3d(dtype=wp.int32), vol: wp.array4d(dtype=wp.float32),
                vel: wp.array3d(dtype=wp.vec3), x: wp.array3d(dtype=wp.vec3),
                r: wp.array3d(dtype=wp.vec3), s: wp.array3d(dtype=wp.vec3),
                c2: float, ce: float, sc: wp.array(dtype=wp.float64)):
    """rhs straight into the residual: dst = 0, residual = rhs, search = residual,
    sc[6:9] = dot(residual, residual)."""
    t = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    rv = wp.vec3(0.0)
    if t < nx * ny * nz:
        p = _unravel(t, ny, nz)
        for a in range(3):
            if _row(flags, a, p):
                rv[a] = _rhs_row(flags, vol, vel, a, p, c2, ce)
        x[p[0], p[1], p[2]] = wp.vec3(0.0)
        r[p[0], p[1], p[2]] = rv
        s[p[0], p[1], p[2]] = rv
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[0] * rv[0]))), offset=6)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[1] * rv[1]))), offset=7)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[2] * rv[2]))), offset=8)


@wp.kernel
def _init_scalars_kernel(sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32),
                         act: wp.array(dtype=wp.int32), iters: wp.array(dtype=wp.int32),
                         res: wp.array(dtype=wp.float32), flag: wp.array(dtype=wp.int32)):
    for a in range(3):
        sig[a] = wp.float32(sc[6 + a])
        act[a] = 1
        iters[a] = 0
        res[a] = 1e20
    flag[0] = 1  # any component still iterating
    flag[1] = 0  # diverged


@wp.kernel
def _zero_kernel(sc: wp.array(dtype=wp.float64)):
    for c in range(9):
        sc[c] = wp.float64(0.0)


@wp.kernel
def _apply_kernel(flags: wp.array3d(dtype=wp.int32), vol: wp.array4d(dtype=wp.float32),
                  s: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.vec3),
                  cst: wp.array(dtype=wp.float32), sc: wp.array(dtype=wp.float64)):
    """tmp = A search for all three components, sc[0:3] = dot(tmp, search).
    cst = (2 * factor * visc, factor * edge visc), on the device so a captured
    graph survives dt changes."""
    t = wp.tid()
    c2 = cst[0]
    ce = cst[1]
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    d = wp.vec3(0.0)
    sv = wp.vec3(0.0)
    if t < nx * ny * nz:
        p = _unravel(t, ny, nz)
        for a in range(3):
            if _row(flags, a, p):
                d[a] = _apply_row(flags, vol, s, a, p, c2, ce)
        tmp[p[0], p[1], p[2]] = d
        sv = s[p[0], p[1], p[2]]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[0] * sv[0]))), offset=0)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[1] * sv[1]))), offset=1)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[2] * sv[2]))), offset=2)


@wp.kernel
def _update_kernel(x: wp.array3d(dtype=wp.vec3), r: wp.array3d(dtype=wp.vec3),
                   s: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.vec3),
                   sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32),
                   act: wp.array(dtype=wp.int32)):
    """Active components: dst += alpha search, residual -= alpha tmp; then
    sc[3:6] = GridSumSqr(residual), sc[6:9] = dot(residual, residual)."""
    t = wp.tid()
    nx, ny, nz = x.shape[0], x.shape[1], x.shape[2]
    rv = wp.vec3(0.0)
    if t < nx * ny * nz:
        p = _unravel(t, ny, nz)
        xv = x[p[0], p[1], p[2]]
        rv = r[p[0], p[1], p[2]]
        sv = s[p[0], p[1], p[2]]
        tv = tmp[p[0], p[1], p[2]]
        for a in range(3):
            if act[a] != 0:
                dp = wp.float32(sc[a])
                alpha = float(0.0)
                if wp.abs(dp) > 0.0:
                    alpha = sig[a] / dp
                xv[a] = xv[a] + sv[a] * alpha
                rv[a] = rv[a] + tv[a] * (-alpha)
        x[p[0], p[1], p[2]] = xv
        r[p[0], p[1], p[2]] = rv
    for a in range(3):
        rd = wp.float64(rv[a])
        wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(rd * rd)), offset=3 + a)
        wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[a] * rv[a]))), offset=6 + a)


@wp.kernel
def _finish_kernel(sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32),
                   beta: wp.array(dtype=wp.float32), act: wp.array(dtype=wp.int32),
                   upd: wp.array(dtype=wp.int32), iters: wp.array(dtype=wp.int32),
                   res: wp.array(dtype=wp.float32), flag: wp.array(dtype=wp.int32),
                   accuracy: float, max_iter: int):
    """The scalar tail of GridCg::iterate per active component, and the
    driver's iteration cap."""
    anyact = int(0)
    for a in range(3):
        upd[a] = 0
        if act[a] != 0:
            iters[a] = iters[a] + 1
            rn = wp.float32(sc[3 + a])
            res[a] = rn
            if rn < accuracy:
                act[a] = 0
            else:
                sn = wp.float32(sc[6 + a])
                beta[a] = sn / sig[a]
                sig[a] = sn
                upd[a] = 1
                if not (rn < 1e35):
                    flag[1] = 1
                if iters[a] >= max_iter:
                    act[a] = 0
        anyact = anyact + act[a]
    flag[0] = anyact


@wp.kernel
def _search_kernel(r: wp.array3d(dtype=wp.vec3), s: wp.array3d(dtype=wp.vec3),
                   beta: wp.array(dtype=wp.float32), upd: wp.array(dtype=wp.int32)):
    """UpdateSearchVec: search = residual + beta search."""
    i, j, k = wp.tid()
    rv = r[i, j, k]
    sv = s[i, j, k]
    for a in range(3):
        if upd[a] != 0:
            sv[a] = rv[a] + beta[a] * sv[a]
    s[i, j, k] = sv


@wp.kernel
def _apply_velocities_kernel(flags: wp.array3d(dtype=wp.int32), x: wp.array3d(dtype=wp.vec3),
                             vel: wp.array3d(dtype=wp.vec3)):
    """KnApplyVelocities: solution on solved faces, 0 elsewhere (whole grid)."""
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    xv = x[i, j, k]
    vel[i, j, k] = wp.vec3(wp.where(_solved(flags, 0, p), xv[0], 0.0),
                           wp.where(_solved(flags, 1, p), xv[1], 0.0),
                           wp.where(_solved(flags, 2, p), xv[2], 0.0))


class ViscositySolver:
    """Volume grids and CG work grids for one domain shape (7 floats + 4 vec3 per cell)."""

    def __init__(self, shape, device):
        self.shape = tuple(int(s) for s in shape)
        self.device = device
        d, sh = device, self.shape
        self.vol = wp.zeros((7,) + sh, dtype=wp.float32, device=d)
        self.x, self.r, self.s, self.tmp = (wp.zeros(sh, dtype=wp.vec3, device=d) for _ in range(4))
        self.sc = wp.zeros(9, dtype=wp.float64, device=d)
        self.sig, self.beta, self.res = (wp.zeros(3, dtype=wp.float32, device=d) for _ in range(3))
        self.act, self.upd, self.iters = (wp.zeros(3, dtype=wp.int32, device=d) for _ in range(3))
        self.flag = wp.zeros(2, dtype=wp.int32, device=d)
        self.cst = wp.zeros(2, dtype=wp.float32, device=d)
        self.flag_host = wp.zeros(2, dtype=wp.int32, device="cpu", pinned=d.is_cuda)
        self.use_graph = d.is_cuda
        self._graph = None  # (key, CUDA graph of CHECK_EVERY iterations)
        self.n = int(np.prod(sh))
        self.padded = (self.n + BLOCK - 1) // BLOCK * BLOCK

    def nbytes(self) -> int:
        return sum(a.capacity for a in (self.vol, self.x, self.r, self.s, self.tmp))

    @staticmethod
    def constants(viscosity: float, dt: float, dx: float):
        """(2 * factor * visc, factor * edge visc) in float32 as the C++ computes them:
        factor = Real(dt * (1 / dx)^2) in double, edge visc = 0.25 * (sum of 4 cells)."""
        f32 = np.float32
        factor = f32(np.float64(f32(dt)) * (1.0 / np.float64(f32(dx))) ** 2)
        v = f32(viscosity)
        ve = f32(0.25 * np.float64(v + v + v + v))
        return float(f32(f32(2 * factor) * v)), float(f32(factor * ve))

    def compute_volumes(self, phi, dx: float):
        dx = np.float32(dx)
        start = float(np.float32(0.25 * np.float64(dx)))
        off = float(np.float32(0.5 * np.float64(np.float32(0.5 * np.float64(dx)))))
        wp.launch(_volumes_kernel, dim=self.shape, inputs=[phi, self.vol, start, off], device=self.device)

    def _iterations(self, flags, accuracy, max_iter):
        d = self.device
        for _ in range(CHECK_EVERY):
            wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=d)
            wp.launch(_apply_kernel, dim=self.padded, block_dim=BLOCK,
                      inputs=[flags, self.vol, self.s, self.tmp, self.cst, self.sc], device=d)
            wp.launch(_update_kernel, dim=self.padded, block_dim=BLOCK,
                      inputs=[self.x, self.r, self.s, self.tmp, self.sc, self.sig, self.act], device=d)
            wp.launch(_finish_kernel, dim=1,
                      inputs=[self.sc, self.sig, self.beta, self.act, self.upd, self.iters, self.res,
                              self.flag, accuracy, max_iter], device=d)
            wp.launch(_search_kernel, dim=self.shape, inputs=[self.r, self.s, self.beta, self.upd], device=d)
        wp.copy(self.flag_host, self.flag)

    def prepare(self, flags, phi, vel, viscosity: float, dt: float, dx: float):
        """Volumes, rhs and GridCg::doInit for all three components."""
        d = self.device
        c2, ce = self.constants(viscosity, dt, dx)
        self.cst.assign(np.array([c2, ce], np.float32))
        self.compute_volumes(phi, dx)
        wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=d)
        wp.launch(_rhs_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[flags, self.vol, vel, self.x, self.r, self.s, c2, ce, self.sc], device=d)
        wp.launch(_init_scalars_kernel, dim=1,
                  inputs=[self.sc, self.sig, self.act, self.iters, self.res, self.flag], device=d)

    def apply(self, flags, src, dst):
        """dst = A src (tests)."""
        wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=self.device)
        wp.launch(_apply_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[flags, self.vol, src, dst, self.cst, self.sc], device=self.device)

    def solve(self, flags, phi, vel, viscosity: float, dt: float, dx: float,
              accuracy: float = 1e-9, max_iter_factor: float = 1.5) -> PressureStats:
        """Mantaflow applyViscosity with a constant viscosity grid; dt and dx in
        Mantaflow units (dx = 1 / maxres). Stats: most iterations of the three
        CGs and the largest final L2 residual."""
        d = self.device
        accuracy = float(np.float32(accuracy))
        max_iter = int(np.float32(max_iter_factor) * np.float32(max(self.shape)))
        self.prepare(flags, phi, vel, viscosity, dt, dx)
        key = (flags.ptr, accuracy, max_iter)
        done = 0
        while done < max_iter:
            if not self.use_graph:
                self._iterations(flags, accuracy, max_iter)
            else:
                if self._graph is None or self._graph[0] != key:
                    with wp.ScopedCapture(device=d) as cap:
                        self._iterations(flags, accuracy, max_iter)
                    self._graph = (key, cap.graph)
                wp.capture_launch(self._graph[1])
            done += CHECK_EVERY
            wp.synchronize_device(d)
            fl = self.flag_host.numpy()
            if fl[1]:
                raise RuntimeError("viscosity solve diverged (residual > 1e35)")
            if not fl[0]:
                break
        wp.launch(_apply_velocities_kernel, dim=self.shape, inputs=[flags, self.x, vel], device=d)
        iters = self.iters.numpy()
        return PressureStats(int(iters.max()), float(self.res.numpy().max()))
