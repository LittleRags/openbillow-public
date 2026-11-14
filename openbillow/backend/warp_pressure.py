# SPDX-License-Identifier: GPL-3.0-or-later
"""Pressure projection for the Warp backend: Mantaflow `solvePressure` with the
PcMGStatic preconditioner (multigrid-preconditioned CG).

Ported from reference/mantaflow: plugin/pressure.cpp and conjugategrad.cpp
(MantaFlow, Apache-2.0, Copyright 2011-2016 Tobias Pfaff, Nils Thuerey) and
multigrid.cpp (Apache-2.0, Copyright 2016 Florian Ferstl; the solver of Dick
et al., "Solving the Fluid Pressure Poisson Equation Using Multigrid -
Evaluation and Improvements", IEEE TVCG 2015).

Steps per solve, as in Mantaflow:
1. rhs = negative divergence in fluid cells (MakeRhs).
2. Laplace matrix A0/Ai/Aj/Ak from the flags (MakeLaplaceMatrix), with one
   cell pinned to zero when asked (fixPressure).
3. CG from pressure = 0, max-norm residual < accuracy, at most 100 iterations
   (Mantaflow's cap with an MG preconditioner). Each iteration applies one
   multigrid V-cycle (GridMg::doVCycle) as the preconditioner.
4. Subtract the pressure gradient (knCorrectVelocity).

Differences from Mantaflow, none of which change the converged answer:
- The matrix and multigrid hierarchy are cached and rebuilt only when the
  FLUID/OBSTACLE/EMPTY layout of the flags (or the pinning) changes. PcMGStatic
  never rebuilds, so the two agree for static obstacles.
- The coarsest level is solved with a precomputed dense inverse (pseudo-inverse when singular) instead
  of Jacobi-CG to 1e-4 * accuracy (the same answer to within that tolerance).
- Coarse-vertex selection (GridMg::genCoarseGrid, a serial greedy pass) runs as
  a one-thread Warp CPU kernel; everything per iteration runs on the GPU.
  High Quality uses a parallel rule on the GPU instead (`parallel_coarsening`):
  a different preconditioner, so other iteration counts, same converged answer.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

from .base import Flags, PressureStats
from .warp_common import BLOCK, border, unravel

FLUID = wp.constant(int(Flags.FLUID))
EMPTY = wp.constant(int(Flags.EMPTY))
OUTFLOW = wp.constant(int(Flags.OUTFLOW))
OBSTACLE = wp.constant(int(Flags.OBSTACLE))
LAYOUT = wp.constant(int(Flags.FLUID | Flags.OBSTACLE | Flags.EMPTY))  # bits the matrix depends on

TRIVIAL_SCALE = wp.constant(1e-6)  # GridMg::mTrivialEquationScale
MG_MAX_ITER = 100                  # solvePressureSystem: maxIter with PcMGStatic/PcMGDynamic
GF_CLAMP = wp.constant(1e-4)        # solvePressure gfClamp

# Vertex types. Level 0: inactive / active / active trivial equation (GridMg::VertexType).
# The greedy coarsening also uses removed / zero / free, as in genCoarseGrid.
VT_INACTIVE = wp.constant(0)
VT_ACTIVE = wp.constant(1)
VT_TRIVIAL = wp.constant(2)
VT_REMOVED = wp.constant(3)
VT_ZERO = wp.constant(4)
VT_FREE = wp.constant(5)
HEAP_K = wp.constant(9)  # NKMinHeap buckets: at most 8 free interpolation vertices in 3D


# ---- small helpers ------------------------------------------------------------

@wp.func
def _odd_weight(p: wp.vec3i):
    """Restriction/interpolation weight 1 / 2^(number of odd coordinates)."""
    return 1.0 / float(1 << ((p[0] & 1) + (p[1] & 1) + (p[2] & 1)))


@wp.func
def _coef(a: wp.array4d(dtype=wp.float32), u: wp.vec3i, w: wp.vec3i):
    """Entry A(u, w) of a coarse 27-point stencil kept in GridMg's symmetric
    14-entry storage (0 = centre, 1..13 = stencil index s = 14..26)."""
    d = w - u
    s = (d[0] + 1) + 3 * (d[1] + 1) + 9 * (d[2] + 1)
    if s >= 13:
        return a[u[0], u[1], u[2], s - 13]
    return a[w[0], w[1], w[2], 13 - s]


@wp.func
def _diag0(a0: wp.array3d(dtype=wp.float32), t0: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    """Level-0 diagonal as GridMg stores it (trivial equations scaled down)."""
    a = a0[i, j, k]
    if t0[i, j, k] == VT_TRIVIAL:
        a = a * TRIVIAL_SCALE
    return a


@wp.func
def _coef7(a0: wp.array3d(dtype=wp.float32), t0: wp.array3d(dtype=wp.int32),
           ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
           ak: wp.array3d(dtype=wp.float32), u: wp.vec3i, w: wp.vec3i):
    """Entry A(u, w) of the level-0 7-point matrix (GridMg level 0)."""
    d = w - u
    r = float(0.0)
    if d[0] == 0 and d[1] == 0 and d[2] == 0:
        r = _diag0(a0, t0, u[0], u[1], u[2])
    elif d[1] == 0 and d[2] == 0:
        if d[0] == 1:
            r = ai[u[0], u[1], u[2]]
        else:
            r = ai[w[0], w[1], w[2]]
    elif d[0] == 0 and d[2] == 0:
        if d[1] == 1:
            r = aj[u[0], u[1], u[2]]
        else:
            r = aj[w[0], w[1], w[2]]
    elif d[0] == 0 and d[1] == 0:
        if d[2] == 1:
            r = ak[u[0], u[1], u[2]]
        else:
            r = ak[w[0], w[1], w[2]]
    return r


@wp.func
def _b0(t0: wp.array3d(dtype=wp.int32), res: wp.array3d(dtype=wp.float32), i: int, j: int, k: int):
    """Level-0 right-hand side of the V-cycle (knSetRhs: trivial equations scaled)."""
    b = res[i, j, k]
    if t0[i, j, k] == VT_TRIVIAL:
        b = b * TRIVIAL_SCALE
    return b


@wp.func
def _offdiag0_sum(ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
                  ak: wp.array3d(dtype=wp.float32), x: wp.array3d(dtype=wp.float32),
                  i: int, j: int, k: int):
    s = float(0.0)
    if i > 0:
        s += ai[i - 1, j, k] * x[i - 1, j, k]
    if i < x.shape[0] - 1:
        s += ai[i, j, k] * x[i + 1, j, k]
    if j > 0:
        s += aj[i, j - 1, k] * x[i, j - 1, k]
    if j < x.shape[1] - 1:
        s += aj[i, j, k] * x[i, j + 1, k]
    if k > 0:
        s += ak[i, j, k - 1] * x[i, j, k - 1]
    if k < x.shape[2] - 1:
        s += ak[i, j, k] * x[i, j, k + 1]
    return s


# ---- matrix, rhs and velocity correction (pressure.cpp) ------------------------

# Ghost fluid (pressure.cpp thetaHelper / ghostFluidHelper), used when a phi is
# passed (Blender: only the guided liquid's PD_fluid_guiding). gfClamp = 1e-4.

@wp.func
def _theta(inside: float, outside: float):
    denom = inside - outside
    if denom > -1e-04:
        return 0.5
    return wp.max(0.0, wp.min(1.0, inside / denom))


@wp.func
def _ghost_fluid(inside: float, outside: float):
    """ghostFluidHelper; Mantaflow's clamp returns gfClamp itself (kept)."""
    a = _theta(inside, outside)
    if a < GF_CLAMP:
        return GF_CLAMP
    return 1.0 - 1.0 / a


@wp.kernel
def _ghost_diag_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32),
                       a0: wp.array3d(dtype=wp.float32)):
    """ApplyGhostFluidDiagonal (bnd 1)."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    if border(i, j, k, nx, ny, nz) or (flags[i, j, k] & FLUID) == 0:
        return
    ph = phi[i, j, k]
    d = a0[i, j, k]
    if (flags[i - 1, j, k] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i - 1, j, k])
    if (flags[i + 1, j, k] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i + 1, j, k])
    if (flags[i, j - 1, k] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i, j - 1, k])
    if (flags[i, j + 1, k] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i, j + 1, k])
    if (flags[i, j, k - 1] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i, j, k - 1])
    if (flags[i, j, k + 1] & EMPTY) != 0:
        d -= _ghost_fluid(ph, phi[i, j, k + 1])
    a0[i, j, k] = d


@wp.func
def _surf_tens(st: float, curv_out: float, curv_in: float, inside: float, outside: float):
    """surfTensHelper: the ghost-fluid pressure jump st * curvature at the surface,
    curvature interpolated between the fluid cell and its EMPTY neighbour."""
    return st * (curv_out - _ghost_fluid(inside, outside) * curv_in)


@wp.kernel
def _curvature_kernel(phi: wp.array3d(dtype=wp.float32), curv: wp.array3d(dtype=wp.float32),
                      clamp: float):
    """CurvatureOp (h = 1, bnd 1): mean curvature (sum of the principal ones, 2/R
    for a sphere) of phi, clamped to +-clamp. Outer layer 0."""
    i, j, k = wp.tid()
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]
    if border(i, j, k, nx, ny, nz):
        curv[i, j, k] = 0.0
        return
    c = phi[i, j, k]
    x = 0.5 * (phi[i + 1, j, k] - phi[i - 1, j, k])
    y = 0.5 * (phi[i, j + 1, k] - phi[i, j - 1, k])
    z = 0.5 * (phi[i, j, k + 1] - phi[i, j, k - 1])
    xx = phi[i + 1, j, k] - 2.0 * c + phi[i - 1, j, k]
    yy = phi[i, j + 1, k] - 2.0 * c + phi[i, j - 1, k]
    zz = phi[i, j, k + 1] - 2.0 * c + phi[i, j, k - 1]
    xy = 0.25 * (phi[i + 1, j + 1, k] + phi[i - 1, j - 1, k] - phi[i - 1, j + 1, k] - phi[i + 1, j - 1, k])
    xz = 0.25 * (phi[i + 1, j, k + 1] + phi[i - 1, j, k - 1] - phi[i - 1, j, k + 1] - phi[i + 1, j, k - 1])
    yz = 0.25 * (phi[i, j + 1, k + 1] + phi[i, j - 1, k - 1] - phi[i, j + 1, k - 1] - phi[i, j - 1, k + 1])
    num = x * x * (yy + zz) + y * y * (xx + zz) + z * z * (xx + yy)
    num -= 2.0 * (x * y * xy + x * z * xz + y * z * yz)
    kappa = num / wp.pow(wp.max(x * x + y * y + z * z, 1e-6), 1.5)
    curv[i, j, k] = wp.clamp(kappa, -clamp, clamp)


@wp.func
def _clamped(inside: float, outside: float):
    return _theta(inside, outside) < GF_CLAMP


@wp.kernel
def _replace_clamped_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                            phi: wp.array3d(dtype=wp.float32)):
    """knReplaceClampedGhostFluidVels (bnd 1): writes EMPTY cells' faces from
    FLUID neighbours' faces only, so it is race-free in place."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    if border(i, j, k, nx, ny, nz) or (flags[i, j, k] & EMPTY) == 0:
        return
    ph = phi[i, j, k]
    v = vel[i, j, k]
    if (flags[i - 1, j, k] & FLUID) != 0 and _clamped(phi[i - 1, j, k], ph):
        v[0] = vel[i - 1, j, k][0]
    if (flags[i, j - 1, k] & FLUID) != 0 and _clamped(phi[i, j - 1, k], ph):
        v[1] = vel[i, j - 1, k][1]
    if (flags[i, j, k - 1] & FLUID) != 0 and _clamped(phi[i, j, k - 1], ph):
        v[2] = vel[i, j, k - 1][2]
    if (flags[i + 1, j, k] & FLUID) != 0 and _clamped(phi[i + 1, j, k], ph):
        v[0] = vel[i + 1, j, k][0]
    if (flags[i, j + 1, k] & FLUID) != 0 and _clamped(phi[i, j + 1, k], ph):
        v[1] = vel[i, j + 1, k][1]
    if (flags[i, j, k + 1] & FLUID) != 0 and _clamped(phi[i, j, k + 1], ph):
        v[2] = vel[i, j, k + 1][2]
    vel[i, j, k] = v


@wp.kernel
def _laplace_kernel(flags: wp.array3d(dtype=wp.int32), a0: wp.array3d(dtype=wp.float32),
                    ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
                    ak: wp.array3d(dtype=wp.float32), frac: wp.array3d(dtype=wp.vec3),
                    use_frac: int, wts: wp.array3d(dtype=wp.vec3), use_w: int):
    """MakeLaplaceMatrix (bnd 1). With fractions: diagonal = sum of the six face
    fractions, off-diagonal = -fraction to FLUID neighbours. Fills zeros everywhere else.
    use_w (not in Mantaflow): each face coefficient is also times its face weight
    wts (1/rho at the face, lower face per axis), i.e. the matrix of div(w grad p)."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    d = float(0.0)
    x = float(0.0)
    y = float(0.0)
    z = float(0.0)
    if border(i, j, k, nx, ny, nz) or (flags[i, j, k] & FLUID) == 0:
        a0[i, j, k] = d
        ai[i, j, k] = x
        aj[i, j, k] = y
        ak[i, j, k] = z
        return
    wl = wp.vec3(1.0, 1.0, 1.0)
    wx = float(1.0)
    wy = float(1.0)
    wz = float(1.0)
    if use_w != 0:
        wl = wts[i, j, k]
        wx = wts[i + 1, j, k][0]
        wy = wts[i, j + 1, k][1]
        wz = wts[i, j, k + 1][2]
    if use_frac != 0:
        f = frac[i, j, k]
        fx = frac[i + 1, j, k][0] * wx
        fy = frac[i, j + 1, k][1] * wy
        fz = frac[i, j, k + 1][2] * wz
        d = f[0] * wl[0] + fx + f[1] * wl[1] + fy + f[2] * wl[2] + fz
        if (flags[i + 1, j, k] & FLUID) != 0:
            x = -fx
        if (flags[i, j + 1, k] & FLUID) != 0:
            y = -fy
        if (flags[i, j, k + 1] & FLUID) != 0:
            z = -fz
    else:
        if (flags[i - 1, j, k] & OBSTACLE) == 0:
            d += wl[0]
        if (flags[i + 1, j, k] & OBSTACLE) == 0:
            d += wx
        if (flags[i, j - 1, k] & OBSTACLE) == 0:
            d += wl[1]
        if (flags[i, j + 1, k] & OBSTACLE) == 0:
            d += wy
        if (flags[i, j, k - 1] & OBSTACLE) == 0:
            d += wl[2]
        if (flags[i, j, k + 1] & OBSTACLE) == 0:
            d += wz
        if (flags[i + 1, j, k] & FLUID) != 0:
            x = -wx
        if (flags[i, j + 1, k] & FLUID) != 0:
            y = -wy
        if (flags[i, j, k + 1] & FLUID) != 0:
            z = -wz
    a0[i, j, k] = d
    ai[i, j, k] = x
    aj[i, j, k] = y
    ak[i, j, k] = z


@wp.kernel
def _pin_kernel(a0: wp.array3d(dtype=wp.float32), ai: wp.array3d(dtype=wp.float32),
                aj: wp.array3d(dtype=wp.float32), ak: wp.array3d(dtype=wp.float32), p: wp.vec3i):
    """fixPressure, matrix part (value 0, so the rhs only changes at p)."""
    i, j, k = p[0], p[1], p[2]
    a0[i, j, k] = 1.0
    ai[i, j, k] = 0.0
    aj[i, j, k] = 0.0
    ak[i, j, k] = 0.0
    ai[i - 1, j, k] = 0.0
    aj[i, j - 1, k] = 0.0
    ak[i, j, k - 1] = 0.0


@wp.kernel
def _types0_kernel(a0: wp.array3d(dtype=wp.float32), ai: wp.array3d(dtype=wp.float32),
                   aj: wp.array3d(dtype=wp.float32), ak: wp.array3d(dtype=wp.float32),
                   t0: wp.array3d(dtype=wp.int32)):
    """knActivateVertices: active where the diagonal is non-zero; trivial rows
    (diagonal 1, no couplings) are marked so GridMg can scale them down."""
    i, j, k = wp.tid()
    t = VT_INACTIVE
    if a0[i, j, k] != 0.0:
        t = VT_ACTIVE
        off = wp.abs(ai[i, j, k]) + wp.abs(aj[i, j, k]) + wp.abs(ak[i, j, k])
        if i > 0:
            off += wp.abs(ai[i - 1, j, k])
        if j > 0:
            off += wp.abs(aj[i, j - 1, k])
        if k > 0:
            off += wp.abs(ak[i, j, k - 1])
        if a0[i, j, k] == 1.0 and off == 0.0:
            t = VT_TRIVIAL
    t0[i, j, k] = t


@wp.kernel
def _types_changed_kernel(t: wp.array3d(dtype=wp.int32), old: wp.array3d(dtype=wp.int32),
                          changed: wp.array(dtype=wp.int32)):
    i, j, k = wp.tid()
    if t[i, j, k] != old[i, j, k]:
        changed[0] = 1


@wp.kernel
def _layout_kernel(flags: wp.array3d(dtype=wp.int32), snap: wp.array3d(dtype=wp.uint8),
                   write: int, changed: wp.array(dtype=wp.int32)):
    """Compare (or store) the flag bits the matrix depends on."""
    i, j, k = wp.tid()
    m = flags[i, j, k] & LAYOUT
    if write != 0:
        snap[i, j, k] = wp.uint8(m)
    elif m != int(snap[i, j, k]):
        changed[0] = 1


@wp.kernel
def _rhs_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                rhs: wp.array3d(dtype=wp.float32), pin: wp.vec3i,
                div: wp.array(dtype=wp.float32), frac: wp.array3d(dtype=wp.vec3),
                obvel: wp.array3d(dtype=wp.vec3), use_frac: int, use_obvel: int,
                divg: wp.array3d(dtype=wp.float32), use_divg: int,
                phi: wp.array3d(dtype=wp.float32), curv: wp.array3d(dtype=wp.float32),
                st: wp.array(dtype=wp.float32), use_st: int):
    """MakeRhs (negative divergence; with fractions face-weighted, plus the
    (1 - fraction) obstacle velocity terms; with use_st the surface tension jump
    st[0] * curv at faces to EMPTY cells) plus the rhs part of fixPressure. Outer
    layer stays 0 (fresh grid in Mantaflow).
    div[0] = target divergence (0 in Mantaflow), kept in an array for the graph;
    divg = per-cell target divergence (combustion gas expansion), added on top."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    s = float(0.0)
    if use_frac != 0 and not border(i, j, k, nx, ny, nz) and (flags[i, j, k] & FLUID) != 0:
        v = vel[i, j, k]
        f = frac[i, j, k]
        fx = frac[i + 1, j, k][0]
        fy = frac[i, j + 1, k][1]
        fz = frac[i, j, k + 1][2]
        s = f[0] * v[0] - fx * vel[i + 1, j, k][0] + f[1] * v[1] - fy * vel[i, j + 1, k][1]
        s += f[2] * v[2] - fz * vel[i, j, k + 1][2]
        if use_obvel != 0:
            o = obvel[i, j, k]
            s += (1.0 - f[0]) * o[0] - (1.0 - fx) * obvel[i + 1, j, k][0]
            s += (1.0 - f[1]) * o[1] - (1.0 - fy) * obvel[i, j + 1, k][1]
            s += (1.0 - f[2]) * o[2] - (1.0 - fz) * obvel[i, j, k + 1][2]
        s += div[0]
    elif not border(i, j, k, nx, ny, nz) and (flags[i, j, k] & FLUID) != 0:
        v = vel[i, j, k]
        s = (v[0] - vel[i + 1, j, k][0]) + (v[1] - vel[i, j + 1, k][1])
        s += v[2] - vel[i, j, k + 1][2] + div[0]
    if use_divg != 0 and (flags[i, j, k] & FLUID) != 0 and not border(i, j, k, nx, ny, nz):
        s += divg[i, j, k]
    if use_st != 0 and (flags[i, j, k] & FLUID) != 0 and not border(i, j, k, nx, ny, nz):
        t = st[0]
        ph = phi[i, j, k]
        c = curv[i, j, k]
        if (flags[i - 1, j, k] & EMPTY) != 0:
            s += _surf_tens(t, curv[i - 1, j, k], c, ph, phi[i - 1, j, k])
        if (flags[i + 1, j, k] & EMPTY) != 0:
            s += _surf_tens(t, curv[i + 1, j, k], c, ph, phi[i + 1, j, k])
        if (flags[i, j - 1, k] & EMPTY) != 0:
            s += _surf_tens(t, curv[i, j - 1, k], c, ph, phi[i, j - 1, k])
        if (flags[i, j + 1, k] & EMPTY) != 0:
            s += _surf_tens(t, curv[i, j + 1, k], c, ph, phi[i, j + 1, k])
        if (flags[i, j, k - 1] & EMPTY) != 0:
            s += _surf_tens(t, curv[i, j, k - 1], c, ph, phi[i, j, k - 1])
        if (flags[i, j, k + 1] & EMPTY) != 0:
            s += _surf_tens(t, curv[i, j, k + 1], c, ph, phi[i, j, k + 1])
    if i == pin[0] and j == pin[1] and k == pin[2]:
        s = 0.0
    rhs[i, j, k] = s


@wp.kernel
def _correct_vel_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                        p: wp.array3d(dtype=wp.float32), phi: wp.array3d(dtype=wp.float32),
                        use_phi: int, curv: wp.array3d(dtype=wp.float32),
                        st: wp.array(dtype=wp.float32), use_st: int,
                        wts: wp.array3d(dtype=wp.vec3), use_w: int):
    """knCorrectVelocity (bnd 1). Faces between fluid and plain OUTFLOW cells
    are left alone, as in Mantaflow (OUTFLOW cells are neither fluid nor empty).
    use_phi: knCorrectVelocityGhostFluid fused in; use_st (needs use_phi): its
    surface tension part. use_w (not with use_phi): u -= w grad p, w = the face
    weight of each lower face (all faces this cell writes)."""
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    if border(i, j, k, nx, ny, nz):
        return
    f = flags[i, j, k]
    v = vel[i, j, k]
    vx = v[0]
    vy = v[1]
    vz = v[2]
    pc = p[i, j, k]
    w = wp.vec3(1.0, 1.0, 1.0)
    if use_w != 0:
        w = wts[i, j, k]
    if (f & FLUID) != 0:
        if (flags[i - 1, j, k] & FLUID) != 0:
            vx -= w[0] * (pc - p[i - 1, j, k])
        if (flags[i, j - 1, k] & FLUID) != 0:
            vy -= w[1] * (pc - p[i, j - 1, k])
        if (flags[i, j, k - 1] & FLUID) != 0:
            vz -= w[2] * (pc - p[i, j, k - 1])
        if (flags[i - 1, j, k] & EMPTY) != 0:
            vx -= w[0] * pc
        if (flags[i, j - 1, k] & EMPTY) != 0:
            vy -= w[1] * pc
        if (flags[i, j, k - 1] & EMPTY) != 0:
            vz -= w[2] * pc
        if use_phi != 0:
            ph = phi[i, j, k]
            if (flags[i - 1, j, k] & EMPTY) != 0:
                vx += pc * _ghost_fluid(ph, phi[i - 1, j, k])
            if (flags[i, j - 1, k] & EMPTY) != 0:
                vy += pc * _ghost_fluid(ph, phi[i, j - 1, k])
            if (flags[i, j, k - 1] & EMPTY) != 0:
                vz += pc * _ghost_fluid(ph, phi[i, j, k - 1])
            if use_st != 0:
                t = st[0]
                c = curv[i, j, k]
                if (flags[i - 1, j, k] & EMPTY) != 0:
                    vx += _surf_tens(t, curv[i - 1, j, k], c, ph, phi[i - 1, j, k])
                if (flags[i, j - 1, k] & EMPTY) != 0:
                    vy += _surf_tens(t, curv[i, j - 1, k], c, ph, phi[i, j - 1, k])
                if (flags[i, j, k - 1] & EMPTY) != 0:
                    vz += _surf_tens(t, curv[i, j, k - 1], c, ph, phi[i, j, k - 1])
    elif (f & EMPTY) != 0 and (f & OUTFLOW) == 0:
        ph = phi[0, 0, 0]
        if use_phi != 0:
            ph = phi[i, j, k]
        if (flags[i - 1, j, k] & FLUID) != 0:
            vx += w[0] * p[i - 1, j, k]
            if use_phi != 0:
                vx -= p[i - 1, j, k] * _ghost_fluid(phi[i - 1, j, k], ph)
        else:
            vx = 0.0
        if (flags[i, j - 1, k] & FLUID) != 0:
            vy += w[1] * p[i, j - 1, k]
            if use_phi != 0:
                vy -= p[i, j - 1, k] * _ghost_fluid(phi[i, j - 1, k], ph)
        else:
            vy = 0.0
        if (flags[i, j, k - 1] & FLUID) != 0:
            vz += w[2] * p[i, j, k - 1]
            if use_phi != 0:
                vz -= p[i, j, k - 1] * _ghost_fluid(phi[i, j, k - 1], ph)
        else:
            vz = 0.0
        if use_st != 0:
            t = st[0]
            c = curv[i, j, k]
            if (flags[i - 1, j, k] & FLUID) != 0:
                vx -= _surf_tens(t, c, curv[i - 1, j, k], phi[i - 1, j, k], ph)
            if (flags[i, j - 1, k] & FLUID) != 0:
                vy -= _surf_tens(t, c, curv[i, j - 1, k], phi[i, j - 1, k], ph)
            if (flags[i, j, k - 1] & FLUID) != 0:
                vz -= _surf_tens(t, c, curv[i, j, k - 1], phi[i, j, k - 1], ph)
    else:
        return
    vel[i, j, k] = wp.vec3(vx, vy, vz)


# ---- CG (conjugategrad.cpp, GridCg::doInit / iterate) ------------------------------
# Scalars live on the device so one CG iteration needs no host round trip
# (and can be captured in a CUDA graph). sc[0] = dot(tmp, search),
# sc[1] = sigma_new; sig[0] = sigma (Real, as mSigma); res[0] = max |residual|.

@wp.kernel
def _cg_init_kernel(rhs: wp.array3d(dtype=wp.float32), p: wp.array3d(dtype=wp.float32),
                    r: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    p[i, j, k] = 0.0
    r[i, j, k] = rhs[i, j, k]


@wp.kernel
def _zero_scalars_kernel(sc: wp.array(dtype=wp.float64), res: wp.array(dtype=wp.float32)):
    sc[0] = wp.float64(0.0)
    sc[1] = wp.float64(0.0)
    res[0] = 0.0


@wp.kernel
def _apply_matrix_kernel(flags: wp.array3d(dtype=wp.int32), a0: wp.array3d(dtype=wp.float32),
                         ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
                         ak: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.float32),
                         dst: wp.array3d(dtype=wp.float32), sc: wp.array(dtype=wp.float64)):
    """ApplyMatrix (dst = A src, dst = src outside fluid), fused with
    sc[0] += dot(dst, src). 1D launch padded to whole blocks."""
    t = wp.tid()
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    prod = float(0.0)
    if t < nx * ny * nz:
        c = unravel(t, ny, nz)
        i, j, k = c[0], c[1], c[2]
        v = src[i, j, k]
        if (flags[i, j, k] & FLUID) != 0:
            v = v * a0[i, j, k] + _offdiag0_sum(ai, aj, ak, src, i, j, k)
        dst[i, j, k] = v
        prod = v * src[i, j, k]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(prod))))


@wp.kernel
def _cg_update_kernel(p: wp.array3d(dtype=wp.float32), r: wp.array3d(dtype=wp.float32),
                      search: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.float32),
                      sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32),
                      res: wp.array(dtype=wp.float32)):
    """dst += alpha search, residual -= alpha tmp, res = max |residual|."""
    t = wp.tid()
    nx, ny, nz = p.shape[0], p.shape[1], p.shape[2]
    dp = wp.float32(sc[0])
    alpha = float(0.0)
    if wp.abs(dp) > 0.0:
        alpha = sig[0] / dp
    m = float(0.0)
    if t < nx * ny * nz:
        c = unravel(t, ny, nz)
        i, j, k = c[0], c[1], c[2]
        p[i, j, k] = p[i, j, k] + alpha * search[i, j, k]
        rv = r[i, j, k] - alpha * tmp[i, j, k]
        r[i, j, k] = rv
        m = wp.abs(rv)
        if not wp.isfinite(m):  # tile_max and atomic_max skip NaN: make it trip the divergence guard
            m = wp.inf
    tm = wp.tile_max(wp.tile(m))
    wp.atomic_max(res, 0, tm[0])


@wp.kernel
def _dot_kernel(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32),
                sc: wp.array(dtype=wp.float64)):
    """sc[1] += dot(a, b) (GridDotProduct, double accumulation)."""
    t = wp.tid()
    nx, ny, nz = a.shape[0], a.shape[1], a.shape[2]
    prod = float(0.0)
    if t < nx * ny * nz:
        c = unravel(t, ny, nz)
        prod = a[c[0], c[1], c[2]] * b[c[0], c[1], c[2]]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(prod))), offset=1)


@wp.kernel
def _set_sigma_kernel(sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32)):
    sig[0] = wp.float32(sc[1])


@wp.kernel
def _search_init_kernel(tmp: wp.array3d(dtype=wp.float32), search: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    search[i, j, k] = tmp[i, j, k]


@wp.kernel
def _search_update_kernel(tmp: wp.array3d(dtype=wp.float32), search: wp.array3d(dtype=wp.float32),
                          sc: wp.array(dtype=wp.float64), sig: wp.array(dtype=wp.float32)):
    """UpdateSearchVec: search = tmp + beta search, beta = sigma_new / sigma."""
    i, j, k = wp.tid()
    beta = wp.float32(sc[1]) / sig[0]
    search[i, j, k] = tmp[i, j, k] + beta * search[i, j, k]


# ---- multigrid V-cycle (multigrid.cpp) ---------------------------------------------

@wp.kernel
def _mg_smooth0_kernel(t0: wp.array3d(dtype=wp.int32), a0: wp.array3d(dtype=wp.float32),
                       ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
                       ak: wp.array3d(dtype=wp.float32), res: wp.array3d(dtype=wp.float32),
                       x: wp.array3d(dtype=wp.float32), color: int):
    """knSmoothColor on level 0: red-black Gauss-Seidel, colour 0 = even i+j+k."""
    i, j, k = wp.tid()
    if ((i + j + k) & 1) != color or t0[i, j, k] == VT_INACTIVE:
        return
    s = _b0(t0, res, i, j, k) - _offdiag0_sum(ai, aj, ak, x, i, j, k)
    x[i, j, k] = s / _diag0(a0, t0, i, j, k)


@wp.kernel
def _mg_residual0_kernel(t0: wp.array3d(dtype=wp.int32), a0: wp.array3d(dtype=wp.float32),
                         ai: wp.array3d(dtype=wp.float32), aj: wp.array3d(dtype=wp.float32),
                         ak: wp.array3d(dtype=wp.float32), res: wp.array3d(dtype=wp.float32),
                         x: wp.array3d(dtype=wp.float32), r: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    if t0[i, j, k] == VT_INACTIVE:
        return
    r[i, j, k] = (_b0(t0, res, i, j, k) - _offdiag0_sum(ai, aj, ak, x, i, j, k)
                  - _diag0(a0, t0, i, j, k) * x[i, j, k])


@wp.func
def _stencil27_sum(a: wp.array4d(dtype=wp.float32), t: wp.array3d(dtype=wp.int32),
                   x: wp.array3d(dtype=wp.float32), v: wp.vec3i, with_centre: int):
    s = float(0.0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0 and with_centre == 0:
                    continue
                n = v + wp.vec3i(dx, dy, dz)
                if n[0] < 0 or n[1] < 0 or n[2] < 0 or n[0] >= x.shape[0] or n[1] >= x.shape[1] or n[2] >= x.shape[2]:
                    continue
                if t[n[0], n[1], n[2]] == VT_INACTIVE:
                    continue
                s += _coef(a, v, n) * x[n[0], n[1], n[2]]
    return s


@wp.kernel
def _mg_smooth_kernel(t: wp.array3d(dtype=wp.int32), a: wp.array4d(dtype=wp.float32),
                      b: wp.array3d(dtype=wp.float32), x: wp.array3d(dtype=wp.float32),
                      off: wp.vec3i):
    """knSmoothColor on levels > 0: one of 8 colours (2x2x2 block offsets)."""
    bi, bj, bk = wp.tid()
    v = wp.vec3i(2 * bi, 2 * bj, 2 * bk) + off
    if v[0] >= x.shape[0] or v[1] >= x.shape[1] or v[2] >= x.shape[2]:
        return
    if t[v[0], v[1], v[2]] == VT_INACTIVE:
        return
    s = b[v[0], v[1], v[2]] - _stencil27_sum(a, t, x, v, 0)
    x[v[0], v[1], v[2]] = s / a[v[0], v[1], v[2], 0]


@wp.kernel
def _mg_residual_kernel(t: wp.array3d(dtype=wp.int32), a: wp.array4d(dtype=wp.float32),
                        b: wp.array3d(dtype=wp.float32), x: wp.array3d(dtype=wp.float32),
                        r: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    if t[i, j, k] == VT_INACTIVE:
        return
    r[i, j, k] = b[i, j, k] - _stencil27_sum(a, t, x, wp.vec3i(i, j, k), 1)


@wp.kernel
def _mg_restrict_kernel(tf: wp.array3d(dtype=wp.int32), rf: wp.array3d(dtype=wp.float32),
                        tc: wp.array3d(dtype=wp.int32), bc: wp.array3d(dtype=wp.float32),
                        xc: wp.array3d(dtype=wp.float32)):
    """knRestrict (full weighting, R = I^T), plus x_coarse = 0."""
    i, j, k = wp.tid()
    xc[i, j, k] = 0.0
    if tc[i, j, k] == VT_INACTIVE:
        return
    s = float(0.0)
    for z in range(wp.max(0, 2 * k - 1), wp.min(rf.shape[2] - 1, 2 * k + 1) + 1):
        for y in range(wp.max(0, 2 * j - 1), wp.min(rf.shape[1] - 1, 2 * j + 1) + 1):
            for x in range(wp.max(0, 2 * i - 1), wp.min(rf.shape[0] - 1, 2 * i + 1) + 1):
                if tf[x, y, z] != VT_INACTIVE:
                    s += _odd_weight(wp.vec3i(x, y, z)) * rf[x, y, z]
    bc[i, j, k] = s


@wp.kernel
def _mg_interp_add_kernel(tf: wp.array3d(dtype=wp.int32), xf: wp.array3d(dtype=wp.float32),
                          tc: wp.array3d(dtype=wp.int32), xc: wp.array3d(dtype=wp.float32)):
    """knInterpolate (trilinear) fused with knAddAssign: x_fine += I x_coarse."""
    i, j, k = wp.tid()
    if tf[i, j, k] == VT_INACTIVE:
        return
    s = float(0.0)
    for z in range(k // 2, (k + 1) // 2 + 1):
        for y in range(j // 2, (j + 1) // 2 + 1):
            for x in range(i // 2, (i + 1) // 2 + 1):
                if tc[x, y, z] != VT_INACTIVE:
                    s += xc[x, y, z]
    xf[i, j, k] = xf[i, j, k] + _odd_weight(wp.vec3i(i, j, k)) * s


@wp.kernel
def _mg_coarsest_kernel(act: wp.array(dtype=wp.vec3i), m: wp.array2d(dtype=wp.float32),
                        b: wp.array3d(dtype=wp.float32), x: wp.array3d(dtype=wp.float32)):
    """Coarsest level: x = minv b over the active vertices (minv from _invert_coarsest)."""
    r = wp.tid()
    s = float(0.0)
    for c in range(act.shape[0]):
        p = act[c]
        s += m[r, c] * b[p[0], p[1], p[2]]
    q = act[r]
    x[q[0], q[1], q[2]] = s


@wp.kernel
def _mg_galerkin_kernel(tf: wp.array3d(dtype=wp.int32), af: wp.array4d(dtype=wp.float32),
                        a0: wp.array3d(dtype=wp.float32), ai: wp.array3d(dtype=wp.float32),
                        aj: wp.array3d(dtype=wp.float32), ak: wp.array3d(dtype=wp.float32),
                        fine_is_l0: int, tc: wp.array3d(dtype=wp.int32),
                        ac: wp.array4d(dtype=wp.float32)):
    """knGenCoarseGridOperator: A_l = R A_{l-1} I over the paths
    V <-R- U <-A- W <-I- N (generic l > 1 form; for l = 1 the fine matrix is
    read from the 7-point level-0 arrays, the same sum as Mantaflow's paths)."""
    i, j, k = wp.tid()
    for s in range(14):
        ac[i, j, k, s] = 0.0
    if tc[i, j, k] == VT_INACTIVE:
        return
    v = wp.vec3i(i, j, k)
    fx, fy, fz = tf.shape[0], tf.shape[1], tf.shape[2]
    cx, cy, cz = tc.shape[0], tc.shape[1], tc.shape[2]
    for uz in range(wp.max(0, 2 * k - 1), wp.min(fz - 1, 2 * k + 1) + 1):
        for uy in range(wp.max(0, 2 * j - 1), wp.min(fy - 1, 2 * j + 1) + 1):
            for ux in range(wp.max(0, 2 * i - 1), wp.min(fx - 1, 2 * i + 1) + 1):
                u = wp.vec3i(ux, uy, uz)
                if tf[ux, uy, uz] == VT_INACTIVE:
                    continue
                rw = _odd_weight(u)
                for nz in range(wp.max(0, uz - 1) // 2, wp.min(cz - 1, (uz + 2) // 2) + 1):
                    for ny in range(wp.max(0, uy - 1) // 2, wp.min(cy - 1, (uy + 2) // 2) + 1):
                        for nx in range(wp.max(0, ux - 1) // 2, wp.min(cx - 1, (ux + 2) // 2) + 1):
                            if tc[nx, ny, nz] == VT_INACTIVE:
                                continue
                            sc = (nx - i + 1) + 3 * (ny - j + 1) + 9 * (nz - k + 1)
                            if sc < 13:
                                continue
                            acc = float(0.0)
                            for wz in range(wp.max(0, wp.max(uz - 1, 2 * nz - 1)), wp.min(fz - 1, wp.min(uz + 1, 2 * nz + 1)) + 1):
                                for wy in range(wp.max(0, wp.max(uy - 1, 2 * ny - 1)), wp.min(fy - 1, wp.min(uy + 1, 2 * ny + 1)) + 1):
                                    for wx in range(wp.max(0, wp.max(ux - 1, 2 * nx - 1)), wp.min(fx - 1, wp.min(ux + 1, 2 * nx + 1)) + 1):
                                        if tf[wx, wy, wz] == VT_INACTIVE:
                                            continue
                                        w = wp.vec3i(wx, wy, wz)
                                        a = float(0.0)
                                        if fine_is_l0 != 0:
                                            a = _coef7(a0, tf, ai, aj, ak, u, w)
                                        else:
                                            a = _coef(af, u, w)
                                        acc += a * _odd_weight(w)
                            ac[i, j, k, sc - 13] = ac[i, j, k, sc - 13] + rw * acc


# ---- coarse-vertex selection (GridMg::genCoarseGrid with NKMinHeap), CPU -----------
# Heap entries: indices 0..K-1 are bucket heads, K + ID the vertices. IDs use
# Mantaflow's linear order (x fastest) so ties break the same way.
# st[0] = heap size, st[1] = min key.

@wp.func
def _heap_unlink(key: wp.array(dtype=wp.int32), prv: wp.array(dtype=wp.int32),
                 nxt: wp.array(dtype=wp.int32), e: int):
    pred = prv[e]
    succ = nxt[e]
    nxt[pred] = succ
    if succ != -1:
        prv[succ] = pred


@wp.func
def _heap_fix_min(nxt: wp.array(dtype=wp.int32), st: wp.array(dtype=wp.int32)):
    m = st[1]
    while m < HEAP_K:
        if nxt[m] != -1:
            break
        m += 1
    st[1] = m


@wp.func
def _heap_set_key(key: wp.array(dtype=wp.int32), prv: wp.array(dtype=wp.int32),
                  nxt: wp.array(dtype=wp.int32), st: wp.array(dtype=wp.int32), vid: int, kv: int):
    e = HEAP_K + vid
    if key[e] != kv:
        if key[e] != -1:
            _heap_unlink(key, prv, nxt, e)
            if key[e] == st[1]:
                if st[0] == 1:
                    st[1] = -1
                else:
                    _heap_fix_min(nxt, st)
            st[0] = st[0] - 1
        key[e] = kv
        if kv == -1:
            nxt[e] = -1
            prv[e] = -1
        else:
            st[0] = st[0] + 1
            if st[1] == -1:
                st[1] = kv
            else:
                st[1] = wp.min(st[1], kv)
            head = nxt[kv]
            nxt[kv] = e
            prv[e] = kv
            nxt[e] = head
            if head != -1:
                prv[head] = e


@wp.kernel
def _coarsen_kernel(tf: wp.array3d(dtype=wp.int32), tc: wp.array3d(dtype=wp.int32),
                    key: wp.array(dtype=wp.int32), prv: wp.array(dtype=wp.int32),
                    nxt: wp.array(dtype=wp.int32), st: wp.array(dtype=wp.int32)):
    fx, fy, fz = tf.shape[0], tf.shape[1], tf.shape[2]
    cx, cy, cz = tc.shape[0], tc.shape[1], tc.shape[2]
    for z in range(cz):
        for y in range(cy):
            for x in range(cx):
                tc[x, y, z] = VT_FREE
    for z in range(fz):
        for y in range(fy):
            for x in range(fx):
                if tf[x, y, z] != VT_INACTIVE:
                    _heap_set_key(key, prv, nxt, st, x + fx * (y + fy * z),
                                  1 << ((x & 1) + (y & 1) + (z & 1)))
    while st[0] > 0:
        # popMin
        e = nxt[st[1]]
        vid = e - HEAP_K
        _heap_unlink(key, prv, nxt, e)
        key[e] = -1
        prv[e] = -1
        nxt[e] = -1
        st[0] = st[0] - 1
        if st[0] == 0:
            st[1] = -1
        else:
            _heap_fix_min(nxt, st)
        vx = vid % fx
        vy = (vid // fx) % fy
        vz = vid // (fx * fy)
        done = int(0)
        for iz in range(vz // 2, (vz + 1) // 2 + 1):
            for iy in range(vy // 2, (vy + 1) // 2 + 1):
                for ix in range(vx // 2, (vx + 1) // 2 + 1):
                    if tc[ix, iy, iz] != VT_FREE:
                        continue
                    if done != 0:
                        tc[ix, iy, iz] = VT_REMOVED
                    else:
                        tc[ix, iy, iz] = VT_ZERO
                        done = 1
                    for rz in range(wp.max(0, 2 * iz - 1), wp.min(fz - 1, 2 * iz + 1) + 1):
                        for ry in range(wp.max(0, 2 * iy - 1), wp.min(fy - 1, 2 * iy + 1) + 1):
                            for rx in range(wp.max(0, 2 * ix - 1), wp.min(fx - 1, 2 * ix + 1) + 1):
                                r = rx + fx * (ry + fy * rz)
                                kr = key[HEAP_K + r]
                                if kr > 1:
                                    _heap_set_key(key, prv, nxt, st, r, kr - 1)
                                elif kr > -1:
                                    _heap_set_key(key, prv, nxt, st, r, -1)
    for z in range(cz):
        for y in range(cy):
            for x in range(cx):
                if tc[x, y, z] == VT_ZERO:
                    tc[x, y, z] = VT_ACTIVE
                else:
                    tc[x, y, z] = VT_INACTIVE


@wp.kernel
def _coarsen_parallel_kernel(tf: wp.array3d(dtype=wp.int32), tc: wp.array3d(dtype=wp.int32)):
    """A coarse vertex is active when any fine vertex in its restriction stencil is
    active (not Mantaflow's rule; see `parallel_coarsening` in MGPressureSolver.solve)."""
    i, j, k = wp.tid()
    t = VT_INACTIVE
    for z in range(wp.max(0, 2 * k - 1), wp.min(tf.shape[2] - 1, 2 * k + 1) + 1):
        for y in range(wp.max(0, 2 * j - 1), wp.min(tf.shape[1] - 1, 2 * j + 1) + 1):
            for x in range(wp.max(0, 2 * i - 1), wp.min(tf.shape[0] - 1, 2 * i + 1) + 1):
                if tf[x, y, z] != VT_INACTIVE:
                    t = VT_ACTIVE
    tc[i, j, k] = t


def _coarse_types(t0: np.ndarray, sizes: list[tuple[int, int, int]]) -> list[np.ndarray]:
    """Run genCoarseGrid for every level on the CPU; returns int32 type grids."""
    cpu = wp.get_device("cpu")
    types = [t0]
    for size in sizes[1:]:
        n = int(np.prod(types[-1].shape)) + HEAP_K
        tf = wp.array(types[-1], dtype=wp.int32, device=cpu)
        tc = wp.empty(size, dtype=wp.int32, device=cpu)
        key, prv, nxt = (wp.full(n, -1, dtype=wp.int32, device=cpu) for _ in range(3))
        st = wp.array([0, -1], dtype=wp.int32, device=cpu)
        wp.launch(_coarsen_kernel, dim=1, inputs=[tf, tc, key, prv, nxt, st], device=cpu)
        types.append(tc.numpy())
    return types


def _level_sizes(shape) -> list[tuple[int, int, int]]:
    """GridMg constructor: halve ((n+2)/2) until all sides <= 5 or <= 1000 vertices."""
    sizes = [tuple(int(s) for s in shape)]
    while True:
        prev = sizes[-1]
        if max(prev) <= 5 or int(np.prod(prev)) <= 1000:
            break
        sizes.append(tuple((s + 2) // 2 for s in prev))
    return sizes


# Eigenvalues below this fraction of the largest count as zero on the coarsest level. Its matrix
# holds float32 Galerkin values, so a block that should be singular (closed domain or sealed pocket
# with no pinned cell) keeps an eigenvalue of 1e-11 to 3e-8 of the largest; inverting that stalls CG.
# Well-posed matrices measured at 128-cubed stay above 9e-5 (CODE-GAS-011).
COARSEST_RCOND = 1e-6


def _invert_coarsest(dense: np.ndarray) -> np.ndarray:
    """Inverse of the coarsest matrix, about 5 times cheaper than a pseudo-inverse (CODE-GAS-002).
    It is kept only when A @ inv maps three random vectors back to themselves to 1e-6 and the
    Frobenius condition number, an upper bound on the 2-norm one, is below 1 / COARSEST_RCOND, so
    the pseudo-inverse would discard nothing and the two agree to rounding. A singular matrix
    (closed domain without a pinned cell) fails that and takes the pseudo-inverse. The check is
    matrix-vector because a full A @ inv costs more than the inverse with the NumPy bundled in
    Blender."""
    if dense.size == 0:
        return dense.copy()
    try:
        with np.errstate(all="ignore"):
            inv = np.linalg.inv(dense)
            probe = np.random.default_rng(0).standard_normal((len(dense), 3))
            err = np.abs(dense @ (inv @ probe) - probe).max()
            cond = np.linalg.norm(dense) * np.linalg.norm(inv)
        if np.isfinite(err) and err < 1e-6 and cond < 1.0 / COARSEST_RCOND:
            return inv
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(dense, COARSEST_RCOND, True)  # positional: the keyword differs across NumPy


def _dense_coarsest(a: np.ndarray, t: np.ndarray):
    """Dense matrix of the coarsest level over its active vertices, inverted. _level_sizes keeps that
    level at 1000 vertices or fewer."""
    act = np.argwhere(t != 0)
    m = len(act)
    index = np.full(t.shape, -1, np.int64)
    index[tuple(act.T)] = np.arange(m)
    dense = np.zeros((m, m), np.float64)
    shape = np.array(t.shape)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
                nb = act + (dx, dy, dz)
                ok = np.all((nb >= 0) & (nb < shape), axis=1)
                rows = np.nonzero(ok)[0]
                cols = index[tuple(nb[rows].T)]
                keep = cols >= 0
                rows, cols = rows[keep], cols[keep]
                if s >= 13:
                    vals = a[tuple(act[rows].T) + (s - 13,)]
                else:
                    vals = a[tuple(nb[rows].T) + (13 - s,)]
                dense[rows, cols] = vals
    dense = 0.5 * (dense + dense.T)  # symmetric by construction; remove rounding noise
    return act.astype(np.int32), _invert_coarsest(dense).astype(np.float32)


def _dense_level0(a0: np.ndarray, ai: np.ndarray, aj: np.ndarray, ak: np.ndarray, t: np.ndarray):
    """Dense inverse for a grid so small that level 0 is also the coarsest (CODE-GAS-005). Puts the
    7-point matrix in the 14-entry storage _coef reads (index s - 13: +x = 1, +y = 3, +z = 9). Trivial
    rows keep the diagonal 1 and the V-cycle passes the unscaled residual, so they give x = b as the
    scaled equation does, and the inverse's conditioning check is not misled by a 1e-6 diagonal."""
    a = np.zeros(t.shape + (14,), np.float32)
    a[..., 0], a[..., 1], a[..., 3], a[..., 9] = a0, ai, aj, ak
    return _dense_coarsest(a, t)


# ---- solver ------------------------------------------------------------------------

class MGPressureSolver:
    """Matrix, multigrid hierarchy and CG work grids for one domain shape."""

    def __init__(self, shape, device):
        self.shape = tuple(int(s) for s in shape)
        self.device = device
        self.builds = 0  # how many times the matrix/hierarchy was (re)built
        self.build_ms = {}  # last build: matrix, coarsen, galerkin, coarsest, total (milliseconds)
        self.use_graph = device.is_cuda
        self._graphs = {}  # name -> (grid pointers, captured CUDA graph)
        self._key = None  # (pins a cell, fractions build key) of the current build
        self.frac = self.obvel = None  # fractions / obstacle velocity of the current solve
        self._dummy_vec = wp.zeros((1, 1, 1), dtype=wp.vec3, device=device)
        self._dummy_scalar = wp.zeros((1, 1, 1), dtype=wp.float32, device=device)
        self.phi = None  # ghost-fluid level set of the current build
        self.wts = None  # per-face weights (1/rho) of the current build, or None
        self._wkey = None  # (ptr, gen) of the weights in the current build
        self._t0_old = None  # level-0 types before a weights-only rebuild
        self.curv = None  # surface tension curvature (with phi) or None
        d, sh = device, self.shape
        f32 = wp.float32
        self.a0, self.ai, self.aj, self.ak = (wp.zeros(sh, dtype=f32, device=d) for _ in range(4))
        self.residual, self.search, self.r0 = (wp.zeros(sh, dtype=f32, device=d) for _ in range(3))
        self.snap = wp.zeros(sh, dtype=wp.uint8, device=d)
        self.changed = wp.zeros(1, dtype=wp.int32, device=d)
        self.sc = wp.zeros(2, dtype=wp.float64, device=d)
        self.sig = wp.zeros(1, dtype=f32, device=d)
        self.res = wp.zeros(1, dtype=f32, device=d)
        self.div = wp.zeros(1, dtype=f32, device=d)  # target divergence (volume correction)
        self._div_value = 0.0
        self.st = wp.zeros(1, dtype=f32, device=d)  # surface tension (Mantaflow surfTens)
        self._st_value = 0.0
        self.res_host = wp.zeros(1, dtype=f32, device="cpu", pinned=d.is_cuda)
        self.sizes = _level_sizes(sh)
        self.t = [wp.zeros(s, dtype=wp.int32, device=d) for s in self.sizes]
        self.x = [wp.zeros(s, dtype=f32, device=d) for s in self.sizes]  # x[0] is CG's tmp
        self.b = [None] + [wp.zeros(s, dtype=f32, device=d) for s in self.sizes[1:]]
        self.r = [self.r0] + [wp.zeros(s, dtype=f32, device=d) for s in self.sizes[1:-1]]
        self.a = [None] + [wp.zeros(s + (14,), dtype=f32, device=d) for s in self.sizes[1:]]
        self._dummy4 = wp.zeros((1, 1, 1, 14), dtype=f32, device=d)
        self.act = None
        self.minv = None
        self.n = int(np.prod(sh))
        self.padded = (self.n + BLOCK - 1) // BLOCK * BLOCK

    def nbytes(self) -> int:
        arrays = [self.a0, self.ai, self.aj, self.ak, self.residual, self.search, self.r0, self.snap,
                  *self.t, *self.x, *self.b[1:], *self.r[1:], *self.a[1:], self.act, self.minv, self._t0_old]
        return sum(a.capacity for a in arrays if a is not None)

    # ---- setup ----

    def _needs_build(self, flags, pin_key) -> bool:
        if self._key != pin_key:
            return True
        self.changed.zero_()
        wp.launch(_layout_kernel, dim=self.shape, inputs=[flags, self.snap, 0, self.changed],
                  device=self.device)
        return bool(self.changed.numpy()[0])

    def _pin_cell(self, fl: np.ndarray):
        """solvePressureSystem's choice of the cell to pin, or None."""
        if np.any(fl & int(Flags.EMPTY)):
            return None
        nx, ny, nz = fl.shape
        top = (nx // 2, ny - 1, nz // 2)
        for dy in (0, 1, 2):
            p = (top[0], top[1] - dy, top[2])
            if fl[p] & int(Flags.FLUID):
                return p
        inner = fl[1:-1, 1:-1, 1:-1] & int(Flags.FLUID)
        hits = np.argwhere(inner.transpose(2, 1, 0))  # FOR_IJK_BND order: k, j, i (i fastest)
        if len(hits) == 0:
            return None
        k, j, i = hits[0]
        return (int(i) + 1, int(j) + 1, int(k) + 1)

    def _coarsen(self, parallel: bool):
        """Fill the type grids self.t[1:] from self.t[0]."""
        if parallel:
            for l in range(1, len(self.sizes)):
                wp.launch(_coarsen_parallel_kernel, dim=self.sizes[l],
                          inputs=[self.t[l - 1], self.t[l]], device=self.device)
            return
        types = _coarse_types(self.t[0].numpy(), self.sizes)
        for l in range(1, len(self.sizes)):
            self.t[l].assign(types[l])

    def _build(self, flags, pin_needed: bool, parallel: bool, weights_only: bool = False):
        """weights_only: same flags and keys, new face weights. The pin cell stays and
        the coarse types are only rebuilt if the level-0 types changed."""
        d = self.device
        t_start = time.perf_counter()
        if weights_only:
            if self._t0_old is None:
                self._t0_old = wp.zeros(self.shape, dtype=wp.int32, device=d)
            wp.copy(self._t0_old, self.t[0])
        else:
            wp.launch(_layout_kernel, dim=self.shape, inputs=[flags, self.snap, 1, self.changed], device=d)
        frac = self._dummy_vec if self.frac is None else self.frac
        wts = self._dummy_vec if self.wts is None else self.wts
        wp.launch(_laplace_kernel, dim=self.shape, inputs=[flags, self.a0, self.ai, self.aj, self.ak,
                                                           frac, int(self.frac is not None),
                                                           wts, int(self.wts is not None)],
                  device=d)
        if self.phi is not None:
            wp.launch(_ghost_diag_kernel, dim=self.shape, inputs=[flags, self.phi, self.a0], device=d)
        if not weights_only:
            self.pin = self._pin_cell(flags.numpy()) if pin_needed else None
        if self.pin is not None:
            wp.launch(_pin_kernel, dim=1, inputs=[self.a0, self.ai, self.aj, self.ak,
                                                  wp.vec3i(*self.pin)], device=d)
        wp.launch(_types0_kernel, dim=self.shape, inputs=[self.a0, self.ai, self.aj, self.ak, self.t[0]],
                  device=d)
        wp.synchronize_device(d)
        t_matrix = time.perf_counter()
        recoarsen = True
        if weights_only:
            self.changed.zero_()
            wp.launch(_types_changed_kernel, dim=self.shape, inputs=[self.t[0], self._t0_old, self.changed],
                      device=d)
            recoarsen = bool(self.changed.numpy()[0])
        if recoarsen:
            self._coarsen(parallel)
        wp.synchronize_device(d)
        t_coarsen = time.perf_counter()
        for l in range(1, len(self.sizes)):
            wp.launch(_mg_galerkin_kernel, dim=self.sizes[l],
                      inputs=[self.t[l - 1], self._dummy4 if l == 1 else self.a[l - 1],
                              self.a0, self.ai, self.aj, self.ak, int(l == 1), self.t[l], self.a[l]],
                      device=d)
        last = len(self.sizes) - 1
        wp.synchronize_device(d)
        t_galerkin = time.perf_counter()
        if last == 0:
            act, minv = _dense_level0(self.a0.numpy(), self.ai.numpy(), self.aj.numpy(), self.ak.numpy(),
                                      self.t[0].numpy())
        else:
            act, minv = _dense_coarsest(self.a[last].numpy(), self.t[last].numpy())
        self.act = wp.array(act.reshape(-1, 3), dtype=wp.vec3i, device=d)
        self.minv = wp.array(minv, dtype=wp.float32, device=d)
        self.builds += 1
        t_end = time.perf_counter()
        self.build_ms = {name: (b - a) * 1000 for name, a, b in (
            ("matrix", t_start, t_matrix), ("coarsen", t_matrix, t_coarsen),
            ("galerkin", t_coarsen, t_galerkin), ("coarsest", t_galerkin, t_end),
            ("total", t_start, t_end))}

    # ---- per solve ----

    def vcycle(self):
        """One V-cycle on A x[0] = residual (GridMg::doVCycle from x = 0),
        1 pre- and 1 post-smoothing step."""
        d = self.device
        top = len(self.sizes) - 1
        lvl0 = [self.t[0], self.a0, self.ai, self.aj, self.ak, self.residual]
        self.x[0].zero_()
        for l in range(top):
            if l == 0:
                for c in (0, 1):
                    wp.launch(_mg_smooth0_kernel, dim=self.shape, inputs=lvl0 + [self.x[0], c], device=d)
                wp.launch(_mg_residual0_kernel, dim=self.shape, inputs=lvl0 + [self.x[0], self.r[0]],
                          device=d)
            else:
                self._smooth(l, False)
                wp.launch(_mg_residual_kernel, dim=self.sizes[l],
                          inputs=[self.t[l], self.a[l], self.b[l], self.x[l], self.r[l]], device=d)
            wp.launch(_mg_restrict_kernel, dim=self.sizes[l + 1],
                      inputs=[self.t[l], self.r[l], self.t[l + 1], self.b[l + 1], self.x[l + 1]], device=d)
        wp.launch(_mg_coarsest_kernel, dim=self.act.shape[0],
                  inputs=[self.act, self.minv, self.b[top] if top else self.residual, self.x[top]], device=d)
        for l in range(top - 1, -1, -1):
            wp.launch(_mg_interp_add_kernel, dim=self.sizes[l],
                      inputs=[self.t[l], self.x[l], self.t[l + 1], self.x[l + 1]], device=d)
            if l == 0:
                for c in (1, 0):
                    wp.launch(_mg_smooth0_kernel, dim=self.shape, inputs=lvl0 + [self.x[0], c], device=d)
            else:
                self._smooth(l, True)

    def _smooth(self, l, reverse):
        blocks = tuple((s + 1) // 2 for s in self.sizes[l])
        offs = [wp.vec3i(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]
        for off in (offs[::-1] if reverse else offs):
            wp.launch(_mg_smooth_kernel, dim=blocks,
                      inputs=[self.t[l], self.a[l], self.b[l], self.x[l], off], device=self.device)

    def _iteration(self, flags):
        """One GridCg::iterate, minus the host-side convergence test. Running
        the V-cycle and search update after convergence leaves pressure alone."""
        d, sh, tmp = self.device, self.shape, self.x[0]
        wp.launch(_zero_scalars_kernel, dim=1, inputs=[self.sc, self.res], device=d)
        wp.launch(_apply_matrix_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[flags, self.a0, self.ai, self.aj, self.ak, self.search, tmp, self.sc], device=d)
        wp.launch(_cg_update_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[self.p, self.residual, self.search, tmp, self.sc, self.sig, self.res], device=d)
        wp.copy(self.res_host, self.res)
        self.vcycle()
        wp.launch(_dot_kernel, dim=self.padded, block_dim=BLOCK, inputs=[tmp, self.residual, self.sc],
                  device=d)
        wp.launch(_search_update_kernel, dim=sh, inputs=[tmp, self.search, self.sc, self.sig], device=d)
        wp.launch(_set_sigma_kernel, dim=1, inputs=[self.sc, self.sig], device=d)

    def _prologue(self, flags, vel, pressure, pin):
        """rhs, then GridCg::doInit: dst = 0, residual = rhs, tmp = M^-1 residual,
        search = tmp, sigma = dot(tmp, residual)."""
        d = self.device
        frac = self._dummy_vec if self.frac is None else self.frac
        obvel = self._dummy_vec if self.obvel is None else self.obvel
        divg = self._dummy_scalar if self.divg is None else self.divg
        phi = self._dummy_scalar if self.phi is None else self.phi
        curv = self._dummy_scalar if self.curv is None else self.curv
        wp.launch(_rhs_kernel, dim=self.shape,
                  inputs=[flags, vel, self.r0, pin, self.div, frac, obvel,
                          int(self.frac is not None), int(self.obvel is not None),
                          divg, int(self.divg is not None), phi, curv, self.st,
                          int(self.curv is not None)], device=d)
        wp.launch(_cg_init_kernel, dim=self.shape, inputs=[self.r0, pressure, self.residual], device=d)
        self.vcycle()
        wp.launch(_search_init_kernel, dim=self.shape, inputs=[self.x[0], self.search], device=d)
        wp.launch(_zero_scalars_kernel, dim=1, inputs=[self.sc, self.res], device=d)
        wp.launch(_dot_kernel, dim=self.padded, block_dim=BLOCK, inputs=[self.x[0], self.residual, self.sc],
                  device=d)
        wp.launch(_set_sigma_kernel, dim=1, inputs=[self.sc, self.sig], device=d)

    def _run(self, name, fn, key):
        """Run fn() directly, or as a CUDA graph captured once per set of grids."""
        if not self.use_graph:
            fn()
            return
        graph = self._graphs.get(name)
        if graph is None or graph[0] != key:
            with wp.ScopedCapture(device=self.device) as cap:
                fn()
            graph = self._graphs[name] = (key, cap.graph)
        wp.capture_launch(graph[1])

    def solve(self, flags, vel, pressure, accuracy, zero_pressure_fixing,
              divergence=0.0, fractions=None, obvel=None, fractions_gen=0,
              phi=None, phi_gen=0, divergence_grid=None, curv=None,
              surface_tension=0.0, parallel_coarsening=False, face_weights=None,
              weights_gen=0) -> PressureStats:
        """fractions: face fractions (Mantaflow's MACGrid) or None; the matrix is
        rebuilt when the flags, `fractions_gen` or `phi_gen` change. obvel only counts
        with fractions. phi: ghost-fluid free surface (matrix diagonal and correction).
        divergence_grid: per-cell target divergence added to the rhs, or None.
        curv (with phi): curvature for the surface tension jump `surface_tension` * curv.
        parallel_coarsening: pick the coarse vertices on the GPU instead of Mantaflow's
        serial CPU pass (High Quality only; Classic stays reproducible).
        face_weights (not in Mantaflow; not with phi): vec3 grid, weight w of the lower
        face per axis; solves div(w grad p) = div u and corrects u -= w grad p (the
        rhs stays the plain or fraction-weighted divergence). The matrix is rebuilt
        when `weights_gen` changes."""
        if face_weights is not None and phi is not None:
            raise ValueError("face_weights cannot be combined with the ghost-fluid phi")
        d = self.device
        if divergence != self._div_value:
            self.div.fill_(divergence)
            self._div_value = divergence
        self.divg = divergence_grid
        self.frac = fractions
        self.obvel = obvel if fractions is not None else None
        self.phi = phi
        self.wts = face_weights
        self.curv = curv if phi is not None and surface_tension != 0.0 else None
        if self.curv is not None and surface_tension != self._st_value:
            self.st.fill_(surface_tension)
            self._st_value = surface_tension
        pin_needed = bool(zero_pressure_fixing or accuracy < 1e-7)
        build_key = (pin_needed, bool(parallel_coarsening), None if fractions is None else (fractions.ptr, fractions_gen),
                     None if phi is None else (phi.ptr, phi_gen), face_weights is not None)
        wkey = None if face_weights is None else (face_weights.ptr, weights_gen)
        if self._needs_build(flags, build_key):
            self._build(flags, pin_needed, bool(parallel_coarsening))
            self._key, self._wkey = build_key, wkey
            self._graphs = {}
        elif wkey != self._wkey:
            self._build(flags, pin_needed, bool(parallel_coarsening), weights_only=True)
            self._wkey = wkey
            self._graphs = {}
        pin = wp.vec3i(*self.pin) if self.pin is not None else wp.vec3i(-1, -1, -1)
        self.p = pressure
        key = (flags.ptr, vel.ptr, pressure.ptr, tuple(pin), self.frac is not None and self.frac.ptr,
               self.obvel is not None and self.obvel.ptr, self.divg is not None and self.divg.ptr,
               self.curv is not None and (self.curv.ptr, self.phi.ptr))
        self._run("prologue", lambda: self._prologue(flags, vel, pressure, pin), key)
        resnorm = 0.0
        it = 0
        while it < MG_MAX_ITER:
            it += 1
            self._run("iteration", lambda: self._iteration(flags), key)
            wp.synchronize_device(d)
            resnorm = float(self.res_host.numpy()[0])
            if resnorm < accuracy:
                break
            if not resnorm < 1e35:
                raise RuntimeError("pressure solve failed: the velocity field contains NaN or infinity, "
                                   "or the solve diverged")

        wp.launch(_correct_vel_kernel, dim=self.shape,
                  inputs=[flags, vel, pressure, self._dummy_scalar if phi is None else phi,
                          int(phi is not None), self._dummy_scalar if self.curv is None else self.curv,
                          self.st, int(self.curv is not None),
                          self._dummy_vec if face_weights is None else face_weights,
                          int(face_weights is not None)], device=d)
        if phi is not None:
            wp.launch(_replace_clamped_kernel, dim=self.shape, inputs=[flags, vel, phi], device=d)
        return PressureStats(iterations=it, residual=resnorm)
