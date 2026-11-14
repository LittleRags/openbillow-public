# SPDX-License-Identifier: GPL-3.0-or-later
"""Liquid kernels: particle sorting, particle level set, level-set extrapolation,
particle/grid velocity transfer, particle resampling, sampling, advection and
push-out, liquid gravity.

Ported from Mantaflow (plugin/flip.cpp `gridParticleIndex`,
`unionParticleLevelset`, `mapPartsToMAC`, `combineGridVel`, `flipVelocityUpdate`,
`adjustNumber`, `sampleLevelsetWithParticles`, `pushOutofObs`; plugin/apic.cpp;
plugin/extforces.cpp `addGravity`; particle.h + helper/util/integrator.h
`advectInGrid`; fastmarch.cpp `extrapolateLsSimple`, `extrapolateMACSimple`,
`extrapolateMACFromWeight`; grid.cpp `FlagGrid::updateFromLevelset`;
levelset.cpp `floodFill`, `subtract`; plugin/initplugins.cpp `setObstacleFlags`;
plugin/extforces.cpp `resetOutflow`; helper/util/interpol.h). Self-contained;
`WarpBackend` inherits `LiquidMixin`.

Particle-to-grid sums gather each face's particles through the stable index, so
they do not depend on thread timing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .base import Flags, ParticleFlags
from .warp_common import (axis, border, cell_of, f4, mac_comp, mac_interp, nb6, sample,
                          sample_vec, shift)

wp.set_module_options({"fuse_fp": False, "enable_backward": False})

FLUID = wp.constant(int(Flags.FLUID))
EMPTY = wp.constant(int(Flags.EMPTY))
OBSTACLE = wp.constant(int(Flags.OBSTACLE))
INFLOW = wp.constant(int(Flags.INFLOW))
OUTFLOW = wp.constant(int(Flags.OUTFLOW))

PDELETE = wp.constant(int(ParticleFlags.DELETE))
PNEW = wp.constant(int(ParticleFlags.NEW))
VECTOR_EPSILON = 1.0e-6
# ParticleBase::globalSeed: adjustNumber seeds RandomStream(parts.getSeed()) afresh each call.
PARTICLE_SEED = 9832
SEED_STREAM_KEEP = 3 << 22  # floats (50 MB) the cache may keep, so reseeding never regenerates it
# FastMarch::InvalidTime(), -1000 (reference/mantaflow/fastmarch.cpp:90, levelset.cpp:387).
INVALID_TIME = -1000.0


@dataclass
class WarpParticles:
    pos: wp.array      # (capacity,) vec3, cell units
    vel: wp.array      # (capacity,) vec3
    flag: wp.array     # (capacity,) int32, ParticleFlags
    n: int = 0         # particles in use (host copy)
    cx: wp.array | None = None  # APIC affine rows (capacity,) vec3, or None without APIC
    cy: wp.array | None = None
    cz: wp.array | None = None
    life: wp.array | None = None  # secondary particles: lifetime (capacity,) float32, or None
    scratch: dict = field(default_factory=dict, repr=False)  # work arrays, see _slot_scratch

    @property
    def capacity(self) -> int:
        return self.pos.shape[0]


@dataclass
class WarpParticleIndex:
    start: wp.array    # (nx,ny,nz) int32: first slot of each cell in `sorted`
    sorted: wp.array   # (capacity,) int32: particle indices, grouped by cell
    total: wp.array    # (1,) int32: number of particles in `sorted`
    shape: tuple


@wp.kernel
def _count_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32), n: int,
                  scale: float, clamp: int, shape: wp.vec3i, counter: wp.array3d(dtype=wp.int32),
                  total: wp.array(dtype=wp.int32), keys: wp.array(dtype=wp.int32),
                  values: wp.array(dtype=wp.int32)):
    # Also writes the sort pair of every slot: the flat cell index, or the number of cells for
    # a particle that is deleted or outside, which sorts it behind all the indexed ones.
    t = wp.tid()
    values[t] = t
    keys[t] = shape[0] * shape[1] * shape[2]
    if t >= n or (flag[t] & PDELETE) != 0:
        return
    p = pos[t] * scale
    c = cell_of(p, shape)
    if c[0] < 0:
        if clamp == 0:
            return
        for a in range(3):
            c[a] = wp.clamp(int(wp.floor(p[a])), 0, shape[a] - 1)
    keys[t] = (c[0] * shape[1] + c[1]) * shape[2] + c[2]
    wp.atomic_add(counter, c[0], c[1], c[2], 1)
    wp.atomic_add(total, 0, 1)


@wp.kernel
def _union_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                  total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                  scale: float, phi: wp.array3d(dtype=wp.float32), radius: float, r: int):
    # ComputeUnionLevelsetPindex; the cell range ends at the next cell in
    # linear order (the last cell ends at the particle total).
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    if i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1:
        phi[i, j, k] = 0.5  # setBound(0.5, 0)
        return
    g = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    phiv = radius * 1.0
    for zj in range(wp.max(0, k - r), wp.min(k + r, nz - 1) + 1):
        for yj in range(wp.max(0, j - r), wp.min(j + r, ny - 1) + 1):
            for xj in range(wp.max(0, i - r), wp.min(i + r, nx - 1) + 1):
                p0 = start[xj, yj, zj]
                lin = (xj * ny + yj) * nz + zj
                p1 = total[0]
                if lin + 1 < nx * ny * nz:
                    p1 = start[(lin + 1) // (ny * nz), ((lin + 1) // nz) % ny, (lin + 1) % nz]
                for q in range(p0, p1):
                    d = g - pos[sorted_idx[q]] * scale
                    phiv = wp.min(phiv, wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - radius)
    phi[i, j, k] = phiv


@wp.kernel
def _averaged_ls_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                        total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                        scale: float, phi: wp.array3d(dtype=wp.float32),
                        pacc_out: wp.array3d(dtype=wp.vec3), racc_out: wp.array3d(dtype=wp.float32),
                        radius: float, r: int):
    # ComputeAveragedLevelsetWeight (all cells, weight max(0, 1 - d^2/(4 r^2))), saving the
    # weighted particle centre and radius for correctLevelset (zero where no particle).
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    g = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    sinv = 1.0 / (4.0 * radius * radius)
    wacc = float(0.0)
    racc = float(0.0)
    pacc = wp.vec3(0.0)
    for zj in range(wp.max(0, k - r), wp.min(k + r, nz - 1) + 1):
        for yj in range(wp.max(0, j - r), wp.min(j + r, ny - 1) + 1):
            for xj in range(wp.max(0, i - r), wp.min(i + r, nx - 1) + 1):
                p0 = start[xj, yj, zj]
                lin = (xj * ny + yj) * nz + zj
                p1 = total[0]
                if lin + 1 < nx * ny * nz:
                    p1 = start[(lin + 1) // (ny * nz), ((lin + 1) // nz) % ny, (lin + 1) % nz]
                for q in range(p0, p1):
                    pp = pos[sorted_idx[q]] * scale
                    w = wp.max(0.0, 1.0 - wp.length_sq(g - pp) * sinv)
                    wacc += w
                    racc += radius * w
                    pacc += pp * w
    phiv = radius
    if wacc > VECTOR_EPSILON:
        racc = racc / wacc
        pacc = pacc / wacc
        phiv = wp.length(g - pacc) - racc
    else:
        racc = 0.0
        pacc = wp.vec3(0.0)
    phi[i, j, k] = phiv
    pacc_out[i, j, k] = pacc
    racc_out[i, j, k] = racc


@wp.func
def _cbrt(x: float):
    return wp.sign(x) * wp.pow(wp.abs(x), 1.0 / 3.0)


@wp.func
def _max_eigenvalue(m: wp.mat33):
    # Largest real part of the eigenvalues: roots of l^3 + a l^2 + b l + c (Cardano / trig).
    a = -(m[0, 0] + m[1, 1] + m[2, 2])
    b = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0] + m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
         + m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
    c = -wp.determinant(m)
    p = b - a * a / 3.0
    q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
    h = q * q / 4.0 + p * p * p / 27.0
    t = float(0.0)
    if h > 0.0:
        sh = wp.sqrt(h)
        u = _cbrt(-0.5 * q + sh) + _cbrt(-0.5 * q - sh)
        t = wp.max(u, -0.5 * u)  # one real root u, complex pair with real part -u/2
    elif p < 0.0:
        rr = wp.sqrt(-p / 3.0)
        t = 2.0 * rr * wp.cos(wp.acos(wp.clamp(-0.5 * q / (rr * rr * rr), -1.0, 1.0)) / 3.0)
    return t - a / 3.0


@wp.kernel
def _correct_ls_kernel(phi: wp.array3d(dtype=wp.float32), pacc: wp.array3d(dtype=wp.vec3),
                       racc: wp.array3d(dtype=wp.float32), radius: float, t_low: float,
                       t_high: float):
    # correctLevelset (Zhu & Bridson concavity fix), interior cells.
    i, j, k = wp.tid()
    if border(i, j, k, phi.shape[0], phi.shape[1], phi.shape[2]) or racc[i, j, k] <= VECTOR_EPSILON:
        return
    dx = 0.5 * (pacc[i + 1, j, k] - pacc[i - 1, j, k])
    dy = 0.5 * (pacc[i, j + 1, k] - pacc[i, j - 1, k])
    dz = 0.5 * (pacc[i, j, k + 1] - pacc[i, j, k - 1])
    jac = wp.mat33(dx[0], dy[0], dz[0], dx[1], dy[1], dz[1], dx[2], dy[2], dz[2])
    ev = _max_eigenvalue(jac)
    corr = float(1.0)
    if ev >= t_low:
        t = (t_high - ev) / (t_high - t_low)
        corr = t * t * t - 3.0 * t * t + 3.0 * t
    corr = wp.clamp(corr, 0.0, 1.0)
    g = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    phi[i, j, k] = wp.min(wp.length(g - pacc[i, j, k]) - racc[i, j, k] * corr, radius)


@wp.kernel
def _smooth_ls_kernel(me: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.float32),
                      neg: int):
    # knSmoothGrid (neg=0: tmp = 7-point mean) / knSmoothGridNeg (neg=1: take the mean only
    # where it is below tmp, else me). Interior cells.
    i, j, k = wp.tid()
    if border(i, j, k, me.shape[0], me.shape[1], me.shape[2]):
        return
    val = (me[i, j, k] + me[i + 1, j, k] + me[i - 1, j, k] + me[i, j + 1, k] + me[i, j - 1, k]
           + me[i, j, k + 1] + me[i, j, k - 1]) * (1.0 / 7.0)
    if neg == 0 or val < tmp[i, j, k]:
        tmp[i, j, k] = val
    else:
        tmp[i, j, k] = me[i, j, k]


@wp.kernel
def _aniso_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                  total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                  flag: wp.array(dtype=wp.int32), n: int, h: float, lam: float, kr: float,
                  nmin: int, scale: float, inv_max: float, cen: wp.array(dtype=wp.vec3),
                  amat: wp.array(dtype=wp.mat33)):
    # Yu & Turk 2013, per particle (simulation-cell index): weighted neighbour mean
    # (w = 1 - (d/h)^3), smoothed centre (1-lam) x + lam mean, covariance -> ellipsoid axes
    # a_k ~ sqrt(eigenvalue), shortest >= longest/kr, normalised to a1 a2 a3 = 1 (same
    # volume as the sphere). amat maps an offset (mesh cells) to the unit-sphere space.
    # inv_max = radius / min_half (0 = off): sheet-like axes keep a half-thickness of at
    # least min_half mesh cells, gated per axis (CODE-LIQ-016).
    t = wp.tid()
    if t >= n or (flag[t] & PDELETE) != 0:
        return
    nx = start.shape[0]
    ny = start.shape[1]
    nz = start.shape[2]
    x = pos[t]
    r = int(wp.ceil(h))
    ci = int(wp.floor(x[0]))
    cj = int(wp.floor(x[1]))
    ck = int(wp.floor(x[2]))
    wsum = float(0.0)
    xw = wp.vec3(0.0)
    cnt = int(0)
    for zj in range(wp.max(0, ck - r), wp.min(ck + r, nz - 1) + 1):
        for yj in range(wp.max(0, cj - r), wp.min(cj + r, ny - 1) + 1):
            for xj in range(wp.max(0, ci - r), wp.min(ci + r, nx - 1) + 1):
                p0 = start[xj, yj, zj]
                lin = (xj * ny + yj) * nz + zj
                p1 = total[0]
                if lin + 1 < nx * ny * nz:
                    p1 = start[(lin + 1) // (ny * nz), ((lin + 1) // nz) % ny, (lin + 1) % nz]
                for q in range(p0, p1):
                    d = wp.length(pos[sorted_idx[q]] - x) / h
                    if d < 1.0:
                        w = 1.0 - d * d * d
                        wsum += w
                        xw += pos[sorted_idx[q]] * w
                        cnt += 1
    xw = xw / wsum
    cen[t] = ((1.0 - lam) * x + lam * xw) * scale
    if cnt < nmin:
        amat[t] = wp.identity(n=3, dtype=float)
        return
    cov = wp.mat33(0.0)
    for zj in range(wp.max(0, ck - r), wp.min(ck + r, nz - 1) + 1):
        for yj in range(wp.max(0, cj - r), wp.min(cj + r, ny - 1) + 1):
            for xj in range(wp.max(0, ci - r), wp.min(ci + r, nx - 1) + 1):
                p0 = start[xj, yj, zj]
                lin = (xj * ny + yj) * nz + zj
                p1 = total[0]
                if lin + 1 < nx * ny * nz:
                    p1 = start[(lin + 1) // (ny * nz), ((lin + 1) // nz) % ny, (lin + 1) % nz]
                for q in range(p0, p1):
                    p = pos[sorted_idx[q]]
                    d = wp.length(p - x) / h
                    if d < 1.0:
                        e = p - xw
                        cov += wp.outer(e, e) * (1.0 - d * d * d)
    cov = cov / wsum
    res = wp.eig3(cov)
    qm = res[0]
    ev = res[1]
    a0 = wp.sqrt(wp.max(ev[0], 0.0))
    a1 = wp.sqrt(wp.max(ev[1], 0.0))
    a2 = wp.sqrt(wp.max(ev[2], 0.0))
    amax = wp.max(a0, wp.max(a1, a2))
    if amax <= VECTOR_EPSILON:
        amat[t] = wp.identity(n=3, dtype=float)
        return
    # gate per axis: 1 for a flat sheet (axis ~ 0), 0 where the kr limit stops binding
    w0 = wp.clamp(1.0 - kr * a0 / amax, 0.0, 1.0)
    w1 = wp.clamp(1.0 - kr * a1 / amax, 0.0, 1.0)
    w2 = wp.clamp(1.0 - kr * a2 / amax, 0.0, 1.0)
    a0 = wp.max(a0, amax / kr)
    a1 = wp.max(a1, amax / kr)
    a2 = wp.max(a2, amax / kr)
    g = _cbrt(a0 * a1 * a2)
    i0 = g / a0
    i1 = g / a1
    i2 = g / a2
    if inv_max > 0.0:
        i0 = wp.min(i0, inv_max / wp.max(w0, 1.0e-6))
        i1 = wp.min(i1, inv_max / wp.max(w1, 1.0e-6))
        i2 = wp.min(i2, inv_max / wp.max(w2, 1.0e-6))
    inv = wp.diag(wp.vec3(i0, i1, i2))
    amat[t] = qm * inv * wp.transpose(qm)


@wp.kernel
def _aniso_splat_kernel(cen: wp.array(dtype=wp.vec3), amat: wp.array(dtype=wp.mat33),
                        flag: wp.array(dtype=wp.int32), n: int, radius: float,
                        phi: wp.array3d(dtype=wp.float32)):
    # phi = min over particles of |A (g - c)| - radius: union of the ellipsoids, each with the
    # volume of a sphere of `radius`. Written where phi < radius, that is |A d| < 2 radius.
    t = wp.tid()
    if t >= n or (flag[t] & PDELETE) != 0:
        return
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    c = cen[t]
    a = amat[t]
    # Bounding box of |A d| < 2 radius: half-extent i = 2 radius |row i of A^-1|.
    ai = wp.inverse(a)
    reach = 2.0 * radius * wp.vec3(wp.length(ai[0]), wp.length(ai[1]), wp.length(ai[2]))
    lo = c - reach - wp.vec3(0.5)
    hi = c + reach - wp.vec3(0.5)
    for k in range(wp.max(1, int(wp.ceil(lo[2]))), wp.min(nz - 2, int(wp.floor(hi[2]))) + 1):
        for j in range(wp.max(1, int(wp.ceil(lo[1]))), wp.min(ny - 2, int(wp.floor(hi[1]))) + 1):
            for i in range(wp.max(1, int(wp.ceil(lo[0]))), wp.min(nx - 2, int(wp.floor(hi[0]))) + 1):
                g = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
                v = wp.length(a * (g - c)) - radius
                if v < radius:
                    wp.atomic_min(phi, i, j, k, v)


@wp.kernel
def _set_border_kernel(phi: wp.array3d(dtype=wp.float32), value: float):
    i, j, k = wp.tid()
    if border(i, j, k, phi.shape[0], phi.shape[1], phi.shape[2]):
        phi[i, j, k] = value


@wp.kernel
def _ls_mark_kernel(phi: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.int32), inside: int):
    # knMarkSkipCells: interior only, everything else starts at 0.
    i, j, k = wp.tid()
    v = int(0)
    if not border(i, j, k, phi.shape[0], phi.shape[1], phi.shape[2]):
        if inside == 0 and phi[i, j, k] < 0.0:
            v = 1
        if inside != 0 and phi[i, j, k] > 0.0:
            v = 1
    tmp[i, j, k] = v


@wp.kernel
def _ls_first_layer_kernel(tmp: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    if border(i, j, k, tmp.shape[0], tmp.shape[1], tmp.shape[2]) or tmp[i, j, k] != 0:
        return
    p = wp.vec3i(i, j, k)
    for n in range(6):
        q = p + nb6(n)
        if tmp[q[0], q[1], q[2]] == 1:
            tmp[i, j, k] = 2
            return


@wp.kernel
def _ls_extrap_kernel(val: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.int32),
                      d: int, direction: float):
    # One layer: written cells get d + 1 and are only read at tmp == d.
    i, j, k = wp.tid()
    if border(i, j, k, val.shape[0], val.shape[1], val.shape[2]) or tmp[i, j, k] != 0:
        return
    p = wp.vec3i(i, j, k)
    nbs = int(0)
    s = float(0.0)
    for n in range(6):
        q = p + nb6(n)
        if tmp[q[0], q[1], q[2]] == d:
            s += val[q[0], q[1], q[2]]
            nbs += 1
    if nbs > 0:
        tmp[i, j, k] = d + 1
        val[i, j, k] = s / float(nbs) + direction


@wp.kernel
def _ls_remaining_kernel(val: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.int32),
                         value: float):
    i, j, k = wp.tid()
    if border(i, j, k, val.shape[0], val.shape[1], val.shape[2]) or tmp[i, j, k] != 0:
        return
    val[i, j, k] = value


# Thread order must not change a total, and float addition depends on order, so the volume
# sums add integers: units of 2^-24 of a cell. Exact for a float32 of 0.5 or more, about 1e-8
# of the total otherwise; an int64 holds totals up to 2^39.
FIXED_ONE = 16777216.0


@wp.func
def _fixed(v: float) -> wp.int64:
    return wp.int64(wp.float64(v) * wp.float64(FIXED_ONE))


@wp.kernel
def _ls_volume_kernel(phi: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32),
                      out: wp.array(dtype=wp.int64), w: int):
    i, j, k = wp.tid()
    # The border layers hold a Neumann copy of phi, obstacles an extrapolated one: no liquid.
    if (i < w or j < w or k < w or i >= phi.shape[0] - w or j >= phi.shape[1] - w
            or k >= phi.shape[2] - w or (flags[i, j, k] & OBSTACLE) != 0):
        return
    f = wp.clamp(0.5 - phi[i, j, k], 0.0, 1.0)
    if f > 0.0:
        wp.atomic_add(out, 0, _fixed(f))


@wp.func
def _src_face(flags: wp.array3d(dtype=wp.int32), phi_in: wp.array3d(dtype=wp.float32),
              i: int, j: int, k: int, u: float) -> float:
    """Flux u leaving a source cell through a face whose far cell (i, j, k) is no source;
    only the outward part counts (backflow into a source is air or earlier jet)."""
    if phi_in[i, j, k] < 0.0 or (flags[i, j, k] & OBSTACLE) != 0:
        return 0.0
    return wp.max(u, 0.0)


@wp.kernel
def _source_outflux_kernel(flags: wp.array3d(dtype=wp.int32), phi_in: wp.array3d(dtype=wp.float32),
                           vel: wp.array3d(dtype=wp.vec3), invel: wp.array3d(dtype=wp.vec3),
                           use_invel: int, out: wp.array(dtype=wp.int64)):
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    if i < 1 or j < 1 or k < 1 or i > nx - 2 or j > ny - 2 or k > nz - 2:
        return
    if phi_in[i, j, k] >= 0.0 or (flags[i, j, k] & FLUID) == 0:
        return
    # Where the flow prescribes a velocity, use it (the grid velocity next to a source is
    # pressure-projected and can even point back into it); elsewhere the MAC face values.
    c = invel[i, j, k]
    if use_invel != 0 and (c[0] != 0.0 or c[1] != 0.0 or c[2] != 0.0):
        s = (_src_face(flags, phi_in, i - 1, j, k, -c[0]) + _src_face(flags, phi_in, i + 1, j, k, c[0])
             + _src_face(flags, phi_in, i, j - 1, k, -c[1]) + _src_face(flags, phi_in, i, j + 1, k, c[1])
             + _src_face(flags, phi_in, i, j, k - 1, -c[2]) + _src_face(flags, phi_in, i, j, k + 1, c[2]))
    else:
        v = vel[i, j, k]
        s = (_src_face(flags, phi_in, i - 1, j, k, -v[0]) + _src_face(flags, phi_in, i + 1, j, k, vel[i + 1, j, k][0])
             + _src_face(flags, phi_in, i, j - 1, k, -v[1]) + _src_face(flags, phi_in, i, j + 1, k, vel[i, j + 1, k][1])
             + _src_face(flags, phi_in, i, j, k - 1, -v[2]) + _src_face(flags, phi_in, i, j, k + 1, vel[i, j, k + 1][2]))
    if s != 0.0:
        wp.atomic_add(out, 0, _fixed(s))


@wp.kernel
def _flags_from_ls_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    f = flags[i, j, k]
    if (f & (OBSTACLE | OUTFLOW)) != 0:
        return
    v = phi[i, j, k]
    if v <= INVALID_TIME:
        return
    f = f & ~(FLUID | EMPTY)
    flags[i, j, k] = f | wp.where(v <= 0.0, FLUID, EMPTY)

# ---- particle/grid transfer (step 20) -----------------------------------------

@wp.func
def _cell_end(start: wp.array3d(dtype=wp.int32), total: wp.array(dtype=wp.int32),
              nx: int, ny: int, nz: int, i: int, j: int, k: int):
    # One past the last index slot of a cell: the start of the next cell in key order.
    if k + 1 < nz:
        return start[i, j, k + 1]
    if j + 1 < ny:
        return start[i, j + 1, 0]
    if i + 1 < nx:
        return start[i + 1, 0, 0]
    return total[0]


@wp.func
def _axis_weight(x: float, n: int, a: int):
    # Weight of grid sample a along one axis (zero unless it is one of the position's two).
    i, s = axis(x, n)
    if a == i:
        return 1.0 - s
    if a == i + 1:
        return s
    return 0.0


@wp.kernel
def _p2g_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                pvel: wp.array(dtype=wp.vec3), vel: wp.array4d(dtype=wp.float32),
                weight: wp.array4d(dtype=wp.float32)):
    # knMapLinearVec3ToMACGrid -> setInterpolMAC; weight per corner = wy * (wx * wz). Each face
    # gathers its particles through the index, so the order of its sum is the same on every run.
    i, j, k = wp.tid()
    nx = vel.shape[0]
    ny = vel.shape[1]
    nz = vel.shape[2]
    for c in range(3):
        # A face reaches the two cells beside it along c, and three across the other axes.
        x0 = wp.max(i - 1, 0)
        x1 = wp.min(wp.where(c == 0, i, i + 1), nx - 1)
        y0 = wp.max(j - 1, 0)
        y1 = wp.min(wp.where(c == 1, j, j + 1), ny - 1)
        z0 = wp.max(k - 1, 0)
        z1 = wp.min(wp.where(c == 2, k, k + 1), nz - 1)
        wsum = float(0.0)
        vsum = float(0.0)
        for xj in range(x0, x1 + 1):
            for yj in range(y0, y1 + 1):
                for q in range(start[xj, yj, z0], _cell_end(start, total, nx, ny, nz, xj, yj, z1)):
                    t = sorted_idx[q]
                    p = pos[t]
                    w = (_axis_weight(shift(p, c, 1), ny, j)
                         * (_axis_weight(shift(p, c, 0), nx, i) * _axis_weight(shift(p, c, 2), nz, k)))
                    wsum = wsum + w
                    vsum = vsum + w * pvel[t][c]
        weight[i, j, k, c] = wsum
        vel[i, j, k, c] = vsum


@wp.kernel
def _stomp_divide_kernel(vel: wp.array3d(dtype=wp.vec3), weight: wp.array3d(dtype=wp.vec3),
                         old: wp.array3d(dtype=wp.vec3), eps: float):
    # weight.stomp(eps); vel.safeDivide(weight); velOld.copyFrom(vel)
    i, j, k = wp.tid()
    w = weight[i, j, k]
    v = vel[i, j, k]
    for c in range(3):
        if w[c] < eps:
            w[c] = 0.0
        if w[c] != 0.0:
            v[c] = v[c] / w[c]
        else:
            v[c] = 0.0
    weight[i, j, k] = w
    vel[i, j, k] = v
    old[i, j, k] = v


@wp.func
def _face_index(nx: int, ny: int, nz: int, fc: wp.vec3i, cc: wp.vec3i, comp: int, off: wp.vec3i):
    # apic.cpp indexU/V/WFace + indexOffset on Mantaflow's linear index (x fastest);
    # an index past a row wraps into the next one, as in Mantaflow. -1 when out of range.
    x = wp.where(comp == 0, fc[0], cc[0])
    y = wp.where(comp == 1, fc[1], cc[1])
    z = wp.where(comp == 2, fc[2], cc[2])
    lin = x + nx * (y + ny * z)
    n = nx * ny * nz
    if lin < 0 or lin >= n:
        return -1
    lin = lin + off[0] + nx * (off[1] + ny * off[2])
    if lin >= n:
        return -1
    return lin


@wp.func
def _trunc3(p: wp.vec3):
    return wp.vec3i(int(p[0]), int(p[1]), int(p[2]))


@wp.func
def _apic_base(fc: wp.vec3i, cc: wp.vec3i, comp: int):
    return wp.vec3i(wp.where(comp == 0, fc[0], cc[0]), wp.where(comp == 1, fc[1], cc[1]),
                    wp.where(comp == 2, fc[2], cc[2]))


@wp.func
def _apic_share(p: wp.vec3, v: float, aff: wp.vec3, fc: wp.vec3i, cc: wp.vec3i, comp: int,
                di: int, dj: int, dk: int):
    # One corner of a particle's splat on component comp: mass, then the two velocity terms.
    wsel = wp.vec3()
    gpos = wp.vec3()
    for d in range(3):
        wsel[d] = wp.where(d == comp, wp.clamp(p[d] - float(fc[d]), 0.0, 1.0),
                           wp.clamp(p[d] - float(cc[d]) - 0.5, 0.0, 1.0))
        gpos[d] = wp.where(d == comp, float(fc[d]), float(cc[d]) + 0.5)
    w = (wp.where(di == 0, 1.0 - wsel[0], wsel[0]) * wp.where(dj == 0, 1.0 - wsel[1], wsel[1])
         * wp.where(dk == 0, 1.0 - wsel[2], wsel[2]))
    r = gpos + wp.vec3(float(di), float(dj), float(dk)) - p
    return w, w * v, w * wp.dot(aff, r)


@wp.kernel
def _apic_p2g_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                     total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                     pvel: wp.array(dtype=wp.vec3), cx: wp.array(dtype=wp.vec3),
                     cy: wp.array(dtype=wp.vec3), cz: wp.array(dtype=wp.vec3),
                     vel: wp.array4d(dtype=wp.float32), mass: wp.array4d(dtype=wp.float32)):
    # knApicMapLinearVec3ToMACGrid, gathered per face through the index.
    i, j, k = wp.tid()
    nx = vel.shape[0]
    ny = vel.shape[1]
    nz = vel.shape[2]
    target = i + nx * (j + ny * k)
    for comp in range(3):
        msum = float(0.0)
        vsum = float(0.0)
        if i > 0 and j > 0:
            # Offsets past a row only land on x = 0 or y = 0, so here the sources are the
            # neighbouring cells: two along comp, three across the other axes.
            x1 = wp.min(wp.where(comp == 0, i, i + 1), nx - 1)
            y1 = wp.min(wp.where(comp == 1, j, j + 1), ny - 1)
            z0 = wp.max(k - 1, 0)
            z1 = wp.min(wp.where(comp == 2, k, k + 1), nz - 1)
            for xj in range(i - 1, x1 + 1):
                for yj in range(j - 1, y1 + 1):
                    for q in range(start[xj, yj, z0], _cell_end(start, total, nx, ny, nz, xj, yj, z1)):
                        t = sorted_idx[q]
                        p = pos[t]
                        fc = _trunc3(p)
                        cc = _trunc3(p - wp.vec3(0.5, 0.5, 0.5))
                        b = _apic_base(fc, cc, comp)
                        di = i - b[0]
                        dj = j - b[1]
                        dk = k - b[2]
                        if di >= 0 and di <= 1 and dj >= 0 and dj <= 1 and dk >= 0 and dk <= 1:
                            aff = wp.where(comp == 0, cx[t], wp.where(comp == 1, cy[t], cz[t]))
                            w, a, c = _apic_share(p, pvel[t][comp], aff, fc, cc, comp, di, dj, dk)
                            msum = msum + w
                            vsum = vsum + a
                            vsum = vsum + c
        else:
            # A particle writes to the eight linear offsets of its base face, and an offset
            # past a row wraps into the next one, as in Mantaflow: try all eight base faces.
            for dk in range(2):
                for dj in range(2):
                    for di in range(2):
                        base = target - (di + nx * (dj + ny * dk))
                        if base >= 0:
                            bx = base % nx
                            by = (base / nx) % ny
                            bz = base / (nx * ny)
                            x1 = wp.min(bx + wp.where(comp == 0, 0, 1), nx - 1)
                            y1 = wp.min(by + wp.where(comp == 1, 0, 1), ny - 1)
                            z1 = wp.min(bz + wp.where(comp == 2, 0, 1), nz - 1)
                            for xj in range(bx, x1 + 1):
                                for yj in range(by, y1 + 1):
                                    for q in range(start[xj, yj, bz], _cell_end(start, total, nx, ny, nz, xj, yj, z1)):
                                        t = sorted_idx[q]
                                        p = pos[t]
                                        fc = _trunc3(p)
                                        cc = _trunc3(p - wp.vec3(0.5, 0.5, 0.5))
                                        if _face_index(nx, ny, nz, fc, cc, comp, wp.vec3i(0, 0, 0)) == base:
                                            aff = wp.where(comp == 0, cx[t], wp.where(comp == 1, cy[t], cz[t]))
                                            w, a, c = _apic_share(p, pvel[t][comp], aff, fc, cc, comp, di, dj, dk)
                                            msum = msum + w
                                            vsum = vsum + a
                                            vsum = vsum + c
        mass[i, j, k, comp] = msum
        vel[i, j, k, comp] = vsum


@wp.kernel
def _apic_g2p_kernel(pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                     flag: wp.array(dtype=wp.int32), cx: wp.array(dtype=wp.vec3),
                     cy: wp.array(dtype=wp.vec3), cz: wp.array(dtype=wp.vec3),
                     vel: wp.array4d(dtype=wp.float32)):
    # knApicMapLinearMACGridToVec3; a face out of range stops the particle there, as in Mantaflow.
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    nx = vel.shape[0]
    ny = vel.shape[1]
    nz = vel.shape[2]
    p = pos[t]
    if cell_of(p, wp.vec3i(nx, ny, nz))[0] < 0:
        return
    fc = _trunc3(p)
    cc = _trunc3(p - wp.vec3(0.5, 0.5, 0.5))
    wf = wp.vec3()
    wc = wp.vec3()
    for d in range(3):
        wf[d] = wp.clamp(p[d] - float(fc[d]), 0.0, 1.0)
        wc[d] = wp.clamp(p[d] - float(cc[d]) - 0.5, 0.0, 1.0)
    vp = wp.vec3()
    a0 = wp.vec3()
    a1 = wp.vec3()
    a2 = wp.vec3()
    stop = int(0)
    for comp in range(3):
        if stop == 0 and _face_index(nx, ny, nz, fc, cc, comp, wp.vec3i(0, 0, 0)) < 0:
            stop = 1
        if stop == 0:
            wsel = wp.vec3()
            for d in range(3):
                wsel[d] = wp.where(d == comp, wf[d], wc[d])
            a = wp.vec3()
            for di in range(2):
                wi = wp.where(di == 0, 1.0 - wsel[0], wsel[0])
                gi = wp.where(di == 0, -1.0, 1.0)
                for dj in range(2):
                    wj = wp.where(dj == 0, 1.0 - wsel[1], wsel[1])
                    gj = wp.where(dj == 0, -1.0, 1.0)
                    for dk in range(2):
                        wk = wp.where(dk == 0, 1.0 - wsel[2], wsel[2])
                        gk = wp.where(dk == 0, -1.0, 1.0)
                        lin = _face_index(nx, ny, nz, fc, cc, comp, wp.vec3i(di, dj, dk))
                        if lin >= 0:
                            g = vel[lin % nx, (lin / nx) % ny, lin / (nx * ny), comp]
                            vp[comp] = vp[comp] + wi * wj * wk * g
                            a[0] = a[0] + gi * wj * wk * g
                            a[1] = a[1] + wi * gj * wk * g
                            a[2] = a[2] + wi * wj * gk * g
            if comp == 0:
                a0 = a
            elif comp == 1:
                a1 = a
            else:
                a2 = a
    pvel[t] = vp
    cx[t] = a0
    cy[t] = a1
    cz[t] = a2


@wp.kernel
def _flip_update_kernel(pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                        flag: wp.array(dtype=wp.int32), vel: wp.array4d(dtype=wp.float32),
                        old: wp.array4d(dtype=wp.float32), ratio: float):
    # knMapLinearMACGridToVec3_FLIP; (1.0 - flipRatio) * v is evaluated in double as in C++.
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    p = pos[t]
    v = mac_interp(vel, p)
    delta = v - mac_interp(old, p)
    pic = wp.float64(1.0) - wp.float64(ratio)
    a = ratio * (pvel[t] + delta)
    r = wp.vec3()
    for c in range(3):
        r[c] = a[c] + wp.float32(wp.float64(v[c]) * pic)
    pvel[t] = r


@wp.kernel
def _weight_reset_kernel(weight: wp.array4d(dtype=wp.float32), c: int):
    # extrapolateMACFromWeight: 1 where weight > 0 (inner cells only), 0 stays 0.
    i, j, k = wp.tid()
    if border(i, j, k, weight.shape[0], weight.shape[1], weight.shape[2]):
        return
    if weight[i, j, k, c] > 0.0:
        weight[i, j, k, c] = 1.0


@wp.kernel
def _mac_extrap_weight_kernel(vel: wp.array4d(dtype=wp.float32), weight: wp.array4d(dtype=wp.float32),
                              d: int, c: int):
    # knExtrapolateMACFromWeight
    i, j, k = wp.tid()
    if border(i, j, k, vel.shape[0], vel.shape[1], vel.shape[2]) or weight[i, j, k, c] != 0.0:
        return
    nbs = int(0)
    s = float(0.0)
    p = wp.vec3i(i, j, k)
    for n in range(6):
        q = p + nb6(n)
        if weight[q[0], q[1], q[2], c] == float(d):
            s += vel[q[0], q[1], q[2], c]
            nbs += 1
    if nbs > 0:
        weight[i, j, k, c] = float(d + 1)
        vel[i, j, k, c] = s / float(nbs)


@wp.kernel
def _mac_mark_kernel(flags: wp.array3d(dtype=wp.int32), tmp: wp.array3d(dtype=wp.int32), c: int,
                     into_obs: int):
    # extrapolateMACSimple: faces next to a fluid cell are known (1), the rest 0.
    i, j, k = wp.tid()
    v = int(0)
    if not border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        q = wp.vec3i(i, j, k) - wp.vec3i(wp.where(c == 0, 1, 0), wp.where(c == 1, 1, 0),
                                         wp.where(c == 2, 1, 0))
        fp = flags[i, j, k]
        fq = flags[q[0], q[1], q[2]]
        if (fp & FLUID) != 0 or (fq & FLUID) != 0:
            v = 1
            if into_obs != 0 and ((fp & OBSTACLE) != 0 or (fq & OBSTACLE) != 0):
                v = 0
    tmp[i, j, k] = v


@wp.kernel
def _mac_extrap_kernel(vel: wp.array4d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.int32),
                       d: int, c: int):
    # knExtrapolateMACSimple
    i, j, k = wp.tid()
    if border(i, j, k, vel.shape[0], vel.shape[1], vel.shape[2]) or tmp[i, j, k] != 0:
        return
    nbs = int(0)
    s = float(0.0)
    p = wp.vec3i(i, j, k)
    for n in range(6):
        q = p + nb6(n)
        if tmp[q[0], q[1], q[2]] == d:
            s += vel[q[0], q[1], q[2], c]
            nbs += 1
    if nbs > 0:
        tmp[i, j, k] = d + 1
        vel[i, j, k, c] = s / float(nbs)


@wp.kernel
def _extrap_into_bnd_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                            src: wp.array3d(dtype=wp.vec3)):
    # knExtrapolateIntoBnd: the last matching side wins, divided by the number of sides.
    i, j, k = wp.tid()
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    obs = (flags[i, j, k] & OBSTACLE) != 0
    v = wp.vec3()
    c = int(0)
    if i == 0:
        v = src[i + 1, j, k]
        if obs and v[0] < 0.0:
            v[0] = 0.0
        c += 1
    elif i == nx - 1:
        v = src[i - 1, j, k]
        if obs and v[0] > 0.0:
            v[0] = 0.0
        c += 1
    if j == 0:
        v = src[i, j + 1, k]
        if obs and v[1] < 0.0:
            v[1] = 0.0
        c += 1
    elif j == ny - 1:
        v = src[i, j - 1, k]
        if obs and v[1] > 0.0:
            v[1] = 0.0
        c += 1
    if k == 0:
        v = src[i, j, k + 1]
        if obs and v[2] < 0.0:
            v[2] = 0.0
        c += 1
    elif k == nz - 1:
        v = src[i, j, k - 1]
        if obs and v[2] > 0.0:
            v[2] = 0.0
        c += 1
    if c > 0:
        vel[i, j, k] = v / float(c)


@wp.kernel
def _combine_vels_kernel(vel: wp.array4d(dtype=wp.float32), w: wp.array4d(dtype=wp.float32),
                         combine: wp.array4d(dtype=wp.float32), phi: wp.array3d(dtype=wp.float32),
                         use_phi: int, band: float, thresh: float):
    # knCombineVels
    i, j, k = wp.tid()
    for c in range(3):
        skip = int(0)
        if use_phi != 0:
            p = wp.vec3(float(i), float(j), float(k))
            p[(c + 1) % 3] = p[(c + 1) % 3] + 0.5
            p[(c + 2) % 3] = p[(c + 2) % 3] + 0.5
            if sample(phi, p) < -band:
                vel[i, j, k, c] = 0.0
                skip = 1
        if skip == 0:
            if w[i, j, k, c] > thresh:
                combine[i, j, k, c] = vel[i, j, k, c]
                vel[i, j, k, c] = -1.0
            else:
                vel[i, j, k, c] = 0.0


@wp.kernel
def _adjust_outside_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                           shape: wp.vec3i):
    # adjustNumber, first loop: the out-of-domain test. The particle index leaves these out.
    t = wp.tid()
    if cell_of(pos[t], shape)[0] < 0:
        flag[t] = flag[t] | PDELETE


@wp.kernel
def _adjust_kill_kernel(start: wp.array3d(dtype=wp.int32), sorted_idx: wp.array(dtype=wp.int32),
                        total: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
                        flag: wp.array(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32),
                        cnt: wp.array3d(dtype=wp.int32), max_parts: int, surface_ls: float,
                        band: float):
    # adjustNumber, first loop, one cell at a time. The index lists a cell's particles in
    # ascending array order, so this is Mantaflow's serial result on every run.
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    lin = (i * ny + j) * nz + k
    p1 = total[0]
    if lin + 1 < nx * ny * nz:
        p1 = start[(lin + 1) // (ny * nz), ((lin + 1) // nz) % ny, (lin + 1) % nz]
    num = int(0)
    for q in range(start[i, j, k], p1):
        t = sorted_idx[q]
        phiv = sample(phi, pos[t])
        if phiv > 0.0 or (band > 0.0 and phiv < -band) or (num > max_parts and phiv <= surface_ls):
            flag[t] = flag[t] | PDELETE
        else:
            num += 1
    cnt[i, j, k] = num


@wp.kernel
def _adjust_need_kernel(cnt: wp.array3d(dtype=wp.int32), flags: wp.array3d(dtype=wp.int32),
                        phi: wp.array3d(dtype=wp.float32), exclude: wp.array3d(dtype=wp.float32),
                        use_exclude: int, min_parts: int, surface_ls: float, band: float,
                        need: wp.array3d(dtype=wp.int32)):
    # adjustNumber, seeding test. `need` is indexed [k, j, i], so its flat order is
    # Mantaflow's FOR_IJK order and a scan gives each cell's offset in the random stream.
    i, j, k = wp.tid()
    m = int(0)
    ph = phi[i, j, k]
    ok = ph <= surface_ls and not (band > 0.0 and ph < -band)
    if ok and use_exclude != 0 and exclude[i, j, k] < 0.0:
        ok = False
    if ok and (flags[i, j, k] & FLUID) != 0 and cnt[i, j, k] < min_parts:
        m = min_parts - cnt[i, j, k]
    need[k, j, i] = m


@wp.kernel
def _alive_kernel(flag: wp.array(dtype=wp.int32), alive: wp.array(dtype=wp.int32)):
    t = wp.tid()
    alive[t] = wp.where((flag[t] & PDELETE) == 0, 1, 0)


@wp.kernel
def _scan_total_kernel(counts: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                       out: wp.array(dtype=wp.int64), slot: int):
    last = counts.shape[0] - 1
    out[slot] = wp.int64(starts[last]) + wp.int64(counts[last])


@wp.kernel
def _compact_kernel(alive: wp.array(dtype=wp.int32), slot: wp.array(dtype=wp.int32),
                    pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                    flag: wp.array(dtype=wp.int32), npos: wp.array(dtype=wp.vec3),
                    nvel: wp.array(dtype=wp.vec3), nflag: wp.array(dtype=wp.int32)):
    # Survivors keep their order; the NEW flag is cleared (insertBufferedParticles).
    t = wp.tid()
    if alive[t] == 0:
        return
    s = slot[t]
    npos[s] = pos[t]
    nvel[s] = vel[t]
    nflag[s] = flag[t] & ~PNEW


@wp.kernel
def _compact_vec_kernel(alive: wp.array(dtype=wp.int32), slot: wp.array(dtype=wp.int32),
                        src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    if alive[t] != 0:
        dst[slot[t]] = src[t]


@wp.kernel
def _compact_float_kernel(alive: wp.array(dtype=wp.int32), slot: wp.array(dtype=wp.int32),
                          src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    t = wp.tid()
    if alive[t] != 0:
        dst[slot[t]] = src[t]


@wp.kernel
def _seed_kernel(need: wp.array3d(dtype=wp.int32), start: wp.array3d(dtype=wp.int32),
                 rnd: wp.array(dtype=wp.float32), base: int, vel: wp.array4d(dtype=wp.float32),
                 pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                 flag: wp.array(dtype=wp.int32), cx: wp.array(dtype=wp.vec3),
                 cy: wp.array(dtype=wp.vec3), cz: wp.array(dtype=wp.vec3), use_apic: int):
    # pos = Vec3(i,j,k) + rand.getVec3(); velocity from the MAC grid (pVel.setSource(vel, isMAC)).
    # New APIC particles start with zero affine rows (free slots are not zeroed).
    i, j, k = wp.tid()
    m = need[k, j, i]
    s = start[k, j, i]
    for q in range(m):
        r = 3 * (s + q)
        p = wp.vec3(float(i), float(j), float(k)) + wp.vec3(rnd[r], rnd[r + 1], rnd[r + 2])
        pos[base + s + q] = p
        pvel[base + s + q] = mac_interp(vel, p)
        flag[base + s + q] = PNEW
        if use_apic != 0:
            cx[base + s + q] = wp.vec3()
            cy[base + s + q] = wp.vec3()
            cz[base + s + q] = wp.vec3()


# ---- substep pieces (step 21) ---------------------------------------------------

@wp.func
def _blocked(flags: wp.array3d(dtype=wp.int32), p: wp.vec3):
    # !flags.isInBounds(pos, 1) || flags.isObstacle(pos)
    c = wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))
    for a in range(3):
        if c[a] < 1 or c[a] >= flags.shape[a] - 1:
            return True
    return (flags[c[0], c[1], c[2]] & OBSTACLE) != 0


@wp.kernel
def _advect_rk4_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                       flags: wp.array3d(dtype=wp.int32), vel: wp.array4d(dtype=wp.float32),
                       dt: float, delete_obs: int, skip_new: int):
    # GridAdvectKernel stages inside integratePointSet(IntRK4), then KnDeleteInObstacle
    # or KnClampPositions (stopInObstacle=False). Each particle only reads its own
    # positions, so running the four stages in one thread equals Mantaflow's four passes.
    t = wp.tid()
    f = flag[t]
    if (f & PDELETE) != 0 or (skip_new != 0 and (f & PNEW) != 0):
        return
    x0 = pos[t]
    p = x0
    tot = wp.vec3()
    for s in range(4):
        if delete_obs != 0 and _blocked(flags, p):
            flag[t] = f | PDELETE
            return
        u = mac_interp(vel, p) * dt
        if s == 0:
            tot = u
            p = x0 + 0.5 * u
        elif s == 1:
            p = x0 + 0.5 * u
            tot = tot + 2.0 * u
        elif s == 2:
            p = x0 + u
            tot = tot + 2.0 * u
        else:
            p = x0 + wp.float32(1.0 / 6.0) * (tot + u)
    if delete_obs != 0:
        if _blocked(flags, p):
            flag[t] = f | PDELETE
    else:
        c = cell_of(p, wp.vec3i(flags.shape[0], flags.shape[1], flags.shape[2]))
        if c[0] < 0:
            for a in range(3):
                p[a] = wp.clamp(p[a], 0.0, float(flags.shape[a] - 1))
    pos[t] = p


@wp.kernel
def _push_out_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                     phi: wp.array3d(dtype=wp.float32), shift: float, thresh: float):
    # knPushOutofObs with getGradient (central differences, indices clamped to [1, n-2]).
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    p = pos[t]
    c = cell_of(p, wp.vec3i(phi.shape[0], phi.shape[1], phi.shape[2]))
    if c[0] < 0:
        return
    v = sample(phi, p)
    if v < thresh:
        i = wp.clamp(c[0], 1, phi.shape[0] - 2)
        j = wp.clamp(c[1], 1, phi.shape[1] - 2)
        k = wp.clamp(c[2], 1, phi.shape[2] - 2)
        g = wp.vec3(phi[i + 1, j, k] - phi[i - 1, j, k], phi[i, j + 1, k] - phi[i, j - 1, k],
                    phi[i, j, k + 1] - phi[i, j, k - 1])
        norm = wp.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        if norm < VECTOR_EPSILON:
            return
        g = g / norm
        pos[t] = p + g * (thresh - v + shift)


@wp.func
def _mirror_of(p: wp.vec3, phi: wp.array3d(dtype=wp.float32), max_dist: float, reach: float):
    # CODE-LIQ-015 ghost particle: the mirror image of p across the obstacle surface (gradient
    # as _push_out_kernel), or a point with x = -1 when there is none: p not within max_dist
    # of a wall, the image in the grid's outer layer (the domain walls, which the VDB
    # writer's wall layers already close), or the wall thinner than the image plus reach.
    none = wp.vec3(-1.0, 0.0, 0.0)
    shape = wp.vec3i(phi.shape[0], phi.shape[1], phi.shape[2])
    c = cell_of(p, shape)
    if c[0] < 0:
        return none
    v = sample(phi, p)
    if v <= 0.0 or v >= max_dist:
        return none
    i = wp.clamp(c[0], 1, phi.shape[0] - 2)
    j = wp.clamp(c[1], 1, phi.shape[1] - 2)
    k = wp.clamp(c[2], 1, phi.shape[2] - 2)
    g = wp.vec3(phi[i + 1, j, k] - phi[i - 1, j, k], phi[i, j + 1, k] - phi[i, j - 1, k],
                phi[i, j, k + 1] - phi[i, j, k - 1])
    norm = wp.length(g)
    if norm < VECTOR_EPSILON:
        return none
    g = g / norm
    q = p - g * (2.0 * v)
    for a in range(3):
        if q[a] < 1.0 or q[a] >= float(phi.shape[a] - 1):
            return none
    deep = q - g * reach
    if cell_of(deep, shape)[0] >= 0 and sample(phi, deep) >= 0.0:
        return none
    return q


@wp.kernel
def _mirror_mask_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                        phi: wp.array3d(dtype=wp.float32), max_dist: float, reach: float,
                        keep: wp.array(dtype=wp.int32)):
    t = wp.tid()
    keep[t] = 0
    if (flag[t] & PDELETE) == 0 and _mirror_of(pos[t], phi, max_dist, reach)[0] >= 0.0:
        keep[t] = 1


@wp.kernel
def _mirror_scatter_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                           phi: wp.array3d(dtype=wp.float32), max_dist: float, reach: float,
                           keep: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32), n: int,
                           out_pos: wp.array(dtype=wp.vec3), out_flag: wp.array(dtype=wp.int32)):
    # Recomputes the image instead of storing it: 12 B less scratch per particle.
    t = wp.tid()
    out_pos[t] = pos[t]
    out_flag[t] = flag[t]
    if keep[t] != 0:
        out_pos[n + start[t]] = _mirror_of(pos[t], phi, max_dist, reach)
        out_flag[n + start[t]] = 0


@wp.kernel
def _clip_obstacle_kernel(phi: wp.array3d(dtype=wp.float32), obs: wp.array3d(dtype=wp.float32),
                          sf: wp.vec3, off: wp.vec3, eps: float):
    # obs sampled at the phi cell centre exactly as interpolate_grid (knInterpolateGridTempl).
    i, j, k = wp.tid()
    p = wp.cw_mul(wp.vec3(float(i), float(j), float(k)), sf) + off
    phi[i, j, k] = wp.max(phi[i, j, k], -(sample(obs, p) + eps))


@wp.func
def _slide(vel: wp.vec3, vobs: wp.vec3, n: wp.vec3):
    vn = wp.dot(vel - vobs, n)
    if vn < 0.0:
        return vel - vn * n
    return vel


@wp.kernel
def _collide_mesh_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                         flag: wp.array(dtype=wp.int32), pos_old: wp.array(dtype=wp.vec3),
                         n_old: int, mesh: wp.uint64, lo: wp.vec3, hi: wp.vec3, margin: float,
                         two_sided: int, max_dist: float):
    """two_sided: the mesh has no inside (open, planar or inside out)."""
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    p = pos[t]
    # Swept test: the first face on the path old -> new; a closed mesh only
    # stops entry through an outward face, a two-sided one stops either way.
    if t < n_old:
        a = pos_old[t]
        d = p - a
        dl = wp.length(d)
        if dl > 1.0e-6:
            d = d / dl
            r = wp.mesh_query_ray(mesh, a, d, dl + margin)
            if r.result:
                n = wp.normalize(r.normal)
                if two_sided != 0 and wp.dot(n, d) > 0.0:
                    n = -n
                if wp.dot(n, d) < 0.0:
                    q = a + d * r.t
                    if wp.dot(p - q, n) < margin:
                        pos[t] = q + n * margin
                        vel[t] = _slide(vel[t], wp.mesh_eval_velocity(mesh, r.face, r.u, r.v), n)
                        return
    # Inside or too close: closest point + margin. Skip particles outside the box.
    for c in range(3):
        if p[c] < lo[c] - margin or p[c] > hi[c] + margin:
            return
    if two_sided != 0:
        # Two-sided: only the margin applies, and the particle stays on its side.
        qn = wp.mesh_query_point_no_sign(mesh, p, margin)
        if not qn.result:
            return
        q = wp.mesh_eval_position(mesh, qn.face, qn.u, qn.v)
        e = p - q
        dist = wp.length(e)
        if dist >= margin:
            return
        n = wp.mesh_eval_face_normal(mesh, qn.face)
        if dist > 1.0e-6:
            n = e / dist
        elif t < n_old and wp.dot(n, pos_old[t] - p) < 0.0:
            n = -n
        pos[t] = q + n * margin
        vel[t] = _slide(vel[t], wp.mesh_eval_velocity(mesh, qn.face, qn.u, qn.v), n)
        return
    # Closed: the sign is Warp's three-ray vote (no winding-number data is built).
    qp = wp.mesh_query_point(mesh, p, max_dist)
    if not qp.result:
        return
    q = wp.mesh_eval_position(mesh, qp.face, qp.u, qp.v)
    e = p - q
    dist = wp.length(e)
    if qp.sign > 0.0 and dist >= margin:
        return
    n = wp.mesh_eval_face_normal(mesh, qp.face)
    if dist > 1.0e-6:
        n = e / dist
        if qp.sign < 0.0:
            n = -n
    pos[t] = q + n * margin
    vel[t] = _slide(vel[t], wp.mesh_eval_velocity(mesh, qp.face, qp.u, qp.v), n)


@wp.kernel
def _thin_obstacle_kernel(mesh: wp.uint64, lo: wp.vec3i, src: wp.array3d(dtype=wp.float32),
                          phi: wp.array3d(dtype=wp.float32), obvel: wp.array3d(dtype=wp.vec3),
                          num: wp.array3d(dtype=wp.float32), use_vel: int):
    a, b, c = wp.tid()
    i = a + lo[0]
    j = b + lo[1]
    k = c + lo[2]
    if src[i, j, k] < 0.0:
        return
    # Only where the grid has no obstacle cell next to it (sub-cell parts).
    if i > 0 and src[i - 1, j, k] < 0.0:
        return
    if i < src.shape[0] - 1 and src[i + 1, j, k] < 0.0:
        return
    if j > 0 and src[i, j - 1, k] < 0.0:
        return
    if j < src.shape[1] - 1 and src[i, j + 1, k] < 0.0:
        return
    if k > 0 and src[i, j, k - 1] < 0.0:
        return
    if k < src.shape[2] - 1 and src[i, j, k + 1] < 0.0:
        return
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    q = wp.mesh_query_point_no_sign(mesh, p, 0.5)
    if not q.result:
        return
    d = wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))
    if d >= 0.5:
        return
    phi[i, j, k] = wp.min(phi[i, j, k], d - 0.5)
    if use_vel != 0:
        num[i, j, k] = num[i, j, k] + 1.0
        obvel[i, j, k] = obvel[i, j, k] + wp.mesh_eval_velocity(mesh, q.face, q.u, q.v)


@wp.kernel
def _collide_walls_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                          flag: wp.array(dtype=wp.int32), lo: wp.vec3, hi: wp.vec3,
                          closed_lo: wp.vec3i, closed_hi: wp.vec3i):
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    p = pos[t]
    v = vel[t]
    for c in range(3):
        if closed_lo[c] != 0 and p[c] < lo[c]:
            p[c] = lo[c]
            v[c] = wp.max(v[c], 0.0)
        if closed_hi[c] != 0 and p[c] > hi[c]:
            p[c] = hi[c]
            v[c] = wp.min(v[c], 0.0)
    pos[t] = p
    vel[t] = v


@wp.kernel
def _gravity_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3), force: wp.vec3):
    # KnApplyForce (additive, no exclude), bnd 1.
    i, j, k = wp.tid()
    if border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        return
    f = flags[i, j, k]
    fluid = (f & FLUID) != 0
    if not fluid and (f & EMPTY) == 0:
        return
    v = vel[i, j, k]
    nb = wp.vec3i(flags[i - 1, j, k], flags[i, j - 1, k], flags[i, j, k - 1])
    for c in range(3):
        if (nb[c] & FLUID) != 0 or (fluid and (nb[c] & EMPTY) != 0):
            v[c] = v[c] + force[c]
    vel[i, j, k] = v


@wp.func
def _sample_sub(i: int, j: int, k: int, q: int, d: int, disp: float, jlen: float,
                rnd: wp.array(dtype=wp.float32), r: int):
    # sampleLevelsetWithParticles: subpos = pos + disp * (0.5 + di, ...) +
    # jlen * (1 - 2.0 * rand.getVec3()); the products with doubles round as in C++.
    di = q % d
    dj = (q / d) % d
    dk = q / (d * d)
    off = wp.vec3(wp.float32(0.5 + float(di)), wp.float32(0.5 + float(dj)), wp.float32(0.5 + float(dk)))
    p = wp.vec3(float(i), float(j), float(k)) + disp * off
    rv = wp.vec3()
    for c in range(3):
        rv[c] = 1.0 - wp.float32(wp.float64(2.0) * wp.float64(rnd[r + c]))
    return p + jlen * rv


@wp.kernel
def _sample_ls_kernel(phi: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32),
                         d: int, disp: float, jlen: float, rnd: wp.array(dtype=wp.float32),
                         cand_start: wp.array3d(dtype=wp.int32), pass_no: int,
                         kept: wp.array3d(dtype=wp.int32), kept_start: wp.array3d(dtype=wp.int32),
                         base: int, pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                         flag: wp.array(dtype=wp.int32), vsrc: wp.array3d(dtype=wp.vec3),
                         use_vsrc: int, cx: wp.array(dtype=wp.vec3), cy: wp.array(dtype=wp.vec3),
                         cz: wp.array(dtype=wp.vec3), use_apic: int):
    # pass 0: candidates per cell ([k, j, i] = FOR_IJK order), pass 1: kept per cell,
    # pass 2: write the kept particles in Mantaflow's insertion order.
    i, j, k = wp.tid()
    ok = (flags[i, j, k] & OBSTACLE) == 0 and phi[i, j, k] < 1.733
    if pass_no == 0:
        kept[k, j, i] = wp.where(ok, d * d * d, 0)
        return
    if not ok:
        if pass_no == 1:
            kept[k, j, i] = 0
        return
    s0 = cand_start[k, j, i]
    w = int(0)
    for q in range(d * d * d):
        p = _sample_sub(i, j, k, q, d, disp, jlen, rnd, 3 * (s0 + q))
        if sample(phi, p) > 0.0:
            continue
        if pass_no == 2:
            o = base + kept_start[k, j, i] + w
            pos[o] = p
            pvel[o] = wp.where(use_vsrc != 0, sample_vec(vsrc, p), wp.vec3())
            flag[o] = PNEW
            if use_apic != 0:
                cx[o] = wp.vec3()
                cy[o] = wp.vec3()
                cz[o] = wp.vec3()
        w += 1
    if pass_no == 1:
        kept[k, j, i] = w


@wp.kernel
def _clear_new_kernel(flag: wp.array(dtype=wp.int32)):
    t = wp.tid()
    flag[t] = flag[t] & ~PNEW


# ---- emission and obstacles (step 22) ------------------------------------------

@wp.kernel
def _flood_seed_kernel(phi: wp.array3d(dtype=wp.float32), vis: wp.array3d(dtype=wp.int32), bnd: int):
    # floodFill (outside): the search starts from every phi >= 0 cell on the inner
    # face of the border; cells in the border itself are never visited.
    i, j, k = wp.tid()
    nx = phi.shape[0]
    ny = phi.shape[1]
    nz = phi.shape[2]
    v = int(0)
    if i >= bnd and j >= bnd and k >= bnd and i < nx - bnd and j < ny - bnd and k < nz - bnd:
        if phi[i, j, k] >= 0.0 and (i == bnd or j == bnd or k == bnd or i == nx - 1 - bnd
                                    or j == ny - 1 - bnd or k == nz - 1 - bnd):
            v = 1
    vis[i, j, k] = v


@wp.func
def _vis(vis: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, bnd: int):
    if i < bnd or j < bnd or k < bnd or i >= vis.shape[0] - bnd or j >= vis.shape[1] - bnd \
            or k >= vis.shape[2] - bnd:
        return 0
    return vis[i, j, k]


@wp.kernel
def _flood_grow_kernel(phi: wp.array3d(dtype=wp.float32), vis: wp.array3d(dtype=wp.int32), bnd: int,
                       changed: wp.array(dtype=wp.int32)):
    # One step of the 6-connected search, in place (visited only ever goes 0 -> 1,
    # so the result is the same reachable set as Mantaflow's stack walk).
    i, j, k = wp.tid()
    if _vis(vis, i, j, k, bnd) != 0 or i < bnd or j < bnd or k < bnd or i >= phi.shape[0] - bnd \
            or j >= phi.shape[1] - bnd or k >= phi.shape[2] - bnd or phi[i, j, k] < 0.0:
        return
    if (_vis(vis, i - 1, j, k, bnd) + _vis(vis, i + 1, j, k, bnd) + _vis(vis, i, j - 1, k, bnd)
            + _vis(vis, i, j + 1, k, bnd) + _vis(vis, i, j, k - 1, bnd) + _vis(vis, i, j, k + 1, bnd)) > 0:
        vis[i, j, k] = 1
        changed[0] = 1


@wp.kernel
def _flood_apply_kernel(phi: wp.array3d(dtype=wp.float32), vis: wp.array3d(dtype=wp.int32), bnd: int,
                        value: float):
    # KnFillApply (outside)
    i, j, k = wp.tid()
    if i < bnd or j < bnd or k < bnd or i >= phi.shape[0] - bnd or j >= phi.shape[1] - bnd \
            or k >= phi.shape[2] - bnd:
        return
    if vis[i, j, k] == 0 and phi[i, j, k] >= 0.0:
        phi[i, j, k] = value


@wp.kernel
def _subtract_kernel(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    if b[i, j, k] < 0.0:
        a[i, j, k] = b[i, j, k] * -1.0


@wp.kernel
def _flags_obs_kernel(flags: wp.array3d(dtype=wp.int32), phi_obs: wp.array3d(dtype=wp.float32),
                      phi_in: wp.array3d(dtype=wp.float32), phi_out: wp.array3d(dtype=wp.float32),
                      use_in: int, use_out: int, bnd: int):
    # KnUpdateFlagsObs without fractions
    i, j, k = wp.tid()
    if i < bnd or j < bnd or k < bnd or i >= flags.shape[0] - bnd or j >= flags.shape[1] - bnd \
            or k >= flags.shape[2] - bnd:
        return
    if phi_obs[i, j, k] < 0.0:
        flags[i, j, k] = OBSTACLE
    elif use_in != 0 and phi_in[i, j, k] < 0.0:
        flags[i, j, k] = FLUID | INFLOW
    elif use_out != 0 and phi_out[i, j, k] < 0.0:
        flags[i, j, k] = EMPTY | OUTFLOW
    else:
        flags[i, j, k] = EMPTY


@wp.kernel
def _outflow_kill_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                         flags: wp.array3d(dtype=wp.int32)):
    t = wp.tid()
    if (flag[t] & PDELETE) != 0:
        return
    c = cell_of(pos[t], wp.vec3i(flags.shape[0], flags.shape[1], flags.shape[2]))
    if c[0] >= 0 and (flags[c[0], c[1], c[2]] & OUTFLOW) != 0:
        flag[t] = flag[t] | PDELETE


@wp.kernel
def _outflow_cells_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    f = flags[i, j, k]
    if (f & OUTFLOW) != 0:
        flags[i, j, k] = (f | EMPTY) & ~FLUID
        phi[i, j, k] = 0.5


def _mt_floats(n, chunk=1 << 20):
    """RandomStream(PARTICLE_SEED).getReal() x n (float build: (float)(randInt / (2^32 - 1))).
    Converted in chunks: the uint64 and float64 intermediates would otherwise take 16 B per float."""
    gen = np.random.RandomState(PARTICLE_SEED)._bit_generator
    out = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        out[s:s + m] = gen.random_raw(m) * (1.0 / 4294967295.0)
    return out


MIN_CAPACITY = 1 << 14  # _compact never shrinks a particle set below this many slots
ANISO_MIN_SLOTS = 1 << 16  # smallest anisotropic mesh scratch pair (3 MB)
SLOT_SCRATCH_MIN = MIN_CAPACITY  # smallest per-particle work array


class LiquidMixin:
    """Backend methods for liquids; WarpBackend inherits them."""

    _seed_stream = None  # cached adjustNumber random stream (identical on every call)
    _seed_recent = 0  # largest non-shrinking request since the last shrinking one

    def particle_count(self, parts):
        return parts.n

    def particle_capacity(self, parts):
        return parts.capacity

    def particles(self, capacity, apic=False, life=False):
        d = self.device
        cap = max(int(capacity), 1)
        parts = WarpParticles(wp.zeros(cap, dtype=wp.vec3, device=d),
                              wp.zeros(cap, dtype=wp.vec3, device=d),
                              wp.zeros(cap, dtype=wp.int32, device=d))
        if apic:
            parts.cx, parts.cy, parts.cz = (wp.zeros(cap, dtype=wp.vec3, device=d) for _ in range(3))
        if life:
            parts.life = wp.zeros(cap, dtype=wp.float32, device=d)
        self._allocated += cap * self._bytes_per_particle(parts)
        return parts

    @staticmethod
    def _bytes_per_particle(parts):
        return 28 + (36 if parts.cx is not None else 0) + (4 if parts.life is not None else 0)

    def set_particles(self, parts, pos, vel=None, flag=None, affine=None, life=None):
        pos = np.asarray(pos, np.float32).reshape(-1, 3)
        n = pos.shape[0]
        if n > parts.capacity:
            raise ValueError(f"set_particles: {n} particles, capacity {parts.capacity}")
        vel = np.zeros_like(pos) if vel is None else np.asarray(vel, np.float32).reshape(n, 3)
        flag = np.zeros(n, np.int32) if flag is None else np.asarray(flag, np.int32).reshape(n)
        for dst, src in ((parts.pos, pos), (parts.vel, vel), (parts.flag, flag)):
            if n:
                dst[:n].assign(src)
        if parts.cx is not None:
            aff = (np.zeros((n, 3, 3), np.float32) if affine is None
                   else np.asarray(affine, np.float32).reshape(n, 3, 3))
            for c, dst in enumerate((parts.cx, parts.cy, parts.cz)):
                if n:
                    dst[:n].assign(np.ascontiguousarray(aff[:, c]))
        if parts.life is not None and n:
            parts.life[:n].assign(np.zeros(n, np.float32) if life is None
                                  else np.asarray(life, np.float32).reshape(n))
        parts.n = n

    @staticmethod
    def _download(a, n):
        # Same reason as WarpBackend.to_numpy: on the CPU numpy() is a view of the array's memory.
        # Slice on the device so only the n live slots are transferred and kept; n == 0 is built on the host.
        if not n:
            if a.dtype == wp.vec3:
                return np.empty((0, 3), np.float32)
            return np.empty(0, wp.dtype_to_numpy(a.dtype))
        h = a[:n].numpy()
        return h.copy() if a.device.is_cpu else h

    def get_particles(self, parts):
        n = parts.n
        return self._download(parts.pos, n), self._download(parts.vel, n), self._download(parts.flag, n)

    def get_particle_affine(self, parts):
        if parts.cx is None:
            return None
        n = parts.n
        return np.stack([self._download(a, n) for a in (parts.cx, parts.cy, parts.cz)], axis=1)

    def get_particle_life(self, parts):
        return None if parts.life is None else self._download(parts.life, parts.n)

    def grid_particle_index(self, parts, shape, scale=1.0, transient=False, clamp=False):
        # clamp: file a particle outside the grid under the nearest cell instead of leaving it out.
        dev = self.device
        shape = tuple(int(s) for s in shape)
        cells = shape[0] * shape[1] * shape[2]
        if cells >= 2 ** 31:
            raise ValueError(f"grid_particle_index: {shape} has too many cells for 32-bit sort keys")
        counter = self._scratch_int(shape)
        counter.zero_()
        # The per-substep index uses cached arrays, so it is valid only until the next such call.
        # Mesh indices are once per frame (transient, or finer than the grid) and would pin their
        # arrays through every pressure solve, so they are allocated per call.
        cached = scale == 1.0 and not transient
        # The sort needs room for twice the pairs.
        slots = 2 * parts.n
        if cached:
            total = self._scratch_int((1,), "index_total")
            total.zero_()
            keys = self._slot_scratch(parts, "index_keys", slots, wp.int32)
            sorted_idx = self._slot_scratch(parts, "index_sorted", slots, wp.int32)
        else:
            total = wp.zeros(1, dtype=wp.int32, device=dev)
            keys = wp.empty(max(slots, 1), dtype=wp.int32, device=dev)
            sorted_idx = wp.empty(max(slots, 1), dtype=wp.int32, device=dev)
        sh = wp.vec3i(*shape)
        if parts.n:
            wp.launch(_count_kernel, dim=parts.n, inputs=[parts.pos, parts.flag, parts.n, float(scale),
                                                          int(clamp), sh, counter, total, keys, sorted_idx], device=dev)
        start = (self._scratch_int(shape, "index_start") if cached
                 else wp.empty(shape, dtype=wp.int32, device=dev))
        wp.utils.array_scan(counter.reshape(-1), start.reshape(-1), inclusive=False)
        # A stable sort, so particles within a cell are in ascending particle order on every run.
        # The largest key is `cells`; sorting only its bits skips the empty high passes.
        wp.utils.radix_sort_pairs(keys, sorted_idx, parts.n, 0, cells.bit_length())
        return WarpParticleIndex(start, sorted_idx, total, shape)

    def union_particle_levelset(self, parts, index, phi, radius_factor=1.0, scale=1.0):
        if index.shape != tuple(phi.shape):
            raise ValueError("union_particle_levelset: index and phi shapes differ")
        radius = float(np.float32(0.5 * math.sqrt(3.0) * (radius_factor + 0.01)))
        wp.launch(_union_kernel, dim=phi.shape,
                  inputs=[index.start, index.sorted, index.total, parts.pos, float(scale), phi,
                          radius, int(radius) + 1], device=self.device)

    def improved_particle_levelset(self, parts, index, phi, radius_factor=1.0, smoothen=1,
                                   smoothen_neg=1, t_low=0.4, t_high=3.5, scale=1.0):
        if index.shape != tuple(phi.shape):
            raise ValueError("improved_particle_levelset: index and phi shapes differ")
        dev = self.device
        shape = tuple(phi.shape)
        radius = float(np.float32(0.5 * math.sqrt(3.0) * (radius_factor + 0.01)))
        pacc = self._scratch_vec(shape, "ls_pacc")
        racc = self._scratch_float(shape, "ls_racc")
        wp.launch(_averaged_ls_kernel, dim=shape,
                  inputs=[index.start, index.sorted, index.total, parts.pos, float(scale), phi,
                          pacc, racc, radius, int(radius) + 1], device=dev)
        wp.launch(_correct_ls_kernel, dim=shape,
                  inputs=[phi, pacc, racc, radius, float(t_low), float(t_high)], device=dev)
        self._smooth_ls(phi, smoothen, smoothen_neg)

    def _aniso_scratch(self, n):
        """The centre (vec3) and matrix (mat33) work arrays for n particle slots, 48 B per
        slot: one pair, resized when too small or far too large. Not zeroed, _aniso_kernel
        writes every live slot before _aniso_splat_kernel reads it."""
        pair = self._tmp.get("aniso")
        size = 0 if pair is None else pair[0].shape[0]
        if size < n or size > 4 * max(n, ANISO_MIN_SLOTS):
            pair = None
            self._tmp.pop("aniso", None)  # the old pair is freed before the new one exists
            new = max(int(n * 1.25), ANISO_MIN_SLOTS)
            pair = self._tmp["aniso"] = (wp.empty(new, dtype=wp.vec3, device=self.device),
                                         wp.empty(new, dtype=wp.mat33, device=self.device))
            self._allocated += (new - size) * 48
        return pair

    def anisotropic_particle_levelset(self, parts, phi, sim_shape, radius_factor=1.0, smoothen=1,
                                      smoothen_neg=1, scale=1.0, h=2.0, lam=0.5, kr=4.0, nmin=10,
                                      min_half=0.0):
        dev = self.device
        shape = tuple(phi.shape)
        radius = float(np.float32(0.5 * math.sqrt(3.0) * (radius_factor + 0.01)))
        phi.fill_(radius)
        if parts.n:
            cen, amat = self._aniso_scratch(parts.n)
            index = self.grid_particle_index(parts, sim_shape, 1.0, transient=True)
            wp.launch(_aniso_kernel, dim=parts.n,
                      inputs=[index.start, index.sorted, index.total, parts.pos, parts.flag, parts.n,
                              float(h), float(lam), float(kr), int(nmin), float(scale),
                              radius / min_half if min_half > 0 else 0.0, cen, amat],
                      device=dev)
            wp.launch(_aniso_splat_kernel, dim=parts.n,
                      inputs=[cen, amat, parts.flag, parts.n, radius, phi], device=dev)
        self._smooth_ls(phi, smoothen, smoothen_neg)

    def mirror_particles(self, parts, phi_obs, max_dist=2.0, reach=1.0):
        dev, n = self.device, parts.n
        keep = self._slot_scratch(parts, "mirror_keep", n, wp.int32)
        start = self._slot_scratch(parts, "mirror_start", n, wp.int32)
        total = 0
        if n:
            wp.launch(_mirror_mask_kernel, dim=n,
                      inputs=[parts.pos, parts.flag, phi_obs, float(max_dist), float(reach), keep],
                      device=dev)
            total = self._scan_totals((keep[:n], start[:n]))[0]
        # One reused position/flag set (16 B per slot); vel aliases pos, nothing reads it.
        out = self._tmp.get("mirror_parts")
        need = n + total
        size = 0 if out is None else out.capacity
        if out is None or size < need or size > 4 * max(need, MIN_CAPACITY):
            out = None
            self._tmp.pop("mirror_parts", None)
            cap = max(int(need * 1.25), MIN_CAPACITY)
            p = wp.empty(cap, dtype=wp.vec3, device=dev)
            out = self._tmp["mirror_parts"] = WarpParticles(p, p, wp.empty(cap, dtype=wp.int32, device=dev))
            self._allocated += (cap - size) * 16
        if n:
            wp.launch(_mirror_scatter_kernel, dim=n,
                      inputs=[parts.pos, parts.flag, phi_obs, float(max_dist), float(reach), keep,
                              start, n, out.pos, out.flag],
                      device=dev)
        out.n = need
        return out

    def clip_to_obstacle(self, phi, phi_obs, eps):
        sf = self._size_factor(phi_obs.shape, phi.shape)
        wp.launch(_clip_obstacle_kernel, dim=phi.shape,
                  inputs=[phi, phi_obs, wp.vec3(*sf), wp.vec3(*(sf * np.float32(0.5))), float(eps)],
                  device=self.device)

    def _smooth_ls(self, phi, smoothen, smoothen_neg):
        dev = self.device
        shape = tuple(phi.shape)
        # Mantaflow's buffer swaps: each pass reads `cur`; the positive pass writes a fresh
        # zeroed grid, the negative one compares against the grid before the positive pass.
        cur, other = phi, self._scratch_float(shape, "ls_smooth")
        for it in range(max(int(smoothen), int(smoothen_neg))):
            other.zero_()
            if it < smoothen:
                wp.launch(_smooth_ls_kernel, dim=shape, inputs=[cur, other, 0], device=dev)
                cur, other = other, cur
            if it < smoothen_neg:
                wp.launch(_smooth_ls_kernel, dim=shape, inputs=[cur, other, 1], device=dev)
                cur, other = other, cur
        if cur is not phi:
            wp.copy(phi, cur)
        wp.launch(_set_border_kernel, dim=shape, inputs=[phi, 0.5], device=dev)

    def extrapolate_ls_simple(self, phi, distance=4, inside=False):
        dev = self.device
        tmp = self._scratch_int(phi.shape)
        direction = -1.0 if inside else 1.0
        wp.launch(_ls_mark_kernel, dim=phi.shape, inputs=[phi, tmp, int(inside)], device=dev)
        wp.launch(_ls_first_layer_kernel, dim=phi.shape, inputs=[tmp], device=dev)
        for d in range(2, 1 + int(distance)):
            wp.launch(_ls_extrap_kernel, dim=phi.shape, inputs=[phi, tmp, d, direction], device=dev)
        wp.launch(_ls_remaining_kernel, dim=phi.shape,
                  inputs=[phi, tmp, float(np.float32(direction * (int(distance) + 2)))], device=dev)

    def update_flags_from_levelset(self, flags, phi):
        wp.launch(_flags_from_ls_kernel, dim=flags.shape, inputs=[flags, phi], device=self.device)

    def levelset_volume(self, phi, flags, border=1):
        out = self._result(wp.int64)
        wp.launch(_ls_volume_kernel, dim=phi.shape, inputs=[phi, flags, out, int(border)],
                  device=self.device)
        return float(out.numpy()[0]) / FIXED_ONE

    def source_outflux(self, flags, phi_in, vel, invel=None):
        out = self._result(wp.int64)
        wp.launch(_source_outflux_kernel, dim=flags.shape,
                  inputs=[flags, phi_in, vel, vel if invel is None else invel, int(invel is not None), out],
                  device=self.device)
        return float(out.numpy()[0]) / FIXED_ONE

    # ---- particle/grid transfer ----

    def map_parts_to_mac(self, flags, vel, vel_old, parts, weight):
        dev = self.device
        vel.zero_()
        weight.zero_()
        if parts.n:
            # Its own index: reset_outflow_particles compacts after the level-set one is built.
            # Clamped, because setInterpolMAC also splats a particle outside the grid.
            index = self.grid_particle_index(parts, vel.shape, clamp=True)
            wp.launch(_p2g_kernel, dim=vel.shape,
                      inputs=[index.start, index.sorted, index.total, parts.pos, parts.vel,
                              f4(vel), f4(weight)], device=dev)
        wp.launch(_stomp_divide_kernel, dim=vel.shape, inputs=[vel, weight, vel_old, VECTOR_EPSILON],
                  device=dev)

    def apic_map_parts_to_mac(self, flags, vel, parts):
        if parts.cx is None:
            raise ValueError("apic_map_parts_to_mac: particles were made without apic=True")
        dev = self.device
        mass = self._scratch_vec(vel.shape, "apic_mass")
        mass.zero_()
        vel.zero_()
        if parts.n:
            index = self.grid_particle_index(parts, vel.shape)
            wp.launch(_apic_p2g_kernel, dim=vel.shape,
                      inputs=[index.start, index.sorted, index.total, parts.pos, parts.vel,
                              parts.cx, parts.cy, parts.cz, f4(vel), f4(mass)], device=dev)
        wp.launch(_stomp_divide_kernel, dim=vel.shape, inputs=[vel, mass, vel, VECTOR_EPSILON],
                  device=dev)

    def apic_map_mac_to_parts(self, parts, vel, flags):
        if parts.cx is None:
            raise ValueError("apic_map_mac_to_parts: particles were made without apic=True")
        if parts.n:
            wp.launch(_apic_g2p_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.vel, parts.flag, parts.cx, parts.cy, parts.cz,
                              f4(vel)], device=self.device)

    def flip_velocity_update(self, flags, vel, vel_old, parts, flip_ratio):
        if parts.n:
            wp.launch(_flip_update_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.vel, parts.flag, f4(vel), f4(vel_old),
                              float(np.float32(flip_ratio))], device=self.device)

    def extrapolate_mac_from_weight(self, vel, weight, distance=2):
        dev = self.device
        v4, w4 = f4(vel), f4(weight)
        for c in range(3):
            wp.launch(_weight_reset_kernel, dim=vel.shape, inputs=[w4, c], device=dev)
            for d in range(1, 1 + int(distance)):
                wp.launch(_mac_extrap_weight_kernel, dim=vel.shape, inputs=[v4, w4, d, c], device=dev)

    def extrapolate_mac_simple(self, flags, vel, distance=4, into_obs=False):
        dev = self.device
        tmp = self._scratch_int(vel.shape)
        v4 = f4(vel)
        for c in range(3):
            wp.launch(_mac_mark_kernel, dim=vel.shape, inputs=[flags, tmp, c, int(into_obs)], device=dev)
            for d in range(1, 1 + int(distance)):
                wp.launch(_mac_extrap_kernel, dim=vel.shape, inputs=[v4, tmp, d, c], device=dev)
        src = self._scratch_vec(vel.shape, "mac_extrap")
        wp.copy(src, vel)
        wp.launch(_extrap_into_bnd_kernel, dim=vel.shape, inputs=[flags, vel, src], device=dev)

    def combine_grid_vel(self, vel, weight, combine_vel, phi=None, narrow_band=0.0, thresh=0.0):
        wp.launch(_combine_vels_kernel, dim=vel.shape,
                  inputs=[f4(vel), f4(weight), f4(combine_vel), phi if phi is not None else
                          self._scratch_float((1, 1, 1)), int(phi is not None),
                          float(np.float32(narrow_band)), float(np.float32(thresh))],
                  device=self.device)

    def adjust_number(self, parts, vel, flags, min_particles, max_particles, phi,
                      radius_factor=1.0, narrow_band=-1.0, exclude=None):
        dev = self.device
        shape = tuple(phi.shape)
        surface_ls = float(np.float32(-math.sqrt(3.0) * (radius_factor + 0.01)))
        band = float(np.float32(narrow_band))
        n = parts.n
        cnt = self._scratch_int(shape)
        if n:
            # Its own index: the one built for the level set is stale once outflow compacts.
            index = self.grid_particle_index(parts, shape)
            wp.launch(_adjust_outside_kernel, dim=n, inputs=[parts.pos, parts.flag, wp.vec3i(*shape)],
                      device=dev)
            wp.launch(_adjust_kill_kernel, dim=shape,
                      inputs=[index.start, index.sorted, index.total, parts.pos, parts.flag, phi, cnt,
                              int(max_particles), surface_ls, band], device=dev)
        else:
            cnt.zero_()
        # Role names (int_b, int_c) share buffers with the other methods: each consumes its grids before returning.
        need = self._scratch_int(shape, "int_b").reshape(shape[::-1])
        start = self._scratch_int(shape, "int_c").reshape(shape[::-1])
        wp.launch(_adjust_need_kernel, dim=shape,
                  inputs=[cnt, flags, phi, exclude if exclude is not None else phi,
                          int(exclude is not None), int(min_particles), surface_ls, band, need],
                  device=dev)
        total = self._scan_totals((need.reshape(-1), start.reshape(-1)))[0]
        n_alive = self._compact(parts, total)
        if total:
            wp.launch(_seed_kernel, dim=shape, inputs=[need, start, self._random_stream(3 * total), n_alive,
                                                       f4(vel), parts.pos, parts.vel, parts.flag,
                                                       *self._affine_args(parts)], device=dev)
        parts.n = n_alive + total

    def _compact(self, parts, extra=0):
        """Drop deleted particles (order kept, NEW cleared), leaving room for
        `extra` more. Grows capacity x1.5 at least; shrinks it to 1.25 x need
        (not below MIN_CAPACITY) when less than half would be used. With nothing
        deleted and the capacity kept, works in place. Free slots: flag, affine rows
        and life are zeroed (the affine zeroing is load-bearing, CODE-LIQ-002);
        pos and vel are not, every seeding kernel writes them for a new particle.
        Sets and returns parts.n."""
        dev = self.device
        n = parts.n
        n_alive = 0
        if n:
            alive = self._slot_scratch(parts, "compact_alive", n, wp.int32)
            slot = self._slot_scratch(parts, "compact_slot", n, wp.int32)
            wp.launch(_alive_kernel, dim=n, inputs=[parts.flag, alive], device=dev)
            n_alive = self._scan_totals((alive[:n], slot[:n]))[0]
        need = n_alive + extra
        old_cap = parts.capacity
        if need > old_cap:
            cap = max(need, int(old_cap * 1.5))
        elif need * 2 < old_cap and old_cap > MIN_CAPACITY:
            cap = max(MIN_CAPACITY, int(need * 1.25))  # shrink, with headroom against regrowth
        else:
            cap = old_cap
        if n_alive == n and cap == old_cap:  # nothing to move: only clear NEW, as the copy would
            if n:
                wp.launch(_clear_new_kernel, dim=n, inputs=[parts.flag], device=dev)
            parts.n = n
            return n
        apic = parts.cx is not None
        npos = wp.empty(cap, dtype=wp.vec3, device=dev)
        nvel = wp.empty(cap, dtype=wp.vec3, device=dev)
        nflag = wp.zeros(cap, dtype=wp.int32, device=dev)
        naff = [wp.empty(cap, dtype=wp.vec3, device=dev) for _ in range(3)] if apic else []
        nlife = wp.empty(cap, dtype=wp.float32, device=dev) if parts.life is not None else None
        if n:
            wp.launch(_compact_kernel, dim=n, inputs=[alive, slot, parts.pos, parts.vel, parts.flag,
                                                      npos, nvel, nflag], device=dev)
            for src, dst in zip((parts.cx, parts.cy, parts.cz) if apic else (), naff):
                wp.launch(_compact_vec_kernel, dim=n, inputs=[alive, slot, src, dst], device=dev)
            if nlife is not None:
                wp.launch(_compact_float_kernel, dim=n, inputs=[alive, slot, parts.life, nlife],
                          device=dev)
        parts.pos, parts.vel, parts.flag, parts.life = npos, nvel, nflag, nlife
        if apic:
            parts.cx, parts.cy, parts.cz = naff
        self._allocated += (cap - old_cap) * self._bytes_per_particle(parts)
        parts.n = n_alive
        return n_alive

    # ---- substep pieces ----

    def _random_stream(self, count, shrink=False):
        """RandomStream(PARTICLE_SEED) as at least `count` floats on the GPU (cached). Grows as
        needed. With shrink, also drops a cache far larger than this call and the other callers
        need, so the first-frame fill does not hold its stream for the whole bake."""
        held = 0 if self._seed_stream is None else self._seed_stream.shape[0]
        if not shrink:
            self._seed_recent = max(self._seed_recent, count)
        want = max(count, self._seed_recent)
        size = 0
        if held < count:
            size = max(count, 2 * held, 3 << 16)
        elif shrink and held > max(8 * want, SEED_STREAM_KEEP):
            size = max(want, SEED_STREAM_KEEP)
        if shrink:
            self._seed_recent = 0
        if size:
            if held:  # free first: the new stream starts with the same values
                self._allocated -= self._seed_stream.capacity
                self._seed_stream = None
            self._seed_stream = wp.array(_mt_floats(size), dtype=wp.float32, device=self.device)
            self._allocated += self._seed_stream.capacity
        return self._seed_stream

    @staticmethod
    def _affine_args(parts):
        """cx, cy, cz, use_apic for the seeding kernels (pos stands in without APIC)."""
        if parts.cx is not None:
            return parts.cx, parts.cy, parts.cz, 1
        return parts.pos, parts.pos, parts.pos, 0

    def _grow(self, parts, need):
        """Make room for `need` particles (x1.5 at least), keeping the first n."""
        old_cap = parts.capacity
        if need <= old_cap:
            return
        cap = max(need, int(old_cap * 1.5))
        n = parts.n
        names = (("pos", "vel", "flag") + (("cx", "cy", "cz") if parts.cx is not None else ())
                 + (("life",) if parts.life is not None else ()))
        for name in names:
            old = getattr(parts, name)
            new = wp.zeros(cap, dtype=old.dtype, device=self.device)
            if n:
                wp.copy(new, old, count=n)
            setattr(parts, name, new)
        self._allocated += (cap - old_cap) * self._bytes_per_particle(parts)

    def sample_levelset_with_particles(self, phi, flags, parts, discretization, randomness,
                                       vel_source=None):
        dev = self.device
        vsrc = vel_source if vel_source is not None else self._dummy_vec
        shape = tuple(phi.shape)
        d = int(discretization)
        disp = float(np.float32(1.0 / d))
        jlen = float(np.float32(np.float32(randomness) / np.float32(d)))
        if parts.n:
            wp.launch(_clear_new_kernel, dim=parts.n, inputs=[parts.flag], device=dev)
        kept = self._scratch_int(shape, "int_a").reshape(shape[::-1])  # not liquid_int: the particle index counter uses that
        cand_start = self._scratch_int(shape, "int_b").reshape(shape[::-1])
        kept_start = self._scratch_int(shape, "int_c").reshape(shape[::-1])

        def run(pass_no, rnd, base=0):
            wp.launch(_sample_ls_kernel, dim=shape,
                      inputs=[phi, flags, d, disp, jlen, rnd, cand_start, pass_no, kept,
                              kept_start, base, parts.pos, parts.vel, parts.flag, vsrc,
                              int(vel_source is not None), *self._affine_args(parts)], device=dev)

        def scan_total(start):
            return self._scan_totals((kept.reshape(-1), start.reshape(-1)))[0]

        run(0, self._random_stream(3))
        n_cand = scan_total(cand_start)
        if int(np.prod(shape, dtype=np.int64)) * d ** 3 * 3 >= 2 ** 31:  # int32 scan may have wrapped
            n_cand = int(kept.numpy().sum(dtype=np.int64))
        if 3 * n_cand >= 2 ** 31:  # the kernels index the stream in int32
            raise MemoryError(f"liquid seeding needs {n_cand} candidate positions; "
                              "lower Particles/Cell or the resolution")
        rnd = self._random_stream(3 * n_cand, shrink=True)
        if n_cand == 0:
            return
        run(1, rnd)
        total = scan_total(kept_start)
        if total == 0:
            return
        self._grow(parts, parts.n + total)
        run(2, rnd, parts.n)
        parts.n += total

    def advect_particles(self, parts, flags, vel, dt, delete_in_obstacle=False, skip_new=True):
        if parts.n:
            wp.launch(_advect_rk4_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.flag, flags, f4(vel), float(dt),
                              int(delete_in_obstacle), int(skip_new)], device=self.device)

    def push_out_of_obstacles(self, parts, phi_obs, shift=0.0, thresh=0.0):
        if parts.n:
            wp.launch(_push_out_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.flag, phi_obs, float(shift), float(thresh)],
                      device=self.device)

    def copy_particle_positions(self, parts):
        """The positions before this substep's move, valid until the next call for these
        particles."""
        out = self._slot_scratch(parts, "old_pos", parts.n, wp.vec3)
        if parts.n:
            wp.copy(out, parts.pos, count=parts.n)
        return out, parts.n

    def collide_particles(self, parts, pos_old, meshes, shape, margin=0.1, closed="xXyYzZ", wall=1.0):
        if not parts.n:
            return
        old, n_old = pos_old if pos_old is not None else (parts.pos, 0)
        dev, m = self.device, float(margin)
        for mesh in meshes:
            wp.launch(_collide_mesh_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.vel, parts.flag, old, int(n_old), mesh.mesh.id,
                              wp.vec3(*map(float, mesh.lo)), wp.vec3(*map(float, mesh.hi)), m,
                              int(mesh.two_sided or mesh.orient < 0.0), mesh.diag + 4.0 * m + 4.0],
                      device=dev)
        n, lo = shape, float(wall) + m
        wp.launch(_collide_walls_kernel, dim=parts.n,
                  inputs=[parts.pos, parts.vel, parts.flag, wp.vec3(lo, lo, lo),
                          wp.vec3(n[0] - lo, n[1] - lo, n[2] - lo),
                          wp.vec3i(*(int(a in closed) for a in "xyz")),
                          wp.vec3i(*(int(a in closed) for a in "XYZ"))], device=dev)

    def thin_obstacles(self, meshes, phi_obs, obvel=None, num_obs=None):
        use_vel = obvel is not None and num_obs is not None
        src = self._scratch_float(phi_obs.shape, "thin_src")
        wp.copy(src, phi_obs)
        for mesh in meshes:
            box = self._mesh_box(mesh, phi_obs.shape)
            if box is None:
                continue
            wp.launch(_thin_obstacle_kernel, dim=box[1],
                      inputs=[mesh.mesh.id, box[0], src, phi_obs,
                              obvel if use_vel else self._dummy_vec,
                              num_obs if use_vel else self._dummy_f, int(use_vel)],
                      device=self.device)

    def add_gravity(self, flags, vel, force):
        wp.launch(_gravity_kernel, dim=flags.shape, inputs=[flags, vel, wp.vec3(*force)],
                  device=self.device)

    # ---- emission and obstacles ----

    def flood_fill(self, phi, boundary_width=1, value=-0.5):
        if value >= 0.0:
            raise ValueError("flood_fill: filling the outside needs a negative value")
        dev = self.device
        sh = tuple(phi.shape)
        bnd = int(boundary_width)
        vis = self._scratch_int(sh, "flood")
        changed = self._result(wp.int32)
        wp.launch(_flood_seed_kernel, dim=sh, inputs=[phi, vis, bnd], device=dev)
        # In-place growth spreads several cells per launch; check for
        # convergence every few launches to keep host syncs rare.
        while True:
            changed.zero_()
            for _ in range(8):
                wp.launch(_flood_grow_kernel, dim=sh, inputs=[phi, vis, bnd, changed], device=dev)
            if not int(changed.numpy()[0]):
                break
        wp.launch(_flood_apply_kernel, dim=sh, inputs=[phi, vis, bnd, float(value)], device=dev)

    def subtract_levelset(self, phi, other):
        wp.launch(_subtract_kernel, dim=phi.shape, inputs=[phi, other], device=self.device)

    def update_flags_obs(self, flags, phi_obs, phi_in=None, phi_out=None, boundary_width=1,
                         fractions=None):
        if fractions is not None:
            self._update_flags_obs_frac(flags, fractions, phi_in, phi_out, boundary_width)
            return
        wp.launch(_flags_obs_kernel, dim=flags.shape,
                  inputs=[flags, phi_obs, phi_obs if phi_in is None else phi_in,
                          phi_obs if phi_out is None else phi_out, int(phi_in is not None),
                          int(phi_out is not None), int(boundary_width)], device=self.device)

    def reset_outflow_particles(self, flags, phi, parts):
        if parts.n:
            wp.launch(_outflow_kill_kernel, dim=parts.n, inputs=[parts.pos, parts.flag, flags],
                      device=self.device)
            self._compact(parts)
        wp.launch(_outflow_cells_kernel, dim=flags.shape, inputs=[flags, phi], device=self.device)

    def _result(self, dtype, size=1):
        """The persistent, zeroed result array of this type, for kernels that return a few
        values. One user at a time: read it before the next call."""
        key = ("result", size, dtype)
        if key not in self._tmp:
            self._tmp[key] = self._new((size,), dtype)
        out = self._tmp[key]
        out.zero_()
        return out

    def _scan_totals(self, *pairs):
        """Exclusive scan of each flat (counts, starts) pair; returns the total of each pair's
        counts with one host sync. At most two pairs."""
        out = self._result(wp.int64, 2)
        for i, (counts, starts) in enumerate(pairs):
            wp.utils.array_scan(counts, starts, inclusive=False)
            wp.launch(_scan_total_kernel, dim=1, inputs=[counts, starts, out, i], device=self.device)
        return [int(v) for v in out.numpy()[:len(pairs)]]

    def _slot_scratch(self, parts, name, n, dtype):
        """A work array of at least n slots kept with this particle set, not zeroed. Reallocated
        only when too small or far too large, so the first-frame seeding peak is not held."""
        arr = parts.scratch.get(name)
        old = 0 if arr is None else arr.capacity
        if arr is None or arr.shape[0] < n or arr.shape[0] > 4 * max(n, SLOT_SCRATCH_MIN):
            arr = None
            parts.scratch.pop(name, None)  # freed before the new one exists
            arr = parts.scratch[name] = wp.empty(max(int(n * 1.25), SLOT_SCRATCH_MIN), dtype=dtype,
                                                 device=self.device)
            self._allocated += arr.capacity - old
        return arr

    def release_scratch(self, *names):
        """Free the scratch grids stored under these role names (used at initialisation only)."""
        for key in [k for k in self._tmp if isinstance(k, tuple) and len(k) == 2 and k[1] in names]:
            arr = self._tmp.pop(key)
            self._allocated -= arr.capacity

    def _scratch_int(self, shape, name="liquid_int"):
        key = (tuple(shape), name)
        if key not in self._tmp:
            self._tmp[key] = self._new(tuple(shape), wp.int32)
        return self._tmp[key]

    def _scratch_vec(self, shape, name):
        key = (tuple(shape), name)
        if key not in self._tmp:
            self._tmp[key] = self._new(tuple(shape), wp.vec3)
        return self._tmp[key]

    def _scratch_float(self, shape, name="liquid_float"):
        key = (tuple(shape), name)
        if key not in self._tmp:
            self._tmp[key] = self._new(tuple(shape), wp.float32)
        return self._tmp[key]
