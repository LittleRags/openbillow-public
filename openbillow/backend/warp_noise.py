# SPDX-License-Identifier: GPL-3.0-or-later
"""Warp kernels for the noise (upres) pass: Mantaflow plugin/waveletturbulence.cpp,
noisefield.{h,cpp} (evaluateCurl, computeCoefficients), the UV-grid helpers of
grid.cpp, and MacCormack advection of cell-centred Vec3 grids (advection.cpp).

Fused multiply-add is off in this module, so float32 results follow the C++
operation order (and openbillow/noise.py) exactly.
"""
import warp as wp

from .warp_backend import (FLT_MAX, FLUID, FLUID_OR_EMPTY, OBSTACLE, _at_flag, _border,
                           _centered_vel, _sample, _sample_comp)

wp.set_module_options({"fuse_fp": False, "enable_backward": False})


@wp.func
def _sample_vec(a: wp.array3d(dtype=wp.vec3), p: wp.vec3):
    # interpol<Vec3>: the scalar stencil per component
    return wp.vec3(_sample_comp(a, 0, p), _sample_comp(a, 1, p), _sample_comp(a, 2, p))


@wp.func
def _index_pos(i: int, j: int, k: int, sf: wp.vec3):
    return wp.cw_mul(wp.vec3(float(i), float(j), float(k)), sf)


# ---- interpolation between grid sizes (orderSpace 1) ----------------------

@wp.kernel
def _interpolate_grid_kernel(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32),
                             sf: wp.vec3, off: wp.vec3):
    # knInterpolateGridTempl
    i, j, k = wp.tid()
    dst[i, j, k] = _sample(src, _index_pos(i, j, k, sf) + off)


@wp.kernel
def _interpolate_mac_kernel(src: wp.array3d(dtype=wp.vec3), dst: wp.array3d(dtype=wp.vec3),
                            sf: wp.vec3, off: wp.vec3):
    # KnInterpolateMACGrid: interpolMAC(pos - 0.5 e_c)[c] is the cell-centred
    # stencil of component c at pos, so values stay per low-res cell.
    i, j, k = wp.tid()
    dst[i, j, k] = _sample_vec(src, _index_pos(i, j, k, sf) + off)


# ---- MacCormack advection of a cell-centred Vec3 grid ----------------------

@wp.kernel
def _sl_vec_kernel(vel: wp.array3d(dtype=wp.vec3), src: wp.array3d(dtype=wp.vec3),
                   dst: wp.array3d(dtype=wp.vec3), dt: float):
    # SemiLagrange<Vec3>; border layer zero as in a fresh grid.
    i, j, k = wp.tid()
    if _border(i, j, k, dst.shape[0], dst.shape[1], dst.shape[2]):
        dst[i, j, k] = wp.vec3()
        return
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) - _centered_vel(vel, i, j, k) * dt
    dst[i, j, k] = _sample_vec(src, p)


@wp.func
def _clamp_vec(flags: wp.array3d(dtype=wp.int32), orig: wp.array3d(dtype=wp.vec3),
               d: wp.vec3, fwd: wp.vec3, p: wp.vec3i, off: wp.vec3, mode: int):
    # doClampComponent<Vec3>: per-component range; mode 2 falls back to fwd
    # as a whole when any component is out of range.
    nx = orig.shape[0]
    ny = orig.shape[1]
    nz = orig.shape[2]
    lo = wp.vec3(FLT_MAX)
    hi = wp.vec3(-FLT_MAX)
    have = bool(False)
    pf = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
    for l in range(2):
        if l == 0 or mode == 1:
            q = pf - off
            if l == 1:
                q = pf + off
            i0 = wp.clamp(int(q[0]), 0, nx - 2)
            j0 = wp.clamp(int(q[1]), 0, ny - 2)
            k0 = wp.clamp(int(q[2]), 0, nz - 2)
            for n in range(8):
                ii = i0 + (n & 1)
                jj = j0 + ((n >> 1) & 1)
                kk = k0 + ((n >> 2) & 1)
                if (flags[ii, jj, kk] & FLUID_OR_EMPTY) != 0:
                    v = orig[ii, jj, kk]
                    lo = wp.min(lo, v)
                    hi = wp.max(hi, v)
                    have = True
    r = d
    if not have:
        r = fwd
    elif mode == 1:
        r = wp.vec3(wp.clamp(d[0], lo[0], hi[0]), wp.clamp(d[1], lo[1], hi[1]), wp.clamp(d[2], lo[2], hi[2]))
    elif (d[0] < lo[0] or d[0] > hi[0] or d[1] < lo[1] or d[1] > hi[1]
          or d[2] < lo[2] or d[2] > hi[2]):
        r = fwd
    return r


@wp.kernel
def _mc_vec_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                   orig: wp.array3d(dtype=wp.vec3), fwd: wp.array3d(dtype=wp.vec3),
                   bwd: wp.array3d(dtype=wp.vec3), dt: float, strength: float, mode: int):
    # MacCormackCorrect<Vec3> + MacCormackClamp<Vec3>, written over bwd (as _mc_cell_kernel).
    i, j, k = wp.tid()
    f = fwd[i, j, k]
    d = f
    if (flags[i, j, k] & FLUID) != 0:
        d = f + strength * 0.5 * (orig[i, j, k] - bwd[i, j, k])
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    if not _border(i, j, k, nx, ny, nz):
        off = _centered_vel(vel, i, j, k) * dt
        d = _clamp_vec(flags, orig, d, f, wp.vec3i(i, j, k), off, mode)
        if mode == 1:
            pc = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
            a = pc - off
            b = pc + off
            ai = wp.vec3i(int(a[0]), int(a[1]), int(a[2]))
            bi = wp.vec3i(int(b[0]), int(b[1]), int(b[2]))
            if (ai[0] < 0 or ai[1] < 0 or ai[2] < 0 or bi[0] < 0 or bi[1] < 0 or bi[2] < 0
                    or ai[0] > nx - 1 or ai[1] > ny - 1 or ai[2] > nz - 1
                    or bi[0] > nx - 1 or bi[1] > ny - 1 or bi[2] > nz - 1):
                d = f
            elif (_at_flag(flags, ai) & OBSTACLE) != 0 or (_at_flag(flags, bi) & OBSTACLE) != 0:
                d = f
    bwd[i, j, k] = d


# ---- UV grids (grid.cpp) ---------------------------------------------------

@wp.kernel
def _reset_uv_kernel(uv: wp.array3d(dtype=wp.vec3), off: wp.vec3):
    # knResetUvGrid
    i, j, k = wp.tid()
    uv[i, j, k] = wp.vec3(float(i), float(j), float(k)) + off


@wp.kernel
def _set_uv_weight_kernel(uv: wp.array3d(dtype=wp.vec3), w: float):
    # updateUvWeight stores the weight in uv[0] (a border cell).
    uv[0, 0, 0] = wp.vec3(w, 0.0, 0.0)


# ---- energy and its extrapolation ------------------------------------------

@wp.kernel
def _energy_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                   energy: wp.array3d(dtype=wp.float32)):
    # KnApplyComputeEnergy. The outer layer is never FLUID (init_domain), so
    # the far-face reads of getCentered stay in range; clamped to be safe.
    i, j, k = wp.tid()
    e = float(0.0)
    if (flags[i, j, k] & FLUID) != 0:
        i1 = wp.min(i + 1, vel.shape[0] - 1)
        j1 = wp.min(j + 1, vel.shape[1] - 1)
        k1 = wp.min(k + 1, vel.shape[2] - 1)
        v = wp.vec3(0.5 * (vel[i, j, k][0] + vel[i1, j, k][0]),
                    0.5 * (vel[i, j, k][1] + vel[i, j1, k][1]),
                    0.5 * (vel[i, j, k][2] + vel[i, j, k1][2]))
        e = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    energy[i, j, k] = e


@wp.func
def _nb6(n: int):
    # +x, -x, +y, -y, +z, -z (extrapolSimpleFlagsHelper order)
    s = 1 - 2 * (n % 2)
    a = n / 2
    return wp.vec3i(wp.where(a == 0, s, 0), wp.where(a == 1, s, 0), wp.where(a == 2, s, 0))


@wp.kernel
def _extrap_init_kernel(flags: wp.array3d(dtype=wp.int32), flag_from: int,
                        tmp: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    tmp[i, j, k] = wp.where((flags[i, j, k] & flag_from) != 0, 1, 0)


# One distance layer. Cells written in this pass get tmp = d + 1 and are only
# read where tmp == d, so the serial C++ loop and this parallel one agree.
@wp.kernel
def _extrap_int_kernel(flags: wp.array3d(dtype=wp.int32), val: wp.array3d(dtype=wp.int32),
                       tmp: wp.array3d(dtype=wp.int32), d: int, flag_to: int):
    i, j, k = wp.tid()
    if _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        return
    if tmp[i, j, k] != 0 or (flags[i, j, k] & flag_to) == 0:
        return
    p = wp.vec3i(i, j, k)
    nbs = int(0)
    s = int(0)
    for n in range(6):
        q = p + _nb6(n)
        if tmp[q[0], q[1], q[2]] == d:
            s += val[q[0], q[1], q[2]]
            nbs += 1
    if nbs > 0:
        tmp[i, j, k] = d + 1
        val[i, j, k] = s / nbs


@wp.kernel
def _extrap_float_kernel(flags: wp.array3d(dtype=wp.int32), val: wp.array3d(dtype=wp.float32),
                         tmp: wp.array3d(dtype=wp.int32), d: int, flag_to: int):
    i, j, k = wp.tid()
    if _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        return
    if tmp[i, j, k] != 0 or (flags[i, j, k] & flag_to) == 0:
        return
    p = wp.vec3i(i, j, k)
    nbs = int(0)
    s = float(0.0)
    for n in range(6):
        q = p + _nb6(n)
        if tmp[q[0], q[1], q[2]] == d:
            s += val[q[0], q[1], q[2]]
            nbs += 1
    if nbs > 0:
        tmp[i, j, k] = d + 1
        val[i, j, k] = s / float(nbs)


# ---- wavelet coefficients (WaveletNoiseField::computeCoefficients) ----------

@wp.kernel
def _wl_down_kernel(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32),
                    axis: int, a: wp.array(dtype=wp.float32)):
    # downsampleNeumann along `axis`: dst has half the cells there.
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    n = src.shape[axis]
    acc = float(0.0)
    for m in range(32):
        q = p
        q[axis] = wp.clamp(2 * p[axis] - 16 + m, 0, n - 1)
        acc += a[m] * src[q[0], q[1], q[2]]
    dst[i, j, k] = acc


@wp.kernel
def _wl_up_kernel(src: wp.array3d(dtype=wp.float32), dst: wp.array3d(dtype=wp.float32), axis: int):
    # upsampleNeumann: the C++ adds a double product to a float, so each step
    # is summed in double and rounded to float.
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    h = src.shape[axis]
    acc = float(0.0)
    for m in range(4):
        q = p
        q[axis] = wp.clamp(p[axis] / 2 - 1 + m, 0, h - 1)
        c = wp.where(m == 0 or m == 3, wp.float64(0.125), wp.float64(0.375))
        acc = wp.float32(wp.float64(acc) + c * wp.float64(src[q[0], q[1], q[2]]))
    dst[i, j, k] = acc


@wp.kernel
def _wl_residual_kernel(g: wp.array3d(dtype=wp.float32), t: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    t[i, j, k] = wp.sqrt(wp.abs(g[i, j, k] - t[i, j, k]))


@wp.kernel
def _wl_smooth_kernel(r: wp.array3d(dtype=wp.float32), g: wp.array3d(dtype=wp.float32), sixth: float):
    # Mean of the 6 neighbours' sqrt|residual|; the border keeps the input.
    i, j, k = wp.tid()
    if _border(i, j, k, g.shape[0], g.shape[1], g.shape[2]):
        return
    s = r[i - 1, j, k] + r[i + 1, j, k]
    s = s + (r[i, j - 1, k] + r[i, j + 1, k])
    s = s + (r[i, j, k - 1] + r[i, j, k + 1])
    g[i, j, k] = s * sixth


# ---- wavelet noise curl (noisefield.h WNoiseVec, evaluateCurl) -------------

@wp.func
def _eval_curl(tile: wp.array3d(dtype=wp.vec3), pos: wp.vec3, gs_inv: float, seed_off: wp.vec3,
               time: float, pos_scale: wp.vec3):
    # The tile holds the three noise tiles as one Vec3 per texel, [z, y, x].
    p = pos * gs_inv
    p = p + seed_off
    p = p + wp.vec3(time)
    p = wp.cw_mul(p, pos_scale)
    ph = p - wp.vec3(0.5)
    mid = wp.vec3i(int(wp.ceil(ph[0])), int(wp.ceil(ph[1])), int(wp.ceil(ph[2])))
    t = wp.vec3(float(mid[0]), float(mid[1]), float(mid[2])) - ph
    one = wp.vec3(1.0)
    omt = one - t
    w0 = wp.cw_mul(t, t) * 0.5
    w2 = wp.cw_mul(omt, omt) * 0.5
    w1 = one - w0 - w2
    dw1 = 2.0 * t - one
    # stencil weights per axis, indexed by the stencil offset 0..2
    wx = wp.vec3(w0[0], w1[0], w2[0])
    wy = wp.vec3(w0[1], w1[1], w2[1])
    wz = wp.vec3(w0[2], w1[2], w2[2])
    dx = wp.vec3(-t[0], dw1[0], omt[0])
    dy = wp.vec3(-t[1], dw1[1], omt[1])
    dz = wp.vec3(-t[2], dw1[2], omt[2])
    # gx[n] = d/dx of tile n, etc.
    gx = wp.vec3()
    gy = wp.vec3()
    gz = wp.vec3()
    for z in range(3):
        zc = (mid[2] + z - 1) & 127
        for y in range(3):
            yc = (mid[1] + y - 1) & 127
            for x in range(3):
                xc = (mid[0] + x - 1) & 127
                v = tile[zc, yc, xc]
                gx = gx + (dx[x] * wy[y] * wz[z]) * v
                gy = gy + (wx[x] * dy[y] * wz[z]) * v
                gz = gz + (wx[x] * wy[y] * dz[z]) * v
    # (d0.y - d1.z, d2.z - d0.x, d1.x - d2.y)
    return wp.vec3(gy[0] - gz[1], gz[2] - gx[0], gx[1] - gy[2])


@wp.kernel
def _apply_noise_kernel(flags: wp.array3d(dtype=wp.int32), target: wp.array3d(dtype=wp.vec3),
                        tile: wp.array3d(dtype=wp.vec3), weight: wp.array3d(dtype=wp.float32),
                        uv: wp.array3d(dtype=wp.vec3), interp: int, sf: wp.vec3, scale: float,
                        scale_spatial: float, gs_inv: float, seed_off: wp.vec3, time: float,
                        pos_scale: wp.vec3):
    # knApplyNoiseVec3 with weight and uv (low-res, interpolated when the sizes differ)
    i, j, k = wp.tid()
    if (flags[i, j, k] & FLUID) == 0:
        return
    w = float(0.0)
    pos = wp.vec3()
    if interp != 0:
        q = _index_pos(i, j, k, sf)
        w = _sample(weight, q)
        pos = wp.cw_div(_sample_vec(uv, q), sf)
    else:
        w = weight[i, j, k]
        pos = uv[i, j, k]
    pos = pos * scale_spatial
    n = _eval_curl(tile, pos, gs_inv, seed_off, time, pos_scale) * scale * w
    target[i, j, k] = target[i, j, k] + n
