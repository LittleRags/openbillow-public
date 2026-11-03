# SPDX-License-Identifier: GPL-3.0-or-later
# NumPy port of Mantaflow's wavelet noise (noisefield.{h,cpp}, randomstream.h).
# Original: MantaFlow fluid solver framework, Copyright 2011 Tobias Pfaff,
# Nils Thuerey, Apache License 2.0.
#
# Float32 arithmetic follows the C++ operation order so the tile matches
# Blender's waveletNoiseTile.bin bit for bit. The evaluation functions are the
# reference for the Warp kernels, not the fast path.
import functools

import numpy as np

TILE_N = 128
TILE_SEED = 13322223          # WaveletNoiseField::randomSeed
BLENDER_NOISE_SEED = 265      # smoke_script.h: create(NoiseField, fixedSeed=265)

_F = np.float32
_A_COEFFS = np.array([
    0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
    -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340,
    0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
    0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334],
    dtype=_F)
_P_COEFFS = (0.25, 0.75, 0.75, 0.25)


def _mt_raw(seed, n):
    """MTRand(seed).randInt() x n. MTRand is plain MT19937 with init_genrand."""
    rs = np.random.RandomState(seed)
    return rs._bit_generator.random_raw(n).astype(np.float64)


def rand_norm_stream(seed, n):
    """n values of RandomStream(seed).getRandNorm(0, 1), as float32."""
    u = _mt_raw(seed, 2 * n)
    dbl_exc = (u[0::2] + 0.5) * (1.0 / 4294967296.0)
    exc = u[1::2] * (1.0 / 4294967296.0)
    r = np.sqrt(-2.0 * np.log(1.0 - dbl_exc))
    phi = 2.0 * 3.14159265358979323846264338328 * exc
    return (r * np.cos(phi)).astype(_F)


def seed_offset(seed=BLENDER_NOISE_SEED):
    """WaveletNoiseField::mSeedOffset = RandomStream(seed).getVec3Norm()."""
    v = (_mt_raw(seed, 3) * (1.0 / 4294967295.0)).astype(_F)
    l = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    eps2 = _F(1e-6) * _F(1e-6)
    if l <= eps2:
        return np.zeros(3, _F)
    if abs(float(l) - 1.0) < eps2:
        return v
    inv = 1.0 / float(np.sqrt(l))
    return (v.astype(np.float64) * inv).astype(_F)


def _down(x, axis, neumann):
    """downsample / downsampleNeumann along axis: n -> n//2 (float32 sums)."""
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    i2 = 2 * np.arange(n // 2)
    acc = np.zeros(x.shape[:-1] + (n // 2,), _F)
    for j in range(32):
        k = i2 - 16 + j
        k = np.clip(k, 0, n - 1) if neumann else k & 127
        acc += _A_COEFFS[j] * x[..., k]
    return np.moveaxis(acc, -1, axis)


def _up(x, axis, n, neumann):
    """upsample / upsampleNeumann along axis: n//2 -> n. The C++ adds the double
    product 0.5 * pp * from to a float, so each step is summed in double and
    rounded to float."""
    x = np.moveaxis(x, axis, -1)
    h = n // 2
    ih = np.arange(n) // 2
    acc = np.zeros(x.shape[:-1] + (n,), _F)
    for j in range(4):
        k = ih - 1 + j
        k = np.clip(k, 0, h - 1) if neumann else k % h
        acc = (acc.astype(np.float64) + (0.5 * _P_COEFFS[j]) * x[..., k].astype(np.float64)).astype(_F)
    return np.moveaxis(acc, -1, axis)


def generate_tile():
    """WaveletNoiseField::generateTile -> float32 (3, z, y, x), 128^3 per tile."""
    n = TILE_N
    noise = rand_norm_stream(TILE_SEED, 3 * n ** 3).reshape(3, n, n, n)
    t = noise
    for axis in (3, 2, 1):                 # x, y, z passes
        t = _up(_down(t, axis, False), axis, n, False)
    noise = noise - t
    off = n // 2
    if off % 2 == 0:
        off += 1
    # temp13 is filled in (tile, ix, iy, iz) order but added in (tile, z, y, x)
    # order, so the shifted copy is added with x and z swapped.
    shifted = np.roll(noise, (-off, -off, -off), axis=(1, 2, 3))
    noise = noise + shifted.transpose(0, 3, 2, 1)
    return np.ascontiguousarray(noise)


@functools.lru_cache(maxsize=1)
def get_tile():
    """The tile, generated once per process (~2 s, so no disk cache)."""
    return generate_tile()


def _wnoise_vec(p, data):
    """WNoiseVec: gradient of the quadratic B-spline noise. p: (N, 3) float32."""
    half = _F(0.5)
    ph = p - half
    mid = np.ceil(ph).astype(np.int64)
    t = mid.astype(_F) - ph
    one = _F(1.0)
    dw = np.stack([-t, _F(2.0) * t - one, one - t], axis=-1)        # (N, 3 axes, 3)
    w0 = t * t * half
    w2 = (one - t) * (one - t) * half
    w = np.stack([w0, one - w0 - w2, w2], axis=-1)
    out = np.zeros(p.shape, _F)
    for c in range(3):
        wx = dw[:, 0] if c == 0 else w[:, 0]
        wy = dw[:, 1] if c == 1 else w[:, 1]
        wz = dw[:, 2] if c == 2 else w[:, 2]
        acc = np.zeros(len(p), _F)
        for z in range(3):
            zc = (mid[:, 2] + z - 1) & 127
            for y in range(3):
                yc = (mid[:, 1] + y - 1) & 127
                for x in range(3):
                    xc = (mid[:, 0] + x - 1) & 127
                    acc += (wx[:, x] * wy[:, y] * wz[:, z]) * data[zc, yc, xc]
        out[:, c] = acc
    return out


def evaluate_curl(tile, pos, gs_inv, seed_off, time, pos_scale, pos_offset=(0.0, 0.0, 0.0)):
    """WaveletNoiseField::evaluateCurl for points pos (N, 3), in grid cells.
    gs_inv = 1 / max(grid size); time = solver time * dx * timeAnim;
    pos_scale per axis (Blender: base_res / noise_pos_scale)."""
    p = np.asarray(pos, _F) * _F(gs_inv)
    p = p + np.asarray(seed_off, _F)
    p = p + _F(time)
    p = p * np.asarray(pos_scale, _F)
    p = p + np.asarray(pos_offset, _F)
    d0, d1, d2 = (_wnoise_vec(p, tile[i]) for i in range(3))
    return np.stack([d0[:, 1] - d1[:, 2], d2[:, 2] - d0[:, 0], d1[:, 0] - d2[:, 1]], axis=-1)


def compute_coefficients(grid):
    """WaveletNoiseField::computeCoefficients (3D) on a float32 grid indexed
    [i, j, k]. Returns the new grid: smoothed sqrt|residual| inside, input on
    the one-cell border."""
    g = np.asarray(grid, _F)
    t = g
    for axis in (0, 1, 2):                 # x, y, z passes
        t = _up(_down(t, axis, True), axis, g.shape[axis], True)
    r = np.sqrt(np.abs(g - t))
    res = r[:-2, 1:-1, 1:-1] + r[2:, 1:-1, 1:-1]
    res = res + (r[1:-1, :-2, 1:-1] + r[1:-1, 2:, 1:-1])
    res = res + (r[1:-1, 1:-1, :-2] + r[1:-1, 1:-1, 2:])
    out = g.copy()
    out[1:-1, 1:-1, 1:-1] = res * _F(1.0 / 6.0)
    return out


def uv_weight(time, dt, reset_time, index, num_uvs):
    """Mantaflow updateUvWeight (grid.cpp) on the host, in float32 like the C++:
    the weight of UV grid `index` and whether that grid resets (its ramp wrapped
    during the last step). time/dt are the low-res solver's time and step."""
    f = _F
    t, dt, rt = f(time), f(dt), f(reset_time)
    time_off = rt / f(num_uvs)

    def grid_time(x):
        return np.fmod(x / rt, f(1.0))

    def ramp(x):
        w = f(2.0) * x
        return f(2.0) - w if w > 1.0 else w

    last = grid_time(t + f(index) * time_off - dt)
    curr = grid_time(t + f(index) * time_off)
    w = ramp(curr)
    total = f(0.0)
    for i in range(num_uvs):
        total = total + ramp(grid_time(t + f(i) * time_off))
    w = f(1.0) if total <= f(1e-6) else w / total
    return float(w), bool(curr < last)
