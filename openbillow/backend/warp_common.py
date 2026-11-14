# SPDX-License-Identifier: GPL-3.0-or-later
"""Small Warp helpers shared by the liquid, secondary, viscosity, diffusion and pressure modules
(CODE-LIQ-012): border test, neighbour offsets, interpolation, flat index. Functions only, no kernels,
so they are compiled under the options of the module that calls them."""
import warp as wp

BLOCK = 256  # block size of the tile-reduction kernels


@wp.func
def border(i: int, j: int, k: int, nx: int, ny: int, nz: int):
    return i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1


@wp.func
def nb6(n: int):
    # +x, -x, +y, -y, +z, -z
    s = 1 - 2 * (n % 2)
    a = n / 2
    return wp.vec3i(wp.where(a == 0, s, 0), wp.where(a == 1, s, 0), wp.where(a == 2, s, 0))


@wp.func
def cell_of(p: wp.vec3, shape: wp.vec3i):
    # toVec3iFloor + isInBounds; returns (-1,-1,-1) when outside.
    c = wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))
    if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] >= shape[0] or c[1] >= shape[1] or c[2] >= shape[2]:
        c = wp.vec3i(-1, -1, -1)
    return c


@wp.func
def axis(x: float, n: int):
    # One axis of interpol.h BUILD_INDEX: lower sample and upper weight, clamped to the grid.
    i = int(x)
    s = x - float(i)
    if x < 0.0:
        i = 0
        s = 0.0
    if i >= n - 1:
        i = n - 2
        s = 1.0
    return i, s


@wp.func
def shift(p: wp.vec3, c: int, d: int):
    # MAC component c is stored at the face centre: unshifted along c, cell-centred elsewhere.
    return wp.where(d == c, p[d], p[d] - 0.5)


@wp.func
def sample(a: wp.array3d(dtype=wp.float32), p: wp.vec3):
    # Grid<Real>::getInterpolated (interpol)
    i, s = axis(p[0] - 0.5, a.shape[0])
    j, t = axis(p[1] - 0.5, a.shape[1])
    k, f = axis(p[2] - 0.5, a.shape[2])
    return (((a[i, j, k] * (1.0 - t) + a[i, j + 1, k] * t) * (1.0 - s)
             + (a[i + 1, j, k] * (1.0 - t) + a[i + 1, j + 1, k] * t) * s) * (1.0 - f)
            + ((a[i, j, k + 1] * (1.0 - t) + a[i, j + 1, k + 1] * t) * (1.0 - s)
               + (a[i + 1, j, k + 1] * (1.0 - t) + a[i + 1, j + 1, k + 1] * t) * s) * f)


@wp.func
def sample_vec(a: wp.array3d(dtype=wp.vec3), p: wp.vec3):
    # Grid<Vec3>::getInterpolated (cell-centred)
    i, s = axis(p[0] - 0.5, a.shape[0])
    j, t = axis(p[1] - 0.5, a.shape[1])
    k, f = axis(p[2] - 0.5, a.shape[2])
    return (((a[i, j, k] * (1.0 - t) + a[i, j + 1, k] * t) * (1.0 - s)
             + (a[i + 1, j, k] * (1.0 - t) + a[i + 1, j + 1, k] * t) * s) * (1.0 - f)
            + ((a[i, j, k + 1] * (1.0 - t) + a[i, j + 1, k + 1] * t) * (1.0 - s)
               + (a[i + 1, j, k + 1] * (1.0 - t) + a[i + 1, j + 1, k + 1] * t) * s) * f)


@wp.func
def mac_comp(v: wp.array4d(dtype=wp.float32), c: int, p: wp.vec3):
    # Component c of interpolMAC.
    i, s = axis(shift(p, c, 0), v.shape[0])
    j, t = axis(shift(p, c, 1), v.shape[1])
    k, f = axis(shift(p, c, 2), v.shape[2])
    return ((((v[i, j, k, c] * (1.0 - t) + v[i, j + 1, k, c] * t) * (1.0 - s)
              + (v[i + 1, j, k, c] * (1.0 - t) + v[i + 1, j + 1, k, c] * t) * s) * (1.0 - f))
            + (((v[i, j, k + 1, c] * (1.0 - t) + v[i, j + 1, k + 1, c] * t) * (1.0 - s)
                + (v[i + 1, j, k + 1, c] * (1.0 - t) + v[i + 1, j + 1, k + 1, c] * t) * s) * f))


@wp.func
def mac_interp(v: wp.array4d(dtype=wp.float32), p: wp.vec3):
    return wp.vec3(mac_comp(v, 0, p), mac_comp(v, 1, p), mac_comp(v, 2, p))


def f4(a):
    """(nx,ny,nz,3) float32 view of a vec3 grid, for per-component atomics and indexing."""
    return wp.array(ptr=a.ptr, dtype=wp.float32, shape=(*a.shape, 3), device=a.device)


@wp.func
def unravel(t: int, ny: int, nz: int):
    return wp.vec3i(t // (ny * nz), (t // nz) % ny, t % nz)
