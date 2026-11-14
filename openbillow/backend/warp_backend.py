# SPDX-License-Identifier: GPL-3.0-or-later
"""NVIDIA Warp backend.

Grid handles are plain `wp.array`s on the backend's device, told apart by
dtype: float32 (nx,ny,nz) scalar grids, vec3 (nx,ny,nz) MAC grids and int32
(nx,ny,nz) flag grids. Warp arrays are row-major, so k is contiguous, as in
Mantaflow.
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

wp.config.log_level = wp.LOG_WARNING  # no banner or per-module compile lines in solver logs

from .base import (Backend, BackendMemory, FlagGrid, Flags, Grid, MACGrid,
                   PressureStats, ScalarGrid, Shape, Vec3, VectorGrid)
from .warp_flame import FlameMixin
from .warp_mixfrac import MixFracMixin
from .warp_fractions import FractionsMixin
from .warp_guiding import GuidingMixin
from .warp_liquid import LiquidMixin
from .warp_secondary import SecondaryMixin
from .warp_pressure import MGPressureSolver, _curvature_kernel
from .warp_diffusion import DiffusionSolver, gas_kelvin_kernel
from .warp_viscosity import ViscositySolver
from .warp_forcefields import compute_forces_kernel, noise_hash, pack_fields
from ..estimate import RAM_RESERVE, default_limit_mb, managed_budget, system_memory  # noqa: F401 - system_memory re-exported
from ..noise import _A_COEFFS as WAVELET_A_COEFFS, uv_weight

vec6i = wp.types.vector(length=6, dtype=wp.int32)

# Flag bits as kernel constants.
FLUID = wp.constant(int(Flags.FLUID))
EMPTY = wp.constant(int(Flags.EMPTY))
INFLOW = wp.constant(int(Flags.INFLOW))
OUTFLOW = wp.constant(int(Flags.OUTFLOW))
OBSTACLE = wp.constant(int(Flags.OBSTACLE))
STICK = wp.constant(int(Flags.STICK))
FLUID_OR_EMPTY = wp.constant(int(Flags.FLUID | Flags.EMPTY))
FLT_MAX = wp.constant(3.402823466e38)
# Cells that fillGrid() leaves alone.
KEEP = wp.constant(int(Flags.OBSTACLE | Flags.INFLOW | Flags.OUTFLOW | Flags.OPEN))


# ---- reductions -------------------------------------------------------------
# One thread per (i, j) column walks k, then one atomic per column.

@wp.kernel
def _max_abs_kernel(a: wp.array3d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    m = float(0.0)
    for k in range(a.shape[2]):
        x = wp.abs(a[i, j, k])
        if not wp.isfinite(x):  # Warp's max is fmax, which skips NaN: report it as inf
            x = wp.inf
        m = wp.max(m, x)
    wp.atomic_max(out, 0, m)


@wp.kernel
def _max_length_sq_kernel(v: wp.array3d(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    m = float(0.0)
    for k in range(v.shape[2]):
        x = wp.length_sq(v[i, j, k])
        if not wp.isfinite(x):  # see _max_abs_kernel
            x = wp.inf
        m = wp.max(m, x)
    wp.atomic_max(out, 0, m)


@wp.kernel
def _join_kernel(dst: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.float32), keep_max: int):
    i, j, k = wp.tid()
    if keep_max:
        dst[i, j, k] = wp.max(dst[i, j, k], src[i, j, k])
    else:
        dst[i, j, k] = wp.min(dst[i, j, k], src[i, j, k])


@wp.kernel
def _mult_const_kernel(g: wp.array3d(dtype=wp.float32), a: float):
    i, j, k = wp.tid()
    g[i, j, k] = g[i, j, k] * a


@wp.kernel
def _mult_const_vec_kernel(g: wp.array3d(dtype=wp.vec3), a: float):
    i, j, k = wp.tid()
    g[i, j, k] = g[i, j, k] * a


@wp.kernel
def _add_const_kernel(g: wp.array3d(dtype=wp.float32), a: float):
    i, j, k = wp.tid()
    g[i, j, k] = g[i, j, k] + a


@wp.kernel
def _bound_neumann_kernel(g: wp.array3d(dtype=wp.float32), w: int):
    # knSetBoundaryNeumann: sources are inner cells, so this is race-free.
    i, j, k = wp.tid()
    nx, ny, nz = g.shape[0], g.shape[1], g.shape[2]
    si, sj, sk = i, j, k
    hit = False
    if i <= w:
        si = w + 1
        hit = True
    if i >= nx - 1 - w:
        si = nx - 1 - w - 1
        hit = True
    if j <= w:
        sj = w + 1
        hit = True
    if j >= ny - 1 - w:
        sj = ny - 1 - w - 1
        hit = True
    if nz > 1:
        if k <= w:
            sk = w + 1
            hit = True
        if k >= nz - 1 - w:
            sk = nz - 1 - w - 1
            hit = True
    if hit:
        g[i, j, k] = g[si, sj, sk]


@wp.kernel
def _mirror_slab_kernel(g: wp.array3d(dtype=wp.float32)):
    i, k = wp.tid()
    v = g[i, 1, k]
    g[i, 0, k] = v
    g[i, 2, k] = v


@wp.kernel
def _mirror_slab_mac_kernel(g: wp.array3d(dtype=wp.vec3)):
    i, k = wp.tid()
    v = g[i, 1, k]
    v = wp.vec3(v[0], 0.0, v[2])
    g[i, 0, k] = v
    g[i, 1, k] = v
    g[i, 2, k] = v


# ---- domain -----------------------------------------------------------------

@wp.func
def _apply_inflow(i: int, j: int, k: int, s: float, emission: wp.array3d(dtype=wp.float32),
                  dens_in: wp.array3d(dtype=wp.float32), heat_in: wp.array3d(dtype=wp.float32),
                  dens: wp.array3d(dtype=wp.float32), heat: wp.array3d(dtype=wp.float32),
                  flow_dens: float, temp: float,
                  fuel_in: wp.array3d(dtype=wp.float32), react_in: wp.array3d(dtype=wp.float32),
                  fuel: wp.array3d(dtype=wp.float32), react: wp.array3d(dtype=wp.float32),
                  use_fire: int, fuel_amount: float, emit_fuel: int):
    """Blender apply_inflow_fields, absolute mode, for emission strength s.
    flow_dens is 0 for FIRE flows; emit_fuel is 0 for SMOKE flows."""
    emission[i, j, k] = wp.max(emission[i, j, k], s)
    if s > 0.0:
        h = heat[i, j, k]
        if temp > 0.0:
            heat_in[i, j, k] = wp.min(h + temp, wp.max(h, temp))
        else:
            heat_in[i, j, k] = wp.max(h + temp, wp.min(h, temp))
    df = s * flow_dens
    if df > dens[i, j, k]:
        dens_in[i, j, k] = wp.max(df, dens_in[i, j, k])
    if use_fire != 0:
        ff = s * fuel_amount
        if emit_fuel != 0 and ff != 0.0 and ff > fuel[i, j, k]:
            fuel_in[i, j, k] = wp.max(ff, fuel_in[i, j, k])
        # Reaction coordinate: slight falloff instead of 1 for all new fuel.
        value = 1.0 - (1.0 - s) * (1.0 - s)
        fi = fuel_in[i, j, k]
        r = react[i, j, k]
        if fi > 1.1920929e-07 and value > r:
            f = ff / fi
            react_in[i, j, k] = wp.clamp(value * f + (1.0 - f) * r, 0.0, value)

# ---- meshes (Blender fluid.cc update_distances / sample_mesh) ---------------
# Mesh points are in cell units; cell centres sit at index + 0.5. Kernels run
# over the object's bounding box only, offset by `lo`.

PHI_MAX = wp.constant(9999.0)
SQRT3 = wp.constant(1.732)  # Blender's nearest-search radius (unit cube diagonal)


@wp.func
def _mesh_distance(mesh: wp.uint64, p: wp.vec3, dirs: wp.array(dtype=wp.vec3),
                   thickness: float, plane_init: int) -> float:
    """update_distances before the min with the existing value."""
    d = float(PHI_MAX)
    if plane_init != 0:
        # Blender adds the thickness to the squared radius, unsquared.
        n = wp.mesh_query_point_no_sign(mesh, p, wp.sqrt(SQRT3 * SQRT3 + thickness))
        if n.result:
            d = -wp.length(p - wp.mesh_eval_position(mesh, n.face, n.u, n.v))
        return d
    # 26 rays: inside iff every ray hits and not every hit faces the ray.
    misses = int(0)
    against = int(0)
    for r in range(26):
        q = wp.mesh_query_ray(mesh, p, dirs[r], PHI_MAX)
        if not q.result:
            misses += 1
            continue
        if wp.dot(dirs[r], q.normal) <= 0.0:
            against += 1
        d = wp.min(d, q.t)
    if misses == 0 and against != 26:
        d = -wp.abs(d)
    if thickness != 0.0:
        d -= thickness
    return d


@wp.kernel
def _rasterize_mesh_flow_kernel(mesh: wp.uint64, dirs: wp.array(dtype=wp.vec3), lo: wp.vec3i,
                                phi: wp.array3d(dtype=wp.float32), emission: wp.array3d(dtype=wp.float32),
                                dens_in: wp.array3d(dtype=wp.float32), heat_in: wp.array3d(dtype=wp.float32),
                                dens: wp.array3d(dtype=wp.float32), heat: wp.array3d(dtype=wp.float32),
                                flow_dens: float, temp: float, surf: float, vol: float, plane_init: int,
                                fuel_in: wp.array3d(dtype=wp.float32), react_in: wp.array3d(dtype=wp.float32),
                                fuel: wp.array3d(dtype=wp.float32), react: wp.array3d(dtype=wp.float32),
                                use_fire: int, fuel_amount: float, emit_fuel: int):
    a, b, c = wp.tid()
    i = a + lo[0]
    j = b + lo[1]
    k = c + lo[2]
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    # sample_mesh: volume factor from a +x ray hitting a face from behind,
    # then a -x ray that must hit anything.
    vf = float(0.0)
    if vol != 0.0:
        q = wp.mesh_query_ray(mesh, p, wp.vec3(1.0, 0.0, 0.0), PHI_MAX)
        if q.result and q.normal[0] >= 0.0:
            if wp.mesh_query_ray(mesh, p, wp.vec3(-1.0, 0.0, 0.0), PHI_MAX).result:
                vf = vol
    s = float(0.0)
    n = wp.mesh_query_point_no_sign(mesh, p, SQRT3)
    if n.result and surf != 0.0:
        dn = wp.length(p - wp.mesh_eval_position(mesh, n.face, n.u, n.v))
        s = wp.sqrt(1.0 - wp.clamp(dn / surf, 0.0, 1.0))
    s = wp.max(vf, s)
    d = wp.clamp(_mesh_distance(mesh, p, dirs, surf, plane_init), -PHI_MAX, PHI_MAX)
    phi[i, j, k] = wp.min(phi[i, j, k], d)
    _apply_inflow(i, j, k, s, emission, dens_in, heat_in, dens, heat, flow_dens, temp,
                  fuel_in, react_in, fuel, react, use_fire, fuel_amount, emit_fuel)


# update_velocities on a zeroed per-object map: OVERRIDE v, AVERAGED (0 + v)/2,
# MAX max(|v|, 0) = |v|, MIN min(|v|, 0) = 0.
_GUIDE_MODE_SCALE = {"OVERRIDE": 1.0, "AVERAGED": 0.5, "MAX": 1.0, "MIN": 0.0}


@wp.kernel
def _rasterize_mesh_obstacle_kernel(mesh: wp.uint64, dirs: wp.array(dtype=wp.vec3), lo: wp.vec3i,
                                    phi: wp.array3d(dtype=wp.float32), thickness: float, plane_init: int,
                                    obvel: wp.array3d(dtype=wp.vec3), num: wp.array3d(dtype=wp.float32),
                                    use_vel: int, vel_scale: float, vel_abs: int):
    a, b, c = wp.tid()
    i = a + lo[0]
    j = b + lo[1]
    k = c + lo[2]
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    d = wp.clamp(_mesh_distance(mesh, p, dirs, thickness, plane_init), -PHI_MAX, PHI_MAX)
    phi[i, j, k] = wp.min(phi[i, j, k], d)
    # update_velocities + apply_effector_fields: only cells inside this mesh count.
    if use_vel != 0 and d < 0.0:
        num[i, j, k] = num[i, j, k] + 1.0
        n = wp.mesh_query_point_no_sign(mesh, p, SQRT3)
        if n.result:
            v = wp.mesh_eval_velocity(mesh, n.face, n.u, n.v) * vel_scale
            if vel_abs != 0:
                v = wp.vec3(wp.abs(v[0]), wp.abs(v[1]), wp.abs(v[2]))
            obvel[i, j, k] = obvel[i, j, k] + v


@wp.kernel
def _rasterize_mesh_liquid_kernel(mesh: wp.uint64, dirs: wp.array(dtype=wp.vec3), lo: wp.vec3i,
                                  phi: wp.array3d(dtype=wp.float32), surf: float, plane_init: int,
                                  invel: wp.array3d(dtype=wp.vec3), use_vel: int,
                                  tris: wp.array(dtype=wp.int32), normals: wp.array(dtype=wp.vec3),
                                  vel_normal: float, vel_multi: float, vel_coord: wp.vec3):
    a, b, c = wp.tid()
    i = a + lo[0]
    j = b + lo[1]
    k = c + lo[2]
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    d = wp.clamp(_mesh_distance(mesh, p, dirs, surf, plane_init), -PHI_MAX, PHI_MAX)
    phi[i, j, k] = wp.min(phi[i, j, k], d)
    if use_vel == 0:
        return
    # sample_mesh (liquid: emission strength 1) and the highest-velocity rule of
    # update_flowsfluids. Barycentric weights (u, v, 1-u-v) as mesh_eval_position.
    n = wp.mesh_query_point_no_sign(mesh, p, SQRT3)
    if not n.result:
        return
    v = wp.vec3(0.0)
    if vel_normal != 0.0:
        w = 1.0 - n.u - n.v
        hn = (normals[tris[3 * n.face]] * n.u + normals[tris[3 * n.face + 1]] * n.v
              + normals[tris[3 * n.face + 2]] * w)
        v = v + wp.normalize(hn) * vel_normal
    if vel_multi != 0.0:
        v = v + wp.mesh_eval_velocity(mesh, n.face, n.u, n.v) * vel_multi
    v = v + vel_coord
    if wp.length_sq(v) > wp.length_sq(invel[i, j, k]):
        invel[i, j, k] = v


# The 26 update_distances ray directions, in Blender's order.
_RAY_DIRS = np.array([
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
    (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1), (1, 1, -1), (1, -1, -1),
    (-1, 1, -1), (-1, -1, -1)], dtype=np.float32)
_RAY_DIRS /= np.linalg.norm(_RAY_DIRS, axis=1, keepdims=True)


def list_devices() -> list[dict]:
    """The CUDA devices, for `openbillow_bake.py devices` (see backend.list_devices)."""
    wp.init()
    return [{"name": d.name, "alias": d.alias, "total_mb": d.total_memory // 2**20,
             "free_mb": d.free_memory // 2**20,
             "default_limit_mb": default_limit_mb(d.total_memory)}
            for d in wp.get_cuda_devices()]


def diagnostics(device: str = "") -> dict:
    """Warp and GPU facts for the support button. Never raises: a failure keeps what was found so
    far and fills device_init_error."""
    out = {}
    try:
        out["warp_version"] = wp.__version__
        out["warp_path"] = getattr(wp, "__file__", None)
        wp.init()
        out["cuda_available"] = bool(wp.is_cuda_available())
        if out["cuda_available"]:
            d = wp.get_cuda_device(device) if device else wp.get_cuda_device()
            out.update(gpu_name=d.name, compute_capability="sm_%d" % d.arch,
                       vram_total_mb=d.total_memory // 2**20, vram_free_mb=d.free_memory // 2**20)
            try:
                v = wp.context.runtime.core.wp_cuda_driver_version()
                out["cuda_driver_version"] = "%d.%d" % (v // 1000, v % 1000 // 10)
            except Exception:  # older/newer Warp without this call
                pass
        c = wp.config.kernel_cache_dir
        if c and os.path.isdir(c):
            out["kernel_cache_exists"] = True
            out["kernel_cache_mb"] = round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(c)
                                               for f in fs) / 2**20, 1)
    except Exception as e:
        out["device_init_error"] = f"{type(e).__name__}: {e}"[:500]
    return out


class WarpMesh:
    """Mesh handle: object-space vertices on the host, cell-space points and
    vertex velocities in a wp.Mesh (BVH) on the device, the last cell-space
    points on the host (Blender's verts_old), and the cell bounding box
    (floor/ceil of the transformed vertices, as Blender's bb_boundInsert)."""

    def __init__(self, verts: np.ndarray, tris: np.ndarray, device):
        self.verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        self.points = wp.array(self.verts, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(len(self.verts), dtype=wp.vec3, device=device)
        self.prev = None
        self.tris = np.ascontiguousarray(tris, dtype=np.int32).reshape(-1, 3)
        self.indices = wp.array(self.tris.ravel(), dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices, velocities=self.velocities)
        self.nbytes = self.points.capacity + self.velocities.capacity + self.indices.capacity  # BVH not counted
        # Closed (watertight, consistently wound) when every directed edge
        # appears once and its reverse appears once. Unwelded duplicate
        # vertices make a mesh count as open, the safe side. Open meshes
        # (a lidless box, a bowl, a single wall) have no inside, so particle
        # collisions treat them as two-sided surfaces.
        t = self.tris.astype(np.int64)
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        nv = max(len(self.verts), 1)
        fwd, rev = e[:, 0] * nv + e[:, 1], e[:, 1] * nv + e[:, 0]
        self.closed = bool(len(t)) and len(np.unique(fwd)) == len(fwd) and np.array_equal(
            np.sort(fwd), np.sort(rev))
        self.two_sided = not self.closed
        self._set_bounds(self.verts)

    def _set_bounds(self, p: np.ndarray) -> None:
        """Bounding box and, for a closed mesh, the orientation (+1 outward
        winding, -1 inside out: flipped normals or a mirroring transform)."""
        self.lo = np.floor(p.min(axis=0)).astype(int)
        self.hi = np.ceil(p.max(axis=0)).astype(int)
        self.diag = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
        self.orient = 1.0
        if self.closed:
            a, b, c = (p[self.tris[:, i]].astype(np.float64) for i in range(3))
            self.orient = 1.0 if np.einsum("ij,ij->", a, np.cross(b, c)) >= 0.0 else -1.0

    def vertex_normals(self) -> np.ndarray:
        """Blender's vertex normals (corner-angle weighted face normals) of the
        current cell-space points. Equal to Blender's quad-based ones for
        planar faces."""
        p = (self.prev if self.prev is not None else self.verts).astype(np.float64)
        t = self.tris
        fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
        vn = np.zeros_like(p)
        for c in range(3):
            e1 = p[t[:, (c + 1) % 3]] - p[t[:, c]]
            e2 = p[t[:, (c + 2) % 3]] - p[t[:, c]]
            cos = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-30)
            np.add.at(vn, t[:, c], fn * np.arccos(np.clip(cos, -1.0, 1.0))[:, None])
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
        return vn.astype(np.float32)


# ---- obstacle velocity (Blender fluid_pre_step, Mantaflow fastmarch.cpp / grid.cpp) ----
# Mantaflow runs these with bnd = 1: the outermost layer is never touched.

@wp.func
def _inner(i: int, j: int, k: int, nx: int, ny: int, nz: int):
    return i > 0 and j > 0 and k > 0 and i < nx - 1 and j < ny - 1 and k < nz - 1


@wp.kernel
def _safe_divide_vec_kernel(v: wp.array3d(dtype=wp.vec3), n: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    if n[i, j, k] != 0.0:
        v[i, j, k] = v[i, j, k] / n[i, j, k]


@wp.kernel
def _mark_skip_kernel(phi: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.int32), inside: int):
    """knMarkSkipCells: 1 = known (inside cells when marching out)."""
    i, j, k = wp.tid()
    t = int(0)
    if _inner(i, j, k, phi.shape[0], phi.shape[1], phi.shape[2]):
        if (inside == 0 and phi[i, j, k] < 0.0) or (inside != 0 and phi[i, j, k] > 0.0):
            t = 1
    tmp[i, j, k] = t


@wp.kernel
def _first_layer_kernel(tmp: wp.array3d(dtype=wp.int32)):
    """knSetFirstLayer: unknown cells next to known ones become layer 2 (value kept)."""
    i, j, k = wp.tid()
    if not _inner(i, j, k, tmp.shape[0], tmp.shape[1], tmp.shape[2]) or tmp[i, j, k] != 0:
        return
    if (tmp[i + 1, j, k] == 1 or tmp[i - 1, j, k] == 1 or tmp[i, j + 1, k] == 1
            or tmp[i, j - 1, k] == 1 or tmp[i, j, k + 1] == 1 or tmp[i, j, k - 1] == 1):
        tmp[i, j, k] = 2


@wp.func
def _nb_add(v: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.int32), i: int, j: int, k: int,
            d: int, acc: wp.vec4):
    if tmp[i, j, k] == d:
        x = v[i, j, k]
        return acc + wp.vec4(x[0], x[1], x[2], 1.0)
    return acc


@wp.kernel
def _extrapolate_vec_kernel(v: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.int32), d: int):
    """knExtrapolateLsSimple<Vec3> (direction 0): unknown cells next to layer d
    take the mean of those neighbours and become layer d + 1. Cells written here
    are never read in the same pass, so in place is safe (as in Mantaflow)."""
    i, j, k = wp.tid()
    if not _inner(i, j, k, tmp.shape[0], tmp.shape[1], tmp.shape[2]) or tmp[i, j, k] != 0:
        return
    acc = wp.vec4(0.0)
    acc = _nb_add(v, tmp, i + 1, j, k, d, acc)
    acc = _nb_add(v, tmp, i - 1, j, k, d, acc)
    acc = _nb_add(v, tmp, i, j + 1, k, d, acc)
    acc = _nb_add(v, tmp, i, j - 1, k, d, acc)
    acc = _nb_add(v, tmp, i, j, k + 1, d, acc)
    acc = _nb_add(v, tmp, i, j, k - 1, d, acc)
    if acc[3] > 0.0:
        tmp[i, j, k] = d + 1
        v[i, j, k] = wp.vec3(acc[0], acc[1], acc[2]) / acc[3]


@wp.kernel
def _set_remaining_vec_kernel(v: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.int32)):
    """knSetRemaining<Vec3>(0): cells beyond the extrapolation distance."""
    i, j, k = wp.tid()
    if _inner(i, j, k, tmp.shape[0], tmp.shape[1], tmp.shape[2]) and tmp[i, j, k] == 0:
        v[i, j, k] = wp.vec3(0.0)


@wp.kernel
def _resample_to_mac_kernel(src: wp.array3d(dtype=wp.vec3), mac: wp.array3d(dtype=wp.vec3)):
    """knResampleVec3ToMac: each face = mean of its two cells."""
    i, j, k = wp.tid()
    if not _inner(i, j, k, src.shape[0], src.shape[1], src.shape[2]):
        return
    c = src[i, j, k]
    mac[i, j, k] = wp.vec3(0.5 * (src[i - 1, j, k][0] + c[0]),
                           0.5 * (src[i, j - 1, k][1] + c[1]),
                           0.5 * (src[i, j, k - 1][2] + c[2]))


@wp.kernel
def _set_inflow_flags_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    f = flags[i, j, k]
    if (f & FLUID) != 0 and phi[i, j, k] < 0.0:
        flags[i, j, k] = f | INFLOW


@wp.kernel
def _set_obstacle_flags_kernel(flags: wp.array3d(dtype=wp.int32), phi: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    if i < 1 or j < 1 or k < 1 or i > flags.shape[0] - 2 or j > flags.shape[1] - 2 or k > flags.shape[2] - 2:
        return
    if phi[i, j, k] < 0.0:
        flags[i, j, k] = OBSTACLE


@wp.kernel
def _apply_emission_kernel(flags: wp.array3d(dtype=wp.int32), target: wp.array3d(dtype=wp.float32),
                           source: wp.array3d(dtype=wp.float32), emission: wp.array3d(dtype=wp.float32),
                           absolute: int):
    i, j, k = wp.tid()
    if (flags[i, j, k] & (INFLOW | OUTFLOW)) != 0 or emission[i, j, k] != 0.0:
        if absolute != 0:
            target[i, j, k] = source[i, j, k]
        else:
            target[i, j, k] = target[i, j, k] + source[i, j, k]


@wp.kernel
def _dissolve_kernel(flags: wp.array3d(dtype=wp.int32), g: wp.array3d(dtype=wp.float32),
                     log_falloff: int, dydx: float, fac: float):
    # KnDissolveSmoke. The linear branch is the heat rule (fade |v| toward 0), which
    # equals the density rule (subtract, clamp at 0) for non-negative values.
    i, j, k = wp.tid()
    if (flags[i, j, k] & FLUID) == 0:
        return
    v = g[i, j, k]
    if log_falloff != 0:
        g[i, j, k] = v * fac
    elif wp.abs(v) < dydx:
        g[i, j, k] = 0.0
    elif v > 0.0:
        g[i, j, k] = v - dydx
    else:
        g[i, j, k] = v + dydx


@wp.kernel
def _buoyancy_kernel(flags: wp.array3d(dtype=wp.int32), g: wp.array3d(dtype=wp.float32),
                     vel: wp.array3d(dtype=wp.vec3), strength: wp.vec3):
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    if i < 1 or j < 1 or k < 1 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return
    if (flags[i, j, k] & FLUID) == 0:
        return
    v = vel[i, j, k]
    if (flags[i - 1, j, k] & FLUID) != 0:
        v[0] += 0.5 * strength[0] * (g[i, j, k] + g[i - 1, j, k])
    if (flags[i, j - 1, k] & FLUID) != 0:
        v[1] += 0.5 * strength[1] * (g[i, j, k] + g[i, j - 1, k])
    if (flags[i, j, k - 1] & FLUID) != 0:
        v[2] += 0.5 * strength[2] * (g[i, j, k] + g[i, j, k - 1])
    vel[i, j, k] = v


@wp.kernel
def _wall_bcs_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                     obvel: wp.array3d(dtype=wp.vec3), use_obvel: int):
    i, j, k = wp.tid()
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    f = flags[i, j, k]
    fluid = (f & FLUID) != 0
    obs = (f & OBSTACLE) != 0
    if not fluid and not obs:
        return
    bc = wp.vec3(0.0)
    if use_obvel != 0:
        bc = obvel[i, j, k]
    v = vel[i, j, k]
    if i > 0:
        fm = flags[i - 1, j, k]
        if (fm & OBSTACLE) != 0 or (obs and (fm & FLUID) != 0):
            v[0] = bc[0]
    if j > 0:
        fm = flags[i, j - 1, k]
        if (fm & OBSTACLE) != 0 or (obs and (fm & FLUID) != 0):
            v[1] = bc[1]
    if k > 0:
        fm = flags[i, j, k - 1]
        if (fm & OBSTACLE) != 0 or (obs and (fm & FLUID) != 0):
            v[2] = bc[2]
    if fluid:
        if (i > 0 and (flags[i - 1, j, k] & STICK) != 0) or (i < nx - 1 and (flags[i + 1, j, k] & STICK) != 0):
            v[1] = 0.0
            v[2] = 0.0
        if (j > 0 and (flags[i, j - 1, k] & STICK) != 0) or (j < ny - 1 and (flags[i, j + 1, k] & STICK) != 0):
            v[0] = 0.0
            v[2] = 0.0
        if (k > 0 and (flags[i, j, k - 1] & STICK) != 0) or (k < nz - 1 and (flags[i, j, k + 1] & STICK) != 0):
            v[0] = 0.0
            v[1] = 0.0
    vel[i, j, k] = v


@wp.kernel
def _reset_outflow_kernel(flags: wp.array3d(dtype=wp.int32), g: wp.array3d(dtype=wp.float32)):
    # Mantaflow also marks outflow cells EMPTY (not FLUID), so the pressure
    # solve treats them as p = 0 and corrects the fluid-outflow faces.
    i, j, k = wp.tid()
    f = flags[i, j, k]
    if (f & OUTFLOW) != 0:
        flags[i, j, k] = (f | EMPTY) & ~FLUID
        g[i, j, k] = 0.0


@wp.kernel
def _init_domain_kernel(flags: wp.array3d(dtype=wp.int32), w: int, types: vec6i):
    # FlagGrid::initDomain (setConst(Empty) + initBoundaries), then fillGrid().
    i, j, k = wp.tid()
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    f = EMPTY
    if i <= w:
        f = types[0]
    if i >= nx - 1 - w:
        f = types[1]
    if j <= w:
        f = types[2]
    if j >= ny - 1 - w:
        f = types[3]
    if nz > 1:
        if k <= w:
            f = types[4]
        if k >= nz - 1 - w:
            f = types[5]
    if (f & KEEP) == 0:
        f = (f & ~(EMPTY | FLUID)) | FLUID
    flags[i, j, k] = f


@wp.kernel
def _wall_levelset_kernel(phi: wp.array3d(dtype=wp.float32), w: int, walls: vec6i):
    # initDomain's phiWalls: setConst(1e9), then Init{Min,Max}{X,Y,Z}Wall for
    # each wall side (min(i - w - .5, phi) in double, stored as float).
    i, j, k = wp.tid()
    n = wp.vec3i(phi.shape[0], phi.shape[1], phi.shape[2])
    c = wp.vec3i(i, j, k)
    v = wp.float64(1.0e9)
    for a in range(3):
        if a == 2 and n[2] == 1:
            continue
        if walls[2 * a] != 0:
            v = wp.min(wp.float64(c[a] - w) - wp.float64(0.5), v)
        if walls[2 * a + 1] != 0:
            v = wp.min(wp.float64(n[a] - c[a]) - wp.float64(1.5) - wp.float64(w), v)
    phi[i, j, k] = wp.float32(v)


# ---- advection --------------------------------------------------------------
# Mantaflow advectSemiLagrange (plugin/advection.cpp) with orderSpace =
# orderTrace = 1, as Blender runs it. Positions are in cells, cell centres at
# index + 0.5. Lookups of MAC component c pass positions shifted by half a cell
# along c, so every lookup uses the cell-centred stencil (interpol.h).

@wp.func
def _axis(x: float, n: int):
    # One axis of interpol.h BUILD_INDEX: lower sample and upper weight, clamped to the grid.
    # The range tests come first so NaN and huge values never reach the float-to-int conversion
    # (undefined on x86: the CPU device read a[INT_MIN] and crashed).
    if not (x > 0.0):  # negative, zero or NaN
        return 0, 0.0
    if x >= float(n - 1):
        return n - 2, 1.0
    i = int(x)
    return i, x - float(i)


@wp.func
def _lerp8(c000: float, c100: float, c010: float, c110: float,
           c001: float, c101: float, c011: float, c111: float, s: float, t: float, f: float):
    return (((c000 * (1.0 - t) + c010 * t) * (1.0 - s) + (c100 * (1.0 - t) + c110 * t) * s) * (1.0 - f)
            + ((c001 * (1.0 - t) + c011 * t) * (1.0 - s) + (c101 * (1.0 - t) + c111 * t) * s) * f)


@wp.func
def _sample(a: wp.array3d(dtype=wp.float32), p: wp.vec3):
    i, s = _axis(p[0] - 0.5, a.shape[0])
    j, t = _axis(p[1] - 0.5, a.shape[1])
    k, f = _axis(p[2] - 0.5, a.shape[2])
    return _lerp8(a[i, j, k], a[i + 1, j, k], a[i, j + 1, k], a[i + 1, j + 1, k],
                  a[i, j, k + 1], a[i + 1, j, k + 1], a[i, j + 1, k + 1], a[i + 1, j + 1, k + 1], s, t, f)


@wp.func
def _sample_comp(a: wp.array3d(dtype=wp.vec3), c: int, p: wp.vec3):
    # interpolComponent<c>
    i, s = _axis(p[0] - 0.5, a.shape[0])
    j, t = _axis(p[1] - 0.5, a.shape[1])
    k, f = _axis(p[2] - 0.5, a.shape[2])
    return _lerp8(a[i, j, k][c], a[i + 1, j, k][c], a[i, j + 1, k][c], a[i + 1, j + 1, k][c],
                  a[i, j, k + 1][c], a[i + 1, j, k + 1][c], a[i, j + 1, k + 1][c],
                  a[i + 1, j + 1, k + 1][c], s, t, f)


@wp.func
def _unit(c: int):
    return wp.vec3i(wp.where(c == 0, 1, 0), wp.where(c == 1, 1, 0), wp.where(c == 2, 1, 0))


@wp.func
def _at(v: wp.array3d(dtype=wp.vec3), p: wp.vec3i):
    return v[p[0], p[1], p[2]]


@wp.func
def _at_flag(flags: wp.array3d(dtype=wp.int32), p: wp.vec3i):
    return flags[p[0], p[1], p[2]]


@wp.func
def _centered_vel(v: wp.array3d(dtype=wp.vec3), i: int, j: int, k: int):
    # MACGrid::getCentered
    return wp.vec3(0.5 * (v[i, j, k][0] + v[i + 1, j, k][0]),
                   0.5 * (v[i, j, k][1] + v[i, j + 1, k][1]),
                   0.5 * (v[i, j, k][2] + v[i, j, k + 1][2]))


@wp.func
def _face_vel(v: wp.array3d(dtype=wp.vec3), c: int, p: wp.vec3i):
    # MACGrid::getAtMACX/Y/Z: component c as stored, the other two averaged from the 4 nearest faces.
    ec = _unit(c)
    r = wp.vec3()
    for d in range(3):
        if d == c:
            r[d] = _at(v, p)[d]
        else:
            ed = _unit(d)
            r[d] = 0.25 * (_at(v, p)[d] + _at(v, p - ec)[d] + _at(v, p + ed)[d] + _at(v, p + ed - ec)[d])
    return r


@wp.func
def _border(i: int, j: int, k: int, nx: int, ny: int, nz: int):
    # Mantaflow kernels with bnd=1 skip the outermost layer.
    return i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1


# ---- vorticity confinement --------------------------------------------------
# Mantaflow vorticityConfinement: GetCentered, CurlOp, GridNorm, KnConfForce and
# KnApplyForceField(isMAC=false), all bnd=1 on fresh zero grids, so the centred
# velocity, curl and force are zero on the outermost layer.

@wp.func
def _center_or_zero(v: wp.array3d(dtype=wp.vec3), i: int, j: int, k: int):
    if _border(i, j, k, v.shape[0], v.shape[1], v.shape[2]):
        return wp.vec3()
    return _centered_vel(v, i, j, k)


@wp.kernel
def _curl_kernel(vel: wp.array3d(dtype=wp.vec3), curl: wp.array3d(dtype=wp.vec3)):
    # GetCentered fused into CurlOp.
    i, j, k = wp.tid()
    if _border(i, j, k, vel.shape[0], vel.shape[1], vel.shape[2]):
        curl[i, j, k] = wp.vec3()
        return
    xp = _center_or_zero(vel, i + 1, j, k)
    xm = _center_or_zero(vel, i - 1, j, k)
    yp = _center_or_zero(vel, i, j + 1, k)
    ym = _center_or_zero(vel, i, j - 1, k)
    zp = _center_or_zero(vel, i, j, k + 1)
    zm = _center_or_zero(vel, i, j, k - 1)
    curl[i, j, k] = wp.vec3(0.5 * ((yp[2] - ym[2]) - (zp[1] - zm[1])),
                            0.5 * ((zp[0] - zm[0]) - (xp[2] - xm[2])),
                            0.5 * ((xp[1] - xm[1]) - (yp[0] - ym[0])))


@wp.kernel
def _conf_force_kernel(curl: wp.array3d(dtype=wp.vec3), strength: float,
                       cell: wp.array3d(dtype=wp.float32), use_cell: int, cell_scale: float,
                       force: wp.array3d(dtype=wp.vec3)):
    # GridNorm fused into KnConfForce; `cell` is strengthCell (fire: fuel * cell_scale).
    i, j, k = wp.tid()
    if _border(i, j, k, curl.shape[0], curl.shape[1], curl.shape[2]):
        force[i, j, k] = wp.vec3()
        return
    grad = 0.5 * wp.vec3(wp.length(curl[i + 1, j, k]) - wp.length(curl[i - 1, j, k]),
                         wp.length(curl[i, j + 1, k]) - wp.length(curl[i, j - 1, k]),
                         wp.length(curl[i, j, k + 1]) - wp.length(curl[i, j, k - 1]))
    l = wp.dot(grad, grad)
    if l > 1.0e-12:  # Mantaflow normalize(): VECTOR_EPSILON^2, else zero
        grad = grad / wp.sqrt(l)
    else:
        grad = wp.vec3()
    st = strength
    if use_cell != 0:
        st = st + cell_scale * cell[i, j, k]
    force[i, j, k] = st * wp.cross(grad, curl[i, j, k])


@wp.kernel
def _process_burn_kernel(fuel: wp.array3d(dtype=wp.float32), density: wp.array3d(dtype=wp.float32),
                         react: wp.array3d(dtype=wp.float32), heat: wp.array3d(dtype=wp.float32),
                         use_heat: int, burn: float, flame_smoke: float, ignition: float,
                         max_temp: float):
    # KnProcessBurn without colours; `burn` = burningRate * dt. No flag check, bnd=1.
    i, j, k = wp.tid()
    if _border(i, j, k, fuel.shape[0], fuel.shape[1], fuel.shape[2]):
        return
    orig = fuel[i, j, k]
    f = wp.max(orig - burn, 0.0)
    fuel[i, j, k] = f
    flame = float(0.0)
    if orig > 1.0e-6:  # VECTOR_EPSILON
        r = react[i, j, k] * (f / orig)
        react[i, j, k] = r
        flame = wp.sqrt(r)
    else:
        react[i, j, k] = 0.0
    emit = float(0.0)
    if orig < 1.0:
        emit = (1.0 - orig) * 0.5
    # Mantaflow clamps a copy, so density is left unclamped (quirk kept).
    density[i, j, k] = density[i, j, k] + (emit + 0.5) * (orig - f) * 0.1 * flame_smoke
    if use_heat != 0 and flame != 0.0:
        heat[i, j, k] = (1.0 - flame) * ignition + flame * max_temp


@wp.kernel
def _update_flame_kernel(react: wp.array3d(dtype=wp.float32), flame: wp.array3d(dtype=wp.float32)):
    # KnUpdateFlame, bnd=1.
    i, j, k = wp.tid()
    if _border(i, j, k, react.shape[0], react.shape[1], react.shape[2]):
        return
    r = react[i, j, k]
    if r > 0.0:
        flame[i, j, k] = wp.sqrt(r)
    else:
        flame[i, j, k] = 0.0


@wp.kernel
def _combust_kernel(fuel: wp.array3d(dtype=wp.float32), density: wp.array3d(dtype=wp.float32),
                    heat: wp.array3d(dtype=wp.float32), expansion_div: wp.array3d(dtype=wp.float32),
                    burn: float, ignition: float, max_temp: float, heat_release: float,
                    smoke: float, cool: float, cool_rad: float, inv_max: float,
                    expansion_rate: float):
    # COMBUSTION model (not Mantaflow): heat cools (implicit Euler, so any dt is
    # stable: linear term plus a radiative one growing as (T/Tmax)^3), then fuel
    # burns only where the gas was above ignition, first order (burn =
    # 1 - exp(-burning_rate * dt), so rate follows the fuel left), releasing heat
    # (capped at max_temp) and smoke. Cooling first lets a burning cell reach max_temp.
    # expansion_div = expansion * burned / dt, the pressure solve's divergence source.
    i, j, k = wp.tid()
    if _border(i, j, k, fuel.shape[0], fuel.shape[1], fuel.shape[2]):
        expansion_div[i, j, k] = 0.0
        return
    f = fuel[i, j, k]
    t0 = heat[i, j, k]
    x = wp.abs(t0) * inv_max
    t = t0 / (1.0 + cool + cool_rad * x * x * x)
    b = float(0.0)
    if f > 1.0e-6:
        # Below ignition the fuel goes at the same rate but as smoke only
        # (no heat, no expansion), so cold fuel does not pile up.
        u = f * burn
        fuel[i, j, k] = f - u
        density[i, j, k] = density[i, j, k] + smoke * u
        # The burning share ramps smoothly from 0 at 0.75 * ignition to 1 at
        # ignition; a hard switch flickers per substep and leaves bands.
        r = wp.clamp((t0 - 0.75 * ignition) / (0.25 * ignition), 0.0, 1.0)
        if r > 0.0:
            b = u * r * r * (3.0 - 2.0 * r)
            if t < max_temp:
                t = wp.min(t + heat_release * b, max_temp)
    heat[i, j, k] = t
    expansion_div[i, j, k] = expansion_rate * b


@wp.kernel
def _flame_from_heat_kernel(heat: wp.array3d(dtype=wp.float32), flame: wp.array3d(dtype=wp.float32),
                            ignition: float, inv_range: float):
    # COMBUSTION flame output: 0 at ignition, 1 at max_temp, linear (no sqrt).
    i, j, k = wp.tid()
    flame[i, j, k] = wp.clamp((heat[i, j, k] - ignition) * inv_range, 0.0, 1.0)


@wp.func
def _gas_rho(h: float, c: float):
    # Ideal-gas density relative to ambient: T_amb / T, T = T_amb (1 + c max(h, 0)).
    return 1.0 / (1.0 + c * wp.max(h, 0.0))


@wp.kernel
def _density_weights_kernel(heat: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.vec3), c: float):
    # Face weight 1/rho_face, rho_face = (rho_lo + rho_hi) / 2, lower face per axis
    # (prototype flame2d.py `project_vardens`); index 0 uses the cell's own density.
    i, j, k = wp.tid()
    r = _gas_rho(heat[i, j, k], c)
    rx = r
    ry = r
    rz = r
    if i > 0:
        rx = _gas_rho(heat[i - 1, j, k], c)
    if j > 0:
        ry = _gas_rho(heat[i, j - 1, k], c)
    if k > 0:
        rz = _gas_rho(heat[i, j, k - 1], c)
    w[i, j, k] = wp.vec3(2.0 / (r + rx), 2.0 / (r + ry), 2.0 / (r + rz))


@wp.func
def _lattice_vec(seed: int, ci: int, cj: int, ck: int) -> wp.vec3:
    # Random vector in [-1, 1]^3 per lattice point (disturbance noise).
    st = wp.rand_init(seed, (ci & 1023) + 1024 * ((cj & 1023) + 1024 * (ck & 1023)))
    return wp.vec3(wp.randf(st, -1.0, 1.0), wp.randf(st, -1.0, 1.0), wp.randf(st, -1.0, 1.0))


@wp.func
def _lattice_noise(seed: int, p: wp.vec3) -> wp.vec3:
    # Trilinear value noise of _lattice_vec, lattice spacing 1.
    i0 = int(wp.floor(p[0]))
    j0 = int(wp.floor(p[1]))
    k0 = int(wp.floor(p[2]))
    fx = p[0] - float(i0)
    fy = p[1] - float(j0)
    fz = p[2] - float(k0)
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)
    a = wp.lerp(_lattice_vec(seed, i0, j0, k0), _lattice_vec(seed, i0 + 1, j0, k0), fx)
    b = wp.lerp(_lattice_vec(seed, i0, j0 + 1, k0), _lattice_vec(seed, i0 + 1, j0 + 1, k0), fx)
    c = wp.lerp(_lattice_vec(seed, i0, j0, k0 + 1), _lattice_vec(seed, i0 + 1, j0, k0 + 1), fx)
    d = wp.lerp(_lattice_vec(seed, i0, j0 + 1, k0 + 1), _lattice_vec(seed, i0 + 1, j0 + 1, k0 + 1), fx)
    return wp.lerp(wp.lerp(a, b, fy), wp.lerp(c, d, fy), fz)


@wp.kernel
def _fire_forces_kernel(heat: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.vec3),
                        force: wp.array3d(dtype=wp.vec3), inv_size: float, seed0: int,
                        seed1: int, ft: float, dist: float, inv_ign: float, shred: float,
                        ign: float, grad_eps: float):
    # COMBUSTION (not Mantaflow), cell-centred force for _apply_force_kernel.
    # Disturbance: smooth random field (lattice every 1/inv_size cells), blended
    # between two time ticks, scaled by clamp(heat / ignition, 0, 1).
    # Shredding: along n = grad(heat)/|grad(heat)|, stretch the normal velocity
    # (v.n) n where heat >= ignition and squash it below, so the flame edge tears.
    i, j, k = wp.tid()
    if _border(i, j, k, heat.shape[0], heat.shape[1], heat.shape[2]):
        force[i, j, k] = wp.vec3(0.0)
        return
    t = heat[i, j, k]
    f = wp.vec3(0.0)
    w = wp.clamp(t * inv_ign, 0.0, 1.0)
    if dist != 0.0 and w > 0.0:
        p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * inv_size
        n0 = _lattice_noise(seed0, p)
        n1 = _lattice_noise(seed1, p)
        f = (dist * w) * wp.lerp(n0, n1, ft * ft * (3.0 - 2.0 * ft))
    if shred != 0.0 and w > 0.1:
        g = 0.5 * wp.vec3(heat[i + 1, j, k] - heat[i - 1, j, k], heat[i, j + 1, k] - heat[i, j - 1, k],
                          heat[i, j, k + 1] - heat[i, j, k - 1])
        gl = wp.length(g)
        if gl > grad_eps:
            n = g / gl
            v = 0.5 * (vel[i, j, k] + wp.vec3(vel[i + 1, j, k][0], vel[i, j + 1, k][1], vel[i, j, k + 1][2]))
            s = shred
            if t < ign:
                s = -wp.min(shred, 1.0)
            f = f + (s * wp.dot(v, n)) * n
    force[i, j, k] = f


@wp.kernel
def _apply_force_kernel(flags: wp.array3d(dtype=wp.int32), force: wp.array3d(dtype=wp.vec3),
                        vel: wp.array3d(dtype=wp.vec3), scale: float):
    # KnApplyForceField, additive, centred force averaged onto faces.
    i, j, k = wp.tid()
    if _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2]):
        return
    f = flags[i, j, k]
    cur_fluid = (f & FLUID) != 0
    if not cur_fluid and (f & EMPTY) == 0:
        return
    fc = force[i, j, k]
    v = vel[i, j, k]
    n = flags[i - 1, j, k]
    if (n & FLUID) != 0 or (cur_fluid and (n & EMPTY) != 0):
        v[0] += 0.5 * scale * (force[i - 1, j, k][0] + fc[0])
    n = flags[i, j - 1, k]
    if (n & FLUID) != 0 or (cur_fluid and (n & EMPTY) != 0):
        v[1] += 0.5 * scale * (force[i, j - 1, k][1] + fc[1])
    n = flags[i, j, k - 1]
    if (n & FLUID) != 0 or (cur_fluid and (n & EMPTY) != 0):
        v[2] += 0.5 * scale * (force[i, j, k - 1][2] + fc[2])
    vel[i, j, k] = v


@wp.func
def _clamp_cell(flags: wp.array3d(dtype=wp.int32), orig: wp.array3d(dtype=wp.float32),
                d: float, fwd: float, p: wp.vec3i, off: wp.vec3, mode: int):
    # doClampComponent: range of the fluid/empty cells around the source (mode 1: and target) point.
    nx = orig.shape[0]
    ny = orig.shape[1]
    nz = orig.shape[2]
    lo = float(FLT_MAX)
    hi = float(-FLT_MAX)
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
        r = wp.clamp(d, lo, hi)
    elif d < lo or d > hi:
        r = fwd
    return r


@wp.func
def _clamp_face(flags: wp.array3d(dtype=wp.int32), orig: wp.array3d(dtype=wp.vec3), c: int,
                d: float, fwd: float, p: wp.vec3i, off: wp.vec3, mode: int):
    # doClampComponentMAC: no flag test on the samples; mode 2 falls back to
    # first order next to non-fluid cells instead.
    nb = p - _unit(c)
    if mode == 2 and ((_at_flag(flags, p) & FLUID_OR_EMPTY) == 0
                      or (_at_flag(flags, nb) & FLUID_OR_EMPTY) == 0):
        return fwd
    nx = orig.shape[0]
    ny = orig.shape[1]
    nz = orig.shape[2]
    lo = float(FLT_MAX)
    hi = float(-FLT_MAX)
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
                v = orig[i0 + (n & 1), j0 + ((n >> 1) & 1), k0 + ((n >> 2) & 1)][c]
                lo = wp.min(lo, v)
                hi = wp.max(hi, v)
    r = d
    if mode == 1:
        r = wp.clamp(d, lo, hi)
    elif d < lo or d > hi:
        r = fwd
    return r


@wp.func
def _is_fluid_at(flags: wp.array3d(dtype=wp.int32), p: wp.vec3i):
    in_bounds = (p[0] >= 0 and p[1] >= 0 and p[2] >= 0 and p[0] < flags.shape[0]
                 and p[1] < flags.shape[1] and p[2] < flags.shape[2])
    r = bool(False)
    if in_bounds:
        r = (_at_flag(flags, p) & FLUID) != 0
    return r


@wp.kernel
def _sl_cell_kernel(vel: wp.array3d(dtype=wp.vec3), src: wp.array3d(dtype=wp.float32),
                    dst: wp.array3d(dtype=wp.float32), dt: float):
    # SemiLagrange<Real>; the border layer is left at zero as in a fresh Mantaflow grid.
    i, j, k = wp.tid()
    if _border(i, j, k, dst.shape[0], dst.shape[1], dst.shape[2]):
        dst[i, j, k] = 0.0
        return
    p = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) - _centered_vel(vel, i, j, k) * dt
    dst[i, j, k] = _sample(src, p)


@wp.kernel
def _sl_mac_kernel(vel: wp.array3d(dtype=wp.vec3), src: wp.array3d(dtype=wp.vec3),
                   dst: wp.array3d(dtype=wp.vec3), dt: float):
    # SemiLagrangeMAC
    i, j, k = wp.tid()
    if _border(i, j, k, dst.shape[0], dst.shape[1], dst.shape[2]):
        dst[i, j, k] = wp.vec3()
        return
    p = wp.vec3i(i, j, k)
    p0 = wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    r = wp.vec3()
    for c in range(3):
        r[c] = _sample_comp(src, c, p0 - _face_vel(vel, c, p) * dt)
    dst[i, j, k] = r


@wp.kernel
def _mc_cell_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                    orig: wp.array3d(dtype=wp.float32), fwd: wp.array3d(dtype=wp.float32),
                    bwd: wp.array3d(dtype=wp.float32), dt: float, strength: float, mode: int):
    # MacCormackCorrect + MacCormackClamp, result written over bwd (each thread
    # reads only its own bwd and fwd cell).
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
        d = _clamp_cell(flags, orig, d, f, wp.vec3i(i, j, k), off, mode)
        if mode == 1:
            # Also fall back when the forward or backward lookup leaves the grid or hits an obstacle.
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


@wp.kernel
def _mc_mac_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                   orig: wp.array3d(dtype=wp.vec3), fwd: wp.array3d(dtype=wp.vec3),
                   bwd: wp.array3d(dtype=wp.vec3), dt: float, strength: float, mode: int):
    # MacCormackCorrectMAC + MacCormackClampMAC, result written over bwd.
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    fluid = (flags[i, j, k] & FLUID) != 0
    border = _border(i, j, k, flags.shape[0], flags.shape[1], flags.shape[2])
    f = fwd[i, j, k]
    o = orig[i, j, k]
    b = bwd[i, j, k]
    r = wp.vec3()
    for c in range(3):
        nb = p - _unit(c)
        skip = not fluid
        if p[c] > 0:
            if (_at_flag(flags, nb) & FLUID) == 0:
                skip = True
        d = f[c]
        if not skip:
            d = f[c] + strength * 0.5 * (o[c] - b[c])
        if not border:
            d = _clamp_face(flags, orig, c, d, f[c], p, _face_vel(vel, c, p) * dt, mode)
        r[c] = d
    bwd[i, j, k] = r


@wp.kernel
def _reflect_kernel(vel: wp.array3d(dtype=wp.vec3), pre: wp.array3d(dtype=wp.vec3)):
    # Advection-reflection: pre = 2 * projected - unprojected.
    i, j, k = wp.tid()
    pre[i, j, k] = 2.0 * vel[i, j, k] - pre[i, j, k]


@wp.kernel
def _outflow_vel_kernel(flags: wp.array3d(dtype=wp.int32), new_vel: wp.array3d(dtype=wp.vec3),
                        vel: wp.array3d(dtype=wp.vec3), time_step: float):
    # applyOutflowBC (extrapolateVelConvectiveBC + copyChangedVels), fused with
    # the copy of `new` into `vel`. `vel` still holds the pre-advection values
    # and each thread reads only its own cell of it.
    i, j, k = wp.tid()
    nv = new_vel[i, j, k]
    if (flags[i, j, k] & OUTFLOW) == 0:
        vel[i, j, k] = nv
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    # getBulkVel: mean of the fluid/outflow cells in the 3x3x3 neighbourhood.
    bulk = wp.vec3()
    cnt = int(0)
    for n in range(27):
        ii = i + n % 3 - 1
        jj = j + (n // 3) % 3 - 1
        kk = k + n // 9 - 1
        if ii >= 0 and jj >= 0 and kk >= 0 and ii < nx and jj < ny and kk < nz:
            if (flags[ii, jj, kk] & (FLUID | OUTFLOW)) != 0:
                bulk = bulk + new_vel[ii, jj, kk]
                cnt = cnt + 1
    if cnt > 0:
        bulk = bulk / float(cnt)
    p = wp.vec3i(i, j, k)
    change = nv - vel[i, j, k]
    r = wp.vec3()
    m = int(0)
    for c in range(3):
        e = _unit(c)
        factor = time_step * wp.max(1.0, wp.abs(bulk[c]))
        found = bool(False)
        for dist in range(1, 3):
            if not found:
                from_lo = _is_fluid_at(flags, p - e * dist)
                from_up = _is_fluid_at(flags, p + e * dist)
                # Mantaflow takes the neighbour value from distance 1 even when
                # the fluid cell it found is 2 away.
                if from_lo:
                    r = r + change / factor + _at(new_vel, p - e)
                    m = m + 1
                if from_up:
                    r = r + change / factor + _at(new_vel, p + e)
                    m = m + 1
                found = from_lo or from_up
    if m > 0:
        r = r / float(m)
    vel[i, j, k] = r


class WarpBackend(GuidingMixin, FlameMixin, MixFracMixin, FractionsMixin, SecondaryMixin, LiquidMixin, Backend):
    name = "warp"

    def __init__(self, device: str | None = None, managed: bool = False):
        wp.init()
        self.device = wp.get_device(device) if device else wp.get_preferred_device()
        # CUDA managed memory: grids may exceed VRAM and page through system RAM (about 2x slower).
        # Set before the first array; it applies to every later allocation on this device.
        # The allocator is per device and global, so a plain backend restores the built-in one.
        self.managed = bool(managed) and self.device.is_cuda
        if self.device.is_cuda:
            wp.set_device_allocator(self.device, wp.CudaManagedAllocator() if self.managed else None)
        self._allocated = 0  # bytes requested through this backend (no frees)
        self._reduce = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._tmp = {}  # advection work grids, see _scratch
        self._wl_coeffs = None  # wavelet filter, uploaded on first use
        self._pressure = None  # MGPressureSolver for the current domain shape
        self._pressure_counted = 0  # bytes of it already in _allocated
        self._diffusion = None  # DiffusionSolver, allocated on first use (48 B/cell)
        self._viscosity = None  # ViscositySolver, allocated on first use (~76 B/cell)
        self._viscosity_coupled = None  # High Quality solve, CODE-LIQ-014 (+24 B/cell on first use)
        self._ray_dirs = wp.array(_RAY_DIRS, dtype=wp.vec3, device=self.device)
        self._dummy_vec = wp.zeros((1, 1, 1), dtype=wp.vec3, device=self.device)  # unused kernel args
        self._dummy_f = wp.zeros((1, 1, 1), dtype=wp.float32, device=self.device)
        self._dummy_vec1 = wp.zeros(1, dtype=wp.vec3, device=self.device)

    # ---- device and memory -------------------------------------------------

    @property
    def device_name(self) -> str:
        return self.device.name

    @property
    def on_cpu(self) -> bool:
        return not self.device.is_cuda

    @property
    def memory_mode(self) -> str:
        return "cpu" if self.on_cpu else ("managed" if self.managed else "gpu")

    def memory(self) -> BackendMemory:
        d = self.device
        if d.is_cuda and wp.is_mempool_enabled(d) and not self.managed:
            allocated = wp.get_mempool_used_mem_current(d)
        else:
            allocated = self._allocated
        if not d.is_cuda:  # Warp reports 0 for the CPU; use system RAM
            total, free = system_memory()
            return BackendMemory(total=total, free=free, allocated=allocated)
        if self.managed:  # VRAM plus the RAM budget (see estimate.managed_budget)
            ram_total, ram_free = system_memory()
            return BackendMemory(total=d.total_memory + max(0, ram_total - RAM_RESERVE),
                                 free=managed_budget(d.free_memory, ram_free), allocated=allocated)
        return BackendMemory(total=d.total_memory, free=d.free_memory, allocated=allocated)

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    # ---- allocation and transfer -------------------------------------------

    def _new(self, shape: Shape, dtype) -> wp.array:
        a = wp.empty(tuple(shape), dtype=dtype, device=self.device)
        self._allocated += a.capacity
        return a

    def scalar_grid(self, shape: Shape, value: float = 0.0) -> ScalarGrid:
        a = self._new(shape, wp.float32)
        a.fill_(value)
        return a

    def mac_grid(self, shape: Shape) -> MACGrid:
        a = self._new(shape, wp.vec3)
        a.zero_()
        return a

    def vector_grid(self, shape: Shape) -> VectorGrid:
        a = self._new(shape, wp.vec3)
        a.zero_()
        return a

    def flag_grid(self, shape: Shape) -> FlagGrid:
        a = self._new(shape, wp.int32)
        a.fill_(int(Flags.FLUID))
        return a

    def fill(self, grid: Grid, value: float | int | Vec3) -> None:
        if grid.dtype == wp.vec3:
            grid.fill_(wp.vec3(*value))
        elif grid.dtype == wp.int32:
            grid.fill_(int(value))
        else:
            grid.fill_(float(value))

    def copy(self, dst: Grid, src: Grid) -> None:
        if dst.dtype != src.dtype or dst.shape != src.shape:
            raise ValueError("copy: grids differ in kind or shape")
        wp.copy(dst, src)

    def join(self, dst, src, keep_max=False):
        if dst.dtype != wp.float32 or dst.shape != src.shape:
            raise ValueError("join: needs two scalar grids of the same shape")
        wp.launch(_join_kernel, dim=dst.shape, inputs=[dst, src, int(keep_max)], device=self.device)

    def mult_const(self, grid, factor):
        kernel = _mult_const_vec_kernel if grid.dtype == wp.vec3 else _mult_const_kernel
        wp.launch(kernel, dim=grid.shape, inputs=[grid, float(factor)], device=self.device)

    def add_const(self, grid, value):
        wp.launch(_add_const_kernel, dim=grid.shape, inputs=[grid, float(value)], device=self.device)

    def set_bound_neumann(self, grid, boundary_width=1):
        wp.launch(_bound_neumann_kernel, dim=grid.shape, inputs=[grid, int(boundary_width)],
                  device=self.device)

    def mirror_slab(self, grid):
        if grid.shape[1] != 3:
            raise ValueError(f"mirror_slab: grid {grid.shape} is not a 2D slab (ny == 3)")
        kernel = _mirror_slab_mac_kernel if grid.dtype == wp.vec3 else _mirror_slab_kernel
        wp.launch(kernel, dim=(grid.shape[0], grid.shape[2]), inputs=[grid], device=self.device)

    def to_numpy(self, grid: Grid) -> np.ndarray:
        a = grid.numpy()
        # On CUDA numpy() is already a fresh download; on the CPU it is a view of the
        # grid's own memory, which later kernels would change under the caller.
        return a.copy() if grid.device.is_cpu else a

    def from_numpy(self, grid: Grid, array: np.ndarray) -> None:
        expected = grid.shape + ((3,) if grid.dtype == wp.vec3 else ())
        if array.shape != expected:
            raise ValueError(f"from_numpy: shape {array.shape}, expected {expected}")
        np_dtype = np.int32 if grid.dtype == wp.int32 else np.float32
        grid.assign(np.ascontiguousarray(array, dtype=np_dtype))

    # ---- reductions --------------------------------------------------------

    def sum(self, grid: ScalarGrid) -> float:
        return float(wp.utils.array_sum(grid))

    def particle_sum(self, parts) -> float:
        if not parts.n:
            return 0.0
        return float(sum(sum(wp.utils.array_sum(a[:parts.n])) for a in (parts.pos, parts.vel)))

    def _column_max(self, kernel, grid) -> float:
        self._reduce.zero_()
        wp.launch(kernel, dim=grid.shape[:2], inputs=[grid, self._reduce], device=self.device)
        return float(self._reduce.numpy()[0])

    def max_abs(self, grid: ScalarGrid) -> float:
        return self._column_max(_max_abs_kernel, grid)

    def max_velocity(self, vel: MACGrid) -> float:
        # Raw face triples, as Mantaflow's MACGrid::getMax() does.
        return float(np.sqrt(self._column_max(_max_length_sq_kernel, vel)))

    # ---- domain ------------------------------------------------------------

    def init_domain(self, flags: FlagGrid, boundary_width: int = 0,
                    outflow: str = "", phi_walls: ScalarGrid | None = None) -> None:
        bad = set(outflow) - set("xXyYzZ")
        if bad:
            raise ValueError(f"init_domain: bad outflow sides {''.join(sorted(bad))!r}")
        types = vec6i(*[int(Flags.OUTFLOW) if side in outflow else int(Flags.OBSTACLE)
                        for side in "xXyYzZ"])
        wp.launch(_init_domain_kernel, dim=flags.shape,
                  inputs=[flags, int(boundary_width), types], device=self.device)
        if phi_walls is not None:
            walls = vec6i(*[int(side not in outflow) for side in "xXyYzZ"])
            wp.launch(_wall_levelset_kernel, dim=phi_walls.shape,
                      inputs=[phi_walls, int(boundary_width), walls], device=self.device)

    # ---- meshes ------------------------------------------------------------

    def create_mesh(self, verts, tris):
        m = WarpMesh(verts, tris, self.device)
        self._allocated += m.nbytes
        return m

    def set_mesh_transform(self, mesh, matrix, dt=0.0):
        # Transformed on the host: the bounding box is needed there anyway,
        # and the vertex upload is a few KB.
        m = np.asarray(matrix, dtype=np.float32)
        p = mesh.verts @ m[:3, :3].T + m[:3, 3]
        mesh.points.assign(p)
        if dt > 0.0 and mesh.prev is not None:
            mesh.velocities.assign(((p - mesh.prev) / np.float32(dt)).astype(np.float32))
        else:
            mesh.velocities.zero_()
        mesh.prev = p
        mesh.mesh.refit()
        mesh._set_bounds(p)

    def _mesh_box(self, mesh, shape):
        """Blender's emission-map box: bbox + ceil(5.196) cells, clamped to
        the domain. None if empty."""
        lo = np.clip(mesh.lo - 6, 0, shape)
        hi = np.clip(mesh.hi + 6, 0, shape)
        if np.any(hi <= lo):
            return None
        return wp.vec3i(*(int(v) for v in lo)), tuple(int(v) for v in hi - lo)

    def rasterize_mesh_flow(self, mesh, phi_in, emission_in, density_in, heat_in, density, heat,
                            flow_density, temperature, surface_distance=1.0, volume_density=0.0,
                            plane_init=False, fuel_in=None, react_in=None, fuel=None, react=None,
                            fuel_amount=0.0, emit_fuel=False):
        box = self._mesh_box(mesh, phi_in.shape)
        if box is None:
            return
        fire = fuel_in is not None
        fg = [fuel_in, react_in, fuel, react] if fire else [self._dummy_f] * 4
        wp.launch(_rasterize_mesh_flow_kernel, dim=box[1],
                  inputs=[mesh.mesh.id, self._ray_dirs, box[0], phi_in, emission_in, density_in,
                          heat_in, density, heat, float(flow_density), float(temperature),
                          float(surface_distance), float(volume_density), int(plane_init),
                          *fg, int(fire), float(fuel_amount), int(emit_fuel)],
                  device=self.device)

    def rasterize_mesh_obstacle(self, mesh, phi_obs, surface_thickness=0.0, plane_init=False,
                                obvel=None, num_obs=None, guide_mode=None, vel_multi=1.0):
        use_vel = obvel is not None and num_obs is not None
        scale = vel_multi * _GUIDE_MODE_SCALE[guide_mode] if guide_mode else 1.0
        box = self._mesh_box(mesh, phi_obs.shape)
        if box is None:
            return
        wp.launch(_rasterize_mesh_obstacle_kernel, dim=box[1],
                  inputs=[mesh.mesh.id, self._ray_dirs, box[0], phi_obs,
                          float(surface_thickness), int(plane_init),
                          obvel if use_vel else self._dummy_vec,
                          num_obs if use_vel else self._dummy_f, int(use_vel),
                          float(scale), int(guide_mode == "MAX")],
                  device=self.device)

    def rasterize_mesh_liquid_flow(self, mesh, phi, surface_distance=0.0, plane_init=False,
                                   invel=None, vel_normal=0.0, vel_multi=1.0,
                                   vel_coord=(0.0, 0.0, 0.0)):
        box = self._mesh_box(mesh, phi.shape)
        if box is None:
            return
        use_vel = invel is not None
        normals = self._dummy_vec1
        if use_vel and vel_normal != 0.0:
            normals = wp.array(mesh.vertex_normals(), dtype=wp.vec3, device=self.device)
        wp.launch(_rasterize_mesh_liquid_kernel, dim=box[1],
                  inputs=[mesh.mesh.id, self._ray_dirs, box[0], phi, float(surface_distance),
                          int(plane_init), invel if use_vel else self._dummy_vec, int(use_vel),
                          mesh.indices, normals, float(vel_normal), float(vel_multi),
                          wp.vec3(*vel_coord)],
                  device=self.device)

    def extrapolate_vec3_simple(self, vel, phi, distance=4, inside=False):
        d = self.device
        sh = vel.shape
        tmp = self._tmp.get("extrap")
        if tmp is None or tmp.shape != sh:
            tmp = self._tmp["extrap"] = self._new(sh, wp.int32)
        wp.launch(_mark_skip_kernel, dim=sh, inputs=[phi, tmp, int(inside)], device=d)
        wp.launch(_first_layer_kernel, dim=sh, inputs=[tmp], device=d)
        for layer in range(2, distance + 1):
            wp.launch(_extrapolate_vec_kernel, dim=sh, inputs=[vel, tmp, layer], device=d)
        wp.launch(_set_remaining_vec_kernel, dim=sh, inputs=[vel, tmp], device=d)

    def obstacle_velocity(self, phi_obs, obvel_c, num_obs, obvel):
        sh = obvel_c.shape
        wp.launch(_safe_divide_vec_kernel, dim=sh, inputs=[obvel_c, num_obs], device=self.device)
        self.extrapolate_vec3_simple(obvel_c, phi_obs, 6, inside=True)
        self.extrapolate_vec3_simple(obvel_c, phi_obs, 3)
        wp.launch(_resample_to_mac_kernel, dim=sh, inputs=[obvel_c, obvel], device=self.device)

    def set_inflow_flags(self, flags, phi_in):
        wp.launch(_set_inflow_flags_kernel, dim=flags.shape, inputs=[flags, phi_in],
                  device=self.device)

    def set_obstacle_flags(self, flags, phi_obs):
        wp.launch(_set_obstacle_flags_kernel, dim=flags.shape, inputs=[flags, phi_obs],
                  device=self.device)

    def apply_emission(self, flags, target, source, emission, absolute=True):
        wp.launch(_apply_emission_kernel, dim=flags.shape,
                  inputs=[flags, target, source, emission, int(absolute)], device=self.device)

    def dissolve(self, flags, grid, speed, log_falloff, scale):
        dydx = float(scale) / float(speed)
        wp.launch(_dissolve_kernel, dim=flags.shape,
                  inputs=[flags, grid, int(log_falloff), dydx, 1.0 - dydx], device=self.device)

    def _scratch(self, like: wp.array, shape: Shape | None = None) -> tuple[wp.array, wp.array]:
        # Two work grids per (shape, dtype), kept for later calls.
        shape = tuple(shape or like.shape)
        key = (shape, like.dtype)
        if key not in self._tmp:
            self._tmp[key] = (self._new(shape, like.dtype), self._new(shape, like.dtype))
        return self._tmp[key]

    @staticmethod
    def _check_advect(shape, order, clamp_mode):
        if min(shape) < 3:
            raise ValueError(f"advect: needs a 3D grid at least 3 cells wide, got {shape}")
        if order not in (1, 2):
            raise ValueError(f"advect: order must be 1 or 2, got {order}")
        if clamp_mode not in (1, 2):
            raise ValueError(f"advect: clamp_mode must be 1 or 2, got {clamp_mode}")

    # CODE-GAS-015 prototype switch ("BFECC", "MC_CUBIC"; see warp_advect_proto.py). No setting sets it.
    advect_scheme = "MACCORMACK"

    def advect(self, flags, vel, grid, dt, order=2, strength=1.0, clamp_mode=2):
        self._check_advect(grid.shape, order, clamp_mode)
        if order == 2 and self.advect_scheme != "MACCORMACK":
            from . import warp_advect_proto as _ap
            if _ap.advect(self, flags, vel, grid, dt, strength, clamp_mode):
                return
        fwd, bwd = self._scratch(grid)
        dev = self.device
        sl, mc = ((_wn._sl_vec_kernel, _wn._mc_vec_kernel) if grid.dtype == wp.vec3
                  else (_sl_cell_kernel, _mc_cell_kernel))
        wp.launch(sl, dim=grid.shape, inputs=[vel, grid, fwd, float(dt)], device=dev)
        if order == 2:
            wp.launch(sl, dim=grid.shape, inputs=[vel, fwd, bwd, -float(dt)], device=dev)
            wp.launch(mc, dim=grid.shape,
                      inputs=[flags, vel, grid, fwd, bwd, float(dt), float(strength), int(clamp_mode)],
                      device=dev)
            fwd = bwd
        wp.copy(grid, fwd)

    def advect_velocity(self, flags, vel, dt, order=2, strength=1.0, clamp_mode=2, field=None):
        self._check_advect(vel.shape, order, clamp_mode)
        field = vel if field is None else field
        if order == 2 and self.advect_scheme != "MACCORMACK":
            from . import warp_advect_proto as _ap
            return _ap.advect_velocity(self, flags, vel, dt, strength, clamp_mode, field)
        fwd, bwd = self._scratch(vel)
        dev = self.device
        wp.launch(_sl_mac_kernel, dim=vel.shape, inputs=[vel, field, fwd, float(dt)], device=dev)
        if order == 2:
            wp.launch(_sl_mac_kernel, dim=vel.shape, inputs=[vel, fwd, bwd, -float(dt)], device=dev)
            wp.launch(_mc_mac_kernel, dim=vel.shape,
                      inputs=[flags, vel, field, fwd, bwd, float(dt), float(strength), int(clamp_mode)],
                      device=dev)
            fwd = bwd
        wp.launch(_outflow_vel_kernel, dim=vel.shape,
                  inputs=[flags, fwd, field, max(1.0, float(dt) * 4.0)], device=dev)

    def reflect(self, vel, pre):
        wp.launch(_reflect_kernel, dim=vel.shape, inputs=[vel, pre], device=self.device)

    def reset_outflow(self, flags, grid):
        wp.launch(_reset_outflow_kernel, dim=flags.shape, inputs=[flags, grid],
                  device=self.device)

    def vorticity_confinement(self, flags, vel, strength, strength_cell=None, cell_scale=1.0):
        if min(vel.shape) < 3:
            raise ValueError(f"vorticity_confinement: needs a 3D grid at least 3 cells wide, got {vel.shape}")
        curl, force = self._scratch(vel)
        dev = self.device
        wp.launch(_curl_kernel, dim=vel.shape, inputs=[vel, curl], device=dev)
        cell = strength_cell if strength_cell is not None else self._dummy_f
        wp.launch(_conf_force_kernel, dim=vel.shape,
                  inputs=[curl, float(strength), cell, int(strength_cell is not None),
                          float(cell_scale), force], device=dev)
        wp.launch(_apply_force_kernel, dim=vel.shape, inputs=[flags, force, vel, 1.0], device=dev)

    def process_burn(self, fuel, density, react, heat, burning_rate, flame_smoke,
                     ignition_temp, max_temp, dt):
        h = heat if heat is not None else self._dummy_f
        wp.launch(_process_burn_kernel, dim=fuel.shape,
                  inputs=[fuel, density, react, h, int(heat is not None),
                          float(np.float32(burning_rate) * np.float32(dt)), float(flame_smoke), float(ignition_temp),
                          float(max_temp)], device=self.device)

    def update_flame(self, react, flame):
        wp.launch(_update_flame_kernel, dim=react.shape, inputs=[react, flame], device=self.device)

    def combust(self, fuel, density, heat, expansion_div, burning_rate, ignition_temp, max_temp,
                heat_release, flame_smoke, cooling, cooling_radiative, expansion, dt, dt_seconds):
        wp.launch(_combust_kernel, dim=fuel.shape,
                  inputs=[fuel, density, heat, expansion_div, float(-np.expm1(-burning_rate * dt)),
                          float(ignition_temp), float(max_temp), float(heat_release),
                          float(0.05 * flame_smoke), float(cooling * dt_seconds),
                          float(cooling_radiative * dt_seconds), float(1.0 / max_temp),
                          float(expansion / dt)], device=self.device)

    def fire_forces(self, flags, vel, heat, disturbance, size_cells, phase, ignition_temp,
                    shredding, max_temp, seed=0):
        if not disturbance and not shredding:
            return
        _, force = self._scratch(vel)
        tick = int(np.floor(phase))
        wp.launch(_fire_forces_kernel, dim=vel.shape,
                  inputs=[heat, vel, force, float(1.0 / max(size_cells, 1.0)),
                          int(seed * 1000003 + tick * 7919), int(seed * 1000003 + (tick + 1) * 7919),
                          float(phase - tick), float(disturbance), float(1.0 / ignition_temp),
                          float(shredding), float(ignition_temp), float(1e-3 * max_temp)],
                  device=self.device)
        wp.launch(_apply_force_kernel, dim=vel.shape, inputs=[flags, force, vel, 1.0],
                  device=self.device)

    def update_flame_temperature(self, heat, flame, ignition_temp, max_temp):
        wp.launch(_flame_from_heat_kernel, dim=heat.shape,
                  inputs=[heat, flame, float(ignition_temp), float(1.0 / (max_temp - ignition_temp))],
                  device=self.device)

    def density_face_weights(self, heat, weights, ambient_kelvin, flame_kelvin, flame_max_temp):
        c = (float(flame_kelvin) - float(ambient_kelvin)) / (float(ambient_kelvin) * float(flame_max_temp))
        wp.launch(_density_weights_kernel, dim=heat.shape, inputs=[heat, weights, c], device=self.device)

    def add_buoyancy(self, flags, vel, grid, strength):
        wp.launch(_buoyancy_kernel, dim=flags.shape,
                  inputs=[flags, grid, vel, wp.vec3(*[float(c) for c in strength])],
                  device=self.device)

    def compute_forces(self, flags, vel, force, fields, weights, origin, cell_size, fps,
                       mat3=None, density=None, fuel=None, phi_obs=None):
        packed = pack_fields(fields, weights, self.device)
        if packed is None:
            return False
        m = np.eye(3) if mat3 is None else np.asarray(mat3, np.float64)
        mode = (int(density is not None) | 2 * int(fuel is not None and density is not None)
                | 4 * int(phi_obs is not None))
        f = self._dummy_f
        wp.launch(compute_forces_kernel, dim=flags.shape,
                  inputs=[flags, vel, packed, wp.vec3(*map(float, origin)),
                          wp.vec3(*map(float, cell_size)), wp.mat33(*m.ravel().tolist()),
                          wp.mat33(*np.linalg.inv(m).ravel().tolist()), 1.0 / max(flags.shape),
                          float(fps), density if density is not None else f,
                          fuel if fuel is not None else f, phi_obs if phi_obs is not None else f,
                          mode, noise_hash(self.device), force], device=self.device)
        return True

    def add_force_field(self, flags, vel, force, scale=1.0):
        wp.launch(_apply_force_kernel, dim=flags.shape, inputs=[flags, force, vel, float(scale)],
                  device=self.device)

    def set_wall_bcs(self, flags, vel, obvel=None, fractions=None, phi_obs=None):
        if fractions is not None and phi_obs is not None:
            self._set_wall_bcs_frac(flags, vel, phi_obs)
            return
        wp.launch(_wall_bcs_kernel, dim=flags.shape,
                  inputs=[flags, vel, self._dummy_vec if obvel is None else obvel, int(obvel is not None)],
                  device=self.device)

    def solve_pressure(self, flags, vel, pressure, accuracy=1e-3,
                       max_iter_factor=1.5,
                       zero_pressure_fixing=False, divergence=0.0, fractions=None,
                       obvel=None, phi=None, _phi_gen=None, divergence_grid=None,
                       surface_tension=0.0, parallel_coarsening=False, face_weights=None,
                       _weights_gen=None) -> PressureStats:
        # max_iter_factor is unused: Mantaflow caps MG-preconditioned CG at 100 iterations.
        # phi may change between calls, so each call rebuilds the matrix unless the
        # caller passes the same _phi_gen (guiding_solve: one build per PD solve).
        if self._pressure is None or self._pressure.shape != tuple(flags.shape):
            self._allocated -= self._pressure_counted  # the old solver is dropped
            self._pressure = MGPressureSolver(flags.shape, self.device)
            self._pressure_counted = 0
        if phi is not None and _phi_gen is None:
            self._phi_gen = _phi_gen = getattr(self, "_phi_gen", 0) + 1
        if face_weights is not None and _weights_gen is None:  # weights change every substep
            self._weights_gen = _weights_gen = getattr(self, "_weights_gen", 0) + 1
        curv = None
        if phi is not None and surface_tension != 0.0:
            curv = self._scratch(phi)[0]
            wp.launch(_curvature_kernel, dim=phi.shape, inputs=[phi, curv, 1.0], device=self.device)
        stats = self._pressure.solve(flags, vel, pressure, float(accuracy), zero_pressure_fixing,
                                    float(divergence), fractions, obvel, self.fractions_gen,
                                    phi, _phi_gen or 0, divergence_grid, curv, float(surface_tension),
                                    parallel_coarsening=parallel_coarsening,
                                    face_weights=face_weights, weights_gen=_weights_gen or 0)
        # Some arrays are built on the first solve, so count what exists now (CUDA reads the pool instead).
        now = self._pressure.nbytes()
        self._allocated += now - self._pressure_counted
        self._pressure_counted = now
        return stats

    def apply_viscosity(self, flags, phi, vel, viscosity, dt, dx, accuracy=1e-9,
                        max_iter_factor=1.5) -> PressureStats:
        if self._viscosity is None or self._viscosity.shape != tuple(flags.shape):
            self._viscosity = ViscositySolver(flags.shape, self.device)
            self._allocated += self._viscosity.nbytes()
        return self._viscosity.solve(flags, phi, vel, float(viscosity), float(dt), float(dx),
                                     float(accuracy), float(max_iter_factor))

    def apply_viscosity_coupled(self, flags, phi, vel, viscosity, dt, dx) -> PressureStats:
        from .warp_viscosity_coupled import CoupledViscositySolver
        if self._viscosity is None or self._viscosity.shape != tuple(flags.shape):
            self._viscosity = ViscositySolver(flags.shape, self.device)
            self._allocated += self._viscosity.nbytes()
        if self._viscosity_coupled is None or self._viscosity_coupled.base is not self._viscosity:
            self._viscosity_coupled = CoupledViscositySolver(self._viscosity)
            self._allocated += self._viscosity_coupled.nbytes()
        return self._viscosity_coupled.solve(flags, phi, vel, float(viscosity), float(dt), float(dx))

    def solve_diffusion(self, flags, vel, alpha, accuracy=1e-4, max_iter_factor=1.0, kelvin=None) -> PressureStats:
        if self._diffusion is None or self._diffusion.shape != tuple(flags.shape):
            self._diffusion = DiffusionSolver(flags.shape, self.device)
            self._allocated += self._diffusion.nbytes()
        return self._diffusion.solve(flags, vel, float(alpha), float(accuracy), float(max_iter_factor), kelvin)

    def gas_kelvin(self, heat, kelvin, ambient_kelvin, flame_kelvin, flame_max_temp):
        k = (float(flame_kelvin) - float(ambient_kelvin)) / float(flame_max_temp)
        wp.launch(gas_kelvin_kernel, dim=heat.shape, inputs=[heat, kelvin, float(ambient_kelvin), k],
                  device=self.device)

    # ---- noise / upres -----------------------------------------------------

    @staticmethod
    def _size_factor(src_shape, dst_shape) -> np.ndarray:
        # calcGridSizeFactor, in float32 like the C++
        return np.float32(src_shape) / np.float32(dst_shape)

    def interpolate_grid(self, target, source):
        sf = self._size_factor(source.shape, target.shape)
        wp.launch(_wn._interpolate_grid_kernel, dim=target.shape,
                  inputs=[source, target, wp.vec3(*sf), wp.vec3(*(sf * np.float32(0.5)))],
                  device=self.device)

    def interpolate_mac(self, target, source):
        sf = self._size_factor(source.shape, target.shape)
        wp.launch(_wn._interpolate_mac_kernel, dim=target.shape,
                  inputs=[source, target, wp.vec3(*sf), wp.vec3(*(sf * np.float32(0.5)))],
                  device=self.device)

    def reset_uv_grid(self, uv, offset=(0.0, 0.0, 0.0)):
        wp.launch(_wn._reset_uv_kernel, dim=uv.shape, inputs=[uv, wp.vec3(*offset)], device=self.device)

    def update_uv_weight(self, uv, time, dt, reset_time, index, num_uvs, offset=(0.0, 0.0, 0.0)):
        w, reset = uv_weight(time, dt, reset_time, index, num_uvs)
        if reset:
            self.reset_uv_grid(uv, offset)
        wp.launch(_wn._set_uv_weight_kernel, dim=1, inputs=[uv, w], device=self.device)
        return w

    def compute_energy(self, flags, vel, energy):
        wp.launch(_wn._energy_kernel, dim=flags.shape, inputs=[flags, vel, energy], device=self.device)

    def extrapolate_simple_flags(self, flags, val, distance=4, flag_from=Flags.FLUID,
                                 flag_to=Flags.OBSTACLE):
        if val.dtype == wp.int32:
            kernel = _wn._extrap_int_kernel
        elif val.dtype == wp.float32:
            kernel = _wn._extrap_float_kernel
        else:
            raise ValueError("extrapolate_simple_flags: needs a flag or scalar grid")
        tmp = self._scratch(flags)[0]
        dev = self.device
        wp.launch(_wn._extrap_init_kernel, dim=flags.shape, inputs=[flags, int(flag_from), tmp], device=dev)
        for d in range(1, int(distance) + 1):
            wp.launch(kernel, dim=flags.shape, inputs=[flags, val, tmp, d, int(flag_to)], device=dev)

    def compute_wavelet_coeffs(self, grid):
        dev = self.device
        if self._wl_coeffs is None:
            self._wl_coeffs = wp.array(WAVELET_A_COEFFS, dtype=wp.float32, device=dev)
        t, t2 = self._scratch(grid)
        src = grid
        for axis in range(3):
            half = list(grid.shape)
            half[axis] //= 2
            h = self._scratch(grid, half)[0]
            wp.launch(_wn._wl_down_kernel, dim=h.shape, inputs=[src, h, axis, self._wl_coeffs], device=dev)
            dst = t if src is not t else t2
            wp.launch(_wn._wl_up_kernel, dim=grid.shape, inputs=[h, dst, axis], device=dev)
            src = dst
        wp.launch(_wn._wl_residual_kernel, dim=grid.shape, inputs=[grid, src], device=dev)
        wp.launch(_wn._wl_smooth_kernel, dim=grid.shape,
                  inputs=[src, grid, float(np.float32(1.0 / 6.0))], device=dev)

    def noise_tile(self, tile):
        a = np.ascontiguousarray(np.moveaxis(np.asarray(tile, np.float32), 0, -1))
        out = self._new(a.shape[:3], wp.vec3)
        out.assign(a)
        return out

    def apply_noise_vec3(self, flags, target, tile, weight, uv, scale, scale_spatial,
                         gs_inv, seed_offset, time, pos_scale):
        if weight.shape != uv.shape:
            raise ValueError("apply_noise_vec3: uv and weight grids must match")
        sf = self._size_factor(uv.shape, target.shape)
        f = lambda x: float(np.float32(x))
        wp.launch(_wn._apply_noise_kernel, dim=target.shape,
                  inputs=[flags, target, tile, weight, uv, int(uv.shape != target.shape),
                          wp.vec3(*sf), f(scale), f(scale_spatial), f(gs_inv),
                          wp.vec3(*seed_offset), f(time), wp.vec3(*pos_scale)],
                  device=self.device)


# Imported last: its kernels use the helpers above.
from . import warp_noise as _wn  # noqa: E402
