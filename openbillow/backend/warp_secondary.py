# SPDX-License-Identifier: GPL-3.0-or-later
"""Secondary particle kernels (spray, foam, bubbles).

Ported from Mantaflow plugin/secondaryparticles.cpp
(`flipComputeSecondaryParticlePotentials`, `flipSampleSecondaryParticles`
mode 'single', `flipUpdateSecondaryParticles` mode 'linear',
`flipDeleteParticlesInObstacle`) and commonkernels.h `GradientOp`, with
vectorbase.h `norm` / `getNormalized` (std::hypot as libstdc++'s float
__hypot3). Mixed float/double expressions of the float build are kept: Mantaflow
writes literals like 1.732 * radius and 0.6 as doubles. `WarpBackend` inherits
`SecondaryMixin`.

Sampling is serial in Mantaflow (one RandomStream for the whole grid); here a
scan over the cells in Mantaflow's order gives each cell its random-number and
particle offsets, so the result is the same particle for particle. Only cos/sin
may differ from the C library by an ulp.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from .base import Flags, ParticleFlags
from .warp_liquid import _f4, _mac_interp

wp.set_module_options({"fuse_fp": False})

FLUID = wp.constant(int(Flags.FLUID))
NOT_NEIGHBOR = wp.constant(int(Flags.OBSTACLE | Flags.OUTFLOW | Flags.INFLOW))
SKIP_CELL = wp.constant(int(Flags.OBSTACLE | Flags.OUTFLOW))
PDELETE = wp.constant(int(ParticleFlags.DELETE))
PSPRAY = wp.constant(int(ParticleFlags.SPRAY))
PBUBBLE = wp.constant(int(ParticleFlags.BUBBLE))
PFOAM = wp.constant(int(ParticleFlags.FOAM))
PTRACER = wp.constant(int(ParticleFlags.TRACER))
VEPS = wp.constant(float(np.float32(1.0e-6)))  # VECTOR_EPSILON (1e-6f)
M_PI = wp.constant(wp.float64(math.pi))


@wp.func
def _hypot3(v: wp.vec3):
    # libstdc++ std::__hypot3<float>
    x = wp.abs(v[0])
    y = wp.abs(v[1])
    z = wp.abs(v[2])
    a = wp.where(x < y, wp.where(y < z, z, y), wp.where(x < z, z, x))
    if a == 0.0:
        return float(0.0)
    return a * wp.sqrt((x / a) * (x / a) + (y / a) * (y / a) + (z / a) * (z / a))


@wp.func
def _norm(v: wp.vec3):
    # vectorbase.h norm(); l - 1 is exact in float wherever the test can pass.
    l = _hypot3(v)
    if l <= VEPS:
        return float(0.0)
    if wp.abs(l - 1.0) < VEPS:
        return float(1.0)
    return l


@wp.func
def _normalized(v: wp.vec3):
    # vectorbase.h getNormalized(); float(1. / l) is the float quotient.
    l = _hypot3(v)
    if wp.abs(l - 1.0) < VEPS:
        return v
    if l > VEPS:
        fac = 1.0 / l
        return wp.vec3(v[0] * fac, v[1] * fac, v[2] * fac)
    return wp.vec3()


@wp.func
def _dot(a: wp.vec3, b: wp.vec3):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def _cross(a: wp.vec3, b: wp.vec3):
    return wp.vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@wp.func
def _min(a: float, b: float):
    # std::min (NaN passes through as Mantaflow's would)
    return wp.where(b < a, b, a)


@wp.func
def _clamp_potential(p: float, lo: float, hi: float):
    return (_min(p, hi) - _min(p, lo)) / (hi - lo)


@wp.func
def _centered(v: wp.array4d(dtype=wp.float32), i: int, j: int, k: int):
    # MACGrid::getCentered
    return wp.vec3(0.5 * (v[i, j, k, 0] + v[i + 1, j, k, 0]),
                   0.5 * (v[i, j, k, 1] + v[i, j + 1, k, 1]),
                   0.5 * (v[i, j, k, 2] + v[i, j, k + 1, 2]))


@wp.kernel
def _gradient_kernel(phi: wp.array3d(dtype=wp.float32), normal: wp.array3d(dtype=wp.vec3)):
    # GradientOp (bnd 1); Mantaflow never writes the border, which stays 0.
    i, j, k = wp.tid()
    if (i < 1 or j < 1 or k < 1 or i >= phi.shape[0] - 1 or j >= phi.shape[1] - 1
            or k >= phi.shape[2] - 1):
        normal[i, j, k] = wp.vec3()
        return
    normal[i, j, k] = wp.vec3(0.5 * (phi[i + 1, j, k] - phi[i - 1, j, k]),
                              0.5 * (phi[i, j + 1, k] - phi[i, j - 1, k]),
                              0.5 * (phi[i, j, k + 1] - phi[i, j, k - 1]))


@wp.kernel
def _potentials_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array4d(dtype=wp.float32),
                       normal: wp.array3d(dtype=wp.vec3), radius: int, h: wp.float64,
                       ta_lo: float, ta_hi: float, wc_lo: float, wc_hi: float,
                       ke_lo: float, ke_hi: float, scale: float,
                       pot_ta: wp.array3d(dtype=wp.float32), pot_wc: wp.array3d(dtype=wp.float32),
                       pot_ke: wp.array3d(dtype=wp.float32), ratio: wp.array3d(dtype=wp.float32)):
    # knFlipComputeSecondaryParticlePotentials (bnd = radius), after the grids are cleared.
    i, j, k = wp.tid()
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    pot_ta[i, j, k] = 0.0
    pot_wc[i, j, k] = 0.0
    pot_ke[i, j, k] = 0.0
    ratio[i, j, k] = 0.0
    if i < radius or j < radius or k < radius or i >= nx - radius or j >= ny - radius or k >= nz - radius:
        return
    if (flags[i, j, k] & FLUID) == 0:
        return
    xi = scale * wp.vec3(float(i), float(j), float(k))
    vi = scale * _centered(vel, i, j, k)
    ni = _normalized(normal[i, j, k])
    vdiff = float(0.0)
    kappa = float(0.0)
    count_fluid = int(0)
    count_max = int(0)
    for x in range(i - radius, i + radius + 1):
        for y in range(j - radius, j + radius + 1):
            for z in range(k - radius, k + radius + 1):
                if x == i and y == j and z == k:
                    continue
                if x < 1 or y < 1 or z < 1 or x >= nx - 1 or y >= ny - 1 or z >= nz - 1:
                    continue
                f = flags[x, y, z]
                if (f & NOT_NEIGHBOR) != 0:
                    continue
                if (f & FLUID) != 0:
                    count_fluid += 1
                count_max += 1
                xj = scale * wp.vec3(float(x), float(y), float(z))
                vj = scale * _centered(vel, x, y, z)
                nj = _normalized(normal[x, y, z])
                xij = xi - xj
                vij = vi - vj
                w = wp.float64(1.0) - wp.float64(_norm(xij)) / h
                nxij = _normalized(xij)
                a = _norm(vij) * (1.0 - _dot(_normalized(vij), nxij))
                vdiff = wp.float32(wp.float64(vdiff) + wp.float64(a) * w)
                if _dot(nxij, ni) < 0.0:
                    kappa = wp.float32(wp.float64(kappa) + wp.float64(1.0 - _dot(ni, nj)) * w)
    ratio[i, j, k] = float(count_fluid) / float(count_max)
    pot_ta[i, j, k] = _clamp_potential(vdiff, ta_lo, ta_hi)
    if wp.float64(_dot(_normalized(vi), ni)) >= wp.float64(0.6):
        pot_wc[i, j, k] = _clamp_potential(kappa, wc_lo, wc_hi)
    ek = 62.5 * (vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2])  # Real(0.5) * 125 * normSquare
    pot_ke[i, j, k] = _clamp_potential(ek, ke_lo, ke_hi)


@wp.func
def _interp_scalar(g: wp.array3d(dtype=wp.float32), p: wp.vec3):
    # Grid::getInterpolated: trilinear, the value of cell i sits at coordinate i, clamped inside.
    nx = g.shape[0]
    ny = g.shape[1]
    nz = g.shape[2]
    x = wp.clamp(p[0], 0.0, float(nx - 1) - 1e-4)
    y = wp.clamp(p[1], 0.0, float(ny - 1) - 1e-4)
    z = wp.clamp(p[2], 0.0, float(nz - 1) - 1e-4)
    i = int(x)
    j = int(y)
    k = int(z)
    fx = x - float(i)
    fy = y - float(j)
    fz = z - float(k)
    return ((1.0 - fz) * ((1.0 - fy) * ((1.0 - fx) * g[i, j, k] + fx * g[i + 1, j, k])
                          + fy * ((1.0 - fx) * g[i, j + 1, k] + fx * g[i + 1, j + 1, k]))
            + fz * ((1.0 - fy) * ((1.0 - fx) * g[i, j, k + 1] + fx * g[i + 1, j, k + 1])
                    + fy * ((1.0 - fx) * g[i, j + 1, k + 1] + fx * g[i + 1, j + 1, k + 1])))


@wp.func
def _sub_point(i: int, j: int, k: int, q: int):
    # the 8 fixed sample points of a cell (multiple mode): cell centre +-0.25
    return wp.vec3(float(i) + wp.where((q & 1) != 0, 0.75, 0.25),
                   float(j) + wp.where((q & 2) != 0, 0.75, 0.25),
                   float(k) + wp.where((q & 4) != 0, 0.75, 0.25))


@wp.kernel
def _sample_count_kernel(flags: wp.array3d(dtype=wp.int32), pot_ta: wp.array3d(dtype=wp.float32),
                         pot_wc: wp.array3d(dtype=wp.float32), pot_ke: wp.array3d(dtype=wp.float32),
                         k_ta: float, k_wc: float, dt: float, mode: int,
                         count: wp.array3d(dtype=wp.int32), draws: wp.array3d(dtype=wp.int32)):
    # Particles per cell and random numbers it takes (single: 3 for the cell offset, 4 per
    # particle; multiple: 4 per particle), stored [k, j, i] so the flat scan runs in
    # Mantaflow's loop order.
    i, j, k = wp.tid()
    n = int(0)
    if (flags[i, j, k] & FLUID) != 0:
        if mode == 0:
            n = int(pot_ke[i, j, k] * (k_ta * pot_ta[i, j, k] + k_wc * pot_wc[i, j, k]) * dt)
        else:
            for q in range(8):
                xp = _sub_point(i, j, k, q)
                n += int(_interp_scalar(pot_ke, xp) * (k_ta * _interp_scalar(pot_ta, xp)
                                                       + k_wc * _interp_scalar(pot_wc, xp)) * dt)
    count[k, j, i] = n
    draws[k, j, i] = wp.where(n != 0, wp.where(mode == 0, 3, 0) + 4 * n, 0)


@wp.func
def _emit_cylinder(vel: wp.array4d(dtype=wp.float32), xi: wp.vec3, n: int, radius: float, dt: float,
                   l_min: float, l_max: float, temp: float, typ: int, rnd: wp.array(dtype=wp.float32),
                   r0: int, s0: int, pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                   life: wp.array(dtype=wp.float32), flag: wp.array(dtype=wp.int32)):
    # n particles in a cylinder of the given radius along the local flow; returns the next slot
    vi = _mac_interp(vel, xi)
    d = dt * vi
    e1 = _normalized(wp.vec3(d[2], 0.0, -d[0]))
    e2 = _normalized(_cross(e1, d))
    vn = _normalized(vi)
    dlen = _norm(d)
    r = r0
    s = s0
    for q in range(n):
        rad = radius * wp.sqrt(rnd[r])
        theta = wp.float32(wp.float64(rnd[r + 1] * 2.0) * M_PI)
        hgt = rnd[r + 2] * dlen
        rc = rad * wp.cos(theta)
        rs = rad * wp.sin(theta)
        pos[s] = xi + rc * e1 + rs * e2 + hgt * vn
        pvel[s] = rc * e1 + rs * e2 + vi
        life[s] = wp.float32(wp.float64((l_max - l_min) * temp + l_min)
                             + wp.float64(rnd[r + 3]) * wp.float64(0.1))
        flag[s] = typ
        r += 4
        s += 1
    return s


@wp.kernel
def _sample_emit_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array4d(dtype=wp.float32),
                        pot_ta: wp.array3d(dtype=wp.float32), pot_wc: wp.array3d(dtype=wp.float32),
                        pot_ke: wp.array3d(dtype=wp.float32), ratio: wp.array3d(dtype=wp.float32),
                        count: wp.array3d(dtype=wp.int32), slot: wp.array3d(dtype=wp.int32),
                        rstart: wp.array3d(dtype=wp.int32), rnd: wp.array(dtype=wp.float32),
                        base: int, dt: float, l_min: float, l_max: float, c_s: float, c_b: float,
                        k_ta: float, k_wc: float, mode: int,
                        pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                        life: wp.array(dtype=wp.float32), flag: wp.array(dtype=wp.int32)):
    # knFlipSampleSecondaryParticles (3D)
    i, j, k = wp.tid()
    n = count[k, j, i]
    if n == 0:
        return
    r = rstart[k, j, i]
    s = base + slot[k, j, i]
    ke = pot_ke[i, j, k]
    ta = pot_ta[i, j, k]
    wc = pot_wc[i, j, k]
    nr = ratio[i, j, k]
    typ = PFOAM
    if nr < c_s:
        typ = PSPRAY
    elif nr > c_b:
        typ = PBUBBLE
    if mode == 0:
        xi = wp.vec3(float(i), float(j), float(k)) + wp.vec3(rnd[r], rnd[r + 1], rnd[r + 2])
        r += 3
        s = _emit_cylinder(vel, xi, n, 0.5, dt, l_min, l_max,
                           (ke + ta + wc) / 3.0, typ, rnd, r, s, pos, pvel, life, flag)
        return
    for sq in range(8):
        xp = _sub_point(i, j, k, sq)
        ke_p = _interp_scalar(pot_ke, xp)
        ta_p = _interp_scalar(pot_ta, xp)
        wc_p = _interp_scalar(pot_wc, xp)
        np_ = int(ke_p * (k_ta * ta_p + k_wc * wc_p) * dt)
        if np_ > 0:
            s = _emit_cylinder(vel, xp, np_, 0.25, dt, l_min, l_max,
                               (ke_p + ta_p + wc_p) / 3.0, typ, rnd, r, s, pos, pvel, life, flag)
            r += 4 * np_


@wp.func
def _in_bounds(p: wp.vec3, shape: wp.vec3i):
    # Grid::isInBounds(Vec3) = isInBounds(toVec3iFloor(p))
    return (wp.floor(p[0]) >= 0.0 and wp.floor(p[1]) >= 0.0 and wp.floor(p[2]) >= 0.0
            and int(wp.floor(p[0])) < shape[0] and int(wp.floor(p[1])) < shape[1]
            and int(wp.floor(p[2])) < shape[2])


@wp.kernel
def _update_kernel(pos: wp.array(dtype=wp.vec3), pvel: wp.array(dtype=wp.vec3),
                   life: wp.array(dtype=wp.float32), flag: wp.array(dtype=wp.int32),
                   vel: wp.array4d(dtype=wp.float32), ratio: wp.array3d(dtype=wp.float32),
                   flags: wp.array3d(dtype=wp.int32), g: wp.vec3, k_b: float, k_d: float,
                   c_s: float, c_b: float, dt: float, at_mode: int, at_steps: int):
    # knFlipUpdateSecondaryParticlesLinear (f_sec = 0). at_mode 0 = off (Blender), 2 = Mantaflow
    # antitunneling with at_steps sub-steps, 1 = per particle steps of at most half a cell.
    # The sub-step points ct/N * dt*v, ct = 1..N-1, are tested like Mantaflow does; the particle
    # is killed when one is out of bounds or in an obstacle.
    t = wp.tid()
    f = flag[t]
    if (f & (PDELETE | PTRACER)) != 0:
        return
    p = pos[t]
    if not _in_bounds(p, wp.vec3i(ratio.shape[0], ratio.shape[1], ratio.shape[2])):
        flag[t] = f | PDELETE
        return
    nr = ratio[int(p[0]), int(p[1]), int(p[2])]
    v = pvel[t]
    dv = wp.vec3(0.0, 0.0, 0.0)  # displacement per dt, for the antitunneling test
    if nr < c_s:
        f = (f | PSPRAY) & ~(PBUBBLE | PFOAM)
        v = v + dt * g
        dv = dt * v
    elif nr > c_b:
        f = (f | PBUBBLE) & ~(PSPRAY | PFOAM)
        vg = _mac_interp(vel, p) - v
        vj = wp.vec3(vg[0] / dt, vg[1] / dt, vg[2] / dt)
        v = v + dt * (k_b * -g + k_d * vj)
        dv = dt * v
    else:
        f = (f | PFOAM) & ~(PBUBBLE | PSPRAY)
        dv = dt * _mac_interp(vel, p)
    if at_mode != 0:
        n = at_steps
        if at_mode == 1:
            n = wp.min(int(wp.ceil(wp.length(dv) / 0.5)) + 1, 16)
        sh = wp.vec3i(flags.shape[0], flags.shape[1], flags.shape[2])
        for ct in range(1, n):
            q = p + (float(ct) / float(n)) * dv
            if not _in_bounds(q, sh):
                f = f | PDELETE
            elif (flags[int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2]))] & SKIP_CELL) != 0:
                f = f | PDELETE
    p = p + dv
    l = life[t] - dt
    if l <= 0.0:
        f = f | PDELETE
    pos[t] = p
    pvel[t] = v
    life[t] = l
    flag[t] = f


@wp.kernel
def _delete_in_obstacle_kernel(pos: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32),
                               flags: wp.array3d(dtype=wp.int32)):
    # knFlipDeleteParticlesInObstacle
    t = wp.tid()
    f = flag[t]
    if (f & PDELETE) != 0:
        return
    p = pos[t]
    if not _in_bounds(p, wp.vec3i(flags.shape[0], flags.shape[1], flags.shape[2])):
        flag[t] = f | PDELETE
        return
    if (flags[int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2]))] & SKIP_CELL) != 0:
        flag[t] = f | PDELETE


class SecondaryMixin:
    """Backend methods for secondary particles; WarpBackend inherits them."""

    def secondary_potentials(self, flags, vel, phi, pot_ta, pot_wc, pot_ke, neighbor_ratio, radius,
                             tau_ta, tau_wc, tau_ke, scale_from_manta):
        radius = int(radius)
        if radius < 1:
            raise ValueError("secondary_potentials: radius must be at least 1")
        dev = self.device
        shape = tuple(phi.shape)
        normal = self._scratch_vec(shape, "snd_normal")
        wp.launch(_gradient_kernel, dim=shape, inputs=[phi, normal], device=dev)
        taus = [float(np.float32(t)) for t in (*tau_ta, *tau_wc, *tau_ke)]
        wp.launch(_potentials_kernel, dim=shape,
                  inputs=[flags, _f4(vel), normal, radius, wp.float64(1.732 * radius), *taus,
                          float(np.float32(scale_from_manta)), pot_ta, pot_wc, pot_ke,
                          neighbor_ratio], device=dev)

    def sample_secondary_particles(self, parts, flags, vel, pot_ta, pot_wc, pot_ke, neighbor_ratio,
                                   k_ta, k_wc, life_min, life_max, dt, c_s=0.4, c_b=0.77,
                                   sampling="SINGLE"):
        dev = self.device
        shape = tuple(flags.shape)
        f32 = lambda x: float(np.float32(x))  # noqa: E731
        mode = {"SINGLE": 0, "MULTIPLE": 1}[sampling]
        count = self._scratch_int(shape[::-1], "snd_count")
        draws = self._scratch_int(shape[::-1], "snd_draws")
        slot = self._scratch_int(shape[::-1], "snd_slot")
        rstart = self._scratch_int(shape[::-1], "snd_rstart")
        wp.launch(_sample_count_kernel, dim=shape,
                  inputs=[flags, pot_ta, pot_wc, pot_ke, f32(k_ta), f32(k_wc), f32(dt), mode, count, draws],
                  device=dev)
        wp.utils.array_scan(count.reshape(-1), slot.reshape(-1), inclusive=False)
        wp.utils.array_scan(draws.reshape(-1), rstart.reshape(-1), inclusive=False)
        total = int(slot.reshape(-1)[-1:].numpy()[0] + count.reshape(-1)[-1:].numpy()[0])
        if total == 0:
            return
        n_rnd = int(rstart.reshape(-1)[-1:].numpy()[0] + draws.reshape(-1)[-1:].numpy()[0])
        self._grow(parts, parts.n + total)
        wp.launch(_sample_emit_kernel, dim=shape,
                  inputs=[flags, _f4(vel), pot_ta, pot_wc, pot_ke, neighbor_ratio, count, slot, rstart,
                          self._random_stream(n_rnd), parts.n, f32(dt), f32(life_min), f32(life_max),
                          f32(c_s), f32(c_b), f32(k_ta), f32(k_wc), mode, parts.pos, parts.vel, parts.life,
                          parts.flag], device=dev)
        parts.n += total

    def update_secondary_particles(self, parts, flags, vel, neighbor_ratio, gravity, k_b, k_d, dt,
                                   c_s=0.4, c_b=0.77, antitunneling="OFF", antitunnel_steps=4):
        if parts.n:
            f32 = lambda x: float(np.float32(x))  # noqa: E731
            wp.launch(_update_kernel, dim=parts.n,
                      inputs=[parts.pos, parts.vel, parts.life, parts.flag, _f4(vel), neighbor_ratio,
                              flags, wp.vec3(*gravity), f32(k_b), f32(k_d), f32(c_s), f32(c_b), f32(dt),
                              {"OFF": 0, "AUTO": 1, "FIXED": 2}[antitunneling],
                              max(2, int(antitunnel_steps))],
                      device=self.device)
        self._compact(parts)

    def delete_particles_in_obstacle(self, parts, flags):
        if parts.n:
            wp.launch(_delete_in_obstacle_kernel, dim=parts.n, inputs=[parts.pos, parts.flag, flags],
                      device=self.device)
        self._compact(parts)
