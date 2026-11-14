# SPDX-License-Identifier: GPL-3.0-or-later
"""Liquid velocity diffusion for the Warp backend: Mantaflow `cgSolveDiffusion`.

Ported from reference/mantaflow/conjugategrad.cpp (cgSolveDiffusion) and the
ApplyMatrix kernel and GridCg in conjugategrad.h (MantaFlow, Apache-2.0,
Copyright 2020 Sebastian Barschkis, Nils Thuerey).

The matrix is (I - alpha * Laplace) built from an all-fluid dummy flag grid, so
the coefficients depend only on the position (interior cell: diagonal
6 alpha + 1, off-diagonal -alpha; border cell: diagonal 1, no off-diagonal)
and on the real OBSTACLE flags (identity row). It is applied only to FLUID
cells; every other cell is the identity. Each velocity component is an
un-preconditioned CG with x = 0 at the start and rhs = the component, L2
residual (sum of squares) < accuracy, at most int(factor * max(shape))
iterations. The three components run in lockstep with the viscosity solver's
scalar kernels; the matrix is recomputed on the fly instead of stored.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

from .base import Flags, PressureStats
from .warp_common import BLOCK, border, unravel
from .warp_viscosity import (CHECK_EVERY, _finish_kernel, _init_scalars_kernel, _search_kernel,
                             _update_kernel, _zero_kernel)

OBSTACLE = wp.constant(int(Flags.OBSTACLE))
FLUID = wp.constant(int(Flags.FLUID))


@wp.func
def _off(flags: wp.array3d(dtype=wp.int32), p: wp.vec3i, alpha: float):
    """Off-diagonal coefficient stored at cell p (towards +x, +y or +z)."""
    c = float(0.0)
    if (flags[p[0], p[1], p[2]] & OBSTACLE) == 0:
        if not border(p[0], p[1], p[2], flags.shape[0], flags.shape[1], flags.shape[2]):
            c = -1.0 * alpha
    return c


@wp.func
def _sutherland_mu(t: float):
    """Dynamic viscosity of air (Pa s) at t K: Sutherland's law mu = mu0 (T / T0)^1.5 (T0 + S) / (T + S),
    mu0 = 1.716e-5 Pa s, T0 = 273.15 K, S = 110.4 K (F. M. White, Viscous Fluid Flow, 3rd ed., Table 1-2)."""
    return 1.716e-5 * wp.pow(t / 273.15, 1.5) * (273.15 + 110.4) / (t + 110.4)


@wp.func
def _face(kel: wp.array3d(dtype=wp.float32), p: wp.vec3i, q: wp.vec3i):
    """Face mu R / p_atm between cells p and q (m^2/s/K): mu at the mean of the two cell values.
    Times T it is the kinematic viscosity of air at 1 atm (ideal gas, R = 287.05 J/(kg K))."""
    mu = 0.5 * (_sutherland_mu(kel[p[0], p[1], p[2]]) + _sutherland_mu(kel[q[0], q[1], q[2]]))
    return mu * (287.05 / 101325.0)


@wp.func
def _row(flags: wp.array3d(dtype=wp.int32), s: wp.array3d(dtype=wp.vec3), a: int,
         p: wp.vec3i, alpha: float, kel: wp.array3d(dtype=wp.float32), use_kel: int):
    """One row of the matrix applied to component a of s, in Mantaflow's order.
    use_kel (gas viscosity, CODE-GAS-029): the variable-density viscous step
    rho u' - alpha div(mu grad u') = rho u with rho = p_atm / (R T), made symmetric by
    w = sqrt(rho) u: off-diagonal -alpha m_f sqrt(T_p T_q), diagonal 1 + alpha T_p sum m_f,
    m_f = mu_f R / p_atm (`_face`). Constant T gives 6 alpha nu + 1 and -alpha nu."""
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    sv = s[p[0], p[1], p[2]][a]
    if (flags[p[0], p[1], p[2]] & FLUID) == 0:
        return sv
    tp = float(1.0)
    if use_kel != 0:
        tp = kel[p[0], p[1], p[2]]
    a0 = float(1.0)
    if (flags[p[0], p[1], p[2]] & OBSTACLE) == 0 and not border(p[0], p[1], p[2], nx, ny, nz):
        if use_kel == 0:
            a0 = 6.0 * alpha + 1.0
        else:
            m = float(0.0)
            for ax in range(3):
                lo = p
                hi = p
                lo[ax] = lo[ax] - 1
                hi[ax] = hi[ax] + 1
                m = m + _face(kel, lo, p) + _face(kel, p, hi)
            a0 = 1.0 + alpha * tp * m
    r = sv * a0
    for ax in range(3):
        lo = p
        hi = p
        lo[ax] = lo[ax] - 1
        hi[ax] = hi[ax] + 1
        if lo[ax] >= 0:
            fl = alpha
            if use_kel != 0:
                fl = alpha * _face(kel, lo, p) * wp.sqrt(tp * kel[lo[0], lo[1], lo[2]])
            r = r + s[lo[0], lo[1], lo[2]][a] * _off(flags, lo, fl)
        if hi[ax] < flags.shape[ax]:
            fh = alpha
            if use_kel != 0:
                fh = alpha * _face(kel, p, hi) * wp.sqrt(tp * kel[hi[0], hi[1], hi[2]])
            r = r + s[hi[0], hi[1], hi[2]][a] * _off(flags, p, fh)
    return r


@wp.kernel
def gas_kelvin_kernel(heat: wp.array3d(dtype=wp.float32), kel: wp.array3d(dtype=wp.float32),
                      ambient: float, k_per_heat: float):
    """FLAME_FRONT temperature T = ambient + k_per_heat max(heat, 0) in K."""
    i, j, k = wp.tid()
    kel[i, j, k] = ambient + k_per_heat * wp.max(heat[i, j, k], 0.0)


@wp.kernel
def _scale_kernel(vel: wp.array3d(dtype=wp.vec3), kel: wp.array3d(dtype=wp.float32), power: float):
    """vel *= T^power: u -> w = sqrt(rho) u (power -0.5, rho ~ 1 / T) and back (+0.5)."""
    i, j, k = wp.tid()
    vel[i, j, k] = vel[i, j, k] * wp.pow(kel[i, j, k], power)


@wp.kernel
def _apply_kernel(flags: wp.array3d(dtype=wp.int32), s: wp.array3d(dtype=wp.vec3),
                  tmp: wp.array3d(dtype=wp.vec3), cst: wp.array(dtype=wp.float32),
                  sc: wp.array(dtype=wp.float64), kel: wp.array3d(dtype=wp.float32), use_kel: int):
    """tmp = A search for the three components, sc[0:3] = dot(tmp, search)."""
    t = wp.tid()
    alpha = cst[0]
    nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
    d = wp.vec3(0.0)
    sv = wp.vec3(0.0)
    if t < nx * ny * nz:
        p = unravel(t, ny, nz)
        for a in range(3):
            d[a] = _row(flags, s, a, p, alpha, kel, use_kel)
        tmp[p[0], p[1], p[2]] = d
        sv = s[p[0], p[1], p[2]]
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[0] * sv[0]))), offset=0)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[1] * sv[1]))), offset=1)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(d[2] * sv[2]))), offset=2)


@wp.kernel
def _rhs_kernel(vel: wp.array3d(dtype=wp.vec3), x: wp.array3d(dtype=wp.vec3),
                r: wp.array3d(dtype=wp.vec3), s: wp.array3d(dtype=wp.vec3),
                sc: wp.array(dtype=wp.float64)):
    """GridCg::doInit: dst = 0, residual = rhs = vel, search = residual, sc[6:9] = dot(r, r)."""
    t = wp.tid()
    nx, ny, nz = vel.shape[0], vel.shape[1], vel.shape[2]
    rv = wp.vec3(0.0)
    if t < nx * ny * nz:
        p = unravel(t, ny, nz)
        rv = vel[p[0], p[1], p[2]]
        x[p[0], p[1], p[2]] = wp.vec3(0.0)
        r[p[0], p[1], p[2]] = rv
        s[p[0], p[1], p[2]] = rv
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[0] * rv[0]))), offset=6)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[1] * rv[1]))), offset=7)
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(wp.float64(rv[2] * rv[2]))), offset=8)


class DiffusionSolver:
    """Work grids for one domain shape (4 vec3 per cell)."""

    def __init__(self, shape, device):
        self.shape = tuple(int(s) for s in shape)
        self.device = device
        d, sh = device, self.shape
        self.x, self.r, self.s, self.tmp = (wp.zeros(sh, dtype=wp.vec3, device=d) for _ in range(4))
        self.sc = wp.zeros(9, dtype=wp.float64, device=d)
        self.sig, self.beta, self.res = (wp.zeros(3, dtype=wp.float32, device=d) for _ in range(3))
        self.act, self.upd, self.iters = (wp.zeros(3, dtype=wp.int32, device=d) for _ in range(3))
        self.flag = wp.zeros(2, dtype=wp.int32, device=d)
        self.cst = wp.zeros(1, dtype=wp.float32, device=d)
        self._one = wp.zeros((1, 1, 1), dtype=wp.float32, device=d)  # unused kelvin grid when constant
        self.flag_host = wp.zeros(2, dtype=wp.int32, device="cpu", pinned=d.is_cuda)
        self.use_graph = d.is_cuda
        self._graph = None
        self.n = int(np.prod(sh))
        self.padded = (self.n + BLOCK - 1) // BLOCK * BLOCK

    def nbytes(self) -> int:
        return sum(a.capacity for a in (self.x, self.r, self.s, self.tmp))

    def _iterations(self, flags, accuracy, max_iter, kelvin):
        d = self.device
        cg, uc = (self._one, 0) if kelvin is None else (kelvin, 1)
        for _ in range(CHECK_EVERY):
            wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=d)
            wp.launch(_apply_kernel, dim=self.padded, block_dim=BLOCK,
                      inputs=[flags, self.s, self.tmp, self.cst, self.sc, cg, uc], device=d)
            wp.launch(_update_kernel, dim=self.padded, block_dim=BLOCK,
                      inputs=[self.x, self.r, self.s, self.tmp, self.sc, self.sig, self.act], device=d)
            wp.launch(_finish_kernel, dim=1,
                      inputs=[self.sc, self.sig, self.beta, self.act, self.upd, self.iters, self.res,
                              self.flag, accuracy, max_iter], device=d)
            wp.launch(_search_kernel, dim=self.shape, inputs=[self.r, self.s, self.beta, self.upd], device=d)
        wp.copy(self.flag_host, self.flag)

    def prepare(self, vel, alpha: float):
        d = self.device
        self.cst.assign(np.array([alpha], np.float32))
        wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=d)
        wp.launch(_rhs_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[vel, self.x, self.r, self.s, self.sc], device=d)
        wp.launch(_init_scalars_kernel, dim=1,
                  inputs=[self.sc, self.sig, self.act, self.iters, self.res, self.flag], device=d)

    def apply(self, flags, src, dst, alpha: float, kelvin=None):
        """dst = A src (tests)."""
        self.cst.assign(np.array([alpha], np.float32))
        wp.launch(_zero_kernel, dim=1, inputs=[self.sc], device=self.device)
        wp.launch(_apply_kernel, dim=self.padded, block_dim=BLOCK,
                  inputs=[flags, src, dst, self.cst, self.sc, self._one if kelvin is None else kelvin,
                          int(kelvin is not None)], device=self.device)

    def solve(self, flags, vel, alpha: float, accuracy: float = 1e-4,
              max_iter_factor: float = 1.0, kelvin=None) -> PressureStats:
        """Mantaflow cgSolveDiffusion on `vel`; alpha = coefficient * dt / dx^2. With `kelvin`
        (gas temperature grid, K) the variable-density viscous step of air instead, alpha = dt / dx^2.
        Stats: most iterations of the three CGs and the largest final residual."""
        d = self.device
        accuracy = float(np.float32(accuracy))
        max_iter = int(np.float32(max_iter_factor) * np.float32(max(self.shape)))
        if kelvin is not None:
            wp.launch(_scale_kernel, dim=self.shape, inputs=[vel, kelvin, -0.5], device=d)
        self.prepare(vel, alpha)
        key = (flags.ptr, accuracy, max_iter, None if kelvin is None else kelvin.ptr)
        done = 0
        while done < max_iter:
            if not self.use_graph:
                self._iterations(flags, accuracy, max_iter, kelvin)
            else:
                if self._graph is None or self._graph[0] != key:
                    with wp.ScopedCapture(device=d) as cap:
                        self._iterations(flags, accuracy, max_iter, kelvin)
                    self._graph = (key, cap.graph)
                wp.capture_launch(self._graph[1])
            done += CHECK_EVERY
            wp.synchronize_device(d)
            fl = self.flag_host.numpy()
            if fl[1]:
                raise RuntimeError("diffusion solve diverged (residual > 1e35)")
            if not fl[0]:
                break
        wp.copy(vel, self.x)
        if kelvin is not None:
            wp.launch(_scale_kernel, dim=self.shape, inputs=[vel, kelvin, 0.5], device=d)
        iters = self.iters.numpy()
        return PressureStats(int(iters.max()), float(self.res.numpy().max()))
