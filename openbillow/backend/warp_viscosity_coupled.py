# SPDX-License-Identifier: GPL-3.0-or-later
"""High Quality viscosity (CODE-LIQ-014): a solve that converges.

Classic stays in warp_viscosity.py, a Mantaflow-identical copy of three independent
CGs run in lockstep on one coupled system. That iteration does not converge on real
liquid states: the iteration cap decides the result.

This solves the same discrete system, with the same volumes, rhs and matrix apply
as warp_viscosity.py, but with one CG over all three components (the operator is
symmetric and positive semi-definite) and a Jacobi preconditioner, because the
diagonal runs from about 1 in air-side faces to 2 * dt * visc * maxres^2 inside.

Stops when sqrt(r.z / r0.z0) < COUPLED_TOL, or at COUPLED_MAX_ITER as a safety
limit. Neither is a user setting. The default for quality HIGH; Classic never uses it.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"fuse_fp": False, "enable_backward": False})

from .base import PressureStats
from .warp_common import BLOCK, unravel
from .warp_viscosity import (CHECK_EVERY, ViscositySolver, _apply_kernel, _apply_velocities_kernel,
                             _fminus, _fplus, _row, _zero_kernel)

COUPLED_TOL = 1e-4        # on sqrt(r.z / r0.z0)
COUPLED_MAX_ITER = 4000   # safety only; a healthy solve stops on the tolerance


@wp.func
def _diag_row(vol: wp.array4d(dtype=wp.float32), a: int, p: wp.vec3i, c2: float, ce: float):
    # the a0 of warp_viscosity._apply_row: face volume plus the six stress weights
    return (vol[a + 1, p[0], p[1], p[2]]
            + _fplus(vol, a, 0, p, c2, ce) + _fminus(vol, a, 0, p, c2, ce)
            + _fplus(vol, a, 1, p, c2, ce) + _fminus(vol, a, 1, p, c2, ce)
            + _fplus(vol, a, 2, p, c2, ce) + _fminus(vol, a, 2, p, c2, ce))


@wp.kernel
def _inv_diag_kernel(flags: wp.array3d(dtype=wp.int32), vol: wp.array4d(dtype=wp.float32),
                     inv: wp.array3d(dtype=wp.vec3), c2: float, ce: float):
    """inv = 1 / diagonal on rows, 0 elsewhere (a zero row stays at x = 0)."""
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    v = wp.vec3(0.0)
    for a in range(3):
        if _row(flags, a, p):
            d = _diag_row(vol, a, p, c2, ce)
            if d > 0.0:
                v[a] = 1.0 / d
    inv[i, j, k] = v


@wp.kernel
def _precond_kernel(r: wp.array3d(dtype=wp.vec3), inv: wp.array3d(dtype=wp.vec3),
                    z: wp.array3d(dtype=wp.vec3), sc: wp.array(dtype=wp.float64)):
    """z = inv * r, sc[0] = r.z (float64 sum)."""
    t = wp.tid()
    nx, ny, nz = r.shape[0], r.shape[1], r.shape[2]
    rz = wp.float64(0.0)
    if t < nx * ny * nz:
        p = unravel(t, ny, nz)
        rv = r[p[0], p[1], p[2]]
        iv = inv[p[0], p[1], p[2]]
        zv = wp.cw_mul(rv, iv)
        z[p[0], p[1], p[2]] = zv
        rz = wp.float64(rv[0]) * wp.float64(zv[0]) + wp.float64(rv[1]) * wp.float64(zv[1]) + wp.float64(rv[2]) * wp.float64(zv[2])
    wp.tile_atomic_add(sc, wp.tile_sum(wp.tile(rz)), offset=0)


@wp.kernel
def _copy_kernel(a: wp.array3d(dtype=wp.vec3), b: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    b[i, j, k] = a[i, j, k]


@wp.kernel
def _init_kernel(sc: wp.array(dtype=wp.float64), rz: wp.array(dtype=wp.float64),
                 st: wp.array(dtype=wp.int32)):
    """rz[0] = r.z, rz[1] = r0.z0; st[0] = iterating, st[1] = iterations, st[2] = broke down."""
    rz[0] = sc[0]
    rz[1] = sc[0]
    st[0] = wp.where(sc[0] > wp.float64(0.0), 1, 0)
    st[1] = 0
    st[2] = 0


@wp.kernel
def _alpha_kernel(sc: wp.array(dtype=wp.float64), rz: wp.array(dtype=wp.float64),
                  alpha: wp.array(dtype=wp.float32), st: wp.array(dtype=wp.int32)):
    """alpha = r.z / s.As, the whole vector; s.As <= 0 ends the solve."""
    pap = sc[0] + sc[1] + sc[2]
    alpha[0] = 0.0
    if st[0] != 0:
        if pap > wp.float64(0.0):
            alpha[0] = wp.float32(rz[0] / pap)
        else:
            st[0] = 0
            st[2] = 1


@wp.kernel
def _advance_kernel(x: wp.array3d(dtype=wp.vec3), r: wp.array3d(dtype=wp.vec3),
                    s: wp.array3d(dtype=wp.vec3), tmp: wp.array3d(dtype=wp.vec3),
                    alpha: wp.array(dtype=wp.float32)):
    i, j, k = wp.tid()
    al = alpha[0]
    x[i, j, k] = x[i, j, k] + s[i, j, k] * al
    r[i, j, k] = r[i, j, k] - tmp[i, j, k] * al


@wp.kernel
def _beta_kernel(sc: wp.array(dtype=wp.float64), rz: wp.array(dtype=wp.float64),
                 beta: wp.array(dtype=wp.float32), st: wp.array(dtype=wp.int32),
                 tol2: wp.float64, max_iter: int):
    """beta = new r.z / old r.z; stop on the tolerance or the safety limit."""
    beta[0] = 0.0
    if st[0] != 0:
        st[1] = st[1] + 1
        new = sc[0]
        beta[0] = wp.float32(new / rz[0])
        rz[0] = new
        if new <= tol2 * rz[1] or st[1] >= max_iter:
            st[0] = 0


@wp.kernel
def _direction_kernel(z: wp.array3d(dtype=wp.vec3), s: wp.array3d(dtype=wp.vec3),
                      beta: wp.array(dtype=wp.float32), st: wp.array(dtype=wp.int32)):
    """s = z + beta s, unless the solve has just ended."""
    i, j, k = wp.tid()
    if st[0] != 0:
        s[i, j, k] = z[i, j, k] + s[i, j, k] * beta[0]


class CoupledViscositySolver:
    """One Jacobi-preconditioned CG over (u, v, w). Reuses a ViscositySolver for the
    volumes, the rhs and the matrix apply; adds 2 vec3 grids (24 B/cell)."""

    def __init__(self, base: ViscositySolver):
        self.base = base
        d, sh = base.device, base.shape
        self.device, self.shape = d, sh
        self.inv, self.z = (wp.zeros(sh, dtype=wp.vec3, device=d) for _ in range(2))
        self.rz = wp.zeros(2, dtype=wp.float64, device=d)
        self.alpha, self.beta = (wp.zeros(1, dtype=wp.float32, device=d) for _ in range(2))
        self.st = wp.zeros(3, dtype=wp.int32, device=d)
        self.st_host = wp.zeros(3, dtype=wp.int32, device="cpu", pinned=d.is_cuda)
        self._graph = None  # (key, CUDA graph of CHECK_EVERY iterations)

    def nbytes(self) -> int:
        return sum(a.capacity for a in (self.inv, self.z))

    def _precond(self):
        b, d = self.base, self.device
        wp.launch(_zero_kernel, dim=1, inputs=[b.sc], device=d)
        wp.launch(_precond_kernel, dim=b.padded, block_dim=BLOCK, inputs=[b.r, self.inv, self.z, b.sc], device=d)

    def _one(self, flags, tol2, max_iter):
        b, d = self.base, self.device
        wp.launch(_zero_kernel, dim=1, inputs=[b.sc], device=d)
        wp.launch(_apply_kernel, dim=b.padded, block_dim=BLOCK,
                  inputs=[flags, b.vol, b.s, b.tmp, b.cst, b.sc], device=d)
        wp.launch(_alpha_kernel, dim=1, inputs=[b.sc, self.rz, self.alpha, self.st], device=d)
        wp.launch(_advance_kernel, dim=self.shape, inputs=[b.x, b.r, b.s, b.tmp, self.alpha], device=d)
        self._precond()
        wp.launch(_beta_kernel, dim=1, inputs=[b.sc, self.rz, self.beta, self.st, tol2, max_iter], device=d)
        wp.launch(_direction_kernel, dim=self.shape, inputs=[self.z, b.s, self.beta, self.st], device=d)

    def _batch(self, flags, tol2, max_iter):
        for _ in range(CHECK_EVERY):
            self._one(flags, tol2, max_iter)
        wp.copy(self.st_host, self.st)

    def solve(self, flags, phi, vel, viscosity: float, dt: float, dx: float,
              tol: float = COUPLED_TOL, max_iter: int = COUPLED_MAX_ITER,
              history: list | None = None) -> PressureStats:
        """Solve and write the velocities; `history` (measurement only) collects
        sqrt(r.z / r0.z0) per iteration and turns the CUDA graph off. Stats: iterations,
        final sqrt(r.z / r0.z0), and whether it stopped without reaching `tol`."""
        b, d = self.base, self.device
        b.prepare(flags, phi, vel, viscosity, dt, dx)
        c2, ce = b.constants(viscosity, dt, dx)
        wp.launch(_inv_diag_kernel, dim=self.shape, inputs=[flags, b.vol, self.inv, c2, ce], device=d)
        self._precond()
        wp.launch(_init_kernel, dim=1, inputs=[b.sc, self.rz, self.st], device=d)
        wp.launch(_copy_kernel, dim=self.shape, inputs=[self.z, b.s], device=d)
        tol2 = wp.float64(tol * tol)
        key = (flags.ptr, tol, max_iter)
        done = 0
        while done < max_iter:
            if history is not None:
                self._one(flags, tol2, max_iter)
                wp.copy(self.st_host, self.st)
                wp.synchronize_device(d)
                rz = self.rz.numpy()
                history.append(float(np.sqrt(max(rz[0], 0.0) / rz[1])) if rz[1] > 0 else 0.0)
                done += 1
            else:
                if not b.use_graph:
                    self._batch(flags, tol2, max_iter)
                else:
                    if self._graph is None or self._graph[0] != key:
                        with wp.ScopedCapture(device=d) as cap:
                            self._batch(flags, tol2, max_iter)
                        self._graph = (key, cap.graph)
                    wp.capture_launch(self._graph[1])
                done += CHECK_EVERY
                wp.synchronize_device(d)
            if not self.st_host.numpy()[0]:
                break
        wp.launch(_apply_velocities_kernel, dim=self.shape, inputs=[flags, b.x, vel], device=d)
        st, rz = self.st.numpy(), self.rz.numpy()
        rel = float(np.sqrt(max(rz[0], 0.0) / rz[1])) if rz[1] > 0 else 0.0
        return PressureStats(int(st[1]), rel, bool(rel > tol) or bool(st[2]))
