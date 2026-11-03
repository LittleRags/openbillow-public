# SPDX-License-Identifier: GPL-3.0-or-later
"""Fluid guiding (Blender "Guides"): Mantaflow `PD_fluid_guiding`, a primal-dual
loop that pulls the velocity toward a guide velocity (blurred) while keeping
it divergence-free, and its separable Gaussian blur.

Ported from reference/mantaflow/plugin/fluidguiding.cpp (MantaFlow,
Apache-2.0, Copyright 2016 Kiwon Um, Nils Thuerey). Mantaflow's chain of grid
operations is fused into a few kernels per iteration; the maths is the same:

    Q = 2 G G (velT - velC) - sigma velC,  invA = 1 / max(2 W^2 + sigma, 0.01)
    loop: x0 = x;  t = sigma (x / sigma + y) + Q
          t = t invA - 2 invA G G (t invA) + velC       (prox_f, approxInvM)
          x = -sigma t + sigma y + x0
          z0 = z;  z -= tau x;  solvePressure(z);  y = z + theta (z - z0)
          stop when iter > 0 and max|z - z0| < sqrt(d) epsAbs + epsRel max|z|
    vel = z

G blurs all three MAC components along x, y and z (2D: Mantaflow's x and y,
which are our x and z; the slab's y axis is skipped), zero-padded, then faces
touching an OBSTACLE cell get their pre-blur value back. Norms are max vector
lengths over raw face triples, as Mantaflow's Grid<Vec3>::getMaxAbs.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .base import Flags, GuidingStats

OBSTACLE = wp.constant(int(Flags.OBSTACLE))


def gaussian_kernel_1d(n: int) -> np.ndarray:
    """Mantaflow get1DGaussianBlurKernel(n, sigma=n), normalised to sum 1.
    Its constant x term cancels in the normalisation."""
    t = np.arange(n, dtype=np.float64) - (n - 1) * 0.5
    g = np.exp(-(t * t) / (2.0 * n * n))
    return (g / g.sum()).astype(np.float32)


@wp.kernel
def _blur_axis_kernel(src: wp.array3d(dtype=wp.vec3), dst: wp.array3d(dtype=wp.vec3),
                      w: wp.array(dtype=wp.float32), axis: int,
                      flags: wp.array3d(dtype=wp.int32), restore: int, two_d: int):
    i, j, k = wp.tid()
    if restore:
        # last pass writes into the grid itself: obstacle faces keep their value
        if flags[i, j, k] & OBSTACLE:
            return
        if i > 0 and (flags[i - 1, j, k] & OBSTACLE):
            return
        if two_d == 0 and j > 0 and (flags[i, j - 1, k] & OBSTACLE):
            return
        if k > 0 and (flags[i, j, k - 1] & OBSTACLE):
            return
    kn = w.shape[0]
    kc = kn // 2
    n = src.shape[axis]
    c = i
    if axis == 1:
        c = j
    elif axis == 2:
        c = k
    acc = wp.vec3(0.0)
    for m in range(kn):
        cc = c - kc + m
        if cc >= 0 and cc < n:
            v = wp.vec3(0.0)
            if axis == 0:
                v = src[cc, j, k]
            elif axis == 1:
                v = src[i, cc, k]
            else:
                v = src[i, j, cc]
            acc += v * w[kn - 1 - m]
    dst[i, j, k] = acc


@wp.kernel
def _inv_a_kernel(weight: wp.array3d(dtype=wp.float32), inv_a: wp.array3d(dtype=wp.float32),
                  sigma: float):
    i, j, k = wp.tid()
    w = weight[i, j, k]
    inv_a[i, j, k] = 1.0 / wp.max(2.0 * w * w + sigma, 0.01)


@wp.kernel
def _sub_kernel(a: wp.array3d(dtype=wp.vec3), b: wp.array3d(dtype=wp.vec3),
                out: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    out[i, j, k] = a[i, j, k] - b[i, j, k]


@wp.kernel
def _q_finish_kernel(q: wp.array3d(dtype=wp.vec3), vel_c: wp.array3d(dtype=wp.vec3), sigma: float):
    i, j, k = wp.tid()
    q[i, j, k] = 2.0 * q[i, j, k] - sigma * vel_c[i, j, k]


@wp.kernel
def _x_update_a_kernel(x: wp.array3d(dtype=wp.vec3), y: wp.array3d(dtype=wp.vec3),
                       q: wp.array3d(dtype=wp.vec3), inv_a: wp.array3d(dtype=wp.float32),
                       x0: wp.array3d(dtype=wp.vec3), w: wp.array3d(dtype=wp.vec3), sigma: float):
    # x0 = x; x = w = (sigma (x / sigma + y) + Q) invA  (w is then blurred twice)
    i, j, k = wp.tid()
    xv = x[i, j, k]
    x0[i, j, k] = xv
    t = (xv * (1.0 / sigma) + y[i, j, k]) * sigma + q[i, j, k]
    t = t * inv_a[i, j, k]
    x[i, j, k] = t
    w[i, j, k] = t


@wp.kernel
def _x_update_b_kernel(x: wp.array3d(dtype=wp.vec3), y: wp.array3d(dtype=wp.vec3),
                       w: wp.array3d(dtype=wp.vec3), inv_a: wp.array3d(dtype=wp.float32),
                       x0: wp.array3d(dtype=wp.vec3), vel_c: wp.array3d(dtype=wp.vec3),
                       sigma: float):
    # t = x - 2 invA G G x + velC;  x = -sigma t + sigma y + x0
    i, j, k = wp.tid()
    t = x[i, j, k] - 2.0 * inv_a[i, j, k] * w[i, j, k] + vel_c[i, j, k]
    x[i, j, k] = -sigma * t + sigma * y[i, j, k] + x0[i, j, k]


@wp.kernel
def _z_update_kernel(z: wp.array3d(dtype=wp.vec3), x: wp.array3d(dtype=wp.vec3),
                     z0: wp.array3d(dtype=wp.vec3), tau: float):
    i, j, k = wp.tid()
    zv = z[i, j, k]
    z0[i, j, k] = zv
    z[i, j, k] = zv - tau * x[i, j, k]


@wp.kernel
def _y_update_kernel(z: wp.array3d(dtype=wp.vec3), z0: wp.array3d(dtype=wp.vec3),
                     y: wp.array3d(dtype=wp.vec3), theta: float, out: wp.array(dtype=wp.float32)):
    # y = z + theta (z - z0); out = (max |z - z0|^2, max |z|^2) over this column
    i, j = wp.tid()
    md = float(0.0)
    mz = float(0.0)
    for k in range(z.shape[2]):
        zv = z[i, j, k]
        d = zv - z0[i, j, k]
        y[i, j, k] = theta * d + zv
        md = wp.max(md, wp.length_sq(d))
        mz = wp.max(mz, wp.length_sq(zv))
    wp.atomic_max(out, 0, md)
    wp.atomic_max(out, 1, mz)


class GuidingMixin:
    """Backend methods for guiding; WarpBackend inherits them."""

    def _guide_scratch(self, shape, names):
        cache = getattr(self, "_guide_tmp", None)
        if cache is None or cache["shape"] != tuple(shape):
            cache = self._guide_tmp = {"shape": tuple(shape)}
        out = []
        for name in names:
            if name not in cache:
                cache[name] = self._new(shape, wp.float32 if name == "inv_a" else wp.vec3)
            out.append(cache[name])
        return out

    def _blur_weights(self, radius):
        cached = getattr(self, "_guide_w", None)
        if cached is None or cached[0] != radius:
            w = wp.array(gaussian_kernel_1d(2 * radius + 1), dtype=wp.float32, device=self.device)
            cached = self._guide_w = (radius, w)
        return cached[1]

    def guide_blur(self, flags, vel, radius=5, two_d=False):
        a, b = self._guide_scratch(vel.shape, ("blur_a", "blur_b"))
        w = self._blur_weights(int(radius))
        d = self.device
        if two_d:
            passes = ((vel, a, 0), (a, vel, 2))
        else:
            passes = ((vel, a, 0), (a, b, 1), (b, vel, 2))
        for n, (src, dst, axis) in enumerate(passes):
            wp.launch(_blur_axis_kernel, dim=vel.shape,
                      inputs=[src, dst, w, axis, flags, int(n == len(passes) - 1), int(two_d)],
                      device=d)

    def guiding_solve(self, flags, vel, vel_target, weight, pressure, blur_radius=5,
                      tau=1.0, sigma=0.99, theta=1.0, eps_rel=1e-3, eps_abs=1e-3, max_iters=200,
                      two_d=False, accuracy=1e-3, zero_pressure_fixing=False, divergence=0.0,
                      fractions=None, phi=None) -> GuidingStats:
        sh = vel.shape
        d = self.device
        x, y, z, x0, z0, q, w, inv_a = self._guide_scratch(
            sh, ("x", "y", "z", "x0", "z0", "q", "w", "inv_a"))
        for g in (x, y, z):
            g.zero_()
        vel_c = vel  # vel is only written at the end, so it serves as velC

        # Q = 2 G G (velT - velC) - sigma velC;  invA from the weight
        wp.launch(_sub_kernel, dim=sh, inputs=[vel_target, vel_c, q], device=d)
        self.guide_blur(flags, q, blur_radius, two_d)
        self.guide_blur(flags, q, blur_radius, two_d)
        wp.launch(_q_finish_kernel, dim=sh, inputs=[q, vel_c, float(sigma)], device=d)
        wp.launch(_inv_a_kernel, dim=sh, inputs=[weight, inv_a, float(sigma)], device=d)

        red = getattr(self, "_guide_reduce", None)
        if red is None:
            red = self._guide_reduce = wp.zeros(2, dtype=wp.float32, device=d)
        eps_dim = np.sqrt(2.0 if two_d else 3.0) * eps_abs
        cg_total = 0
        phi_gen = None
        if phi is not None:  # phi is fixed during the loop: one matrix build
            self._phi_gen = phi_gen = getattr(self, "_phi_gen", 0) + 1
        stats = None
        it = 0
        for it in range(max_iters):
            wp.launch(_x_update_a_kernel, dim=sh, inputs=[x, y, q, inv_a, x0, w, float(sigma)], device=d)
            self.guide_blur(flags, w, blur_radius, two_d)
            self.guide_blur(flags, w, blur_radius, two_d)
            wp.launch(_x_update_b_kernel, dim=sh, inputs=[x, y, w, inv_a, x0, vel_c, float(sigma)],
                      device=d)

            wp.launch(_z_update_kernel, dim=sh, inputs=[z, x, z0, float(tau)], device=d)
            stats = self.solve_pressure(flags, z, pressure, accuracy,
                                        zero_pressure_fixing=zero_pressure_fixing,
                                        divergence=divergence, fractions=fractions,
                                        phi=phi, _phi_gen=phi_gen)
            cg_total += stats.iterations

            red.zero_()
            wp.launch(_y_update_kernel, dim=sh[:2], inputs=[z, z0, y, float(theta), red], device=d)
            md, mz = np.sqrt(red.numpy())
            if it > 0 and md < eps_dim + eps_rel * mz:
                break
        wp.copy(vel, z)
        return GuidingStats(it + 1, cg_total, stats.residual if stats else 0.0)
