# SPDX-License-Identifier: GPL-3.0-or-later
"""Flame-front fire model kernels (A6 variant C, CODE-GAS-020): a level-set fuel core that
burns at speed S (Nguyen, Fedkiw and Jensen 2002), the heat, smoke, fuel and expansion it
makes, the ideal-gas density lift, and the fuel mass source at the emitters.

Ported from the research prototype (research/flame2d/flame_gpu.py, numpy reference in
flame2d.py `FlameSolver._front_step` / `_add_forces`). Every tuning value is a parameter;
the prototype's numbers are the defaults of the solver settings (CODE-GAS-021). 3D only:
the solver refuses the model in 2D. The variable-density pressure solve is
`density_face_weights` + `solve_pressure(face_weights=)` (CODE-GAS-019).

Known shortcuts kept from the prototype (see the finding): the front moves with the mixed
velocity, not the fuel velocity plus S; no expansion within `source_gap` cells of an
emitter (a fix for the source acting as a fuel pump, not physics).
"""
from __future__ import annotations

import math

import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

from .base import Flags

OBSTACLE = wp.constant(int(Flags.OBSTACLE))
EPS_BAND = wp.constant(1.5)  # half-width (cells) of the smoothed delta that carries the expansion


@wp.func
def _at(a: wp.array3d(dtype=wp.float32), i: int, j: int, k: int):
    """Edge-clamped read (numpy edge padding)."""
    return a[wp.clamp(i, 0, a.shape[0] - 1), wp.clamp(j, 0, a.shape[1] - 1), wp.clamp(k, 0, a.shape[2] - 1)]


@wp.func
def _g1(c: float, m: float, p: float, minus: int):
    dm = c - m
    dp = p - c
    if minus != 0:
        return wp.min(dm, 0.0) * wp.min(dm, 0.0) + wp.max(dp, 0.0) * wp.max(dp, 0.0)
    return wp.max(dm, 0.0) * wp.max(dm, 0.0) + wp.min(dp, 0.0) * wp.min(dp, 0.0)


@wp.func
def _god(a: wp.array3d(dtype=wp.float32), i: int, j: int, k: int, minus: int):
    """Godunov |grad a|: minus = 1 for a front moving into a < 0, 0 for the other side."""
    c = a[i, j, k]
    s = _g1(c, _at(a, i - 1, j, k), _at(a, i + 1, j, k), minus)
    s += _g1(c, _at(a, i, j - 1, k), _at(a, i, j + 1, k), minus)
    s += _g1(c, _at(a, i, j, k - 1), _at(a, i, j, k + 1), minus)
    return wp.sqrt(s)


@wp.func
def _edge(i: int, j: int, k: int, nx: int, ny: int, nz: int):
    return i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1


@wp.kernel
def _src_kernel(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32),
                src: wp.array3d(dtype=wp.int32)):
    """Emitter cells: inside a flow's emission shell and not obstacle."""
    i, j, k = wp.tid()
    s = 0
    if phi_in[i, j, k] < 0.0 and (flags[i, j, k] & OBSTACLE) == 0:
        s = 1
    src[i, j, k] = s


@wp.kernel
def _pin_kernel(phi: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.int32),
                out: wp.array3d(dtype=wp.float32), frac0: wp.array3d(dtype=wp.float32), band: float):
    """Border outside the core (no fuel enters through it), emitters inside it; fuel fraction."""
    i, j, k = wp.tid()
    v = phi[i, j, k]
    if _edge(i, j, k, phi.shape[0], phi.shape[1], phi.shape[2]):
        v = band
    if src[i, j, k] != 0:
        v = wp.min(v, -1.0)
    out[i, j, k] = v
    frac0[i, j, k] = wp.clamp(0.5 - v, 0.0, 1.0)


@wp.kernel
def _burn_kernel(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32), c: float):
    """phi_t = S |grad phi|: the front moves into the core by c cells."""
    i, j, k = wp.tid()
    b[i, j, k] = a[i, j, k] + c * _god(a, i, j, k, 1)


@wp.kernel
def _repin_kernel(phi: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.int32),
                  s0: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    v = phi[i, j, k]
    if src[i, j, k] != 0:
        v = wp.min(v, -1.0)
        phi[i, j, k] = v
    s0[i, j, k] = v / wp.sqrt(v * v + 1.0)


@wp.kernel
def _reinit_kernel(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32),
                   s0: wp.array3d(dtype=wp.float32)):
    """One pseudo-time step (dtau 0.5) of phi_tau = s0 (1 - |grad phi|)."""
    i, j, k = wp.tid()
    s = s0[i, j, k]
    m = 1
    if s > 0.0:
        m = 0
    b[i, j, k] = a[i, j, k] - 0.5 * s * (_god(a, i, j, k, m) - 1.0)


@wp.kernel
def _fields_kernel(phi: wp.array3d(dtype=wp.float32), src: wp.array3d(dtype=wp.int32),
                   frac0: wp.array3d(dtype=wp.float32), heat: wp.array3d(dtype=wp.float32),
                   dens: wp.array3d(dtype=wp.float32), fuel: wp.array3d(dtype=wp.float32),
                   fuel_in: wp.array3d(dtype=wp.float32), div: wp.array3d(dtype=wp.float32),
                   phi_out: wp.array3d(dtype=wp.float32),
                   fmt: float, cool: float, crad: float, dt_s: float, burn: float, heat_fuel: float,
                   glow: float, smoke_yield: float, amount: float, expand: float, gap: int, band: float):
    """Heat, smoke, fuel and expansion from the burned front, per cell."""
    i, j, k = wp.tid()
    nx, ny, nz = phi.shape[0], phi.shape[1], phi.shape[2]
    p = phi[i, j, k]
    f0 = frac0[i, j, k]
    frac = wp.clamp(0.5 - p, 0.0, 1.0)
    burned = wp.max(f0 - frac, 0.0)
    is_src = src[i, j, k] != 0
    # Cooling of the old heat (as `combust`).
    h = heat[i, j, k]
    x = wp.abs(h) / fmt
    h = h / (1.0 + cool * dt_s + crad * dt_s * x * x * x)
    core = p < 0.0
    near = core
    if i > 0 and phi[i - 1, j, k] < 0.0:
        near = True
    if i < nx - 1 and phi[i + 1, j, k] < 0.0:
        near = True
    if j > 0 and phi[i, j - 1, k] < 0.0:
        near = True
    if j < ny - 1 and phi[i, j + 1, k] < 0.0:
        near = True
    if k > 0 and phi[i, j, k - 1] < 0.0:
        near = True
    if k < nz - 1 and phi[i, j, k + 1] < 0.0:
        near = True
    # Product side of the front at flame_max_temp, weighted smoothly (a binary mask speckles).
    w_burn = wp.clamp(burned / wp.max(burn, 1e-6), 0.0, 1.0)
    w_band = float(0.0)
    if near and not core:
        w_band = wp.clamp(1.0 - p, 0.0, 1.0)
    h = wp.max(h, fmt * wp.max(w_burn, w_band))
    # Diffusion-flame glow: fuel-side soot heats up near the front (glow cells deep);
    # the deeper core and the emitters stay at the fuel gas temperature.
    if core:
        if is_src:
            h = heat_fuel
        else:
            h = heat_fuel + (fmt - heat_fuel) * wp.clamp(1.0 + p / glow, 0.0, 1.0)
    heat[i, j, k] = h
    # Fuel = amount A * fraction. amount > 0: one A everywhere (the prototype). Else A per
    # cell: the emitter's fuel_in, elsewhere the carried fuel over its fraction before the
    # step, and away from the emitters fuel never grows (reinit may enlarge frac).
    fo = fuel[i, j, k]
    A = amount
    if amount <= 0.0:
        if is_src:
            A = fuel_in[i, j, k]
        elif f0 > 0.0:
            A = fo / f0
        else:
            A = fo
        A = wp.max(A, 0.0)
    fn = A * frac
    if amount <= 0.0 and not is_src:
        fn = wp.min(fn, wp.max(fo, 0.0))
    dens[i, j, k] = dens[i, j, k] + smoke_yield * A * burned
    fuel[i, j, k] = fn
    # Expansion (rho_f / rho_h - 1) S on the product side only (0 < phi < 2 eps): a source
    # inside the core would inflate it.
    d = float(0.0)
    q = p - EPS_BAND
    if not _edge(i, j, k, nx, ny, nz) and wp.abs(q) < EPS_BAND:
        gx = 0.5 * (phi[i + 1, j, k] - phi[i - 1, j, k])
        gy = 0.5 * (phi[i, j + 1, k] - phi[i, j - 1, k])
        gz = 0.5 * (phi[i, j, k + 1] - phi[i, j, k - 1])
        g = wp.sqrt(gx * gx + gy * gy + gz * gz)
        # The band sits outside the front, so where the front curves its level sets are larger
        # than the front. Scale by the area ratio front / own level set, (1 - p kappa / 2)^2 with
        # the local mean curvature kappa = div(grad phi / |grad phi|): exact on a sphere (CODE-GAS-024).
        xx = phi[i + 1, j, k] - 2.0 * p + phi[i - 1, j, k]
        yy = phi[i, j + 1, k] - 2.0 * p + phi[i, j - 1, k]
        zz = phi[i, j, k + 1] - 2.0 * p + phi[i, j, k - 1]
        xy = 0.25 * (phi[i + 1, j + 1, k] - phi[i + 1, j - 1, k] - phi[i - 1, j + 1, k] + phi[i - 1, j - 1, k])
        xz = 0.25 * (phi[i + 1, j, k + 1] - phi[i + 1, j, k - 1] - phi[i - 1, j, k + 1] + phi[i - 1, j, k - 1])
        yz = 0.25 * (phi[i, j + 1, k + 1] - phi[i, j + 1, k - 1] - phi[i, j - 1, k + 1] + phi[i, j - 1, k - 1])
        gs = wp.max(g, 1e-6)
        kap = ((xx + yy + zz) * gs * gs - gx * gx * xx - gy * gy * yy - gz * gz * zz
               - 2.0 * (gx * gy * xy + gx * gz * xz + gy * gz * yz)) / (gs * gs * gs)
        sc = wp.clamp(1.0 - 0.5 * p * kap, 0.25, 1.75)
        d = expand * (1.0 + wp.cos(wp.pi * q / EPS_BAND)) / (2.0 * EPS_BAND) * g * sc * sc
        # No expansion within gap cells (box) of an emitter: the emitters are pinned to fuel,
        # so expansion there blows fuel off them and the source becomes an unlimited pump.
        for a in range(wp.max(i - gap, 0), wp.min(i + gap, nx - 1) + 1):
            for b in range(wp.max(j - gap, 0), wp.min(j + gap, ny - 1) + 1):
                for c in range(wp.max(k - gap, 0), wp.min(k + gap, nz - 1) + 1):
                    if src[a, b, c] != 0:
                        d = 0.0
    div[i, j, k] = d
    phi_out[i, j, k] = wp.clamp(p, -band, band)


@wp.kernel
def _lift_kernel(heat: wp.array3d(dtype=wp.float32), lift: wp.array3d(dtype=wp.float32), c: float):
    """lift = rho_amb / rho - 1 = T / T_amb - 1 = c max(heat, 0)."""
    i, j, k = wp.tid()
    lift[i, j, k] = c * wp.max(heat[i, j, k], 0.0)


@wp.kernel
def _perim_kernel(src: wp.array3d(dtype=wp.int32), cnt: wp.array(dtype=wp.int32)):
    """cnt[0] += emitter cells, cnt[1] += faces between emitter and non-emitter cells."""
    i, j, k = wp.tid()
    s = src[i, j, k]
    if s != 0:
        wp.atomic_add(cnt, 0, 1)
    n = 0
    if i + 1 < src.shape[0] and src[i + 1, j, k] != s:
        n += 1
    if j + 1 < src.shape[1] and src[i, j + 1, k] != s:
        n += 1
    if k + 1 < src.shape[2] and src[i, j, k + 1] != s:
        n += 1
    if n > 0:
        wp.atomic_add(cnt, 1, n)


@wp.kernel
def _inject_kernel(src: wp.array3d(dtype=wp.int32), cnt: wp.array(dtype=wp.int32),
                   div: wp.array3d(dtype=wp.float32), inj: float):
    i, j, k = wp.tid()
    if src[i, j, k] != 0 and cnt[0] > 0:
        div[i, j, k] = div[i, j, k] + inj * float(cnt[1]) / float(cnt[0])


class FlameMixin:
    """Backend methods of the flame-front fire model; WarpBackend inherits them."""

    def _flame_scratch(self, shape):
        cache = getattr(self, "_flame_tmp", None)
        if cache is None or cache[0] != tuple(shape):
            cache = self._flame_tmp = (tuple(shape), self._new(shape, wp.float32), self._new(shape, wp.float32))
        return cache[1], cache[2]

    def _flame_src(self, flags, phi_in):
        src = self._scratch(flags)[0]
        wp.launch(_src_kernel, dim=flags.shape, inputs=[phi_in, flags, src], device=self.device)
        return src

    def flame_front_step(self, flags, phi_in, phi, heat, density, fuel, fuel_in, expansion_div, *,
                         burn_cells, expansion_rate, fuel_heat, flame_max_temp, cooling,
                         cooling_radiative, dt_seconds, smoke_yield, fuel_amount=0.0,
                         glow_cells=6.0, source_gap=3, band=6.0, reinit_iterations=2):
        sh = flags.shape
        dev = self.device
        src = self._flame_src(flags, phi_in)
        a, b = self._scratch(heat)  # free between calls (advection work grids)
        frac0, s0 = self._flame_scratch(sh)
        band = float(band)
        wp.launch(_pin_kernel, dim=sh, inputs=[phi, src, a, frac0, band], device=dev)
        burn = float(burn_cells)
        n = max(1, int(math.ceil(burn / 0.5)))
        for _ in range(n):
            wp.launch(_burn_kernel, dim=sh, inputs=[a, b, burn / n], device=dev)
            a, b = b, a
        wp.launch(_repin_kernel, dim=sh, inputs=[a, src, s0], device=dev)
        for _ in range(int(reinit_iterations)):
            wp.launch(_reinit_kernel, dim=sh, inputs=[a, b, s0], device=dev)
            a, b = b, a
        wp.launch(_fields_kernel, dim=sh, inputs=[
            a, src, frac0, heat, density, fuel, fuel_in, expansion_div, phi,
            float(flame_max_temp), float(cooling), float(cooling_radiative), float(dt_seconds), burn,
            float(fuel_heat), float(glow_cells), float(smoke_yield), float(fuel_amount),
            float(expansion_rate), int(source_gap), band], device=dev)

    def density_lift(self, heat, lift, ambient_kelvin, flame_kelvin, flame_max_temp):
        c = (float(flame_kelvin) - float(ambient_kelvin)) / (float(ambient_kelvin) * float(flame_max_temp))
        wp.launch(_lift_kernel, dim=heat.shape, inputs=[heat, lift, c], device=self.device)

    def fuel_mass_source(self, flags, phi_in, expansion_div, inject_cells):
        src = self._flame_src(flags, phi_in)
        cnt = getattr(self, "_flame_cnt", None)
        if cnt is None:
            cnt = self._flame_cnt = wp.zeros(2, dtype=wp.int32, device=self.device)
        cnt.zero_()
        wp.launch(_perim_kernel, dim=flags.shape, inputs=[src, cnt], device=self.device)
        wp.launch(_inject_kernel, dim=flags.shape, inputs=[src, cnt, expansion_div, float(inject_cells)],
                  device=self.device)
