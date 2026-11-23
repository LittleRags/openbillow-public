# SPDX-License-Identifier: GPL-3.0-or-later
"""Mixture-fraction / eddy-dissipation combustion for the Warp backend (CODE-GAS-036): fuel
inflow, the burn, heat release with a local radiative loss, the expansion it drives, soot, and
conservative finite-volume transport of the species.

Model from the FDS Technical Reference Guide 6.11.1 (McGrattan et al., NIST), sec. 5.1-5.2.4
(fast chemistry, mixing time), as outlined in CODE-GAS-032. No FDS source is copied. FDS is
software of the US National Institute of Standards and Technology (public domain in the US);
credit and thanks to NIST for the model. Ported from the prototype research/mixfrac/mixfrac.py
(`--variant a --fv`, CODE-GAS-034 and CODE-GAS-040) with the same kernel boundaries; the only
change is that the fuel source is a per-cell rate grid instead of a mask and one rate.

State, per cell: Y_F (fuel), Y_P (products), air Y_A = 1 - Y_F - Y_P, and sig = rho / rho_amb,
the transported density, which persists between steps (`MixtureFraction.sig`). Fuel and air
enter at ambient temperature and all species share one cp, so the temperature is a function of
Y_P alone, T = T_amb + (1 - chi_r) dH_c / cp * Y_P / (1 + s); it is written to the solver's
heat grid as (T - T_amb) / kelvin_per_heat, the inverse of `gas_kelvin`, so `density_lift`,
output and the renderer read it unchanged.

One step, in the order the solver calls it:
1. `burn`: k_sgs (warp_sgs) for the mixing time, then per cell: add the source fuel mass
   mdot dt; burn dY_F = min(Y_F, Y_A / s) (1 - exp(-dt / tau_mix)), tau_mix = max(dx / 100 m/s,
   min(dx^2 / nu(T), 0.4 dx / sqrt(2/3 k_sgs), sqrt(2 dx / g))), and dY_F = 0 in a cell whose
   centre is within the quenching distance `Fuel.quench` of an obstacle (CODE-GAS-045: heat loss
   to a cold solid quenches the flame; this model is adiabatic, T = f(Y_P), so FDS's critical
   flame temperature test would never fire; off when quench < dx / 2, e.g. at pool scale);
   dT = (1 - chi_r) dH_c dY_F / cp;
   smoke += soot yield * dY_F; target divergence = [ln(T1 / T0) + ln(sig1 / sig0) +
   rho_gain ln(sig0 / rho(T0))] / dt (expansion, the source's volume, and a drift correction of
   sig towards the ideal-gas density).
2. pressure projection with that divergence (the solver's).
3. `transport`: rho, rho Y_F, rho Y_P as a conservative finite-volume update, species flux =
   mass flux x limited face Y (CHARM capped at min(2, 2 r)) plus Fick diffusion
   -rho D grad Y with rho D = mu(T) / Sc (Sutherland, Sc = 0.7 = Pr of air, so Le = 1 and the
   heat, a function of Y_P, diffuses with it; CODE-GAS-044: sets the length of small laminar
   flames, negligible at pool scale), SSP-RK2 sub-cycled at CFL 0.4 and at the explicit
   diffusion limit, each stage with an outflow limiter on the donor cell (CODE-GAS-046: no cell
   loses more than it holds, so no negative mass for the unpack to clamp into new fuel). No
   diffusion through the domain's outer faces.
   Solids have zero flux; outside the domain is ambient air. Smoke is not moved here.

3D only: the constructor refuses dimensions != 3 (as the FLAME_FRONT model, CODE-GAS-020).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .base import Flags
from .warp_diffusion import _sutherland_mu
from .warp_sgs import k_sgs

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

FLUID = wp.constant(int(Flags.FLUID))
OBSTACLE = wp.constant(int(Flags.OBSTACLE))

RHO_AIR = 1.204  # kg/m^3 at 293 K and 1 atm


@dataclass(frozen=True)
class Fuel:
    """Single-step reaction constants. Default: propane as FDS 6.11.1 ran it in pool3d
    (pool3d.out, "Gas Phase Reaction Information"); soot yield 0.01 as the FDS input."""
    dhc: float = 46007.0e3   # J/kg heat of combustion
    s_air: float = 15.592    # kg air per kg fuel
    chi_r: float = 0.29      # radiative fraction, removed locally (no radiation transport)
    soot: float = 0.01       # kg soot per kg fuel burned
    cp: float = 1005.0       # J/(kg K), all species
    # m, no burning this close to a cold solid (CODE-GAS-045): half the parallel-plate quenching
    # distance of alkane-air flames (~2 mm), since every point of such a channel is that close to a wall
    quench: float = 1.0e-3

    @property
    def dt_per_y(self) -> float:
        """K per unit of fuel mass fraction burned: (1 - chi_r) dH_c / cp."""
        return (1.0 - self.chi_r) * self.dhc / self.cp


PROPANE = Fuel()


def source_rate(hrrpua_kw_m2: float, dx: float, fuel: Fuel = PROPANE) -> float:
    """Fuel mass source (kg / (m^3 s)) for a one-cell-thick source layer that releases
    `hrrpua_kw_m2` kW per m^2 of its area: HRRPUA / (dH_c dx)."""
    return float(hrrpua_kw_m2) * 1e3 / (fuel.dhc * float(dx))


@wp.func
def _near_cold_solid(flags: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, r: int, q2: float):
    """True if an obstacle cell within r cells has a point closer than sqrt(q2) cells to the
    centre of cell (i, j, k). Fuel emitters are fluid cells, so a burning wick is not cold."""
    for a in range(-r, r + 1):
        ia = i + a
        if ia < 0 or ia >= flags.shape[0]:
            continue
        da = wp.max(float(wp.abs(a)) - 0.5, 0.0)
        for b in range(-r, r + 1):
            jb = j + b
            if jb < 0 or jb >= flags.shape[1]:
                continue
            db = wp.max(float(wp.abs(b)) - 0.5, 0.0)
            for c in range(-r, r + 1):
                kc = k + c
                if kc < 0 or kc >= flags.shape[2]:
                    continue
                dc = wp.max(float(wp.abs(c)) - 0.5, 0.0)
                if da * da + db * db + dc * dc < q2 and (flags[ia, jb, kc] & OBSTACLE) != 0:
                    return True
    return False


@wp.kernel
def _burn_kernel(flags: wp.array3d(dtype=wp.int32), mdot: wp.array3d(dtype=wp.float32),
                 sig: wp.array3d(dtype=wp.float32),
                 yf_g: wp.array3d(dtype=wp.float32), yp_g: wp.array3d(dtype=wp.float32),
                 heat: wp.array3d(dtype=wp.float32), smoke: wp.array3d(dtype=wp.float32),
                 divg: wp.array3d(dtype=wp.float32), ks: wp.array3d(dtype=wp.float32),
                 burned: wp.array(dtype=wp.float64),
                 amb: float, kph: float, dx: float, dt_s: float, rho_amb: float,
                 s_air: float, dt_per_y: float, soot: float, div_scale: float, gain: float,
                 qr: int, q2: float):
    """Source, burn, heat, smoke and target divergence of one cell (prototype `_chem_fv_kernel`).
    burned[1] += fuel mass burned in source cells (mdot > 0), burned[0] elsewhere (kg).
    No burning within the quenching distance (sqrt(q2) cells, search radius qr) of an obstacle."""
    i, j, k = wp.tid()
    f = flags[i, j, k]
    if (f & OBSTACLE) != 0:
        yf_g[i, j, k] = 0.0
        yp_g[i, j, k] = 0.0
        heat[i, j, k] = 0.0
        smoke[i, j, k] = 0.0
        divg[i, j, k] = 0.0
        sig[i, j, k] = 1.0
        return
    if (f & FLUID) == 0:
        divg[i, j, k] = 0.0
        return
    yf = wp.clamp(yf_g[i, j, k], 0.0, 1.0)
    yp = wp.clamp(yp_g[i, j, k], 0.0, 1.0 - yf)
    g = wp.max(sig[i, j, k], 1e-4)
    g0 = g
    m = mdot[i, j, k]
    if m > 0.0:
        ma = m * dt_s / rho_amb
        gn = g + ma
        yf = (g * yf + ma) / gn
        yp = g * yp / gn
        g = gn
    t0 = amb + dt_per_y * yp / (1.0 + s_air)
    ya = wp.max(1.0 - yf - yp, 0.0)
    nu = _sutherland_mu(t0) * 287.05 * t0 / 101325.0
    tau = wp.min(dx * dx / nu, wp.sqrt(2.0 * dx / 9.81))
    tau = wp.min(tau, 0.4 * dx / wp.sqrt(2.0 / 3.0 * ks[i, j, k] + 1e-12))
    tau = wp.max(tau, dx / 100.0)
    dyf = wp.min(yf, ya / s_air) * (1.0 - wp.exp(-dt_s / tau))
    if dyf > 0.0 and qr > 0:
        if _near_cold_solid(flags, i, j, k, qr, q2):
            dyf = 0.0
    yf -= dyf
    yp += (1.0 + s_air) * dyf
    t1 = t0 + dt_per_y * dyf
    sig[i, j, k] = g
    yf_g[i, j, k] = yf
    yp_g[i, j, k] = wp.min(yp, 1.0)
    heat[i, j, k] = (t1 - amb) / kph
    smoke[i, j, k] = smoke[i, j, k] + soot * dyf
    divg[i, j, k] = (wp.log(t1 / t0) + wp.log(g / g0) + gain * wp.log(g0 * t0 / amb)) / dt_s * div_scale
    if dyf > 0.0:
        slot = int(0)
        if m > 0.0:
            slot = 1
        wp.atomic_add(burned, slot, wp.float64(rho_amb * g * dyf * dx * dx * dx))


@wp.kernel
def _vmax_kernel(vel: wp.array3d(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i, j, k = wp.tid()
    v = vel[i, j, k]
    wp.atomic_max(out, 0, wp.max(wp.abs(v[0]), wp.max(wp.abs(v[1]), wp.abs(v[2]))))


SCHMIDT = 0.7  # species (and, Le = 1, heat) diffusion: rho D = mu / Sc, Sc = Pr of air


@wp.func
def _charm(r: float):
    """CHARM flux limiter, capped at min(2, 2 r) so the face value stays between the upwind
    and downwind cells and the update is TVD (CHARM itself tends to 3)."""
    if r <= 0.0:
        return 0.0
    return wp.min(r * (3.0 * r + 1.0) / ((r + 1.0) * (r + 1.0)), wp.min(2.0, 2.0 * r))


@wp.func
def _fv_val(flags: wp.array3d(dtype=wp.int32), q: wp.array3d(dtype=wp.vec3), p: wp.vec3i, a: int, c: int,
            amb_q: wp.vec3):
    """(rho, Y_F, Y_P) of cell p shifted by c along a; ambient outside the domain."""
    n = flags.shape[a]
    pp = wp.vec3i(p[0], p[1], p[2])
    pp[a] = pp[a] + c
    if pp[a] < 0 or pp[a] >= n:
        return amb_q
    v = q[pp[0], pp[1], pp[2]]
    r = wp.max(v[0], 1e-6)
    yf = wp.clamp(v[1] / r, 0.0, 1.0)
    return wp.vec3(wp.max(v[0], 0.0), yf, wp.clamp(v[2] / r, 0.0, 1.0 - yf))


@wp.func
def _fv_flux(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3), q: wp.array3d(dtype=wp.vec3),
             lo: wp.vec3i, a: int, inv_vs: float, dif: wp.vec3):
    """Flux of (rho, rho Y_F, rho Y_P) / rho_amb in m/s through the face between lo and lo + e_a,
    + along a: mass flux from the limited rho, species from the limited Y times the mass flux,
    plus species diffusion mu(T_face) / (Sc rho_amb dx) (Y_lo - Y_hi) between two domain cells.
    dif = (1 / (Sc rho_amb dx), T_amb, dT per Y_P). Zero at a solid; past the last face the
    velocity is extrapolated (zero gradient)."""
    n = flags.shape[a]
    hi = wp.vec3i(lo[0], lo[1], lo[2])
    hi[a] = hi[a] + 1
    amb_q = wp.vec3(1.0, 0.0, 0.0)
    if lo[a] >= 0 and (flags[lo[0], lo[1], lo[2]] & OBSTACLE) != 0:
        return wp.vec3(0.0)
    if hi[a] < n and (flags[hi[0], hi[1], hi[2]] & OBSTACLE) != 0:
        return wp.vec3(0.0)
    fi = wp.vec3i(hi[0], hi[1], hi[2])
    if fi[a] >= n:
        fi[a] = n - 1
    if fi[a] < 0:
        fi[a] = 0
    u = vel[fi[0], fi[1], fi[2]][a] * inv_vs
    # upwind U, far upwind UU, downwind D, as offsets from lo
    cu = 0
    cuu = -1
    cd = 1
    if u < 0.0:
        cu = 1
        cuu = 2
        cd = 0
    qu = _fv_val(flags, q, lo, a, cu, amb_q)
    quu = _fv_val(flags, q, lo, a, cuu, amb_q)
    qd = _fv_val(flags, q, lo, a, cd, amb_q)
    face = wp.vec3(qu[0], qu[1], qu[2])
    for m in range(3):
        dd = qd[m] - qu[m]
        if wp.abs(dd) > 1e-12:
            face[m] = qu[m] + 0.5 * _charm((qu[m] - quu[m]) / dd) * dd
    mf = u * face[0]
    out = wp.vec3(mf, mf * face[1], mf * face[2])
    if dif[0] > 0.0 and lo[a] >= 0 and hi[a] < n:
        ql = qu
        qh = qd
        if u < 0.0:
            ql = qd
            qh = qu
        g = _sutherland_mu(dif[1] + dif[2] * 0.5 * (ql[2] + qh[2])) * dif[0]
        out = out + wp.vec3(0.0, g * (ql[1] - qh[1]), g * (ql[2] - qh[2]))
    return out


@wp.kernel
def _fv_limit_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                     q: wp.array3d(dtype=wp.vec3), lam: wp.array3d(dtype=wp.vec3), inv_vs: float,
                     dtdx: float, dif: wp.vec3):
    """Outflow limiter of one cell, per component of q (CODE-GAS-046): lam = min(1, q / (dt/dx x
    the sum of the fluxes leaving through its six faces)). The limited face values can each be
    up to twice the cell's, and a cell can drain through every face, so without it the stage
    leaves rho Y_F < 0 at the flame edge, which the unpack clamps to 0: fuel made from nothing."""
    i, j, k = wp.tid()
    l = wp.vec3(1.0)
    if (flags[i, j, k] & OBSTACLE) == 0:
        p = wp.vec3i(i, j, k)
        o = wp.vec3(0.0)
        for a in range(3):
            lo = wp.vec3i(i, j, k)
            lo[a] = lo[a] - 1
            fp = _fv_flux(flags, vel, q, p, a, inv_vs, dif)   # leaves this cell where > 0
            fm = _fv_flux(flags, vel, q, lo, a, inv_vs, dif)  # leaves this cell where < 0
            for m in range(3):
                o[m] = o[m] + wp.max(fp[m], 0.0) - wp.min(fm[m], 0.0)
        v = q[i, j, k]
        for m in range(3):
            d = dtdx * o[m]
            if d > v[m]:
                l[m] = wp.max(v[m], 0.0) / d
    lam[i, j, k] = l


@wp.func
def _donor_lim(flux: wp.vec3, lam: wp.array3d(dtype=wp.vec3), lo: wp.vec3i, a: int):
    """The flux through the face between lo and lo + e_a, each component scaled by the outflow
    limiter of the cell it leaves (none from outside the domain), so both sides see the same flux."""
    hi = wp.vec3i(lo[0], lo[1], lo[2])
    hi[a] = hi[a] + 1
    out = wp.vec3(flux[0], flux[1], flux[2])
    for m in range(3):
        if flux[m] > 0.0 and lo[a] >= 0:
            out[m] = flux[m] * lam[lo[0], lo[1], lo[2]][m]
        if flux[m] < 0.0 and hi[a] < lam.shape[a]:
            out[m] = flux[m] * lam[hi[0], hi[1], hi[2]][m]
    return out


@wp.kernel
def _fv_stage_kernel(flags: wp.array3d(dtype=wp.int32), vel: wp.array3d(dtype=wp.vec3),
                     q: wp.array3d(dtype=wp.vec3), q0: wp.array3d(dtype=wp.vec3),
                     out: wp.array3d(dtype=wp.vec3), lam: wp.array3d(dtype=wp.vec3), inv_vs: float,
                     dtdx: float, a0: float, dif: wp.vec3):
    """SSP-RK2 stage: out = a0 q0 + (1 - a0) (q - dt/dx sum of the donor-limited face fluxes),
    lam from `_fv_limit_kernel` on q; each stage keeps q >= 0, so their convex sum does too."""
    i, j, k = wp.tid()
    p = wp.vec3i(i, j, k)
    if (flags[i, j, k] & OBSTACLE) != 0:
        out[i, j, k] = q[i, j, k]
        return
    div = wp.vec3(0.0)
    for a in range(3):
        lo = wp.vec3i(i, j, k)
        lo[a] = lo[a] - 1
        div = (div + _donor_lim(_fv_flux(flags, vel, q, p, a, inv_vs, dif), lam, p, a)
               - _donor_lim(_fv_flux(flags, vel, q, lo, a, inv_vs, dif), lam, lo, a))
    out[i, j, k] = a0 * q0[i, j, k] + (1.0 - a0) * (q[i, j, k] - dtdx * div)


@wp.kernel
def _fv_pack_kernel(sig: wp.array3d(dtype=wp.float32), yf: wp.array3d(dtype=wp.float32),
                    yp: wp.array3d(dtype=wp.float32), q: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    g = sig[i, j, k]
    q[i, j, k] = wp.vec3(g, g * yf[i, j, k], g * yp[i, j, k])


@wp.kernel
def _fv_unpack_kernel(flags: wp.array3d(dtype=wp.int32), q: wp.array3d(dtype=wp.vec3),
                      sig: wp.array3d(dtype=wp.float32), yf: wp.array3d(dtype=wp.float32),
                      yp: wp.array3d(dtype=wp.float32), heat: wp.array3d(dtype=wp.float32),
                      dt_per_y: float, s_air: float, kph: float):
    i, j, k = wp.tid()
    if (flags[i, j, k] & OBSTACLE) != 0:
        return
    v = q[i, j, k]
    g = wp.max(v[0], 1e-4)
    a = wp.clamp(v[1] / g, 0.0, 1.0)
    b = wp.clamp(v[2] / g, 0.0, 1.0 - a)
    sig[i, j, k] = g
    yf[i, j, k] = a
    yp[i, j, k] = b
    heat[i, j, k] = dt_per_y * b / (1.0 + s_air) / kph


@wp.kernel
def _drift_kernel(flags: wp.array3d(dtype=wp.int32), sig: wp.array3d(dtype=wp.float32),
                  yp: wp.array3d(dtype=wp.float32), out: wp.array(dtype=wp.float64),
                  amb: float, dpy: float, s_air: float):
    """out[0] += |sig - rho(T) / rho_amb|, out[1] += sig, over non-solid cells."""
    i, j, k = wp.tid()
    if (flags[i, j, k] & OBSTACLE) != 0:
        return
    g = sig[i, j, k]
    r = amb / (amb + dpy * wp.clamp(yp[i, j, k], 0.0, 1.0) / (1.0 + s_air))
    wp.atomic_add(out, 0, wp.float64(wp.abs(g - r)))
    wp.atomic_add(out, 1, wp.float64(g))


class MixtureFraction:
    """Buffers and launches of the mixture-fraction model on one grid. The caller owns the
    solver grids (flags, MAC vel, Y_F, Y_P, heat, smoke, target divergence, source rate) and
    passes them in; this object owns sig (persistent state: checkpoint it), k_sgs, the
    transport buffers and the burned-fuel counter.

    dx in m; vel_scale converts the solver's velocity to m/s (u = vel / vel_scale) and its
    divergence per cell is d (1/s) * dx * vel_scale; kelvin_per_heat is gas_kelvin's
    (flame_kelvin - ambient_kelvin) / flame_max_temp."""

    def __init__(self, shape, device, dx: float, vel_scale: float, ambient_kelvin: float,
                 kelvin_per_heat: float, fuel: Fuel = PROPANE, rho_amb: float = RHO_AIR,
                 rho_gain: float = 0.3, cfl: float = 0.4, dimensions: int = 3):
        if dimensions != 3:
            raise ValueError("the mixture-fraction fire model needs a 3D domain")
        self.shape, self.device = tuple(shape), device
        self.dx, self.vel_scale = float(dx), float(vel_scale)
        self.amb, self.kph, self.fuel = float(ambient_kelvin), float(kelvin_per_heat), fuel
        self.rho_amb, self.rho_gain, self.cfl = float(rho_amb), float(rho_gain), float(cfl)
        self.sig = wp.full(self.shape, 1.0, dtype=wp.float32, device=device)
        self.ks = wp.zeros(self.shape, dtype=wp.float32, device=device)
        self.q = [wp.zeros(self.shape, dtype=wp.vec3, device=device) for _ in range(3)]
        self.lam = wp.zeros(self.shape, dtype=wp.vec3, device=device)  # outflow limiter (CODE-GAS-046)
        self.vmax = wp.zeros(1, dtype=wp.float32, device=device)
        self.burned = wp.zeros(2, dtype=wp.float64, device=device)
        self.substeps = 0

    def burn(self, flags, vel, mdot, yf, yp, heat, smoke, divg, dt_s: float) -> None:
        """Step 1 (see the module docstring); dt_s in seconds. Writes yf, yp, heat, smoke, divg
        and sig in place."""
        f = self.fuel
        k_sgs(vel, self.ks, self.vel_scale)
        q = f.quench / self.dx  # cells; an obstacle face at n - 0.5 cells is within reach if n < q + 0.5
        qr = max(math.ceil(q + 0.5) - 1, 0)
        wp.launch(_burn_kernel, dim=self.shape,
                  inputs=[flags, mdot, self.sig, yf, yp, heat, smoke, divg, self.ks, self.burned,
                          self.amb, self.kph, self.dx, float(dt_s), self.rho_amb, f.s_air, f.dt_per_y,
                          f.soot, self.dx * self.vel_scale, self.rho_gain, qr, q * q], device=self.device)

    def transport(self, flags, vel, yf, yp, heat, dt_s: float) -> int:
        """Step 3: move sig, Y_F, Y_P with the (projected) velocity over dt_s seconds and rewrite
        heat from Y_P. Returns the number of RK2 substeps (one host sync for the CFL)."""
        dev, f = self.device, self.fuel
        q0, q1, q2 = self.q
        self.vmax.zero_()
        wp.launch(_vmax_kernel, dim=vel.shape, inputs=[vel, self.vmax], device=dev)
        c = float(self.vmax.numpy()[0]) / self.vel_scale * float(dt_s) / self.dx
        # advection and explicit diffusion together: c / cfl + 6 dt D_max / (0.5 dx^2) <= 1 per
        # substep (stability; positivity is the outflow limiter's job); D_max at the adiabatic T
        t_max = self.amb + f.dt_per_y / (1.0 + f.s_air)
        mu = 1.716e-5 * (t_max / 273.15) ** 1.5 * (273.15 + 110.4) / (t_max + 110.4)
        d_max = mu / SCHMIDT / (self.rho_amb * self.amb / t_max)
        n = max(1, math.ceil(c / self.cfl + float(dt_s) * d_max * 6.0 / (0.5 * self.dx * self.dx)))
        dtdx = float(dt_s) / n / self.dx
        ivs = 1.0 / self.vel_scale
        dif = wp.vec3(1.0 / (SCHMIDT * self.rho_amb * self.dx), self.amb, f.dt_per_y / (1.0 + f.s_air))
        wp.launch(_fv_pack_kernel, dim=self.shape, inputs=[self.sig, yf, yp, q0], device=dev)
        lam = self.lam
        for _ in range(n):
            wp.launch(_fv_limit_kernel, dim=self.shape, inputs=[flags, vel, q0, lam, ivs, dtdx, dif], device=dev)
            wp.launch(_fv_stage_kernel, dim=self.shape, inputs=[flags, vel, q0, q0, q1, lam, ivs, dtdx, 0.0, dif],
                      device=dev)
            wp.launch(_fv_limit_kernel, dim=self.shape, inputs=[flags, vel, q1, lam, ivs, dtdx, dif], device=dev)
            wp.launch(_fv_stage_kernel, dim=self.shape, inputs=[flags, vel, q1, q0, q2, lam, ivs, dtdx, 0.5, dif],
                      device=dev)
            q0, q2 = q2, q0
        self.q = [q0, q1, q2]
        wp.launch(_fv_unpack_kernel, dim=self.shape,
                  inputs=[flags, q0, self.sig, yf, yp, heat, f.dt_per_y, f.s_air, self.kph], device=dev)
        self.substeps += n
        return n

    def burned_mass(self) -> tuple[float, float]:
        """Fuel burned so far (kg): (outside source cells, in source cells)."""
        b = self.burned.numpy()
        return float(b[0]), float(b[1])

    def density_drift(self, flags, yp) -> float:
        """Mean |sig - rho(T) / rho_amb| / sig over non-solid cells (a health check of the
        drift correction)."""
        out = wp.zeros(2, dtype=wp.float64, device=self.device)
        wp.launch(_drift_kernel, dim=self.shape,
                  inputs=[flags, self.sig, yp, out, self.amb, self.fuel.dt_per_y, self.fuel.s_air],
                  device=self.device)
        e = out.numpy()
        return float(e[0] / max(e[1], 1e-30))

    @staticmethod
    def rate_grid(mask: np.ndarray, rate: float, device):
        """Source rate grid (kg / (m^3 s)) of `rate` on the nonzero cells of `mask`."""
        return wp.array((np.asarray(mask) != 0).astype(np.float32) * np.float32(rate), dtype=wp.float32,
                        device=device)


@wp.func
def _is_emitter(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    return phi_in[i, j, k] < 0.0 and (flags[i, j, k] & OBSTACLE) == 0


@wp.func
def _gas_face(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    """1 if the neighbour (i, j, k) is inside the grid, not an emitter and not an obstacle."""
    if i < 0 or j < 0 or k < 0 or i >= flags.shape[0] or j >= flags.shape[1] or k >= flags.shape[2]:
        return 0
    if (flags[i, j, k] & OBSTACLE) != 0 or phi_in[i, j, k] < 0.0:
        return 0
    return 1


@wp.func
def _exposed_faces(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32), i: int, j: int, k: int):
    """Faces of emitter cell (i, j, k) that border gas (0 for other cells)."""
    if not _is_emitter(phi_in, flags, i, j, k):
        return 0
    return (_gas_face(phi_in, flags, i - 1, j, k) + _gas_face(phi_in, flags, i + 1, j, k)
            + _gas_face(phi_in, flags, i, j - 1, k) + _gas_face(phi_in, flags, i, j + 1, k)
            + _gas_face(phi_in, flags, i, j, k - 1) + _gas_face(phi_in, flags, i, j, k + 1))


@wp.kernel
def _count_faces_kernel(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32),
                        count: wp.array(dtype=wp.int32)):
    i, j, k = wp.tid()
    f = _exposed_faces(phi_in, flags, i, j, k)
    if f > 0:
        wp.atomic_add(count, 0, f)


@wp.kernel
def _surface_source_kernel(phi_in: wp.array3d(dtype=wp.float32), flags: wp.array3d(dtype=wp.int32),
                           count: wp.array(dtype=wp.int32), mdot: wp.array3d(dtype=wp.float32),
                           rate_per_face: float):
    """mdot = rate_per_face x (exposed faces of the cell) / (all exposed faces); rate_per_face is
    the whole source in kg/(m^3 s) of one cell, so the cells together release it (CODE-GAS-042)."""
    i, j, k = wp.tid()
    f = _exposed_faces(phi_in, flags, i, j, k)
    v = float(0.0)
    if f > 0:
        v = rate_per_face * float(f) / float(count[0])
    mdot[i, j, k] = v


class MixFracMixin:
    """Backend methods of the DIFFUSION fire model (CODE-GAS-037); WarpBackend inherits them.
    The state handle is a MixtureFraction with its source rate grid in `.mdot`."""

    def mixfrac_init(self, shape, *, dx, vel_scale, ambient_kelvin, kelvin_per_heat, heat_of_combustion,
                     air_fuel_ratio, radiative_fraction, soot_yield, rho_gain=0.3):
        fuel = Fuel(dhc=float(heat_of_combustion), s_air=float(air_fuel_ratio),
                    chi_r=float(radiative_fraction), soot=float(soot_yield))
        mf = MixtureFraction(shape, self.device, dx, vel_scale, ambient_kelvin, kelvin_per_heat, fuel=fuel,
                             rho_gain=rho_gain)
        mf.mdot = wp.zeros(mf.shape, dtype=wp.float32, device=self.device)
        return mf

    def mixfrac_source(self, mf, flags, phi_in, hrrpua_kw_m2, area_m2):
        if getattr(mf, "face_count", None) is None:
            mf.face_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        mf.face_count.zero_()
        wp.launch(_count_faces_kernel, dim=mf.shape, inputs=[phi_in, flags, mf.face_count], device=self.device)
        # kg/s of fuel over one cell volume; no host sync: the kernel divides by the count on the device.
        rate = float(hrrpua_kw_m2) * 1e3 * float(area_m2) / (mf.fuel.dhc * float(mf.dx) ** 3)
        wp.launch(_surface_source_kernel, dim=mf.shape, inputs=[phi_in, flags, mf.face_count, mf.mdot, rate],
                  device=self.device)

    def mixfrac_burn(self, mf, flags, vel, fuel, products, heat, smoke, expansion_div, dt_seconds):
        mf.burn(flags, vel, mf.mdot, fuel, products, heat, smoke, expansion_div, dt_seconds)

    def mixfrac_transport(self, mf, flags, vel, fuel, products, heat, dt_seconds):
        return mf.transport(flags, vel, fuel, products, heat, dt_seconds)
