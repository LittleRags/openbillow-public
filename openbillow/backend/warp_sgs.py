# SPDX-License-Identifier: GPL-3.0-or-later
"""Subgrid-scale turbulence for the Warp backend: Deardorff k_sgs and eddy viscosity
(CODE-GAS-035), for the mixing time of the mixture-fraction fire model.

Model from the FDS Technical Reference Guide 6.11.1 (McGrattan et al., NIST), sec. 4.2:
k_sgs = |u - u_test|^2 / 2 with u_test a test filter of the cell-centre velocity, and
nu_t = C_v dx sqrt(k_sgs), C_v = 0.1 (eq. 4.17). The test filter is the [1 2 1] / 4 filter
in each direction (27 cells, weights / 64). No FDS source is copied. FDS is software of the
US National Institute of Standards and Technology (public domain in the US); credit to NIST
for the model. Ported unchanged from the prototype research/mixfrac/mixfrac.py
(`_ksgs_kernel`, CODE-GAS-034), where the eddy-dissipation burn uses k_sgs in
tau_u = 0.4 dx / sqrt(2/3 k_sgs).

Cell-centre velocities are the averages of a cell's two MAC faces; indices past the grid
are clamped to the edge (so a uniform or linear field gives k_sgs = 0 away from the last two
cells of each axis). Obstacle cells are not treated specially (as in the prototype).
CODE-GAS-034 chose variant (a): nu_t feeds only the burn's mixing time, not the momentum
equation, so nothing here touches the viscosity solves.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})  # the solver never differentiates (CODE-GAS-006)

C_V = 0.1  # Deardorff constant, FDS guide eq. 4.17


@wp.func
def _uc(vel: wp.array3d(dtype=wp.vec3), i: int, j: int, k: int):
    """Cell-centre velocity from the MAC faces, edge-clamped."""
    nx, ny, nz = vel.shape[0], vel.shape[1], vel.shape[2]
    i = wp.clamp(i, 0, nx - 1)
    j = wp.clamp(j, 0, ny - 1)
    k = wp.clamp(k, 0, nz - 1)
    v = vel[i, j, k]
    return wp.vec3(0.5 * (v[0] + vel[wp.min(i + 1, nx - 1), j, k][0]),
                   0.5 * (v[1] + vel[i, wp.min(j + 1, ny - 1), k][1]),
                   0.5 * (v[2] + vel[i, j, wp.min(k + 1, nz - 1)][2]))


@wp.kernel
def _ksgs_kernel(vel: wp.array3d(dtype=wp.vec3), ks: wp.array3d(dtype=wp.float32), inv_vs: float):
    """k_sgs = |u - u_test|^2 / 2 of u = vel * inv_vs, u_test the [1 2 1]^3 / 64 filter of u."""
    i, j, k = wp.tid()
    ut = wp.vec3(0.0, 0.0, 0.0)
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                w = float((2 - wp.abs(a)) * (2 - wp.abs(b)) * (2 - wp.abs(c))) / 64.0
                ut = ut + w * _uc(vel, i + a, j + b, k + c)
    du = (_uc(vel, i, j, k) - ut) * inv_vs
    ks[i, j, k] = 0.5 * wp.dot(du, du)


@wp.kernel
def _nut_kernel(ks: wp.array3d(dtype=wp.float32), nt: wp.array3d(dtype=wp.float32), cvdx: float):
    i, j, k = wp.tid()
    nt[i, j, k] = cvdx * wp.sqrt(ks[i, j, k])


def k_sgs(vel, out, vel_scale: float = 1.0) -> None:
    """out (scalar grid) = k_sgs of the MAC velocity `vel`, in (vel / vel_scale)^2: m^2/s^2
    when vel / vel_scale is in m/s (the solver's `vel_scale`)."""
    wp.launch(_ksgs_kernel, dim=out.shape, inputs=[vel, out, 1.0 / vel_scale], device=out.device)


def eddy_viscosity(ks, out, dx: float, c_v: float = C_V) -> None:
    """out = nu_t = c_v dx sqrt(k_sgs), kinematic (m^2/s with dx in m and k_sgs in m^2/s^2);
    the dynamic mu_t is rho nu_t."""
    wp.launch(_nut_kernel, dim=out.shape, inputs=[ks, out, float(c_v) * float(dx)], device=out.device)
