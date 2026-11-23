# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender's default primitive meshes as triangle lists (outward winding).

The cube and sphere are convex, so each triangle is oriented away from the origin. Blender
splits the UV sphere's (planar) quads along a different diagonal in places,
which covers the same surface.
"""
from __future__ import annotations

import numpy as np


def _fan(polys, v):
    tris = []
    for p in polys:
        for t in range(1, len(p) - 1):
            a, b, c = p[0], p[t], p[t + 1]
            n = np.cross(v[b] - v[a], v[c] - v[a])
            tris.append((a, b, c) if n @ (v[a] + v[b] + v[c]) > 0 else (a, c, b))
    return np.array(tris, np.int32)


def plane() -> tuple[np.ndarray, np.ndarray]:
    """Blender's default plane: corners at ±1 in z = 0, normal +z."""
    v = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float)
    return v, np.array([(0, 1, 2), (0, 2, 3)], np.int32)


def mesh_area(verts, tris, matrix) -> float:
    """Surface area in m^2 of the triangles after the (4, 4) matrix."""
    w = np.asarray(verts, float) @ np.asarray(matrix, float)[:3, :3].T
    t = np.asarray(tris)
    return 0.5 * float(np.linalg.norm(np.cross(w[t[:, 1]] - w[t[:, 0]], w[t[:, 2]] - w[t[:, 0]]), axis=1).sum())


def cube(open_top: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Blender's default cube: corners at ±1. open_top drops the +z face."""
    v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    if open_top:
        quads = quads[:-1]
    return v, _fan(quads, v)


def uv_sphere(segments: int = 32, rings: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Blender's default UV sphere, radius 1, poles on z."""
    v = [(0.0, 0.0, 1.0)]
    for r in range(1, rings):
        th = np.pi * r / rings
        for s in range(segments):
            ph = 2 * np.pi * s / segments
            v.append((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    v.append((0.0, 0.0, -1.0))
    v = np.array(v)
    ring = lambda r, s: 1 + (r - 1) * segments + s % segments
    polys = [(0, ring(1, s), ring(1, s + 1)) for s in range(segments)]
    polys += [(ring(r, s), ring(r + 1, s), ring(r + 1, s + 1), ring(r, s + 1))
              for r in range(1, rings - 1) for s in range(segments)]
    polys += [(ring(rings - 1, s), len(v) - 1, ring(rings - 1, s + 1)) for s in range(segments)]
    return v, _fan(polys, v)


def tube(points, r, sides=12, phase=0.0):
    """A closed tube (prism with end caps) of radius r round the polyline `points` (n, 3) in
    world metres -> (verts, tris) with outward winding. sides=4, phase=pi/4, r=a/sqrt(2): an a x a
    square section."""
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    v = []
    for pk, tk in zip(p, t):
        n = np.cross((0.0, 1.0, 0.0), tk)
        n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-9 else np.array((1.0, 0.0, 0.0))
        b2 = np.cross(tk, n)
        for j in range(sides):
            a = phase + 2.0 * np.pi * j / sides
            v.append(pk + r * (np.cos(a) * n + np.sin(a) * b2))
    nr = len(p)
    v += [p[0], p[-1]]
    c0, c1 = nr * sides, nr * sides + 1
    tris = []
    for k in range(nr - 1):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            c, d = (k + 1) * sides + (j + 1) % sides, (k + 1) * sides + j
            tris += [(a, b, c), (a, c, d)]
    for j in range(sides):
        tris += [(c0, (j + 1) % sides, j), (c1, (nr - 1) * sides + j, (nr - 1) * sides + (j + 1) % sides)]
    v, tris = np.array(v), np.array(tris, np.int32)
    w = v[tris] - p.mean(axis=0)
    if np.einsum("ij,ij->i", w[:, 0], np.cross(w[:, 1], w[:, 2])).sum() < 0:  # signed volume
        tris = tris[:, ::-1].copy()
    return v, tris


def wick_burning_length(d, length):
    """Sunderland et al. 2011 (Fig. 6): burning share of a straight wick, Lb/L = 0.8 exp(-0.6 D/L)."""
    return 0.8 * np.exp(-0.6 * d / length) * length


# Air at Sunderland's film temperature (1151 K): k W/(m K), cp J/(kg K), interpolated from the
# usual air tables (Incropera A.4, from memory, not checked); B and Ra as Sunderland 2011.
WICK_K_AIR, WICK_CP_AIR, WICK_B = 0.0758, 1167.0, 1.89


def wick_burning_rate(d, burning):
    """Sunderland et al. 2011, Eqs. 5-9 (stagnant-layer flame sheet, tetracosane B = 1.89; paraffin
    taken the same): mass burning flux of a straight wick of diameter d with `burning` m of its
    side alight, kg/(m^2 s) -> (side, top). The side's flux falls as burning grows, (d/Lb)^(1/4),
    so a long wick burns less per mm^2 (CODE-GAS-041). The paper measured no burning rates; for
    d 1.2 mm, Lb 9 mm this gives 0.11 g/min, Hamins 2005 measured 0.105 g/min."""
    ra4 = (1.46 * (d * 1e3) ** 3) ** 0.25
    g = WICK_K_AIR / (WICK_CP_AIR * d) * np.log1p(WICK_B)
    return g * (0.52 + 0.547 * ra4 * (d / burning) ** 0.25), g * (0.43 + 0.6 * ra4)


def wick_hrrpua(d, length, burning=None, square=False, heat_of_combustion=43.8):
    """HRR per unit burning surface (side + top) of a straight wick, kW/m^2, from
    wick_burning_rate; heat_of_combustion in kJ/g. -> (hrrpua, mass rate kg/s)."""
    lb = wick_burning_length(d, length) if burning is None else burning
    m_side, m_top = wick_burning_rate(d, lb)
    a_side, a_top = (4 * d * lb, d * d) if square else (np.pi * d * lb, np.pi * d * d / 4)
    mdot = m_side * a_side + m_top * a_top
    return heat_of_combustion * 1e3 * mdot / (a_side + a_top), mdot



def wick_meshes(d, length, burning=None, curl=0.0, square=False, z0=0.0):
    """A candle wick (CODE-GAS-041, CODE-ADD-034) of diameter d (square: a d x d section) standing
    `length` above the wax pool at z = z0, bent so the tip sits `curl` off axis in +x (Hamins 2005:
    5 mm over 12 mm). The lowest length - burning (None: Sunderland's fit) is a quenched solid that
    reaches 2 mm into the wax; the rest is a closed tube that emits fuel.
    -> (base, burn), each (verts, tris) in metres; base is None when the whole wick burns."""
    lb = wick_burning_length(d, length) if burning is None else min(burning, length)
    r, sides, ph = (d / np.sqrt(2.0), 4, np.pi / 4) if square else (0.5 * d, 12, 0.0)
    s0 = (length - lb) / length                      # path parameter where burning starts
    s = np.linspace(0.0, 1.0, 25)
    path = lambda u: np.stack([curl * u ** 2, 0.0 * u, z0 + length * u], 1)  # noqa: E731
    burn = path(np.concatenate([[s0], s[s > s0 + 1e-6]]))
    if s0 <= 1e-6:
        return None, tube(burn, r, sides, ph)
    base = path(np.concatenate([s[s < s0 - 1e-6], [s0]]))
    base[0, 2] -= 2e-3                               # into the wax
    return tube(base, r, sides, ph), tube(burn, r, sides, ph)
