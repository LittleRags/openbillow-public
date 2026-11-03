# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender's default primitive meshes as triangle lists (outward winding).

Both are convex, so each triangle is oriented away from the origin. Blender
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
