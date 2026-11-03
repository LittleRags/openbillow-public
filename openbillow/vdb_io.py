# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenVDB output in the layout of Blender's own smoke cache.

Grids `density`, `temperature` (fog volumes) and `velocity` (Vec3s, cell-centred m/s),
saved as half floats, with only voxels above the clipping threshold active.
Unlike Blender's cache (no translation), the transform is world space: voxel
size dx, index 0 centred at min_corner + dx/2, so the file lines up when loaded
as a Volume object. With noise, `density` and `flame` are the high-res grids
(voxel size dx/upres, their own transform); temperature and velocity stay
low-res. Needs `openvdb`, which Blender's bundled Python has.

Liquid frames (`write_liquid_vdb`): a narrow-band level set `phi` (world
units, negative inside, for Volume to Mesh) and `velocity` where the liquid
is. Particles go to a binary PLY (`write_ply`), since pyopenvdb cannot build
point grids.
"""
import glob
import sys

import numpy as np

try:
    import openvdb as vdb
except ImportError:  # Blender's Python keeps it in site-packages, which -I hides
    sys.path += glob.glob(sys.base_prefix + "/lib/python3*/site-packages")
    import openvdb as vdb


def _grid(cls, name, arr, transform, meta, clip, mask=None):
    g = cls()
    g.name = name
    g.transform = transform
    g.saveFloatAsHalf = True
    if mask is None:
        g.copyFromArray(arr, ijk=(0, 0, 0), tolerance=clip)
    else:  # vector grid: active where the mask is, zeros elsewhere
        g.copyFromArray(np.where(mask[..., None], arr, 0.0).astype(np.float32),
                        ijk=(0, 0, 0), tolerance=(0.0, 0.0, 0.0))
    for k, v in meta.items():
        g[k] = v
    return g


def cell_centred_mps(vel, vel_scale):
    """MAC face velocity (nx,ny,nz,3, solver units) -> cell-centred m/s: the mean of a
    cell's low and high face (the last cell keeps its low face), divided by vel_scale
    (m/s -> cells per solver time). Blender's volume motion blur reads this."""
    v = np.asarray(vel, np.float32)
    c = v.copy()
    for a in range(3):
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[a], sl_lo[a] = slice(0, -1), slice(1, None)
        c[tuple(sl_hi) + (a,)] = 0.5 * (v[tuple(sl_hi) + (a,)] + v[tuple(sl_lo) + (a,)])
    return c / np.float32(vel_scale)


def _space(shape, min_corner, dx):
    transform = vdb.createLinearTransform(voxelSize=float(dx))
    transform.postTranslate(tuple(float(m) + 0.5 * dx for m in min_corner))
    meta = {"file_base_resolution": tuple(int(n) for n in shape),
            "file_voxel_size": float(dx), "is_saved_as_half_float": True}
    return transform, meta


def write_vdb(path, density, temperature, velocity, min_corner, dx, clip=1e-6, flame=None,
              noise=None, vel_scale=None):
    """density, temperature (and flame, with fire): (nx,ny,nz) arrays; velocity:
    (nx,ny,nz,3) MAC face values (solver units); with vel_scale written cell-centred in
    m/s (`cell_centred_mps`), else as is. Active where
    density or flame is. noise = (density, flame or None, upres): high-res
    grids written as `density`/`flame` instead of the low-res ones.
    Returns the active density voxel count."""
    f32 = lambda a: np.ascontiguousarray(a, np.float32)
    density, temperature, velocity = f32(density), f32(temperature), f32(velocity)
    flame = None if flame is None else f32(flame)
    if vel_scale:
        velocity = cell_centred_mps(velocity, vel_scale)
    transform, meta = _space(density.shape, min_corner, dx)
    mask = density > clip
    if flame is not None:
        mask |= flame > clip
    d_tr, d_meta = transform, meta
    if noise is not None:
        density, nf, up = noise
        density = f32(density)
        flame = None if nf is None else f32(nf)
        d_tr, d_meta = _space(density.shape, min_corner, dx / up)
    grids = [_grid(vdb.FloatGrid, "density", density, d_tr, d_meta, clip),
             _grid(vdb.FloatGrid, "temperature", temperature, transform, meta, clip)]
    if flame is not None:
        grids.append(_grid(vdb.FloatGrid, "flame", flame, d_tr, d_meta, clip))
    for g in grids:
        g.gridClass = vdb.GridClass.FOG_VOLUME
    grids.append(_grid(vdb.Vec3SGrid, "velocity", velocity, transform, meta, clip, mask=mask))
    grids[-1].gridClass = vdb.GridClass.UNKNOWN if vel_scale else vdb.GridClass.STAGGERED
    vdb.write(path, grids)
    return grids[0].activeVoxelCount()


def write_liquid_vdb(path, phi, velocity, min_corner, dx, band=3.0, surface=None,
                     mesh_scale=1, wall_layers=None, vel_scale=None):
    """phi: (nx,ny,nz) level set in cells (Mantaflow phi); velocity (nx,ny,nz,3)
    MAC values, solver units. surface: the mesh level set (`LiquidSolver.surface`,
    mesh_scale times finer, voxel dx/mesh_scale); without it phi itself is written.
    The outer `wall_layers` (default mesh_scale) layers are set to 0.5 so the surface
    closes inside the domain walls (as Blender's mesh, phi.setBound(0.5, mesh_scale - 1);
    with fractions 2 * mesh_scale - 1 layers).
    Written as a LEVEL_SET grid: phi * voxel, background band * voxel; outside
    voxels beyond the band are inactive, inside ones stay active, clamped to
    -band * voxel (pruning them would need a sign flood fill, which the forced
    wall layer breaks where liquid touches a wall). Velocity (simulation grid) is
    active where phi < 0.5. Returns the active phi voxel count."""
    phi = np.asarray(phi, np.float32)
    ls_src, sc = (phi, 1) if surface is None else (surface, int(mesh_scale))
    ls_src = np.array(ls_src, np.float32)
    w = sc if wall_layers is None or surface is None else int(wall_layers)
    ls_src[:w] = ls_src[-w:] = 0.5
    ls_src[:, :w] = ls_src[:, -w:] = 0.5
    ls_src[:, :, :w] = ls_src[:, :, -w:] = 0.5
    hdx = dx / sc
    transform, meta = _space(ls_src.shape, min_corner, hdx)
    bg = float(band * hdx)
    ls = np.where(ls_src < band, np.maximum(ls_src, -band) * np.float32(hdx), bg).astype(np.float32)
    g = vdb.FloatGrid(bg)
    g.name = "phi"
    g.transform = transform
    g.saveFloatAsHalf = True
    g.copyFromArray(ls, ijk=(0, 0, 0), tolerance=0.0)
    g.gridClass = vdb.GridClass.LEVEL_SET
    for k, v in meta.items():
        g[k] = v
    vt, vmeta = _space(phi.shape, min_corner, dx)
    if vel_scale:
        velocity = cell_centred_mps(velocity, vel_scale)
    mask = phi < 0.5
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    v = _grid(vdb.Vec3SGrid, "velocity", np.ascontiguousarray(velocity, np.float32), vt,
              vmeta, 0.0, mask=mask)
    v.gridClass = vdb.GridClass.UNKNOWN if vel_scale else vdb.GridClass.STAGGERED
    vdb.write(path, [g, v])
    return g.activeVoxelCount()


def write_ply(out, pos, vel):
    """out: path or binary file object. Binary little-endian PLY point cloud: x y z (world metres) and vx vy vz
    (m/s) per vertex, no faces. Blender's PLY importer loads it as vertices."""
    n = len(pos)
    data = np.empty(n, [(c, "<f4") for c in ("x", "y", "z", "vx", "vy", "vz")])
    for i, c in enumerate("xyz"):
        data[c] = pos[:, i]
        data["v" + c] = vel[:, i]
    head = ("ply\nformat binary_little_endian 1.0\ncomment OpenBillow liquid particles\n"
            f"element vertex {n}\n"
            + "".join(f"property float {c}\n" for c in data.dtype.names) + "end_header\n")
    blob = head.encode("ascii") + data.tobytes()
    if hasattr(out, "write"):
        out.write(blob)
        return
    with open(out, "wb") as fh:
        fh.write(blob)
