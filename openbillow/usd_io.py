# SPDX-License-Identifier: GPL-3.0-or-later
"""Secondary-particle output: USD `Points` prims (`/snd/spray`, `/snd/foam`, `/snd/bubble`).

The bake writes one small file per frame (`particles_snd_####.usdc`, atomic, resumable) and,
when it finishes, merges them into one time-sampled `particles_snd.usdc` that
`wm.usd_import` loads as point clouds with a Mesh Sequence Cache. Widths are one constant
value per prim (the file would double otherwise). Needs `pxr`, which Blender's bundled Python has.
"""
from __future__ import annotations

import numpy as np

TYPES = ("spray", "foam", "bubble")
FPS = 24
# merge_frames saves after this much new array data: authored samples stay in RAM until a Save
MERGE_BATCH_BYTES = 512 * 2**20


def have_pxr() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return True


def _stage(path, f0, f1, fps):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.CreateNew(path)
    st.SetStartTimeCode(f0)
    st.SetEndTimeCode(f1)
    st.SetTimeCodesPerSecond(fps)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/snd")
    return st


def _points(st, name, width):
    from pxr import UsdGeom, Vt
    p = UsdGeom.Points.Define(st, f"/snd/{name}")
    p.GetWidthsAttr().Set(Vt.FloatArray([width]))
    p.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    return p


def _set(prim, frame, pos, vel):
    from pxr import Vt
    prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pos, np.float32)), frame)
    prim.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(vel, np.float32)), frame)


def write_frame(path, frame, data, width, fps=FPS):
    """data: {type: (pos (n,3) world metres, vel (n,3) m/s)} for the types to write."""
    st = _stage(path, frame, frame, fps)
    for name, (pos, vel) in data.items():
        _set(_points(st, name, width), frame, pos, vel)
    st.GetRootLayer().Save()


def merge_frames(path, frames, types, width, fps=FPS, batch_bytes=MERGE_BATCH_BYTES):
    """frames: {frame: per-frame file}. Writes the time-sampled file at `path`; returns its size in bytes."""
    import os
    from pxr import Usd, UsdGeom
    fs = sorted(frames)
    st = _stage(path, fs[0], fs[-1], fps)
    prims = {t: _points(st, t, width) for t in types}
    unsaved = 0
    for f in fs:
        src = Usd.Stage.Open(frames[f])
        for t in types:
            p = UsdGeom.Points(src.GetPrimAtPath(f"/snd/{t}"))
            pts, vel = p.GetPointsAttr().Get(f), p.GetVelocitiesAttr().Get(f)
            prims[t].GetPointsAttr().Set(pts, f)
            prims[t].GetVelocitiesAttr().Set(vel, f)
            unsaved += 12 * (len(pts) + len(vel))
        if unsaved >= batch_bytes:  # the saved samples then live in the file, not in RAM
            st.GetRootLayer().Save()
            unsaved = 0
    st.GetRootLayer().Save()
    return os.path.getsize(path)


def read_frame(path, frame, name):
    """(pos, vel) float32 arrays of one prim at `frame` (tests)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(path)  # keep a reference: the prim dies with its stage
    p = UsdGeom.Points(st.GetPrimAtPath(f"/snd/{name}"))
    return (np.array(p.GetPointsAttr().Get(frame), np.float32).reshape(-1, 3),
            np.array(p.GetVelocitiesAttr().Get(frame), np.float32).reshape(-1, 3))
