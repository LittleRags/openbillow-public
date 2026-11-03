# SPDX-License-Identifier: GPL-3.0-or-later
"""Force fields (step 35): pure-NumPy reference of Blender's effector evaluation for fluids.

Follows `update_effectors_task_cb` (fluid.cc) and `BKE_effectors_apply` /
`get_effector_data` / `effector_falloff` / `do_physical_effector` (effect.cc) for object
effectors. Blender calls it once per substep; the point velocity is the raw MAC component
at the cell index times dx (1/max res), `vel_to_sec` is the scene's integer fps.

Supported: Force (incl. gravitation), Wind, Vortex (point = old, plane/line = new),
Magnetic, Harmonic (incl. rest length), Drag, Turbulence (BLI_noise_generic_turbulence with
improved Perlin, 2 octaves, local or global coordinates); shapes POINT/PLANE/LINE; falloff
SPHERE/TUBE/CONE; Z direction; per-type and "All" effector weights.
Charge is accepted but has no effect: `pd_point_from_loc` never sets the point's charge,
so Blender's result is undefined there. Not ported: noise (random per call, `wind_func`),
Texture, Guide, Boid, Lennard-Jones, Fluid Flow, surface/points shapes, visibility.

No Warp import: this is the reference the GPU kernel is tested against, and the field
description that the add-on exports."""
from dataclasses import dataclass, field, replace

import numpy as np

TYPES = ("FORCE", "WIND", "VORTEX", "MAGNET", "HARMONIC", "CHARGE", "DRAG", "TURBULENCE")


@dataclass
class ForceField:
    """One field object, in world space. Names follow Blender's PartDeflect (DNA)."""
    type: str = "FORCE"
    location: tuple = (0.0, 0.0, 0.0)
    z_axis: tuple = (0.0, 0.0, 1.0)   # object matrix column 2 (normalised here)
    shape: str = "POINT"              # POINT, PLANE, LINE
    strength: float = 1.0             # f_strength
    flow: float = 0.0                 # f_flow
    damp: float = 1.0                 # f_damp (Harmonic, Drag)
    size: float = 0.0                 # f_size: Harmonic rest length, Turbulence noise size
    use_global_coords: bool = False   # PFIELD_GLOBAL_CO (Turbulence)
    use_gravity_falloff: bool = False  # PFIELD_GRAVITATION (Force)
    apply_to_location: bool = True    # PFIELD_DO_LOCATION
    z_direction: str = "BOTH"         # BOTH, POSITIVE, NEGATIVE
    falloff: str = "SPHERE"           # SPHERE, TUBE, CONE
    power: float = 0.0                # f_power
    use_min_distance: bool = False
    distance_min: float = 0.0
    use_max_distance: bool = False
    distance_max: float = 0.0
    radial_power: float = 0.0         # f_power_r (tube/cone)
    use_radial_min: bool = False
    radial_min: float = 0.0
    use_radial_max: bool = False
    radial_max: float = 0.0
    # Animation: (4, 4) or (k, 4, 4) object -> world matrices at scene times `frames`
    # (linear, held outside the range); when set they replace location / z_axis.
    matrix: np.ndarray | None = None
    frames: np.ndarray | None = None

    def at(self, frame):
        """The field with location / z_axis taken from the matrix at scene time `frame`."""
        if self.matrix is None:
            return self
        m = np.asarray(self.matrix, np.float64)
        if m.ndim == 3:
            if len(m) == 1:
                m = m[0]
            else:
                t = np.asarray(self.frames, np.float64)
                i = int(np.clip(np.searchsorted(t, frame) - 1, 0, len(t) - 2))
                u = min(max((frame - t[i]) / (t[i + 1] - t[i]), 0.0), 1.0)
                m = m[i] + (m[i + 1] - m[i]) * u
        return replace(self, location=tuple(m[:3, 3]), z_axis=tuple(m[:3, 2]), matrix=None, frames=None)


@dataclass
class EffectorWeights:
    """Domain effector weights: `all` (weight[0]) times the per-type weight."""
    all: float = 1.0
    types: dict = field(default_factory=dict)  # type name -> weight, missing = 1

    def weight(self, t):
        return self.types.get(t, 1.0)


def is_relevant(f, weights):
    """is_effector_relevant: weight, Affect Location, nonzero strength."""
    if weights.weight(f.type) == 0.0 or not f.apply_to_location:
        return False
    if f.strength != 0.0 or f.flow != 0.0:
        return True
    if f.type == "VORTEX":
        return f.shape != "POINT"
    if f.type == "DRAG":
        return f.damp != 0.0
    return False


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _normalize(v):
    """normalize_v3: returns (unit vector or 0, length)."""
    n = np.linalg.norm(v, axis=-1)
    safe = np.where(n > 0, n, 1.0)
    return np.where((n > 0)[..., None], v / safe[..., None], 0.0), n


def _falloff_func(fac, usemin, mindist, usemax, maxdist, power):
    if not usemin:
        mindist = 0.0
    out = np.power(1.0 + fac - mindist, -power)
    if usemin:
        out = np.where(fac < mindist, 1.0, out)
    if usemax:
        out = np.where(fac > maxdist, 0.0, out)
    return out


def effector_data(f, pos):
    """get_effector_data for an object effector: vec_to_point, distance, nor, vec_to_point2."""
    loc = np.asarray(f.location, np.float64)
    nor, _ = _normalize(np.asarray(f.z_axis, np.float64))
    if f.shape in ("PLANE", "LINE"):
        translate = _dot(pos - loc, nor)[..., None] * nor
        if f.type == "VORTEX" or f.shape == "LINE":
            eloc = loc + translate
        else:  # closest point on the effector's XY plane
            eloc = pos - translate
    else:
        eloc = np.broadcast_to(loc, pos.shape)
    vtp = pos - eloc
    dist = np.linalg.norm(vtp, axis=-1)
    if f.type == "HARMONIC" and f.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            vtp = vtp * np.where(dist > 0, (dist - f.size) / dist, 0.0)[..., None]
    return vtp, dist, nor, pos - loc


def effector_falloff(f, weights, dist, nor, vtp2):
    """effector_falloff. NaN (cone at the field centre) behaves like 0, as in Blender."""
    fall = np.full(dist.shape, weights.all * weights.weight(f.type))
    fac = _dot(vtp2, nor)
    dist_f = lambda x: _falloff_func(x, f.use_min_distance, f.distance_min,
                                     f.use_max_distance, f.distance_max, f.power)
    rad_f = lambda x: _falloff_func(x, f.use_radial_min, f.radial_min,
                                    f.use_radial_max, f.radial_max, f.radial_power)
    with np.errstate(divide="ignore", invalid="ignore"):
        if f.falloff == "SPHERE":
            fall = fall * dist_f(dist)
        elif f.falloff == "TUBE":
            fall = fall * dist_f(np.abs(fac))
            fall = fall * rad_f(np.linalg.norm(vtp2 - nor * fac[..., None], axis=-1))
        elif f.falloff == "CONE":
            fall = fall * dist_f(np.abs(fac))
            cos = np.clip(fac / np.linalg.norm(vtp2, axis=-1), -1.0, 1.0)
            fall = fall * rad_f(np.degrees(np.arccos(cos)))
    if f.z_direction == "POSITIVE":
        fall = np.where(fac < 0.0, 0.0, fall)
    elif f.z_direction == "NEGATIVE":
        fall = np.where(fac > 0.0, 0.0, fall)
    return fall


# BLI_noise_hash_uchar_512 (noise_c.cc): 256 bytes, stored twice in Blender.
_HASH = np.tile(np.frombuffer(bytes.fromhex(
    "a2a0193bf8ebaaeef31c67281ded00de952edc3f3a82354d6cba36d0f60c7932d159f4088b63892fb8b49783f28f18c751146587482042a880b54013b2227e57bc7f6b9d864cc8db7cd5254e5a557450cdb37abbc3cbb6e2e4ecfd980b96d39e5ca164f18161e1c42472498c904b843438ab78ca1f01d79311c158a931f9446dbf339c5f0994a38506c69a1e7b461530272b1b713c5bd66f62ac4fc2c00eb123a7df47b0776905e9e6e776730ffe6e9b56ef12a537fcaed9038edd10b9cec98dda2abd68179fbed40accd2e8433d70b7027d99d80d608a042c3e92e5af5307e029a6c5e3f5f74a41266a165e522d21adf091ffea54fa661a4539cf75a488fb5d"), np.uint8).astype(np.int64), 2)


def _grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def _new_perlin(x, y, z):
    """newPerlin (noise_c.cc), float32 like Blender."""
    f32 = np.float32
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    X, Y, Z = (a.astype(np.int64) & 255 for a in (fx, fy, fz))
    x, y, z = x - fx, y - fy, z - fz
    fade = lambda t: t * t * t * (t * (t * f32(6) - f32(15)) + f32(10))
    u, v, w = fade(x), fade(y), fade(z)
    lerp = lambda t, a, b: a + t * (b - a)
    H = _HASH
    A = H[X] + Y
    AA, AB = H[A] + Z, H[A + 1] + Z
    B = H[X + 1] + Y
    BA, BB = H[B] + Z, H[B + 1] + Z
    one = f32(1)
    return lerp(w, lerp(v, lerp(u, _grad(H[AA], x, y, z), _grad(H[BA], x - one, y, z)),
                        lerp(u, _grad(H[AB], x, y - one, z), _grad(H[BB], x - one, y - one, z))),
                lerp(v, lerp(u, _grad(H[AA + 1], x, y, z - one), _grad(H[BA + 1], x - one, y, z - one)),
                        lerp(u, _grad(H[AB + 1], x, y - one, z - one),
                             _grad(H[BB + 1], x - one, y - one, z - one))))


def noise_turbulence(size, x, y, z):
    """BLI_noise_generic_turbulence(size, x, y, z, oct=2, hard=false, basis=2 newPerlinU)."""
    f32 = np.float32
    x, y, z = (np.asarray(a, f32) for a in (x, y, z))
    if size != 0.0:
        inv = f32(1) / f32(size)
        x, y, z = x * inv, y * inv, z * inv
    total = np.zeros_like(x)
    amp, fs = f32(1), f32(1)
    for _ in range(3):
        total = total + (f32(0.5) + f32(0.5) * _new_perlin(fs * x, fs * y, fs * z)) * amp
        amp, fs = amp * f32(0.5), fs * f32(2)
    return total * f32(4.0 / 7.0)


def physical_effector(f, falloff, vtp, dist, nor, vtp2, vel, vel_to_sec):
    """do_physical_effector without noise: the force added to the total (location part)."""
    s = f.strength
    fl = falloff[..., None]
    t = f.type
    if t == "WIND":
        force = nor * (s * fl)
    elif t == "FORCE":
        force, _ = _normalize(vtp)
        if f.use_gravity_falloff:
            with np.errstate(divide="ignore"):
                g = np.where(dist < np.finfo(np.float32).eps, 0.0, s / dist**2)
            force = force * (g[..., None] * fl)
        else:
            force = force * (s * fl)
    elif t == "VORTEX":
        if f.shape == "POINT":  # old vortex
            force, _ = _normalize(np.cross(nor, vtp))
            force = force * (s * dist[..., None] * fl)
        else:  # new vortex
            temp = np.cross(nor, vtp2) * (s * fl)
            force = np.cross(nor, temp) * (s * fl)
            force = force + temp - vel * vel_to_sec
    elif t == "MAGNET":
        temp = np.cross(nor, vtp) if f.shape in ("POINT", "LINE") else np.broadcast_to(nor, vtp.shape)
        temp, _ = _normalize(temp)
        force = np.cross(vel, temp * (s * fl)) * vel_to_sec
    elif t == "HARMONIC":
        force = vtp * (-s * fl) + vel * (-f.damp * 2.0 * np.sqrt(abs(s)) * vel_to_sec)
    elif t == "DRAG":
        force, n = _normalize(vel)
        fac = (n * vel_to_sec)[..., None]
        sq, lin = min(s, 2.0), min(f.damp, 2.0)
        force = force * (-fl * fac * (sq * fac + lin))
    elif t == "TURBULENCE":
        # global: the point's world location; local: offset from the object plus its Z axis
        tp = vtp2 + (np.asarray(f.location, np.float64) if f.use_global_coords else nor)
        a, b, c = tp[..., 0], tp[..., 1], tp[..., 2]
        force = np.stack([noise_turbulence(f.size, a, b, c), noise_turbulence(f.size, b, c, a),
                          noise_turbulence(f.size, c, a, b)], -1).astype(np.float64)
        force = (force * 2.0 - 1.0) * (s * fl)
    else:  # CHARGE: the fluid point has no defined charge
        force = np.zeros_like(vtp)
    total = force / vel_to_sec
    if t not in ("HARMONIC", "DRAG") and f.flow != 0.0:
        total = total - vel * (f.flow * fl)
    return total


def effectors_apply(fields, weights, pos, vel, vel_to_sec):
    """BKE_effectors_apply: summed force at world positions pos (N,3), world velocity vel."""
    pos = np.asarray(pos, np.float64)
    vel = np.asarray(vel, np.float64)
    total = np.zeros_like(pos)
    for f in fields:
        if not is_relevant(f, weights):
            continue
        vtp, dist, nor, vtp2 = effector_data(f, pos)
        fall = effector_falloff(f, weights, dist, nor, vtp2)
        on = fall > 0.0  # also drops NaN
        fall = np.where(on, fall, 0.0)
        out = physical_effector(f, fall, vtp, dist, nor, vtp2, vel, vel_to_sec)
        total += np.where(on[..., None], out, 0.0)
    return total


def _keep_length(v, m):
    """Blender's direction-only transform: v' = normalize(m @ v) * |v|."""
    d, _ = _normalize(v @ np.asarray(m, np.float64).T)
    return d * np.linalg.norm(v, axis=-1)[..., None]


def compute_forces(fields, weights, flags, vel_mac, origin, cell_size, fps, mat3=None,
                   density=None, fuel=None, phi_obs=None, obstacle_bit=2):
    """update_effectors_task_cb on the whole grid: returns the `forces` grid (nx,ny,nz,3).

    vel_mac: Mantaflow MAC velocity (nx,ny,nz,3), read per index like Blender does.
    origin: world min corner; cell_size: world cell size (3,); mat3: domain rotation/scale
    (only used for Blender's direction-keeping transforms; identity if None)."""
    sh = flags.shape
    m = np.eye(3) if mat3 is None else np.asarray(mat3, np.float64)
    skip = (flags & obstacle_bit) != 0
    if fuel is not None:
        skip |= np.maximum(density, fuel) < np.finfo(np.float32).eps
    elif density is not None:
        skip |= density < np.finfo(np.float32).eps
    if phi_obs is not None:
        skip |= phi_obs < 0.0
    idx = np.indices(sh, dtype=np.float64).transpose(1, 2, 3, 0)
    pos = np.asarray(origin, np.float64) + (idx + 0.5) * np.asarray(cell_size, np.float64)
    dx = 1.0 / max(sh)
    vel = _keep_length(np.asarray(vel_mac, np.float64) * dx, m)
    keep = ~skip
    f = np.zeros(sh + (3,))
    world = effectors_apply(fields, weights, pos[keep], vel[keep], float(fps))
    f[keep] = np.clip(_keep_length(world, np.linalg.inv(m)) * 0.2, -1.0, 1.0)
    return f
