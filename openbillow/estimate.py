# SPDX-License-Identifier: GPL-3.0-or-later
"""VRAM estimate for a bake. Pure Python (no Warp), so the add-on can show it live.

Constants were measured with tests/measure_vram.py (3 frames, resolution_max
64/128/192/256, default 2x2x4 domain, RTX 12 GB): backend-allocated bytes per
cell were 124.0 (no obstacle) and 126.0 (with obstacle) at every size, i.e.
exactly linear; with a moving obstacle 160 (moving only) and 168 (moving +
static), also linear; fire adds 20. Noise was measured the same way (upres 2 and 4,
64-192/256, smoke and fire; tests/measure_vram.py plume). `FIXED_MB` is the CUDA context + Warp modules (measured
~200 MB over a bare process; rounded up). Mesh BVHs are KBs and ignored."""
import os
import sys

MB = 2**20
PER_CELL = 124          # bytes: grids, advection work grids, MGPCG vectors/levels
PER_CELL_OBSTACLE = 4   # phi_obs
# Moving obstacles: obvel_c, obvel, num_obs, extrapolation scratch, moving phi,
# plus static phi when mixed. Measured +34 (moving only) / +42 (mixed); use the
# mixed value, so moving-only over-estimates by ~5%.
PER_CELL_MOVING = 42
# Fire (flow type FIRE/BOTH): fuel, react, flame, fuel_in, react_in = 5 float32 grids.
# Measured 20.0 B/cell at 64-256 (BOTH, with and without obstacle).
PER_CELL_FIRE = 20
PER_CELL_COMBUSTION = 4  # COMBUSTION: the expansion (divergence) grid
PER_CELL_HIGH_QUALITY = 12  # quality HIGH (advection-reflection): the unprojected velocity (MAC)
# Noise (upres): per high-res cell flags, vel (MAC), phi, density, inflow and
# emission grids + advection scratch = 40; fire adds fuel, react, flame = 12.
# Per low-res cell vel_tmp, emission, 2 UV grids, energy, flags + scratch = 68.
# Plus the 128^3 vec3 noise tile. Fits all 14 runs to within 1 MB.
PER_CELL_NOISE = 40
PER_CELL_NOISE_FIRE = 12
PER_CELL_NOISE_LOW = 68
NOISE_TILE_BYTES = 128 ** 3 * 12
# Liquid (tests/measure_vram.py dam/pool/paddle, 128-256, 4 frames; fits within 3%):
# grids incl. FLIP velParts/mapWeights and MGPCG = 190 B/cell (APIC: 166, no
# velParts/mapWeights); per particle slot 40 B (pos, vel, flag, sort index,
# compaction scratch), APIC 88 (+ 3 affine rows). Moving obstacles add the smoke
# value (measured +40). The particle capacity peaks on the first frame, when
# initial liquid is seeded with particle_number^3 per cell (adjustNumber then
# keeps only the surface band, ~0.1-0.3 per domain cell): capacity =
# max(particle_number^3 * fill, LIQUID_MIN_PARTICLES) per domain cell. The
# compaction transient did not show in the measured peaks.
PER_CELL_LIQUID = 190
PER_CELL_LIQUID_APIC = 166
PER_PARTICLE = 40
PER_PARTICLE_APIC = 88
LIQUID_MIN_PARTICLES = 0.5
# Mesh level set (LiquidSolver.surface), per cell of the mesh grid (mesh_scale^3 per
# domain cell), counted not measured: phi, particle phi, index start and counter,
# pAcc (12), rAcc and a smoothing grid = 36 B.
PER_MESH_CELL = 36
# Secondary particles: phi_tmp + vel_tmp per simulation cell; flags, 4 potential/ratio grids,
# phi, phi_obs, vel per particle-grid cell (x scale^3, plus phi_out); 32 B per particle
# (pos, vel, flag, life) with growth headroom. The count is a rough budget (dam at f60: 16/cell).
PER_CELL_SND = 16
PER_SND_GRID_CELL = 44
PER_SND_PARTICLE = 48
SND_PARTICLES_PER_CELL = 16
# Viscosity (ViscositySolver, counted): 7 volume grids + vec3 x, r, s, tmp.
PER_CELL_VISCOSITY = 76
PER_CELL_DIFFUSION = 48
PER_CELL_FRACTIONS = 24   # face fractions + KnSetWallBcsFrac's temp (two vec3)
PER_CELL_FORCES = 12      # the force-field grid (one vec3)
# Guiding (counted): PD scratch 9 MAC + invA (112), velT, weight, guidevel, guide cell
# velocity, guide phi and count (48), extrapolation marks (4).
PER_CELL_GUIDING = 164
FIXED_MB = 256
HEADROOM_MB = 1024      # default limit = device total - this


def grid_shape(size, resolution_max, dimensions=3):
    """Same rule as SmokeSolver: longest side gets resolution_max cells
    (in 2D the longest of X and Z, and Y is the 3-cell slab)."""
    if dimensions == 2:
        size = slab((0.0, 0.0, 0.0), size, resolution_max)[1]
    dx = max(size) / resolution_max
    return tuple(max(3, int(n / dx + 0.5)) for n in size)


def slab(min_corner, size, resolution_max):
    """2D domain (X-Z plane): min_corner and size of the 3-cell slab in Y at
    the domain's Y centre. dx comes from the X and Z sides only."""
    dx = max(size[0], size[2]) / resolution_max
    cy = min_corner[1] + 0.5 * size[1]
    return ((min_corner[0], cy - 1.5 * dx, min_corner[2]),
            (size[0], 3.0 * dx, size[2]))


def cells(size, resolution_max, dimensions=3):
    a, b, c = grid_shape(size, resolution_max, dimensions)
    return a * b * c


def liquid_particles_per_cell(fill, particle_number=2):
    """Peak particle capacity per domain cell; fill = initial liquid volume /
    domain volume."""
    return max(particle_number ** 3 * min(max(fill, 0.0), 1.0), LIQUID_MIN_PARTICLES)


def mesh_volume(verts, tris, matrix):
    """Volume (m^3) of a closed triangle mesh in object space under a (4, 4)
    world matrix (divergence theorem; sign-free)."""
    import numpy as np
    v = np.asarray(verts, float) @ np.asarray(matrix, float)[:3, :3].T
    a, b, c = (v[np.asarray(tris)[:, i]] for i in range(3))
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def estimate_bytes(size, resolution_max, has_obstacles=False, moving_obstacles=False, fire=False,
                   noise_upres=0, liquid=False, apic=False, particles_per_cell=0.0, mesh_scale=1,
                   secondary=False, particle_scale=1, viscosity=False, diffusion=False,
                   force_fields=False, dimensions=3, fractions=False, guiding=False, combustion=False,
                   high_quality=False, snd_multiple=False):
    """Bytes the solver allocates on the device (excluding the context).
    noise_upres 0 = noise off. liquid=True: a liquid domain (fire and noise
    ignored) with `particles_per_cell` from liquid_particles_per_cell and the
    mesh level set `mesh_scale` times finer. secondary=True adds the secondary-particle
    grids (`particle_scale` times finer) and a particle budget; viscosity=True / diffusion=True the solves,
    fractions=True the face-fraction grids, guiding=True the guide grids and PD solve.
    dimensions=2: the 2D slab grid. combustion=True (with fire): the COMBUSTION fire model.
    high_quality=True: gas quality HIGH (ignored for liquid)."""
    n = cells(size, resolution_max, dimensions)
    if liquid:
        per = PER_CELL_LIQUID_APIC if apic else PER_CELL_LIQUID
        return int(n * (per + (PER_CELL_MOVING if moving_obstacles else 0)
                        + (PER_CELL_VISCOSITY if viscosity else 0)
                        + (PER_CELL_DIFFUSION if diffusion else 0)
                        + (PER_CELL_FRACTIONS if fractions else 0)
                        + (PER_CELL_FORCES if force_fields else 0)
                        + (PER_CELL_GUIDING if guiding else 0)
                        + particles_per_cell * (PER_PARTICLE_APIC if apic else PER_PARTICLE)
                        + PER_MESH_CELL * max(1, int(mesh_scale)) ** 3
                        + ((PER_CELL_SND + PER_SND_GRID_CELL * max(1, int(particle_scale)) ** 3
                            + SND_PARTICLES_PER_CELL * (4 if snd_multiple else 1) * PER_SND_PARTICLE) if secondary else 0)))
    per = PER_CELL + (PER_CELL_OBSTACLE if has_obstacles or moving_obstacles else 0)
    total = n * (per + (PER_CELL_MOVING if moving_obstacles else 0)
                 + (PER_CELL_FIRE if fire else 0) + (PER_CELL_COMBUSTION if fire and combustion else 0) + (PER_CELL_FORCES if force_fields else 0)
                 + (PER_CELL_GUIDING if guiding else 0) + (PER_CELL_HIGH_QUALITY if high_quality else 0))
    if noise_upres:
        total += (NOISE_TILE_BYTES + n * PER_CELL_NOISE_LOW
                  + n * noise_upres ** 3 * (PER_CELL_NOISE + (PER_CELL_NOISE_FIRE if fire else 0)))
    return total


def estimate_total_mb(size, resolution_max, *args, **kw):
    """estimate_bytes (same arguments) in MB plus the fixed context cost."""
    return estimate_bytes(size, resolution_max, *args, **kw) / MB + FIXED_MB


def default_limit_mb(total_bytes):
    return max(0, total_bytes // MB - HEADROOM_MB)


def system_memory() -> tuple[int, int]:
    """(total, available) system RAM in bytes; (0, 0) if unknown."""
    if sys.platform == "win32":
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MemStatus()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys, st.ullAvailPhys
        return 0, 0
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemFree"])
    except (OSError, KeyError, ValueError):
        try:  # macOS and other POSIX
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return total, total // 2
        except (ValueError, OSError, AttributeError):
            return 0, 0


def managed_ok():
    """CUDA managed memory beyond VRAM works on Linux; Windows (WDDM) may refuse it, so it stays
    off there until tested (Auto then falls back to the CPU)."""
    return sys.platform != "win32"


RAM_RESERVE = 4 * 2**30  # system RAM kept free for Blender and the OS when paging the GPU grid


def managed_budget(vram_free, ram_free):
    """Bytes a managed-memory bake may use: free VRAM plus free RAM minus RAM_RESERVE."""
    return vram_free + max(0, ram_free - RAM_RESERVE)


def device_plan(need, vram_free, ram_free, managed_ok):
    """Where Device Auto bakes `need` bytes: "gpu" (fits in free VRAM), "managed" (GPU with CUDA
    managed memory paging through system RAM, about 2x slower), "cpu" (no GPU, or managed
    unavailable, and it fits in free RAM) or "none". vram_free is None without a CUDA GPU.
    One source of truth for the CLI and the panel."""
    if vram_free is not None:
        if need <= vram_free:
            return "gpu"
        if managed_ok and need <= managed_budget(vram_free, ram_free):
            return "managed"
    if need <= ram_free:
        return "cpu"
    return "none"


def max_resolution_for(limit_mb, size, *args, **kw):
    """Largest resolution_max whose estimate (incl. fixed) fits in limit_mb; 0 if none.
    Other arguments as estimate_bytes."""
    if estimate_total_mb(size, 1, *args, **kw) > limit_mb:
        return 0
    lo, hi = 1, 1
    while estimate_total_mb(size, hi * 2, *args, **kw) <= limit_mb and hi < 16384:
        hi *= 2
    hi *= 2
    while hi - lo > 1:  # lo fits, hi doesn't (monotonic)
        mid = (lo + hi) // 2
        if estimate_total_mb(size, mid, *args, **kw) <= limit_mb:
            lo = mid
        else:
            hi = mid
    return lo


def surface_tension_substeps(size, resolution_max, sigma, fps=24.0, time_scale=1.0, timesteps_max=4):
    """Substeps per frame the capillary limit forces (liquid.py), or 0 if it is below the Max Substeps."""
    import math
    dt0 = 0.1
    res_per_m = resolution_max / max(size)
    scale_acc = res_per_m * (1.0 / (dt0 * 25.0)) ** 2
    st = sigma / 1000.0 * res_per_m ** 2 * scale_acc
    if st <= 0.0:
        return 0
    cap = math.sqrt(1.0 / (4.0 * math.pi * st))
    n = math.ceil(dt0 * (25.0 / fps) * time_scale / cap)
    return n if n > timesteps_max else 0
