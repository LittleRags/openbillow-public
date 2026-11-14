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
# FLAME_FRONT on top of COMBUSTION: level set, lift, face weights (MAC), 2 front scratch grids and the
# weighted pressure matrix. Measured +40.0 B/cell at 128 and 256 (plume BOTH, tests/measure_vram.py --fire-model).
PER_CELL_FLAME_FRONT = 40
# DIFFUSION on top of COMBUSTION (CODE-GAS-037): lift, face weights (MAC) and the weighted pressure matrix as
# FLAME_FRONT (28), plus sig, k_sgs, the source rate and 3 vec3 transport buffers (48). Not yet measured.
PER_CELL_MIXTURE_FRACTION = 76
# FLAME_FRONT / DIFFUSION viscosity (CODE-GAS-029): the nu grid and the diffusion CG work grids (4 vec3).
PER_CELL_GAS_VISCOSITY = 4 + 48
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
# velParts/mapWeights); per particle slot 55 B (pos, vel, flag, sort pairs,
# compaction scratch), APIC 103 (+ 3 affine rows). The stable index sort added 15 B to
# the measured 40 and 88: keys and values, each 2.5 x the particle count (CODE-LIQ-006). Moving obstacles add the smoke
# value (measured +40). The particle capacity peaks on the first frame, when
# initial liquid is seeded with particle_number^3 per cell (adjustNumber then
# keeps only the surface band, ~0.1-0.3 per domain cell): capacity =
# max(particle_number^3 * fill, LIQUID_MIN_PARTICLES) per domain cell. The first
# compaction holds old and new arrays at once: 64 B per slot (APIC 136) plus the
# 20 B sort pairs; the per-frame samples above cannot see that transient (re-measure
# with Warp's get_mempool_used_mem_high). Afterwards _compact shrinks the capacity to
# about 1.25x the live count (CODE-LIQ-002).
PER_CELL_LIQUID = 190
PER_CELL_LIQUID_APIC = 166
PER_PARTICLE_SORT = 15
PER_PARTICLE = 40 + PER_PARTICLE_SORT
PER_PARTICLE_APIC = 88 + PER_PARTICLE_SORT
LIQUID_MIN_PARTICLES = 0.5
# Mesh level set (LiquidSolver.surface), per cell of the mesh grid (mesh_scale^3 per
# domain cell), counted not measured: phi, particle phi, index start and counter,
# pAcc (12), rAcc and a smoothing grid = 36 B.
PER_MESH_CELL = 36
# High Quality (anisotropic) mesh: centre + matrix per particle slot = 48 B on about
# 1.25 x the live count, ~0.3 live particles per domain cell (see above), counted not measured.
PER_CELL_ANISO_MESH = 18
# Wall contact (mesh_wall_contact, CODE-LIQ-015): per particle slot the keep/start scratch (8 B)
# and a position/flag copy with up to one ghost each (2 x 16 B), on 1.25 x the live count, plus
# the mesh index sort pairs for the ghosts; ~0.3 live particles per domain cell, counted not measured.
PER_CELL_WALL_CONTACT = 20
# Secondary particles: phi_tmp + vel_tmp per simulation cell; flags, 4 potential/ratio grids,
# phi, phi_obs, vel per particle-grid cell (x scale^3, plus phi_out); 32 B per particle
# (pos, vel, flag, life) with growth headroom. The count is a rough budget (dam at f60: 16/cell at
# particle_scale 1, 8x that x SND_SCALE_BOOST at scale 2, about 5x with MULTIPLE sampling;
# tests/test_estimate_live.py, 60 frames at 64^3, 2026-11-09). A finer particle grid seeds more than
# scale^3 (live particles per cell at f60: 7.6 at scale 1, 88.5 at scale 2), and the peak is at a
# capacity growth, when the old and the new arrays are both allocated.
PER_CELL_SND = 16
PER_SND_GRID_CELL = 44
PER_SND_PARTICLE = 48
SND_PARTICLES_PER_CELL = 16
SND_SCALE_BOOST = 1.45
# Viscosity (ViscositySolver, counted): 7 volume grids + vec3 x, r, s, tmp.
PER_CELL_VISCOSITY = 76
PER_CELL_VISCOSITY_COUPLED = 24  # High Quality solve on top of that: vec3 inverse diagonal, z
PER_CELL_DIFFUSION = 48
PER_CELL_FRACTIONS = 24   # face fractions + KnSetWallBcsFrac's temp (two vec3)
PER_CELL_FORCES = 12      # the force-field grid (one vec3)
# Guiding (counted): PD scratch 9 MAC + invA (112), velT, weight, guidevel, guide cell
# velocity, guide phi and count (48), extrapolation marks (4).
PER_CELL_GUIDING = 164
PER_CELL_INVEL = 12       # liquid: the initial-velocity grid (one vec3) when a flow sets an initial velocity
PER_CELL_FLOW_PHI = 4     # liquid: each extra flow level set (static inflow, static outflow, moving outflow)
PER_CELL_GUIDE_VEL = 12   # the guide velocity grid, which has the parent domain's resolution
# System RAM for output, measured on a fully active 256-cubed grid (peak RSS during write_vdb)
HOST_TREE = 4.5        # OpenVDB float tree per active voxel
HOST_VELOCITY = 26     # velocity download 12, its tree 12.5, the mask 1, the clearing mask 0.5
HOST_LEVELSET = 9.5    # a level set or density being written: array 4, mask 1, its tree 4.5
HOST_GAS_STATE = 20    # checkpoint arrays: velocity, density, heat
HOST_FIRE_STATE = 12   # fuel, react, flame
HOST_NOISE_UV = 24     # the two uv grids
HOST_LIQUID_STATE = 16  # velocity, phi
HOST_PARTICLE = 40     # checkpoint arrays per particle
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
                   secondary=False, particle_scale=1, viscosity=False, viscosity_coupled=False, diffusion=False,
                   force_fields=False, dimensions=3, fractions=False, guiding=False, combustion=False, flame_front=False,
                   gas_viscosity=False, mixture_fraction=False,
                   high_quality=False, snd_multiple=False, aniso_mesh=False, wall_contact=False,
                   initial_velocity=False, flow_levelsets=0, guide_cells=0):
    """Bytes the solver allocates on the device (excluding the context).
    noise_upres 0 = noise off. liquid=True: a liquid domain (fire and noise
    ignored) with `particles_per_cell` from liquid_particles_per_cell and the
    mesh level set `mesh_scale` times finer. secondary=True adds the secondary-particle
    grids (`particle_scale` times finer) and a particle budget; viscosity=True / diffusion=True the solves,
    fractions=True the face-fraction grids, guiding=True the guide grids and PD solve.
    dimensions=2: the 2D slab grid. combustion=True (with fire): the COMBUSTION fire model (also True for FLAME_FRONT);
    flame_front=True (with fire): FLAME_FRONT on top of it; mixture_fraction=True (with fire): DIFFUSION on top
    of it; gas_viscosity=True with either: its viscosity solve.
    high_quality=True: gas quality HIGH (ignored for liquid). aniso_mesh=True: liquid meshed
    with the High Quality (anisotropic) generator. wall_contact=True: mesh_wall_contact on (Final or
    High Quality). initial_velocity=True: a liquid flow has an initial
    velocity; flow_levelsets (0-3): the extra liquid flow level sets; guide_cells: cell count of the parent's
    guide grid for a domain-guided bake (0: assume the same as this domain).
    """
    n = cells(size, resolution_max, dimensions)
    guide_extra = PER_CELL_GUIDE_VEL * (guide_cells - n) if guiding and guide_cells else 0
    if liquid:
        per = PER_CELL_LIQUID_APIC if apic else PER_CELL_LIQUID
        return int(n * (per + (PER_CELL_MOVING if moving_obstacles else 0)
                        + (PER_CELL_VISCOSITY if viscosity else 0)
                        + (PER_CELL_VISCOSITY_COUPLED if viscosity_coupled else 0)
                        + (PER_CELL_DIFFUSION if diffusion else 0)
                        + (PER_CELL_FRACTIONS if fractions else 0)
                        + (PER_CELL_FORCES if force_fields else 0)
                        + (PER_CELL_GUIDING if guiding else 0) + (PER_CELL_INVEL if initial_velocity else 0)
                        + PER_CELL_FLOW_PHI * max(0, min(3, int(flow_levelsets)))
                        + particles_per_cell * (PER_PARTICLE_APIC if apic else PER_PARTICLE)
                        + PER_MESH_CELL * max(1, int(mesh_scale)) ** 3
                        + (PER_CELL_ANISO_MESH if aniso_mesh else 0)
                        + (PER_CELL_WALL_CONTACT if wall_contact else 0)
                        + ((PER_CELL_SND + PER_SND_GRID_CELL * max(1, int(particle_scale)) ** 3
                            + SND_PARTICLES_PER_CELL * (5 if snd_multiple else 1) * max(1, int(particle_scale)) ** 3
                            * (SND_SCALE_BOOST if int(particle_scale) > 1 else 1)
                            * PER_SND_PARTICLE) if secondary else 0)) + guide_extra)
    per = PER_CELL + (PER_CELL_OBSTACLE if has_obstacles or moving_obstacles else 0)
    total = guide_extra + n * (per + (PER_CELL_MOVING if moving_obstacles else 0)
                 + (PER_CELL_FIRE if fire else 0) + (PER_CELL_COMBUSTION if fire and combustion else 0)
                 + (PER_CELL_FLAME_FRONT if fire and flame_front else 0)
                 + (PER_CELL_MIXTURE_FRACTION if fire and mixture_fraction else 0)
                 + (PER_CELL_GAS_VISCOSITY if fire and (flame_front or mixture_fraction) and gas_viscosity else 0) + (PER_CELL_FORCES if force_fields else 0)
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


def host_bytes(size, resolution_max, fire=False, noise_upres=0, liquid=False, particles_per_cell=0.0,
               mesh_scale=1, dimensions=3, **_):
    """Peak system RAM (bytes) a bake needs beside the solver's grids, whatever the device: the larger
    of writing a frame (vdb_io, grids fully active, the worst case for the OpenVDB trees) and saving
    a checkpoint; the two never overlap. Takes estimate_bytes' keywords."""
    n = cells(size, resolution_max, dimensions)
    if liquid:
        m = n * max(1, int(mesh_scale)) ** 3
        write = max(HOST_LEVELSET * m + n, HOST_TREE * m + (HOST_VELOCITY + HOST_TREE) * n)
        return int(max(write, n * HOST_LIQUID_STATE + n * particles_per_cell * HOST_PARTICLE))
    hi = n * noise_upres ** 3 if noise_upres else n
    fog = 2 if fire else 1  # density, and flame with fire: at the noise resolution if there is noise
    write = max(HOST_LEVELSET * hi + n, fog * HOST_TREE * hi + (HOST_VELOCITY + HOST_TREE) * n)
    state = n * (HOST_GAS_STATE + (HOST_FIRE_STATE if fire else 0))
    if noise_upres:
        state += n * HOST_NOISE_UV + hi * 4 * (3 if fire else 1)
    return int(max(write, state))


def device_plan(need, vram_free, ram_free, managed_ok, host=0):
    """Where Device Auto bakes `need` bytes: "gpu" (fits in free VRAM), "managed" (GPU with CUDA
    managed memory paging through system RAM, about 2x slower), "cpu" (no GPU, or managed
    unavailable, and it fits in free RAM) or "none". vram_free is None without a CUDA GPU.
    host (`host_bytes`) is system RAM for frame output, which competes with paged or CPU grids.
    One source of truth for the CLI and the panel."""
    if vram_free is not None:
        if need <= vram_free:
            return "gpu"
        if managed_ok and need <= managed_budget(vram_free, ram_free - host):
            return "managed"
    if need + host <= ram_free:
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


def capillary_limit(sigma, res_per_m, scale_acc):
    """(sigma / rho in cells^3 per solver time^2, largest stable dt). The capillary limit is
    dt < sqrt(rho_avg dx^3 / (2 pi sigma)) with rho_avg = rho / 2 (air). Infinite without tension."""
    import math
    st = sigma / 1000.0 * res_per_m ** 2 * scale_acc
    return st, math.sqrt(1.0 / (4.0 * math.pi * st)) if st > 0.0 else math.inf


def surface_tension_substeps(size, resolution_max, sigma, fps=24.0, time_scale=1.0, timesteps_max=4):
    """Substeps per frame the capillary limit forces (liquid.py), or 0 if it is below the Max Substeps."""
    import math
    dt0 = 0.1
    res_per_m = resolution_max / max(size)
    scale_acc = res_per_m * (1.0 / (dt0 * 25.0)) ** 2
    st, cap = capillary_limit(sigma, res_per_m, scale_acc)
    if st <= 0.0:
        return 0
    n = math.ceil(dt0 * (25.0 / fps) * time_scale / cap)
    return n if n > timesteps_max else 0
