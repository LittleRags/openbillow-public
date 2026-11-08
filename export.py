# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene exporter: builds a bake folder (bake.json + objects.npz) from a Blender
scene. See the bake-folder layout in openbillow/cli.py. No UI here; the operators (4.5) collect the
objects and their settings and call `export_bake`.

The `*_kw` helpers read settings with Blender's FluidDomain/Flow/Effector
attribute names, so they work on the Fluid modifier's settings and on our own
PropertyGroups (which copy those names).
"""
from __future__ import annotations

import bpy
import numpy as np

from .openbillow.bakefile import write_bake
from .openbillow.forcefields import TYPES as FIELD_TYPES, EffectorWeights, ForceField
from .openbillow.liquid import LiquidFlow, LiquidSettings
from .openbillow.solver import DomainSettings, MeshFlow, MeshGuide, MeshObstacle

STATIC_TOL = 1e-7
# MANTA_main.cpp: a side without border collision is open (outflow).
_BORDERS = (("use_collision_border_left", "x"), ("use_collision_border_right", "X"),
            ("use_collision_border_front", "y"), ("use_collision_border_back", "Y"),
            ("use_collision_border_bottom", "z"), ("use_collision_border_top", "Z"))

class ExportError(Exception):
    """A scene that v1 can't bake; the message is for the user."""

def domain_kw(scene, ds) -> dict:
    """DomainSettings fields from Blender-named domain settings `ds`
    (min_corner/size come from the domain mesh, see `export_bake`)."""
    if scene.use_gravity:  # fluid.cc update_final_gravity
        g = scene.gravity
    else:
        g = ds.gravity
    g = tuple(float(x) * ds.effector_weights.gravity for x in g)
    if ds.use_2d and ds.use_noise:
        raise ExportError("2D domains do not support High Resolution (noise)")
    return dict(resolution_max=ds.resolution_max, dimensions=2 if ds.use_2d else 3,
                fps=scene.render.fps / scene.render.fps_base, render_fps=int(scene.render.fps),
                time_scale=ds.time_scale, cfl=ds.cfl_condition,
                timesteps_min=ds.timesteps_min, timesteps_max=ds.timesteps_max,
                gravity=g, alpha=ds.alpha, beta=ds.beta, vorticity=ds.vorticity,
                dissolve=ds.use_dissolve_smoke, dissolve_speed=ds.dissolve_speed,
                dissolve_log=ds.use_dissolve_smoke_log,
                burning_rate=ds.burning_rate, flame_smoke=ds.flame_smoke,
                flame_vorticity=ds.flame_vorticity, flame_ignition=ds.flame_ignition,
                flame_max_temp=ds.flame_max_temp, fire_model=ds.fire_model, quality=ds.quality,
                heat_release=ds.heat_release, cooling=ds.cooling,
                cooling_radiative=ds.cooling_radiative, expansion=ds.expansion,
                flame_kelvin=ds.flame_kelvin, ambient_kelvin=ds.ambient_kelvin,
                disturbance=ds.disturbance, disturbance_size=ds.disturbance_size,
                disturbance_rate=ds.disturbance_rate, shredding=ds.shredding,
                use_noise=ds.use_noise, noise_upres=ds.noise_upres,
                noise_strength=ds.noise_strength, noise_pos_scale=ds.noise_pos_scale,
                noise_time_anim=ds.noise_time_anim,
                outflow="".join(c for a, c in _BORDERS if not getattr(ds, a)), **guide_domain_kw(ds))


def guide_domain_kw(ds) -> dict:
    """Guiding fields shared by gas and liquid domains (guide_parent / write_guide
    are set by the operator, which knows the bake folders)."""
    return dict(use_guide=ds.use_guide, guide_alpha=ds.guide_alpha, guide_beta=ds.guide_beta,
                guide_vel_factor=ds.guide_vel_factor, guide_source=ds.guide_source)

# LiquidSettings field -> Blender's property name (without the sndparticle_ prefix)
_SND = {"sndparticle_tau_min_wc": "potential_min_wavecrest", "sndparticle_tau_max_wc": "potential_max_wavecrest",
        "sndparticle_tau_min_ta": "potential_min_trappedair", "sndparticle_tau_max_ta": "potential_max_trappedair",
        "sndparticle_tau_min_k": "potential_min_energy", "sndparticle_tau_max_k": "potential_max_energy",
        "sndparticle_k_wc": "sampling_wavecrest", "sndparticle_k_ta": "sampling_trappedair",
        "sndparticle_k_b": "bubble_buoyancy", "sndparticle_k_d": "bubble_drag",
        "sndparticle_l_min": "life_min", "sndparticle_l_max": "life_max"}


# Multiple sampling throws about 3x the particles at equal rates (A/B 2026-11-03: 200/40 -> 65/13
# gave the same count), so the user's Blender-style rates are divided to keep the amount comparable.
_MULTIPLE_RATE_SCALE = 0.325


def _multiple_rates(ds):
    if ds.quality == "CLASSIC" or ds.sndparticle_sampling != "MULTIPLE":
        return {}
    return {"sndparticle_k_wc": round(ds.sndparticle_sampling_wavecrest * _MULTIPLE_RATE_SCALE),
            "sndparticle_k_ta": round(ds.sndparticle_sampling_trappedair * _MULTIPLE_RATE_SCALE)}


_FIELD_SHAPES = ("POINT", "PLANE", "LINE")


def field_supported(ob) -> bool:
    """A force field object the solver handles (type and shape, see forcefields.py)."""
    f = ob.field
    return f is not None and f.type in FIELD_TYPES and f.shape in _FIELD_SHAPES


def field_warnings(scene) -> list:
    """Messages for force fields the solver skips or only partly follows."""
    out = []
    for ob in scene.objects:
        f = ob.field
        if f is None or f.type == "NONE":
            continue
        if not field_supported(ob):
            out.append(f"{ob.name}: {f.type.title()} / {f.shape.title()} not supported, skipped")
        elif f.noise > 0.0:
            out.append(f"{ob.name}: Noise is ignored (random per call in Blender)")
    return out


def field_kw(fl) -> dict:
    """ForceField fields (without the matrix) from Blender's PartDeflect `ob.field`.
    DNA f_damp is RNA `harmonic_damping` (= `quadratic_drag`), f_strength is `strength`
    (= `linear_drag`), f_size is `size` (= `rest_length`)."""
    return dict(type=fl.type, shape=fl.shape, strength=fl.strength, flow=fl.flow,
                damp=fl.harmonic_damping, size=fl.size, use_gravity_falloff=fl.use_gravity_falloff,
                apply_to_location=fl.apply_to_location, z_direction=fl.z_direction,
                falloff=fl.falloff_type, power=fl.falloff_power,
                use_min_distance=fl.use_min_distance, distance_min=fl.distance_min,
                use_max_distance=fl.use_max_distance, distance_max=fl.distance_max,
                radial_power=fl.radial_falloff, use_radial_min=fl.use_radial_min,
                radial_min=fl.radial_min, use_radial_max=fl.use_radial_max,
                radial_max=fl.radial_max, use_global_coords=fl.use_global_coords)


def weights_kw(w) -> EffectorWeights:
    """EffectorWeights from the domain's weights (All plus one weight per field type)."""
    names = {"FORCE": "force", "WIND": "wind", "VORTEX": "vortex", "MAGNET": "magnetic",
             "HARMONIC": "harmonic", "CHARGE": "charge", "DRAG": "drag",
             "TURBULENCE": "turbulence"}
    return EffectorWeights(float(w.all), {t: float(getattr(w, n)) for t, n in names.items()})


def liquid_domain_kw(scene, ds) -> dict:
    """LiquidSettings fields from the domain settings (same gravity/outflow rules as `domain_kw`)."""
    if ds.use_2d:
        raise ExportError("2D domains are gas only for now")
    kw = domain_kw(scene, ds)
    keep = ("resolution_max", "fps", "time_scale", "cfl", "timesteps_min", "timesteps_max",
            "gravity", "outflow", "quality", *guide_domain_kw(ds))
    return {**{k: kw[k] for k in keep}, "use_apic": ds.use_apic == "APIC", "flip_ratio": ds.flip_ratio,
            "particle_number": ds.particle_number, "particle_min": ds.particle_minimum,
            "particle_max": ds.particle_maximum, "particle_radius": ds.particle_radius,
            "particle_band_width": ds.particle_band_width,
            "particle_randomness": ds.particle_randomness,
            "delete_in_obstacle": ds.use_delete_in_obstacle, "output_particles": ds.output_particles,
            "flip_wall_fix": ds.flip_wall_fix, "robust_collisions": ds.robust_collisions,
            "collision_margin": ds.collision_margin, "volume_correction": ds.volume_correction,
            "volume_correction_rate": ds.volume_correction_rate,
            "use_viscosity": ds.use_viscosity, "viscosity_value": ds.viscosity_value,
            "use_diffusion": ds.use_diffusion, "viscosity_base": ds.viscosity_base,
            "viscosity_exponent": ds.viscosity_exponent, "surface_tension": ds.surface_tension,
            "use_fractions": ds.use_fractions,
            "fractions_threshold": ds.fractions_threshold, "fractions_distance": ds.fractions_distance,
            **{k: getattr(ds, k) for k in ("use_spray_particles", "use_foam_particles",
                                           "use_bubble_particles", "particle_scale",
                                           "sndparticle_boundary", "sndparticle_potential_radius",
                                           "sndparticle_width")},
            "sndparticle_antitunneling": ("OFF" if ds.quality == "CLASSIC"
                                          else ds.sndparticle_antitunneling),
            "sndparticle_antitunnel_steps": ds.sndparticle_antitunnel_steps,
            "sndparticle_sampling": ("SINGLE" if ds.quality == "CLASSIC" else ds.sndparticle_sampling),
            **{k: getattr(ds, "sndparticle_" + v) for k, v in _SND.items()},
            **_multiple_rates(ds),
            **{k: getattr(ds, k) for k in ("mesh_generator", "mesh_scale", "mesh_particle_radius",
                                           "mesh_smoothen_pos", "mesh_smoothen_neg",
                                           "mesh_concave_upper", "mesh_concave_lower")}}

def liquid_flow_kw(fs) -> dict:
    return dict(behavior=fs.behavior, surface_distance=fs.liquid_surface_distance,
                plane_init=fs.use_plane_init, use_initial_velocity=fs.use_initial_velocity,
                velocity_factor=fs.velocity_factor, velocity_normal=fs.velocity_normal,
                velocity_coord=tuple(float(x) for x in fs.velocity_coord),
                emit_start=fs.emit_start if fs.use_emit_range else 0,
                emit_end=fs.emit_end if fs.use_emit_range else 0)

def flow_kw(fs) -> dict:
    return dict(density=fs.density, temperature=fs.temperature,
                surface_distance=fs.surface_distance, volume_density=fs.volume_density,
                plane_init=fs.use_plane_init, flow_type=fs.flow_type, fuel_amount=fs.fuel_amount,
                emit_start=fs.emit_start if fs.use_emit_range else 0,
                emit_end=fs.emit_end if fs.use_emit_range else 0)

def obstacle_kw(es) -> dict:
    return dict(surface_thickness=es.surface_distance, plane_init=es.use_plane_init)

_GUIDE_MODES = {"MAXIMUM": "MAX", "MINIMUM": "MIN"}  # Blender's guide_mode -> MeshGuide

def guide_kw(gs) -> dict:
    return dict(obstacle_kw(gs), guide_mode=_GUIDE_MODES.get(gs.guide_mode, gs.guide_mode),
                vel_multi=gs.velocity_factor)

def _mesh_arrays(ob, depsgraph):
    """Evaluated mesh (modifiers applied) in object space + loop triangles."""
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        v = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", v)
        t = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", t)
    finally:
        ev.to_mesh_clear()
    return v.reshape(-1, 3), t.reshape(-1, 3)

def _domain_box(ob, depsgraph, flat_y=False):
    """World-space AABB of the domain's evaluated mesh; refuses rotation/shear.
    flat_y: a 2D domain may be a plane in Y."""
    m = np.array(ob.matrix_world, np.float64)
    lin = m[:3, :3]
    if np.abs(lin - np.diag(np.diag(lin))).max() > 1e-6 or (np.diag(lin) <= 0).any():
        raise ExportError(f"Domain '{ob.name}' is rotated, sheared or mirrored; "
                          "v1 needs an unrotated domain (apply rotation)")
    v, _ = _mesh_arrays(ob, depsgraph)
    if len(v) == 0:
        raise ExportError(f"Domain '{ob.name}' has no vertices")
    w = v.astype(np.float64) @ lin.T + m[:3, 3]
    lo, hi = w.min(0), w.max(0)
    if (hi - lo <= 0)[[0, 2] if flat_y else [0, 1, 2]].any():
        raise ExportError(f"Domain '{ob.name}' is flat")
    return tuple(lo.tolist()), tuple((hi - lo).tolist())

def export_bake(folder: str, scene, domain, flows, obstacles, domain_settings: dict,
                frame_start: int | None = None, frame_end: int | None = None,
                samples_per_frame: int = 4, liquid: bool = False,
                fields=(), weights: EffectorWeights | None = None,
                guides=()) -> DomainSettings | LiquidSettings:
    """Write `folder`/bake.json + objects.npz.

    flows / obstacles / guides: lists of (object, settings dict) with MeshFlow /
    MeshObstacle / MeshGuide field names (`flow_kw` / `obstacle_kw` / `guide_kw`). domain_settings:
    DomainSettings fields (LiquidSettings and `liquid_flow_kw` if `liquid`) except min_corner/size/flows/obstacles (`domain_kw`).
    fields: force field objects (`field_kw(ob.field)`), their matrices sampled like the others;
    weights: `weights_kw`.
    Meshes and the domain box are taken at frame_start; world matrices are
    sampled at f + s/N for f = frame_start-1 .. frame_end. The scene's frame is
    restored afterwards.
    """
    fs = scene.frame_start if frame_start is None else int(frame_start)
    fe = scene.frame_end if frame_end is None else int(frame_end)
    if fe < fs:
        raise ExportError(f"Frame end {fe} is before frame start {fs}")
    meshed = list(flows) + list(obstacles) + list(guides)
    for ob, _ in meshed:
        if ob.type != "MESH":
            raise ExportError(f"'{ob.name}' is not a mesh object")
    objs = [ob for ob, _ in meshed] + list(fields)
    n = max(1, int(samples_per_frame))
    times = np.array([f + s / n for f in range(fs - 1, fe + 1) for s in range(n)], np.float64)

    old_frame, old_sub = scene.frame_current, scene.frame_subframe
    try:
        scene.frame_set(fs)
        dg = bpy.context.evaluated_depsgraph_get()
        min_corner, size = _domain_box(domain, dg, domain_settings.get("dimensions") == 2)
        meshes = [_mesh_arrays(ob, dg) for ob, _ in meshed]
        mats = np.empty((len(objs), len(times), 4, 4), np.float64)
        for j, t in enumerate(times):
            f = int(np.floor(t))
            scene.frame_set(f, subframe=float(t - f))
            for i, ob in enumerate(objs):
                mats[i, j] = np.array(ob.matrix_world, np.float64)
    finally:
        scene.frame_set(old_frame, subframe=old_sub)

    made = []
    nf, no = len(flows), len(flows) + len(obstacles)
    for i, (ob, kw) in enumerate(meshed):
        cls = (LiquidFlow if liquid else MeshFlow) if i < nf else MeshObstacle if i < no else MeshGuide
        (v, t), m = meshes[i], mats[i]
        if len(t) == 0:
            raise ExportError(f"'{ob.name}' has no faces")
        if np.ptp(m, axis=0).max() < STATIC_TOL:
            made.append(cls(v, t, m[0], None, **kw))
        else:
            made.append(cls(v, t, m, times.copy(), **kw))
    nfo = len(meshed)
    made_fields = []
    for j, ob in enumerate(fields):
        m = mats[nfo + j]
        kw = field_kw(ob.field)
        if np.ptp(m, axis=0).max() < STATIC_TOL:
            made_fields.append(ForceField(**kw, matrix=m[0]))
        else:
            made_fields.append(ForceField(**kw, matrix=m, frames=times.copy()))
    settings = (LiquidSettings if liquid else DomainSettings)(
        **domain_settings, min_corner=min_corner, size=size,
        domain_mat3=tuple(np.array(domain.matrix_world, np.float64)[:3, :3].ravel().tolist()), flows=made[:nf],
        obstacles=made[nf:no], guides=made[no:],
        force_fields=made_fields, effector_weights=weights or EffectorWeights())
    write_bake(folder, settings, fs, fe, [ob.name for ob, _ in flows],
               [ob.name for ob, _ in obstacles], [ob.name for ob, _ in guides])
    return settings
