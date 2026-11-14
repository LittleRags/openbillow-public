# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties > Physics panels."""
import bpy
from bpy.types import Panel

from . import bakeexit, export, ops, presets
from .flavor import STORE
from .props import prefs



_PLAN_TEXT = {"gpu": ("Fits in VRAM", "CHECKMARK"),
              "managed": ("Will page through system RAM (about 2x slower)", "INFO"),
              "cpu": ("Will bake on the CPU (much slower)", "INFO"),
              "none": ("Does not fit in VRAM or RAM; lower the resolution", "ERROR")}
_ram = [0.0, 0]  # (time read, free bytes): the panel redraws often, /proc/meminfo is read at most every 2 s


def _ram_free():
    import time
    from .openbillow import estimate
    now = time.monotonic()
    if now - _ram[0] > 2.0:
        _ram[:] = [now, estimate.system_memory()[1]]
    return _ram[1]

class _Panel(Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"
    bl_options = {"DEFAULT_CLOSED"}
    role = "NONE"

    @classmethod
    def poll(cls, context):
        ob = context.object
        return ob is not None and ob.type == "MESH" and ob.openbillow.role == cls.role


class OPENBILLOW_PT_main(Panel):
    bl_label = "OpenBillow"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        ob, lay = context.object, self.layout
        if not ops.warp_available():
            box = lay.box()
            st = {"state": "idle", "msg": ""}
            if not STORE:
                from . import warp_install
                st = warp_install._install
            if st["state"] == "running":
                box.label(text="Downloading GPU engine (about 105 MB)...", icon="TIME")
            else:
                box.label(text="The GPU engine (Warp) is not installed.", icon="ERROR")
                if not STORE:
                    row = box.row()
                    row.enabled = bpy.app.online_access
                    row.operator("openbillow.install_warp", icon="IMPORT")
                    if not bpy.app.online_access:
                        box.label(text="Online access is off (Preferences > System > Network).", icon="ERROR")
                if st["state"] == "error":
                    box.label(text=st["msg"])
            box.label(text="The first bake compiles GPU kernels and takes longer.")
        if ob.openbillow.role == "NONE":
            col = lay.column(align=True)
            for role, label in (("DOMAIN", "Domain"), ("FLOW", "Flow"), ("OBSTACLE", "Obstacle"), ("GUIDE", "Guide")):
                col.operator("openbillow.set_role", text=label).role = role
        else:
            row = lay.row()
            row.prop(ob.openbillow, "role", text="Type")
            row.operator("openbillow.set_role", text="", icon="X").role = "NONE"
        lay.operator("openbillow.report_bug", icon="URL")


class OPENBILLOW_PT_domain(_Panel):
    bl_label = "Domain"
    bl_parent_id = "OPENBILLOW_PT_main"
    role = "DOMAIN"
    bl_options = set()

    def draw(self, context):
        from .openbillow import estimate
        ob = context.object
        d = ob.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        presets.draw_dropdown(lay, d)
        col = lay.column()
        col.prop(d, "domain_type")
        col.prop(d, "resolution_max")
        if d.domain_type == "GAS":
            col.prop(d, "use_2d", text="2D (X-Z plane)")
        size = ops.domain_size(ob)
        est = ops.estimate_args(context.scene, ob)
        shape = estimate.grid_shape(size, d.resolution_max, est.get("dimensions", 3))
        mb = estimate.estimate_total_mb(size, d.resolution_max, **est)
        limit = prefs().limit_mb()
        mode = prefs().device_mode
        over = mode != "CPU" and limit is not None and mb > limit
        plan = None
        if mode == "AUTO" and limit is not None:  # the limit stands in for free VRAM (no Warp here)
            plan = estimate.device_plan(mb * estimate.MB, limit * estimate.MB, _ram_free(),
                                        estimate.managed_ok(),
                                        estimate.host_bytes(size, d.resolution_max, **est))
        row = col.row()
        row.alert = (over and mode == "GPU") or plan == "none"
        mem = "RAM" if mode == "CPU" else "VRAM"
        row.label(text=f"{shape[0]}×{shape[1]}×{shape[2]}  ≈ {mb:.0f} MB {mem}"
                       + (f" (limit {limit:.0f})" if limit is not None and mode != "CPU" else ""))
        if plan is not None:
            text, icon = _PLAN_TEXT[plan]
            r = col.row()
            r.alert = plan == "none"
            r.label(text=text, icon=icon)
        col.prop(d, "use_scene_range")
        if not d.use_scene_range:
            col.prop(d, "frame_start")
            col.prop(d, "frame_end")
        col.prop(d, "cache_dir")
        col.prop(d, "refresh_vdb")
        self._status(col, ob)

    def _status(self, col, ob):
        try:
            folder = ops.bake_folder(ob)
        except Exception as e:  # noqa: BLE001 — shown, not raised, in a draw callback
            col.label(text=str(e), icon="ERROR")
            return
        st = ops.read_status(folder)
        running = ops.is_running(ob)
        row = col.row(align=True)
        row.operator("openbillow.bake")
        row.operator("openbillow.resume")
        row.operator("openbillow.cancel")
        row.operator("openbillow.free")
        last = None if running else ops.last_result(folder)
        if last and bakeexit.failed(last["exit"]) and (
                st is None or st.get("pid") != last["pid"] or st.get("state") == "running"):
            r = col.row()
            r.alert = True
            r.label(text=f"Bake stopped ({bakeexit.describe(last['exit'])}): {last['tail'][:80]}", icon="ERROR")
            return
        if st is None:
            return
        state = st.get("state", "?")
        if state == "running" and not running:
            state = ops.running_elsewhere(folder) or "interrupted"
        f0, f1, f = st.get("frame_start"), st.get("frame_end"), st.get("frame")
        if state == "running" and f0 is not None and f1 and f is not None:
            col.progress(factor=max(0.0, min(1.0, (f - f0 + 1) / (f1 - f0 + 1))),
                         text=f"Frame {f} / {f1}  ETA {st.get('eta_s') or 0:.0f} s")
        else:
            col.label(text=f"{state}, frame {f} / {f1}", icon="ERROR" if state == "error" else "INFO")
        if st.get("message"):
            col.label(text=st["message"][:90])
        if st.get("vram_peak_mb"):
            col.label(text=f"Peak VRAM {st['vram_peak_mb']} MB, {st.get('sec_per_frame') or 0:.2f} s/frame")


class OPENBILLOW_PT_domain_physics(_Panel):
    bl_label = "Physics"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    def draw(self, context):
        d = context.object.openbillow.domain
        liquid = d.domain_type == "LIQUID"
        lay = self.layout
        lay.use_property_split = True
        col = lay.column()
        col.prop(d, "quality")
        for p in ("time_scale", "cfl_condition", "timesteps_min", "timesteps_max") + (
                () if liquid else ("alpha", "beta", "vorticity")) + ("samples_per_frame",):
            col.prop(d, p)
        if not context.scene.use_gravity:
            col.prop(d, "gravity")
        col.prop(d.effector_weights, "gravity", text="Gravity weight")
        if liquid:
            return
        col.prop(d, "use_dissolve_smoke")
        if d.use_dissolve_smoke:
            col.prop(d, "dissolve_speed")
            col.prop(d, "use_dissolve_smoke_log")


class OPENBILLOW_PT_domain_weights(_Panel):
    bl_label = "Effector Weights"
    bl_parent_id = "OPENBILLOW_PT_domain"
    bl_options = {"DEFAULT_CLOSED"}
    role = "DOMAIN"

    def draw(self, context):
        w = context.object.openbillow.domain.effector_weights
        lay = self.layout
        lay.use_property_split = True
        col = lay.column()
        col.label(text="Force fields: Physics tab > Force Fields on any object", icon="INFO")
        for p in ("all", "force", "wind", "vortex", "magnetic", "harmonic", "drag", "turbulence"):
            col.prop(w, p)
        for msg in export.field_warnings(context.scene):
            col.label(text=msg, icon="ERROR")


class _GasPanel(_Panel):
    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.object.openbillow.domain.domain_type == "GAS"


class OPENBILLOW_PT_domain_liquid(_Panel):
    bl_label = "Liquid"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    @classmethod
    def poll(cls, context):
        return super().poll(context) and context.object.openbillow.domain.domain_type == "LIQUID"

    def draw(self, context):
        ob = context.object
        d = ob.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        col = lay.column()
        col.prop(d, "use_apic", text="Method")
        if d.use_apic == "FLIP":
            col.prop(d, "flip_ratio")
        for p in ("particle_number", "particle_minimum", "particle_maximum", "particle_radius",
                  "particle_band_width", "particle_randomness", "use_delete_in_obstacle",
                  "output_particles", "flip_wall_fix", "robust_collisions", "liquid_threshold", "liquid_adaptivity"):
            col.prop(d, p)
        if d.robust_collisions:
            col.prop(d, "collision_margin")
        col.prop(d, "volume_correction")
        if d.volume_correction:
            col.prop(d, "volume_correction_rate")
        col.prop(d, "use_fractions")
        if d.use_fractions:
            col.prop(d, "fractions_threshold")
            col.prop(d, "fractions_distance")
        col = lay.column()
        col.prop(d, "use_viscosity")
        if d.use_viscosity:
            col.prop(d, "viscosity_value")
        col.prop(d, "use_diffusion")
        if d.use_diffusion:
            col.prop(d, "viscosity_base")
            col.prop(d, "viscosity_exponent")
        row = col.row()
        row.active = d.quality == "HIGH"
        row.prop(d, "surface_tension")
        if d.quality == "HIGH" and d.surface_tension > 0.0:
            from .openbillow import estimate
            r = context.scene.render
            n = estimate.surface_tension_substeps(ops.domain_size(ob), d.resolution_max, d.surface_tension,
                                                  r.fps / r.fps_base, d.time_scale, d.timesteps_max)
            if n:
                col.label(text=f"Needs about {n} substeps per frame: much slower bake", icon="INFO")
        col = lay.column(heading="Mesh")
        for p in ("mesh_generator", "mesh_scale", "mesh_particle_radius"):
            col.prop(d, p)
        if d.mesh_generator != "UNION":
            col.prop(d, "mesh_smoothen_pos")
            col.prop(d, "mesh_smoothen_neg")
        if d.mesh_generator == "IMPROVED":
            for p in ("mesh_concave_upper", "mesh_concave_lower"):
                col.prop(d, p)
        if d.mesh_generator != "UNION":
            col.prop(d, "mesh_wall_contact")


class OPENBILLOW_PT_domain_secondary(_Panel):
    bl_label = "Secondary Particles"
    bl_parent_id = "OPENBILLOW_PT_domain_liquid"
    role = "DOMAIN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        d = context.object.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        col = lay.column(heading="Types")
        for p in ("use_spray_particles", "use_foam_particles", "use_bubble_particles"):
            col.prop(d, p)
        if not (d.use_spray_particles or d.use_foam_particles or d.use_bubble_particles):
            return
        col = lay.column()
        for p in ("particle_scale", "sndparticle_boundary", "sndparticle_antitunneling",
                  "sndparticle_sampling", "sndparticle_potential_radius",
                  "sndparticle_sampling_wavecrest", "sndparticle_sampling_trappedair",
                  "sndparticle_life_min", "sndparticle_life_max", "sndparticle_bubble_buoyancy",
                  "sndparticle_bubble_drag", "sndparticle_width"):
            col.prop(d, p)
            if p == "sndparticle_antitunneling" and d.sndparticle_antitunneling == "FIXED":
                col.prop(d, "sndparticle_antitunnel_steps")
        sub = col.column(align=True)
        for p in ("sndparticle_potential_min_wavecrest", "sndparticle_potential_max_wavecrest",
                  "sndparticle_potential_min_trappedair", "sndparticle_potential_max_trappedair",
                  "sndparticle_potential_min_energy", "sndparticle_potential_max_energy"):
            sub.prop(d, p)
        if d.volume_correction:
            box = lay.box()
            box.label(text="Counts are about half of Blender's with", icon="INFO")
            box.label(text="Volume Correction on (raise the sampling or turn it off).")
        lay.operator("openbillow.import_particles", icon="IMPORT")


class OPENBILLOW_PT_domain_fire(_GasPanel):
    bl_label = "Fire"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    def draw(self, context):
        d = context.object.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        col = lay.column()
        col.prop(d, "fire_model")
        for p in ("burning_rate", "flame_smoke", "flame_vorticity", "flame_ignition", "flame_max_temp"):
            col.prop(d, p)
        if d.fire_model == "COMBUSTION":
            box = lay.box()
            box.label(text="Combustion")
            col = box.column()
            for p in ("heat_release", "cooling", "cooling_radiative", "expansion",
                      "flame_kelvin", "ambient_kelvin"):
                col.prop(d, p)
            col = box.column(heading="Disturbance")
            for p in ("disturbance", "disturbance_size", "disturbance_rate", "shredding"):
                col.prop(d, p)
        elif d.fire_model == "FLAME_FRONT":
            box = lay.box()
            box.label(text="Flame Front (experimental)", icon="EXPERIMENTAL")
            col = box.column()
            for p in ("flame_speed", "fuel_kelvin", "flame_inject", "cooling",
                      "flame_kelvin", "ambient_kelvin", "use_viscosity"):
                col.prop(d, p)


class OPENBILLOW_PT_domain_noise(_GasPanel):
    bl_label = "Noise"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    def draw_header(self, context):
        self.layout.enabled = not context.object.openbillow.domain.use_2d
        self.layout.prop(context.object.openbillow.domain, "use_noise", text="")

    def draw(self, context):
        d = context.object.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        lay.active = d.use_noise
        col = lay.column()
        for p in ("noise_upres", "noise_strength", "noise_pos_scale", "noise_time_anim"):
            col.prop(d, p)


class OPENBILLOW_PT_domain_guides(_Panel):
    bl_label = "Guides"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    def draw_header(self, context):
        self.layout.prop(context.object.openbillow.domain, "use_guide", text="")

    def draw(self, context):
        ob = context.object
        d = ob.openbillow.domain
        lay = self.layout
        lay.use_property_split = True
        lay.active = d.use_guide
        col = lay.column()
        for p in ("guide_alpha", "guide_beta", "guide_vel_factor", "guide_source"):
            col.prop(d, p)
        if d.guide_source == "DOMAIN":
            col.prop(d, "guide_parent")
            if d.guide_parent is not None:
                col.label(text=f"Bake '{d.guide_parent.name}' first, then this domain", icon="INFO")
        elif not ops.scene_objects(context.scene, "GUIDE"):
            col.label(text="No Guide objects: the flow is only slowed down", icon="ERROR")
        kids = ops.guide_children(context.scene, ob)
        if kids:
            lay.label(text="Guides " + ", ".join(k.name for k in kids) + " (writes its velocity)", icon="LINKED")


class OPENBILLOW_PT_domain_borders(_Panel):
    bl_label = "Border Collisions"
    bl_parent_id = "OPENBILLOW_PT_domain"
    role = "DOMAIN"

    def draw(self, context):
        d = context.object.openbillow.domain
        col = self.layout.column(align=True)
        for s in ("front", "back", "left", "right", "top", "bottom"):
            col.prop(d, "use_collision_border_" + s)


class OPENBILLOW_PT_flow(_Panel):
    bl_label = "Flow"
    bl_parent_id = "OPENBILLOW_PT_main"
    role = "FLOW"
    bl_options = set()

    def draw(self, context):
        f = context.object.openbillow.flow
        lay = self.layout
        lay.use_property_split = True
        if ops.domain_type_in_scene(context.scene) == "LIQUID":
            lay.prop(f, "behavior")
            lay.prop(f, "liquid_surface_distance")
            lay.prop(f, "use_plane_init")
            lay.prop(f, "use_initial_velocity")
            if f.use_initial_velocity:
                col = lay.column()
                col.prop(f, "velocity_factor")
                col.prop(f, "velocity_normal")
                col.prop(f, "velocity_coord")
            if f.behavior == "INFLOW":
                lay.prop(f, "use_emit_range")
                if f.use_emit_range:
                    col = lay.column(align=True)
                    col.prop(f, "emit_start")
                    col.prop(f, "emit_end")
            return
        lay.prop(f, "flow_type")
        if f.flow_type != "SMOKE":
            lay.prop(f, "fuel_amount")
        for p in ("density", "temperature", "surface_distance", "volume_density", "use_plane_init"):
            lay.prop(f, p)
        lay.prop(f, "use_emit_range")
        if f.use_emit_range:
            col = lay.column(align=True)
            col.prop(f, "emit_start")
            col.prop(f, "emit_end")


class OPENBILLOW_PT_obstacle(_Panel):
    bl_label = "Obstacle"
    bl_parent_id = "OPENBILLOW_PT_main"
    role = "OBSTACLE"
    bl_options = set()

    def draw(self, context):
        e = context.object.openbillow.obstacle
        lay = self.layout
        lay.use_property_split = True
        lay.prop(e, "surface_distance")
        lay.prop(e, "use_plane_init")


class OPENBILLOW_PT_guide(_Panel):
    bl_label = "Guide"
    bl_parent_id = "OPENBILLOW_PT_main"
    role = "GUIDE"
    bl_options = set()

    def draw(self, context):
        g = context.object.openbillow.guide
        lay = self.layout
        lay.use_property_split = True
        for p in ("guide_mode", "velocity_factor", "surface_distance", "use_plane_init"):
            lay.prop(g, p)
        lay.label(text="Used by domains with Guides on, source Effector", icon="INFO")


_classes = (OPENBILLOW_PT_main, OPENBILLOW_PT_domain, OPENBILLOW_PT_domain_physics, OPENBILLOW_PT_domain_weights,
            OPENBILLOW_PT_domain_liquid, OPENBILLOW_PT_domain_secondary, OPENBILLOW_PT_domain_fire, OPENBILLOW_PT_domain_noise,
            OPENBILLOW_PT_domain_guides, OPENBILLOW_PT_domain_borders, OPENBILLOW_PT_flow, OPENBILLOW_PT_obstacle,
            OPENBILLOW_PT_guide)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
