# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators, the bake process and the status poll timer. The add-on never
imports Warp: it exports a bake folder and runs openbillow_bake.py in a subprocess
(Blender's own Python). Progress comes from <folder>/status.json.
"""
import importlib.util
import json
import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time

import bpy
from bpy.types import Operator

from . import errlog, export
from . import ROOT
from .props import prefs

POLL_S = 0.5
CANCEL_KILL_S = 15.0
_runs = {}  # domain object name -> {"proc", "folder", "cancel_at", "last_vdb"}


# ---- scene helpers ---------------------------------------------------------
def domain_of(context):
    ob = context.object
    if ob and ob.openbillow.role == "DOMAIN":
        return ob
    return None


def scene_objects(scene, role):
    return [o for o in scene.objects if o.type == "MESH" and o.openbillow.role == role]


def bake_folder(ob):
    d = ob.openbillow.domain.cache_dir.strip()
    if not d:
        d = os.path.join(prefs().cache_root or "//openbillow_cache", ob.name)
    if d.startswith("//") and not bpy.data.filepath:
        raise export.ExportError("Save the .blend first, or set an absolute cache folder")
    return os.path.normpath(bpy.path.abspath(d))


def _moves(ob):
    """True if the object is animated (transform keyframes, drivers or a parent chain with them)."""
    while ob is not None:
        ad = ob.animation_data
        if ad and (ad.action or ad.drivers):
            return True
        ob = ob.parent
    return False


def frame_range(scene, ds):
    if ds.use_scene_range:
        return scene.frame_start, scene.frame_end
    return ds.frame_start, ds.frame_end


def domain_size(ob):
    return tuple(max(1e-6, d) for d in ob.dimensions)


def domain_type_in_scene(scene):
    """Type of the first domain object in the scene (flows follow it); GAS if none."""
    for o in scene_objects(scene, "DOMAIN"):
        return o.openbillow.domain.domain_type
    return "GAS"


def scene_fields(scene):
    """Objects with a force field the solver handles."""
    return [o for o in scene.objects if o.field is not None and o.field.type != "NONE"
            and export.field_supported(o)]


def estimate_args(scene, ob):
    """Keyword extras for estimate.* for domain `ob` and the scene's flows/obstacles."""
    from .openbillow import estimate
    d = ob.openbillow.domain
    obs = scene_objects(scene, "OBSTACLE")
    kw = dict(has_obstacles=bool(obs), moving_obstacles=any(_moves(o) for o in obs),
              force_fields=bool(scene_fields(scene)))
    flows = scene_objects(scene, "FLOW")
    if d.domain_type == "LIQUID":
        vol = sum(estimate.mesh_volume([tuple(v.co) for v in o.data.vertices],
                                       [tuple(p.vertices) for p in _tris(o)], o.matrix_world)
                  for o in flows if o.openbillow.flow.behavior == "GEOMETRY")
        dx, dy, dz = domain_size(ob)
        return {**kw, "liquid": True, "apic": d.use_apic == "APIC",
                "particles_per_cell": estimate.liquid_particles_per_cell(
                    vol / (dx * dy * dz), d.particle_number), "mesh_scale": d.mesh_scale,
                "secondary": d.use_spray_particles or d.use_foam_particles or d.use_bubble_particles,
                "particle_scale": d.particle_scale, "viscosity": d.use_viscosity,
                "diffusion": d.use_diffusion, "fractions": d.use_fractions, "guiding": d.use_guide}
    return {**kw, "fire": any(o.openbillow.flow.flow_type != "SMOKE" for o in flows),
            "noise_upres": d.noise_upres if d.use_noise else 0, "dimensions": 2 if d.use_2d else 3,
            "guiding": d.use_guide, "combustion": d.fire_model == "COMBUSTION",
            "high_quality": d.quality == "HIGH"}


def guide_children(scene, ob):
    """Domains that use `ob` as their guide parent (source Domain)."""
    return [o for o in scene_objects(scene, "DOMAIN") if o != ob and o.openbillow.domain.use_guide
            and o.openbillow.domain.guide_source == "DOMAIN" and o.openbillow.domain.guide_parent == ob]


def guide_kw(scene, ob, f0, f1):
    """guide_parent (the parent's bake folder) and write_guide for domain `ob`.
    The parent must already be baked with this domain naming it, since a domain
    writes its guide velocity only when another domain names it as parent."""
    from .openbillow.guiding import guide_frame_path
    ds = ob.openbillow.domain
    kw = {"write_guide": bool(guide_children(scene, ob))}
    if ds.use_guide and ds.guide_source == "DOMAIN":
        par = ds.guide_parent
        if par is None or par.openbillow.role != "DOMAIN":
            raise export.ExportError("Guides: set a Guide Parent domain, or use Velocity Source Effector")
        folder = bake_folder(par)
        if not all(os.path.isfile(guide_frame_path(folder, f)) for f in range(f0, f1 + 1)):
            raise export.ExportError(f"Guides: bake the guide parent '{par.name}' first (frames {f0}-{f1}; "
                                     "rebake it if it was baked before it became a parent)")
        kw["guide_parent"] = folder
    return kw


def _tris(ob):
    me = ob.data
    me.calc_loop_triangles()
    return me.loop_triangles


def read_status(folder):
    try:
        with open(os.path.join(folder, "status.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def deps_dir():
    """Where Warp is installed: the extension's user folder (survives add-on updates),
    or ROOT/deps in a dev checkout."""
    pkg = __package__ or ""
    if pkg.startswith("bl_ext."):
        return bpy.utils.extension_path_user(pkg, path="deps", create=True)
    return os.path.join(ROOT, "deps")


def _use_deps():
    d = deps_dir()
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)
    importlib.invalidate_caches()


def warp_available():
    _use_deps()
    return importlib.util.find_spec("warp") is not None


def _cli(*args):
    """Command line and env for openbillow_bake.py. -I hides the site-packages folder
    Blender installs the extension's wheels into, so pass it in OPENBILLOW_DEPS
    (find_spec locates warp without importing it)."""
    env = dict(os.environ)
    spec = importlib.util.find_spec("warp")
    if spec and spec.submodule_search_locations:
        env["OPENBILLOW_DEPS"] = os.path.dirname(list(spec.submodule_search_locations)[0])
    return [sys.executable, "-I", os.path.join(ROOT, "openbillow_bake.py"), *args], env


def run_cli(*args):
    """Run a CLI subcommand, return the last stdout line parsed as JSON (Warp prints a banner)."""
    cmd, env = _cli(*args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if r.returncode:
        raise RuntimeError((r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}").splitlines()[-1])
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---- Volume object ---------------------------------------------------------
def update_volume(ob, folder, status):
    """Create/refresh the `<domain>_OpenBillow` Volume pointing at the VDB sequence."""
    last = status.get("last_vdb") if status else None
    f0, f1 = (status.get("frame_start"), status.get("frame")) if status else (None, None)
    if not last or f0 is None or f1 is None or f1 < f0:
        return
    name = ob.name + "_OpenBillow"
    liquid = ob.openbillow.domain.domain_type == "LIQUID"
    vo = bpy.data.objects.get(name)
    if vo is None or vo.type != "VOLUME":
        vol = bpy.data.volumes.new(name)
        vo = bpy.data.objects.new(name, vol)
        for c in ob.users_collection or [bpy.context.scene.collection]:
            c.objects.link(vo)
        if not liquid:
            vol.materials.append(ob.openbillow.domain.material or
                             _volume_material(ob.openbillow.domain.fire_model == "COMBUSTION"))
    vol = vo.data
    first = os.path.join(folder, "vdb", "openbillow_%04d.vdb" % f0)
    if bpy.data.filepath:
        try:
            first = bpy.path.relpath(first)
        except ValueError:  # other drive
            pass
    if vol.filepath != first:
        vol.filepath = first
        vol.is_sequence = True
        vol.frame_start = f0
        vol.frame_offset = 0
    # Hold the first/last frame outside the baked range (default CLIP leaves the Volume with no
    # grids there, and Volume to Mesh then warns "Cannot find 'phi' grid").
    if vol.sequence_mode != "EXTEND":
        vol.sequence_mode = "EXTEND"
    n = f1 - f0 + 1
    if vol.frame_duration != n:
        vol.frame_duration = n
    vo.update_tag()
    if liquid:
        _liquid_mesh(ob, vo)


def refresh_volume(ob, now=False):
    """Drop the Volume's cached grids so Blender re-reads the VDB files (stale after a re-bake).
    Blender reloads and rebuilds (liquid: the mesh too) in one blocking step with no progress API,
    so by default show a status-bar message first and do the work on the next timer tick."""
    name = ob.name

    def work():
        wm = bpy.context.window_manager
        o = bpy.data.objects.get(name + "_OpenBillow")
        if o is not None and o.type == "VOLUME":
            o.data.grids.unload()
            o.data.update_tag()
            o.update_tag()
        wm.progress_end()
        return None

    if bpy.data.objects.get(name + "_OpenBillow") is None:
        return
    if now:
        work()
        return
    wm = bpy.context.window_manager
    wm.progress_begin(0, 1)
    wm.progress_update(0.5)  # half-full bar = busy
    bpy.context.workspace.status_text_set("OpenBillow: refreshing VDB...")
    bpy.app.timers.register(lambda: (work(), bpy.context.workspace.status_text_set(None))[1], first_interval=0.3)


def _liquid_mesh(ob, vo):
    """`<domain>_Liquid` mesh object: Volume to Mesh on the `phi` level set (threshold 0 = the
    surface; Blender uses the threshold as a plain iso value). The Volume itself is hidden with
    the eye icon (not the monitor icon, which greys it out where the user cannot undo it)."""
    d = ob.openbillow.domain
    mo = bpy.data.objects.get(ob.name + "_Liquid")
    new = mo is None or mo.type != "MESH"
    if new:
        mo = bpy.data.objects.new(ob.name + "_Liquid", bpy.data.meshes.new(ob.name + "_Liquid"))
        for c in ob.users_collection or [bpy.context.scene.collection]:
            c.objects.link(mo)
        mo.data.materials.append(ob.openbillow.domain.material or _liquid_material())
    if new or vo.hide_viewport:  # hide_viewport: files from 0.5.0 and older
        vo.hide_viewport = False
        vo.hide_render = True
        try:
            vo.hide_set(True)
        except RuntimeError:  # not in the active view layer
            pass
    mod = mo.modifiers.get("OpenBillow Liquid")
    if mod is None:
        mod = mo.modifiers.new("OpenBillow Liquid", "VOLUME_TO_MESH")
        mod.grid_name = "phi"
        mod.resolution_mode = "GRID"
    if mod.object != vo:  # also repairs a link lost when the Volume was deleted or replaced
        mod.object = vo
    mod.threshold = d.liquid_threshold
    mod.adaptivity = d.liquid_adaptivity


def _liquid_material():
    m = bpy.data.materials.get("OpenBillow Liquid")
    if m:
        return m
    m = bpy.data.materials.new("OpenBillow Liquid")
    m.use_nodes = True
    n = m.node_tree.nodes.get("Principled BSDF")
    if n:  # clear water-like glass
        n.inputs["Base Color"].default_value = (0.8, 0.9, 1.0, 1.0)
        n.inputs["Roughness"].default_value = 0.0
        n.inputs["Transmission Weight"].default_value = 1.0
        n.inputs["IOR"].default_value = 1.33
    return m


def _material(name, smoke, flame, kelvin=False):
    """Volume material from the bake's `density` and `flame` grids (`temperature` in K when `kelvin`,
    the Combustion model). Users can replace it."""
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)

    def attr(n, y):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = n
        a.location = (-800, y)
        return a

    def math(op, a, b, x, y):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation, n.location = op, (x, y)
        n.inputs[1].default_value = b
        nt.links.new(a, n.inputs[0])
        return n.outputs[0]

    shaders = []
    if smoke:  # dark grey smoke driven by density
        pv = nt.nodes.new("ShaderNodeVolumePrincipled")
        pv.location = (300, 300)
        pv.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
        nt.links.new(math("MULTIPLY", attr("density", 300).outputs["Fac"], 5.0, -400, 300),
                     pv.inputs["Density"])
        shaders.append(pv.outputs["Volume"])
    if flame:  # emission only: blackbody colour from the flame grid, no density of its own
        pv = nt.nodes.new("ShaderNodeVolumePrincipled")
        pv.location = (300, -300)
        pv.inputs["Density"].default_value = 0.0
        pv.inputs["Blackbody Intensity"].default_value = 1.0
        if kelvin:  # Combustion: the grid is real Kelvin (0 where cold)
            # Colour from a Blackbody node at 1.25 x T (soot at 1800 K reads too red on screen);
            # strength ramps 900..1800 K -> 0..1, cubed, x20. Principled's own blackbody term is off.
            pv.inputs["Blackbody Intensity"].default_value = 0.0
            tk = attr("temperature", -300).outputs["Fac"]
            bb = nt.nodes.new("ShaderNodeBlackbody")
            bb.location = (0, -150)
            nt.links.new(math("MULTIPLY", tk, 1.25, -400, -100), bb.inputs["Temperature"])
            nt.links.new(bb.outputs["Color"], pv.inputs["Emission Color"])
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.location = (-400, -450)
            mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 900.0, 1800.0
            nt.links.new(tk, mr.inputs["Value"])
            nt.links.new(math("MULTIPLY", math("POWER", mr.outputs["Result"], 3.0, -200, -450), 20.0, 0, -450),
                         pv.inputs["Emission Strength"])
            shaders.append(pv.outputs["Volume"])
        else:
            fl = attr("flame", -300).outputs["Fac"]
            mr = nt.nodes.new("ShaderNodeMapRange")  # flame 0..1 -> 800..1800 K: red to yellow
            mr.location = (-400, -450)
            mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 800.0, 1800.0
            nt.links.new(fl, mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], pv.inputs["Temperature"])
            pv.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            nt.links.new(math("MULTIPLY", fl, 0.1, -400, -250), pv.inputs["Emission Strength"])
            shaders.append(pv.outputs["Volume"])
    if len(shaders) == 2:
        add = nt.nodes.new("ShaderNodeAddShader")
        add.location = (600, 0)
        nt.links.new(shaders[0], add.inputs[0])
        nt.links.new(shaders[1], add.inputs[1])
        shaders = [add.outputs["Shader"]]
    nt.links.new(shaders[0], out.inputs["Volume"])
    return m


def _volume_material(combustion=False):
    """Smoke + flame; also creates the separate Smoke and Flame materials to pick from.
    `combustion` reads the Kelvin `temperature` grid instead of `flame`."""
    if combustion:
        _material("OpenBillow Combustion Flame", False, True, kelvin=True)
        return _material("OpenBillow Combustion Fire", True, True, kelvin=True)
    _material("OpenBillow Smoke", True, False)
    _material("OpenBillow Flame", False, True)
    return _material("OpenBillow Fire", True, True)


# ---- process + poll --------------------------------------------------------
def is_running(ob):
    r = _runs.get(ob.name)
    return bool(r and r["proc"].poll() is None)


def _poll():
    now = time.time()
    for name, r in list(_runs.items()):
        ob = bpy.data.objects.get(name)
        code = r["proc"].poll()
        if code is None and r["cancel_at"] and now - r["cancel_at"] > CANCEL_KILL_S:
            r["proc"].terminate()
            r["cancel_at"] = None
        st = read_status(r["folder"])
        if ob and st and (st.get("last_vdb") != r["last_vdb"] or code is not None):
            r["last_vdb"] = st.get("last_vdb")
            update_volume(ob, r["folder"], st)
        if code is not None:
            if ob and ob.openbillow.domain.refresh_vdb:
                refresh_volume(ob)
            r["exit"] = code
            del _runs[name]
            _redraw()
    _redraw()
    return POLL_S if _runs else None


def _redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "PROPERTIES":
                a.tag_redraw()


def _start_timer():
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_S)


def _launch(ob, folder, resume):
    args = ["bake", folder]
    if resume:
        args.append("--resume")
    args += ["--device", prefs().device_arg()]
    # The CLI writes bake.log itself; keep its stdout out of that file.
    out = open(os.path.join(folder, "bake.stdout"), "w")
    cmd, env = _cli(*args)
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    out.close()
    _runs[ob.name] = {"proc": proc, "folder": folder, "cancel_at": None, "last_vdb": None}
    _start_timer()


# ---- operators -------------------------------------------------------------
class _DomainOp(Operator):
    @classmethod
    def poll(cls, context):
        return domain_of(context) is not None


@contextlib.contextmanager
def _results_unevaluated(dom):
    """The export calls scene.frame_set for every sub-frame; each one re-evaluates the result
    objects (Volume, Volume to Mesh, user node groups on top: ~25 ms each on a heavy scene, ~1000
    calls). Take them out of the depsgraph meanwhile."""
    objs = [o for k in ("OpenBillow", "Liquid", "Spray", "Foam", "Bubble")
            if (o := bpy.data.objects.get(f"{dom.name}_{k}")) is not None and o.openbillow.role == "NONE"]
    state = [o.hide_viewport for o in objs]
    for o in objs:
        o.hide_viewport = True
    try:
        yield
    finally:
        for o, h in zip(objs, state):
            o.hide_viewport = h


class _BakeBase(_DomainOp):
    resume = False

    @classmethod
    def poll(cls, context):
        ob = domain_of(context)
        return ob is not None and not is_running(ob)

    def execute(self, context):
        from .openbillow import estimate
        sc, dom = context.scene, domain_of(context)
        ds = dom.openbillow.domain
        liquid = ds.domain_type == "LIQUID"
        flows, obstacles = scene_objects(sc, "FLOW"), scene_objects(sc, "OBSTACLE")
        est = estimate_args(sc, dom)
        try:
            folder = bake_folder(dom)
            limit = prefs().limit_mb()
            total = estimate.estimate_total_mb(domain_size(dom), ds.resolution_max, **est)
            if prefs().device_mode == "GPU" and limit is not None and total > limit:
                fit = estimate.max_resolution_for(limit, domain_size(dom), **est)
                raise export.ExportError(f"Estimated {total:.0f} MB exceeds the {limit:.0f} MB limit; "
                                         f"largest resolution that fits: {fit}")
            if not flows:
                raise export.ExportError("No Flow objects in the scene")
            f0, f1 = frame_range(sc, ds)
            dkw = (export.liquid_domain_kw if liquid else export.domain_kw)(sc, ds)
            dkw.update(guide_kw(sc, dom, f0, f1))
            guides = scene_objects(sc, "GUIDE") if ds.use_guide and ds.guide_source == "EFFECTOR" else []
            with _results_unevaluated(dom):
                export.export_bake(
                    folder, sc, dom,
                    [(o, (export.liquid_flow_kw if liquid else export.flow_kw)(o.openbillow.flow)) for o in flows],
                    [(o, export.obstacle_kw(o.openbillow.obstacle)) for o in obstacles],
                    dkw, f0, f1, ds.samples_per_frame, liquid, scene_fields(sc),
                    export.weights_kw(ds.effector_weights),
                    [(o, export.guide_kw(o.openbillow.guide)) for o in guides])
            _launch(dom, folder, self.resume)
        except export.ExportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except OSError as e:
            self.report({"ERROR"}, f"Cannot start the bake: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class OPENBILLOW_OT_bake(_BakeBase):
    bl_idname = "openbillow.bake"
    bl_label = "Bake"
    bl_description = "Start baking the fluid on your graphics card. Blender stays usable while it runs."


class OPENBILLOW_OT_resume(_BakeBase):
    bl_idname = "openbillow.resume"
    bl_label = "Resume"
    bl_description = "Carry on from where the bake stopped, or extend a finished bake after you raise the end frame."
    resume = True


class OPENBILLOW_OT_cancel(_DomainOp):
    bl_idname = "openbillow.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the bake after the frame it is on. You can continue later with Resume."

    @classmethod
    def poll(cls, context):
        ob = domain_of(context)
        return ob is not None and is_running(ob)

    def execute(self, context):
        r = _runs[context.object.name]
        open(os.path.join(r["folder"], "CANCEL"), "w").close()
        r["cancel_at"] = time.time()
        return {"FINISHED"}


class OPENBILLOW_OT_free(_DomainOp):
    bl_idname = "openbillow.free"
    bl_label = "Free Bake"
    bl_description = "Delete everything this domain has baked so far, so you can start again from scratch."

    @classmethod
    def poll(cls, context):
        ob = domain_of(context)
        return ob is not None and not is_running(ob)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            folder = bake_folder(context.object)
        except export.ExportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        for kind in ("spray", "foam", "bubble"):
            old = bpy.data.objects.get(_snd_name(context.object, kind))
            if old:
                bpy.data.objects.remove(old)
        for sub in ("vdb", "state"):
            shutil.rmtree(os.path.join(folder, sub), ignore_errors=True)
        for f in ("status.json", "bake.log", "bake.stdout", "CANCEL"):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
        vo = bpy.data.objects.get(context.object.name + "_OpenBillow")
        if vo and vo.type == "VOLUME":
            vol = vo.data
            bpy.data.objects.remove(vo)
            if vol.users == 0:
                bpy.data.volumes.remove(vol)
        return {"FINISHED"}


_SND_COLORS = {"spray": (0.9, 0.95, 1.0, 0.6), "foam": (0.95, 0.97, 1.0, 0.9), "bubble": (0.7, 0.85, 1.0, 0.3)}


def _snd_material(kind):
    name = "OpenBillow " + kind.title()
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            r, g, b, a = _SND_COLORS[kind]
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.2
            bsdf.inputs["Alpha"].default_value = a
    return m


def _snd_name(ob, kind):
    return "%s_%s" % (ob.name, kind.title())


class OPENBILLOW_OT_import_particles(_DomainOp):
    bl_idname = "openbillow.import_particles"
    bl_label = "Import Particles"
    bl_description = "Bring the baked spray, foam and bubbles into the scene as objects you can render."

    @classmethod
    def poll(cls, context):
        ob = domain_of(context)
        return ob is not None and not is_running(ob) and ob.openbillow.domain.domain_type == "LIQUID"

    def execute(self, context):
        dom = context.object
        try:
            path = os.path.join(bake_folder(dom), "vdb", "particles_snd.usdc")
        except export.ExportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if not os.path.exists(path):
            self.report({"ERROR"}, "No secondary particles baked yet (switch a type on and bake)")
            return {"CANCELLED"}
        for kind in ("spray", "foam", "bubble"):  # replace an earlier import
            old = bpy.data.objects.get(_snd_name(dom, kind))
            if old:
                bpy.data.objects.remove(old)
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.usd_import(filepath=path, import_materials=False, import_points=True,
                                  import_meshes=False, import_volumes=False, set_frame_range=False,
                                  create_collection=False)
        except RuntimeError as e:
            self.report({"ERROR"}, "USD import failed: %s" % e)
            return {"CANCELLED"}
        for o in set(bpy.data.objects) - before:
            kind = o.name.split(".")[0].lower()
            if kind not in _SND_COLORS:
                continue
            o.name = _snd_name(dom, kind)
            if o.data is not None:
                o.data.materials.clear()
                o.data.materials.append(_snd_material(kind))
            for c in list(o.users_collection):
                c.objects.unlink(o)
            for c in dom.users_collection or [context.scene.collection]:
                c.objects.link(o)
        return {"FINISHED"}


class OPENBILLOW_OT_detect_gpu(Operator):
    bl_idname = "openbillow.detect_gpu"
    bl_label = "Detect GPU"
    bl_description = "Find your graphics card and how much memory it has, so OpenBillow can warn you before a bake that will not fit."

    def execute(self, context):
        p = prefs()
        try:
            res = run_cli("devices")
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
            self.report({"ERROR"}, f"Device query failed: {e}")
            return {"CANCELLED"}
        if not res.get("warp_installed", True):
            self.report({"ERROR"}, "GPU engine (Warp) not installed - click Install GPU Engine")
            return {"CANCELLED"}
        devs = res["devices"]
        if not devs:
            self.report({"ERROR"}, "No CUDA device found")
            return {"CANCELLED"}
        p.device_total_mb, p.device_name = devs[0]["total_mb"], devs[0]["name"]
        return {"FINISHED"}


class OPENBILLOW_OT_set_role(Operator):
    bl_idname = "openbillow.set_role"
    bl_label = "Set OpenBillow Type"
    bl_options = {"UNDO"}
    role: bpy.props.StringProperty()

    def execute(self, context):
        context.object.openbillow.role = self.role
        return {"FINISHED"}


_classes = (OPENBILLOW_OT_bake, OPENBILLOW_OT_resume, OPENBILLOW_OT_cancel, OPENBILLOW_OT_free,
            OPENBILLOW_OT_detect_gpu, OPENBILLOW_OT_set_role, OPENBILLOW_OT_import_particles)


def register():
    _use_deps()
    for c in _classes:
        bpy.utils.register_class(c)
    errlog.install()


def unregister():
    errlog.uninstall()
    for r in _runs.values():
        r["proc"].terminate()
    _runs.clear()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
