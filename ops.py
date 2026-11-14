# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators, the bake process and the status poll timer. The add-on never
imports Warp: it exports a bake folder and runs openbillow_bake.py in a subprocess
(Blender's own Python). Progress comes from <folder>/status.json.
"""
import importlib.util
import json
import contextlib
import os
import subprocess
import sys
import threading
import time
import traceback

import bpy
from bpy.props import IntProperty, StringProperty
from bpy.types import Operator

from . import bakeexit, errlog, export
from . import ROOT
from .openbillow import bakefiles
from .props import RADIUS_HIGH_QUALITY, RADIUS_OTHER, prefs

POLL_S = 0.5
CANCEL_KILL_S = 15.0
_runs = {}  # normalised bake folder -> {"proc", "folder", "cancel_at", "last_vdb", "uid"}
_last = {}  # bake folder -> {"exit", "pid", "tail"} of the last finished run


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


_moves = export.may_move


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


_vol_cache = {}  # object name -> (key, volume)


def _flow_volume(o, exact=False):
    """Volume of a flow mesh; cached on (counts, world matrix) so panel redraws stay cheap.
    A sculpt that keeps the vertex count is missed (fine for an estimate; bakes pass exact)."""
    import numpy as np
    from .openbillow import estimate
    me = o.data
    key = (len(me.vertices), len(me.polygons), tuple(round(x, 5) for row in o.matrix_world for x in row))
    hit = _vol_cache.get(o.name)
    if hit and hit[0] == key and not exact:
        return hit[1]
    v = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", v)
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    vol = estimate.mesh_volume(v.reshape(-1, 3), t.reshape(-1, 3), o.matrix_world)
    _vol_cache[o.name] = (key, vol)
    return vol


def estimate_args(scene, ob, exact=False):
    """Keyword extras for estimate.* for domain `ob` and the scene's flows/obstacles."""
    from .openbillow import estimate
    d = ob.openbillow.domain
    obs = scene_objects(scene, "OBSTACLE")
    kw = dict(has_obstacles=bool(obs), moving_obstacles=any(_moves(o) for o in obs),
              force_fields=bool(scene_fields(scene)))
    flows = scene_objects(scene, "FLOW")
    if d.domain_type == "LIQUID":
        vol = sum(_flow_volume(o, exact) for o in flows if o.openbillow.flow.behavior == "GEOMETRY")
        dx, dy, dz = domain_size(ob)
        return {**kw, "liquid": True, "apic": d.use_apic == "APIC",
                "particles_per_cell": estimate.liquid_particles_per_cell(
                    vol / (dx * dy * dz), d.particle_number), "mesh_scale": d.mesh_scale,
                "secondary": d.use_spray_particles or d.use_foam_particles or d.use_bubble_particles,
                "particle_scale": d.particle_scale, "viscosity": d.use_viscosity,
                "diffusion": d.use_diffusion, "fractions": d.use_fractions, "guiding": d.use_guide,
                "snd_multiple": d.sndparticle_sampling == "MULTIPLE" and d.quality != "CLASSIC",
                "aniso_mesh": d.mesh_generator == "ANISOTROPIC",
                "wall_contact": d.mesh_wall_contact and d.mesh_generator != "UNION"}
    return {**kw, "fire": any(o.openbillow.flow.flow_type != "SMOKE" for o in flows),
            "noise_upres": d.noise_upres if d.use_noise else 0, "dimensions": 2 if d.use_2d else 3,
            "guiding": d.use_guide, "combustion": d.fire_model in ("COMBUSTION", "FLAME_FRONT"),
            "flame_front": d.fire_model == "FLAME_FRONT",
            "gas_viscosity": d.fire_model == "FLAME_FRONT" and d.use_viscosity, "high_quality": d.quality == "HIGH"}


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
    global _warp_ok
    _warp_ok = importlib.util.find_spec("warp") is not None


_warp_ok = None


def warp_available():
    """Cached; recomputed by _use_deps (register, install finished, Detect GPU). Draw-safe."""
    if _warp_ok is None:
        _use_deps()
    return _warp_ok


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
                             _volume_material(ob.openbillow.domain.fire_model in ("COMBUSTION", "FLAME_FRONT")))
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
def _run_key(folder):
    return os.path.normcase(os.path.abspath(folder))


def _domain_key(ob):
    """The _runs key for this domain's cache folder, or None if it has no usable folder."""
    try:
        return _run_key(bake_folder(ob))
    except export.ExportError:
        return None


def _run_of(ob):
    """The live run of this domain, or None. Found by cache folder; failing that by object
    identity, because renaming a domain moves its default cache folder while the bake carries on
    in the old one."""
    r = _runs.get(_domain_key(ob)) or next((x for x in _runs.values() if x["uid"] == ob.session_uid), None)
    return r if r and r["proc"].poll() is None else None


def is_running(ob):
    return _run_of(ob) is not None


def _run_domain(r):
    """The domain in the open file that this run belongs to, or None (another file is open).
    After a file load session_uid has changed, so a domain using the same folder counts."""
    key, same_folder = _run_key(r["folder"]), None
    for ob in bpy.data.objects:
        if ob.openbillow.role != "DOMAIN":
            continue
        if ob.session_uid == r["uid"]:
            return ob
        if same_folder is None and _domain_key(ob) == key:
            same_folder = ob
    return same_folder


def _lock_error(folder):
    """Why this folder must not be touched now, or None."""
    lock = bakefiles.folder_lock(folder)
    if not (lock and lock["alive"]):
        return None
    if lock["pid"] is None:
        return "This cache folder's bake.lock is empty or damaged, so a bake may be starting. Try again in a minute."
    text = f"This cache folder is being baked by pid {lock['pid']} on {lock['host']}. Wait for it, or cancel it there."
    if not lock["local"]:
        text += f" If no bake is running on {lock['host']}, delete bake.lock in that folder."
    return text


def running_elsewhere(folder):
    """Status text if another process holds this folder's lock, else None."""
    lock = bakefiles.folder_lock(folder)
    if not (lock and lock["alive"]):
        return None
    return f"baking in another process (pid {lock['pid']} on {lock['host']})" if lock["pid"] is not None \
        else "baking in another process"


def last_result(folder):
    return _last.get(folder)


def _sync_scene(r, st, finished):
    ob = _run_domain(r)
    if ob is None:
        return
    if st:
        update_volume(ob, r["folder"], st)
    if finished and ob.openbillow.domain.refresh_vdb:
        refresh_volume(ob)


def _poll():
    now = time.time()
    for key, r in list(_runs.items()):
        code = r["proc"].poll()
        if code is None and r["cancel_at"] and now - r["cancel_at"] > CANCEL_KILL_S:
            if r.get("terminated"):  # stuck in native code: SIGTERM is only handled between Python calls
                r["proc"].kill()
                r["cancel_at"] = None
            else:
                r["proc"].terminate()
                r["terminated"] = True
                r["cancel_at"] = now
        st = read_status(r["folder"])
        if code is not None or (st and st.get("last_vdb") != r["last_vdb"]):
            r["last_vdb"] = st.get("last_vdb") if st else None
            try:
                _sync_scene(r, st, code is not None)
            except Exception:  # noqa: BLE001 - a scene problem must not end monitoring of the process
                errlog.record("ERROR", traceback.format_exc())
        if code is not None:
            res = bakeexit.result(r["folder"], code, r["proc"].pid)
            _last[r["folder"]] = res
            if bakeexit.failed(code):
                errlog.record("BAKE", f"exit {code}: {res['tail']}")
            del _runs[key]
            _redraw()
    _redraw()
    return POLL_S if _runs else None


def wait_for_bakes(timeout_s, cancel_on_timeout=True):
    """Background-mode helper for scripts: poll until no bake runs. -> True if all finished.

    On timeout the bakes are cancelled (CANCEL file, then terminate after CANCEL_KILL_S, then kill)
    so no solver is left holding GPU memory; the result is then False.
    """
    t0 = time.time()
    while _runs and time.time() - t0 < timeout_s:
        _poll()
        time.sleep(POLL_S)
    if not _runs:
        return True
    if cancel_on_timeout:
        for r in _runs.values():
            with contextlib.suppress(OSError):
                open(os.path.join(r["folder"], "CANCEL"), "w").close()
            r["cancel_at"] = time.time()  # _poll terminates after CANCEL_KILL_S, then kills after another
        t1 = time.time()
        while _runs and time.time() - t1 < 2 * CANCEL_KILL_S + 10:
            _poll()
            time.sleep(POLL_S)
        for r in list(_runs.values()):
            r["proc"].kill()
        while _runs:
            _poll()
            time.sleep(POLL_S)
    return False


def _redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "PROPERTIES":
                a.tag_redraw()


def _start_timer():
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_S, persistent=True)  # survives a file load


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
    _last.pop(folder, None)
    _runs[_run_key(folder)] = {"proc": proc, "folder": folder, "cancel_at": None, "last_vdb": None,
                               "uid": ob.session_uid}
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
        est = estimate_args(sc, dom, exact=True)
        try:
            folder = bake_folder(dom)
            if err := _lock_error(folder):
                raise export.ExportError(err)
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
        r = _run_of(context.object)
        if r is None:  # ended since the button was drawn
            return {"CANCELLED"}
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

    shown_folder: StringProperty(options={"HIDDEN", "SKIP_SAVE"})  # for the dialog only
    shown_files: IntProperty(options={"HIDDEN", "SKIP_SAVE"})

    @staticmethod
    def _refusal(folder):
        if err := _lock_error(folder):
            return err
        if not bakefiles.is_bake_folder(folder):
            return f"Not an OpenBillow bake folder (no bake.json): {folder}"
        return None

    def invoke(self, context, event):
        try:
            folder = bake_folder(context.object)
        except export.ExportError:
            return self.execute(context)
        if self._refusal(folder):
            return self.execute(context)  # reports why
        self.shown_folder = folder
        self.shown_files = len(bakefiles.output_files(folder))
        return context.window_manager.invoke_props_dialog(self, width=600, title="Free Bake", confirm_text="Delete")

    def draw(self, context):
        col = self.layout.column()
        col.label(text=f"Delete {self.shown_files} baked files from this folder?", icon="ERROR")
        col.label(text=self.shown_folder)

    def execute(self, context):
        try:
            folder = bake_folder(context.object)
        except export.ExportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if err := self._refusal(folder):
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        for kind in ("spray", "foam", "bubble"):
            old = bpy.data.objects.get(_snd_name(context.object, kind))
            if old:
                bpy.data.objects.remove(old)
        _last.pop(folder, None)
        for path in bakefiles.output_files(folder):
            with contextlib.suppress(OSError):
                os.remove(path)
        for sub in bakefiles.OUTPUT_DIRS:
            with contextlib.suppress(OSError):  # still holds files that are not ours, or never existed
                os.rmdir(os.path.join(folder, sub))
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
        _use_deps()
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


def _volume_behind(ob, st):
    f0, f1 = st.get("frame_start"), st.get("frame")
    if not st.get("last_vdb") or f0 is None or f1 is None:
        return False
    vo = bpy.data.objects.get(ob.name + "_OpenBillow")
    return vo is None or (vo.type == "VOLUME" and vo.data.frame_duration < f1 - f0 + 1)


RADIUS_STAMP = "mesh_radius_v"  # on a domain: its Particle Radius follows the CODE-LIQ-004 meaning


def _migrate_radius(d):
    """Files saved before CODE-LIQ-004: High Quality ignored the radius and looked like 1.1447,
    so 2.0 maps to that. Other generators used the old 2.0 default, which is no longer the
    property's default, so an unset radius has to be written out."""
    if d.get(RADIUS_STAMP):
        return
    d[RADIUS_STAMP] = 1
    if d.mesh_generator == "ANISOTROPIC":
        if abs(d.mesh_particle_radius - RADIUS_OTHER) < 1e-4:
            d.mesh_particle_radius = RADIUS_HIGH_QUALITY
    elif not d.is_property_set("mesh_particle_radius"):
        d.mesh_particle_radius = RADIUS_OTHER


@bpy.app.handlers.persistent
def _on_save(*_):
    """Mark domains as current, so a 2.0 chosen from now on is not mapped at the next load."""
    for ob in bpy.data.objects:
        if ob.openbillow.role == "DOMAIN":
            ob.openbillow.domain[RADIUS_STAMP] = 1


@bpy.app.handlers.persistent
def _on_load(*_):
    """Catch up with a bake that advanced or finished while another file was open."""
    for ob in bpy.data.objects:
        if ob.openbillow.role != "DOMAIN":
            continue
        try:
            _migrate_radius(ob.openbillow.domain)
            folder = bake_folder(ob)
            st = read_status(folder)
            if st and _volume_behind(ob, st):
                update_volume(ob, folder, st)
        except export.ExportError:
            pass
        except Exception:  # noqa: BLE001 - never disturb a file load
            errlog.record("ERROR", traceback.format_exc())


def register():
    _use_deps()
    for c in _classes:
        bpy.utils.register_class(c)
    errlog.install()
    bpy.app.handlers.load_post.append(_on_load)
    bpy.app.handlers.save_pre.append(_on_save)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if _on_save in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save)
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    errlog.uninstall()
    for r in _runs.values():
        r["proc"].terminate()
    _runs.clear()
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
