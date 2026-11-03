# SPDX-License-Identifier: GPL-3.0-or-later
"""Licensed preset downloads. Presets are not shipped with the add-on: Download/Update in the
Preferences fetches them from the Pipelineflow server with the support licence key, verifies
each file's sha256 and keeps them in the user extension folder. The server decides who is
entitled; the add-on enforces nothing.

Preset file (`<name>.json`), format 1 (format 2 when the domain uses the COMBUSTION fire model):
    {"format": 1, "name": "Campfire", "description": "...",
     "domain": {<OpenBillowDomain property>: value, "effector_weights": {...}},
     "flow": {<OpenBillowFlow property>: value},   # applied to every Flow object in the scene
     "material": "OB Campfire"}                     # optional, appended from materials.blend
Unknown keys are ignored, so older add-ons can read newer presets."""
import hashlib
import json
import os
import shutil
import tempfile
import textwrap
import threading
import urllib.error
import urllib.request

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

from . import activation, ops
from .openbillow import __version__
from .props import prefs

API = "https://support.pipelineflow.co.uk/presets"
FORMAT = 2  # 2: COMBUSTION fire model (0.5.x add-ons would load it as MANTAFLOW, so they refuse it)
MATERIALS = "materials.blend"
INDEX = "index.json"
MAX_FILE = 64 * 2**20
SKIP = {"cache_dir", "refresh_vdb", "use_scene_range", "frame_start", "frame_end", "guide_parent", "material"}
ROOT_OVERRIDE = None  # tests
_result = {"state": "idle", "msg": ""}  # idle | busy | ok | error
_items = {"key": None, "items": []}  # enum items must stay referenced from Python


def folder(create=False):
    if ROOT_OVERRIDE:
        if create:
            os.makedirs(ROOT_OVERRIDE, exist_ok=True)
        return ROOT_OVERRIDE
    try:
        return bpy.utils.extension_path_user(__package__, path="presets", create=create)
    except (ValueError, KeyError):  # legacy add-on install, not an extension
        p = bpy.utils.user_resource("CONFIG", path="openbillow_presets", create=create)
        return p


def installed():
    """-> local index dict, or None when nothing is downloaded."""
    try:
        with open(os.path.join(folder(), INDEX), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def load(name):
    with open(os.path.join(folder(), name), encoding="utf-8") as fh:
        return json.load(fh)


# ---- applying and capturing ------------------------------------------------
def _settable(group):
    return {p.identifier: p for p in group.bl_rna.properties
            if p.identifier != "rna_type" and p.identifier not in SKIP}


def _assign(group, values):
    """Set the known properties; -> number of keys skipped (unknown or out of range)."""
    props, skipped = _settable(group), 0
    for k, v in values.items():
        if k in SKIP:
            continue
        p = props.get(k)
        try:
            if p is None or p.is_readonly and p.type != "POINTER":
                raise TypeError
            if p.type == "POINTER":
                skipped += _assign(getattr(group, k), v)
            else:
                setattr(group, k, set(v) if p.type == "ENUM" and p.is_enum_flag else v)
        except (TypeError, ValueError, AttributeError):
            skipped += 1
    return skipped


def _capture(group):
    out = {}
    for k, p in _settable(group).items():
        v = getattr(group, k)
        if p.type == "POINTER":
            if p.fixed_type.identifier.startswith("OpenBillow"):
                out[k] = _capture(v)
        elif p.type in {"INT", "FLOAT", "BOOLEAN", "ENUM"}:
            if p.type == "ENUM" and p.is_enum_flag:
                v = sorted(v)
            elif p.type != "ENUM" and p.array_length:
                v = [round(x, 6) if isinstance(x, float) else x for x in v]
            elif isinstance(v, float):
                v = round(v, 6)
            out[k] = v
    return out


def capture(context, name, description=""):
    """Preset dict from the active domain and the first Flow object (for authoring the pack)."""
    ob = ops.domain_of(context)
    d = ob.openbillow.domain
    preset = {"format": 2 if d.fire_model == "COMBUSTION" else 1, "name": name, "description": description,
              "domain": _capture(d)}
    flows = ops.scene_objects(context.scene, "FLOW")
    if flows:
        preset["flow"] = _capture(flows[0].openbillow.flow)
    if ob.openbillow.domain.material:
        preset["material"] = ob.openbillow.domain.material.name
    return preset


def _material(name):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    path = os.path.join(folder(), MATERIALS)
    if not os.path.isfile(path):
        return None
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        if name in src.materials:
            dst.materials = [name]
    return dst.materials[0] if dst.materials else None


def apply(context, preset):
    """-> (number of objects changed, message)."""
    if preset.get("format", 0) > FORMAT:
        return 0, "This preset needs a newer OpenBillow"
    ob = ops.domain_of(context)
    if ob is None:
        return 0, "Select a domain"
    skipped = _assign(ob.openbillow.domain, preset.get("domain", {}))
    n = 1
    if "flow" in preset:
        for fo in ops.scene_objects(context.scene, "FLOW"):
            skipped += _assign(fo.openbillow.flow, preset["flow"])
            n += 1
    msg = f"Applied '{preset.get('name', '?')}' to {n} object{'s' * (n > 1)}"
    if preset.get("material"):
        m = _material(preset["material"])
        if m is None:
            msg += f"; material '{preset['material']}' not found (update presets)"
        else:
            ob.openbillow.domain.material = m
            res = bpy.data.objects.get(ob.name + ("_Liquid" if ob.openbillow.domain.domain_type == "LIQUID"
                                                  else "_OpenBillow"))
            if res is not None and res.data is not None:
                res.data.materials.clear()
                res.data.materials.append(m)
    if skipped:
        msg += f" ({skipped} setting{'s' * (skipped > 1)} not supported by this version)"
    return n, msg


# ---- downloading -----------------------------------------------------------
class ServerError(Exception):
    pass


def _request(path, payload):
    """POST JSON; -> response bytes. Raises ServerError with the server's message."""
    req = urllib.request.Request(API + path, json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": f"OpenBillow/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = int(r.headers.get("Content-Length") or 0)
            if n > MAX_FILE:
                raise ServerError(f"File too large ({n} bytes)")
            body = r.read(MAX_FILE + 1)
            if len(body) > MAX_FILE or n and len(body) != n:
                raise ServerError("Download incomplete, try again")
            return body
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("message")
        except ValueError:
            msg = None
        raise ServerError(msg or f"Server error {e.code}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise ServerError(f"Could not reach the preset server: {e}") from None


def _sha(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _safe(name):
    return (isinstance(name, str) and name == os.path.basename(name) and not name.startswith(".")
            and os.path.splitext(name)[1] in {".json", ".blend"} and name != INDEX)


def download(key, email, machine_id=""):
    """Fetch the index, then each file whose sha256 differs; drop files no longer listed.
    Runs in a thread; reports through _result."""
    try:
        auth = {"licence_key": key, "email": email, "machine_id": machine_id}
        try:
            index = json.loads(_request("/index", auth))
            files = [f for f in index["files"] if _safe(f["name"])]
        except (ValueError, KeyError, TypeError):
            raise ServerError("Unexpected reply from the preset server") from None
        root = folder(create=True)
        if not root:
            raise ServerError("Cannot create the preset folder")
        got = 0
        for f in files:
            dst = os.path.join(root, f["name"])
            if _sha(dst) == f["sha256"]:
                continue
            body = _request("/get", dict(auth, name=f["name"]))
            if hashlib.sha256(body).hexdigest() != f["sha256"]:
                raise ServerError(f"{f['name']}: checksum mismatch, try again")
            fd, tmp = tempfile.mkstemp(dir=root, suffix=".part")
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
            os.replace(tmp, dst)
            got += 1
        keep = {f["name"] for f in files}
        for name in os.listdir(root):
            if name not in keep and name != INDEX and _safe(name):
                os.remove(os.path.join(root, name))
        with open(os.path.join(root, INDEX), "w", encoding="utf-8") as fh:
            json.dump({"version": index.get("version", "?"), "files": files}, fh, indent=1)
        _items["key"] = None
        _result.update(state="ok", msg=f"Presets {index.get('version', '?')}: {len(files)} files, "
                                       f"{got} downloaded" + (", up to date" if not got else ""))
    except ServerError as e:
        _result.update(state="error", msg=str(e))
    except OSError as e:
        _result.update(state="error", msg=f"Could not save presets: {e}")


def remove():
    root = folder()
    if root and os.path.isdir(root):
        shutil.rmtree(root)
    _items["key"] = None


def _wait_redraw():
    if _result["state"] == "busy":
        return 0.5
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    return None


# ---- operators -------------------------------------------------------------
def _preset_items(self, context):
    idx = installed()
    key = json.dumps(idx, sort_keys=True) if idx else ""
    if key != _items["key"]:
        items = []
        for f in (idx or {}).get("files", []):
            if not f["name"].endswith(".json"):
                continue
            try:
                p = load(f["name"])
            except (OSError, ValueError):
                continue
            items.append((f["name"], p.get("name") or f["name"][:-5], p.get("description", "")))
        items.sort(key=lambda it: it[1])
        _items.update(key=key, items=items or [("NONE", "No presets", "Download them in the Preferences")])
    return _items["items"]


class OPENBILLOW_OT_preset_apply(Operator):
    bl_idname = "openbillow.preset_apply"
    bl_label = "Apply Preset"
    bl_description = "Apply a ready-made look to the domain and its Flow objects, such as campfire or honey."
    bl_options = {"REGISTER", "UNDO"}
    preset: EnumProperty(name="Preset", items=_preset_items)

    def execute(self, context):
        if self.preset == "NONE":
            return {"CANCELLED"}
        try:
            n, msg = apply(context, load(self.preset))
        except (OSError, ValueError) as e:
            self.report({"ERROR"}, f"Cannot read preset: {e}")
            return {"CANCELLED"}
        self.report({"INFO"} if n else {"ERROR"}, msg)
        return {"FINISHED"} if n else {"CANCELLED"}


class OPENBILLOW_OT_presets_download(Operator):
    bl_idname = "openbillow.presets_download"
    bl_label = "Download / Update Presets"
    bl_description = "Download the ready-made presets and materials. Only files that changed are fetched."

    def execute(self, context):
        p = prefs()
        if not bpy.app.online_access:
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        if not activation.is_activated(p):
            self.report({"ERROR"}, "Activate your licence key first (Support box)")
            return {"CANCELLED"}
        if not p.support_key.strip() or "@" not in p.support_email:
            self.report({"ERROR"}, "Enter your licence key and email in the Support box")
            return {"CANCELLED"}
        _result.update(state="busy", msg="Downloading presets...")
        threading.Thread(target=download, args=(p.support_key.strip(), p.support_email.strip(),
                                                activation.machine_id()),
                         daemon=True).start()
        bpy.app.timers.register(_wait_redraw, first_interval=0.5)
        return {"FINISHED"}


class OPENBILLOW_OT_presets_remove(Operator):
    bl_idname = "openbillow.presets_remove"
    bl_label = "Remove Presets"
    bl_description = "Remove the downloaded presets and materials from this computer."

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        remove()
        _result.update(state="idle", msg="")
        return {"FINISHED"}


def draw_box(layout):
    box = layout.box()
    box.label(text="Presets and materials (licensed)", icon="PRESET")
    idx = installed()
    if idx:
        n = sum(f["name"].endswith(".json") for f in idx.get("files", []))
        box.label(text=f"Installed: version {idx.get('version', '?')}, {n} presets")
    else:
        box.label(text="Not downloaded. Uses the activated licence key and email from the Support box.")
    if not activation.is_activated():
        box.label(text="Activate your licence key in the Support box first.", icon="ERROR")
    row = box.row()
    sub = row.row()
    sub.enabled = bpy.app.online_access and _result["state"] != "busy" and activation.is_activated()
    sub.operator("openbillow.presets_download", icon="IMPORT")
    sub = row.row()
    sub.enabled = idx is not None
    sub.operator("openbillow.presets_remove", icon="TRASH")
    if not bpy.app.online_access:
        box.label(text="Online access is off (Preferences > System > Network).", icon="ERROR")
    if _result["msg"]:
        for line in textwrap.wrap(_result["msg"], 90):
            box.label(text=line, icon="CHECKMARK" if _result["state"] == "ok" else "NONE")


def draw_dropdown(layout, domain):
    row = layout.row(align=True)
    if installed():
        row.operator_menu_enum("openbillow.preset_apply", "preset", text="Presets", icon="PRESET")
    else:
        op = row.operator("preferences.addon_show", text="Presets: download in Preferences", icon="PRESET")
        op.module = __package__
    layout.prop(domain, "material")


_classes = (OPENBILLOW_OT_preset_apply, OPENBILLOW_OT_presets_download, OPENBILLOW_OT_presets_remove)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
