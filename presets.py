# SPDX-License-Identifier: GPL-3.0-or-later
"""Presets. A small free base pack ships inside the add-on (`presets_base/`, listed as "(base)").
The premium pack is not shipped: Download/Update in the
Preferences fetches them from the Pipelineflow server with the support licence key, verifies
each file's sha256 and keeps them in the user extension folder. The server decides who is
entitled; the add-on enforces nothing.

Preset file (`<name>.json`), format 1 (format 2 when the domain uses the COMBUSTION fire model):
    {"format": 1, "name": "Campfire", "description": "...",
     "domain": {<OpenBillowDomain property>: value, "effector_weights": {...}},
     "flow": {<OpenBillowFlow property>: value},   # applied to every Flow object in the scene
     "material": "OB Campfire",                     # optional, appended from materials.blend
     "recommended_resolution": 320}                 # optional: Resolution Divisions the look was tuned at
Unknown keys are ignored, so older add-ons can read newer presets."""
import json
import os

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

from . import ops

FORMAT = 2  # 2: COMBUSTION fire model (0.5.x add-ons would load it as MANTAFLOW, so they refuse it)
MATERIALS = "materials.blend"
INDEX = "index.json"
SKIP = {"cache_dir", "refresh_vdb", "use_scene_range", "frame_start", "frame_end", "guide_parent", "material"}
ROOT_OVERRIDE = None  # tests
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


BASE_DIR = os.path.join(os.path.dirname(__file__), "presets_base")  # free base pack, ships in the add-on
BASE = "base/"  # dropdown id prefix of base presets


def base_names():
    try:
        return sorted(f for f in os.listdir(BASE_DIR) if f.endswith(".json"))
    except OSError:
        return []


def load(name):
    path = os.path.join(BASE_DIR, name[len(BASE):]) if name.startswith(BASE) else os.path.join(folder(), name)
    with open(path, encoding="utf-8") as fh:
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
    for root in (folder(), BASE_DIR):
        path = os.path.join(root, MATERIALS)
        if not os.path.isfile(path):
            continue
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            if name in src.materials:
                dst.materials = [name]
        if dst.materials:
            return dst.materials[0]
    return None


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
    rec = preset.get("recommended_resolution")
    if isinstance(rec, int) and rec > 0:
        cur = ob.openbillow.domain.resolution_max
        msg += (f". Recommended resolution: {rec}" + (f" (now {cur}, set it in Resolution Divisions)" if cur != rec
                                                   else " (already set)"))
    if skipped:
        msg += f" ({skipped} setting{'s' * (skipped > 1)} not supported by this version)"
    return n, msg


# ---- operators -------------------------------------------------------------
def _preset_items(self, context):
    idx = installed()
    key = json.dumps([idx, base_names()], sort_keys=True)
    if key != _items["key"]:
        items = []
        names = [BASE + n for n in base_names()] + [f["name"] for f in (idx or {}).get("files", [])
                                                     if f["name"].endswith(".json")]
        for n in names:
            try:
                p = load(n)
            except (OSError, ValueError):
                continue
            label = p.get("name") or os.path.basename(n)[:-5]
            items.append((n, label + (" (base)" if n.startswith(BASE) else ""), p.get("description", "")))
        items.sort(key=lambda it: it[1])
        _items.update(key=key, items=items or [("NONE", "No presets", "Premium presets: download in the Preferences")])
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


def draw_dropdown(layout, domain):
    row = layout.row(align=True)
    if installed() or base_names():
        row.operator_menu_enum("openbillow.preset_apply", "preset", text="Presets", icon="PRESET")
    else:
        op = row.operator("preferences.addon_show", text="Presets: download in Preferences", icon="PRESET")
        op.module = __package__
    layout.prop(domain, "material")


_classes = (OPENBILLOW_OT_preset_apply,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
