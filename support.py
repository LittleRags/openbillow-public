# SPDX-License-Identifier: GPL-3.0-or-later
"""Support button: collects diagnostics, shows the exact JSON before sending, posts it to
the support API. The add-on enforces nothing; the server decides who is entitled. The Warp
probe runs in the CLI subprocess (this module never imports Warp)."""
import json
import os
import platform
import re
import subprocess
import textwrap
import threading
import urllib.error
import urllib.request

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import activation, errlog, ops
from .openbillow import __version__
from .props import prefs

API = "https://support.pipelineflow.co.uk/tickets"
SITE = "https://pipelineflow.co.uk"
EMAIL = "support@pipelineflow.co.uk"
MAX_DIAG = 32 * 1024
MAX_LOG = 64 * 1024
TEXT_NAME = "OpenBillow support request"
_result = {"state": "idle", "msg": ""}  # idle | sending | ok | error
_pending = {}  # what the preview showed, which is exactly what Send posts


# ---- collection ------------------------------------------------------------
def scrub(text):
    """Home folders and user names out of paths."""
    text = re.sub(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\s\"']+", "~", text)
    text = re.sub(r"/(?:home|Users)/[^/\s\"']+", "~", text)
    home = os.path.expanduser("~")
    return text.replace(home, "~") if len(home) > 3 else text


def _tail(path, n):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _jit_error(folder):
    """Last error block from the bake stdout/log, trimmed."""
    for name in ("bake.stdout", "bake.log"):
        lines = _tail(os.path.join(folder, name), 16384).splitlines()
        hits = [i for i, l in enumerate(lines) if re.search(r"error|nvrtc|failed|traceback", l, re.I)]
        if hits:
            return scrub("\n".join(lines[hits[-1]:hits[-1] + 12]))[:2000]
    return None


def _domain_settings(ob):
    """Numbers, booleans and enums of the domain. No strings, so no names or paths."""
    ds = ob.openbillow.domain
    out = {}
    for p in ds.bl_rna.properties:
        if p.identifier == "rna_type" or p.type not in {"INT", "FLOAT", "BOOLEAN", "ENUM"}:
            continue
        v = getattr(ds, p.identifier)
        if p.type == "ENUM" and p.is_enum_flag:
            v = sorted(v)
        elif p.type != "ENUM" and p.array_length:
            v = [round(x, 6) if isinstance(x, float) else x for x in v]
        elif isinstance(v, float):
            v = round(v, 6)
        out[p.identifier] = v
    return out


def collect(context, want_log, want_errors=False):
    """-> (diagnostics dict, log string or None). Blocks for the Warp probe (a few seconds)."""
    d = {"os": platform.platform(), "blender_version": bpy.app.version_string,
         "python_version": platform.python_version(), "openbillow_version": __version__}
    try:
        args = ["diagnostics"] + (["--device", prefs().device] if prefs().device else [])
        d.update(ops.run_cli(*args))
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
        d["device_init_error"] = scrub(f"diagnostics probe failed: {e}")[:500]
    d["last_jit_error"] = None
    d["recent_errors"] = [scrub(l) for l in errlog.recent()] if want_errors else None
    log = None
    ob = ops.domain_of(context)
    if ob:
        d["domain"] = _domain_settings(ob)
        try:
            folder = ops.bake_folder(ob)
        except Exception:  # unsaved .blend with a relative cache folder
            folder = None
        if folder:
            d["last_jit_error"] = _jit_error(folder)
            if want_log:
                log = scrub(_tail(os.path.join(folder, "bake.log"), MAX_LOG)) or None
    while len(json.dumps(d)) > MAX_DIAG and d.get("domain"):
        d["domain"].popitem()
    return d, log


def build_payload(context):
    p = prefs()
    diag, log = collect(context, p.support_attach_log, p.support_attach_errors)
    return {"licence_key": p.support_key.strip(), "email": p.support_email.strip(),
            "machine_id": activation.machine_id(), "description": p.support_description.strip(),
            "diagnostics": diag, "log": log}


def _online():
    return bpy.app.online_access


# ---- sending ---------------------------------------------------------------
def _post(payload):
    req = urllib.request.Request(API, json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": f"OpenBillow/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body, ok = r.read(), True
    except urllib.error.HTTPError as e:
        body, ok = e.read(), False
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        _result.update(state="error", msg=f"Could not reach the support server: {e}. "
                                          f"Use Save Diagnostics and email the file to {EMAIL}.")
        return
    try:
        j = json.loads(body)
    except ValueError:
        j = {}
    msg = j.get("message") or "Unexpected reply from the support server."
    if ok and j.get("ticket"):
        msg = f"{msg} (ticket {j['ticket']})"
    _result.update(state="ok" if ok else "error", msg=msg)


def _wait_redraw():
    if _result["state"] == "sending":
        return 0.5
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    return None


# ---- operators -------------------------------------------------------------
def _check(op, p):
    if not activation.is_activated(p):
        op.report({"ERROR"}, "Activate your licence key first (Support box above)")
    elif not p.support_key.strip():
        op.report({"ERROR"}, "Enter your licence key")
    elif "@" not in p.support_email:
        op.report({"ERROR"}, "Enter your email address")
    elif not p.support_description.strip():
        op.report({"ERROR"}, "Describe the problem")
    else:
        return True
    return False


class OPENBILLOW_OT_support_send(Operator):
    bl_idname = "openbillow.support_send"
    bl_label = "Send Support Request"
    bl_description = "Show you exactly what will be sent, then send your support request."

    def invoke(self, context, event):
        p = prefs()
        if not _online():
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        if not _check(self, p):
            return {"CANCELLED"}
        _pending.clear()
        _pending.update(build_payload(context))
        shown = dict(_pending, licence_key="(your key)")
        txt = bpy.data.texts.get(TEXT_NAME) or bpy.data.texts.new(TEXT_NAME)
        txt.from_string(json.dumps(shown, indent=2))
        return context.window_manager.invoke_props_dialog(self, width=640, confirm_text="Send")

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="This will be sent to support.pipelineflow.co.uk:", icon="URL")
        col.label(text=f"email: {_pending['email']}")
        for line in textwrap.wrap(_pending["description"], 90)[:6]:
            col.label(text=line)
        col.separator()
        for k, v in _pending["diagnostics"].items():
            if k == "recent_errors":
                col.label(text=f"recent errors: {len(v)} lines" if v else "recent errors: none")
                continue
            if isinstance(v, dict):
                col.label(text=f"{k}: {len(v)} settings (numbers and switches only)")
                continue
            for i, line in enumerate(textwrap.wrap(f"{k}: {v}", 90) or [k]):
                col.label(text=line if i == 0 else "    " + line)
        lg = _pending["log"]
        col.label(text=f"bake log: {len(lg.splitlines())} lines" if lg else "bake log: not attached")
        col.separator()
        col.label(text=f"The full JSON is in the Text Editor as '{TEXT_NAME}'.")

    def execute(self, context):
        if not _pending:
            return {"CANCELLED"}
        _result.update(state="sending", msg="Sending...")
        threading.Thread(target=_post, args=(dict(_pending),), daemon=True).start()
        bpy.app.timers.register(_wait_redraw, first_interval=0.5)
        return {"FINISHED"}


class OPENBILLOW_OT_support_save(Operator):
    bl_idname = "openbillow.support_save"
    bl_label = "Save Diagnostics to File"
    bl_description = f"Write the same information to a JSON file you can email to {EMAIL}"
    filepath: StringProperty(subtype="FILE_PATH", default="openbillow_diagnostics.json")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        p = prefs()
        payload = build_payload(context)
        payload.pop("licence_key")
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.report({"INFO"}, f"Saved. Email it to {EMAIL} with your licence key and a description.")
        return {"FINISHED"}


def draw_box(layout):
    p = prefs()
    box = layout.box()
    box.label(text="Support", icon="QUESTION")
    row = box.row()
    row.operator("wm.url_open", text="Get a key (Superhive buyers)").url = SITE + "/support/claim/"
    row.operator("wm.url_open", text="How support works").url = SITE + "/support/"
    col = box.column()
    col.prop(p, "support_key")
    col.prop(p, "support_email")
    activation.draw(box)
    col = box.column()
    col.prop(p, "support_description")
    col.prop(p, "support_attach_log")
    col.prop(p, "support_attach_errors")
    row = box.row()
    sub = row.row()
    sub.enabled = _online() and _result["state"] != "sending" and activation.is_activated(p)
    sub.operator("openbillow.support_send", icon="EXPORT")
    row.operator("openbillow.support_save", icon="FILE_TICK")
    if not _online():
        box.label(text="Online access is off (Preferences > System > Network). Save diagnostics and email them.",
                  icon="ERROR")
    if _result["msg"]:
        for line in textwrap.wrap(_result["msg"], 90):
            box.label(text=line, icon="CHECKMARK" if _result["state"] == "ok" else "NONE")


_classes = (OPENBILLOW_OT_support_send, OPENBILLOW_OT_support_save)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
