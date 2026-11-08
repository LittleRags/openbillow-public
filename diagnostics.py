# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostics for bug reports: what the support box sends, and what "Copy Diagnostics" puts on
the clipboard for a public GitHub issue. No network. In both builds."""
import json
import os
import platform
import re
import subprocess

import bpy
from bpy.types import Operator

from . import errlog, ops
from .openbillow import __version__
from .props import prefs

ISSUES = "https://github.com/LittleRags/openbillow-public/issues"
MAX_DIAG = 32 * 1024
MAX_LOG = 64 * 1024


def scrub(text):
    """Home folders and user names out of paths."""
    home = os.path.expanduser("~")
    if len(home) > 3:
        spellings = {home, home.replace("\\", "/"), home.replace("\\", "\\\\"), os.path.realpath(home)}
        for h in sorted(spellings, key=len, reverse=True):
            text = re.sub(re.escape(h), "~", text, flags=re.I if os.name == "nt" else 0)
    # other users' folders: up to the next separator, quote or line end (spaces allowed)
    text = re.sub(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\"'\r\n]+", "~", text)
    text = re.sub(r"/(?:home|Users)/[^/\"'\r\n]+", "~", text)
    return text


_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/)(?:[^\\/\s\"':]+[\\/])+([^\\/\s\"':]*)")


def reduce_paths(text):
    """Absolute paths down to their last component: the file name stays, the folders go."""
    return _PATH.sub(r"<path>/\1", text)


def _own(line):
    return "openbillow" in line.lower()


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


def collect(context, want_log, want_errors=False, own_only=False, short_paths=False):
    """-> (diagnostics dict, log string or None). Blocks for the Warp probe (a few seconds)."""
    d = {"os": platform.platform(), "blender_version": bpy.app.version_string,
         "python_version": platform.python_version(), "openbillow_version": __version__}
    try:
        args = ["diagnostics"] + (["--device", prefs().device] if prefs().device else [])
        d.update({k: scrub(v) if isinstance(v, str) else v for k, v in ops.run_cli(*args).items()})
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as e:
        d["device_init_error"] = scrub(f"diagnostics probe failed: {e}")[:500]
    d["last_jit_error"] = None
    fix = (lambda t: reduce_paths(scrub(t))) if short_paths else scrub
    errs = errlog.recent() if want_errors else None
    if errs is not None and own_only:
        errs = [l for l in errs if _own(l)]
    d["recent_errors"] = [fix(l) for l in errs] if errs is not None else None
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
            if short_paths and d["last_jit_error"]:
                d["last_jit_error"] = reduce_paths(d["last_jit_error"])
            if want_log:
                log = fix(_tail(os.path.join(folder, "bake.log"), MAX_LOG)) or None
    while len(json.dumps(d)) > MAX_DIAG and d.get("domain"):
        d["domain"].popitem()
    return d, log



class OPENBILLOW_OT_report_bug(Operator):
    bl_idname = "openbillow.report_bug"
    bl_label = "Report a Bug"
    bl_description = "Open the OpenBillow issue tracker in your browser. Nothing is sent from Blender"

    def execute(self, context):
        bpy.ops.wm.url_open(url=ISSUES + "/new/choose")
        return {"FINISHED"}


class OPENBILLOW_OT_copy_diagnostics(Operator):
    bl_idname = "openbillow.copy_diagnostics"
    bl_label = "Copy Diagnostics"
    bl_description = ("Copy versions, GPU, the domain settings, the last bake error and, if ticked, recent "
                      "OpenBillow warnings to the clipboard. Folder names are removed from paths")

    def execute(self, context):
        diag, log = collect(context, want_log=False, want_errors=prefs().support_attach_errors,
                            own_only=True, short_paths=True)
        context.window_manager.clipboard = "```json\n" + json.dumps(diag, indent=1) + "\n```"
        self.report({"INFO"}, "Diagnostics copied. Read them before posting in public.")
        return {"FINISHED"}


def draw_report(layout):
    row = layout.row(align=True)
    row.operator("openbillow.copy_diagnostics", icon="COPYDOWN")
    row.operator("openbillow.report_bug", icon="URL")


_classes = (OPENBILLOW_OT_report_bug, OPENBILLOW_OT_copy_diagnostics)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
