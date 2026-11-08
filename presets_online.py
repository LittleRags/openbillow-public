# SPDX-License-Identifier: GPL-3.0-or-later
"""Premium preset download (Pipelineflow server, licence key). Left out of the Extensions-platform build."""
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
from bpy.types import Operator

from . import activation, presets
from .openbillow import __version__
from .presets import FORMAT, INDEX, MATERIALS, folder, installed
from .props import prefs

API = "https://support.pipelineflow.co.uk/presets"
MAX_FILE = 64 * 2**20
_result = {"state": "idle", "msg": ""}  # idle | busy | ok | error


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
        presets._items["key"] = None
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
    presets._items["key"] = None


def _wait_redraw():
    if _result["state"] == "busy":
        return 0.5
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    return None




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



_classes = (OPENBILLOW_OT_presets_download, OPENBILLOW_OT_presets_remove)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
