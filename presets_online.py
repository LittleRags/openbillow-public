# SPDX-License-Identifier: GPL-3.0-or-later
"""Premium preset download (Pipelineflow server, licence key). Left out of the Extensions-platform build."""
import hashlib
import json
import os
import shutil
import tempfile
import textwrap

import bpy
from bpy.types import Operator

from . import activation, net, presets, signing
from .presets import FORMAT, INDEX, MATERIALS, folder, installed
from .props import prefs

API = "https://support.pipelineflow.co.uk/presets"
MAX_FILE = 64 * 2**20
_result = {"state": "idle", "msg": ""}  # idle | busy | ok | error


# ---- downloading -----------------------------------------------------------
ServerError = net.NetError


def _request(path, payload):
    """POST JSON; -> response bytes. Raises NetError with the server's message."""
    code, body = net.post(API + path, payload, timeout=60, limit=MAX_FILE, what="the preset server")
    if code >= 400:
        j = net.as_dict(body) or {}
        raise ServerError(str(j.get("message") or f"Server error {code}"))
    return body


def _sha(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _safe(name):
    return (isinstance(name, str) and name == os.path.basename(name) and not name.startswith(".")
            and os.path.splitext(name)[1] in {".json", ".blend"} and name != INDEX)


def _sha_ok(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _files(index):
    """The index's entries with a safe name; an entry without a usable sha256 rejects the index."""
    if not isinstance(index, dict) or not isinstance(index.get("files"), list):
        raise ServerError("Unexpected reply from the preset server")
    files = []
    for f in index["files"]:
        if not isinstance(f, dict):
            raise ServerError("Unexpected reply from the preset server")
        if _safe(f.get("name")):
            if not _sha_ok(f.get("sha256")):
                raise ServerError(f"Unexpected reply from the preset server (no checksum for {f['name']})")
            files.append(f)
    return files


def _signed_index(reply):
    """The pack's manifest from the signed text of a /presets/index view (CODE-API-004); never the unsigned copy."""
    if not isinstance(reply, dict):
        raise ServerError("Unexpected reply from the preset server")
    try:
        return signing.signed_json(reply, "preset index")
    except signing.BadSignature as e:
        raise ServerError(str(e)) from None


def download(key, email, machine_id, root):
    """Fetch the index, then each file whose sha256 differs; drop files no longer listed.
    Runs in a thread (plain values only, no bpy; `root` is resolved on the main thread); reports through _result."""
    try:
        auth = {"licence_key": key, "email": email, "machine_id": machine_id}
        index = _signed_index(net.as_dict(_request("/index", auth)))
        files = _files(index)
        for name in os.listdir(root):  # leftovers of an interrupted earlier download
            if name.endswith(".part"):
                os.remove(os.path.join(root, name))
        got = 0
        for f in files:
            dst = os.path.join(root, f["name"])
            if _sha(dst) == f["sha256"]:
                continue
            body = _request("/get", dict(auth, name=f["name"]))
            if hashlib.sha256(body).hexdigest() != f["sha256"]:
                raise ServerError(f"{f['name']}: checksum mismatch, try again")
            fd, tmp = tempfile.mkstemp(dir=root, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(body)
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            got += 1
        keep = {f["name"] for f in files}
        for name in os.listdir(root):
            if name not in keep and name != INDEX and _safe(name):
                os.remove(os.path.join(root, name))
        with open(os.path.join(root, INDEX), "w", encoding="utf-8") as fh:
            json.dump({"version": index.get("version", "?"), "files": files}, fh, indent=1)
        presets._items["key"] = None
        presets.clear_cache()
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
    presets.clear_cache()


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
        root = folder(create=True)
        if not root:
            self.report({"ERROR"}, "Cannot create the preset folder")
            return {"CANCELLED"}
        _result.update(state="busy", msg="Downloading presets...")
        net.run(download, _result, p.support_key.strip(), p.support_email.strip(), activation.machine_id(), root)
        net.wait_redraw(_result, "busy")
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
