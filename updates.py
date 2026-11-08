# SPDX-License-Identifier: GPL-3.0-or-later
"""Early-access update button. Asks the site for the latest version (sends only the current
version and platform), downloads the zip (sha256 checked) when the user clicks, and installs
it with Blender's extension installer. Never downgrades. Free for everyone; once the add-on
is on extensions.blender.org, Blender's own updater replaces this."""
import hashlib
import json
import os
import platform
import re
import tempfile
import textwrap
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse

import bpy
from bpy.types import Operator

from .openbillow import __version__

try:  # written only by `build_extension.py --store`
    from ._store_build import STORE
except ImportError:
    STORE = False

def repo_install(package=__package__ or ""):
    """True when installed from a remote extension repository (bl_ext.<repo>.openbillow); a zip
    installed from disk lands in user_default (or the development repo), so keeps the custom updater."""
    parts = package.split(".")
    return len(parts) > 2 and parts[0] == "bl_ext" and parts[1] not in ("user_default", "vscode_development")


REPO = repo_install()
API = "https://support.pipelineflow.co.uk/version"
HOST = "pipelineflow.co.uk"
MAX_ZIP = 200 * 1024 * 1024
# state: idle | checking | current | available | downloading | ready | installed | error
_s = {"state": "idle", "msg": "", "info": None, "zip": None, "auto_done": False}


def parse(v):
    """'0.5.0-early-access' -> (0, 5, 0). Anything unparsable -> None."""
    m = re.match(r"\s*v?(\d+)\.(\d+)\.(\d+)", str(v))
    return tuple(int(x) for x in m.groups()) if m else None


def _set(**kw):
    _s.update(kw)


def _trusted(url):
    u = urlparse(url)
    return u.scheme == "https" and (u.hostname == HOST or (u.hostname or "").endswith("." + HOST))


# ---- worker threads --------------------------------------------------------
def _check():
    payload = {"version": __version__, "platform": platform.system().lower(),
               "blender": ".".join(map(str, bpy.app.version))}
    req = urllib.request.Request(API, json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": f"OpenBillow/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            info = json.loads(r.read(1 << 20))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        _set(state="error", msg=f"Could not check for updates: {e}")
        return
    latest, cur = parse(info.get("latest")), parse(__version__)
    z = info.get("zip") or {}
    if latest is None or cur is None:
        _set(state="error", msg="Unexpected reply from the update server.")
    elif latest <= cur:  # never offer an older or equal build
        _set(state="current", msg=f"Up to date ({__version__}).", info=info)
    elif not (z.get("url") and z.get("sha256") and _trusted(z["url"])):
        _set(state="error", msg="The update server sent no usable download.")
    else:
        _set(state="available", info=info, msg=f"Version {info['latest']} is available.")


def _download():
    z = _s["info"]["zip"]
    try:
        req = urllib.request.Request(z["url"], headers={"User-Agent": f"OpenBillow/{__version__}"})
        h, n = hashlib.sha256(), 0
        fd, path = tempfile.mkstemp(suffix=".zip", prefix="openbillow_update_")
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=60) as r:
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > MAX_ZIP:
                    raise ValueError("file too large")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != z["sha256"].lower():
            os.remove(path)
            _set(state="error", msg="Download failed its checksum; nothing was installed.")
            return
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        _set(state="error", msg=f"Download failed: {e}")
        return
    _set(state="ready", zip=path, msg="Downloaded and verified. Click Install.")


def _start(target, state, msg):
    _set(state=state, msg=msg)
    threading.Thread(target=target, daemon=True).start()
    bpy.app.timers.register(_wait_redraw, first_interval=0.5)


def _wait_redraw():
    if _s["state"] in ("checking", "downloading"):
        return 0.5
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    return None


def _auto_check():
    """Once per session, shortly after start-up, only with online access on."""
    if not _s["auto_done"] and bpy.app.online_access:
        _s["auto_done"] = True
        _start(_check, "checking", "Checking for updates...")


# ---- operators -------------------------------------------------------------
class OPENBILLOW_OT_update_check(Operator):
    bl_idname = "openbillow.update_check"
    bl_label = "Check for Updates"
    bl_description = "Look online for a newer OpenBillow. Only your version and system type are sent."

    def execute(self, context):
        if not bpy.app.online_access:
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        _start(_check, "checking", "Checking for updates...")
        return {"FINISHED"}


class OPENBILLOW_OT_update_download(Operator):
    bl_idname = "openbillow.update_download"
    bl_label = "Download Update"
    bl_description = "Download the new version and check that the file is intact."

    def execute(self, context):
        if _s["state"] != "available" or not bpy.app.online_access:
            return {"CANCELLED"}
        _start(_download, "downloading", "Downloading...")
        return {"FINISHED"}


class OPENBILLOW_OT_update_install(Operator):
    bl_idname = "openbillow.update_install"
    bl_label = "Install Update"
    bl_description = "Install the downloaded version over this one. Restart Blender afterwards."

    def execute(self, context):
        path = _s["zip"]
        if _s["state"] != "ready" or not path or not os.path.isfile(path):
            self.report({"ERROR"}, "No verified download to install")
            return {"CANCELLED"}
        try:
            r = bpy.ops.extensions.package_install_files(
                filepath=path, repo="user_default", enable_on_install=True, overwrite=True)
        except Exception as e:
            # Blender may have written the files and then failed to unregister the running copy.
            _set(state="error", msg=f"Install reported an error ({str(e).strip()[:120]}). The files may "
                                    "have been updated: restart Blender and check the version.")
            self.report({"ERROR"}, _s["msg"])
            return {"CANCELLED"}
        if "FINISHED" not in r:
            _set(state="error", msg="Blender did not install the update; see the console.")
            return {"CANCELLED"}
        try:
            os.remove(path)
        except OSError:
            pass
        _set(state="installed", zip=None, msg="Installed. Restart Blender to use the new version.")
        return {"FINISHED"}


def draw_box(layout):
    if STORE:
        return
    if REPO:
        box = layout.box()
        box.label(text=f"Installed {__version__}", icon="CHECKMARK")
        box.label(text="Updates come through Blender's Get Extensions.")
        return
    box = layout.box()
    box.label(text=f"Updates (installed {__version__})", icon="FILE_REFRESH")
    on = bpy.app.online_access
    st = _s["state"]
    row = box.row()
    sub = row.row()
    sub.enabled = on and st not in ("checking", "downloading")
    sub.operator("openbillow.update_check", icon="FILE_REFRESH")
    if st == "available":
        row.operator("openbillow.update_download", text=f"Download {_s['info']['latest']}", icon="IMPORT")
    elif st == "ready":
        row.operator("openbillow.update_install", icon="CHECKMARK")
    if not on:
        box.label(text="Online access is off (Preferences > System > Network).", icon="ERROR")
    if _s["msg"]:
        notes = (_s["info"] or {}).get("notes_url")
        for line in textwrap.wrap(_s["msg"], 90):
            box.label(text=line, icon="ERROR" if st == "error" else "NONE")
        if notes and st in ("available", "ready"):
            box.operator("wm.url_open", text="Release notes", icon="URL").url = notes


_classes = (OPENBILLOW_OT_update_check, OPENBILLOW_OT_update_download, OPENBILLOW_OT_update_install)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    if not STORE and not REPO:
        bpy.app.timers.register(_auto_check, first_interval=5.0)


def unregister():
    if bpy.app.timers.is_registered(_auto_check):
        bpy.app.timers.unregister(_auto_check)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
