# SPDX-License-Identifier: GPL-3.0-or-later
"""Early-access update button. Asks the site for the latest version (sends only the current
version and platform), downloads the zip (sha256 checked) when the user clicks, and installs
it with Blender's extension installer. Never downgrades. Free for everyone; once the add-on
is on extensions.blender.org, Blender's own updater replaces this."""
import hashlib
import http.client
import os
import platform
import re
import sys
import tempfile
import textwrap
import tomllib
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlparse

import bpy
from bpy.types import Operator

from . import net, ops, signing
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


def _install_repo(package=__package__ or ""):
    """The local repository this copy can be replaced in, or None: a legacy install, the development
    copy (bl_ext.vscode_development.addon), or a remote or system repository."""
    parts = package.split(".")
    if len(parts) != 3 or parts[0] != "bl_ext" or parts[2] != "openbillow":
        return None
    repo = next((r for r in bpy.context.preferences.extensions.repos if r.module == parts[1]), None)
    return repo.module if repo and not repo.use_remote_url and repo.source == "USER" else None


REPO = repo_install()
API = "https://support.pipelineflow.co.uk/version"
HOST = "pipelineflow.co.uk"
MAX_ZIP = 200 * 1024 * 1024
# state: idle | checking | current | available | downloading | ready | installing | installed | error
_s = {"state": "idle", "msg": "", "info": None, "zip": None, "repo": None, "auto_done": False}


def parse(v):
    """'0.5.0-early-access' -> (0, 5, 0). Anything unparsable -> None."""
    m = re.match(r"\s*v?(\d+)\.(\d+)\.(\d+)", str(v))
    return tuple(int(x) for x in m.groups()) if m else None


def _set(**kw):
    _s.update(kw)


def _trusted(url):
    u = urlparse(url)
    return u.scheme == "https" and (u.hostname == HOST or (u.hostname or "").endswith("." + HOST))


class _SameHost(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted(newurl):
            raise urllib.error.URLError("redirect to an untrusted host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_open = urllib.request.build_opener(_SameHost).open


def _drop_zip():
    """Delete a downloaded but not installed zip."""
    path, _s["zip"] = _s["zip"], None
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


# ---- worker threads (run through net.run: any failure ends in state "error") ----
def _payload():
    """Main thread: the request body, so the worker never reads bpy."""
    return {"version": __version__, "platform": platform.system().lower(),
            "blender": ".".join(map(str, bpy.app.version))}


def _check(payload):
    ok, info = net.post_json(API, payload, timeout=15, what="the update server")
    if not ok:
        raise net.NetError(f"Could not check for updates: {info.get('message') or 'server error'}")
    # CODE-API-004: everything but the download URL comes from the signed latest.json text.
    try:
        signed = signing.signed_json(info, "update")
    except signing.BadSignature as e:
        raise net.NetError(str(e)) from None
    top = info.get("zip")
    z = signed.get("zip")
    z = dict(z, url=top.get("url")) if isinstance(z, dict) and isinstance(top, dict) else {}
    info = dict(signed, zip=z)
    latest, cur = parse(info.get("latest")), parse(__version__)
    sha, url = z.get("sha256"), z.get("url")
    if latest is None or cur is None:
        _set(state="error", msg="Unexpected reply from the update server.")
    elif latest <= cur:  # never offer an older or equal build
        _set(state="current", msg=f"Up to date ({__version__}).", info=info)
    elif not (isinstance(url, str) and isinstance(sha, str) and len(sha) == 64 and _trusted(url)):
        _set(state="error", msg="The update server sent no usable download.")
    else:
        _set(state="available", info=info, msg=f"Version {info['latest']} is available.")


def _check_zip(path):
    """Raise NetError unless the zip is an OpenBillow package newer than the running copy."""
    try:
        with zipfile.ZipFile(path) as zf, zf.open("blender_manifest.toml") as f:
            m = tomllib.load(f)
    except (zipfile.BadZipFile, KeyError, OSError, ValueError):
        raise net.NetError("The download is not an OpenBillow package; nothing was installed.") from None
    new, cur = parse(m.get("version")), parse(__version__)
    if m.get("id") != "openbillow" or new is None or cur is None or new <= cur:
        raise net.NetError("The download is not a newer OpenBillow; nothing was installed.")


def _download():
    z = _s["info"]["zip"]
    req = urllib.request.Request(z["url"], headers={"User-Agent": net.user_agent()})
    h, n, done, path = hashlib.sha256(), 0, False, None
    try:
        try:
            fd, path = tempfile.mkstemp(suffix=".zip", prefix="openbillow_update_")
            with os.fdopen(fd, "wb") as out, _open(req, timeout=60) as r:
                while chunk := r.read(1 << 20):
                    n += len(chunk)
                    if n > MAX_ZIP:
                        raise net.NetError("Download failed: file too large")
                    h.update(chunk)
                    out.write(chunk)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            raise net.NetError(f"Download failed: {e}") from None
        if h.hexdigest() != z["sha256"].lower():
            raise net.NetError("Download failed its checksum; nothing was installed.")
        _check_zip(path)
        done = True
    finally:
        if path and not done:
            try:
                os.remove(path)
            except OSError:
                pass
    _set(state="ready", zip=path, msg="Downloaded and verified. Click Install.")


def _installed_version(repo):
    """Version in the installed manifest of the local repository `repo`, or None if unreadable."""
    try:
        r = next(r for r in bpy.context.preferences.extensions.repos if r.module == repo)
        with open(os.path.join(r.directory, "openbillow", "blender_manifest.toml"), "rb") as f:
            return parse(tomllib.load(f).get("version"))
    except Exception:
        return None


def _zip_version(path):
    try:
        with zipfile.ZipFile(path) as zf, zf.open("blender_manifest.toml") as f:
            return parse(tomllib.load(f).get("version"))
    except Exception:
        return None


def _install_end(**kw):
    """Record the result here and in the new version's module, which replaces this one when enabled."""
    _set(**kw)
    getattr(sys.modules.get(__name__), "_s", _s).update(kw)
    net.redraw()


def _install_now():
    """Timer body, never inside an operator of this add-on: Blender unregisters the add-on's classes
    to replace it, which frees a running operator. The zip leaves _s first because unregister() deletes it."""
    path, repo, _s["zip"] = _s["zip"], _s["repo"], None
    want = _zip_version(path)
    try:
        r = bpy.ops.extensions.package_install_files(
            filepath=path, repo=repo, enable_on_install=True, overwrite=True)
    except Exception as e:
        # Blender may have written the files and then failed to unregister the running copy.
        _install_end(state="error", msg=f"Install reported an error ({str(e).strip()[:120]}). The files may "
                                        "have been updated: restart Blender and check the version.")
    else:
        # Blender returns FINISHED even when it could not extract the zip: check what is on disk.
        if "FINISHED" in r and (want is None or _installed_version(repo) != want):
            _install_end(state="error", msg="Blender did not report the update as installed. Restart Blender "
                                            "and check the version; see the console.")
        elif "FINISHED" in r:
            _install_end(state="installed", zip=None, msg="Installed. Restart Blender to use the new version.")
        else:
            _install_end(state="error", msg="Blender did not install the update; see the console.")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return None


def _start(target, state, msg, *args):
    if target is _check:
        _drop_zip()
    _set(state=state, msg=msg)
    t = net.run(target, _s, *args)
    net.wait_redraw(_s, ("checking", "downloading"))
    return t


def _auto_check():
    """Once per session, shortly after start-up, only with online access on."""
    if not _s["auto_done"] and bpy.app.online_access:
        try:
            from .props import prefs
            if not prefs().auto_update_check:
                _s["auto_done"] = True
                return None
        except Exception:
            pass
        _s["auto_done"] = True
        _start(_check, "checking", "Checking for updates...", _payload())


# ---- operators -------------------------------------------------------------
class OPENBILLOW_OT_update_check(Operator):
    bl_idname = "openbillow.update_check"
    bl_label = "Check for Updates"
    bl_description = "Look online for a newer OpenBillow. Only the OpenBillow version, the Blender version and the system type are sent."

    def execute(self, context):
        if not bpy.app.online_access:
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        _start(_check, "checking", "Checking for updates...", _payload())
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
        # Blender disables the running copy to replace it, and that stops every bake.
        if any(r["proc"].poll() is None for r in ops._runs.values()):
            self.report({"ERROR"}, "A bake is running. Cancel it or wait, then install the update.")
            return {"CANCELLED"}
        repo = _install_repo()
        if repo is None:
            self.report({"ERROR"}, "This copy was not installed from a zip; update it where you got it")
            return {"CANCELLED"}
        _set(state="installing", repo=repo, msg="Installing...")
        bpy.app.timers.register(_install_now, first_interval=0.1, persistent=True)
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
    sub.enabled = on and st not in ("checking", "downloading", "installing")
    sub.operator("openbillow.update_check", icon="FILE_REFRESH")
    if st == "available":
        row.operator("openbillow.update_download", text=f"Download {_s['info']['latest']}", icon="IMPORT")
    elif st == "ready":
        row.operator("openbillow.update_install", icon="CHECKMARK")
    try:
        from .props import prefs
        box.prop(prefs(), "auto_update_check")
    except Exception:
        pass
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
    _drop_zip()
    if bpy.app.timers.is_registered(_auto_check):
        bpy.app.timers.unregister(_auto_check)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
