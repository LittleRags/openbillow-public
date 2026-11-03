# SPDX-License-Identifier: GPL-3.0-or-later
"""Licence activation. The key is activated on this computer through the support API; the server
allows at most MAX new computers (distinct system identifiers) per key in any 12 months, and is the authority: it
checks the identifier again on every presets download and support request. The add-on only
remembers the result (`activated_for`) so it can enable or grey out the buttons.

The identifier is a SHA-256 of the operating system's machine ID (Linux /etc/machine-id, Windows
MachineGuid, macOS IOPlatformUUID), salted with "openbillow". The raw ID never leaves the
computer, and nothing else about the computer is sent for activation."""
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
import uuid

import bpy
from bpy.types import Operator

from .openbillow import __version__
from .props import prefs

API = "https://support.pipelineflow.co.uk"
SITE = "https://pipelineflow.co.uk"
LICENCE_URL = SITE + "/legal/licence/"
TERMS_URL = SITE + "/legal/terms/"
PRIVACY_URL = SITE + "/legal/privacy/"
MAX = 5   # new computers per key in 12 months (the server enforces it; shown in the text)
_result = {"state": "idle", "msg": "", "done": None}  # idle | busy | ok | error
_cache = {}


# ---- system identifier -----------------------------------------------------
def _raw_machine_id():
    if sys.platform.startswith("linux"):
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(path, encoding="ascii") as fh:
                    v = fh.read().strip()
                if v:
                    return v
            except OSError:
                pass
    elif sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
                                0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                return str(winreg.QueryValueEx(k, "MachineGuid")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True,
                                 text=True, timeout=5).stdout
            m = re.search(r'"IOPlatformUUID" = "([^"]+)"', out)
            if m:
                return m.group(1)
        except (OSError, subprocess.SubprocessError):
            pass
    # No system ID: a random one kept in the user's config folder (a new Blender config counts as a new computer).
    path = os.path.join(bpy.utils.user_resource("CONFIG", create=True), "openbillow_machine_id")
    try:
        with open(path, encoding="ascii") as fh:
            v = fh.read().strip()
        if v:
            return v
    except OSError:
        pass
    v = uuid.uuid4().hex
    try:
        with open(path, "w", encoding="ascii") as fh:
            fh.write(v)
    except OSError:
        pass
    return v


def machine_id():
    """64 hex characters, stable on this computer, not reversible to the system ID."""
    if "id" not in _cache:
        _cache["id"] = hashlib.sha256(("openbillow:" + _raw_machine_id()).encode()).hexdigest()
    return _cache["id"]


def token(p):
    """What `activated_for` holds when this key and email are activated on this computer."""
    key, email = p.support_key.strip(), p.support_email.strip().lower()
    if not key or not email:
        return ""
    return hashlib.sha256(f"{key}|{email}|{machine_id()}".encode()).hexdigest()


def is_activated(p=None):
    p = p or prefs()
    return bool(p.activated_for) and p.activated_for == token(p)


# ---- server calls ----------------------------------------------------------
def _call(action, key, email):
    """POST /activate. Runs in a thread; the result is applied by `_poll` on the main thread."""
    body = {"licence_key": key, "email": email, "machine_id": machine_id()}
    req = urllib.request.Request(f"{API}/{action}", json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": f"OpenBillow/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw, ok = r.read(), True
    except urllib.error.HTTPError as e:
        raw, ok = e.read(), False
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        _result["done"] = (action, False, {"message": f"Could not reach the licence server: {e}"})
        return
    try:
        j = json.loads(raw)
    except ValueError:
        j = {}
    _result["done"] = (action, ok, j)


def _apply():
    """Main thread: turn the finished call into prefs state and a message."""
    action, ok, j = _result["done"]
    _result["done"] = None
    p = prefs()
    used = j.get("used")
    if ok:
        p.activated_for = token(p)
        msg = f"Activated on this computer ({used} of {j.get('max', MAX)} activations used in the last 12 months)."
    else:
        msg = j.get("message") or "Unexpected reply from the licence server."
        if j.get("error") in ("key_invalid", "key_revoked", "email_mismatch"):
            p.activated_for = ""
    _result.update(state="ok" if ok else "error", msg=msg)


def _poll():
    if _result["done"] is None:
        return 0.3
    _apply()
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            a.tag_redraw()
    return None


# ---- operators -------------------------------------------------------------
def _start(op, action):
    p = prefs()
    key, email = p.support_key.strip(), p.support_email.strip()
    if not bpy.app.online_access:
        op.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
    elif not key:
        op.report({"ERROR"}, "Enter your licence key")
    elif "@" not in email:
        op.report({"ERROR"}, "Enter your email address")
    elif not p.accept_terms:
        op.report({"ERROR"}, "Accept the Licence and Terms of Service first")
    elif _result["state"] == "busy":
        return {"CANCELLED"}
    else:
        _result.update(state="busy", msg="Contacting the licence server...", done=None)
        threading.Thread(target=_call, args=(action, key, email), daemon=True).start()
        bpy.app.timers.register(_poll, first_interval=0.3)
        return {"FINISHED"}
    return {"CANCELLED"}


class OPENBILLOW_OT_activate(Operator):
    bl_idname = "openbillow.activate"
    bl_label = "Activate"
    bl_description = (f"Activate your licence key on this computer. A key allows up to {MAX} new computers "
                      "in any 12 months; only a one-way hash of your system ID is sent")

    def execute(self, context):
        return _start(self, "activate")


def draw(layout):
    """Links, accept checkbox, Activate / Deactivate and the status line (Support box)."""
    p = prefs()
    on = is_activated(p)
    if not on:
        row = layout.row()
        row.operator("wm.url_open", text="Licence", icon="URL").url = LICENCE_URL
        row.operator("wm.url_open", text="Terms of Service", icon="URL").url = TERMS_URL
        row.operator("wm.url_open", text="Privacy", icon="URL").url = PRIVACY_URL
        layout.prop(p, "accept_terms")
    if on:
        layout.label(text="Licence activated on this computer.", icon="CHECKMARK")
    else:
        row = layout.row()
        row.enabled = bpy.app.online_access and _result["state"] != "busy" and p.accept_terms
        row.operator("openbillow.activate", icon="KEY_HLT")
        layout.label(text=f"Not activated. Presets and support need an activated key ({MAX} new computers per 12 months).",
                     icon="ERROR")
    if _result["msg"]:
        for line in textwrap.wrap(_result["msg"], 90):
            layout.label(text=line, icon="CHECKMARK" if _result["state"] == "ok" else "NONE")


_classes = (OPENBILLOW_OT_activate,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
