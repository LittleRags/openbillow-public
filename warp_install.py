# SPDX-License-Identifier: GPL-3.0-or-later
"""First-use download of NVIDIA Warp with pip (site build only). The Extensions-platform build
bundles Warp as a wheel instead and leaves this file out."""
import os
import subprocess
import sys
import tempfile
import threading

import bpy
from bpy.types import Operator

from .ops import _use_deps, deps_dir

WARP_PIP = "warp-lang==1.18.0"  # tools/build_extension.py reads this and WARP_HASHES as text
WARP_HASHES = (  # the two wheels the add-on supports: Linux x64 and Windows x64
    "sha256:090962bfae7dd22ade0d8e0258f41635e0e58980f15bcc9e7a7d312819dfe440",
    "sha256:1769e55746739f2e244a2b412123003c462ec3d977dfb5f420f924a8553d30aa",
)
PIP_TIMEOUT = 1800  # seconds; a stalled connection must not leave the panel at "Downloading"
_install = {"state": "idle", "msg": ""}  # idle | running | done | error


def _pip(args):
    # --isolated: PIP_* variables and the user's pip.conf do not decide where the native code comes from
    cmd = [sys.executable, "-m", "pip", "--isolated", "--no-input", "--disable-pip-version-check", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PIP_TIMEOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out; check your connection and try again") from None
    if r.returncode:
        out = (r.stdout + r.stderr).strip()
        if "No module named pip" in out:
            raise RuntimeError("Blender's Python has no pip; install Warp by hand, see the README")
        if "DO NOT MATCH THE HASHES" in out:
            raise RuntimeError("The downloaded GPU engine failed its checksum test and was not installed")
        raise RuntimeError(out.splitlines()[-1] if out else "pip failed")


def _install_worker(target):
    try:
        with tempfile.TemporaryDirectory(prefix="openbillow_warp_") as d:
            req = os.path.join(d, "requirements.txt")
            with open(req, "w") as fh:
                fh.write(" ".join((WARP_PIP, *(f"--hash={h}" for h in WARP_HASHES))) + "\n")
            _pip(["install", "--target", target, "--no-deps", "--upgrade", "--only-binary=:all:",
                  "--require-hashes", "-r", req])
        _install.update(state="done", msg="")
    except Exception as e:  # shown in the panel
        _install.update(state="error", msg=str(e)[:200])


class OPENBILLOW_OT_install_warp(Operator):
    bl_idname = "openbillow.install_warp"
    bl_label = "Install GPU Engine"
    bl_description = "Download the graphics-card library OpenBillow needs (about 105 MB). Needed once."

    def execute(self, context):
        if not bpy.app.online_access:
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        if _install["state"] == "running":
            return {"CANCELLED"}
        _install.update(state="running", msg="")
        threading.Thread(target=_install_worker, args=(deps_dir(),), daemon=True).start()

        def poll():
            if _install["state"] == "running":
                for w in bpy.context.window_manager.windows:
                    for a in w.screen.areas:
                        a.tag_redraw()
                return 1.0
            _use_deps()
            return None
        bpy.app.timers.register(poll, first_interval=1.0, persistent=True)
        return {"FINISHED"}


def register():
    bpy.utils.register_class(OPENBILLOW_OT_install_warp)


def unregister():
    bpy.utils.unregister_class(OPENBILLOW_OT_install_warp)
