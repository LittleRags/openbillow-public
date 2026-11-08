# SPDX-License-Identifier: GPL-3.0-or-later
"""First-use download of NVIDIA Warp with pip (site build only). The Extensions-platform build
bundles Warp as a wheel instead and leaves this file out."""
import subprocess
import sys
import threading

import bpy
from bpy.types import Operator

from .ops import _use_deps, deps_dir

WARP_PIP = "warp-lang==1.18.0"
_install = {"state": "idle", "msg": ""}  # idle | running | done | error


def _pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "pip failed")


def _install_worker(target):
    try:
        try:
            _pip(["--version"])
        except RuntimeError:
            subprocess.run([sys.executable, "-m", "ensurepip", "--user"], capture_output=True, check=True)
        _pip(["install", "--target", target, "--no-deps", "--upgrade", WARP_PIP])
        _install.update(state="done", msg="")
    except Exception as e:  # shown in the panel
        _install.update(state="error", msg=str(e)[:200])


class OPENBILLOW_OT_install_warp(Operator):
    bl_idname = "openbillow.install_warp"
    bl_label = "Install GPU Engine"
    bl_description = "Download the graphics-card library OpenBillow needs (about 105 MB). Needed once."

    def execute(self, context):
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
        bpy.app.timers.register(poll, first_interval=1.0)
        return {"FINISHED"}


def register():
    bpy.utils.register_class(OPENBILLOW_OT_install_warp)


def unregister():
    bpy.utils.unregister_class(OPENBILLOW_OT_install_warp)
