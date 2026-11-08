# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

# Packaged: openbillow/ and openbillow_bake.py sit next to this file. In the repo the
# add-on folder is `addon/` and they sit one level up.
_here = os.path.dirname(os.path.realpath(__file__))
ROOT = _here if os.path.isdir(os.path.join(_here, "openbillow")) else os.path.dirname(_here)

# Load the solver package as a sub-package of this add-on (<add-on>.openbillow). A
# top-level `openbillow` module or a sys.path entry inside the extension is a
# Blender extension policy violation.
if __name__ + ".openbillow" not in sys.modules:
    import importlib.util
    _dir = os.path.join(ROOT, "openbillow")
    _spec = importlib.util.spec_from_file_location(
        __name__ + ".openbillow", os.path.join(_dir, "__init__.py"), submodule_search_locations=[_dir])
    _core = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _core
    _spec.loader.exec_module(_core)

from .flavor import STORE  # noqa: E402

# Network modules exist only in the site build (see flavor.py).
_NET = ("activation", "support", "presets_online", "updates", "warp_install")
_ORDER = ("errlog", "props", "export", "ops", "diagnostics", "activation", "support", "presets", "presets_online",
          "warp_install", "updates", "ui")
_MODS = [m for m in _ORDER if not (STORE and m in _NET)]
if "_loaded" in locals():  # F8 reload
    import importlib
    for _m in _loaded:
        importlib.reload(_m)
else:
    import importlib
    _loaded = [importlib.import_module("." + m, __name__) for m in _MODS]
_by = {m.__name__.rsplit(".", 1)[1]: m for m in _loaded}
props, ops, ui = _by["props"], _by["ops"], _by["ui"]


def register():
    for m in _loaded:
        if hasattr(m, "register"):
            m.register()


def unregister():
    for m in reversed(_loaded):
        if hasattr(m, "unregister"):
            m.unregister()
