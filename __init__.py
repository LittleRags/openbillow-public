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

if "props" in locals():  # F8 reload
    import importlib
    for _m in (errlog, props, export, ops, activation, support, presets, updates, ui):
        importlib.reload(_m)
else:
    from . import errlog, props, export, ops, activation, support, presets, updates, ui  # noqa: F401


def register():
    props.register()
    ops.register()
    activation.register()
    support.register()
    presets.register()
    updates.register()
    ui.register()


def unregister():
    ui.unregister()
    updates.unregister()
    presets.unregister()
    support.unregister()
    activation.unregister()
    ops.unregister()
    props.unregister()
