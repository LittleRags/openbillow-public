# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenBillow: a GPU smoke solver that follows Blender's Mantaflow gas algorithm.

Runs on Blender's bundled Python without bpy. Third-party packages (Warp) live
in a `deps` folder next to this package (dev checkout) or, when installed as an
extension, in the site-packages folder the add-on passes in OPENBILLOW_DEPS
(`python -I` ignores PYTHONPATH). Both are put on sys.path here.
"""
import os
import sys

__version__ = "0.5.2-early-access"

_DEPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps")
for _d in (_DEPS, os.environ.get("OPENBILLOW_DEPS", "")):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
