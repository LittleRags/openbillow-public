# SPDX-License-Identifier: GPL-3.0-or-later
"""Which build this is. `tools/build_extension.py --store` writes `_store_build.py`
(STORE = True) and leaves out the network modules (activation, support, updates,
presets_online, warp_install); Warp then comes from the wheel bundled in the zip."""
try:
    from ._store_build import STORE
except ImportError:
    STORE = False
