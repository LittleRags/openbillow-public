# SPDX-License-Identifier: GPL-3.0-or-later
"""Launcher: `python -I openbillow_bake.py bake <folder> ...` (see openbillow/cli.py).
-I drops the cwd from sys.path, so put the add-on folder on it here."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openbillow.cli import main  # noqa: E402  (openbillow adds deps/ to sys.path)

sys.exit(main())
