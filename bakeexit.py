# SPDX-License-Identifier: GPL-3.0-or-later
"""How a finished bake subprocess ended: exit code text and the last stdout line.
No bpy import, so tests can load it with plain Python."""
import os

EXIT_TEXT = {1: "error", 3: "out of memory", 4: "settings changed or folder in use"}


def describe(code):
    if code < 0:
        return f"killed by signal {-code}"
    return EXIT_TEXT.get(code, f"exit {code}")


def tail_line(path, n=8192):
    """Last non-empty line of a text file, or ''."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    return lines[-1] if lines else ""


def result(folder, code, pid):
    return {"exit": code, "pid": pid, "tail": tail_line(os.path.join(folder, "bake.stdout"))}


def failed(code):
    return code not in (0, 2)
