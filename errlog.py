# SPDX-License-Identifier: GPL-3.0-or-later
"""Ring buffer of the last warnings/errors, for the support ticket. Blender has no API to read
its own log, so we collect what we can see: the window manager's report history (all operators,
Blender's own included, via `ui.reports_to_textblock`), lines written to sys.stderr (Python
tracebacks, add-on warnings) and bake subprocess errors (record()). Python cannot wrap
`Operator.report`: the RNA function always wins over a class attribute."""
import collections
import re
import sys

MAX_LINES = 20
_buf = collections.deque(maxlen=MAX_LINES)
_orig = {}
_WORD = re.compile(r"error|warning|traceback|exception|cannot|failed", re.I)


def record(level, text):
    for line in str(text).strip().splitlines()[-4:]:
        if line.strip():
            _buf.append(f"{level}: {line.strip()[:300]}")


def _reports():
    """WARNING/ERROR lines from Blender's report history (empty without a window manager)."""
    try:
        import bpy
        before = set(bpy.data.texts)
        bpy.ops.ui.reports_to_textblock()
        new = [t for t in bpy.data.texts if t not in before]
    except Exception:
        return []
    lines = []
    for t in new:
        lines += [l.body.strip() for l in t.lines if _WORD.search(l.body)]
        bpy.data.texts.remove(t)
    return [l[:300] for l in lines if l][-MAX_LINES:]


def recent():
    return (_reports() + list(_buf))[-MAX_LINES:]


class _Tee:
    def __init__(self, stream):
        self._s = stream

    def write(self, s):
        if s and _WORD.search(s):
            record("STDERR", s)
        return self._s.write(s)

    def __getattr__(self, name):
        return getattr(self._s, name)


def install():
    if sys.stderr is not None and not isinstance(sys.stderr, _Tee):
        _orig["stderr"] = sys.stderr
        sys.stderr = _Tee(sys.stderr)


def uninstall():
    if isinstance(sys.stderr, _Tee):
        sys.stderr = _orig.pop("stderr", sys.stderr._s)
