# SPDX-License-Identifier: GPL-3.0-or-later
"""Known-issue rules: match the published list of problems against the diagnostics, on this computer.
Rules are data and are only ever compared, never executed. Format 1 is documented in
OpenBillowSite api/README.md. A rule that cannot be understood does not match, and nothing here
raises on bad input. No bpy import, so tests can load it with plain Python."""
import json
import operator
import os
import re

FORMAT = 1
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MAX_TEXT = 200   # build.py enforces the same limit on the published file
MAX_STEPS = 8

_ORDER = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
_COMPARE = {"eq": operator.eq, "ne": operator.ne, **_ORDER}
_NULL = {"is_null", "not_null"}
_VERSION = {"version_lt", "version_ge"}
_OPS = set(_COMPARE) | _NULL | _VERSION | {"contains_any"}
_NUMBER = re.compile(r"\s*v?(\d+(?:\.\d+)*)")

_bundled = {}  # one entry once read, so the file is opened once


def parse(raw):
    """File contents -> the knowledge base, or None if it cannot be used (damaged or a newer format)."""
    try:
        kb = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(kb, dict):
        return None
    fmt = kb.get("format")
    if not isinstance(fmt, int) or isinstance(fmt, bool) or not 1 <= fmt <= FORMAT:
        return None
    return kb if isinstance(kb.get("fields"), dict) and isinstance(kb.get("issues"), list) else None


def bundled():
    """The copy shipped in the add-on, for offline use. None if it is missing or damaged."""
    if "kb" not in _bundled:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_issues.json"), "rb") as fh:
                _bundled["kb"] = parse(fh.read())
        except OSError:
            _bundled["kb"] = None
    return _bundled["kb"]


def _kind(v):
    """Values of different kinds never compare equal or ordered (Python would say False == 0)."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    return "text" if isinstance(v, str) else None


def _version(v):
    """'0.5.4-early-access' -> [0, 5, 4]. None if it does not start with a number."""
    m = _NUMBER.match(v) if isinstance(v, str) else None
    return [int(x) for x in m.group(1).split(".")] if m else None


def _version_lt(a, b):
    n = max(len(a), len(b))
    return a + [0] * (n - len(a)) < b + [0] * (n - len(b))


def _understood(cond, fields):
    """False for an unknown field or op, which fails the whole rule (even inside "any")."""
    if not isinstance(cond, dict):
        return False
    field, op = cond.get("field"), cond.get("op")
    if not isinstance(field, str) or field not in fields or not isinstance(op, str) or op not in _OPS:
        return False
    if "value_field" in cond:
        other = cond["value_field"]
        return op in _COMPARE and isinstance(other, str) and other in fields
    return op in _NULL or "value" in cond


def _contains_any(have, wanted):
    if not isinstance(wanted, list):
        return False
    texts = [have] if isinstance(have, str) else have if isinstance(have, list) else []
    texts = [t.lower() for t in texts if isinstance(t, str)]
    return any(w.lower() in t for w in wanted if isinstance(w, str) and w for t in texts)


def _holds(cond, data):
    v, op = data.get(cond["field"]), cond["op"]
    if op in _NULL:
        return (v is None) == (op == "is_null")
    if v is None:
        return False
    other = data.get(cond["value_field"]) if "value_field" in cond else cond.get("value")
    if op == "contains_any":
        return _contains_any(v, other)
    if op in _VERSION:
        a, b = _version(v), _version(other)
        return a is not None and b is not None and _version_lt(a, b) == (op == "version_lt")
    kind = _kind(v)
    if kind is None or kind != _kind(other):
        return False
    if op in _ORDER and kind != "number":
        return False
    return bool(_COMPARE[op](v, other))


def _detected(detect, fields, data):
    if not isinstance(detect, dict):
        return False
    modes = [m for m in ("all", "any") if m in detect]
    if len(modes) != 1 or not isinstance(detect[modes[0]], list) or not detect[modes[0]]:
        return False
    conds = detect[modes[0]]
    if not all(_understood(c, fields) for c in conds):
        return False
    return (all if modes[0] == "all" else any)(_holds(c, data) for c in conds)


def _applies(scope, data):
    """`applies_to`. A key this version cannot check means the fix may not be for this user: no match."""
    if not isinstance(scope, dict) or set(scope) - {"addon_min", "addon_max", "os"}:
        return False  # includes "packs": the diagnostics do not say which packs are installed
    have = _version(data.get("openbillow_version"))
    for key, want_lt in (("addon_min", True), ("addon_max", False)):
        if scope.get(key):
            limit = _version(scope[key])
            if have is None or limit is None:
                return False
            if _version_lt(have, limit) if want_lt else _version_lt(limit, have):
                return False
    if scope.get("os"):
        names = scope["os"]
        if not isinstance(names, list):
            return False
        if not _contains_any(data.get("os") or "", names):
            return False
    return True


def _fit_to_show(e):
    """Only entries that can be drawn safely reach the user."""
    def text(s):
        return isinstance(s, str) and 0 < len(s) <= MAX_TEXT

    return (isinstance(e, dict) and isinstance(e.get("id"), str) and bool(ID.fullmatch(e["id"]))
            and text(e.get("symptom")) and text(e.get("cause")) and isinstance(e.get("fix"), list)
            and 0 < len(e["fix"]) <= MAX_STEPS and all(text(s) for s in e["fix"]))


def matches(kb, data):
    """Entries of `kb` whose rule holds for `data` (diagnostics plus "log"), in file (priority) order."""
    fields = kb["fields"]
    return [e for e in kb["issues"] if _fit_to_show(e) and _detected(e.get("detect"), fields, data)
            and _applies(e.get("applies_to"), data)]
