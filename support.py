# SPDX-License-Identifier: GPL-3.0-or-later
"""Support button: collects diagnostics, shows the exact JSON before sending, posts it to
the support API. The add-on enforces nothing; the server decides who is entitled. The Warp
probe runs in the CLI subprocess (this module never imports Warp)."""
import json
import os
import textwrap
import time
import traceback

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import activation, diagnostics, errlog, known_issues, net, ops
from .diagnostics import MAX_DIAG, MAX_LOG, collect, scrub  # noqa: F401  (re-exported for tests)
from .props import prefs

API = "https://support.pipelineflow.co.uk/tickets"
SITE = "https://pipelineflow.co.uk"
EMAIL = "support@pipelineflow.co.uk"
TEXT_NAME = "OpenBillow support request"
_result = {"state": "idle", "msg": ""}  # idle | sending | ok | error
_pending = {}  # what the preview showed, which is exactly what Send posts

KB_URL = "https://openbillow.com/support/known_issues.json"
FAQ = "https://openbillow.com/support/faq/#"
KB_LIMIT = 256 * 1024
KB_TIMEOUT = 5
KB_RETRY_S = 3600
MAX_SHOWN = 3
MAX_COUNTS = 64
MAX_COUNT = 1_000_000
CHECKS_KEY = "known_issue_checks"
_kb = {"state": "idle", "msg": "", "data": None, "at": None}  # idle | fetching | error; data is the newest good file
_shown = []  # the known problems the open dialog lists
_counted = set()  # ids already counted in this session, so pressing the button twice is one find
_sent = {}  # counts that went out with the ticket being posted


def _drop_preview():
    """Remove the preview text block and the pending payload: the block holds the email address and
    machine id and would otherwise be saved into the .blend (CODE-ADD-022)."""
    _pending.clear()
    _shown.clear()
    txt = bpy.data.texts.get(TEXT_NAME)
    if txt is not None:
        bpy.data.texts.remove(txt)


def build_payload(context):
    p = prefs()
    diag, log = collect(context, p.support_attach_log, p.support_attach_errors)
    return {"licence_key": p.support_key.strip(), "email": p.support_email.strip(),
            "machine_id": activation.machine_id(), "description": p.support_description.strip(),
            "diagnostics": diag, "log": log}


def _online():
    return bpy.app.online_access


# ---- known problems --------------------------------------------------------
def _fetch_kb():
    """Runs in a thread (net.run). A missing, damaged or newer file leaves the copy we already have."""
    code, raw = net.get(KB_URL, timeout=KB_TIMEOUT, limit=KB_LIMIT, what="openbillow.com")
    kb = known_issues.parse(raw) if code == 200 else None
    if kb is not None:
        _kb["data"] = kb
    _kb["state"] = "idle"


def _refresh_kb():
    """Start a background fetch of the published list, if online access allows it. Only ever
    called from a button the user pressed. Never waits: the check uses what is here already."""
    try:
        recent = _kb["at"] is not None and time.monotonic() - _kb["at"] < KB_RETRY_S
        if not _online() or _kb["state"] == "fetching" or recent:
            return None
        _kb.update(state="fetching", at=time.monotonic())
        return net.run(_fetch_kb, _kb)
    except Exception:  # noqa: BLE001 - the check is optional, a ticket is not
        _kb["state"] = "idle"
        errlog.record("ERROR", traceback.format_exc())
        return None


def _find_matches(diag, log):
    """Known problems for these diagnostics, from the published list or the bundled copy.
    Never raises: a fault here must not get in the way of a ticket."""
    try:
        kb = _kb["data"] or known_issues.bundled()
        return known_issues.matches(kb, dict(diag, log=log)) if kb else []
    except Exception:  # noqa: BLE001
        errlog.record("ERROR", traceback.format_exc())
        return []


def _counts(p):
    """{issue id: finds since the last ticket}, stored in the preferences. Damaged text counts as none."""
    try:
        j = json.loads(p.support_issue_counts)
    except (ValueError, TypeError):
        return {}
    if not isinstance(j, dict):
        return {}
    return {k: v for k, v in j.items()
            if known_issues.ID.fullmatch(k) and isinstance(v, int) and not isinstance(v, bool) and v > 0}


def _count_matches(matches):
    """Add this check's finds, at most once per problem per Blender session."""
    try:
        new = [m["id"] for m in matches if m["id"] not in _counted]
        if not new:
            return
        p = prefs()
        c = _counts(p)
        for i in new:
            if i in c or len(c) < MAX_COUNTS:
                c[i] = min(c.get(i, 0) + 1, MAX_COUNT)
        p.support_issue_counts = json.dumps(c)
        _counted.update(new)
    except Exception:  # noqa: BLE001
        errlog.record("ERROR", traceback.format_exc())


def _attach_counts(payload, matches):
    """Ids and whole numbers only. Deflections for an id = counts - (1 if it is in matched)."""
    try:
        counts, d = _counts(prefs()), payload["diagnostics"]
        if not counts and not matches:
            return
        d[CHECKS_KEY] = {"matched": [m["id"] for m in matches], "counts": counts}
        while len(json.dumps(d)) > diagnostics.MAX_DIAG and d.get("domain"):
            d["domain"].popitem()
        if len(json.dumps(d)) > diagnostics.MAX_DIAG:
            del d[CHECKS_KEY]
    except Exception:  # noqa: BLE001
        payload["diagnostics"].pop(CHECKS_KEY, None)
        errlog.record("ERROR", traceback.format_exc())


def _after_send():
    """Main thread, after the post. Counts that reached the server are not reported again."""
    sent = dict(_sent)
    _sent.clear()
    if _result["state"] != "ok" or not sent:
        return
    try:
        p = prefs()
        left = {k: v - sent.get(k, 0) for k, v in _counts(p).items() if v > sent.get(k, 0)}
        p.support_issue_counts = json.dumps(left)
        _counted.clear()
    except Exception:  # noqa: BLE001 - the ticket has already gone
        errlog.record("ERROR", traceback.format_exc())


def _draw_matches(layout, matches):
    box = layout.box()
    box.label(text="This looks like a known problem. Try this first:", icon="INFO")
    for m in matches[:MAX_SHOWN]:
        box.separator()
        for i, line in enumerate(textwrap.wrap(m["symptom"], 90)):
            box.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
        for line in textwrap.wrap(m["cause"], 90):
            box.label(text=line)
        for n, step in enumerate(m["fix"], 1):
            for i, line in enumerate(textwrap.wrap(step, 86)):
                box.label(text=f"{n}. {line}" if i == 0 else "    " + line)
        box.operator("wm.url_open", text="Read more", icon="URL").url = FAQ + m["id"]
    if len(matches) > MAX_SHOWN:
        box.label(text=f"{len(matches) - MAX_SHOWN} more found. Fix these first, then check again.")


# ---- sending ---------------------------------------------------------------
def _post(payload):
    """Runs in a thread (net.run)."""
    try:
        ok, j = net.post_json(API, payload, what="the support server")
    except net.NetError as e:
        raise net.NetError(f"{e}. Use Save Diagnostics and email the file to {EMAIL}.") from None
    msg = str(j.get("message") or "Unexpected reply from the support server.")
    if ok and j.get("ticket"):
        msg = f"{msg} (ticket {j['ticket']})"
    _result.update(state="ok" if ok else "error", msg=msg)


# ---- operators -------------------------------------------------------------
def _check(op, p):
    if not activation.is_activated(p):
        op.report({"ERROR"}, "Activate your licence key first (Support box above)")
    elif not p.support_key.strip():
        op.report({"ERROR"}, "Enter your licence key")
    elif "@" not in p.support_email:
        op.report({"ERROR"}, "Enter your email address")
    elif not p.support_description.strip():
        op.report({"ERROR"}, "Describe the problem")
    else:
        return True
    return False


def _check_setup(context):
    """Collect, match and count. -> (payload, matches)."""
    _refresh_kb()
    payload = build_payload(context)
    matches = _find_matches(payload["diagnostics"], payload["log"])
    _count_matches(matches)
    _shown[:] = matches
    return payload, matches


def _prepare(context):
    """Fill _pending with what Send will post."""
    payload, matches = _check_setup(context)
    _attach_counts(payload, matches)
    _pending.clear()
    _pending.update(payload)


class OPENBILLOW_OT_support_check(Operator):
    bl_idname = "openbillow.support_check"
    bl_label = "Check for Known Problems"
    bl_description = "Compare this computer's diagnostics with the list of known problems. Nothing about you is sent"

    def invoke(self, context, event):
        _check_setup(context)
        return context.window_manager.invoke_props_dialog(self, width=640, confirm_text="Close")

    def draw(self, context):
        if _shown:
            _draw_matches(self.layout, _shown)
        else:
            self.layout.label(text="No known problem matches this computer.", icon="CHECKMARK")

    def execute(self, context):
        return {"FINISHED"}


class OPENBILLOW_OT_support_send(Operator):
    bl_idname = "openbillow.support_send"
    bl_label = "Send Support Request"
    bl_description = "Show you exactly what will be sent, then send your support request."

    def invoke(self, context, event):
        p = prefs()
        if not _online():
            self.report({"ERROR"}, "Online access is off (Preferences > System > Network)")
            return {"CANCELLED"}
        if not _check(self, p):
            return {"CANCELLED"}
        _prepare(context)
        shown = dict(_pending, licence_key="(your key)")
        txt = bpy.data.texts.get(TEXT_NAME) or bpy.data.texts.new(TEXT_NAME)
        txt.from_string(json.dumps(shown, indent=2))
        return context.window_manager.invoke_props_dialog(self, width=640,
                                                          confirm_text="Send anyway" if _shown else "Send")

    def draw(self, context):
        if _shown:
            _draw_matches(self.layout, _shown)
        col = self.layout.column(align=True)
        col.label(text="This will be sent to support.pipelineflow.co.uk:", icon="URL")
        col.label(text=f"email: {_pending['email']}")
        for line in textwrap.wrap(_pending["description"], 90)[:6]:
            col.label(text=line)
        col.separator()
        for k, v in _pending["diagnostics"].items():
            if k == "recent_errors":
                col.label(text=f"recent errors: {len(v)} lines" if v else "recent errors: none")
                continue
            if k == CHECKS_KEY:
                col.label(text=f"known problems: {len(v['matched'])} found now, "
                               f"{sum(v['counts'].values())} since your last request (counts only)")
                continue
            if isinstance(v, dict):
                col.label(text=f"{k}: {len(v)} settings (numbers and switches only)")
                continue
            for i, line in enumerate(textwrap.wrap(f"{k}: {v}", 90) or [k]):
                col.label(text=line if i == 0 else "    " + line)
        lg = _pending["log"]
        col.label(text=f"bake log: {len(lg.splitlines())} lines" if lg else "bake log: not attached")
        col.separator()
        col.label(text=f"The full JSON is in the Text Editor as '{TEXT_NAME}' until you send or cancel.")

    def execute(self, context):
        if not _pending:
            return {"CANCELLED"}
        payload = dict(_pending)
        _sent.clear()
        _sent.update(((payload.get("diagnostics") or {}).get(CHECKS_KEY) or {}).get("counts") or {})
        _drop_preview()
        _result.update(state="sending", msg="Sending...")
        net.run(_post, _result, payload)
        net.wait_redraw(_result, "sending", then=_after_send)
        return {"FINISHED"}

    def cancel(self, context):
        _drop_preview()


class OPENBILLOW_OT_support_save(Operator):
    bl_idname = "openbillow.support_save"
    bl_label = "Save Diagnostics to File"
    bl_description = f"Write the same information to a JSON file you can email to {EMAIL}"
    filepath: StringProperty(subtype="FILE_PATH", default="openbillow_diagnostics.json")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        p = prefs()
        payload = build_payload(context)
        payload.pop("licence_key")
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.report({"INFO"}, f"Saved. Email it to {EMAIL} with your licence key and a description.")
        return {"FINISHED"}


def draw_box(layout):
    p = prefs()
    box = layout.box()
    box.label(text="Support", icon="QUESTION")
    row = box.row()
    row.operator("wm.url_open", text="How support works").url = SITE + "/support/"
    col = box.column()
    col.prop(p, "support_key")
    col.prop(p, "support_email")
    activation.draw(box)
    col = box.column()
    col.prop(p, "support_description")
    col.prop(p, "support_attach_log")
    col.prop(p, "support_attach_errors")
    box.operator("openbillow.support_check", icon="VIEWZOOM")  # free and offline users too
    row = box.row()
    sub = row.row()
    sub.enabled = _online() and _result["state"] != "sending" and activation.is_activated(p)
    sub.operator("openbillow.support_send", icon="EXPORT")
    row.operator("openbillow.support_save", icon="FILE_TICK")
    if not _online():
        box.label(text="Online access is off (Preferences > System > Network). Save diagnostics and email them.",
                  icon="ERROR")
    if _result["msg"]:
        for line in textwrap.wrap(_result["msg"], 90):
            box.label(text=line, icon="CHECKMARK" if _result["state"] == "ok" else "NONE")


_classes = (OPENBILLOW_OT_support_check, OPENBILLOW_OT_support_send, OPENBILLOW_OT_support_save)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
