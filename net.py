# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared HTTP and worker-thread helpers for the site build's online features (activation,
support, preset download, updates). Every call ends in a state dict whose "state" leaves its busy
value, whatever the server sends or however the connection fails. Left out of the store build."""
import http.client
import json
import threading
import traceback
import urllib.error
import urllib.request

import bpy

from . import errlog
from .openbillow import __version__


class NetError(Exception):
    """A failure with a message fit to show the user."""


def user_agent():
    return f"OpenBillow/{__version__}"


def post(url, payload, timeout=20, limit=1 << 20, what="the server"):
    """POST JSON -> (status, body bytes). An HTTP error reply is returned, not raised.
    Raises NetError when the server cannot be reached, the body is cut off or exceeds `limit`."""
    req = urllib.request.Request(url, json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": user_agent()})
    return _send(req, timeout, limit, what)


def get(url, timeout=20, limit=1 << 20, what="the server"):
    """GET -> (status, body bytes), with the same failure rules as post()."""
    return _send(urllib.request.Request(url, headers={"User-Agent": user_agent()}), timeout, limit, what)


def _send(req, timeout, limit, what):
    n = 0
    try:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                try:
                    n = int(r.headers.get("Content-Length") or 0)
                except (AttributeError, TypeError, ValueError):
                    n = 0
                if n > limit:
                    raise NetError(f"Reply from {what} too large ({n} bytes)")
                raw, code = r.read(limit + 1), getattr(r, "status", None) or 200
        except urllib.error.HTTPError as e:
            raw, code = e.read(limit + 1), e.code
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        raise NetError(f"Could not reach {what}: {e}") from None
    if len(raw) > limit:
        raise NetError(f"Reply from {what} too large")
    if n and len(raw) != n:
        raise NetError(f"The connection to {what} dropped; try again")
    return code, raw


def as_dict(raw):
    """JSON object -> dict; anything else -> None."""
    try:
        j = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return j if isinstance(j, dict) else None


def post_json(url, payload, timeout=20, limit=1 << 20, what="the server"):
    """POST JSON -> (ok, dict). Raises NetError if the reply is not a JSON object."""
    code, raw = post(url, payload, timeout, limit, what)
    j = as_dict(raw)
    if j is None:
        raise NetError(f"Unexpected reply from {what}" + ("" if code < 400 else f" (error {code})") + ".")
    return code < 400, j


def guard(target, state, *args):
    """Run target(*args); any exception ends with state["state"] = "error" and a message."""
    try:
        target(*args)
    except NetError as e:
        state.update(state="error", msg=str(e))
    except Exception as e:  # noqa: BLE001 - a worker must never leave the UI busy
        state.update(state="error", msg=f"Unexpected error ({type(e).__name__}). Try again.")
        errlog.record("ERROR", traceback.format_exc())


def run(target, state, *args):
    """guard() in a daemon thread; returns the thread."""
    t = threading.Thread(target=guard, args=(target, state) + args, daemon=True)
    t.start()
    return t


def redraw():
    try:
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                a.tag_redraw()
    except Exception:  # noqa: BLE001 - no window manager (background) or a closing window
        pass


def wait_redraw(state, busy, then=None, interval=0.5):
    """Register a timer that waits while state["state"] is in `busy`, then runs `then()` (main
    thread; an exception there also ends in "error") and redraws."""
    busy = (busy,) if isinstance(busy, str) else tuple(busy)

    def tick():
        if state["state"] in busy:
            return interval
        if then is not None:
            guard(then, state)
        redraw()
        return None

    bpy.app.timers.register(tick, first_interval=interval, persistent=True)  # a file load must not strand "busy"
