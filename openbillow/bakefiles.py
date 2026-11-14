# SPDX-License-Identifier: GPL-3.0-or-later
"""What a bake folder holds: the bake.lock and the names of the files a bake writes.
Standard library only, so the add-on can import it inside Blender as well as the solver."""
import glob
import json
import os
import re
import socket
import time

LOCK_NAME = "bake.lock"
LOCK_STALE_S = 60  # an empty or damaged bake.lock older than this is left by a dead writer

# (sub-folder, pattern) of the files named <prefix>####.<ext>
FRAME_FILES = (("vdb", "openbillow_*.vdb"), ("vdb", "particles_*.ply"), ("vdb", "particles_snd_*.usdc"),
               ("state", "state_*.npz"), ("guide", "????.npy"))
# Half-written files of a killed bake, and the merged particle file.
OTHER_FILES = (("vdb", ".openbillow_*.vdb.tmp"), ("vdb", ".particles_*.ply.tmp"), ("vdb", ".particles_snd*.usdc"),
               ("vdb", "particles_snd.usdc"), ("state", ".state_*.npz.tmp"), ("guide", ".????.npy.tmp"))
ROOT_FILES = ("status.json", "bake.log", "bake.stdout", "CANCEL", LOCK_NAME)
OUTPUT_DIRS = ("vdb", "state", "guide")


def frames_in(folder, pattern):
    """{frame: path} for files named <prefix>####.<ext> matching pattern."""
    out = {}
    for p in glob.glob(os.path.join(glob.escape(folder), pattern)):
        m = re.search(r"(-?\d+)\.\w+$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def is_bake_folder(folder):
    return os.path.isfile(os.path.join(folder, "bake.json"))


def output_files(folder):
    """Every file a bake wrote to this folder and Free Bake may delete. The inputs (bake.json,
    objects.npz) stay, and so does anything else the user keeps in the same folders."""
    out = []
    for sub, pattern in FRAME_FILES:
        out += frames_in(os.path.join(folder, sub), pattern).values()
    for sub, pattern in OTHER_FILES:
        out += glob.glob(os.path.join(glob.escape(os.path.join(folder, sub)), pattern))
    out += [p for p in (os.path.join(folder, f) for f in ROOT_FILES) if os.path.isfile(p)]
    return out


def pid_alive(pid):
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k.GetLastError() == 5  # access denied: it exists
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        k.CloseHandle(ctypes.c_void_p(h))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(path):
    """(host, pid), or None if the file is unreadable, empty or damaged."""
    try:
        with open(path) as fh:
            other = json.load(fh)
        return str(other["host"]), int(other["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lock_age(path):
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:  # just released: let the caller retry
        return float("inf")


def folder_lock(folder):
    """None if the folder has no bake.lock, else {"pid", "host", "local", "alive"}. Same rules as
    the solver's Lock: another host's pid cannot be checked, so that lock counts as alive; pid and
    host are None for a damaged lock, which counts as alive for LOCK_STALE_S."""
    path = os.path.join(folder, LOCK_NAME)
    if not os.path.exists(path):
        return None
    other = read_lock(path)
    if other is None:
        return {"pid": None, "host": None, "local": False, "alive": lock_age(path) <= LOCK_STALE_S}
    host, pid = other
    local = host == socket.gethostname()
    return {"pid": pid, "host": host, "local": local, "alive": not local or pid_alive(pid)}
