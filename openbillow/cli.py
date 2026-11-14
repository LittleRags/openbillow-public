# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line: `openbillow_bake.py bake <folder> [--resume] [--frame-end N]
[--device cuda:0] [--checkpoint-every 10] [--keep-checkpoints 1] [--break-lock]`;
other sub-commands: `estimate`, `devices`, `diagnostics`.

Bake folder layout: bake.json + objects.npz in,
vdb/openbillow_####.vdb, state/state_####.npz, status.json, bake.log (and guide/####.npy
for guide parents) out;
bake.lock while running, CANCEL to ask a running bake to stop.
Exit codes: 0 done, 2 cancelled, 3 out of VRAM, 4 bad settings / stale / command-line usage error, 1 other.
"""
import argparse
import dataclasses
import glob
import json
import os
import re
import signal
import socket
import sys
import time
import traceback
import zipfile

import numpy as np

from . import __version__, usd_io
from .bakefiles import FRAME_FILES, LOCK_NAME, LOCK_STALE_S, lock_age, read_lock
from .bakefiles import frames_in as _frames_in, pid_alive as _pid_alive

EXIT_DONE, EXIT_ERROR, EXIT_CANCELLED, EXIT_VRAM, EXIT_SETTINGS = 0, 1, 2, 3, 4
BACKEND = "warp"  # the one place a second backend changes


class BakeError(Exception):
    def __init__(self, message, code=EXIT_ERROR):
        super().__init__(message)
        self.code = code


# ---- small file helpers ----------------------------------------------------

def _replace(src, dst, tries=40):
    """os.replace that waits out Windows sharing violations: Blender holds the loaded VDB and
    polls status.json, and Windows refuses to replace a file another process has open
    (PermissionError). Retries for about 8 s, then raises."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 if i < 10 else 0.25)


def _replace_write(path, write):
    """write(fh) into a hidden temp file next to path, then os.replace, so
    readers (and a killed bake) never see a partial file. A failed or interrupted write removes its temp file."""
    d, name = os.path.split(path)
    tmp = os.path.join(d, f".{name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
        _replace(tmp, path)
    except BaseException:
        _remove(tmp)
        raise


def _remove_retry(path, tries=40):
    """os.remove with the same wait for Windows sharing violations as _replace (Blender may be reading the frame)."""
    for i in range(tries):
        try:
            os.remove(path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 if i < 10 else 0.25)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _write_json(path, doc):
    _replace_write(path, lambda fh: fh.write(json.dumps(doc, indent=1).encode()))


class Lock:
    """bake.lock = {"pid", "host"}. A lock from a dead pid on this host is taken over, and so is
    an empty or damaged one older than LOCK_STALE_S. A live one, or one from another host (its
    pid cannot be checked from here), refuses the bake. `break_foreign` (--break-lock) also takes
    a lock from another host and a damaged one of any age, never a live one on this host."""

    def __init__(self, path, break_foreign=False):
        self.path = path
        self.break_foreign = break_foreign
        self.held = False

    def acquire(self):
        me = {"pid": os.getpid(), "host": socket.gethostname()}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                other = read_lock(self.path)
                if other is None:
                    if not self.break_foreign and lock_age(self.path) <= LOCK_STALE_S:
                        raise BakeError(f"bake.lock is empty or damaged ({self.path}); if no bake "
                                        "is running, wait a minute or pass --break-lock", EXIT_SETTINGS)
                else:
                    host, pid = other
                    locked = f"bake folder is locked by pid {pid} on {host} ({self.path})"
                    if host == me["host"] and _pid_alive(pid):
                        raise BakeError(locked, EXIT_SETTINGS)
                    if host != me["host"] and not self.break_foreign:
                        raise BakeError(f"{locked}; if no bake is running on {host}, "
                                        "delete bake.lock or pass --break-lock", EXIT_SETTINGS)
                _remove(self.path)  # stale, or the user said to break it
                continue
            self.held = True  # from here on release() removes it, even if the write below fails
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            return
        raise BakeError(f"could not take the lock {self.path}")

    def release(self):
        if self.held:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self.held = False


class Log:
    """Messages go to stdout and are appended to bake.log."""

    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8", errors="replace")

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        try:  # a console that cannot encode the line must not stop the bake or hide its error
            print(line, flush=True)
        except (UnicodeError, OSError):
            pass

    def close(self):
        self.fh.close()


# ---- bake ------------------------------------------------------------------

class Bake:
    def __init__(self, folder, args):
        self.folder = folder
        self.args = args
        self.vdb_dir = os.path.join(folder, "vdb")
        self.state_dir = os.path.join(folder, "state")
        self.guide_dir = os.path.join(folder, "guide")  # only for guide parents
        self.status_path = os.path.join(folder, "status.json")
        self.cancel_path = os.path.join(folder, "CANCEL")
        self.lock_path = os.path.join(folder, LOCK_NAME)
        self.status = {"state": "running", "frame": None, "frame_start": None,
                       "frame_end": None, "last_vdb": None, "sec_per_frame": None,
                       "eta_s": None, "vram_peak_mb": 0, "settings_hash": None,
                       "pid": os.getpid(), "message": ""}

    def set_status(self, **kw):
        self.status.update(kw)
        try:
            _write_json(self.status_path, self.status)
        except PermissionError:  # Blender still has it open after the retries: status is informational
            pass

    def cancel_is_new(self):
        """True if CANCEL was made after this bake took its lock, i.e. it is meant for this bake.
        Compares two files in the bake folder, so a share's clock can't disagree with ours."""
        try:
            return os.path.getmtime(self.cancel_path) > os.path.getmtime(self.lock_path)
        except OSError:
            return False

    def interrupted(self):
        """Clean up after SIGINT/SIGTERM; returns the message saying where Resume continues."""
        for d in (self.vdb_dir, self.state_dir, self.guide_dir):
            for p in glob.glob(os.path.join(glob.escape(d), ".*")):  # half-written frame
                _remove(p)
        done = _frames_in(self.state_dir, "state_*.npz")
        return (f"interrupted; Resume continues from frame {max(done)}" if done
                else "interrupted before the first checkpoint; Resume starts from the first frame")

    def checkpoint(self, sol, hasher):
        st = sol.save_state()
        h = hasher.digest(upto_frame=sol.frame)  # what this state depends on, not later frames
        path = os.path.join(self.state_dir, f"state_{sol.frame:04d}.npz")
        _replace_write(path, lambda fh: np.savez(fh, settings_hash=np.array(h), frame_start=np.array(self.frame_start), **st))
        keep = max(1, self.args.keep_checkpoints)
        for f, p in sorted(_frames_in(self.state_dir, "state_*.npz").items())[:-keep]:
            _remove_retry(p)

    def find_checkpoint(self, hasher, legacy, frame_start, log=lambda m: None):
        """Newest checkpoint whose settings hash matches -> (frame, path), or None.
        `legacy`: the whole-file hash that checkpoints from 0.5.3 and earlier store.
        Raises if there are checkpoints but none match (settings changed)."""
        found = sorted(_frames_in(self.state_dir, "state_*.npz").items(), reverse=True)
        for f, p in found:
            try:
                with np.load(p) as z:
                    if "frame_start" in z.files:
                        ok = (str(z["settings_hash"]) == hasher.digest(upto_frame=int(z["frame"]))
                              and int(z["frame_start"]) == frame_start)
                    else:  # 0.5.3 and earlier: no start frame stored, so it must not be later than the checkpoint
                        ok = str(z["settings_hash"]) == legacy and int(z["frame"]) >= frame_start - 1
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
                log(f"checkpoint {os.path.basename(p)} is unreadable, skipped")
                continue
            if ok:
                return f, p
        if found:
            raise BakeError("settings changed since the checkpoints were written; "
                            "free the bake to re-bake", EXIT_SETTINGS)
        return None

    def run(self, log):
        from .bakefile import SettingsHasher, _legacy_hash, read_bake
        from .backend import get_backend
        from .guiding import guide_frame_path
        from .liquid import LiquidSolver
        from .solver import SimulationDiverged, SmokeSolver
        from .vdb_io import write_liquid_vdb, write_ply, write_vdb

        args = self.args
        try:
            settings, f0, f1 = read_bake(self.folder)
            hasher = SettingsHasher(self.folder)
            h = hasher.digest()
        except (OSError, KeyError, ValueError, TypeError, AttributeError, zipfile.BadZipFile) as e:
            raise BakeError(f"cannot read bake settings: {e}", EXIT_SETTINGS)
        if args.frame_end is not None:
            f1 = args.frame_end
        if f1 < f0:
            raise BakeError(f"frame_end {f1} < frame_start {f0}", EXIT_SETTINGS)
        if settings.use_guide and settings.guide_source == "DOMAIN":
            if not settings.guide_parent:
                raise BakeError("guide source DOMAIN needs a guide parent", EXIT_SETTINGS)
            parent = os.path.normpath(os.path.join(self.folder, settings.guide_parent))
            missing = [f for f in range(f0, f1 + 1) if not os.path.isfile(guide_frame_path(parent, f))]
            if missing:
                raise BakeError(f"guide parent {parent}: no guide frames {missing[0]}-{missing[-1]} "
                                f"({len(missing)} missing); bake the parent domain first", EXIT_SETTINGS)
            settings = dataclasses.replace(settings, guide_parent=parent)
        os.makedirs(self.vdb_dir, exist_ok=True)  # up front: NFS mkdir races
        os.makedirs(self.state_dir, exist_ok=True)
        if settings.write_guide:
            os.makedirs(self.guide_dir, exist_ok=True)
        for sub in ("vdb", "state", "guide"):  # leftovers of an interrupted write
            d = glob.escape(os.path.join(self.folder, sub))
            for pat in (".*.tmp", ".particles_snd*.usdc"):
                for p in glob.glob(os.path.join(d, pat)):
                    _remove(p)
        cancel_now = self.cancel_is_new()  # an older CANCEL is left over from an old cancel
        _remove(self.cancel_path)
        self.frame_start = f0
        self.set_status(frame_start=f0, frame_end=f1, settings_hash=h, frame=f0 - 1)
        if cancel_now:  # before anything below touches the previous bake's output
            self.set_status(state="cancelled", eta_s=None, message="cancelled before the first frame")
            log("cancelled before the first frame")
            return EXIT_CANCELLED

        start = self.find_checkpoint(hasher, _legacy_hash(self.folder), f0, log) if args.resume else None
        from . import estimate
        obs = settings.estimate_kwargs()
        if settings.use_guide and settings.guide_source == "DOMAIN" and settings.guide_parent:
            from .guiding import guide_frame_path
            try:  # the guide velocity grid has the parent's resolution
                obs["guide_cells"] = int(np.prod(np.load(guide_frame_path(settings.guide_parent, f0), mmap_mode="r").shape[:3]))
            except (OSError, ValueError):
                pass
        need = estimate.estimate_bytes(settings.size, settings.resolution_max, **obs)
        host = estimate.host_bytes(settings.size, settings.resolution_max, **obs)
        b = _pick_backend(args.device, need, log, managed=getattr(args, "managed", False), host=host)
        m0 = b.memory()
        base_used = m0.total - m0.free  # other processes (Blender, desktop) + our CUDA context
        mode = b.memory_mode
        self.set_status(device=b.device_name, device_mode=mode)
        log(f"OpenBillow {__version__}, device {b.device_name} ({mode}), frames {f0}-{f1}, "
            f"resolution_max {settings.resolution_max}")
        mem = b.memory()
        output = host if mode != "gpu" else 0  # frame output comes out of the same RAM as the grids
        free = max(0, mem.free - output)
        if mem.total and need > free:  # the context already exists here, so only grids are missing
            res = estimate.max_resolution_for(free / 2**20 + estimate.FIXED_MB, settings.size, **obs)
            what = {"cpu": "RAM", "managed": "VRAM + RAM"}.get(mode, "VRAM")
            raise BakeError(
                f"needs about {need / 2**20:.0f} MB of {what} but only {free / 2**20:.0f} MB "
                + (f"(after {output / 2**20:.0f} MB for writing frames) " if output else "")
                + f"are free on {b.device_name}; try resolution_max {res} or free memory",
                EXIT_VRAM)
        liquid = is_liquid(settings)
        snd = liquid and bool(settings.use_spray_particles or settings.use_foam_particles
                              or settings.use_bubble_particles)
        if snd and not usd_io.have_pxr():
            raise BakeError("secondary particles need the 'pxr' (USD) module, which Blender's "
                            "bundled Python has; run the bake with it", EXIT_SETTINGS)
        sol = (LiquidSolver if liquid else SmokeSolver)(b, settings, frame_start=f0)
        if not args.resume:  # fresh bake: drop old output, only once nothing can refuse to start
            for sub, pattern in FRAME_FILES:
                for p in _frames_in(os.path.join(self.folder, sub), pattern).values():
                    _remove_retry(p)
            _remove(os.path.join(self.vdb_dir, "particles_snd.usdc"))
        log(f"{'liquid' if liquid else 'gas'} grid {sol.shape}, dx {sol.dx:.5g} m"
            + (f", noise grid {sol.noise.shape}" if not liquid and sol.noise else ""))
        if start is not None:
            frame, path = start
            with np.load(path) as z:
                sol.load_state(z)
            log(f"resumed from checkpoint frame {frame}")
            last = os.path.join(self.vdb_dir, f"openbillow_{frame:04d}.vdb")
            self.set_status(frame=frame, last_vdb=os.path.relpath(last, self.folder)
                            if os.path.exists(last) else None)
        elif args.resume:
            log("no checkpoint, starting from frame_start")
        for sub, pattern in FRAME_FILES:
            if sub == "state":
                continue
            for f, p in _frames_in(os.path.join(self.folder, sub), pattern).items():
                if f > sol.frame:  # stale frames after the resume point
                    _remove_retry(p)

        times = []
        visc_warned = False
        first_frame = sol.frame
        while sol.frame < f1:
            if os.path.exists(self.cancel_path):
                return self.cancel(sol, hasher, log)
            try:
                st = sol.step_frame()
            except SimulationDiverged as e:  # nothing of this frame is on disk yet
                done = _frames_in(self.state_dir, "state_*.npz")
                raise BakeError(f"The simulation became unstable in frame {e.frame}. "
                                + (f"The last good checkpoint is frame {max(done)}. " if done else "")
                                + "Bake again with more Max Substeps or a lower Speed.") from e
            path = os.path.join(self.vdb_dir, f"openbillow_{st.frame:04d}.vdb")
            tmp = os.path.join(self.vdb_dir, f".openbillow_{st.frame:04d}.vdb.tmp")
            tmp_usd = os.path.join(self.vdb_dir, f".particles_snd_{st.frame:04d}.usdc")
            try:
                if liquid:
                    write_liquid_vdb(tmp, lambda: b.to_numpy(sol.phi), lambda: b.to_numpy(sol.vel),
                                     settings.min_corner, sol.dx, band=sol.NARROW_BAND,
                                     surface=sol.surface, mesh_scale=sol.mesh_scale,
                                     wall_layers=sol.mesh_wall_layers, vel_scale=sol.vel_scale)
                    if settings.output_particles:
                        ply = os.path.join(self.vdb_dir, f"particles_{st.frame:04d}.ply")
                        _replace_write(ply, lambda fh: write_ply(fh, *sol.particles_world()))
                else:
                    write_vdb(tmp, lambda: sol.output(sol.density), sol.output_temperature,
                              lambda: sol.output(sol.vel), sol.output_min_corner, sol.dx,
                              flame=(lambda: sol.output(sol.flame)) if sol.fire else None,
                              vel_scale=sol.vel_scale,
                              noise=(lambda: b.to_numpy(sol.noise.density),
                                     (lambda: b.to_numpy(sol.noise.flame)) if sol.fire else None,
                                     sol.noise.upres) if sol.noise else None)
                if snd:
                    usd = os.path.join(self.vdb_dir, f"particles_snd_{st.frame:04d}.usdc")
                    usd_io.write_frame(tmp_usd, st.frame, sol.secondary_world(),
                                       settings.sndparticle_width, settings.fps)
                    _replace(tmp_usd, usd)
                if settings.write_guide:  # MAC velocity for child domains
                    vel = sol.guide_velocity()
                    _replace_write(guide_frame_path(self.folder, st.frame), lambda fh: np.save(fh, vel))
                _replace(tmp, path)
            except BaseException:
                _remove(tmp)
                _remove(tmp_usd)
                raise
            if (st.frame - f0 + 1) % max(1, args.checkpoint_every) == 0 and st.frame < f1:
                self.checkpoint(sol, hasher)
            times = (times + [st.seconds])[-10:]
            spf = sum(times) / len(times)
            mem = b.memory()
            # only what this bake added, plus the CUDA context the estimate counts as FIXED_MB
            used = (mem.total - mem.free - base_used) // 2**20 + estimate.FIXED_MB
            peak = max(self.status["vram_peak_mb"], used)
            self.set_status(frame=st.frame, last_vdb=os.path.relpath(path, self.folder),
                            sec_per_frame=round(spf, 3), eta_s=round(spf * (f1 - st.frame), 1),
                            vram_peak_mb=peak)
            log(f"frame {st.frame}: {st.substeps} substeps, pressure iters "
                f"{st.pressure_iters}, " + (f"guiding PD iters {st.guide_iters}, " if st.guide_iters else "")
                + (f"{st.particles} particles, " if liquid else "")
                + (f"viscosity iters {st.viscosity_iters} (residual {st.viscosity_residual:.2e}), "
                   if liquid and st.viscosity_iters else "")
                + f"{st.seconds:.2f} s")
            if liquid and st.viscosity_capped and not visc_warned:
                visc_warned = True
                log(f"warning: the viscosity solve stopped at its iteration limit on frame {st.frame} without "
                    f"reaching its tolerance (residual {st.viscosity_residual:.2e}). "
                    + ("Mantaflow's solver stops the same way; " if settings.quality != "HIGH"
                       or settings.viscosity_solver != "COUPLED" else "")
                    + "shown once per bake.")
        self.checkpoint(sol, hasher)
        message = ""
        if snd:  # every frame is on disk by now: a failure here must not fail the bake
            try:
                message = self.merge_secondary(settings, f0, f1, sol.frame > first_frame, log)
            except Exception as e:  # noqa: BLE001 - disk full, USD error, destination in use
                log(traceback.format_exc().rstrip())
                message = f"Frames are complete but particles_snd.usdc failed; Resume retries the merge. ({e})"
                log(f"warning: {message}")
            _remove(os.path.join(self.vdb_dir, ".particles_snd.usdc"))  # after the except: Windows locks open files
        self.set_status(state="done", eta_s=0, message=message)
        log("done")
        return EXIT_DONE

    def merge_secondary(self, settings, f0, f1, baked, log):
        """Merge the per-frame secondary particle files into particles_snd.usdc. `baked`: this
        run wrote new frames. Returns a message for status.json ("" when all is well)."""
        frames = _frames_in(self.vdb_dir, "particles_snd_*.usdc")
        if not frames:
            log("warning: no per-frame secondary particle files found, skipping the merge")
            return ""
        out = os.path.join(self.vdb_dir, "particles_snd.usdc")
        if not baked and os.path.exists(out) and os.path.getmtime(out) > max(map(os.path.getmtime, frames.values())):
            log("secondary particles: particles_snd.usdc is up to date")
            return ""
        missing = [f for f in range(f0, f1 + 1) if f not in frames]
        if missing:  # merging what is left would replace the full file with a part of it
            message = (f"Particles not merged: {len(missing)} per-frame files are missing "
                       f"(--snd-drop-frames?). Bake again from the start.")
            log(f"warning: {message}")
            return message
        types = [t for t in usd_io.TYPES if getattr(settings, f"use_{t}_particles")]
        tmp_usd = os.path.join(self.vdb_dir, ".particles_snd.usdc")
        _remove(tmp_usd)
        size = usd_io.merge_frames(tmp_usd, frames, types, settings.sndparticle_width, settings.fps)
        _replace(tmp_usd, out)
        per_frame = sum(map(os.path.getsize, frames.values()))
        if self.args.snd_drop_frames:
            for p in frames.values():
                _remove(p)
            per_frame = 0
        log(f"secondary particles: {len(frames)} frames, {size / 2**20:.0f} MB in particles_snd.usdc, "
            f"{per_frame / 2**20:.0f} MB in per-frame files, {(size + per_frame) / 2**20:.0f} MB on disk")
        return ""

    def cancel(self, sol, hasher, log):
        if sol.frame >= sol.frame_start:
            self.checkpoint(sol, hasher)
        try:
            os.remove(self.cancel_path)
        except FileNotFoundError:
            pass
        self.set_status(state="cancelled", eta_s=None, message=f"cancelled after frame {sol.frame}")
        log(f"cancelled after frame {sol.frame}")
        return EXIT_CANCELLED


def _pick_backend(device, need, log, managed=False, host=0):
    """`auto` (or empty) follows estimate.device_plan: the GPU if the bake fits in free VRAM,
    else the GPU with managed memory paging through system RAM, else the CPU (slower, uses
    system RAM). When nothing fits it returns the option with the most room, so the caller's
    over-limit check names the largest resolution. `auto:cuda:1` picks the GPU to try. Any
    other value is passed to the backend as is. `managed` forces managed memory (tests).
    `host`: system RAM frame output needs (estimate.host_bytes)."""
    from . import estimate
    from .backend import get_backend
    device = device or "auto"
    if not device.startswith("auto"):
        return get_backend(BACKEND, device, managed=managed)
    gpu = device[5:] or "cuda"
    if managed:
        return get_backend(BACKEND, gpu, managed=True)
    try:
        b = get_backend(BACKEND, gpu)
    except Exception as e:  # noqa: BLE001 - no CUDA device or driver
        log(f"no usable GPU ({type(e).__name__}: {str(e)[:120]}); running on the CPU")
        return get_backend(BACKEND, "cpu")
    vram_free = b.memory().free
    ram_free = estimate.system_memory()[1]
    ok = estimate.managed_ok()
    plan = estimate.device_plan(need, vram_free, ram_free, ok, host)
    if plan == "none":  # does not fit anywhere: take the most room for the error message
        if ok:
            plan = "managed"
        elif ram_free > vram_free:
            plan = "cpu"
        else:
            plan = "gpu"
    if plan == "gpu":
        return b
    name = b.device_name
    del b  # drop the probe; a managed backend sets the allocator before its first array
    if plan == "managed":
        log(f"needs about {need / 2**20:.0f} MB, {vram_free / 2**20:.0f} MB of VRAM free on {name}: "
            f"paging through system RAM (about 2x slower)")
        return get_backend(BACKEND, gpu, managed=True)
    log(f"needs about {need / 2**20:.0f} MB but only {vram_free / 2**20:.0f} MB of VRAM are free on "
        f"{name}; running on the CPU (slower)")
    return get_backend(BACKEND, "cpu")


def _is_oom(e):
    from .backend import BackendOutOfMemory
    if isinstance(e, (MemoryError, BackendOutOfMemory)):
        return True
    s = str(e).lower()
    return any(k in s for k in ("out of memory", "cudaerrormemoryallocation", "failed to allocate"))


def _raise_interrupt(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # one is enough; a repeat must not break the cleanup
    raise KeyboardInterrupt


def cmd_bake(args):
    folder = os.path.abspath(args.folder)
    if not os.path.isfile(os.path.join(folder, "bake.json")):
        print(f"no bake.json in {folder}", file=sys.stderr)
        return EXIT_SETTINGS
    log = Log(os.path.join(folder, "bake.log"))
    bake = Bake(folder, args)
    lock = Lock(bake.lock_path, getattr(args, "break_lock", False))
    # Popen.terminate(), systemctl stop and farm workers send SIGTERM; end through the cleanup
    # below. On Windows terminate() can't be caught: the dead-pid takeover in Lock recovers.
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _raise_interrupt)
    try:
        try:
            lock.acquire()
            return bake.run(log)
        except BakeError as e:
            code, msg = e.code, str(e)
        except KeyboardInterrupt:
            code, msg = EXIT_CANCELLED, bake.interrupted() if lock.held else "interrupted"
        except Exception as e:  # noqa: BLE001 — report everything through status.json
            code = EXIT_VRAM if _is_oom(e) else EXIT_ERROR
            msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc().rstrip()
            if code == EXIT_VRAM:
                frame = (getattr(bake, "status", None) or {}).get("frame")
                msg = (f"Ran out of memory at frame {frame}. Lower the resolution or free "
                       f"memory; Resume continues from the last checkpoint. ({e})")
            if lock.held:  # status first: a logging failure must not hide the error from the panel
                bake.set_status(state="error", message=msg)
            log(tb)
        # A refused lock must not touch the running bake's status.json.
        if lock.held:
            bake.set_status(state="cancelled" if code == EXIT_CANCELLED else "error",
                            message=msg)
        log(msg if code == EXIT_CANCELLED else f"error: {msg}")
        return code
    finally:
        log.close()
        lock.release()


def is_liquid(settings):
    from .liquid import LiquidSettings
    return isinstance(settings, LiquidSettings)


def cmd_estimate(args):
    from . import estimate
    from .bakefile import read_bake
    try:
        settings, _, _ = read_bake(os.path.abspath(args.folder))
    except (OSError, KeyError, ValueError, TypeError, AttributeError, zipfile.BadZipFile) as e:
        print(f"cannot read bake settings: {e}", file=sys.stderr)
        return EXIT_SETTINGS
    obs = settings.estimate_kwargs()
    res = args.resolution or settings.resolution_max
    out = {"resolution_max": res, "shape": list(estimate.grid_shape(settings.size, res, obs.get("dimensions", 3))),
           "grid_mb": round(estimate.estimate_bytes(settings.size, res, **obs) / 2**20, 1),
           "total_mb": round(estimate.estimate_total_mb(settings.size, res, **obs), 1)}
    if args.limit_mb is not None:
        out["limit_mb"] = args.limit_mb
        out["fits"] = out["total_mb"] <= args.limit_mb
        out["max_resolution"] = estimate.max_resolution_for(args.limit_mb, settings.size, **obs)
    print(json.dumps(out))
    return EXIT_DONE


def cmd_devices(args):
    from .backend import list_devices
    try:
        devs = list_devices(BACKEND)
    except ImportError:
        print(json.dumps({"warp_installed": False, "devices": []}))
        return EXIT_DONE
    print(json.dumps({"warp_installed": True, "devices": devs}))
    return EXIT_DONE


def cmd_diagnostics(args):
    """Warp/GPU facts for the support button (JSON). Never raises: failures become device_init_error."""
    import subprocess
    out = {"warp_version": None, "cuda_available": False, "device_init_error": None, "gpu_name": None,
           "compute_capability": None, "vram_total_mb": None, "vram_free_mb": None,
           "driver_version": None, "cuda_driver_version": None, "kernel_cache_exists": False,
           "kernel_cache_mb": None, "numpy_path": None, "warp_path": None}
    out["numpy_path"] = getattr(np, "__file__", None)
    try:
        from .backend import diagnostics
        out.update(diagnostics(BACKEND, args.device or ""))
    except Exception as e:  # the backend module could not be imported
        out["device_init_error"] = f"{type(e).__name__}: {e}"[:500]
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            out["driver_version"] = r.stdout.strip().splitlines()[0]
        if out["cuda_driver_version"] is None:
            m = re.search(r"CUDA Version:\s*([\d.]+)", subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout)
            out["cuda_driver_version"] = m.group(1) if m else None
    except (OSError, subprocess.SubprocessError):
        pass
    print(json.dumps(out))
    return EXIT_DONE


class _Parser(argparse.ArgumentParser):
    """Usage errors exit with EXIT_SETTINGS; argparse's own 2 would read as "cancelled"."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(EXIT_SETTINGS, f"{self.prog}: error: {message}\n")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # python -I ignores PYTHONUTF8; the locale code page can fail
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    p = _Parser(prog="openbillow_bake.py", description="OpenBillow smoke bake")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    b = sub.add_parser("bake", help="bake a folder written by the add-on")
    b.add_argument("folder")
    b.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    b.add_argument("--frame-end", type=int, help="override frame_end from bake.json")
    b.add_argument("--device", help="auto (default: GPU, or the CPU when the bake does not fit), auto:cuda:1, cpu, or cuda:0")
    b.add_argument("--managed", action="store_true", help=argparse.SUPPRESS)  # force managed memory (tests)
    b.add_argument("--checkpoint-every", type=int, default=10, metavar="N")
    b.add_argument("--keep-checkpoints", type=int, default=1, metavar="N")
    b.add_argument("--snd-drop-frames", action="store_true",
                   help="delete the per-frame secondary particle files after a successful merge "
                        "(saves half the disk space; extending the bake then needs a full re-bake)")
    b.add_argument("--break-lock", action="store_true",
                   help="take over a bake.lock left by another machine; only if no bake is running there")
    e = sub.add_parser("estimate", help="print the VRAM estimate for a bake folder (JSON)")
    e.add_argument("folder")
    e.add_argument("--resolution", type=int, help="override resolution_max")
    e.add_argument("--limit-mb", type=float, help="also report fits / largest resolution")
    sub.add_parser("devices", help="list CUDA devices with memory (JSON)")
    dg = sub.add_parser("diagnostics", help="GPU/Warp facts for support requests (JSON)")
    dg.add_argument("--device", default="")
    args = p.parse_args(argv)
    return {"bake": cmd_bake, "estimate": cmd_estimate, "devices": cmd_devices, "diagnostics": cmd_diagnostics}[args.cmd](args)
