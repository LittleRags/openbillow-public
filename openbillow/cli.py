# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line: `openbillow_bake.py bake <folder> [--resume] [--frame-end N]
[--device cuda:0] [--checkpoint-every 10] [--keep-checkpoints 1]`.

Bake folder layout: bake.json + objects.npz in,
vdb/openbillow_####.vdb, state/state_####.npz, status.json, bake.log (and guide/####.npy
for guide parents) out;
bake.lock while running, CANCEL to ask a running bake to stop.
Exit codes: 0 done, 2 cancelled, 3 out of VRAM, 4 bad settings / stale, 1 other.
"""
import argparse
import dataclasses
import glob
import json
import os
import re
import socket
import sys
import time
import traceback

import numpy as np

from . import __version__, usd_io

EXIT_DONE, EXIT_ERROR, EXIT_CANCELLED, EXIT_VRAM, EXIT_SETTINGS = 0, 1, 2, 3, 4


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
    readers (and a killed bake) never see a partial file."""
    d, name = os.path.split(path)
    tmp = os.path.join(d, f".{name}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    _replace(tmp, path)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _write_json(path, doc):
    _replace_write(path, lambda fh: fh.write(json.dumps(doc, indent=1).encode()))


def _frames_in(folder, pattern):
    """{frame: path} for files named <prefix>####.<ext> matching pattern."""
    out = {}
    for p in glob.glob(os.path.join(folder, pattern)):
        m = re.search(r"(\d+)\.\w+$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def _pid_alive(pid):
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


class Lock:
    """bake.lock = {"pid", "host"}. A lock from a dead pid on this host is taken
    over; a live one, or one from another host, refuses the bake."""

    def __init__(self, path):
        self.path = path
        self.held = False

    def acquire(self):
        me = {"pid": os.getpid(), "host": socket.gethostname()}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.path) as fh:
                        other = json.load(fh)
                except (OSError, ValueError):
                    other = {}
                if other.get("host") == me["host"] and not _pid_alive(int(other.get("pid", -1))):
                    os.remove(self.path)  # stale lock from a dead bake here
                    continue
                raise BakeError(f"bake folder is locked by pid {other.get('pid')} "
                                f"on {other.get('host')} ({self.path})")
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            self.held = True
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
        self.fh = open(path, "a")

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

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

    def checkpoint(self, sol, h):
        st = sol.save_state()
        path = os.path.join(self.state_dir, f"state_{sol.frame:04d}.npz")
        _replace_write(path, lambda fh: np.savez(fh, settings_hash=np.array(h), **st))
        keep = max(1, self.args.keep_checkpoints)
        for f, p in sorted(_frames_in(self.state_dir, "state_*.npz").items())[:-keep]:
            os.remove(p)

    def find_checkpoint(self, h, frame_end):
        """Newest checkpoint whose settings hash matches -> (frame, path), or None.
        Raises if there are checkpoints but none match (settings changed)."""
        found = sorted(_frames_in(self.state_dir, "state_*.npz").items(), reverse=True)
        for f, p in found:
            with np.load(p) as z:
                if str(z["settings_hash"]) == h:
                    return f, p
        if found:
            raise BakeError("settings changed since the checkpoints were written; "
                            "free the bake to re-bake", EXIT_SETTINGS)
        return None

    def run(self, log):
        from .bakefile import read_bake, settings_hash
        from .backend import get_backend
        from .guiding import guide_frame_path
        from .liquid import LiquidSolver
        from .solver import SmokeSolver
        from .vdb_io import write_liquid_vdb, write_ply, write_vdb

        args = self.args
        try:
            settings, f0, f1 = read_bake(self.folder)
            h = settings_hash(self.folder)
        except (OSError, KeyError, ValueError, TypeError) as e:
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
        if os.path.exists(self.cancel_path):  # left over from an old cancel
            os.remove(self.cancel_path)
        self.set_status(frame_start=f0, frame_end=f1, settings_hash=h, frame=f0 - 1)

        start = self.find_checkpoint(h, f1) if args.resume else None
        if not args.resume:  # fresh bake: drop old output
            for p in [*_frames_in(self.vdb_dir, "openbillow_*.vdb").values(),
                      *_frames_in(self.vdb_dir, "particles_*.ply").values(),
                      *_frames_in(self.vdb_dir, "particles_snd_*.usdc").values(),
                      *_frames_in(self.state_dir, "state_*.npz").values(),
                      *_frames_in(self.guide_dir, "????.npy").values()]:
                os.remove(p)
            _remove(os.path.join(self.vdb_dir, "particles_snd.usdc"))

        from . import estimate
        obs = settings.estimate_kwargs()
        need = estimate.estimate_bytes(settings.size, settings.resolution_max, **obs)
        b = _pick_backend(args.device, need, log, managed=getattr(args, "managed", False))
        m0 = b.memory()
        base_used = m0.total - m0.free  # other processes (Blender, desktop) + our CUDA context
        mode = "cpu" if b.on_cpu else ("managed" if getattr(b, "managed", False) else "gpu")
        self.set_status(device=b.device_name, device_mode=mode)
        log(f"OpenBillow {__version__}, device {b.device_name} ({mode}), frames {f0}-{f1}, "
            f"resolution_max {settings.resolution_max}")
        mem = b.memory()
        if mem.total and need > mem.free:  # the context already exists here, so only grids are missing
            res = estimate.max_resolution_for(
                mem.free / 2**20 + estimate.FIXED_MB, settings.size, **obs)
            what = "RAM" if b.on_cpu else ("VRAM + RAM" if mode == "managed" else "VRAM")
            raise BakeError(
                f"needs about {need / 2**20:.0f} MB of {what} but only {mem.free / 2**20:.0f} MB "
                f"are free on {b.device_name}; try resolution_max {res} or free memory",
                EXIT_VRAM)
        liquid = is_liquid(settings)
        snd = liquid and bool(settings.use_spray_particles or settings.use_foam_particles
                              or settings.use_bubble_particles)
        if snd and not usd_io.have_pxr():
            raise BakeError("secondary particles need the 'pxr' (USD) module, which Blender's "
                            "bundled Python has; run the bake with it", EXIT_SETTINGS)
        sol = (LiquidSolver if liquid else SmokeSolver)(b, settings, frame_start=f0)
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
        for folder, pattern in ((self.vdb_dir, "openbillow_*.vdb"), (self.vdb_dir, "particles_*.ply"),
                                (self.vdb_dir, "particles_snd_*.usdc"), (self.guide_dir, "????.npy")):
            for f, p in _frames_in(folder, pattern).items():
                if f > sol.frame:  # stale frames after the resume point
                    os.remove(p)

        times = []
        while sol.frame < f1:
            if os.path.exists(self.cancel_path):
                return self.cancel(sol, h, log)
            st = sol.step_frame()
            path = os.path.join(self.vdb_dir, f"openbillow_{st.frame:04d}.vdb")
            tmp = os.path.join(self.vdb_dir, f".openbillow_{st.frame:04d}.vdb.tmp")
            if liquid:
                write_liquid_vdb(tmp, b.to_numpy(sol.phi), b.to_numpy(sol.vel),
                                 settings.min_corner, sol.dx, band=sol.NARROW_BAND,
                                 surface=sol.surface(), mesh_scale=sol.mesh_scale,
                                 wall_layers=sol.mesh_wall_layers, vel_scale=sol.vel_scale)
                if settings.output_particles:
                    ply = os.path.join(self.vdb_dir, f"particles_{st.frame:04d}.ply")
                    _replace_write(ply, lambda fh: write_ply(fh, *sol.particles_world()))
            else:
                write_vdb(tmp, sol.output(sol.density), sol.output_temperature(),
                          sol.output(sol.vel), sol.output_min_corner, sol.dx,
                          flame=sol.output(sol.flame) if sol.fire else None,
                          vel_scale=sol.vel_scale,
                          noise=(b.to_numpy(sol.noise.density),
                                 b.to_numpy(sol.noise.flame) if sol.fire else None,
                                 sol.noise.upres) if sol.noise else None)
            if snd:
                usd = os.path.join(self.vdb_dir, f"particles_snd_{st.frame:04d}.usdc")
                tmp_usd = os.path.join(self.vdb_dir, f".particles_snd_{st.frame:04d}.usdc")
                usd_io.write_frame(tmp_usd, st.frame, sol.secondary_world(),
                                   settings.sndparticle_width, settings.fps)
                _replace(tmp_usd, usd)
            if settings.write_guide:  # MAC velocity for child domains
                vel = sol.guide_velocity()
                _replace_write(guide_frame_path(self.folder, st.frame), lambda fh: np.save(fh, vel))
            _replace(tmp, path)
            if (st.frame - f0 + 1) % max(1, args.checkpoint_every) == 0 and st.frame < f1:
                self.checkpoint(sol, h)
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
                + f"{st.seconds:.2f} s")
        self.checkpoint(sol, h)
        if snd:
            frames = _frames_in(self.vdb_dir, "particles_snd_*.usdc")
            types = [t for t, on in (("spray", settings.use_spray_particles),
                                     ("foam", settings.use_foam_particles),
                                     ("bubble", settings.use_bubble_particles)) if on]
            out = os.path.join(self.vdb_dir, "particles_snd.usdc")
            tmp_usd = os.path.join(self.vdb_dir, ".particles_snd.usdc")
            _remove(tmp_usd)
            size = usd_io.merge_frames(tmp_usd, frames, types, settings.sndparticle_width, settings.fps)
            _replace(tmp_usd, out)
            log(f"secondary particles: {len(frames)} frames, {size / 2**20:.0f} MB in particles_snd.usdc")
        self.set_status(state="done", eta_s=0, message="")
        log("done")
        return EXIT_DONE

    def cancel(self, sol, h, log):
        if sol.frame >= sol.frame_start:
            self.checkpoint(sol, h)
        try:
            os.remove(self.cancel_path)
        except FileNotFoundError:
            pass
        self.set_status(state="cancelled", eta_s=None, message=f"cancelled after frame {sol.frame}")
        log(f"cancelled after frame {sol.frame}")
        return EXIT_CANCELLED


def _pick_backend(device, need, log, managed=False):
    """`auto` (or empty) follows estimate.device_plan: the GPU if the bake fits in free VRAM,
    else the GPU with managed memory paging through system RAM, else the CPU (slower, uses
    system RAM). When nothing fits it returns the option with the most room, so the caller's
    over-limit check names the largest resolution. `auto:cuda:1` picks the GPU to try. Any
    other value is passed to the backend as is. `managed` forces managed memory (tests)."""
    from . import estimate
    from .backend import get_backend
    device = device or "auto"
    if not device.startswith("auto"):
        return get_backend("warp", device, managed=managed)
    gpu = device[5:] or "cuda"
    if managed:
        return get_backend("warp", gpu, managed=True)
    try:
        b = get_backend("warp", gpu)
    except Exception as e:  # noqa: BLE001 - no CUDA device or driver
        log(f"no usable GPU ({type(e).__name__}: {str(e)[:120]}); running on the CPU")
        return get_backend("warp", "cpu")
    vram_free = b.memory().free
    ram_free = estimate.system_memory()[1]
    ok = estimate.managed_ok()
    plan = estimate.device_plan(need, vram_free, ram_free, ok)
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
        return get_backend("warp", gpu, managed=True)
    log(f"needs about {need / 2**20:.0f} MB but only {vram_free / 2**20:.0f} MB of VRAM are free on "
        f"{name}; running on the CPU (slower)")
    return get_backend("warp", "cpu")


def _is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "cudaerrormemoryallocation" in s


def cmd_bake(args):
    folder = os.path.abspath(args.folder)
    if not os.path.isfile(os.path.join(folder, "bake.json")):
        print(f"no bake.json in {folder}", file=sys.stderr)
        return EXIT_SETTINGS
    lock = Lock(os.path.join(folder, "bake.lock"))
    log = Log(os.path.join(folder, "bake.log"))
    bake = Bake(folder, args)
    try:
        try:
            lock.acquire()
            return bake.run(log)
        except BakeError as e:
            code, msg = e.code, str(e)
        except KeyboardInterrupt:
            code, msg = EXIT_CANCELLED, "interrupted"
        except Exception as e:  # noqa: BLE001 — report everything through status.json
            code = EXIT_VRAM if _is_oom(e) else EXIT_ERROR
            msg = f"{type(e).__name__}: {e}"
            log(traceback.format_exc().rstrip())
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
    except (OSError, KeyError, ValueError, TypeError) as e:
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
    from . import estimate
    try:
        import warp as wp
    except ImportError:
        print(json.dumps({"warp_installed": False, "devices": []}))
        return EXIT_DONE
    wp.init()
    devs = [{"name": d.name, "alias": d.alias, "total_mb": d.total_memory // 2**20,
             "free_mb": d.free_memory // 2**20,
             "default_limit_mb": estimate.default_limit_mb(d.total_memory)}
            for d in wp.get_cuda_devices()]
    print(json.dumps({"warp_installed": True, "devices": devs}))
    return EXIT_DONE


def cmd_diagnostics(args):
    """Warp/GPU facts for the support button (JSON). Never raises: failures become device_init_error."""
    import subprocess
    out = {"warp_version": None, "cuda_available": False, "device_init_error": None, "gpu_name": None,
           "compute_capability": None, "vram_total_mb": None, "vram_free_mb": None,
           "driver_version": None, "cuda_driver_version": None, "kernel_cache_exists": False,
           "kernel_cache_mb": None}
    try:
        import warp as wp
        out["warp_version"] = wp.__version__
        wp.init()
        out["cuda_available"] = bool(wp.is_cuda_available())
        if out["cuda_available"]:
            d = wp.get_cuda_device(args.device) if args.device else wp.get_cuda_device()
            out.update(gpu_name=d.name, compute_capability="sm_%d" % d.arch,
                       vram_total_mb=d.total_memory // 2**20, vram_free_mb=d.free_memory // 2**20)
            try:
                v = wp.context.runtime.core.wp_cuda_driver_version()
                out["cuda_driver_version"] = "%d.%d" % (v // 1000, v % 1000 // 10)
            except Exception:  # older/newer Warp without this call
                pass
        c = wp.config.kernel_cache_dir
        if c and os.path.isdir(c):
            out["kernel_cache_exists"] = True
            out["kernel_cache_mb"] = round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(c)
                                               for f in fs) / 2**20, 1)
    except Exception as e:
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


def main(argv=None):
    p = argparse.ArgumentParser(prog="openbillow_bake.py", description="OpenBillow smoke bake")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bake", help="bake a folder written by the add-on")
    b.add_argument("folder")
    b.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    b.add_argument("--frame-end", type=int, help="override frame_end from bake.json")
    b.add_argument("--device", help="auto (default: GPU, or the CPU when the bake does not fit), auto:cuda:1, cpu, or cuda:0")
    b.add_argument("--managed", action="store_true", help=argparse.SUPPRESS)  # force managed memory (tests)
    b.add_argument("--checkpoint-every", type=int, default=10, metavar="N")
    b.add_argument("--keep-checkpoints", type=int, default=1, metavar="N")
    e = sub.add_parser("estimate", help="print the VRAM estimate for a bake folder (JSON)")
    e.add_argument("folder")
    e.add_argument("--resolution", type=int, help="override resolution_max")
    e.add_argument("--limit-mb", type=float, help="also report fits / largest resolution")
    sub.add_parser("devices", help="list CUDA devices with memory (JSON)")
    dg = sub.add_parser("diagnostics", help="GPU/Warp facts for support requests (JSON)")
    dg.add_argument("--device", default="")
    args = p.parse_args(argv)
    return {"bake": cmd_bake, "estimate": cmd_estimate, "devices": cmd_devices, "diagnostics": cmd_diagnostics}[args.cmd](args)
