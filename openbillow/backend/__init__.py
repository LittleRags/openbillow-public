# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU backends. The solver only talks to the `Backend` interface in `base`."""
from .base import Backend, Flags, ParticleFlags, BackendMemory, BackendOutOfMemory


def get_backend(name: str = "warp", device: str | None = None, managed: bool = False) -> Backend:
    """Create a backend by name. Imports are lazy so missing GPU libraries only
    fail when that backend is asked for."""
    if name == "warp":
        from .warp_backend import WarpBackend
        return WarpBackend(device, managed=managed)
    raise ValueError(f"unknown backend {name!r}")


def list_devices(name: str = "warp") -> list[dict]:
    """Devices of a backend as JSON-ready dicts. ImportError when its GPU library is missing."""
    if name == "warp":
        from .warp_backend import list_devices as devices
        return devices()
    raise ValueError(f"unknown backend {name!r}")


def diagnostics(name: str = "warp", device: str = "") -> dict:
    """GPU facts for the support button, only the keys it found. ImportError when the library is missing."""
    if name == "warp":
        from .warp_backend import diagnostics as facts
        return facts(device)
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["Backend", "Flags", "ParticleFlags", "BackendMemory", "BackendOutOfMemory", "diagnostics",
           "get_backend", "list_devices"]
