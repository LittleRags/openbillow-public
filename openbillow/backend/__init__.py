# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU backends. The solver only talks to the `Backend` interface in `base`."""
from .base import Backend, Flags, ParticleFlags, BackendMemory


def get_backend(name: str = "warp", device: str | None = None) -> Backend:
    """Create a backend by name. Imports are lazy so missing GPU libraries only
    fail when that backend is asked for."""
    if name == "warp":
        from .warp_backend import WarpBackend
        return WarpBackend(device)
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["Backend", "Flags", "ParticleFlags", "BackendMemory", "get_backend"]
