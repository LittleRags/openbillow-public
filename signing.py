# SPDX-License-Identifier: GPL-3.0-or-later
"""Ed25519 signature check for the update and preset index replies (CODE-API-004).

The server returns `signed` (the exact text of latest.json / manifest.json as published) and `sig`
(base64 detached Ed25519 signature over its UTF-8 bytes). The private key is kept offline by the
publisher; only the public key below is shipped. Python has no Ed25519 in the standard library,
so this is a small verify-only implementation after RFC 8032 section 5.1.7 (not constant time,
which is fine: it handles public data only). Tests: tests/test_ed25519.py (RFC 8032 vectors).
"""
import base64
import binascii
import hashlib
import json

# Raw 32-byte public key, base64. Rotation: see OpenBillowSite api/README.md.
PUBLIC_KEY = base64.b64decode("8Bvhr5YCURarhFHdEbiR9f5xXRy/SbvgCz5724/cmEc=")

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _add(a, b):
    """Point addition in extended coordinates (X, Y, Z, T)."""
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    A = (y1 - x1) * (y2 - x2) % _P
    B = (y1 + x1) * (y2 + x2) % _P
    C = 2 * t1 * t2 * _D % _P
    D = 2 * z1 * z2 % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s, pt):
    q = (0, 1, 1, 0)
    while s:
        if s & 1:
            q = _add(q, pt)
        pt = _add(pt, pt)
        s >>= 1
    return q


def _equal(a, b):
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _x(y, sign):
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    return _P - x if (x & 1) != sign else x


def _point(b):
    """Decode a 32-byte point; None if it is not on the curve."""
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


_GY = 4 * pow(5, _P - 2, _P) % _P
_G = (_x(_GY, 0), _GY, 1, _x(_GY, 0) * _GY % _P)


def verify(public, msg, sig):
    """True if `sig` (64 bytes) is a valid Ed25519 signature of `msg` under `public` (32 bytes)."""
    if not (isinstance(public, bytes) and isinstance(sig, bytes) and len(public) == 32 and len(sig) == 64):
        return False
    a, r = _point(public), _point(sig[:32])
    s = int.from_bytes(sig[32:], "little")
    if a is None or r is None or s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + public + msg).digest(), "little") % _L
    return _equal(_mul(s, _G), _add(r, _mul(h, a)))


class BadSignature(Exception):
    pass


def signed_json(reply, what):
    """Verify `reply["sig"]` over `reply["signed"]` with PUBLIC_KEY; -> the parsed signed JSON (a dict).
    Raises BadSignature with a message for the user; `what` names the data ("update", "preset index")."""
    text, sig = (reply.get("signed"), reply.get("sig")) if isinstance(reply, dict) else (None, None)
    if not isinstance(text, str) or not isinstance(sig, str) or not sig:
        raise BadSignature(f"The {what} from the server is not signed; it was refused. "
                           "Try again later, or reinstall OpenBillow from openbillow.com if this keeps happening.")
    try:
        raw = base64.b64decode(sig, validate=True)
    except (binascii.Error, ValueError):
        raw = b""
    if not verify(PUBLIC_KEY, text.encode("utf-8"), raw):
        raise BadSignature(f"The {what} from the server has an invalid signature; it was refused. "
                           "It may be a publish in progress (try again in a few minutes) or tampered data.")
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise BadSignature(f"The signed {what} could not be read; it was refused.")
    return data
