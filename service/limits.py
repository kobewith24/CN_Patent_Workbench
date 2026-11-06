# -*- coding: utf-8 -*-
"""Generation concurrency limit.

The 4B model on an 8GB card serves one generation at a time. Requests that
arrive while a slot is busy queue here and time out with 429.

Streaming responses are sync generators; if a client disconnects, Starlette
abandons the generator and its ``finally`` may never run. A watchdog therefore
force-releases any lease held longer than RAG_HOLD_MAX seconds, so a dead
connection can delay the next request but can never leak the slot forever.
"""
import os
import threading
import time
from contextlib import contextmanager

MAX_CONCURRENT = int(os.environ.get("RAG_MAX_CONCURRENT", "1"))
QUEUE_TIMEOUT = float(os.environ.get("RAG_QUEUE_TIMEOUT", "30"))
HOLD_MAX = float(os.environ.get("RAG_HOLD_MAX", "90"))

_sem = threading.BoundedSemaphore(MAX_CONCURRENT)
_lock = threading.Lock()
_state = {
    "active": 0,
    "total": 0,
    "rejected": 0,
    "forced": 0,
    "queue_ms": [],
}


class Busy(Exception):
    pass


class _Lease:
    def __init__(self):
        self._done = False
        self._mu = threading.Lock()
        self._timer = threading.Timer(HOLD_MAX, self.force)
        self._timer.daemon = True

    def start(self):
        self._timer.start()

    def _finish(self):
        with self._mu:
            if self._done:
                return False
            self._done = True
        return True

    def release(self):
        if not self._finish():
            return
        self._timer.cancel()
        with _lock:
            _state["active"] -= 1
        _sem.release()

    def force(self):
        if not self._finish():
            return
        with _lock:
            _state["active"] -= 1
            _state["forced"] += 1
        _sem.release()


@contextmanager
def slot():
    t0 = time.time()
    if not _sem.acquire(timeout=QUEUE_TIMEOUT):
        with _lock:
            _state["rejected"] += 1
        raise Busy()
    wait = (time.time() - t0) * 1000
    lease = _Lease()
    lease.start()
    with _lock:
        _state["active"] += 1
        _state["total"] += 1
        _state["queue_ms"] = (_state["queue_ms"] + [wait])[-200:]
    try:
        yield wait
    finally:
        lease.release()


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(p * len(s)))], 1)


def snapshot():
    with _lock:
        q = list(_state["queue_ms"])
        return {
            "max_concurrent": MAX_CONCURRENT,
            "queue_timeout_s": QUEUE_TIMEOUT,
            "hold_max_s": HOLD_MAX,
            "active": _state["active"],
            "total": _state["total"],
            "rejected": _state["rejected"],
            "forced_release": _state["forced"],
            "queue_ms_p50": _pct(q, 0.5),
            "queue_ms_p95": _pct(q, 0.95),
        }


def limited(fn):
    """Decorator for synchronous endpoints that own a generation slot."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            with slot():
                return fn(*args, **kwargs)
        except Busy:
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail="server busy, retry later")

    return wrapper
