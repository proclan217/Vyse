"""Tool-result cache: expensive or repetitive read-only results (system info, searches, directory listings).

Entries expire after the tool's own TTL and are dropped when a write tool declares it makes them stale
(`Tool.invalidates`). The cache never stores failures, and a result is only reused for the exact same arguments.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


def make_key(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


class ToolCache:
    def __init__(self, max_entries: int = 256, clock: Callable[[], float] | None = None) -> None:
        self.max_entries = max(1, max_entries)
        self._clock = clock or time.monotonic
        self._data: OrderedDict[str, tuple[float, str, Any]] = OrderedDict()   # key -> (expires, tool, result)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = make_key(tool, args)
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < self._clock():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(item[2])

    def put(self, tool: str, args: dict[str, Any], result: dict[str, Any], ttl: float) -> None:
        if ttl <= 0 or not result.get("ok", True):
            return
        key = make_key(tool, args)
        with self._lock:
            self._data[key] = (self._clock() + ttl, tool, copy.deepcopy(result))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def invalidate(self, tools: tuple[str, ...] | list[str]) -> int:
        names = set(tools)
        with self._lock:
            doomed = [k for k, (_, t, _) in self._data.items() if t in names]
            for k in doomed:
                del self._data[k]
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"entries": len(self._data), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
