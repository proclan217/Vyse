"""Observability: model latency, tool latency, failures, token usage, routing decisions, command outcomes.

Everything is appended to a small SQLite database (`metrics.db`) so `/stats` can answer "what is slow, what fails,
what does a request cost" without any external service. Recording never raises: metrics must not break the agent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_calls(
    ts REAL NOT NULL, model TEXT, latency_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
    estimated INTEGER, tools_offered INTEGER, tool_calls INTEGER, ok INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS tool_calls(
    ts REAL NOT NULL, tool TEXT, latency_ms REAL, ok INTEGER, error_type TEXT, cached INTEGER, repaired INTEGER, parallel INTEGER);
CREATE TABLE IF NOT EXISTS routing(
    ts REAL NOT NULL, request TEXT, tools TEXT, tool_count INTEGER, forced INTEGER, matched INTEGER);
CREATE TABLE IF NOT EXISTS commands(
    ts REAL NOT NULL, request TEXT, ok INTEGER, steps INTEGER, tool_calls INTEGER, failed_calls INTEGER,
    duration_ms REAL, outcome TEXT);
"""


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 1)


class Metrics:
    def __init__(self, path: Path | str = ":memory:", enabled: bool = True) -> None:
        self.enabled = enabled
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def _write(self, sql: str, params: tuple[Any, ...]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self._db.execute(sql, params)
                self._db.commit()
        except sqlite3.Error:
            pass

    # ---- recording ----
    def record_model(self, model: str, latency_ms: float, prompt_tokens: int, completion_tokens: int, *,
                     estimated: bool, tools_offered: int, tool_calls: int, ok: bool, error: str = "") -> None:
        self._write("INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (time.time(), model, latency_ms, prompt_tokens, completion_tokens, int(estimated),
                     tools_offered, tool_calls, int(ok), error[:200]))

    def record_tool(self, tool: str, latency_ms: float, ok: bool, *, error_type: str = "", cached: bool = False,
                    repaired: bool = False, parallel: bool = False) -> None:
        self._write("INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), tool, latency_ms, int(ok), error_type, int(cached), int(repaired), int(parallel)))

    def record_routing(self, request: str, tools: list[str], *, forced: bool, matched: bool) -> None:
        self._write("INSERT INTO routing VALUES (?,?,?,?,?,?)",
                    (time.time(), request[:200], json.dumps(tools), len(tools), int(forced), int(matched)))

    def record_command(self, request: str, ok: bool, steps: int, tool_calls: int, failed_calls: int,
                       duration_ms: float, outcome: str = "") -> None:
        self._write("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), request[:200], int(ok), steps, tool_calls, failed_calls, duration_ms, outcome[:200]))

    # ---- reporting ----
    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._db.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]       # type: ignore[misc]

    def summary(self, hours: float = 24.0) -> dict[str, Any]:
        since = time.time() - hours * 3600
        models = self._rows("SELECT * FROM model_calls WHERE ts>=?", (since,))
        tools = self._rows("SELECT * FROM tool_calls WHERE ts>=?", (since,))
        cmds = self._rows("SELECT * FROM commands WHERE ts>=?", (since,))
        routes = self._rows("SELECT * FROM routing WHERE ts>=?", (since,))
        lat = [m["latency_ms"] for m in models if m["ok"]]
        per_tool: dict[str, dict[str, Any]] = {}
        for t in tools:
            d = per_tool.setdefault(t["tool"], {"calls": 0, "failures": 0, "cached": 0, "lat": []})
            d["calls"] += 1
            d["failures"] += 0 if t["ok"] else 1
            d["cached"] += t["cached"]
            if not t["cached"]:
                d["lat"].append(t["latency_ms"])
        errors: dict[str, int] = {}
        for t in tools:
            if not t["ok"]:
                errors[t["error_type"] or "unknown"] = errors.get(t["error_type"] or "unknown", 0) + 1
        return {
            "window_hours": hours,
            "model": {"calls": len(models), "failures": sum(1 for m in models if not m["ok"]),
                      "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95),
                      "prompt_tokens": sum(m["prompt_tokens"] or 0 for m in models),
                      "completion_tokens": sum(m["completion_tokens"] or 0 for m in models),
                      "tokens_estimated": any(m["estimated"] for m in models)},
            "tools": {n: {"calls": d["calls"], "failures": d["failures"], "cached": d["cached"],
                          "p50_ms": _pct(d["lat"], 0.5), "p95_ms": _pct(d["lat"], 0.95)}
                      for n, d in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])},
            "tool_errors": errors,
            "repairs": sum(t["repaired"] for t in tools),
            "parallel_calls": sum(t["parallel"] for t in tools),
            "commands": {"total": len(cmds), "succeeded": sum(c["ok"] for c in cmds),
                         "failed": sum(1 for c in cmds if not c["ok"]),
                         "avg_ms": round(sum(c["duration_ms"] for c in cmds) / len(cmds), 1) if cmds else 0.0},
            "routing": {"requests": len(routes), "forced_tool": sum(r["forced"] for r in routes),
                        "no_tool_match": sum(1 for r in routes if not r["matched"]),
                        "avg_tools_offered": round(sum(r["tool_count"] for r in routes) / len(routes), 1) if routes else 0.0},
        }

    def recent_commands(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM commands ORDER BY ts DESC LIMIT ?", (limit,))

    def recent_failures(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._rows("SELECT ts, tool, error_type FROM tool_calls WHERE ok=0 ORDER BY ts DESC LIMIT ?", (limit,))

    def prune(self, days: int) -> None:
        cutoff = time.time() - days * 86400
        for table in ("model_calls", "tool_calls", "routing", "commands"):
            self._write(f"DELETE FROM {table} WHERE ts<?", (cutoff,))

    def close(self) -> None:
        try:
            self._db.close()
        except sqlite3.Error:
            pass
