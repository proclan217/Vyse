"""Persistent scheduling and reminders on APScheduler.

Definitions live in the SQLite `schedules` table; APScheduler's own job store is in memory and is rebuilt from that
table every time Vyse starts. Consequence (by design, and stated in the README): schedules survive restarts, but a job
can only fire while Vyse is running. A one-time reminder whose moment passed while Vyse was closed is reported as
missed at the next start instead of being silently dropped or fired late without saying so.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .memory import Memory

KINDS = ("once", "interval", "daily", "cron")
_UNITS = {"s": "seconds", "sec": "seconds", "second": "seconds", "m": "minutes", "min": "minutes", "minute": "minutes",
          "h": "hours", "hr": "hours", "hour": "hours", "d": "days", "day": "days", "w": "weeks", "week": "weeks"}
_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def _unit(u: str) -> str:
    u = u.lower()
    u = u[:-1] if u.endswith("s") and u != "s" else u
    if u not in _UNITS:
        raise ValueError(f"unknown time unit '{u}'")
    return _UNITS[u]


def parse_when(text: str, now: datetime | None = None) -> datetime:
    """'in 10 minutes', '2h', '18:30' (next occurrence), 'tomorrow 9:00', or an ISO date/time -> a future datetime."""
    now = now or datetime.now()
    s = text.strip().lower()
    m = re.fullmatch(r"(?:in\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)", s)
    if m:
        return now + timedelta(**{_unit(m.group(2)): float(m.group(1))})
    m = re.fullmatch(r"(today|tomorrow)?\s*(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?", s)
    if m:
        h, mi = int(m.group(2)), int(m.group(3))
        if m.group(4) == "pm" and h < 12:
            h += 12
        if m.group(4) == "am" and h == 12:
            h = 0
        if not (0 <= h < 24 and 0 <= mi < 60):
            raise ValueError(f"'{text}' is not a valid time")
        at = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if m.group(1) == "tomorrow" or (at <= now and m.group(1) != "today"):
            at += timedelta(days=1)
        return at
    try:
        return datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"Could not understand the time '{text}'. Use e.g. 'in 10 minutes', '18:30', 'tomorrow 09:00' or '2026-10-06 09:00'.")


def interval_kwargs(text: str) -> dict[str, float]:
    m = re.fullmatch(r"(?:every\s+)?(\d+(?:\.\d+)?)\s*([a-z]+)", text.strip().lower())
    if not m:
        raise ValueError(f"Could not understand the interval '{text}'. Use e.g. '30 minutes' or '2 hours'.")
    return {_unit(m.group(2)): float(m.group(1))}


def build_trigger(kind: str, spec: dict[str, Any]):
    if kind == "once":
        at = datetime.fromisoformat(spec["at"])
        return DateTrigger(run_date=at)
    if kind == "interval":
        kw = {k: spec[k] for k in ("seconds", "minutes", "hours", "days", "weeks") if k in spec}
        if not kw:
            raise ValueError("interval needs seconds/minutes/hours/days")
        if sum(kw.get(k, 0) * f for k, f in (("seconds", 1), ("minutes", 60), ("hours", 3600), ("days", 86400), ("weeks", 604800))) < 30:
            raise ValueError("Intervals shorter than 30 seconds are not allowed.")
        return IntervalTrigger(**kw)
    if kind == "daily":
        h, mi = (int(x) for x in str(spec["time"]).split(":"))
        days = str(spec.get("days") or "").lower().replace(" ", "")
        if days and not all(d in _DAYS for d in re.split(r"[,\-]", days)):
            raise ValueError("days must look like 'mon-fri' or 'sat,sun'")
        return CronTrigger(hour=h, minute=mi, day_of_week=days or "*")
    if kind == "cron":
        return CronTrigger.from_crontab(str(spec["expr"]))
    raise ValueError(f"unknown schedule kind '{kind}'")


def describe(s: dict[str, Any]) -> str:
    sp, k = s["spec"], s["kind"]
    if k == "once":
        return "once at " + sp["at"].replace("T", " ")[:16]
    if k == "interval":
        return "every " + ", ".join(f"{v:g} {u}" for u, v in sp.items())
    if k == "daily":
        return f"daily at {sp['time']}" + (f" ({sp['days']})" if sp.get("days") else "")
    return f"cron {sp.get('expr')}"


class Scheduler:
    def __init__(self, memory: Memory, runner: Callable[[dict[str, Any]], str], notify: Callable[[str], None] | None = None) -> None:
        """`runner(action) -> result text` executes a tool/routine action; `notify(text)` shows the user a message."""
        self.memory = memory
        self.runner = runner
        self.notify = notify or (lambda text: None)
        self._sched = BackgroundScheduler(daemon=True)
        self.started = False

    # ---- lifecycle ----
    def start(self) -> list[str]:
        """Start APScheduler and rebuild its jobs from the database. Returns notices about reminders missed while closed."""
        missed: list[str] = []
        if self.started:
            return missed
        self._sched.start()
        self.started = True
        now = datetime.now()
        for s in self.memory.list_schedules(only_enabled=True):
            if s["kind"] == "once" and datetime.fromisoformat(s["spec"]["at"]) <= now:
                if not s["runs"]:
                    missed.append(f"Missed while Vyse was closed ({s['spec']['at'].replace('T', ' ')[:16]}): {self._label(s)}")
                self.memory.update_schedule(s["id"], enabled=0, last_result="missed (Vyse was not running)")
                continue
            self._add_job(s)
        return missed

    def stop(self) -> None:
        if self.started:
            self._sched.shutdown(wait=False)
            self.started = False

    # ---- jobs ----
    def _add_job(self, s: dict[str, Any]) -> None:
        self._sched.add_job(self._fire, build_trigger(s["kind"], s["spec"]), args=[s["id"]], id=f"s{s['id']}",
                            replace_existing=True, misfire_grace_time=60, coalesce=True)

    @staticmethod
    def _label(s: dict[str, Any]) -> str:
        a = s["action"]
        return a.get("text") or s["name"] if a.get("type") == "reminder" else f"{s['name']} ({a.get('tool') or a.get('name')})"

    def _fire(self, sid: int) -> None:
        s = self.memory.get_schedule(sid)
        if not s or not s["enabled"]:
            return
        a = s["action"]
        try:
            result = f"Reminder: {a.get('text') or s['name']}" if a.get("type") == "reminder" else self.runner(a)
            ok = True
        except Exception as e:
            result, ok = f"{type(e).__name__}: {e}", False
        self.memory.update_schedule(sid, last_run=time.time(), runs=s["runs"] + 1, last_result=str(result)[:300],
                                    **({"enabled": 0} if s["kind"] == "once" else {}))
        if not ok:
            self.notify(f"⚠ Scheduled task '{s['name']}' failed: {result}")
        elif a.get("type") == "reminder":
            self.notify(f"⏰ {result}")
        else:
            self.notify(f"⏰ {s['name']}: {result}")

    # ---- api ----
    def add(self, name: str, kind: str, spec: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        build_trigger(kind, spec)           # validate before persisting (raises ValueError)
        if kind == "once" and datetime.fromisoformat(spec["at"]) <= datetime.now():
            raise ValueError("That time is already in the past.")
        sid = self.memory.add_schedule(name, kind, spec, action)
        s = self.memory.get_schedule(sid)
        assert s is not None
        if self.started:
            self._add_job(s)
        return s

    def remove(self, sid: int) -> bool:
        if self.started:
            try:
                self._sched.remove_job(f"s{sid}")
            except Exception:
                pass
        return self.memory.delete_schedule(sid)

    def set_enabled(self, sid: int, enabled: bool) -> bool:
        s = self.memory.get_schedule(sid)
        if not s:
            return False
        self.memory.update_schedule(sid, enabled=int(enabled))
        if self.started:
            if enabled:
                self._add_job(s)
            else:
                try:
                    self._sched.remove_job(f"s{sid}")
                except Exception:
                    pass
        return True

    def next_run(self, sid: int) -> str:
        if not self.started:
            return ""
        job = self._sched.get_job(f"s{sid}")
        return job.next_run_time.strftime("%Y-%m-%d %H:%M") if job and job.next_run_time else ""

    def listing(self) -> list[dict[str, Any]]:
        out = []
        for s in self.memory.list_schedules():
            out.append({"id": s["id"], "name": s["name"], "when": describe(s), "enabled": s["enabled"],
                        "next": self.next_run(s["id"]), "runs": s["runs"], "last_result": s["last_result"]})
        return out
