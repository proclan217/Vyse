"""Background tasks, schedules/reminders and self-inspection (stats, index) tools."""
from __future__ import annotations

from typing import Any

from ..context import Context
from ..scheduler import interval_kwargs, parse_when
from .registry import Registry, ToolError

_NO_BACKGROUND = {"start_background_task", "schedule_task", "set_reminder", "run_routine"}


def register(reg: Registry, ctx: Context) -> None:
    mem = ctx.memory

    def action_for(tool: str, args: dict[str, Any] | None, routine: str) -> dict[str, Any]:
        """A validated tool/routine action for background or scheduled execution."""
        if routine.strip():
            if mem.get_routine(routine) is None:
                known = ", ".join(x["name"] for x in mem.list_routines()) or "none"
                raise ToolError(f"No routine called '{routine}'. Saved routines: {known}.")
            return {"type": "routine", "name": routine.strip()}
        t = reg.get(tool.strip())
        if t is None:
            raise ToolError(f"Unknown tool '{tool}'.")
        if t.name in _NO_BACKGROUND:
            raise ToolError(f"'{tool}' cannot be run in the background.")
        from ..validation import validate_args
        clean, err = validate_args(t, dict(args or {}))
        if err:
            raise ToolError(str(err.get("message") or err))
        return {"type": "tool", "tool": t.name, "args": clean}

    def run_action(action: dict[str, Any]) -> dict[str, Any]:
        """Unattended: whatever needs a confirmation is declined, never silently approved."""
        if ctx.run_tool_unattended is None:
            raise RuntimeError("Tools are not available in this context.")
        if action.get("type") == "routine":
            r = mem.get_routine(action["name"])
            if r is None:
                raise RuntimeError(f"routine '{action['name']}' no longer exists")
            results = [ctx.run_tool_unattended(s["tool"], dict(s["args"])) for s in r["steps"]]
            bad = [x for x in results if not x.get("ok") or x.get("verified") is False]
            summary = f"routine '{r['name']}': {len(results) - len(bad)}/{len(results)} steps ok"
            if bad:
                raise RuntimeError(summary + f" ({bad[0].get('error') or bad[0].get('display')})")
            return {"ok": True, "display": summary}
        res = ctx.run_tool_unattended(action["tool"], dict(action.get("args") or {}))
        if not res.get("ok", True):
            raise RuntimeError(str(res.get("error") or "failed"))
        return res

    # Used by the scheduler: returns text, raises on failure.
    def run_text(action: dict[str, Any]) -> str:
        res = run_action(action)
        if not res.get("ok", True):
            raise RuntimeError(str(res.get("error") or res.get("display") or "failed"))
        return str(res.get("display") or "done")

    ctx.action_runner = run_text

    # ------------------------------------------------------------ background tasks
    @reg.tool(risk="safe", group="tasks", parallel=False, final=True,
              keywords=("background", "later", "async", "long", "run", "start", "task", "without", "waiting", "meanwhile"))
    def start_background_task(tool: str = "", args: dict | None = None, routine: str = "", name: str = "") -> dict:
        """Run a slow tool (or a saved routine) in the background so the conversation is not blocked. Returns a task id immediately; the user is told when it finishes. Anything needing confirmation is declined in the background.

        Args:
            tool: Name of the tool to run, e.g. 'find_files'. Leave empty when running a routine.
            args: The tool's arguments as an object.
            routine: Name of a saved routine to run instead of a single tool.
            name: Optional short label for the task.
        """
        action = action_for(tool, args, routine)
        label = name.strip() or (f"routine {action['name']}" if action["type"] == "routine" else action["tool"])
        task = ctx.tasks.submit(label, lambda cancel: run_action(action))
        return {"task_id": task.id, "name": label, "verified": True,
                "display": f"Started background task #{task.id} ({label}). Ask for its status any time."}

    @reg.tool(risk="safe", group="tasks", final=True, keywords=("tasks", "background", "status", "running", "progress", "jobs", "list"))
    def list_background_tasks(active_only: bool = False) -> dict:
        """List background tasks with their status (pending, running, done, failed, cancelled).

        Args:
            active_only: Only show tasks that are still running.
        """
        rows = [t.brief() for t in ctx.tasks.list(active_only)]
        return {"tasks": rows, "display": f"{len(rows)} task(s)" if rows else "No background tasks"}

    @reg.tool(risk="safe", group="tasks", final=True, keywords=("task", "status", "result", "background", "done", "finished", "progress"))
    def background_task_status(task_id: int) -> dict:
        """Status and result of one background task.

        Args:
            task_id: The task id returned by start_background_task.
        """
        t = ctx.tasks.get(int(task_id))
        if t is None:
            raise ToolError(f"No background task #{task_id}.", suggestions=[str(x.id) for x in ctx.tasks.list()][:5])
        return {**t.brief(), "result": t.summary(), "display": f"Task #{t.id} {t.status}" + (f": {t.summary()}" if t.summary() else "")}

    @reg.tool(risk="write", group="tasks", final=True, keywords=("cancel", "stop", "abort", "task", "background"))
    def cancel_background_task(task_id: int) -> dict:
        """Cancel a background task that has not finished yet.

        Args:
            task_id: The task id.
        """
        return {"display": ctx.tasks.cancel(int(task_id)), "verified": True}

    # ------------------------------------------------------------ schedules / reminders
    def sched():
        if ctx.scheduler is None:
            raise ToolError("The scheduler is not running in this session.")
        return ctx.scheduler

    @reg.tool(risk="write", group="schedule", final=True,
              keywords=("remind", "reminder", "alert", "schedule", "minutes", "tomorrow"))
    def set_reminder(text: str, when: str = "", every: str = "") -> dict:
        """Set a persistent reminder. Vyse shows it when the time comes (Vyse must be running then; missed ones are reported at the next start).

        Args:
            text: What to remind the user about.
            when: One-time: 'in 10 minutes', '18:30', 'tomorrow 09:00' or an ISO date-time.
            every: Repeating instead of when: e.g. '2 hours' or '30 minutes'.
        """
        try:
            if every.strip():
                s = sched().add(text[:40], "interval", interval_kwargs(every), {"type": "reminder", "text": text})
            elif when.strip():
                s = sched().add(text[:40], "once", {"at": parse_when(when).isoformat(timespec="seconds")}, {"type": "reminder", "text": text})
            else:
                raise ToolError("Say when ('in 10 minutes', '18:30') or how often (every '2 hours').")
        except ValueError as e:
            raise ToolError(str(e))
        nxt = sched().next_run(s["id"])
        return {"id": s["id"], "next": nxt, "verified": True, "display": f"Reminder #{s['id']} set" + (f" for {nxt}" if nxt else "")}

    @reg.tool(risk="write", group="schedule", final=True,
              keywords=("schedule", "every", "daily", "recurring", "cron", "automatically", "run", "task", "weekly", "morning", "night"))
    def schedule_task(name: str, kind: str = "daily", at: str = "", every: str = "", cron: str = "", days: str = "",
                      tool: str = "", args: dict | None = None, routine: str = "") -> dict:
        """Schedule a tool or routine to run unattended (persisted). Anything that needs a confirmation is declined when it runs, so only schedule safe actions.

        Args:
            name: Short name for the schedule.
            kind: 'once', 'interval', 'daily' or 'cron'.
            at: For once: when ('in 1 hour', '2026-10-06 09:00'). For daily: the time 'HH:MM'.
            every: For interval: e.g. '30 minutes'.
            cron: For cron: a 5-field expression, e.g. '0 9 * * mon-fri'.
            days: For daily: optional days, e.g. 'mon,wed,fri'.
            tool: Tool to run, e.g. 'system_info'. Leave empty for a routine.
            args: The tool's arguments.
            routine: Name of a saved routine to run.
        """
        action = action_for(tool, args, routine)
        try:
            if kind == "once":
                spec: dict[str, Any] = {"at": parse_when(at).isoformat(timespec="seconds")}
            elif kind == "interval":
                spec = interval_kwargs(every)
            elif kind == "daily":
                spec = {"time": at.strip(), **({"days": days.strip()} if days.strip() else {})}
            elif kind == "cron":
                spec = {"expr": cron.strip()}
            else:
                raise ToolError("kind must be once, interval, daily or cron.")
            s = sched().add(name, kind, spec, action)
        except ValueError as e:
            raise ToolError(str(e))
        nxt = sched().next_run(s["id"])
        return {"id": s["id"], "next": nxt, "verified": True, "display": f"Scheduled #{s['id']} '{name}'" + (f", next run {nxt}" if nxt else "")}

    @reg.tool(risk="safe", group="schedule", final=True, keywords=("schedules", "reminders", "scheduled", "list", "upcoming", "show", "what"))
    def list_schedules() -> dict:
        """List reminders and scheduled tasks with their next run time."""
        rows = sched().listing()
        return {"schedules": rows, "display": f"{len(rows)} schedule(s)" if rows else "Nothing scheduled"}

    @reg.tool(risk="write", group="schedule", final=True, keywords=("cancel", "delete", "remove", "stop", "reminder", "schedule", "unschedule"))
    def cancel_schedule(id: int) -> dict:
        """Delete a reminder or scheduled task.

        Args:
            id: The schedule id from list_schedules.
        """
        if not sched().remove(int(id)):
            raise ToolError(f"No schedule #{id}.", suggestions=[str(r["id"]) for r in sched().listing()][:5])
        return {"verified": True, "display": f"Cancelled schedule #{id}"}

    # ------------------------------------------------------------ introspection
    @reg.tool(risk="safe", group="stats", final=True,
              keywords=("stats", "statistics", "latency", "slow", "performance", "tokens", "failures", "errors", "metrics", "usage"))
    def vyse_stats(hours: float = 24.0) -> dict:
        """Vyse's own metrics: model/tool latency, failures, token use, routing decisions and command success rate.

        Args:
            hours: Time window in hours (default 24).
        """
        s = ctx.metrics.summary(max(0.1, float(hours)))
        c = s["commands"]
        return {**s, "display": f"Last {hours:g}h: {c['total']} command(s), {c['succeeded']} ok, {c['failed']} failed; "
                                f"model p50 {s['model']['p50_ms']} ms, {s['model']['prompt_tokens'] + s['model']['completion_tokens']} tokens"}

    @reg.tool(risk="safe", group="stats", final=True, keywords=("index", "indexed", "files", "search", "status", "refresh", "rebuild"))
    def file_index_status(rebuild: bool = False) -> dict:
        """Status of the fast file index used by find_files; optionally rebuild it in the background.

        Args:
            rebuild: Start a background rebuild.
        """
        if ctx.index is None:
            return {"enabled": False, "display": "The file index is disabled ([index] enabled = false)."}
        if rebuild and not ctx.index.building:
            ctx.index.build_async()
        st = ctx.index.stats()
        return {"enabled": True, **st, "display": f"Index: {st.get('files', 0)} file(s)" + (" (building...)" if ctx.index.building else "")}
