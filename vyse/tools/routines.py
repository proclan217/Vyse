"""Routines: named sequences of tool calls the user teaches Vyse ("study session = clock, timer, lofi, ...").

Nothing is hard-coded: the model composes the steps from the existing tools, and every step still runs
through the normal validation + policy + confirmation path when the routine is executed."""
from __future__ import annotations

from typing import Any

from ..context import Context
from .registry import Registry, ToolError

_NO_NEST = {"run_routine", "save_routine", "delete_routine", "list_routines"}


def register(reg: Registry, ctx: Context) -> None:
    mem = ctx.memory

    def save_routine(name: str, description: str, steps: list[dict]) -> dict:
        """Save a named routine: an ordered list of tool calls to run LATER with run_routine. Use this when the user is teaching or saving a setup ('save a routine', 'study session = ...', 'when I say X do Y'); do NOT perform the steps now. Use the exact links/names the user gave or that appear in remembered facts; never invent URLs or placeholder ids (ask_user if one is missing). Reusing a name replaces it.

        Args:
            name: Short name, e.g. 'study session'.
            description: One sentence on what it does, e.g. 'Clock, 25 min timer, lofi on Spotify, close games.'
            steps: Ordered steps, each {"tool": "<tool name>", "args": {<arguments>}}, e.g. [{"tool": "open_app", "args": {"name": "Clock"}}].
        """
        if not steps:
            raise ToolError("A routine needs at least one step.")
        from ..validation import validate_args  # same validation a live call gets
        clean = []
        for i, s in enumerate(steps, 1):
            tool = reg.get(str(s.get("tool", "")))
            if tool is None or tool.name in _NO_NEST:
                names = ', '.join(t.name for t in reg.all() if t.name not in _NO_NEST)
                raise ToolError(f"Step {i}: unknown or not allowed tool '{s.get('tool')}'. Available tools: {names}.")
            # Flat form {"tool": "open_app", "name": "Clock"} (easiest for small models) or nested {"tool":..., "args": {...}}.
            args = s["args"] if "args" in s else {k: v for k, v in s.items() if k != "tool"}
            args = args or {}
            if not isinstance(args, dict):
                raise ToolError(f"Step {i}: args must be an object.")
            checked, err = validate_args(tool, args)
            if err:
                raise ToolError(f"Step {i} ({tool.name}): {err}")
            unknown = sorted(set(args) - set(checked))
            if unknown:
                raise ToolError(f"Step {i} ({tool.name}): unknown argument(s) {', '.join(unknown)}. Use: {', '.join(tool.parameters.get('properties', {}))}.")
            clean.append({"tool": tool.name, "args": checked})
        mem.save_routine(name, description, clean)
        return {"verified": mem.get_routine(name) is not None, "display": f"Saved routine '{name.strip().lower()}' ({len(clean)} steps)"}

    reg.add(save_routine, risk="safe", group="routines",
            keywords=("routine", "session", "mode", "setup", "workflow", "whenever", "save", "teach", "macro", "shortcut"),
            parameters={"type": "object", "required": ["name", "description", "steps"], "properties": {
                "name": {"type": "string", "description": "Short name, e.g. 'study session'."},
                "description": {"type": "string", "description": "One sentence on what it does."},
                "steps": {"type": "array", "description": (
                    "Ordered tool calls. Each step is the tool name plus that tool's own arguments as sibling keys, e.g. "
                    "{\"tool\": \"open_app\", \"name\": \"Clock\"}, {\"tool\": \"set_timer\", \"minutes\": 25, \"label\": \"study\"}, "
                    "{\"tool\": \"open_path\", \"path\": \"https://...\"}, {\"tool\": \"close_app\", \"name\": \"Valorant\"}. "
                    "Fill every required argument."), "items": {
                        "type": "object", "required": ["tool"], "additionalProperties": True,
                        "properties": {"tool": {"type": "string", "description": "Tool name, e.g. open_app, open_path, set_timer, close_app."}}}}}})

    @reg.tool(risk="safe", group="routines", final=True,
              keywords=("routine", "session", "mode", "start", "begin", "run", "setup", "workflow", "go"))
    def run_routine(name: str) -> dict:
        """Run a saved routine by name. Each step still goes through safety checks and confirmations.

        Args:
            name: The routine's name.
        """
        r = mem.get_routine(name)
        if r is None:
            known = ", ".join(x["name"] for x in mem.list_routines()) or "none"
            raise ToolError(f"No routine called '{name}'. Saved routines: {known}.")
        if ctx.run_tool is None:
            raise ToolError("Routines cannot run in this context.")
        lines, failed = [], 0
        for s in r["steps"]:
            res: dict[str, Any] = ctx.run_tool(s["tool"], dict(s["args"]))
            ok = bool(res.get("ok")) and res.get("verified") is not False
            failed += not ok
            lines.append(("OK " if ok else "FAILED ") + str(res.get("display") or res.get("error") or s["tool"]))
        return {"routine": r["name"], "verified": failed == 0, "steps": lines,
                "display": f"Ran '{r['name']}': " + "; ".join(lines)}

    @reg.tool(risk="safe", group="routines", final=True, always=True, name="ask_user",
              keywords=("which", "choose", "or", "ambiguous", "clarify"))
    def ask_user(question: str) -> dict:
        """Ask the user a short clarifying question instead of guessing, e.g. when "open spotify" could mean the app or a remembered playlist. Offer the concrete options.

        Args:
            question: The question, naming the options.
        """
        return {"asked": True, "verified": True, "display": question.strip()}

    @reg.tool(risk="safe", group="routines", final=True, keywords=("routines", "list", "saved", "what", "have"))
    def list_routines() -> dict:
        """List saved routines and what they do."""
        rs = mem.list_routines()
        return {"routines": [{"name": r["name"], "description": r["description"]} for r in rs],
                "display": "; ".join(f"{r['name']}: {r['description']}" for r in rs) or "No saved routines"}

    @reg.tool(risk="write", group="routines", keywords=("forget", "delete", "remove", "routine"))
    def delete_routine(name: str) -> dict:
        """Delete a saved routine.

        Args:
            name: The routine's name.
        """
        if not mem.delete_routine(name):
            raise ToolError(f"No routine called '{name}'.")
        return {"verified": mem.get_routine(name) is None, "display": f"Deleted routine '{name}'"}
