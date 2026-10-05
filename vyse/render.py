"""Turn a tool result into the user-facing answer without asking the LLM to narrate it."""
from __future__ import annotations

from typing import Any


def _more(shown: int, total: int) -> str:
    return f" (+{total - shown} more)" if total > shown else ""


def render(tool: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        if result.get("declined"):
            return "Okay, I won't do that."
        return str(result.get("error") or "That didn't work.")
    if tool == "get_current_time":
        return f"It's {result.get('pretty') or result.get('display')}."
    if tool == "list_dir":
        folders, files = result.get("folders") or [], result.get("files") or []
        lines = [f"{result.get('display')}:"]
        if folders:
            lines.append("Folders: " + ", ".join(folders[:15]) + _more(15, len(folders)))
        if files:
            lines.append("Files: " + ", ".join(files[:15]) + _more(15, len(files)))
        return "\n".join(lines)
    if tool == "find_files":
        files = result.get("files") or []
        if not files:
            return str(result.get("display"))
        lines = [str(result.get("display")) + ":"]
        lines += [f"- {f['name']}  ({f['modified']})  {f['path']}" for f in files[:10]]
        if len(files) > 10:
            lines.append(f"... and {len(files) - 10} more")
        return "\n".join(lines)
    if tool in ("remember", "forget", "create_note", "append_note", "undo_last") and result.get("verified") is False:
        return f"{result.get('display')} (could not verify)"
    text = str(result.get("display") or "Done.")
    if result.get("verified") is False and tool in ("open_app",):
        return text          # open_app's display already says the process wasn't detected yet
    return text
