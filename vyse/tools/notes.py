"""Markdown notes stored in the configured notes directory."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..context import Context
from .registry import Registry, ToolError


def slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower(), flags=re.U)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:60] or "note"


def register(reg: Registry, ctx: Context) -> None:
    notes_dir = ctx.cfg.notes_dir

    def ensure() -> Path:
        notes_dir.mkdir(parents=True, exist_ok=True)
        return notes_dir

    def find_note(name: str) -> Path:
        """Resolve a note by file name or title; never leaves the notes directory."""
        base = ensure()
        name = Path(name.strip()).name
        for cand in (name, name + ".md"):
            p = base / cand
            if p.is_file():
                return p
        slug = slugify(Path(name).stem)
        matches = sorted((p for p in base.glob("*.md") if slug in p.stem.lower()), reverse=True)
        if matches:
            return matches[0]
        raise ToolError(f"No note matching '{name}'.")

    @reg.tool(risk="write", group="notes", keywords=("note", "notes", "write", "jot", "save", "memo", "create", "idea"))
    def create_note(title: str, content: str, tags: str = "") -> dict:
        """Create a new Markdown note.

        Args:
            title: Short title for the note.
            content: The body text of the note (Markdown).
            tags: Optional comma-separated tags.
        """
        base = ensure()
        stamp = time.strftime("%Y-%m-%d")
        path = base / f"{stamp}-{slugify(title)}.md"
        n = 1
        while path.exists():
            n += 1
            path = base / f"{stamp}-{slugify(title)}-{n}.md"
        front = f"---\ntitle: {title}\ncreated: {time.strftime('%Y-%m-%d %H:%M')}\ntags: [{tags}]\n---\n\n"
        path.write_text(front + f"# {title}\n\n{content}\n", encoding="utf-8")
        return {"path": str(path), "verified": path.is_file(), "display": f"Created note {path.name}"}

    @reg.tool(risk="write", group="notes", keywords=("note", "notes", "append", "add", "update", "extend"))
    def append_note(name: str, content: str) -> dict:
        """Append text to an existing note.

        Args:
            name: The note's file name or part of its title.
            content: Text to add at the end.
        """
        p = find_note(name)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")
        return {"path": str(p), "verified": True, "display": f"Updated {p.name}"}

    @reg.tool(risk="safe", group="notes", keywords=("note", "notes", "search", "find", "look", "wrote", "written"))
    def search_notes(query: str, max_results: int = 10) -> dict:
        """Search all notes for the words in a query (case-insensitive, all words must match).

        Args:
            query: Words to look for.
            max_results: Maximum notes to return.
        """
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        hits: list[dict[str, Any]] = []
        for p in sorted(ensure().glob("*.md"), reverse=True):
            text = p.read_text(encoding="utf-8", errors="replace")
            low = (p.stem + " " + text).lower()
            if words and all(w in low for w in words):
                idx = min((low.find(w) for w in words if w in text.lower()), default=-1)
                body = text.replace("\n", " ")
                snippet = body[max(0, idx - 40): idx + 120] if idx >= 0 else body[:120]
                hits.append({"name": p.name, "snippet": snippet.strip()})
            if len(hits) >= max_results:
                break
        return {"count": len(hits), "notes": hits, "display": f"{len(hits)} matching note(s)"}

    @reg.tool(risk="safe", group="notes", keywords=("note", "notes", "read", "open", "show", "list"))
    def read_note(name: str = "") -> dict:
        """Read a note by file name or title. With no name, lists the most recent notes.

        Args:
            name: The note's file name or part of its title. Leave empty to list recent notes.
        """
        if not name.strip():
            names = [p.name for p in sorted(ensure().glob("*.md"), reverse=True)[:20]]
            return {"notes": names, "display": f"{len(names)} recent note(s)"}
        p = find_note(name)
        return {"name": p.name, "content": p.read_text(encoding="utf-8", errors="replace")[:6000], "display": f"Read {p.name}"}
