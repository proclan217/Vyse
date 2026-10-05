"""Tools that let the model store and retrieve persistent facts about the user."""
from __future__ import annotations

from ..context import Context
from .registry import Registry, ToolError


def register(reg: Registry, ctx: Context) -> None:
    mem = ctx.memory

    @reg.tool(risk="safe", group="memory", keywords=("remember", "memorize", "save", "store", "my", "fact", "note"))
    def remember(key: str, text: str, tags: str = "") -> dict:
        """Store a lasting fact about the user or their setup, including favorites and how to open them. Reusing a key updates that fact.

        Args:
            key: Short unique label, e.g. 'printer' or 'favorite_editor'.
            text: The full fact as a sentence. For favorites include how to open it, e.g. 'Favorite show: Severance, open with open_path https://www.netflix.com/title/81152350' or 'Favorite lofi playlist on Spotify: open_path https://open.spotify.com/playlist/xyz'.
            tags: Optional comma-separated tags.
        """
        f = mem.remember(key, text, tags)
        return {"key": f.key, "verified": mem.get_fact(f.key) is not None, "display": f"Remembered: {f.text}"}

    @reg.tool(risk="safe", group="memory", keywords=("recall", "remember", "what", "my", "do", "know", "memory", "forgot"))
    def recall(query: str, limit: int = 5) -> dict:
        """Search remembered facts about the user.

        Args:
            query: What to look for, e.g. 'printer'.
            limit: Maximum facts to return.
        """
        facts = mem.recall(query, limit)
        return {"facts": [{"key": f.key, "text": f.text} for f in facts], "display": f"{len(facts)} fact(s) recalled"}

    @reg.tool(risk="write", group="memory", keywords=("forget", "remove", "erase", "memory", "fact"))
    def forget(key: str) -> dict:
        """Delete a remembered fact by its key.

        Args:
            key: The key of the fact to forget.
        """
        if not mem.forget(key):
            raise ToolError(f"No remembered fact with key '{key}'.")
        return {"verified": mem.get_fact(key) is None, "display": f"Forgot '{key}'"}
