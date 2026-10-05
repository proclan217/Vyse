"""Context manager: keeps prompts small and fast.

Two jobs: (1) shrink every tool result before it goes back to the model (long lists and strings are cut, with an
explicit note of what was omitted), and (2) fit the whole message list into a token budget by trimming the oldest,
least useful parts first: old tool results, then old conversation turns. The system prompt and the current request
are never dropped.
"""
from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def message_tokens(m: dict[str, Any]) -> int:
    n = estimate_tokens(str(m.get("content") or "")) + 4
    if m.get("tool_calls"):
        n += estimate_tokens(json.dumps(m["tool_calls"], default=str))
    return n


def total_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    n = sum(message_tokens(m) for m in messages)
    if tools:
        n += estimate_tokens(json.dumps(tools))
    return n


def _shrink(value: Any, max_items: int, max_str: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + f"...[+{len(value) - max_str} chars]"
    if isinstance(value, list):
        out = [_shrink(v, max_items, max_str) for v in value[:max_items]]
        if len(value) > max_items:
            out.append(f"...[{len(value) - max_items} more omitted]")
        return out
    if isinstance(value, dict):
        return {k: _shrink(v, max_items, max_str) for k, v in value.items()}
    return value


def compact_result(result: Any, limit: int = 1800) -> str:
    """Serialise a tool result for the model, within `limit` characters, keeping its structure readable."""
    s = json.dumps(result, default=str, ensure_ascii=False)
    if len(s) <= limit:
        return s
    if isinstance(result, dict):
        slim = dict(result)
        if "display" in slim and "items" in slim:       # `items` repeats what folders/files/display already say
            slim.pop("items")
        for items, strs in ((12, 700), (8, 400), (5, 250), (3, 150)):
            s = json.dumps(_shrink(slim, items, strs), default=str, ensure_ascii=False)
            if len(s) <= limit:
                return s
    return s[:limit] + f'..."[truncated {len(s) - limit} chars]'


def digest(content: str, keep: int = 160) -> str:
    return content if len(content) <= keep else content[:keep] + f"...[trimmed {len(content) - keep} chars]"


def fit(messages: list[dict[str, Any]], budget: int, tools: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (messages, stats) trimmed to roughly `budget` tokens. Input is not mutated."""
    msgs = [dict(m) for m in messages]
    before = total_tokens(msgs, tools)
    stats = {"before": before, "after": before, "trimmed_results": 0, "dropped_messages": 0}
    if before <= budget or not msgs:
        return msgs, stats
    last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=len(msgs) - 1)

    # 1) old tool results (not the two most recent) shrink to a digest
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    for i in tool_idx[:-2]:
        if total_tokens(msgs, tools) <= budget:
            break
        c = str(msgs[i].get("content", ""))
        if len(c) > 200:
            msgs[i]["content"] = digest(c)
            stats["trimmed_results"] += 1

    # 2) oldest conversation turns before the current request (keep system messages)
    i = 1 if msgs and msgs[0].get("role") == "system" else 0
    while total_tokens(msgs, tools) > budget and i < last_user:
        if msgs[i].get("role") in ("user", "assistant"):
            del msgs[i]
            last_user -= 1
            stats["dropped_messages"] += 1
        else:
            i += 1

    # 3) still over: shrink remaining tool results harder
    for i in [k for k, m in enumerate(msgs) if m.get("role") == "tool"]:
        if total_tokens(msgs, tools) <= budget:
            break
        c = str(msgs[i].get("content", ""))
        if len(c) > 120:
            msgs[i]["content"] = digest(c, 100)
            stats["trimmed_results"] += 1
    stats["after"] = total_tokens(msgs, tools)
    return msgs, stats
