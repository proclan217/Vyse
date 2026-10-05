"""Tool registry: @tool decorator, JSON-schema generation, relevance-based subsetting."""
from __future__ import annotations

import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

RISKS = ("safe", "write", "risky")


@dataclass
class Tool:
    name: str
    description: str
    risk: str
    fn: Callable[..., Any]
    parameters: dict[str, Any]
    keywords: tuple[str, ...] = ()
    group: str = "general"
    always: bool = False  # always exposed to the model
    # True when the tool's own result is already a complete answer: a single-intent request that
    # ends in one successful call of such tools skips the second (narration) LLM call.
    final: bool = False
    # Optional dynamic policy hook: (args) -> Decision | None. Can only make a tool stricter.
    assess: Callable[[dict[str, Any]], Any] | None = None

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolError(Exception):
    """Raised by a tool for expected failures; surfaced to the model as a structured error."""


def _json_type(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    if origin is Literal:
        vals = list(get_args(tp))
        return {"type": "string", "enum": vals}
    if origin in (list, typing.List):
        inner = get_args(tp)
        return {"type": "array", "items": _json_type(inner[0]) if inner else {"type": "string"}}
    if origin is dict or tp is dict:
        return {"type": "object"}
    return {"type": {int: "integer", float: "number", bool: "boolean", str: "string"}.get(tp, "string")}


def _parse_arg_docs(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into a summary and per-arg descriptions from an 'Args:' section."""
    lines = inspect.cleandoc(doc or "").splitlines()
    summary, args, in_args = [], {}, False
    for line in lines:
        if re.match(r"^\s*Args:\s*$", line):
            in_args = True
            continue
        if in_args:
            m = re.match(r"^\s+(\w+)\s*(?:\(.*?\))?:\s*(.*)$", line)
            if m:
                args[m.group(1)] = m.group(2).strip()
            elif not line.strip():
                in_args = False
        else:
            summary.append(line)
    return " ".join(s.strip() for s in summary if s.strip()), args


def build_schema(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    summary, arg_docs = _parse_arg_docs(fn.__doc__ or "")
    hints = typing.get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in inspect.signature(fn).parameters.items():
        if pname.startswith("_") or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        prop = _json_type(hints.get(pname, str))
        if pname in arg_docs:
            prop["description"] = arg_docs[pname]
        props[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return summary, {"type": "object", "properties": props, "required": required}


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.risk not in RISKS:
            raise ValueError(f"invalid risk {tool.risk!r} for {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def add(self, fn: Callable[..., Any], *, risk: str, name: str | None = None,
            description: str | None = None, keywords: tuple[str, ...] = (),
            group: str = "general", always: bool = False, final: bool = False,
            parameters: dict[str, Any] | None = None, assess: Callable | None = None) -> Tool:
        summary, params = build_schema(fn) if parameters is None else ("", parameters)
        return self.register(Tool(
            name=name or fn.__name__, description=description or summary or (name or fn.__name__),
            risk=risk, fn=fn, parameters=params, keywords=keywords, group=group, always=always,
            final=final, assess=assess))

    def tool(self, *, risk: str = "safe", name: str | None = None, keywords: tuple[str, ...] = (),
             group: str = "general", always: bool = False, final: bool = False, assess: Callable | None = None):
        """Decorator bound to this registry (used by tool modules so tests can use private registries)."""
        def deco(fn):
            self.add(fn, risk=risk, name=name, keywords=keywords, group=group, always=always, final=final, assess=assess)
            return fn
        return deco

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def _scored(self, query: str) -> list[tuple[int, Tool]]:
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, Tool]] = []
        for t in self._tools.values():
            if t.always:
                continue
            name_words = set(re.findall(r"[a-z0-9]+", t.name.replace("_", " ").lower()))
            kw = set(re.findall(r"[a-z0-9]+", " ".join(t.keywords).lower()))
            score = 3 * len(words & name_words) + 2 * len(words & kw) + len(words & {t.group})
            if score:
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        return scored

    def has_match(self, query: str) -> bool:
        """True if some non-always tool looks relevant to the query."""
        return bool(self._scored(query))

    def select(self, query: str, limit: int = 14) -> list[Tool]:
        """Pick tools relevant to the query: always-on tools plus best keyword/name matches.

        The best match brings its whole group (plan/apply/undo travel together); the rest
        of the budget goes to the other individually matching tools.
        """
        scored = self._scored(query)
        chosen = [t for t in self._tools.values() if t.always]
        picked: list[Tool] = []

        def add(t: Tool) -> None:
            if t not in picked and t not in chosen:
                picked.append(t)

        if scored:
            top = scored[0][1]
            add(top)
            for x in self._tools.values():
                if x.group == top.group:
                    add(x)
            for _, t in scored:
                add(t)
        if not picked:      # typo or unfamiliar wording: offer a small core set rather than nothing
            core = [self._tools[n] for n in CORE_TOOLS if n in self._tools]
            picked = [t for t in core if t not in chosen] or list(self._tools.values())
        return (chosen + picked)[:limit]


CORE_TOOLS = ("open_app", "open_path", "find_files", "list_dir", "remember", "recall")


# Global default registry used by the @tool decorator.
REGISTRY = Registry()


def tool(*, risk: str = "safe", name: str | None = None, keywords: tuple[str, ...] = (),
         group: str = "general", always: bool = False, registry: Registry | None = None):
    """Register a function as an agent tool. Schema comes from type hints + docstring."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        (registry or REGISTRY).add(fn, risk=risk, name=name, keywords=keywords, group=group, always=always)
        return fn
    return deco
