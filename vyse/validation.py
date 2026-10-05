"""Tool validation and structured errors.

Every LLM-generated tool name and argument is checked here before anything executes. Failures are returned to the
model as *structured* errors (type, message, hint, the tool's expected signature, "did you mean" suggestions) so a
small model can correct itself on the next step instead of guessing what went wrong.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .fuzzy import suggestions
from .repair import repair_arguments, repair_tool_name

MAX_STRING = 20_000
MAX_ARRAY = 200

# Error types the model (and the metrics) can rely on.
UNKNOWN_TOOL = "unknown_tool"
INVALID_JSON = "invalid_json"
MISSING_ARGUMENT = "missing_argument"
INVALID_ARGUMENT = "invalid_argument"
BLOCKED = "blocked"
DECLINED = "declined"
TOOL_FAILED = "tool_failed"
BAD_ARGUMENTS = "bad_arguments"
TIMEOUT = "timeout"
LOOP = "repeated_call"
UNRESOLVED_REF = "unresolved_reference"
RETRYABLE = {UNKNOWN_TOOL, INVALID_JSON, MISSING_ARGUMENT, INVALID_ARGUMENT, BAD_ARGUMENTS, UNRESOLVED_REF}


def signature(tool: Any) -> str:
    """'open_app(name: string)' / 'find_files(query?: string, ...)': what the model should have sent."""
    props = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))
    parts = []
    for k, spec in props.items():
        t = spec.get("type", "string")
        if spec.get("enum"):
            t = "|".join(map(str, spec["enum"]))
        parts.append(f"{k}{'' if k in required else '?'}: {t}")
    return f"{tool.name}({', '.join(parts)})"


def tool_error(kind: str, message: str, *, hint: str = "", tool: Any = None, suggestions_: list[str] | None = None,
               **extra: Any) -> dict[str, Any]:
    """The one shape every failed tool call takes. `error` stays a plain string for simple consumers."""
    out: dict[str, Any] = {"ok": False, "error": message, "error_type": kind, "retryable": kind in RETRYABLE}
    if hint:
        out["hint"] = hint
    if tool is not None:
        out["expected"] = signature(tool)
    if suggestions_:
        out["suggestions"] = suggestions_
    out.update(extra)
    return out


@dataclass
class Validated:
    args: dict[str, Any]
    error: dict[str, Any] | None = None
    repairs: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def _coerce(value: Any, spec: dict[str, Any]) -> Any:
    t = spec.get("type")
    if t == "integer" and isinstance(value, (str, float)) and not isinstance(value, bool):
        f = float(str(value).strip())
        if f != int(f):
            raise ValueError("not a whole number")
        return int(f)
    if t == "number" and isinstance(value, str):
        return float(value.strip())
    if t == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False
        raise ValueError("not a boolean")
    if t == "string" and isinstance(value, (int, float, bool)):
        return str(value)
    if t == "array":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = parsed if isinstance(parsed, list) else [value]
            except json.JSONDecodeError:
                value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list):
            raise ValueError("not a list")
        item_spec = spec.get("items") or {}
        return [_coerce(v, item_spec) for v in value] if item_spec.get("type") in ("integer", "number", "string", "boolean") else value
    if t == "object":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("not an object")
        if not isinstance(value, dict):
            raise ValueError("not an object")
    return value


def _check_type(value: Any, spec: dict[str, Any]) -> str | None:
    t = spec.get("type")
    ok = {"string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool),
          "number": isinstance(value, (int, float)) and not isinstance(value, bool),
          "boolean": isinstance(value, bool), "array": isinstance(value, list), "object": isinstance(value, dict)}
    if t in ok and not ok[t]:
        return f"expected {t}, got {type(value).__name__}"
    if isinstance(value, str):
        if "\x00" in value:
            return "contains a NUL character"
        if len(value) > MAX_STRING:
            return f"is too long ({len(value)} chars, limit {MAX_STRING})"
    if isinstance(value, list) and len(value) > MAX_ARRAY:
        return f"has too many items ({len(value)}, limit {MAX_ARRAY})"
    return None


def validate_call(tool: Any, args: dict[str, Any]) -> Validated:
    """Repair, coerce and check one call's arguments against the tool's schema."""
    params = tool.parameters
    props = params.get("properties", {})
    required = params.get("required", [])
    args, repairs = repair_arguments(params, args)
    missing = [r for r in required if r not in args or args[r] in (None, "")]
    if missing:
        return Validated(args, tool_error(
            MISSING_ARGUMENT, f"Missing required argument(s): {', '.join(missing)}. Parameters: {', '.join(props) or 'none'}.",
            hint="Call the tool again with the missing argument(s) filled in.", tool=tool, missing=missing), repairs)
    clean: dict[str, Any] = {}
    ignored: list[str] = []
    for k, v in args.items():
        if k not in props:
            ignored.append(k)               # hallucinated extra argument: dropped, but reported
            continue
        spec = props[k]
        try:
            v = _coerce(v, spec)
        except (ValueError, TypeError):
            return Validated(args, tool_error(
                INVALID_ARGUMENT, f"Argument '{k}' has the wrong type; expected {spec.get('type')}.",
                hint=f"Send '{k}' as {spec.get('type')}.", tool=tool, argument=k), repairs, ignored)
        problem = _check_type(v, spec)
        if problem:
            return Validated(args, tool_error(
                INVALID_ARGUMENT, f"Argument '{k}' is invalid: {problem}.", tool=tool, argument=k), repairs, ignored)
        enum = spec.get("enum")
        if enum and v not in enum:
            low = {str(e).lower(): e for e in enum}
            if isinstance(v, str) and v.strip().lower() in low:
                v = low[v.strip().lower()]
                repairs.append(f"fixed case of '{k}'")
            else:
                near = suggestions(str(v), [str(e) for e in enum], limit=2, threshold=60)
                return Validated(args, tool_error(
                    INVALID_ARGUMENT, f"Argument '{k}' must be one of {enum}.",
                    hint=f"Use one of: {', '.join(map(str, enum))}.", tool=tool, argument=k, suggestions_=near), repairs, ignored)
        clean[k] = v
    return Validated(clean, None, repairs, ignored)


def validate_args(tool: Any, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Back-compatible wrapper: (clean_args, error_message | None)."""
    v = validate_call(tool, args)
    return (v.args, v.error["error"]) if v.error else (v.args, None)


def resolve_tool(name: str, registry: Any) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Validate a tool *name*: exact, repaired (case/prefix/typo) or a structured 'unknown tool' error.

    Returns (tool, error_result, repair_note)."""
    names = [t.name for t in registry.all()]
    tool = registry.get(name)
    if tool is not None:
        return tool, None, None
    fixed, how = repair_tool_name(name or "", names)
    if fixed:
        return registry.get(fixed), None, how
    near = suggestions(name or "", names, limit=3, threshold=50)
    msg = f"Unknown tool '{name}'." + (f" Did you mean: {', '.join(near)}?" if near else "") + f" Available tools: {', '.join(names)}"
    return None, tool_error(UNKNOWN_TOOL, msg, hint="Pick a tool from the available list.", suggestions_=near), None
