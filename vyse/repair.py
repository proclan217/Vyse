"""Tool-call repair: fix malformed JSON, wrong tool names and sloppy argument names before anything is rejected.

Small local models regularly emit almost-valid calls: trailing commas, single quotes, unquoted keys, a code fence
around the JSON, a truncated closing brace, ``Path`` instead of ``path``. Rejecting those costs a whole model
round-trip; repairing them is free and deterministic. Every repair is reported so it can be logged and shown.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .fuzzy import best_match

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)
_BARE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_LITERALS = {"True": "true", "False": "false", "None": "null", "undefined": "null", "NaN": "null"}
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')
_SMART =str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _normalize(s: str) -> str:
    """One pass over the text, outside strings only: requote 'single' strings, quote bare keys,
    map Python literals, drop trailing commas. Strings are copied (and re-escaped) faithfully."""
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in "\"'":
            j, buf = i + 1, []
            while j < n and s[j] != ch:
                if s[j] == "\\" and j + 1 < n:
                    buf.append(s[j:j + 2])
                    j += 2
                    continue
                buf.append(s[j])
                j += 1
            raw = "".join(buf).replace("\n", "\\n").replace("\r", "")
            if ch == "'":
                raw = re.sub(r'(?<!\\)"', r'\\"', raw.replace("\\'", "'"))
            raw = _BAD_ESCAPE.sub(lambda m: "\\" + m.group(0), raw)     # C:\Users -> C:\\Users
            out.append('"' + raw + '"')
            i = j + 1
            continue
        if ch == ",":
            k = i + 1
            while k < n and s[k].isspace():
                k += 1
            if k >= n or s[k] in "}]":
                i += 1                      # trailing comma
                continue
        m = _BARE.match(s, i) if (ch.isalpha() or ch == "_") else None
        if m:
            word = m.group(0)
            k = m.end()
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k] == ":":
                out.append(json.dumps(word))          # bare key
            else:
                out.append(_LITERALS.get(word, word))
            i = m.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _close(s: str) -> str:
    """Append whatever is needed to close a truncated string/object/array."""
    stack: list[str] = []
    in_str, esc = False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    tail = '"' if in_str else ""
    body = re.sub(r",\s*$", "", s + tail)
    body = re.sub(r':\s*$', ": null", body)          # key with no value at all
    return body + "".join(reversed(stack))


def _extract(text: str) -> str:
    """Cut the JSON-looking part out of surrounding chatter."""
    starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
    return text[min(starts):] if starts else text


def repair_json(text: str) -> tuple[Any, list[str]]:
    """Parse `text` as JSON, repairing common damage. Returns (value, repairs). Raises ValueError if hopeless."""
    repairs: list[str] = []
    s = text.strip().translate(_SMART)
    if s != text.strip():
        repairs.append("replaced typographic quotes")
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
        repairs.append("removed code fence")
    try:
        return json.loads(s), repairs
    except json.JSONDecodeError:
        pass
    inner = _extract(s)
    if inner != s:
        repairs.append("dropped text around the JSON")
    s = inner
    if s and s[0] not in "{[" and re.match(r"""\s*["']?\w+["']?\s*:""", s):
        s = "{" + s + "}"
        repairs.append("wrapped bare key/value pairs in braces")
    try:                                              # valid JSON followed by chatter
        value, end = json.JSONDecoder().raw_decode(s)
        if s[end:].strip():
            repairs.append("dropped text after the JSON")
            return value, repairs
    except json.JSONDecodeError:
        pass
    for label, fn in (("normalized quotes/keys/commas", _normalize),
                      ("closed truncated JSON", lambda x: _close(_normalize(x)))):
        try:
            value = json.loads(fn(s))
        except (json.JSONDecodeError, RecursionError):
            continue
        repairs.append(label)
        return value, repairs
    raise ValueError("not repairable as JSON")


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """Normalise whatever the model sent as `arguments` into a dict.

    Returns (args, repairs, error). `error` is a message for the model when nothing sensible can be recovered."""
    repairs: list[str] = []
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, repairs, None
    value = raw
    for _ in range(2):                                   # JSON that is itself a JSON string (double encoding)
        if isinstance(value, str):
            try:
                value, fixes = repair_json(value)
            except ValueError as e:
                return None, repairs, (f"Arguments were not valid JSON ({e}). "
                                       "Send one JSON object such as {\"name\": \"value\"}.")
            repairs += fixes
            if fixes == [] and isinstance(value, dict):
                break
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
        repairs.append("unwrapped single-item list")
    if not isinstance(value, dict):
        return None, repairs, "Arguments must be a JSON object, e.g. {\"name\": \"value\"}."
    fixed = _fix_paths(value)
    if fixed != value:
        repairs.append("restored backslashes in a Windows path")
    return fixed, repairs, None


_CTRL_BACK = {"\x08": "\\b", "\x0c": "\\f", "\t": "\\t", "\r": "\\r", "\n": "\\n"}


def _fix_paths(value: Any) -> Any:
    """'C:\\temp\\file' written with single backslashes parses as a path full of tab/form-feed characters.
    A string that starts like a drive path but holds control characters is almost surely that mistake."""
    if isinstance(value, str):
        if re.match(r"^[A-Za-z]:[\\/]", value) or (len(value) > 2 and value[1:2] == ":" and value[0].isalpha()):
            if any(c in value for c in _CTRL_BACK):
                return "".join(_CTRL_BACK.get(c, c) for c in value)
        return value
    if isinstance(value, dict):
        return {k: _fix_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fix_paths(v) for v in value]
    return value


def repair_tool_name(name: str, known: list[str]) -> tuple[str | None, str | None]:
    """Map a slightly wrong tool name to a real one. Returns (name, how) or (None, None)."""
    if name in known:
        return name, None
    cleaned = re.sub(r"^(functions?|tools?)[./:]", "", name.strip(), flags=re.I)
    cleaned = re.sub(r"\(.*$", "", cleaned).strip().replace("-", "_").replace(" ", "_").replace(".", "_")
    lowered = {k.lower(): k for k in known}
    if cleaned.lower() in lowered:
        return lowered[cleaned.lower()], f"normalized tool name '{name}'"
    hits = best_match(cleaned, known, threshold=86, limit=2)
    if hits and (len(hits) == 1 or hits[0].score - hits[1].score >= 6):
        return hits[0].value, f"fuzzy-matched tool name '{name}'"
    return None, None


_WRAPPERS = ("arguments", "args", "parameters", "params", "input", "kwargs")


def repair_arguments(parameters: dict[str, Any], args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fit sloppy argument names to a tool's schema (unwrap {"arguments": {...}}, fix case, fuzzy-match names)."""
    props = list(parameters.get("properties", {}))
    required = parameters.get("required", [])
    repairs: list[str] = []
    args = dict(args)
    for w in _WRAPPERS:                                     # {"arguments": {"path": ...}} wrapper
        if set(args) == {w} and isinstance(args[w], dict):
            args = dict(args[w])
            repairs.append(f"unwrapped '{w}'")
            break
    if not props:
        return args, repairs
    fixed: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for k, v in args.items():
        if k in props:
            fixed[k] = v
            continue
        norm = re.sub(r"[^a-z0-9]", "", k.lower())
        exact = [p for p in props if re.sub(r"[^a-z0-9]", "", p.lower()) == norm and p not in args]
        if exact:
            fixed[exact[0]] = v
            repairs.append(f"renamed argument '{k}' -> '{exact[0]}'")
            continue
        free = [p for p in props if p not in args and p not in fixed]
        hit = best_match(k, free, threshold=82, limit=1) if free else []
        if hit:
            fixed[hit[0].value] = v
            repairs.append(f"renamed argument '{k}' -> '{hit[0].value}'")
        else:
            unknown[k] = v
    missing = [r for r in required if r not in fixed]
    if len(unknown) == 1 and len(missing) == 1:           # one stray key, one hole: obviously the same thing
        (k, v), = unknown.items()
        fixed[missing[0]] = v
        repairs.append(f"used '{k}' as '{missing[0]}'")
        unknown = {}
    fixed.update(unknown)                                  # unknown keys are dropped later by validation, with a note
    return fixed, repairs
