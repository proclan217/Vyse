"""Model-agnostic LLM interface plus an Ollama implementation (native tool calling)."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import httpx

from .config import ModelConfig
from .repair import parse_arguments


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set when the model produced a malformed call
    repairs: list[str] = field(default_factory=list)  # fixes applied to the raw call (JSON repair etc.)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int = 0       # reported by the backend (0 = not reported; the agent then estimates)
    completion_tokens: int = 0


class LLMError(Exception):
    pass


class LLMClient(ABC):
    """Anything that can take chat messages (+ tool schemas) and return text and/or tool calls."""

    model: str

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             on_token: Callable[[str], None] | None = None, force_tool: bool = False) -> LLMResponse: ...

    def set_model(self, name: str) -> None:
        self.model = name


def parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Normalize raw tool calls; malformed ones become ToolCall(error=...) instead of raising."""
    out: list[ToolCall] = []
    for rc in raw_calls or []:
        fn = (rc or {}).get("function") if isinstance(rc, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            out.append(ToolCall(name="", error="Malformed tool call: missing function name."))
            continue
        args, repairs, err = parse_arguments(fn.get("arguments", {}))
        if args is None:
            out.append(ToolCall(name=fn["name"], error=err or "Arguments must be a JSON object."))
            continue
        out.append(ToolCall(name=fn["name"], arguments=args, repairs=repairs))
    return out


class OllamaClient(LLMClient):
    def __init__(self, cfg: ModelConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.name
        self._http = client or httpx.Client(base_url=cfg.ollama_url, timeout=cfg.timeout)

    def chat(self, messages, tools=None, on_token=None, force_tool=False) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": True,
            "think": self.cfg.think,
            "options": {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx,
                        "num_predict": self.cfg.num_predict, "num_gpu": self.cfg.num_gpu},
            "keep_alive": self.cfg.keep_alive,   # keep the model loaded between turns (no reload lag)
        }
        if tools:
            payload["tools"] = tools
        resp = LLMResponse()
        raw_calls: list[dict[str, Any]] = []
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")
                    raise LLMError(f"Ollama returned {r.status_code}: {body[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError(chunk["error"])
                    msg = chunk.get("message", {})
                    if msg.get("thinking"):
                        resp.thinking += msg["thinking"]
                    if msg.get("content"):
                        resp.content += msg["content"]
                        if on_token:
                            on_token(msg["content"])
                    raw_calls.extend(msg.get("tool_calls") or [])
                    if chunk.get("done"):
                        resp.prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                        resp.completion_tokens = int(chunk.get("eval_count") or 0)
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach Ollama at {self.cfg.ollama_url}. Is it running? ({e})") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        resp.tool_calls = parse_tool_calls(raw_calls)
        return resp

    def warm(self) -> None:
        """Load the model into memory now (empty request) so the first real turn has no load delay."""
        try:
            self._http.post("/api/chat", json={"model": self.model, "messages": [], "keep_alive": self.cfg.keep_alive,
                                               "options": {"num_ctx": self.cfg.num_ctx, "num_gpu": self.cfg.num_gpu}})
        except httpx.HTTPError:
            pass

    def list_models(self) -> list[str]:
        try:
            r = self._http.get("/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except httpx.HTTPError as e:
            raise LLMError(f"Cannot list models: {e}") from e


def read_api_key(name: str) -> str:
    """Environment first; on Windows fall back to the saved user variable (setx only affects *new* terminals)."""
    import os
    key = os.environ.get(name, "")
    if not key and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            key = ""
    return key.strip()


class RateLimiter:
    """Sliding-window limiter: acquire() sleeps just long enough to stay under `limit` requests per `window` seconds."""

    def __init__(self, limit: int, window: float = 60.0, clock=None, sleep=None) -> None:
        import threading
        import time
        from collections import deque
        self.limit, self.window = max(1, limit), window
        self._clock, self._sleep = clock or time.monotonic, sleep or time.sleep
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be sent; returns how long we waited."""
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                while self._times and now - self._times[0] >= self.window:
                    self._times.popleft()
                if len(self._times) < self.limit:
                    self._times.append(now)
                    return waited
                delay = self.window - (now - self._times[0]) + 0.05
                self._sleep(delay)
                waited += delay


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat API with streaming tool calls (NVIDIA NIM at integrate.api.nvidia.com)."""

    def __init__(self, cfg: ModelConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.name
        key = read_api_key(cfg.api_key_env)
        self._key_missing = not key
        self._limiter = RateLimiter(cfg.rpm_limit)
        self._http = client or httpx.Client(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout,
                                            headers={"Authorization": f"Bearer {key}"})

    def chat(self, messages, tools=None, on_token=None, force_tool=False) -> LLMResponse:
        if self._key_missing:
            raise LLMError(f"No API key: set the {self.cfg.api_key_env} environment variable.")
        payload: dict[str, Any] = {
            "model": self.model, "stream": True, "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.num_predict,
            "messages": [_to_openai(m) for m in messages],
            "chat_template_kwargs": {"enable_thinking": bool(self.cfg.think)},
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required" if force_tool else "auto"
        resp = LLMResponse()
        parts: dict[int, dict[str, Any]] = {}
        try:
            for attempt in range(3):
                self._limiter.acquire()
                with self._http.stream("POST", "/chat/completions", json=payload) as r:
                    if r.status_code == 429 and attempt < 2:      # still rate limited: honour Retry-After, retry
                        try:
                            wait = float(r.headers.get("retry-after", "5"))
                        except ValueError:
                            wait = 5.0
                        r.read()
                        time.sleep(min(max(wait, 1.0), 30.0))
                        continue
                    if r.status_code != 200:
                        raise LLMError(f"API returned {r.status_code}: {r.read().decode('utf-8', 'replace')[:300]}")
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        usage = obj.get("usage") or {}
                        if usage:
                            resp.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                            resp.completion_tokens = int(usage.get("completion_tokens") or 0)
                        for ch in obj.get("choices") or []:
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                resp.content += d["content"]
                                if on_token:
                                    on_token(d["content"])
                            for tc in d.get("tool_calls") or []:
                                p = parts.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                                fn = tc.get("function") or {}
                                p["name"] += fn.get("name") or ""
                                p["args"] += fn.get("arguments") or ""
                break
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach {self.cfg.base_url}: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"API request failed: {e}") from e
        resp.tool_calls = parse_tool_calls(
            [{"function": {"name": p["name"], "arguments": p["args"]}} for _, p in sorted(parts.items())])
        return resp

    def warm(self) -> None:
        pass

    def list_models(self) -> list[str]:
        return [self.model]


def _to_openai(m: dict[str, Any]) -> dict[str, Any]:
    """Ollama-style history messages -> OpenAI schema (tool calls need ids and string arguments)."""
    if m.get("role") == "tool":
        return {"role": "tool", "tool_call_id": m.get("tool_call_id") or f"call_{m.get('tool_name', 'x')}",
                "content": m.get("content", "")}
    if m.get("role") == "assistant" and m.get("tool_calls"):
        calls = [{"id": f"call_{c['function']['name']}", "type": "function",
                  "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"].get("arguments", {}))}}
                 for c in m["tool_calls"]]
        return {"role": "assistant", "content": m.get("content") or "", "tool_calls": calls}
    return {"role": m["role"], "content": m.get("content", "")}


def make_client(cfg: ModelConfig) -> LLMClient:
    return OpenAIClient(cfg) if cfg.provider == "openai" else OllamaClient(cfg)
