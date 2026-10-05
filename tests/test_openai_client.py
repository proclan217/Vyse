from __future__ import annotations

import json

import httpx
import pytest

from vyse.config import ModelConfig
from vyse.llm import RateLimiter, LLMError, OpenAIClient, make_client, OllamaClient


def sse(*chunks) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def client(handler, monkeypatch, key="nvapi-test"):
    monkeypatch.setattr("vyse.llm.read_api_key", lambda name: key)      # never touch the real key / registry
    cfg = ModelConfig(provider="openai", name="nvidia/x")
    http = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler),
                        headers={"Authorization": f"Bearer {key}"})
    return OpenAIClient(cfg, http)


def test_streams_text_and_tool_calls_and_sends_payload(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen["body"], seen["auth"], seen["path"] = json.loads(req.content), req.headers["authorization"], req.url.path
        body = sse({"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
                   {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "echo", "arguments": '{"te'}}]}}]},
                   {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'xt": "hi"}'}}]}}]})
        return httpx.Response(200, text=body)

    tokens: list[str] = []
    r = client(handler, monkeypatch).chat(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tools=[{"type": "function"}], on_token=tokens.append)
    assert r.content == "Hello" and tokens == ["Hel", "lo"]
    assert r.tool_calls[0].name == "echo" and r.tool_calls[0].arguments == {"text": "hi"}
    assert seen["path"].endswith("/chat/completions") and seen["auth"] == "Bearer nvapi-test"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False} and seen["body"]["stream"] is True


def test_history_is_converted_to_openai_schema(monkeypatch):
    seen = {}

    def handler(req):
        seen["msgs"] = json.loads(req.content)["messages"]
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "ok"}}]}))

    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "content": "(calling tools)", "tool_calls": [{"function": {"name": "t", "arguments": {"a": 1}}}]},
            {"role": "tool", "tool_name": "t", "content": "{}"}]
    client(handler, monkeypatch).chat(msgs)
    a, t = seen["msgs"][1], seen["msgs"][2]
    assert a["tool_calls"][0]["function"]["arguments"] == '{"a": 1}' and t["tool_call_id"] == a["tool_calls"][0]["id"]


def test_missing_key_and_http_errors_are_reported(monkeypatch):
    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        client(lambda r: httpx.Response(200), monkeypatch, key="").chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError, match="401"):
        client(lambda r: httpx.Response(401, text="bad key"), monkeypatch).chat([{"role": "user", "content": "x"}])


def test_factory_picks_provider():
    assert isinstance(make_client(ModelConfig(provider="ollama")), OllamaClient)
    assert isinstance(make_client(ModelConfig(provider="openai")), OpenAIClient)


def test_rate_limiter_waits_instead_of_exceeding_limit():
    now = [0.0]
    slept: list[float] = []

    def sleep(d):
        slept.append(d)
        now[0] += d
    rl = RateLimiter(3, window=60, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        assert rl.acquire() == 0 and not slept          # under the limit: no waiting
        now[0] += 1
    assert rl.acquire() > 55 and slept                    # 4th request must wait for the window to slide
    assert max(len([t for t in rl._times if now[0] - t < 60]), 0) <= 3


def test_429_is_retried(monkeypatch):
    monkeypatch.setattr("vyse.llm.time.sleep", lambda s: None)
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        if n["c"] == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, text="slow down")
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "ok"}}]}))
    assert client(handler, monkeypatch).chat([{"role": "user", "content": "x"}]).content == "ok" and n["c"] == 2
