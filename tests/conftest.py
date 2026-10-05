from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vyse.config import AgentConfig, AppEntry, Config
from vyse.context import Context
from vyse.llm import LLMClient, LLMResponse, ToolCall
from vyse.memory import Memory
from vyse.tools import build_registry


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    home = tmp_path / "home"
    for d in ("Desktop", "Documents", "Downloads", "outside", "protected"):
        (home / d).mkdir(parents=True)
    c = Config(
        data_dir=tmp_path / "data", notes_dir=home / "Documents" / "VyseNotes",
        allowed_roots=[home / "Desktop", home / "Documents", home / "Downloads"],
        read_roots=[home],
        protected_paths=[home / "protected"],
        agent=AgentConfig(max_steps=5, history_messages=6, summarize_after=8),
    )
    c.organize_confirm_threshold = 5
    c.app_scan = False            # hermetic: never scan the real Start Menu in tests
    c.index.enabled = False        # tests that need the file index build their own
    c.apps = {"notepad": AppEntry("notepad", ["notepad"], path="notepad.exe", process="notepad.exe")}
    return c


@pytest.fixture
def home(cfg: Config) -> Path:
    return cfg.allowed_roots[0].parent


@pytest.fixture
def ctx(cfg: Config) -> Context:
    return Context.build(cfg, Memory(":memory:"))


@pytest.fixture
def registry(ctx: Context):
    return build_registry(ctx, extra=False)


class FakeLLM(LLMClient):
    """Scripted LLM: each chat() pops the next response. Records what it was sent."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.model = "fake"
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, on_token=None, force_tool=False) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "force": force_tool})
        if not self.script:
            return LLMResponse(content="(script exhausted)")
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, arguments=args)])


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text)
