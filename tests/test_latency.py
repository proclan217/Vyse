"""Latency shortcuts: skip the narration call, deferred summarization, stable prompt prefix."""
from __future__ import annotations

import re
import time

import pytest

from tests.conftest import FakeLLM, call, say
from vyse.agent import Agent, Hooks


# ---------------------------------------------------------------- LLM-path shortcuts
def test_final_tool_skips_second_llm_call(ctx, registry):
    llm = FakeLLM([call("get_ram_usage"), say("SHOULD NOT BE USED")])
    out = Agent(llm, registry, ctx).run_turn("is my ram ok?")
    assert len(llm.calls) == 1 and "SHOULD NOT" not in out


def test_compound_request_still_gets_second_llm_call(ctx, registry):
    llm = FakeLLM([call("get_ram_usage"), say("all done")])
    out = Agent(llm, registry, ctx).run_turn("check my ram and then tell me a joke")
    assert len(llm.calls) == 2 and out == "all done"


def test_summarization_can_be_deferred(ctx, registry):
    llm = FakeLLM([say("a")] * 6 + [say("SUMMARY")])
    a = Agent(llm, registry, ctx)
    for i in range(6):
        a.run_turn(f"chat message {i}", maintain=False)
    assert ctx.memory.latest_summary("default") is None and len(llm.calls) == 6
    a.maintain()
    assert ctx.memory.latest_summary("default") is not None


def test_system_prompt_prefix_is_stable(ctx, registry):
    assert not re.search(r"\d\d:\d\d", Agent(FakeLLM([]), registry, ctx)._system())


def test_find_files_skips_protected(ctx, registry, home):
    from tests.test_files import agent_for, run
    (home / "protected" / "secret.pdf").write_text("x")
    (home / "Documents" / "ok.pdf").write_text("x")
    res = run(agent_for(ctx, registry), "find_files", extension="pdf", folder=str(home))
    assert [f["name"] for f in res["files"]] == ["ok.pdf"]


def test_action_request_forces_a_tool_call_on_first_step_only(ctx, registry):
    llm = FakeLLM([call("get_ram_usage")])
    Agent(llm, registry, ctx).run_turn("check my ram")
    assert llm.calls[0]["force"] is True
    llm = FakeLLM([say("Why did the cat..."), ])
    Agent(llm, registry, ctx).run_turn("tell me a joke about cats")
    assert llm.calls[0]["force"] is False
