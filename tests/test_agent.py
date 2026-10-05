import json

from tests.conftest import FakeLLM, call, say
from vyse.agent import Agent, Hooks, validate_args
from vyse.llm import LLMError, LLMResponse, ToolCall, parse_tool_calls
from vyse.tools.registry import Registry, ToolError


def make_agent(ctx, script, registry=None, confirm=None, tools=None):
    reg = registry or Registry()
    if tools:
        for fn, kw in tools:
            reg.add(fn, **kw)
    llm = FakeLLM(script)
    events = []
    hooks = Hooks(on_tool_start=lambda n, a: events.append(("start", n, a)),
                  on_tool_end=lambda n, ok, s: events.append(("end", n, ok, s)),
                  confirm=confirm or (lambda p, r: False))
    return Agent(llm, reg, ctx, hooks), llm, events


def echo(text: str) -> dict:
    """Echo text back.

    Args:
        text: What to echo.
    """
    return {"echo": text, "display": "echoed"}


def test_plain_answer_without_tools(ctx):
    a, llm, _ = make_agent(ctx, [say("Hello!")])
    assert a.run_turn("hi") == "Hello!"
    assert len(llm.calls) == 1


def test_tool_loop_executes_and_feeds_result_back(ctx):
    a, llm, ev = make_agent(ctx, [call("echo", text="ping"), say("It said ping.")], tools=[(echo, {"risk": "safe"})])
    assert a.run_turn("echo ping") == "It said ping."
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"]) == {"ok": True, "echo": "ping", "display": "echoed"}
    assert ("end", "echo", True, "echoed") in ev
    assert a.last_state.phase == "finish" and a.last_state.steps[0]["tool"] == "echo"


def test_multiple_sequential_tool_calls(ctx):
    a, llm, ev = make_agent(ctx, [call("echo", text="1"), call("echo", text="2"), say("done")], tools=[(echo, {"risk": "safe"})])
    assert a.run_turn("x") == "done"
    assert [e[1] for e in ev if e[0] == "start"] == ["echo", "echo"]


def test_unknown_tool_recovery(ctx):
    a, llm, _ = make_agent(ctx, [call("does_not_exist"), say("sorry")], tools=[(echo, {"risk": "safe"})])
    assert a.run_turn("x") == "sorry"
    err = json.loads([m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert not err["ok"] and "Unknown tool" in err["error"] and "echo" in err["error"]


def test_malformed_tool_call_json_recovers():
    # string args that are not JSON -> ToolCall.error, not an exception
    calls = parse_tool_calls([{"function": {"name": "echo", "arguments": "{not json"}}])
    assert calls[0].error and "valid JSON" in calls[0].error
    assert parse_tool_calls([{"function": {"arguments": {}}}])[0].error
    assert parse_tool_calls(["garbage"])[0].error
    assert parse_tool_calls([{"function": {"name": "echo", "arguments": '{"text": "ok"}'}}])[0].arguments == {"text": "ok"}
    assert parse_tool_calls([{"function": {"name": "echo", "arguments": "[1]"}}])[0].error


def test_agent_survives_malformed_call_then_succeeds(ctx):
    bad = LLMResponse(tool_calls=[ToolCall(name="echo", error="Arguments were not valid JSON")])
    a, llm, ev = make_agent(ctx, [bad, call("echo", text="fixed"), say("ok")], tools=[(echo, {"risk": "safe"})])
    assert a.run_turn("x") == "ok"
    first_tool = json.loads([m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert not first_tool["ok"] and "valid JSON" in first_tool["error"]
    # the malformed call must NOT be echoed back as a real tool_call
    assistant = [m for m in llm.calls[1]["messages"] if m["role"] == "assistant"][-1]
    assert "tool_calls" not in assistant
    assert ("end", "echo", True, "echoed") in ev


def test_missing_and_wrong_args_return_structured_errors(ctx):
    a, llm, _ = make_agent(ctx, [call("echo"), say("x")], tools=[(echo, {"risk": "safe"})])
    a.run_turn("x")
    err = json.loads([m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert not err["ok"] and "Missing required" in err["error"]


def test_arg_coercion_and_extra_args_ignored():
    reg = Registry()

    def f(n: int, flag: bool = False, items: list[str] | None = None) -> dict:
        return {}

    reg.add(f, risk="safe")
    args, err = validate_args(reg.get("f"), {"n": "5", "flag": "true", "items": "a, b", "bogus": 1})
    assert err is None and args == {"n": 5, "flag": True, "items": ["a", "b"]}
    _, err = validate_args(reg.get("f"), {"n": "abc"})
    assert err and "wrong type" in err


def test_tool_exception_becomes_error_result(ctx):
    def boom() -> dict:
        """Explode."""
        raise RuntimeError("kaboom")

    def known() -> dict:
        """Fail nicely."""
        raise ToolError("nice failure")

    a, llm, _ = make_agent(ctx, [call("boom"), call("known"), say("handled")],
                           tools=[(boom, {"risk": "safe"}), (known, {"risk": "safe"})])
    assert a.run_turn("x") == "handled"
    msgs = [json.loads(m["content"]) for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert "RuntimeError: kaboom" in msgs[0]["error"] and msgs[1]["error"] == "nice failure"


def test_confirmation_declined_blocks_execution_and_tells_model(ctx):
    ran = []

    def danger() -> dict:
        """Danger."""
        ran.append(1)
        return {}

    a, llm, _ = make_agent(ctx, [call("danger"), say("ok, not doing it")], tools=[(danger, {"risk": "risky"})],
                           confirm=lambda p, r: False)
    assert a.run_turn("x") == "ok, not doing it"
    assert not ran
    res = json.loads([m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert res["declined"] and "declined" in res["error"]


def test_confirmation_approved_executes_and_preview_shown(ctx):
    ran, shown = [], []

    def danger() -> dict:
        """Danger."""
        ran.append(1)
        return {"x": 1}

    a, _, _ = make_agent(ctx, [call("danger"), say("done")], tools=[(danger, {"risk": "risky"})],
                         confirm=lambda p, r: shown.append((p, r)) or True)
    a.run_turn("x")
    assert ran == [1] and shown and "danger" in shown[0][0]


def test_auto_yes_skips_confirm_but_never_block(ctx):
    from vyse.policy import BLOCK, Decision
    ran = []

    def risky() -> dict:
        """r"""
        ran.append("r")
        return {}

    def blocked() -> dict:
        """b"""
        ran.append("b")
        return {}

    asked = []
    a, _, _ = make_agent(ctx, [call("risky"), call("blocked"), say("x")],
                         tools=[(risky, {"risk": "risky"}), (blocked, {"risk": "safe", "assess": lambda a: Decision(BLOCK, "nope")})],
                         confirm=lambda p, r: asked.append(p) or False)
    a.auto_yes = True
    a.run_turn("x")
    assert ran == ["r"] and not asked


def test_model_cannot_override_policy_via_args(ctx, home):
    """The model passing 'confirm=true'-style args must not bypass the policy."""
    reg = Registry()
    ran = []

    def wipe(path: str) -> dict:
        """wipe"""
        ran.append(path)
        return {}

    reg.add(wipe, risk="risky")
    a, _, _ = make_agent(ctx, [LLMResponse(tool_calls=[ToolCall("wipe", {"path": "x", "confirmed": True, "force": True})]), say("ok")],
                         registry=reg, confirm=lambda p, r: False)
    a.run_turn("x")
    assert ran == []


def test_repeated_identical_call_is_cut_off(ctx):
    n = []

    def tick() -> dict:
        """tick"""
        n.append(1)
        return {}

    a, _, _ = make_agent(ctx, [call("tick")] * 4 + [say("stop")], tools=[(tick, {"risk": "safe"})])
    a.run_turn("x")
    assert len(n) == 2


def test_max_steps_enforced_with_wrapup(ctx):
    a, llm, _ = make_agent(ctx, [call("echo", text=str(i)) for i in range(10)] + [say("wrapped up")],
                           tools=[(echo, {"risk": "safe"})])
    out = a.run_turn("x")
    assert len(llm.calls) == ctx.cfg.agent.max_steps + 1
    assert llm.calls[-1]["tools"] is None
    assert out == "(script exhausted)" or out == "wrapped up" or out


def test_llm_error_is_reported_not_raised(ctx):
    a, _, _ = make_agent(ctx, [LLMError("ollama down")])
    assert "ollama down" in a.run_turn("hi")


def test_think_tags_stripped(ctx):
    a, _, _ = make_agent(ctx, [say("<think>secret reasoning</think>Answer.")])
    assert a.run_turn("hi") == "Answer."


def test_unverified_result_triggers_honesty_nudge(ctx):
    def flaky() -> dict:
        """flaky"""
        return {"verified": False}

    a, llm, _ = make_agent(ctx, [call("flaky"), say("It failed.")], tools=[(flaky, {"risk": "safe"})])
    a.run_turn("x")
    assert any(m["role"] == "system" and "not verified" in m["content"] for m in llm.calls[1]["messages"])


def test_relevant_facts_injected_not_whole_db(ctx):
    ctx.memory.remember("printer", "The user's printer is an HP M15")
    ctx.memory.remember("pet", "The user has a cat named Luna")
    a, llm, _ = make_agent(ctx, [say("ok")])
    a.run_turn("what printer do I have")
    system = llm.calls[0]["messages"][0]["content"]
    assert "HP M15" in system and "Luna" not in system


def test_history_is_bounded_and_summarised(ctx):
    a, llm, _ = make_agent(ctx, [say(f"a{i}") for i in range(30)] + [say("SUMMARY")] * 30)
    for i in range(6):
        a.run_turn(f"q{i}")
    sent = [m for m in llm.calls[-1]["messages"] if m["role"] != "system"]
    assert len(sent) <= ctx.cfg.agent.history_messages + 1 + 2
    assert ctx.memory.latest_summary("default") is not None or ctx.memory.message_count("default") <= ctx.cfg.agent.summarize_after


def test_reset_clears_conversation_but_not_facts(ctx):
    ctx.memory.remember("k", "v")
    a, _, _ = make_agent(ctx, [say("a")])
    a.run_turn("hi")
    a.reset()
    assert ctx.memory.message_count("default") == 0 and ctx.memory.all_facts()


def test_tool_subset_sent_to_model(ctx, registry):
    llm = FakeLLM([say("ok")])
    a = Agent(llm, registry, ctx)
    a.run_turn("what's the weather in Paris")
    names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert "weather" in names and "move" not in names
    assert len(names) <= ctx.cfg.agent.max_tools_per_turn


def test_nudge_when_model_answers_instead_of_calling_tool(ctx):
    a, llm, ev = make_agent(ctx, [say("It's sunny!"), call("echo", text="echo"), say("done")], tools=[(echo, {"risk": "safe"})])
    assert a.run_turn("please echo hello") == "done"
    assert any(m["role"] == "system" and "without calling a tool" in m["content"] for m in llm.calls[1]["messages"])
    assert ("end", "echo", True, "echoed") in ev


def test_select_prefers_name_matches_and_includes_app_names(ctx, registry):
    names = [t.name for t in registry.select("Open Notepad", 12)]
    assert "open_app" in names
    assert "weather" in [t.name for t in registry.select("weather in Rome", 6)]
