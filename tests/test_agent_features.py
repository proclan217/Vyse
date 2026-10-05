"""Features 6-10, 16: chaining, parallel execution, permissions, confirmation, undo, tool-result cache."""
from __future__ import annotations

import threading
import time

from tests.conftest import FakeLLM, say
from vyse.agent import Agent, Hooks, resolve_refs
from vyse.llm import LLMResponse, ToolCall
from vyse.tools.registry import Registry


def calls(*specs) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in specs])


def agent(ctx, reg, script, confirm=None):
    llm = FakeLLM(script)
    return Agent(llm, reg, ctx, Hooks(confirm=confirm or (lambda p, r: False))), llm


# ---- 6. chaining ----
def make_chain_reg():
    reg = Registry()
    seen = []

    def make_thing() -> dict:
        """Make."""
        return {"path": "C:/x/y.txt", "info": {"size": 5}, "display": "made"}

    def use_thing(path: str) -> dict:
        """Use.

        Args:
            path: A path.
        """
        seen.append(path)
        return {"display": f"used {path}"}

    reg.add(make_thing, risk="safe", parallel=False)
    reg.add(use_thing, risk="safe", parallel=False)
    return reg, seen


def test_resolve_refs_paths_and_embedding():
    res = {1: {"path": "a.txt", "info": {"size": 5}}}
    assert resolve_refs({"p": "$1.path"}, res) == {"p": "a.txt"}
    assert resolve_refs("${1.info.size} bytes", res) == "5 bytes"
    try:
        resolve_refs("$1.nope", res)
    except KeyError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError


def test_dependent_tools_chain_in_one_model_step(ctx):
    reg, seen = make_chain_reg()
    a, llm = agent(ctx, reg, [calls(("make_thing", {}), ("use_thing", {"path": "$1.path"})), say("done")])
    assert a.run_turn("chain") == "done"
    assert seen == ["C:/x/y.txt"]
    assert len(llm.calls) == 2           # one step for both tools, one for the answer


def test_dependent_call_is_skipped_when_its_dependency_failed(ctx):
    reg, seen = make_chain_reg()

    def broken() -> dict:
        """Break."""
        raise RuntimeError("x")

    reg.add(broken, risk="safe", parallel=False)
    a, llm = agent(ctx, reg, [calls(("broken", {}), ("use_thing", {"path": "$1.path"})), say("ok")])
    a.run_turn("x")
    assert seen == []
    msgs = " ".join(m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "depends on call 1" in msgs


# ---- 7. parallel execution ----
def test_independent_safe_tools_run_concurrently(ctx):
    reg = Registry()
    barrier = threading.Barrier(2, timeout=3)

    def slow_a() -> dict:
        """A."""
        barrier.wait()          # only passes when slow_b runs at the same time
        return {"display": "a"}

    def slow_b() -> dict:
        """B."""
        barrier.wait()
        return {"display": "b"}

    reg.add(slow_a, risk="safe")
    reg.add(slow_b, risk="safe")
    a, llm = agent(ctx, reg, [calls(("slow_a", {}), ("slow_b", {})), say("both")])
    t0 = time.time()
    assert a.run_turn("go") == "both"
    assert time.time() - t0 < 3
    msgs = " ".join(m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert '"ok": true' in msgs.lower() or "'ok': True" in msgs


def test_slow_parallel_tool_times_out_without_blocking_the_others(ctx):
    ctx.cfg.agent.tool_timeout = 0.3
    reg = Registry()

    def hang() -> dict:
        """Hang."""
        time.sleep(1.5)
        return {"display": "late"}

    def quick() -> dict:
        """Quick."""
        return {"display": "fast"}

    reg.add(hang, risk="safe")
    reg.add(quick, risk="safe")
    a, llm = agent(ctx, reg, [calls(("hang", {}), ("quick", {})), say("x")])
    a.run_turn("go")
    msgs = " ".join(m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "timeout" in msgs and "fast" in msgs


# ---- 8/9. permissions + confirmation ----
def test_blocked_permission_never_runs(ctx):
    reg = Registry()
    ran = []
    reg.add(lambda: ran.append(1) or {"display": "x"}, risk="safe", name="nope", permission="blocked")
    a, llm = agent(ctx, reg, [calls(("nope", {})), say("refused")])
    a.run_turn("x")
    assert ran == []


def test_confirmation_permission_asks_and_respects_decline(ctx):
    reg = Registry()
    ran = []
    reg.add(lambda: ran.append(1) or {"display": "x"}, risk="safe", name="ask_me", permission="confirmation")
    asked = []
    a, _ = agent(ctx, reg, [calls(("ask_me", {})), say("a"), ], confirm=lambda p, r: asked.append(r) or False)
    a.run_turn("x")
    assert asked and ran == []


def test_config_permission_override_can_only_tighten(ctx):
    reg = Registry()
    reg.add(lambda: {"display": "x"}, risk="risky", name="danger")
    reg.add(lambda: {"display": "y"}, risk="safe", name="calm")
    ctx.cfg.permissions = {"danger": "safe", "calm": "blocked"}
    pol = ctx.policy
    assert pol.permission_of(reg.get("danger")) == "confirmation"      # cannot be relaxed
    assert pol.permission_of(reg.get("calm")) == "blocked"             # can be tightened


def test_destructive_tools_confirm_even_with_auto_yes(ctx):
    reg = Registry()
    ran = []
    reg.add(lambda: ran.append(1) or {"display": "gone"}, risk="write", name="wipe", destructive=True)
    asked = []
    a, _ = agent(ctx, reg, [calls(("wipe", {})), say("n")], confirm=lambda p, r: asked.append(1) or False)
    a.auto_yes = True
    a.run_turn("wipe")
    assert asked and ran == []


def test_unattended_runs_decline_anything_needing_confirmation(ctx):
    reg = Registry()
    ran = []
    reg.add(lambda: ran.append(1) or {"display": "x"}, risk="risky", name="risky_op")
    a, _ = agent(ctx, reg, [])
    res = ctx.run_tool_unattended("risky_op", {})
    assert not res["ok"] and ran == []


def test_trash_tool_is_destructive_and_kill_process_is_registered(registry):
    assert registry.get("trash").destructive
    kp = registry.get("kill_process")
    assert kp is not None and kp.destructive


def test_kill_process_refuses_critical_and_own_processes(registry, ctx):
    import os
    d = registry.get("kill_process").assess({"name": "lsass.exe"})
    assert d.action == "block"
    d = registry.get("kill_process").assess({"pid": os.getpid()})
    assert d.action == "block"


# ---- 10. undo / transaction log ----
def test_undo_history_and_multi_step_undo(ctx, registry, home):
    f1, f2 = home / "Desktop" / "a.txt", home / "Desktop" / "b.txt"
    f1.write_text("1")
    f2.write_text("2")
    dest = home / "Documents"
    registry.get("move").fn(str(f1), str(dest))
    registry.get("move").fn(str(f2), str(dest))
    hist = registry.get("undo_history").fn()
    assert len(hist["transactions"]) == 2 and not hist["transactions"][0]["undone"]
    res = registry.get("undo").fn(steps=2)
    assert res["reverted"] == 2 and f1.exists() and f2.exists()
    assert registry.get("undo_history").fn()["transactions"][0]["undone"]


def test_undo_specific_transaction_by_id(ctx, registry, home):
    a, b = home / "Desktop" / "a.txt", home / "Desktop" / "b.txt"
    a.write_text("1")
    b.write_text("2")
    registry.get("move").fn(str(a), str(home / "Documents"))
    registry.get("move").fn(str(b), str(home / "Documents"))
    old = registry.get("undo_history").fn()["transactions"][-1]["transaction"]
    registry.get("undo").fn(transaction=old)
    assert a.exists() and not b.exists()
    again = registry.get("undo").fn(transaction=old)
    assert again["problems"]


# ---- 16. tool-result caching ----
def test_cached_tool_result_is_reused_and_write_invalidates(ctx):
    reg = Registry()
    n = []

    def info() -> dict:
        """Info."""
        n.append(1)
        return {"display": f"call {len(n)}"}

    def change() -> dict:
        """Change."""
        return {"display": "changed"}

    reg.add(info, risk="safe", cache_ttl=60)
    reg.add(change, risk="write", invalidates=("info",))
    a, _ = agent(ctx, reg, [calls(("info", {})), say("1"), calls(("info", {})), say("2"),
                            calls(("change", {})), say("3"), calls(("info", {})), say("4")])
    a.run_turn("a")
    a.run_turn("b")
    assert len(n) == 1
    a.auto_yes = True
    a.run_turn("c")
    a.run_turn("d")
    assert len(n) == 2


def test_cache_expires_and_is_bounded():
    from vyse.cache import ToolCache
    now = [0.0]
    c = ToolCache(max_entries=2, clock=lambda: now[0])
    c.put("t", {"a": 1}, {"ok": True}, ttl=10)
    assert c.get("t", {"a": 1}) == {"ok": True}
    now[0] = 11
    assert c.get("t", {"a": 1}) is None
    for i in range(5):
        c.put("t", {"i": i}, {"ok": True}, ttl=100)
    assert len(c) <= 2
    assert c.stats()["hits"] == 1
