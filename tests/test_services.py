"""Features 11-19: file index, organize rules, memory retrieval, context budget, background tasks, scheduler, metrics."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from vyse.budget import compact_result, estimate_tokens, fit, total_tokens
from vyse.config import OrganizeRule
from vyse.indexer import FileIndex
from vyse.memory import Memory
from vyse.observability import Metrics
from vyse.organize_rules import build_rule_plan, classify, rules_for
from vyse.scheduler import Scheduler, build_trigger, interval_kwargs, parse_when
from vyse.tasks import TaskManager


# ---- 11. file index ----
@pytest.fixture
def index(home, ctx):
    ix = FileIndex(":memory:", [home], pruner_factory=ctx.policy.walk_pruner, is_protected=ctx.policy.is_protected)
    yield ix


def test_index_finds_files_by_name_extension_and_folder(index, home):
    (home / "Documents" / "Tax Return 2025.pdf").write_text("x")
    (home / "Documents" / "notes.txt").write_text("x")
    (home / "Downloads" / "holiday.jpg").write_text("x")
    index.build()
    assert index.ready and index.count() >= 3
    res = index.search("tax return")
    assert [r["name"] for r in res["rows"]] == ["Tax Return 2025.pdf"]
    assert {r["name"] for r in index.search("", ext="jpg")["rows"]} == {"holiday.jpg"}
    assert {r["name"] for r in index.search("", folder=home / "Documents")["rows"]} == {"Tax Return 2025.pdf", "notes.txt"}


def test_index_is_typo_tolerant_and_skips_protected_folders(index, home):
    (home / "Documents" / "Resume_final.docx").write_text("x")
    (home / "protected" / "secret.txt").write_text("x")
    index.build()
    res = index.search("resume_fnial")
    assert res["fuzzy"] and res["rows"][0]["name"] == "Resume_final.docx"
    assert not index.search("secret")["rows"]


def test_index_drops_files_deleted_since_the_scan(index, home):
    f = home / "Documents" / "gone.txt"
    f.write_text("x")
    index.build()
    f.unlink()
    assert not index.search("gone")["rows"]


def test_index_upsert_and_remove_keep_it_current(index, home):
    index.build()
    f = home / "Desktop" / "fresh.txt"
    f.write_text("x")
    index.upsert(f)
    assert index.search("fresh")["rows"]
    index.remove(f)
    assert not index.search("fresh")["rows"]


def test_find_files_uses_index_when_available(ctx, registry, home):
    (home / "Documents" / "quarterly report.pdf").write_text("x")
    ctx.index = FileIndex(":memory:", [home], pruner_factory=ctx.policy.walk_pruner, is_protected=ctx.policy.is_protected)
    ctx.index.build()
    res = registry.get("find_files").fn(query="quarterly")
    assert res.get("indexed") and res["count"] == 1


# ---- 12. organize rules ----
def test_rules_classify_deterministically(home):
    d = home / "Downloads"
    for n in ("Screenshot 2026-01-01.png", "setup.exe", "invoice_march.pdf", "photo.jpg", "thesis.pdf", "mystery.xyz"):
        (d / n).write_text("x")
    plan = {Path(m["src"]).name: m["category"] for m in build_rule_plan(d, rules_for(None))}
    assert plan["Screenshot 2026-01-01.png"] == "Images/Screenshots"
    assert plan["setup.exe"] == "Installers"
    assert plan["invoice_march.pdf"] == "Documents/Finance"
    assert plan["photo.jpg"] == "Images" and plan["thesis.pdf"] == "Documents"
    assert "mystery.xyz" not in plan                                    # unmatched files stay put
    assert build_rule_plan(d, rules_for(None)) == build_rule_plan(d, rules_for(None))      # same input, same plan


def test_custom_rules_size_age_and_first_match_wins(home):
    d = home / "Desktop"
    big = d / "big.bin"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    old = d / "old.txt"
    old.write_text("x")
    os.utime(old, (time.time() - 90 * 86400,) * 2)
    rules = [OrganizeRule("Big", "Big", min_size_mb=1), OrganizeRule("Old", "Old", older_than_days=30), OrganizeRule("All", "All", ext=["bin", "txt"])]
    assert classify(big, rules).name == "Big"
    assert classify(old, rules).name == "Old"
    (d / "x.rs").write_text("x")
    assert classify(d / "x.rs", rules) is None


def test_organize_by_rules_plan_apply_and_undo(ctx, registry, home):
    d = home / "Downloads"
    (d / "setup.exe").write_text("x")
    (d / "invoice1.pdf").write_text("x")
    plan = registry.get("plan_organize").fn(folder=str(d))
    assert plan["strategy"] == "by_rules" and plan["moves"] == 2
    registry.get("apply_plan").fn(plan_id=plan["plan_id"])
    assert (d / "Installers" / "setup.exe").exists() and (d / "Documents" / "Finance" / "invoice1.pdf").exists()
    registry.get("undo_last").fn()
    assert (d / "setup.exe").exists() and (d / "invoice1.pdf").exists()


# ---- 13/14. memory + retrieval ----
def test_memory_persists_kind_and_importance(tmp_path):
    m = Memory(tmp_path / "m.db")
    m.remember("editor", "Prefers VS Code", kind="preference", importance=3)
    m.close() if hasattr(m, "close") else None
    m2 = Memory(tmp_path / "m.db")
    assert m2.get_fact("editor").text == "Prefers VS Code"


def test_retrieval_returns_only_relevant_memories():
    m = Memory(":memory:")
    m.remember("printer", "Printer is an HP LaserJet M15 in the office", tags="hardware")
    m.remember("wifi", "Home wifi network is called Nebula")
    m.remember("pet", "The cat is called Pixel")
    got = m.retrieve("how do I print on my printer", limit=3)
    assert got and got[0]["key"] == "printer"
    assert all(g["key"] != "pet" for g in got)
    assert m.retrieve("zzzz qqqq") == []


def test_retrieval_is_typo_tolerant_and_respects_the_budget():
    m = Memory(":memory:")
    m.remember("printer", "Printer is an HP LaserJet", tags="hardware")
    assert m.retrieve("my pritner")[0]["key"] == "printer"
    for i in range(30):
        m.remember(f"fact{i}", "long fact about gardening " * 10)
    assert sum(len(x["text"]) for x in m.retrieve("gardening", limit=20, max_chars=500)) <= 700


def test_important_memories_outrank_trivia():
    m = Memory(":memory:")
    m.remember("a", "coffee preference is oat milk", importance=1)
    m.remember("b", "coffee machine is broken", importance=3)
    assert m.retrieve("coffee", limit=2)[0]["key"] == "b"


def test_past_tasks_are_recorded_and_retrievable():
    m = Memory(":memory:")
    m.record_task("organize my downloads folder", "Moved 12 files", ["plan_organize", "apply_plan"])
    assert m.recent_tasks(1)[0]["request"].startswith("organize")
    assert any(r["source"] == "task" for r in m.retrieve("organize downloads"))


# ---- 15. context manager ----
def test_budget_trims_old_tool_results_and_old_turns_but_keeps_the_request():
    big = "x" * 4000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a" * 400},
                 {"role": "tool", "content": big}]
    msgs.append({"role": "user", "content": "the current request"})
    out, stats = fit(msgs, 1200)
    assert total_tokens(out) < total_tokens(msgs)
    assert out[0]["role"] == "system" and out[-1]["content"] == "the current request"
    assert stats["trimmed_results"] or stats["dropped_messages"]
    assert len(msgs) == 20                                              # input untouched


def test_budget_leaves_small_prompts_alone_and_compacts_big_results():
    msgs = [{"role": "user", "content": "hi"}]
    assert fit(msgs, 1000)[0] == msgs
    huge = {"files": [{"path": f"C:/f{i}.txt", "name": f"f{i}.txt"} for i in range(500)], "display": "500 files"}
    assert len(compact_result(huge, limit=800)) <= 1000 and estimate_tokens("abcd" * 100) > 50


# ---- 17. background tasks ----
def test_background_task_runs_without_blocking_and_reports():
    events = []
    tm = TaskManager(on_event=events.append)
    t = tm.submit("slow", lambda: (time.sleep(0.2), {"display": "finished"})[1])
    assert t.status in ("pending", "running")                           # submit returned immediately
    tm.wait(t.id, 3)
    assert t.status == "done" and t.summary() == "finished" and events and events[0].id == t.id
    bad = tm.submit("bad", lambda: 1 / 0)
    tm.wait(bad.id, 3)
    assert bad.status == "failed" and "division" in bad.error
    tm.shutdown()


def test_background_task_can_be_cancelled_cooperatively():
    tm = TaskManager()

    def loop(cancel):
        while not cancel.is_set():
            time.sleep(0.01)
        return "stopped"

    t = tm.submit("loop", loop)
    time.sleep(0.1)
    tm.cancel(t.id)
    tm.wait(t.id, 3)
    assert t.status in ("cancelled", "done")
    tm.shutdown()


def test_background_task_tools_run_registered_tools_unattended(ctx, registry, home):
    from tests.conftest import FakeLLM
    from vyse.agent import Agent, Hooks
    Agent(FakeLLM([]), registry, ctx, Hooks(confirm=lambda p, r: True))
    (home / "Documents" / "bg.txt").write_text("x")
    out = registry.get("start_background_task").fn(tool="list_dir", args={"path": str(home / "Documents")})
    ctx.tasks.wait(out["task_id"], 5)
    st = registry.get("background_task_status").fn(task_id=out["task_id"])
    assert st["status"] == "done"
    assert registry.get("list_background_tasks").fn()["tasks"]
    risky = registry.get("start_background_task").fn(tool="run_command", args={"command": "echo hi"})
    ctx.tasks.wait(risky["task_id"], 5)
    st = registry.get("background_task_status").fn(task_id=risky["task_id"])
    assert st["status"] == "failed" and "confirmation" in st["display"].lower() + str(st)


# ---- 18. scheduler ----
def test_time_parsing():
    now = datetime(2026, 10, 5, 12, 0)
    assert parse_when("in 10 minutes", now) == now + timedelta(minutes=10)
    assert parse_when("2h", now) == now + timedelta(hours=2)
    assert parse_when("18:30", now) == datetime(2026, 10, 5, 18, 30)
    assert parse_when("09:00", now) == datetime(2026, 10, 6, 9, 0)      # already passed today -> tomorrow
    assert parse_when("tomorrow 09:00", now) == datetime(2026, 10, 6, 9, 0)
    assert parse_when("2026-12-01 08:00", now) == datetime(2026, 12, 1, 8, 0)
    with pytest.raises(ValueError):
        parse_when("whenever", now)
    assert interval_kwargs("every 30 minutes") == {"minutes": 30.0}


def test_scheduler_persists_and_rebuilds_jobs_from_the_database(tmp_path):
    db = tmp_path / "s.db"
    m = Memory(db)
    s = Scheduler(m, runner=lambda a: "ran")
    s.start()
    at = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
    row = s.add("tea", "once", {"at": at}, {"type": "reminder", "text": "Drink tea"})
    s.add("hourly", "interval", {"hours": 1}, {"type": "tool", "tool": "x", "args": {}})
    s.stop()
    s2 = Scheduler(Memory(db), runner=lambda a: "ran")        # "restart"
    s2.start()
    listing = s2.listing()
    assert len(listing) == 2 and all(r["next"] for r in listing)
    assert s2.remove(row["id"]) and len(s2.listing()) == 1
    s2.stop()


def test_reminder_fires_and_notifies(tmp_path):
    notes = []
    m = Memory(":memory:")
    s = Scheduler(m, runner=lambda a: "ran", notify=notes.append)
    s.start()
    row = s.add("tea", "once", {"at": (datetime.now() + timedelta(seconds=1)).isoformat(timespec="seconds")},
                {"type": "reminder", "text": "Drink tea"})
    for _ in range(60):
        if notes:
            break
        time.sleep(0.1)
    s.stop()
    assert notes and "Drink tea" in notes[0]
    assert m.get_schedule(row["id"])["runs"] == 1 and not m.get_schedule(row["id"])["enabled"]


def test_missed_one_shot_reminders_are_reported_at_startup():
    m = Memory(":memory:")
    s = Scheduler(m, runner=lambda a: "ran")
    m.add_schedule("old", "once", {"at": (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")},
                   {"type": "reminder", "text": "Call mom"})
    missed = s.start()
    s.stop()
    assert len(missed) == 1 and "Call mom" in missed[0]


def test_failed_scheduled_task_is_reported_not_swallowed():
    notes = []

    def runner(a):
        raise RuntimeError("disk on fire")

    m = Memory(":memory:")
    s = Scheduler(m, runner=runner, notify=notes.append)
    sid = m.add_schedule("job", "interval", {"hours": 1}, {"type": "tool", "tool": "x", "args": {}})
    s._fire(sid)
    assert notes and "failed" in notes[0] and "disk on fire" in notes[0]


def test_invalid_schedules_are_rejected():
    s = Scheduler(Memory(":memory:"), runner=lambda a: "")
    with pytest.raises(ValueError):
        s.add("past", "once", {"at": (datetime.now() - timedelta(hours=1)).isoformat()}, {"type": "reminder", "text": "x"})
    with pytest.raises(ValueError):
        build_trigger("cron", {"expr": "not a cron"})


def test_reminder_and_schedule_tools(ctx, registry):
    from tests.conftest import FakeLLM
    from vyse.agent import Agent, Hooks
    Agent(FakeLLM([]), registry, ctx, Hooks(confirm=lambda p, r: True))
    ctx.scheduler.start()
    try:
        r = registry.get("set_reminder").fn(text="stretch", when="in 30 minutes")
        assert r["next"]
        j = registry.get("schedule_task").fn(name="sysinfo", kind="interval", every="1 hour", tool="system_info")
        assert j["id"] != r["id"]
        assert len(registry.get("list_schedules").fn()["schedules"]) == 2
        registry.get("cancel_schedule").fn(id=r["id"])
        assert len(registry.get("list_schedules").fn()["schedules"]) == 1
        with pytest.raises(Exception):
            registry.get("schedule_task").fn(name="x", kind="interval", every="1 hour", tool="no_such_tool")
    finally:
        ctx.scheduler.stop()


# ---- 19. observability ----
def test_metrics_summary_aggregates_latency_failures_tokens_and_routing():
    m = Metrics(":memory:")
    m.record_model("q", 100, 50, 10, estimated=False, tools_offered=5, tool_calls=1, ok=True)
    m.record_model("q", 300, 70, 20, estimated=True, tools_offered=5, tool_calls=0, ok=False, error="timeout")
    m.record_tool("open_app", 40, True)
    m.record_tool("open_app", 60, False, error_type="tool_failed", repaired=True)
    m.record_tool("find_files", 5, True, cached=True)
    m.record_routing("open discord", ["open_app"], forced=True, matched=True)
    m.record_command("open discord", True, 2, 1, 0, 450)
    m.record_command("break", False, 3, 2, 2, 900)
    s = m.summary(1)
    assert s["model"]["calls"] == 2 and s["model"]["failures"] == 1 and s["model"]["prompt_tokens"] == 120
    assert s["model"]["completion_tokens"] == 30 and s["model"]["tokens_estimated"]
    assert s["tools"]["open_app"]["calls"] == 2 and s["tools"]["open_app"]["failures"] == 1
    assert s["tools"]["find_files"]["cached"] == 1
    assert s["tool_errors"] == {"tool_failed": 1} and s["repairs"] == 1
    assert s["commands"] == {"total": 2, "succeeded": 1, "failed": 1, "avg_ms": 675.0}
    assert s["routing"]["forced_tool"] == 1
    assert m.recent_failures(1)[0]["tool"] == "open_app"
    assert m.recent_commands(1)[0]["request"] == "break"


def test_metrics_can_be_disabled_and_pruned(tmp_path):
    off = Metrics(":memory:", enabled=False)
    off.record_tool("x", 1, True)
    assert off.summary(1)["tools"] == {}
    m = Metrics(tmp_path / "m.db")
    m.record_tool("x", 1, True)
    m.prune(0)
    assert m.summary(1)["tools"] == {}


def test_agent_records_metrics_for_a_real_turn(ctx):
    from tests.conftest import FakeLLM, say
    from tests.test_agent_features import calls
    from vyse.agent import Agent, Hooks
    from vyse.tools.registry import Registry
    reg = Registry()
    reg.add(lambda: {"display": "ok"}, risk="safe", name="ping")
    Agent(FakeLLM([calls(("ping", {})), say("pong")]), reg, ctx, Hooks()).run_turn("ping please")
    s = ctx.metrics.summary(1)
    assert s["model"]["calls"] == 2 and s["tools"]["ping"]["calls"] == 1
    assert s["commands"]["total"] == 1 and s["commands"]["succeeded"] == 1 and s["routing"]["requests"] == 1
    assert s["model"]["prompt_tokens"] > 0
