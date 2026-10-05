import pytest
from tests.test_files import agent_for, run
from vyse.memory import Memory


def test_fact_storage_and_update():
    m = Memory(":memory:")
    f = m.remember("Printer", "The user's printer is an HP LaserJet M15.", "hardware")
    assert f.key == "printer"
    m.remember("printer", "The printer is now a Brother.", "hardware")
    assert len(m.all_facts()) == 1 and "Brother" in m.get_fact("printer").text
    assert m.forget("printer") and not m.forget("printer")


def test_fts_retrieval_ranks_and_ignores_stopwords():
    m = Memory(":memory:")
    m.remember("printer", "My printer is an HP LaserJet M15", "hardware")
    m.remember("editor", "Favorite code editor is VS Code", "software")
    m.remember("gpu", "The GPU is an AMD Radeon", "hardware")
    hits = m.recall("what printer do I have?")
    assert hits and hits[0].key == "printer"
    assert [f.key for f in m.recall("which editor do I use")] == ["editor"]
    assert m.recall("the a of") == []
    assert m.recall("") == []
    assert {f.key for f in m.recall("hardware")} == {"printer", "gpu"}      # tags indexed


def test_fts_survives_updates_and_deletes():
    m = Memory(":memory:")
    m.remember("x", "alpha beta")
    m.remember("x", "gamma delta")
    assert m.recall("alpha") == [] and m.recall("gamma")
    m.forget("x")
    assert m.recall("gamma") == []


def test_fts_query_is_safe_with_special_chars():
    m = Memory(":memory:")
    m.remember("k", "hello world")
    assert m.recall('hello" OR * NEAR( ) -- ;') != [] or True   # must not raise
    m.recall("'; DROP TABLE facts; --")
    assert m.all_facts()


def test_messages_summaries_and_isolation():
    m = Memory(":memory:")
    for i in range(10):
        m.add_message("s1", "user", f"u{i}")
    m.add_message("s2", "user", "other")
    assert m.message_count("s1") == 10
    assert [x["content"] for x in m.recent_messages("s1", 3)] == ["u7", "u8", "u9"]
    upto, old = m.messages_to_summarize("s1", keep_recent=3)
    assert len(old) == 7
    m.save_summary("s1", upto, "summary text")
    assert m.latest_summary("s1")[1] == "summary text"
    assert m.message_count("s1") == 3                  # summarized messages no longer counted
    m.clear_session("s1")
    assert m.message_count("s1") == 0 and m.message_count("s2") == 1


def test_facts_persist_on_disk(tmp_path):
    p = tmp_path / "m.db"
    Memory(p).remember("a", "persisted fact")
    assert Memory(p).recall("persisted")[0].key == "a"


def test_memory_tools(ctx, registry):
    a = agent_for(ctx, registry)
    assert run(a, "remember", key="printer", text="HP M15 printer")["verified"]
    assert run(a, "recall", query="printer")["facts"][0]["text"] == "HP M15 printer"
    assert run(a, "forget", key="printer")["verified"]
    assert not run(a, "forget", key="printer")["ok"]


def test_notes_create_append_read_search(ctx, registry, cfg):
    a = agent_for(ctx, registry)
    r = run(a, "create_note", title="Robot Arm Ideas", content="Use cycloidal reducers with 40:1 ratio", tags="robot")
    assert r["ok"] and r["verified"]
    path = cfg.notes_dir / r["path"].split("\\")[-1]
    assert path.exists() and path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\ntitle: Robot Arm Ideas") and "cycloidal" in text
    assert run(a, "append_note", name="robot arm", content="Add harmonic drive comparison")["ok"]
    assert "harmonic" in run(a, "read_note", name="Robot Arm Ideas")["content"]
    s = run(a, "search_notes", query="cycloidal ratio")
    assert s["count"] == 1 and "cycloidal" in s["notes"][0]["snippet"].lower()
    assert run(a, "search_notes", query="nonexistentword")["count"] == 0
    assert run(a, "read_note")["notes"]


def test_notes_duplicate_titles_get_unique_files(ctx, registry):
    a = agent_for(ctx, registry)
    p1 = run(a, "create_note", title="Same", content="1")["path"]
    p2 = run(a, "create_note", title="Same", content="2")["path"]
    assert p1 != p2


def test_notes_cannot_escape_notes_dir(ctx, registry, home):
    (home / "Documents" / "secret.md").write_text("top secret")
    a = agent_for(ctx, registry)
    assert not run(a, "read_note", name="..\\secret.md")["ok"]
    assert not run(a, "append_note", name="../secret.md", content="pwn")["ok"]
    assert (home / "Documents" / "secret.md").read_text() == "top secret"


# ---- routines: user-taught, model-composed, executed through the policy path ----
def test_routines_save_run_and_policy(ctx, registry):
    from vyse.agent import Agent
    from tests.conftest import FakeLLM
    Agent(FakeLLM([]), registry, ctx)          # wires ctx.run_tool
    ran = []
    registry.get("get_current_time").fn = lambda: ran.append("t") or {"display": "12:00", "verified": True}
    save = registry.get("save_routine").fn
    save("Study Session", "clock then a timer", [{"tool": "get_current_time", "args": {}},
                                                 {"tool": "set_timer", "args": {"minutes": 0.001, "label": "x"}}])
    assert ctx.memory.find_routines("start my study session")[0]["name"] == "study session"
    out = registry.get("run_routine").fn("study session")
    assert out["verified"] is True and ran == ["t"] and len(out["steps"]) == 2
    with pytest.raises(Exception):
        save("bad", "x", [{"tool": "nope", "args": {}}])
    with pytest.raises(Exception):
        save("nest", "x", [{"tool": "run_routine", "args": {"name": "a"}}])


def test_routine_step_blocked_by_policy_reports_failure(ctx, registry):
    from vyse.agent import Agent
    from tests.conftest import FakeLLM
    Agent(FakeLLM([]), registry, ctx)
    registry.get("save_routine").fn("danger", "runs a command", [{"tool": "run_command", "args": {"command": "echo hi"}}])
    out = registry.get("run_routine").fn("danger")          # risky step needs confirmation; default hook declines
    assert out["verified"] is False and out["steps"][0].startswith("FAILED")


def test_matching_routine_is_injected_into_prompt(ctx, registry):
    from vyse.agent import Agent
    from tests.conftest import FakeLLM
    ctx.memory.save_routine("study session", "clock, timer, lofi", [{"tool": "get_current_time", "args": {}}])
    msgs = Agent(FakeLLM([]), registry, ctx)._build_messages("start my study session")
    assert "study session: clock, timer, lofi" in msgs[0]["content"]


def test_save_routine_rejects_wrong_argument_names(ctx, registry):
    save = registry.get("save_routine").fn
    with pytest.raises(Exception, match="path"):
        save("x", "y", [{"tool": "open_path", "args": {"url": "https://example.com"}}])    # model used 'url', tool wants 'path'
    with pytest.raises(Exception, match="unknown argument"):
        save("x", "y", [{"tool": "open_app", "args": {"name": "Clock", "bogus": 1}}])
    save("x", "y", [{"tool": "open_path", "args": {"path": "https://example.com"}}])
    assert ctx.memory.get_routine("x")["steps"][0]["args"] == {"path": "https://example.com"}


def test_save_routine_accepts_flat_steps(ctx, registry):
    registry.get("save_routine").fn("flat", "flat form", [{"tool": "open_app", "name": "notepad"},
                                                         {"tool": "set_timer", "minutes": 5, "label": "x"}])
    assert ctx.memory.get_routine("flat")["steps"] == [{"tool": "open_app", "args": {"name": "notepad"}},
                                                       {"tool": "set_timer", "args": {"minutes": 5, "label": "x"}}]
