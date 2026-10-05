import os
import time

from vyse.agent import Agent
from vyse.llm import ToolCall
from vyse.tools.files import resolve_path


def agent_for(ctx, registry, confirm=lambda p, r: False):
    from vyse.agent import Hooks
    from tests.conftest import FakeLLM
    return Agent(FakeLLM([]), registry, ctx, Hooks(confirm=confirm))


def run(agent, _tool, /, **args):
    from vyse.agent import TaskState
    return agent.execute_call(ToolCall(_tool, args), TaskState("t"))


def test_resolve_friendly_names(home, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    assert resolve_path("Downloads") == home / "Downloads"
    assert resolve_path("downloads/sub") == home / "Downloads" / "sub"
    assert resolve_path("") == home


def test_find_files_by_extension_and_recency(ctx, registry, home):
    (home / "Documents" / "a.pdf").write_text("x")
    old = home / "Documents" / "old.pdf"
    old.write_text("x")
    os.utime(old, (time.time() - 30 * 86400,) * 2)
    (home / "Documents" / "b.txt").write_text("x")
    a = agent_for(ctx, registry)
    r = run(a, "find_files", extension="pdf", folder=str(home / "Documents"), modified_within_days=7)
    assert r["ok"] and r["count"] == 1 and r["files"][0]["name"] == "a.pdf"
    r = run(a, "find_files", query="b*", folder=str(home / "Documents"))
    assert [f["name"] for f in r["files"]] == ["b.txt"]


def test_find_files_skips_protected(ctx, registry, home):
    (home / "protected" / "secret.pdf").write_text("x")
    r = run(agent_for(ctx, registry), "find_files", extension="pdf", folder=str(home))
    assert all("protected" not in f["path"] for f in r.get("files", []))


def test_read_and_list(ctx, registry, home):
    f = home / "Documents" / "n.txt"
    f.write_text("hello world")
    a = agent_for(ctx, registry)
    assert run(a, "read_file", path=str(f))["content"] == "hello world"
    (home / "Documents" / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert not run(a, "read_file", path=str(home / "Documents" / "bin.dat"))["ok"]
    assert run(a, "list_dir", path=str(home / "Documents"))["total"] >= 2


def test_move_inside_allowed_root_is_automatic_and_verified(ctx, registry, home):
    src = home / "Downloads" / "x.txt"
    src.write_text("1")
    (home / "Documents" / "dest").mkdir()
    r = run(agent_for(ctx, registry), "move", src=str(src), dst=str(home / "Documents" / "dest"))
    assert r["ok"] and r["verified"] is True
    assert not src.exists() and (home / "Documents" / "dest" / "x.txt").exists()


def test_move_collision_never_overwrites(ctx, registry, home):
    (home / "Downloads" / "x.txt").write_text("new")
    (home / "Documents" / "x.txt").write_text("old")
    run(agent_for(ctx, registry), "move", src=str(home / "Downloads" / "x.txt"), dst=str(home / "Documents"))
    assert (home / "Documents" / "x.txt").read_text() == "old"
    assert (home / "Documents" / "x (1).txt").read_text() == "new"


def test_move_outside_roots_needs_confirmation(ctx, registry, home):
    src = home / "Downloads" / "x.txt"
    src.write_text("1")
    asked = []
    a = agent_for(ctx, registry, confirm=lambda p, r: asked.append(p) or False)
    r = run(a, "move", src=str(src), dst=str(home / "outside"))
    assert r.get("declined") and asked and src.exists()
    a2 = agent_for(ctx, registry, confirm=lambda p, r: True)
    assert run(a2, "move", src=str(src), dst=str(home / "outside"))["ok"]
    assert (home / "outside" / "x.txt").exists()


def test_move_into_protected_is_blocked_even_with_auto_yes(ctx, registry, home):
    src = home / "Downloads" / "x.txt"
    src.write_text("1")
    a = agent_for(ctx, registry)
    a.auto_yes = True
    r = run(a, "move", src=str(src), dst=str(home / "protected"))
    assert r.get("blocked") and src.exists()


def test_trash_never_hard_deletes_and_undo_restores(ctx, registry, home, cfg):
    f = home / "Downloads" / "gone.txt"
    f.write_text("keep me")
    a = agent_for(ctx, registry)
    r = run(a, "trash", path=str(f))
    assert r["ok"] and r["verified"] and not f.exists()
    assert os.path.exists(r["trashed_to"]) and open(r["trashed_to"]).read() == "keep me"
    assert str(cfg.trash_dir) in r["trashed_to"]
    u = run(a, "undo_last")
    assert u["ok"] and f.read_text() == "keep me"


def test_copy_and_make_dir(ctx, registry, home):
    f = home / "Documents" / "c.txt"
    f.write_text("c")
    a = agent_for(ctx, registry)
    assert run(a, "copy", src=str(f), dst=str(home / "Downloads"))["verified"]
    assert f.exists() and (home / "Downloads" / "c.txt").exists()
    assert run(a, "make_dir", path=str(home / "Documents" / "new" / "deep"))["verified"]
    run(a, "undo_last")
    assert not (home / "Documents" / "new").exists()


def test_resolve_relative_falls_back_to_common_folders(home, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    (home / "Desktop" / "Proj").mkdir()
    (home / "Desktop" / "Proj" / "r.md").write_text("x")
    assert resolve_path("Proj/r.md") == home / "Desktop" / "Proj" / "r.md"
    assert resolve_path("nothing/here") == home / "nothing" / "here"
