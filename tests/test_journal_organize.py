from pathlib import Path

from tests.test_files import agent_for, run
from vyse.journal import unique_path


def test_unique_path(tmp_path):
    f = tmp_path / "a.txt"
    assert unique_path(f) == f
    f.write_text("x")
    assert unique_path(f) == tmp_path / "a (1).txt"
    (tmp_path / "a (1).txt").write_text("x")
    assert unique_path(f) == tmp_path / "a (2).txt"


def test_journal_move_undo_roundtrip_and_collision_on_undo(ctx, home):
    j = ctx.journal
    a = home / "Downloads" / "a.txt"
    a.write_text("A")
    new = j.move(a, home / "Documents" / "a.txt")
    assert new.exists() and not a.exists()
    a.write_text("someone recreated it")     # original spot now occupied
    done, problems = j.undo_last()
    assert not problems and not new.exists()
    assert a.read_text() == "someone recreated it"          # not overwritten
    assert (home / "Downloads" / "a (1).txt").read_text() == "A"


def test_journal_undo_is_batch_wise_and_ordered(ctx, home):
    j = ctx.journal
    (home / "Downloads" / "1.txt").write_text("1")
    (home / "Downloads" / "2.txt").write_text("2")
    j.move(home / "Downloads" / "1.txt", home / "Documents" / "1.txt")
    b = j.new_batch()
    j.move(home / "Downloads" / "2.txt", home / "Documents" / "2.txt", b)
    j.undo_last()
    assert (home / "Downloads" / "2.txt").exists() and (home / "Documents" / "1.txt").exists()
    j.undo_last()
    assert (home / "Downloads" / "1.txt").exists()
    assert j.undo_last() == ([], ["Nothing to undo."])


def test_journal_trash_goes_to_trash_dir_not_deleted(ctx, home, cfg):
    f = home / "Documents" / "t.txt"
    f.write_text("t")
    dst = ctx.journal.trash(f)
    assert dst.exists() and cfg.trash_dir in dst.parents
    ctx.journal.undo_last()
    assert f.read_text() == "t"


def test_journal_persists_across_instances(ctx, home, cfg):
    from vyse.journal import Journal
    f = home / "Documents" / "p.txt"
    f.write_text("p")
    ctx.journal.move(f, home / "Downloads" / "p.txt")
    j2 = Journal(cfg.journal_path, cfg.trash_dir)
    j2.undo_last()
    assert f.exists()


def make_messy(folder: Path):
    for n in ("a.pdf", "b.png", "c.jpg", "d.zip", "e.xyz", "f.txt", "z_notes.docx"):
        (folder / n).write_text(n)
    (folder / "subdir").mkdir()
    (folder / "subdir" / "keep.txt").write_text("k")
    (folder / "partial.crdownload").write_text("p")


def snapshot(folder: Path):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def test_plan_organize_is_pure_dry_run(ctx, registry, home):
    f = home / "Downloads"
    make_messy(f)
    before = snapshot(f)
    r = run(agent_for(ctx, registry), "plan_organize", folder=str(f), strategy="by_type")
    assert r["ok"] and r["dry_run"] and r["moves"] == 7           # skips dir + partial download
    assert snapshot(f) == before                                  # nothing touched
    assert "Images: 2" in r["preview"] and "Documents" in r["preview"]


def test_apply_plan_moves_verifies_and_undo_restores(ctx, registry, home):
    f = home / "Downloads"
    make_messy(f)
    before = snapshot(f)
    a = agent_for(ctx, registry, confirm=lambda p, r: True)    # 7 files > threshold 5 -> confirmation
    pid = run(a, "plan_organize", folder=str(f))["plan_id"]
    r = run(a, "apply_plan", plan_id=pid)
    assert r["ok"] and r["moved"] == 7 and r["verified"] is True
    assert (f / "Images" / "b.png").exists() and (f / "Documents" / "a.pdf").exists() and (f / "Other" / "e.xyz").exists()
    assert not (f / "a.pdf").exists()
    assert (f / "subdir" / "keep.txt").exists()                  # untouched
    u = run(a, "undo_last")
    assert u["ok"] and u["verified"]
    assert [p for p in snapshot(f) if p in before] == before
    assert not (f / "b.png").parent.joinpath("Images", "b.png").exists()


def test_apply_plan_over_threshold_requires_confirmation(ctx, registry, home):
    f = home / "Downloads"
    make_messy(f)
    asked = []
    a = agent_for(ctx, registry, confirm=lambda p, r: asked.append(p) or False)
    pid = run(a, "plan_organize", folder=str(f))["plan_id"]
    r = run(a, "apply_plan", plan_id=pid)
    assert r.get("declined") and (f / "a.pdf").exists()
    assert asked and "7 file(s)" in asked[0]


def test_small_plan_applies_without_confirmation(ctx, registry, home):
    f = home / "Downloads"
    (f / "a.pdf").write_text("a")
    (f / "b.png").write_text("b")
    asked = []
    a = agent_for(ctx, registry, confirm=lambda p, r: asked.append(p) or False)
    pid = run(a, "plan_organize", folder=str(f))["plan_id"]
    assert run(a, "apply_plan", plan_id=pid)["ok"] and not asked


def test_apply_plan_collision_safe(ctx, registry, home):
    f = home / "Downloads"
    (f / "a.pdf").write_text("new")
    (f / "Documents").mkdir()
    (f / "Documents" / "a.pdf").write_text("existing")
    a = agent_for(ctx, registry)
    pid = run(a, "plan_organize", folder=str(f))["plan_id"]
    r = run(a, "apply_plan", plan_id=pid)
    assert r["verified"]
    assert (f / "Documents" / "a.pdf").read_text() == "existing"
    assert (f / "Documents" / "a (1).pdf").read_text() == "new"


def test_apply_plan_unknown_or_traversal_id(ctx, registry):
    a = agent_for(ctx, registry)
    assert not run(a, "apply_plan", plan_id="nope")["ok"]
    assert not run(a, "apply_plan", plan_id="..\\..\\evil")["ok"]


def test_plan_is_single_use(ctx, registry, home):
    f = home / "Downloads"
    (f / "a.pdf").write_text("a")
    a = agent_for(ctx, registry)
    pid = run(a, "plan_organize", folder=str(f))["plan_id"]
    assert run(a, "apply_plan", plan_id=pid)["ok"]
    assert not run(a, "apply_plan", plan_id=pid)["ok"]


def test_plan_organize_protected_folder_blocked(ctx, registry, home):
    (home / "protected" / "x.txt").write_text("x")
    r = run(agent_for(ctx, registry), "plan_organize", folder=str(home / "protected"))
    assert r.get("blocked")


def test_by_date_and_by_name(ctx, registry, home):
    f = home / "Downloads"
    (f / "alpha.txt").write_text("a")
    (f / "1file.txt").write_text("b")
    a = agent_for(ctx, registry)
    r = run(a, "plan_organize", folder=str(f), strategy="by_name")
    assert "A: 1" in r["preview"] and "0-9 & symbols: 1" in r["preview"]
    r = run(a, "plan_organize", folder=str(f), strategy="by_date")
    import time
    assert time.strftime("%Y-%m") in r["preview"]
    assert not run(a, "plan_organize", folder=str(f), strategy="by_color")["ok"]   # enum validation


def test_apply_plan_without_id_uses_latest_plan(ctx, registry, home):
    f = home / "Downloads"
    (f / "a.pdf").write_text("a")
    a = agent_for(ctx, registry)
    run(a, "plan_organize", folder=str(f))
    assert run(a, "apply_plan")["ok"] and (f / "Documents" / "a.pdf").exists()
    assert not run(a, "apply_plan")["ok"]


def test_undo_removes_folders_created_by_organize(ctx, registry, home):
    f = home / "Downloads"
    (f / "a.pdf").write_text("a")
    (f / "b.png").write_text("b")
    a = agent_for(ctx, registry)
    run(a, "plan_organize", folder=str(f))
    run(a, "apply_plan")
    assert (f / "Images").exists()
    run(a, "undo_last")
    assert not (f / "Images").exists() and not (f / "Documents").exists()
    assert (f / "a.pdf").exists() and (f / "b.png").exists()
