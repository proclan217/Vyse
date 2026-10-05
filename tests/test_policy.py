from vyse.policy import ALLOW, BLOCK, CONFIRM, Decision, Policy
from vyse.tools.registry import Registry


def test_path_inside_allowed_root_write_ok(ctx, home):
    assert ctx.policy.check_path(home / "Downloads" / "a.txt", write=True).action == ALLOW


def test_path_outside_allowed_root_write_needs_confirmation(ctx, home):
    assert ctx.policy.check_path(home / "outside" / "a.txt", write=True).action == CONFIRM


def test_protected_path_blocked_for_read_and_write(ctx, home):
    p = home / "protected" / "secret.txt"
    assert ctx.policy.check_path(p, write=True).action == BLOCK
    assert ctx.policy.check_path(p, write=False).action == BLOCK


def test_dotdot_escape_is_caught(ctx, home):
    sneaky = home / "Downloads" / ".." / "protected" / "x.txt"
    assert ctx.policy.check_path(sneaky, write=True).action == BLOCK
    sneaky2 = home / "Downloads" / ".." / "outside" / "x.txt"
    assert ctx.policy.check_path(sneaky2, write=True).action == CONFIRM


def test_vyse_data_dir_is_off_limits_except_trash(ctx, cfg):
    assert ctx.policy.check_path(cfg.data_dir / "vyse.db", write=False).action == BLOCK
    assert ctx.policy.check_path(cfg.data_dir / "google_token.json", write=False).action == BLOCK
    assert ctx.policy.check_path(cfg.trash_dir / "x", write=False).action != BLOCK


def test_real_windows_protected_defaults():
    from vyse.config import load_config
    pol = Policy(load_config())
    assert pol.is_protected("C:\\Windows\\System32\\cmd.exe")
    assert pol.is_protected("C:\\Program Files\\Foo\\bar.dll")
    assert pol.check_path("C:\\Windows", write=True).action == BLOCK


def test_risk_classification(ctx):
    reg = Registry()
    reg.add(lambda: {}, risk="safe", name="s")
    reg.add(lambda: {}, risk="write", name="w")
    reg.add(lambda: {}, risk="risky", name="r")
    assert ctx.policy.decide(reg.get("s"), {}).action == ALLOW
    assert ctx.policy.decide(reg.get("w"), {}).action == ALLOW
    assert ctx.policy.decide(reg.get("r"), {}).action == CONFIRM


def test_assessor_can_only_tighten(ctx):
    reg = Registry()
    reg.add(lambda: {}, risk="risky", name="r", assess=lambda a: Decision(ALLOW))
    assert ctx.policy.decide(reg.get("r"), {}).action == CONFIRM   # assessor cannot loosen risky
    reg.add(lambda: {}, risk="safe", name="b", assess=lambda a: Decision(BLOCK, "no"))
    assert ctx.policy.decide(reg.get("b"), {}).action == BLOCK


def test_broken_assessor_fails_closed(ctx):
    reg = Registry()

    def boom(a):
        raise RuntimeError("x")

    reg.add(lambda: {}, risk="safe", name="b", assess=boom)
    assert ctx.policy.decide(reg.get("b"), {}).action == CONFIRM


def test_commands_always_confirm_and_destructive_blocked(ctx):
    assert ctx.policy.check_command("Get-Date").action == CONFIRM
    for bad in ("format C:", "shutdown /s /t 0", "reg delete HKLM\\Software\\X /f",
                "Remove-Item -Recurse -Force C:\\", "diskpart"):
        assert ctx.policy.check_command(bad).action == BLOCK, bad


def test_confirm_risky_flag_off_allows_risky(cfg):
    from vyse.context import Context
    from vyse.memory import Memory
    cfg.confirm_risky = False
    c = Context.build(cfg, Memory(":memory:"))
    reg = Registry()
    reg.add(lambda: {}, risk="risky", name="r")
    assert c.policy.decide(reg.get("r"), {}).action == ALLOW
