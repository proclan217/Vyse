"""Features 1-5: tool validation, fuzzy resolution, app launcher, JSON/argument repair, structured errors."""
from __future__ import annotations

from tests.conftest import FakeLLM, call, say
from vyse.agent import Agent, Hooks
from vyse.apps import AppCatalog, AppRecord
from vyse.llm import LLMResponse, ToolCall
from vyse.fuzzy import best_match, fuzzy_path, resolve, suggestions
from vyse.repair import parse_arguments, repair_json, repair_tool_name
from vyse.tools.registry import Registry, ToolError
from vyse.validation import (INVALID_ARGUMENT, MISSING_ARGUMENT, UNKNOWN_TOOL, resolve_tool, validate_call)


def greet(name: str, times: int = 1, mode: str = "polite") -> dict:
    """Greet.

    Args:
        name: Who.
        times: How often.
        mode: Style.
    """
    return {"display": f"hi {name} x{times}"}


def make_reg() -> Registry:
    reg = Registry()
    reg.add(greet, risk="safe", parameters={
        "type": "object", "required": ["name"],
        "properties": {"name": {"type": "string"}, "times": {"type": "integer"}, "mode": {"type": "string", "enum": ["polite", "loud"]}}})
    return reg


# ---- 1. validation ----
def test_validation_missing_wrong_type_and_extra_args():
    t = make_reg().get("greet")
    v = validate_call(t, {})
    assert v.error and v.error["error_type"] == MISSING_ARGUMENT and v.error["retryable"]
    assert "greet(" in v.error["expected"]
    v = validate_call(t, {"name": "a", "times": "abc"})
    assert v.error and v.error["error_type"] == INVALID_ARGUMENT
    v = validate_call(t, {"name": "a", "times": "3", "bogus": 1})
    assert v.error is None and v.args == {"name": "a", "times": 3} and v.ignored == ["bogus"]


def test_validation_enum_case_fix_and_suggestion():
    t = make_reg().get("greet")
    assert validate_call(t, {"name": "a", "mode": "LOUD"}).args["mode"] == "loud"
    v = validate_call(t, {"name": "a", "mode": "lod"})
    assert v.error and v.error["error_type"] == INVALID_ARGUMENT


def test_unknown_tool_name_is_repaired_or_reported():
    reg = make_reg()
    tool, err, note = resolve_tool("Greet", reg)
    assert tool is not None and err is None
    tool, err, _ = resolve_tool("totally_other", reg)
    assert tool is None and err["error_type"] == UNKNOWN_TOOL
    assert repair_tool_name("functions.greet", ["greet"])[0] == "greet"


# ---- 2. fuzzy ----
def test_fuzzy_matching_corrects_typos_but_not_short_or_ambiguous():
    assert best_match("discrod", ["Discord", "Spotify"])[0].value == "Discord"
    assert resolve("discrod", ["Discord", "Spotify"]).status == "fuzzy"
    assert resolve("Discord", ["Discord"]).status == "exact"
    assert not resolve("ab", ["abc", "abd"]).ok          # too short to guess, never auto-picked
    assert resolve("zzzzzz", ["Discord"]).status == "none"
    assert suggestions("spotfy", ["Spotify", "Steam"])[0] == "Spotify"


def test_fuzzy_path_corrects_the_missing_tail_only(home):
    real = home / "Documents" / "Invoices"
    real.mkdir()
    fixed, _ = fuzzy_path(home / "Documents" / "Invoces")
    assert fixed == real
    fixed, _ = fuzzy_path(home / "Documents" / "qqqqqqq")
    assert fixed is None


def test_read_tools_autocorrect_but_write_tools_only_suggest(ctx, registry, home):
    (home / "Documents" / "Budget2026.txt").write_text("hello")
    r = registry.get("read_file").fn(str(home / "Documents" / "Budget2026.txt".replace("2026", "2O26")))
    assert r["corrected_from"] and "hello" in r["content"]
    try:
        registry.get("move").fn(str(home / "Documents" / "Budjet2026.txt"), str(home / "Desktop"))
    except ToolError as e:
        assert e.suggestions
    else:
        raise AssertionError("move must not guess a source file")


def test_fuzzy_correction_cannot_escape_policy(ctx, registry, home):
    (home / "protected" / "secrets.txt").write_text("x")
    try:
        registry.get("read_file").fn(str(home / "protected" / "secret.txt"))
    except ToolError:
        pass
    else:
        raise AssertionError("a protected file must not be reached through fuzzy correction")


# ---- 3. app launcher ----
def test_app_catalog_fuzzy_resolves_installed_apps(cfg):
    cfg.app_scan = True
    cat = AppCatalog(cfg, scanner=lambda: [AppRecord("Discord", "lnk", "C:/x/Discord.lnk"), AppRecord("Medal", "lnk", "C:/x/Medal.lnk"),
                                           AppRecord("Visual Studio Code", "lnk", "C:/x/vsc.lnk")],
                     cache_path=cfg.data_dir / "apps.json")
    assert cat.resolve("discrod").match.key.name == "Discord"
    assert cat.resolve("visual studio").ok
    assert cat.resolve("qwertyuiop").status == "none"
    assert "Medal" in cat.names()


# ---- 4. repair ----
def test_repair_malformed_json_arguments():
    args, repairs, err = parse_arguments("{'name': 'Bob', 'times': 2,}")
    assert err is None and args == {"name": "Bob", "times": 2} and repairs
    args, _, err = parse_arguments('```json\n{"name": "x"}\n```')
    assert args == {"name": "x"}
    args, _, err = parse_arguments('{"name": "x"')
    assert args == {"name": "x"}
    args, _, err = parse_arguments("not json at all")
    assert args is None and err
    assert repair_json("{'a': 1}")[0] == {"a": 1}


def test_windows_path_with_backslash_escapes_survives_repair():
    args, _, err = parse_arguments('{"path": "C:\\Users\\me\\file.txt"}')
    assert err is None and args["path"] == "C:\\Users\\me\\file.txt"


# ---- 5. error recovery (the model gets a structured error and corrects itself) ----
def test_model_recovers_from_structured_error(ctx):
    reg = make_reg()
    llm = FakeLLM([call("greet"), LLMResponse(tool_calls=[ToolCall(name="greet", arguments={"name": "Ann"})]), say("Done.")])
    a = Agent(llm, reg, ctx, Hooks(confirm=lambda p, r: False))
    assert a.run_turn("greet someone") == "Done."
    tool_msgs = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert any("missing_argument" in m and "retryable" in m for m in tool_msgs)


def test_tool_error_suggestions_reach_the_model(ctx):
    def boom() -> dict:
        """Fail."""
        raise ToolError("nope", suggestions=["alpha", "beta"], hint="try alpha")

    reg = Registry()
    reg.add(boom, risk="safe")
    llm = FakeLLM([call("boom"), say("ok")])
    Agent(llm, reg, ctx, Hooks(confirm=lambda p, r: False)).run_turn("boom")
    msg = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "alpha" in msg and "tool_failed" in msg
