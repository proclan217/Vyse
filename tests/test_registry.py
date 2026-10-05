from typing import Literal, Optional

import pytest

from vyse.tools.registry import Registry, build_schema


def sample(path: str, count: int = 3, flag: bool = False, mode: Literal["a", "b"] = "a",
           tags: list[str] | None = None, ratio: float = 1.0, maybe: Optional[str] = None) -> dict:
    """Do a sample thing.

    Args:
        path: Where to look.
        count: How many.
    """
    return {}


def test_schema_generation_types_and_required():
    summary, schema = build_schema(sample)
    props = schema["properties"]
    assert summary == "Do a sample thing."
    assert schema["required"] == ["path"]
    assert props["path"] == {"type": "string", "description": "Where to look."}
    assert props["count"]["type"] == "integer" and props["count"]["description"] == "How many."
    assert props["flag"]["type"] == "boolean"
    assert props["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["ratio"]["type"] == "number"
    assert props["maybe"]["type"] == "string"


def test_tool_registration_and_schema_shape():
    reg = Registry()
    reg.add(sample, risk="write", keywords=("sample",))
    t = reg.get("sample")
    assert t and t.risk == "write" and "sample" in reg
    s = t.schema()
    assert s["type"] == "function" and s["function"]["name"] == "sample"
    assert s["function"]["parameters"]["required"] == ["path"]


def test_decorator_registers_on_given_registry():
    reg = Registry()

    @reg.tool(risk="safe")
    def hello(name: str) -> dict:
        """Say hi."""
        return {"hi": name}

    assert reg.get("hello").fn("x") == {"hi": "x"}


def test_invalid_risk_rejected():
    with pytest.raises(ValueError):
        Registry().add(sample, risk="dangerous")


def test_select_prefers_relevant_group_and_keeps_always_tools():
    reg = Registry()
    reg.add(lambda: {}, risk="safe", name="get_time", always=True)
    reg.add(lambda: {}, risk="safe", name="plan_organize", group="organize", keywords=("organize", "tidy"))
    reg.add(lambda: {}, risk="write", name="apply_plan", group="organize")
    reg.add(lambda: {}, risk="safe", name="weather", group="web", keywords=("weather",))
    names = [t.name for t in reg.select("please tidy my downloads")]
    assert names[0] == "get_time"
    assert "plan_organize" in names and "apply_plan" in names   # whole group travels together
    assert "weather" not in names


def test_select_falls_back_to_everything_when_nothing_matches():
    reg = Registry()
    reg.add(lambda: {}, risk="safe", name="a", keywords=("x",))
    reg.add(lambda: {}, risk="safe", name="b", keywords=("y",))
    assert {t.name for t in reg.select("zzz qqq")} == {"a", "b"}


def test_real_registry_exposes_documented_params(registry):
    for t in registry.all():
        assert t.description, t.name
        assert t.risk in ("safe", "write", "risky")
    plan = registry.get("plan_organize").parameters
    assert plan["properties"]["strategy"]["enum"] == ["by_type", "by_date", "by_name"]
    assert registry.get("run_command").risk == "risky"
