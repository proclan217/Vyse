"""Tool packages. `build_registry` wires every tool module into a Registry."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import Registry, Tool, ToolError, tool  # noqa: F401

if TYPE_CHECKING:
    from ..context import Context


def build_registry(ctx: "Context", *, extra: bool = True) -> Registry:
    """Create a registry with all built-in tools. Optional integrations register only when configured."""
    from . import automation, files, memory_tools, notes, organize, routines, system, web

    reg = Registry()
    for mod in (files, organize, notes, memory_tools, routines, automation, web, system):
        mod.register(reg, ctx)
    if extra:
        from . import google, mcp_client
        google.register(reg, ctx)
        mcp_client.register(reg, ctx)
    return reg
