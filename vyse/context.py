"""Shared runtime context handed to tool modules (config, policy, journal, memory)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path

from .config import Config
from .journal import Journal
from .memory import Memory
from .policy import Policy


@dataclass
class Context:
    cfg: Config
    policy: Policy
    journal: Journal
    memory: Memory
    # Set by the Agent: runs a tool by name through validation + policy (+ confirmation). Used by routines.
    run_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None

    @classmethod
    def build(cls, cfg: Config, memory: Memory | None = None) -> "Context":
        cfg.ensure_dirs()
        return cls(cfg=cfg, policy=Policy(cfg),
                   journal=Journal(cfg.journal_path, cfg.trash_dir),
                   memory=memory or Memory(cfg.db_path))
