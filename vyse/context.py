"""Shared runtime context handed to tool modules (config, policy, journal, memory and the feature services)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .apps import AppCatalog
from .cache import ToolCache
from .config import Config, expand
from .indexer import FileIndex
from .journal import Journal
from .memory import Memory
from .observability import Metrics
from .policy import Policy
from .scheduler import Scheduler
from .tasks import TaskManager


@dataclass
class Context:
    cfg: Config
    policy: Policy
    journal: Journal
    memory: Memory
    # Set by the Agent: runs a tool by name through validation + policy (+ confirmation). Used by routines.
    run_tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    # Same, but unattended: anything that would need a confirmation is declined (background tasks, schedules).
    run_tool_unattended: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    cache: ToolCache = field(default_factory=ToolCache)
    metrics: Metrics = field(default_factory=lambda: Metrics(":memory:"))
    tasks: TaskManager = field(default_factory=TaskManager)
    apps: AppCatalog | None = None
    index: FileIndex | None = None
    scheduler: Any = None                    # vyse.scheduler.Scheduler (created by build(); optional in tests)
    # Set by the automation tools: runs a tool/routine action unattended and returns a result text (raises on failure).
    action_runner: Callable[[dict[str, Any]], str] | None = None
    notify: Callable[[str], None] = field(default=lambda text: None)   # the CLI replaces this to print background events

    @classmethod
    def build(cls, cfg: Config, memory: Memory | None = None) -> "Context":
        cfg.ensure_dirs()
        policy = Policy(cfg)
        ctx = cls(cfg=cfg, policy=policy,
                  journal=Journal(cfg.journal_path, cfg.trash_dir),
                  memory=memory or Memory(cfg.db_path),
                  cache=ToolCache(cfg.cache_max_entries),
                  metrics=Metrics(cfg.data_dir / "metrics.db", enabled=cfg.metrics_enabled),
                  apps=AppCatalog(cfg))
        if cfg.index.enabled:
            roots = [expand(r) for r in cfg.index.roots]
            ctx.index = FileIndex(cfg.data_dir / "file_index.db", roots, pruner_factory=policy.walk_pruner,
                                  is_protected=policy.is_protected,
                                  max_files=cfg.index.max_files, stale_minutes=cfg.index.stale_minutes)
        ctx.tasks.on_event = lambda t: ctx.emit(
            f"Background task #{t.id} ({t.name}) {t.status}" + (f": {t.summary()}" if t.summary() else "") + (f" - {t.error}" if t.error else ""))
        if cfg.scheduler_enabled:
            ctx.scheduler = Scheduler(ctx.memory, lambda action: _run_action(ctx, action), notify=ctx.emit)
        return ctx

    def emit(self, text: str) -> None:
        """Show a background/scheduled event to the user."""
        try:
            self.notify(text)
        except Exception:
            pass


def _run_action(ctx: Context, action: dict[str, Any]) -> str:
    if ctx.action_runner is None:
        raise RuntimeError("scheduled actions are not available (tools not loaded)")
    return ctx.action_runner(action)
