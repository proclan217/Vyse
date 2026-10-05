"""Policy: the single authority on what Vyse may do. The model proposes; the policy decides."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .tools.registry import Tool

ALLOW, CONFIRM, BLOCK = "allow", "confirm", "block"
_ORDER = {ALLOW: 0, CONFIRM: 1, BLOCK: 2}


@dataclass
class Decision:
    action: str = ALLOW
    reason: str = ""
    preview: str = ""
    destructive: bool = False       # destructive confirmations are never skipped by auto-approve

    def stricter(self, other: "Decision | None") -> "Decision":
        if other is None:
            return self
        if _ORDER[other.action] > _ORDER[self.action]:
            other.destructive = other.destructive or self.destructive
            return other
        self.destructive = self.destructive or other.destructive
        return self


def _norm(p: str | Path) -> Path:
    # resolve() collapses .. and follows symlinks/junctions, so escapes are caught.
    return Path(p).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# Command safety: blocked outright vs. confirmation-only (run_command always confirms anyway).
_BLOCKED_COMMANDS = [
    r"\bformat\s+[a-z]:", r"\bdiskpart\b", r"\bbcdedit\b", r"\bcipher\s+/w", r"\bmkfs\b",
    r"\brm\s+(-\w*\s+)*-?\w*r\w*f?\w*\s+[/\\]", r"\brd\s+/s\b.*\b[a-z]:\\\s*$", r"\bdel\s+/[fsq].*\b[a-z]:\\(windows|users)?\s*$",
    r"remove-item\b.*-recurse.*\b[a-z]:\\(windows|program files|users)?\s*$",
    r"\breg\s+delete\s+hk(lm|cr)", r"\bshutdown\b", r"\btakeown\b.*\\windows", r"\bvssadmin\s+delete",
    r"\bnet\s+user\b.*\s/(add|delete)", r"\bwmic\b.*\bdelete\b", r"clear-disk", r"remove-partition",
    r"\bformat-volume\b", r"set-executionpolicy\s+unrestricted", r"invoke-expression|\biex\b.*(http|downloadstring)",
]
_DANGEROUS_HINTS = [r"\bdel\b", r"\brmdir\b", r"\brd\b", r"remove-item", r"\brm\b", r"\bmove\b", r"\breg\b",
                    r"\bpip\b.*install", r"\bnpm\b.*install", r"\bcurl\b|\bwget\b|invoke-webrequest", r"\bsc\b\s+(delete|stop)",
                    r"\btaskkill\b", r"\bnet\b"]


class Policy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.allowed = [_norm(p) for p in cfg.allowed_roots]
        self.read_roots = [_norm(p) for p in cfg.read_roots]
        self.protected = [_norm(p) for p in cfg.protected_paths]
        # Vyse's own data dir (db, tokens) is off limits to file tools except the trash dir.
        self.data_dir = _norm(cfg.data_dir)
        self.trash_dir = _norm(cfg.trash_dir)

    # ---- paths ----
    def is_protected(self, path: str | Path) -> bool:
        p = _norm(path)
        if _is_within(p, self.trash_dir):
            return False
        if _is_within(p, self.data_dir):
            return True
        return any(_is_within(p, r) for r in self.protected)

    def walk_pruner(self, root: str | Path):
        """Fast 'is this sub-directory protected?' check for a tree walk below `root`.

        Calling is_protected() on every directory costs milliseconds each (pathlib + resolve).
        Protected locations are fixed, so only those lying inside `root` can ever be hit by name;
        junctions/symlinks (which could lead into one) still get the full resolving check.
        """
        base = _norm(root)
        hits = {os.path.normcase(str(p)) for p in [*self.protected, self.data_dir] if _is_within(p, base)}

        def pruned(parent: str, name: str) -> bool:
            full = os.path.join(parent, name)
            if os.path.islink(full) or os.path.isjunction(full):
                return self.is_protected(full)
            return bool(hits) and os.path.normcase(full) in hits

        return pruned

    def in_allowed(self, path: str | Path) -> bool:
        p = _norm(path)
        return any(_is_within(p, r) for r in self.allowed)

    def check_path(self, path: str | Path, *, write: bool) -> Decision:
        p = _norm(path)
        if self.is_protected(p):
            return Decision(BLOCK, f"'{p}' is a protected location.")
        if write:
            if not self.in_allowed(p):
                return Decision(CONFIRM, f"'{p}' is outside Vyse's allowed folders.")
            return Decision(ALLOW)
        if self.read_roots and not any(_is_within(p, r) for r in self.read_roots) and not self.in_allowed(p):
            return Decision(CONFIRM, f"'{p}' is outside the folders Vyse normally reads.")
        return Decision(ALLOW)

    def check_paths(self, paths: list[str | Path], *, write: bool) -> Decision:
        d = Decision()
        for p in paths:
            d = d.stricter(self.check_path(p, write=write))
        return d

    # ---- commands ----
    def check_command(self, command: str) -> Decision:
        low = command.lower()
        for pat in _BLOCKED_COMMANDS:
            if re.search(pat, low):
                return Decision(BLOCK, "This command looks destructive to the system and is blocked.")
        warn = [h for h in _DANGEROUS_HINTS if re.search(h, low)]
        reason = "Shell commands always need your approval."
        if warn:
            reason += " This one may modify files or the system."
        return Decision(CONFIRM, reason, preview=f"Run command:\n  {command}")

    # ---- tools ----
    def permission_of(self, tool: Tool) -> str:
        """The tool's permission level after config overrides (config may only tighten, never relax)."""
        rank = {"safe": 0, "confirmation": 1, "blocked": 2}
        level = tool.effective_permission
        override = self.cfg.permissions.get(tool.name)
        if override and rank[override] > rank[level]:
            level = override
        if tool.destructive and level == "safe":
            level = "confirmation"
        return level

    def decide(self, tool: Tool, args: dict[str, Any]) -> Decision:
        level = self.permission_of(tool)
        if level == "blocked":
            return Decision(BLOCK, f"'{tool.name}' is disabled (permission: blocked).")
        if level == "confirmation":
            ask = self.cfg.confirm_risky or tool.risk != "risky" or tool.destructive
            base = Decision(CONFIRM if ask else ALLOW,
                            f"'{tool.name}' is a destructive action." if tool.destructive else f"'{tool.name}' needs your approval.",
                            destructive=tool.destructive)
        else:
            base = Decision(ALLOW)
        if tool.assess is not None:
            try:
                base = base.stricter(tool.assess(args))
            except Exception as e:  # a broken assessor must fail closed
                return Decision(CONFIRM, f"Could not assess '{tool.name}' safely ({e}).")
        if base.action == CONFIRM and not base.preview:
            base.preview = f"{tool.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
        return base
