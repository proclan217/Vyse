"""Folder organizing: plan (pure dry-run) -> apply (policy-gated, journaled, verified) -> undo."""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from ..context import Context
from ..organize_rules import build_rule_plan, rules_for
from ..policy import ALLOW, CONFIRM, Decision
from .files import resolve_path
from .registry import Registry, ToolError

CATEGORIES: dict[str, set[str]] = {
    "Images": {"jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "heic", "tiff", "ico", "raw"},
    "Videos": {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv"},
    "Audio": {"mp3", "wav", "flac", "aac", "ogg", "m4a", "wma"},
    "Documents": {"pdf", "doc", "docx", "txt", "rtf", "odt", "md", "epub", "tex"},
    "Spreadsheets": {"xls", "xlsx", "csv", "ods", "tsv"},
    "Presentations": {"ppt", "pptx", "odp", "key"},
    "Archives": {"zip", "rar", "7z", "tar", "gz", "bz2", "xz"},
    "Installers": {"exe", "msi", "msix", "appx", "dmg", "iso"},
    "Code": {"py", "js", "ts", "html", "css", "json", "cpp", "c", "h", "java", "rs", "go", "ino", "ipynb", "toml", "yaml", "yml"},
    "3D & CAD": {"stl", "step", "stp", "obj", "3mf", "f3d", "dxf", "dwg", "gcode", "fbx", "blend", "sldprt"},
}
EXT_TO_CAT = {e: c for c, exts in CATEGORIES.items() for e in exts}
SKIP_SUFFIXES = {".crdownload", ".part", ".tmp", ".partial", ".download", ".lnk", ".ini"}
Strategy = Literal["by_rules", "by_type", "by_date", "by_name"]


def category_for(p: Path, strategy: str) -> str:
    if strategy == "by_type":
        return EXT_TO_CAT.get(p.suffix.lower().lstrip("."), "Other")
    if strategy == "by_date":
        return time.strftime("%Y-%m", time.localtime(p.stat().st_mtime))
    if strategy == "by_name":
        c = p.name[:1].upper()
        return c if c.isalpha() else "0-9 & symbols"
    raise ToolError(f"Unknown strategy '{strategy}'. Use by_rules, by_type, by_date or by_name.")


def build_plan(folder: Path, strategy: str, rules: list | None = None) -> list[dict[str, str]]:
    """Pure function: compute proposed moves for the files directly inside `folder`. Touches nothing."""
    if strategy == "by_rules":      # deterministic rules (config [[organize.rules]] or the built-in defaults)
        return build_rule_plan(folder, rules_for(rules), catch_all="Other")
    moves = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.name.startswith(".") or p.name.startswith("~$") or p.suffix.lower() in SKIP_SUFFIXES:
            continue
        cat = category_for(p, strategy)
        moves.append({"src": str(p), "dst": str(folder / cat / p.name), "category": cat})
    return moves


def _summary(moves: list[dict[str, str]], limit: int = 12) -> str:
    counts = Counter(m["category"] for m in moves)
    lines = [f"{len(moves)} file(s) into {len(counts)} folder(s):"]
    lines += [f"  {c}: {n}" for c, n in counts.most_common()]
    lines.append("Examples:")
    lines += [f"  {Path(m['src']).name} -> {m['category']}/" for m in moves[:limit]]
    return "\n".join(lines)


def register(reg: Registry, ctx: Context) -> None:
    pol = ctx.policy
    plans_dir = ctx.cfg.data_dir / "plans"

    def save_plan(plan: dict[str, Any]) -> None:
        plans_dir.mkdir(parents=True, exist_ok=True)
        (plans_dir / f"{plan['id']}.json").write_text(json.dumps(plan), encoding="utf-8")

    def load_plan(plan_id: str) -> dict[str, Any]:
        if not plan_id.strip():     # no id given: use the most recent pending plan
            plans = sorted(plans_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not plans:
                raise ToolError("There is no pending plan. Run plan_organize first.")
            plan_id = plans[-1].stem
        f = plans_dir / f"{Path(plan_id).name}.json"   # Path().name blocks traversal
        if not f.is_file():
            raise ToolError(f"No plan with id '{plan_id}'. Run plan_organize first.")
        return json.loads(f.read_text(encoding="utf-8"))

    @reg.tool(risk="safe", group="organize", keywords=("organize", "organise", "clean", "tidy", "sort", "folder", "downloads", "desktop", "mess"),
              assess=lambda a: pol.check_path(resolve_path(a.get("folder", "")), write=False))
    def plan_organize(folder: str, strategy: Strategy = "by_rules") -> dict:
        """Create a DRY-RUN plan for organizing the files in a folder into sub-folders. Nothing is changed. Show the plan to the user, then use apply_plan.

        Args:
            folder: Folder to organize, e.g. 'Downloads' or a full path.
            strategy: by_rules (default: deterministic rules from config, e.g. screenshots, invoices, installers), by_type (Images, Documents...), by_date (YYYY-MM) or by_name (first letter).
        """
        f = resolve_path(folder)
        if not f.is_dir():
            raise ToolError(f"Not a folder: {f}")
        moves = build_plan(f, strategy, ctx.cfg.organize_rules)
        if not moves:
            return {"plan_id": None, "moves": 0, "display": f"Nothing to organize in {f.name}"}
        plan = {"id": uuid.uuid4().hex[:8], "folder": str(f), "strategy": strategy, "moves": moves, "created": time.time()}
        save_plan(plan)
        return {"plan_id": plan["id"], "folder": str(f), "strategy": strategy, "moves": len(moves),
                "preview": _summary(moves), "dry_run": True,
                "display": f"Plan {plan['id']}: {len(moves)} file(s) would be moved (nothing changed yet)"}

    def apply_assess(args: dict[str, Any]) -> Decision:
        plan = load_plan(str(args.get("plan_id", "")))
        folder = Path(plan["folder"])
        d = pol.check_path(folder, write=True)
        n = len(plan["moves"])
        if d.action == ALLOW and n > ctx.cfg.organize_confirm_threshold:
            d = Decision(CONFIRM, f"This moves {n} files (limit for automatic runs is {ctx.cfg.organize_confirm_threshold}).")
        if d.action != ALLOW:
            d.preview = f"Organize {folder} ({plan['strategy']})\n" + _summary(plan["moves"])
        return d

    @reg.tool(risk="write", group="organize", keywords=("organize", "organise", "apply", "plan", "sort", "tidy", "clean"),
              assess=apply_assess)
    def apply_plan(plan_id: str = "") -> dict:
        """Execute a plan created by plan_organize. Moves are journaled (undo_last reverses them) and verified afterwards.

        Args:
            plan_id: The plan_id returned by plan_organize (leave empty to use the latest plan).
        """
        plan = load_plan(plan_id)
        plan_id = plan.get("id") or plan_id
        batch = ctx.journal.new_batch()
        moved, failed, missing = [], [], []
        for m in plan["moves"]:
            src, dst = Path(m["src"]), Path(m["dst"])
            if not src.exists():
                missing.append(src.name)
                continue
            try:
                moved.append((src, ctx.journal.move(src, dst, batch)))
            except Exception as e:
                failed.append(f"{src.name}: {e}")
        # verify filesystem state: every destination exists, no source remains
        bad = [s.name for s, d in moved if not d.exists() or s.exists()]
        verified = not bad and not failed
        (plans_dir / f"{plan_id}.json").unlink(missing_ok=True)  # a plan is single-use
        return {"moved": len(moved), "failed": failed, "skipped_missing": missing, "verification_problems": bad,
                "verified": verified, "undo": "Use undo_last to revert.",
                "display": f"Moved {len(moved)} file(s)" + (f", {len(failed)} failed" if failed else "") + (" — verified" if verified else " — NOT fully verified")}

    def _undone(done: list[str], problems: list[str]) -> dict:
        return {"reverted": len(done), "details": done[:20], "problems": problems,
                "verified": not problems and bool(done),
                "display": f"Reverted {len(done)} item(s)" + (f"; {len(problems)} problem(s)" if problems else "")}

    @reg.tool(risk="write", group="organize", keywords=("undo", "revert", "rollback", "restore", "back", "oops"))
    def undo_last() -> dict:
        """Undo the most recent file operation batch (a move, organize run, copy, folder creation or trash)."""
        return _undone(*ctx.journal.undo_last())

    @reg.tool(risk="write", group="organize", keywords=("undo", "revert", "rollback", "restore", "back", "oops", "steps", "transaction"))
    def undo(steps: int = 1, transaction: str = "") -> dict:
        """Undo several recent file operations, or one specific transaction from undo_history.

        Args:
            steps: How many of the most recent transactions to undo (default 1).
            transaction: A transaction id from undo_history (overrides steps).
        """
        return _undone(*ctx.journal.undo(steps=max(1, min(int(steps), 20)), batch=transaction.strip() or None))

    @reg.tool(risk="safe", group="organize", final=True, keywords=("undo", "history", "log", "transactions", "changes", "did", "recent"))
    def undo_history(limit: int = 10) -> dict:
        """List recent reversible file operations (the transaction log) and whether each can still be undone.

        Args:
            limit: How many transactions to show (default 10).
        """
        rows = ctx.journal.history(limit=max(1, min(int(limit), 50)))
        out = [{"transaction": r["batch"], "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
                "what": r["summary"], "files": r["files"], "undone": r["undone"]} for r in rows]
        return {"transactions": out, "display": f"{len(out)} transaction(s)" if out else "No file operations recorded yet"}
