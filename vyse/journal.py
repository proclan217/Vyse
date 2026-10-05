"""Undo journal: every write operation is logged so it can be reversed. Never hard-deletes."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def unique_path(dst: Path) -> Path:
    """Return dst, or 'name (1).ext', 'name (2).ext'... if it exists. Collision-safe."""
    if not dst.exists():
        return dst
    stem, suffix, n = dst.stem, dst.suffix, 1
    while True:
        cand = dst.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists():
            return cand
        n += 1


@dataclass
class Entry:
    id: str
    batch: str
    op: str            # move | copy | mkdir | trash
    src: str
    dst: str
    ts: float
    undone: bool = False
    label: str = ""      # what caused it (tool name), shown in the history


class Journal:
    def __init__(self, path: Path, trash_dir: Path) -> None:
        self.path = path
        self.trash_dir = trash_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)

    # ---- storage ----
    def _load(self) -> list[Entry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Entry(**{k: v for k, v in d.items() if k in Entry.__dataclass_fields__}))
        return out

    def _save(self, entries: list[Entry]) -> None:
        self.path.write_text("".join(json.dumps(e.__dict__) + "\n" for e in entries), encoding="utf-8")

    def new_batch(self) -> str:
        return uuid.uuid4().hex[:8]

    label: str = ""      # set by the agent for the duration of a tool call so every entry says what did it

    def _append(self, batch: str, op: str, src: Path, dst: Path) -> Entry:
        e = Entry(uuid.uuid4().hex[:8], batch, op, str(src), str(dst), time.time(), label=self.label)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e.__dict__) + "\n")
        return e

    def _mkparents(self, folder: Path, batch: str) -> None:
        """Create missing folders, journaling each so undo can remove them again."""
        missing = [p for p in [folder, *folder.parents] if not p.exists()]
        folder.mkdir(parents=True, exist_ok=True)
        for p in reversed(missing):          # shallowest first; undo walks the batch in reverse
            self._append(batch, "mkdir", p, p)

    # ---- operations (all logged) ----
    def move(self, src: Path, dst: Path, batch: str | None = None) -> Path:
        batch = batch or self.new_batch()
        self._mkparents(dst.parent, batch)
        final = unique_path(dst)
        shutil.move(str(src), str(final))
        self._append(batch, "move", src, final)
        return final

    def copy(self, src: Path, dst: Path, batch: str | None = None) -> Path:
        batch = batch or self.new_batch()
        self._mkparents(dst.parent, batch)
        final = unique_path(dst)
        if src.is_dir():
            shutil.copytree(src, final)
        else:
            shutil.copy2(src, final)
        self._append(batch, "copy", src, final)
        return final

    def mkdir(self, path: Path, batch: str | None = None) -> bool:
        batch = batch or self.new_batch()
        if path.exists():
            return False
        # log each created ancestor so undo can remove them (deepest first)
        # logged shallowest-first because undo walks the batch in reverse
        missing = [path] + [p for p in path.parents if not p.exists()]
        path.mkdir(parents=True)
        for p in reversed(missing):
            self._append(batch, "mkdir", p, p)
        return True

    def trash(self, src: Path, batch: str | None = None) -> Path:
        """'Delete' = move into Vyse's trash dir (timestamped), never remove."""
        batch = batch or self.new_batch()
        dst = unique_path(self.trash_dir / time.strftime("%Y%m%d-%H%M%S") / src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        self._append(batch, "trash", src, dst)
        return dst

    # ---- transaction log / undo ----
    def history(self, limit: int = 10, include_undone: bool = True) -> list[dict]:
        """Recent transactions (one per batch), newest first: what was done and whether it can still be undone."""
        batches: dict[str, dict] = {}
        for e in self._load():
            b = batches.setdefault(e.batch, {"batch": e.batch, "label": e.label, "ts": e.ts, "ops": [], "undone": True})
            b["ops"].append({"op": e.op, "src": e.src, "dst": e.dst, "undone": e.undone})
            b["undone"] = b["undone"] and e.undone
            b["label"] = b["label"] or e.label
        rows = sorted(batches.values(), key=lambda b: -b["ts"])
        if not include_undone:
            rows = [r for r in rows if not r["undone"]]
        for r in rows:
            moves = [o for o in r["ops"] if o["op"] != "mkdir"]
            r["summary"] = describe_ops(r["ops"])
            r["files"] = len(moves)
        return rows[:limit]

    def last_batch(self) -> str | None:
        for e in reversed(self._load()):
            if not e.undone:
                return e.batch
        return None

    def undo_last(self) -> tuple[list[str], list[str]]:
        """Reverse the most recent batch. Returns (reverted descriptions, problems)."""
        return self.undo(steps=1)

    def undo(self, steps: int = 1, batch: str | None = None) -> tuple[list[str], list[str]]:
        """Reverse the last `steps` batches, or one specific batch by id. Returns (reverted, problems)."""
        entries = self._load()
        if batch:
            targets = [batch] if any(e.batch == batch and not e.undone for e in entries) else []
            if not targets:
                known = any(e.batch == batch for e in entries)
                return [], [f"Transaction {batch} was already undone." if known else f"No transaction with id {batch}."]
        else:
            order: list[str] = []
            for e in reversed(entries):
                if not e.undone and e.batch not in order:
                    order.append(e.batch)
            targets = order[:max(1, steps)]
        if not targets:
            return [], ["Nothing to undo."]
        done: list[str] = []
        problems: list[str] = []
        for b in targets:
            self._undo_batch(entries, b, done, problems)
        self._save(entries)
        return done, problems

    def _undo_batch(self, entries: list[Entry], batch: str, done: list[str], problems: list[str]) -> None:
        for e in reversed([x for x in entries if x.batch == batch and not x.undone]):
            src, dst = Path(e.src), Path(e.dst)
            try:
                if e.op in ("move", "trash"):
                    if not dst.exists():
                        raise FileNotFoundError(f"{dst} no longer exists")
                    back = unique_path(src) if src.exists() else src
                    back.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dst), str(back))
                    done.append(f"{dst.name} -> {back}")
                elif e.op == "copy":
                    if dst.exists():
                        t = unique_path(self.trash_dir / "undone-copies" / dst.name)
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(dst), str(t))
                    done.append(f"removed copy {dst.name}")
                elif e.op == "mkdir":
                    if dst.exists() and not any(dst.iterdir()):
                        dst.rmdir()
                        done.append(f"removed empty folder {dst.name}")
                    elif dst.exists():
                        problems.append(f"kept non-empty folder {dst}")
                e.undone = True
            except Exception as ex:  # keep going; report
                problems.append(f"{e.op} {e.src}: {ex}")
                e.undone = True  # don't retry forever


def describe_ops(ops: list[dict]) -> str:
    """'moved 12 files', 'trashed report.pdf', 'copied 2 files' for a transaction's operations."""
    real = [o for o in ops if o["op"] != "mkdir"]
    if not real:
        return f"created {len(ops)} folder(s)"
    verbs = {"move": "moved", "trash": "trashed", "copy": "copied"}
    first = real[0]["op"]
    verb = verbs.get(first, first)
    if len(real) == 1:
        return f"{verb} {Path(real[0]['src']).name}"
    return f"{verb} {len(real)} items"
