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
                out.append(Entry(**json.loads(line)))
        return out

    def _save(self, entries: list[Entry]) -> None:
        self.path.write_text("".join(json.dumps(e.__dict__) + "\n" for e in entries), encoding="utf-8")

    def new_batch(self) -> str:
        return uuid.uuid4().hex[:8]

    def _append(self, batch: str, op: str, src: Path, dst: Path) -> Entry:
        e = Entry(uuid.uuid4().hex[:8], batch, op, str(src), str(dst), time.time())
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

    # ---- undo ----
    def last_batch(self) -> str | None:
        for e in reversed(self._load()):
            if not e.undone:
                return e.batch
        return None

    def undo_last(self) -> tuple[list[str], list[str]]:
        """Reverse the most recent batch. Returns (reverted descriptions, problems)."""
        entries = self._load()
        batch = next((e.batch for e in reversed(entries) if not e.undone), None)
        if batch is None:
            return [], ["Nothing to undo."]
        done: list[str] = []
        problems: list[str] = []
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
        self._save(entries)
        return done, problems
