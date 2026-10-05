"""Fast indexed file search: one SQLite table of every file under the configured roots.

The model never walks folders itself. `search()` answers from the index in milliseconds (all words must appear in the
name, wildcards work, typos fall back to a fuzzy match), and results are verified against the disk before they are
returned so a stale entry can never point at a file that is gone. The first build runs in the background; callers
fall back to a live scan until it is ready.
"""
from __future__ import annotations

import fnmatch
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "$RECYCLE.BIN", "System Volume Information",
             "site-packages", ".gradle", ".cache", ".npm", "AppData"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
    path TEXT PRIMARY KEY, root TEXT NOT NULL, dir TEXT NOT NULL, name TEXT NOT NULL, name_lc TEXT NOT NULL,
    ext TEXT NOT NULL, size INTEGER NOT NULL, mtime REAL NOT NULL);
CREATE INDEX IF NOT EXISTS files_name ON files(name_lc);
CREATE INDEX IF NOT EXISTS files_ext ON files(ext);
CREATE INDEX IF NOT EXISTS files_mtime ON files(mtime);
CREATE TABLE IF NOT EXISTS roots(root TEXT PRIMARY KEY, built_at REAL NOT NULL, count INTEGER NOT NULL);
"""


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class FileIndex:
    def __init__(self, path: Path | str, roots: list[Path], pruner_factory: Callable[[Path], Callable[[str, str], bool]] | None = None,
                 is_protected: Callable[[Path], bool] | None = None, max_files: int = 400_000, stale_minutes: float = 30.0) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self.roots = [Path(r) for r in roots]
        self._pruner_factory = pruner_factory
        self._is_protected = is_protected or (lambda p: False)
        self.max_files = max_files
        self.stale_seconds = stale_minutes * 60
        self._building: threading.Thread | None = None
        self.last_error = ""

    # ---- state ----
    def covers(self, folder: Path) -> bool:
        f = Path(os.path.normcase(str(folder.resolve(strict=False))))
        return any(f == r or r in f.parents for r in (Path(os.path.normcase(str(x.resolve(strict=False)))) for x in self.roots))

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def built_at(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT MIN(built_at) FROM roots").fetchone()
        return row[0] or 0.0

    @property
    def ready(self) -> bool:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM roots").fetchone()[0]
        return n >= len([r for r in self.roots if r.exists()]) and n > 0

    @property
    def building(self) -> bool:
        return bool(self._building and self._building.is_alive())

    def is_stale(self) -> bool:
        return (not self.ready) or (time.time() - self.built_at() > self.stale_seconds)

    # ---- building ----
    def _walk(self, root: Path, deadline: float | None):
        pruned = self._pruner_factory(root) if self._pruner_factory else (lambda parent, name: False)
        stack = [str(root)]
        while stack:
            d = stack.pop()
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name not in SKIP_DIRS and not pruned(d, e.name):
                                stack.append(e.path)
                            continue
                        st = e.stat()
                    except OSError:
                        continue
                    yield e.path, e.name, st.st_size, st.st_mtime
            if deadline and time.time() > deadline:
                return

    def build(self, roots: list[Path] | None = None, time_limit: float | None = None) -> int:
        """(Re)index the roots. Returns the number of files indexed."""
        total = 0
        deadline = time.time() + time_limit if time_limit else None
        for root in roots or self.roots:
            root = Path(root)
            if not root.exists() or self._is_protected(root):
                continue
            rows: list[tuple[Any, ...]] = []
            root_s = str(root)
            for path, name, size, mtime in self._walk(root, deadline):
                rows.append((path, root_s, str(Path(path).parent), name, name.lower(),
                             Path(name).suffix.lower().lstrip("."), size, mtime))
                if total + len(rows) >= self.max_files:
                    break
            with self._lock:
                self._db.execute("DELETE FROM files WHERE root=?", (root_s,))
                self._db.executemany("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)", rows)
                self._db.execute("INSERT OR REPLACE INTO roots VALUES (?,?,?)", (root_s, time.time(), len(rows)))
                self._db.commit()
            total += len(rows)
        return total

    def build_async(self) -> None:
        if self.building:
            return

        def run() -> None:
            try:
                self.build()
            except Exception as e:      # an index failure must never take the app down
                self.last_error = f"{type(e).__name__}: {e}"

        self._building = threading.Thread(target=run, daemon=True, name="vyse-index")
        self._building.start()

    def wait(self, timeout: float = 30.0) -> bool:
        if self._building:
            self._building.join(timeout)
        return self.ready

    # ---- incremental updates (called after Vyse's own file operations) ----
    def upsert(self, path: str | Path) -> None:
        p = Path(path)
        try:
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file():
                        self.upsert(sub)
                return
            st = p.stat()
        except OSError:
            return
        root = next((str(r) for r in self.roots if r in p.parents), None)
        if root is None:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?)",
                             (str(p), root, str(p.parent), p.name, p.name.lower(), p.suffix.lower().lstrip("."),
                              st.st_size, st.st_mtime))
            self._db.commit()

    def remove(self, path: str | Path) -> None:
        p = str(path)
        with self._lock:
            self._db.execute("DELETE FROM files WHERE path=? OR path LIKE ? ESCAPE '\\'", (p, _like_escape(p + os.sep) + "%"))
            self._db.commit()

    # ---- searching ----
    def search(self, query: str = "", *, ext: str = "", folder: str | Path | None = None, modified_within_days: float = 0,
               limit: int = 40, fuzzy: bool = True) -> dict[str, Any]:
        """Returns {'rows': [...], 'total': n, 'fuzzy': bool, 'matched_as': str|None}. Rows are newest first."""
        where, params = ["1=1"], []
        if ext:
            where.append("ext=?")
            params.append(ext.lower().lstrip("."))
        if folder:
            f = str(Path(folder))
            where.append("(dir=? OR dir LIKE ? ESCAPE '\\')")
            params += [f, _like_escape(f + os.sep) + "%"]
        if modified_within_days and modified_within_days > 0:
            where.append("mtime>=?")
            params.append(time.time() - modified_within_days * 86400)
        q = (query or "").strip().lower()
        tokens = [t for t in re.split(r"\s+", q) if t]
        base_where, base_params = list(where), list(params)
        for t in tokens:
            if any(c in t for c in "*?"):
                where.append("name_lc LIKE ? ESCAPE '\\'")
                params.append(re.sub(r"([%_\\])", r"\\\1", t).replace("*", "%").replace("?", "_"))
            else:
                where.append("name_lc LIKE ? ESCAPE '\\'")
                params.append("%" + _like_escape(t) + "%")
        sql_where = " AND ".join(where)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM files WHERE {sql_where}", params).fetchone()[0]
            rows = self._db.execute(
                f"SELECT path,name,size,mtime,ext FROM files WHERE {sql_where} ORDER BY mtime DESC LIMIT ?",
                (*params, limit * 2)).fetchall()
        used_fuzzy, matched_as = False, None
        if not rows and tokens and fuzzy and len(q.replace(" ", "")) >= 4:
            rows, matched_as = self._fuzzy_rows(q, " AND ".join(base_where), base_params, limit)
            used_fuzzy, total = bool(rows), len(rows)
        out = []
        stale = []
        for path, name, size, mtime, e in rows:
            if os.path.exists(path):
                out.append({"path": path, "name": name, "size": size, "mtime": mtime, "ext": e})
            else:
                stale.append(path)
        for p in stale:
            self.remove(p)
        total = max(0, total - len(stale))
        return {"rows": out[:limit], "total": total, "fuzzy": used_fuzzy, "matched_as": matched_as}

    def _fuzzy_rows(self, q: str, where: str, params: list[Any], limit: int):
        from rapidfuzz import fuzz, process, utils
        with self._lock:
            names = self._db.execute(f"SELECT DISTINCT name_lc FROM files WHERE {where} LIMIT 150000", params).fetchall()
        by_stem: dict[str, list[str]] = {}
        for (n,) in names:                         # compare against the name without its extension ("report.pdf" -> "report")
            by_stem.setdefault(os.path.splitext(n)[0] or n, []).append(n)
        hits = process.extract(q, list(by_stem), scorer=fuzz.WRatio, processor=utils.default_process, score_cutoff=82, limit=5)
        if not hits:
            return [], None
        best = [n for h in hits for n in by_stem[h[0]]]
        marks = ",".join("?" * len(best))
        with self._lock:
            rows = self._db.execute(
                f"SELECT path,name,size,mtime,ext FROM files WHERE name_lc IN ({marks}) ORDER BY mtime DESC LIMIT ?",
                (*best, limit * 2)).fetchall()
        return rows, best[0]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            roots = self._db.execute("SELECT root, built_at, count FROM roots").fetchall()
        return {"files": self.count(), "ready": self.ready, "building": self.building,
                "roots": [{"root": r, "files": c, "age_min": round((time.time() - b) / 60, 1)} for r, b, c in roots]}
