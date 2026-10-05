"""Persistent memory in SQLite: messages, facts (FTS5-searchable) and rolling summaries."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Fact:
    id: int
    key: str
    text: str
    tags: str
    created_at: float
    kind: str = "fact"           # fact | preference | context
    importance: int = 1          # 1 normal, 2 important, 3 pinned-worthy
    uses: int = 0                # how often retrieval surfaced it (a cheap usefulness signal)
    last_used: float = 0.0


_STOP = set("a an the is are was were be to of in on at for and or my me i you your what which who do does did "
            "it this that with about from can could would should please tell show".split())


def _fts_query(text: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


class Memory:
    def __init__(self, db_path: Path | str) -> None:
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS facts(
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS summaries(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
            upto_message_id INTEGER NOT NULL, text TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS routines(
            name TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '',
            steps TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS task_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, request TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT '',
            tools TEXT NOT NULL DEFAULT '', ok INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS task_fts USING fts5(
            request, outcome, tools, content='task_history', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS task_ai AFTER INSERT ON task_history BEGIN
            INSERT INTO task_fts(rowid,request,outcome,tools) VALUES (new.id,new.request,new.outcome,new.tools); END;
        CREATE TRIGGER IF NOT EXISTS task_ad AFTER DELETE ON task_history BEGIN
            INSERT INTO task_fts(task_fts,rowid,request,outcome,tools) VALUES('delete',old.id,old.request,old.outcome,old.tools); END;
        CREATE TABLE IF NOT EXISTS schedules(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, kind TEXT NOT NULL, spec TEXT NOT NULL,
            action TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL,
            last_run REAL, runs INTEGER NOT NULL DEFAULT 0, last_result TEXT NOT NULL DEFAULT '');
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            key, text, tags, content='facts', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid,key,text,tags) VALUES (new.id,new.key,new.text,new.tags); END;
        CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts,rowid,key,text,tags) VALUES('delete',old.id,old.key,old.text,old.tags); END;
        CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts,rowid,key,text,tags) VALUES('delete',old.id,old.key,old.text,old.tags);
            INSERT INTO facts_fts(rowid,key,text,tags) VALUES (new.id,new.key,new.text,new.tags); END;
        """)
        self._migrate()
        self.db.commit()

    def _migrate(self) -> None:
        """Databases created before kind/importance/uses existed are upgraded in place."""
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(facts)")}
        for col, ddl in (("kind", "TEXT NOT NULL DEFAULT 'fact'"), ("importance", "INTEGER NOT NULL DEFAULT 1"),
                         ("uses", "INTEGER NOT NULL DEFAULT 0"), ("last_used", "REAL NOT NULL DEFAULT 0")):
            if col not in have:
                self.db.execute(f"ALTER TABLE facts ADD COLUMN {col} {ddl}")

    # ---- facts ----
    def remember(self, key: str, text: str, tags: str = "", kind: str = "fact", importance: int = 1) -> Fact:
        key = key.strip().lower()
        kind = kind if kind in ("fact", "preference", "context") else "fact"
        importance = min(3, max(1, int(importance)))
        now = time.time()
        cur = self.db.execute("SELECT id FROM facts WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            self.db.execute("UPDATE facts SET text=?, tags=?, created_at=?, kind=?, importance=? WHERE id=?",
                            (text, tags, now, kind, importance, row["id"]))
        else:
            self.db.execute("INSERT INTO facts(key,text,tags,created_at,kind,importance) VALUES (?,?,?,?,?,?)",
                            (key, text, tags, now, kind, importance))
        self.db.commit()
        return self.get_fact(key)  # type: ignore[return-value]

    def get_fact(self, key: str) -> Fact | None:
        r = self.db.execute("SELECT * FROM facts WHERE key=?", (key.strip().lower(),)).fetchone()
        return Fact(**dict(r)) if r else None

    def recall(self, query: str, limit: int = 5) -> list[Fact]:
        q = _fts_query(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT f.* FROM facts_fts JOIN facts f ON f.id=facts_fts.rowid "
            "WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT ?", (q, limit)).fetchall()
        return [Fact(**dict(r)) for r in rows]

    # ---- task history: what the user asked before and how it went ----
    def record_task(self, request: str, outcome: str, tools: list[str], ok: bool = True) -> None:
        self.db.execute("INSERT INTO task_history(request,outcome,tools,ok,created_at) VALUES (?,?,?,?,?)",
                        (request[:300], outcome[:300], " ".join(dict.fromkeys(tools)), int(ok), time.time()))
        self.db.execute("DELETE FROM task_history WHERE id <= (SELECT MAX(id) FROM task_history) - 500")   # bounded
        self.db.commit()

    def recent_tasks(self, limit: int = 10) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM task_history ORDER BY id DESC LIMIT ?", (limit,))]

    def retrieve(self, query: str, limit: int = 5, max_chars: int = 900) -> list[dict]:
        """Only what is relevant to `query`: ranked facts and past tasks, within a character budget.

        Score = text relevance (bm25 rank) + importance + a little recency and past usefulness. Typos fall back to
        fuzzy matching on keys/tags. Returns [{'source': 'fact'|'task', 'key', 'text', 'score'}]."""
        q = _fts_query(query)
        cands: dict[tuple[str, str], dict] = {}
        now = time.time()
        if q:
            rows = self.db.execute(
                "SELECT f.*, bm25(facts_fts) AS r FROM facts_fts JOIN facts f ON f.id=facts_fts.rowid "
                "WHERE facts_fts MATCH ? ORDER BY r LIMIT ?", (q, limit * 3)).fetchall()
            for n, r in enumerate(rows):
                age_days = (now - r["created_at"]) / 86400
                score = 10 - n * 0.7 + r["importance"] * 1.5 + min(r["uses"], 5) * 0.2 + (0.5 if age_days < 7 else 0)
                cands[("fact", r["key"])] = {"source": "fact", "key": r["key"], "text": r["text"], "score": score, "id": r["id"]}
            rows = self.db.execute(
                "SELECT t.*, bm25(task_fts) AS r FROM task_fts JOIN task_history t ON t.id=task_fts.rowid "
                "WHERE task_fts MATCH ? AND t.ok=1 ORDER BY r LIMIT ?", (q, 3)).fetchall()
            for n, r in enumerate(rows):
                cands[("task", str(r["id"]))] = {"source": "task", "key": str(r["id"]), "score": 5 - n,
                                                 "text": f"Earlier you asked: {r['request']} -> {r['outcome']}"[:240]}
        if not any(c["source"] == "fact" for c in cands.values()):
            from .fuzzy import best_match
            words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP and len(w) > 3]
            labels = {}                         # one entry per key/tag word, so 'pritner' can match 'printer'
            for f in self.all_facts():
                for tok in re.findall(r"[a-z0-9]+", f"{f.key.replace('_', ' ')} {f.tags.replace(',', ' ')}".lower()):
                    if len(tok) > 3:
                        labels.setdefault(tok, f)
            for w in words:
                for m in best_match(w, list(labels), threshold=82, limit=2):
                    f = labels[m.value]
                    cands[("fact", f.key)] = {"source": "fact", "key": f.key, "text": f.text, "id": f.id,
                                              "score": 4 + f.importance}
        ranked = sorted(cands.values(), key=lambda c: -c["score"])
        out, used = [], 0
        for c in ranked[:limit]:
            if used + len(c["text"]) > max_chars and out:
                break
            out.append(c)
            used += len(c["text"])
        for c in out:
            if c["source"] == "fact":
                self.db.execute("UPDATE facts SET uses=uses+1, last_used=? WHERE id=?", (now, c["id"]))
        if out:
            self.db.commit()
        return out

    # ---- schedules (persistent definitions; the scheduler rebuilds its jobs from these) ----
    def add_schedule(self, name: str, kind: str, spec: dict, action: dict) -> int:
        cur = self.db.execute("INSERT INTO schedules(name,kind,spec,action,created_at) VALUES (?,?,?,?,?)",
                              (name, kind, json.dumps(spec), json.dumps(action), time.time()))
        self.db.commit()
        return cur.lastrowid or 0

    @staticmethod
    def _sched(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["spec"], d["action"], d["enabled"] = json.loads(d["spec"]), json.loads(d["action"]), bool(d["enabled"])
        return d

    def list_schedules(self, only_enabled: bool = False) -> list[dict]:
        sql = "SELECT * FROM schedules" + (" WHERE enabled=1" if only_enabled else "") + " ORDER BY id"
        return [self._sched(r) for r in self.db.execute(sql)]

    def get_schedule(self, sid: int) -> dict | None:
        r = self.db.execute("SELECT * FROM schedules WHERE id=?", (sid,)).fetchone()
        return self._sched(r) if r else None

    def update_schedule(self, sid: int, **fields) -> None:
        cols = [k for k in fields if k in {"enabled", "last_run", "runs", "last_result"}]
        if cols:
            self.db.execute(f"UPDATE schedules SET {', '.join(f'{c}=?' for c in cols)} WHERE id=?",
                            (*[fields[c] for c in cols], sid))
            self.db.commit()

    def delete_schedule(self, sid: int) -> bool:
        cur = self.db.execute("DELETE FROM schedules WHERE id=?", (sid,))
        self.db.commit()
        return cur.rowcount > 0

    def forget(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM facts WHERE key=?", (key.strip().lower(),))
        self.db.commit()
        return cur.rowcount > 0

    def all_facts(self) -> list[Fact]:
        return [Fact(**dict(r)) for r in self.db.execute("SELECT * FROM facts ORDER BY created_at DESC")]

    # ---- routines: named sequences of tool calls the user taught Vyse ----
    def save_routine(self, name: str, description: str, steps: list[dict]) -> None:
        self.db.execute("INSERT OR REPLACE INTO routines(name,description,steps,created_at) VALUES (?,?,?,?)",
                        (name.strip().lower(), description.strip(), json.dumps(steps), time.time()))
        self.db.commit()

    def get_routine(self, name: str) -> dict | None:
        r = self.db.execute("SELECT * FROM routines WHERE name=?", (name.strip().lower(),)).fetchone()
        return {"name": r["name"], "description": r["description"], "steps": json.loads(r["steps"])} if r else None

    def list_routines(self) -> list[dict]:
        return [{"name": r["name"], "description": r["description"], "steps": json.loads(r["steps"])}
                for r in self.db.execute("SELECT * FROM routines ORDER BY name")]

    def find_routines(self, query: str, limit: int = 3) -> list[dict]:
        """Routines whose name/description share a meaningful word with the query (tiny table: no index needed)."""
        words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP and len(w) > 2}
        hits = []
        for r in self.list_routines():
            hay = set(re.findall(r"[a-z0-9]+", f"{r['name']} {r['description']}".lower()))
            score = len(words & hay)
            if score:
                hits.append((score, r))
        return [r for _, r in sorted(hits, key=lambda t: -t[0])[:limit]]

    def delete_routine(self, name: str) -> bool:
        cur = self.db.execute("DELETE FROM routines WHERE name=?", (name.strip().lower(),))
        self.db.commit()
        return cur.rowcount > 0

    # ---- messages ----
    def add_message(self, session: str, role: str, content: str) -> int:
        cur = self.db.execute("INSERT INTO messages(session,role,content,created_at) VALUES (?,?,?,?)",
                              (session, role, content, time.time()))
        self.db.commit()
        return cur.lastrowid or 0

    def recent_messages(self, session: str, limit: int) -> list[dict[str, str]]:
        since = self.latest_summary(session)
        floor = since[0] if since else 0
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session=? AND id>? ORDER BY id DESC LIMIT ?",
            (session, floor, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def message_count(self, session: str) -> int:
        since = self.latest_summary(session)
        floor = since[0] if since else 0
        return self.db.execute("SELECT COUNT(*) c FROM messages WHERE session=? AND id>?",
                               (session, floor)).fetchone()["c"]

    def clear_session(self, session: str) -> None:
        self.db.execute("DELETE FROM messages WHERE session=?", (session,))
        self.db.execute("DELETE FROM summaries WHERE session=?", (session,))
        self.db.commit()

    # ---- summaries ----
    def latest_summary(self, session: str) -> tuple[int, str] | None:
        r = self.db.execute("SELECT upto_message_id, text FROM summaries WHERE session=? ORDER BY id DESC LIMIT 1",
                            (session,)).fetchone()
        return (r["upto_message_id"], r["text"]) if r else None

    def messages_to_summarize(self, session: str, keep_recent: int) -> tuple[int, list[dict[str, str]]]:
        """Messages older than the last `keep_recent` that haven't been summarized yet."""
        since = self.latest_summary(session)
        floor = since[0] if since else 0
        rows = self.db.execute("SELECT id, role, content FROM messages WHERE session=? AND id>? ORDER BY id",
                               (session, floor)).fetchall()
        old = rows[:-keep_recent] if keep_recent else rows
        if not old:
            return 0, []
        return old[-1]["id"], [{"role": r["role"], "content": r["content"]} for r in old]

    def save_summary(self, session: str, upto_message_id: int, text: str) -> None:
        self.db.execute("INSERT INTO summaries(session,upto_message_id,text,created_at) VALUES (?,?,?,?)",
                        (session, upto_message_id, text, time.time()))
        self.db.commit()
