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
        self.db.commit()

    # ---- facts ----
    def remember(self, key: str, text: str, tags: str = "") -> Fact:
        key = key.strip().lower()
        now = time.time()
        cur = self.db.execute("SELECT id FROM facts WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            self.db.execute("UPDATE facts SET text=?, tags=?, created_at=? WHERE id=?", (text, tags, now, row["id"]))
        else:
            self.db.execute("INSERT INTO facts(key,text,tags,created_at) VALUES (?,?,?,?)", (key, text, tags, now))
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
