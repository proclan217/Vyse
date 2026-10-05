"""Fuzzy entity resolution on top of RapidFuzz: app names, files, folders, commands, tools, routines.

One place decides what "close enough" means, so every caller gets the same, conservative behaviour:
- very short queries never fuzzy-match (``cal`` must not become ``Calculator`` by accident),
- near-ties are reported as *ambiguous* instead of guessed,
- callers decide what to do with a fuzzy hit (read-only tools may auto-correct, destructive ones only suggest).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from rapidfuzz import fuzz, process, utils

DEFAULT_THRESHOLD = 78
AMBIGUITY_GAP = 5          # top two scores closer than this (and not exact) => ambiguous
MIN_FUZZY_LEN = 3


@dataclass(frozen=True)
class Match:
    value: str             # the candidate as given
    score: float           # 0-100
    key: str | None = None  # the mapped payload when choices were a mapping


@dataclass
class Resolution:
    """status: exact | fuzzy | ambiguous | none"""
    status: str
    match: Match | None = None
    candidates: list[Match] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("exact", "fuzzy")


def _norm(s: str) -> str:
    return utils.default_process(s) or ""


def best_match(query: str, choices: Iterable[str] | Mapping[str, str], threshold: float = DEFAULT_THRESHOLD,
               limit: int = 5) -> list[Match]:
    """Top fuzzy matches, best first. `choices` may be a mapping label -> payload (payload goes in .key)."""
    q = (query or "").strip()
    if not q:
        return []
    mapping = dict(choices) if isinstance(choices, Mapping) else None
    labels = list(mapping) if mapping is not None else list(choices)
    if not labels:
        return []
    short = len(_norm(q)) < MIN_FUZZY_LEN
    out: list[Match] = []
    for label, score, _ in process.extract(q, labels, scorer=fuzz.WRatio, processor=utils.default_process,
                                           score_cutoff=threshold, limit=limit):
        n, c = _norm(q), _norm(label)
        if short and not (c == n or c.startswith(n)):
            continue
        out.append(Match(label, float(score), mapping[label] if mapping is not None else None))
    return out


def resolve(query: str, choices: Iterable[str] | Mapping[str, str], threshold: float = DEFAULT_THRESHOLD) -> Resolution:
    """Exact (case/punctuation-insensitive) match first, then a confident fuzzy one, else ambiguous/none."""
    mapping = dict(choices) if isinstance(choices, Mapping) else None
    labels = list(mapping) if mapping is not None else list(choices)
    nq = re.sub(r"[^a-z0-9]", "", (query or "").lower())
    if not nq:
        return Resolution("none")
    for label in labels:
        if re.sub(r"[^a-z0-9]", "", label.lower()) == nq:
            return Resolution("exact", Match(label, 100.0, mapping[label] if mapping is not None else None))
    hits = best_match(query, mapping if mapping is not None else labels, threshold, limit=5)
    if not hits:
        return Resolution("none")
    if len(hits) > 1 and hits[0].score - hits[1].score < AMBIGUITY_GAP and hits[0].score < 95:
        return Resolution("ambiguous", None, hits)
    return Resolution("fuzzy", hits[0], hits)


def suggestions(query: str, choices: Iterable[str], limit: int = 3, threshold: float = 55) -> list[str]:
    """'Did you mean' candidates: looser than resolve(), never auto-applied."""
    return [m.value for m in best_match(query, choices, threshold=threshold, limit=limit)]


# ---------------------------------------------------------------- filesystem names
def _listdir(p: Path) -> list[str]:
    try:
        return os.listdir(p)
    except OSError:
        return []


def fuzzy_path(path: Path, *, max_parts: int = 6) -> tuple[Path | None, list[Path]]:
    """Resolve a path whose last components don't exist by fuzzy-matching them against real siblings.

    Returns (unique_corrected_path | None, other_candidates). Only the missing tail is corrected; the existing
    prefix is trusted. Never touches anything: pure lookup."""
    path = Path(path)
    if path.exists():
        return path, []
    parts = list(path.parts)
    # find the longest existing prefix
    k = len(parts)
    while k > 0 and not Path(*parts[:k]).exists():
        k -= 1
    if k == 0 or len(parts) - k > max_parts:
        return None, []
    current = Path(*parts[:k])
    alternatives: list[Path] = []
    for i in range(k, len(parts)):
        names = _listdir(current)
        res = resolve(parts[i], names, threshold=72)
        if res.ok and res.match:
            current = current / res.match.value
            continue
        if res.status == "ambiguous":
            alternatives = [current / m.value for m in res.candidates]
        elif not alternatives:
            alternatives = [current / s for s in suggestions(parts[i], names)]
        return None, alternatives
    return current, []


def path_hint(path: Path) -> str:
    """Human-readable 'Did you mean ...' for an error message, or ''."""
    fixed, alts = fuzzy_path(path)
    cands = [fixed] if fixed else alts
    if not cands:
        return ""
    return " Did you mean: " + "; ".join(str(c) for c in cands[:3]) + "?"
