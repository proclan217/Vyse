"""Deterministic file-organization rules. No model involved: the same folder always produces the same plan.

Rules come from config.toml ([[organize.rules]]); when none are configured, DEFAULT_RULES are used. The first rule
that matches a file wins. Files that match nothing stay where they are (they are not shoved into an 'Other' folder),
unless the caller asks for a catch-all.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import OrganizeRule

MB = 1024 * 1024
SKIP_SUFFIXES = {".crdownload", ".part", ".tmp", ".partial", ".download", ".lnk", ".ini"}

DEFAULT_RULES: list[OrganizeRule] = [
    OrganizeRule("Screenshots", "Images/Screenshots", ext=["png", "jpg"], name_contains=["screenshot", "screen shot", "snip"]),
    OrganizeRule("Installers", "Installers", ext=["exe", "msi", "msix", "appx", "iso"]),
    OrganizeRule("Invoices & receipts", "Documents/Finance", ext=["pdf"], name_contains=["invoice", "receipt", "statement", "bill"]),
    OrganizeRule("Images", "Images", ext=["jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "heic", "tiff", "ico", "raw"]),
    OrganizeRule("Videos", "Videos", ext=["mp4", "mkv", "avi", "mov", "wmv", "webm", "flv"]),
    OrganizeRule("Audio", "Audio", ext=["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma"]),
    OrganizeRule("Documents", "Documents", ext=["pdf", "doc", "docx", "txt", "rtf", "odt", "md", "epub", "tex"]),
    OrganizeRule("Spreadsheets", "Spreadsheets", ext=["xls", "xlsx", "csv", "ods", "tsv"]),
    OrganizeRule("Presentations", "Presentations", ext=["ppt", "pptx", "odp", "key"]),
    OrganizeRule("Archives", "Archives", ext=["zip", "rar", "7z", "tar", "gz", "bz2", "xz"]),
    OrganizeRule("3D & CAD", "3D & CAD", ext=["stl", "step", "stp", "obj", "3mf", "f3d", "dxf", "dwg", "gcode", "fbx", "blend", "sldprt"]),
    OrganizeRule("Code", "Code", ext=["py", "js", "ts", "html", "css", "json", "cpp", "c", "h", "java", "rs", "go", "ino", "ipynb", "toml", "yaml", "yml"]),
]


@dataclass
class FileFacts:
    name: str
    ext: str
    size: int
    mtime: float


def facts_of(p: Path) -> FileFacts:
    st = p.stat()
    return FileFacts(p.name, p.suffix.lower().lstrip("."), st.st_size, st.st_mtime)


def matches(rule: OrganizeRule, f: FileFacts, now: float | None = None) -> bool:
    """A rule matches when EVERY condition it declares holds (conditions it leaves empty are ignored)."""
    now = now if now is not None else time.time()
    conds = 0
    if rule.ext:
        conds += 1
        if f.ext not in {e.lower().lstrip(".") for e in rule.ext}:
            return False
    if rule.name_contains:
        conds += 1
        low = f.name.lower()
        if not any(s.lower() in low for s in rule.name_contains):
            return False
    if rule.name_regex:
        conds += 1
        try:
            if not re.search(rule.name_regex, f.name, re.I):
                return False
        except re.error:
            return False
    if rule.min_size_mb:
        conds += 1
        if f.size < rule.min_size_mb * MB:
            return False
    if rule.max_size_mb:
        conds += 1
        if f.size > rule.max_size_mb * MB:
            return False
    if rule.older_than_days:
        conds += 1
        if now - f.mtime < rule.older_than_days * 86400:
            return False
    return conds > 0


def rules_for(configured: list[OrganizeRule] | None) -> list[OrganizeRule]:
    return list(configured) if configured else list(DEFAULT_RULES)


def classify(p: Path, rules: list[OrganizeRule], now: float | None = None) -> OrganizeRule | None:
    f = facts_of(p)
    return next((r for r in rules if matches(r, f, now)), None)


def build_rule_plan(folder: Path, rules: list[OrganizeRule], catch_all: str = "", now: float | None = None) -> list[dict[str, str]]:
    """Moves for the files directly inside `folder`. Pure: reads metadata only. Skips partial downloads and hidden files."""
    moves: list[dict[str, str]] = []
    for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.name.startswith((".", "~$")) or p.suffix.lower() in SKIP_SUFFIXES:
            continue
        rule = classify(p, rules, now)
        dest = rule.dest if rule else catch_all
        if not dest:
            continue
        dst = folder / dest / p.name
        if dst.parent == p.parent:
            continue
        moves.append({"src": str(p), "dst": str(dst), "category": dest, "rule": rule.name if rule else "catch-all"})
    return moves
