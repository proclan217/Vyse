"""Installed-app catalog: detect what is installed and fuzzy-match a spoken/typed name to it.

Sources, merged and de-duplicated by name (earlier wins):
  1. apps declared in config.toml                 (explicit, with aliases and process names)
  2. Start Menu shortcuts (.lnk)                   (launchable file, easy to verify by process name)
  3. `Get-StartApps`                               (everything in the Start menu, including Store/UWP apps)
  4. registry "App Paths"                          (classic apps that register an exe name)

The scan runs once, is cached on disk and refreshed in the background when stale, so `open_app` never waits on it
after the first run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config, expand
from .fuzzy import Resolution, resolve, suggestions

START_MENU_DIRS = [Path(os.environ.get("ProgramData", "C:/ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
                   Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs"]
_SKIP = re.compile(r"uninstall|readme|documentation|release notes|\bfaq\b|homepage|web site|website|on the web|support|manual|"
                   r"visit |help|sample|command prompt for vs|developer powershell", re.I)


@dataclass
class AppRecord:
    name: str
    kind: str                     # config | lnk | uwp | apppath
    target: str = ""              # exe path, .lnk path or AppUserModelID
    process: str = ""             # expected process name for verification ('' = unknown)
    args: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


Scanner = Callable[[], list[AppRecord]]


# ---------------------------------------------------------------- scanning (Windows)
def scan_lnk(dirs: list[Path] | None = None) -> list[AppRecord]:
    out: list[AppRecord] = []
    for d in dirs if dirs is not None else START_MENU_DIRS:
        if not d.is_dir():
            continue
        for lnk in d.rglob("*.lnk"):
            if not _SKIP.search(lnk.stem):
                out.append(AppRecord(lnk.stem, "lnk", str(lnk)))
    return out


def scan_start_apps(timeout: int = 20) -> list[AppRecord]:
    """UWP/Store and Start-menu apps via PowerShell. Empty list when PowerShell is unavailable."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "Get-StartApps | ConvertTo-Json -Compress"], capture_output=True, text=True, timeout=timeout)
        data = json.loads(r.stdout or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = [data]
    return [AppRecord(str(d["Name"]), "uwp", str(d["AppID"])) for d in data
            if d.get("Name") and d.get("AppID") and not _SKIP.search(str(d["Name"]))]


def scan_app_paths() -> list[AppRecord]:
    out: list[AppRecord] = []
    try:
        import winreg
    except ImportError:
        return out
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\App Paths") as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    exe = winreg.EnumKey(root, i)
                    try:
                        with winreg.OpenKey(root, exe) as k:
                            path = str(winreg.QueryValueEx(k, "")[0]).strip('"')
                    except OSError:
                        continue
                    if exe.lower().endswith(".exe") and path:
                        out.append(AppRecord(Path(exe).stem, "apppath", path, process=exe))
        except OSError:
            continue
    return out


def default_scan() -> list[AppRecord]:
    return [*scan_lnk(), *scan_start_apps(), *scan_app_paths()]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class AppCatalog:
    def __init__(self, cfg: Config, scanner: Scanner | None = None, cache_path: Path | None = None) -> None:
        self.cfg = cfg
        self.scanner = scanner or default_scan
        self.cache_path = cache_path if cache_path is not None else cfg.data_dir / "apps_cache.json"
        self._scanned: list[AppRecord] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ---- loading ----
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.app_scan)

    def _read_cache(self) -> bool:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._scanned = [AppRecord(**a) for a in raw["apps"]]
            self._loaded_at = float(raw["ts"])
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def refresh(self) -> int:
        """Scan now (blocking) and cache the result. Returns the number of apps found."""
        apps = self.scanner()
        with self._lock:
            self._scanned, self._loaded_at = apps, time.time()
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"ts": self._loaded_at, "apps": [asdict(a) for a in apps]}), encoding="utf-8")
        except OSError:
            pass
        return len(apps)

    def refresh_async(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.refresh, daemon=True, name="vyse-app-scan")
        self._thread.start()

    def _ensure(self) -> None:
        if not self.enabled:
            return
        if not self._scanned and not self._read_cache():
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=25)
            if not self._scanned:
                self.refresh()
            return
        if time.time() - self._loaded_at > self.cfg.app_scan_ttl_hours * 3600:
            self.refresh_async()

    # ---- queries ----
    def records(self) -> list[AppRecord]:
        """Config apps first, then scanned ones; one record per normalized name."""
        self._ensure()
        seen: dict[str, AppRecord] = {}
        for e in self.cfg.apps.values():
            seen[_key(e.name)] = AppRecord(e.name, "config", e.path or "", e.process or "", list(e.args), list(e.aliases))
        order = {"lnk": 0, "uwp": 1, "apppath": 2}
        for a in sorted(self._scanned, key=lambda a: order.get(a.kind, 9)):
            seen.setdefault(_key(a.name), a)
        return list(seen.values())

    def labels(self) -> dict[str, AppRecord]:
        out: dict[str, AppRecord] = {}
        for rec in self.records():
            out.setdefault(rec.name, rec)
            for al in rec.aliases:
                out.setdefault(al, rec)
        return out

    def resolve(self, query: str) -> Resolution:
        """Exact, fuzzy or ambiguous match of a requested app name. Resolution.match.key holds the AppRecord."""
        labels = self.labels()
        mapping = {label: rec for label, rec in labels.items()}
        res = resolve(query, mapping, threshold=80)
        # collapse "ambiguous" when every candidate is the same app under different aliases
        if res.status == "ambiguous" and len({id(c.key) for c in res.candidates}) == 1:
            return Resolution("fuzzy", res.candidates[0], res.candidates)
        return res

    def suggest(self, query: str, limit: int = 3) -> list[str]:
        names = [r.name for r in self.records()]
        return suggestions(query, names, limit=limit, threshold=55)

    def names(self) -> list[str]:
        return sorted(r.name for r in self.records())
