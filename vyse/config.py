"""Configuration loading. Everything is a plain dataclass so tests can build one directly."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


@dataclass
class ModelConfig:
    provider: str = "ollama"         # "ollama" (local) or "openai" (any OpenAI-compatible API, e.g. NVIDIA NIM)
    base_url: str = "https://integrate.api.nvidia.com/v1"
    rpm_limit: int = 38               # client-side cap on requests per minute (NVIDIA free tier allows 40)
    api_key_env: str = "NVIDIA_API_KEY"   # the key is read from this environment variable, never stored in files
    name: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    think: bool = False
    temperature: float = 0.3
    num_ctx: int = 3072
    num_predict: int = 400        # hard cap on generated tokens (a tool call is ~30; stops rambling)
    num_gpu: int = 0              # 0 = run entirely on CPU/RAM (partial GPU offload is slower on this 4 GB card)
    keep_alive: str = "30m"       # how long Ollama keeps the model in memory after a request
    timeout: float = 300.0


@dataclass
class AgentConfig:
    max_steps: int = 8
    history_messages: int = 6
    summarize_after: int = 20
    relevant_facts: int = 5
    max_tools_per_turn: int = 14
    parallel_tools: bool = True       # run independent tools of one model step concurrently
    max_parallel: int = 4
    max_calls_per_step: int = 6       # a model step may chain/batch at most this many calls
    context_tokens: int = 2200        # soft budget for the prompt (history + tool results); older parts get trimmed
    tool_result_chars: int = 1800     # cap per tool result sent back to the model
    tool_timeout: float = 60.0        # seconds before a parallel tool call is reported as timed out
    relevant_memories: int = 5


@dataclass
class IndexConfig:
    enabled: bool = True
    roots: list[str] = field(default_factory=lambda: ["~/Desktop", "~/Documents", "~/Downloads"])
    max_files: int = 400_000
    stale_minutes: float = 30.0       # a search triggers a background refresh when the index is older than this


@dataclass
class OrganizeRule:
    """One deterministic sorting rule: files that match go to `dest` (relative to the folder being organized)."""
    name: str
    dest: str
    ext: list[str] = field(default_factory=list)
    name_contains: list[str] = field(default_factory=list)
    name_regex: str = ""
    min_size_mb: float = 0.0
    max_size_mb: float = 0.0
    older_than_days: float = 0.0


@dataclass
class AppEntry:
    name: str
    aliases: list[str] = field(default_factory=list)
    path: str | None = None
    args: list[str] = field(default_factory=list)
    process: str | None = None


@dataclass
class McpServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    trusted_tools: list[str] = field(default_factory=list)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_dir: Path = field(default_factory=lambda: expand("~/.vyse"))
    notes_dir: Path = field(default_factory=lambda: expand("~/Documents/VyseNotes"))
    allowed_roots: list[Path] = field(default_factory=list)
    read_roots: list[Path] = field(default_factory=list)
    protected_paths: list[Path] = field(default_factory=list)
    organize_confirm_threshold: int = 25
    confirm_risky: bool = True
    auto_yes: bool = False
    apps: dict[str, AppEntry] = field(default_factory=dict)
    mcp_servers: list[McpServer] = field(default_factory=list)
    google_enabled: bool = True
    # --- feature settings ---
    index: IndexConfig = field(default_factory=IndexConfig)
    organize_rules: list[OrganizeRule] = field(default_factory=list)
    permissions: dict[str, str] = field(default_factory=dict)   # tool name -> safe | confirmation | blocked (stricter only)
    app_scan: bool = True              # detect installed apps (Start Menu, App Paths, UWP) for fuzzy launching
    app_scan_ttl_hours: float = 12.0
    cache_enabled: bool = True
    cache_max_entries: int = 256
    scheduler_enabled: bool = True
    metrics_enabled: bool = True
    metrics_retention_days: int = 30

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vyse.db"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"

    @property
    def journal_path(self) -> Path:
        return self.data_dir / "undo_journal.jsonl"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.trash_dir, self.log_dir, self.notes_dir):
            d.mkdir(parents=True, exist_ok=True)


def _paths(values: list[str]) -> list[Path]:
    return [expand(v) for v in values]


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.toml. Search order: explicit path, $VYSE_CONFIG, ./config.toml, ~/.vyse/config.toml."""
    candidates = [path, os.environ.get("VYSE_CONFIG"), Path.cwd() / "config.toml",
                  Path(__file__).resolve().parent.parent / "config.toml",
                  expand("~/.vyse/config.toml")]
    raw: dict = {}
    for c in candidates:
        if c and Path(c).is_file():
            with open(c, "rb") as f:
                raw = tomllib.load(f)
            break

    m, a, p, pol = raw.get("model", {}), raw.get("agent", {}), raw.get("paths", {}), raw.get("policy", {})
    cfg = Config(
        model=ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__}),
        agent=AgentConfig(**{k: v for k, v in a.items() if k in AgentConfig.__dataclass_fields__}),
    )
    cfg.data_dir = expand(p.get("data_dir", "~/.vyse"))
    cfg.notes_dir = expand(p.get("notes_dir", "~/Documents/VyseNotes"))
    cfg.allowed_roots = _paths(p.get("allowed_roots", ["~/Desktop", "~/Documents", "~/Downloads"]))
    cfg.read_roots = _paths(p.get("read_roots", ["~"]))
    cfg.protected_paths = _paths(p.get("protected_paths", [
        "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)"]))
    # notes dir is always writable
    if cfg.notes_dir not in cfg.allowed_roots:
        cfg.allowed_roots.append(cfg.notes_dir)
    cfg.organize_confirm_threshold = pol.get("organize_confirm_threshold", 25)
    cfg.confirm_risky = pol.get("confirm_risky", True)
    cfg.auto_yes = pol.get("auto_yes", False)
    for name, entry in raw.get("apps", {}).items():
        cfg.apps[name] = AppEntry(
            name=name, aliases=[x.lower() for x in entry.get("aliases", [name])],
            path=entry.get("path"), args=entry.get("args", []), process=entry.get("process"))
    for s in raw.get("mcp", {}).get("servers", []):
        cfg.mcp_servers.append(McpServer(
            name=s["name"], command=s["command"], args=s.get("args", []),
            trusted_tools=s.get("trusted_tools", [])))
    cfg.google_enabled = raw.get("google", {}).get("enabled", True)
    ix = raw.get("index", {})
    cfg.index = IndexConfig(**{k: v for k, v in ix.items() if k in IndexConfig.__dataclass_fields__})
    for r in raw.get("organize", {}).get("rules", []):
        cfg.organize_rules.append(OrganizeRule(**{k: v for k, v in r.items() if k in OrganizeRule.__dataclass_fields__}))
    perms = raw.get("permissions", {})
    cfg.permissions = {str(k): str(v) for k, v in perms.items() if str(v) in ("safe", "confirmation", "blocked")}
    feat = raw.get("features", {})
    for key in ("app_scan", "app_scan_ttl_hours", "cache_enabled", "cache_max_entries", "scheduler_enabled",
                "metrics_enabled", "metrics_retention_days"):
        if key in feat:
            setattr(cfg, key, feat[key])
    return cfg
