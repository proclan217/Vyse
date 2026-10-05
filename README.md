# Vyse

A local-first AI receptionist for your Windows PC. **Qwen3** is the model, **Ollama** runs it, **Vyse** is the agent layer: it opens apps, finds and organizes files, takes notes, remembers facts, checks system status, searches the web, and (optionally) reads Gmail/Calendar/Drive or talks to MCP servers.

```
You: Find my PDFs modified this week
  ⚙ find_files(extension='pdf', modified_within_days=7)
  ✓ Found 3 file(s)
Vyse: ...
```

## Setup

```powershell
# 1. Ollama + a model
ollama pull qwen3:1.7b        # works, but small; qwen3:8b or qwen3:14b is far better at multi-step tasks
# 2. Vyse
cd C:\Users\procl\Desktop\Vyse
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -m vyse                 # interactive REPL
.venv\Scripts\python -m vyse "What's my RAM usage?"   # one-shot
```

Options: `-c path\to\config.toml`, `-m qwen3:8b`. In the REPL: `/help /tools /memory /model [name] /yes /reset /quit`.

## Configuration (`config.toml`)

Searched in this order: `-c` path, `$VYSE_CONFIG`, current dir, project dir, `~/.vyse/config.toml`.

| Section | What it controls |
|---|---|
| `[model]` | `name`, `ollama_url`, `think` (Qwen reasoning on/off), `temperature`, `num_ctx` |
| `[agent]` | `max_steps` (default 8), tools per turn, history/summary sizes |
| `[paths]` | `allowed_roots` (writes allowed without asking), `read_roots`, `protected_paths`, `notes_dir` |
| `[policy]` | `organize_confirm_threshold`, `confirm_risky`, `auto_yes` |
| `[apps.*]` | App registry for `open_app` (path, args, aliases, process name): Discord, Medal, Valorant, Chrome, Steam, Spotify... |
| `[[mcp.servers]]` | MCP servers (stdio) and `trusted_tools` |
| `[google]` | `enabled` |

All state lives in `%USERPROFILE%\.vyse\`: `vyse.db` (memory), `logs\`, `trash\`, `journal.jsonl` (undo), `plans\`, Google token. Notes go to `%USERPROFILE%\Documents\VyseNotes`.

Switching models is just `/model qwen3:8b` or `[model] name = ...`; the agent talks to an abstract `LLMClient`, so other backends can be added in `vyse/llm.py`.

## Tools

- **Files**: find_files, list_dir, read_file, file_info, move, copy, make_dir, trash
- **Organize**: plan_organize (dry run) → apply_plan → undo_last / undo / undo_history; strategies by_rules (default) / by_type / by_date / by_name
- **Notes**: create_note, append_note, search_notes, read_note (Markdown with front matter)
- **Memory**: remember, recall, forget (SQLite + FTS5; only relevant facts are injected each turn; old chat is rolled into a summary)
- **System**: open_app, open_path, get_running_apps, system_info, CPU/RAM/GPU/disk, volume, current time, clipboard, run_command
- **Web**: web_search (DuckDuckGo), fetch_url, weather (Open-Meteo)
- **Google** (optional), **MCP** (optional)

Only a relevant subset of tools is shown to the model each turn, which matters for small models.

## Safety model

The model proposes; the policy decides, and no tool argument can override it.

- **Safe** actions run automatically. **Write** actions inside allowed roots run automatically and are journaled. **Risky** actions show a preview and ask `y/n`.
- **Nothing is permanently deleted.** "Delete" moves to `~\.vyse\trash`. File moves/copies/folder creation are journaled; `undo_last` reverses the last batch (collision-safe, never overwrites).
- **Sandbox**: writes outside allowed roots need confirmation; `C:\Windows`, `C:\Program Files*`, other protected paths and Vyse's own data dir are blocked outright (even with `/yes`).
- `run_command` always asks, and a blocklist (format, diskpart, `rm -rf`, registry deletes, shutdown...) is blocked.
- Web pages, file contents and MCP output are untrusted data. `fetch_url` refuses private/loopback addresses (SSRF guard).
- Organize is always plan first; applying more than `organize_confirm_threshold` files asks for confirmation.
- Actions are verified (process started? file moved?) and the model is told when verification failed.

`/yes` auto-approves confirmations for the session; blocked actions stay blocked.

## Optional: Google (Gmail / Calendar / Drive)

1. `pip install -e ".[google]"`
2. Create an OAuth *Desktop* client in Google Cloud Console, enable the Gmail, Calendar and Drive APIs.
3. Save it as `%USERPROFILE%\.vyse\credentials.json`; keep `[google] enabled = true`.

Tools register only when credentials exist. Reads are read-only scopes and safe; `gmail_send` and `calendar_create` always ask for confirmation. The first use opens a browser for consent.

## Optional: MCP

```toml
[[mcp.servers]]
name = "fs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\you\\Documents"]
trusted_tools = ["read_file"]
```

Tools appear as `fs__read_file`. Anything not in `trusted_tools` is risky (confirmation required).

## Reliability, speed and automation features

**Tool calls are checked before they run**
- *Validation*: every tool name and argument the model produces is checked against the tool's schema (required fields, types, enums, unknown arguments). Harmless slips are fixed (`"3"` -> `3`, wrong case); everything else is rejected.
- *Repair*: malformed JSON arguments (single quotes, trailing commas, code fences, unclosed braces, bad Windows-path escapes) and misspelled tool names are repaired automatically before the call is retried.
- *Structured errors*: a failed call returns `{ok:false, error_type, retryable, hint, expected, suggestions}` so the model can correct itself instead of giving up.
- *Chaining*: one model step may contain several dependent calls; later calls reference earlier results with `$1.path` / `${2.info.size}`. A call whose dependency failed is skipped, not run with garbage.
- *Parallel execution*: independent read-only calls in one step run concurrently (thread pool driven by asyncio) with a per-call timeout (`[agent] tool_timeout`).

**Fuzzy matching (RapidFuzz)**: app names, files, folders and process names tolerate typos. Corrections are applied automatically only for read-only lookups (reading, listing, opening an app), and a corrected path is re-checked by the safety policy. Write operations never guess: they fail with "did you mean ...?" suggestions. Very short queries are never fuzzy-matched and near-ties are reported as ambiguous.

**App launcher**: Vyse scans the Start Menu, App Paths and Store apps (cached, refreshed in the background, `/apps` lists them), then fuzzy-matches the request. Your `[apps.*]` entries still win.

**Permissions and confirmation**: every tool has a permission: `safe` (just runs), `confirmation` (asks first) or `blocked` (never runs). Defaults derive from the tool's risk; `[permissions]` in the config can only make a tool stricter. *Destructive* tools (`trash`, `kill_process`) always ask, even with `/yes`. Critical system processes and Vyse itself cannot be killed. Scheduled and background runs are unattended, so anything that would need a confirmation is declined and the task is reported as failed.

**Undo**: every file operation is written to a transaction log (the undo journal). `/history` (or `undo_history`) lists it; `/undo [n]` or the `undo` tool reverses the last *n* transactions or one by id. "Deleting" is a move to Vyse's trash, so it is recoverable.

**Fast file search**: `find_files` is answered from a SQLite index of your configured roots (built in the background, refreshed when stale; protected folders are never indexed). Results are verified against the disk and typos fall back to fuzzy name matching. If the index is not ready the tool falls back to a live scan. `/index [rebuild]`.

**File organization**: deterministic Python rules (no model involved), same folder in, same plan out, plus your own `[[organize.rules]]` (extension, name text/regex, size, age). Always dry-run first, then apply, then undo if you like.

**Memory**: SQLite + FTS5. Facts have a kind and an importance; finished tasks are recorded too. Each turn only the memories relevant to the request (ranked, typo-tolerant, within a size budget) are injected, never the whole history.

**Context manager**: the prompt is kept under `[agent] context_tokens`: old tool results shrink first, then the oldest turns drop; huge tool results are compacted before the model sees them.

**Tool-result cache**: read-only tools (system info, CPU/RAM, listings) declare a TTL; results are reused until they expire or a write tool that could change them runs. `/cache [clear]`.

**Background tasks**: `start_background_task` runs a tool or routine in a worker thread and returns immediately; you are notified when it finishes. `/tasks`, `list_background_tasks`, `background_task_status`, `cancel_background_task`.

**Scheduling and reminders**: `set_reminder` and `schedule_task` (once / interval / daily / cron) are stored in SQLite and run by APScheduler. **Limitation:** jobs fire only while Vyse is running. The scheduler is rebuilt from the database at each start; one-shot reminders that came due while Vyse was closed are reported (not run) at the next start. Firing while Vyse is closed would need Windows Task Scheduler, which is not set up. `/schedules`.

**Observability**: model latency (p50/p95), tool latency, failures by error type, token usage, routing decisions and command success are stored in a local SQLite database and shown by `/stats [hours]` or the `vyse_stats` tool. Retention: `metrics_retention_days`.

New slash commands: `/stats /history /undo /tasks /schedules /index /apps /cache`. New config sections: `[features]`, `[index]`, `[permissions]`, `[[organize.rules]]`, and more `[agent]` keys (`max_calls_per_step`, `context_tokens`, `tool_result_chars`, `tool_timeout`, `relevant_memories`, `parallel_tools`, `max_parallel`).

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Tests use temp directories and a scripted fake LLM; they never touch your real files or Ollama.

## Notes on small models

`qwen3:1.7b` sometimes answers instead of calling a tool; Vyse nudges it once with a narrowed tool list, but if a past conversation contains made-up "I did it" replies the model tends to imitate them. Use `/reset` if it starts doing that, or move to `qwen3:8b`+.

## Latency design

Vyse is a receptionist, so it is tuned for speed with `qwen3:4b-instruct-2507` (non-thinking) running entirely on CPU/RAM (`num_gpu = 0`):

- Every request goes through the model; nothing is hard-coded, so typos and paraphrases work.
- Thinking is off, the system prompt asks for summarized answers, `num_ctx` is 3072, history is 4 messages, 8 tools per turn.
- No second "narration" call when a tool's own result is the answer (single-intent requests).
- Stable system-prompt prefix (Ollama KV-cache hits), model preloaded at REPL start, `keep_alive` keeps it resident,
  history summarization runs after the answer is shown.

## Hosted model (NVIDIA NIM)

`config.toml` defaults to `provider = "openai"` with `nvidia/nemotron-3.5-lightning-30b-a3b` at
`https://integrate.api.nvidia.com/v1`. Generate a key on build.nvidia.com and store it outside the repo:

    setx NVIDIA_API_KEY "nvapi-..."      # then open a new terminal

Set `provider = "ollama"` to go back to local models. Note: with a hosted provider your prompts and tool results leave the PC.


## Teaching Vyse (memory & routines)

Nothing here is hard-coded: Vyse stores what you tell it and the model decides how to use it.

- **Favorites** - "my favorite show is Severance, here's the Netflix link ..." / "my lofi playlist is <Spotify link>". Stored as facts (SQLite + FTS5) together with how to open them, and recalled automatically when a request matches.
- **Routines** - "save a routine called study session: open Clock, 25 minute timer, my lofi playlist, my favorite show, close Valorant and Roblox Player". The model turns this into a list of tool calls (`save_routine`); "start my study session" runs them (`run_routine`). Every step still goes through normal validation, safety policy and confirmations. Steps are checked when saved, so wrong argument names are rejected immediately.
- **Clarifying questions** - if "open spotify" could mean the app or a remembered playlist, Vyse asks (`ask_user`) instead of guessing.
- New tools used by routines: `close_app` (graceful close, never a force kill) and `set_timer` (beep + popup). Manage with `/memory`, "list my routines", "delete the study session routine".
