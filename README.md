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
- **Organize**: plan_organize (dry run) → apply_plan → undo_last; strategies by_type / by_date / by_name
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
