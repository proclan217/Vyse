"""Rich terminal REPL for Vyse."""
from __future__ import annotations

import argparse
import sys
import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from . import __version__
from .agent import Agent, Hooks
from .config import load_config
from .context import Context
from .llm import LLMError, make_client
from .tools import build_registry

HELP = """[bold]Commands[/bold]
  /help            show this help
  /tools           list available tools and their risk level
  /memory          show remembered facts
  /stats [hours]   latency, failures, token use and routing (default 24h)
  /history         recent file operations (transaction log)
  /undo [n]        undo the last n file operations (default 1)
  /tasks           background tasks
  /schedules       reminders and scheduled tasks
  /index [rebuild] file-search index status
  /apps            installed apps Vyse can launch
  /cache [clear]   tool-result cache stats
  /model [name]    show/list models, or switch model
  /yes             toggle auto-approve for risky actions (blocked actions stay blocked)
  /reset           clear this conversation (facts are kept)
  /quit            exit

Try: "Open Discord and Medal" · "Find my PDFs modified this week" · "Organize my Downloads"
     "Remember my printer is an HP M15" · "Note: ideas for the robot arm" · "Weather in Berlin\""""


class App:
    def __init__(self, console: Console | None = None, config_path: str | None = None) -> None:
        self.console = console or Console()
        self.cfg = load_config(config_path)
        self.ctx = Context.build(self.cfg)
        self.ctx.notify = self._notify
        self.llm = make_client(self.cfg.model)
        self.registry = build_registry(self.ctx)
        self.agent = Agent(self.llm, self.registry, self.ctx, Hooks(
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end, confirm=self._confirm))

    def _notify(self, text: str) -> None:
        """Background tasks and reminders print here (from worker threads)."""
        self.console.print(f"
[magenta]{text}[/magenta]", highlight=False)

    def startup(self) -> None:
        """Warm the slow services in the background so the first request is fast."""
        ctx = self.ctx
        try:
            ctx.metrics.prune(self.cfg.metrics_retention_days)
        except Exception:
            pass
        if ctx.index is not None and (not ctx.index.ready or ctx.index.is_stale()):
            ctx.index.build_async()
        if ctx.apps is not None and ctx.apps.enabled:
            ctx.apps.refresh_async()
        if ctx.scheduler is not None:
            for notice in ctx.scheduler.start():
                self._notify(notice)

    # ---- hooks ----
    def _on_tool_start(self, name: str, args: dict) -> None:
        shown = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items())
        self.console.print(f"[dim]  ⚙ {name}({shown})[/dim]")

    def _on_tool_end(self, name: str, ok: bool, summary: str) -> None:
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        self.console.print(f"  {mark} {summary if ok else f'{name}: {summary}'}", highlight=False)

    def _confirm(self, preview: str, reason: str) -> bool:
        self.console.print(Panel(preview, title="[yellow]Confirm action[/yellow]", subtitle=reason, border_style="yellow"))
        return Confirm.ask("Allow?", default=False, console=self.console)

    # ---- commands ----
    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False to quit."""
        cmd, _, arg = line.strip().partition(" ")
        c = self.console
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            c.print(HELP)
        elif cmd == "/reset":
            self.agent.reset()
            c.print("[dim]Conversation cleared.[/dim]")
        elif cmd == "/yes":
            self.agent.auto_yes = not self.agent.auto_yes
            c.print(f"Auto-approve is now [bold]{'ON' if self.agent.auto_yes else 'OFF'}[/bold].")
        elif cmd == "/memory":
            facts = self.ctx.memory.all_facts()
            if not facts:
                c.print("[dim]No facts remembered yet.[/dim]")
            t = Table("key", "fact", "tags")
            for f in facts:
                t.add_row(f.key, f.text, f.tags)
            if facts:
                c.print(t)
        elif cmd == "/stats":
            self._stats(float(arg) if arg.strip().replace(".", "", 1).isdigit() else 24.0)
        elif cmd == "/history":
            rows = self.ctx.journal.history(15)
            t = Table("id", "when", "what", "undone")
            for r in rows:
                t.add_row(r["batch"], time.strftime("%m-%d %H:%M", time.localtime(r["ts"])), r["summary"], "yes" if r["undone"] else "")
            c.print(t if rows else "[dim]No file operations recorded yet.[/dim]")
        elif cmd == "/undo":
            n = int(arg) if arg.strip().isdigit() else 1
            done, problems = self.ctx.journal.undo(steps=n)
            c.print(f"Reverted {len(done)} item(s)." + "".join(f"
  [yellow]{p}[/yellow]" for p in problems))
        elif cmd == "/tasks":
            rows = self.ctx.tasks.list()
            t = Table("id", "task", "status", "seconds")
            for x in rows:
                t.add_row(str(x.id), x.name, x.status, str(x.elapsed()))
            c.print(t if rows else "[dim]No background tasks.[/dim]")
        elif cmd == "/schedules":
            if self.ctx.scheduler is None:
                c.print("[dim]Scheduler is disabled.[/dim]")
            else:
                rows = self.ctx.scheduler.listing()
                t = Table("id", "name", "when", "next", "runs", "on")
                for r in rows:
                    t.add_row(str(r["id"]), r["name"], r["when"], r["next"], str(r["runs"]), "yes" if r["enabled"] else "no")
                c.print(t if rows else "[dim]Nothing scheduled.[/dim]")
                c.print("[dim]Reminders only fire while Vyse is running.[/dim]")
        elif cmd == "/index":
            ix = self.ctx.index
            if ix is None:
                c.print("[dim]File index is disabled.[/dim]")
            else:
                if arg.strip() == "rebuild":
                    ix.build_async()
                st = ix.stats()
                c.print(f"Indexed files: {st['files']} · ready: {st['ready']} · building: {st['building']}")
        elif cmd == "/apps":
            cat = self.ctx.apps
            names = cat.names() if cat is not None and cat.enabled else []
            c.print(f"{len(names)} app(s): " + ", ".join(names[:80]) if names else "[dim]No apps detected (or scanning is off).[/dim]")
        elif cmd == "/cache":
            if arg.strip() == "clear":
                self.ctx.cache.clear()
            c.print(str(self.ctx.cache.stats()))
        elif cmd == "/tools":
            t = Table("tool", "permission", "description")
            colors = {"safe": "green", "confirmation": "yellow", "blocked": "red"}
            for tool in sorted(self.registry.all(), key=lambda x: (x.group, x.name)):
                perm = self.ctx.policy.permission_of(tool)
                t.add_row(tool.name, f"[{colors.get(perm, 'white')}]{perm}[/]" + (" !" if tool.destructive else ""), tool.description[:80])
            c.print(t)
        elif cmd == "/model":
            if arg:
                self.llm.set_model(arg.strip())
                c.print(f"Model set to [bold]{self.llm.model}[/bold].")
            else:
                try:
                    c.print(f"Current: [bold]{self.llm.model}[/bold]\nInstalled: {', '.join(self.llm.list_models())}")
                except LLMError as e:
                    c.print(f"[red]{e}[/red]")
        else:
            c.print(f"[red]Unknown command {cmd}.[/red] Type /help.")
        return True

    def _stats(self, hours: float) -> None:
        s = self.ctx.metrics.summary(hours)
        m, cm, r = s["model"], s["commands"], s["routing"]
        c = self.console
        c.print(f"[bold]Last {hours:g}h[/bold]: {cm['total']} command(s) ({cm['succeeded']} ok, {cm['failed']} failed), "
                f"avg {cm['avg_ms']} ms")
        c.print(f"Model: {m['calls']} call(s), {m['failures']} failed, p50 {m['p50_ms']} ms, p95 {m['p95_ms']} ms, "
                f"{m['prompt_tokens']}+{m['completion_tokens']} tokens" + (" (estimated)" if m["tokens_estimated"] else ""))
        c.print(f"Routing: {r['requests']} request(s), {r['forced_tool']} forced a tool, {r['no_tool_match']} matched no tool, "
                f"~{r['avg_tools_offered']} tools offered")
        c.print(f"Repairs: {s['repairs']} · parallel tool calls: {s['parallel_calls']} · cache: {self.ctx.cache.stats()}")
        if s["tools"]:
            t = Table("tool", "calls", "failed", "cached", "p50 ms", "p95 ms")
            for n, d in list(s["tools"].items())[:12]:
                t.add_row(n, str(d["calls"]), str(d["failures"]), str(d["cached"]), str(d["p50_ms"]), str(d["p95_ms"]))
            c.print(t)
        if s["tool_errors"]:
            c.print("Errors: " + ", ".join(f"{k}={v}" for k, v in s["tool_errors"].items()))

    def ask(self, text: str) -> None:
        try:
            answer = self.agent.run_turn(text)
        except KeyboardInterrupt:
            self.console.print("\n[dim]Interrupted.[/dim]")
            return
        self.console.print(f"[bold cyan]Vyse:[/bold cyan] {answer}", highlight=False)

    def repl(self) -> None:
        self.startup()
        threading.Thread(target=self.llm.warm, daemon=True).start()     # load the model while the user types
        self.console.print(Panel.fit(
            f"[bold]Vyse[/bold] v{__version__} · model [cyan]{self.llm.model}[/cyan] · "
            f"{len(self.registry.all())} tools · /help for commands", border_style="cyan"))
        while True:
            try:
                line = self.console.input("[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    break
                continue
            self.ask(line)
        self.console.print("[dim]Goodbye.[/dim]")
        self.close()

    def close(self) -> None:
        from .tools import mcp_client
        mcp_client.shutdown()
        if self.ctx.scheduler is not None:
            self.ctx.scheduler.stop()
        self.ctx.tasks.shutdown()
        self.ctx.metrics.close()


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):      # legacy cp1252 consoles/pipes choke on symbols
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="vyse", description="Vyse: local AI receptionist for your PC")
    ap.add_argument("-c", "--config", help="path to config.toml")
    ap.add_argument("-m", "--model", help="override the Ollama model")
    ap.add_argument("prompt", nargs="*", help="run a single prompt and exit")
    args = ap.parse_args(argv)
    app = App(config_path=args.config)
    if args.model:
        app.llm.set_model(args.model)
    if args.prompt:
        app.ask(" ".join(args.prompt))
        app.close()
        return
    app.repl()


if __name__ == "__main__":
    main(sys.argv[1:])
