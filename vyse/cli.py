"""Rich terminal REPL for Vyse."""
from __future__ import annotations

import argparse
import sys
import threading

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
        self.llm = make_client(self.cfg.model)
        self.registry = build_registry(self.ctx)
        self.agent = Agent(self.llm, self.registry, self.ctx, Hooks(
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end, confirm=self._confirm))

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
        elif cmd == "/tools":
            t = Table("tool", "risk", "description")
            colors = {"safe": "green", "write": "yellow", "risky": "red"}
            for tool in sorted(self.registry.all(), key=lambda x: (x.group, x.name)):
                t.add_row(tool.name, f"[{colors[tool.risk]}]{tool.risk}[/]", tool.description[:80])
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

    def ask(self, text: str) -> None:
        try:
            answer = self.agent.run_turn(text)
        except KeyboardInterrupt:
            self.console.print("\n[dim]Interrupted.[/dim]")
            return
        self.console.print(f"[bold cyan]Vyse:[/bold cyan] {answer}", highlight=False)

    def repl(self) -> None:
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
