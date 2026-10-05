"""Every module must at least compile and import (the CLI is not otherwise exercised by the suite)."""
from __future__ import annotations

import compileall
import importlib
import pathlib

import vyse


def test_all_modules_compile():
    assert compileall.compile_dir(str(pathlib.Path(vyse.__file__).parent), quiet=1, force=True)


def test_cli_imports_and_handles_slash_commands(ctx, monkeypatch):
    cli = importlib.import_module("vyse.cli")
    assert hasattr(cli, "main")
    assert "/stats" in cli.HELP and "/undo" in cli.HELP
