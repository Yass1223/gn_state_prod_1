#!/usr/bin/env python
"""Import every pipeline stage up front.

Hydra instantiates modules lazily, one stage at a time, so a stale transitive
dependency in stage 7 is only discovered after stages 1-6 have already run --
turning dependency debugging into one failure per (slow) evaluation run.

This walks the same `_target_` list the configs declare and imports each one,
so every breakage surfaces in a single pass that takes seconds and needs no GPU.

    python scripts/preflight_imports.py
    python scripts/preflight_imports.py -v     # full traceback for each failure

Exit code 0 means every stage is importable and `tracklab` will get as far as
actually running. Non-zero is the count of broken stages.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import traceback
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "sn_gamestate" / "configs"
TARGET_RE = re.compile(r"^\s*_target_:\s*(\S+)\s*$", re.MULTILINE)

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[0m"
)


def discover_targets() -> list[str]:
    """Collect every _target_ declared under configs/ (deduped, sorted)."""
    if not CONFIG_DIR.is_dir():
        sys.exit(f"config directory not found: {CONFIG_DIR}")
    targets: set[str] = set()
    for yaml_file in CONFIG_DIR.rglob("*.yaml"):
        targets.update(TARGET_RE.findall(yaml_file.read_text(encoding="utf-8")))
    return sorted(targets)


def try_import(dotted: str) -> tuple[bool, str, str]:
    """Import 'pkg.mod.Class'. Returns (ok, short_reason, full_traceback)."""
    module_path, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - report anything, don't crash
        return False, f"{type(exc).__name__}: {exc}", traceback.format_exc()
    if not hasattr(module, attr):
        return False, f"module '{module_path}' has no attribute '{attr}'", ""
    return True, "", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the full traceback for each failure")
    args = ap.parse_args(argv)

    targets = discover_targets()
    print(f"Importing {len(targets)} pipeline stages "
          f"(python {sys.version.split()[0]})\n")

    failures: list[tuple[str, str, str]] = []
    for dotted in targets:
        ok, reason, tb = try_import(dotted)
        if ok:
            print(f"  {GREEN}OK  {RESET} {dotted}")
        else:
            print(f"  {RED}FAIL{RESET} {dotted}")
            print(f"       {DIM}{reason}{RESET}")
            failures.append((dotted, reason, tb))

    print()
    if not failures:
        print(f"{GREEN}All {len(targets)} stages import cleanly.{RESET} "
              "tracklab should reach execution.")
        return 0

    print(f"{RED}{len(failures)} of {len(targets)} stages failed to import.{RESET}")
    print(f"{YELLOW}These are almost always unpinned transitive dependencies "
          f"that drifted forward.{RESET}")
    print("Fix by adding an upper bound to [project.dependencies] in "
          "pyproject.toml, then:")
    print("    rm -rf .venv && uv venv --clear --python 3.9 .venv "
          "&& uv pip install --python .venv -e .\n")

    if args.verbose:
        for dotted, _, tb in failures:
            if tb:
                print(f"{DIM}{'-' * 70}{RESET}\n{dotted}\n{tb}")

    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
