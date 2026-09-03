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

import os

# A hosted-notebook kernel (Kaggle, Jupyter) exports MPLBACKEND=module://matplotlib_inline.
# backend_inline. matplotlib rejects that value in a plain subprocess, and 16 of the 21
# stages import matplotlib.pyplot through tracklab.visualization, so every one of them
# would fail here for a reason that has nothing to do with the pipeline. Force a headless
# backend before any stage is imported.
if "inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"

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
    ap.add_argument("--skip-backbone", action="store_true",
                    help="skip building the osnet_ain_x1_0 backbone (torch import, "
                         "a few seconds; no download)")
    args = ap.parse_args(argv)

    targets = discover_targets()
    # Two runtime dependencies of the `track` stage live OUTSIDE the config graph
    # and deserve the same seconds-not-minutes failure: boxmot (installed with
    # --no-deps, so nothing else vouches for it - see pyproject.toml) and the
    # shared OSNet-AIN embedder module both `track` and `split_merge` import. The
    # split_merge algorithm module is listed too: the stage imports it, no config does.
    targets += [
        "boxmot.trackers.botsort.botsort.BotSort",
        "boxmot.motion.cmc.get_cmc_method",
        "sn_gamestate.reid.osnet_ain.from_config",
        "sn_gamestate.track.split_merge.split_merge_video",
        "sn_gamestate.reid.osnet_team.from_config",
        "sn_gamestate.team.rules.run_sequence",
        "sn_gamestate.refine.traj_refine.refine_video",
    ]
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

    # The osnet_ain_x1_0 backbone must be BUILDABLE, not just importable: the
    # torchreid in this environment is the VlSomers/bpbreid fork, and whether its
    # model factory carries osnet_ain decides whether the appearance model can
    # exist at all. No network, no GPU - the factory builds from code alone.
    if not args.skip_backbone and not any(
            d.startswith("sn_gamestate.reid.osnet_ain") for d, _, _ in failures):
        from sn_gamestate.reid.osnet_ain import build_backbone
        # osnet_x1_0 is the team-appearance backbone (sn_gamestate/reid/osnet_team.py);
        # the same factory must carry both, or the role_team stage cannot embed a crop.
        for name in ("osnet_ain_x1_0", "osnet_x1_0"):
            try:
                _, factory = build_backbone(name)
                print(f"  {GREEN}OK  {RESET} {name} buildable via {factory}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {RED}FAIL{RESET} {name} backbone build")
                print(f"       {DIM}{type(exc).__name__}: {exc}{RESET}")
                failures.append((f"{name} backbone", str(exc),
                                 traceback.format_exc()))

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
