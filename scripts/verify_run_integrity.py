#!/usr/bin/env python
"""Refuse to trust metrics from a run in which any stage silently failed.

WHY THIS EXISTS
---------------
Neither the BroadTrack calibration stage nor the jersey stage aborts the run when
its subprocess fails: both log an ERROR and return, and the pipeline continues to
completion. A run can therefore finish, write an evaluation table, and report
numbers that were computed with NO camera calibration (so no pitch coordinates)
or NO jersey numbers. Those numbers look plausible and are wrong.

This gate inspects a finished run and exits non-zero if anything silently failed.
It is read-only: it never modifies the pipeline or its outputs.

    python scripts/verify_run_integrity.py                      # newest eval log
    python scripts/verify_run_integrity.py --log eval_results/eval_test.log
    python scripts/verify_run_integrity.py --expect-sequences 49

Exit code 0 = every stage produced its artifacts and logged no failure.
Anything else = DO NOT REPORT THESE METRICS.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "broadtrack_calib"
JN_CACHE = ROOT / "jn_cache"
PROVENANCE = ROOT / "plugins" / "jn_gsr" / "models" / "weights_provenance.json"

RED, GREEN, YELLOW, BOLD, RESET = (
    "\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[1m", "\033[0m"
)

# Signatures that mean a stage failed but the run continued anyway.
FAILURE_PATTERNS = [
    (r"\[BroadTrack\] binary failed", "BroadTrack calibration subprocess failed"),
    (r"unrecognised option", "BroadTrack rejected its command line"),
    (r"\[jn_gsr\] worker \d+ FAILED", "jersey-number worker failed"),
    (r"ModuleNotFoundError", "a module was missing at runtime"),
    (r"Tracker .* was unable to be evaluated", "TrackEval could not evaluate"),
    (r"\[BroadTrack\].*fall(ing)? back", "BroadTrack silently fell back"),
]


def newest_log() -> Path | None:
    """Most recent evaluation log, from either place a run can leave one.

    ``eval_results/eval_*.log`` is what scripts/lightning_eval.sh writes. A plain
    ``tracklab -cn soccernet`` writes nothing there: soccernet.yaml sets
    ``hydra.job.chdir: True`` with
    ``hydra.run.dir: outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}``,
    so the log lands at ``outputs/sn-gamestate/<date>/<time>/main.log``. Looking
    only in the first location made this gate report "no evaluation log found"
    and exit 1 for a run that had evaluated perfectly well.
    """
    candidates = list((ROOT / "eval_results").glob("eval_*.log"))
    candidates += list((ROOT / "outputs").glob("sn-gamestate/*/*/main.log"))
    logs = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--expect-sequences", type=int, default=None,
                    help="how many sequences the run should have covered (test=49)")
    args = ap.parse_args(argv)

    problems: list[str] = []
    notes: list[str] = []

    log = args.log or newest_log()
    if log is None or not log.is_file():
        problems.append("no evaluation log found - cannot verify the run at all")
    else:
        print(f"{BOLD}log:{RESET} {log}")
        text = log.read_text(encoding="utf-8", errors="replace")
        for pattern, desc in FAILURE_PATTERNS:
            hits = len(re.findall(pattern, text))
            if hits:
                problems.append(f"{desc}  ({hits} occurrence(s) matching /{pattern}/)")
        if "HOTA" not in text:
            notes.append("no HOTA table in the log - evaluation may not have finished")

    # Calibration artifacts: one JSON per sequence, and it must contain frames.
    calibs = sorted(CALIB_DIR.glob("*.json")) if CALIB_DIR.is_dir() else []
    if not calibs:
        problems.append(f"no calibration JSON in {CALIB_DIR} - "
                        "every detection lacks pitch coordinates")
    else:
        empty = []
        for c in calibs:
            try:
                data = json.loads(c.read_text())
                if not data:
                    empty.append(c.name)
            except Exception as exc:
                empty.append(f"{c.name} (unreadable: {exc})")
        print(f"{BOLD}calibration:{RESET} {len(calibs)} JSON file(s)")
        if empty:
            problems.append(f"empty/unreadable calibration JSON: {', '.join(empty[:5])}")
        if args.expect_sequences and len(calibs) < args.expect_sequences:
            problems.append(f"calibration covers {len(calibs)} sequence(s), "
                            f"expected {args.expect_sequences}")

    # Jersey artifacts.
    jn = sorted(JN_CACHE.glob("*")) if JN_CACHE.is_dir() else []
    if not jn:
        problems.append(f"no jersey output in {JN_CACHE} - "
                        "jersey numbers are absent from the results")
    else:
        print(f"{BOLD}jersey:{RESET} {len(jn)} cache entr(ies)")
        if args.expect_sequences and len(jn) < args.expect_sequences:
            notes.append(f"jersey cache covers {len(jn)} sequence(s), "
                         f"expected {args.expect_sequences}")

    # Weights provenance: stage_weights.py records whether the PARSeq checkpoint
    # matches the published upstream hash, but treats a mismatch as non-fatal
    # (a private fine-tune is legitimate). Nothing downstream reads that record,
    # so surface it here - metrics from a substituted checkpoint are not
    # comparable to published numbers.
    if PROVENANCE.is_file():
        try:
            prov = json.loads(PROVENANCE.read_text())
            if prov.get("matches_upstream") is True:
                print(f"{BOLD}weights:{RESET} PARSeq checkpoint matches upstream")
            else:
                problems.append(
                    "PARSeq checkpoint does NOT match the published upstream hash "
                    f"(source: {prov.get('source', 'unknown')}) - metrics are not "
                    "comparable to published results")
        except Exception as exc:
            notes.append(f"could not read {PROVENANCE.name}: {exc}")
    else:
        notes.append("no weights_provenance.json - PARSeq checkpoint origin unverified")

    print()
    for n in notes:
        print(f"  {YELLOW}note{RESET}  {n}")
    if problems:
        print(f"\n{RED}{BOLD}RUN INTEGRITY: FAILED{RESET}")
        for p in problems:
            print(f"  {RED}x{RESET} {p}")
        print(f"\n{RED}Do NOT report these metrics.{RESET} A stage failed silently and "
              "the pipeline continued;\nthe resulting numbers were computed from "
              "incomplete data.")
        return 1

    print(f"{GREEN}{BOLD}RUN INTEGRITY: OK{RESET} - every stage produced its artifacts "
          "and no silent failure was logged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
