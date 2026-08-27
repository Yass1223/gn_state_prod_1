#!/usr/bin/env python3
"""stage_data.py -- notebook section 3 (GSR-2024 download) as a resumable script.

Semantics are a faithful port of the section-3 cell -- per-split, idempotent,
verified, corrupt-archive self-healing, nested-layout flattening, encrypted-
archive guidance, empty-split hard stop -- with two Kaggle-driven additions:

  --splits      download ONLY what this session needs. A Kaggle session that
                only evaluates needs `test` alone; re-pulling train+valid's
                ~20 GB every session would burn both time and the ~60 GiB
                scratch disk.
  --delete-zip  remove each archive after its extraction is VERIFIED (default
                on for Kaggle). The Lightning cell keeps zips around, which is
                fine on a Studio but doubles the footprint here: with the full
                train+valid+test set at ~30 GB extracted, keeping the archives
                too would not fit the scratch disk alongside the venv.

Run it under the pipeline venv (network + SoccerNet package needed):
    $PY stage_data.py --root data/gsr --splits train valid
    $PY stage_data.py --root data/gsr --splits test
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile


def seq_dirs(root, split):
    """Sequence folders for a split, located at ANY depth (some archives carry an
    extra top-level directory), returned as the parents of Labels-GameState.json."""
    return sorted(os.path.dirname(p) for p in glob.glob(
        f"{root}/{split}/**/Labels-GameState.json", recursive=True))


def flatten(root, split):
    """Guarantee the layout every later stage globs for: {root}/{split}/<SEQ>/.
    An archive that extracts one level deeper is moved up here rather than
    silently matching nothing."""
    troot, moved = os.path.abspath(f"{root}/{split}"), 0
    for d in seq_dirs(root, split):
        if os.path.abspath(os.path.dirname(d)) != troot:
            dest = os.path.join(troot, os.path.basename(d))
            if not os.path.exists(dest):
                shutil.move(d, dest); moved += 1
    if moved:
        print(f"  [{split}] flattened {moved} sequence folder(s) up one level")
    return moved


def ensure_split(root, split, zipdir, sn_pwd=None, delete_zip=True,
                 downloader_factory=None):
    """Download+extract+verify ONE split; no-op when already present. Returns
    the number of sequence dirs. Raises SystemExit on unrecoverable states,
    exactly like the notebook cell."""
    if seq_dirs(root, split):
        n = len(seq_dirs(root, split))
        print(f"[{split}] already present: {n} sequences -- skipping")
        return n

    z = f"{zipdir}/{split}.zip"
    if os.path.exists(z) and not zipfile.is_zipfile(z):
        print(f"[{split}] archive is corrupt or partial -- deleting so the retry can work")
        os.remove(z)                       # otherwise a partial file blocks every retry

    if not os.path.exists(z):
        print(f"[{split}] downloading (this is the slow step)...")
        if downloader_factory is None:
            from SoccerNet.Downloader import SoccerNetDownloader
            downloader_factory = SoccerNetDownloader
        dl = downloader_factory(LocalDirectory=root)
        dl.downloadDataTask(task="gamestate-2024", split=[split])   # public password default
        if not os.path.exists(z):
            raise SystemExit(
                f"[{split}] download produced no archive at {z}.\n"
                f"  The SoccerNet API reports HTTP errors on stdout rather than raising,"
                f" so scroll up for the cause.\n"
                f"  On Kaggle, also confirm Settings -> Internet is ON (needs a "
                f"phone-verified account), and that 'gamestate-2024' is still the "
                f"correct task name.")

    if not zipfile.is_zipfile(z):
        raise SystemExit(f"[{split}] {z} downloaded but is not a valid zip -- delete it and re-run.")

    print(f"[{split}] extracting {os.path.getsize(z)/1e9:.2f} GB...")
    try:
        zipfile.ZipFile(z).extractall(f"{root}/{split}",
                                      pwd=sn_pwd.encode() if sn_pwd else None)
    except (RuntimeError, NotImplementedError) as e:
        raise SystemExit(
            f"[{split}] archive is encrypted ({e}). Set the SoccerNet NDA password:\n"
            f"  export SN_PWD='...' (or pass --password) then re-run.")
    flatten(root, split)
    if not seq_dirs(root, split):
        raise SystemExit(
            f"[{split}] extracted, but no Labels-GameState.json was found anywhere under "
            f"{root}/{split}.\n  The archive layout is not what the pipeline expects; "
            f"inspect it before continuing.")
    if delete_zip:
        # Only after the extraction VERIFIED above -- a deleted archive plus a
        # broken extraction would force a full re-download.
        os.remove(z)
        print(f"[{split}] archive deleted after verified extraction (disk budget)")
    return len(seq_dirs(root, split))


def main(argv=None, downloader_factory=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gsr")
    ap.add_argument("--splits", nargs="+", required=True,
                    help="e.g. --splits train valid   (or: test / challenge)")
    ap.add_argument("--delete-zip", dest="delete_zip", action="store_true", default=True)
    ap.add_argument("--keep-zip", dest="delete_zip", action="store_false",
                    help="keep archives after extraction (Lightning behaviour)")
    ap.add_argument("--password", default=None,
                    help="SoccerNet NDA password (default: $SN_PWD)")
    a = ap.parse_args(argv)

    sn_pwd = a.password or os.environ.get("SN_PWD")
    zipdir = f"{a.root}/gamestate-2024"
    os.makedirs(a.root, exist_ok=True)

    splits = list(dict.fromkeys(a.splits))          # dedupe, keep order
    counts = {}
    for s in splits:
        counts[s] = ensure_split(a.root, s, zipdir, sn_pwd=sn_pwd,
                                 delete_zip=a.delete_zip,
                                 downloader_factory=downloader_factory)

    # ---- verification gate: every later stage depends on these counts -------
    print()
    for s, n in counts.items():
        print(f"  {s:<9} {n:>3} sequences")
    empty = [s for s, n in counts.items() if n == 0]
    if empty:
        raise SystemExit(
            f"\nSTOP -- no sequences for: {', '.join(empty)}.\n"
            f"Later stages would build empty manifests and every subsequent stage "
            f"would produce nothing without erroring. Resolve the download first.")
    print("\ndataset OK")
    return counts


if __name__ == "__main__":
    main()
