#!/usr/bin/env python3
"""stage_weights.py -- resolve the PARSeq recognition checkpoint, verifiably.

One canonical output (default models/parseq_gsr_ft_s1.ckpt), produced by the
first source that works:

  1. --attached <path>   a file the user attached as a Kaggle dataset (fastest,
                         reproducible, immune to outages) -- preferred;
  2. HF  Ynniss/final_parseq_jn  'parseq_gsr_ft_s1.zip'.

There is no second mirror and no Drive fallback for this checkpoint: the
Ynniss/final_parseq_jn repository holds this file and .gitattributes and
nothing else. The old Drive id served the SUPERSEDED Koshkina checkpoint and
is gone from the chain rather than left to hand back the wrong weights.

THE UPSTREAM FILE IS NAMED .zip BUT IS NOT AN ARCHIVE. Its interior is
archive/data.pkl + archive/data/0..703 + archive/version -- the layout
torch.save writes. It needs a RENAME, not an unzip, which is exactly what
src_hf's shutil.copyfile onto --out does. Do not add an extraction step.

THE STAGED FILENAME MATTERS. strhub's load_from_checkpoint picks the model
class by substring-matching the checkpoint PATH against, in order, 'abinet',
'crnn', 'parseq', 'trba', 'trbc', 'vitstr' -- so the resolved absolute path
must contain 'parseq' and none of the others, in the filename OR in any parent
directory. The load gate below asserts the class it actually got, which is
what turns a misroute into an immediate, legible failure.

After ANY source wins:
  * sha256 is computed and compared against the published upstream hash --
    reported, NOT fatal on mismatch (a mismatch means the user supplied a
    legitimately different checkpoint, e.g. a private fine-tune; the run must
    say so, not refuse). audit_parseq.py is the arm that asserts;
  * a LOAD GATE: the file must load through strhub's loader (the exact code
    path run_eval.build_models uses) AND come back as a PARSeq instance. A
    file that fails here would have crashed the pipeline hours later
    mid-stage -- fail now instead;
  * the checkpoint's own internal metadata (epoch, global_step, whether
    optimizer state is present, PARSeq hparams) is read FROM THE FILE and
    written to weights_provenance.json next to the output. Filenames lie;
    the file's contents are the ground truth about what is being loaded.

    $PY stage_weights.py --attached "/kaggle/input/mydata/parseq_gsr_ft_s1.zip" \\
        --out models/parseq_gsr_ft_s1.ckpt
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

# Before ANY import that can reach matplotlib (strhub -> pytorch_lightning ->
# torchmetrics -> matplotlib). Kaggle presets MPLBACKEND to its inline backend,
# whose module is absent from this venv; matplotlib raises at import time.
os.environ["MPLBACKEND"] = "Agg"

# Published by Ynniss/final_parseq_jn. Single source; no mirror.
UPSTREAM_SHA256 = "22d936444e09b0358b5b7339c2971ab5e792fee9d53dd30a98917abcd3ee1887"
UPSTREAM_NAME = "parseq_gsr_ft_s1.zip"
HF_A = ("Ynniss/final_parseq_jn", UPSTREAM_NAME)


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ sources --
def src_attached(a):
    if not a.attached:
        raise RuntimeError("no --attached path given")
    if not os.path.exists(a.attached):
        raise RuntimeError(f"attached path does not exist: {a.attached}")
    shutil.copyfile(a.attached, a.out)
    return f"attached: {a.attached}"


def src_hf(repo, fname):
    def fetch(a):
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, fname)
        # RENAME, not unzip: the upstream .zip IS the torch.save file.
        shutil.copyfile(p, a.out)
        return f"huggingface: {repo}/{fname}"
    return fetch


# ---------------------------------------------------------------- load gate --
def inspect_and_gate(path):
    """Load through the SAME strhub path the pipeline uses, then read the
    checkpoint's own metadata. Everything here is best-effort-defensive:
    metadata fields vary by Lightning version, but the LOAD must succeed."""
    from strhub.models.utils import load_from_checkpoint
    model = load_from_checkpoint(path)            # raises on a broken/wrong file
    # load_from_checkpoint routes on the PATH string, not the file contents, so
    # a path containing 'abinet'/'crnn'/'trba'/'trbc'/'vitstr' anywhere -- a
    # parent directory counts -- silently builds the wrong class and fails much
    # later with a confusing shape error. Catch it here, at the source.
    got = type(model).__name__
    if got != "PARSeq":
        raise RuntimeError(
            f"strhub built a {got}, not a PARSeq, from {path!r}. "
            f"load_from_checkpoint selects the model class by substring-"
            f"matching the checkpoint PATH ('abinet', 'crnn', 'parseq', "
            f"'trba', 'trbc', 'vitstr', in that order) -- the resolved "
            f"absolute path must contain 'parseq' and none of the others, "
            f"including in every parent directory.")
    meta = {"model_class": got}
    try:
        import torch
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict):
            meta["epoch"] = raw.get("epoch")
            meta["global_step"] = raw.get("global_step")
            meta["has_optimizer_state"] = bool(raw.get("optimizer_states"))
            hp = raw.get("hyper_parameters") or {}
            meta["hparams"] = {k: hp[k] for k in
                               ("charset_train", "img_size", "max_label_length")
                               if k in hp}
    except Exception as e:                        # metadata is optional; load isn't
        meta["metadata_note"] = f"unreadable ({type(e).__name__}: {e})"
    del model
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attached", default="",
                    help="path to a user-attached checkpoint (tried first)")
    ap.add_argument("--out", default="models/parseq_gsr_ft_s1.ckpt")
    ap.add_argument("--skip-load-gate", action="store_true",
                    help="hash + provenance only (used by offline tests where "
                         "the stub loader would accept any bytes anyway)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if os.path.exists(a.out):
        print(f"[weights] {a.out} already present -- verifying, not re-fetching")
        source = "already-present"
    else:
        chain = [("attached dataset", src_attached),
                 (f"HF {HF_A[0]}", src_hf(*HF_A))]
        source, errors = None, []
        for name, fn in chain:
            try:
                print(f"[weights] trying {name} ...")
                source = fn(a)
                print(f"[weights] OK from {name}")
                break
            except Exception as e:
                msg = f"{name}: {type(e).__name__}: {e}"
                errors.append(msg)
                print(f"[weights] {msg}")
        if source is None:
            sys.exit("[weights] every source failed:\n  " + "\n  ".join(errors) +
                     "\n  Attach the checkpoint as a Kaggle dataset "
                     "(Add Input) and re-run.")

    digest = sha256_of(a.out)
    size = os.path.getsize(a.out)
    match = digest == UPSTREAM_SHA256
    print(f"[weights] sha256 {digest}  ({size/1e6:.1f} MB)")
    if match:
        print(f"[weights] MATCHES the published upstream checkpoint "
              f"({UPSTREAM_NAME})")
    else:
        print(f"[weights] does NOT match the published upstream hash "
              f"{UPSTREAM_SHA256[:12]}... -- this is a DIFFERENT checkpoint. "
              f"Not an error (a private fine-tune is legitimate), but the "
              f"provenance record will say so and results carry it.")

    meta = {}
    if a.skip_load_gate:
        print("[weights] load gate SKIPPED (--skip-load-gate)")
    else:
        meta = inspect_and_gate(a.out)
        print(f"[weights] load gate PASSED; checkpoint metadata: {meta}")

    prov = {"path": a.out, "source": source, "sha256": digest, "bytes": size,
            "matches_upstream": match, "upstream_name": UPSTREAM_NAME,
            "upstream_sha256": UPSTREAM_SHA256, **meta}
    pp = os.path.join(os.path.dirname(a.out) or ".", "weights_provenance.json")
    with open(pp, "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[weights] provenance -> {pp}")


if __name__ == "__main__":
    main()
