#!/usr/bin/env python3
"""fetch_weights.py -- the HuggingFace checkpoints this build needs, hash-checked.

    DBNet++      Ynniss/dbnetppp_jn               111 MB   single .pth   FETCHED
    ResNet-34    Ynniss/Legibility_classifier     85.3 MB  single .pth   FETCHED
    SATRN        Ynniss/satrn_small               47.7 MB  .zip = torch   FETCHED
                 (second recogniser; staged as recog2/*.pth, see below)
    ResNet-18    Ynniss/Resnet18_multi_single_player 44.8 MB EXPLODED    RETIRED

The ResNet-18 single/multi frame filter measured <= no-op in every ablation
cell and is no longer part of the configuration, so main() fetches THREE
files. fetch_classifier() below is kept because it is the only implementation
of the exploded-archive reassembly, and --clf-attached still reaches it.

THE SATRN FILE IS NAMED .zip BUT IS NOT AN ARCHIVE OF FILES. Like the PARSeq
download, it is the torch.save container itself (entries archive/data.pkl,
archive/data/N, archive/version), so it is staged by COPY under a .pth name,
never extracted; fetch_single refuses a download that is a real archive of
files. The checkpoint was saved by mmengine and carries its own training
config; after staging, that text is written next to it as
recog2/config_from_checkpoint.py (mmocr_reader.config_from_checkpoint) so the
reader has a config even though the download ships none. The config names the
dictionary by an absolute path from the training machine; mmocr_reader
resolves it by basename inside the installed mmocr and checks the resulting
size against the checkpoint's classifier width at load.

Its published sha256 is not asserted in full: the value on record is the
16-hex-character prefix observed by the audited fusion run of 2026-08-17
(audit check recog.recog2.sha), so the download is checked against that
prefix and the full digest is written to the provenance file. Replace
`sha256_prefix` with a full `sha256` once one has been read off a download.

PARSeq is NOT here: stage_weights.py fetches it, because it also has to gate the
strhub Lightning load, which is a different kind of check.

TWO SHAPES OF REPOSITORY. Two of these are ordinary single-file repos and are
fetched with hf_hub_download plus a sha256 comparison. The ResNet-18 repo is a
torch.save zip archive uploaded EXPLODED into its 128 member files, so there is
no single file to download or hash. crop_classifier.resolve_weights snapshot-
downloads it and reassembles the archive; provenance is pinned on data.pkl's
sha256, which fixes every key name, dtype and shape in the checkpoint. The
reassembled .pt is deliberately NOT hashed -- zip records carry mtimes, so it is
not byte-reproducible and a hash of it would fail for no reason.

A MISMATCH IS REPORTED, NOT FATAL. Re-uploading a private fine-tune under the
same path is legitimate; quietly claiming a provenance you do not have is not.
Every mismatch is printed and written into the provenance record.

    $PY fetch_weights.py --out-dir models
    $PY fetch_weights.py --out-dir models --dbnet-attached /kaggle/input/x/y.pth
"""
import argparse
import hashlib
import json
import os
import sys

# repo, filename, sha256, approximate size, destination, description
SOURCES = {
    "dbnet": {
        "repo": "Ynniss/dbnetppp_jn",
        "file": "best_icdar_hmean_epoch_10.pth",
        "sha256": "1e8ee32969a0264dd8d26918a3dea7ab9914c90033b087b1bf97c5eefecfe6c9",
        "size_mb": 111,
        "out": "best_icdar_hmean_epoch_10.pth",
        "what": "DBNet++ text detector (number localisation)",
    },
    "legibility": {
        "repo": "Ynniss/Legibility_classifier",
        "file": "legibility_resnet34_soccer_20240215.pth",
        "sha256": "b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41",
        "size_mb": 85.3,
        "out": "sn_legibility.pth",
        "what": "Koshkina ResNet-34 legibility classifier (SoccerNet fine-tuned)",
    },
    "satrn": {
        "repo": "Ynniss/satrn_small",
        "file": "best_recog_word_acc_epoch_10.zip",
        # observed, not published: prefix from the audited run (see header)
        "sha256_prefix": "9e8f73b300754c35",
        "size_mb": 47.7,
        "out": "recog2/best_recog_word_acc_epoch_10.pth",
        "torch_container": True,   # the .zip IS the checkpoint; copy, never unzip
        "what": "SATRN small (mmocr 1.x) second jersey-number recogniser",
    },
}

# Exploded-archive repo: no single file, so it has its own fetch path.
CLASSIFIER = {
    "repo": "Ynniss/Resnet18_multi_single_player",
    "out": "resnet18_multi_single.pt",
    "size_mb": 44.8,
    "what": "ResNet-18 single/multi player frame filter",
}

# For reference only -- stage_weights.py fetches and gates this one. The
# upstream file is named .zip but IS the torch.save checkpoint (see
# stage_weights.py); it is staged by rename, not extraction.
PARSEQ = {
    "repo": "Ynniss/final_parseq_jn",
    "file": "parseq_gsr_ft_s1.zip",
    "sha256": "22d936444e09b0358b5b7339c2971ab5e792fee9d53dd30a98917abcd3ee1887",
    "size_mb": 286,
    "staged_as": "parseq_gsr_ft_s1.ckpt",
}


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def classify(path):
    """'torch' (torch.save zip container), 'archive' (zip of files), or 'raw'."""
    import zipfile
    if not zipfile.is_zipfile(path):
        return "raw"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    base = {n.rsplit("/", 1)[-1] for n in names}
    if "data.pkl" in base and "version" in base:
        return "torch"
    return "archive"


def write_config_from_checkpoint(pth):
    """For an mmengine checkpoint: write the config it carries next to it as
    config_from_checkpoint.py. -> path | None (a checkpoint without one is
    not an error here; mmocr_reader reports it at load)."""
    try:                       # numpy>=2 pickles under this numpy<2 venv
        import crop_classifier as CC
        CC.numpy2_pickle_compat()
    except Exception:
        pass
    from mmocr_reader import config_from_checkpoint
    out = os.path.join(os.path.dirname(pth) or ".", "config_from_checkpoint.py")
    return config_from_checkpoint(pth, out)


def fetch_single(key, out_dir, attached=None, refresh=False):
    """An ordinary one-file HuggingFace repo."""
    spec = SOURCES[key]
    dest = os.path.join(out_dir, spec["out"])
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    if refresh and os.path.exists(dest):
        # A re-upload under the SAME repo/filename is invisible to both caches:
        # the local copy short-circuits, and huggingface_hub may serve a stale
        # blob. --refresh forces both to be re-checked.
        os.remove(dest)

    if attached and os.path.exists(attached):
        import shutil
        if os.path.abspath(attached) != os.path.abspath(dest):
            shutil.copyfile(attached, dest)
        source = f"attached: {attached}"
    elif os.path.exists(dest):
        source = "already-present"
    else:
        print(f"[weights] {key}: downloading ~{spec['size_mb']} MB from "
              f"{spec['repo']}...")
        import shutil

        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(spec["repo"], spec["file"],
                                force_download=refresh)
        except Exception as e:
            raise SystemExit(
                f"[weights] {key}: could not fetch {spec['repo']}/"
                f"{spec['file']}: {type(e).__name__}: {e}\n"
                f"  * Internet must be ON in the Kaggle notebook settings.\n"
                f"  * If the repo is private: huggingface-cli login "
                f"(or set $HF_TOKEN).\n"
                f"  * Or attach the file as a Kaggle dataset and pass "
                f"--{key}-attached <path>.") from e
        shutil.copyfile(p, dest)
        source = f"huggingface: {spec['repo']}/{spec['file']}"

    size = os.path.getsize(dest)
    digest = sha256_of(dest)
    if "sha256" in spec:
        expect, match = spec["sha256"], digest == spec["sha256"]
        how = "sha256"
    else:                                   # prefix on record only (SATRN)
        expect = spec["sha256_prefix"]
        match = digest.startswith(expect)
        how = f"sha256 prefix[{len(expect)}]"
    print(f"[weights] {key}: {dest}  {size / 1e6:.1f} MB  "
          f"sha256 {digest[:16]}...  {'OK' if match else 'MISMATCH'} ({how})")
    if not match:
        print(f"[weights]   WARNING: expected {how} {expect[:16]}... -- this "
              f"is a DIFFERENT file. Not an error (a private re-upload is "
              f"legitimate), but every result carries this flag.")
    rec = {"key": key, "path": dest, "source": source, "sha256": digest,
           "expected": expect, "expected_kind": how, "sha256_matches": match,
           "bytes": size, "repo": spec["repo"], "what": spec["what"]}
    if spec.get("torch_container"):
        kind = classify(dest)
        rec["container"] = kind
        if kind == "archive":
            raise SystemExit(
                f"[weights] {key}: {dest} is a zip archive of files, not a "
                f"torch.save container. This build expects the published "
                f"file to BE the checkpoint (as it was on 2026-08-17); "
                f"inspect the download and stage the .pth inside it by hand.")
        if kind == "raw":
            print(f"[weights]   NOTE: not a zip container (legacy torch "
                  f"pickle?) -- staged as-is, the load gate will judge it.")
        cfg = write_config_from_checkpoint(dest)
        rec["config_from_checkpoint"] = cfg
        print(f"[weights]   container={kind}  config -> "
              f"{cfg or 'NONE inside the checkpoint'}")
    return rec


def fetch_classifier(out_dir, attached=None, refresh=False):
    """The exploded-archive repo. Delegates to the loader that understands it,
    so there is exactly ONE implementation of the reassembly."""
    import crop_classifier as CC
    dest = os.path.join(out_dir, CLASSIFIER["out"])
    if refresh and os.path.exists(dest):
        os.remove(dest)
    if not (attached or os.path.exists(dest)):
        print(f"[weights] classifier: downloading ~{CLASSIFIER['size_mb']} MB "
              f"from {CLASSIFIER['repo']} (128 files, then reassembled)...")
    path, source, info = CC.resolve_weights(attached, dest,
                                            repo=CLASSIFIER["repo"])
    size = os.path.getsize(path)
    print(f"[weights] classifier: {path}  {size / 1e6:.1f} MB  ({source})")
    if info.get("data_pkl_sha256"):
        print(f"[weights]   data.pkl sha256 {info['data_pkl_sha256'][:16]}...  "
              f"{'OK' if info.get('data_pkl_matches') else 'MISMATCH'}  "
              f"({info.get('n_storages')} tensor storages)")
    return {"key": "classifier", "path": path, "source": source,
            "bytes": size, "repo": CLASSIFIER["repo"],
            "what": CLASSIFIER["what"], **info}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--dbnet-attached", default=None,
                    help="skip the 111 MB download (Kaggle dataset path)")
    ap.add_argument("--legibility-attached", default=None)
    ap.add_argument("--satrn-attached", default=None,
                    help="skip the 47.7 MB download (the published .zip or "
                         "an already-staged .pth)")
    ap.add_argument("--clf-attached", default=None,
                    help="an already-reassembled ResNet-18 .pt")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if a local copy exists")
    # NOT weights_provenance.json: stage_weights.py writes THAT file (the
    # PARSeq record scripts/verify_run_integrity.py reads) into the same
    # directory, and it ran after this script, so this record was overwritten.
    ap.add_argument("--provenance", default=None,
                    help="default: <out-dir>/fetch_weights_provenance.json")
    a = ap.parse_args(argv)

    # final config: the ResNet-18 single/multi filter is retired (measured
    # <= no-op in every ablation cell), so its weights are no longer fetched.
    recs = [fetch_single("dbnet", a.out_dir, a.dbnet_attached, a.refresh),
            fetch_single("legibility", a.out_dir, a.legibility_attached,
                         a.refresh),
            fetch_single("satrn", a.out_dir, a.satrn_attached, a.refresh)]

    out = a.provenance or os.path.join(a.out_dir,
                                       "fetch_weights_provenance.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"weights": recs, "parseq_note": PARSEQ}, f, indent=2)
    print(f"\n[weights] provenance -> {out}")

    bad = [r["key"] for r in recs if r.get("sha256_matches") is False
           or r.get("data_pkl_matches") is False]
    if bad:
        print(f"[weights] NOTE: {bad} did not match the published hash. "
              f"Proceeding; the flag travels with the results.")
    print(f"[weights] {len(recs)} checkpoints present "
          f"({', '.join(r['key'] for r in recs)}). "
          f"PARSeq: run stage_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
