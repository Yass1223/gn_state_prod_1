"""Report what the OSNet-AIN checkpoint contains and which factory can build it.

Answers the two questions the tracker and GTA-Link rewrites depend on:

  1. What is in the checkpoint — backbone name, input size, embedding width, number of
     training identities, number of role classes, tensor count.
  2. Whether the `torchreid` already installed in this environment (VlSomers/bpbreid, a
     dependency of prtreid) or tracklab's bundled `strong_sort.deep.models` can build
     that backbone, which decides whether the model definition has to be vendored.

Also prints the digest of the rezipped archive so it can be pinned in config.

    python scripts/inspect_ain_checkpoint.py
    python scripts/inspect_ain_checkpoint.py --local-path ../osnet_ain/best_ain_full.pth

CPU only; no dataset required. Needs network access on first run unless --local-path is
given, in which case it needs none.
"""
import argparse
import json
import logging
import platform
import sys


def _version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _probe(factory_path, backbone_name):
    """Try to import a model factory and check whether it knows `backbone_name`."""
    module_path, attr = factory_path.rsplit(".", 1)
    result = {"factory": factory_path, "importable": False, "has_backbone": None,
              "osnet_names": [], "error": None}
    try:
        import importlib
        module = importlib.import_module(module_path)
        build_model = getattr(module, attr)
        result["importable"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    # Most torchreid-derived factories expose their registry as a dict of name -> class.
    names = []
    for registry_attr in ("__model_factory", "_model_factory", "model_factory"):
        registry = getattr(module, registry_attr, None)
        if isinstance(registry, dict):
            names = sorted(registry)
            break
    if not names:
        mangled = [v for k, v in vars(module).items()
                   if k.endswith("__model_factory") and isinstance(v, dict)]
        if mangled:
            names = sorted(mangled[0])
    result["osnet_names"] = [n for n in names if "osnet" in n.lower()]

    if names:
        result["has_backbone"] = backbone_name in names
    else:
        # No visible registry: ask the factory directly.
        try:
            build_model(name=backbone_name, num_classes=1, loss="softmax", pretrained=False)
            result["has_backbone"] = True
        except Exception as exc:
            result["has_backbone"] = False
            result["error"] = f"build failed: {type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, help="Hub repository id")
    parser.add_argument("--revision", default=None, help="Hub commit to pin")
    parser.add_argument("--file", default=None,
                        help="packed checkpoint filename in the repo "
                             "(default: the module's FILENAME; pass '' to force the "
                             "exploded-snapshot path)")
    parser.add_argument("--local-path", default=None,
                        help="read the checkpoint from disk instead of the Hub; accepts "
                             "an archive file or the exploded directory")
    parser.add_argument("--cache-dir", default=None, help="where to write the rezipped archive")
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from sn_gamestate.reid import osnet_ain
    except ImportError as exc:
        print(f"cannot import sn_gamestate.reid.osnet_ain: {exc}\n"
              f"run this with the project environment, e.g. "
              f"uv run python scripts/inspect_ain_checkpoint.py", file=sys.stderr)
        return 2

    repo = args.repo or osnet_ain.REPO_ID
    revision = args.revision or osnet_ain.REVISION
    filename = getattr(args, "file", None) or getattr(osnet_ain, "FILENAME", None)

    import torch

    report = {
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in
                     ("torch", "numpy", "huggingface_hub", "torchreid", "tracklab")},
        "repo": repo,
        "revision": revision,
    }
    print(f"python {report['python']} | " +
          " | ".join(f"{k} {v}" for k, v in report["packages"].items() if v))

    path = osnet_ain.resolve_checkpoint(repo, revision, cache_dir=args.cache_dir,
                                        local_path=args.local_path, filename=filename)
    report["source"] = args.local_path or f"{repo}@{revision[:12]}"
    report["archive"] = {"path": str(path), "bytes": path.stat().st_size,
                         "sha256": osnet_ain.sha256(path)}
    print(f"\nsource   {report['source']}\n"
          f"archive  {path}\n"
          f"  {report['archive']['bytes'] / 1e6:.1f} MB  sha256 {report['archive']['sha256']}")

    checkpoint = osnet_ain.load_checkpoint(repo, revision, cache_dir=args.cache_dir,
                                           local_path=args.local_path, filename=filename)

    cfg = checkpoint["cfg"]
    ema = checkpoint["ema"]
    report["top_level_keys"] = sorted(checkpoint)
    report["cfg"] = {k: (list(v) if isinstance(v, (tuple, list)) else v)
                     for k, v in cfg.items()} if isinstance(cfg, dict) else repr(cfg)
    report["epoch"] = int(checkpoint["epoch"])
    report["train_identities"] = len(checkpoint["pid2idx"])
    report["ema_tensors"] = len(ema)
    report["ema_parameters"] = int(sum(t.numel() for t in ema.values()
                                       if hasattr(t, "numel")))
    report["dtypes"] = sorted({str(t.dtype) for t in ema.values() if torch.is_tensor(t)})

    prefixes = {}
    for key in ema:
        prefixes[key.split(".", 1)[0]] = prefixes.get(key.split(".", 1)[0], 0) + 1
    report["ema_prefixes"] = dict(sorted(prefixes.items(), key=lambda kv: -kv[1]))

    shapes = {name: list(ema[name].shape) for name in
              ("bnneck.weight", "classifier.weight", "role_head.weight", "role_head.bias")
              if name in ema}
    report["head_shapes"] = shapes

    print(f"\ntop-level keys  {report['top_level_keys']}")
    print(f"cfg             {report['cfg']}")
    print(f"epoch           {report['epoch']}")
    print(f"identities      {report['train_identities']:,}")
    print(f"ema             {report['ema_tensors']} tensors, "
          f"{report['ema_parameters']:,} parameters, dtypes {report['dtypes']}")
    print(f"key prefixes    {report['ema_prefixes']}")
    print(f"head shapes     {shapes if shapes else 'none of the expected head keys found'}")

    if "bnneck.weight" in shapes:
        print(f"  embedding width  {shapes['bnneck.weight'][0]}")
    if "role_head.weight" in shapes:
        print(f"  role classes     {shapes['role_head.weight'][0]}")
    if "classifier.weight" in shapes:
        print(f"  classifier       {shapes['classifier.weight'][0]} ids "
              f"x {shapes['classifier.weight'][1]}")

    backbone = cfg.get("BACKBONE") if isinstance(cfg, dict) else None
    report["backbone"] = backbone
    print(f"\nbackbone named in the checkpoint: {backbone!r}")
    report["factories"] = []
    if backbone:
        for factory in ("torchreid.models.build_model",
                        "strong_sort.deep.models.build_model"):
            probe = _probe(factory, backbone)
            report["factories"].append(probe)
            if not probe["importable"]:
                print(f"  {factory}: not importable ({probe['error']})")
            else:
                verdict = {True: "yes", False: "no", None: "unknown"}[probe["has_backbone"]]
                print(f"  {factory}: builds {backbone!r}? {verdict}")
                if probe["osnet_names"]:
                    print(f"    osnet names available: {probe['osnet_names']}")
                if probe["error"]:
                    print(f"    {probe['error']}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
