"""OSNet-AIN appearance model, shared by the tracker and by the split_merge stage.

Checkpoint
----------
Sourced by default from `Ynniss/osnet_ain` as the packed single file
`best_ain_full.zip` — which is itself a `torch.save` archive (torch archives are zip
files), so `torch.load` reads the download directly. It is fetched with
`hf_hub_download`, digest-verified against the pinned sha256, and never rezipped.
The same weights also exist exploded (`Ynniss/osnet_ain_ckp`: `data.pkl` and the
`data/<n>` tensor storages as individual repository files); the two were proven
bitwise-identical tensor-by-tensor. The exploded form is kept as a fallback: with no
`ain_file`, a snapshot is downloaded and deterministically rezipped under one prefix
before loading. A local copy (`ain_local_path`) may be either form.

`torch.load` cannot read the exploded form. Measured against a reference `torch.save`
archive:

    torch.load(<snapshot directory>)      IsADirectoryError
    rezipped with members at the root     RuntimeError: [enforce fail at
                                          inline_container.cc:180] file in archive is
                                          not in a subdirectory: data.pkl
    rezipped under a single prefix        loads

so the snapshot is rezipped under one prefix directory before loading. The rezip is
deterministic — sorted member order, stored rather than deflated, fixed timestamps — so
the archive is a byte-reproducible function of the snapshot and can be pinned by digest
exactly as a single-file checkpoint would be.

The pickle references `numpy._core.multiarray._reconstruct`, the NumPy 2.x module path.
NumPy 1.26.4 ships `numpy/_core/multiarray.py` as a stub for precisely this case,
re-exporting `numpy.core.multiarray` (including `_reconstruct`), so the checkpoint
unpickles under the pinned NumPy without a compatibility shim.

Network
-------
`backbone -> GAP -> backbone.fc -> BNNeck -> {classifier, role_head}`, with the
embedding taken as the L2-normalised BNNeck output. Every width is read from the
checkpoint's own tensors — embedding size from `bnneck.weight`, identity count from
`classifier.weight`, role count from `role_head.weight` — and the backbone name and
input size from `cfg`, so nothing here has to agree with a hardcoded constant. Weights
load with `strict=False` and any missing tensor is fatal: a partially initialised
appearance model produces plausible embeddings and silently degrades association.

Read from the checkpoint's `data.pkl`, for the record rather than as constants:
`BACKBONE` is `osnet_ain_x1_0` and `INPUT_HW` is (256, 128); `ema` holds 560 tensors,
552 of them under `backbone.`, plus `bnneck` (512), `classifier.weight` (2205 x 512,
no bias) and `role_head` (4 x 512). Building `osnet_ain_x1_0` from the torchreid fork
this project installs, wrapped in the network below and sized from those tensors,
reproduces that key set exactly — 560 against 560, nothing missing, nothing unexpected,
no shape disagreement.

The archive also carries `model`, `opt`, `sched`, `scaler`, `rng` and `history`, which
is why it is 67 MB. Only `ema` is used, matching the reference pipeline.

The role head is built because the checkpoint carries it and the load is strict about
missing tensors, but its output is unused: in this pipeline role and team come from
the `team_embed` + `role_team` stages downstream (sn_gamestate/team).
"""
import contextlib
import hashlib
import importlib
import inspect
import logging
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

REPO_ID = "Ynniss/osnet_ain"
FILENAME = "best_ain_full.zip"
REVISION = "d78f65ded828f0cbd8dff2a06dcbff4fc6835dfe"
SHA256 = "a0a7e42676edad0cbc3a4ba0f7d0f8ded75612443ef7a9eabb57cb9e5e245293"

# Members a packed download may carry if it is a plain zip *around* a checkpoint
# rather than a torch archive itself (torch archives always contain a data.pkl).
CKPT_SUFFIXES = (".pth", ".pt", ".ckpt", ".bin", ".pth.tar", ".safetensors")

# Keys the training script writes. Asserted on load so a wrong or truncated artefact
# fails here rather than producing a partly initialised network further down.
REQUIRED_KEYS = ("cfg", "ema", "pid2idx", "epoch")

# Repository metadata that is not part of the torch archive. Hub snapshots may also
# contain a `.cache/` directory of download bookkeeping when `local_dir` is used.
_SKIP_NAMES = frozenset({".gitattributes", ".gitignore", "README.md"})
_SKIP_DIRS = frozenset({".cache", ".git"})

# Any name works; torch only requires that every member share one prefix directory.
_ARCHIVE_PREFIX = "archive"

# DOS epoch. Fixed so the rezip does not vary with download mtimes.
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------- checkpoint

def _members(snapshot_dir: Path):
    """Torch archive members under `snapshot_dir`, as (arcname, path), sorted."""
    out = []
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_dir)
        if rel.name in _SKIP_NAMES or _SKIP_DIRS.intersection(rel.parts):
            continue
        out.append((rel.as_posix(), path))
    out.sort()
    return out


def _rezip(snapshot_dir: Path, out_path: Path) -> list:
    members = _members(snapshot_dir)
    if not any(name == "data.pkl" for name, _ in members):
        raise RuntimeError(
            f"{snapshot_dir} has no data.pkl, so it is not an exploded torch archive. "
            f"Found: {[n for n, _ in members][:12]}"
        )
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for arcname, path in members:
            info = zipfile.ZipInfo(f"{_ARCHIVE_PREFIX}/{arcname}", date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, path.read_bytes())
    tmp.replace(out_path)
    return [name for name, _ in members]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rezip_cache_root(cache_dir) -> Path:
    """Where rezipped archives are written.

    Not next to the source: on Kaggle a mounted dataset is read-only, and the Hub cache
    entry is meant to be immutable.
    """
    if cache_dir:
        return Path(cache_dir)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE) / "rezipped"
    except Exception:
        return Path(tempfile.gettempdir()) / "sn_gamestate_ain"


def _use_cached(out_path: Path, expected_sha256) -> bool:
    if not out_path.exists():
        return False
    got = sha256(out_path)
    if expected_sha256 is None or got == expected_sha256:
        log.info(f"[ain] cached archive {out_path} sha256 {got[:16]}")
        return True
    log.warning(f"[ain] cached archive digest {got[:16]} != expected "
                f"{expected_sha256[:16]}; rebuilding")
    return False


def _finish_rezip(source_dir: Path, out_path: Path, label: str, expected_sha256) -> Path:
    names = _rezip(source_dir, out_path)
    got = sha256(out_path)
    if expected_sha256 is not None and got != expected_sha256:
        raise RuntimeError(
            f"rezipped {label} digest mismatch\n"
            f"  expected {expected_sha256}\n  got      {got}"
        )
    log.info(f"[ain] {label}: {len(names)} members -> {out_path} "
             f"({out_path.stat().st_size / 1e6:.1f} MB), sha256 {got[:16]}")
    return out_path


def _resolve_local(source: Path, expected_sha256, cache_dir) -> Path:
    """Use a checkpoint already on disk, in either the packed or the exploded form.

    A packed file is digest-verified and used as-is; an exploded directory gets
    rezipped exactly as a Hub snapshot would. Note some exploded copies carry
    misleading names — a directory called `best_ain_full.pth` is still a directory.
    """
    if not source.exists():
        raise RuntimeError(f"ain_local_path does not exist: {source}")
    if source.is_file():
        got = sha256(source)
        if expected_sha256 is not None and got != expected_sha256:
            raise RuntimeError(
                f"{source} digest mismatch\n"
                f"  expected {expected_sha256}\n  got      {got}"
            )
        log.info(f"[ain] local archive {source} "
                 f"({source.stat().st_size / 1e6:.1f} MB), sha256 {got[:16]}")
        return source

    out_path = _rezip_cache_root(cache_dir) / f"local--{source.name}.pt"
    if _use_cached(out_path, expected_sha256):
        return out_path
    return _finish_rezip(source, out_path, str(source), expected_sha256)


def _resolve_packed(repo_id: str, filename: str, revision: str,
                    expected_sha256, cache_dir) -> Path:
    """Fetch one packed checkpoint file from the Hub and verify its digest.

    The published `best_ain_full.zip` is itself a torch archive (confirmed against the
    reference run: "the download is itself a checkpoint"), so the verified download is
    returned as-is. If a future artefact is instead a zip *around* a checkpoint, the
    largest member with a checkpoint suffix is extracted next to the rezip cache and
    returned; the sha256 pin covers the download either way.
    """
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    got = sha256(path)
    if expected_sha256 is not None and got != expected_sha256:
        raise RuntimeError(
            f"{repo_id}/{filename}@{revision[:12]} digest mismatch\n"
            f"  expected {expected_sha256}\n  got      {got}"
        )
    log.info(f"[ain] {repo_id}/{filename} ({path.stat().st_size / 1e6:.1f} MB), "
             f"sha256 {got[:16]}, revision {revision[:7]}")

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if any(n == "data.pkl" or n.endswith("/data.pkl") for n in names):
            return path                      # a torch archive: directly loadable
        members = [i for i in zf.infolist() if not i.is_dir() and i.file_size > 0]
        named = [m for m in members if m.filename.lower().endswith(CKPT_SUFFIXES)]
        if not (named or members):
            raise RuntimeError(f"{path} is a zip with no checkpoint member; "
                               f"members: {names[:12]}")
        pick = sorted(named or members, key=lambda m: -m.file_size)[0]
        dst = _rezip_cache_root(cache_dir) / "unpacked"
        dst.mkdir(parents=True, exist_ok=True)
        zf.extract(pick, dst)
        log.info(f"[ain] extracted {pick.filename} from {path.name}")
        return dst / pick.filename


def resolve_checkpoint(repo_id: str = REPO_ID, revision: str = REVISION,
                       expected_sha256: str = None, cache_dir=None,
                       local_path=None, filename: str = None) -> Path:
    """Return a path `torch.load` can read.

    Precedence: `local_path` (packed file or exploded directory) if given; else the
    packed Hub file `filename`; else the exploded-snapshot fallback, whose rezipped
    archive is cached and rebuilt rather than trusted on a digest mismatch.
    """
    if local_path:
        return _resolve_local(Path(local_path), expected_sha256, cache_dir)

    if filename:
        return _resolve_packed(repo_id, filename, revision, expected_sha256, cache_dir)

    from huggingface_hub import snapshot_download

    label = f"{repo_id}@{revision[:12]}"
    out_path = (_rezip_cache_root(cache_dir)
                / f"{repo_id.replace('/', '--')}--{revision[:12]}.pt")
    if _use_cached(out_path, expected_sha256):
        return out_path

    snapshot_dir = Path(snapshot_download(
        repo_id=repo_id, revision=revision,
        ignore_patterns=["*.md", ".gitattributes"],
    ))
    return _finish_rezip(snapshot_dir, out_path, label, expected_sha256)


def _torch_load(path: Path):
    """`torch.load` with `weights_only` pinned off where the parameter exists.

    The checkpoint stores a config dict and NumPy arrays alongside tensors, so the
    restricted unpickler cannot read it. The parameter is absent in older torch and
    defaults to True in torch 2.6+, so it is passed only when supported.
    """
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def _validate_checkpoint(checkpoint, path) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{path} holds {type(checkpoint).__name__}, expected a dict")
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise RuntimeError(
            f"{path} is missing {missing}; its keys are {sorted(checkpoint)[:12]}"
        )
    if not checkpoint["ema"]:
        raise RuntimeError(f"{path} has an empty 'ema' state dict")
    return checkpoint


def load_checkpoint(repo_id: str = REPO_ID, revision: str = REVISION,
                    expected_sha256: str = None, cache_dir=None,
                    local_path=None, filename: str = None) -> dict:
    """Resolve, load and structurally validate the checkpoint."""
    path = resolve_checkpoint(repo_id, revision, expected_sha256, cache_dir,
                              local_path, filename)
    return _validate_checkpoint(_torch_load(path), path)


# -------------------------------------------------------------------------- crops

# Height/width the crop is padded to before the resize, so a wide box is padded rather
# than stretched. Matches the training-time preprocessing.
TARGET_ASPECT = 2.0

# Stand-in for a box that clamps to nothing, so batch order stays aligned with the
# detection order. Callers that would rather leave a zero feature skip instead.
EMPTY_CROP = np.zeros((8, 4, 3), np.uint8)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(crop: np.ndarray, aspect: float = TARGET_ASPECT) -> np.ndarray:
    """Pad `crop` to height/width == `aspect` with its own mean colour. No resize."""
    height, width = crop.shape[:2]
    if crop.ndim == 3:
        fill = crop.reshape(-1, crop.shape[2]).mean(0).astype(crop.dtype)
    else:
        fill = int(crop.mean())
    if height / max(width, 1) >= aspect:            # too tall: pad width
        new_width = int(np.ceil(height / aspect))
        left = (new_width - width) // 2
        out = np.empty((height, new_width) + crop.shape[2:], dtype=crop.dtype)
        out[...] = fill
        out[:, left:left + width] = crop
    else:                                           # too wide: pad height
        new_height = int(np.ceil(width * aspect))
        top = (new_height - height) // 2
        out = np.empty((new_height, width) + crop.shape[2:], dtype=crop.dtype)
        out[...] = fill
        out[top:top + height] = crop
    return out


def crop_ltrb(image: np.ndarray, box) -> np.ndarray:
    """Integer crop of `box` clamped to the frame, or None if it clamps to nothing.

    The ordering test is explicit rather than a `patch.size` check afterwards: a box
    entirely outside the frame clamps to something like `x1=0, x2=-10`, and
    `image[y1:y2, 0:-10]` is a perfectly valid non-empty slice of almost the whole
    frame. Testing emptiness alone would hand the appearance model that slice.

    Callers decide what a degenerate box means: the tracker substitutes `EMPTY_CROP`
    to keep batch order aligned with the detection order, while split_merge skips the
    detection and leaves it a zero feature so it is never clustered or merged.
    """
    height, width = image.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(width, int(box[2]))
    y2 = min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


# -------------------------------------------------------------------------- model

# Tried in order. torchreid is VlSomers/bpbreid (declared in pyproject.toml); strong_sort
# is bundled by tracklab. Which of them carries the backbone is resolved at runtime.
_FACTORIES = ("torchreid.models.build_model", "strong_sort.deep.models.build_model")


def _call_factory(factory, name: str):
    """Call a torchreid-style factory, passing only the arguments it accepts."""
    params = inspect.signature(factory).parameters
    kwargs = {"num_classes": 1, "pretrained": False}
    if "loss" in params:
        kwargs["loss"] = "softmax"
    if "use_gpu" in params:
        kwargs["use_gpu"] = False
    return factory(name, **kwargs)


def build_backbone(name: str):
    """Build `name` from whichever model factory in this environment provides it.

    Returns (module, factory path). Raises with every failure listed rather than
    falling back to a different architecture, because a substituted backbone would
    load almost no weights and fail silently.
    """
    failures = []
    for path in _FACTORIES:
        module_path, attr = path.rsplit(".", 1)
        try:
            factory = getattr(importlib.import_module(module_path), attr)
        except Exception as exc:
            failures.append(f"{path}: not importable ({type(exc).__name__}: {exc})")
            continue
        try:
            return _call_factory(factory, name), path
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"no model factory in this environment can build {name!r}.\n  "
        + "\n  ".join(failures)
        + "\nEither install a torchreid that provides it, or vendor the definition."
    )


class _Net(nn.Module):
    """backbone -> GAP -> backbone.fc -> BNNeck -> {classifier, role_head}."""

    def __init__(self, backbone, feat_dim: int, num_ids: int, num_roles: int):
        super().__init__()
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_ids, bias=False)
        self.role_head = nn.Linear(feat_dim, num_roles)

    def features(self, x):
        if hasattr(self.backbone, "featuremaps"):
            maps = self.backbone.featuremaps(x)
        else:
            maps = self.backbone(x, return_featuremaps=True)
        vec = nn.functional.adaptive_avg_pool2d(maps, 1).flatten(1)
        fc = getattr(self.backbone, "fc", None)
        if fc is not None:
            vec = fc(vec)
        return vec


def _tensor_shape(state: dict, key: str):
    if key not in state:
        raise RuntimeError(
            f"the checkpoint's 'ema' has no {key!r}, so the network cannot be sized "
            f"from it; its keys begin {sorted(state)[:8]}"
        )
    return tuple(state[key].shape)


class OsnetAin:
    """BGR crops in, L2-normalised float32 embeddings out.

    Arithmetic is fp16 by construction: the backbone runs under torch autocast on CUDA
    (batch norms stay fp32, matmuls/convs run half) and casts back to fp32 before the
    L2 normalisation, so the output dtype never changes. Baked in after the Kaggle
    validation run (fp16 vs fp32 on the 5-sequence test set: tracking HOTA 71.61 vs
    71.08, GS-HOTA 64.98 vs 64.70, calibration bit-identical - deltas inside the
    observed run-to-run noise), so there is no precision switch. On CPU, autocast is
    unavailable and the forward runs fp32; `info` records the effective arithmetic.
    The tracking pass and the split_merge pass build this same module, so both embed a
    crop identically by construction.
    """

    def __init__(self, checkpoint: dict, device, batch_size: int = 64,
                 provenance: dict = None):
        cfg = checkpoint["cfg"]
        state = checkpoint["ema"]
        if not isinstance(cfg, dict):
            raise RuntimeError(f"checkpoint 'cfg' is {type(cfg).__name__}, expected a dict")
        for key in ("BACKBONE", "INPUT_HW"):
            if key not in cfg:
                raise RuntimeError(f"checkpoint 'cfg' has no {key!r}; it holds {sorted(cfg)}")

        feat_dim = _tensor_shape(state, "bnneck.weight")[0]
        num_ids = _tensor_shape(state, "classifier.weight")[0]
        num_roles = _tensor_shape(state, "role_head.weight")[0]

        backbone, factory = build_backbone(str(cfg["BACKBONE"]))
        net = _Net(backbone, feat_dim, num_ids, num_roles)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"{len(missing)} tensor(s) missing from the checkpoint, so the network "
                f"would be partly random: {list(missing)[:5]}"
            )

        self.net = net.to(device).eval()
        self.device = device
        self.dim = int(feat_dim)
        self.height, self.width = (int(v) for v in cfg["INPUT_HW"])
        self.batch_size = int(batch_size)
        # fp16 autocast on CUDA, baked in; on CPU autocast is unavailable and the
        # forward runs fp32. `info` records the EFFECTIVE arithmetic for the audit.
        self.use_fp16 = getattr(device, "type", str(device)) == "cuda"
        self.info = dict(backbone=str(cfg["BACKBONE"]), factory=factory,
                         input_hw=[self.height, self.width], embedding_dim=self.dim,
                         train_identities=len(checkpoint["pid2idx"]),
                         role_classes=int(num_roles), epoch=int(checkpoint["epoch"]),
                         unexpected_tensors=len(unexpected),
                         precision="fp16" if self.use_fp16 else "fp32")
        self.info.update(provenance or {})
        log.info(f"[ain] {cfg['BACKBONE']} via {factory} @ {self.height}x{self.width}, "
                 f"{self.dim}-d, {len(checkpoint['pid2idx']):,} train ids, "
                 f"epoch {checkpoint['epoch']}, {len(unexpected)} unexpected tensor(s), "
                 f"{self.info['precision']}")

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(letterbox(crop), (self.width, self.height),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)

    @torch.no_grad()
    def embed(self, crops) -> np.ndarray:
        """(N, dim) float32, L2-normalised. `crops` are BGR uint8 arrays."""
        if len(crops) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(crops), self.dim), dtype=np.float32)
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start:start + self.batch_size]
            batch = torch.from_numpy(
                np.stack([self._prepare(c) for c in chunk])
            ).to(self.device)
            amp = (torch.autocast(device_type="cuda") if self.use_fp16
                   else contextlib.nullcontext())
            with amp:
                vectors = self.net.bnneck(self.net.features(batch))
            # Back to fp32 BEFORE normalising, so the embedding dtype and the norm
            # arithmetic are identical in both precision modes.
            out[start:start + len(chunk)] = (
                nn.functional.normalize(vectors.float(), dim=1).cpu().numpy()
            )
        return out


def from_config(cfg, device, batch_size: int = 64) -> OsnetAin:
    """Build the embedder from a module config carrying the checkpoint's location.

    Also computes the resolved artefact's sha256 and passes it into `info` together
    with the effective precision, so the tracker's audit sidecar records exactly which
    weights, at which arithmetic, produced the run's embeddings.
    """
    repo_id = str(getattr(cfg, "ain_repo", REPO_ID))
    revision = str(getattr(cfg, "ain_revision", REVISION))
    expected = getattr(cfg, "ain_sha256", None) or None
    local_path = getattr(cfg, "ain_local_path", None) or None
    filename = getattr(cfg, "ain_file", None) or None

    # Every resolver branch enforces `expected` against the artefact it fetched, so the
    # digest here is provenance for the audit sidecar, not a second gate (the resolved
    # path may legitimately be a member extracted FROM the pinned download).
    path = resolve_checkpoint(repo_id, revision, expected, None, local_path, filename)
    digest = sha256(path)
    checkpoint = _validate_checkpoint(_torch_load(path), path)
    provenance = dict(
        sha256=digest,
        source=(str(local_path) if local_path
                else f"{repo_id}/{filename}@{revision[:12]}" if filename
                else f"{repo_id}@{revision[:12]} (rezipped snapshot)"),
    )
    return OsnetAin(checkpoint, device, batch_size=batch_size, provenance=provenance)
