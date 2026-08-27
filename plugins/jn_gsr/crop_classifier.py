#!/usr/bin/env python3
"""crop_classifier.py -- ResNet-18 single/multi-player frame filter.

Stage 1 of the pipeline: ResNet-18 -> legibility -> DBNet++ -> PARSeq -> sum rule.

WHAT THIS IS. The question is unchanged from the retired Mask R-CNN filter:
"does a second person cover >= 10% of this player's box?". This classifier is a
learned approximation of that rule -- same question, exact computation swapped
for a cheap forward pass with its own error rate.

WHAT CHANGED FROM THE PREVIOUS CHECKPOINT. The weights are now

    Ynniss/Resnet18_multi_single_player          (was: Ynniss/anomaly_classifier)

and they are a DIFFERENT ARCHITECTURE, not a re-train of the same one:

                       old best.pt              new checkpoint
    stem               conv1 3x3 stride-1       conv1 7x7 stride-2 (STOCK)
    head               fc -> 2 logits           fc -> 1 logit
    read as            softmax(logits)[:, 1]    sigmoid(logit)
    threshold          0.8639 (valid max-F1)    NOT PUBLISHED -- see below

Both heads express the same binary question; 1 logit + BCEWithLogits and
2 logits + CrossEntropy are the two standard encodings of it. Koshkina's
legibility ResNet-34 in this same pipeline uses the 1-logit form too
(legibility.py: "nn.BCELoss during training ... the model emits a SIGMOID
PROBABILITY, one value per image").

Nothing here hard-codes that reading. build_model() is told the head width, and
CropClassifier derives BOTH the stem and the head width from the checkpoint's
own tensor shapes, then loads strictly -- so a wrong guess raises immediately
instead of producing plausible garbage. What the file actually contains is
printed at load.

THE THRESHOLD IS NOT IN THE CHECKPOINT, and the old 0.8639 does not transfer:
it was tuned on the old model's softmax output, and has no meaning on this
model's sigmoid. DEFAULT_THR is therefore 0.5, the neutral point, and
run_eval.py sweeps it at merge time over cached probabilities at no extra cost.
Read the printed histogram: a distribution piled into the ambiguous middle is
how a wrong input size or a flipped polarity announces itself.

THE REPOSITORY IS NOT A .pt FILE. It is a torch.save zip archive uploaded
EXPLODED into its 128 member files (data.pkl, data/0..121, version, byteorder,
.format_version, .storage_alignment, .data/serialization_id). hf_hub_download of
a "best.pt" would 404. resolve_weights() snapshot-downloads the repo and
reassembles the archive before torch.load; see reassemble_archive().

    $PY crop_classifier.py          # self-tests (pure logic needs no torch)
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys

# Before any import that can reach matplotlib (torchvision does, transitively).
# Kaggle presets MPLBACKEND to an inline backend whose module is absent here.
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from PIL import Image

# --------------------------------------------------------------- constants --
HF_REPO = "Ynniss/Resnet18_multi_single_player"

# data.pkl fixes every key name, dtype, shape and storage key in the archive, so
# pinning it detects any architecture drift even though the reassembled .pt is
# not byte-reproducible (zip records carry mtimes). Read off the HuggingFace
# blob page for this repo.
DATA_PKL_SHA256 = "4c789a18a6be9c13efbf3452352ff9b5410b9a364e6d89a1605f6f777a1dab7f"
N_STORAGES = 122          # a ResNet-18 state_dict: 20 conv + 20 BN x 5 + fc x 2

# Fallbacks only. A cfg dict inside the checkpoint WINS, and --img-size wins
# over both. ResNet pools adaptively, so a wrong size does not crash -- it
# silently shifts every probability. Same trap legibility.py documents.
IMG_H, IMG_W = 96, 48

DEFAULT_THR = 0.5         # neutral; the old 0.8639 belonged to the old model
CLASSES = ("single", "multi")   # index 0, index 1


# ------------------------------------------------------- numpy 2 -> 1 pickles --
class _ModuleAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve modules named `new.*` out of the `old.*` package that exists."""

    def __init__(self, new, old):
        self.new, self.old = new, old

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.new or fullname.startswith(self.new + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(self.old + spec.name[len(self.new):])

    def exec_module(self, module):
        pass


def numpy2_pickle_compat():
    """Let numpy-2-written pickles load under numpy 1.x.

    numpy 2.0 renamed `numpy.core` to `numpy._core`, and a pickle records the
    module path of every global it references. Checkpoints saved on a modern
    machine therefore name `numpy._core...`, which does not exist in this venv
    -- numpy is pinned below 2 because torch 2.0.1 and mmcv 2.0.1 are built
    against the 1.x C ABI, so 'just upgrade numpy' would break the detector.

    No-op on numpy 2, safe to call repeatedly. True if the shim was installed.
    """
    import numpy as _np
    if hasattr(_np, "_core"):
        return False
    if any(isinstance(f, _ModuleAliasFinder) and f.new == "numpy._core"
           for f in sys.meta_path):
        return False
    sys.meta_path.insert(0, _ModuleAliasFinder("numpy._core", "numpy.core"))
    return True


def sha256_of(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------- exploded-archive handling --
_SKIP_PREFIXES = (".cache", ".git", ".huggingface")
_SKIP_NAMES = (".gitattributes", "README.md")


def archive_members(snapshot_dir):
    """Relative paths of the archive's real members inside a snapshot dir.

    Excludes the hub's own bookkeeping (.cache/, .gitattributes, a model card),
    which are not part of the torch archive and would confuse the reader.
    """
    out = []
    for dirpath, _dirs, files in os.walk(snapshot_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f),
                                  snapshot_dir).replace(os.sep, "/")
            if rel.startswith(_SKIP_PREFIXES) or rel in _SKIP_NAMES:
                continue
            out.append(rel)
    return sorted(out)


def reassemble_archive(snapshot_dir, out_pt, arcname="archive"):
    """Exploded snapshot dir -> a real torch .pt. Returns (out_pt, members).

    torch's zip serialisation requires every record under ONE top-level
    directory; the directory's NAME is arbitrary and is not recorded anywhere in
    the pickle, so any value loads. ZIP_STORED because the payload is raw fp32
    tensor bytes: deflate would cost CPU for ~nothing and torch reads records by
    offset either way. data.pkl is written first so a reader hits the manifest
    without seeking to the end of 44 MB.
    """
    import zipfile
    members = archive_members(snapshot_dir)
    if "data.pkl" not in members:
        raise RuntimeError(
            f"{snapshot_dir}: no data.pkl among {members[:6]}... -- this does "
            f"not look like an exploded torch archive.")
    order = {"data.pkl": 0, "byteorder": 1, ".format_version": 2,
             ".storage_alignment": 3, "version": 4}
    members.sort(key=lambda r: (order.get(r, 5), r))
    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    tmp = out_pt + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for rel in members:
            z.write(os.path.join(snapshot_dir, rel), f"{arcname}/{rel}")
    os.replace(tmp, out_pt)
    return out_pt, members


def verify_snapshot(snapshot_dir, strict=False):
    """Provenance check on the exploded repo. Returns a dict; never raises
    unless `strict`. A private re-upload is legitimate, but no run may quietly
    claim a provenance it does not have -- so a mismatch is reported and travels
    with the result."""
    members = archive_members(snapshot_dir)
    n_storage = sum(1 for m in members if m.startswith("data/"))
    pkl = os.path.join(snapshot_dir, "data.pkl")
    got = sha256_of(pkl) if os.path.exists(pkl) else None
    info = {"n_members": len(members), "n_storages": n_storage,
            "data_pkl_sha256": got, "published_data_pkl_sha256": DATA_PKL_SHA256,
            "data_pkl_matches": got == DATA_PKL_SHA256,
            "n_storages_expected": N_STORAGES}
    if not info["data_pkl_matches"]:
        msg = (f"[classifier] WARNING: data.pkl sha256 {str(got)[:16]}... does "
               f"not match the published {DATA_PKL_SHA256[:16]}... -- a "
               f"DIFFERENT checkpoint structure. Legitimate for a private "
               f"re-upload; every result carries this flag.")
        if strict:
            raise RuntimeError(msg)
        print(msg)
    if n_storage != N_STORAGES:
        print(f"[classifier] WARNING: {n_storage} tensor storages, expected "
              f"{N_STORAGES} for a ResNet-18 state_dict.")
    return info


def resolve_weights(path=None, out="models/resnet18_multi_single.pt",
                    repo=HF_REPO):
    """Attached path -> already-present -> HuggingFace snapshot + reassembly.

    Returns (path, source, info). `info` is the provenance record.
    """
    if path and os.path.exists(path):
        return path, f"attached: {path}", {"attached": True}
    if os.path.exists(out):
        return out, "already-present", {"attached": False}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(repo_id=repo)
    except Exception as e:
        raise RuntimeError(
            f"could not fetch {repo} from HuggingFace: {type(e).__name__}: {e}\n"
            f"  * Internet must be ON in the Kaggle notebook settings.\n"
            f"  * If the repo is private, log in first:\n"
            f"      huggingface-cli login   (or set $HF_TOKEN)\n"
            f"  * Or attach a reassembled .pt as a Kaggle dataset and pass\n"
            f"      --clf-weights /kaggle/input/<slug>/<file>.pt") from e
    info = verify_snapshot(snap)
    _, members = reassemble_archive(snap, out)
    info["reassembled_from"] = len(members)
    return out, f"huggingface: {repo} (exploded archive, reassembled)", info


# ------------------------------------------------------------- preprocessing --
def letterbox(img, h=IMG_H, w=IMG_W):
    """Aspect-preserving resize onto a BLACK h*w canvas, centred.

    This is NOT a stretch -- contrast PARSeq's transform, which forces the exact
    target size and does distort. A silent stretch here would shift every
    probability while looking entirely normal in the output.
    """
    img = img.convert("RGB")
    iw, ih = img.size
    sc = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * sc)), max(1, round(ih * sc))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def letterbox_geometry(src_size, h=IMG_H, w=IMG_W):
    """(nw, nh, ox, oy) for a source (iw, ih) -- the pasted extent and offset."""
    iw, ih = src_size
    sc = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * sc)), max(1, round(ih * sc))
    return nw, nh, (w - nw) // 2, (h - nh) // 2


def crop_box(frame_pil, xywh):
    """The RAW GT annotation box, rasterised and clipped.

    Deliberately the unpadded box: coverage `s` in the training set was measured
    on the annotation box, so the classifier's input domain is that box. The 18%
    pad belongs to DBNet++, one stage later, and is a different region for a
    different purpose. legibility.py scores this same crop.
    """
    W, H = frame_pil.size
    x, y, w, h = xywh
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None, (x0, y0, x1, y1)
    return frame_pil.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


# -------------------------------------------------------------------- model --
def build_model(n_out=1, stem="7x7"):
    """torchvision resnet18 with the head width and stem the CHECKPOINT asks for.

    stem "7x7" is the stock ImageNet stem (what the new checkpoint has).
    stem "3x3" is the small-input variant the OLD checkpoint had -- kept so an
    older file still loads rather than failing with a shape error the user then
    has to decode. A strict load_state_dict against the result is itself the
    architecture assertion.
    """
    import torch.nn as nn
    import torchvision
    m = torchvision.models.resnet18(weights=None)
    if stem == "3x3":
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    elif stem != "7x7":
        raise ValueError(f"unknown stem {stem!r}")
    m.fc = nn.Linear(m.fc.in_features, int(n_out))
    return m


def strip_prefix(sd):
    """Drop a uniform key prefix (module., model., backbone., net.) if present.

    Same defence legibility.py applies. Returns (state_dict, prefix|'').
    """
    sd = {k: v for k, v in sd.items() if k != "_metadata"}
    for p in ("module.", "model.", "backbone.", "net."):
        if sd and all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}, p
    return sd, ""


def extract_state_dict(ck):
    """A checkpoint may be a bare state_dict or a wrapper around one.

    Returns (state_dict, meta) where meta holds every non-tensor scalar found
    alongside it -- that is where an img size or an operating threshold would
    live if the trainer saved one. Printed at load; never silently applied.
    """
    import torch
    meta = {}
    if isinstance(ck, dict) and not any(
            isinstance(v, torch.Tensor) for v in ck.values()):
        for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
            if key in ck and isinstance(ck[key], dict):
                sd = ck[key]
                for k, v in ck.items():
                    if k == key:
                        continue
                    if isinstance(v, (int, float, str, bool, type(None))):
                        meta[k] = v
                    elif isinstance(v, dict):
                        meta.update({f"{k}.{kk}": vv for kk, vv in v.items()
                                     if isinstance(vv, (int, float, str, bool))})
                return sd, meta
        raise RuntimeError(
            f"checkpoint has no tensors and no recognised wrapper key "
            f"(found {sorted(ck)[:8]}).")
    if not isinstance(ck, dict):
        raise RuntimeError(f"checkpoint is a {type(ck).__name__}, not a dict; "
                           f"torch.save(model) is not supported -- save "
                           f"model.state_dict() instead.")
    return ck, meta


def describe_checkpoint(sd):
    """Derive architecture facts from tensor shapes alone. No assumptions."""
    if "conv1.weight" not in sd or "fc.weight" not in sd:
        raise RuntimeError(
            f"not a torchvision resnet state_dict: missing conv1.weight/"
            f"fc.weight (have {sorted(sd)[:6]}...)")
    k = tuple(sd["conv1.weight"].shape)          # (64, 3, K, K)
    stem = "7x7" if k[-1] == 7 else "3x3" if k[-1] == 3 else f"{k[-1]}x{k[-1]}"
    n_out, n_feat = tuple(sd["fc.weight"].shape)
    return {"stem": stem, "n_out": int(n_out), "fc_in_features": int(n_feat),
            "n_keys": len(sd), "conv1_shape": list(k)}


class CropClassifier:
    """Frozen single/multi crop classifier. Inference only, eval mode.

    p_multi is P(class 'multi'):
      1-logit head -> sigmoid(logit)          (BCEWithLogits convention)
      2-logit head -> softmax(logits)[:, 1]   (CrossEntropy convention)
    `polarity='invert'` flips it, for a checkpoint whose positive class is
    'single'. Nothing in the file records which; the histogram and the ablation
    table are how you tell.
    """

    def __init__(self, weights=None, device=None, thr=DEFAULT_THR, batch=256,
                 out="models/resnet18_multi_single.pt", img_size=None,
                 norm="half", polarity="multi", repo=HF_REPO, quiet=False):
        import torch
        import torchvision.transforms as T

        self.path, self.source, self.provenance = resolve_weights(
            weights, out, repo=repo)
        if numpy2_pickle_compat() and not quiet:
            print("[classifier] numpy<2 detected: mapping numpy._core -> "
                  "numpy.core so the checkpoint's pickle can be read")
        # weights_only=False: a wrapped checkpoint may carry a cfg dict beside
        # the tensors. Explicit because torch>=2.6 flipped the default.
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        sd, self.meta = extract_state_dict(ck)
        sd, self.key_prefix = strip_prefix(sd)
        self.arch = describe_checkpoint(sd)

        # size: --img-size > checkpoint cfg > module fallback
        cfg_h = self.meta.get("cfg.img_h", self.meta.get("img_h"))
        cfg_w = self.meta.get("cfg.img_w", self.meta.get("img_w"))
        if img_size:
            self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
            self.size_source = "--img-size"
        elif cfg_h and cfg_w:
            self.img_h, self.img_w = int(cfg_h), int(cfg_w)
            self.size_source = "checkpoint cfg"
        else:
            self.img_h, self.img_w = IMG_H, IMG_W
            self.size_source = "module fallback (NOT from the checkpoint)"

        self.model = build_model(n_out=self.arch["n_out"],
                                 stem=self.arch["stem"])
        self.model.load_state_dict(sd)          # strict -> architecture gate
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        if norm == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif norm == "half":
            mean, std = [0.5] * 3, [0.5] * 3
        else:
            raise ValueError(f"unknown norm {norm!r} (half|imagenet)")
        self.norm = norm
        self.tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        self.polarity = polarity
        self.thr = float(thr)
        self.batch = int(batch)
        self.n_calls = self.n_crops = 0

        # Fail-fast: a broken device or an incompatible torchvision raises HERE,
        # in the first seconds, not at the first real crop hours in.
        with torch.inference_mode():
            probe = self.model(torch.zeros(2, 3, self.img_h, self.img_w,
                                           device=self.device))
        if tuple(probe.shape) != (2, self.arch["n_out"]):
            raise RuntimeError(f"forward produced {tuple(probe.shape)}, "
                               f"expected (2, {self.arch['n_out']})")
        if not quiet:
            self.report()

    def report(self):
        a = self.arch
        print(f"[classifier] {self.path}  ({self.source})")
        print(f"[classifier]   resnet18 stem={a['stem']} fc={a['fc_in_features']}"
              f"->{a['n_out']}  keys={a['n_keys']}"
              f"{'  prefix=' + self.key_prefix if self.key_prefix else ''}")
        print(f"[classifier]   p_multi = "
              f"{'sigmoid(logit)' if a['n_out'] == 1 else 'softmax(logits)[:,1]'}"
              f"{'  INVERTED' if self.polarity == 'invert' else ''}"
              f"   thr={self.thr}")
        print(f"[classifier]   input {self.img_h}x{self.img_w} "
              f"({self.size_source}), norm={self.norm}")
        if self.meta:
            print(f"[classifier]   checkpoint metadata: {self.meta}")
        else:
            print("[classifier]   checkpoint metadata: none (bare state_dict) "
                  "-- no img size, no threshold, no class order recorded")

    def preprocess(self, crop):
        """PIL crop -> (canvas, tensor). The tensor IS what the model consumes."""
        canvas = letterbox(crop, self.img_h, self.img_w)
        return canvas, self.tf(canvas)

    def score_crops(self, crops, keep_inputs=False):
        """[PIL] -> (p_multi array, [canvas], input tensor batch | None)."""
        import torch
        if not crops:
            return np.zeros(0, np.float64), [], None
        canvases, tensors = [], []
        for c in crops:
            cv, t = self.preprocess(c)
            canvases.append(cv)
            tensors.append(t)
        probs, kept = [], []
        with torch.inference_mode():
            for i in range(0, len(tensors), self.batch):
                xb = torch.stack(tensors[i:i + self.batch]).to(self.device)
                logits = self.model(xb).float()
                if self.arch["n_out"] == 1:
                    p = torch.sigmoid(logits)[:, 0]
                else:
                    p = torch.softmax(logits, 1)[:, 1]
                if self.polarity == "invert":
                    p = 1.0 - p
                probs.append(p.cpu().numpy())
                self.n_calls += 1
                if keep_inputs:
                    kept.append(xb.detach().cpu())
        self.n_crops += len(crops)
        inputs = torch.cat(kept) if keep_inputs and kept else None
        return np.concatenate(probs).astype(np.float64), canvases, inputs


# ---------------------------------------------------------- the frame filter --
def keep_from_probs(probs, thr):
    """(p_multi array, threshold) -> (keep mask, guard_fired).

    The rule is p_multi < thr: keep what looks like one player. The direct
    analogue of the retired n(b) <= 1.

    NEVER-EMPTY GUARD. If every frame is rejected, keep the argmin of p_multi.
    A continuous score is a total order, so the guard admits THE least
    crowded-looking frame rather than an arbitrary set of equally-bad ones.
    A guarded tracklet's surviving frames are ones the filter wanted to reject,
    so `guard_fired` must be recorded and must travel with the verdict.

    Pure function of the probabilities, so run_eval.py's merge re-derives the
    identical mask from cached scores -- ONE source of truth for the rule.
    """
    probs = np.asarray(probs, np.float64)
    if probs.size == 0:
        return np.zeros(0, bool), False
    keep = probs < float(thr)
    if not keep.any():
        return probs == probs.min(), True
    return keep, False


def keep_and_diag(frames, clf, thr=None, cache=None):
    """(frames, classifier) -> (keep, p_multi, 'p_multi', guard_fired)."""
    thr = clf.thr if thr is None else float(thr)
    if not frames:
        return (np.zeros(0, bool), np.zeros(0, np.float64), "p_multi", False)

    crops, ok = [], []
    for fr in frames:
        key = (fr["frame_path"], tuple(fr["xywh"]))
        img = None
        if cache is not None and key in cache:
            img = cache[key]
        if img is None:
            frame = Image.open(fr["frame_path"]).convert("RGB")
            img, _ = crop_box(frame, fr["xywh"])
            if cache is not None and img is not None:
                cache[key] = img
        ok.append(img is not None)
        if img is not None:
            crops.append(img)

    probs = np.ones(len(frames), np.float64)     # degenerate box -> reject...
    if crops:
        p, _, _ = clf.score_crops(crops)
        probs[np.array(ok, bool)] = p
    # ...but the guard still cannot leave a tracklet empty. A degenerate box
    # scores 1.0, so it is chosen only when nothing else exists.
    keep, guard = keep_from_probs(probs, thr)
    return keep, probs, "p_multi", guard


def make_frame_filter(clf, thr=None):
    """frames -> kept frames."""
    if clf is None:
        return lambda frames: frames

    def _filt(frames):
        if not frames:
            return frames
        keep, _, _, _ = keep_and_diag(frames, clf, thr)
        return [f for f, k in zip(frames, keep) if k]
    return _filt


def histogram(probs, bins=10):
    """Printable 0..1 histogram. A pile in the middle means the operating point
    -- or the input size, or the polarity -- is wrong."""
    probs = np.asarray(probs, np.float64)
    if probs.size == 0:
        return "  (no scores)"
    counts, _ = np.histogram(probs, bins=bins, range=(0.0, 1.0))
    top = max(1, counts.max())
    lines = []
    for i, c in enumerate(counts):
        lo, hi = i / bins, (i + 1) / bins
        lines.append(f"  [{lo:.1f},{hi:.1f})  {'#' * int(40 * c / top):<40} {c}")
    return "\n".join(lines)


# ------------------------------------------------------------- self-tests ----
if __name__ == "__main__":
    import tempfile

    # ---- letterbox geometry: aspect preserved, black padding, centred -------
    for (iw, ih) in [(46, 98), (51, 109), (200, 50), (10, 10), (1, 300)]:
        src = Image.new("RGB", (iw, ih), (200, 30, 30))
        cv = letterbox(src, 96, 48)
        assert cv.size == (48, 96), cv.size
        nw, nh, ox, oy = letterbox_geometry((iw, ih), 96, 48)
        assert nw <= 48 and nh <= 96, (nw, nh)
        assert abs(nw / nh - iw / ih) <= max(0.02, 1.5 / min(nw, nh)), (iw, ih)
        a = np.asarray(cv)
        if oy > 0:
            assert a[:oy].max() == 0
        if ox > 0:
            assert a[:, :ox].max() == 0
        assert a[oy:oy + nh, ox:ox + nw].max() > 0
    assert letterbox_geometry((200, 50), 96, 48)[0] == 48
    assert letterbox_geometry((46, 98), 96, 48)[1] == 96

    # ---- crop_box: rasterisation, clipping, degeneracy ----------------------
    frame = Image.new("RGB", (60, 40), (7, 7, 7))
    c, ext = crop_box(frame, (10, 10, 20, 20))
    assert c.size == (20, 20) and ext == (10, 10, 30, 30)
    c, ext = crop_box(frame, (55, 35, 20, 20))
    assert c.size == (5, 5) and ext == (55, 35, 60, 40)
    assert crop_box(frame, (10, 10, 0, 20))[0] is None
    assert crop_box(frame, (10.4, 10.6, 20.0, 20.0))[0].size == (20, 20)

    # ---- keep_from_probs: the rule, the guard, monotonicity -----------------
    p = np.array([0.1, 0.9, 0.2, 0.95])
    k, g = keep_from_probs(p, 0.5)
    assert list(k) == [True, False, True, False] and not g
    k, g = keep_from_probs(p, 0.0)
    assert g and k.sum() == 1 and k[0]
    k, g = keep_from_probs(np.array([0.4, 0.4, 0.9]), 0.0)   # tie at the argmin
    assert g and list(k) == [True, True, False]
    assert keep_from_probs(np.zeros(0), 0.5)[0].shape == (0,)
    prev = None
    for t in (0.99, 0.8, 0.5, 0.2, 0.05):
        k, g = keep_from_probs(p, t)
        if prev is not None and not g:
            assert k.sum() <= prev, (t, k.sum(), prev)
        prev = k.sum()

    # ---- keep_and_diag agrees with keep_from_probs on real crops ------------
    class _FakeClf:
        """Scores crops by mean red / 255 -- deterministic, no torch."""
        thr = 0.5

        def score_crops(self, crops, keep_inputs=False):
            p = np.array([float(np.asarray(c)[:, :, 0].mean()) / 255.0
                          for c in crops], np.float64)
            return p, list(crops), None

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, red in enumerate((10, 200, 40, 250)):
            fp = os.path.join(td, f"f{i}.png")
            Image.new("RGB", (60, 40), (red, 0, 0)).save(fp)
            paths.append(fp)
        frames = [{"frame_path": fp, "xywh": (10, 10, 20, 20)} for fp in paths]

        keep, probs, kind, guard = keep_and_diag(frames, _FakeClf(), thr=0.5)
        assert kind == "p_multi" and not guard
        assert list(keep) == [True, False, True, False]
        assert np.array_equal(keep, keep_from_probs(probs, 0.5)[0])
        assert abs(probs[0] - 10 / 255) < 1e-9 and abs(probs[3] - 250 / 255) < 1e-9

        keep, probs, _, guard = keep_and_diag(frames, _FakeClf(), thr=0.0)
        assert guard and keep.sum() == 1 and keep[int(np.argmin(probs))]

        deg = [{"frame_path": paths[0], "xywh": (10, 10, 0, 20)}]
        keep, probs, _, guard = keep_and_diag(deg, _FakeClf(), thr=0.5)
        assert probs[0] == 1.0 and guard and keep[0]

        keep, probs, _, guard = keep_and_diag([], _FakeClf())
        assert len(keep) == 0 and not guard

        f = make_frame_filter(_FakeClf(), thr=0.5)
        assert [x["frame_path"] for x in f(frames)] == [paths[0], paths[2]]
        assert make_frame_filter(None)(frames) is frames

        cache = {}
        k1, p1, _, _ = keep_and_diag(frames, _FakeClf(), thr=0.5, cache=cache)
        k2, p2, _, _ = keep_and_diag(frames, _FakeClf(), thr=0.5, cache=cache)
        assert list(k1) == list(k2) and np.array_equal(p1, p2) and len(cache) == 4

    assert "no scores" in histogram([])
    assert histogram([0.1, 0.9]).count("\n") == 9

    # ---- ARCHITECTURE + EXPLODED-ARCHIVE TESTS (need torch) ----------------
    try:
        import torch
        import torch.nn as nn
        import torchvision                                   # noqa: F401
    except ImportError:
        print("crop_classifier: torch tests SKIPPED (no torch)")
    else:
        import shutil
        import zipfile

        def _explode(obj, d):
            """Write obj as a .pt then unpack it exactly as the HF repo is."""
            src = os.path.join(d, "src.pt")
            torch.save(obj, src)
            snap = os.path.join(d, "snap")
            with zipfile.ZipFile(src) as z:
                root = z.namelist()[0].split("/")[0]
                for n in z.namelist():
                    rel = n[len(root) + 1:]
                    if not rel:
                        continue
                    dst = os.path.join(snap, rel)
                    os.makedirs(os.path.dirname(dst) or snap, exist_ok=True)
                    with z.open(n) as fi, open(dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
            return src, snap

        _td = tempfile.mkdtemp()
        try:
            # the published architecture: stock 7x7 stem, 1-logit head
            ref = build_model(n_out=1, stem="7x7")
            ref_sd = ref.state_dict()
            _src, _snap = _explode(ref_sd, _td)

            # hub bookkeeping must be ignored, not zipped into the archive
            os.makedirs(os.path.join(_snap, ".cache"), exist_ok=True)
            open(os.path.join(_snap, ".cache", "junk"), "w").write("x")
            open(os.path.join(_snap, ".gitattributes"), "w").write("x")

            mem = archive_members(_snap)
            # The STORAGE count (data/*) is the architecture fingerprint and is
            # invariant. The TOTAL member count is NOT: torch writes a
            # version-dependent set of sidecar files -- 2.0.1 emits ~124,
            # newer torch ~128 (it adds byteorder, .format_version,
            # .storage_alignment, .data/serialization_id). Asserting the total
            # here made the test pass on the build box and fail in the Kaggle
            # venv, which is the opposite of what a test is for. Reassembly is
            # proven bit-exact below regardless of the sidecar set.
            n_store = sum(1 for m in mem if m.startswith("data/"))
            assert n_store == N_STORAGES, n_store
            assert len(mem) >= N_STORAGES + 1, len(mem)   # at least data.pkl too
            assert ".cache/junk" not in mem and ".gitattributes" not in mem

            _out = os.path.join(_td, "rebuilt.pt")
            reassemble_archive(_snap, _out)
            got = torch.load(_out, map_location="cpu", weights_only=True)
            assert set(got) == set(ref_sd)
            assert not [k for k in ref_sd
                        if not torch.equal(got[k], ref_sd[k])], "not bit-exact"

            d = describe_checkpoint(got)
            assert d == {"stem": "7x7", "n_out": 1, "fc_in_features": 512,
                         "n_keys": 122, "conv1_shape": [64, 3, 7, 7]}, d
            build_model(**{"n_out": d["n_out"],
                           "stem": d["stem"]}).load_state_dict(got)  # strict

            # the OLD architecture must still be described and built correctly
            old = build_model(n_out=2, stem="3x3")
            d_old = describe_checkpoint(old.state_dict())
            assert (d_old["stem"], d_old["n_out"]) == ("3x3", 2), d_old
            build_model(n_out=2, stem="3x3").load_state_dict(old.state_dict())
            # ...and must NOT accept the new file
            try:
                build_model(n_out=2, stem="3x3").load_state_dict(got)
                raise AssertionError("3x3/2-logit must reject the 7x7/1 file")
            except RuntimeError as e:
                assert "conv1.weight" in str(e)

            # wrapper + prefix handling
            sd_w, meta = extract_state_dict(
                {"model": ref_sd, "cfg": {"img_h": 96, "img_w": 48},
                 "epoch": 28, "val_pr_auc": 0.918})
            assert meta["epoch"] == 28 and meta["cfg.img_h"] == 96
            assert describe_checkpoint(sd_w)["n_out"] == 1
            pre, p = strip_prefix({"module." + k: v for k, v in ref_sd.items()})
            assert p == "module." and describe_checkpoint(pre)["n_out"] == 1
            assert strip_prefix(dict(ref_sd))[1] == ""

            # END-TO-END: CropClassifier over the reassembled file, both heads
            for n_out, stem, want in ((1, "7x7", "sigmoid"), (2, "3x3", "softmax")):
                m = build_model(n_out=n_out, stem=stem)
                fp = os.path.join(_td, f"ck_{n_out}.pt")
                torch.save(m.state_dict(), fp)
                c = CropClassifier(weights=fp, device="cpu", quiet=True)
                assert c.arch["n_out"] == n_out and c.arch["stem"] == stem
                assert (c.img_h, c.img_w) == (96, 48)
                pr, cv, xin = c.score_crops(
                    [Image.new("RGB", (46, 98), (9, 9, 9)),
                     Image.new("RGB", (51, 109), (240, 10, 10))], keep_inputs=True)
                assert pr.shape == (2,) and pr.min() >= 0 and pr.max() <= 1, pr
                assert tuple(xin.shape) == (2, 3, 96, 48)
                assert cv[0].size == (48, 96) and c.model.training is False
                solo, _, _ = c.score_crops([Image.new("RGB", (46, 98), (9, 9, 9))])
                assert abs(float(solo[0]) - float(pr[0])) < 2e-3, "batch invariance"
                assert c.score_crops([])[0].shape == (0,)
                inv = CropClassifier(weights=fp, device="cpu", quiet=True,
                                     polarity="invert")
                pi, _, _ = inv.score_crops([Image.new("RGB", (46, 98), (9, 9, 9))])
                assert abs(float(pi[0]) - (1.0 - float(solo[0]))) < 1e-6
                # cfg in the checkpoint must beat the module fallback
                torch.save({"model": m.state_dict(),
                            "cfg": {"img_h": 64, "img_w": 32}}, fp)
                c2 = CropClassifier(weights=fp, device="cpu", quiet=True)
                assert (c2.img_h, c2.img_w) == (64, 32), (c2.img_h, c2.img_w)
                # ...and --img-size must beat the cfg
                c3 = CropClassifier(weights=fp, device="cpu", quiet=True,
                                    img_size=(96, 48))
                assert (c3.img_h, c3.img_w) == (96, 48)
            print("crop_classifier: torch tests passed (122-storage archive "
                  "invariant, bit-exact reassembly, stem/head derived from "
                  "shapes, strict load, sigmoid + softmax heads, polarity, "
                  "size precedence)")
        finally:
            shutil.rmtree(_td, ignore_errors=True)

    print("crop_classifier: all self-tests passed")
