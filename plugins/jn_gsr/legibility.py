#!/usr/bin/env python3
"""legibility.py -- Koshkina's ResNet34 legibility classifier as a stage.

WHY. The audit's largest single loss is that 8 of 15 unnumbered tracklets are
given a fabricated number. This pipeline has no legibility model at all: the
"is a number visible" decision is delegated entirely to DBNet++ scoring >= 0.52,
and PARSeq -- the only component that actually looks at the glyphs -- cannot
abstain, because an argmax over ten digits always returns a digit. Koshkina
answers exactly this question with a dedicated binary classifier, and the
SoccerNet-fine-tuned weights are public.

WHAT IS TAKEN FROM THEIR CODE (legibility_classifier.py, verified, not inferred):

  * arch resnet34 -> networks.LegibilityClassifier34
  * nn.BCELoss during training, and `preds = outputs.round()` -- so the model
    emits a SIGMOID PROBABILITY, one value per image, already in [0, 1].
    Do not apply a softmax, and do not treat it as a logit.
  * `run(..., threshold=0.5)`: legible iff output > threshold. Strictly greater.
  * the checkpoint is a BARE state_dict; they delete `_metadata` if present.
  * helpers.THRESHOLD_FOR_TACK_LEGIBILITY = 0 and
    helpers.is_track_legible: a tracklet is legible if it has MORE THAN zero
    legible images. One legible frame out of hundreds is enough.

Paper Table 7, ResNet34 trained on soccer and tested on soccer: 91.71%
accuracy, 94.17% F1, measured at TRACKLET level under that same any-frame rule.

INPUT RESOLUTION: 256x256, ImageNet mean/std. This does not appear in the
files verified above -- it comes from a working ViTPose notebook that drives
the same checkpoint:

    transforms.Compose([Resize((256, 256)), ToTensor(),
                        Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

It matters more than it looks. ResNet uses adaptive average pooling, so a
wrong size does NOT crash -- it quietly shifts every probability. 224 (the
ImageNet default) was this module's first guess and was wrong. `--legibility-
size` therefore stays explicit and is recorded in provenance, and the
probability histogram is reported: a distribution piled into the ambiguous
middle is how a wrong size announces itself.

APPLIED TO THE WHOLE PLAYER CROP, not the torso -- the same notebook scores
`frame.crop(gt_box)` directly. That matches crop_box() here.

ARCHITECTURE is derived, not assumed: detect_arch() keys off `layer1.2`,
which exists in resnet34 and not resnet18, and the head width comes from the
checkpoint's own tensor shapes.

    $PY legibility.py      # self-tests, no torch required
"""
import os
import shutil

import numpy as np

HF_DRIVE_ID = "18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw"   # README: SoccerNet legibility
DRIVE_URL = f"https://drive.google.com/uc?id={HF_DRIVE_ID}"

# HuggingFace mirror of the SAME checkpoint. Preferred over Drive because Drive
# throttles unauthenticated downloads and serves an HTML quota page instead of
# the file -- which the size check below catches, but only after wasting the
# attempt. The MODEL is unchanged: same ResNet-34, same weights, same 0.5
# threshold, same any-frame tracklet rule. Only the host differs.
HF_REPO = "Ynniss/Legibility_classifier"
HF_FILE = "legibility_resnet34_soccer_20240215.pth"
HF_SHA256 = "b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41"

DEFAULT_THRESHOLD = 0.5      # legibility_classifier.run(threshold=0.5)
DEFAULT_SIZE = 256           # transforms.Resize((256, 256)) -- see docstring
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# helpers.THRESHOLD_FOR_TACK_LEGIBILITY -- a tracklet needs MORE THAN this many
# legible frames. Zero means one frame is enough.
TRACKLET_MIN_LEGIBLE = 0


def detect_arch(sd):
    """resnet34's layer1 has three blocks, resnet18's has two, so a
    `layer1.2.*` key exists only for resnet34. Derived, never assumed."""
    return "resnet34" if any("layer1.2" in k for k in sd) else "resnet18"


def build_model(arch="resnet34", n_out=1):
    """Faithful reconstruction of networks.LegibilityClassifier34.

    The resnet lives under `.model_ft` and forward ENDS IN torch.sigmoid. Two
    consequences worth being deliberate about: the published state_dict's keys
    are prefixed `model_ft.`, so building it this shape lets them load
    STRICTLY -- a real architecture assertion -- and the squash is structural
    rather than bolted on afterwards, so nothing has to decide at runtime
    whether the head emits logits or probabilities.

    A module-level factory so the offline dry run can substitute a tiny
    network and still exercise every rule here without an 85 MB download.
    """
    import torch
    import torch.nn as nn
    import torchvision

    class _Leg(nn.Module):
        def __init__(self):
            super().__init__()
            fn = (torchvision.models.resnet34 if arch == "resnet34"
                  else torchvision.models.resnet18)
            self.model_ft = fn(weights=None)
            self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, n_out)

        def forward(self, x):
            return torch.sigmoid(self.model_ft(x))

    return _Leg()


def resolve_weights(path=None, out="models/sn_legibility.pth"):
    """Local -> attached -> HuggingFace -> Google Drive. Returns (path, source).

    HF is tried first; Drive stays as the fallback so an HF outage or a private
    repo does not strand the run. Both serve the same checkpoint.
    """
    if path and os.path.exists(path):
        return path, f"attached: {path}"
    if os.path.exists(out):
        return out, "already-present"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    errs = []
    try:
        import shutil

        from huggingface_hub import hf_hub_download
        shutil.copyfile(hf_hub_download(HF_REPO, HF_FILE), out)
        source = f"huggingface: {HF_REPO}/{HF_FILE}"
    except Exception as e:
        errs.append(f"huggingface: {type(e).__name__}: {e}")
        try:
            import gdown
            gdown.download(DRIVE_URL, out, quiet=False)
            source = "google-drive (HF fallback)"
        except Exception as e2:
            errs.append(f"google-drive: {type(e2).__name__}: {e2}")
            raise RuntimeError(
                "could not fetch the legibility checkpoint from either source:\n"
                + "\n".join("  * " + s for s in errs) + "\n"
                f"  * Internet must be ON in the Kaggle notebook settings.\n"
                f"  * Or download https://huggingface.co/{HF_REPO}/blob/main/"
                f"{HF_FILE} by hand, attach it as a Kaggle dataset and pass\n"
                f"      --legibility-weights /kaggle/input/<slug>/{HF_FILE}"
            ) from e

    if not os.path.exists(out) or os.path.getsize(out) < 1 << 20:
        raise RuntimeError(f"{out} is missing or implausibly small after "
                           f"download -- Drive probably served an HTML "
                           f"quota page instead of the file.")
    got = _sha256(out)
    if got != HF_SHA256:
        # Reported, not fatal: a legitimate re-upload is possible, but no run
        # may quietly claim a provenance it does not have.
        print(f"[legibility] WARNING: sha256 {got[:16]}... does not match the "
              f"published {HF_SHA256[:16]}... -- a different checkpoint.")
    return out, source


def _sha256(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def strip_prefix(sd):
    """Their checkpoints are bare state_dicts, but a wrapper module changes the
    key prefix. Strip a single common prefix if every key shares one."""
    if not sd:
        return sd
    keys = list(sd)
    if all("." in k for k in keys):
        heads = {k.split(".", 1)[0] for k in keys}
        if len(heads) == 1 and heads.pop() in ("model", "module", "model_ft",
                                               "backbone", "net"):
            return {k.split(".", 1)[1]: v for k, v in sd.items()}
    return sd


def head_out_features(sd):
    """Derive the classifier width from the checkpoint rather than assuming it.
    Returns None when no 2-D final weight can be found."""
    best = None
    for k, v in sd.items():
        shape = tuple(getattr(v, "shape", ()))
        if len(shape) == 2 and k.endswith("weight"):
            best = shape[0]
    return best


class LegibilityClassifier:
    """Frozen binary legibility model. Inference only."""

    def __init__(self, weights=None, device=None, threshold=DEFAULT_THRESHOLD,
                 size=DEFAULT_SIZE, batch=128, force_arch=None,
                 out="models/sn_legibility.pth"):
        import torch
        import torchvision.transforms as T

        self.path, self.source = resolve_weights(weights, out)
        sd = torch.load(self.path, map_location="cpu", weights_only=False)
        if hasattr(sd, "_metadata"):
            del sd._metadata
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # Do NOT strip the prefix here. build_model reconstructs their wrapper,
        # so the published keys (model_ft.*) are exactly what it expects;
        # stripping first would guarantee the strict load fails and silently
        # drop us into the tolerant path. Stripping belongs in the fallback,
        # for a checkpoint saved from the bare backbone instead.
        n_out = head_out_features(sd)
        if n_out is None:
            raise RuntimeError(f"{self.path}: no 2-D head weight found; this "
                               f"does not look like a ResNet classifier "
                               f"state_dict (keys: {list(sd)[:6]})")
        self.n_out = int(n_out)
        self.arch = force_arch or detect_arch(sd)

        m = build_model(self.arch, self.n_out)
        try:
            m.load_state_dict(sd)                      # strict: keys are model_ft.*
            self.missing, self.unexpected, self.strict = [], [], True
        except RuntimeError:
            # a checkpoint saved from the bare backbone: normalise to model_ft.*
            miss, unexp = m.load_state_dict(
                {f"model_ft.{k}": v for k, v in strip_prefix(sd).items()},
                strict=False)
            self.missing, self.unexpected, self.strict = (
                list(miss), list(unexp), False)
            hard = [k for k in self.missing if "fc" not in k]
            if hard:
                raise RuntimeError(
                    f"{self.path}: {len(hard)} backbone tensors missing after "
                    f"load, e.g. {hard[:4]}. Detected arch {self.arch}; the "
                    f"checkpoint does not match it.")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = m.to(self.device).eval()
        self.threshold = float(threshold)
        self.size = int(size)
        self.batch = int(batch)
        self.tf = T.Compose([T.Resize((self.size, self.size)),
                             T.ToTensor(),
                             T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.n_crops = 0
        # Fail-fast probe, and the evidence for the sigmoid claim. forward()
        # ends in sigmoid, so score() must NOT squash again -- head_range is
        # what proves that, rather than an assumption.
        with torch.inference_mode():
            probe = self.model(torch.zeros(2, 3, self.size, self.size,
                                           device=self.device))
        self.head_range = [float(probe.min()), float(probe.max())]
        self.model_squashes = bool(0.0 <= self.head_range[0]
                                   and self.head_range[1] <= 1.0)

    def score(self, crops, keep_inputs=False):
        """[PIL] -> (p_legible array, input tensor batch | None)."""
        import torch
        if not crops:
            return np.zeros(0, np.float64), None
        tensors = [self.tf(c.convert("RGB")) for c in crops]
        out, kept = [], []
        with torch.inference_mode():
            for i in range(0, len(tensors), self.batch):
                xb = torch.stack(tensors[i:i + self.batch]).to(self.device)
                y = self.model(xb).float().reshape(len(xb), -1)
                y = y[:, -1] if y.shape[1] > 1 else y[:, 0]
                # no sigmoid here: build_model's forward already applied it
                out.append(y.cpu().numpy())
                if keep_inputs:
                    kept.append(xb.detach().cpu())
        self.n_crops += len(crops)
        return (np.concatenate(out).astype(np.float64),
                torch.cat(kept) if keep_inputs and kept else None)


def frame_verdicts(probs, threshold=DEFAULT_THRESHOLD):
    """Strictly greater, matching legibility_classifier.run."""
    return np.asarray(probs, float) > float(threshold)


def tracklet_is_legible(probs, threshold=DEFAULT_THRESHOLD,
                        min_legible=TRACKLET_MIN_LEGIBLE):
    """helpers.is_track_legible: legible iff the count of legible frames
    exceeds `min_legible`.

    Note how permissive the published setting is: with min_legible = 0 and
    seventy-odd frames per tracklet, a SINGLE false positive makes the whole
    tracklet legible. That is the right bias for their pipeline, where the cost
    of wrongly declaring a tracklet illegible is losing it entirely. It is
    worth sweeping here, because the audit's failure runs the other way -- 8
    unnumbered tracklets each received a fabricated number.
    """
    return int(frame_verdicts(probs, threshold).sum()) > int(min_legible)


if __name__ == "__main__":
    # ---- threshold semantics: strictly greater, per their run() ------------
    p = np.array([0.0, 0.4999, 0.5, 0.5001, 1.0])
    assert list(frame_verdicts(p)) == [False, False, False, True, True]
    assert list(frame_verdicts(p, 0.0)) == [False, True, True, True, True]

    # ---- tracklet rule: one legible frame is enough at the published setting
    assert tracklet_is_legible([0.9] + [0.01] * 74)
    assert not tracklet_is_legible([0.01] * 75)
    assert not tracklet_is_legible([])
    # ...and raising min_legible makes it stricter, monotonically
    probs = [0.9, 0.8, 0.7] + [0.1] * 20
    prev = True
    for m in range(0, 6):
        cur = tracklet_is_legible(probs, min_legible=m)
        assert not (cur and not prev), m
        prev = cur
    assert tracklet_is_legible(probs, min_legible=2)
    assert not tracklet_is_legible(probs, min_legible=3)

    # ---- state_dict handling ----------------------------------------------
    class _T:
        def __init__(self, *shape):
            self.shape = shape
    assert strip_prefix({"model.fc.weight": _T(1, 512),
                         "model.conv1.weight": _T(64, 3, 7, 7)}) \
        == {"fc.weight": _T(1, 512), "conv1.weight": _T(64, 3, 7, 7)} \
        or True   # identity of _T objects differs; check keys instead
    got = strip_prefix({"model.fc.weight": _T(1, 512),
                        "model.conv1.weight": _T(64, 3, 7, 7)})
    assert sorted(got) == ["conv1.weight", "fc.weight"], sorted(got)
    # a shared prefix that is NOT a known wrapper name must be left alone
    keep = {"layer1.0.weight": _T(64, 64, 3, 3), "layer1.1.weight": _T(64, 64, 3, 3)}
    assert sorted(strip_prefix(keep)) == sorted(keep)

    assert head_out_features({"fc.weight": _T(1, 512),
                              "conv1.weight": _T(64, 3, 7, 7)}) == 1
    assert head_out_features({"fc.weight": _T(2, 512)}) == 2
    assert head_out_features({"conv1.weight": _T(64, 3, 7, 7)}) is None

    # ---- arch detection: layer1.2 exists only in resnet34 ------------------
    assert detect_arch({"model_ft.layer1.2.conv1.weight": _T(64, 64, 3, 3)}) == "resnet34"
    assert detect_arch({"model_ft.layer1.1.conv1.weight": _T(64, 64, 3, 3)}) == "resnet18"
    assert detect_arch({}) == "resnet18"

    # ---- the squash: logits -> probabilities, monotone and centred --------
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
    z = _sig([-6, -1, 0, 1, 6])
    assert abs(z[2] - 0.5) < 1e-12                    # logit 0 -> exactly 0.5
    assert all(z[i] < z[i + 1] for i in range(4))     # monotone
    assert z.min() > 0 and z.max() < 1
    # a logit of 0 lands EXACTLY on the threshold and is therefore illegible
    # under the strictly-greater rule -- the reason auto-detecting whether the
    # head already emits probabilities was unsafe.
    assert not frame_verdicts([0.5])[0]

    # ---- the all-illegible tracklet MUST be illegible -----------------------
    # SNGS-117_11 in the real run: gt=-1, every frame called illegible, and it
    # still produced "1". This is the property that makes that impossible.
    assert not tracklet_is_legible([0.01, 0.02, 0.4, 0.499])
    assert not tracklet_is_legible([0.5] * 20)        # exactly at thr = illegible
    assert tracklet_is_legible([0.01, 0.02, 0.51])    # one legible frame is enough
    # and no "keep the best anyway" escape: the max being high is irrelevant if
    # it never crosses the threshold
    assert not tracklet_is_legible([0.499] * 50)

    # ---- CONSTRUCTION SMOKE TEST ------------------------------------------
    # The pure-function tests above all passed once while __init__ raised
    # NameError, because nothing here had ever built a LegibilityClassifier.
    # This closes that gap: it exercises the real constructor, load, probe and
    # score path against a tiny stub. Skipped where torch is absent, so it
    # costs nothing locally and runs for real on Kaggle.
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("legibility: construction smoke test SKIPPED (no torch here) -- "
              "it runs in the Kaggle venv, where it is the only thing that "
              "exercises __init__")
    else:
        import tempfile
        from PIL import Image

        class _Stub(nn.Module):
            def __init__(self, n_out=1):
                super().__init__()
                self.model_ft = nn.Linear(3, n_out)
                with torch.no_grad():
                    self.model_ft.weight.zero_()
                    self.model_ft.bias.fill_(6.0)

            def forward(self, x):
                return torch.sigmoid(self.model_ft(x.mean(dim=(2, 3))))

        _real_build, _real_resolve = build_model, resolve_weights
        td = tempfile.mkdtemp()
        ck = os.path.join(td, "stub.pth")
        torch.save(_Stub().state_dict(), ck)
        try:
            build_model = lambda arch="resnet34", n_out=1: _Stub(n_out)   # noqa: E731
            resolve_weights = lambda path=None, out=None: (ck, "stub")     # noqa: E731
            clf = LegibilityClassifier(size=256, out=ck)
            assert clf.n_out == 1, clf.n_out
            assert clf.strict is True, "stub keys should load strictly"
            assert clf.size == 256 and clf.threshold == 0.5
            assert 0.0 <= clf.head_range[0] <= clf.head_range[1] <= 1.0, clf.head_range
            assert clf.model_squashes, clf.head_range
            assert clf.model.training is False
            p_, x_ = clf.score([Image.new("RGB", (40, 90), (10, 20, 30)),
                                Image.new("RGB", (55, 110), (200, 210, 220))],
                               keep_inputs=True)
            assert p_.shape == (2,) and p_.min() >= 0.0 and p_.max() <= 1.0, p_
            assert tuple(x_.shape) == (2, 3, 256, 256), tuple(x_.shape)
            assert all(frame_verdicts(p_)), p_          # stub says legible
            assert tracklet_is_legible(p_)
            assert clf.score([])[0].shape == (0,)
        finally:
            build_model, resolve_weights = _real_build, _real_resolve
            __import__('shutil').rmtree(td, ignore_errors=True)
        print("legibility: construction smoke test passed (strict load, "
              "256x256 tensor, probe in [0,1], score path, empty input)")

    print("legibility: all self-tests passed (strict-greater threshold, "
          "any-frame tracklet rule + monotonicity, prefix stripping, "
          "head width and arch derived from the checkpoint, sigmoid squash)")
