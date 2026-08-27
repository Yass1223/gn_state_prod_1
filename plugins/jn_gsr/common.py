# Shared pipeline helpers -- ONE code path for train & test, envelope off & on,
# so the two phases (and the two arms, when the A/B is run) are consistent by
# construction. The notebook and any later challenge-eval script import from here.
#
# CHANGED for the DBNet++ pipeline:
#   * padded_torso_crop (ViTPose) -> padded_roi_crop (DBNet++ number quad).
#     ViTPose is gone entirely; pose_vitpose_hf.py is not part of this pipeline.
#
# A LEGIBILITY GATE **IS** IN THE PATH. An earlier revision of this header said
# it was not, and that the detector score was the frame-inclusion decision.
# That is wrong and was corrected in place. Frame inclusion is CONJUNCTIVE:
#     legibility.LegibilityClassifier -> p_legible > 0.72  (frame_verdicts)
#   AND dbnet_infer.DetectorGate      -> top-quad ds > 0.52
# run_eval.py applies both at merge; jn_recognizer.py applies both before
# PARSeq reads. Same semantics, both strict.
#
# WHAT IN THIS FILE IS NOT IN THE PATH (confirmed by repository-wide search --
# every reference is inside this file or its own self-tests):
#   * class LegibilityGate and LEG_THRESHOLD = 0.5. The production classifier
#     is legibility.LegibilityClassifier, whose threshold comes from
#     --legibility-thr / cfg.legibility_thr. This class is a separate ResNet-34
#     wrapper kept only so an audit can compare the two implementations; its
#     0.5 is the upstream published default and is unrelated to the 0.72 gate.
#   * padded_roi_crop. The production ROI path is
#     dbnet_infer.DetectorGate.roi_crop_scored, which also caches detections
#     and returns the top-quad score. padded_roi_crop is retained because its
#     self-test below is the only coverage of the 18%-pad -> detect -> ROI
#     wiring that does not need mmocr.
# Both are RETAINED, not dead-by-accident. Delete them only together with the
# self-tests that exercise them.
#
# Torch imports are lazy (inside functions) so the pure-logic helpers import and
# test without a GPU. The model-dependent pieces are exercised on the Studio.

import numpy as np
from gsr_adapter import pad_box   # single source of the 18% pad

SEED = 0
# NOT the production gate. Upstream's published default for the LegibilityGate
# class below, which is not in the path. The gate that runs is
# --legibility-thr / cfg.legibility_thr (0.72). See the header.
LEG_THRESHOLD = 0.5
LEG_PAD = 0.18
FRAME_STRIDE = 5            # subsample every Nth legible frame (budget + near-dup removal)
OVERLAY_WINDOW = 3          # frames fused per multi-frame-overlay composite (train, on-arm)
OVERLAY_PER_TRACK = 2       # multi-frame-overlay composites added per tracklet (train, on-arm)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHARSET = "0123456789"     # digits-only recogniser


# ----------------------------- determinism ---------------------------------
def seed_everything(seed=SEED):
    """Seed python/numpy/torch identically in every run and both arms (fairness)."""
    import os, random
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ----------------------------- frame subsampling ---------------------------
def subsample(seq, stride=FRAME_STRIDE):
    """Every `stride`-th element (>=1 kept). Consecutive frames are near-identical,
    so this is the main budget lever and barely costs accuracy. Applied identically
    when selecting train frames and test frames."""
    stride = max(1, int(stride))
    return list(seq)[::stride]


# ----------------------------- framing-parity ROI crop ---------------------
def padded_roi_crop(detector, frame_pil, box_xywh, frac=LEG_PAD,
                    thr=None, pad_frac=None):
    """Number ROI: pad the PLAYER box by 18% (real frame pixels), then DBNet++ ->
    highest-scoring quad -> AABB -> glyph-height padding -> integer extraction.
    Deterministic -- no augmentation -- so it is legitimate at test; it removes the
    train/test framing skew (train ROI came from a padded box, so test must too).
    Returns (roi_pil | None, source) with source in {'det', 'none'}.

    NOT the production path -- dbnet_infer.DetectorGate.roi_crop_scored is,
    because it memoises detections and returns the top-quad score the merge
    gates on. Kept because its self-test below is the only mmocr-free coverage
    of this wiring. 'none' here means "no quad above thr", which is the
    DETECTOR's verdict; the legibility gate is a separate, conjunctive test
    applied by the caller."""
    from roi_dbnet import DET_THR, PAD_FRAC, roi_from_detections
    W, H = frame_pil.size
    x, y, w, h = box_xywh
    x0, y0, x1, y1 = pad_box(x, y, w, h, W, H, frac)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, "none"
    crop = frame_pil.crop((x0, y0, x1, y1))
    quads, scores = detector.detect(crop)
    return roi_from_detections(crop, quads, scores,
                               thr=DET_THR if thr is None else thr,
                               pad_frac=PAD_FRAC if pad_frac is None else pad_frac)


# ------------- alternate legibility wrapper (NOT in the path) ---------------
# Not called anywhere outside this file. The production legibility gate is
# legibility.LegibilityClassifier + legibility.frame_verdicts at
# --legibility-thr (0.72); this is a second ResNet-34 wrapper retained only so
# an audit can compare the two implementations on the same crops. Its default
# `thr` is LEG_THRESHOLD = 0.5, which is upstream's number, NOT the gate.
class LegibilityGate:
    """ResNet34 soccer legibility (Koshkina) -- exact arch + 256x256 + ImageNet norm.
    Same instance/threshold used to select train frames (cell 5) and to gate test
    frames (cell 8), so 'which frames count' is identical on both sides."""

    def __init__(self, weights_path, device=None, thr=LEG_THRESHOLD):
        import torch, torch.nn as nn
        from torchvision import models, transforms as T
        self.torch, self.thr = torch, thr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                             T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        class LegibilityClassifier34(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_ft = models.resnet34(weights=None)
                self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)
            def forward(self, x):
                return torch.sigmoid(self.model_ft(x))

        sd = torch.load(weights_path, map_location=self.device)
        if hasattr(sd, "_metadata"):
            del sd._metadata
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}  # defensive strip
        self.model = LegibilityClassifier34()
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device).eval()

    def score(self, frame_path, box_xywh, pad=LEG_PAD):
        from PIL import Image
        torch = self.torch
        img = Image.open(frame_path).convert("RGB"); W, H = img.size
        x0, y0, x1, y1 = pad_box(*box_xywh, W, H, pad)
        crop = img.crop((x0, y0, x1, y1))
        with torch.no_grad():
            return self.model(self.tf(crop).unsqueeze(0).to(self.device)).item()

    def is_legible(self, frame_path, box_xywh, pad=LEG_PAD):
        return self.score(frame_path, box_xywh, pad) > self.thr


# ----------------------------- PARSeq reader (hardened) --------------------
def _charset_indices(tokenizer):
    """Resolve digit->column and EOS index across PARSeq tokenizer variants. Raises a
    clear error rather than silently mis-indexing (this WAS a verify-on-Studio spot)."""
    stoi = getattr(tokenizer, "_stoi", None)
    if isinstance(stoi, dict) and all(c in stoi for c in CHARSET):
        digit_idx = [stoi[c] for c in CHARSET]
    else:
        charset = getattr(tokenizer, "charset", None) or getattr(tokenizer, "_itos", None)
        if charset is None:
            raise AttributeError("PARSeq tokenizer exposes neither _stoi nor charset/_itos; "
                                 "inspect the tokenizer and set the digit index map explicitly.")
        idx = {c: i for i, c in enumerate(charset)}
        if not all(c in idx for c in CHARSET):
            raise ValueError(f"charset {charset!r} does not contain all of {CHARSET!r}")
        digit_idx = [idx[c] for c in CHARSET]
    eos = getattr(tokenizer, "eos_id", None)
    if eos is None:
        eos = getattr(tokenizer, "_eos_id", 0)     # PARSeq convention: EOS is index 0
    return digit_idx, eos


def build_parseq_batch_reader(str_model, device=None, batch=64):
    """v2: batched variant of build_parseq_reader. read_many([crops]) ->
    [(tens_logl[11], units_logl[11])], numerically identical to per-crop reads
    (the transform and marginalisation are the same; only the forward is
    batched). Empty input -> []."""
    import torch
    from torchvision import transforms as T
    device = device or next(str_model.parameters()).device
    img_h, img_w = str_model.hparams.img_size
    tf = T.Compose([T.Resize((img_h, img_w), T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(0.5, 0.5)])
    digit_idx, eos = _charset_indices(str_model.tokenizer)
    cols = list(digit_idx) + [eos]

    @torch.no_grad()
    def read_many(crops):
        out = []
        for i in range(0, len(crops), batch):
            x = torch.stack([tf(c.convert("RGB")) for c in crops[i:i + batch]])
            probs = str_model(x.to(device)).softmax(-1)          # [B, L, C]
            m = probs[:, :2, :][:, :, cols].double().cpu().numpy()  # [B, 2, 11]
            logl = np.log(m + 1e-9)
            out.extend((logl[b, 0], logl[b, 1]) for b in range(len(logl)))
        return out
    return read_many


def build_parseq_reader(str_model, device=None):
    """Return read(crop) -> (tens_logl[11], units_logl[11]) over [0-9, EOL], using the
    model's own val transform (resize+normalise, NO augmentation). Validated against
    the actual loaded checkpoint's tokenizer at call time."""
    import torch
    from torchvision import transforms as T
    device = device or next(str_model.parameters()).device
    img_h, img_w = str_model.hparams.img_size
    tf = T.Compose([T.Resize((img_h, img_w), T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(0.5, 0.5)])
    digit_idx, eos = _charset_indices(str_model.tokenizer)

    @torch.no_grad()
    def read(crop):
        x = tf(crop.convert("RGB")).unsqueeze(0).to(device)
        probs = str_model(x).softmax(-1)[0]           # [L, C]
        def marg(pos):
            v = np.array([probs[pos, i].item() for i in digit_idx] + [probs[pos, eos].item()])
            return np.log(v + 1e-9)
        return marg(0), marg(1)
    return read


if __name__ == "__main__":
    # pure-logic tests (no torch needed)
    assert subsample(list(range(10)), 5) == [0, 5]
    assert subsample([1, 2, 3], 1) == [1, 2, 3]
    assert subsample([1, 2, 3], 0) == [1, 2, 3]          # guarded to stride>=1
    assert subsample(range(20), 7) == [0, 7, 14]
    assert pad_box(100, 100, 100, 100, 1000, 1000) == (82, 82, 218, 218)

    # padded_roi_crop wiring: 18% player pad, then the detector, then the ROI rules.
    # Stubbed detector -> no mmocr / GPU needed here.
    import math
    from PIL import Image

    def _rq(cx, cy, w, h, deg):
        t = math.radians(deg); c, s = math.cos(t), math.sin(t)
        pts = [(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]
        return np.array([(cx + x*c - y*s, cy + x*s + y*c) for x, y in pts], np.float32)

    class _Stub:
        def __init__(self, sc): self.sc = sc; self.seen = None
        def detect(self, img): self.seen = img.size; return [_rq(60, 70, 34, 24, 6)], [self.sc]

    frame = Image.fromarray(np.zeros((1000, 1000, 3), np.uint8))
    d_hi = _Stub(0.9)
    roi, src = padded_roi_crop(d_hi, frame, (100, 100, 100, 100))
    assert src == "det" and roi is not None
    assert d_hi.seen == (136, 136), f"detector must see the 18%-padded player crop, got {d_hi.seen}"
    roi2, src2 = padded_roi_crop(_Stub(0.4), frame, (100, 100, 100, 100))
    assert (roi2, src2) == (None, "none"), "below 0.52 must be 'none'"
    # degenerate player box -> 'none', never an exception
    assert padded_roi_crop(_Stub(0.9), frame, (0, 0, 1, 1))[1] == "none"
    print("common: pure-logic tests passed (subsample, pad re-export, "
          "padded_roi_crop wiring + 0.52 verdict)")
