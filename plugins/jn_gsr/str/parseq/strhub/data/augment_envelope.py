# Measured-envelope augmentation for jersey-number PARSeq (Option A).
#
# Reproduces the applied constants of the PaddleOCR/PARSeq JN augmentation
# document, Table 3 / Appendix A, verbatim. This is a training-time-only PIL->PIL
# transform inserted into PARSeq's data pipeline in place of the default
# rand_augment_transform() (see strhub/data/module.py get_transform).
#
# OPTION A NOTE (important, and deliberate):
#   The released weakly-labelled SoccerNet LMDB contains *torso* crops (cut from
#   ViTPose output by helpers.generate_crops). We therefore apply the whole
#   envelope here, at the recogniser input, on the torso ROI. The *geometric*
#   framing transforms (anisotropic + isotropic scale, shift, rotation) were
#   calibrated for the PLAYER box in the source document (the +/-8.5% / +/-4.5%
#   shifts are fractions of the player box). Applying them to the torso ROI is
#   the Section-9 single-stage APPROXIMATION. This is the acknowledged cost of
#   Option A; the two-stage split (Option B) removes it by jittering the player
#   crop pre-ViTPose. Nothing here is applied at test time.
#
# Flips are DISABLED unconditionally: a mirrored digit is a different digit.

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

# ----------------------------------------------------------------------------
# Applied constants  ==  Appendix A of the augmentation document (verbatim)
# ----------------------------------------------------------------------------
ISO_SCALE     = (0.90, 1.08)   # isotropic scale (box-relative)
ANISO_W       = (0.85, 1.15)   # independent width scale  -> aspect gap
ANISO_H       = (0.85, 1.15)   # independent height scale -> aspect gap
SHIFT_X       = 0.085          # crop-relative half-range (= 1.3 * 0.066)
SHIFT_Y       = 0.045          # crop-relative half-range (= 1.3 * 0.035); x > y
ROTATE        = (-4.0, 4.0)    # degrees; camera tilt only
DOWNSCALE     = (0.40, 0.90)   # resolution drop -> off-GT boxes + legibility floor
CROP_PAD_FRAC = 0.18           # context padding when cutting the crop
# Flips: DISABLED (mirroring corrupts digit identity).
# NOT a 2.5-sigma truncation: shift is a uniform half-range = 1.3 * sigma_crop.

# Photometric (Section 7.3 step 4: "mild"): small brightness/contrast + HSV jitter.
BRIGHTNESS    = 0.20           # +/- fraction
CONTRAST      = 0.20           # +/- fraction
HUE           = 0.03           # +/- fraction of full hue circle
SATURATION    = 0.20           # +/- fraction

# Blur / compression accompanying the downscale (Section 7.3 step 3).
BLUR_RADIUS   = (0.0, 1.2)     # Gaussian blur radius in px
JPEG_QUALITY  = (40, 90)       # mild JPEG compression range

# Probabilities (each op is independently sampled; the geometric envelope is
# always on, the degradations are stochastic so clean-ish crops still appear).
P_DOWNSCALE   = 0.7
P_BLUR        = 0.5
P_JPEG        = 0.5
P_PHOTOMETRIC = 0.8

# Multi-frame overlay (recogniser-stage, Section 6): TRUE registered fusion of
# several same-tracklet TORSO crops is done by multiframe_overlay() below, invoked
# by the Option B materializer during TRAINING crop generation. It fuses multiple
# frames, so it is NOT a per-image op and deliberately lives outside JerseyEnvelope.


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _context_pad(img, frac):
    """Pad by `frac` of each side so the geometric transform pulls in surrounding
    pixels instead of black borders (the CROP_PAD_FRAC context-pad idea, applied
    to the already-cut crop for Option A).

    Uses EDGE replication rather than reflection: reflection would mirror a digit
    that sits near the crop edge and invent a spurious mirrored digit fragment,
    which contradicts the document's core rule that mirroring corrupts digit
    identity. Edge replication only streaks border pixels outward, never creating
    a flipped digit. On real torso crops the border is jersey fabric, so this is
    a mild, safe fill either way."""
    w, h = img.size
    px, py = int(round(w * frac)), int(round(h * frac))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")
    if padded.shape[2] == 1:
        padded = padded[:, :, 0]
    return Image.fromarray(padded), px, py


def _affine_about_center(img, sx, sy, angle_deg, tx, ty):
    """Single affine: anisotropic+isotropic scale, rotation, translation, about
    the image centre. Uses PIL's inverse-mapping AFFINE. Out-of-range samples
    are unlikely because we operate on a reflect-padded canvas, then centre-crop."""
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)

    # Forward transform F(p) = R*S*(p - c) + c + t. PIL needs the INVERSE
    # mapping (output -> input): p_in = S^-1 R^-1 (p_out - c - t) + c.
    a = cos / sx
    b = sin / sx
    d = -sin / sy
    e = cos / sy
    # inverse translation of (c + t) then + c
    ox = a * (-(cx + tx)) + b * (-(cy + ty)) + cx
    oy = d * (-(cx + tx)) + e * (-(cy + ty)) + cy
    coeffs = (a, b, ox, d, e, oy)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR)


def _geometric(img, rng):
    """Step 1-2 of Section 7.3: anisotropic scale -> isotropic scale + crop-relative
    shift -> rotation. Applied on a reflect-padded canvas, then centre-cropped back."""
    w, h = img.size
    padded, px, py = _context_pad(img, CROP_PAD_FRAC)

    sw = _u(rng, *ANISO_W)
    sh = _u(rng, *ANISO_H)
    iso = _u(rng, *ISO_SCALE)
    sx = sw * iso
    sy = sh * iso
    angle = _u(rng, *ROTATE)
    tx = _u(rng, -SHIFT_X, SHIFT_X) * w    # crop-relative to ORIGINAL width
    ty = _u(rng, -SHIFT_Y, SHIFT_Y) * h

    out = _affine_about_center(padded, sx, sy, angle, tx, ty)
    # centre-crop back to original size
    W, H = out.size
    left = (W - w) // 2
    top = (H - h) // 2
    return out.crop((left, top, left + w, top + h))


def _downscale_blur_jpeg(img, rng):
    """Step 3: random downscale-then-up (resolution floor), plus mild blur and JPEG."""
    w, h = img.size
    if rng.random() < P_DOWNSCALE:
        f = _u(rng, *DOWNSCALE)
        small = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    if rng.random() < P_BLUR:
        img = img.filter(ImageFilter.GaussianBlur(_u(rng, *BLUR_RADIUS)))
    if rng.random() < P_JPEG:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(_u(rng, *JPEG_QUALITY)))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def _photometric(img, rng):
    """Step 4: mild brightness/contrast + HSV (hue/saturation) jitter. No flips."""
    if rng.random() >= P_PHOTOMETRIC:
        return img
    img = ImageEnhance.Brightness(img).enhance(1.0 + _u(rng, -BRIGHTNESS, BRIGHTNESS))
    img = ImageEnhance.Contrast(img).enhance(1.0 + _u(rng, -CONTRAST, CONTRAST))
    img = ImageEnhance.Color(img).enhance(1.0 + _u(rng, -SATURATION, SATURATION))
    # hue shift via HSV roll
    hsv = np.asarray(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(_u(rng, -HUE, HUE) * 255)) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


class JerseyEnvelope:
    """Measured-envelope transform. Callable PIL.Image -> PIL.Image.

    Order (Section 7.3): geometric framing -> downscale/blur/jpeg -> photometric.
    The final aspect-preserving resize is left to PARSeq's own T.Resize that
    follows this transform in get_transform().
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        img = img.convert("RGB")
        img = _geometric(img, self.rng)
        img = _downscale_blur_jpeg(img, self.rng)
        img = _photometric(img, self.rng)
        return img


def build_envelope_transform(seed=None):
    """Factory used by the patched get_transform() (Option A, single-stage)."""
    return JerseyEnvelope(seed=seed)


# ---------------------------------------------------------------------------
# Two-stage split (Option B). Same constants/internals as JerseyEnvelope, but
# the two halves are exposed separately so the geometric framing can act on the
# PLAYER crop (pre-ViTPose) and the recogniser-stage degradations on the TORSO
# ROI (at the PARSeq input), exactly as Table 3 intends.
# ---------------------------------------------------------------------------
def apply_geometric(img, rng=None):
    """Stage 1: anisotropic+isotropic scale, crop-relative shift, +/-4 deg rotation
    on the PLAYER crop (edge-padded, then centre-cropped). Pre-ViTPose."""
    if rng is None:
        rng = np.random.default_rng()
    return _geometric(img.convert("RGB"), rng)


def apply_recognizer_stage(img, rng=None):
    """Stage 2: resolution downscale+blur+jpeg + photometric jitter on the TORSO ROI,
    at the PARSeq input. Flips never applied. (Multi-frame overlay is the separate
    recogniser-stage op multiframe_overlay(), because it fuses several frames.)"""
    if rng is None:
        rng = np.random.default_rng()
    img = img.convert("RGB")
    img = _downscale_blur_jpeg(img, rng)
    img = _photometric(img, rng)
    return img


def multiframe_overlay(torsos, rng=None, size=None):
    """Multi-frame overlay / input-level fusion (recogniser-stage, Section 6; the
    2023 winner's trick). Register a window of same-tracklet TORSO crops to a common
    frame and average them into ONE composite, so the recogniser sees the number
    fused across frames. Registration here is by the torso box: every input is a
    ViTPose torso ROI of the SAME player's number, so resizing them to a common size
    aligns the digit region (the document's 'register each frame's patch by its torso
    box' option) -- no separate homography needed. Digits stay legible because the
    number sits in the same part of every torso ROI; per-frame motion averages out.

    torsos : list of PIL torso crops from ONE tracklet (>=2 to actually fuse).
    size   : common registration size; default = per-window median torso size.
    Returns one composite PIL.Image (or the single crop unchanged if only one given)."""
    if rng is None:
        rng = np.random.default_rng()
    imgs = [t.convert("RGB") for t in torsos if t is not None]
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    if size is None:
        ws = sorted(im.size[0] for im in imgs)
        hs = sorted(im.size[1] for im in imgs)
        size = (max(8, ws[len(ws) // 2]), max(8, hs[len(hs) // 2]))   # median = registration target
    acc = np.zeros((size[1], size[0], 3), dtype=np.float32)
    for im in imgs:
        acc += np.asarray(im.resize(size, Image.BILINEAR), dtype=np.float32)
    comp = (acc / len(imgs)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(comp, "RGB")
