"""Team-appearance model (``osnet_team``) used by the role/team stage.

The model of ``role-team-gt-tracks-3.ipynb`` (cell 3): an OSNet x1.0 backbone,
global average pooling, the backbone's own ``fc`` block, a BNNeck and a bias-free
512 -> ``emb_dim`` projection, L2-normalised. Checkpoint ``osnet_team_best.pt``
from ``Ynniss/osnet_team``: a ``torch.save`` dict with ``config`` (``img_h``,
``img_w``, ``emb_dim``, ``mean``, ``std``, ``letterbox``), ``state_dict`` and
``test_metrics``. Input size, embedding width and normalisation are read from
``config``, so nothing here has to agree with a hardcoded constant; the notebook
printed 128x64 -> 256-d.

Preprocessing is the notebook's ``letterbox``: the RGB crop is resized with PIL
bilinear filtering to fit 128x64 preserving its aspect ratio, centred on a canvas
filled with grey 110, scaled to [0, 1] and normalised with the checkpoint's
mean/std. Inference is the notebook's ``embed_osnet``: the crop and its horizontal
flip are embedded, the two vectors summed and L2-normalised (flip test-time
augmentation). Arithmetic is fp32, as in the notebook (the model is 9.5 MB and
embeds at most 16 crops per tracklet, so there is nothing to gain from fp16).

Crops are RGB. ``tracklab.utils.cv2.cv2_load_image`` returns RGB; unlike the
OSNet-AIN path (which converts to BGR because its own preprocessing converts
back), the frame is used as loaded.

Backbone factory: ``sn_gamestate.reid.osnet_ain.build_backbone("osnet_x1_0")``,
i.e. the ``torchreid`` fork this project installs (VlSomers/bpbreid). Its
``osnet.py`` is the file the notebook downloaded from KaiyangZhou/deep-person-reid
apart from two cosmetic lines (a docstring year and a ``self.feature_dim``
attribute), verified by diff; ``osnet_x1_0(num_classes=1, pretrained=False,
loss="softmax")`` therefore builds the same module and the notebook's state dict
loads with no missing and no unexpected tensor — asserted below, as the notebook
asserted it.

Provenance: the resolved file's sha256 is computed and reported (``info``) so the
run audit records which weights produced the embeddings. ``team_sha256`` in the
config, when set, is enforced; when null the digest is logged and recorded but not
enforced. Pin it after the first verified run.
"""
import hashlib
import inspect
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from sn_gamestate.reid.osnet_ain import build_backbone

log = logging.getLogger(__name__)

REPO_ID = "Ynniss/osnet_team"
FILENAME = "osnet_team_best.pt"
BACKBONE = "osnet_x1_0"
GREY = 110
REQUIRED_KEYS = ("state_dict",)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(repo_id=REPO_ID, filename=FILENAME, revision=None,
                       expected_sha256=None, local_path=None):
    """Path to the checkpoint: ``local_path`` if given, else the Hub file.
    Returns (path, sha256, source). Enforces the digest when one is expected."""
    if local_path:
        path = Path(local_path)
        if not path.is_file():
            raise RuntimeError(f"team_local_path does not exist: {path}")
        source = str(path)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
        source = f"{repo_id}/{filename}@{revision or 'main'}"
    digest = sha256(path)
    if expected_sha256 and digest != str(expected_sha256):
        raise RuntimeError(f"{source} digest mismatch\n  expected {expected_sha256}\n  got      {digest}")
    log.info(f"[osnet_team] {source} ({path.stat().st_size / 1e6:.1f} MB), sha256 {digest[:16]}"
             + ("" if expected_sha256 else " (not pinned)"))
    return path, digest, source


def _torch_load(path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def load_checkpoint(path):
    ck = _torch_load(path)
    if not isinstance(ck, dict):
        raise RuntimeError(f"{path} holds {type(ck).__name__}, expected a dict")
    missing = [k for k in REQUIRED_KEYS if k not in ck]
    if missing:
        raise RuntimeError(f"{path} is missing {missing}; its keys are {sorted(ck)[:12]}")
    if not ck["state_dict"]:
        raise RuntimeError(f"{path} has an empty 'state_dict'")
    if not isinstance(ck.get("config", {}), dict):
        raise RuntimeError(f"{path}: 'config' is {type(ck['config']).__name__}, expected a dict")
    return ck


class TeamNet(nn.Module):
    """backbone -> GAP -> backbone.fc -> BNNeck -> proj -> L2 (the notebook's TeamEmbedder)."""

    def __init__(self, backbone, feat_dim: int, emb_dim: int):
        super().__init__()
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.proj = nn.Linear(feat_dim, emb_dim, bias=False)

    def forward(self, x):
        f = self.backbone.featuremaps(x)
        v = nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        if self.backbone.fc is not None:
            v = self.backbone.fc(v)
        return nn.functional.normalize(self.proj(self.bnneck(v)), dim=1)


class OsnetTeam:
    """RGB crops in, L2-normalised float32 team descriptors out (flip TTA, fp32)."""

    def __init__(self, checkpoint: dict, device, batch_size: int = 128, provenance: dict = None):
        cfg = checkpoint.get("config", {}) or {}
        state = checkpoint["state_dict"]
        self.height, self.width = int(cfg.get("img_h", 128)), int(cfg.get("img_w", 64))
        self.dim = int(cfg.get("emb_dim", 256))
        self.mean = np.asarray(cfg.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(cfg.get("std", [0.229, 0.224, 0.225]), np.float32)
        if not cfg.get("letterbox", True):
            raise RuntimeError("checkpoint config says letterbox=False; the notebook asserted letterbox")
        if "bnneck.weight" not in state or "proj.weight" not in state:
            raise RuntimeError(f"state_dict lacks bnneck/proj tensors; keys begin {sorted(state)[:8]}")
        feat_dim = int(state["bnneck.weight"].shape[0])
        emb_dim = int(state["proj.weight"].shape[0])
        if emb_dim != self.dim:
            raise RuntimeError(f"config emb_dim {self.dim} != proj.weight rows {emb_dim}")
        backbone, factory = build_backbone(BACKBONE)
        net = TeamNet(backbone, feat_dim, emb_dim)
        missing, unexpected = net.load_state_dict(state, strict=False)
        # The notebook asserted both empty against the same architecture.
        if missing or unexpected:
            raise RuntimeError(f"osnet_team state_dict mismatch: {len(missing)} missing "
                               f"{list(missing)[:5]}, {len(unexpected)} unexpected {list(unexpected)[:5]}")
        self.net = net.to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)
        self.info = dict(backbone=BACKBONE, factory=factory, input_hw=[self.height, self.width],
                         embedding_dim=self.dim, precision="fp32", flip_tta=True,
                         state_tensors=len(state))
        self.info.update(provenance or {})
        log.info(f"[osnet_team] {BACKBONE} via {factory} @ {self.height}x{self.width}, "
                 f"{self.dim}-d, {len(state)} tensors, fp32 + flip TTA")

    def letterbox(self, img: np.ndarray) -> np.ndarray:
        """Notebook ``letterbox``: PIL bilinear resize into a grey-110 canvas, normalise, CHW."""
        h, w = img.shape[:2]
        s = min(self.height / h, self.width / w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ox, oy = (self.width - nw) // 2, (self.height - nh) // 2
        canvas = np.full((self.height, self.width, 3), GREY, np.uint8)
        canvas[oy:oy + nh, ox:ox + nw] = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        x = ((canvas.astype(np.float32) / 255.0) - self.mean) / self.std
        return x.transpose(2, 0, 1)

    @torch.no_grad()
    def embed(self, crops) -> np.ndarray:
        """(N, dim) float32, L2-normalised; ``crops`` are RGB uint8 HxWx3 arrays."""
        if len(crops) == 0:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(crops), self.batch_size):
            xs = np.stack([self.letterbox(c) for c in crops[i:i + self.batch_size]])
            xt = torch.from_numpy(xs).to(self.device)
            e = self.net(xt) + self.net(torch.flip(xt, dims=[3]))     # flip TTA as the notebook
            out.append(nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def crop_rgb(img: np.ndarray, box_ltwh) -> np.ndarray:
    """Notebook ``crop`` on an ltwh box: rounded, clamped to the frame, at least 2 px."""
    l, t, w, h = [float(v) for v in box_ltwh]
    x1, y1, x2, y2 = [int(round(v)) for v in (l, t, l + w, t + h)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], max(x2, x1 + 2)), min(img.shape[0], max(y2, y1 + 2))
    return img[y1:y2, x1:x2]


def from_config(cfg, device, batch_size: int = 128) -> OsnetTeam:
    repo_id = str(getattr(cfg, "team_repo", REPO_ID))
    filename = str(getattr(cfg, "team_file", FILENAME))
    revision = getattr(cfg, "team_revision", None) or None
    expected = getattr(cfg, "team_sha256", None) or None
    local_path = getattr(cfg, "team_local_path", None) or None
    path, digest, source = resolve_checkpoint(repo_id, filename, revision, expected, local_path)
    ck = load_checkpoint(path)
    try:
        worst5 = [w[0] for w in (ck.get("test_metrics", {}) or {}).get("worst5", [])]
    except (TypeError, IndexError, KeyError):
        worst5 = []
    provenance = dict(sha256=digest, source=source, sha256_pinned=bool(expected),
                      checkpoint_worst5=worst5)
    return OsnetTeam(ck, device, batch_size=batch_size, provenance=provenance)
