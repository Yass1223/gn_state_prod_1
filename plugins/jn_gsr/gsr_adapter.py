# GSR-2024 -> jersey-number-pipeline adapter.
#
# Reconstructs sn-jersey-style tracklets from GSR: groups player/GK boxes by GT
# track id, attaches the tracklet's jersey label, produces 18%-padded crops, and
# IoU-matches detector boxes to GT for the detector-crop test population.
#
# DEFENSIVE TYPING (hard lesson from GSR's JSON): fields are loosely typed --
# `num_tracklets` is a STRING, `track_id` may be int or str, `jersey` may be int,
# str, or null. Every read is coerced here (`_to_int`, `_norm_label`, `_tid`) so a
# schema quirk can't silently break grouping, sizing, or labels downstream.

import json
from pathlib import Path

CROP_PAD_FRAC = 0.18
ROLES = {"player", "goalkeeper"}


# ----------------------------- type coercion -------------------------------
def _to_int(v, default=None):
    """Coerce a possibly-stringy numeric to int; default on failure."""
    try:
        return int(str(v).strip())
    except (ValueError, TypeError, AttributeError):
        return default


def has_number(j):
    """True iff a real jersey number is present. null/-1/''/'null' -> False (-> -1 class)."""
    if j in (None, "", "null", "None"):
        return False
    return _to_int(j, default=-1) not in (-1, None)


def _norm_label(j):
    """Normalise a jersey value to its canonical digit string, or '-1'."""
    return str(_to_int(j)) if has_number(j) else "-1"


def jersey_label(attr):
    return _norm_label(attr.get("jersey"))


def _tid(seq_name, track_id):
    """Namespaced, string track id (avoids int/str mismatch when grouping)."""
    return f"{seq_name}_{str(track_id)}"


# ----------------------------- geometry ------------------------------------
def bbox_image_to_xywh(b):
    """GSR 'bbox_image' -> (x, y, w, h) top-left (floats). Raises on unknown keys so
    a wrong schema surfaces loudly rather than producing offset crops."""
    if all(k in b for k in ("x", "y", "w", "h")):
        return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
    if all(k in b for k in ("x_topleft", "y_topleft", "w", "h")):
        return float(b["x_topleft"]), float(b["y_topleft"]), float(b["w"]), float(b["h"])
    raise KeyError(f"Unexpected bbox_image keys: {sorted(b.keys())}")


def pad_box(x, y, w, h, W, H, frac=CROP_PAD_FRAC):
    """18% context pad, clipped to the frame. Single source of the pad used at both
    train and test (imported by common.padded_torso_crop)."""
    px, py = w * frac, h * frac
    x0 = max(0, int(round(x - px))); y0 = max(0, int(round(y - py)))
    x1 = min(W, int(round(x + w + px))); y1 = min(H, int(round(y + h + py)))
    return x0, y0, x1, y1


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def match_to_gt(det_boxes, gt_entries, thr=0.5):
    """Greedy IoU match of detector boxes to GT entries; label transferred from GT
    for scoring only. Returns [(det_box, gt_entry), ...]."""
    pairs, used = [], set()
    for db in det_boxes:
        best, best_i = None, thr
        for i, g in enumerate(gt_entries):
            if i in used:
                continue
            v = iou(db, g["xywh"])
            if v >= best_i:
                best, best_i = i, v
        if best is not None:
            used.add(best)
            pairs.append((db, gt_entries[best]))
    return pairs


# ----------------------------- extraction ----------------------------------
def find_sequences(root, split_filter=None):
    paths = sorted(Path(root).rglob("Labels-GameState.json"))
    return [p for p in paths if not split_filter
            or f"/{split_filter.lower()}/" in str(p).lower().replace("\\", "/")]


def count_tracklets(seq_jsons):
    """Sum info.num_tracklets across clips, coercing the STRING field to int."""
    return sum(_to_int(json.loads(Path(j).read_text()).get("info", {}).get("num_tracklets"), 0)
               for j in seq_jsons)


def build_pool(seq_jsons):
    """Group player/GK detections by GT track id -> {tid: {'number':str,'frames':[...]}}."""
    tracks = {}
    for jp in seq_jsons:
        seq_dir = Path(jp).parent
        data = json.loads(Path(jp).read_text())
        id2path = {}
        for img in data.get("images", []):
            iid = img.get("image_id", img.get("id"))
            fn = img.get("file_name", "")
            cand = seq_dir / "img1" / Path(fn).name
            id2path[iid] = cand if cand.exists() else (seq_dir / fn)
        for a in data.get("annotations", []):
            attr = a.get("attributes", {}) or {}
            if attr.get("role") not in ROLES or "bbox_image" not in a:
                continue
            try:
                xywh = bbox_image_to_xywh(a["bbox_image"])
            except KeyError:
                continue
            tid = _tid(seq_dir.name, a.get("track_id"))
            fp = id2path.get(a.get("image_id"))
            if fp is None:
                continue
            t = tracks.setdefault(tid, {"number": jersey_label(attr), "frames": []})
            t["frames"].append({"frame_path": str(fp), "xywh": xywh})
    return tracks


if __name__ == "__main__":
    # ---- type-coercion + geometry unit tests (no download) ----
    assert _to_int("22") == 22 and _to_int(7) == 7 and _to_int("x", -1) == -1 and _to_int(None, 0) == 0
    assert has_number("23") and has_number(7) and not has_number("-1") and not has_number(None)
    assert _norm_label("07") == "7" and _norm_label(23) == "23" and _norm_label(None) == "-1"  # normalised
    assert jersey_label({"jersey": "9"}) == "9" and jersey_label({"jersey": None}) == "-1"
    assert _tid("SNGS-108", 3) == "SNGS-108_3" and _tid("SNGS-108", "3") == "SNGS-108_3"       # int==str

    assert bbox_image_to_xywh({"x": 1, "y": 2, "w": 3, "h": 4}) == (1, 2, 3, 4)
    assert bbox_image_to_xywh({"x_topleft": 5, "y_topleft": 6, "w": 7, "h": 8}) == (5, 6, 7, 8)
    assert pad_box(100, 100, 100, 100, 1000, 1000) == (82, 82, 218, 218)
    assert pad_box(10, 10, 100, 100, 1000, 1000) == (0, 0, 128, 128)
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - (50 / 150)) < 1e-9

    gt = [{"xywh": (0, 0, 10, 20)}, {"xywh": (100, 0, 10, 20)}]
    dets = [(1, 1, 10, 20), (300, 300, 5, 5)]
    matched = match_to_gt(dets, gt, thr=0.5)
    assert len(matched) == 1 and matched[0][1] is gt[0]

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "Labels-GameState.json")
        json.dump({"info": {"num_tracklets": "22"}, "images": [], "annotations": []}, open(p, "w"))
        assert count_tracklets([p]) == 22
    print("gsr_adapter: all type-coercion + geometry tests passed")
