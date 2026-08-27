# Evaluation harness for the envelope A/B (Option B).
#
# Reports, per cell (dataset x crop-source x arm): image accuracy, 1-NED, tracklet
# accuracy split TWO ways (all-tracklets-incl-`-1`, and numbered-only), the `-1`
# decision's precision/recall/F1, a single- vs double-digit accuracy breakdown, and
# FPS. Consolidation follows Koshkina's per-digit-position log-likelihood aggregation.
#
# The `-1` fix (vs the earlier dead `-850` threshold): `-1` is decided by the
# LEGIBILITY GATE, exactly as the document states ("the absence of any sufficiently
# confident legible reading across the whole tracklet maps to -1"). A tracklet with
# no legible frames -> `-1` (its frame list is empty here). An OPTIONAL average
# per-frame confidence floor (`min_conf`, off by default) can also force `-1`; its
# threshold is meant to be calibrated on VAL via `sweep_min_conf`, not guessed.

import time
import numpy as np

ANCHORS = {"koshkina_soccer_tracklet": 87.4, "winner_2023": 92.85}
_TOK = "0123456789E"   # 10 digits then 'E' (end-of-label), per helpers.py


# ------------------------------- string metrics ----------------------------
def edit_distance(a, b):
    a, b = str(a), str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normed_edit_similarity(pred, gt):
    d = edit_distance(pred, gt)
    m = max(len(str(pred)), len(str(gt)), 1)
    return 1.0 - d / m


def image_metrics(preds, gts):
    """Per-image accuracy + 1-NED over equal-length lists of number strings."""
    assert len(preds) == len(gts) and preds, "empty or mismatched"
    n = len(preds)
    acc = sum(p == g for p, g in zip(preds, gts)) / n
    ned1 = sum(normed_edit_similarity(p, g) for p, g in zip(preds, gts)) / n
    return {"accuracy": acc, "one_minus_ned": ned1, "n_images": n}


# ------------------------------- consolidation + -1 ------------------------
def consolidate_tracklet(frames_tens_logl, frames_units_logl, min_conf=None):
    """Koshkina per-position aggregation: sum per-digit log-likelihoods across the
    tracklet's legible frames SEPARATELY for tens and units, argmax each, read the
    number ('E' in a position ends it -> single digit). `-1` when there are no
    legible frames, OR (if `min_conf` is set) when the average per-frame log-prob of
    the chosen tokens is below `min_conf`. Returns (label, avg_conf)."""
    if not frames_tens_logl:
        return "-1", float("-inf")                      # gate-driven -1 (no legible frames)
    st = np.sum(np.asarray(frames_tens_logl, dtype=float), axis=0)
    su = np.sum(np.asarray(frames_units_logl, dtype=float), axis=0)
    ti, ui = int(np.argmax(st)), int(np.argmax(su))
    tok = _TOK[ti] + _TOK[ui]
    chosen = [st[ti], su[ui]]
    for i in range(2):                                  # 'E' ends the number
        if tok[i] == "E":
            tok, chosen = tok[:i], chosen[:i]
            break
    n_frames = len(frames_tens_logl)
    avg_conf = (sum(chosen) / len(chosen)) / n_frames if chosen else float("-inf")  # per-frame, per-pos
    if tok == "":
        return "-1", avg_conf
    if min_conf is not None and avg_conf < min_conf:
        return "-1", avg_conf                           # confidence-driven -1 (optional)
    return tok, avg_conf


def compare_box_reader(results, metric="tracklet_accuracy"):
    """The 2x2 evaluation matrix: {GT boxes, YOLO boxes} x {PARSeq fine-tuned,
    PARSeq zero-shot Koshkina}, on one metric, with its three derived quantities:

      fine-tune gain, per box source   (ft - zs)      what GSR fine-tuning buys
      detection cost, per reader       (GT - YOLO)    what imperfect boxes cost
      interaction                      gain@GT - gain@YOLO
                                       > 0: fine-tuning helps MORE on clean GT
                                            framing than on detector framing
                                       < 0: fine-tuning also absorbs detector
                                            framing error (the more useful case)

    All four cells share the SAME ROIs per box source and the SAME sum-rule
    consolidation, so each delta isolates exactly one factor. Missing cells
    (ZS_EVAL off, or no detector) yield None and None-valued deltas."""
    g = lambda k: results.get(k, {}).get(metric)
    m = {"gt_ft": g("gt"), "gt_zs": g("gt_zs"),
         "yolo_ft": g("detector"), "yolo_zs": g("detector_zs")}
    d = lambda a, b: None if (m[a] is None or m[b] is None) else round(m[a] - m[b], 4)
    m["finetune_gain_gt"] = d("gt_ft", "gt_zs")
    m["finetune_gain_yolo"] = d("yolo_ft", "yolo_zs")
    m["detection_cost_ft"] = d("gt_ft", "yolo_ft")
    m["detection_cost_zs"] = d("gt_zs", "yolo_zs")
    m["interaction"] = (None if None in (m["finetune_gain_gt"], m["finetune_gain_yolo"])
                        else round(m["finetune_gain_gt"] - m["finetune_gain_yolo"], 4))
    return m


def format_2x2(results, metrics=("tracklet_accuracy", "accuracy")):
    """Human-readable rendering of compare_box_reader for one or more metrics."""
    out = []
    fmt = lambda v: "   --  " if v is None else f"{100 * v:6.2f}%"
    dfm = lambda v: "    -- " if v is None else f"{100 * v:+6.2f}"
    for metric in metrics:
        m = compare_box_reader(results, metric)
        if all(m[k] is None for k in ("gt_ft", "gt_zs", "yolo_ft", "yolo_zs")):
            continue
        out.append(f"[{metric}]  boxes \\ reader     fine-tuned   zero-shot   ft-gain")
        out.append(f"  GT (annotation)        {fmt(m['gt_ft'])}    {fmt(m['gt_zs'])}"
                   f"   {dfm(m['finetune_gain_gt'])}")
        out.append(f"  YOLO (detector)        {fmt(m['yolo_ft'])}    {fmt(m['yolo_zs'])}"
                   f"   {dfm(m['finetune_gain_yolo'])}")
        out.append(f"  detection cost         {dfm(m['detection_cost_ft'])}"
                   f"     {dfm(m['detection_cost_zs'])}"
                   f"     interaction {dfm(m['interaction'])}")
        out.append("")
    return "\n".join(out) if out else "(no populated cells -- run section 7 first)"


def detection_metrics(frames, thr=0.5, height_bins=(0, 40, 80, float("inf"))):
    """Score the PLAYER DETECTOR on its own terms, independently of recognition.

    frames: [(det_boxes, gt_boxes), ...] one entry per image, boxes as (x, y, w, h).

    Proper one-to-one assignment: all candidate pairs above `thr` are sorted by IoU
    descending and assigned greedily, so each detection and each ground-truth box is
    used at most once. This is deliberately STRICTER than the recognition path's
    match_to_gt, which walks detections in the order the detector emitted them
    (confidence order) and lets the first one clearing `thr` claim the box -- fine for
    "which box would a detector hand me for this player", wrong for scoring the
    detector.

    Returns precision / recall / F1, the mean IoU of matched pairs (localisation
    quality, separate from whether the player was found at all), and recall split by
    ground-truth box height -- the breakdown that matters most here, because distant
    players are small and small players are exactly where a generic COCO model
    diverges from a football-fine-tuned one."""
    tp = fp = fn = 0
    ious = []
    bins = list(zip(height_bins[:-1], height_bins[1:]))
    bin_tot = [0] * len(bins)
    bin_hit = [0] * len(bins)

    for dets, gts in frames:
        cand = []
        for di, d in enumerate(dets):
            for gi, g in enumerate(gts):
                v = _iou_xywh(d, g)
                if v >= thr:
                    cand.append((v, di, gi))
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        used_d, used_g = set(), set()
        for v, di, gi in cand:
            if di in used_d or gi in used_g:
                continue
            used_d.add(di); used_g.add(gi); ious.append(v)
        tp += len(used_g)
        fp += len(dets) - len(used_d)
        fn += len(gts) - len(used_g)
        for gi, g in enumerate(gts):
            h = float(g[3])
            for bi, (lo, hi) in enumerate(bins):
                if lo <= h < hi:
                    bin_tot[bi] += 1
                    if gi in used_g:
                        bin_hit[bi] += 1
                    break

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "mean_iou_matched": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_frames": len(frames),
        "recall_by_height": {
            (f"{int(lo)}-{'inf' if hi == float('inf') else int(hi)}px"):
                {"n": bin_tot[bi],
                 "recall": round(bin_hit[bi] / bin_tot[bi], 4) if bin_tot[bi] else None}
            for bi, (lo, hi) in enumerate(bins)},
    }


def _iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def consolidate_tracklet_vote(frames_tens_logl, frames_units_logl):
    """Majority vote over PER-FRAME decoded labels. Each frame is decoded exactly as
    a single-frame sum would be (argmax per position, 'E' terminates); the tracklet
    label is the most frequent per-frame label. Ties break by summed log-likelihood
    of the tied labels' frames (deterministic, and falls back toward the sum rule).

    Complements consolidate_tracklet (the sum rule): the sum is the MAP estimate
    when confidence tracks correctness (blur, distance); the vote is robust when it
    anti-tracks (a numbered occluder read MORE confidently than the target). Their
    DISAGREEMENT on a tracklet is itself a signal for that failure mode.
    Returns (label, vote_share)."""
    if not frames_tens_logl:
        return "-1", 0.0
    votes, strength = {}, {}
    for t, u in zip(frames_tens_logl, frames_units_logl):
        ti, ui = int(np.argmax(t)), int(np.argmax(u))
        tok = _TOK[ti] + _TOK[ui]
        for i in range(2):
            if tok[i] == "E":
                tok = tok[:i]
                break
        lab = tok if tok else "-1"
        votes[lab] = votes.get(lab, 0) + 1
        strength[lab] = strength.get(lab, 0.0) + float(t[ti]) + float(u[ui])
    best = max(votes, key=lambda l: (votes[l], strength[l]))
    return best, votes[best] / len(frames_tens_logl)


def _logsoftmax(v):
    """Defensive per-position normalisation: PARSeq emits log-probs already,
    but re-normalising is a no-op on normalised input and makes the rule
    correct on raw logits too."""
    v = np.asarray(v, dtype=float)
    m = v.max()
    return v - (m + np.log(np.exp(v - m).sum()))


def consolidate_tracklet_weighted_vote(frames_tens_logl, frames_units_logl):
    """WEIGHTED majority vote:  score(L) = count(L) * sum_conf(L).

    Each frame is decoded to a hard label exactly as the plain vote (argmax
    per position, 'E' terminates), so a single confidently-wrong occluder
    frame still cannot veto the tracklet the way it can under the sum rule.
    Its confidence -- exp of the (log-softmax-normalised) log-probs of the
    chosen tokens, a joint probability in (0, 1] -- is then summed per label,
    and the label maximising count * summed-confidence wins. The count factor
    keeps one near-certain frame (score <= 1 * 1) below a repeated label
    (score = k * sum of k confidences); the confidence factor lets legible
    frames outvote an equal or larger pile of barely-decoded ones.

    Ties break deterministically: count, then summed raw log-likelihood,
    then label -- the first two matching the plain vote's tie behaviour.
    Returns (label, score_share) with score_share = score(best) / sum scores."""
    if not frames_tens_logl:
        return "-1", 0.0
    votes, conf_sum, strength = {}, {}, {}
    for t, u in zip(frames_tens_logl, frames_units_logl):
        t = np.asarray(t, dtype=float); u = np.asarray(u, dtype=float)
        ti, ui = int(np.argmax(t)), int(np.argmax(u))
        tok = _TOK[ti] + _TOK[ui]
        for i in range(2):
            if tok[i] == "E":
                tok = tok[:i]
                break
        lab = tok if tok else "-1"
        lt, lu = _logsoftmax(t), _logsoftmax(u)
        conf = float(np.exp(lt[ti] + lu[ui]))
        votes[lab] = votes.get(lab, 0) + 1
        conf_sum[lab] = conf_sum.get(lab, 0.0) + conf
        strength[lab] = strength.get(lab, 0.0) + float(t[ti]) + float(u[ui])
    score = {l: votes[l] * conf_sum[l] for l in votes}
    best = max(score, key=lambda l: (score[l], votes[l], strength[l], l))
    tot = sum(score.values())
    share = score[best] / tot if tot > 0 else votes[best] / len(frames_tens_logl)
    return best, share


def _decode_frame(t, u):
    """Shared per-frame hard decode (argmax per position, 'E' terminates).
    Returns (label_or_'-1', ti, ui) on float arrays."""
    t = np.asarray(t, dtype=float); u = np.asarray(u, dtype=float)
    ti, ui = int(np.argmax(t)), int(np.argmax(u))
    tok = _TOK[ti] + _TOK[ui]
    for i in range(2):
        if tok[i] == "E":
            tok = tok[:i]
            break
    return (tok if tok else "-1"), ti, ui


def consolidate_tracklet_slogconf(frames_tens_logl, frames_units_logl):
    """v7.2 experimental rule 'slogconf':
        score(L) = psum(L) * conf_sum(L)
    where, over the frames whose hard decode is L,
        psum(L)     = sum of exp(raw joint log-likelihood of the chosen
                      tokens)  -- the sign-consistent form of "sum of log
                      prediction" (raw log-liks are <= 0, so summing them and
                      multiplying by a positive weight would invert the
                      ordering; exp first, then sum)
        conf_sum(L) = sum of exp(log-softmax-normalised joint log-prob),
                      exactly wvote's per-frame confidence.
    On PARSeq output (already log-probs) the two factors coincide, so this
    reduces to argmax of the SQUARED summed confidence == argmax of summed
    confidence: wvote with the count factor removed. Kept as its own rule
    precisely to measure what the count factor buys. Winner is invariant to
    a uniform logit shift (psum scales all labels equally). Ties break
    (count, summed raw log-lik, label) like the other votes.
    Returns (label, score_share)."""
    if not frames_tens_logl:
        return "-1", 0.0
    psum, conf_sum, votes, strength = {}, {}, {}, {}
    for t, u in zip(frames_tens_logl, frames_units_logl):
        t = np.asarray(t, dtype=float); u = np.asarray(u, dtype=float)
        lab, ti, ui = _decode_frame(t, u)
        raw = float(t[ti]) + float(u[ui])
        lt, lu = _logsoftmax(t), _logsoftmax(u)
        conf = float(np.exp(lt[ti] + lu[ui]))
        psum[lab] = psum.get(lab, 0.0) + float(np.exp(raw))
        conf_sum[lab] = conf_sum.get(lab, 0.0) + conf
        votes[lab] = votes.get(lab, 0) + 1
        strength[lab] = strength.get(lab, 0.0) + raw
    score = {l: psum[l] * conf_sum[l] for l in votes}
    best = max(score, key=lambda l: (score[l], votes[l], strength[l], l))
    tot = sum(score.values())
    share = score[best] / tot if tot > 0 else votes[best] / len(frames_tens_logl)
    return best, share


def consolidate_tracklet_sumexp(frames_tens_logl, frames_units_logl):
    """v7.2 experimental rule 'sumexp': the Koshkina sum rule with an
    exponentiated confidence,  conf = exp(st[argmax st] + su[argmax su]),
    where st/su are the per-position log-likelihood sums over all frames.
    exp is monotone, so the LABEL is identical to `sum` by construction (the
    head-to-head row exists to verify that, and the rule to compare the
    confidence scale). The formula is applied literally at the UNTRIMMED
    argmax positions, i.e. a single-digit read includes the 'E' position's
    summed log-lik. The value is a product of per-frame probabilities, so it
    underflows toward 0.0 on long tracklets -- it ranks, it is not calibrated.
    Returns (label, exp_conf)."""
    if not frames_tens_logl:
        return "-1", 0.0
    st = np.sum(np.asarray(frames_tens_logl, dtype=float), axis=0)
    su = np.sum(np.asarray(frames_units_logl, dtype=float), axis=0)
    ti, ui = int(np.argmax(st)), int(np.argmax(su))
    conf = float(np.exp(st[ti] + su[ui]))
    tok = _TOK[ti] + _TOK[ui]
    for i in range(2):
        if tok[i] == "E":
            tok = tok[:i]
            break
    return (tok if tok else "-1"), conf


def consolidate_tracklet_maxconf(frames_tens_logl, frames_units_logl):
    """v7.2 experimental rule 'maxconf':
        score(L) = exp( max over L's frames of the raw joint log-lik
                        log P_f(L | j) ) * conf_sum(L)
    i.e. argmax over (frame, label) of the single-frame log-likelihood,
    weighted by the label's summed (normalised) confidence -- the sign-
    consistent exp form, per the approved plan. One stellar frame can carry
    a label here (max, not sum/count), but the conf_sum factor still lets a
    pile of legible frames outweigh it; the discriminating case vs wvote is
    several moderate frames against one near-certain one. Winner is invariant
    to a uniform logit shift. Ties break (count, summed raw log-lik, label).
    Returns (label, score_share)."""
    if not frames_tens_logl:
        return "-1", 0.0
    mx, conf_sum, votes, strength = {}, {}, {}, {}
    for t, u in zip(frames_tens_logl, frames_units_logl):
        t = np.asarray(t, dtype=float); u = np.asarray(u, dtype=float)
        lab, ti, ui = _decode_frame(t, u)
        raw = float(t[ti]) + float(u[ui])
        lt, lu = _logsoftmax(t), _logsoftmax(u)
        conf = float(np.exp(lt[ti] + lu[ui]))
        mx[lab] = max(mx.get(lab, float("-inf")), raw)
        conf_sum[lab] = conf_sum.get(lab, 0.0) + conf
        votes[lab] = votes.get(lab, 0) + 1
        strength[lab] = strength.get(lab, 0.0) + raw
    score = {l: float(np.exp(mx[l])) * conf_sum[l] for l in votes}
    best = max(score, key=lambda l: (score[l], votes[l], strength[l], l))
    tot = sum(score.values())
    share = score[best] / tot if tot > 0 else votes[best] / len(frames_tens_logl)
    return best, share


def consolidate_both(frames_tens_logl, frames_units_logl, min_conf=None):
    """Run ALL THREE rules on one tracklet. Returns
    {'sum', 'vote', 'wvote': labels; 'sum_conf', 'vote_share', 'wvote_share';
     'agree' (sum == vote, kept for backward compatibility) and the pairwise
     flags 'agree_sum_vote', 'agree_vote_wvote', 'agree_sum_wvote'}.
    Disagreement is worth surfacing: under a numbered occluder the sum rule
    follows the (more confident) wrong frames first, so sum != vote flags
    exactly the tracklets deserving manual review; vote != wvote isolates the
    tracklets where confidence weighting changes the answer -- the entire
    delta between the two vote rules lives there."""
    s_lab, s_conf = consolidate_tracklet(frames_tens_logl, frames_units_logl,
                                         min_conf=min_conf)
    v_lab, v_share = consolidate_tracklet_vote(frames_tens_logl, frames_units_logl)
    w_lab, w_share = consolidate_tracklet_weighted_vote(frames_tens_logl,
                                                        frames_units_logl)
    sc_lab, sc_share = consolidate_tracklet_slogconf(frames_tens_logl,
                                                     frames_units_logl)
    se_lab, se_conf = consolidate_tracklet_sumexp(frames_tens_logl,
                                                  frames_units_logl)
    mc_lab, mc_share = consolidate_tracklet_maxconf(frames_tens_logl,
                                                    frames_units_logl)
    return {"sum": s_lab, "vote": v_lab, "wvote": w_lab,
            "slogconf": sc_lab, "sumexp": se_lab, "maxconf": mc_lab,
            "agree": s_lab == v_lab,
            "agree_sum_vote": s_lab == v_lab,
            "agree_vote_wvote": v_lab == w_lab,
            "agree_sum_wvote": s_lab == w_lab,
            "agree_wvote_slogconf": w_lab == sc_lab,
            "agree_wvote_maxconf": w_lab == mc_lab,
            "agree_sum_sumexp": s_lab == se_lab,   # True by construction when
                                                   # min_conf is None; a sum
                                                   # forced to -1 by min_conf
                                                   # can differ
            "sum_conf": s_conf, "vote_share": v_share, "wvote_share": w_share,
            "slogconf_share": sc_share, "sumexp_conf": se_conf,
            "maxconf_share": mc_share}


def sweep_min_conf(val_logls, val_gts, grid=None):
    """Calibrate the optional confidence floor on VAL. val_logls: list of
    (tens_list, units_list) per tracklet; val_gts: matching GT labels ('-1' or number).
    Returns the threshold maximising overall accuracy, plus the full table."""
    if grid is None:
        grid = [None] + list(np.linspace(-8.0, -0.1, 40))
    rows = []
    for thr in grid:
        preds = [consolidate_tracklet(t, u, min_conf=thr)[0] for t, u in val_logls]
        pr = minus_one_prf(dict(enumerate(preds)), dict(enumerate(val_gts)))
        acc = sum(p == g for p, g in zip(preds, val_gts)) / len(val_gts)
        rows.append({"min_conf": thr, "accuracy": acc, **pr})
    best = max(rows, key=lambda r: r["accuracy"])
    return best["min_conf"], rows


# ------------------------------- tracklet metrics --------------------------
def tracklet_accuracy(pred_by_track, gt_by_track):
    """All tracklets, `-1` counted on equal footing (the challenge metric)."""
    assert gt_by_track, "empty GT"
    correct = sum(str(pred_by_track.get(t, "-1")) == str(g) for t, g in gt_by_track.items())
    return correct / len(gt_by_track)


def tracklet_accuracy_numbered(pred_by_track, gt_by_track):
    """Only tracklets whose GT is a real number (excludes GT `-1`). Isolates
    reading skill from not-visible handling."""
    items = [(t, g) for t, g in gt_by_track.items() if str(g) != "-1"]
    if not items:
        return float("nan")
    correct = sum(str(pred_by_track.get(t, "-1")) == str(g) for t, g in items)
    return correct / len(items)


def minus_one_prf(pred_by_track, gt_by_track):
    """Precision/recall/F1 for the `-1` (not-visible) class -- does the gate agree
    with GT visibility. Positive class = `-1`."""
    tp = fp = fn = 0
    for t, g in gt_by_track.items():
        p = str(pred_by_track.get(t, "-1"))
        g = str(g)
        if g == "-1" and p == "-1":
            tp += 1
        elif g != "-1" and p == "-1":
            fp += 1
        elif g == "-1" and p != "-1":
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (tp + fp and tp + fn and prec + rec) else float("nan")
    return {"minus1_precision": prec, "minus1_recall": rec, "minus1_f1": f1,
            "minus1_tp": tp, "minus1_fp": fp, "minus1_fn": fn}


def digit_length_breakdown(pred_by_track, gt_by_track):
    """Tracklet accuracy split by GT digit count (1 vs 2), numbered tracklets only."""
    out = {}
    for k, keep in (("single_digit", lambda g: len(g) == 1),
                    ("double_digit", lambda g: len(g) == 2)):
        items = [(t, g) for t, g in gt_by_track.items() if str(g) != "-1" and keep(str(g))]
        out[k + "_n"] = len(items)
        out[k + "_acc"] = (sum(str(pred_by_track.get(t, "-1")) == str(g)
                               for t, g in items) / len(items)) if items else float("nan")
    return out


class FPS:
    def __init__(self, n_items):
        self.n = n_items
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        self.fps = self.n / self.dt if self.dt > 0 else float("inf")


# ------------------------------- packaging + report ------------------------
def cell_metrics(img_preds, img_gts, pred_by_track, gt_by_track, fps_crops=None, fps_tracks=None):
    """Assemble ALL metrics for one (dataset, crop-source, arm) cell into one dict --
    the single results schema both arms write and the merge consumes."""
    m = image_metrics(img_preds, img_gts) if img_preds else {
        "accuracy": float("nan"), "one_minus_ned": float("nan"), "n_images": 0}
    m["tracklet_accuracy"] = tracklet_accuracy(pred_by_track, gt_by_track)
    m["tracklet_accuracy_numbered"] = tracklet_accuracy_numbered(pred_by_track, gt_by_track)
    m.update(minus_one_prf(pred_by_track, gt_by_track))
    m.update(digit_length_breakdown(pred_by_track, gt_by_track))
    m["n_tracklets"] = len(gt_by_track)
    m["fps_crops_per_s"] = round(fps_crops, 1) if fps_crops else None
    m["fps_tracklets_per_s"] = round(fps_tracks, 1) if fps_tracks else None
    return m


def format_matrix(cells):
    """cells: {(dataset, crop, arm): metrics}. Printable table + GT->detector gap per arm."""
    lines, gaps = [], {}
    hdr = (f"{'dataset':<22}{'crop':<10}{'arm':<10}{'img_acc':>9}{'1-NED':>8}"
           f"{'trk_all':>9}{'trk_num':>9}{'-1 F1':>8}")
    lines += [hdr, "-" * len(hdr)]
    for (ds, crop, arm), m in cells.items():
        lines.append(f"{ds:<22}{crop:<10}{arm:<10}"
                     f"{m.get('accuracy', float('nan'))*100:>8.2f}%"
                     f"{m.get('one_minus_ned', float('nan')):>8.3f}"
                     f"{m.get('tracklet_accuracy', float('nan'))*100:>8.2f}%"
                     f"{m.get('tracklet_accuracy_numbered', float('nan'))*100:>8.2f}%"
                     f"{m.get('minus1_f1', float('nan')):>8.2f}")
    for arm in {k[2] for k in cells}:
        g = cells.get(("GSR gamestate-2024", "GT", arm))
        d = cells.get(("GSR gamestate-2024", "detector", arm))
        if g and d:
            gaps[arm] = (g["tracklet_accuracy"] - d["tracklet_accuracy"]) * 100
    lines.append("")
    for arm, gap in gaps.items():
        lines.append(f"GSR GT->detector tracklet-acc gap [{arm}]: {gap:+.2f} pts "
                     f"(large => envelope too narrow, Section 10)")
    lines.append(f"anchors: Koshkina soccer tracklet={ANCHORS['koshkina_soccer_tracklet']}%, "
                 f"2023 winner={ANCHORS['winner_2023']}%")
    return "\n".join(lines)


if __name__ == "__main__":
    # ------------------------------ unit tests --------------------------------
    assert edit_distance("23", "25") == 1 and edit_distance("7", "17") == 1
    assert abs(normed_edit_similarity("23", "25") - 0.5) < 1e-9

    def ll(idx, hi=-0.1, lo=-6.0):
        v = [lo] * 11; v[idx] = hi; return v

    lab, conf = consolidate_tracklet([ll(2)] * 3, [ll(3)] * 3)
    assert lab == "23" and conf > -1.0                              # confident "23"
    assert consolidate_tracklet([ll(7)] * 3, [ll(10)] * 3)[0] == "7"   # 'E' -> single digit
    assert consolidate_tracklet([], [])[0] == "-1"                 # no legible frames -> gate -1
    weak = consolidate_tracklet([ll(2, hi=-5.0)] * 2, [ll(3, hi=-5.0)] * 2, min_conf=-2.0)
    assert weak[0] == "-1"                                          # confidence floor forces -1
    assert consolidate_tracklet([ll(2)] * 2, [ll(3)] * 2, min_conf=-2.0)[0] == "23"

    gt = {"t1": "23", "t2": "-1", "t3": "9", "t4": "10"}
    pred = {"t1": "23", "t2": "-1", "t3": "-1", "t4": "10"}         # t3 wrongly -1
    assert abs(tracklet_accuracy(pred, gt) - 3/4) < 1e-9
    assert abs(tracklet_accuracy_numbered(pred, gt) - 2/3) < 1e-9  # t2 excluded
    prf = minus_one_prf(pred, gt)
    assert prf["minus1_tp"] == 1 and prf["minus1_fp"] == 1 and prf["minus1_fn"] == 0
    assert abs(prf["minus1_precision"] - 0.5) < 1e-9 and prf["minus1_recall"] == 1.0
    dl = digit_length_breakdown(pred, gt)
    assert dl["single_digit_n"] == 1 and dl["double_digit_n"] == 2
    assert dl["single_digit_acc"] == 0.0 and dl["double_digit_acc"] == 1.0

    val_logls = [([ll(2)] * 3, [ll(3)] * 3), ([ll(1, hi=-5.5)], [ll(1, hi=-5.5)])]
    val_gts = ["23", "-1"]
    # ---- 2x2 comparison matrix ----
    r = {"gt": {"tracklet_accuracy": 0.90}, "gt_zs": {"tracklet_accuracy": 0.70},
         "detector": {"tracklet_accuracy": 0.84}, "detector_zs": {"tracklet_accuracy": 0.60}}
    m = compare_box_reader(r)
    assert m["finetune_gain_gt"] == 0.20 and m["finetune_gain_yolo"] == 0.24, m
    assert m["detection_cost_ft"] == 0.06 and m["detection_cost_zs"] == 0.10, m
    assert m["interaction"] == -0.04, m         # ft helps MORE on YOLO framing here
    # missing cells degrade to None, never crash
    m2 = compare_box_reader({"gt": {"tracklet_accuracy": 0.9}})
    assert m2["gt_ft"] == 0.9 and m2["yolo_ft"] is None and m2["interaction"] is None
    assert "GT (annotation)" in format_2x2(r)
    # the image-level key in cell_metrics is 'accuracy' -- the 2x2 must use it
    r3 = {"gt": {"accuracy": 0.8}, "gt_zs": {"accuracy": 0.5}}
    assert compare_box_reader(r3, "accuracy")["finetune_gain_gt"] == 0.30
    assert "no populated cells" in format_2x2({})
    print("evaluate_jn: 2x2 comparison tests passed (deltas, interaction, "
          "missing-cell degradation)")

    # ---- detection metrics (detector scored on its own, not via recognition) ----
    # one frame: 2 GT, 1 perfect detection + 1 spurious -> P=.5 R=.5 F1=.5
    m = detection_metrics([([(0, 0, 10, 40), (500, 500, 10, 40)],
                            [(0, 0, 10, 40), (100, 0, 10, 40)])])
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1, m
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5, m
    assert m["mean_iou_matched"] == 1.0, m
    # one-to-one: two detections on ONE gt -> 1 TP, 1 FP (not 2 TP)
    m = detection_metrics([([(0, 0, 10, 40), (1, 0, 10, 40)], [(0, 0, 10, 40)])])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0), m
    # best-IoU wins the assignment, regardless of detection order
    m = detection_metrics([([(4, 0, 10, 40), (0, 0, 10, 40)], [(0, 0, 10, 40)])])
    assert m["mean_iou_matched"] == 1.0, ("best-IoU pair must win", m)
    # empty frame both ways
    assert detection_metrics([([], [])])["f1"] == 0.0
    assert detection_metrics([([], [(0, 0, 10, 40)])])["fn"] == 1
    assert detection_metrics([([(0, 0, 10, 40)], [])])["fp"] == 1
    # height breakdown: a small player missed, a large one found
    m = detection_metrics([([(0, 0, 10, 100)], [(0, 0, 10, 100), (200, 0, 5, 20)])])
    rb = m["recall_by_height"]
    assert rb["0-40px"]["n"] == 1 and rb["0-40px"]["recall"] == 0.0, rb
    assert rb["80-infpx"]["n"] == 1 and rb["80-infpx"]["recall"] == 1.0, rb
    print("evaluate_jn: detection-metric tests passed (one-to-one assignment, "
          "best-IoU precedence, empty frames, height breakdown)")

    # ---- vote rule ----
    def ll(d, p=0.85):
        v = np.full(11, (1 - p) / 10.0); v[d] = p
        return np.log(v + 1e-9)
    def ll_conf(d):                       # near-certain frame (the 'veto' shape)
        v = np.full(11, 1e-10); v[d] = 1.0
        return np.log(v + 1e-9)

    # empty -> -1 under both rules
    assert consolidate_tracklet_vote([], [])[0] == "-1"
    # 1 confidently-wrong frame vs 3 moderate correct: sum is vetoed, vote is not
    T = [ll_conf(7)] + [ll(2)] * 3
    U = [ll_conf(1)] + [ll(3)] * 3
    assert consolidate_tracklet(T, U)[0] != "23"          # the veto effect
    lab, share = consolidate_tracklet_vote(T, U)
    assert lab == "23" and abs(share - 0.75) < 1e-9       # vote resists it
    both = consolidate_both(T, U)
    assert both["vote"] == "23" and both["agree"] is False  # disagreement flagged
    # unanimous confident frames: rules agree
    both2 = consolidate_both([ll(2, 0.95)] * 4, [ll(3, 0.95)] * 4)
    assert both2["sum"] == both2["vote"] == "23" and both2["agree"] is True
    # per-frame 'E' termination inside the vote: ('7','E') frames -> label '7'
    T7 = [ll(7)] * 3; UE = [ll(10)] * 3
    assert consolidate_tracklet_vote(T7, UE)[0] == "7"
    # tie (1 vs 1) breaks by summed log-likelihood -> the stronger frame's label
    lab_tie, _ = consolidate_tracklet_vote([ll(2, 0.95), ll(7, 0.60)],
                                           [ll(3, 0.95), ll(8, 0.60)])
    assert lab_tie == "23", lab_tie
    print("evaluate_jn: vote-rule tests passed (empty, veto resistance, agreement "
          "flag, E-termination, deterministic tie-break)")

    # ---- weighted vote rule (count x summed confidence) ----
    # empty -> -1
    assert consolidate_tracklet_weighted_vote([], [])[0] == "-1"
    # veto resistance is PRESERVED: 1 near-certain wrong frame (score <= 1*1)
    # vs 3 moderate correct (score = 3 * ~2.2) -> still "23"
    w_lab, w_share = consolidate_tracklet_weighted_vote(T, U)
    assert w_lab == "23" and 0.0 < w_share <= 1.0, (w_lab, w_share)
    # the discriminating case, where wvote and plain vote DIFFER:
    # 3 barely-decoded "23" frames (conf ~0.12^2 each, score ~3*0.043) vs
    # 2 legible "78" frames (conf ~0.95^2 each, score ~2*1.8) -> wvote follows
    # the legible frames, plain vote follows the count
    T2 = [ll(2, 0.12)] * 3 + [ll(7, 0.95)] * 2
    U2 = [ll(3, 0.12)] * 3 + [ll(8, 0.95)] * 2
    assert consolidate_tracklet_vote(T2, U2)[0] == "23"
    assert consolidate_tracklet_weighted_vote(T2, U2)[0] == "78"
    b3 = consolidate_both(T2, U2)
    assert b3["vote"] == "23" and b3["wvote"] == "78"
    assert b3["agree_vote_wvote"] is False and b3["agree"] == b3["agree_sum_vote"]
    # ...and when the low-conf label is repeated ENOUGH, count wins again:
    # 50 barely-decoded frames (score ~50*0.72) beat the same 2 legible ones
    T2b = [ll(2, 0.12)] * 50 + [ll(7, 0.95)] * 2
    U2b = [ll(3, 0.12)] * 50 + [ll(8, 0.95)] * 2
    assert consolidate_tracklet_weighted_vote(T2b, U2b)[0] == "23"
    # per-frame 'E' termination inside the weighted vote: ('7','E') -> '7'
    assert consolidate_tracklet_weighted_vote(T7, UE)[0] == "7"
    # 1-vs-1 tie: equal count, the higher-confidence frame's label wins
    assert consolidate_tracklet_weighted_vote(
        [ll(2, 0.95), ll(7, 0.60)], [ll(3, 0.95), ll(8, 0.60)])[0] == "23"
    # unanimous confident frames: all three rules agree, flags all True
    b4 = consolidate_both([ll(2, 0.95)] * 4, [ll(3, 0.95)] * 4)
    assert b4["sum"] == b4["vote"] == b4["wvote"] == "23"
    assert b4["agree_sum_vote"] and b4["agree_vote_wvote"] and b4["agree_sum_wvote"]
    # normalisation defence: adding a constant to every logit (unnormalised
    # input) must not change the winner
    T2c = [[x + 3.0 for x in v] for v in T2]
    U2c = [[x + 3.0 for x in v] for v in U2]
    assert consolidate_tracklet_weighted_vote(T2c, U2c)[0] == "78"
    print("evaluate_jn: weighted-vote tests passed (empty, veto resistance, "
          "conf-vs-count crossover both ways, E-termination, tie-break, "
          "three-way agreement flags, shift-invariance)")

    # ---- v7.2 experimental rules (slogconf / sumexp / maxconf) ----
    # empty -> -1 under all three
    assert consolidate_tracklet_slogconf([], [])[0] == "-1"
    assert consolidate_tracklet_sumexp([], [])[0] == "-1"
    assert consolidate_tracklet_maxconf([], [])[0] == "-1"
    # sumexp label == sum label BY CONSTRUCTION (min_conf=None), conf in (0, 1]
    for TT, UU in ((T, U), (T2, U2), (T2b, U2b), (T7, UE),
                   ([ll(2, 0.95)] * 4, [ll(3, 0.95)] * 4)):
        se_l, se_c = consolidate_tracklet_sumexp(TT, UU)
        assert se_l == consolidate_tracklet(TT, UU)[0], (se_l, TT is T)
        assert 0.0 < se_c <= 1.0 + 1e-12, se_c
    # veto resistance: 1 near-certain wrong frame vs 3 moderate correct --
    # slogconf (score ~2.2*2.2 vs 1*1) and maxconf (~0.72*2.2 vs 1*1) both
    # resist it, like the votes and unlike the sum
    assert consolidate_tracklet_slogconf(T, U)[0] == "23"
    assert consolidate_tracklet_maxconf(T, U)[0] == "23"
    # count-independence: 50 barely-decoded "23" frames flip wvote (count
    # factor) but NOT slogconf (conf_sum 50*.014=0.72 still < 2*.90=1.8)
    assert consolidate_tracklet_weighted_vote(T2b, U2b)[0] == "23"
    assert consolidate_tracklet_slogconf(T2b, U2b)[0] == "78"
    # ...and on the 3-vs-2 case slogconf sides with the legible pair too
    assert consolidate_tracklet_slogconf(T2, U2)[0] == "78"
    # maxconf's discriminating case vs wvote: 3 moderate (p=0.7) "23" frames
    # vs 1 near-certain "78" frame -- wvote keeps the pile (3*1.47 > 1*1),
    # maxconf follows the stellar frame (0.49*1.47 < 1*1)
    T3 = [ll(2, 0.7)] * 3 + [ll_conf(7)]
    U3 = [ll(3, 0.7)] * 3 + [ll_conf(8)]
    assert consolidate_tracklet_weighted_vote(T3, U3)[0] == "23"
    assert consolidate_tracklet_maxconf(T3, U3)[0] == "78"
    # per-frame 'E' termination: ('7','E') frames -> label '7'
    assert consolidate_tracklet_slogconf(T7, UE)[0] == "7"
    assert consolidate_tracklet_maxconf(T7, UE)[0] == "7"
    # 1-vs-1 tie on equal scores is impossible here, but the deterministic
    # chain must at least pick the stronger frame's label
    assert consolidate_tracklet_slogconf(
        [ll(2, 0.95), ll(7, 0.60)], [ll(3, 0.95), ll(8, 0.60)])[0] == "23"
    assert consolidate_tracklet_maxconf(
        [ll(2, 0.95), ll(7, 0.60)], [ll(3, 0.95), ll(8, 0.60)])[0] == "23"
    # shift-invariance of the WINNER (scores scale uniformly across labels)
    assert consolidate_tracklet_slogconf(T2c, U2c)[0] == \
        consolidate_tracklet_slogconf(T2, U2)[0]
    T3c = [[x + 3.0 for x in np.asarray(v)] for v in T3]
    U3c = [[x + 3.0 for x in np.asarray(v)] for v in U3]
    assert consolidate_tracklet_maxconf(T3c, U3c)[0] == "78"
    # consolidate_both carries all six labels + the new agreement flags
    b6 = consolidate_both([ll(2, 0.95)] * 4, [ll(3, 0.95)] * 4)
    assert b6["slogconf"] == b6["sumexp"] == b6["maxconf"] == "23"
    assert b6["agree_wvote_slogconf"] and b6["agree_wvote_maxconf"] \
        and b6["agree_sum_sumexp"]
    b7 = consolidate_both(T3, U3)
    assert b7["wvote"] == "23" and b7["maxconf"] == "78" \
        and b7["agree_wvote_maxconf"] is False
    print("evaluate_jn: v7.2 rule tests passed (empty, sumexp==sum invariant, "
          "veto resistance, slogconf count-independence, maxconf stellar-frame "
          "crossover, E-termination, tie-break, shift-invariance, six-way "
          "consolidate_both)")

    thr, table = sweep_min_conf(val_logls, val_gts)
    assert any(r["accuracy"] == 1.0 for r in table)

    cell = cell_metrics(["23", "7"], ["23", "9"], pred, gt, fps_crops=120.0, fps_tracks=30.0)
    assert cell["n_tracklets"] == 4 and cell["fps_crops_per_s"] == 120.0
    cells = {("GSR gamestate-2024", "GT", "envelope"): cell,
             ("GSR gamestate-2024", "detector", "envelope"): dict(cell, tracklet_accuracy=0.6)}
    print(format_matrix(cells))
    print("\nevaluate_jn: all unit tests passed")
