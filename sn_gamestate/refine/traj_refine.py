"""Label-aware trajectory refinement -- the algorithm (``traj_refine`` stage).

Runs AFTER role/team assignment and jersey recognition, on the trajectories the
``split_merge`` stage produced (and the pitch gate kept): it merges trajectories
that the appearance-only merge could not join because their evidence -- team
side, jersey number, exit/entry geometry -- did not exist yet at that point of
the pipeline, and it resolves contradictory jersey-number claims.

Scope: trajectories whose role is player or goalkeeper. Referee trajectories
and unassigned rows are untouched. Role is NOT a merge condition (a goalkeeper
fragment may join a player cluster); the caller unifies the per-row role of a
merged cluster afterwards.

Inputs, per video (aligned arrays, one entry per TRACKED detection):

    E        (n, d) OSNet-AIN embeddings (unit rows; a zero row carries no
             signal), same checkpoint pin as the tracker and split_merge so the
             cosine-distance scale of ``tau`` transfers
    single   (n,)  bool, the crop filter's ``crop_single`` label
    frames   (n,)  int, CHRONOLOGICAL frame index (the dataset's ``frame``
             column; equality == same frame, order == time order)
    boxes    (n, 4) float, ``bbox_ltwh`` in image space (for the re-enter test)
    tids     (n,)  int, the trajectory id each row carries when the stage runs

    tracks   {tid: dict(team, number, cand, scope)} per-trajectory labels:
             ``team`` in {"left", "right"} or None; ``number`` a digit string
             or None; ``cand`` the jersey stage's pooled candidate list
             ``[[label, mx, conf_sum, votes], ...]``; ``scope`` bool (role in
             {player, goalkeeper})
    img_w    image width in pixels, or None (re-enter checks become vacuous)

Jersey confidence model (the maxconf consolidation rule): a trajectory's score
for label L is ``exp(mx(L)) * conf_sum(L)`` over the pooled frame decodes of
the two recognisers. When two trajectories merge, the pooled statistics
combine exactly per that rule -- ``mx = max``, ``conf_sum``/``votes`` add --
which is why the stage consumes the stats, not precomputed scores.

Phase 2a -- same-team, same-number.  All pairs of in-scope clusters with equal
team (both known) and equal number (both known) are processed in descending
order of the pair score ``score_F(n) + score_G(n)``:

  * frame sets disjoint AND re-enter consistent AND distance <= tau -> merge;
  * frame sets disjoint but re-enter or tau fails -> the pair is set aside
    (re-examined if either side later changes through a merge);
  * frame sets overlap -> two trajectories claiming one shirt at the same
    time: the one with the LOWER maxconf for that number is reassigned to its
    best-ranked candidate not yet lost in a conflict (labels it lost on are
    banned, so a cascade of conflicts walks strictly down its candidate list);
    a non-digit best candidate ("-1", or nothing left) leaves it unnumbered.
    Reassignment can create new same-number pairs; the loop re-derives the
    pair set until no eligible pair remains, and terminates because every
    action either removes a cluster (merge), shrinks a candidate list's
    unbanned prefix (conflict), or grows the set-aside set (reject).

Phase 2b -- distance-based agglomerative merging (average linkage, as in
split_merge: group distance = 1 minus the dot product of the two mean unit
vectors over clean detections).  Compatible(F, G) holds iff ALL of:

    C2  time overlap: the frame sets are disjoint (every occupied frame);
    C3  re-enter: when one cluster ends before the other begins and BOTH the
        earlier cluster's last box and the later cluster's first box touch a
        lateral image edge (within ``edge_margin`` * width), the two sides
        must be equal; in every other case the condition is vacuous;
    C4  labels, vacuous-when-unknown: teams must agree when both are known,
        numbers must agree when both are known.  This yields the five cases:
        team+number vs team+number (same team, same number), team and no
        number (attach to a same-team cluster), number and no team (attach on
        the number, team vacuous), no number and no team (distance only).

The closest compatible pair merges while its distance <= tau; the merged
cluster inherits the union of the known labels (compatibility guarantees no
contradiction), combines the candidate statistics, and only its row/column of
the distance matrix is re-evaluated.

A cluster with no valid centroid (no clean detection with a non-zero
embedding, and no non-zero embedding at all) never merges, in either phase; it
can still lose a 2a number conflict, which needs no appearance.

Determinism: every choice breaks ties on explicit keys ending in the cluster
key (the smallest source trajectory id), so the output is a function of the
inputs alone.

By construction the output holds at most one detection per (frame, cluster):
C2 is evaluated on every occupied frame, so no merge can create a collision.
"""
import math

import numpy as np

DIGITS_1_99 = frozenset(str(i) for i in range(1, 100))


# ------------------------------------------------------------------ helpers

def score_of(cand, label):
    """maxconf score exp(mx) * conf_sum of ``label`` in a stats dict
    {label: [mx, conf_sum, votes]}; 0.0 for an unseen label."""
    s = cand.get(label)
    return math.exp(s[0]) * s[1] if s else 0.0


def combine_cand(a, b):
    """Combine two per-label stats dicts per the maxconf rule:
    mx = max, conf_sum and votes add."""
    out = {l: list(v) for l, v in a.items()}
    for l, v in b.items():
        if l in out:
            out[l] = [max(out[l][0], v[0]), out[l][1] + v[1], out[l][2] + v[2]]
        else:
            out[l] = list(v)
    return out


def ranked_labels(cand):
    """Labels of a stats dict, best first: (score, votes, label) descending --
    the emission-time tie rule minus per-label strength, which is not carried
    (score ties across distinct real trajectories are not expected; votes and
    the label string keep the order deterministic regardless)."""
    return sorted(cand, key=lambda l: (score_of(cand, l), cand[l][2], l),
                  reverse=True)


def edge_side(box, img_w, margin_px):
    """'left' / 'right' when the box touches exactly one lateral image edge
    within ``margin_px``; None when it touches neither or both (ambiguous)."""
    l, _, w, _ = (float(v) for v in box)
    left = l <= margin_px
    right = l + w >= img_w - margin_px
    if left == right:
        return None
    return "left" if left else "right"


class _Cluster:
    """Mutable merge state of one (possibly merged) trajectory."""

    __slots__ = ("tids", "rows", "sum_clean", "n_clean", "sum_any", "n_any",
                 "frames", "team", "number", "cand", "scope", "first_row",
                 "last_row", "banned")

    def __init__(self, tid, rows, E, single, frames, info):
        self.tids = [int(tid)]
        self.rows = np.asarray(rows, dtype=np.int64)
        nz = np.linalg.norm(E[self.rows], axis=1) > 1e-6
        clean = np.asarray(single, dtype=bool)[self.rows] & nz
        self.sum_clean = E[self.rows[clean]].sum(axis=0).astype(np.float64) \
            if clean.any() else np.zeros(E.shape[1], dtype=np.float64)
        self.n_clean = int(clean.sum())
        self.sum_any = E[self.rows[nz]].sum(axis=0).astype(np.float64) \
            if nz.any() else np.zeros(E.shape[1], dtype=np.float64)
        self.n_any = int(nz.sum())
        fr = np.asarray(frames, dtype=np.int64)[self.rows]
        self.frames = set(int(x) for x in fr)
        self.first_row = int(self.rows[int(np.argmin(fr))])
        self.last_row = int(self.rows[int(np.argmax(fr))])
        self.team = info.get("team") or None
        number = info.get("number")
        self.number = str(number) if number not in (None, "", "-1") else None
        self.cand = {str(c[0]): [float(c[1]), float(c[2]), int(c[3])]
                     for c in (info.get("cand") or [])}
        self.scope = bool(info.get("scope"))
        self.banned = set()

    @property
    def key(self):
        return min(self.tids)

    def centroid(self):
        """Mean unit vector over clean rows, else over any non-zero rows, else
        None (the cluster then never merges)."""
        if self.n_clean:
            return self.sum_clean / self.n_clean
        if self.n_any:
            return self.sum_any / self.n_any
        return None

    def first_last(self, frames):
        return int(frames[self.first_row]), int(frames[self.last_row])

    def absorb(self, other, frames):
        self.tids += other.tids
        self.rows = np.concatenate([self.rows, other.rows])
        self.sum_clean += other.sum_clean
        self.n_clean += other.n_clean
        self.sum_any += other.sum_any
        self.n_any += other.n_any
        self.frames |= other.frames
        if int(frames[other.first_row]) < int(frames[self.first_row]):
            self.first_row = other.first_row
        if int(frames[other.last_row]) > int(frames[self.last_row]):
            self.last_row = other.last_row
        self.team = self.team or other.team
        self.number = self.number or other.number
        self.cand = combine_cand(self.cand, other.cand)
        self.banned |= other.banned


# ------------------------------------------------------------------ conditions

def _dist(a, b):
    ca, cb = a.centroid(), b.centroid()
    if ca is None or cb is None:
        return float("inf")
    return 1.0 - float(ca @ cb)


def _reenter_ok(a, b, frames, boxes, img_w, margin_frac, record=None):
    """C3. Vacuous unless one cluster ends strictly before the other begins and
    both boundary boxes touch a lateral edge; then the sides must be equal."""
    if img_w is None:
        return True
    fa, la = a.first_last(frames)
    fb, lb = b.first_last(frames)
    if la < fb:
        earlier, later = a, b
    elif lb < fa:
        earlier, later = b, a
    else:
        return True                     # interleaved intervals: vacuous
    margin_px = float(margin_frac) * float(img_w)
    exit_side = edge_side(boxes[earlier.last_row], img_w, margin_px)
    entry_side = edge_side(boxes[later.first_row], img_w, margin_px)
    if record is not None:
        record.update(exit_side=exit_side, entry_side=entry_side)
    if exit_side is None or entry_side is None:
        return True
    return exit_side == entry_side


def _labels_ok(a, b):
    """C4 (2b): teams agree when both known, numbers agree when both known."""
    if a.team and b.team and a.team != b.team:
        return False
    if a.number and b.number and a.number != b.number:
        return False
    return True


# ------------------------------------------------------------------ phase 2a

def _phase2a(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
             report):
    """Same-team same-number merges and overlap conflict resolution.
    Mutates ``clusters`` (dict key -> cluster). See the module docstring."""
    aside = set()               # pair keys set aside on a re-enter/tau reject
    while True:
        live = sorted(clusters)
        pairs = []
        for i, ka in enumerate(live):
            a = clusters[ka]
            if not a.scope or a.number is None or a.team is None:
                continue
            for kb in live[i + 1:]:
                b = clusters[kb]
                if not b.scope or b.number != a.number or b.team != a.team:
                    continue
                if (ka, kb) in aside:
                    continue
                sc = score_of(a.cand, a.number) + score_of(b.cand, b.number)
                pairs.append((sc, ka, kb))
        if not pairs:
            return
        # descending pair maxconf; the key pair keeps equal scores deterministic
        sc, ka, kb = max(pairs, key=lambda p: (p[0], -p[1], -p[2]))
        a, b = clusters[ka], clusters[kb]
        if a.frames & b.frames:
            _resolve_conflict(a, b, sc, report)
            continue
        entry = dict(phase="2a", pair=[ka, kb], number=a.number,
                     pair_score=round(sc, 6))
        d = _dist(a, b)
        entry["distance"] = None if not np.isfinite(d) else round(d, 4)
        ok_re = (not use_reenter) or _reenter_ok(a, b, frames, boxes, img_w,
                                                 edge_margin, entry)
        if ok_re and d <= tau:
            a.absorb(b, frames)
            del clusters[kb]
            aside = {p for p in aside if ka not in p and kb not in p}
            report["merges"].append(entry)
        else:
            entry["rejected"] = "reenter" if not ok_re else "tau"
            report["rejected_2a"].append(entry)
            aside.add((ka, kb))


def _resolve_conflict(a, b, pair_score, report):
    """Two overlapping clusters claim one number: the lower maxconf side walks
    to its best-ranked candidate it has not lost a conflict on."""
    n = a.number
    sa, sb = score_of(a.cand, n), score_of(b.cand, n)
    # deterministic loser on a perfect tie: the larger key (the smaller keeps)
    loser = b if (sb < sa or (sb == sa and b.key > a.key)) else a
    winner = a if loser is b else b
    loser.banned.add(n)
    new = None
    for lab in ranked_labels(loser.cand):
        if lab in loser.banned:
            continue
        if lab in DIGITS_1_99:
            new = lab
        break                       # the best unbanned label decides either way
    loser.number = new
    report["conflicts"].append(dict(
        phase="2a", number=n, pair=[a.key, b.key],
        winner=winner.key, winner_score=round(score_of(winner.cand, n), 6),
        loser=loser.key, loser_score=round(min(sa, sb), 6),
        reassigned_to=new, pair_score=round(pair_score, 6)))


# ------------------------------------------------------------------ phase 2b

def _phase2b(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
             report):
    """Agglomerative average-linkage merging under C2 + C3 + C4 and tau.
    Mutates ``clusters``."""
    keys = sorted(clusters)
    idx = {k: i for i, k in enumerate(keys)}
    k = len(keys)
    alive = np.ones(k, dtype=bool)

    def compat(a, b):
        if not (a.scope and b.scope):
            return False
        if a.frames & b.frames:
            return False
        if not _labels_ok(a, b):
            return False
        if use_reenter and not _reenter_ok(a, b, frames, boxes, img_w,
                                           edge_margin):
            return False
        return True

    D = np.full((k, k), np.inf)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = clusters[keys[i]], clusters[keys[j]]
            if compat(a, b):
                D[i, j] = D[j, i] = _dist(a, b)
    while alive.sum() > 1:
        sub = D.copy()
        sub[~alive, :] = np.inf
        sub[:, ~alive] = np.inf
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if not np.isfinite(sub[i, j]) or sub[i, j] > tau:
            break
        ka, kb = keys[i], keys[j]
        if kb < ka:                       # keep the smaller key
            i, j, ka, kb = j, i, kb, ka
        a, b = clusters[ka], clusters[kb]
        report["merges"].append(dict(
            phase="2b", pair=[ka, kb], distance=round(float(sub[min(i, j), max(i, j)]), 4),
            team=a.team or b.team, number=a.number or b.number))
        a.absorb(b, frames)
        del clusters[kb]
        alive[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        for m in range(k):
            if m != i and alive[m]:
                other = clusters[keys[m]]
                v = _dist(a, other) if compat(a, other) else np.inf
                D[i, m] = D[m, i] = v


# ------------------------------------------------------------------ driver

def refine_video(E, single, frames, boxes, tids, tracks, img_w,
                 tau, use_reenter=True, edge_margin=0.02):
    """Whole method for one video.

    Returns ``(new_tid_of_row, resolved, report)``:

    * ``new_tid_of_row`` (n,) int64 -- the cluster id (the smallest source
      trajectory id) each row belongs to after refinement;
    * ``resolved`` {cluster id: dict(tids, team, number, confidence, maxconf)}
      -- the label state of every final cluster.  ``confidence`` is the
      number's share of the cluster's pooled frame votes (the jersey stage's
      definition, extended to combined clusters), ``maxconf`` its combined
      maxconf score; both 0.0 with no number;
    * ``report`` -- merges, conflicts, 2a rejections and counts for the audit
      sidecar.
    """
    E = np.asarray(E, dtype=np.float32)
    single = np.asarray(single, dtype=bool)
    frames = np.asarray(frames, dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64)
    tids = np.asarray(tids, dtype=np.int64)
    n = len(E)
    if not (len(single) == len(frames) == len(tids) == n and boxes.shape == (n, 4)):
        raise ValueError("E, single, frames, boxes and tids must have one entry "
                         "per detection")
    tau = float(tau)
    if not (0.0 <= tau <= 2.0):
        raise ValueError(f"tau must be in [0, 2], got {tau}")
    edge_margin = float(edge_margin)
    if not (0.0 <= edge_margin < 0.5):
        raise ValueError(f"edge_margin must be in [0, 0.5), got {edge_margin}")

    report = dict(merges=[], conflicts=[], rejected_2a=[],
                  clusters_in=0, clusters_out=0, out_of_scope=0,
                  no_centroid=[], img_w=img_w)
    clusters = {}
    for tid in np.unique(tids):
        rows = np.where(tids == tid)[0]
        info = tracks.get(int(tid), {})
        c = _Cluster(tid, rows, E, single, frames, info)
        clusters[c.key] = c
        if not c.scope:
            report["out_of_scope"] += 1
        elif c.centroid() is None:
            report["no_centroid"].append(int(tid))
    report["clusters_in"] = len(clusters)

    _phase2a(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
             report)
    report["clusters_after_2a"] = len(clusters)
    _phase2b(clusters, frames, boxes, img_w, tau, use_reenter, edge_margin,
             report)
    report["clusters_out"] = len(clusters)

    new_tid = np.full(n, -1, dtype=np.int64)
    resolved = {}
    for key, c in clusters.items():
        new_tid[c.rows] = key
        total_votes = sum(v[2] for v in c.cand.values())
        number = c.number
        conf = (c.cand[number][2] / total_votes
                if number and number in c.cand and total_votes else 0.0)
        resolved[key] = dict(
            tids=sorted(c.tids), team=c.team, number=number,
            confidence=float(conf),
            maxconf=float(score_of(c.cand, number)) if number else 0.0)
    if (new_tid < 0).any():
        raise RuntimeError("a tracked row was left without a cluster; the "
                           "bookkeeping is broken")
    return new_tid, resolved, report
