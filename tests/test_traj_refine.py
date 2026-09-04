"""Unit tests for sn_gamestate/refine/traj_refine.py (pure numpy, no GPU).

Synthetic embeddings: orthogonal-ish unit vectors per identity, so distances
between same-identity clusters are ~0 and between different identities ~1.
Run directly (``python tests/test_traj_refine.py``) or under pytest.
"""
import math

import numpy as np

try:
    from sn_gamestate.refine.traj_refine import (
        DIGITS_1_99, combine_cand, edge_side, pair_maxconf, ranked_labels,
        refine_video, score_of)
except ImportError:                      # sandbox layout
    from traj_refine import (DIGITS_1_99, combine_cand, edge_side,
                             pair_maxconf, ranked_labels, refine_video,
                             score_of)

W = 1920.0
D = 8


def unit(i):
    v = np.zeros(D, dtype=np.float32)
    v[i % D] = 1.0
    return v


def rows_for(tid, ident, frame_list, x=900.0, singles=None, zero=False):
    """One synthetic trajectory: (E, single, frames, boxes, tids) row lists."""
    out = []
    for k, f in enumerate(frame_list):
        e = np.zeros(D, dtype=np.float32) if zero else unit(ident)
        s = True if singles is None else bool(singles[k])
        out.append((e, s, int(f), [float(x), 400.0, 40.0, 80.0], int(tid)))
    return out


def build(*trajs):
    rows = [r for t in trajs for r in t]
    E = np.stack([r[0] for r in rows])
    single = np.array([r[1] for r in rows])
    frames = np.array([r[2] for r in rows])
    boxes = np.array([r[3] for r in rows])
    tids = np.array([r[4] for r in rows])
    return E, single, frames, boxes, tids


def cand(*entries):
    """entries: (label, p, votes) -> [label, mx, conf_sum, votes] with
    mx = log p and conf_sum = votes * p, so score = p * votes * p."""
    return [[lab, math.log(p), votes * p, votes] for lab, p, votes in entries]


CLU = {None: None, "left": 0.0, "right": 1.0}


def track(team=None, number=None, c=None, scope=True):
    """Helper kept call-compatible with the old team-string tests: "left"/"right"
    map to team-cluster ids 0.0/1.0, None stays unclustered."""
    return dict(cluster=CLU.get(team, team), number=number, cand=c or [], scope=scope)


def run(trajs, tracks, tau=0.6, use_reenter=True, edge_margin=0.02, img_w=W):
    E, single, frames, boxes, tids = build(*trajs)
    return refine_video(E, single, frames, boxes, tids, tracks, img_w,
                        tau, use_reenter, edge_margin)


def check_invariants(frames_arr, new_tid):
    seen = set()
    for f, t in zip(frames_arr, new_tid):
        if int(t) < 0:
            continue                     # unassigned by stage 3b
        assert (int(f), int(t)) not in seen, "frame collision"
        seen.add((int(f), int(t)))


# ---------------------------------------------------------------- helpers ----

def test_helpers():
    assert edge_side([0, 0, 40, 80], W, 38.4) == "left"
    assert edge_side([W - 30, 0, 40, 80], W, 38.4) == "right"
    assert edge_side([900, 0, 40, 80], W, 38.4) is None
    assert edge_side([0, 0, W, 80], W, 38.4) is None          # both edges
    a = {"7": [math.log(0.9), 1.8, 2]}
    b = {"7": [math.log(0.5), 0.5, 1], "9": [math.log(0.8), 0.8, 1]}
    m = combine_cand(a, b)
    assert m["7"] == [math.log(0.9), 2.3, 3] and "9" in m
    assert abs(score_of(m, "7") - 0.9 * 2.3) < 1e-12
    assert score_of(m, "77") == 0.0
    assert ranked_labels({"3": [math.log(.5), 1.0, 2],
                          "8": [math.log(.5), 1.0, 2]})[0] == "8"
    # pair_maxconf == score_of on the combined stats, for every holding pattern
    ca = {"7": [math.log(0.9), 1.0, 1]}
    cb = {"7": [math.log(0.1), 20.0, 20], "9": [math.log(0.5), 0.5, 1]}
    assert abs(pair_maxconf(ca, cb, "7") - score_of(combine_cand(ca, cb), "7")) < 1e-12
    assert abs(pair_maxconf(ca, cb, "9") - score_of(combine_cand(ca, cb), "9")) < 1e-12
    assert pair_maxconf(ca, cb, "42") == 0.0
    assert abs(pair_maxconf(ca, {}, "7") - score_of(ca, "7")) < 1e-12
    assert "0" not in DIGITS_1_99 and "1" in DIGITS_1_99 and "99" in DIGITS_1_99


# ------------------------------------------------------------------ 2a --------

def test_2a_merge_disjoint_same_team_number():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    assert res[1]["number"] == "7" and res[1]["cluster"] == 0.0
    assert [m["phase"] for m in rep["merges"]] == ["2a"]
    # combined confidence: 5+4 votes on "7" of 9 pooled -> 1.0
    assert abs(res[1]["confidence"] - 1.0) < 1e-12
    # combined maxconf: exp(max mx) * (conf_sum sum) = .9 * (4.5 + 3.2)
    assert abs(res[1]["maxconf"] - 0.9 * (4.5 + 3.2)) < 1e-9


def test_2a_tau_blocks():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 1, range(10, 16))          # orthogonal: distance ~1
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, 2}
    assert rep["rejected_2a"] and rep["rejected_2a"][0]["rejected"] == "tau"


def test_2a_reenter_blocks_and_vacuous():
    # earlier exits LEFT, later enters RIGHT -> blocked even at distance 0
    t1 = rows_for(1, 0, range(0, 6), x=2.0)
    t2 = rows_for(2, 0, range(10, 16), x=W - 42.0)
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, 2}
    assert rep["rejected_2a"][0]["rejected"] == "reenter"
    assert rep["rejected_2a"][0]["exit_side"] == "left"
    assert rep["rejected_2a"][0]["entry_side"] == "right"
    # same sides -> merge
    t2 = rows_for(2, 0, range(10, 16), x=2.0)
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    # earlier ends mid-image (occlusion) -> vacuous -> merge
    t1 = rows_for(1, 0, range(0, 6), x=900.0)
    t2 = rows_for(2, 0, range(10, 16), x=W - 42.0)
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    # re-enter off -> geometry ignored
    t1 = rows_for(1, 0, range(0, 6), x=2.0)
    t2 = rows_for(2, 0, range(10, 16), x=W - 42.0)
    new, res, rep = run([t1, t2], tracks, use_reenter=False)
    assert set(new.tolist()) == {1}
    # no image width known -> vacuous
    new, res, rep = run([t1, t2], tracks, img_w=None)
    assert set(new.tolist()) == {1}


def test_2a_overlap_conflict_second_candidate():
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))            # overlaps frames 4..7
    tracks = {1: track("left", "7", cand(("7", .9, 6), ("9", .5, 2))),
              2: track("left", "7", cand(("7", .6, 3), ("4", .55, 2)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, 2}
    assert res[1]["number"] == "7"               # higher maxconf keeps
    assert res[2]["number"] == "4"               # loser walks to its 2nd
    cf = rep["conflicts"][0]
    assert cf["winner"] == 1 and cf["loser"] == 2 and cf["reassigned_to"] == "4"
    # confidence of the reassigned number = its share of pooled votes
    assert abs(res[2]["confidence"] - 2 / 5) < 1e-12


def test_2a_conflict_to_unnumbered():
    # loser's best remaining candidate is "-1" -> unnumbered
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))
    tracks = {1: track("left", "7", cand(("7", .9, 6))),
              2: track("left", "7", cand(("7", .6, 3), ("-1", .95, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert res[1]["number"] == "7"
    assert res[2]["number"] is None
    assert res[2]["confidence"] == 0.0 and res[2]["maxconf"] == 0.0


def test_2a_conflict_cascade_then_merge():
    # 2 loses "7" to 1, walks to "9"; now 2 and 3 share "9" disjointly and merge.
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))
    t3 = rows_for(3, 1, range(20, 26))
    tracks = {1: track("left", "7", cand(("7", .9, 6))),
              2: track("left", "7", cand(("7", .6, 3), ("9", .5, 2))),
              3: track("left", "9", cand(("9", .7, 4)))}
    new, res, rep = run([t1, t2, t3], tracks)
    assert set(new.tolist()) == {1, 2}
    assert res[1]["number"] == "7"
    assert res[2]["number"] == "9" and res[2]["tids"] == [2, 3]
    assert rep["conflicts"][0]["reassigned_to"] == "9"
    assert any(m["phase"] == "2a" and m["pair"] == [2, 3] for m in rep["merges"])


def test_2a_double_conflict_walks_banned_list():
    # 2 loses "7" to 1, walks to "9"; then loses "9" to 3 (overlap, higher
    # score); "7" is banned so it ends unnumbered, not oscillating back.
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 1, range(4, 12))
    t3 = rows_for(3, 2, range(6, 14))            # overlaps 2 in time
    tracks = {1: track("left", "7", cand(("7", .9, 6))),
              2: track("left", "7", cand(("7", .6, 3), ("9", .5, 2))),
              3: track("left", "9", cand(("9", .8, 5)))}
    new, res, rep = run([t1, t2, t3], tracks)
    assert set(new.tolist()) == {1, 2, 3}
    assert res[1]["number"] == "7" and res[3]["number"] == "9"
    assert res[2]["number"] is None
    assert len(rep["conflicts"]) == 2


def test_2a_ordering_merge_changes_later_pair():
    # 1-2 (score .9-family) and 1-3 (weaker) both claim "7" against 1;
    # the STRONGER pair merges first, then 1's frames include t2's, which
    # overlap t3's -> the weaker pair becomes a conflict, not a merge.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 0, range(12, 18))           # overlaps t2 only
    tracks = {1: track("left", "7", cand(("7", .9, 6))),
              2: track("left", "7", cand(("7", .9, 6))),
              3: track("left", "7", cand(("7", .3, 1), ("5", .3, 1)))}
    new, res, rep = run([t1, t2, t3], tracks)
    assert res[1]["tids"] == [1, 2]
    assert res[3]["number"] == "5"
    assert rep["merges"][0]["pair"] == [1, 2]
    assert rep["conflicts"][0]["pair"] == [1, 3]


def test_2a_ordering_is_joint_maxconf_not_sum():
    # Pair X (1,2): equal mx -> joint == sum == 3.2.
    # Pair Y (3,4): mx .9 with cs 1  +  mx .1 with cs 20:
    #   sum of separate scores = 0.9 + 2.0 = 2.9  (< X)
    #   joint = exp(max mx) * (cs1+cs2) = .9 * 21 = 18.9  (> X)
    # Under the joint rule Y merges FIRST; under the old sum rule X would.
    t1 = rows_for(1, 0, range(0, 4))
    t2 = rows_for(2, 0, range(10, 14))
    t3 = rows_for(3, 1, range(0, 4))
    t4 = rows_for(4, 1, range(10, 14))
    tracks = {1: track("left", "7", [["7", math.log(.8), 2.0, 2]]),
              2: track("left", "7", [["7", math.log(.8), 2.0, 2]]),
              3: track("left", "9", [["9", math.log(.9), 1.0, 1]]),
              4: track("left", "9", [["9", math.log(.1), 20.0, 20]])}
    new_tid, res, rep = run([t1, t2, t3, t4], tracks)
    assert set(new_tid.tolist()) == {1, 3}
    order = [m["pair"] for m in rep["merges"]]
    assert order == [[3, 4], [1, 2]], order
    assert abs(rep["merges"][0]["pair_maxconf"] - 0.9 * 21.0) < 1e-6
    assert abs(rep["merges"][1]["pair_maxconf"] - 3.2) < 1e-6
    # merged maxconf recomputed from the combined stats, not from old scores
    assert abs(res[3]["maxconf"] - 0.9 * 21.0) < 1e-6


# ------------------------------------------------------------------ 2b --------

def test_2b_team_no_number_attaches():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", None)}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    assert res[1]["number"] == "7" and res[1]["cluster"] == 0.0
    assert rep["merges"][0]["phase"] == "2b"


def test_2b_number_no_team_attaches():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track(None, "7", cand(("7", .5, 2)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    assert res[1]["cluster"] == 0.0 and res[1]["number"] == "7"


def test_2b_no_number_no_team_distance_only():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(None, None), 2: track(None, None)}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}
    assert res[1]["cluster"] is None and res[1]["number"] is None


def test_2b_blocks():
    base = {  # identical appearance, disjoint times, mid-image
        1: rows_for(1, 0, range(0, 6)),
        2: rows_for(2, 0, range(10, 16)),
    }
    # different teams
    new, _, _ = run(list(base.values()),
                    {1: track("left", None), 2: track("right", None)})
    assert set(new.tolist()) == {1, 2}
    # conflicting numbers
    new, _, _ = run(list(base.values()),
                    {1: track("left", "7", cand(("7", .9, 5))),
                     2: track("left", "9", cand(("9", .9, 5)))})
    assert set(new.tolist()) == {1, 2}
    # time overlap
    new, _, _ = run([rows_for(1, 0, range(0, 8)), rows_for(2, 0, range(4, 12))],
                    {1: track("left", None), 2: track("left", None)})
    assert set(new.tolist()) == {1, 2}
    # distance above tau
    new, _, _ = run([rows_for(1, 0, range(0, 6)), rows_for(2, 1, range(10, 16))],
                    {1: track("left", None), 2: track("left", None)})
    assert set(new.tolist()) == {1, 2}
    # re-enter contradiction
    new, _, _ = run([rows_for(1, 0, range(0, 6), x=2.0),
                     rows_for(2, 0, range(10, 16), x=W - 42.0)],
                    {1: track("left", None), 2: track("left", None)})
    assert set(new.tolist()) == {1, 2}


def test_2b_average_linkage_and_inherited_number_gates_later_merges():
    # 1 (numbered 7) + 2 (unnumbered) merge; 3 carries 9 on the same team and
    # identical appearance -> blocked by the INHERITED number.
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 0, range(20, 26))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", None),
              3: track("left", "9", cand(("9", .9, 5)))}
    new, res, rep = run([t1, t2, t3], tracks)
    assert res[1]["tids"] == [1, 2] and 3 in res
    assert res[1]["number"] == "7" and res[3]["number"] == "9"


# ------------------------------------------------------ cluster-label rules ----

def test_same_cluster_different_numbers_never_merge():
    # two same-cluster fragments with two DIFFERENT known numbers are two
    # different players: no merge in 2a (numbers differ) nor 2b (number veto),
    # at any distance
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "9", cand(("9", .9, 5)))}
    new, res, rep = run([t1, t2], tracks)
    assert rep["merges"] == [] and set(new.tolist()) == {1, 2}


def test_unclustered_numbered_merges_on_same_number():
    # an UNCLUSTERED numbered fragment merges with a clustered fragment of the
    # same number (cluster imposes no condition when either side is unknown)
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track(None, "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert rep["merges"] and rep["merges"][0]["phase"] == "2a"
    assert set(new.tolist()) == {1}
    assert res[1]["cluster"] == 0.0        # the known cluster survives the merge


def test_unclustered_unnumbered_merges_on_distance_alone():
    # an unclustered, unnumbered fragment merges purely on distance <= tau (2b)
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(None, None), 2: track(None, None)}
    new, res, rep = run([t1, t2], tracks)
    assert rep["merges"] and rep["merges"][0]["phase"] == "2b"
    assert set(new.tolist()) == {1}


def test_stage3b_dynamic_centroid_reassignment():
    # After each 3b assignment the receiving trajectory's centroid is
    # recomputed over ALL its detections. Construction: held m1 (0.6*u0+0.8*u2)
    # joins T_a (a single clean u0 row) first; T_a's centroid then leans toward
    # u2, so held m2 (0.3*u1+0.954*u2) lands in T_a (d ~0.62) instead of T_b
    # (d 0.7), which a static centroid would have chosen.
    import numpy as np
    w = 0.6 * unit(0) + 0.8 * unit(2)
    w = (w / np.linalg.norm(w)).astype(np.float32)
    v = 0.3 * unit(1) + 0.954 * unit(2)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    t1 = rows_for(1, 3, range(0, 6)) + [
        (w, False, 20, [900.0, 400.0, 40.0, 80.0], 1),
        (v, False, 21, [900.0, 400.0, 40.0, 80.0], 1)]
    t2 = rows_for(2, 3, range(20, 26))          # clean 20-25 -> collisions at 20, 21
    t3 = [(unit(0), True, 30, [900.0, 400.0, 40.0, 80.0], 3)]      # T_a
    t4 = rows_for(4, 1, range(40, 46))                              # T_b
    tracks = {1: track(None, None), 2: track(None, None),
              3: track("left", None), 4: track("right", None)}
    new, res, rep = run([t1, t2, t3, t4], tracks)
    s3 = rep["stage3"]
    assert s3["held"] == 2 and s3["placed"] == 2 and s3["unassigned"] == 0
    E, single, frames, boxes, tids = build(t1, t2, t3, t4)
    for f, vec in ((20, w), (21, v)):
        m = [i for i in range(len(frames)) if frames[i] == f and not single[i]]
        assert len(m) == 1 and new[m[0]] == 3, (f, new[m[0]])   # BOTH land in T_a
    check_invariants(frames, new)


# ------------------------------------------------------------------ stage 3 ----

def test_stage3_clean_disjoint_multi_overlap_resolved():
    # t1 clean frames 0-5 plus MULTI rows on frames 10,11; t2 clean 10-15.
    # Clean frame sets are disjoint -> 2b merges them; the multi rows collide
    # with t2's clean rows on 10 and 11 -> 3a keeps the clean ones; with no
    # other trajectory to accept the two multis, they are unassigned.
    t1 = rows_for(1, 0, range(0, 6)) + rows_for(1, 0, [10, 11], singles=[False, False])
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", None), 2: track("left", None)}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, -1}
    assert rep["merges"] and rep["merges"][0]["phase"] == "2b"
    s3 = rep["stage3"]
    assert s3["collided_frames"] == 2 and s3["held"] == 2
    assert s3["placed"] == 0 and s3["unassigned"] == 2
    check_invariants(build(t1, t2)[2], new)
    # the kept detections in frames 10 and 11 are the clean ones
    E, single, frames, boxes, tids = build(t1, t2)
    for f in (10, 11):
        kept = [i for i in range(len(frames)) if frames[i] == f and new[i] >= 0]
        assert len(kept) == 1 and single[kept[0]]


def test_stage3_among_multi_keeps_nearest_and_places_rest():
    # One cluster (t1+t2 merged on clean disjointness) holds TWO multi rows in
    # frame 20: identity-0 (near the cluster) and identity-1 (far). 3a keeps
    # the near one. The far one is placed by 3b into t3 (identity 1, frame 20
    # free, in scope).
    t1 = rows_for(1, 0, range(0, 6)) + [
        (unit(0), False, 20, [900.0, 400.0, 40.0, 80.0], 1),
        (unit(1), False, 20, [900.0, 400.0, 40.0, 80.0], 1)]
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 1, range(30, 36))
    tracks = {1: track("left", None), 2: track("left", None),
              3: track("right", None)}
    new, res, rep = run([t1, t2, t3], tracks)
    s3 = rep["stage3"]
    assert s3["collided_frames"] == 1 and s3["held"] == 1
    assert s3["placed"] == 1 and s3["unassigned"] == 0
    E, single, frames, boxes, tids = build(t1, t2, t3)
    kept = [i for i in range(len(frames)) if frames[i] == 20 and new[i] == 1]
    moved = [i for i in range(len(frames)) if frames[i] == 20 and new[i] == 3]
    assert len(kept) == 1 and E[kept[0]][0] == 1.0     # identity-0 stays
    assert len(moved) == 1 and E[moved[0]][1] == 1.0   # identity-1 -> t3
    check_invariants(frames, new)


def test_stage3_no_op_without_collisions():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    s3 = rep["stage3"]
    assert s3 == dict(collided_frames=0, held=0, clean_anomaly=0, placed=0,
                      unassigned=0, unassigned_rows=[])


# ------------------------------------------------------------- scope/centroid --

def test_out_of_scope_untouched():
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16))
    tracks = {1: track(None, None, scope=False),      # referee
              2: track(None, None, scope=False)}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, 2}
    assert rep["out_of_scope"] == 2 and not rep["merges"]


def test_no_centroid_never_merges_but_can_lose_conflict():
    # zero embeddings everywhere on t2: no merging in either phase...
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16), zero=True)
    tracks = {1: track("left", "7", cand(("7", .9, 5))),
              2: track("left", "7", cand(("7", .8, 4)))}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1, 2}
    assert rep["no_centroid"] == [2]
    # ...but an overlap conflict is still resolved (needs no appearance)
    t2 = rows_for(2, 0, range(3, 9), zero=True)
    new, res, rep = run([t1, t2], tracks)
    assert res[1]["number"] == "7" and res[2]["number"] is None
    assert rep["conflicts"]


def test_centroid_fallback_to_nonclean():
    # t2 has no clean row, but its multi rows embed -> it can still merge (2b)
    t1 = rows_for(1, 0, range(0, 6))
    t2 = rows_for(2, 0, range(10, 16), singles=[False] * 6)
    tracks = {1: track("left", None), 2: track("left", None)}
    new, res, rep = run([t1, t2], tracks)
    assert set(new.tolist()) == {1}


# ------------------------------------------------------------ invariants ------

def test_invariants_and_determinism():
    t1 = rows_for(1, 0, range(0, 8))
    t2 = rows_for(2, 0, range(10, 16))
    t3 = rows_for(3, 1, range(4, 12))
    t4 = rows_for(4, 2, range(0, 20), singles=[True, False] * 10)
    tracks = {1: track("left", "7", cand(("7", .9, 6), ("9", .5, 2))),
              2: track("left", "7", cand(("7", .8, 4))),
              3: track("left", "7", cand(("7", .6, 3), ("4", .5, 2))),
              4: track("right", None)}
    E, single, frames, boxes, tids = build(t1, t2, t3, t4)
    out1 = refine_video(E, single, frames, boxes, tids, tracks, W, 0.6)
    out2 = refine_video(E, single, frames, boxes, tids, tracks, W, 0.6)
    assert np.array_equal(out1[0], out2[0]) and out1[1] == out2[1]
    new, res, rep = out1
    assert len(new) == len(tids)                      # no row lost
    check_invariants(frames, new)
    # every final cluster id is the min of its source ids
    for key, r in res.items():
        assert key == min(r["tids"])
    # input validation
    try:
        refine_video(E, single, frames, boxes[:2], tids, tracks, W, 0.6)
        raise AssertionError("shape mismatch must raise")
    except ValueError:
        pass
    try:
        refine_video(E, single, frames, boxes, tids, tracks, W, 3.0)
        raise AssertionError("tau out of range must raise")
    except ValueError:
        pass


def _all_tests():
    g = dict(globals())
    names = sorted(n for n in g if n.startswith("test_"))
    for n in names:
        g[n]()
        print(f"  ok {n}")
    print(f"test_traj_refine: {len(names)} tests passed")


if __name__ == "__main__":
    _all_tests()
