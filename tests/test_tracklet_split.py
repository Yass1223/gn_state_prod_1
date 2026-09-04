"""Unit tests for sn_gamestate/track/tracklet_split.py (pure numpy, no GPU).

Run directly (``python tests/test_tracklet_split.py``) or under pytest.
"""
import numpy as np

try:
    from sn_gamestate.track.tracklet_split import (FRAG_BASE, split_tracklet,
                                                   split_video, _unit)
except ImportError:                      # sandbox layout
    from tracklet_split import FRAG_BASE, split_tracklet, split_video, _unit

D = 8
EPS, MINS = 0.2, 5


def unit(i):
    v = np.zeros(D, dtype=np.float32)
    v[i % D] = 1.0
    return v


def near(i, wobble=0.05, seed=0):
    rng = np.random.RandomState(seed)
    v = unit(i) + wobble * rng.randn(D).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def block(ident, count, single=True, zero=False, wobble=0.03, seed0=0):
    rows = []
    for k in range(count):
        e = np.zeros(D, np.float32) if zero else near(ident, wobble, seed0 + k)
        rows.append((e, single))
    return rows


def run_split(rows):
    E = np.stack([r[0] for r in rows])
    single = np.array([r[1] for r in rows])
    return split_tracklet(_unit(E), single, EPS, MINS)


def test_two_identities_split():
    rows = block(0, 8) + block(1, 8)
    lab, k, n_noise, n_diss = run_split(rows)
    assert k == 2 and n_diss == 0
    assert len(set(lab[:8])) == 1 and len(set(lab[8:])) == 1
    assert lab[0] != lab[8]


def test_single_identity_one_fragment():
    lab, k, _, _ = run_split(block(0, 12))
    assert k == 1 and set(lab) == {0}


def test_small_tracklet_one_fragment():
    lab, k, n_noise, _ = run_split(block(0, 3))
    assert k == 1 and n_noise == 0 and set(lab) == {0}


def test_all_noise_one_fragment():
    # five mutually orthogonal points: DBSCAN(min_samples=5) finds no core
    rows = [(unit(i), True) for i in range(5)]
    lab, k, _, _ = run_split(rows)
    assert k == 1 and set(lab) == {0}


def test_noise_single_and_multi_attach_to_nearest():
    # two clear identities + one single-crop and one multi-crop outlier that
    # lean toward id 1 but sit outside eps (cos sim 0.6 -> distance 0.4 > 0.2)
    rows = block(0, 8) + block(1, 8)
    v = 0.6 * unit(1) + 0.8 * unit(3)
    out_vec = (v / np.linalg.norm(v)).astype(np.float32)
    outlier_single = (out_vec, True)
    outlier_multi = (out_vec.copy(), False)
    lab, k, n_noise, _ = run_split(rows + [outlier_single, outlier_multi])
    assert k == 2
    id1_frag = lab[8]
    assert lab[-2] == id1_frag and lab[-1] == id1_frag   # both attached to id-1
    assert n_noise >= 2


def test_zero_embedding_noise_deterministic():
    rows = block(0, 8) + block(1, 8) + [(np.zeros(D, np.float32), True)]
    lab1 = run_split(rows)[0]
    lab2 = run_split(rows)[0]
    assert np.array_equal(lab1, lab2)
    assert lab1[-1] == min(lab1[:16])     # distance 1 to all -> lowest label


def test_allmulti_fragment_dissolved_per_detection():
    # id-0 clean cluster; id-1 cluster entirely multi-crop -> dissolved into
    # the remaining (clean-holding) fragment, detection by detection
    rows = block(0, 8, single=True) + block(1, 8, single=False)
    lab, k, _, n_diss = run_split(rows)
    assert n_diss == 1 and k == 1
    assert set(lab) == {0}


def test_allmulti_everywhere_kept():
    # no clean detection anywhere: nothing to dissolve into -> fragments kept
    rows = block(0, 8, single=False) + block(1, 8, single=False)
    lab, k, _, n_diss = run_split(rows)
    assert k == 2 and n_diss == 0


def test_centroids_clean_only_drive_attachment():
    # id-0 fragment: clean members at identity 0 plus multi members at identity 2
    # (which would drag an all-rows centroid away). A noise point midway is
    # closer to the CLEAN centroid of fragment A than to fragment B's.
    a = block(0, 6, single=True) + block(2, 4, single=False)
    b = block(1, 8, single=True)
    v = unit(0) * 0.8 + unit(1) * 0.35
    noise = ((v / np.linalg.norm(v)).astype(np.float32), True)
    lab, k, _, _ = run_split(a + b + [noise])
    if k == 2:                             # DBSCAN grouping as constructed
        assert lab[-1] == lab[0]           # attached via the clean centroid


def test_split_video_invariant_and_frag_ids():
    E, single, frames, tids = [], [], [], []
    for f in range(12):
        E.append(near(0, seed=f)); single.append(True); frames.append(f); tids.append(4)
    for f in range(12):
        E.append(near(1, seed=50 + f)); single.append(True); frames.append(f); tids.append(7)
    frag, per = split_video(np.stack(E), np.array(single), np.array(frames),
                            np.array(tids), EPS, MINS)
    assert set(frag[:12]) == {4 * FRAG_BASE} and set(frag[12:]) == {7 * FRAG_BASE}
    assert [p["track_id"] for p in per] == [4, 7]
    assert all(p["k"] == 1 for p in per)
    # source tracklet recoverable
    assert all(int(f) // FRAG_BASE in (4, 7) for f in frag)


def test_split_video_rejects_duplicate_frame():
    E = np.stack([near(0, seed=k) for k in range(4)])
    single = np.array([True] * 4)
    frames = np.array([0, 1, 1, 2])
    tids = np.array([3, 3, 3, 3])
    try:
        split_video(E, single, frames, tids, EPS, MINS)
        raise AssertionError("duplicate (tracklet, frame) must raise")
    except ValueError:
        pass


def test_no_cross_tracklet_mixing_and_determinism():
    rng = np.random.RandomState(7)
    E, single, frames, tids = [], [], [], []
    for tid, ident in ((1, 0), (2, 1), (3, 2)):
        for f in range(10):
            E.append(near(ident, seed=rng.randint(10000)))
            single.append(bool(rng.rand() > 0.3))
            frames.append(f)
            tids.append(tid)
    args = (np.stack(E), np.array(single), np.array(frames), np.array(tids), EPS, MINS)
    f1, _ = split_video(*args)
    f2, _ = split_video(*args)
    assert np.array_equal(f1, f2)
    for i, t in enumerate(tids):
        assert int(f1[i]) // FRAG_BASE == t     # fragments never cross tracklets


def _all_tests():
    g = dict(globals())
    names = sorted(n for n in g if n.startswith("test_"))
    for n in names:
        g[n]()
        print(f"  ok {n}")
    print(f"test_tracklet_split: {len(names)} tests passed")


if __name__ == "__main__":
    _all_tests()
