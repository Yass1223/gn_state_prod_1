"""Ported rules == notebook functions, on synthetic sequences.

Builds the notebook's own inputs (T = stride-5 detection table, S = sampled crops,
Z = embeddings) for random sequences with two kits, a keeper per half, a centre
referee and two assistants, runs the ORIGINAL notebook code (extracted verbatim
into notebook_reference.py) and the port, and asserts identical per-tracklet
roles, teams, reasons, outlier flags and identical tracklet tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import notebook_reference as nbref                      # noqa: E402
from sn_gamestate.team import rules                     # noqa: E402


def make_sequence(rng, n_frames=750, n_tracks=24, dim=16, variant=0):
    """Synthetic tracklets on the stride grid, notebook table layout.

    variant 0: keeper per half, centre referee, two assistants, players.
    variant 1: one assistant only, one extra referee-kit tracklet mid-pitch (extra
               referee channel), a second same-half keeper candidate (gk_confirmed),
               and an assistant-positioned player in a team kit (assistant_rejected)."""
    kits = rng.normal(size=(2, dim)); kits /= np.linalg.norm(kits, axis=1, keepdims=True)
    ref_kit = rng.normal(size=dim); ref_kit /= np.linalg.norm(ref_kit)
    gk2_kit = rng.normal(size=dim); gk2_kit /= np.linalg.norm(gk2_kit)
    rows, samples = [], []
    for tr in range(n_tracks):
        kind = "player"
        if tr < 2: kind = "gk"
        elif tr == 2: kind = "centre"
        elif tr in (3, 4): kind = "assist"
        if variant == 1:
            if tr == 4: kind = "assist_player"   # assistant position, team kit
            if tr == 5: kind = "extra_ref"       # referee kit, mid-pitch, low movement
            if tr == 6: kind = "gk2"             # second keeper candidate, same half as tr 0
        length = int(rng.integers(20, n_frames))
        start = int(rng.integers(0, n_frames - length + 1))
        frames = [f for f in range(start, start + length) if f % 5 == 0]
        if kind == "gk":
            cx = (-48.0 if tr == 0 else 48.0); cy = 0.0; spread = (2.0, 3.0)
        elif kind == "centre":
            cx, cy, spread = 0.0, 0.0, (25.0, 12.0)
        elif kind == "assist":
            cx = 0.0; cy = (-33.0 if tr == 3 else 33.0); spread = (20.0, 0.5)
        elif kind == "assist_player":
            cx = 0.0; cy = 33.0; spread = (20.0, 0.5)
        elif kind == "extra_ref":
            cx, cy, spread = 10.0, 5.0, (6.0, 5.0)
        elif kind == "gk2":
            cx, cy, spread = -46.0, 2.0, (2.0, 3.0)
        else:
            cx = rng.uniform(-30, 30); cy = rng.uniform(-20, 20); spread = (8.0, 6.0)
        for f in frames:
            px = cx + rng.normal(0, spread[0]); py = cy + rng.normal(0, spread[1])
            if rng.random() < 0.02: px = py = np.nan
            rows.append(dict(sequence="S", frame=f, track=tr, cls=0, px=px, py=py, fallback=False,
                             rT=float(rng.uniform(0, 0.5)), rB=float(rng.uniform(0, 0.6))))
    T = pd.DataFrame(rows)
    keep = []
    for (seq, tr), g in T.groupby(["sequence", "track"], sort=False):
        g = g.sort_values("frame")
        idx = np.linspace(0, len(g) - 1, min(16, len(g))).astype(int)
        keep.append(g.iloc[sorted(set(idx))])
    S = pd.concat(keep, ignore_index=True)
    E = np.zeros((len(S), dim), np.float32)
    for i, r in S.iterrows():
        tr = int(r.track)
        kind = "player" if tr >= 5 else ("gk" if tr < 2 else "ref")
        if variant == 1:
            if tr == 4: kind = "player"
            if tr == 5: kind = "ref"
            if tr == 6: kind = "gk2"
        base = kits[tr % 2] if kind not in ("ref", "gk2") else (ref_kit if kind == "ref" else gk2_kit)
        e = base + rng.normal(0, 0.15 if kind != "ref" else 0.3, size=dim)
        E[i] = (e / np.linalg.norm(e)).astype(np.float32)
    Z = {"emb_osnet_team": E}
    return T, S, Z


def port_table(T, S, Z, rt, rb):
    rows = []
    for (seq, tr), gS in S.groupby(["sequence", "track"], sort=False):
        gT = T[T.track == tr]
        ii = gS.index.to_numpy()
        single = (gS.rT <= rt) & (gS.rB < rb)
        rows.append(rules.tracklet_row(tr, gT.px.to_numpy(float), gT.py.to_numpy(float), len(gT),
                                       Z["emb_osnet_team"][ii], single.to_numpy(), gS.rT.to_numpy()))
    return pd.DataFrame(rows)


def compare(P, seeds=range(12), variant=0):
    for sd in seeds:
        rng = np.random.default_rng(sd)
        T, S, Z = make_sequence(rng, variant=variant)
        D_ref = nbref.tracklets(T, S, Z, "osnet_team", 0.25, 0.40)
        D_port = port_table(T, S, Z, 0.25, 0.40)
        for col in ("n", "n_single", "filt_fallback", "mx", "sx", "q75", "my", "sy"):
            a, b = D_ref[col].to_numpy(float), D_port[col].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True, rtol=0, atol=0), (sd, col)
        Eref = np.stack(D_ref.emb.to_numpy()); Eport = np.stack(D_port.emb.to_numpy())
        assert Eref.dtype == Eport.dtype == np.float32 and np.array_equal(Eref, Eport), (sd, "emb")
        nbref.KM_CACHE.clear()
        R_ref = nbref.run_sequence(D_ref, P)
        R_port = rules.run_sequence(D_port, P)
        for key in ("role", "team", "why", "out_rule", "out_db", "confirmed", "gk_c"):
            assert np.array_equal(np.asarray(R_ref[key]), np.asarray(R_port[key])), (sd, key)
        assert np.array_equal(R_ref["z"], R_port["z"]), (sd, "z")
        assert R_ref["named"] == R_port["named"] and R_ref["s_ok"] == R_port["s_ok"], (sd, "named/s_ok")
        assert R_ref["cues"] == R_port["cues"], (sd, "cues")
        roles = np.asarray(R_port["role"])
        yield sd, dict(ref=int((roles == 2).sum()), gk=int((roles == 1).sum()),
                       s_ok=R_port["s_ok"], why=sorted(set(R_port["why"])))


if __name__ == "__main__":
    P = dict(rules.FROZEN_PARAMS)
    seen = set()
    for sd, info in compare(P):
        seen.update(info["why"])
        print(f"seed {sd}: referees {info['ref']}, keepers {info['gk']}, s_ok {info['s_ok']}")
    # a second parameter set exercises the other branches
    P2 = dict(P, confirm="geometry", extra_ref="both", extra_pool="nogk", centre_confirm="margin", side_rule="vote2")
    for sd, info in compare(P2, seeds=range(12, 20)):
        seen.update(info["why"])
    P3 = dict(P, extra_ref="rule", side_rule="vote")
    for sd, info in compare(P3, seeds=range(20, 26)):
        seen.update(info["why"])
    for sd, info in compare(P, seeds=range(30, 42), variant=1):
        seen.update(info["why"])
        print(f"variant 1 seed {sd}: referees {info['ref']}, keepers {info['gk']}, reasons {info['why']}")
    print("reasons exercised:", sorted(seen))
    print("OK: port == notebook on", 38, "synthetic sequences x 3 parameter sets")
