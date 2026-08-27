"""fuse_jn.py -- two-recogniser tracklet consolidation rules.

Both recognisers read the SAME surviving frames of a tracklet (identical ROI,
identical gates), each emitting per frame two 11-way log-probability vectors
(tens, units) over "0123456789E". Model A is PARSeq, model B the second
(mmocr) recogniser; the rules below are written for any A/B pair.

Per model and per label L, `label_stats` collects exactly the four quantities
evaluate_jn.consolidate_tracklet_maxconf uses:
    mx[L]        max over L's frames of the raw joint log-lik  s(f) = t[ti]+u[ui]
    conf_sum[L]  sum over L's frames of c(f) = exp(logsoftmax(t)[ti]+logsoftmax(u)[ui])
    votes[L]     number of frames whose hard decode is L
    strength[L]  sum over L's frames of s(f)
and maxconf(L) = exp(mx[L]) * conf_sum[L]. A label unseen by a model has
maxconf 0, votes 0, strength 0 for that model.

Rules (all deterministic; ties break on (score, total votes, total raw
log-lik strength, label string), the maxconf tie rule extended to the pooled
frames):

    a_maxconf        maxconf over A's frames alone (baseline; identical to
                     evaluate_jn.consolidate_tracklet_maxconf on A)
    b_maxconf        same for B
    vote_pool        majority vote over the pooled frame labels of A and B
    a_mc_x_bcount    maxconf_A(L) * votes_B(L)
    b_mc_x_acount    maxconf_B(L) * votes_A(L)
    a_mc_x_bcount1   maxconf_A(L) * (1 + votes_B(L))
    b_mc_x_acount1   maxconf_B(L) * (1 + votes_A(L))
    sum              maxconf_A(L) + maxconf_B(L)
    prod             maxconf_A(L) * maxconf_B(L)
    prod_plus1       (1 + maxconf_A(L)) * (1 + maxconf_B(L)) - 1
                     = sum + prod, non-zero for one-sided labels
    joint_maxconf    maxconf over the pooled frames of A and B as if one
                     recogniser: exp(max(mx_A, mx_B)) * (conf_sum_A + conf_sum_B)
    agree_first      a_maxconf if a_maxconf == b_maxconf, else joint_maxconf

A tracklet with no surviving frame returns "-1" under every rule. As in
evaluate_jn, a frame whose first character is 'E' decodes to "-1", and "-1"
is a candidate label that can win.
"""
import numpy as np

from evaluate_jn import _decode_frame, _logsoftmax

RULES = ("a_maxconf", "b_maxconf", "vote_pool",
         "a_mc_x_bcount", "b_mc_x_acount", "a_mc_x_bcount1", "b_mc_x_acount1",
         "sum", "prod", "prod_plus1", "joint_maxconf", "agree_first")


def label_stats(frames_t, frames_u):
    """{label: dict(mx, conf_sum, votes, strength)} over one model's frames."""
    st = {}
    for t, u in zip(frames_t, frames_u):
        t = np.asarray(t, dtype=float)
        u = np.asarray(u, dtype=float)
        lab, ti, ui = _decode_frame(t, u)
        raw = float(t[ti]) + float(u[ui])
        lt, lu = _logsoftmax(t), _logsoftmax(u)
        conf = float(np.exp(lt[ti] + lu[ui]))
        s = st.setdefault(lab, {"mx": float("-inf"), "conf_sum": 0.0,
                                "votes": 0, "strength": 0.0})
        s["mx"] = max(s["mx"], raw)
        s["conf_sum"] += conf
        s["votes"] += 1
        s["strength"] += raw
    return st


def maxconf_of(st, lab):
    s = st.get(lab)
    return float(np.exp(s["mx"])) * s["conf_sum"] if s else 0.0


def votes_of(st, lab):
    s = st.get(lab)
    return s["votes"] if s else 0


def strength_of(st, lab):
    s = st.get(lab)
    return s["strength"] if s else 0.0


def _pick(score, sa, sb):
    """Winner under the extended maxconf tie rule."""
    return max(score, key=lambda l: (score[l],
                                     votes_of(sa, l) + votes_of(sb, l),
                                     strength_of(sa, l) + strength_of(sb, l),
                                     l))


def _single(st):
    """maxconf over one model's frames; tie rule (score, votes, strength, label)
    -- byte-for-byte the evaluate_jn.consolidate_tracklet_maxconf ordering."""
    if not st:
        return "-1"
    score = {l: maxconf_of(st, l) for l in st}
    return max(score, key=lambda l: (score[l], st[l]["votes"],
                                     st[l]["strength"], l))


def fuse(frames_ta, frames_ua, frames_tb, frames_ub, rules=RULES):
    """-> {rule: label}. Frame lists of A and B may differ in length only if
    one model produced no output; normally they cover the same frames."""
    sa = label_stats(frames_ta, frames_ua)
    sb = label_stats(frames_tb, frames_ub)
    return fuse_stats(sa, sb, rules)


def fuse_stats(sa, sb, rules=RULES):
    out = {}
    if not sa and not sb:
        return {r: "-1" for r in rules}
    labels = sorted(set(sa) | set(sb))
    mca = {l: maxconf_of(sa, l) for l in labels}
    mcb = {l: maxconf_of(sb, l) for l in labels}
    va = {l: votes_of(sa, l) for l in labels}
    vb = {l: votes_of(sb, l) for l in labels}

    def joint():
        sc = {}
        for l in labels:
            mx = max(sa[l]["mx"] if l in sa else float("-inf"),
                     sb[l]["mx"] if l in sb else float("-inf"))
            cs = (sa[l]["conf_sum"] if l in sa else 0.0) + \
                 (sb[l]["conf_sum"] if l in sb else 0.0)
            sc[l] = float(np.exp(mx)) * cs
        return _pick(sc, sa, sb)

    a_win = _single(sa)
    b_win = _single(sb)
    for r in rules:
        if r == "a_maxconf":
            out[r] = a_win
        elif r == "b_maxconf":
            out[r] = b_win
        elif r == "vote_pool":
            out[r] = _pick({l: va[l] + vb[l] for l in labels}, sa, sb)
        elif r == "a_mc_x_bcount":
            out[r] = _pick({l: mca[l] * vb[l] for l in labels}, sa, sb)
        elif r == "b_mc_x_acount":
            out[r] = _pick({l: mcb[l] * va[l] for l in labels}, sa, sb)
        elif r == "a_mc_x_bcount1":
            out[r] = _pick({l: mca[l] * (1 + vb[l]) for l in labels}, sa, sb)
        elif r == "b_mc_x_acount1":
            out[r] = _pick({l: mcb[l] * (1 + va[l]) for l in labels}, sa, sb)
        elif r == "sum":
            out[r] = _pick({l: mca[l] + mcb[l] for l in labels}, sa, sb)
        elif r == "prod":
            out[r] = _pick({l: mca[l] * mcb[l] for l in labels}, sa, sb)
        elif r == "prod_plus1":
            out[r] = _pick({l: (1 + mca[l]) * (1 + mcb[l]) - 1 for l in labels},
                           sa, sb)
        elif r == "joint_maxconf":
            out[r] = joint()
        elif r == "agree_first":
            out[r] = a_win if a_win == b_win else joint()
        else:
            raise ValueError(f"unknown rule {r!r}")
    return out


# ------------------------------------------------------------- self-tests --
if __name__ == "__main__":
    import math
    from evaluate_jn import consolidate_tracklet_maxconf

    def vec(ch, p):
        """11-vector of log-probs: p on `ch`, rest of mass spread evenly."""
        idx = "0123456789E".index(ch)
        v = np.full(11, math.log((1 - p) / 10))
        v[idx] = math.log(p)
        return v

    def frame(label, p):
        if label == "-1":
            return vec("E", p), vec("E", p)
        if len(label) == 1:
            return vec(label, p), vec("E", p)
        return vec(label[0], p), vec(label[1], p)

    def frames(*spec):
        T, U = [], []
        for lab, p in spec:
            t, u = frame(lab, p)
            T.append(t); U.append(u)
        return T, U

    # 1. a_maxconf / b_maxconf equal the production rule on any frame set.
    for spec in ([("23", 0.9), ("23", 0.6), ("28", 0.95)],
                 [("7", 0.5), ("-1", 0.9), ("7", 0.5)],
                 [("10", 0.99)], []):
        T, U = frames(*spec)
        ref = consolidate_tracklet_maxconf(T, U)[0]
        r = fuse(T, U, T, U)
        assert r["a_maxconf"] == ref and r["b_maxconf"] == ref, (spec, r, ref)
        # identical inputs: every symmetric rule must agree with the baseline
        for k in ("joint_maxconf", "agree_first", "sum", "prod", "prod_plus1"):
            assert r[k] == ref, (k, spec, r[k], ref)

    # 2. empty tracklet -> "-1" under every rule
    assert set(fuse([], [], [], []).values()) == {"-1"}

    # 3. one-sided labels: A reads 23 confidently, B reads 28 (never 23).
    TA, UA = frames(("23", 0.9), ("23", 0.9))
    TB, UB = frames(("28", 0.6), ("28", 0.6))
    r = fuse(TA, UA, TB, UB)
    assert r["a_maxconf"] == "23" and r["b_maxconf"] == "28"
    # a_mc x b_count: 23 has B-count 0 -> score 0; 28 has A-maxconf 0 -> 0.
    # All-zero scores fall to the tie rule: equal votes (2 vs 2), strength
    # decides: 23's frames are more confident (larger raw log-lik) -> 23.
    assert r["a_mc_x_bcount"] == "23"
    assert r["b_mc_x_acount"] == "23"
    assert r["prod"] == "23"
    # +1 variants: 23 -> mcA*1 vs 28 -> 0  => 23 ; b side: 28 -> mcB*1 => 28
    assert r["a_mc_x_bcount1"] == "23" and r["b_mc_x_acount1"] == "28"
    # sum: mcA(23) = 0.81*2*0.81=1.31 > mcB(28)=0.36*0.72=0.26 -> 23
    assert r["sum"] == "23" and r["prod_plus1"] == "23"
    assert r["joint_maxconf"] == "23"
    assert r["agree_first"] == "23"          # disagree -> joint -> 23
    assert r["vote_pool"] == "23"            # 2 vs 2, strength breaks -> 23

    # 4. count-weighting flips a label: A best frame says 45 (p .95) once, but
    #    A also read 23 twice at .6; B read 23 four times at .7 and never 45.
    TA, UA = frames(("45", 0.95), ("23", 0.6), ("23", 0.6))
    TB, UB = frames(("23", 0.7), ("23", 0.7), ("23", 0.7), ("23", 0.7))
    r = fuse(TA, UA, TB, UB)
    assert r["a_maxconf"] == "45"            # exp(2*log .95)*.9025 = .81 > .36*.72=.26
    assert r["a_mc_x_bcount"] == "23"        # 45 has b_count 0
    assert r["b_mc_x_acount"] == "23"
    assert r["vote_pool"] == "23"
    assert r["sum"] == "23"                  # .26 + .49*4*.49=.96 -> 1.22 > .81
    assert r["joint_maxconf"] == "23"        # exp(max mx 23)=.49 * (0.72+1.96)=1.31 > .81
    assert r["agree_first"] == "23"

    # 5. agreement short-circuits: both models pick 9 -> 9 regardless of joint
    TA, UA = frames(("9", 0.8), ("6", 0.5))
    TB, UB = frames(("9", 0.7), ("6", 0.5), ("6", 0.5))
    r = fuse(TA, UA, TB, UB)
    assert r["a_maxconf"] == "9" and r["b_maxconf"] == "9"
    assert r["agree_first"] == "9"

    # 6. determinism: label string breaks a perfect tie
    TA, UA = frames(("3", 0.7)); TB, UB = frames(("8", 0.7))
    r = fuse(TA, UA, TB, UB)
    assert r["vote_pool"] == "8" and r["sum"] == "8"

    print("fuse_jn self-tests OK:", len(RULES), "rules")
