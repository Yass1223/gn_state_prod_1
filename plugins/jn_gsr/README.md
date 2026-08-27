# JN pipeline -- GSR production build (PARSeq + SATRN, vote_pool)

Jersey-number recognition pipeline for integration into a GSR (SoccerNet
GameState) pipeline. One configuration, one consolidation rule, four metrics.

## Configuration

    recogniser A      PARSeq parseq_gsr_ft_s1.ckpt   (Ynniss/final_parseq_jn)
    recogniser B      SATRN  recog2/best_recog_word_acc_epoch_10.pth
                      (Ynniss/satrn_small; mmocr 1.x, SATRN encoder + NRTR
                      decoder, 93-class english_digits_symbols dictionary --
                      all read from the config stored inside the checkpoint)
    legibility gate   pl > 0.72       (strict; frame_verdicts is the rule)
    det gate          ds > 0.52       (cached DBNet++ top-quad score)
    consolidation     vote_pool       both recognisers decode every surviving
                                      frame; score(L) = n_A(L) + n_B(L); ties
                                      on pooled votes, then summed raw
                                      log-likelihood, then label string
                                      (fuse_jn.py)
    -1                a tracklet with no surviving ROI frame

`vote_pool` is the only rule in `jn_recognizer.py`, `predict_tracklets.py` and
the host module. The single-model rules (`maxconf`, `wvote`, `vote`, ...)
remain in `evaluate_jn.py` / `run_eval.py --rule` for the evaluation harness
only.

The legibility gate is a real parameter, not a constant: `jn_gsr.yaml` ->
`jn_gsr_api` -> `predict_tracklets.py --legibility-thr` ->
`JerseyNumberRecognizer(legibility_thr=...)`, and it is part of both cache
keys, so changing it in the config re-runs rather than serving the previous
numbers. The same chain carries `--parseq-ckpt` / `--satrn-ckpt`, and the
host cache key hashes the CONTENT of both checkpoints.

Both recogniser downloads carry a `.zip` extension but are `torch.save`
containers, not archives of files: they are staged by copy/rename, never
unzipped. PARSeq's staged name must contain `parseq` and none of `abinet`,
`crnn`, `trba`, `trbc`, `vitstr` (strhub routes on the path). SATRN's config
is written next to it as `recog2/config_from_checkpoint.py` by
`fetch_weights.py`; its dictionary is named there by an absolute path from the
training machine and is resolved by `mmocr_reader.py` to the same file inside
the installed mmocr, then checked against the checkpoint's classifier width.

`jersey_number_confidence` (host output) is the winner's share of the pooled
frame votes, `(n_A(L) + n_B(L)) / (2 * n_used)`; 0.0 when no frame survives.

## Measured numbers

Fusion run of 2026-08-17 (`kaggle-fusion-1.ipynb` in the fusion package,
harness `run_eval.py --recog2` + `run_fusion.py`), GT-box tracklets,
legibility > 0.72, det > 0.52, stride 5, every audit stage 0 fail
(weights 21/0/0; cache 32/2/0 and 33/2/0 -- the two WARNs are the known
"0"/"00" labels; merge 14/0/0 on both splits; `a_maxconf` reproduced the
one-model `run_eval` merge to the last digit):

| split | rule | trk_acc | numbered | -1 F1 | roi kept |
|---|---|---|---|---|---|
| GSR-2024 test, 1038 trk | PARSeq maxconf | 87.48% | 90.83% | 0.82 | 57.2% |
| GSR-2024 test, 1038 trk | **vote_pool** | **87.86%** | **91.34%** | 0.82 | 57.2% |
| SN jersey-2023 test, 1211 trk | PARSeq maxconf | 84.72% | 83.76% | 0.81 | 58.6% |
| SN jersey-2023 test, 1211 trk | **vote_pool** | **85.30%** | **84.58%** | 0.81 | 58.6% |

Versus PARSeq maxconf: GSR 19 tracklets changed, 5 fixed, 1 broken (+4);
SN-JNR 18 changed, 9 fixed, 2 broken (+7). The two single-model answers
already agreed on 96.1% of GSR tracklets.

Caveats: GT boxes, not predicted tracklets; the rule was chosen on these
test splits (no held-out selection). No number exists yet for this build
through the host pipeline on predicted tracklets -- `scripts/reference_metrics.py`
on a saved state is what produces it.

The SUPERSEDED single-model reference row (previous checkpoint, legibility
0.9: maxconf 84.20 / 85.01 / 0.83) is retained only as the harness check in
`audit_parseq.py --stages d6`.

## Kaggle validation

Provision with `bash scripts/setup_jn_gsr.sh` from the host repo (venv ->
hash-checked weights incl. SATRN -> offline self-tests -> PARSeq audit), then
run the host pipeline. `kaggle_gsr_maxconf.ipynb` is the stand-alone
validation notebook of the PREVIOUS (single-model maxconf) build; it has not
been updated for this build and its commands do not exercise SATRN.

## Integration surface

    from jn_recognizer import JerseyNumberRecognizer
    rec = JerseyNumberRecognizer("models")        # 4 nets; vote_pool
    number, confidence, n_used = rec.predict(tracklet)

`tracklet` = iterable of (pil_image_or_path, xywh) or bare pre-cropped
images; `number` is "1".."99" or "-1"; `confidence` the pooled vote share.
Constructor arguments `parseq_ckpt` / `satrn_ckpt` are paths relative to
`models_dir`. The worker the host pipeline executes is `predict_tracklets.py`
(`--parseq-ckpt`, `--satrn-ckpt`, `--legibility-thr`; no `--rule`).

The evaluation harness (`run_eval.py` worker + `--merge`, `--rule
vote|wvote|sum|slogconf|sumexp|maxconf`) is single-model PARSeq and unchanged;
the two-model cache path (`--recog2`) lives in the fusion package, not here.

The cache fingerprint of `run_eval.py` keys on the PARSeq checkpoint's
**content**, so a merge normally needs the file present. For a merge-only
session whose cache outlived its 286 MB checkpoint (a resumed Kaggle session),
`--parseq-sha <digest>` declares the identity instead -- take it from the
cache's own `provenance.parseq.sha256`. It is accepted at `--merge` only.

## Weights

`fetch_weights.py --out-dir models` fetches DBNet++ (sha256 asserted),
the legibility ResNet-34 (sha256 asserted) and SATRN (checked against the
16-hex sha256 prefix `9e8f73b300754c35` observed by the audited run; the
full digest is written to `models/fetch_weights_provenance.json`).
`stage_weights.py --out models/parseq_gsr_ft_s1.ckpt` fetches PARSeq and
writes `models/weights_provenance.json` (read by
`scripts/verify_run_integrity.py`). Mismatches are reported, not fatal;
`audit_parseq.py` is the arm that asserts for PARSeq.

## Files

run_eval.py (single-model harness: worker + merge), jn_recognizer.py
(per-tracklet API, two recognisers, vote_pool), predict_tracklets.py (the
worker the host pipeline executes), fuse_jn.py (two-recogniser rules;
`python fuse_jn.py` self-tests), mmocr_reader.py (SATRN loader under the
PARSeq reader contract; `python mmocr_reader.py` self-tests), audit_parseq.py
(PARSeq checkpoint audit, D1-D6), evaluate_jn.py (single-model rules +
metrics; `python evaluate_jn.py` self-tests), legibility / crop_classifier /
roi_dbnet / dbnet_infer (models), gsr_adapter + stage_data (GSR loading),
fetch_weights + stage_weights (hash-checked checkpoints), dual_gpu (sharded
worker), common / stage_utils / setup_env / setup_kaggle (infra),
kaggle_gsr_maxconf.ipynb (previous build's validation notebook),
MANIFEST.sha256 (verified by the notebook bootstrap). SN-JNR support was
removed from this build; use the full harness for cross-dataset work.
