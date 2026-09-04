# Tests

Run from the repository root with the project venv (`uv venv --python 3.9 .venv && uv pip install --python .venv -e .`):

```bash
.venv/bin/python tests/test_rules_equivalence.py   # ported rules == notebook functions (numpy/sklearn only)
.venv/bin/python tests/test_tracklet_split.py      # DBSCAN split-only stage algorithm (numpy/sklearn only)
.venv/bin/python tests/test_traj_refine.py         # label-aware merge + stage-3 duplicate-frame resolution (numpy only)
.venv/bin/python tests/test_jersey_single_crops.py # jersey manifest = single crops only; cache key; audit jersey check with negative controls (needs tracklab, no torch)
.venv/bin/python tests/test_stages.py              # crop_filter -> team_embed -> role_team on synthetic frames (needs torch + torchreid)
.venv/bin/python tests/test_audit.py               # the audit checks for the three stages, with negative controls
```

* `notebook_reference.py` is the original notebook code (cells 11 and 13 of
  `role-team-gt-tracks-3.ipynb`), kept verbatim as the reference the port is compared to.
* `test_tracklet_split.py` covers the split-only stage algorithm
  (`sn_gamestate/track/tracklet_split.py`): per-tracklet DBSCAN over all
  detections, noise attachment, all-multi dissolution, clean-only centroids,
  the (tracklet, frame) input invariant and determinism.
* `test_stages.py` builds a synthetic `osnet_team` checkpoint from the same architecture,
  so it verifies loading, preprocessing, flip TTA and the column contracts — not the real
  weights. The real checkpoint is exercised by a pipeline run on Kaggle.
