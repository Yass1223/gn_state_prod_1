# Tests

Run from the repository root with the project venv (`uv venv --python 3.9 .venv && uv pip install --python .venv -e .`):

```bash
.venv/bin/python tests/test_rules_equivalence.py   # ported rules == notebook functions (numpy/sklearn only)
.venv/bin/python tests/test_split_merge.py         # split_merge port == notebook cells; behaviour; stage + audit check (needs torch + tracklab for part 3)
.venv/bin/python tests/test_stages.py              # crop_filter -> team_embed -> role_team on synthetic frames (needs torch + torchreid)
.venv/bin/python tests/test_audit.py               # the audit checks for the three stages, with negative controls
```

* `notebook_reference.py` is the original notebook code (cells 11 and 13 of
  `role-team-gt-tracks-3.ipynb`), kept verbatim as the reference the port is compared to.
* `notebook_split_merge_reference.py` is the production part of
  `splitter-plus-merger-final.ipynb` (splitter and merger cells), kept verbatim; `test_split_merge.py`
  runs it against `sn_gamestate/track/split_merge.py` on random videos and checks the stage's
  `track_id` output, its audit sidecar and the audit's split_merge check (positive and negative).
* `test_stages.py` builds a synthetic `osnet_team` checkpoint from the same architecture,
  so it verifies loading, preprocessing, flip TTA and the column contracts — not the real
  weights. The real checkpoint is exercised by a pipeline run on Kaggle.
