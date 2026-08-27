# sn_gamestate.track — BoT-SORT + GTA-Link integration for the SoccerNet GSR pipeline.
# Kept intentionally empty: importing this package must stay cheap (no torch import here),
# because sn_gamestate.track.hf_resolver is imported early (at plugin discovery) to register
# the `${hf:...}` OmegaConf resolver. The heavy GTA-Link module is loaded lazily by Hydra
# via its `_target_` only when the pipeline is built.
