# sn_gamestate.track — BoT-SORT + tracklet_split integration for the SoccerNet GSR pipeline.
# Kept intentionally empty: importing this package must stay cheap (no torch import here),
# because sn_gamestate.track.hf_resolver is imported early (at plugin discovery) to register
# the `${hf:...}` OmegaConf resolver. The heavy stage modules (bot_sort, tracklet_split_api)
# are loaded lazily by Hydra via their `_target_` only when the pipeline is built;
# tracklet_split (the algorithm) imports only numpy and scikit-learn.
