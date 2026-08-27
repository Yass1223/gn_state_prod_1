# sn_gamestate.bbox_detector — SoccerNet-conformant YOLO detector wrapper.
# Kept intentionally empty (no torch import here): the Hydra config targets the
# submodule directly (sn_gamestate.bbox_detector.yolo_snft_api.YOLOUltralyticsSNFT),
# so the heavy module is imported lazily only when the pipeline is built.
