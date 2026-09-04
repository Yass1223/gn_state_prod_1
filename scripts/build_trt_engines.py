#!/usr/bin/env python3
"""Build a TensorRT FP16 engine for the SoccerNet GSR detector.

Run this ON THE MACHINE THAT WILL RUN INFERENCE — TensorRT engines are specific to the
GPU, driver, CUDA and TensorRT version, so they must be built where they'll be used.

What it builds
--------------
* YOLO11-L detector  -> ``<trt_dir>/yolov11_sn_best.engine``   (ultralytics native export)

Documented fallbacks (NOT converted here)
-----------------------------------------
* OSNet-AIN (tracker appearance and tracklet_split) runs on PyTorch in fp16 (torch autocast on
  CUDA, fp32 outputs), baked in after the Kaggle fp16/fp32 validation: both stages build
  the same embedder module so they produce the same embedding for the same crop, and the
  run audit fails on a checkpoint-pin mismatch. Neither stage has a ``use_tensorrt`` path.
* NBJW-Calib HRNet (pitch + calibration), prtreid (team ReID), MMOCR (jersey number)
  keep running on PyTorch. They use custom ops / multi-stage mm* pipelines / geometry
  post-processing that don't export to a single clean ONNX graph without mmdeploy or a
  bespoke exporter, which is out of scope.

Everything is best-effort and isolated: if a build step fails (missing tensorrt/onnx, an
unsupported op, an OOM at build time), it's caught, reported in the final status table, and
the corresponding runtime path falls back to PyTorch automatically (the wrappers check for
the ``.engine`` file and warn+fallback if it's absent). So a partial build never breaks
inference — it just means fewer engines.

Enable at runtime by setting ``use_tensorrt: true`` in ``soccernet.yaml`` (or
``+use_tensorrt=true`` on the CLI). NOTE: FP16 engines are not bit-identical to the
PyTorch pipeline, so keep ``use_tensorrt: false`` for exact HOTA-reproduction runs.

Examples
--------
    python scripts/build_trt_engines.py                      # fetch weights from HF, fp16
    python scripts/build_trt_engines.py --weights-dir pretrained_models
    python scripts/build_trt_engines.py --imgsz 1280 --det-batch 4
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_trt_engines")

DEFAULT_HF_REPO = "Ynniss/sn-gamestate-weights"
YOLO_PT = "yolov11_sn_best.pt"


# --------------------------------------------------------------------------- weights
def resolve_weight(name: str, weights_dir: str | None, hf_repo: str) -> Path:
    """Return a local path to ``name``: prefer a local weights dir, else fetch from HF."""
    if weights_dir:
        cand = Path(weights_dir) / name
        if cand.is_file():
            log.info(f"[weights] {name}: {cand}")
            return cand
        # also try a recursive search under the dir
        hits = list(Path(weights_dir).rglob(name))
        if hits:
            log.info(f"[weights] {name}: {hits[0]}")
            return hits[0]
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(repo_id=hf_repo, filename=name))
    log.info(f"[weights] {name}: {p}  (from HF {hf_repo})")
    return p


# --------------------------------------------------------------------------- YOLO
def build_yolo_engine(pt_path: Path, out_path: Path, imgsz: int, batch: int, fp16: bool) -> str:
    """Export the YOLO detector to a TensorRT engine via ultralytics; return status str."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        return f"SKIP (ultralytics import failed: {e})"
    try:
        model = YOLO(str(pt_path), task="detect")
        # ultralytics writes the engine next to the .pt and returns its path.
        exported = model.export(
            format="engine",
            half=fp16,
            imgsz=imgsz,
            dynamic=True,
            batch=batch,
            device=0,
            verbose=False,
        )
        exported = Path(str(exported))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != out_path.resolve():
            shutil.copy(str(exported), str(out_path))
        return f"OK -> {out_path.name}  (imgsz={imgsz}, max_batch={batch}, fp16={fp16})"
    except Exception as e:
        return f"SKIP (export failed: {e})"


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", default=None,
                    help="Local dir holding the .pt/.pth weights (else fetched from HF).")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                    help="HF repo id for weights (default: %(default)s).")
    ap.add_argument("--trt-dir", default="pretrained_models/trt",
                    help="Where to write engines (default: %(default)s).")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Detector inference size — must match the config (default: 1280).")
    ap.add_argument("--det-batch", type=int, default=4,
                    help="Max detector batch for the engine profile (default: 4).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="Build FP32 engines instead of FP16.")
    ap.set_defaults(fp16=True)
    args = ap.parse_args(argv)

    trt_dir = Path(args.trt_dir)
    trt_dir.mkdir(parents=True, exist_ok=True)

    status = {}

    # 1) YOLO detector
    try:
        yolo_pt = resolve_weight(YOLO_PT, args.weights_dir, args.hf_repo)
        status["YOLO11-L detector"] = build_yolo_engine(
            yolo_pt, trt_dir / "yolov11_sn_best.engine",
            imgsz=args.imgsz, batch=args.det_batch, fp16=args.fp16,
        )
    except Exception as e:
        status["YOLO11-L detector"] = f"SKIP (weights unavailable: {e})"

    # 2) documented fallbacks (not converted; run on PyTorch)
    status["OSNet-AIN (tracker + tracklet_split)"] = (
        "FALLBACK (PyTorch, fp16 autocast) — by design: both stages must embed a "
        "crop identically, enforced by the run audit")
    status["NBJW-Calib HRNet (pitch/calibration)"] = (
        "FALLBACK (PyTorch) — geometry post-processing / custom pipeline, no clean ONNX")
    status["prtreid (team ReID)"] = (
        "FALLBACK (PyTorch) — torchreid multi-branch head, not exported here")
    status["MMOCR (jersey number)"] = (
        "FALLBACK (PyTorch) — mm* det+recog multi-stage, needs mmdeploy")

    # --- report ---
    print("\n" + "=" * 74)
    print(" TensorRT build summary")
    print("=" * 74)
    width = max(len(k) for k in status)
    for k, v in status.items():
        print(f"  {k.ljust(width)} : {v}")
    print("=" * 74)
    print(f" Engines dir: {trt_dir.resolve()}")
    print(" Enable with `use_tensorrt: true` (keep it false for exact HOTA reproduction).")
    print("=" * 74 + "\n")

    # Non-zero exit if the one convertible engine failed (nothing to gain from TRT).
    return 0 if status["YOLO11-L detector"].startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main())
