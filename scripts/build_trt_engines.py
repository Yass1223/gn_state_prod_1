#!/usr/bin/env python3
"""Build TensorRT FP16 engines for the SoccerNet GSR (BoT-SORT · SOF + GTA-Link) pipeline.

Run this ON THE MACHINE THAT WILL RUN INFERENCE — TensorRT engines are specific to the
GPU, driver, CUDA and TensorRT version, so they must be built where they'll be used.

What it builds
--------------
* YOLO11-L detector  -> ``<trt_dir>/yolov11_sn_best.engine``   (ultralytics native export)
* sports-OSNet ReID  -> ``<trt_dir>/osnet_x1_0_sports.engine``  (ONNX -> TensorRT)
      One engine, consumed by BOTH the BoT-SORT ReID backend and GTA-Link. Bindings are
      named ``images`` / ``output`` and the filename contains ``osnet_x1_0`` so tracklab's
      ReIDDetectMultiBackend recognises the architecture and loads it.

Documented fallbacks (NOT converted here)
-----------------------------------------
* NBJW-Calib HRNet (pitch + calibration), prtreid (team ReID), MMOCR (jersey number)
  keep running on PyTorch. They use custom ops / multi-stage mm* pipelines / geometry
  post-processing that don't export to a single clean ONNX graph without mmdeploy or a
  bespoke exporter, which is out of scope. Forcing FP16 on them is also not done, to keep
  the pipeline coherent (these nets aren't validated for half precision). The pipeline
  runs them exactly as before; only the detector and the two OSNet instances are TRT-FP16.

Everything is best-effort and isolated: if a build step fails (missing tensorrt/onnx, an
unsupported op, an OOM at build time), it's caught, reported in the final status table, and
the corresponding runtime path falls back to PyTorch automatically (the wrappers check for
the ``.engine`` file and warn+fallback if it's absent). So a partial build never breaks
inference — it just means fewer engines.

Enable at runtime by setting ``use_tensorrt: true`` in ``soccernet_botsort.yaml`` (or
``+use_tensorrt=true`` on the CLI). NOTE: FP16 engines are not bit-identical to the
PyTorch pipeline, so keep ``use_tensorrt: false`` for exact HOTA-reproduction runs.

Examples
--------
    python scripts/build_trt_engines.py                      # fetch weights from HF, fp16
    python scripts/build_trt_engines.py --weights-dir pretrained_models
    python scripts/build_trt_engines.py --imgsz 1280 --det-batch 4 --reid-max-batch 64
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
OSNET_CKPT = "sports_model.pth.tar-60"


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


# --------------------------------------------------------------------------- OSNet
def _load_osnet(ckpt_path: Path):
    """Build osnet_x1_0 with an Identity classifier and load the sports weights."""
    import torch.nn as nn
    from strong_sort.deep.models import build_model
    from strong_sort.deep.reid_model_factory import load_pretrained_weights

    model = build_model("osnet_x1_0", num_classes=1, pretrained=False, use_gpu=False)
    load_pretrained_weights(model, str(ckpt_path))
    model.classifier = nn.Identity()
    model.eval()
    return model


def _export_osnet_onnx(model, onnx_path: Path) -> None:
    import torch
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["images"], output_names=["output"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    )


def _onnx_to_trt(onnx_path: Path, engine_path: Path, max_batch: int, opt_batch: int,
                 fp16: bool, workspace_gib: float) -> None:
    """Parse the ONNX and build a serialized TensorRT engine with a dynamic batch profile."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed: {msgs}")

    config = builder.create_builder_config()
    ws = int(workspace_gib * (1 << 30))
    # Workspace limit API differs across TRT versions.
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)  # TRT 8.4+/10
    except Exception:
        try:
            config.max_workspace_size = ws  # TRT < 8.4
        except Exception:
            pass
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 256, 128),
                      (opt_batch, 3, 256, 128), (max_batch, 3, 256, 128))
    config.add_optimization_profile(profile)

    # build_serialized_network is available on TRT 8.0+ and 10.x.
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)


def build_osnet_engine(ckpt_path: Path, out_path: Path, tmp_dir: Path,
                       max_batch: int, opt_batch: int, fp16: bool) -> str:
    try:
        import tensorrt  # noqa: F401
    except Exception as e:
        return f"SKIP (tensorrt import failed: {e})"
    try:
        model = _load_osnet(ckpt_path)
    except Exception as e:
        return f"SKIP (osnet load failed: {e})"
    try:
        onnx_path = tmp_dir / "osnet_x1_0_sports.onnx"
        _export_osnet_onnx(model, onnx_path)
    except Exception as e:
        return f"SKIP (onnx export failed: {e})"
    try:
        _onnx_to_trt(onnx_path, out_path, max_batch=max_batch, opt_batch=opt_batch,
                     fp16=fp16, workspace_gib=2.0)
        return (f"OK -> {out_path.name}  (dynamic batch 1..{max_batch}, "
                f"opt={opt_batch}, fp16={fp16})")
    except Exception as e:
        return f"SKIP (trt build failed: {e})"


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
    ap.add_argument("--reid-max-batch", type=int, default=64,
                    help="Max OSNet batch for the engine profile (default: 64).")
    ap.add_argument("--reid-opt-batch", type=int, default=32,
                    help="Optimised OSNet batch for the engine profile (default: 32).")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false",
                    help="Build FP32 engines instead of FP16.")
    ap.set_defaults(fp16=True)
    args = ap.parse_args(argv)

    trt_dir = Path(args.trt_dir)
    tmp_dir = trt_dir / "_tmp"
    trt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

    # 2) sports-OSNet (shared by BoT-SORT ReID + GTA-Link)
    try:
        osnet_ckpt = resolve_weight(OSNET_CKPT, args.weights_dir, args.hf_repo)
        status["sports-OSNet ReID (BoT-SORT + GTA-Link)"] = build_osnet_engine(
            osnet_ckpt, trt_dir / "osnet_x1_0_sports.engine", tmp_dir,
            max_batch=args.reid_max_batch, opt_batch=args.reid_opt_batch, fp16=args.fp16,
        )
    except Exception as e:
        status["sports-OSNet ReID (BoT-SORT + GTA-Link)"] = f"SKIP (weights unavailable: {e})"

    # 3) documented fallbacks (not converted; run on PyTorch)
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

    # Non-zero exit only if BOTH convertible engines failed (nothing to gain from TRT).
    built = [k for k in ("YOLO11-L detector", "sports-OSNet ReID (BoT-SORT + GTA-Link)")
             if status[k].startswith("OK")]
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
