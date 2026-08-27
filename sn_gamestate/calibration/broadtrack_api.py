"""BroadTrack (EVS, WACV'25) as a TrackLab ``VideoLevelModule``.

Replaces the NBJW pair (``pitch`` = ``NBJW_Calib_Keypoints`` + ``calibration`` =
``NBJW_Calib``) with BroadTrack's temporal camera tracking, and performs the
image-to-pitch projection with the *same* code path the repo already uses
(``sn_calibration_baseline.Camera.unproject_point_on_planeZ0``), so the ``bbox_pitch``
contract is byte-compatible with the previous stages.

Engine contract (verified against tracklab 1.3.24 sources)
----------------------------------------------------------
* ``OfflineTrackingEngine.video_loop`` dispatches video-level modules as
  ``detections = model.process(detections, image_pred)`` — the return value MUST be a
  DataFrame (returning a ``(detections, metadatas)`` tuple would corrupt the loop).
* ``image_pred`` is passed by reference and later persisted via
  ``TrackerState.on_video_loop_end -> update -> save``, so the ``parameters`` image
  column is written **in place** on ``metadatas`` here and does persist.
* ``Pipeline.validate`` starts from empty column sets on a fresh run and accumulates
  declared outputs, so inputs below are producible upstream (``bbox_ltwh``/``image_id``
  come from the detector wrapper) and image-level ``parameters`` is declared as an
  output (dict form) so bookkeeping and state save/load stay coherent.

Verified behaviours of the BroadTrack binary this wrapper compensates for
-------------------------------------------------------------------------
* Frames must be ``%06d.jpg`` starting at ``000001.jpg``; ``frames_number`` counts the
  *regular files* in ``-f`` and the loop is ``for (i = 1; i < frames_number; i++)`` so
  the **last frame is never calibrated** -> carry-forward (``use_prev_parameters``).
* Output JSON keys are full path strings built from ``-f`` -> matched by basename.
* Player masking is read from ``<frames_dir>/human-bboxes/%06d.json`` (path hardcoded
  in ``main.cpp``; the ``-b`` flag is parsed but unused) -> ``write_human_bboxes``.
* ``-t <file>`` existing switches the binary to tripod ("soft") mode; otherwise "free"
  mode with the ``--X/--Y/--Z`` prior. The binary's built-in defaults are ``0/90/-18``
  (Bundesliga), *not* the SoccerNet prior, so priors are always passed explicitly.

Schema conversion
-----------------
See :func:`broadtrack_cp_to_sncalib`. Validated numerically against an independent
reimplementation of the C++ projection model driven only by raw JSON values:
max projection disagreement ~3e-5 px, plane-unprojection roundtrip < 1 mm across
zoomed/wide/rolled configurations; omitting the distortion rescale diverges by
>1000 px (negative control). Run ``scripts/verify_broadtrack_conversion.py`` on real
output before benchmarking.

No Docker: build the binary natively with ``scripts/setup_broadtrack.sh``. Nothing from
EVS is vendored in this repo (licence: noncommercial research, no redistribution).
"""
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sn_calibration_baseline.camera import Camera
from tracklab.pipeline.videolevel_module import VideoLevelModule

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Schema conversion : BroadTrack "cp" object  ->  sn-calibration camera parameters
# --------------------------------------------------------------------------------------
def broadtrack_cp_to_sncalib(cp: dict) -> dict:
    """Convert one BroadTrack ``cp`` object to the sn-calibration parameter dict.

    BroadTrack (``Camera::toJSONString``) emits::

        sensorResolutionWidthPixels, sensorResolutionHeightPixels,
        horizontalFieldOfViewDegrees, panDegrees, tiltDegrees, rollDegrees,
        positionXMeters, positionYMeters, positionZMeters,
        normalizedRadialDistortionCoefficients

    ``sn_calibration_baseline.Camera.from_json_parameters`` requires::

        principal_point, x_focal_length, y_focal_length,
        pan_degrees, tilt_degrees, roll_degrees, position_meters,
        radial_distortion (6), tangential_distortion (2), thin_prism_distortion (4)

    Mapping rationale (verified against the C++ and BroadTrack's own
    ``scripts/compute_tripod.py``):

    * **Angles / position** are identical conventions: ``compute_tripod.py`` parses this
      JSON with the same ``pan_tilt_roll_to_orientation`` + transpose used by
      ``sn_calibration_baseline``. No sign flips, no reordering.
    * **Focal length**: ``getHorizontalFieldOfView() = 2*atan2(W/2, f)`` so
      ``f = (W/2) / tan(hFoV/2)``; square pixels => ``fx = fy``; principal point at the
      image centre (the paper reduces the intrinsics to a single ``f``).
    * **Radial distortion**: ``Camera::distort`` applies ``1 + sum k_i r^(2i)`` with
      ``r`` on the normalized image plane (OpenCV / sn-calibration convention), but
      ``getNormalizedRadialDistortion()`` exports
      ``k_i_json = k_i_internal * (H/f)^(2(i+1))`` (radius rescaled by image *height*).
      Inverted here: ``k_i_sncalib = k_i_json * (f/H)^(2(i+1))``. Only ``k1`` is
      modelled by BroadTrack; the remaining slots stay zero. This scaling is covered by
      a machine-precision test in ``scripts/verify_broadtrack_conversion.py``.
    """
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    hfov = float(cp["horizontalFieldOfViewDegrees"]) * np.pi / 180.0
    focal = (w / 2.0) / np.tan(hfov / 2.0)

    radial = [0.0] * 6
    for i, k_json in enumerate(cp.get("normalizedRadialDistortionCoefficients", []) or []):
        if i >= 3:  # sn-calibration numerator terms are k1..k3 (slots 3..5 = k4..k6)
            break
        radial[i] = float(k_json) * (focal / h) ** (2.0 * (i + 1))

    return {
        "principal_point": [w / 2.0, h / 2.0],
        "x_focal_length": focal,
        "y_focal_length": focal,
        "pan_degrees": float(cp["panDegrees"]),
        "tilt_degrees": float(cp["tiltDegrees"]),
        "roll_degrees": float(cp["rollDegrees"]),
        "position_meters": [
            float(cp["positionXMeters"]),
            float(cp["positionYMeters"]),
            float(cp["positionZMeters"]),
        ],
        "radial_distortion": radial,
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }


def get_bbox_pitch(cam):
    """Bottom-left / bottom-right / bottom-middle unprojection onto the Z=0 plane.

    Identical to ``sn_gamestate.calibration.bbox2pitch.get_bbox_pitch`` so the
    ``bbox_pitch`` schema and numerics match the rest of the pipeline exactly.
    ``unproject_point_on_planeZ0`` undistorts by default, so BroadTrack's k1 is honoured.
    """
    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = np.array([l, b, 1])
        br = np.array([r, b, 1])
        bm = np.array([l + (r - l) / 2, b, 1])
        pbl_x, pbl_y, _ = cam.unproject_point_on_planeZ0(bl)
        pbr_x, pbr_y, _ = cam.unproject_point_on_planeZ0(br)
        pbm_x, pbm_y, _ = cam.unproject_point_on_planeZ0(bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox


# --------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------
class BroadTrackCalibration(VideoLevelModule):
    """Camera calibration + image-to-pitch projection driven by the BroadTrack binary."""

    # Dict form so image-level "parameters" is declared (Pipeline.validate and the
    # TrackerState load/save column bookkeeping both use get_*_columns, which handles
    # dicts; the deprecated validate_input/validate_output are never called in 1.3.24).
    input_columns = {"detection": ["bbox_ltwh", "image_id"], "image": []}
    output_columns = {"detection": ["bbox_pitch"], "image": ["parameters"]}

    def __init__(self, cfg, device, tracking_dataset=None):
        self.cfg = cfg
        self.device = device

        self.binary = Path(str(cfg.binary))
        self.kp_model = Path(str(cfg.keypoint_model))
        self.line_model = Path(str(cfg.line_model))
        self.libtorch_lib = str(getattr(cfg, "libtorch_lib", "") or "")

        self.calib_dir = Path(str(cfg.calib_dir))
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.use_cached_json = bool(getattr(cfg, "use_cached_json", True))
        self.timeout = int(getattr(cfg, "timeout", 7200))

        self.prior_xyz = [float(v) for v in getattr(cfg, "prior_xyz", [0.0, 55.0, -12.0])]
        self.tripod_mode = str(getattr(cfg, "tripod_mode", "none"))  # none|per_sequence|per_game
        self.tripod_dir = Path(str(getattr(cfg, "tripod_dir", self.calib_dir / "tripod")))

        self.write_human_bboxes = bool(getattr(cfg, "write_human_bboxes", True))
        self.staging_dir = str(getattr(cfg, "staging_dir", "") or "")
        # Stage even when the frames dir IS writable. main.cpp hardcodes reading
        # player masks from <frames_dir>/human-bboxes/, so writing them in place
        # leaves a SUBDIRECTORY inside img1/. That is harmless for SoccerNet
        # sequences (the loader takes nframes from Labels-GameState.json's
        # seq_length) but breaks CUSTOM/UNLABELED video: with no labels file the
        # loader falls back to len(os.listdir(img1)), which counts the extra
        # directory and then requests a frame index one past the end ->
        # cv2.error: (-215:Assertion failed) !_src.empty(). Set true whenever
        # running inference on video that has no Labels-GameState.json.
        self.always_stage = bool(getattr(cfg, "always_stage", False))
        self.min_score = float(getattr(cfg, "min_score", 0.0))
        self.use_prev_parameters = bool(getattr(cfg, "use_prev_parameters", True))

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _frames_dir(metadatas: pd.DataFrame) -> Path:
        """Folder holding the %06d.jpg frames of this video (SNGS ``.../img1``)."""
        return Path(str(metadatas["file_path"].iloc[0])).parent

    @staticmethod
    def _sequence_name(frames_dir: Path) -> str:
        """`.../SNGS-116/img1` -> `SNGS-116` (falls back to the folder name)."""
        return frames_dir.parent.name or frames_dir.name

    def _prepare_frames_dir(self, frames_dir: Path, detections: pd.DataFrame,
                            metadatas: pd.DataFrame) -> Path:
        """Return the directory to pass to ``-f``, writing human-bboxes when possible.

        The binary reads masks from ``<frames_dir>/human-bboxes/%06d.json`` (hardcoded).
        If the dataset is mounted read-only (Kaggle), a directory of symlinks is staged
        so the masks can still be written; symlinks count as regular files, so the
        binary's ``frames_number`` stays correct. The ``human-bboxes`` *subdirectory*
        itself is not a regular file, so it does not shift ``frames_number`` either.
        """
        if not self.write_human_bboxes:
            return frames_dir

        target = frames_dir
        if self.always_stage or not os.access(frames_dir, os.W_OK):
            if not self.staging_dir:
                log.warning(
                    "[BroadTrack] frames dir is read-only and no staging_dir configured; "
                    "running without player masking."
                )
                return frames_dir
            target = Path(self.staging_dir) / self._sequence_name(frames_dir)
            target.mkdir(parents=True, exist_ok=True)
            for src in sorted(frames_dir.glob("*.jpg")):
                dst = target / src.name
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except OSError:  # no symlink permission -> copy
                        shutil.copy2(src, dst)

        try:
            bbox_dir = target / "human-bboxes"
            bbox_dir.mkdir(parents=True, exist_ok=True)
            # image_id -> frame stem, via the frame file name (%06d.jpg).
            id2stem = {idx: Path(str(p)).stem for idx, p in metadatas["file_path"].items()}
            for image_id, group in detections.groupby("image_id"):
                stem = id2stem.get(image_id)
                if stem is None:
                    continue
                boxes = []
                for ltwh in group["bbox_ltwh"]:
                    l, t, w, h = [float(v) for v in ltwh]
                    boxes.append([l, t, l + w, t + h])  # binary expects [x1, y1, x2, y2]
                (bbox_dir / f"{stem}.json").write_text(json.dumps({"bboxes": boxes}))
        except OSError as e:
            log.warning(f"[BroadTrack] could not write human-bboxes ({e}); "
                        f"running without player masking.")
        return target

    def _run_binary(self, frames_dir: Path, out_json: Path, tripod_file=None) -> bool:
        cmd = [
            str(self.binary),
            # Boost.ProgramOptions declares these as LONG option names ('f', 'o',
            # 'k', 'l', 't'); only 'help' has a short form ("help,h"). Single-dash
            # '-f' is therefore rejected with "unrecognised option '-f'".
            "--f", str(frames_dir),
            "--o", str(out_json),
            "--k", str(self.kp_model),
            "--l", str(self.line_model),
            "--X", str(self.prior_xyz[0]),
            "--Y", str(self.prior_xyz[1]),
            "--Z", str(self.prior_xyz[2]),
        ]
        if tripod_file is not None and Path(tripod_file).is_file():
            cmd += ["--t", str(tripod_file)]

        env = os.environ.copy()
        if self.libtorch_lib:
            env["LD_LIBRARY_PATH"] = f"{self.libtorch_lib}:{env.get('LD_LIBRARY_PATH', '')}"

        log.info(f"[BroadTrack] running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            log.error(f"[BroadTrack] timed out after {self.timeout}s on {frames_dir}")
            return False
        except OSError as e:
            log.error(f"[BroadTrack] could not execute '{self.binary}': {e}")
            return False
        if proc.returncode != 0:
            log.error(
                f"[BroadTrack] binary failed (rc={proc.returncode}).\n"
                f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
            )
            return False
        return out_json.is_file()

    def _tripod_file(self, seq_name: str, frames_dir: Path):
        """Two-pass tripod estimation (free run -> compute_tripod.py -> soft run).

        ``per_game`` shares one tripod file between every clip mapped to the same game
        key via ``cfg.game_of`` (the file is estimated from the first clip of that game
        that gets processed, then reused). SNGS clip names do not encode the game, so
        without a ``game_of`` mapping this degrades to per-sequence behaviour.
        """
        if self.tripod_mode == "none":
            return None
        self.tripod_dir.mkdir(parents=True, exist_ok=True)
        key = seq_name
        if self.tripod_mode == "per_game":
            mapping = dict(getattr(self.cfg, "game_of", {}) or {})
            key = mapping.get(seq_name, seq_name)
        tripod = self.tripod_dir / f"{key}.tripod.json"
        if tripod.is_file():
            return tripod

        free_json = self.calib_dir / f"{seq_name}.free.json"
        if not free_json.is_file():
            if not self._run_binary(frames_dir, free_json, tripod_file=None):
                return None
        script = Path(str(getattr(self.cfg, "compute_tripod_script", "")))
        if not script.is_file():
            log.warning(f"[BroadTrack] compute_tripod.py not found at '{script}'; "
                        f"continuing in free mode.")
            return None
        try:
            proc = subprocess.run(
                ["python", str(script), "-i", str(free_json), "-o", str(tripod)],
                capture_output=True, text=True, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning(f"[BroadTrack] compute_tripod.py failed ({e}); "
                        f"continuing in free mode.")
            return None
        if proc.returncode != 0 or not tripod.is_file():
            log.warning(f"[BroadTrack] compute_tripod.py failed; continuing in free "
                        f"mode. stderr: {proc.stderr[-1000:]}")
            return None
        return tripod

    # ---------------------------------------------------------------- main
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # NOTE (engine contract): must return a DataFrame (detections) only; the image
        # 'parameters' column is written on `metadatas` in place, which the offline
        # engine persists (image_pred is the same object across the video loop).
        if len(metadatas) == 0:
            return detections

        frames_dir = self._frames_dir(metadatas)
        seq_name = self._sequence_name(frames_dir)
        out_json = self.calib_dir / f"{seq_name}.json"

        calibs = None
        if self.use_cached_json and out_json.is_file():
            log.info(f"[BroadTrack] using cached calibration: {out_json}")
        else:
            if not self.binary.is_file():
                log.error(
                    f"[BroadTrack] binary not found at '{self.binary}'. Build it first "
                    f"with scripts/setup_broadtrack.sh. Skipping {seq_name}."
                )
                return self._empty_outputs(detections, metadatas)
            run_dir = self._prepare_frames_dir(frames_dir, detections, metadatas)
            tripod = self._tripod_file(seq_name, run_dir)
            if not self._run_binary(run_dir, out_json, tripod):
                return self._empty_outputs(detections, metadatas)

        try:
            calibs = json.loads(out_json.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.error(f"[BroadTrack] could not read '{out_json}': {e}")
            return self._empty_outputs(detections, metadatas)

        # Keys are full path strings built from -f -> index by basename.
        by_name = {Path(k).name: v for k, v in calibs.items()}
        log.info(f"[BroadTrack] {seq_name}: {len(by_name)} calibrated frames "
                 f"for {len(metadatas)} images")

        params_rows, bbox_pitch = {}, {}
        last_params = None
        n_carried = n_lowscore = 0

        # Temporal order matters for carry-forward: iterate frames sorted by file name
        # (%06d.jpg sorts chronologically); do not rely on the incoming row order.
        order = metadatas["file_path"].astype(str).map(lambda p: Path(p).name)
        for image_id in order.sort_values().index:
            entry = by_name.get(order[image_id])
            params = None
            if entry is not None:
                score = float(entry.get("score", 0.0))
                if score >= self.min_score:
                    params = broadtrack_cp_to_sncalib(entry["cp"])
                else:
                    n_lowscore += 1
            if params is None and self.use_prev_parameters and last_params is not None:
                params = last_params
                n_carried += 1
            if params is None:
                params_rows[image_id] = {}
                continue
            last_params = params
            params_rows[image_id] = params

            cam = Camera()
            cam.from_json_parameters(params)
            image_dets = detections[detections.image_id == image_id]
            if len(image_dets):
                projected = image_dets.bbox.ltrb().apply(get_bbox_pitch(cam))
                bbox_pitch.update(projected.to_dict())

        if n_carried or n_lowscore:
            log.info(f"[BroadTrack] {seq_name}: carried-forward {n_carried} frame(s), "
                     f"{n_lowscore} below min_score={self.min_score}")

        detections = detections.copy()
        col = pd.Series(bbox_pitch, dtype=object).reindex(detections.index)
        detections["bbox_pitch"] = col.where(col.notna(), None)  # NaN -> None (NBJW parity)

        params_col = pd.Series(params_rows, dtype=object).reindex(metadatas.index)
        metadatas["parameters"] = params_col.apply(
            lambda v: v if isinstance(v, dict) else {}
        )
        return detections

    @staticmethod
    def _empty_outputs(detections: pd.DataFrame, metadatas: pd.DataFrame):
        detections = detections.copy()
        detections["bbox_pitch"] = None
        metadatas["parameters"] = pd.Series(
            [{} for _ in range(len(metadatas))], index=metadatas.index, dtype=object
        )
        return detections
