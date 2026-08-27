#!/usr/bin/env python
"""Verify the BroadTrack -> sn-calibration conversion on REAL output before benchmarking.

The conversion (``sn_gamestate/calibration/broadtrack_api.py``) is exact for angles,
position and focal length; the radial-distortion rescale
``k_sncalib = k_json * (f/H)^2`` is derived from the C++ source. This script checks the
whole chain on the actual JSON your run produced.

Checks
------
A. model-equivalence  - project 3D pitch-template points through (i) the converted
                        ``sn_calibration_baseline.Camera`` and (ii) an independent
                        reimplementation of BroadTrack's projection driven ONLY by raw
                        JSON values. Max disagreement must be < 0.01 px (measured
                        ~3e-5 px in synthetic tests; a forgotten rescale diverges by
                        1000+ px).
B. plane roundtrip    - unproject the projected pixels back onto Z=0; error < 1 cm on
                        genuinely visible points (gated by the *undistorted* projection
                        being in-frame: strong barrel distortion can fold far-outside
                        points into the frame, which is a model property, not a bug).
C. overlay            - draw the pitch wireframe with the converted camera; it must sit
                        on the painted lines (this is the ground-truth check; also
                        cross-check the binary's own overlays produced with ``-r``).
D. coverage/scores    - expect ``len(json) == n_frames - 1`` (the binary's loop is
                        ``i < frames_number`` so the last frame is never calibrated).

Usage
-----
    python scripts/verify_broadtrack_conversion.py \
        -c broadtrack_calib/SNGS-116.json \
        -f data/SoccerNetGS/test/SNGS-116/img1 \
        -o broadtrack_check --n 6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sn_gamestate.calibration.broadtrack_api import broadtrack_cp_to_sncalib  # noqa: E402
from sn_calibration_baseline.camera import (  # noqa: E402
    Camera,
    pan_tilt_roll_to_orientation,
)
from sn_calibration_baseline.soccerpitch import SoccerPitch  # noqa: E402


def project_direct_from_json(cp, P):
    """BroadTrack forward model reconstructed from raw JSON only.

    Returns (distorted_pixel, pinhole_pixel) or (None, None) if behind the camera.
    Distortion: offset scaling ``1 + sum k_json_i * (pixel_radius/H)^(2(i+1))`` — the
    algebraic identity of the C++ normalized-plane model with height-normalized export.
    """
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    hfov = np.radians(cp["horizontalFieldOfViewDegrees"])
    f = (w / 2.0) / np.tan(hfov / 2.0)
    R = np.transpose(pan_tilt_roll_to_orientation(
        np.radians(cp["panDegrees"]), np.radians(cp["tiltDegrees"]),
        np.radians(cp["rollDegrees"])))
    C = np.array([cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]])
    pc = R @ (np.asarray(P, float) - C)
    if pc[2] <= 1e-3:
        return None, None
    pp = np.array([w / 2.0, h / 2.0])
    d0 = f * (pc[:2] / pc[2])
    L = 1.0
    rho2 = (np.linalg.norm(d0) / h) ** 2
    for i, kj in enumerate(cp.get("normalizedRadialDistortionCoefficients", []) or []):
        L += float(kj) * rho2 ** (i + 1)
    return pp + d0 * L, pp + d0


def check_frame(cp):
    """Run checks A + B on one cp entry. Returns (max_proj_diff_px, max_roundtrip_m, n)."""
    params = broadtrack_cp_to_sncalib(cp)
    cam = Camera()
    cam.from_json_parameters(params)
    w = 2 * params["principal_point"][0]
    h = 2 * params["principal_point"][1]

    field = SoccerPitch()
    diffs, rts, n = [], [], 0
    for p3d in field.point_dict.values():
        P = np.asarray(p3d, float)
        qd, q0 = project_direct_from_json(cp, P)
        if qd is None:
            continue
        # visibility gate on the UNDISTORTED projection (see module docstring)
        if not (0 < q0[0] < w and 0 < q0[1] < h):
            continue
        qc = cam.project_point(P, distort=True)
        if qc[2] == 0:
            continue
        qc = (qc / qc[2])[:2]
        diffs.append(float(np.linalg.norm(qd - qc)))
        n += 1
        if abs(P[2]) < 1e-6:  # ground-plane points only for the Z=0 roundtrip
            back = cam.unproject_point_on_planeZ0(np.array([qc[0], qc[1], 1.0]))
            rts.append(float(np.linalg.norm(np.asarray(back[:2]) - P[:2])))
    dmax = max(diffs) if diffs else float("nan")
    rmax = max(rts) if rts else float("nan")
    return dmax, rmax, n


def corner_distortion_px(cp):
    """Informative: pixel displacement induced by k1 at the image corner."""
    ks = cp.get("normalizedRadialDistortionCoefficients", []) or []
    if not ks:
        return 0.0
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    r = np.hypot(w / 2.0, h / 2.0)
    L = 1.0
    for i, kj in enumerate(ks):
        L += float(kj) * (r / h) ** (2 * (i + 1))
    return abs(L - 1.0) * r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--calib", required=True, help="BroadTrack output JSON")
    ap.add_argument("-f", "--frames", required=True, help="frames directory (%%06d.jpg)")
    ap.add_argument("-o", "--out", default="broadtrack_check", help="output folder")
    ap.add_argument("--n", type=int, default=6, help="number of frames to check/overlay")
    ap.add_argument("--min-score", type=float, default=0.6)
    args = ap.parse_args()

    calibs = json.loads(Path(args.calib).read_text())
    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- D. coverage ---------------------------------------------------------------
    n_frames = len(list(frames_dir.glob("*.jpg")))
    print(f"[coverage] frames on disk: {n_frames} | calibrated entries: {len(calibs)}")
    if len(calibs) != max(n_frames - 1, 0):
        print("  !! expected n_frames - 1 (the binary never calibrates the last frame).")
        print("     Check for stray non-image files in the frames directory.")

    by_name = {Path(k).name: v for k, v in calibs.items()}
    scores = [float(v.get("score", 0.0)) for v in by_name.values()]
    reinits = sum(bool(v.get("reinit", False)) for v in by_name.values())
    print(f"[scores] mean={np.mean(scores):.3f} min={np.min(scores):.3f} "
          f"| reinit frames: {reinits}")
    good = [(k, v) for k, v in sorted(by_name.items())
            if float(v.get("score", 0.0)) >= args.min_score]
    print(f"[scores] frames with score >= {args.min_score}: {len(good)}/{len(by_name)}")
    if not good:
        print("  !! no high-confidence frames; inspect the run before continuing.")
        return 1

    try:
        import cv2
    except ImportError:
        cv2 = None
        print("[overlay] opencv not available - skipping image overlays (check C)")

    # -- A + B on sampled high-confidence frames ------------------------------------
    step = max(1, len(good) // max(args.n, 1))
    all_dmax, all_rmax = [], []
    for name, entry in good[::step][:args.n]:
        cp = entry["cp"]
        dmax, rmax, n = check_frame(cp)
        all_dmax.append(dmax)
        all_rmax.append(rmax)
        print(f"[{name}] score={entry['score']:.3f} "
              f"k1_json={(cp.get('normalizedRadialDistortionCoefficients') or [0])[0]:+.5f} "
              f"corner-distortion={corner_distortion_px(cp):.1f}px "
              f"| A: max diff={dmax:.2e}px  B: roundtrip={rmax:.2e}m  ({n} pts)")

        if cv2 is not None:
            img = cv2.imread(str(frames_dir / name))
            if img is not None:
                cam = Camera()
                cam.from_json_parameters(broadtrack_cp_to_sncalib(cp))
                cam.draw_pitch(img)
                cv2.imwrite(str(out_dir / f"overlay_{name}"), img)

    dmax = float(np.nanmax(all_dmax))
    rmax = float(np.nanmax(all_rmax))
    ok_a = dmax < 1e-2
    ok_b = rmax < 1e-2
    print("\n================ verdict ================")
    print(f"  A. model equivalence : max {dmax:.2e} px  -> {'OK' if ok_a else 'FAIL'}"
          f" (threshold 1e-2 px; a wrong distortion rescale gives 1e2..1e3 px)")
    print(f"  B. plane roundtrip   : max {rmax:.2e} m   -> {'OK' if ok_b else 'FAIL'}"
          f" (threshold 1 cm on visible points)")
    if cv2 is not None:
        print(f"  C. overlays written to {out_dir}/ - the wireframe MUST sit on the")
        print(f"     painted lines (ground truth). Cross-check the binary's -r overlays.")
    print("=========================================")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
