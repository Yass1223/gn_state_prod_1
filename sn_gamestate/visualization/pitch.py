"""Broadcast-style radar (minimap) visualizer — single pitch template, predictions only.

Professional inference display: ONE pitch panel (no ground-truth twin), rendered
bottom-centre with low transparency, players as filled team-color discs carrying their
jersey number, referee in its own color. Ground truth is intentionally never drawn here:
inference videos show the prediction state only (bbox + JN + ID on the frame come from
``CompletePlayerBBox``; this class adds the tactical minimap).

Coordinate mapping is unchanged from the validated original: ``bbox_pitch`` is in the
SoccerNet pitch frame (metres, origin at the pitch centre), drawn at ``scale`` px/m on a
105x68 template with 10 m / 5 m margins.

Config (all optional, see configs/visualization/gamestate.yaml)::

    radar:
      _target_: sn_gamestate.visualization.Radar
      scale: 4          # px per metre
      alpha: 0.8        # panel opacity (1.0 = fully opaque; the old GT/pred twin used 0.3)
      margin_bottom: 12 # px above the lower frame edge
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from tracklab.utils.cv2 import draw_text
from tracklab.visualization import ImageVisualizer

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"

# SoccerNet pitch (m) + drawing margins (m) — identical to the validated mapping.
PITCH_LENGTH, PITCH_WIDTH = 105, 68
MARGIN_X, MARGIN_Y = 10, 5

# Team palette (RGB, tracklab frames are RGB): left=blue, right=red, referee=yellow —
# consistent with configs/visualization/colors_gs.yaml.
COLOR_LEFT = (0, 0, 255)
COLOR_RIGHT = (255, 0, 0)
COLOR_REFEREE = (238, 210, 2)
COLOR_NEUTRAL = (230, 230, 230)


class Radar(ImageVisualizer):
    """Single prediction radar panel, bottom-centre, low transparency."""

    def __init__(self, scale: int = 4, alpha: float = 0.8, margin_bottom: int = 12):
        self.scale = int(scale)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.margin_bottom = int(margin_bottom)
        self._template = None  # lazily built, cached per scale

    # ------------------------------------------------------------------ template
    def _panel_template(self):
        if self._template is not None:
            return self._template
        w = (PITCH_LENGTH + 2 * MARGIN_X) * self.scale
        h = (PITCH_WIDTH + 2 * MARGIN_Y) * self.scale
        img = cv2.imread(str(pitch_file))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frames are RGB
            img = cv2.resize(img, (w, h))
        else:  # template asset missing -> plain dark-green pitch, never crash
            img = np.full((h, w, 3), (30, 90, 30), dtype=np.uint8)
        # subtle attack-direction accents matching the team palette
        cv2.line(img, (0, 0), (0, h), color=COLOR_LEFT, thickness=4)
        cv2.line(img, (w - 1, 0), (w - 1, h), color=COLOR_RIGHT, thickness=4)
        self._template = img
        return img

    # ------------------------------------------------------------------ drawing
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        # Predictions only, by design: no ground-truth panel in inference videos.
        if detections_pred is None or "bbox_pitch" not in detections_pred:
            return
        H, W = image.shape[:2]
        panel = self._panel_template()
        ph, pw = panel.shape[:2]
        if pw > W or ph + self.margin_bottom > H:
            log.warning("[Radar] frame too small for the radar panel; skipping")
            return

        top_x = (W - pw) // 2
        top_y = H - ph - self.margin_bottom
        cx = top_x + pw // 2   # pitch-centre pixel
        cy = top_y + ph // 2

        roi = image[top_y:top_y + ph, top_x:top_x + pw, :]
        roi[:] = cv2.addWeighted(roi, 1.0 - self.alpha, panel, self.alpha, 0.0)
        cv2.rectangle(image, (top_x - 1, top_y - 1),
                      (top_x + pw, top_y + ph), (255, 255, 255), 1, cv2.LINE_AA)

        r = max(4, int(round(2.2 * self.scale)))
        for _, det in detections_pred.iterrows():
            role = det.role if "role" in det else None
            if role == "ball":
                continue
            bp = det["bbox_pitch"]
            if not isinstance(bp, dict):
                continue
            x = float(np.clip(bp["x_bottom_middle"], -(PITCH_LENGTH / 2 + MARGIN_X),
                              PITCH_LENGTH / 2 + MARGIN_X))
            y = float(np.clip(bp["y_bottom_middle"], -(PITCH_WIDTH / 2 + MARGIN_Y),
                              PITCH_WIDTH / 2 + MARGIN_Y))
            px = cx + int(round(x * self.scale))
            py = cy + int(round(y * self.scale))

            team = det.team if "team" in det else None
            if role == "referee":
                color = COLOR_REFEREE
            elif team == "left":
                color = COLOR_LEFT
            elif team == "right":
                color = COLOR_RIGHT
            else:
                color = COLOR_NEUTRAL

            cv2.circle(image, (px, py), r, color, -1, cv2.LINE_AA)
            cv2.circle(image, (px, py), r, (255, 255, 255), 1, cv2.LINE_AA)

            label = None
            jn = det.jersey_number if "jersey_number" in det else None
            if role == "goalkeeper":
                label = "GK"
            elif role == "player" and jn is not None and not (
                isinstance(jn, float) and np.isnan(jn)
            ):
                label = f"{int(jn)}"
            if label:
                draw_text(
                    image, label, (px, py),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.075 * self.scale,
                    thickness=1,
                    color_txt=(255, 255, 255),
                    color_bg=None,
                    alignH="c", alignV="c",
                )
