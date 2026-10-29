from __future__ import annotations

import math

import torch

try:
    from .depth_runner import finite_quantile, is_scene_start
except ImportError:
    from depth_runner import finite_quantile, is_scene_start


SCENE_Z_SHIFT_QUANTILE = 0.01  # p1 of normalized depth is placed on the screen plane for each scene.
# Subject mode (zero_parallax = "subject"). Depths are in the renderer's curved 0-1 space.
SUBJECT_SCREEN_DEPTH = 0.1  # The chosen face sits this far behind the screen.
SUBJECT_MAX_POP_OUT = 0.15  # p1 may come at most this far in front of the screen.
SUBJECT_FACE_SCORE_THRESHOLD = 0.8  # YuNet confidence.
SUBJECT_MIN_FACE_HEIGHT = 0.04  # Fraction of frame height; drops background extras and texture false positives.
SUBJECT_FACE_SAMPLE_FRACTION = 0.5  # Depth is sampled from the central part of the face box (width and height).


class SubjectFaceDetector:
    """YuNet face detector (OpenCV DNN, CPU). Created by the Face Detector Loader node."""

    def __init__(self, model_path: str):
        import cv2  # Imported here so constant/adaptive users never need OpenCV.

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0), SUBJECT_FACE_SCORE_THRESHOLD)
        self._input_size = None

    def detect(self, frame_rgb_uint8) -> list[tuple[float, float, float, float, float]]:
        """Returns (x, y, w, h, score) for faces that pass the size filter."""
        height, width = frame_rgb_uint8.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        _, faces = self._detector.detect(self._cv2.cvtColor(frame_rgb_uint8, self._cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        # YuNet rows: x, y, w, h, 10 landmark coordinates, score.
        return [
            (float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14]))
            for f in faces
            if f[3] >= SUBJECT_MIN_FACE_HEIGHT * height
        ]


def choose_face(faces):
    """Multiple-face rule: highest YuNet confidence."""
    return max(faces, key=lambda face: face[4])


def face_depth(depth_1hw: torch.Tensor, face) -> float:
    """Median depth over the central part of the face box, clipped to the frame; NaN if empty."""
    x, y, w, h = face[:4]
    height, width = depth_1hw.shape[-2:]
    margin = (1.0 - SUBJECT_FACE_SAMPLE_FRACTION) / 2.0
    x0 = min(max(int(x + w * margin), 0), width)
    x1 = min(max(math.ceil(x + w * (1.0 - margin)), 0), width)
    y0 = min(max(int(y + h * margin), 0), height)
    y1 = min(max(math.ceil(y + h * (1.0 - margin)), 0), height)
    return finite_quantile(depth_1hw[..., y0:y1, x0:x1], 0.5)


class AdaptiveZeroParallax:
    """Per-scene zero parallax, held for the whole scene.

    Without a face detector ("adaptive"): puts each scene's p1 depth on the screen plane.
    With one ("subject"): puts the most confident face SUBJECT_SCREEN_DEPTH behind the screen,
    falling back to p1 when no face is found. Both are pure shifts.
    """

    def __init__(self, scene_starts: list[int], stride: int = 1, face_detector: SubjectFaceDetector | None = None):
        self.scene_starts = sorted(set(int(x) for x in scene_starts)) or [0]
        self.stride = max(1, int(stride))
        self.face_detector = face_detector
        self.applied = 0.0

    def shifts(
        self,
        frame_indices: list[int],
        depth_b1hw: torch.Tensor,
        depth_powers: float | list[float],
        frames_rgb_uint8=None,
    ) -> list[float]:
        """depth_powers must be the final per-frame powers (after adaptive depth power).

        frames_rgb_uint8 is the chunk's source frames (B, H, W, 3), needed only with a face detector.
        """
        if not isinstance(depth_powers, list):
            depth_powers = [float(depth_powers)] * len(frame_indices)
        results = []
        for item, frame_index in enumerate(frame_indices):
            # Measured once, on the scene's first frame; every later frame reuses self.applied.
            if is_scene_start(self.scene_starts, int(frame_index), self.stride):
                self._measure(int(frame_index), depth_b1hw[item], float(depth_powers[item]),
                              None if frames_rgb_uint8 is None else frames_rgb_uint8[item])
            results.append(self.applied)
        return results

    def _measure(self, frame_index: int, depth_1hw: torch.Tensor, depth_power: float, frame_rgb_uint8) -> None:
        p1 = finite_quantile(depth_1hw, SCENE_Z_SHIFT_QUANTILE)
        if not math.isfinite(p1):
            return
        power = max(depth_power, 1e-3)  # same floor as the renderer
        p1_shift = min(1.0, max(0.0, p1)) ** power

        faces = []
        if self.face_detector is not None and frame_rgb_uint8 is not None:
            faces = self.face_detector.detect(frame_rgb_uint8)
        face = choose_face(faces) if faces else None
        depth = face_depth(depth_1hw, face) if face is not None else float("nan")

        if not math.isfinite(depth):
            self.applied = min(1.0, p1_shift)
            return

        face_shift = min(1.0, max(0.0, depth)) ** power - SUBJECT_SCREEN_DEPTH
        max_shift = p1_shift + SUBJECT_MAX_POP_OUT  # pop-out guard: p1 at most SUBJECT_MAX_POP_OUT in front
        # Lower clamp: never move the scene away from the viewer.
        self.applied = min(1.0, max(0.0, min(face_shift, max_shift)))
