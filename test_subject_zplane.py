"""CPU checks for zero_parallax = adaptive / subject (plans/SUBJECT_AWARE_ADAPTIVE_ZPLANE.md, Phase 6).

Run from the plugin folder with ComfyUI's Python:  python -m unittest test_subject_zplane
Set YUNET_MODEL to a face_detection_yunet_2023mar.onnx path to also check the real detector loads.
"""
import os
import unittest

import numpy as np
import torch

from job_types import StereoVideoJob
from z_plane import SUBJECT_MAX_POP_OUT, SUBJECT_SCREEN_DEPTH, AdaptiveZeroParallax, SubjectFaceDetector

SIZE = 100
FACE_BOX = (40.0, 40.0, 20.0, 20.0)


class StubDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return self.faces


def scene(p1, face=None, background=0.9):
    """Depth with a near strip (5% of the frame) at p1 and an optional face box at `face`."""
    depth = torch.full((1, 1, SIZE, SIZE), background)
    depth[0, 0, :5, :] = p1
    if face is not None:
        x, y, w, h = (int(v) for v in FACE_BOX)
        depth[0, 0, y:y + h, x:x + w] = face
    return depth


def shift(depth, power, faces=None, scene_starts=(0,), frame=0):
    detector = None if faces is None else StubDetector(faces)
    zp = AdaptiveZeroParallax(scene_starts=list(scene_starts), face_detector=detector)
    frames = np.zeros((1, SIZE, SIZE, 3), dtype=np.uint8)
    return zp, zp.shifts([frame], depth, power, frames)[0]


class SubjectZPlaneTests(unittest.TestCase):
    def test_face_lands_at_screen_depth(self):
        _, applied = shift(scene(p1=0.30, face=0.40), 0.5, [FACE_BOX + (0.9,)])
        self.assertAlmostEqual(applied, 0.40 ** 0.5 - SUBJECT_SCREEN_DEPTH, places=5)

    def test_pop_out_guard_caps_p1_in_front(self):
        # Worked example in the plan: face 0.40, p1 0.10, power 0.5 -> capped at 0.316 + 0.15.
        _, applied = shift(scene(p1=0.10, face=0.40), 0.5, [FACE_BOX + (0.9,)])
        self.assertLess(applied, 0.40 ** 0.5 - SUBJECT_SCREEN_DEPTH)  # the guard did trigger
        self.assertAlmostEqual(applied, 0.10 ** 0.5 + SUBJECT_MAX_POP_OUT, places=5)
        self.assertAlmostEqual(0.10 ** 0.5 - applied, -SUBJECT_MAX_POP_OUT, places=5)

    def test_no_face_falls_back_to_p1(self):
        _, applied = shift(scene(p1=0.20, face=0.40), 0.5, [])
        self.assertAlmostEqual(applied, 0.20 ** 0.5, places=5)

    def test_adaptive_is_p1_without_constant(self):
        _, applied = shift(scene(p1=0.20), 0.7)
        self.assertAlmostEqual(applied, 0.20 ** 0.7, places=5)

    def test_face_nearer_than_screen_depth_never_moves_scene_away(self):
        _, applied = shift(scene(p1=0.005, face=0.005), 0.5, [FACE_BOX + (0.9,)])
        self.assertEqual(applied, 0.0)

    def test_sweep_applied_never_negative(self):
        for power in (0.1, 0.3, 0.5, 1.0, 2.0, 4.0):
            for p1 in (0.0, 0.001, 0.05, 0.2, 0.5, 0.9):
                for face in (0.0, 0.001, 0.01, 0.1, 0.3, 0.6, 1.0):
                    for faces in (None, [], [FACE_BOX + (0.9,)]):
                        _, applied = shift(scene(p1=p1, face=face), power, faces)
                        self.assertGreaterEqual(applied, 0.0)
                        self.assertLessEqual(applied, 1.0)

    def test_highest_score_face_is_chosen(self):
        depth = scene(p1=0.30, face=0.40)
        far_face = (0.0, 70.0, 20.0, 20.0)
        depth[0, 0, 70:90, 0:20] = 0.80
        _, applied = shift(depth, 0.5, [FACE_BOX + (0.85,), far_face + (0.95,)])
        expected = min(0.80 ** 0.5 - SUBJECT_SCREEN_DEPTH, 0.30 ** 0.5 + SUBJECT_MAX_POP_OUT)
        self.assertAlmostEqual(applied, expected, places=5)

    def test_face_box_outside_frame_is_clipped(self):
        depth = scene(p1=0.30)
        depth[0, 0, 90:, 90:] = 0.40
        _, applied = shift(depth, 0.5, [(90.0, 90.0, 40.0, 40.0, 0.9)])  # inner box 100-120 is clipped away
        self.assertAlmostEqual(applied, 0.30 ** 0.5, places=5)
        _, applied = shift(depth, 0.5, [(85.0, 85.0, 20.0, 20.0, 0.9)])  # inner box 90-100 is inside
        self.assertAlmostEqual(applied, 0.40 ** 0.5 - SUBJECT_SCREEN_DEPTH, places=5)

    def test_value_is_held_until_next_scene(self):
        zp, first = shift(scene(p1=0.30, face=0.40), 0.5, [FACE_BOX + (0.9,)], scene_starts=(0, 10))
        frames = np.zeros((1, SIZE, SIZE, 3), dtype=np.uint8)
        self.assertEqual(zp.shifts([5], scene(p1=0.0), 0.5, frames), [first])
        zp.face_detector.faces = []
        self.assertAlmostEqual(zp.shifts([10], scene(p1=0.25), 0.5, frames)[0], 0.25 ** 0.5, places=5)

    def test_job_handle_modes(self):
        job = StereoVideoJob(
            source_video_path="x", width=1, height=1, source_fps=24, target_fps=24, every_nth=1,
            audio_mode="none", stereo_layout="sbs", depth_model="da3_small", depth_use_source_resolution=True,
            depth_inference_resolution=512, depth_edge_refine_method="none", chunk_size=4, depth_power=0.5,
            invert_depth=True,
        )
        handle = job.to_handle()
        self.assertEqual(StereoVideoJob.from_handle(handle).zero_parallax, "constant")
        handle["zero_parallax"] = "bogus"
        with self.assertRaises(ValueError):
            StereoVideoJob.from_handle(handle)

    @unittest.skipUnless(os.environ.get("YUNET_MODEL"), "set YUNET_MODEL to check the real detector")
    def test_real_yunet_loads(self):
        detector = SubjectFaceDetector(os.environ["YUNET_MODEL"])
        self.assertEqual(detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
