import unittest
from types import SimpleNamespace

import numpy as np
import torch

from depth_runner import DepthAnythingRunner
from job_types import StereoVideoJob


class GlobalDepthNormalizationTests(unittest.TestCase):
    def test_simple_normalization_keeps_per_frame_min_max(self):
        first = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        second = first * 10.0
        normalized_first = DepthAnythingRunner._normalize_depth(first)
        normalized_second = DepthAnythingRunner._normalize_depth(second)

        self.assertTrue(torch.allclose(normalized_first, normalized_second))
        self.assertEqual(normalized_first.shape, first.shape)

    def test_global_mode_uses_800_when_p99_is_under_limit(self):
        depth = torch.full((1, 1, 10, 10), 400.0)
        normalized = DepthAnythingRunner._normalize_depth_global(depth)
        self.assertTrue(torch.allclose(normalized, torch.full_like(depth, 0.5)))

    def test_global_mode_hard_clamps_and_warns_for_p99_exceedance(self):
        ordinary = torch.full((1, 1, 10, 10), 35.0)
        exceeding = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
        depth = torch.cat([ordinary, exceeding], dim=0)

        with self.assertLogs("depth_runner", level="WARNING") as captured:
            normalized = DepthAnythingRunner._normalize_depth_global(
                depth,
                global_depth_max=50.0,
                source_frame_index_start=0,
                source_frame_stride=30,
                source_fps=30.0,
            )

        self.assertTrue(torch.allclose(normalized[0], torch.full_like(ordinary[0], 0.7)))
        self.assertAlmostEqual(float(normalized[1, 0, 4, 9]), 0.98, places=5)
        self.assertEqual(float(normalized[1, 0, 9, 8]), 1.0)
        self.assertEqual(float(normalized[1, 0, 9, 9]), 1.0)
        warning = captured.output[0]
        self.assertIn("source frame 31", warning)
        self.assertIn("1.000 s", warning)
        self.assertIn("global_depth_max: 50.000", warning)
        self.assertIn("P99: 98.010", warning)
        self.assertIn("frame max: 99.000", warning)
        self.assertIn(">max: 49.000%", warning)

    def test_inversion_is_applied_after_global_normalization(self):
        class FakeDepthBackend:
            def inference(self, _frames):
                values = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
                return SimpleNamespace(depth=values)

        runner = DepthAnythingRunner(
            device=torch.device("cpu"),
            depth_normalization_method="global",
            global_depth_max=750.0,
        )
        runner._api_backend = FakeDepthBackend()
        frames = torch.zeros((1, 3, 10, 10))
        depth = runner.infer(frames, invert_depth=True)

        self.assertAlmostEqual(float(depth[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(depth[0, 0, 9, 9]), 1.0 - 99.0 / 750.0, places=5)

    def test_nonfinite_values_are_sanitized(self):
        depth = torch.tensor([[[[0.0, 750.0], [float("nan"), float("inf")]]]])
        normalized = DepthAnythingRunner._normalize_depth_global(depth)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(float(normalized[0, 0, 0, 1]), 750.0 / 850.0)
        self.assertEqual(float(normalized[0, 0, 1, 1]), 1.0)

    def test_global_maximum_must_be_positive_and_finite(self):
        for invalid_max in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid_max=invalid_max):
                with self.assertRaises(ValueError):
                    DepthAnythingRunner._normalize_depth_global(
                        torch.ones((1, 1, 2, 2)),
                        global_depth_max=invalid_max,
                    )

    def test_old_and_intermediate_handles_default_to_simple(self):
        base = {
            "type": "STEREO_VIDEO_JOB",
            "source_video_path": "clip.mp4",
            "width": 10, "height": 10, "source_fps": 30.0, "target_fps": 30.0,
            "every_nth": 1, "audio_mode": "none", "stereo_layout": "sbs",
            "depth_model": "da3_small", "depth_use_source_resolution": True,
            "depth_inference_resolution": 512, "depth_edge_refine_method": "none",
            "chunk_size": 1, "depth_power": 1.0, "invert_depth": False,
        }
        old_job = StereoVideoJob.from_handle(base)
        self.assertEqual(old_job.depth_normalization_method, "simple")
        self.assertEqual(old_job.global_depth_max, 850.0)

        intermediate = dict(base, depth_normalization_method="ema", depth_temporal_ema=0.95)
        intermediate_job = StereoVideoJob.from_handle(intermediate)
        self.assertEqual(intermediate_job.depth_normalization_method, "simple")
        self.assertEqual(intermediate_job.global_depth_max, 850.0)

        invalid_handle = dict(base, global_depth_max=float("nan"))
        with self.assertRaises(ValueError):
            StereoVideoJob.from_handle(invalid_handle)


if __name__ == "__main__":
    unittest.main()
