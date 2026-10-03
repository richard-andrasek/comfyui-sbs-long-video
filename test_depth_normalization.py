import unittest

import torch

from depth_runner import DepthAnythingRunner
from job_types import StereoVideoJob


class TemporalDepthNormalizationTests(unittest.TestCase):
    def test_simple_mode_keeps_per_frame_min_max(self):
        runner = DepthAnythingRunner(device=torch.device("cpu"))
        first = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        second = first * 10.0
        expected = torch.cat([runner._normalize_depth(first), runner._normalize_depth(second)])

        self.assertTrue(torch.allclose(runner._normalize_depth(first), runner._normalize_depth(second)))
        self.assertEqual(expected.shape, (2, 1, 2, 2))

    def test_ema_bounds_are_smoothed_across_calls(self):
        runner = DepthAnythingRunner(
            device=torch.device("cpu"),
            depth_normalization_method="ema",
        )
        frame = torch.zeros((1, 3, 2, 2))
        first_depth = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        shifted_depth = first_depth + 0.1

        first = runner._normalize_depth_temporal(frame, first_depth)
        second = runner._normalize_depth_temporal(frame, shifted_depth)

        self.assertTrue(torch.allclose(first, torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])))
        self.assertGreater(float(second[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(runner._depth_range_ema[0], 0.005, places=5)
        self.assertAlmostEqual(runner._depth_range_ema[1], 1.005, places=5)

    def test_constant_and_nonfinite_depth_produce_finite_output(self):
        runner = DepthAnythingRunner(
            device=torch.device("cpu"),
            depth_normalization_method="ema",
        )
        frame = torch.zeros((1, 3, 2, 2))
        constant = torch.full((1, 1, 2, 2), 4.0)
        constant_result = runner._normalize_depth_temporal(frame, constant)
        self.assertTrue(torch.isfinite(constant_result).all())
        self.assertTrue(torch.equal(constant_result, torch.zeros_like(constant_result)))

        varied = torch.tensor([[[[0.0, 1.0], [float("nan"), float("inf")]]]])
        result = runner._normalize_depth_temporal(frame, varied)
        self.assertTrue(torch.isfinite(result).all())

    def test_cut_resets_bounds(self):
        runner = DepthAnythingRunner(
            device=torch.device("cpu"),
            depth_normalization_method="ema",
        )
        depth = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        runner._normalize_depth_temporal(torch.zeros((1, 3, 2, 2)), depth)
        result = runner._normalize_depth_temporal(torch.ones((1, 3, 2, 2)), depth + 10.0)
        self.assertTrue(torch.allclose(result, runner._normalize_depth(depth)))
        self.assertAlmostEqual(runner._depth_range_ema[0], 10.0, places=5)
        self.assertAlmostEqual(runner._depth_range_ema[1], 11.0, places=5)

    def test_old_job_handles_default_to_simple(self):
        old_handle = {
            "type": "STEREO_VIDEO_JOB",
            "source_video_path": "clip.mp4",
            "width": 10, "height": 10, "source_fps": 24.0, "target_fps": 24.0,
            "every_nth": 1, "audio_mode": "none", "stereo_layout": "sbs",
            "depth_model": "da3_small", "depth_use_source_resolution": True,
            "depth_inference_resolution": 512, "depth_edge_refine_method": "none",
            "chunk_size": 1, "depth_power": 1.0, "invert_depth": False,
        }
        job = StereoVideoJob.from_handle(old_handle)
        self.assertEqual(job.depth_normalization_method, "simple")


if __name__ == "__main__":
    unittest.main()
