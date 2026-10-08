import unittest

import cv2
import torch
import torch.nn.functional as F

from particle_depth import apply_particle_depth, detect_particle_mask


def _old_detect(frames, min_area=1, max_area=200, mask_blur=1.0):
    masks = []
    for frame in frames:
        rgb = frame[..., :3].float().clamp_(0, 1)
        maximum = rgb.max(dim=-1).values
        minimum = rgb.min(dim=-1).values
        saturation = (maximum - minimum) / maximum.clamp_min_(1e-6)
        saturation = torch.where(maximum > 1e-6, saturation, 0)
        brightness = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        candidates = (brightness > 0.85) & (saturation < 0.25)
        candidate = candidates.cpu().to(torch.uint8).numpy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        keep = torch.zeros(candidate.shape, dtype=torch.float32).numpy()
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                keep[labels == label] = 1.0
        masks.append(torch.from_numpy(keep)[None, None])
    mask = torch.cat(masks)
    if mask_blur > 0:
        radius = max(1, int(__import__('math').ceil(mask_blur * 2)))
        coords = torch.arange(-radius, radius + 1, dtype=mask.dtype)
        kernel = torch.exp(-0.5 * (coords / mask_blur).square())
        kernel /= kernel.sum()
        mask = F.conv2d(mask, kernel.view(1, 1, 1, -1), padding=(0, radius))
        mask = F.conv2d(mask, kernel.view(1, 1, -1, 1), padding=(radius, 0)).clamp(0, 1)
    return mask


class ParticleDepthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.frames = torch.zeros((3, 32, 40, 3), dtype=torch.float32)
        for batch, y, x in ((0, 2, 2), (0, 10, 12), (1, 20, 4), (2, 5, 30)):
            self.frames[batch, y:y+2, x:x+2] = 1
        self.frames[1, 3:25, 20:38] = 1

    def test_detection_matches_reference_float_uint8_and_blur_options(self):
        for blur in (0.0, 1.0):
            expected = _old_detect(self.frames, mask_blur=blur)
            for frames in (self.frames, (self.frames * 255).to(torch.uint8)):
                actual = detect_particle_mask(frames, mask_blur=blur)
                torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=0)
                self.assertEqual(actual.dtype, torch.float16)

    def test_detection_does_not_mutate_input(self):
        frames = self.frames.clone()
        frames[0, 0, 0] = 2.5
        before = frames.clone()
        detect_particle_mask(frames, mask_blur=0)
        torch.testing.assert_close(frames, before)

    def test_apply_is_independent_of_micro_batch_and_chunk_boundaries(self):
        scene = torch.rand(16, 1, 12, 13)
        mask = torch.rand_like(scene)
        for mode in ('relative', 'absolute'):
            whole, synth = apply_particle_depth(scene, mask, mode=mode, micro_batch=16)
            small_batch, small_synth = apply_particle_depth(scene, mask, mode=mode, micro_batch=3)
            first, first_synth = apply_particle_depth(scene[:8], mask[:8], mode=mode, frame_index_start=0)
            second, second_synth = apply_particle_depth(scene[8:], mask[8:], mode=mode, frame_index_start=8)
            torch.testing.assert_close(small_batch, whole)
            torch.testing.assert_close(small_synth, synth)
            torch.testing.assert_close(torch.cat((first, second)), whole)
            torch.testing.assert_close(torch.cat((first_synth, second_synth)), synth)
            combined, no_synth = apply_particle_depth(scene, mask, mode=mode, return_synthetic=False)
            torch.testing.assert_close(combined, whole)
            self.assertIsNone(no_synth)


if __name__ == '__main__':
    unittest.main()
