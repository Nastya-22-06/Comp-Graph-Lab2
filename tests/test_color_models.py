"""Тесты вычислений для лабораторной работы."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1_grayscale import to_grayscale
from part2_rgb_channels import extract_channel
from part3_hsv_editor import hsv_to_rgb, rgb_to_hsv


class ColorModelTests(unittest.TestCase):
    def test_grayscale_bt601_primary_colors(self) -> None:
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        actual = to_grayscale(rgb, (0.299, 0.587, 0.114))
        np.testing.assert_array_equal(
            actual,
            np.array([[76, 150, 29]], dtype=np.uint8),
        )
        
    def test_extract_rgb_channels(self) -> None:
        rgb = np.array([[[100, 150, 200]]], dtype=np.uint8)

        for index, expected_values in enumerate(
            ([100, 0, 0], [0, 150, 0], [0, 0, 200])
        ):
            with self.subTest(channel=index):
                actual = extract_channel(rgb, index)
                expected = np.array([[expected_values]], dtype=np.uint8)
                np.testing.assert_array_equal(actual, expected)    

    def test_rgb_to_hsv_known_colors(self) -> None:
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], dtype=np.uint8)
        hsv = rgb_to_hsv(rgb)
        np.testing.assert_allclose(hsv[0, :, 0], [0, 120, 240, 0], atol=1e-10)
        np.testing.assert_allclose(hsv[0, :, 1], [1, 1, 1, 0], atol=1e-10)
        np.testing.assert_allclose(hsv[0, :, 2], [1, 1, 1, 128 / 255], atol=1e-10)

    def test_hsv_round_trip(self) -> None:
        generator = np.random.default_rng(42)
        rgb = generator.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
        restored = hsv_to_rgb(rgb_to_hsv(rgb))
        error = np.abs(restored.astype(np.int16) - rgb.astype(np.int16))
        self.assertLessEqual(int(error.max()), 1)

    def test_hsv_ranges(self) -> None:
        rgb = np.array([[[12, 34, 56], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        hsv = rgb_to_hsv(rgb)
        self.assertTrue(np.all((hsv[..., 0] >= 0) & (hsv[..., 0] < 360)))
        self.assertTrue(np.all((hsv[..., 1:] >= 0) & (hsv[..., 1:] <= 1)))

if __name__ == "__main__":
    unittest.main()