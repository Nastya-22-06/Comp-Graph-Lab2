"""Тесты вычислений для лабораторной работы."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1_grayscale import to_grayscale
from part2_rgb_channels import extract_channel


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


if __name__ == "__main__":
    unittest.main()