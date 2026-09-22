"""Тесты вычислений для лабораторной работы."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from part1_grayscale import to_grayscale


class ColorModelTests(unittest.TestCase):
    def test_grayscale_bt601_primary_colors(self) -> None:
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        actual = to_grayscale(rgb, (0.299, 0.587, 0.114))
        np.testing.assert_array_equal(
            actual,
            np.array([[76, 150, 29]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()