"""Часть 1: два преобразования RGB в оттенки серого."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import load_rgb, save_gray


def to_grayscale(rgb: np.ndarray, coefficients: tuple[float, float, float]) -> np.ndarray:
    """Вычислить яркость как взвешенную сумму каналов R, G и B."""
    source = rgb.astype(np.float64)
    return np.clip(np.rint(source @ np.asarray(coefficients)), 0, 255).astype(np.uint8)


def process(image_path: str | Path, output_dir: str | Path) -> None:
    rgb = load_rgb(image_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gray_bt601 = to_grayscale(rgb, (0.299, 0.587, 0.114))
    gray_bt709 = to_grayscale(rgb, (0.2126, 0.7152, 0.0722))
    difference = np.abs(gray_bt601.astype(np.int16) - gray_bt709.astype(np.int16)).astype(np.uint8)

    save_gray(gray_bt601, output / "gray_bt601.png")
    save_gray(gray_bt709, output / "gray_bt709.png")
    save_gray(difference, output / "difference.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, data, title in (
        (axes[0], gray_bt601, "BT.601"),
        (axes[1], gray_bt709, "BT.709"),
    ):
        axis.hist(data.ravel(), bins=256, range=(0, 256), color="#364f6b")
        axis.set_title(f"Гистограмма интенсивности {title}")
        axis.set_xlabel("Интенсивность")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Количество пикселей")
    fig.tight_layout()
    fig.savefig(output / "grayscale_histograms.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Преобразование RGB в оттенки серого")
    parser.add_argument("image", help="путь к исходному изображению")
    parser.add_argument("--output", default="results/part1", help="каталог результатов")
    args = parser.parse_args()
    process(args.image, args.output)
    print(f"Часть 1 выполнена. Результаты: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
