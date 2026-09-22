"""Часть 2: выделение каналов R, G и B и построение гистограмм."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import load_rgb, save_rgb


CHANNELS = ((0, "R", "red"), (1, "G", "green"), (2, "B", "blue"))


def extract_channel(rgb: np.ndarray, index: int) -> np.ndarray:
    """Оставить один цветовой канал, обнулив остальные."""
    result = np.zeros_like(rgb)
    result[..., index] = rgb[..., index]
    return result


def process(image_path: str | Path, output_dir: str | Path) -> None:
    rgb = load_rgb(image_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, (index, name, color) in zip(axes, CHANNELS):
        save_rgb(extract_channel(rgb, index), output / f"channel_{name.lower()}.png")
        axis.hist(rgb[..., index].ravel(), bins=256, range=(0, 256), color=color, alpha=0.82)
        axis.set_title(f"Канал {name}")
        axis.set_xlabel("Значение канала")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Количество пикселей")
    fig.suptitle("Гистограммы каналов RGB")
    fig.tight_layout()
    fig.savefig(output / "rgb_histograms.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Выделение каналов RGB")
    parser.add_argument("image", help="путь к исходному изображению")
    parser.add_argument("--output", default="results/part2", help="каталог результатов")
    args = parser.parse_args()
    process(args.image, args.output)
    print(f"Часть 2 выполнена. Результаты: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
