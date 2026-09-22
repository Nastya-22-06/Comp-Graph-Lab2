"""Общие функции лабораторной работы."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    """Загрузить изображение в массив RGB типа uint8."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)



def save_gray(array: np.ndarray, path: str | Path) -> None:
    """Сохранить одноканальное изображение."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    Image.fromarray(data, mode="L").save(destination)
