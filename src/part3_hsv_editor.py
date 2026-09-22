"""Часть 3: ручное преобразование RGB ↔ HSV и редактор с ползунками."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import load_rgb, save_rgb


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Преобразовать RGB 0..255 в HSV: H 0..360, S и V 0..1."""
    data = rgb.astype(np.float64) / 255.0
    red, green, blue = np.moveaxis(data, -1, 0)
    maximum = data.max(axis=-1)
    minimum = data.min(axis=-1)
    delta = maximum - minimum

    hue = np.zeros_like(maximum)
    non_gray = delta != 0
    red_max = non_gray & (maximum == red)
    green_max = non_gray & (maximum == green)
    blue_max = non_gray & (maximum == blue)

    hue[red_max] = 60.0 * np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)

    saturation = np.zeros_like(maximum)
    non_black = maximum != 0
    saturation[non_black] = 1.0 - minimum[non_black] / maximum[non_black]
    return np.stack((hue, saturation, maximum), axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Преобразовать HSV: H 0..360, S и V 0..1 в RGB 0..255."""
    hue = np.mod(hsv[..., 0], 360.0)
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)
    value = np.clip(hsv[..., 2], 0.0, 1.0)

    sector = np.floor(hue / 60.0).astype(np.int16) % 6
    fraction = (hue % 60.0) / 60.0
    value_min = (1.0 - saturation) * value
    value_inc = value_min + (value - value_min) * fraction
    value_dec = value - (value - value_min) * fraction

    table = (
        (value, value_inc, value_min),
        (value_dec, value, value_min),
        (value_min, value, value_inc),
        (value_min, value_dec, value),
        (value_inc, value_min, value),
        (value, value_min, value_dec),
    )
    rgb = np.empty(hsv.shape, dtype=np.float64)
    for index, components in enumerate(table):
        mask = sector == index
        rgb[mask] = np.stack(components, axis=-1)[mask]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def adjust_hsv(rgb: np.ndarray, hue_shift: float, saturation_factor: float, value_factor: float) -> np.ndarray:
    """Изменить оттенок, насыщенность и яркость и вернуть RGB."""
    hsv = rgb_to_hsv(rgb)
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 360.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * value_factor, 0.0, 1.0)
    return hsv_to_rgb(hsv)


class HsvEditor:
    """Небольшой редактор на Tkinter с тремя ползунками."""

    def __init__(self, image_path: str | Path, output_path: str | Path) -> None:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk

        self.tk = tk
        self.ImageTk = ImageTk
        self.messagebox = messagebox
        self.source = load_rgb(image_path)
        self.output_path = Path(output_path)
        self.root = tk.Tk()
        self.root.title("Редактор HSV")

        self.preview_label = tk.Label(self.root)
        self.preview_label.grid(row=0, column=0, columnspan=3, padx=12, pady=12)

        self.hue = self._slider("Оттенок, °", -180, 180, 0, 1)
        self.saturation = self._slider("Насыщенность, %", 0, 200, 100, 2)
        self.value = self._slider("Яркость, %", 0, 200, 100, 3)

        tk.Button(self.root, text="Сбросить", command=self.reset).grid(row=4, column=0, pady=12)
        tk.Button(self.root, text="Сохранить", command=self.save).grid(row=4, column=2, pady=12)
        self.update_preview()

    def _slider(self, label: str, minimum: int, maximum: int, initial: int, row: int):
        self.tk.Label(self.root, text=label).grid(row=row, column=0, sticky="w", padx=12)
        scale = self.tk.Scale(
            self.root,
            from_=minimum,
            to=maximum,
            orient=self.tk.HORIZONTAL,
            length=460,
            command=lambda _: self.update_preview(),
        )
        scale.set(initial)
        scale.grid(row=row, column=1, columnspan=2, padx=12, sticky="ew")
        return scale

    def current_result(self) -> np.ndarray:
        return adjust_hsv(
            self.source,
            float(self.hue.get()),
            float(self.saturation.get()) / 100.0,
            float(self.value.get()) / 100.0,
        )

    def update_preview(self) -> None:
        result = self.current_result()
        image = Image.fromarray(result, mode="RGB")
        image.thumbnail((720, 480))
        self.photo = self.ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.photo)

    def reset(self) -> None:
        self.hue.set(0)
        self.saturation.set(100)
        self.value.set(100)
        self.update_preview()

    def save(self) -> None:
        save_rgb(self.current_result(), self.output_path)
        self.messagebox.showinfo("Готово", f"Изображение сохранено:\n{self.output_path.resolve()}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Редактор HSV")
    parser.add_argument("image", help="путь к исходному изображению")
    parser.add_argument("--output", default="results/part3/hsv_adjusted.png", help="выходной файл")
    parser.add_argument("--no-gui", action="store_true", help="обработать без окна")
    parser.add_argument("--hue", type=float, default=0.0, help="сдвиг оттенка в градусах")
    parser.add_argument("--saturation", type=float, default=1.0, help="множитель насыщенности")
    parser.add_argument("--value", type=float, default=1.0, help="множитель яркости")
    args = parser.parse_args()

    if args.no_gui:
        source = load_rgb(args.image)
        result = adjust_hsv(source, args.hue, args.saturation, args.value)
        save_rgb(result, args.output)
        print(f"Часть 3 выполнена. Результат: {Path(args.output).resolve()}")
    else:
        HsvEditor(args.image, args.output).run()


if __name__ == "__main__":
    main()
