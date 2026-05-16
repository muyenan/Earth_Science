from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path("C:/Windows/Fonts") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _canvas(width: int = 1100, height: int = 680) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def _scale(value: float, lower: float, upper: float, start: int, end: int, reverse: bool = False) -> float:
    if upper == lower:
        upper = lower + 1.0
    ratio = (value - lower) / (upper - lower)
    if reverse:
        ratio = 1.0 - ratio
    return start + ratio * (end - start)


def line_chart(path: Path, title: str, frame: pd.DataFrame, series: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas()
    width, height = image.size
    left, top, right, bottom = 88, 92, width - 48, height - 78

    years = frame["year"].astype(float)
    y_values = []
    for column, _, _ in series:
        y_values.extend(frame[column].dropna().astype(float).tolist())
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = (y_max - y_min) * 0.12 or 0.1
    y_min -= y_pad
    y_max += y_pad

    draw.text((left, 32), title, font=_font(28, bold=True), fill="#222222")
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    for tick in range(5):
        y = top + tick * (bottom - top) / 4
        value = y_max - tick * (y_max - y_min) / 4
        draw.line((left - 6, y, right, y), fill="#e5e5e5", width=1)
        draw.text((18, y - 8), f"{value:.2f}", font=_font(14), fill="#444444")

    x_min = float(years.min())
    x_max = float(years.max())
    for year in frame["year"].astype(int):
        x = _scale(float(year), x_min, x_max, left, right)
        draw.line((x, bottom, x, bottom + 6), fill="#333333", width=1)
        draw.text((x - 16, bottom + 16), str(year), font=_font(14), fill="#444444")

    legend_x = left
    legend_y = 66
    for column, label, color in series:
        points = []
        for _, row in frame[["year", column]].dropna().iterrows():
            x = _scale(float(row["year"]), x_min, x_max, left, right)
            y = _scale(float(row[column]), y_min, y_max, top, bottom, reverse=True)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.line((legend_x, legend_y, legend_x + 34, legend_y), fill=color, width=4)
        draw.text((legend_x + 42, legend_y - 9), label, font=_font(15), fill="#222222")
        legend_x += 245

    image.save(path)


def scatter_plot(path: Path, title: str, frame: pd.DataFrame, actual_column: str, predicted_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas()
    width, height = image.size
    left, top, right, bottom = 88, 92, width - 58, height - 78
    data = frame[[actual_column, predicted_column]].dropna().astype(float)
    values = data.to_numpy().reshape(-1)
    lower = float(values.min()) - 0.08
    upper = float(values.max()) + 0.08

    draw.text((left, 32), title, font=_font(28, bold=True), fill="#222222")
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    draw.line((left, bottom, right, top), fill="#999999", width=2)
    draw.text((left, bottom + 28), "Actual temperature anomaly (C)", font=_font(16), fill="#333333")
    draw.text((16, top - 30), "Predicted", font=_font(16), fill="#333333")
    for _, row in data.iterrows():
        x = _scale(float(row[actual_column]), lower, upper, left, right)
        y = _scale(float(row[predicted_column]), lower, upper, top, bottom, reverse=True)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2f6db3")

    image.save(path)


def bar_chart(path: Path, title: str, frame: pd.DataFrame, label_column: str, value_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas()
    width, height = image.size
    left, top, right, bottom = 110, 92, width - 70, height - 100
    labels = frame[label_column].astype(str).tolist()
    values = frame[value_column].astype(float).tolist()
    max_value = max(values) * 1.2 or 1.0
    bar_width = (right - left) / max(1, len(values)) * 0.58

    draw.text((left, 32), title, font=_font(28, bold=True), fill="#222222")
    draw.line((left, bottom, right, bottom), fill="#333333", width=2)
    draw.line((left, top, left, bottom), fill="#333333", width=2)
    for index, (label, value) in enumerate(zip(labels, values)):
        center = left + (index + 0.5) * (right - left) / len(values)
        x0 = center - bar_width / 2
        x1 = center + bar_width / 2
        y0 = _scale(value, 0.0, max_value, top, bottom, reverse=True)
        draw.rectangle((x0, y0, x1, bottom), fill="#4c78a8")
        draw.text((center - 38, bottom + 16), label[:12], font=_font(14), fill="#333333")
        draw.text((center - 28, y0 - 24), f"{value:.3f}", font=_font(14), fill="#333333")

    image.save(path)


def workflow_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas(width=1250, height=520)
    title_font = _font(26, bold=True)
    box_font = _font(16)
    draw.text((50, 28), "Climate Temperature Prediction Workflow", font=title_font, fill="#222222")

    boxes = [
        (50, 120, 250, 230, "Temperature\nGISTEMP"),
        (50, 290, 250, 400, "Greenhouse gases\nCO2 / CH4 / N2O"),
        (320, 205, 540, 315, "Feature engineering\nforcing + causal FFT\nlagged ENSO"),
        (610, 120, 820, 230, "Gas forecast\nrecursive model"),
        (610, 290, 820, 400, "Temperature model\nhybrid ML"),
        (890, 205, 1140, 315, "Evaluation\nstrict future +\nrandom decade"),
    ]
    for x0, y0, x1, y1, text in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline="#333333", width=2, fill="#f7f9fb")
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            draw.text((x0 + 18, y0 + 20 + idx * 24), line, font=box_font, fill="#222222")

    arrows = [
        ((250, 175), (320, 250)),
        ((250, 345), (320, 270)),
        ((540, 250), (610, 175)),
        ((540, 270), (610, 345)),
        ((820, 175), (890, 250)),
        ((820, 345), (890, 270)),
    ]
    for start, end in arrows:
        draw.line((start, end), fill="#555555", width=3)
        ex, ey = end
        draw.polygon([(ex, ey), (ex - 12, ey - 6), (ex - 8, ey + 8)], fill="#555555")

    image.save(path)
