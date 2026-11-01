#!/usr/bin/env python
"""レポート用の図を作る（シーン外観・観測画像の比較・学習曲線）。

matplotlib を入れていないので、折れ線は PIL で直接描く。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import TidyUpEnv  # noqa: E402
from pai_final.overlay import load_font  # noqa: E402

OUT = Path("report/figures")


def label_strip(img: np.ndarray, text: str, height: int = 26) -> np.ndarray:
    """画像の下にキャプション帯を付ける。"""
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h + height), (24, 26, 32))
    canvas.paste(Image.fromarray(img), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(11, int(w * 0.055)))
    tw = int(draw.textlength(text, font=font))
    draw.text(((w - tw) // 2, h + (height - font.size) // 2 - 1), text,
              font=font, fill=(222, 228, 240))
    return np.asarray(canvas)


def hstack(images: list[np.ndarray], gap: int = 10) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), 24, np.uint8)
            im = np.concatenate([im, pad], axis=0)
        parts.append(im)
        if i != len(images) - 1:
            parts.append(np.full((h, gap, 3), 24, np.uint8))
    return np.concatenate(parts, axis=1)


def fig_scene() -> None:
    """シーン全体（斜め前方・俯瞰）。"""
    env = TidyUpEnv(seed=11, image_size=96, render_images=False)
    env.reset("red", "B")
    front = env.render("front", size=420)
    top = env.render("top", size=420)
    img = hstack([label_strip(front, "斜め前方 (front)"), label_strip(top, "俯瞰 (top)")])
    Image.fromarray(img).save(OUT / "fig1_scene.png")
    env.close()
    print("fig1_scene.png")


def fig_observation() -> None:
    """方策が実際に見る観測画像を v1 / v2 / v3 で並べる。"""
    panels = []
    for title, kwargs in (
        ("v1: 96px / 広角", dict(image_size=96, camera_preset="wide")),
        ("v2: 160px / 寄り", dict(image_size=160, camera_preset="tight")),
        ("v3: + ゴール描画", dict(image_size=160, camera_preset="tight", goal_cue=True)),
    ):
        env = TidyUpEnv(seed=11, **kwargs)
        obs = env.reset("blue", "A")
        cells = []
        for cam in ("top", "front"):
            up = np.asarray(
                Image.fromarray(obs["images"][cam]).resize((240, 240), Image.NEAREST)
            )
            cells.append(label_strip(up, f"{title} / {cam}", height=24))
        panels.append(hstack(cells, gap=6))
        env.close()
    gap = np.full((8, panels[0].shape[1], 3), 24, np.uint8)
    img = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=0)
    Image.fromarray(img).save(OUT / "fig2_observation.png")
    print("fig2_observation.png")


def fig_loss(
    runs: list[tuple[str, Path, tuple[int, int, int]]],
    out: Path,
    *,
    size: tuple[int, int] = (860, 400),
) -> None:
    """学習曲線（片対数）を PIL で描く。"""
    w, h = size
    pad_l, pad_r, pad_t, pad_b = 76, 160, 26, 52
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = load_font(15)
    small = load_font(13)

    series = []
    for name, path, color in runs:
        if not path.exists():
            continue
        xs, ys = [], []
        with path.open() as fh:
            for row in csv.DictReader(fh):
                step, loss = int(row["step"]), float(row["loss"])
                if step % 200 == 0 and loss > 0:
                    xs.append(step)
                    ys.append(loss)
        if xs:
            series.append((name, xs, ys, color))
    if not series:
        print("学習ログが無いので fig_loss をとばす")
        return

    x_max = max(max(xs) for _, xs, _, _ in series)
    y_min = min(min(ys) for _, _, ys, _ in series)
    y_max = max(max(ys) for _, _, ys, _ in series)
    lo, hi = np.log10(max(y_min, 1e-4)), np.log10(y_max)

    def px(x: float) -> float:
        return pad_l + (w - pad_l - pad_r) * x / x_max

    def py(y: float) -> float:
        t = (np.log10(max(y, 1e-4)) - lo) / (hi - lo)
        return pad_t + (h - pad_t - pad_b) * (1.0 - t)

    # 目盛り
    for dec in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
        val = 10.0**dec
        if not (y_min * 0.9 <= val <= y_max * 1.1):
            continue
        y = py(val)
        draw.line([(pad_l, y), (w - pad_r, y)], fill=(228, 230, 236))
        draw.text((10, y - 8), f"{val:g}", font=small, fill=(90, 96, 110))
    for i in range(6):
        x = pad_l + (w - pad_l - pad_r) * i / 5
        step = int(x_max * i / 5)
        draw.line([(x, pad_t), (x, h - pad_b)], fill=(238, 240, 244))
        draw.text((x - 22, h - pad_b + 8), f"{step // 1000}k", font=small, fill=(90, 96, 110))
    draw.rectangle([pad_l, pad_t, w - pad_r, h - pad_b], outline=(150, 156, 168))
    draw.text((pad_l, h - 22), "学習ステップ", font=small, fill=(60, 64, 74))
    draw.text((8, 6), "損失（対数軸）", font=small, fill=(60, 64, 74))

    for i, (name, xs, ys, color) in enumerate(series):
        pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
        draw.line(pts, fill=color, width=2)
        ly = pad_t + 12 + i * 24
        draw.line([(w - pad_r + 12, ly), (w - pad_r + 40, ly)], fill=color, width=3)
        draw.text((w - pad_r + 46, ly - 9), name, font=font, fill=(40, 44, 52))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(out.name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_scene()
    fig_observation()
    fig_loss(
        [
            ("v2 (160px, ノイズ注入)", Path("outputs/checkpoints/act_v2/train_log.csv"), (31, 102, 214)),
            ("v3 (+ ゴール描画)", Path("outputs/checkpoints/act_v3/train_log.csv"), (214, 86, 31)),
        ],
        OUT / "fig3_loss.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
