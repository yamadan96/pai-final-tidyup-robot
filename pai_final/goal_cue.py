"""観測画像の隅にゴール（目標色・目標トレイ）を描き込むためのヘルパー。

v2 の診断で、学習した方策は「掴んで運ぶ」動作は獲得しているのに、
指示された色ではないブロックを掴むことが分かった。ゴールは状態ベクタ 13 次元の
うち 5 次元に one-hot で入っているだけで、ResNet が出す約 50 個の画像トークンに
比べて信号が弱い。そこでゴールを画像そのものにも描き込み、視覚側の経路からも
読めるようにする。

左上に 2 つの正方形を並べる:
  1 つ目 = 目標ブロックと同じ色
  2 つ目 = 目標トレイ（A=白 / B=黄）
"""

from __future__ import annotations

import numpy as np

# MJCF の block_* geom と同じ rgba（0-255 に変換したもの）
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (217, 38, 38),
    "blue": (38, 77, 230),
    "green": (38, 179, 64),
}

TRAY_RGB: dict[str, tuple[int, int, int]] = {
    "A": (255, 255, 255),
    "B": (255, 214, 0),
}

PATCH_RATIO = 0.11  # 画像の一辺に対するマーカーの大きさ
MARGIN_RATIO = 0.02


def patch_geometry(size: int) -> tuple[int, int]:
    """(マーカー1辺のピクセル数, 余白) を返す。"""
    side = max(6, int(round(size * PATCH_RATIO)))
    margin = max(1, int(round(size * MARGIN_RATIO)))
    return side, margin


def apply_goal_cue(image: np.ndarray, color: str, tray: str) -> np.ndarray:
    """画像の左上にゴールのマーカーを描き込んだ配列を返す（入力は変更しない）。

    image は HWC uint8。学習データ生成時と推論時で必ず同じ関数を通すこと。
    """
    if color not in COLOR_RGB:
        raise ValueError(f"unknown color: {color}")
    if tray not in TRAY_RGB:
        raise ValueError(f"unknown tray: {tray}")

    out = np.array(image, dtype=np.uint8, copy=True)
    h, w = out.shape[:2]
    side, margin = patch_geometry(min(h, w))

    y0, y1 = margin, margin + side
    x0, x1 = margin, margin + side
    out[y0:y1, x0:x1] = COLOR_RGB[color]

    x2, x3 = x1 + margin, x1 + margin + side
    out[y0:y1, x2:x3] = TRAY_RGB[tray]
    return out
