"""動画に日本語の字幕・情報パネルを重ねるための描画ヘルパー。

MuJoCo のレンダラは文字を描けないので、RGB の numpy 配列を PIL に渡して
その上にテキストを書く。提出動画で「いま何をしているのか」が分かるように、
指示文・LLM が出した計画・実行中のサブタスクを画面内に表示する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# macOS に標準で入っている日本語フォント（上から順に探す）
FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """日本語が出せるフォントを size で読み込む（見つからなければ既定フォント）。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        _FONT_CACHE[size] = font
        return font
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    alpha: int = 170,
) -> None:
    """半透明の黒い帯を敷いて、文字を背景から浮かせる。"""
    draw.rectangle(box, fill=(0, 0, 0, alpha))


def annotate(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    *,
    header: str | None = None,
    footer: str | None = None,
    font_scale: float = 1.0,
) -> np.ndarray:
    """1 フレームに情報パネル（左上）とフッター（左下）を重ねる。

    lines は (テキスト, RGB色) の並び。header は少し大きい文字で先頭に出す。
    """
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    pad = max(8, int(w * 0.018))
    body_size = max(11, int(w * 0.027 * font_scale))
    head_size = max(13, int(w * 0.034 * font_scale))
    body_font = load_font(body_size)
    head_font = load_font(head_size)
    gap = int(body_size * 0.45)

    entries: list[tuple[str, tuple[int, int, int], ImageFont.FreeTypeFont]] = []
    if header:
        entries.append((header, (255, 255, 255), head_font))
    entries.extend((text, color, body_font) for text, color in lines)

    if entries:
        text_h = sum(int(f.size * 1.25) for _, _, f in entries) + gap * (len(entries) - 1)
        panel_w = max(
            int(draw.textlength(text, font=f)) for text, _, f in entries
        ) + pad * 2
        panel_w = min(panel_w, w - pad * 2)
        _draw_panel(draw, (pad, pad, pad + panel_w, pad + text_h + pad * 2))

        y = pad * 2
        for text, color, font in entries:
            draw.text((pad * 2, y), text, font=font, fill=(*color, 255))
            y += int(font.size * 1.25) + gap

    if footer:
        f_font = load_font(max(10, int(w * 0.023 * font_scale)))
        f_w = int(draw.textlength(footer, font=f_font)) + pad * 2
        f_h = int(f_font.size * 1.5)
        _draw_panel(draw, (pad, h - pad - f_h, pad + min(f_w, w - pad * 2), h - pad))
        draw.text(
            (pad * 2, h - pad - f_h + int(f_font.size * 0.2)),
            footer,
            font=f_font,
            fill=(255, 255, 255, 255),
        )

    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def title_card(
    size: tuple[int, int],
    title: str,
    body: list[str],
    *,
    n_frames: int,
    bg: tuple[int, int, int] = (16, 18, 24),
) -> list[np.ndarray]:
    """タイトル・説明用の静止画カードを n_frames 枚ぶん作る。"""
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    title_font = load_font(max(18, int(w * 0.055)))
    body_font = load_font(max(13, int(w * 0.032)))

    title_h = int(title_font.size * 1.4)
    body_h = int(body_font.size * 1.6)
    total = title_h + body_h * len(body)
    y = (h - total) // 2

    tw = int(draw.textlength(title, font=title_font))
    draw.text(((w - tw) // 2, y), title, font=title_font, fill=(255, 255, 255))
    y += title_h + int(body_font.size * 0.4)
    for line in body:
        lw = int(draw.textlength(line, font=body_font))
        draw.text(((w - lw) // 2, y), line, font=body_font, fill=(190, 198, 214))
        y += body_h

    frame = np.asarray(img)
    return [frame.copy() for _ in range(n_frames)]


def info_card(
    size: tuple[int, int],
    title: str,
    bullets: list[str],
    *,
    n_frames: int,
    bg: tuple[int, int, int] = (16, 18, 24),
    note: str | None = None,
) -> list[np.ndarray]:
    """左寄せの箇条書きカード（タイトル + 本文 + 下部の注記）。"""
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    title_font = load_font(max(16, int(w * 0.046)))
    body_font = load_font(max(12, int(w * 0.030)))
    note_font = load_font(max(10, int(w * 0.024)))

    left = int(w * 0.09)
    line_h = int(body_font.size * 1.75)
    total = int(title_font.size * 2.0) + line_h * len(bullets)
    y = max(int(h * 0.12), (h - total) // 2)

    draw.text((left, y), title, font=title_font, fill=(255, 255, 255))
    draw.line(
        [(left, y + int(title_font.size * 1.45)), (w - left, y + int(title_font.size * 1.45))],
        fill=(90, 100, 120),
        width=2,
    )
    y += int(title_font.size * 2.0)
    for line in bullets:
        draw.text((left, y), line, font=body_font, fill=(206, 214, 228))
        y += line_h

    if note:
        draw.text(
            (left, h - int(h * 0.09)), note, font=note_font, fill=(140, 150, 168)
        )

    frame = np.asarray(img)
    return [frame.copy() for _ in range(n_frames)]
