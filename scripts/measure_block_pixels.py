#!/usr/bin/env python
"""観測画像の中でブロックが何ピクセルに写るかを実測する CLI。

使い方:
    ./.venv/bin/python scripts/measure_block_pixels.py

カメラのプリセット（wide / tight）と解像度の組み合わせごとに
    - ブロックの外接矩形の幅 [px]
    - 面積換算の一辺 sqrt(画素数) [px]
    - 作業領域の端に置いたブロックが画面からはみ出さないか
を測る。色マスクは最大連結成分だけを残すので、床の微妙な色味や
アンチエイリアスの画素は混ざらない。

はみ出し判定は描画ではなく射影計算で行う（腕にブロックが隠れても
結果が変わらないようにするため）。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import (  # noqa: E402
    BLOCK_HALF_Z,
    CAMERA_PRESETS,
    COLORS,
    WORK_X,
    WORK_Y,
    TidyUpEnv,
)

# ブロックの半径 [m]（tidyup_scene.xml の geom size と一致させる）
BLOCK_HALF_XY = 0.022
BLOCK_SIDE_M = 2 * BLOCK_HALF_XY

# 各色がどのチャンネルで支配的か
DOMINANT_CHANNEL = {"red": 0, "green": 1, "blue": 2}
CHANNEL_MIN = 70  # 支配チャンネルの最小値
CHANNEL_MARGIN = 55  # 2 番目に大きいチャンネルとの差


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets", type=str, default="wide,tight", help="比較するプリセット（カンマ区切り）"
    )
    parser.add_argument(
        "--sizes", type=str, default="96,160", help="比較する画像サイズ（カンマ区切り）"
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="サンプル画像の保存先")
    return parser.parse_args(argv)


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    """その色のブロックらしい画素のマスク。"""
    pixels = image.astype(np.int16)
    channel = DOMINANT_CHANNEL[color]
    others = [i for i in range(3) if i != channel]
    rest = np.maximum(pixels[:, :, others[0]], pixels[:, :, others[1]])
    return (pixels[:, :, channel] > CHANNEL_MIN) & (
        pixels[:, :, channel] - rest > CHANNEL_MARGIN
    )


def largest_component(mask: np.ndarray) -> list[tuple[int, int]]:
    """8 近傍での最大連結成分（scipy を使わない BFS）。"""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        component: list[tuple[int, int]] = []
        queue = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for dy, dx in itertools.product((-1, 0, 1), repeat=2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(component) > len(best):
            best = component
    return best


def block_pixels(image: np.ndarray, color: str) -> dict:
    """1 色のブロックについて、外接矩形と面積換算の一辺を測る。"""
    component = largest_component(color_mask(image, color))
    if not component:
        return {"n": 0, "width": 0, "height": 0, "side": 0.0}
    ys = np.array([p[0] for p in component])
    xs = np.array([p[1] for p in component])
    return {
        "n": len(component),
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
        "side": float(np.sqrt(len(component))),
    }


def place_blocks(env: TidyUpEnv, positions: list[tuple[float, float]], yaw: float) -> None:
    """3 個のブロックを指定位置・指定ヨー角に直接置く。"""
    for color, (x, y) in zip(COLORS, positions):
        adr = env.block_qadr[color]
        env.data.qpos[adr : adr + 3] = [x, y, BLOCK_HALF_Z]
        env.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(env.model, env.data)


# 作業領域の端を舐める配置（画角の下限を決めるワーストケース）
EDGE_LAYOUTS: list[list[tuple[float, float]]] = [
    [(WORK_X[0], WORK_Y[0]), (WORK_X[1], WORK_Y[0]), (0.48, WORK_Y[0])],
    [(WORK_X[0], WORK_Y[1]), (WORK_X[1], WORK_Y[1]), (0.48, WORK_Y[1])],
    [(WORK_X[0], WORK_Y[0]), (WORK_X[0], 0.0), (WORK_X[0], WORK_Y[1])],
    [(WORK_X[1], WORK_Y[0]), (WORK_X[1], 0.0), (WORK_X[1], WORK_Y[1])],
]


def frame_margin(env: TidyUpEnv, cam_xml: str) -> float:
    """作業領域の端に 45deg 傾けて置いたブロックの全頂点が画面内に残る余裕。

    返り値は「半画面に対する余白の割合」。正なら完全に画面内。
    描画ではなく射影計算なので、腕による遮蔽の影響を受けない。
    """
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_xml)
    mujoco.mj_forward(env.model, env.data)
    cam_pos = env.data.cam_xpos[cam_id].copy()
    cam_mat = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
    half_tan = np.tan(np.radians(float(env.model.cam_fovy[cam_id])) / 2.0)

    worst = np.inf
    for x in np.linspace(WORK_X[0], WORK_X[1], 5):
        for y in np.linspace(WORK_Y[0], WORK_Y[1], 9):
            for yaw in np.linspace(-np.pi / 4, np.pi / 4, 5):
                cos, sin = np.cos(yaw), np.sin(yaw)
                rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
                for sx, sy, sz in itertools.product(
                    (-BLOCK_HALF_XY, BLOCK_HALF_XY),
                    (-BLOCK_HALF_XY, BLOCK_HALF_XY),
                    (-BLOCK_HALF_Z, BLOCK_HALF_Z),
                ):
                    world = np.array([x, y, BLOCK_HALF_Z]) + rot @ np.array([sx, sy, sz])
                    rel = cam_mat.T @ (world - cam_pos)
                    depth = -rel[2]
                    worst = min(
                        worst, 1.0 - max(abs(rel[0]), abs(rel[1])) / (depth * half_tan)
                    )
    return float(worst)


def measure(preset: str, size: int, save_dir: Path | None) -> list[dict]:
    env = TidyUpEnv(seed=0, image_size=size, render_images=False, camera_preset=preset)
    env.reset("red", "A")
    rows: list[dict] = []
    for cam in env.cameras:
        cam_xml = env.camera_xml[cam]
        measurements: list[dict] = []
        for layout_idx, layout in enumerate(EDGE_LAYOUTS):
            place_blocks(env, layout, yaw=np.pi / 4)
            image = env.render(cam_xml, size=size)
            if save_dir is not None and layout_idx == 0:
                import imageio.v2 as imageio

                save_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(save_dir / f"{preset}_{cam}_{size}.png", image)
            for color in COLORS:
                stat = block_pixels(image, color)
                if stat["n"] > 0:  # 腕に完全に隠れた分は平均から除く
                    measurements.append(stat)
        widths = [m["width"] for m in measurements]
        sides = [m["side"] for m in measurements]
        rows.append(
            {
                "preset": preset,
                "camera": cam,
                "camera_xml": cam_xml,
                "size": size,
                "fovy": float(
                    env.model.cam_fovy[
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_xml)
                    ]
                ),
                "width_mean": float(np.mean(widths)),
                "width_min": int(np.min(widths)),
                "side_mean": float(np.mean(sides)),
                "frame_margin": frame_margin(env, cam_xml),
                "samples": len(measurements),
            }
        )
    env.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    presets = [p for p in args.presets.split(",") if p]
    sizes = [int(s) for s in args.sizes.split(",") if s]

    print(f"ブロック実寸: 一辺 {BLOCK_SIDE_M * 100:.1f} cm (対角 {BLOCK_SIDE_M * 141.4:.1f} cm)")
    print("測定は 45deg 傾け（見かけが対角幅になる向き）で行う\n")
    header = (
        f"{'preset':7s} {'camera':6s} {'xml':12s} {'px':>4s} {'fovy':>5s} "
        f"{'幅平均':>7s} {'幅最小':>7s} {'面積換算':>8s} {'画面内余白':>10s}"
    )
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for preset in presets:
        for size in sizes:
            for row in measure(preset, size, args.save_dir):
                results.append(row)
                print(
                    f"{row['preset']:7s} {row['camera']:6s} {row['camera_xml']:12s} "
                    f"{row['size']:4d} {row['fovy']:5.0f} "
                    f"{row['width_mean']:6.1f}px {row['width_min']:6d}px "
                    f"{row['side_mean']:7.1f}px {row['frame_margin']:+9.1%}"
                )

    print("\n--- 変更前後の比較（wide 96px -> tight 160px）---")
    for cam in ("top", "front"):
        before = next(
            (r for r in results if r["preset"] == "wide" and r["camera"] == cam and r["size"] == 96),
            None,
        )
        after = next(
            (r for r in results if r["preset"] == "tight" and r["camera"] == cam and r["size"] == 160),
            None,
        )
        if before is None or after is None:
            continue
        ratio = after["width_mean"] / before["width_mean"]
        print(
            f"  {cam:6s} 幅 {before['width_mean']:.1f}px -> {after['width_mean']:.1f}px "
            f"({ratio:.1f}x)   面積換算 {before['side_mean']:.1f}px -> {after['side_mean']:.1f}px"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
