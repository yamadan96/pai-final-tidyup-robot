#!/usr/bin/env python
"""把持の直前区間で、対象ブロックが各カメラから見えているかを実測する。

収集済みの npz（画像つき）から、グリッパを閉じ始める直前の区間を取り出し、
対象色の画素が各カメラに何フレーム写っているかを数える。
カメラを 2 台にしたことに意味があったのかを確かめるための計測。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# MJCF の rgba を 0-255 にしたもの。レンダリングでは陰影が乗るので幅を持たせる
TARGET_HUE = {
    "red": lambda f: (f[..., 0] > 90) & (f[..., 0] > f[..., 1] * 1.8) & (f[..., 0] > f[..., 2] * 1.8),
    "blue": lambda f: (f[..., 2] > 90) & (f[..., 2] > f[..., 0] * 1.8) & (f[..., 2] > f[..., 1] * 1.6),
    "green": lambda f: (f[..., 1] > 70) & (f[..., 1] > f[..., 0] * 1.6) & (f[..., 1] > f[..., 2] * 1.6),
}
CAMERAS = ("top", "front")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("outputs/dataset_v2"))
    p.add_argument("--limit", type=int, default=60, help="調べるエピソード数")
    p.add_argument("--window", type=int, default=20, help="把持直前の何フレームを見るか")
    p.add_argument("--min-pixels", type=int, default=4, help="見えていると判定する画素数")
    args = p.parse_args()

    paths = sorted(args.src.glob("*.npz"))[: args.limit]
    if not paths:
        print(f"[ERROR] npz が無い: {args.src}")
        return 1

    counts = {cam: [0, 0] for cam in CAMERAS}  # [見えたフレーム, 総フレーム]
    either = [0, 0]
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            actions = npz["actions"]
            color = str(npz["target_color"])
            imgs = {cam: npz[f"images_{cam}"] for cam in CAMERAS}

        grip = actions[:, 7]
        closing = np.flatnonzero(grip < 0.5)
        if closing.size == 0:
            continue
        t_grasp = int(closing[0])
        t0 = max(0, t_grasp - args.window)
        mask_fn = TARGET_HUE[color]

        for t in range(t0, t_grasp):
            seen_any = False
            for cam in CAMERAS:
                n = int(mask_fn(imgs[cam][t].astype(np.int16)).sum())
                visible = n >= args.min_pixels
                counts[cam][0] += int(visible)
                counts[cam][1] += 1
                seen_any |= visible
            either[0] += int(seen_any)
            either[1] += 1

    print(f"対象: {len(paths)} エピソード / 把持直前 {args.window} フレーム / "
          f"{args.min_pixels} 画素以上を「見えている」とする\n")
    for cam in CAMERAS:
        seen, total = counts[cam]
        print(f"  {cam:6s}  {seen}/{total}  = {seen / total:.1%}")
    print(f"  どちらか  {either[0]}/{either[1]}  = {either[0] / either[1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
