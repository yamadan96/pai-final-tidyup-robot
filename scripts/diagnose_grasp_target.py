#!/usr/bin/env python
"""ACT ポリシーが「指定された色のブロック」を掴みに行けているかを調べる診断。

全体成功率だけでは、失敗が「狙いは正しいが掴めない」のか
「そもそも別のブロックへ向かっている」のかが分からない。
各エピソードで 3 個のブロックの移動量と、TCP が最接近したブロックを記録する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import CAMERA_PRESETS, COLORS, TRAYS, TidyUpEnv  # noqa: E402

CAMERAS = ("top", "front")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/checkpoints/act_v2/100000/pretrained_model"))
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--camera-preset", default="tight", choices=sorted(CAMERA_PRESETS))
    p.add_argument("--temporal-ensemble", type=float, default=0.01)
    p.add_argument("--json-out", type=Path, default=Path("outputs/diag_grasp_target.json"))
    p.add_argument("--goal-cue", action="store_true",
                   help="観測画像にゴールのマーカーを描き込む（学習データと揃える）")
    p.add_argument("--goal-repeat", type=int, default=None,
                   help="state のゴール one-hot の繰り返し回数（既定: checkpoint から自動判定）")
    args = p.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    if args.temporal_ensemble is not None:
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble, policy.config.chunk_size
        )
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # state の次元から goal_repeat を逆算する（学習時との食い違いを防ぐ）
    state_dim = int(policy.config.input_features["observation.state"].shape[0])
    if args.goal_repeat is None:
        if (state_dim - 8) % 5 != 0:
            raise SystemExit(f"state_dim={state_dim} からゴールの繰り返し回数を決められない")
        args.goal_repeat = (state_dim - 8) // 5
    print(f"state_dim={state_dim} goal_repeat={args.goal_repeat} goal_cue={args.goal_cue}")

    env = TidyUpEnv(seed=args.seed, image_size=args.image_size,
                    max_steps=args.max_steps, camera_preset=args.camera_preset,
                    goal_cue=args.goal_cue, goal_repeat=args.goal_repeat)
    combos = [(c, t) for c in COLORS for t in TRAYS]
    rows = []

    for ep in range(args.episodes):
        color, tray = combos[ep % len(combos)]
        obs = env.reset(color, tray)
        policy.reset()
        start = {c: env.block_pos(c).copy() for c in COLORS}
        min_dist = {c: 1e9 for c in COLORS}

        done = False
        info: dict = {}
        while not done:
            batch = {"observation.state": torch.from_numpy(
                np.asarray(obs["state"], dtype=np.float32))}
            for cam in CAMERAS:
                img = np.asarray(obs["images"][cam], dtype=np.uint8)
                batch[f"observation.images.{cam}"] = (
                    torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0)
            action = post(policy.select_action(pre(batch)))
            obs, done, info = env.step(action.squeeze(0).to(torch.float32).cpu().numpy())

            tcp, _ = env.tcp_pose()
            for c in COLORS:
                d = float(np.linalg.norm(env.block_pos(c) - tcp))
                min_dist[c] = min(min_dist[c], d)

        moved = {c: float(np.linalg.norm(env.block_pos(c) - start[c])) for c in COLORS}
        nearest = min(COLORS, key=lambda c: min_dist[c])
        most_moved = max(COLORS, key=lambda c: moved[c])
        row = {
            "episode": ep,
            "target_color": color,
            "target_tray": tray,
            "success": bool(info.get("success", False)),
            "steps": int(env.steps),
            "min_tcp_dist": {c: round(min_dist[c], 4) for c in COLORS},
            "displacement": {c: round(moved[c], 4) for c in COLORS},
            "nearest_block": nearest,
            "most_moved_block": most_moved,
            "approached_target": nearest == color,
            "moved_target": most_moved == color and moved[color] > 0.02,
        }
        rows.append(row)
        print(f"ep{ep:02d} 目標={color:5s}->{tray}  成功={row['success']!s:5s} "
              f"最接近={nearest:5s} 最大移動={most_moved:5s} "
              f"dist={row['min_tcp_dist']} move={row['displacement']}")

    n = len(rows)
    appr = sum(r["approached_target"] for r in rows)
    mov = sum(r["moved_target"] for r in rows)
    print(f"\n正しい色に最接近した割合: {appr}/{n} ({appr / n:.0%})")
    print(f"正しい色を 2cm 以上動かした割合: {mov}/{n} ({mov / n:.0%})")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"rows": rows, "approached_target": appr, "moved_target": mov, "n": n},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
