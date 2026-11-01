#!/usr/bin/env python
"""学習済み ACT ポリシーを TidyUpEnv 上でロールアウトして成功率を測る CLI。

使い方:
    ./.venv/bin/python scripts/eval_policy.py \
        --episodes 6 --checkpoint outputs/checkpoints/act/last/pretrained_model \
        --video outputs/eval.mp4

色（赤/青/緑）× トレイ（A/B）の 6 通りをラウンドロビンで回し、
ゴール別の成功率と全体の成功率を表示する。

lerobot 0.4.4 では正規化が processor パイプライン側にあるので、推論も
    batch = preprocessor(obs)      # 正規化 + batch 次元 + device 転送
    action = policy.select_action(batch)
    action = postprocessor(action) # 逆正規化 + CPU 転送
の順で通す必要がある。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import CAMERA_PRESETS, COLORS, TRAYS, VIDEO_CAMERA, TidyUpEnv  # noqa: E402

CAMERAS: tuple[str, ...] = ("top", "front")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/act/last/pretrained_model"),
        help="pretrained_model ディレクトリ",
    )
    parser.add_argument("--episodes", type=int, default=6, help="評価エピソード数")
    parser.add_argument("--video", type=Path, default=Path("outputs/eval.mp4"), help="mp4 の出力先")
    parser.add_argument("--max-steps", type=int, default=400, help="1 エピソードの最大制御ステップ")
    parser.add_argument("--image-size", type=int, default=96, help="観測画像の一辺 [px]（学習時と揃える）")
    parser.add_argument("--seed", type=int, default=100, help="乱数シード（収集時と変えて未知配置で評価）")
    parser.add_argument("--video-size", type=int, default=480, help="動画の一辺 [px]")
    parser.add_argument("--video-fps", type=int, default=50, help="動画のフレームレート")
    parser.add_argument("--video-episodes", type=int, default=None, help="動画に含める本数（既定: 全部）")
    parser.add_argument("--device", type=str, default=None, help="torch device（既定: mps があれば mps）")
    parser.add_argument("--temporal-ensemble", type=float, default=None,
                        help="時間アンサンブルの係数（指定すると n_action_steps=1 で有効化）")
    parser.add_argument("--json-out", type=Path, default=None, help="結果を JSON で書き出す先")
    parser.add_argument(
        "--camera-preset",
        type=str,
        default="wide",
        choices=sorted(CAMERA_PRESETS),
        help="観測カメラのプリセット（学習データの収集時と揃える）",
    )
    parser.add_argument(
        "--goal-cue", action="store_true",
        help="観測画像にゴールのマーカーを描き込む（学習データと揃える）",
    )
    parser.add_argument(
        "--goal-repeat", type=int, default=None,
        help="state のゴール one-hot の繰り返し回数（既定: checkpoint から自動判定）",
    )
    return parser.parse_args(argv)


def select_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def goal_schedule(n_episodes: int) -> list[tuple[str, str]]:
    """色 × トレイの 6 通りをまんべんなく回す並び（collect_demos.py と同じ規則）。"""
    combos = [(color, tray) for color in COLORS for tray in TRAYS]
    return [combos[i % len(combos)] for i in range(n_episodes)]


def obs_to_batch(obs: dict) -> dict[str, torch.Tensor]:
    """環境の観測を policy の入力形式に変換する。

    学習時のデータセットは画像を float32 CHW の [0, 1] で返すため、
    環境の uint8 HWC (96,96,3) をそれに合わせる。
    batch 次元・正規化・device 転送は preprocessor が行う。
    """
    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)),
    }
    for cam in CAMERAS:
        img = np.asarray(obs["images"][cam], dtype=np.uint8)
        batch[f"observation.images.{cam}"] = (
            torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0
        )
    return batch


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if not args.checkpoint.exists():
        print(f"[ERROR] checkpoint が無い: {args.checkpoint}", file=sys.stderr)
        return 1

    device = select_device(args.device)
    print(f"device = {device}")
    print(f"checkpoint = {args.checkpoint}")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    if args.temporal_ensemble is not None:
        # 時間アンサンブル: 過去の推論結果を指数重みで混ぜ、行動のばらつきを抑える
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.config.n_action_steps = 1
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensemble, policy.config.chunk_size)
    policy.to(device)
    policy.eval()
    # 学習時の device 設定が残っていても、推論側の device に合わせる
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(
        f"policy: chunk_size={policy.config.chunk_size} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"inputs={sorted(policy.config.input_features)}"
    )

    # state の次元から goal_repeat を逆算する（学習時との食い違いを防ぐ）
    state_dim = int(policy.config.input_features["observation.state"].shape[0])
    if args.goal_repeat is None:
        if (state_dim - 8) % 5 != 0:
            print(f"[ERROR] state_dim={state_dim} からゴールの繰り返し回数を決められない", file=sys.stderr)
            return 1
        args.goal_repeat = (state_dim - 8) // 5
    print(f"state_dim={state_dim} goal_repeat={args.goal_repeat} goal_cue={args.goal_cue}")

    env = TidyUpEnv(
        seed=args.seed,
        image_size=args.image_size,
        max_steps=args.max_steps,
        camera_preset=args.camera_preset,
        goal_cue=args.goal_cue,
        goal_repeat=args.goal_repeat,
    )
    print(f"camera_preset={args.camera_preset} ({env.camera_xml}) image_size={args.image_size}")
    schedule = goal_schedule(args.episodes)
    n_video = args.episodes if args.video_episodes is None else args.video_episodes

    # ゴール別の集計（6 通りの順序を固定して表示するため OrderedDict）
    per_goal: OrderedDict[str, dict] = OrderedDict(
        (f"{color}->{tray}", {"n": 0, "success": 0, "steps": []})
        for color in COLORS
        for tray in TRAYS
    )

    video_frames: list[np.ndarray] = []
    results: list[dict] = []
    started = time.time()

    for ep, (color, tray) in enumerate(schedule):
        obs = env.reset(color, tray)
        policy.reset()  # action queue をエピソードごとにクリアする
        record = ep < n_video

        done = False
        info: dict = {"success": False}
        while not done:
            if record:
                video_frames.append(env.render(VIDEO_CAMERA, size=args.video_size))

            batch = obs_to_batch(obs)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            action_np = action.squeeze(0).to(torch.float32).cpu().numpy()

            obs, done, info = env.step(action_np)

        if record:
            video_frames.append(env.render(VIDEO_CAMERA, size=args.video_size))

        success = bool(info.get("success", False))
        key = f"{color}->{tray}"
        per_goal[key]["n"] += 1
        per_goal[key]["success"] += int(success)
        per_goal[key]["steps"].append(int(env.steps))
        results.append(
            {"episode": ep, "color": color, "tray": tray, "success": success, "steps": int(env.steps)}
        )
        mark = "OK " if success else "NG "
        print(f"  ep{ep:03d} {key:12s} {mark} steps={env.steps:4d}")

    elapsed = time.time() - started
    n_success = sum(r["success"] for r in results)
    overall = n_success / len(results) if results else 0.0

    print("\n--- ゴール別成功率 ---")
    for key, agg in per_goal.items():
        if agg["n"] == 0:
            print(f"  {key:12s}   -   (0 本)")
            continue
        rate = agg["success"] / agg["n"]
        mean_steps = sum(agg["steps"]) / len(agg["steps"])
        print(f"  {key:12s} {rate:6.1%}  ({agg['success']}/{agg['n']})  平均 {mean_steps:.0f} steps")
    print(f"\n全体成功率: {overall:.1%}  ({n_success}/{len(results)})")
    print(f"評価時間: {elapsed:.1f}s")

    if args.video is not None and video_frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, video_frames, fps=args.video_fps, macro_block_size=1)
        size_kb = args.video.stat().st_size / 1024
        size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"
        print(f"動画: {args.video} ({len(video_frames)} frames, {size_str})")

    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "overall_success_rate": round(overall, 4),
        "n_success": n_success,
        "per_goal": {
            key: {
                "n": agg["n"],
                "success": agg["success"],
                "rate": round(agg["success"] / agg["n"], 4) if agg["n"] else None,
            }
            for key, agg in per_goal.items()
        },
        "episodes_detail": results,
        "elapsed_s": round(elapsed, 1),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "camera_preset": args.camera_preset,
        "image_size": args.image_size,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"JSON: {args.json_out}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
