#!/usr/bin/env python
"""スクリプト型エキスパートで模倣学習用のデモを収集する CLI。

使い方:
    ./.venv/bin/python scripts/collect_demos.py \
        --episodes 60 --out outputs/demos --video outputs/expert_demo.mp4 --seed 0

色（赤/青/緑）× トレイ（A/B）の 6 通りをラウンドロビンで回し、成功したエピソードだけを
.npz に保存する。成功率は必ず標準出力に表示する。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import CAMERA_PRESETS, COLORS, TRAYS, VIDEO_CAMERA, TidyUpEnv  # noqa: E402
from pai_final.expert import ScriptedExpert  # noqa: E402
from pai_final.noise import DEFAULT_SKIP_PHASES, NOISE_MODES, CommandNoise  # noqa: E402

# エキスパートの台本が終わってからブロックが静止するのを待つ上限ステップ数
SETTLE_STEPS = 40


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60, help="収集を試行する本数")
    parser.add_argument("--out", type=Path, default=Path("outputs/demos"), help="npz の保存先")
    parser.add_argument("--video", type=Path, default=None, help="mp4 の出力先（省略時は書き出さない）")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--image-size", type=int, default=96, help="観測画像の一辺 [px]")
    parser.add_argument("--max-steps", type=int, default=400, help="1 エピソードの最大制御ステップ")
    parser.add_argument("--video-episodes", type=int, default=3, help="動画に含める先頭エピソード数")
    parser.add_argument("--video-size", type=int, default=480, help="動画の一辺 [px]")
    parser.add_argument("--video-fps", type=int, default=50, help="動画のフレームレート")
    parser.add_argument(
        "--camera-preset",
        type=str,
        default="wide",
        choices=sorted(CAMERA_PRESETS),
        help="観測カメラのプリセット（wide=旧設定 / tight=作業領域を画面いっぱいに）",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="指令関節角に加えるガウスノイズの標準偏差 [rad]（0 でノイズなし）",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="ctrl",
        choices=list(NOISE_MODES),
        help="ctrl=実行指令にだけ載せる / cmd=エキスパート内部指令にも積み上げる",
    )
    parser.add_argument(
        "--noise-skip-phases",
        type=str,
        default=",".join(DEFAULT_SKIP_PHASES),
        help="ノイズを加えないフェーズ（カンマ区切り、空文字で全フェーズに加える）",
    )
    return parser.parse_args(argv)


def goal_schedule(n_episodes: int) -> list[tuple[str, str]]:
    """色 × トレイの 6 通りをまんべんなく回す並び。

    6 本ごとに全組み合わせがちょうど 1 回ずつ現れる。連続するエピソードで
    トレイが交互になるよう並べてあるので、先頭数本だけを動画にしても両方の
    トレイが映る。
    """
    combos = [(color, tray) for color in COLORS for tray in TRAYS]
    return [combos[i % len(combos)] for i in range(n_episodes)]


def run_episode(
    env: TidyUpEnv,
    expert: ScriptedExpert,
    color: str,
    tray: str,
    video_frames: list[np.ndarray] | None = None,
    video_size: int = 480,
    noise: CommandNoise | None = None,
) -> dict:
    """1 エピソードを実行し、観測・行動の列と成否を返す。

    noise を渡すと、記録するラベルは素のエキスパート指令のまま、環境には
    揺らした指令を送る。ずれた状態でエキスパートが出す立て直しの指令が
    そのまま (s_t, a_t) ペアになる。
    """
    obs = env.reset(color, tray)
    expert.reset()

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: dict[str, list[np.ndarray]] = {cam: [] for cam in env.cameras}

    settle = 0
    info: dict = {"success": False}
    while True:
        # act() はフェーズを進めることがあるので、ノイズ判定用に呼ぶ前の値を取る
        phase = expert.phase
        action = expert.act(obs)

        # (s_t, a_t) のペアになるよう、行動を適用する前の観測を記録する
        states.append(obs["state"])
        actions.append(action.astype(np.float32))
        for cam in env.cameras:
            images[cam].append(obs["images"][cam])
        if video_frames is not None:
            video_frames.append(env.render(VIDEO_CAMERA, size=video_size))

        exec_action = action if noise is None else noise.perturb(action, phase, expert)
        obs, done, info = env.step(exec_action)
        if done:
            break
        if expert.done:
            settle += 1
            if settle > SETTLE_STEPS:
                break

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "images": {cam: np.asarray(frames, dtype=np.uint8) for cam, frames in images.items()},
        "goal": env.goal_vector(),
        "success": bool(env.success()),
        "steps": env.steps,
        "final_block_pos": env.block_pos(color).copy(),
        "tray_pos": env.tray_pos(tray).copy(),
        "timeout": bool(info.get("timeout", False)),
        "last_phase": expert.phase,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    env = TidyUpEnv(
        seed=args.seed,
        image_size=args.image_size,
        max_steps=args.max_steps,
        camera_preset=args.camera_preset,
    )
    expert = ScriptedExpert(env)
    skip_phases = tuple(p for p in args.noise_skip_phases.split(",") if p)
    noise = CommandNoise(
        std=args.noise_std,
        mode=args.noise_mode,
        seed=args.seed + 1_000,
        skip_phases=skip_phases,
        joint_lo=env.arm_ctrl_lo,
        joint_hi=env.arm_ctrl_hi,
    )
    print(
        f"camera_preset={args.camera_preset} ({env.camera_xml})  image_size={args.image_size}"
    )
    print(
        f"noise: mode={noise.mode} std={noise.std} rad skip_phases={list(skip_phases)}"
    )

    schedule = goal_schedule(args.episodes)
    video_frames: list[np.ndarray] | None = [] if args.video is not None else None

    n_success = 0
    saved: list[dict] = []
    started = time.time()

    for idx, (color, tray) in enumerate(schedule):
        want_video = video_frames is not None and idx < args.video_episodes
        episode = run_episode(
            env,
            expert,
            color,
            tray,
            video_frames=video_frames if want_video else None,
            video_size=args.video_size,
            noise=noise,
        )
        n_success += int(episode["success"])

        dist = float(np.linalg.norm(episode["final_block_pos"][:2] - episode["tray_pos"][:2]))
        status = "OK  " if episode["success"] else "FAIL"
        print(
            f"[{idx + 1:3d}/{args.episodes}] {status} {color:5s} -> tray {tray} "
            f"steps={episode['steps']:3d} tray_dist={dist:.3f}m "
            f"block_z={episode['final_block_pos'][2]:.3f}m phase={episode['last_phase']}"
        )

        if not episode["success"]:
            continue

        path = args.out / f"ep{idx:04d}_{color}_{tray}.npz"
        np.savez_compressed(
            path,
            states=episode["states"],
            actions=episode["actions"],
            goal=episode["goal"],
            success=episode["success"],
            target_color=color,
            target_tray=tray,
            control_dt=env.control_dt,
            final_block_pos=episode["final_block_pos"],
            tray_pos=episode["tray_pos"],
            **{f"images_{cam}": frames for cam, frames in episode["images"].items()},
        )
        saved.append(
            {
                "file": path.name,
                "color": color,
                "tray": tray,
                "steps": int(episode["steps"]),
                "length": int(episode["states"].shape[0]),
            }
        )

    elapsed = time.time() - started
    rate = n_success / args.episodes if args.episodes else 0.0
    print("-" * 72)
    print(f"success rate: {n_success}/{args.episodes} = {rate * 100:.1f}%")
    print(f"noise applied: {noise.stats()}")
    print(f"saved episodes: {len(saved)} -> {args.out}  ({elapsed:.1f}s)")
    if saved:
        lengths = [item["length"] for item in saved]
        print(
            f"episode length: mean={np.mean(lengths):.1f} min={min(lengths)} max={max(lengths)} steps"
        )

    if video_frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, video_frames, fps=args.video_fps, quality=8)
        print(
            f"video: {args.video} ({len(video_frames)} frames, "
            f"{len(video_frames) / args.video_fps:.1f}s @ {args.video_fps}fps)"
        )

    meta = {
        "episodes_attempted": args.episodes,
        "episodes_saved": len(saved),
        "success_rate": rate,
        "seed": args.seed,
        "image_size": args.image_size,
        "max_steps": args.max_steps,
        "control_dt": env.control_dt,
        "state_dim": 13,
        "action_dim": 8,
        "cameras": list(env.cameras),
        "camera_preset": args.camera_preset,
        "camera_xml": dict(env.camera_xml),
        "noise": noise.stats(),
        "state_layout": "arm_joint[0:7], gripper_width_norm[7], color_onehot[8:11], tray_onehot[11:13]",
        "action_layout": "arm_joint_target[0:7] (rad), gripper_cmd[7] (1=open, 0=closed)",
        "colors": list(COLORS),
        "trays": list(TRAYS),
        "episodes": saved,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
