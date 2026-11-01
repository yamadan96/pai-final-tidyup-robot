#!/usr/bin/env python
"""自然言語の指示 → ローカル LLM の計画 → ロボット実行 を一気に通すデモ CLI。

    ./.venv/bin/python scripts/demo_pipeline.py \
        --instruction "ブロックを全部手前のトレイに入れて" \
        --policy expert --video outputs/demo_expert.mp4

--policy expert はスクリプト型エキスパート（減衰最小二乗 IK）、
--policy act は模倣学習した ACT ポリシーで実行する。
1 つの指示から複数のサブタスクが出た場合は、シーンを初期化せずにゴールだけ
差し替えて続けて実行する（連続した片付けとして見えるようにするため）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.demo_common import (  # noqa: E402
    COLOR_JA,
    POLICY_JA,
    TRAY_JA,
    load_act_policy,
    plan_lines,
)
from pai_final.env import CAMERA_PRESETS, VIDEO_CAMERA, TidyUpEnv  # noqa: E402
from pai_final.expert import ScriptedExpert  # noqa: E402
from pai_final.overlay import annotate, title_card  # noqa: E402
from pai_final.planner import PlannerConnectionError, TaskPlanner  # noqa: E402

DEFAULT_INSTRUCTIONS = [
    "赤いブロックを手前のトレイに入れて",
    "ブロックを全部奥のトレイに片付けて",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instruction", action="append", default=None, help="自然言語の指示（複数指定可）")
    p.add_argument("--policy", choices=("expert", "act"), default="expert")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/act_v2/100000/pretrained_model"))
    p.add_argument("--temporal-ensemble", type=float, default=0.01)
    p.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS), default="tight")
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--goal-cue", action="store_true", help="観測画像にゴールのマーカーを描き込む")
    p.add_argument("--goal-repeat", type=int, default=1, help="state のゴール one-hot の繰り返し回数")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-size", type=int, default=560)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--frame-skip", type=int, default=1, help="動画に残すフレームの間隔（2 で 2 倍速相当）")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--model", type=str, default="qwen2.5-coder:latest")
    p.add_argument("--no-title", action="store_true", help="タイトルカードを入れない")
    p.add_argument("--plan-seconds", type=float, default=2.5, help="計画カードを映す秒数")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    instructions = args.instruction or DEFAULT_INSTRUCTIONS

    planner = TaskPlanner(model=args.model)
    env = TidyUpEnv(
        seed=args.seed,
        image_size=args.image_size,
        max_steps=args.max_steps,
        camera_preset=args.camera_preset,
        goal_cue=args.goal_cue,
        goal_repeat=args.goal_repeat,
    )
    expert = ScriptedExpert(env) if args.policy == "expert" else None
    if args.policy == "act":
        policy_reset, act_fn, desc = load_act_policy(
            args.checkpoint, args.temporal_ensemble
        )
        print(f"policy: {desc}")

    frames: list[np.ndarray] = []
    vs = args.video_size
    speed_note = "等速" if args.frame_skip == 1 else f"{args.frame_skip} 倍速"
    footer = f"MuJoCo / Franka Emika Panda · {speed_note}"
    records: list[dict] = []

    if not args.no_title:
        frames += title_card(
            (vs, vs),
            "自然言語で指示する片付けロボット",
            [
                "ローカル LLM が指示をタスク列に分解し",
                "模倣学習した方策が実行する",
                "MuJoCo / Franka Emika Panda",
            ],
            n_frames=int(args.fps * 2.0),
        )

    for idx, instruction in enumerate(instructions):
        print(f"\n[指示 {idx + 1}/{len(instructions)}] {instruction}")
        t0 = time.time()
        try:
            tasks = planner.plan(instruction)
        except PlannerConnectionError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        plan_s = time.time() - t0
        report = planner.last_report
        print(f"  計画 ({plan_s:.2f}s): {tasks}" + (f"  / {report}" if report else ""))

        if not tasks:
            print("  → 実行できるタスクがないのでスキップ")
            records.append(
                {"instruction": instruction, "tasks": [], "report": report, "results": []}
            )
            continue

        # 計画だけを映すカード（LLM が何を出したかを見せる）
        header = f"指示: 「{instruction}」"
        plan_frame = annotate(
            np.full((vs, vs, 3), 18, np.uint8),
            [(f"ローカル LLM: {args.model}", (150, 200, 255)), ("", (0, 0, 0))]
            + plan_lines(tasks, None),
            header=header,
            footer=f"推論 {plan_s:.2f} 秒 · 温度 0 · JSON 出力",
        )
        frames += [plan_frame] * int(args.fps * args.plan_seconds)

        results = []
        for ti, task in enumerate(tasks):
            if ti == 0:
                obs = env.reset(task["color"], task["tray"])
            else:
                obs = env.set_goal(task["color"], task["tray"])
            if expert is not None:
                expert.reset()
            else:
                policy_reset()

            lines = (
                [(POLICY_JA[args.policy], (150, 200, 255)), ("", (0, 0, 0))]
                + plan_lines(tasks, ti)
            )
            done = False
            info: dict = {"success": False}
            step = 0
            while not done:
                if step % args.frame_skip == 0:
                    frames.append(
                        annotate(
                            env.render(VIDEO_CAMERA, size=vs),
                            lines,
                            header=header,
                            footer=footer,
                        )
                    )
                action = expert.act(obs) if expert is not None else act_fn(obs)
                obs, done, info = env.step(action)
                step += 1

            success = bool(info.get("success", False))
            mark = "OK " if success else "NG "
            print(f"  {mark} {task['color']:5s} -> {task['tray']}  steps={env.steps}")
            results.append(
                {
                    "color": task["color"],
                    "tray": task["tray"],
                    "success": success,
                    "steps": int(env.steps),
                }
            )
            # 結果を 0.6 秒だけ静止表示
            res_lines = (
                [(POLICY_JA[args.policy], (150, 200, 255)), ("", (0, 0, 0))]
                + plan_lines(tasks, ti + 1 if success else ti)
            )
            tail = annotate(
                env.render(VIDEO_CAMERA, size=vs),
                res_lines,
                header=header,
                footer="成功" if success else "失敗（制御ステップ上限）",
            )
            frames += [tail] * int(args.fps * 0.6)

        records.append(
            {
                "instruction": instruction,
                "tasks": tasks,
                "report": report,
                "plan_seconds": round(plan_s, 2),
                "results": results,
            }
        )

    if args.video is not None and frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=args.fps, macro_block_size=1)
        mb = args.video.stat().st_size / 1024 / 1024
        print(
            f"\n動画: {args.video} ({len(frames)} frames, "
            f"{len(frames) / args.fps:.1f} 秒, {mb:.1f} MB)"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"policy": args.policy, "seed": args.seed, "records": records},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"結果: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
