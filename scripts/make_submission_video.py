#!/usr/bin/env python
"""提出用動画を章ごとに書き出す CLI。

章を分けてあるのは、学習や評価のやり直しで一部だけ作り直したいため。
最後に --concat で 1 本につなぐ。

    # 章ごとに作る
    ./.venv/bin/python scripts/make_submission_video.py --sections intro,llm,expert
    ./.venv/bin/python scripts/make_submission_video.py --sections act,outro
    # つなぐ
    ./.venv/bin/python scripts/make_submission_video.py --concat \
        outputs/video/intro.mp4 outputs/video/llm.mp4 ... --out outputs/video/final.mp4
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
    C_DONE,
    C_INFO,
    C_PENDING,
    C_PLAIN,
    COLOR_JA,
    POLICY_JA,
    TRAY_JA,
    load_act_policy,
    plan_lines,
)
from pai_final.env import VIDEO_CAMERA, TidyUpEnv  # noqa: E402
from pai_final.expert import ScriptedExpert  # noqa: E402
from pai_final.overlay import annotate, info_card, title_card  # noqa: E402
from pai_final.planner import TaskPlanner  # noqa: E402

SIZE = 560
FPS = 50

# LLM 章で見せる指示（最後の 1 つは意図的に実行不能なもの）
LLM_INSTRUCTIONS = [
    "赤いブロックを手前のトレイに入れて",
    "青と緑を奥のトレイに片付けて",
    "ブロックを全部手前のトレイに入れて",
    "黄色いのを取って",
]

EXPERT_INSTRUCTION = "ブロックを全部手前のトレイに入れて"


def sec(n: float) -> int:
    return int(round(FPS * n))


def write(frames: list[np.ndarray], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=FPS, macro_block_size=1)
    mb = path.stat().st_size / 1024 / 1024
    print(f"  -> {path}  {len(frames)} frames / {len(frames) / FPS:.1f}s / {mb:.1f} MB")


# --------------------------------------------------------------------- 章ごと


def section_intro() -> list[np.ndarray]:
    frames = title_card(
        (SIZE, SIZE),
        "自然言語で指示する片付けロボット",
        ["ローカル LLM × 模倣学習", "MuJoCo / Franka Emika Panda"],
        n_frames=sec(3.0),
    )
    frames += info_card(
        (SIZE, SIZE),
        "タスク",
        [
            "机の上に赤・青・緑のブロックが 1 個ずつ",
            "手前と奥に受け皿のトレイが 1 つずつ",
            "　（この視点では 手前 = 画面左 / 奥 = 画面右）",
            "置かれる位置と向きは毎回ランダム",
            "日本語の指示どおりに片付ける",
        ],
        n_frames=sec(5.0),
        note="MuJoCo 3.x / Franka Emika Panda（7 自由度 + グリッパ）",
    )
    frames += info_card(
        (SIZE, SIZE),
        "構成",
        [
            "① 指示文  「赤いブロックを手前のトレイに入れて」",
            "② ローカル LLM (qwen2.5-coder / Ollama) が JSON に分解",
            "③ ゴール（色 one-hot 3 + トレイ one-hot 2）に変換",
            "④ 方策がゴールと 2 台のカメラ画像から関節指令を出す",
            "⑤ MuJoCo が物理を進める（制御 50 Hz）",
        ],
        n_frames=sec(6.0),
        note="LLM はローカル実行・推論のみ。API も通信も使わない",
    )
    return frames


def section_llm(model: str) -> list[np.ndarray]:
    planner = TaskPlanner(model=model)
    frames = info_card(
        (SIZE, SIZE),
        "① 指示を解釈する",
        [
            "ローカル LLM に温度 0・JSON 出力で問い合わせ",
            "色とトレイの表記ゆれは Python 側の対応表で吸収",
            "指示にない色は実行せず、警告を返す",
        ],
        n_frames=sec(3.5),
        note=f"モデル: {model}（Ollama, 手元の Mac で実行）",
    )
    for instruction in LLM_INSTRUCTIONS:
        t0 = time.time()
        tasks = planner.plan(instruction)
        dt = time.time() - t0
        report = planner.last_report
        print(f"  [LLM] {instruction} -> {tasks} ({dt:.2f}s) {report}")

        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"ローカル LLM: {model}", C_INFO),
            ("", (0, 0, 0)),
        ]
        if tasks:
            lines += plan_lines(tasks, None)
        else:
            lines += [("実行できるタスクなし", (255, 140, 140))]
        if report:
            lines += [("", (0, 0, 0)), (report[:38], C_PENDING)]

        card = annotate(
            np.full((SIZE, SIZE, 3), 18, np.uint8),
            lines,
            header=f"指示: 「{instruction}」",
            footer=f"推論 {dt:.2f} 秒 · 温度 0 · JSON 出力",
        )
        frames += [card] * sec(3.2)
    return frames


def _rollout(
    env: TidyUpEnv,
    act_fn,
    tasks: list[dict],
    *,
    header: str,
    policy_label: str,
    footer: str,
    frames: list[np.ndarray],
    on_reset=None,
    frame_skip: int = 1,
    first_reset: bool = True,
    continuous: bool = True,
) -> list[dict]:
    """tasks を順に実行しながらフレームを積む。戻り値は各タスクの結果。"""
    results = []
    for ti, task in enumerate(tasks):
        if ti == 0 or not continuous:
            obs = env.reset(task["color"], task["tray"]) if (first_reset or ti > 0) else env.reset(task["color"], task["tray"])
        else:
            obs = env.set_goal(task["color"], task["tray"])
        if on_reset is not None:
            on_reset()

        lines = [(policy_label, C_INFO), ("", (0, 0, 0))] + plan_lines(tasks, ti)
        done = False
        info: dict = {}
        step = 0
        while not done:
            if step % frame_skip == 0:
                frames.append(
                    annotate(env.render(VIDEO_CAMERA, size=SIZE), lines, header=header, footer=footer)
                )
            obs, done, info = env.step(act_fn(obs))
            step += 1

        success = bool(info.get("success", False))
        results.append({**task, "success": success, "steps": int(env.steps)})
        print(f"  {'OK ' if success else 'NG '} {task['color']:5s} -> {task['tray']}  steps={env.steps}")

        tail_lines = [(policy_label, C_INFO), ("", (0, 0, 0))] + plan_lines(
            tasks, ti + 1 if success else ti
        )
        frames += [
            annotate(
                env.render(VIDEO_CAMERA, size=SIZE),
                tail_lines,
                header=header,
                footer="成功" if success else "失敗（上限 400 ステップ）",
            )
        ] * sec(0.7)
    return results


def section_expert(model: str, seed: int) -> list[np.ndarray]:
    frames = info_card(
        (SIZE, SIZE),
        "② まず規範方策で解く",
        [
            "減衰最小二乗 IK + 8 フェーズの状態機械",
            "接近 → 降下 → 把持 → 持ち上げ → 搬送 → 設置 → 解放 → 退避",
            "重力たわみ 8.6 mm を積分補正で打ち消す",
            "300 本中 294 本成功（98.0%）: これが模倣学習の教師データ",
        ],
        n_frames=sec(5.0),
        note="この方策は学習したものではなく、手で書いた制御器",
    )
    planner = TaskPlanner(model=model)
    tasks = planner.plan(EXPERT_INSTRUCTION)
    print(f"  [LLM] {EXPERT_INSTRUCTION} -> {tasks}")
    env = TidyUpEnv(seed=seed, image_size=96, max_steps=400, render_images=False)
    expert = ScriptedExpert(env)
    header = f"指示: 「{EXPERT_INSTRUCTION}」"
    _rollout(
        env,
        lambda obs: expert.act(obs),
        tasks,
        header=header,
        policy_label=POLICY_JA["expert"],
        footer="規範方策 · 等速 · 手前 = 画面左 / 奥 = 画面右",
        frames=frames,
        on_reset=expert.reset,
    )
    env.close()
    return frames


def section_act(args) -> list[np.ndarray]:
    frames = info_card(
        (SIZE, SIZE),
        "③ 模倣学習（ACT）で置き換える",
        [
            "294 本・66,815 フレームを LeRobot 形式に変換",
            "Action Chunking Transformer（ResNet18 × 2 + Transformer, 51.6M）",
            "入力: 俯瞰 160×160 + 斜め前方 160×160 + 関節 8 + ゴール",
            "出力: 100 ステップ分の関節指令（時間アンサンブルで混合）",
            f"学習: {args.train_steps} ステップ / Apple M 系 GPU (MPS)",
        ],
        n_frames=sec(5.5),
        note="ブロック位置は方策に与えない。画像から読み取る必要がある",
    )
    reset_fn, act_fn, desc = load_act_policy(args.checkpoint, args.temporal_ensemble)
    print(f"  policy: {desc}")
    env = TidyUpEnv(
        seed=args.seed,
        image_size=args.image_size,
        max_steps=400,
        camera_preset=args.camera_preset,
        goal_cue=args.goal_cue,
        goal_repeat=args.goal_repeat,
    )
    tasks = [{"color": c, "tray": t} for c, t in args.act_goals]
    header = "指示: 「{} ブロックを{}に入れて」"
    results = []
    for task in tasks:
        h = header.format(COLOR_JA[task["color"]], TRAY_JA[task["tray"]])
        results += _rollout(
            env,
            act_fn,
            [task],
            header=h,
            policy_label=POLICY_JA["act"],
            footer="模倣学習ポリシー · 等速 · 手前 = 画面左 / 奥 = 画面右",
            frames=frames,
            on_reset=reset_fn,
            continuous=False,
        )
    env.close()
    Path("outputs/video").mkdir(parents=True, exist_ok=True)
    Path("outputs/video/act_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frames


def section_outro(args) -> list[np.ndarray]:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    frames = info_card(
        (SIZE, SIZE),
        "結果",
        summary["result_lines"],
        n_frames=sec(6.0),
        note=summary.get("result_note", ""),
    )
    frames += info_card(
        (SIZE, SIZE),
        "分かったこと",
        summary["finding_lines"],
        n_frames=sec(6.5),
        note=summary.get("finding_note", ""),
    )
    frames += title_card(
        (SIZE, SIZE),
        "コードは GitHub に公開",
        summary.get("closing_lines", []),
        n_frames=sec(3.0),
    )
    return frames


# ------------------------------------------------------------------- エントリ


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", type=str, default="intro,llm,expert,act,outro")
    p.add_argument("--outdir", type=Path, default=Path("outputs/video"))
    p.add_argument("--model", type=str, default="qwen2.5-coder:latest")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/checkpoints/act_v2/100000/pretrained_model"))
    p.add_argument("--temporal-ensemble", type=float, default=0.01)
    p.add_argument("--camera-preset", type=str, default="tight")
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--goal-cue", action="store_true")
    p.add_argument("--goal-repeat", type=int, default=1)
    p.add_argument("--train-steps", type=str, default="100,000")
    p.add_argument("--act-goals", type=str, default="red:A,blue:A")
    p.add_argument("--summary", type=Path, default=Path("report/video_summary.json"))
    p.add_argument("--concat", nargs="*", default=None, help="つなぐ mp4 を順に並べる")
    p.add_argument("--out", type=Path, default=Path("outputs/video/final.mp4"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.concat:
        import imageio.v2 as imageio

        frames: list[np.ndarray] = []
        for path in args.concat:
            reader = imageio.get_reader(path)
            n = 0
            for frame in reader:
                frames.append(np.asarray(frame))
                n += 1
            reader.close()
            print(f"  {path}: {n} frames")
        write(frames, args.out)
        print(f"合計 {len(frames) / FPS:.1f} 秒")
        return 0

    args.act_goals = [
        tuple(part.split(":")) for part in args.act_goals.split(",") if part
    ]

    builders = {
        "intro": lambda: section_intro(),
        "llm": lambda: section_llm(args.model),
        "expert": lambda: section_expert(args.model, args.seed),
        "act": lambda: section_act(args),
        "outro": lambda: section_outro(args),
    }
    for name in args.sections.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in builders:
            print(f"[ERROR] 未知の章: {name}", file=sys.stderr)
            return 1
        print(f"\n=== 章: {name} ===")
        write(builders[name](), args.outdir / f"{name}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
