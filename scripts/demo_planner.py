#!/usr/bin/env python3
"""TaskPlanner の動作確認用 CLI。

使い方:
    ./.venv/bin/python scripts/demo_planner.py --instruction "赤を左のトレイに"
    ./.venv/bin/python scripts/demo_planner.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.planner import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_MODEL,
    PlannerConnectionError,
    TaskPlanner,
)
from pai_final.planner_testset import PLANNER_CASES  # noqa: E402


def _format_plan(plan: list[dict]) -> str:
    if not plan:
        return "(なし)"
    return " -> ".join(f"{p['color']}:{p['tray']}" for p in plan)


def run_single(planner: TaskPlanner, instruction: str) -> int:
    start = time.perf_counter()
    plan = planner.plan(instruction)
    elapsed = time.perf_counter() - start

    print(f"指示文      : {instruction}")
    print(f"解釈結果    : {_format_plan(plan)}")
    for step, goal in enumerate(plan, start=1):
        print(f"  step {step}: env.reset(target_color={goal['color']!r}, "
              f"target_tray={goal['tray']!r})")
    if planner.last_report:
        print(f"注意・理由  : {planner.last_report}")
    print(f"推論時間    : {elapsed:.2f} 秒")
    return 0 if plan else 1


def run_self_test(planner: TaskPlanner) -> int:
    total = len(PLANNER_CASES)
    correct = 0
    durations: list[float] = []
    failures: list[tuple[str, list[dict], list[dict]]] = []

    print(f"モデル: {planner.model}  ホスト: {planner.host}")
    print(f"テスト件数: {total}")
    print("-" * 78)

    for instruction, expected in PLANNER_CASES:
        start = time.perf_counter()
        actual = planner.plan(instruction)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)

        ok = actual == expected
        correct += int(ok)
        mark = "OK  " if ok else "NG  "
        print(f"{mark} [{elapsed:5.2f}s] {instruction}")
        print(f"       期待: {_format_plan(expected)}")
        if not ok:
            print(f"       実際: {_format_plan(actual)}")
            failures.append((instruction, expected, actual))
        if planner.last_report:
            print(f"       note: {planner.last_report}")

    print("-" * 78)
    print(f"正解数: {correct}/{total} ({100.0 * correct / total:.1f}%)")
    print(
        f"推論時間: 合計 {sum(durations):.2f}s / 平均 {sum(durations) / total:.2f}s "
        f"/ 最小 {min(durations):.2f}s / 最大 {max(durations):.2f}s"
    )
    if failures:
        print("\n失敗した指示文:")
        for instruction, expected, actual in failures:
            print(f"  - {instruction}")
            print(f"      期待 {_format_plan(expected)} / 実際 {_format_plan(actual)}")
    return 0 if correct == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="TaskPlanner の動作確認")
    parser.add_argument("--instruction", help="解釈させる指示文")
    parser.add_argument("--self-test", action="store_true", help="テスト集合を一括実行")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama のモデル名")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama のホスト")
    args = parser.parse_args()

    if not args.instruction and not args.self_test:
        parser.error("--instruction か --self-test のどちらかを指定してください")

    planner = TaskPlanner(model=args.model, host=args.host)
    try:
        if args.self_test:
            return run_self_test(planner)
        return run_single(planner, args.instruction)
    except PlannerConnectionError as exc:
        print(f"[エラー] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
