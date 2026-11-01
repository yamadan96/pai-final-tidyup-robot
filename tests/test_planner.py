"""TaskPlanner のテスト。

- LLM を実際に呼ぶパラメトリックテスト（15 件、planner_testset.PLANNER_CASES）
- LLM を呼ばない単体テスト（語彙正規化・JSON 抽出・接続エラー）

実行:
    ./.venv/bin/python -m pytest tests/test_planner.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.planner import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_MODEL,
    PlannerConnectionError,
    TaskPlanner,
    _extract_tasks,
    _mentions_tray,
    _normalize_color,
    _normalize_tray,
)
from pai_final.planner_testset import PLANNER_CASES  # noqa: E402


def _ollama_available() -> bool:
    try:
        response = requests.get(f"{DEFAULT_HOST}/api/tags", timeout=3)
    except requests.exceptions.RequestException:
        return False
    return response.status_code == 200


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason=f"Ollama ({DEFAULT_HOST}) が起動していないためスキップ",
)


@pytest.fixture(scope="module")
def planner() -> TaskPlanner:
    return TaskPlanner(model=DEFAULT_MODEL)


# --------------------------------------------------------- LLM を呼ぶテスト
@requires_ollama
@pytest.mark.parametrize(
    ("instruction", "expected"),
    PLANNER_CASES,
    ids=[case[0] for case in PLANNER_CASES],
)
def test_plan_matches_expected(
    planner: TaskPlanner, instruction: str, expected: list[dict]
) -> None:
    actual = planner.plan(instruction)
    assert actual == expected, (
        f"指示文={instruction!r} 期待={expected} 実際={actual} "
        f"note={planner.last_report} raw={planner.last_raw!r}"
    )


@requires_ollama
def test_plan_output_schema_is_valid_for_env(planner: TaskPlanner) -> None:
    """返り値が TidyUpEnv.reset() にそのまま渡せる形になっていること。"""
    plan = planner.plan("赤を左に、緑を右に片付けて")
    assert plan, planner.last_report
    for goal in plan:
        assert set(goal) == {"color", "tray"}
        assert goal["color"] in ("red", "blue", "green")
        assert goal["tray"] in ("A", "B")


@requires_ollama
def test_plan_on_empty_instruction_returns_empty_with_reason(
    planner: TaskPlanner,
) -> None:
    assert planner.plan("   ") == []
    assert "空" in planner.last_report


@requires_ollama
def test_out_of_range_color_sets_warning(planner: TaskPlanner) -> None:
    """範囲外の色は要素が落ちて、理由が残ること。"""
    assert planner.plan("黄色いブロックをトレイAに入れて") == []
    assert planner.last_report  # 警告かエラーのどちらかが入る


# --------------------------------------------------- LLM を呼ばない単体テスト
def test_connection_error_is_raised_when_ollama_is_down() -> None:
    dead = TaskPlanner(host="http://127.0.0.1:1", timeout=3.0)
    with pytest.raises(PlannerConnectionError) as excinfo:
        dead.plan("赤をトレイAに")
    assert "ollama serve" in str(excinfo.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("red", "red"), ("赤", "red"), ("レッド", "red"), ("赤いブロック", "red"),
        ("blue", "blue"), ("青", "blue"), ("ブルー", "blue"),
        ("green", "green"), ("みどり", "green"), ("green cube", "green"),
        ("yellow", None), ("黄色", None), ("", None), (None, None), (3, None),
    ],
)
def test_normalize_color(value: object, expected: str | None) -> None:
    assert _normalize_color(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("A", "A"), ("tray A", "A"), ("left", "A"), ("左", "A"), ("奥", "A"),
        ("B", "B"), ("トレイB", "B"), ("right", "B"), ("右", "B"), ("手前", "B"),
        ("C", None), ("", None), (None, None),
    ],
)
def test_normalize_tray(value: object, expected: str | None) -> None:
    assert _normalize_tray(value) == expected


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("赤を右に", True),
        ("緑を左のトレイへ", True),
        ("put it in tray b", True),
        ("move the red cube to B", True),
        ("全部片付けて", False),
        ("赤いのを取って", False),
    ],
)
def test_mentions_tray(instruction: str, expected: bool) -> None:
    assert _mentions_tray(instruction.lower()) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"tasks": [{"color": "red", "tray": "A"}]}', [{"color": "red", "tray": "A"}]),
        ('```json\n{"tasks": []}\n```', []),
        ('ここまでが結果です {"tasks": [{"color": "blue", "tray": "B"}]}',
         [{"color": "blue", "tray": "B"}]),
        ('{"plan": [{"color": "green", "tray": "A"}]}', [{"color": "green", "tray": "A"}]),
        ('[{"color": "red", "tray": "B"}]', [{"color": "red", "tray": "B"}]),
        ('{"color": "red", "tray": "A"}', [{"color": "red", "tray": "A"}]),
        ("", None),
        ("申し訳ありませんが分かりません", None),
    ],
)
def test_extract_tasks(raw: str, expected: list | None) -> None:
    assert _extract_tasks(raw) == expected


def test_normalize_drops_bad_elements_without_raising() -> None:
    """LLM が壊れた要素を返しても例外にせず、落として警告を残すこと。"""
    planner = TaskPlanner()
    result = planner._normalize(
        [
            {"color": "red", "tray": "left"},
            {"color": "yellow", "tray": "A"},
            "not a dict",
            {"color": "blue"},
        ],
        "赤を左に、黄色をAに、青も",
    )
    assert result == [
        {"color": "red", "tray": "A"},
        {"color": "blue", "tray": "A"},
    ]
    assert len(planner.last_warnings) >= 3
