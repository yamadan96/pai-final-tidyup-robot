"""プランナ評価用の指示文セット（demo_planner.py と pytest で共有する）。

各要素は ``(指示文, 期待される plan() の出力)``。
期待値の根拠は planner.py 冒頭の対応表（左=A / 右=B / 手前=B / 奥=A、
「全部」= red,blue,green の順、トレイ未指定時は A）に従う。
"""

from __future__ import annotations

# (instruction, expected)
PLANNER_CASES: tuple[tuple[str, list[dict]], ...] = (
    # --- 日本語・単一タスク（明示的なトレイ名） ---
    ("赤いブロックをトレイAに入れて", [{"color": "red", "tray": "A"}]),
    ("レッドブロックをBに移動して", [{"color": "red", "tray": "B"}]),
    # --- 日本語・言い換え（左右／口語） ---
    ("青いやつを右のトレイに置いて", [{"color": "blue", "tray": "B"}]),
    ("緑のブロックを左に片付けて", [{"color": "green", "tray": "A"}]),
    # --- 日本語・手前／奥（規約: 手前=B, 奥=A） ---
    ("赤いのを手前のトレイに入れて", [{"color": "red", "tray": "B"}]),
    # --- 日本語・複数タスク ---
    (
        "赤を左に、青は右に",
        [{"color": "red", "tray": "A"}, {"color": "blue", "tray": "B"}],
    ),
    (
        "緑を奥のトレイに入れてから、青を手前に",
        [{"color": "green", "tray": "A"}, {"color": "blue", "tray": "B"}],
    ),
    # --- 英語・単一タスク ---
    ("Put the red cube in tray A", [{"color": "red", "tray": "A"}]),
    ("Move the blue block to the right tray", [{"color": "blue", "tray": "B"}]),
    ("Please place the green one into tray B", [{"color": "green", "tray": "B"}]),
    # --- 英語・複数タスク ---
    (
        "First put the blue block in tray B, then the red one in tray A",
        [{"color": "blue", "tray": "B"}, {"color": "red", "tray": "A"}],
    ),
    # --- 「全部」の展開（red, blue, green の順） ---
    (
        "ブロックを全部トレイBに入れて",
        [
            {"color": "red", "tray": "B"},
            {"color": "blue", "tray": "B"},
            {"color": "green", "tray": "B"},
        ],
    ),
    (
        "全部片付けて",
        [
            {"color": "red", "tray": "A"},
            {"color": "blue", "tray": "A"},
            {"color": "green", "tray": "A"},
        ],
    ),
    # --- 範囲外の色（要素を落として空リスト） ---
    ("黄色いのを取って", []),
    ("Move the purple block to tray A", []),
)
