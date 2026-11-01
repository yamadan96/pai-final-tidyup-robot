"""自然言語の指示を TidyUpEnv のゴール指定（色・トレイ）へ変換するプランナ。

ローカル LLM（Ollama 上の qwen2.5-coder）に JSON だけを返させ、Python 側で
語彙の正規化と妥当性検証を行う。LLM が多少崩れた表記（「赤」「左」など）を
返しても Python 側で吸収できるようにしている。

出力形式
--------
``plan()`` は実行順のリストを返す。各要素は ``{"color": "red", "tray": "A"}``。
解釈できなかった場合は空リストを返し、理由は ``last_error`` /
``last_warnings`` に格納する（例外では落とさない）。
Ollama へ接続できない場合のみ ``PlannerConnectionError`` を送出する。

語彙 → トレイ 対応表
--------------------
トレイ A : ワールド座標 (x=0.30, y=+0.38) = ロボットから見て左手
トレイ B : ワールド座標 (x=0.30, y=-0.38) = ロボットから見て右手

    "A", "tray A", "トレイA", "Aのトレイ", "a"        -> "A"
    "B", "tray B", "トレイB", "Bのトレイ", "b"        -> "B"
    "left", "左", "左側", "左手", "レフト"             -> "A"
    "right", "右", "右側", "右手", "ライト"            -> "B"
    "far", "奥", "奥側", "向こう"                      -> "A"
    "near", "手前", "手前側", "こっち"                 -> "B"

注意: 2 つのトレイは同じ奥行き (x=0.30) に置かれているため、「手前/奥」は
幾何的には区別できない。ここでは動画撮影用の front カメラ (1.35, -0.75, 0.75)
が -y 側にあることから、「カメラ（視点）に近い側 = B」「遠い側 = A」という
規約を採用する。あいまいな語なので、使われた場合は警告を残す。

色の語彙
--------
    "red",   "赤", "あか", "レッド", "赤色"   -> "red"
    "blue",  "青", "あお", "ブルー", "青色"   -> "blue"
    "green", "緑", "みどり", "グリーン", "緑色" -> "green"

既定の補完
----------
- 「全部 / すべて / all / everything」= 赤・青・緑の 3 件に展開する。
- トレイが一切言及されない場合は既定で "A" を使い、警告を残す。
- red/blue/green 以外の色（黄色など）はその要素を落として警告を残す。
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

COLORS: tuple[str, ...] = ("red", "blue", "green")
TRAYS: tuple[str, ...] = ("A", "B")

DEFAULT_MODEL = "qwen2.5-coder:latest"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 120.0

# LLM 出力の表記ゆれを正規化するための対応表（キーは小文字化して比較する）
COLOR_ALIASES: dict[str, str] = {
    "red": "red", "赤": "red", "あか": "red", "レッド": "red", "赤色": "red",
    "赤い": "red", "r": "red",
    "blue": "blue", "青": "blue", "あお": "blue", "ブルー": "blue",
    "青色": "blue", "青い": "blue", "b": "blue",
    "green": "green", "緑": "green", "みどり": "green", "グリーン": "green",
    "緑色": "green", "g": "green",
}

TRAY_ALIASES: dict[str, str] = {
    "a": "A", "tray a": "A", "traya": "A", "トレイa": "A", "aのトレイ": "A",
    "left": "A", "左": "A", "左側": "A", "左手": "A", "レフト": "A",
    "far": "A", "奥": "A", "奥側": "A", "向こう": "A",
    "b": "B", "tray b": "B", "trayb": "B", "トレイb": "B", "bのトレイ": "B",
    "right": "B", "右": "B", "右側": "B", "右手": "B", "ライト": "B",
    "near": "B", "手前": "B", "手前側": "B", "こっち": "B",
}

# 「手前/奥」は幾何的に一意でないため、使われたら警告を出す語
AMBIGUOUS_TRAY_WORDS = ("手前", "奥", "near", "far", "こっち", "向こう")

SYSTEM_PROMPT = """You convert an instruction for a tidy-up robot into a JSON plan.

The robot can move exactly three blocks: red, blue, green.
There are exactly two trays: "A" (the robot's left hand side) and "B" (the robot's right hand side).

Reply with ONLY this JSON object, nothing else:
{"tasks": [{"color": "<red|blue|green>", "tray": "<A|B>"}]}

Rules:
- One element per block to move, listed in execution order.
- Tray vocabulary (apply exactly):
    "tray A" / "A" / "トレイA" / left / 左 / 左側 / 左手  -> "A"
    "tray B" / "B" / "トレイB" / right / 右 / 右側 / 右手  -> "B"
    far  / 奥 / 奥側   -> "A"
    near / 手前 / 手前側 -> "B"
- Color vocabulary: 赤/あか/レッド -> "red", 青/あお/ブルー -> "blue", 緑/みどり/グリーン -> "green".
- Drop any block whose color is not red, blue or green.
- "all" / "everything" / "全部" / "すべて" / "みんな" means every block: output three
  tasks in this exact order: red, blue, green.
- If no tray is mentioned anywhere, use "A".
- If nothing can be moved, reply {"tasks": []}.
Output JSON only. No explanation, no markdown fence."""

# few-shot 例（JSON 以外を返させないための見本。chat の履歴として渡す）
FEW_SHOT: tuple[tuple[str, str], ...] = (
    (
        "赤いブロックをトレイAに入れて",
        '{"tasks": [{"color": "red", "tray": "A"}]}',
    ),
    (
        "Move the blue block to the right tray",
        '{"tasks": [{"color": "blue", "tray": "B"}]}',
    ),
    (
        "緑を奥のトレイに入れてから、青を手前に",
        '{"tasks": [{"color": "green", "tray": "A"}, {"color": "blue", "tray": "B"}]}',
    ),
    (
        "全部トレイBに片付けて",
        '{"tasks": [{"color": "red", "tray": "B"}, {"color": "blue", "tray": "B"},'
        ' {"color": "green", "tray": "B"}]}',
    ),
    (
        "黄色いのを取って",
        '{"tasks": []}',
    ),
)

RETRY_NUDGE = (
    "Your previous reply was not valid JSON. Reply with ONLY the JSON object "
    '{"tasks": [...]} and nothing else.'
)


class PlannerConnectionError(RuntimeError):
    """Ollama に接続できない / モデルが無いときに送出する。"""


class TaskPlanner:
    """自然言語の指示を ``[{"color": ..., "tray": ...}, ...]`` に変換する。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.last_error: str | None = None
        self.last_warnings: list[str] = []
        self.last_raw: str = ""

    # ------------------------------------------------------------------ public
    def plan(self, instruction: str) -> list[dict]:
        """指示文を解釈してゴールのリストを返す。失敗時は空リスト。"""
        self.last_error = None
        self.last_warnings = []
        self.last_raw = ""

        if not instruction or not instruction.strip():
            self.last_error = "指示文が空です"
            return []

        messages = self._build_messages(instruction)
        raw = self._chat(messages)
        self.last_raw = raw
        tasks = _extract_tasks(raw)

        if tasks is None:
            # 1 回だけ再試行する（JSON だけを返すよう念押しする）
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_NUDGE},
            ]
            raw = self._chat(messages)
            self.last_raw = raw
            tasks = _extract_tasks(raw)

        if tasks is None:
            self.last_error = f"LLM の応答を JSON として解釈できませんでした: {raw[:200]!r}"
            return []

        return self._normalize(tasks, instruction)

    @property
    def last_report(self) -> str:
        """直前の plan() の警告・エラーを 1 行にまとめた説明。"""
        parts = list(self.last_warnings)
        if self.last_error:
            parts.append(self.last_error)
        return " / ".join(parts)

    # ----------------------------------------------------------------- private
    def _build_messages(self, instruction: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for user_text, assistant_text in FEW_SHOT:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": instruction})
        return messages

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        url = f"{self.host}/api/chat"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise PlannerConnectionError(
                f"Ollama ({self.host}) に接続できません。"
                "`ollama serve` が起動しているか確認してください。"
                f" 元のエラー: {exc.__class__.__name__}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise PlannerConnectionError(
                f"Ollama ({self.host}) が {self.timeout:.0f} 秒以内に応答しませんでした。"
                f" モデル {self.model} のロードが終わっていない可能性があります。"
            ) from exc

        if response.status_code != 200:
            raise PlannerConnectionError(
                f"Ollama が HTTP {response.status_code} を返しました: "
                f"{response.text[:200]} (model={self.model})"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise PlannerConnectionError(
                f"Ollama の応答が JSON ではありません: {response.text[:200]}"
            ) from exc
        return str(body.get("message", {}).get("content", ""))

    def _normalize(self, tasks: list[Any], instruction: str) -> list[dict]:
        """LLM が返した要素を検証し、色・トレイを正規化する。"""
        lowered = instruction.lower()
        tray_mentioned = _mentions_tray(lowered)
        normalized: list[dict] = []

        for index, item in enumerate(tasks):
            if not isinstance(item, dict):
                self.last_warnings.append(f"要素{index}が辞書ではないため除外: {item!r}")
                continue

            color = _normalize_color(item.get("color"))
            if color is None:
                self.last_warnings.append(
                    f"対応していない色のため除外: {item.get('color')!r}"
                    f" (使えるのは {', '.join(COLORS)})"
                )
                continue

            tray = _normalize_tray(item.get("tray"))
            if tray is None:
                if tray_mentioned:
                    self.last_warnings.append(
                        f"トレイ指定 {item.get('tray')!r} を解釈できないため既定の A を使用"
                    )
                else:
                    self.last_warnings.append("トレイが未指定のため既定の A を使用")
                tray = "A"

            normalized.append({"color": color, "tray": tray})

        if not tray_mentioned and normalized:
            self.last_warnings.append("指示文にトレイの言及がありません（既定 A と解釈）")
        if any(word in lowered for word in AMBIGUOUS_TRAY_WORDS):
            self.last_warnings.append(
                "「手前/奥」は幾何的に一意でないため規約（手前=B, 奥=A）で解釈"
            )
        if not normalized and not self.last_error:
            self.last_error = "実行可能なタスクを抽出できませんでした"

        return normalized


# ------------------------------------------------------------------- helpers
def _is_safe_substring_alias(alias: str) -> bool:
    """部分一致に使ってよいエイリアスか。

    ASCII の 1 文字（"a", "b", "r" 等）は無関係な単語に誤ヒットするため除外する。
    一方で「左」「右」「奥」のような 1 文字の日本語は部分一致に使う必要がある。
    """
    return len(alias) >= 2 or not alias.isascii()


def _normalize_color(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in COLOR_ALIASES:
        return COLOR_ALIASES[key]
    # 「赤いブロック」「red cube」のように余計な語が付いた場合も拾う
    for alias, color in COLOR_ALIASES.items():
        if _is_safe_substring_alias(alias) and alias in key:
            return color
    return None


def _normalize_tray(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in TRAY_ALIASES:
        return TRAY_ALIASES[key]
    for alias, tray in TRAY_ALIASES.items():
        if _is_safe_substring_alias(alias) and alias in key:
            return tray
    return None


def _mentions_tray(lowered_instruction: str) -> bool:
    """指示文がトレイを何らかの形で指しているか判定する。"""
    for alias in TRAY_ALIASES:
        if _is_safe_substring_alias(alias) and alias in lowered_instruction:
            return True
    # 単独の A / B（「Bに入れて」「to B」など）
    return bool(re.search(r"(?<![a-z])[ab](?![a-z])", lowered_instruction))


def _extract_tasks(raw: str) -> list[Any] | None:
    """LLM 応答から tasks のリストを取り出す。失敗時は None。"""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # ```json ... ``` のフェンスが付いた場合に備えて剥がす
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = [text]
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    bracket = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket:
        candidates.append(bracket.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            for key in ("tasks", "plan", "steps", "actions", "result"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
            # {"color": ..., "tray": ...} が単体で返ってきた場合
            if "color" in parsed or "tray" in parsed:
                return [parsed]
            if not parsed:
                return []
        if isinstance(parsed, list):
            return parsed
    return None
