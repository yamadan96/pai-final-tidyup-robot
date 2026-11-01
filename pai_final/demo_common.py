"""デモ動画のスクリプトが共通で使う小道具（日本語ラベル、ACT ポリシーの読み込み）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAMERAS: tuple[str, ...] = ("top", "front")

COLOR_JA = {"red": "赤", "blue": "青", "green": "緑"}
TRAY_JA = {"A": "奥のトレイ", "B": "手前のトレイ"}
POLICY_JA = {
    "expert": "規範方策（減衰最小二乗 IK + 状態機械）",
    "act": "模倣学習ポリシー（ACT, 51.6M パラメータ）",
}

# パネルの文字色
C_PENDING = (150, 158, 172)
C_ACTIVE = (255, 214, 102)
C_DONE = (120, 200, 140)
C_PLAIN = (215, 222, 235)
C_INFO = (150, 200, 255)


def plan_lines(tasks: list[dict], current: int | None) -> list[tuple[str, tuple[int, int, int]]]:
    """計画をパネル用の行に変換する（実行中の行を目立たせる）。"""
    lines = []
    for i, t in enumerate(tasks):
        body = f"{COLOR_JA[t['color']]}ブロック → {TRAY_JA[t['tray']]}"
        if current is None:
            lines.append((f"{i + 1}) {body}", C_PLAIN))
        elif i < current:
            lines.append((f"✓ {body}", C_DONE))
        elif i == current:
            lines.append((f"▶ {body}", C_ACTIVE))
        else:
            lines.append((f"{i + 1}) {body}", C_PENDING))
    return lines


def load_act_policy(
    checkpoint: Path,
    temporal_ensemble: float | None = 0.01,
):
    """ACT のチェックポイントを読み、(reset_fn, act_fn, 説明文) を返す。"""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(checkpoint)
    if temporal_ensemble is not None:
        # 時間アンサンブル: 毎ステップ推論し、過去の予測を指数重みで混ぜる
        policy.config.temporal_ensemble_coeff = temporal_ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble, policy.config.chunk_size
        )
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    def act_fn(obs: dict) -> np.ndarray:
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["state"], dtype=np.float32)
            )
        }
        for cam in CAMERAS:
            img = np.asarray(obs["images"][cam], dtype=np.uint8)
            batch[f"observation.images.{cam}"] = (
                torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0
            )
        action = post(policy.select_action(pre(batch)))
        return action.squeeze(0).to(torch.float32).cpu().numpy()

    desc = (
        f"chunk={policy.config.chunk_size} "
        f"n_action_steps={policy.config.n_action_steps} device={device}"
    )
    return policy.reset, act_fn, desc
