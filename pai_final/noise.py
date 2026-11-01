"""デモ収集時に指令関節角へ加えるガウスノイズ（復帰動作をラベルに含めるため）。

完璧な軌道だけを学習すると、少しずれた状態からどう戻せばよいかをポリシーが
知らないまま終わる（covariate shift）。収集中に指令を揺らして実際に機体をずらし、
そこからエキスパートが立て直す動作をラベルとして記録することで、
復帰動作をデモに含める。

mode:
    "none" : ノイズなし（従来の収集と同じ）
    "ctrl" : 毎ステップ独立なノイズを「実際に実行する指令」にだけ加える。
             エキスパート内部の指令列は汚れないため、フェーズの収束判定が
             壊れにくい。観測だけが公称軌道からずれる。
    "cmd"  : ノイズをエキスパート内部の指令 q_cmd にも積み上げる（DART 風）。
             ずれが持続するので復帰動作が強く現れるが、過大にすると
             フェーズの位置許容誤差に収束できず成功率が落ちる。

記録するラベルは常に「ノイズを載せる前のエキスパート指令」にする。
ずれた状態で専門家が出す正しい指令、という対応をポリシーに学習させたいため。
"""

from __future__ import annotations

import numpy as np

from pai_final.env import N_ARM_JOINTS

NOISE_MODES: tuple[str, ...] = ("none", "ctrl", "cmd")

# 既定でノイズを加えないフェーズ。
#   grasp / release : グリッパが開閉している最中に腕を揺らすとブロックを弾いてしまい、
#                     復帰動作の学習にならずに失敗が増えるだけになる。
#   finished        : 台本が終わった後はエキスパートが IK を解かないので、
#                     mode="cmd" だと補正の入らないランダムウォークになってしまう。
DEFAULT_SKIP_PHASES: tuple[str, ...] = ("grasp", "release", "finished")


class CommandNoise:
    """エキスパートの 8 次元 action にノイズを載せて返す。"""

    def __init__(
        self,
        std: float = 0.0,
        mode: str = "ctrl",
        seed: int | None = None,
        skip_phases: tuple[str, ...] = DEFAULT_SKIP_PHASES,
        clip_sigma: float = 3.0,
        joint_lo: np.ndarray | None = None,
        joint_hi: np.ndarray | None = None,
    ) -> None:
        if mode not in NOISE_MODES:
            raise ValueError(f"mode must be one of {NOISE_MODES}, got {mode}")
        self.std = float(std)
        self.mode = "none" if self.std <= 0.0 else mode
        self.rng = np.random.default_rng(seed)
        self.skip_phases = tuple(skip_phases)
        self.clip_sigma = float(clip_sigma)
        self.joint_lo = joint_lo
        self.joint_hi = joint_hi
        self.reset_stats()

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def reset_stats(self) -> None:
        self._n_steps = 0
        self._n_perturbed = 0
        self._sq_sum = 0.0
        self._abs_max = 0.0

    def perturb(self, action: np.ndarray, phase: str, expert=None) -> np.ndarray:
        """実行用の action を返す。mode="cmd" のときは expert.q_cmd も書き換える。

        action はラベルとして使うので破壊しない（コピーを返す）。
        """
        self._n_steps += 1
        if not self.enabled or phase in self.skip_phases:
            return np.asarray(action, dtype=np.float64).copy()

        limit = self.clip_sigma * self.std
        noise = np.clip(
            self.rng.normal(0.0, self.std, size=N_ARM_JOINTS), -limit, limit
        )
        out = np.asarray(action, dtype=np.float64).copy()
        out[:N_ARM_JOINTS] = out[:N_ARM_JOINTS] + noise
        if self.joint_lo is not None and self.joint_hi is not None:
            out[:N_ARM_JOINTS] = np.clip(
                out[:N_ARM_JOINTS], self.joint_lo, self.joint_hi
            )

        if self.mode == "cmd" and expert is not None:
            # ずれを引き継がせる: 次ステップの IK はこのずれた指令から解き直す
            expert.q_cmd = out[:N_ARM_JOINTS].copy()

        self._n_perturbed += 1
        applied = out[:N_ARM_JOINTS] - np.asarray(action[:N_ARM_JOINTS])
        self._sq_sum += float(np.square(applied).sum())
        self._abs_max = max(self._abs_max, float(np.abs(applied).max()))
        return out

    def stats(self) -> dict:
        """報告用の統計（実際に適用したノイズの RMS など）。"""
        n_values = self._n_perturbed * N_ARM_JOINTS
        return {
            "mode": self.mode,
            "std": self.std,
            "skip_phases": list(self.skip_phases),
            "steps": self._n_steps,
            "steps_perturbed": self._n_perturbed,
            "perturbed_ratio": (
                round(self._n_perturbed / self._n_steps, 4) if self._n_steps else 0.0
            ),
            "applied_rms_rad": (
                round(float(np.sqrt(self._sq_sum / n_values)), 5) if n_values else 0.0
            ),
            "applied_abs_max_rad": round(self._abs_max, 5),
        }
