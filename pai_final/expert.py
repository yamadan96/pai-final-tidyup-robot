"""スクリプト型エキスパート（減衰最小二乗 IK + 状態機械）。

6 自由度（位置 3 + 姿勢 3）の減衰最小二乗 IK で TCP を目標姿勢へ導き、
「上空へ移動 → 降下 → 把持 → 持ち上げ → トレイ上空へ移動 → 設置 → 解放 → 退避」
のフェーズを順に進める。各フェーズにタイムアウトを設け、無限ループを避ける。
"""

from __future__ import annotations

import mujoco
import numpy as np

from pai_final.env import N_ARM_JOINTS, TCP_OFFSET, TidyUpEnv

# フェーズごとの高さ [m]（TCP = 指パッド中心の目標 z）
Z_HOVER = 0.13  # ブロック上空
Z_GRASP = 0.026  # 把持高さ（ブロック重心 0.025 とほぼ一致）
Z_LIFT = 0.22  # 持ち上げ後の搬送高さ
Z_PLACE = 0.055  # トレイ上での解放高さ（指先がトレイ壁上端 0.036 に当たらない）
Z_RETREAT = 0.20  # 退避高さ

# 収束判定
POS_TOL = 0.008  # 位置誤差 [m]
POS_TOL_FINE = 0.005  # 降下・設置時の位置誤差 [m]
ROT_TOL = 0.10  # 姿勢誤差 [rad]

# フェーズ順と各フェーズのタイムアウト（制御ステップ数）
PHASE_ORDER = (
    "approach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "finished",
)
PHASE_TIMEOUT = {
    "approach": 120,
    "descend": 80,
    "grasp": 45,
    "lift": 80,
    "transport": 130,
    "place": 80,
    "release": 35,
    "retreat": 60,
}
GRASP_HOLD = 35  # グリッパを閉じてから持ち上げるまでの待機ステップ
RELEASE_HOLD = 25  # グリッパを開いてから退避するまでの待機ステップ

GRIP_OPEN = 1.0
GRIP_CLOSED = 0.0


def _down_frame(theta: float) -> np.ndarray:
    """グリッパを真下に向け、閉じ方向（hand-y）を角度 theta に合わせた回転行列。"""
    c, s = np.cos(theta), np.sin(theta)
    x_axis = np.array([-s, c, 0.0])
    y_axis = np.array([c, s, 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x_axis, y_axis, z_axis])


def _natural_yaw(xy: np.ndarray) -> float:
    """その位置を狙うときに腕にとって自然なグリッパ閉じ方向（半径方向に直交）。"""
    return float(np.arctan2(xy[1], xy[0]) - np.pi / 2)


def _align_yaw(block_yaw: float, natural: float) -> float:
    """正方形断面の 90deg 対称性を使い、自然な向きに最も近い等価ヨー角を選ぶ。"""
    k = np.round((block_yaw - natural) / (np.pi / 2))
    return float(block_yaw - k * (np.pi / 2))


class DampedLeastSquaresIK:
    """TCP の位置・姿勢誤差から関節角の増分を求める減衰最小二乗 IK。

    実機の関節角ではなく「指令値 q_cmd」の仮想運動学上で解く。
    位置制御アクチュエータの追従遅れに IK が引きずられず、滑らかな指令列になる。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        damping: float = 0.12,
        rot_weight: float = 0.35,
        max_dq: float = 0.09,
        pos_cap: float = 0.010,
        rot_cap: float = 0.07,
        integral_gain: float = 0.2,
        integral_limit: float = 0.04,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.damping = damping
        self.rot_weight = rot_weight
        self.max_dq = max_dq
        self.pos_cap = pos_cap
        self.rot_cap = rot_cap
        self.integral_gain = integral_gain
        self.integral_limit = integral_limit
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        # 位置制御アクチュエータの定常偏差（重力たわみ）を打ち消す積分補正
        self.bias = np.zeros(3)
        # 関節限界に余裕をもたせる（限界に貼り付くと IK が詰まる）
        self.q_lo = model.jnt_range[:N_ARM_JOINTS, 0] + 0.02
        self.q_hi = model.jnt_range[:N_ARM_JOINTS, 1] - 0.02

    def reset(self) -> None:
        self.bias[:] = 0.0

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """指令値 q に対する TCP 位置と hand 回転行列。"""
        self.data.qpos[:N_ARM_JOINTS] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mat = self.data.xmat[self.hand_id].reshape(3, 3)
        tcp = self.data.xpos[self.hand_id] + mat[:, 2] * TCP_OFFSET
        return tcp.copy(), mat.copy()

    def solve_step(
        self,
        q: np.ndarray,
        pos_des: np.ndarray,
        mat_des: np.ndarray,
        tcp_real: np.ndarray | None = None,
    ) -> np.ndarray:
        """1 制御ステップ分だけ目標へ近づけた関節角指令を返す。

        tcp_real を渡すと、実機 TCP と目標の差を積分して指令側の目標に足し込み、
        位置制御アクチュエータの定常偏差（重力たわみで 8mm 程度）を打ち消す。
        """
        if tcp_real is not None:
            self.bias += self.integral_gain * np.clip(pos_des - tcp_real, -0.02, 0.02)
            np.clip(self.bias, -self.integral_limit, self.integral_limit, out=self.bias)
        pos_des = pos_des + self.bias

        tcp, mat = self.forward(q)
        err_pos = np.clip(pos_des - tcp, -self.pos_cap, self.pos_cap)
        err_rot = np.clip(
            self._rot_error(mat, mat_des) * self.rot_weight, -self.rot_cap, self.rot_cap
        )
        mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, tcp, self.hand_id)
        jac = np.vstack(
            [self.jacp[:, :N_ARM_JOINTS], self.jacr[:, :N_ARM_JOINTS] * self.rot_weight]
        )
        err = np.concatenate([err_pos, err_rot])
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + (self.damping**2) * np.eye(6), err
        )
        dq = np.clip(dq, -self.max_dq, self.max_dq)
        return np.clip(q + dq, self.q_lo, self.q_hi)

    @staticmethod
    def _rot_error(mat: np.ndarray, mat_des: np.ndarray) -> np.ndarray:
        """現在姿勢から目標姿勢への回転誤差（軸角ベクトル）。"""
        quat_cur = np.zeros(4)
        quat_des = np.zeros(4)
        quat_err = np.zeros(4)
        vel = np.zeros(3)
        mujoco.mju_mat2Quat(quat_cur, mat.flatten())
        mujoco.mju_mat2Quat(quat_des, mat_des.flatten())
        mujoco.mju_negQuat(quat_cur, quat_cur)
        mujoco.mju_mulQuat(quat_err, quat_des, quat_cur)
        mujoco.mju_quat2Vel(vel, quat_err, 1.0)
        return vel


class ScriptedExpert:
    """フェーズ状態機械で片付け動作を生成するエキスパート。"""

    def __init__(self, env: TidyUpEnv, **ik_kwargs) -> None:
        self.env = env
        self.ik = DampedLeastSquaresIK(env.model, **ik_kwargs)
        self.reset()

    def reset(self) -> None:
        self.ik.reset()
        self.q_cmd = self.env.data.qpos[:N_ARM_JOINTS].copy()
        self.phase_idx = 0
        self.phase_steps = 0
        self.grip = GRIP_OPEN
        self.grasp_xy = np.zeros(2)
        self.grasp_yaw = 0.0

    # ------------------------------------------------------------------ phase

    @property
    def phase(self) -> str:
        return PHASE_ORDER[self.phase_idx]

    @property
    def done(self) -> bool:
        return self.phase == "finished"

    def _advance(self) -> None:
        self.phase_idx = min(self.phase_idx + 1, len(PHASE_ORDER) - 1)
        self.phase_steps = 0

    def _target(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """現フェーズの (目標位置, 目標姿勢, グリッパ指令, 位置許容誤差)。"""
        env = self.env
        phase = self.phase
        tray = env.tray_pos(env.target_tray)

        if phase == "approach":
            # ブロックは静止しているので毎ステップ最新の姿勢を見てよい
            block = env.block_pos(env.target_color)
            self.grasp_xy = block[:2].copy()
            self.grasp_yaw = _align_yaw(
                env.block_yaw(env.target_color), _natural_yaw(block[:2])
            )
            pos = np.array([block[0], block[1], Z_HOVER])
            return pos, _down_frame(self.grasp_yaw), GRIP_OPEN, POS_TOL

        if phase == "descend":
            pos = np.array([self.grasp_xy[0], self.grasp_xy[1], Z_GRASP])
            return pos, _down_frame(self.grasp_yaw), GRIP_OPEN, POS_TOL_FINE

        if phase == "grasp":
            pos = np.array([self.grasp_xy[0], self.grasp_xy[1], Z_GRASP])
            return pos, _down_frame(self.grasp_yaw), GRIP_CLOSED, POS_TOL_FINE

        if phase == "lift":
            pos = np.array([self.grasp_xy[0], self.grasp_xy[1], Z_LIFT])
            return pos, _down_frame(self.grasp_yaw), GRIP_CLOSED, POS_TOL

        tray_yaw = _natural_yaw(tray[:2])
        if phase == "transport":
            pos = np.array([tray[0], tray[1], Z_LIFT])
            return pos, _down_frame(tray_yaw), GRIP_CLOSED, POS_TOL
        if phase == "place":
            pos = np.array([tray[0], tray[1], Z_PLACE])
            return pos, _down_frame(tray_yaw), GRIP_CLOSED, POS_TOL_FINE
        if phase == "release":
            pos = np.array([tray[0], tray[1], Z_PLACE])
            return pos, _down_frame(tray_yaw), GRIP_OPEN, POS_TOL_FINE
        # retreat / finished
        pos = np.array([tray[0], tray[1], Z_RETREAT])
        return pos, _down_frame(tray_yaw), GRIP_OPEN, POS_TOL

    def _reached(self, pos_des: np.ndarray, mat_des: np.ndarray, tol: float) -> bool:
        """実機の TCP が目標に到達したか（指令値ではなく実測で判定する）。"""
        tcp, mat = self.env.tcp_pose()
        if np.linalg.norm(pos_des - tcp) > tol:
            return False
        rot_err = DampedLeastSquaresIK._rot_error(mat, mat_des)
        return bool(np.linalg.norm(rot_err) <= ROT_TOL)

    # -------------------------------------------------------------------- act

    def act(self, obs: dict | None = None) -> np.ndarray:
        """次に送る 8 次元 action を返す。"""
        pos_des, mat_des, grip, tol = self._target()
        phase = self.phase

        if phase != "finished":
            tcp_real, _ = self.env.tcp_pose()
            self.q_cmd = self.ik.solve_step(self.q_cmd, pos_des, mat_des, tcp_real)
            self.phase_steps += 1

            if phase == "grasp":
                advance = self.phase_steps >= GRASP_HOLD
            elif phase == "release":
                advance = self.phase_steps >= RELEASE_HOLD
            else:
                advance = self._reached(pos_des, mat_des, tol)
            if advance or self.phase_steps >= PHASE_TIMEOUT[phase]:
                self._advance()

        self.grip = grip
        return np.concatenate([self.q_cmd, [grip]])
