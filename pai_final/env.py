"""色付きブロックをトレイへ片付けるタスクの MuJoCo 環境ラッパー。

シーンは Franka Emika Panda + 赤/青/緑ブロック3個 + トレイ A/B で構成される。
エピソードごとに「どの色のブロックを、どちらのトレイに入れるか」をゴールとして与え、
その条件を観測 (state) の末尾 5 次元に one-hot で埋め込む（ゴール条件付きの模倣学習用）。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pai_final.goal_cue import apply_goal_cue

COLORS: tuple[str, ...] = ("red", "blue", "green")
TRAYS: tuple[str, ...] = ("A", "B")

SCENE_XML = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "franka_emika_panda"
    / "tidyup_scene.xml"
)

# hand 原点から指パッド中心までの距離（hand-z 方向、モデルから実測）。
# left/right_finger は hand+0.0584 に置かれ、パッド中心はその先 0.0445。
TCP_OFFSET = 0.103

# ブロックをランダム配置する作業領域（テーブル上、ロボットの到達範囲内）
WORK_X = (0.38, 0.58)
WORK_Y = (-0.25, 0.25)
MIN_BLOCK_SEP = 0.08  # ブロック同士の最小間隔 [m]
BLOCK_HALF_Z = 0.025  # ブロック半径（高さ方向）[m]
BLOCK_YAW_RANGE = (-np.pi / 4, np.pi / 4)  # 正方形断面なので ±45deg で全姿勢を網羅

N_ARM_JOINTS = 7
GRIPPER_CTRL_MAX = 255.0  # アクチュエータ 7 の ctrlrange 上限（255 が開、0 が閉）
GRIPPER_MAX_WIDTH = 0.08  # finger_joint1 + finger_joint2 の最大値 [m]

# 成功判定のしきい値
SUCCESS_XY_RADIUS = 0.07  # トレイ中心からの水平距離 [m]
SUCCESS_Z_MAX = 0.12  # ブロック重心高さの上限 [m]
SUCCESS_SPEED = 0.03  # 静止判定の並進速度 [m/s]

STATE_DIM = 13
ACTION_DIM = 8

# 観測カメラのプリセット。キーが観測 dict / データセットで使う論理名、
# 値が tidyup_scene.xml 内の実際のカメラ名。
#   "wide"  : 旧設定。テーブル全体が入るがブロックが数 px にしかならない
#   "tight" : 作業領域を画面いっぱいに入れ、ブロックの見かけを約 3 倍にした設定
CAMERA_PRESETS: dict[str, dict[str, str]] = {
    "wide": {"top": "top", "front": "front"},
    "tight": {"top": "top_tight", "front": "front_tight"},
}
# 動画は常にこのカメラ（全体が入る広角）で描画し、プリセットに依存させない
VIDEO_CAMERA = "front"


class TidyUpEnv:
    """片付けタスクの環境ラッパー。

    action は 8 次元:
        action[0:7] : 腕 7 関節の目標角 [rad]（位置制御アクチュエータへの指令）
        action[7]   : グリッパ指令を [0, 1] に正規化した値（1 = 開, 0 = 閉）
                      内部で 0〜255 の ctrlrange にスケールして渡す。
                      関節角 [rad] と桁を揃えるため正規化している（模倣学習での回帰を容易にする）。

    obs は dict:
        obs["state"]  : float32 (13,) = 関節角 7 + グリッパ開度 1（正規化）+ 色 one-hot 3 + トレイ one-hot 2
        obs["images"] : {"top": (H,W,3) uint8, "front": (H,W,3) uint8}
    """

    def __init__(
        self,
        seed: int | None = None,
        image_size: int = 96,
        control_decimation: int = 10,
        max_steps: int = 600,
        cameras: tuple[str, ...] = ("top", "front"),
        render_images: bool = True,
        camera_preset: str = "wide",
        goal_cue: bool = False,
        goal_repeat: int = 1,
    ) -> None:
        if camera_preset not in CAMERA_PRESETS:
            raise ValueError(
                f"camera_preset must be one of {sorted(CAMERA_PRESETS)}, got {camera_preset}"
            )
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.image_size = image_size
        self.control_decimation = control_decimation
        self.max_steps = max_steps
        self.cameras = tuple(cameras)
        self.render_images = render_images
        self.camera_preset = camera_preset
        # ゴールを画像にも描き込むか / 状態ベクタで何回繰り返すか
        self.goal_cue = goal_cue
        self.goal_repeat = max(1, int(goal_repeat))
        # 論理名 -> XML 上のカメラ名。プリセットに無い名前はそのまま使う
        mapping = CAMERA_PRESETS[camera_preset]
        self.camera_xml = {name: mapping.get(name, name) for name in self.cameras}

        self.control_dt = self.model.opt.timestep * control_decimation

        # 解像度ごとに Renderer をキャッシュする（生成コストが高いため）
        self._renderers: dict[int, mujoco.Renderer] = {}

        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.block_body = {
            c: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"block_{c}")
            for c in COLORS
        }
        self.block_geom = {
            c: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"block_{c}")
            for c in COLORS
        }
        # freejoint の qpos 開始インデックス（pos 3 + quat 4）
        self.block_qadr = {
            c: self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block_{c}_free"
                )
            ]
            for c in COLORS
        }
        self.block_dofadr = {
            c: self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block_{c}_free"
                )
            ]
            for c in COLORS
        }
        self.tray_site = {
            "A": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tray_a_center"),
            "B": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tray_b_center"),
        }
        self.finger_geoms = self._collect_finger_geoms()

        self.arm_ctrl_lo = self.model.actuator_ctrlrange[:N_ARM_JOINTS, 0].copy()
        self.arm_ctrl_hi = self.model.actuator_ctrlrange[:N_ARM_JOINTS, 1].copy()

        self.target_color = COLORS[0]
        self.target_tray = TRAYS[0]
        self.steps = 0

    # ------------------------------------------------------------------ setup

    def _collect_finger_geoms(self) -> set[int]:
        """指 2 本に属する geom id を集める（把持判定の接触チェックに使う）。"""
        finger_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_finger", "right_finger")
        }
        return {
            gi
            for gi in range(self.model.ngeom)
            if self.model.geom_bodyid[gi] in finger_bodies
        }

    def get_renderer(self, size: int | None = None) -> mujoco.Renderer:
        size = self.image_size if size is None else int(size)
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, size, size)
        return self._renderers[size]

    @property
    def renderer(self) -> mujoco.Renderer:
        return self.get_renderer(self.image_size)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    # ------------------------------------------------------------------ reset

    def _sample_block_positions(self) -> np.ndarray:
        """3 個のブロック位置を、互いに MIN_BLOCK_SEP 以上離してサンプリングする。"""
        for _ in range(500):
            xy = np.stack(
                [
                    self.rng.uniform(WORK_X[0], WORK_X[1], size=3),
                    self.rng.uniform(WORK_Y[0], WORK_Y[1], size=3),
                ],
                axis=1,
            )
            dists = [
                np.linalg.norm(xy[i] - xy[j]) for i in range(3) for j in range(i + 1, 3)
            ]
            if min(dists) >= MIN_BLOCK_SEP:
                return xy
        raise RuntimeError("ブロック配置のサンプリングに失敗した")

    def reset(self, target_color: str, target_tray: str) -> dict:
        """ゴール（対象色・対象トレイ）を指定してエピソードを初期化する。"""
        if target_color not in COLORS:
            raise ValueError(f"target_color must be one of {COLORS}, got {target_color}")
        if target_tray not in TRAYS:
            raise ValueError(f"target_tray must be one of {TRAYS}, got {target_tray}")

        self.target_color = target_color
        self.target_tray = target_tray
        self.steps = 0

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.nkey - 1)

        xy = self._sample_block_positions()
        for i, color in enumerate(COLORS):
            adr = self.block_qadr[color]
            yaw = self.rng.uniform(*BLOCK_YAW_RANGE)
            self.data.qpos[adr : adr + 3] = [xy[i, 0], xy[i, 1], BLOCK_HALF_Z]
            self.data.qpos[adr + 3 : adr + 7] = [
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ]
            dof = self.block_dofadr[color]
            self.data.qvel[dof : dof + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def set_goal(self, target_color: str, target_tray: str) -> dict:
        """シーンを初期化せずにゴールだけ差し替える（連続タスク実行用）。

        1 回の指示から複数のサブタスクが出てくる場合、途中でブロックを並べ直すと
        連続した片付けに見えないため、物理状態をそのまま引き継いでゴールだけ更新する。
        """
        if target_color not in COLORS:
            raise ValueError(f"target_color must be one of {COLORS}, got {target_color}")
        if target_tray not in TRAYS:
            raise ValueError(f"target_tray must be one of {TRAYS}, got {target_tray}")

        self.target_color = target_color
        self.target_tray = target_tray
        self.steps = 0
        return self._observation()

    # ------------------------------------------------------------------- step

    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]:
        """8 次元 action を適用して control_decimation 回だけ物理を進める。"""
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        self.data.ctrl[:N_ARM_JOINTS] = np.clip(
            action[:N_ARM_JOINTS], self.arm_ctrl_lo, self.arm_ctrl_hi
        )
        self.data.ctrl[N_ARM_JOINTS] = float(
            np.clip(action[N_ARM_JOINTS], 0.0, 1.0) * GRIPPER_CTRL_MAX
        )

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs = self._observation()
        success = self.success()
        done = bool(success or self.steps >= self.max_steps)
        info = {
            "success": success,
            "steps": self.steps,
            "timeout": bool(self.steps >= self.max_steps and not success),
            "target_block_pos": self.block_pos(self.target_color).copy(),
            "tray_pos": self.tray_pos(self.target_tray).copy(),
        }
        return obs, done, info

    # ------------------------------------------------------------ observation

    def goal_vector(self) -> np.ndarray:
        """ゴール条件 5 次元（色 one-hot 3 + トレイ one-hot 2）。"""
        goal = np.zeros(5, dtype=np.float32)
        goal[COLORS.index(self.target_color)] = 1.0
        goal[3 + TRAYS.index(self.target_tray)] = 1.0
        return goal

    def gripper_width(self) -> float:
        """グリッパの開き幅 [m]（finger_joint1 + finger_joint2）。"""
        return float(self.data.qpos[7] + self.data.qpos[8])

    def state(self) -> np.ndarray:
        """state 13 次元 = 関節角 7 + グリッパ開度（正規化）1 + ゴール 5。

        グリッパ開度は GRIPPER_MAX_WIDTH で割った値。MuJoCo の関節限界はソフト
        拘束なので、全開時に 1.0 を 1e-3 程度超えることがある（クリップしていない）。
        """
        return np.concatenate(
            [
                self.data.qpos[:N_ARM_JOINTS].astype(np.float32),
                np.array(
                    [self.gripper_width() / GRIPPER_MAX_WIDTH], dtype=np.float32
                ),
                np.tile(self.goal_vector(), self.goal_repeat),
            ]
        ).astype(np.float32)

    def render(self, camera: str = VIDEO_CAMERA, size: int | None = None) -> np.ndarray:
        """XML 上のカメラ名を指定して RGB 画像を返す（論理名の変換はしない）。

        動画用の呼び出し元が camera_preset に引きずられないよう、ここは常に
        XML のカメラ名をそのまま受け取る。観測画像は _observation() が作る。
        """
        renderer = self.get_renderer(size)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def _observation(self) -> dict:
        images: dict[str, np.ndarray] = {}
        if self.render_images:
            for cam in self.cameras:
                self.renderer.update_scene(self.data, camera=self.camera_xml[cam])
                img = self.renderer.render().copy()
                if self.goal_cue:
                    img = apply_goal_cue(img, self.target_color, self.target_tray)
                images[cam] = img
        return {"state": self.state(), "images": images}

    # --------------------------------------------------------------- geometry

    def block_pos(self, color: str) -> np.ndarray:
        return self.data.xpos[self.block_body[color]]

    def block_yaw(self, color: str) -> float:
        """ブロックの z 軸まわりの回転角 [rad]。"""
        mat = self.data.xmat[self.block_body[color]].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def block_speed(self, color: str) -> float:
        dof = self.block_dofadr[color]
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def tray_pos(self, tray: str) -> np.ndarray:
        """トレイ中心（site の world 座標、z は床から 0.05）。"""
        return self.data.site_xpos[self.tray_site[tray]]

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP（指パッド中心）の位置と hand の回転行列。"""
        mat = self.data.xmat[self.hand_id].reshape(3, 3)
        tcp = self.data.xpos[self.hand_id] + mat[:, 2] * TCP_OFFSET
        return tcp.copy(), mat.copy()

    def is_grasped(self, color: str) -> bool:
        """対象ブロックが指と接触しているか（把持中かどうか）。"""
        geom = self.block_geom[color]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == geom and g2 in self.finger_geoms) or (
                g2 == geom and g1 in self.finger_geoms
            ):
                return True
        return False

    # ---------------------------------------------------------------- success

    def success(self) -> bool:
        """対象ブロックがトレイ内に静止して置かれているか。

        - トレイ中心からの水平距離 < SUCCESS_XY_RADIUS
        - 重心高さ < SUCCESS_Z_MAX
        - 並進速度 < SUCCESS_SPEED（静止）
        - 指と接触していない（グリッパで保持したままを成功としない）
        """
        pos = self.block_pos(self.target_color)
        tray = self.tray_pos(self.target_tray)
        if np.linalg.norm(pos[:2] - tray[:2]) >= SUCCESS_XY_RADIUS:
            return False
        if pos[2] >= SUCCESS_Z_MAX:
            return False
        if self.block_speed(self.target_color) >= SUCCESS_SPEED:
            return False
        return not self.is_grasped(self.target_color)
