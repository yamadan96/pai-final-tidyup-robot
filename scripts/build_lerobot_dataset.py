#!/usr/bin/env python
"""collect_demos.py が出力した npz 群を LeRobotDataset に変換する CLI。

使い方:
    ./.venv/bin/python scripts/build_lerobot_dataset.py \
        --src outputs/dataset --out outputs/lerobot/tidyup --overwrite

lerobot 0.4.4 の API を使う:
    LeRobotDataset.create(repo_id, fps, features, root=..., use_videos=...)
    -> add_frame({...,"task": str}) を 1 フレームずつ
    -> save_episode()
    -> finalize()

画像は use_videos=False（dtype="image"、PNG で保存）で格納する。
96x96 と小さいので容量は問題にならず、macOS で ffmpeg / torchcodec の
動画エンコード・デコードに依存しないため再現性が高い。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai_final.env import ACTION_DIM, COLORS, STATE_DIM, TRAYS  # noqa: E402
from pai_final.goal_cue import apply_goal_cue  # noqa: E402

CAMERAS: tuple[str, ...] = ("top", "front")

# state の内訳（LeRobotDataset の features.names に入れる）
def state_names(goal_repeat: int = 1) -> list[str]:
    """goal_repeat 回だけゴール one-hot を繰り返した state の名前一覧。"""
    names = [f"arm_joint_{i}" for i in range(1, 8)] + ["gripper_width"]
    for k in range(goal_repeat):
        suffix = "" if goal_repeat == 1 else f"_{k}"
        names += [f"goal_color_{c}{suffix}" for c in COLORS]
        names += [f"goal_tray_{t}{suffix}" for t in TRAYS]
    return names


STATE_NAMES: list[str] = state_names()
# action 8 次元の内訳
ACTION_NAMES: list[str] = [f"arm_joint_{i}_target" for i in range(1, 8)] + ["gripper_cmd"]

TRAY_WORD = {"A": "tray A", "B": "tray B"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("outputs/dataset"), help="npz のあるディレクトリ")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/lerobot/tidyup"), help="LeRobotDataset の出力先"
    )
    parser.add_argument("--repo-id", type=str, default="local/tidyup", help="LeRobotDataset の repo_id")
    parser.add_argument("--image-size", type=int, default=96, help="観測画像の一辺 [px]")
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 本だけ変換する（デバッグ用）")
    parser.add_argument("--overwrite", action="store_true", help="既存の出力先を削除してから作る")
    parser.add_argument(
        "--use-videos", action="store_true", help="画像を mp4 にエンコードする（既定は PNG）"
    )
    parser.add_argument(
        "--goal-cue", action="store_true",
        help="観測画像の左上にゴール（目標色・目標トレイ）のマーカーを描き込む",
    )
    parser.add_argument(
        "--goal-repeat", type=int, default=1,
        help="state ベクタでゴール one-hot を繰り返す回数（既定 1）",
    )
    return parser.parse_args(argv)


def task_string(goal: np.ndarray) -> str:
    """ゴール 5 次元（色 one-hot 3 + トレイ one-hot 2）から自然言語のタスク文を作る。"""
    color = COLORS[int(np.argmax(goal[:3]))]
    tray = TRAYS[int(np.argmax(goal[3:5]))]
    return f"put the {color} block into {TRAY_WORD[tray]}"


def build_features(image_size: int, use_videos: bool, goal_repeat: int = 1) -> dict:
    """LeRobotDataset.create に渡す features 定義。

    画像の names は末尾を "channels" にしておく必要がある。
    dataset_to_policy_features() がこの名前を見て (h,w,c) -> (c,h,w) に
    並べ替えるため、別の名前だと ACT 側の入力 shape が崩れる。
    """
    image_dtype = "video" if use_videos else "image"
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM + 5 * (goal_repeat - 1),),
            "names": state_names(goal_repeat),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ACTION_NAMES,
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": image_dtype,
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class ChannelStatsAccumulator:
    """画像のチャンネル別 mean / std を float64 で厳密に累積する。

    lerobot 0.4.4 の compute_stats には画像統計量のバグがある:
    sample_images() が uint8 で画像を読み、RunningQuantileStats.update() の中で
    `np.mean(batch**2)` を計算するため、uint8 の二乗がラップアラウンドして
    分散が壊れる（std がほぼ 0 になる）。
    std ≈ 0 のまま MEAN_STD 正規化すると画像が ±300 程度に発散し、
    ResNet バックボーンに入る入力が壊れるため、ここで正しい値を計算して
    meta/stats.json を上書きする。
    """

    def __init__(self) -> None:
        self.count = 0  # チャンネルあたりの画素数
        self.total = np.zeros(3, dtype=np.float64)
        self.total_sq = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update(self, images_u8: np.ndarray) -> None:
        """(T, H, W, 3) の uint8 画像列を受け取り、[0,1] スケールで累積する。"""
        flat = images_u8.reshape(-1, 3).astype(np.float64) / 255.0
        self.count += flat.shape[0]
        self.total += flat.sum(axis=0)
        self.total_sq += np.square(flat).sum(axis=0)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))

    def result(self) -> dict[str, np.ndarray]:
        """stats.json と同じ (3, 1, 1) 形状で mean / std / min / max を返す。"""
        mean = self.total / self.count
        var = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.reshape(3, 1, 1),
            "std": np.sqrt(var).reshape(3, 1, 1),
            "min": self.minimum.reshape(3, 1, 1),
            "max": self.maximum.reshape(3, 1, 1),
        }


def fix_image_stats(root: Path, accumulators: dict[str, ChannelStatsAccumulator]) -> None:
    """meta/stats.json の画像統計量を、厳密に計算した値で上書きする。"""
    from lerobot.datasets.utils import load_stats, write_stats

    stats = load_stats(root)
    if stats is None:
        raise RuntimeError(f"{root} に meta/stats.json が無い")

    print("\n--- 画像統計量の補正 (lerobot 0.4.4 の uint8 オーバーフローバグ対策) ---")
    for cam, acc in accumulators.items():
        key = f"observation.images.{cam}"
        fixed = acc.result()
        before_std = np.asarray(stats[key]["std"]).reshape(-1)
        for stat_name, value in fixed.items():
            stats[key][stat_name] = value
        after = fixed["std"].reshape(-1)
        print(f"  {key}")
        print(f"    std  {np.round(before_std, 6)} -> {np.round(after, 6)}")
        print(f"    mean {np.round(fixed['mean'].reshape(-1), 6)}  ({acc.count} px/channel)")
        if float(after.min()) < 1e-3:
            print(f"    [WARN] 補正後も std が極端に小さい: {after.min():.2e}")

    write_stats(stats, root)
    print(f"  meta/stats.json を更新した")


def load_episode(path: Path) -> dict:
    """npz を読み、形状を検証して必要な配列だけ返す。"""
    with np.load(path, allow_pickle=False) as npz:
        states = npz["states"].astype(np.float32)
        actions = npz["actions"].astype(np.float32)
        images = {cam: npz[f"images_{cam}"] for cam in CAMERAS}
        goal = npz["goal"].astype(np.float32)
        control_dt = float(npz["control_dt"])
        success = bool(npz["success"])

    n = len(states)
    if len(actions) != n:
        raise ValueError(f"{path.name}: states {n} と actions {len(actions)} の長さが違う")
    for cam, arr in images.items():
        if len(arr) != n:
            raise ValueError(f"{path.name}: states {n} と images_{cam} {len(arr)} の長さが違う")

    return {
        "states": states,
        "actions": actions,
        "images": images,
        "goal": goal,
        "control_dt": control_dt,
        "success": success,
        "length": n,
    }


def augment_episode(ep: dict, *, goal_cue: bool, goal_repeat: int) -> dict:
    """ゴール条件づけを強める加工をエピソード単位で適用する。

    - goal_cue: 観測画像の左上に目標色・目標トレイのマーカーを描き込む
    - goal_repeat: state 末尾のゴール one-hot 5 次元を goal_repeat 回繰り返す

    推論側（TidyUpEnv）と必ず同じ関数・同じ規則を使うこと。
    """
    if not goal_cue and goal_repeat <= 1:
        return ep

    color = COLORS[int(np.argmax(ep["goal"][:3]))]
    tray = TRAYS[int(np.argmax(ep["goal"][3:5]))]

    images = ep["images"]
    if goal_cue:
        images = {
            cam: np.stack([apply_goal_cue(frame, color, tray) for frame in arr])
            for cam, arr in ep["images"].items()
        }

    states = ep["states"]
    if goal_repeat > 1:
        base, goal = states[:, :8], states[:, 8:13]
        states = np.concatenate([base, np.tile(goal, (1, goal_repeat))], axis=1).astype(
            np.float32
        )

    return {**ep, "images": images, "states": states}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # LeRobotDataset は重いので、引数チェックが通ってから import する
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = sorted(args.src.glob("*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"[ERROR] {args.src} に npz が無い", file=sys.stderr)
        return 1

    # 収集プロセスが書き込み中の npz を掴まないよう、先に全件読んで検証する。
    # 壊れている／書き込み途中のファイルはスキップし、理由を表示する。
    episodes: list[tuple[Path, dict]] = []
    skipped: list[tuple[str, str]] = []
    for path in paths:
        try:
            episodes.append((path, load_episode(path)))
        except Exception as exc:  # 書き込み途中 / 破損
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))

    if not episodes:
        print("[ERROR] 読み込めた npz が 0 件", file=sys.stderr)
        for name, reason in skipped:
            print(f"  skip {name}: {reason}", file=sys.stderr)
        return 1

    control_dts = {round(ep["control_dt"], 6) for _, ep in episodes}
    if len(control_dts) != 1:
        raise ValueError(f"control_dt がエピソード間で揃っていない: {sorted(control_dts)}")
    control_dt = control_dts.pop()
    fps = int(round(1.0 / control_dt))

    image_shapes = {ep["images"][cam].shape[1:] for _, ep in episodes for cam in CAMERAS}
    if len(image_shapes) != 1:
        raise ValueError(f"画像 shape が揃っていない: {image_shapes}")
    img_h, img_w, _ = image_shapes.pop()
    if img_h != img_w or img_h != args.image_size:
        print(f"[WARN] --image-size {args.image_size} だが実データは {img_h}x{img_w}。実データに合わせる")

    if args.out.exists():
        if not args.overwrite:
            print(f"[ERROR] {args.out} が既に存在する。--overwrite を付けるか別の --out を指定", file=sys.stderr)
            return 1
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"npz {len(paths)} 件中 {len(episodes)} 件を変換 (skip {len(skipped)} 件)")
    for name, reason in skipped:
        print(f"  skip {name}: {reason}")
    print(f"fps={fps} (control_dt={control_dt}), image={img_h}x{img_w}, use_videos={args.use_videos}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=build_features(img_h, args.use_videos, args.goal_repeat),
        root=args.out,
        robot_type="franka_panda",
        use_videos=args.use_videos,
        image_writer_threads=4,
    )

    started = time.time()
    total_frames = 0
    task_counts: dict[str, int] = {}
    accumulators = {cam: ChannelStatsAccumulator() for cam in CAMERAS}
    for i, (path, ep) in enumerate(episodes):
        task = task_string(ep["goal"])
        task_counts[task] = task_counts.get(task, 0) + 1
        ep = augment_episode(ep, goal_cue=args.goal_cue, goal_repeat=args.goal_repeat)
        for cam in CAMERAS:
            accumulators[cam].update(ep["images"][cam])
        for t in range(ep["length"]):
            frame = {
                "observation.state": ep["states"][t],
                "action": ep["actions"][t],
                "task": task,
            }
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = ep["images"][cam][t]
            dataset.add_frame(frame)
        dataset.save_episode()
        total_frames += ep["length"]
        if (i + 1) % 20 == 0 or i + 1 == len(episodes):
            elapsed = time.time() - started
            print(f"  [{i + 1}/{len(episodes)}] {path.name} frames={total_frames} {elapsed:.1f}s")

    dataset.finalize()
    print(f"変換完了: {len(episodes)} episodes / {total_frames} frames / {time.time() - started:.1f}s")
    fix_image_stats(args.out, accumulators)
    print(f"タスク内訳: {len(task_counts)} 種")
    for task, count in sorted(task_counts.items()):
        print(f"  {count:4d}  {task}")

    # ---- 読み戻して検証する（受け入れ条件 1）----
    print("\n--- 読み戻し検証 ---")
    reloaded = LeRobotDataset(repo_id=args.repo_id, root=args.out)
    print(f"num_episodes = {reloaded.num_episodes}")
    print(f"num_frames   = {reloaded.num_frames}")
    print(f"fps          = {reloaded.fps}")
    print(f"tasks        = {reloaded.meta.total_tasks}")
    item = reloaded[0]
    print("sample[0] の feature shape:")
    for key in sorted(item):
        val = item[key]
        shape = tuple(val.shape) if hasattr(val, "shape") else type(val).__name__
        print(f"  {key:34s} {shape}")

    print("stats (補正後):")
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        mean = np.asarray(reloaded.meta.stats[key]["mean"]).reshape(-1)
        std = np.asarray(reloaded.meta.stats[key]["std"]).reshape(-1)
        # MEAN_STD 正規化後に取りうる最大の絶対値（画素が 0 または 1 のとき）
        worst = float(np.max(np.abs((np.array([0.0, 1.0])[:, None] - mean) / std)))
        print(f"  {key:34s} mean={np.round(mean, 4)} std={np.round(std, 4)} 正規化後|最大|={worst:.1f}")

    if reloaded.num_episodes != len(episodes) or reloaded.num_frames != total_frames:
        print("[ERROR] 書き出した件数と読み戻した件数が一致しない", file=sys.stderr)
        return 1
    print("OK: 書き出し件数と読み戻し件数が一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
