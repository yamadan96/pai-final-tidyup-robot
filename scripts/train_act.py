#!/usr/bin/env python
"""LeRobotDataset 上で ACT (Action Chunking Transformer) を学習する CLI。

使い方:
    ./.venv/bin/python scripts/train_act.py --steps 2000

lerobot 0.4.4 の API を使う。0.4.4 では正規化がポリシー本体から切り離され、
processor パイプラインに移っている点に注意:
    cfg = ACTConfig(input_features=..., output_features=..., device="mps")
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    batch = preprocessor(batch)        # 正規化 + デバイス転送
    loss, out = policy.forward(batch)  # 正規化済みの batch を渡す

device は Apple Silicon の MPS を既定とする（CUDA ではない）。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/lerobot/tidyup"), help="LeRobotDataset の root")
    parser.add_argument("--repo-id", type=str, default="local/tidyup", help="LeRobotDataset の repo_id")
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoints/act"), help="チェックポイント出力先")
    parser.add_argument("--steps", type=int, default=2000, help="学習ステップ数")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT の action chunk 長")
    parser.add_argument("--n-action-steps", type=int, default=100, help="1 回の推論で実行する action 数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学習率")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="ResNet バックボーンの学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="勾配クリップのノルム（0 で無効）")
    parser.add_argument("--dim-model", type=int, default=512, help="Transformer の隠れ次元")
    parser.add_argument("--dim-feedforward", type=int, default=3200, help="FFN の次元")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader のワーカ数（0 でメインプロセス）")
    parser.add_argument("--log-freq", type=int, default=50, help="ログを出す間隔 [step]")
    parser.add_argument("--save-freq", type=int, default=0, help="途中保存の間隔 [step]（0 で最後だけ）")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument(
        "--device", type=str, default=None, help="torch device（既定: mps があれば mps、無ければ cpu）"
    )
    parser.add_argument("--overwrite", action="store_true", help="既存の出力先を削除してから作る")
    return parser.parse_args(argv)


def select_device(requested: str | None) -> torch.device:
    """使用する device を決める。Apple Silicon では mps を既定にする。"""
    if requested is not None:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    if not args.dataset.exists():
        print(f"[ERROR] {args.dataset} が無い。先に build_lerobot_dataset.py を実行", file=sys.stderr)
        return 1

    device = select_device(args.device)
    print(f"device = {device}")
    print(f"  mps available = {torch.backends.mps.is_available()} / built = {torch.backends.mps.is_built()}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("[ERROR] mps を要求されたが利用できない", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)

    # ---- dataset の features から ACT の入出力定義を作る ----
    # 一度 delta_timestamps なしで開いてメタデータだけ読む
    meta_only = LeRobotDataset(repo_id=args.repo_id, root=args.dataset)
    policy_features = dataset_to_policy_features(meta_only.meta.features)
    output_features = {k: v for k, v in policy_features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in policy_features.items() if v.type is not FeatureType.ACTION}
    print("input_features:")
    for key, ft in input_features.items():
        print(f"  {key:34s} {ft.type.name:7s} {tuple(ft.shape)}")
    print("output_features:")
    for key, ft in output_features.items():
        print(f"  {key:34s} {ft.type.name:7s} {tuple(ft.shape)}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        dim_model=args.dim_model,
        dim_feedforward=args.dim_feedforward,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
        optimizer_weight_decay=args.weight_decay,
        device=str(device),
        push_to_hub=False,
    )

    # action chunk を取り出すための delta_timestamps を config から導出して開き直す
    delta_timestamps = resolve_delta_timestamps(cfg, meta_only.meta)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset, delta_timestamps=delta_timestamps)
    print(f"dataset: {dataset.num_episodes} episodes / {dataset.num_frames} frames / fps={dataset.fps}")
    print(f"delta_timestamps keys = {sorted(delta_timestamps)} (action chunk = {args.chunk_size})")

    policy = ACTPolicy(cfg)
    policy.to(device)
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"ACTPolicy params = {n_params / 1e6:.1f}M")

    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # MPS では pin_memory は効かない
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "loss", "l1_loss", "kld_loss", "grad_norm", "lr", "elapsed_s"])

    def save(step: int) -> Path:
        ckpt_dir = args.out / f"{step:06d}" / "pretrained_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(ckpt_dir)
        preprocessor.save_pretrained(ckpt_dir)
        postprocessor.save_pretrained(ckpt_dir)
        # 最新を指す symlink を張り替える（eval から参照しやすくするため）
        link = args.out / "last"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(ckpt_dir.parent.name, target_is_directory=True)
        return ckpt_dir

    # loss の推移を見る窓幅。短い run でも前半 / 後半を比較できるようにする
    window = min(50, max(1, args.steps // 10))
    print(f"\n--- 学習開始: {args.steps} steps, batch={args.batch_size}, loss窓={window} ---")
    dl_iter = iter(dataloader)
    first_loss: float | None = None
    last_loss: float | None = None
    recent: list[float] = []
    first_window: list[float] = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
        optimizer.step()

        loss_val = loss.item()
        last_loss = loss_val
        if first_loss is None:
            first_loss = loss_val
        if len(first_window) < window:
            first_window.append(loss_val)
        recent.append(loss_val)
        if len(recent) > window:
            recent.pop(0)

        elapsed = time.perf_counter() - started
        writer.writerow(
            [
                step,
                f"{loss_val:.6f}",
                f"{loss_dict.get('l1_loss', float('nan')):.6f}",
                f"{loss_dict.get('kld_loss', float('nan')):.6f}",
                f"{grad_norm.item():.4f}",
                f"{optimizer.param_groups[0]['lr']:.3e}",
                f"{elapsed:.2f}",
            ]
        )

        if step % args.log_freq == 0 or step == 1:
            log_file.flush()
            avg = sum(recent) / len(recent)
            sps = step / elapsed
            eta = (args.steps - step) / sps if sps > 0 else float("nan")
            print(
                f"step {step:6d}/{args.steps}  loss={loss_val:8.4f}  avg{window}={avg:8.4f}  "
                f"l1={loss_dict.get('l1_loss', float('nan')):7.4f}  "
                f"kld={loss_dict.get('kld_loss', float('nan')):7.4f}  "
                f"{sps:.2f} step/s  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
            )

        if args.save_freq > 0 and step % args.save_freq == 0 and step != args.steps:
            print(f"  -> checkpoint: {save(step)}")

    total = time.perf_counter() - started
    log_file.close()
    ckpt = save(args.steps)

    first_avg = sum(first_window) / len(first_window)
    last_avg = sum(recent) / len(recent)
    summary = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "params_M": round(n_params / 1e6, 2),
        "elapsed_s": round(total, 1),
        "step_per_s": round(args.steps / total, 3),
        "loss_first": round(first_loss, 4) if first_loss is not None else None,
        "loss_last": round(last_loss, 4) if last_loss is not None else None,
        "loss_window": window,
        f"loss_avg_first{window}": round(first_avg, 4),
        f"loss_avg_last{window}": round(last_avg, 4),
        "checkpoint": str(ckpt),
        "train_log": str(log_path),
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n--- 学習結果 ---")
    for key, val in summary.items():
        print(f"  {key:18s} {val}")
    trend = "下降" if last_avg < first_avg else "下降せず"
    print(f"loss: 先頭{window}step平均 {first_avg:.4f} -> 末尾{window}step平均 {last_avg:.4f} ({trend})")
    print(f"所要時間: {total:.1f}s ({args.steps / total:.2f} step/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
