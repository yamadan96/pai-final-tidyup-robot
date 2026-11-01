#!/usr/bin/env python
"""学習済み ACT が画像をどれだけ使っているかを切り分ける診断スクリプト。

1) in-distribution 検査: 学習データの観測を入れて、専門家の行動をどれだけ再現できるか
2) 反実仮想検査: state はそのまま、画像だけ別エピソードのものに差し替えて予測の変化量を見る
   → 変化がほぼ無ければ、ポリシーは画像を無視している
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# 第1引数: checkpoint, 第2引数: 参照するデモ npz のディレクトリ
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/checkpoints/act/100000/pretrained_model")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/dataset")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy = ACTPolicy.from_pretrained(CKPT); policy.to(dev); policy.eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(CKPT),
                                     preprocessor_overrides={"device_processor": {"device": str(dev)}})
print("chunk", policy.config.chunk_size, "n_action_steps", policy.config.n_action_steps)

def make_batch(state, imgs):
    b = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))}
    for cam, im in imgs.items():
        b[f"observation.images.{cam}"] = torch.from_numpy(im).permute(2, 0, 1).to(torch.float32) / 255.0
    return b

def predict(state, imgs):
    policy.reset()
    with torch.inference_mode():
        return post(policy.select_action(pre(make_batch(state, imgs)))).squeeze(0).cpu().numpy()

files = sorted(glob.glob(str(DATA / "*.npz")))
a, b = np.load(files[0]), np.load(files[7])   # 別ゴール・別配置の2本
print("ep A:", files[0].split('/')[-1], " ep B:", files[7].split('/')[-1])

idxs = [0, 30, 60, 90, 120]
err_true, delta_img = [], []
for t in idxs:
    s = a["states"][t]
    img_a = {"top": a["images_top"][t], "front": a["images_front"][t]}
    img_b = {"top": b["images_top"][t], "front": b["images_front"][t]}
    p_true = predict(s, img_a)
    p_swap = predict(s, img_b)          # state は同じ、画像だけ差し替え
    err_true.append(np.abs(p_true - a["actions"][t]).mean())
    delta_img.append(np.abs(p_true - p_swap).mean())

print(f"\n専門家行動との平均絶対誤差 : {np.mean(err_true):.4f} rad")
print(f"画像を差し替えた時の予測変化: {np.mean(delta_img):.4f} rad")
print(f"参考: 行動の標準偏差        : {a['actions'].std():.4f} rad")
if np.mean(delta_img) < 0.02:
    print("\n=> 画像をほぼ無視している（state だけで予測している疑いが濃厚）")
else:
    print("\n=> 画像に反応している")
