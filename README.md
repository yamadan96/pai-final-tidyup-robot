# 自然言語で指示する片付けロボット（MuJoCo × ローカル LLM × 模倣学習）

Physical AI 講座 2026 応用編1 最終課題のコード一式です。

机の上に置かれた 3 色のブロックを、日本語の指示どおりにトレイへ片付けます。
指示の解釈はローカルで動かす LLM（Ollama / qwen2.5-coder）が担当し、
実行は「手で書いた制御器」と「その制御器の実演を模倣学習した ACT ポリシー」の 2 通りを用意しています。

```
「ブロックを全部手前のトレイに入れて」
        │
        ▼  ローカル LLM（温度 0・JSON 出力）
[{"color":"red","tray":"B"}, {"color":"blue","tray":"B"}, {"color":"green","tray":"B"}]
        │
        ▼  方策（IK + 状態機械 / ACT）
   MuJoCo（Franka Emika Panda, 物理 500 Hz・制御 50 Hz）
```

クラウド GPU も外部 API も使わず、Apple Silicon の Mac 1 台で完結します。

## できること

| 要素 | 実測 |
| --- | --- |
| LLM プランナ | テスト 15 問に 15/15 正解、1 問あたり平均 0.84 秒 |
| 規範方策（IK + 状態機械） | 300 本中 300 本成功（ノイズ注入時は 294/300） |
| 模倣学習（ACT） | 未知配置 18 本中 2 本成功（11.1%、最良の v2）※原因究明の記録はレポート参照 |

## セットアップ

```bash
uv venv
uv sync

# ローカル LLM
ollama serve
ollama pull qwen2.5-coder
```

MuJoCo Menagerie の Franka Emika Panda モデルを `assets/franka_emika_panda/` に同梱しています（Apache-2.0、`NOTICE` 参照）。

## 使い方

```bash
# LLM プランナだけ試す
./.venv/bin/python scripts/demo_planner.py --self-test
./.venv/bin/python scripts/demo_planner.py "青と緑を奥のトレイに片付けて"

# 指示 → 計画 → 実行 を通す
./.venv/bin/python scripts/demo_pipeline.py \
    --instruction "ブロックを全部手前のトレイに入れて" \
    --policy expert --video outputs/demo.mp4

# 実演を集める（約 20 分）
./.venv/bin/python scripts/collect_demos.py --episodes 300 \
    --out outputs/dataset_v2 --image-size 160 --camera-preset tight \
    --noise-mode ctrl --noise-std 0.015

# LeRobot 形式へ変換 → 学習 → 評価
./.venv/bin/python scripts/build_lerobot_dataset.py \
    --src outputs/dataset_v2 --out outputs/lerobot/tidyup_v3 \
    --repo-id local/tidyup_v3 --image-size 160 --goal-cue --goal-repeat 8
./.venv/bin/python scripts/train_act.py \
    --dataset outputs/lerobot/tidyup_v3 --repo-id local/tidyup_v3 \
    --out outputs/checkpoints/act_v3 --steps 60000
./.venv/bin/python scripts/eval_policy.py \
    --checkpoint outputs/checkpoints/act_v3/last/pretrained_model \
    --episodes 18 --seed 777 --camera-preset tight --image-size 160 \
    --temporal-ensemble 0.01 --goal-cue
```

## ディレクトリ

```
pai_final/
  env.py          MuJoCo 環境（観測・行動・成功判定）
  expert.py       減衰最小二乗 IK + 8 フェーズの状態機械
  noise.py        実演収集時のノイズ注入（教師ラベルはきれいなまま）
  planner.py      ローカル LLM で指示をゴール列に変換
  planner_testset.py  プランナの 15 問テスト
  goal_cue.py     観測画像にゴールを描き込む（学習・推論で共通）
  overlay.py      動画の日本語字幕・情報パネル
  demo_common.py  デモ用の共通部品
scripts/
  collect_demos.py          実演の収集
  build_lerobot_dataset.py  npz → LeRobotDataset（統計量の補正込み）
  train_act.py              ACT の学習
  eval_policy.py            未知配置での評価
  diagnose_policy.py        画像が出力に効いているかの診断
  diagnose_grasp_target.py  どのブロックを掴みに行ったかの診断
  measure_block_pixels.py   観測画像内のブロックの大きさ
  measure_occlusion.py      把持直前の遮蔽率
  demo_pipeline.py          指示 → 計画 → 実行
  make_submission_video.py  提出動画の生成
  make_report_figures.py    レポートの図
assets/franka_emika_panda/  Panda モデル（Menagerie）+ 本課題のシーン
report/                     レポートと図
tests/                      pytest
```

## 既知の制約

- ACT ポリシーの成功率は最良で 11.1% です。掴んで運ぶ動作は学習できている一方、
  ゴールで指定した色ではないブロックを掴むことが診断で分かっています。正しい色に
  最も近づけた割合は 18 本中 7 本（39%）で、3 色からでたらめに選ぶ 33% と区別できません。
- ゴールの信号を強める改修（state のゴール one-hot を 8 回繰り返して 48 次元化、
  観測画像へのマーカー描き込み）を v3 として試しましたが、学習ステップ数をそろえた
  比較で v2 と同じ 1 / 18 にとどまり、改善は検出できませんでした。詳細は `report/` のレポート 8 節を
  参照してください。
- LeRobot 0.4.4 の統計量計算に uint8 オーバーフローがあり、画像の標準偏差が 0 になります。
  `scripts/build_lerobot_dataset.py` で float64 で計算し直して回避しています。

## ライセンス

Apache License 2.0（`LICENSE`）。同梱物については `NOTICE` を参照してください。
