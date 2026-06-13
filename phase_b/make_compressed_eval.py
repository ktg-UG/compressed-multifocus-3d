"""cell の多焦点画像 + 学習済みパターンから、評価用のスナップショットと masks.mat を生成する。

cell には volume GT が無いため、圧縮シミュレーションは PSF を使わず多焦点画像を直接マスク:
    snapshot = clamp( Σ_s (mask_s ⊙ focal_s) / D , 0, 1 )
これは MaskedImagingModel（DIP 推論側）の正規化（/n_masks）・clamp と整合する。

使用例:
  python3 phase_b/make_compressed_eval.py \
    --cell inputs/multi_focus_data/cell/0005 \
    --pattern experiments/phase_b_m2/learned_pattern.pt \
    -o inputs/compressed_data/learned_phase_b/cell/0005
"""

import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image
from scipy import io as scio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase_b.dataset import _load_focal_stack


def tile_pattern(pattern, height, width):
    """(D, b, b) を (D, H, W) にタイリング。"""
    d, bh, bw = pattern.shape
    reps_h = (height + bh - 1) // bh
    reps_w = (width + bw - 1) // bw
    tiled = np.tile(pattern, (1, reps_h, reps_w))[:, :height, :width]
    return tiled  # (D, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="cell サンプルディレクトリ（00.jpg..10.jpg）")
    ap.add_argument("--pattern", help="学習済みパターン .pt（無ければ --random で代替）")
    ap.add_argument("--random", action="store_true",
                    help="学習パターンの代わりに 50%% ランダム 2x2 パターンを使う（比較用 baseline）")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", required=True, help="出力ディレクトリ")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    focal = _load_focal_stack(args.cell)  # (D, H, W) float32 [0,1]
    d, h, w = focal.shape

    if args.random:
        rng = np.random.default_rng(args.seed)
        # 各層 50% ランダム。ただし全0/全1層（snapshot に寄与しない / 符号化なし）は再サンプル
        cells = args.block * args.block
        pattern = np.zeros((d, args.block, args.block), dtype=np.float32)
        for s in range(d):
            while True:
                blk = (rng.random((args.block, args.block)) > 0.5).astype(np.float32)
                ssum = blk.sum()
                if 0 < ssum < cells:
                    break
            pattern[s] = blk
        tag = "random"
    else:
        if not args.pattern:
            raise ValueError("--pattern か --random のどちらかを指定してください")
        ckpt = torch.load(args.pattern, map_location="cpu")
        pattern = np.rint(ckpt["binary_pattern"].numpy()).astype(np.float32)  # (D, b, b)
        tag = "learned"

    masks = tile_pattern(pattern, h, w)  # (D, H, W)

    # 圧縮シミュレーション（PSF なし）
    snapshot = (masks * focal).sum(axis=0) / d  # (H, W)
    snapshot = np.clip(snapshot, 0.0, 1.0)

    # snapshot を uint8 PNG で保存
    snap_path = os.path.join(args.output, "snapshot.png")
    Image.fromarray((snapshot * 255).astype(np.uint8), mode="L").save(snap_path)

    # masks を ExpPtn (H, W, N) で .mat 保存（main.py の loader と互換）
    exp_ptn = np.transpose(masks, (1, 2, 0)).astype(np.float32)  # (H, W, N)
    mat_path = os.path.join(args.output, "masks.mat")
    scio.savemat(mat_path, {"ExpPtn": exp_ptn})

    print(f"[make_compressed_eval] tag={tag} cell={args.cell}")
    print(f"  focal stack: {focal.shape}, aperture={pattern.mean():.3f}")
    print(f"  snapshot -> {snap_path}  range=[{snapshot.min():.3f},{snapshot.max():.3f}]")
    print(f"  masks.mat (ExpPtn {exp_ptn.shape}) -> {mat_path}")


if __name__ == "__main__":
    main()
