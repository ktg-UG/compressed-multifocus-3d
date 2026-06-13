#!/usr/bin/env python3
"""
ブロックランダムマスクを使用して圧縮画像（スナップショット）を生成

仕様:
  - 画像を A×A ピクセルのブロックに分割 (A = 2, 4, 8 から選択)
  - フレームごとに A×A ブロック内の 50% をランダムに選んで開口するパターンを 1 つ生成
  - その 1 パターンを画像全体にタイリング（全ブロックで同じパターン）
  - フレーム間はそれぞれ異なるランダムパターン

使用例:
  python snapshot_block_random.py -i volume.npy -o output/ --block-size 4
"""

import os
import sys
import numpy as np
from PIL import Image
import scipy.io as scio
import argparse
import json
import torch
import math
import random

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from img_model import MaskedImagingModel


def generate_block_random_masks(h, w, n_slices, block_size, seed=None):
    """
    各フレームに対して A×A ブロック内でランダムに 50% を開口し、
    そのパターンを画像全体にタイリングしたマスクを生成する。
    さらに、フレーム間で 4x4(や2x2/8x8)パターンが重複しないことを保証する。

    Args:
        h, w       : 画像サイズ
        n_slices   : フレーム数
        block_size : ブロックサイズ A (2, 4, 8 など)
        seed       : 乱数シード (None で毎回異なる)

    Returns:
        masks      : shape (n_slices, h, w), dtype float32
    """
    rng = random.Random(seed)
    n_pixels = block_size * block_size
    n_open   = n_pixels // 2  # 50% 開口
    total_patterns = math.comb(n_pixels, n_open)

    if n_slices > total_patterns:
        raise ValueError(
            f"n_slices={n_slices} は一意パターン数の上限 {total_patterns} を超えています"
        )

    masks = np.zeros((n_slices, h, w), dtype=np.float32)

    def unrank_combination(n, k, rank):
        """
        0 <= rank < C(n, k) を、辞書順の k-combination (0-index) に復元する。
        """
        comb = []
        start = 0
        r = rank
        for remaining in range(k, 0, -1):
            for candidate in range(start, n):
                count = math.comb(n - candidate - 1, remaining - 1)
                if r < count:
                    comb.append(candidate)
                    start = candidate + 1
                    break
                r -= count
        return comb

    # C(n_pixels, n_open) 通りから重複なしで n_slices 個をサンプリング
    selected_ranks = rng.sample(range(total_patterns), n_slices)

    for s in range(n_slices):
        open_idx = unrank_combination(n_pixels, n_open, selected_ranks[s])
        pattern = np.zeros(n_pixels, dtype=np.float32)
        pattern[open_idx] = 1.0
        pattern_2d = pattern.reshape(block_size, block_size)

        # 画像サイズを超えないようにタイリング後クロップ
        tiles_h = (h + block_size - 1) // block_size
        tiles_w = (w + block_size - 1) // block_size
        tiled = np.tile(pattern_2d, (tiles_h, tiles_w))[:h, :w]
        masks[s] = tiled

    return masks


def main():
    parser = argparse.ArgumentParser(
        description="Generate snapshot using block-random masks (tiled 50% pattern)"
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        dest='vol_path', help='入力ボリューム (.npy ファイルパス)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        dest='output_dir', help='出力ディレクトリ')
    parser.add_argument('--block-size', type=int, default=4, choices=[2, 4, 8],
                        help='ブロックサイズ A (2, 4, 8 のいずれか, デフォルト: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (デフォルト: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用デバイス (cuda or cpu)')
    args = parser.parse_args()

    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Block size  : {args.block_size}x{args.block_size}")
    print(f"Seed        : {args.seed}")

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. ボリュームデータ読み込み
    print(f"\n[1/4] Loading volume from {args.vol_path}")
    volume_np = np.load(args.vol_path)  # (D, H, W)
    n_slices, h, w = volume_np.shape
    print(f"  Volume shape: {volume_np.shape}")
    print(f"  Value range : [{volume_np.min():.4f}, {volume_np.max():.4f}]")

    # 2. ブロックランダムマスク生成
    print(f"\n[2/4] Generating block-random masks "
          f"({args.block_size}x{args.block_size} block, 50% open, seed={args.seed})")
    masks_np = generate_block_random_masks(h, w, n_slices, args.block_size, seed=args.seed)
    print(f"  Masks shape : {masks_np.shape}")
    open_ratio = masks_np.mean()
    print(f"  Open ratio  : {open_ratio:.4f} (target: 0.5000)")

    # マスクを tensor に変換 (1, N, H, W)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()

    # 3. MaskedImagingModel でスナップショット生成
    print(f"\n[3/4] Generating snapshot with MaskedImagingModel (PSF + Block-Random Mask)")
    imaging_model = MaskedImagingModel(
        device=device,
        psf_mask=None,
        snapshot_masks_tensor=masks_tensor
    )

    volume_normalized = volume_np / volume_np.max()
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    volume_log = torch.log(volume_tensor + 1e-6)

    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        snapshot_tensor = snapshot_list[0]
    snapshot_np = snapshot_tensor.cpu().numpy()

    # 4. 結果の保存
    print(f"\n[4/4] Saving outputs to {args.output_dir}")

    # 4-1. Snapshot 画像 (PNG)
    snapshot_8bit = (snapshot_np * 255).clip(0, 255).astype(np.uint8)
    snapshot_path = os.path.join(args.output_dir, 'snapshot.png')
    Image.fromarray(snapshot_8bit).save(snapshot_path)
    print(f"  ✓ Saved: {snapshot_path}")

    # 4-2. masks.mat (DIP 学習用, 変数名: ExpPtn)
    masks_hwk = np.transpose(masks_np, (1, 2, 0))  # (H, W, N)
    masks_path = os.path.join(args.output_dir, 'masks.mat')
    scio.savemat(masks_path, {'ExpPtn': masks_hwk.astype(np.float32)}, do_compression=True)
    print(f"  ✓ Saved: {masks_path}")

    # 4-3. 全スライスのマスクパターン画像
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    for s in range(n_slices):
        mask_s_8bit = (masks_np[s] * 255).astype(np.uint8)
        Image.fromarray(mask_s_8bit).save(os.path.join(masks_dir, f'mask_slice_{s:02d}.png'))
    print(f"  ✓ Saved: {masks_dir}/ (mask_slice_*.png)")

    # 4-4. mapping.json (メタデータ)
    meta = {
        'method': 'block_random',
        'description': (
            f'Snapshot generated with PSF convolution + '
            f'{args.block_size}x{args.block_size} block-random masks (50% open, tiled, unique per slice)'
        ),
        'block_size': args.block_size,
        'open_ratio': float(open_ratio),
        'unique_patterns_per_slice': True,
        'seed': args.seed,
        'n_slices': int(n_slices),
        'image_size': [int(h), int(w)],
        'input_volume': args.vol_path,
        'value_range': {
            'volume': [float(volume_np.min()), float(volume_np.max())],
            'snapshot': [float(snapshot_np.min()), float(snapshot_np.max())]
        }
    }
    meta_path = os.path.join(args.output_dir, 'mapping.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"  ✓ Saved: {meta_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
