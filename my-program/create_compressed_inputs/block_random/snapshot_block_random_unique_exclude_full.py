#!/usr/bin/env python3
"""
ブロックパターンマスクを使用して圧縮画像（スナップショット）を生成

仕様:
  - 画像を A×A ピクセルのブロックに分割 (A = 2, 4, 8 から選択)
  - 各スライスで A×A ブロック内の2値パターンを1つ選択
  - 全マスク(全0)と全露光(全1)は除外
  - スライス間でパターンは重複なし
  - 選んだ1パターンを画像全体へタイリング（全ブロック同一）

例:
  python snapshot_block_random_unique_exclude_full.py \
    -i inputs/raw_data/open_scivis_128x128/aneurism_0/vol.npy \
    -o test_run_output/block_random_2x2_unique14 \
    --block-size 2 --seed 42
"""

import os
import sys
import numpy as np
from PIL import Image
import scipy.io as scio
import argparse
import json
import torch
import random
import csv

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from img_model import MaskedImagingModel


def generate_unique_nontrivial_masks(h, w, n_slices, block_size, seed=None):
    """
    全0/全1を除く全パターンから、重複なしで n_slices 個を選んでマスク生成。

    Returns:
        masks: (n_slices, h, w) float32
        total_available_patterns: 使用可能な一意パターン数
        selected_codes: 各sliceに割り当てた整数コード
        pattern_blocks: 各sliceのA×Aパターン（list of list）
    """
    rng = random.Random(seed)
    n_pixels = block_size * block_size

    # 2^n 通りのうち、全0 と 全1 を除外
    total_available_patterns = (1 << n_pixels) - 2
    if n_slices > total_available_patterns:
        raise ValueError(
            f"n_slices={n_slices} は一意パターン数の上限 {total_available_patterns} を超えています"
        )

    # コード値 1..(2^n - 2): 全0(0) と 全1(2^n-1) を除外
    selected_codes = rng.sample(range(1, (1 << n_pixels) - 1), n_slices)

    masks = np.zeros((n_slices, h, w), dtype=np.float32)
    pattern_blocks = []
    for s, code in enumerate(selected_codes):
        pattern = np.zeros(n_pixels, dtype=np.float32)
        for bit in range(n_pixels):
            if (code >> bit) & 1:
                pattern[bit] = 1.0

        pattern_2d = pattern.reshape(block_size, block_size)
        pattern_blocks.append(pattern_2d.astype(int).tolist())

        # 画像全体にタイリングしてクロップ
        tiles_h = (h + block_size - 1) // block_size
        tiles_w = (w + block_size - 1) // block_size
        tiled = np.tile(pattern_2d, (tiles_h, tiles_w))[:h, :w]
        masks[s] = tiled

    return masks, total_available_patterns, selected_codes, pattern_blocks


def main():
    parser = argparse.ArgumentParser(
        description="Generate snapshot with unique tiled block masks excluding all-0/all-1"
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        dest='vol_path', help='入力ボリューム (.npy ファイルパス)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        dest='output_dir', help='出力ディレクトリ')
    parser.add_argument('--block-size', type=int, default=2, choices=[2, 4, 8],
                        help='ブロックサイズ A (2, 4, 8 のいずれか, デフォルト: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード (デフォルト: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用デバイス (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Block size  : {args.block_size}x{args.block_size}")
    print(f"Seed        : {args.seed}")

    # 1. 入力読み込み
    print(f"\n[1/4] Loading volume from {args.vol_path}")
    volume_np = np.load(args.vol_path)  # (D, H, W)
    n_slices, h, w = volume_np.shape
    print(f"  Volume shape: {volume_np.shape}")
    print(f"  Value range : [{volume_np.min():.4f}, {volume_np.max():.4f}]")

    # 2. マスク生成
    print(f"\n[2/4] Generating unique nontrivial masks "
          f"({args.block_size}x{args.block_size}, exclude all-0/all-1)")
    masks_np, total_available, selected_codes, pattern_blocks = generate_unique_nontrivial_masks(
        h, w, n_slices, args.block_size, seed=args.seed
    )
    print(f"  Masks shape              : {masks_np.shape}")
    print(f"  Available unique patterns: {total_available}")
    print(f"  Open ratio (mean)        : {masks_np.mean():.4f}")

    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()

    # 3. スナップショット生成
    print("\n[3/4] Generating snapshot with MaskedImagingModel")
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

    # 4. 保存
    print(f"\n[4/4] Saving outputs to {args.output_dir}")

    snapshot_8bit = (snapshot_np * 255).clip(0, 255).astype(np.uint8)
    snapshot_path = os.path.join(args.output_dir, 'snapshot.png')
    Image.fromarray(snapshot_8bit).save(snapshot_path)
    print(f"  ✓ Saved: {snapshot_path}")

    masks_hwk = np.transpose(masks_np, (1, 2, 0))  # (H, W, N)
    masks_path = os.path.join(args.output_dir, 'masks.mat')
    scio.savemat(masks_path, {'ExpPtn': masks_hwk.astype(np.float32)}, do_compression=True)
    print(f"  ✓ Saved: {masks_path}")

    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    for s in range(n_slices):
        mask_s_8bit = (masks_np[s] * 255).astype(np.uint8)
        Image.fromarray(mask_s_8bit).save(os.path.join(masks_dir, f'mask_slice_{s:02d}.png'))
    print(f"  ✓ Saved: {masks_dir}/ (mask_slice_*.png)")

    # 各sliceに対応するブロックパターン情報を保存（見やすさ重視）
    pattern_rows = []
    for s in range(n_slices):
        block = pattern_blocks[s]
        open_pixels = int(sum(sum(r) for r in block))
        pattern_rows.append({
            'slice': int(s),
            'code': int(selected_codes[s]),
            'open_pixels': open_pixels,
            'open_ratio': float(open_pixels / (args.block_size * args.block_size)),
            'pattern_2d': block,
        })

    pattern_json_path = os.path.join(args.output_dir, 'pattern_assignments.json')
    with open(pattern_json_path, 'w') as f:
        json.dump(pattern_rows, f, indent=4, ensure_ascii=False)
    print(f"  ✓ Saved: {pattern_json_path}")

    pattern_csv_path = os.path.join(args.output_dir, 'pattern_assignments.csv')
    row_fields = [f'row_{r:02d}' for r in range(args.block_size)]
    with open(pattern_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['slice', 'code', 'open_pixels', 'open_ratio', *row_fields])
        for row in pattern_rows:
            row_strings = [''.join(str(v) for v in r) for r in row['pattern_2d']]
            writer.writerow([
                row['slice'],
                row['code'],
                row['open_pixels'],
                f"{row['open_ratio']:.6f}",
                *row_strings,
            ])
    print(f"  ✓ Saved: {pattern_csv_path}")

    pattern_txt_path = os.path.join(args.output_dir, 'pattern_assignments.txt')
    with open(pattern_txt_path, 'w') as f:
        f.write(f"block_size={args.block_size}x{args.block_size}, n_slices={n_slices}\n")
        f.write("legend: 1=open, 0=masked\n\n")
        for row in pattern_rows:
            f.write(
                f"slice {row['slice']:02d} | code={row['code']} | "
                f"open={row['open_pixels']}/{args.block_size * args.block_size} "
                f"({row['open_ratio']:.3f})\n"
            )
            for rr in row['pattern_2d']:
                f.write('  ' + ' '.join(str(v) for v in rr) + '\n')
            f.write('\n')
    print(f"  ✓ Saved: {pattern_txt_path}")

    meta = {
        'method': 'block_random_unique_exclude_full',
        'description': (
            f'Snapshot generated with PSF convolution + '
            f'{args.block_size}x{args.block_size} tiled masks '
            f'(unique per slice, exclude all-0/all-1)'
        ),
        'block_size': int(args.block_size),
        'n_pixels_per_block': int(args.block_size * args.block_size),
        'total_available_patterns': int(total_available),
        'unique_patterns_per_slice': True,
        'exclude_all_mask_and_all_open': True,
        'pattern_files': {
            'json': 'pattern_assignments.json',
            'csv': 'pattern_assignments.csv',
            'txt': 'pattern_assignments.txt'
        },
        'seed': int(args.seed),
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
