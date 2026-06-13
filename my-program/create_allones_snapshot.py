#!/usr/bin/env python3
"""
全開放マスク（全1）スナップショット生成 - A-5 Step3 用

全スライスのマスクを 1 にした場合のスナップショットを生成する。
snapshot[y,x] = (1/N) × Σ_s focal_s[y,x]  （全スライスの平均投影）

使用例:
  python my-program/create_allones_snapshot.py \
    -i inputs/raw_data/open_scivis_128x128/aneurism_0/vol.npy \
    -o inputs/compressed_data/allones/open_scivis_128x128/aneurism_0
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
import json
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from img_model import MaskedImagingModel


def main():
    parser = argparse.ArgumentParser(
        description='全開放マスク（全1）スナップショット生成 - A-5 Step3 用'
    )
    parser.add_argument('-i', '--input', required=True, dest='vol_path',
                        help='入力ボリューム (.npy)')
    parser.add_argument('-o', '--output', required=True, dest='output_dir',
                        help='出力ディレクトリ')
    parser.add_argument('--device', default='cuda',
                        help='使用デバイス (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")

    # 1. ボリューム読み込み
    print(f"\n[1/3] Loading volume from {args.vol_path}")
    volume_np = np.load(args.vol_path)  # (D, H, W)
    n_slices, h, w = volume_np.shape
    print(f"  shape: {volume_np.shape}, range [{volume_np.min():.4f}, {volume_np.max():.4f}]")

    # 2. 全1マスク生成
    print(f"\n[2/3] Generating all-ones masks ({n_slices} slices, {h}×{w})")
    masks_np = np.ones((n_slices, h, w), dtype=np.float32)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()  # (1, N, H, W)

    # 3. スナップショット生成
    print("\n[3/3] Generating snapshot with MaskedImagingModel (all-ones masks)")
    imaging_model = MaskedImagingModel(
        device=device,
        psf_mask=None,
        snapshot_masks_tensor=masks_tensor,
    )

    volume_normalized = volume_np / volume_np.max()
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    volume_log = torch.log(volume_tensor + 1e-6)

    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        snapshot_np = snapshot_list[0].cpu().numpy().astype(np.float32)

    print(f"  snapshot range: [{snapshot_np.min():.4f}, {snapshot_np.max():.4f}]")

    # 保存
    snapshot_path = os.path.join(args.output_dir, 'snapshot.png')
    Image.fromarray((snapshot_np * 255).clip(0, 255).astype(np.uint8)).save(snapshot_path)
    print(f"  ✓ {snapshot_path}")

    masks_hwk = np.transpose(masks_np, (1, 2, 0))  # (H, W, N)
    masks_path = os.path.join(args.output_dir, 'masks.mat')
    scio.savemat(masks_path, {'ExpPtn': masks_hwk}, do_compression=True)
    print(f"  ✓ {masks_path}")

    meta = {
        'method': 'allones',
        'description': 'All-ones masks (A-5 Step3 diagnostic)',
        'n_slices': int(n_slices),
        'image_size': [int(h), int(w)],
        'open_ratio': 1.0,
        'input_volume': args.vol_path,
        'value_range': {
            'volume': [float(volume_np.min()), float(volume_np.max())],
            'snapshot': [float(snapshot_np.min()), float(snapshot_np.max())]
        }
    }
    with open(os.path.join(args.output_dir, 'mapping.json'), 'w') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"  ✓ mapping.json")

    print("\nDone.")


if __name__ == '__main__':
    main()
