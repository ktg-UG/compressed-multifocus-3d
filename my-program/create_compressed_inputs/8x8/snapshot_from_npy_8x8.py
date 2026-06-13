#!/usr/bin/env python3
"""
8x8ブロックごとに1つだけ開くマスキングパターンでスナップショットを生成
"""
import sys
import os
import numpy as np
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from img_model import MaskedImagingModel

def generate_8x8_masks(h, w, n_slices):
    """
    各8x8ブロック内で1つだけ開くマスクをn_slices分生成
    """
    masks = np.zeros((n_slices, h, w), dtype=np.float32)
    block_size = 8
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    for s_idx in range(n_slices):
        open_idx = s_idx % (block_size * block_size)
        r_in_block = open_idx // block_size
        c_in_block = open_idx % block_size
        for bh in range(n_blocks_h):
            for bw in range(n_blocks_w):
                r = bh * block_size + r_in_block
                c = bw * block_size + c_in_block
                masks[s_idx, r, c] = 1.0
    return masks

def main():
    parser = argparse.ArgumentParser(description="Generate 8x8 block mask snapshot")
    parser.add_argument('-i', '--input', type=str, required=True, dest='vol_path', help='Input volume (.npy file path)')
    parser.add_argument('-o', '--output', type=str, required=True, dest='output_dir', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    volume_np = np.load(args.vol_path)
    n_slices, h, w = volume_np.shape
    masks_np = generate_8x8_masks(h, w, n_slices)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()

    imaging_model = MaskedImagingModel(device=device, psf_mask=None, snapshot_masks_tensor=masks_tensor)
    volume_normalized = volume_np / volume_np.max()
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    volume_log = torch.log(volume_tensor + 1e-6)

    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        snapshot_tensor = snapshot_list[0]
    snapshot_np = snapshot_tensor.cpu().numpy()

    # 1. Snapshot画像 (PNG)
    snapshot_8bit = (snapshot_np * 255).clip(0, 255).astype(np.uint8)
    snapshot_path = os.path.join(args.output_dir, 'snapshot.png')
    Image.fromarray(snapshot_8bit).save(snapshot_path)
    print(f"  ✓ Saved: {snapshot_path}")

    # 2. masks.mat (DIP学習用, 変数名: ExpPtn)
    import scipy.io as scio
    masks_hwk = np.transpose(masks_np, (1, 2, 0))  # (H, W, N)
    masks_path = os.path.join(args.output_dir, 'masks.mat')
    scio.savemat(masks_path, {'ExpPtn': masks_hwk.astype(np.float32)}, do_compression=True)
    print(f"  ✓ Saved: {masks_path}")

    # 3. 全スライスのマスクパターン画像
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    for s in range(n_slices):
        mask_s_8bit = (masks_np[s] * 255).astype(np.uint8)
        Image.fromarray(mask_s_8bit).save(os.path.join(masks_dir, f'mask_slice_{s:02d}.png'))
    print(f"  ✓ Saved: {masks_dir}/ (mask_slice_*.png)")

    # 4. mapping.json (メタデータ)
    import json
    meta = {
        'method': 'MaskedImagingModel',
        'description': 'Snapshot generated with PSF convolution + 8x8 masks',
        'n_slices': int(n_slices),
        'image_size': [int(h), int(w)],
        'input_volume': args.vol_path,
        'value_range': {
            'volume': [float(volume_np.min()), float(volume_np.max())],
            'snapshot': [float(snapshot_np.min()), float(snapshot_np.max())]
        },
        'assignments': {
            'block_size': 8,
            'open_pixel_per_slice': 'Each slice opens a unique pixel in every 8x8 block'
        }
    }
    meta_path = os.path.join(args.output_dir, 'mapping.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"  ✓ Saved: {meta_path}")

if __name__ == '__main__':
    main()
