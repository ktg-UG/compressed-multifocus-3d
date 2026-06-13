#!/usr/bin/env python3
"""
MaskedImagingModelを使用して圧縮画像（スナップショット）を生成

物理的に正確なPSF畳み込み + マスクで圧縮画像を作成する。
これにより、MaskedImagingModelでの復元と前向きモデルが一致する。
"""

import os
import sys
import numpy as np
from PIL import Image
import scipy.io as scio
import argparse
import json
import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from img_model import MaskedImagingModel


def generate_mauf19_masks_flexible(h, w, n_slices):
    """
    MAUF19の物理構造に基づき、任意の枚数Nを2x2ユニットに分配するマスクを生成する
    (mauf-snapshot.pyと同じロジック)
    """
    masks = np.zeros((n_slices, h, w), dtype=np.float32)
    
    # 2x2のユニット座標定義
    unit_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 全N枚を4つのグループに分ける境界線を計算
    boundaries = np.linspace(0, n_slices, 5).astype(int)
    
    for i in range(4):
        r_mod, c_mod = unit_coords[i]
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        # この画素(r_mod, c_mod)が担当するスライスの範囲に1を立てる
        for s_idx in range(start_idx, end_idx):
            masks[s_idx, r_mod::2, c_mod::2] = 1.0
                
    return masks, boundaries


def main():
    parser = argparse.ArgumentParser(
        description="Generate snapshot image using MaskedImagingModel (PSF + Mask)"
    )
    parser.add_argument('-i', '--input', type=str, required=True, 
                       dest='vol_path', help='Input volume (.npy file path)')
    parser.add_argument('-o', '--output', type=str, required=True,
                       dest='output_dir', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()

    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. ボリュームデータ読み込み
    print(f"\n[1/4] Loading volume from {args.vol_path}")
    volume_np = np.load(args.vol_path)  # (D, H, W)
    
    n_slices, h, w = volume_np.shape
    print(f"  Volume shape: {volume_np.shape}")
    print(f"  Value range: [{volume_np.min():.4f}, {volume_np.max():.4f}]")

    # 2. MAUF19マスクパターン生成
    print(f"\n[2/4] Generating MAUF19 masks for {n_slices} slices")
    masks_np, boundaries = generate_mauf19_masks_flexible(h, w, n_slices)
    print(f"  Masks shape: {masks_np.shape}")
    print(f"  Slice assignments:")
    print(f"    Pixel A(0,0): slice {boundaries[0]}-{boundaries[1]-1}")
    print(f"    Pixel B(0,1): slice {boundaries[1]}-{boundaries[2]-1}")
    print(f"    Pixel C(1,0): slice {boundaries[2]}-{boundaries[3]-1}")
    print(f"    Pixel D(1,1): slice {boundaries[3]}-{boundaries[4]-1}")

    # マスクをtensorに変換 (1, N, H, W)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()

    # 3. MaskedImagingModelで順変換（スナップショット生成）
    print(f"\n[3/4] Generating snapshot with MaskedImagingModel (PSF + Mask)")
    
    # MaskedImagingModelのインスタンス化
    imaging_model = MaskedImagingModel(
        device=device,
        psf_mask=None,  # PSFマスクなし（全光線を使用）
        snapshot_masks_tensor=masks_tensor
    )
    
    # ボリュームをtensorに変換 (1, 1, D, H, W)
    # 値域を[0, 1]に正規化してからlog空間に変換
    volume_normalized = volume_np / volume_np.max()  # 正規化
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # log変換（MaskedImagingModelはlog入力を期待）
    volume_log = torch.log(volume_tensor + 1e-6)  # log(0)回避
    
    # 順変換
    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        snapshot_tensor = snapshot_list[0]  # (H, W)
    
    # CPU/numpyに変換
    snapshot_np = snapshot_tensor.cpu().numpy()
    print(f"  Snapshot shape: {snapshot_np.shape}")
    print(f"  Snapshot range: [{snapshot_np.min():.4f}, {snapshot_np.max():.4f}]")

    # 4. 保存
    print(f"\n[4/4] Saving results to {args.output_dir}")
    
    # 4-1. Snapshot画像 (PNG)
    snapshot_8bit = (snapshot_np * 255).clip(0, 255).astype(np.uint8)
    snapshot_path = os.path.join(args.output_dir, 'snapshot.png')
    Image.fromarray(snapshot_8bit).save(snapshot_path)
    print(f"  ✓ Saved: {snapshot_path}")

    # 4-2. masks.mat (DIP学習用, 変数名: ExpPtn)
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
        'method': 'MaskedImagingModel',
        'description': 'Snapshot generated with PSF convolution + MAUF19 masks',
        'n_slices': int(n_slices),
        'image_size': [int(h), int(w)],
        'input_volume': args.vol_path,
        'value_range': {
            'volume': [float(volume_np.min()), float(volume_np.max())],
            'snapshot': [float(snapshot_np.min()), float(snapshot_np.max())]
        },
        'assignments': {
            'pixel_A(0,0)': f"slice {boundaries[0]}-{boundaries[1]-1}",
            'pixel_B(0,1)': f"slice {boundaries[1]}-{boundaries[2]-1}",
            'pixel_C(1,0)': f"slice {boundaries[2]}-{boundaries[3]-1}",
            'pixel_D(1,1)': f"slice {boundaries[3]}-{boundaries[4]-1}",
        }
    }
    meta_path = os.path.join(args.output_dir, 'mapping.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"  ✓ Saved: {meta_path}")

    print(f"\n{'='*60}")
    print(f"✓ Successfully generated MaskedImagingModel snapshot!")
    print(f"{'='*60}")
    print(f"\nUsage for DIP reconstruction:")
    print(f"  python main.py -m dip -p v3 \\")
    print(f"    --snapshot {os.path.join(args.output_dir, 'snapshot.png')} \\")
    print(f"    --snapshot-masks {os.path.join(args.output_dir, 'masks.mat')} \\")
    print(f"    --snapshot-use-masked-model \\")
    print(f"    -o <output_dir> -v <version>")
    print()


if __name__ == '__main__':
    main()
