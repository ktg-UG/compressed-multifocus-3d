import scipy.io as scio
#!/usr/bin/env python3
"""
ランダムマスクを使用して圧縮画像（スナップショット）を生成

物理的に正確なPSF畳み込み + ランダムマスクで圧縮画像を作成する。
MaskedImagingModelでの復元と前向きモデルが一致する。
"""

import os
import sys
import numpy as np
import numpy as np
# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from PIL import Image
import argparse
import json
import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from img_model import MaskedImagingModel

def generate_random_masks(h, w, n_slices, seed=42):
    """
    ランダムなバイナリマスクをN枚生成 (各スライスごとに独立)
    """
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(n_slices, h, w)).astype(np.float32)
    return masks

def main():
    parser = argparse.ArgumentParser(
        description="Generate snapshot image using MaskedImagingModel (PSF + Random Mask)"
    )
    parser.add_argument('-i', '--input', type=str, required=True, 
                       dest='vol_path', help='Input volume (.npy file path)')
    parser.add_argument('-o', '--output', type=str, required=True,
                       dest='output_dir', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for mask generation')
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

    # 2. ランダムマスク生成
    print(f"\n[2/4] Generating random masks for {n_slices} slices (seed={args.seed})")
    masks_np = generate_random_masks(h, w, n_slices, seed=args.seed)
    print(f"  Masks shape: {masks_np.shape}")

    # マスクをtensorに変換 (1, N, H, W)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(0).float()

    # 3. MaskedImagingModelで順変換（スナップショット生成）
    print(f"\n[3/4] Generating snapshot with MaskedImagingModel (PSF + Random Mask)")
    imaging_model = MaskedImagingModel(
        device=device,
        psf_mask=None,  # PSFマスクなし（全光線を使用）
        snapshot_masks_tensor=masks_tensor
    )
    volume_normalized = volume_np / volume_np.max()  # 正規化
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    volume_log = torch.log(volume_tensor + 1e-6)  # log(0)回避
    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        snapshot_tensor = snapshot_list[0]  # (H, W)
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
        'description': 'Snapshot generated with PSF convolution + random masks',
        'n_slices': int(n_slices),
        'image_size': [int(h), int(w)],
        'input_volume': args.vol_path,
        'value_range': {
            'volume': [float(volume_np.min()), float(volume_np.max())],
            'snapshot': [float(snapshot_np.min()), float(snapshot_np.max())]
        },
        'mask_seed': int(args.seed)
    }
    meta_path = os.path.join(args.output_dir, 'mapping.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"  ✓ Saved: {meta_path}")

    print(f"\n{'='*60}")
    print(f"✓ Successfully generated MaskedImagingModel snapshot with random masks!")
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
