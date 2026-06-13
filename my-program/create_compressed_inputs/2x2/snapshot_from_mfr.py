import os
import numpy as np
from PIL import Image
import scipy.io as scio
import argparse
import json

def generate_mauf19_masks_flexible(h, w, n_slices):
    """
    MAUF19の物理構造に基づき、任意の枚数Nを2x2ユニットに分配するマスクを生成する
    - 2x2ユニットの各画素が、全N枚を4分割したブロックをそれぞれ担当する
    """
    masks = np.zeros((n_slices, h, w), dtype=np.float32)
    
    # 2x2のユニット座標定義
    # (0,0)=A, (0,1)=B, (1,0)=C, (1,1)=D
    unit_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # 全N枚を4つのグループに分ける境界線を計算 (np.linspaceで端数を調整)
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
    parser = argparse.ArgumentParser(description="MAUF19 Flexible Slicing Simulation")
    parser.add_argument('-i', type=str, required=True, dest='image_dir', help='多焦点画像のディレクトリ')
    parser.add_argument('-o', type=str, default='./output_mauf19_flex', dest='output_dir', help='保存先')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. 画像の読み込み
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    filepaths = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                       if f.lower().endswith(exts)])
    
    n_slices = len(filepaths)
    if n_slices == 0:
        print(f"エラー: {args.image_dir} に画像が見つかりません。")
        return
    print(f"検出された画像枚数: {n_slices}枚")

    # 初枚を読み込んでサイズ取得
    img0 = np.array(Image.open(filepaths[0]).convert('L'))
    h, w = img0.shape
    images = np.zeros((n_slices, h, w), dtype=np.float32)

    for i, fp in enumerate(filepaths):
        images[i] = np.array(Image.open(fp).convert('L')) / 255.0

    # 2. 汎用MAUF19マスクの生成
    masks, boundaries = generate_mauf19_masks_flexible(h, w, n_slices)
    
    # 3. 圧縮画像 (Snapshot) の生成
    # S = sum(I * M) / sum(M)
    mask_sum = np.sum(masks, axis=0)
    mask_sum_safe = np.where(mask_sum > 0, mask_sum, 1.0) # 0除算回避
    
    snapshot = np.sum(images * masks, axis=0) / mask_sum_safe

    # 4. 保存
    # Snapshot (PNG)
    snapshot_8bit = (snapshot * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(snapshot_8bit).save(os.path.join(args.output_dir, 'snapshot.png'))

    # masks.mat (DIP学習用, 変数名: ExpPtn)
    masks_hwk = np.transpose(masks, (1, 2, 0)) # (H, W, N)
    scio.savemat(os.path.join(args.output_dir, 'masks.mat'), {'ExpPtn': masks_hwk.astype(np.float32)}, do_compression=True)
    
    # 全スライスのマスクパターン画像の保存
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    for s in range(n_slices):
        mask_s_8bit = (masks[s] * 255).astype(np.uint8)
        Image.fromarray(mask_s_8bit).save(os.path.join(masks_dir, f'mask_slice_{s:02d}.png'))

    # mapping.json (メタデータ)
    meta = {
        'n_slices': n_slices,
        'image_size': [h, w],
        'assignments': {
            'pixel_A(0,0)': f"slice {boundaries[0]}-{boundaries[1]-1}",
            'pixel_B(0,1)': f"slice {boundaries[1]}-{boundaries[2]-1}",
            'pixel_C(1,0)': f"slice {boundaries[2]}-{boundaries[3]-1}",
            'pixel_D(1,1)': f"slice {boundaries[3]}-{boundaries[4]-1}",
        }
    }
    with open(os.path.join(args.output_dir, 'mapping.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"成功: {args.output_dir} にデータを保存しました。")

if __name__ == '__main__':
    main()