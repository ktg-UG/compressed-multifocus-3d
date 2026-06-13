#!/usr/bin/env python3
"""
A-5 Step1: Forward-only テスト

forward model 実装と snapshot.png 生成プロセスの整合性を検証する。

手順:
  1. GT スライス群 → 正規化 → log 変換 → MaskedImagingModel で合成スナップショット
  2. 実スナップショット（snapshot.png）と比較
  3. MAE / max_abs_err を主判定、PSNR を参考値として併記

主判定が uint8 量子化しきい値（1/255 ≈ 0.0039）内に収まれば整合。
乖離が大きければ正規化ずれ・clip 処理・forward 実装にバグの可能性。

使用例:
  python my-program/forward_only_test.py \
    -gt inputs/raw_data/open_scivis_128x128/aneurism_0/slice \
    -sm inputs/compressed_data/unique14/open_scivis_128x128/aneurism_0/masks.mat \
    -s  inputs/compressed_data/unique14/open_scivis_128x128/aneurism_0/snapshot.png \
    -o  experiments/forward_only_aneurism_0_snapshot
"""

import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
import scipy.io as scio
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.loading import get_image_paths, imread_uint
from img_model import MaskedImagingModel


def load_gt_volume(gt_dir):
    """GT スライス (PNG 群) を (N, H, W) float32 [0,1] で読み込む。"""
    paths = get_image_paths(gt_dir)
    slices = []
    for p in paths:
        img = imread_uint(p, n_channels=1)  # (H, W, 1) uint8
        slices.append(img[:, :, 0].astype(np.float32) / 255.0)
    return np.stack(slices, axis=0)


def load_masks(mat_path):
    """masks.mat → (N, H, W) float32 binary。"""
    data = scio.loadmat(mat_path)
    if 'ExpPtn' not in data:
        available = [k for k in data if not k.startswith('_')]
        raise KeyError(f"Key 'ExpPtn' not found in {mat_path}. Available: {available}")
    masks = data['ExpPtn'].astype(np.float32)
    if masks.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got shape {masks.shape}")
    d0, d1, d2 = masks.shape
    if d2 < d0 and d2 < d1:
        masks = np.transpose(masks, (2, 0, 1))
    return masks


def load_snapshot_png(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def run_forward(gt_volume, masks, device):
    """
    snapshot_block_random_unique_exclude_full.py:131-138 と同じ変換を適用。

    volume_normalized = gt_volume / gt_volume.max()
    volume_log = log(volume_normalized + 1e-6)
    snapshot = MaskedImagingModel(volume_log)
    """
    v_max = gt_volume.max()
    volume_normalized = gt_volume / v_max if v_max > 0 else gt_volume
    volume_tensor = torch.from_numpy(volume_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
    volume_log = torch.log(volume_tensor + 1e-6)

    masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()  # (1, N, H, W)
    imaging_model = MaskedImagingModel(
        device=device,
        psf_mask=None,
        snapshot_masks_tensor=masks_tensor,
    )

    with torch.no_grad():
        snapshot_list = imaging_model(volume_log)
        synthetic = snapshot_list[0].cpu().numpy().astype(np.float32)
    return synthetic, float(v_max)


def save_diff_images(synthetic, actual, out_dir):
    """差分画像を保存（|diff| とカラーマップ符号付き）。"""
    diff = synthetic - actual
    abs_diff = np.abs(diff)

    # 絶対差分（0..max を 0..255 にストレッチ）
    max_abs = abs_diff.max()
    if max_abs > 0:
        abs_vis = (abs_diff / max_abs * 255).clip(0, 255).astype(np.uint8)
    else:
        abs_vis = np.zeros_like(abs_diff, dtype=np.uint8)
    Image.fromarray(abs_vis).save(os.path.join(out_dir, 'diff_abs.png'))

    # 符号付き（-max..+max を 0..255 に）
    if max_abs > 0:
        signed_vis = ((diff + max_abs) / (2 * max_abs) * 255).clip(0, 255).astype(np.uint8)
    else:
        signed_vis = np.full_like(abs_diff, 127, dtype=np.uint8)
    Image.fromarray(signed_vis).save(os.path.join(out_dir, 'diff_signed.png'))


def classify_max_err(max_abs):
    """uint8 量子化を基準とした判定。"""
    if max_abs < 2.0 / 255.0:
        return "整合 (uint8 量子化範囲内)"
    if max_abs < 5.0 / 255.0:
        return "要調査 (正規化・clip 処理のずれ疑い)"
    return "バグ (forward 実装に乖離)"


def classify_mae(mae):
    if mae < 1.0 / 255.0:
        return "整合"
    if mae < 3.0 / 255.0:
        return "要調査"
    return "バグ"


def main():
    parser = argparse.ArgumentParser(description='A-5 Step1: Forward-only テスト')
    parser.add_argument('-gt', '--ground-truth', required=True, help='GT スライスディレクトリ')
    parser.add_argument('-sm', '--snapshot-masks', required=True, help='Masks .mat パス')
    parser.add_argument('-s', '--snapshot', required=True, help='実スナップショット PNG パス')
    parser.add_argument('-o', '--output', default='experiments/forward_only_output',
                        help='出力ディレクトリ')
    parser.add_argument('--device', default='cuda', help='cuda または cpu')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading GT       : {args.ground_truth}")
    gt = load_gt_volume(args.ground_truth)
    print(f"  shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]")

    print(f"Loading masks    : {args.snapshot_masks}")
    masks = load_masks(args.snapshot_masks)
    print(f"  shape: {masks.shape}")

    print(f"Loading snapshot : {args.snapshot}")
    actual = load_snapshot_png(args.snapshot)
    print(f"  shape: {actual.shape}, range [{actual.min():.4f}, {actual.max():.4f}]")

    # Shape 整合性チェック
    assert gt.shape[0] == masks.shape[0], \
        f"GT slices ({gt.shape[0]}) != n_masks ({masks.shape[0]})"
    assert gt.shape[1:] == masks.shape[1:] == actual.shape, \
        f"Spatial mismatch: GT {gt.shape[1:]}, masks {masks.shape[1:]}, snapshot {actual.shape}"

    # Forward pass
    print("\nRunning forward model on GT...")
    synthetic, v_max = run_forward(gt, masks, device)
    print(f"  synthetic range: [{synthetic.min():.4f}, {synthetic.max():.4f}]")
    print(f"  GT vol.max() used for normalization: {v_max:.4f}")

    # Save synthetic & copy actual for reference
    synth_uint8 = (synthetic * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(synth_uint8).save(os.path.join(args.output, 'synthetic_snapshot.png'))
    shutil.copyfile(args.snapshot, os.path.join(args.output, 'actual_snapshot_copy.png'))

    # 評価 (float [0,1] 同士)
    diff = synthetic - actual
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    max_abs = float(abs_diff.max())
    rmse = float(np.sqrt((diff ** 2).mean()))
    psnr_val = float(psnr_fn(actual, synthetic, data_range=1.0))
    ssim_val = float(ssim_fn(actual, synthetic, data_range=1.0))

    # uint8 round-trip 一致率（synthetic.png を保存 → 再読込 → 一致ピクセル割合）
    synth_reloaded = np.array(Image.open(os.path.join(args.output, 'synthetic_snapshot.png')),
                              dtype=np.uint8)
    actual_uint8 = (actual * 255).clip(0, 255).astype(np.uint8)  # 元PNGの近似再現
    exact_match_ratio = float((synth_reloaded == actual_uint8).mean())

    print("\n=== Forward-only 一致度（snapshot 空間） ===")
    print(f"  MAE          : {mae:.6f}  ({mae * 255:.3f} / 255)")
    print(f"  max_abs_err  : {max_abs:.6f}  ({max_abs * 255:.3f} / 255)")
    print(f"  RMSE         : {rmse:.6f}")
    print(f"  PSNR (参考)  : {psnr_val:.2f} dB")
    print(f"  SSIM         : {ssim_val:.4f}")
    print(f"  uint8 一致率 : {exact_match_ratio * 100:.2f} %")

    verdict_max = classify_max_err(max_abs)
    verdict_mae = classify_mae(mae)
    print(f"\n判定:")
    print(f"  max_abs_err 基準: {verdict_max}")
    print(f"  MAE 基準        : {verdict_mae}")

    # diff 画像保存
    save_diff_images(synthetic, actual, args.output)

    # result.txt
    result_path = os.path.join(args.output, 'result.txt')
    with open(result_path, 'w') as f:
        f.write("A-5 Step1: Forward-only テスト結果 (snapshot-space metric)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GT directory : {args.ground_truth}\n")
        f.write(f"Masks        : {args.snapshot_masks}\n")
        f.write(f"Actual PNG   : {args.snapshot}\n\n")
        f.write(f"GT volume shape: {gt.shape}, range [{gt.min():.4f}, {gt.max():.4f}]\n")
        f.write(f"GT vol.max() for normalization: {v_max:.4f}\n\n")
        f.write("一致度指標（synthetic vs actual, float [0,1] 同士）:\n")
        f.write(f"  MAE          : {mae:.6f}  ({mae * 255:.3f} / 255)\n")
        f.write(f"  max_abs_err  : {max_abs:.6f}  ({max_abs * 255:.3f} / 255)\n")
        f.write(f"  RMSE         : {rmse:.6f}\n")
        f.write(f"  PSNR (参考)  : {psnr_val:.2f} dB\n")
        f.write(f"  SSIM         : {ssim_val:.4f}\n")
        f.write(f"  uint8 一致率 : {exact_match_ratio * 100:.2f} %\n\n")
        f.write("判定しきい値 (uint8 量子化 1/255 ≈ 0.0039 基準):\n")
        f.write("  max_abs_err  : < 2/255 整合 / 2-5/255 要調査 / > 5/255 バグ\n")
        f.write("  MAE          : < 1/255 整合 / 1-3/255 要調査 / > 3/255 バグ\n\n")
        f.write(f"判定: max_abs_err → {verdict_max}\n")
        f.write(f"判定: MAE         → {verdict_mae}\n\n")
        f.write("注記:\n")
        f.write("  - これは snapshot 空間の一致度。baseline 36.94 dB は volume 空間なので直接比較不可。\n")
        f.write("  - 整合していれば、合成 snapshot で DIP を走らせて volume 空間上限を測る (Step1-b) に進む。\n")

    print(f"\nSaved to {args.output}/")
    print("  synthetic_snapshot.png, actual_snapshot_copy.png")
    print("  diff_abs.png, diff_signed.png")
    print("  result.txt")

    # 次手提案
    out_version = os.path.basename(os.path.abspath(args.output))
    dip_version = out_version.replace('forward_only_', 'forward_only_dip_', 1)
    dip_out = os.path.join(os.path.dirname(os.path.abspath(args.output)), dip_version)

    print("\n次手:")
    if "バグ" in verdict_max or "バグ" in verdict_mae:
        print("  → forward 実装 / 正規化ロジックの見直しが必要")
    elif "要調査" in verdict_max or "要調査" in verdict_mae:
        print("  → diff_abs.png を確認。局所誤差かグローバルずれか切り分け")
    else:
        print("  → 整合。合成 snapshot を使った DIP 実行 (Step1-b) で volume 空間上限を測れる:")
        print(f"    python main.py -m dip -p sc \\")
        print(f"      -s {os.path.join(args.output, 'synthetic_snapshot.png')} \\")
        print(f"      -sm {args.snapshot_masks} \\")
        print(f"      -gt {args.ground_truth} \\")
        print(f"      -v {dip_out} \\")
        print(f"      -o {dip_out}")


if __name__ == '__main__':
    main()
