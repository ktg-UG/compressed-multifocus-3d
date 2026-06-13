#!/usr/bin/env python3
"""
Oracle 復元（A-5 Step2）- 診断用近似最小二乗法

近似前提:
  1. PSF を無視（各深度スライスが対応するスナップショットピクセルに直接寄与）
  2. 2×2 ブロック内は均一（ブロック内 4 ピクセルが同じ深度プロファイル V を共有）

これらの前提により「厳密な情報上限」ではなく「診断用近似上限」である点に注意。
（PSF 込みの厳密 Oracle を求めるには非線形逆問題を解く必要がある）

線形システム:
  y[i] = (1/N) × A[i,:] @ V    i=0..3（2×2 ブロック内ピクセル位置）
  A[i,s] = mask[s, i//2, i%2]  (4×N バイナリ行列、全ブロック共通)
  → np.linalg.lstsq(A/N, y) で最小ノルム解 V を推定

使用例:
  python my-program/oracle_reconstruction.py \
    -s inputs/compressed_data/unique14/open_scivis_128x128/aneurism_0/snapshot.png \
    -sm inputs/compressed_data/unique14/open_scivis_128x128/aneurism_0/masks.mat \
    -gt inputs/raw_data/open_scivis_128x128/aneurism_0/slice \
    -o experiments/oracle_aneurism_0_snapshot
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.loading import get_image_paths, imread_uint

SVD_RANK_THRESH = 1e-6
SVD_COND_THRESH = 1e-10


def load_snapshot(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def load_masks(mat_path):
    data = scio.loadmat(mat_path)

    key = 'ExpPtn'
    if key not in data:
        available = [k for k in data if not k.startswith('_')]
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available: {available}")

    masks = data[key].astype(np.float32)

    if masks.ndim != 3:
        raise ValueError(f"Expected 3D mask array, got shape {masks.shape}")

    # 形状判定: (H,W,N) → (N,H,W)、(N,H,W) はそのまま
    d0, d1, d2 = masks.shape
    if d2 < d0 and d2 < d1:
        # 最小次元が第3軸 → (H,W,N)
        masks = np.transpose(masks, (2, 0, 1))

    return masks  # (N, H, W)


def load_gt(gt_dir):
    paths = get_image_paths(gt_dir)
    slices = []
    for p in paths:
        img = imread_uint(p, n_channels=1)  # (H, W, 1) uint8
        slices.append(img[:, :, 0].astype(np.float32) / 255.0)
    return np.stack(slices, axis=0)  # (N, H, W)


def build_measurement_matrix(masks):
    """
    A[i,s] = masks[s, i//2, i%2]  (i=0..3)
    Returns A (4, N) float32
    """
    A = np.stack([
        masks[:, 0, 0],  # pixel (0,0)
        masks[:, 0, 1],  # pixel (0,1)
        masks[:, 1, 0],  # pixel (1,0)
        masks[:, 1, 1],  # pixel (1,1)
    ], axis=0).astype(np.float32)  # (4, N)
    assert A.shape == (4, masks.shape[0]), f"Expected (4, N), got {A.shape}"
    return A


def oracle_reconstruct(snapshot, masks):
    """
    PSF 無視 + ブロック均一の近似 Oracle 復元。

    Args:
        snapshot: (H, W) float32 [0,1]
        masks:    (N, H, W) float32 binary

    Returns:
        volume: (N, H, W) float32 [0,1]
    """
    N, H, W = masks.shape
    assert H % 2 == 0 and W % 2 == 0, f"H,W は偶数が必要 (got {H},{W})"

    A = build_measurement_matrix(masks)  # (4, N)

    # 各ピクセル位置を明示的スライスで取り出す（reshape 並び順ミスを回避）
    y0 = snapshot[0::2, 0::2]  # (H//2, W//2) ← pixel (0,0)
    y1 = snapshot[0::2, 1::2]  # ← pixel (0,1)
    y2 = snapshot[1::2, 0::2]  # ← pixel (1,0)
    y3 = snapshot[1::2, 1::2]  # ← pixel (1,1)

    n_blocks = (H // 2) * (W // 2)
    y_flat = np.stack([
        y0.ravel(), y1.ravel(), y2.ravel(), y3.ravel()
    ], axis=0)  # (4, n_blocks)
    assert y_flat.shape == (4, n_blocks), f"y_flat shape mismatch: {y_flat.shape}"

    # Solve A/N @ V = y for all blocks at once
    # lstsq: A_norm (4,N), y_flat (4, n_blocks) → V_all (N, n_blocks)
    A_norm = A / N  # (4, N)
    V_all, _, _, _ = np.linalg.lstsq(A_norm, y_flat, rcond=None)  # (N, n_blocks)
    V_all = np.clip(V_all, 0.0, 1.0)  # (N, n_blocks)

    # Reshape → (N, H//2, W//2) → upsample 2× → (N, H, W)
    V_blocks = V_all.reshape(N, H // 2, W // 2)
    volume = np.repeat(np.repeat(V_blocks, 2, axis=1), 2, axis=2)

    return volume


def analyze_matrix(A):
    """SVD による測定行列の条件数・ランク解析。"""
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    sigma_max = sv[0]
    sigma_min_nonzero = sv[sv > SVD_COND_THRESH]
    cond = sigma_max / sigma_min_nonzero[-1] if len(sigma_min_nonzero) > 0 else float('inf')
    rank = int(np.sum(sv > SVD_RANK_THRESH))
    return sv, cond, rank


def compute_metrics(volume, gt):
    N = volume.shape[0]
    psnr_vals, ssim_vals = [], []
    for s in range(N):
        p = psnr_fn(gt[s], volume[s], data_range=1.0)
        s_val = ssim_fn(gt[s], volume[s], data_range=1.0)
        psnr_vals.append(p)
        ssim_vals.append(s_val)
    return psnr_vals, ssim_vals


def main():
    parser = argparse.ArgumentParser(
        description='Oracle 復元（A-5 Step2）: 最小二乗・PSF 無視・ブロック均一近似'
    )
    parser.add_argument('-s', '--snapshot', required=True, help='Snapshot PNG パス')
    parser.add_argument('-sm', '--snapshot-masks', required=True, help='Masks .mat パス')
    parser.add_argument('-gt', '--ground-truth', required=True, help='GT スライスディレクトリ')
    parser.add_argument('-o', '--output', default='experiments/oracle_output', help='出力ディレクトリ')
    parser.add_argument('--baseline-psnr', type=float, default=None,
                        help='比較用 baseline DIP の PSNR (dB)。指定時のみ差分・判定を表示')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading snapshot : {args.snapshot}")
    snapshot = load_snapshot(args.snapshot)
    H, W = snapshot.shape
    print(f"  shape: {snapshot.shape}, range [{snapshot.min():.3f}, {snapshot.max():.3f}]")

    print(f"Loading masks    : {args.snapshot_masks}")
    masks = load_masks(args.snapshot_masks)
    N = masks.shape[0]
    print(f"  shape: {masks.shape}")

    print(f"Loading GT       : {args.ground_truth}")
    gt = load_gt(args.ground_truth)
    print(f"  shape: {gt.shape}")

    # 測定行列解析
    A = build_measurement_matrix(masks)  # (4, N)
    sv, cond, rank = analyze_matrix(A)
    print(f"\nMeasurement matrix A ({A.shape[0]}×{A.shape[1]}):")
    print(f"  Singular values : {np.round(sv, 4).tolist()}")
    print(f"  Condition number: {cond:.4f}  (σmax/σmin, thresh={SVD_COND_THRESH})")
    print(f"  Rank            : {rank}  (thresh={SVD_RANK_THRESH})")

    # Oracle 復元
    print("\nRunning oracle reconstruction...")
    volume = oracle_reconstruct(snapshot, masks)
    print(f"  Volume shape: {volume.shape}, range [{volume.min():.3f}, {volume.max():.3f}]")

    # PSNR / SSIM
    psnr_vals, ssim_vals = compute_metrics(volume, gt)
    mean_psnr = float(np.mean(psnr_vals))
    mean_ssim = float(np.mean(ssim_vals))
    print(f"\nResults:")
    print(f"  Mean PSNR: {mean_psnr:.2f} dB")
    print(f"  Mean SSIM: {mean_ssim:.4f}")

    if args.baseline_psnr is not None:
        diff = mean_psnr - args.baseline_psnr
        print(f"\n  Baseline DIP : {args.baseline_psnr:.2f} dB")
        print(f"  Oracle       : {mean_psnr:.2f} dB")
        print(f"  Difference   : {diff:+.2f} dB")
        if diff > 2.0:
            print("  → Oracle >> baseline: DIP 側に改善余地あり → A-3 (Early Stopping) を優先")
        elif diff < -1.0:
            print("  → Oracle < baseline: rank-4 ボトルネック or 近似誤差が主因 → A-4 / Phase B を検討")
        else:
            print("  → Oracle ≈ baseline: 測定行列が原理的な上限に近い → A-4 / Phase B が本質")

    # 保存
    np.save(os.path.join(args.output, 'oracle_volume.npy'), volume)

    for s in range(N):
        img_arr = (volume[s] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_arr).save(
            os.path.join(args.output, f'oracle_slice_{s:02d}.png')
        )

    result_path = os.path.join(args.output, 'result.txt')
    with open(result_path, 'w') as f:
        f.write("Oracle Reconstruction (A-5 Step2) - Approximate Oracle\n")
        f.write("Assumptions: no PSF, block-uniform (2x2 super-pixel)\n")
        f.write(f"Snapshot : {args.snapshot}\n")
        f.write(f"Masks    : {args.snapshot_masks}\n")
        f.write(f"GT       : {args.ground_truth}\n\n")
        f.write(f"Measurement Matrix A ({A.shape[0]}x{A.shape[1]}):\n")
        f.write(f"  Singular values : {np.round(sv, 4).tolist()}\n")
        f.write(f"  Condition number: {cond:.4f}  (thresh={SVD_COND_THRESH})\n")
        f.write(f"  Rank            : {rank}  (thresh={SVD_RANK_THRESH})\n\n")
        f.write("Per-slice metrics:\n")
        for s in range(N):
            f.write(f"  slice {s:02d}: PSNR={psnr_vals[s]:.2f} dB  SSIM={ssim_vals[s]:.4f}\n")
        f.write(f"\nMean PSNR: {mean_psnr:.2f} dB\n")
        f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
        if args.baseline_psnr is not None:
            diff = mean_psnr - args.baseline_psnr
            f.write(f"\nBaseline DIP: {args.baseline_psnr:.2f} dB\n")
            f.write(f"Difference (Oracle - baseline): {diff:+.2f} dB\n")

    print(f"\nSaved to {args.output}/")
    print(f"  oracle_volume.npy, oracle_slice_*.png, result.txt")


if __name__ == '__main__':
    main()
