#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a snapshot image by multiplexing a stack of multi-focal images with coded masks.

- Reads N images from --image-dir (default: ./image)
- Generates N masks (random overlapped or disjoint partition)
- Forms a single snapshot: S = sum_i I_i * M_i, optionally normalized by mask sum
- Saves:
  * snapshot.png        : 8-bit grayscale snapshot
  * snapshot.npy        : float32 snapshot array (H, W)
  * masks.mat           : 'ExpPtn' with shape (H, W, N)
  * mapping.json        : filenames list and parameters
  * optional previews   : preview.png (masks grid + snapshot)

This script aims to mirror the measurement model used in the notebook's Exposuref:
  meas = mean_t(x * mask_b)  ~  sum_i(I_i * M_i) / sum_i(M_i)

Dependencies: numpy, scipy, PIL or OpenCV (prefer PIL; falls back to cv2 if available).

Example usage:
  # Basic usage with random masks
  python sensing/snapshot.py --image-dir open_scivis_128x128/aneurism_0/image \
                             --output-dir outputs/snapshot_random \
                             --mask-type random --seed 0
  
  # Using disjoint masks (recommended for reconstruction)
  python sensing/snapshot.py --image-dir open_scivis_128x128/aneurism_0/slice \
                             --output-dir outputs/snapshot_disjoint_correct \
                             --mask-type disjoint --seed 0
  
  # Random masks with custom duty cycle
  python sensing/snapshot.py --image-dir open_scivis_128x128/backpack_0/slice \
                             --output-dir outputs/snapshot_backpack \
                             --mask-type random --duty-cycle 0.3 --seed 42
  
  # Without normalization
  python sensing/snapshot.py --image-dir open_scivis_128x128/aneurism_0/slice \
                             --output-dir outputs/snapshot_no_norm \
                             --mask-type disjoint --no-normalize
  
  # Limit number of images used
  python sensing/snapshot.py --image-dir open_scivis_128x128/aneurism_0/slice \
                             --output-dir outputs/snapshot_limited \
                             --mask-type disjoint --limit 5
"""

import os
import json
import argparse
from typing import List, Tuple

import numpy as np
try:
    from scipy import io as scio  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    scio = None  # type: ignore
    SCIPY_AVAILABLE = False

# Try to use PIL for broad availability; fall back to cv2 if PIL isn't present
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    try:
        import cv2  # type: ignore
        CV2_AVAILABLE = True
    except Exception:
        CV2_AVAILABLE = False


def read_grayscale_image(path: str) -> np.ndarray:
    """Read image file as grayscale float32 in [0,1]."""
    if PIL_AVAILABLE:
        img = Image.open(path).convert('L')  # grayscale
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr
    elif CV2_AVAILABLE:
        arr8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr8 is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return (arr8.astype(np.float32) / 255.0)
    else:
        raise RuntimeError("Neither Pillow (PIL) nor OpenCV (cv2) is available to read images.")


def resize_image(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize grayscale image array to (H, W) using bilinear interpolation."""
    H, W = size
    if PIL_AVAILABLE:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode='L')
        img = img.resize((W, H), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0
    elif CV2_AVAILABLE:
        arr8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        out = cv2.resize(arr8, (W, H), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32) / 255.0
    else:
        raise RuntimeError("Neither Pillow (PIL) nor OpenCV (cv2) is available to resize images.")


def collect_images(image_dir: str) -> List[str]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = [f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in exts]
    files.sort()
    return [os.path.join(image_dir, f) for f in files]


def ensure_same_size(images: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    sizes = [(im.shape[0], im.shape[1]) for im in images]
    if len(set(sizes)) == 1:
        H, W = sizes[0]
        return images, (H, W)
    # Auto-resize all to smallest H and W among the set
    minH = min(s[0] for s in sizes)
    minW = min(s[1] for s in sizes)
    resized = [resize_image(im, (minH, minW)) for im in images]
    return resized, (minH, minW)


def make_masks_random(n: int, H: int, W: int, duty: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = (rng.random((n, H, W)) < float(duty)).astype(np.float32)
    return masks


def make_masks_disjoint(n: int, H: int, W: int, seed: int) -> np.ndarray:
    """Partition pixels among n channels so that sum of masks equals 1 everywhere."""
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, n, size=(H, W), endpoint=False)
    masks = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        masks[i] = (assign == i).astype(np.float32)
    return masks


def compose_snapshot(images: np.ndarray, masks: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    images: (N, H, W) in [0,1]
    masks : (N, H, W) float in {0,1}
    Returns snapshot (H,W) and mask_sum (H,W).
    """
    assert images.shape == masks.shape, f"Shape mismatch images {images.shape} vs masks {masks.shape}"
    meas = np.sum(images * masks, axis=0)  # (H, W)
    mask_sum = np.sum(masks, axis=0)       # (H, W)
    if normalize:
        # Avoid divide-by-zero
        mask_sum_safe = np.where(mask_sum > 0, mask_sum, 1.0)
        meas = meas / mask_sum_safe
    # Clip to [0,1]
    meas = np.clip(meas, 0.0, 1.0)
    return meas.astype(np.float32), mask_sum.astype(np.float32)


def save_png_gray(path: str, arr01: np.ndarray) -> None:
    arr8 = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    if PIL_AVAILABLE:
        Image.fromarray(arr8, mode='L').save(path)
    elif CV2_AVAILABLE:
        cv2.imwrite(path, arr8)
    else:
        raise RuntimeError("Neither Pillow (PIL) nor OpenCV (cv2) is available to save images.")


def make_preview(masks: np.ndarray, snapshot: np.ndarray, out_path: str, max_cols: int = 6) -> None:
    """Create a simple preview grid of masks and the snapshot (requires PIL)."""
    if not PIL_AVAILABLE:
        return
    n, H, W = masks.shape
    scale = 1  # no scale; keep original size
    cell_w, cell_h = W * scale, H * scale
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols)) + 1  # extra row for snapshot
    gap = 4
    grid_w = cols * cell_w + (cols + 1) * gap
    grid_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new('L', (grid_w, grid_h), color=255)

    # paste masks
    for i in range(n):
        r = i // cols
        c = i % cols
        x0 = gap + c * (cell_w + gap)
        y0 = gap + r * (cell_h + gap)
        tile = (masks[i] * 255.0).astype(np.uint8)
        tile_img = Image.fromarray(tile, mode='L')
        canvas.paste(tile_img, (x0, y0))

    # paste snapshot at the last row, first column spanning 2 cells if possible
    snap8 = (snapshot * 255.0).astype(np.uint8)
    snap_img = Image.fromarray(snap8, mode='L')
    y0 = gap + (rows - 1) * (cell_h + gap)
    x0 = gap
    # Span across min(2, cols) cells
    span_cols = min(2, cols)
    snap_w = span_cols * cell_w + (span_cols - 1) * gap
    snap_img = snap_img.resize((snap_w, cell_h), resample=Image.BILINEAR)
    canvas.paste(snap_img, (x0, y0))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Create a coded snapshot from multi-focal images")
    parser.add_argument('--image-dir', type=str, default='./image', help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, default='./snapshot_output', help='Directory to save outputs')
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'disjoint'], help='Mask generation strategy')
    parser.add_argument('--duty-cycle', type=float, default=0.5, help='Probability of 1s for random masks (0-1)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--no-normalize', action='store_true', help='Disable normalization by mask sum')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images used (0 = use all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filepaths = collect_images(args.image_dir)
    if args.limit and args.limit > 0:
        filepaths = filepaths[:args.limit]

    if len(filepaths) == 0:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    print(f"Found {len(filepaths)} images.")

    # Load images
    imgs = [read_grayscale_image(p) for p in filepaths]
    imgs, (H, W) = ensure_same_size(imgs)
    n = len(imgs)
    stack = np.stack(imgs, axis=0).astype(np.float32)  # (N, H, W)

    # Masks
    if args.mask_type == 'random':
        masks = make_masks_random(n, H, W, duty=args.duty_cycle, seed=args.seed)
    else:
        masks = make_masks_disjoint(n, H, W, seed=args.seed)

    snapshot, mask_sum = compose_snapshot(stack, masks, normalize=(not args.no_normalize))

    # Save outputs
    snap_png = os.path.join(args.output_dir, 'snapshot.png')
    snap_npy = os.path.join(args.output_dir, 'snapshot.npy')
    masks_mat = os.path.join(args.output_dir, 'masks.mat')
    mapping_json = os.path.join(args.output_dir, 'mapping.json')
    preview_png = os.path.join(args.output_dir, 'preview.png')

    save_png_gray(snap_png, snapshot)
    np.save(snap_npy, snapshot)

    # Save masks as (H, W, N) to match typical 'ExpPtn' layout
    masks_hwk = np.transpose(masks, (1, 2, 0))  # (H, W, N)
    if SCIPY_AVAILABLE:
        scio.savemat(masks_mat, {'ExpPtn': masks_hwk.astype(np.float32)}, do_compression=True)
    else:
        # Fallback: save as NPZ if SciPy is unavailable
        np.savez_compressed(os.path.splitext(masks_mat)[0] + '.npz', ExpPtn=masks_hwk.astype(np.float32))

    meta = {
        'image_dir': os.path.abspath(args.image_dir),
        'output_dir': os.path.abspath(args.output_dir),
        'files': [os.path.relpath(p, args.image_dir) for p in filepaths],
        'H': int(H),
        'W': int(W),
        'N': int(n),
        'mask_type': args.mask_type,
        'duty_cycle': float(args.duty_cycle),
        'seed': int(args.seed),
        'normalized': not args.no_normalize,
        'stats': {
            'snapshot_min': float(snapshot.min()),
            'snapshot_max': float(snapshot.max()),
            'mask_sum_min': float(mask_sum.min()),
            'mask_sum_max': float(mask_sum.max()),
        }
    }
    with open(mapping_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Optional preview
    try:
        make_preview(masks, snapshot, preview_png)
    except Exception as e:
        print(f"Preview not created: {e}")

    print(f"Saved snapshot to: {snap_png}")
    print(f"Saved masks to:    {masks_mat}")
    print(f"Saved mapping to:  {mapping_json}")


if __name__ == '__main__':
    main()
