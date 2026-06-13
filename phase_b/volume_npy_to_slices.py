"""DIP 復元 volume(.npy) を、main.py の -gt として使えるクリーンな PNG スライス群に変換する。

trainer.save_results は volume を *_volume.npy で保存するが、スライス画像は matplotlib 経由で
padding/dpi が入りピクセル不正確。-gt ローダ（cv2 grayscale）には素のスライスが必要なため、
.npy から 128x128 等の素の uint8 PNG を書き出す。

使用例:
  python3 phase_b/volume_npy_to_slices.py \
    outputs/cell_mf_0005/dip_final_volume.npy \
    -o inputs/pseudo_gt/cell/0005
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume_npy", help="*_volume.npy（shape (D,H,W)）")
    ap.add_argument("-o", "--output", required=True, help="スライス出力ディレクトリ")
    args = ap.parse_args()

    vol = np.load(args.volume_npy)
    if vol.ndim == 5:
        vol = vol[0, 0]
    elif vol.ndim == 4:
        vol = vol[0]
    assert vol.ndim == 3, f"expected (D,H,W), got {vol.shape}"

    os.makedirs(args.output, exist_ok=True)
    vol = np.clip(vol, 0.0, 1.0)
    for s in range(vol.shape[0]):
        img = (vol[s] * 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(os.path.join(args.output, f"{s:02d}.png"))

    print(f"[volume_npy_to_slices] {args.volume_npy} {vol.shape} "
          f"-> {vol.shape[0]} slices in {args.output}")


if __name__ == "__main__":
    main()
