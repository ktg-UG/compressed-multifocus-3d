"""学習済みパターンの初期↔学習後を可視化する。

使用例:
  python3 phase_b/visualize_pattern.py experiments/phase_b_m2/learned_pattern.pt
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ascii_block(p):
    """2x2 を ■(開)/□(閉) で表す。"""
    sym = {1.0: "■", 0.0: "□"}
    rows = ["".join(sym[float(v)] for v in row) for row in p]
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase_b_m2/learned_pattern.pt"
    ckpt = torch.load(path, map_location="cpu")
    init = np.rint(ckpt["init_binary"].numpy())    # (D, b, b) STE 残差を丸める
    final = np.rint(ckpt["final_binary"].numpy())  # (D, b, b)
    D = init.shape[0]

    print(f"=== {path} ===")
    print("層 | 初期(open率) | 学習後(open率) | 変化")
    flipped_total = 0
    for s in range(D):
        i_rows, f_rows = ascii_block(init[s]), ascii_block(final[s])
        i_open, f_open = init[s].mean(), final[s].mean()
        flips = int((init[s] != final[s]).sum())
        flipped_total += flips
        print(f"L{s:02d} | {i_rows[0]} {f_rows[0]}   "
              f"init={i_open:.2f} -> learned={f_open:.2f}  flips={flips}")
        print(f"    | {i_rows[1]} {f_rows[1]}")
    print(f"\n全開口率: init={init.mean():.3f} -> learned={final.mean():.3f}")
    print(f"反転セル合計: {flipped_total}/{D*4} ({100*flipped_total/(D*4):.1f}%)")
    print(f"全0層: init={int((init.sum((1,2))==0).sum())}, "
          f"learned={int((final.sum((1,2))==0).sum())}")
    print(f"全1層: init={int((init.sum((1,2))==4).sum())}, "
          f"learned={int((final.sum((1,2))==4).sum())}")

    # 図: 上段=初期, 下段=学習後
    fig, axes = plt.subplots(2, D, figsize=(D * 1.1, 2.6))
    for s in range(D):
        axes[0, s].imshow(init[s], cmap="gray", vmin=0, vmax=1)
        axes[0, s].set_title(f"L{s}", fontsize=8)
        axes[1, s].imshow(final[s], cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, s].set_xticks([]); axes[r, s].set_yticks([])
    axes[0, 0].set_ylabel("init", fontsize=10)
    axes[1, 0].set_ylabel("learned", fontsize=10)
    fig.suptitle("Phase B pattern: init (top) vs learned (bottom)  [white=open]")
    fig.tight_layout()
    out = path.replace(".pt", "_compare.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n図を保存: {out}")


if __name__ == "__main__":
    main()
