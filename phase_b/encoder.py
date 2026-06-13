"""学習可能シャッターパターン（Encoder）と圧縮シミュレーション（B-1）。

- 2×2 タイリングの D 層パターン = D*4 個の Bernoulli パラメータを学習する
- Straight-Through Estimator (STE) で hard binarize（順伝播は 0/1、逆伝播は probs へ）
- PSF 不要: snapshot = Σ_s (mask_s ⊙ focal_s) / n（改善案）
"""

import torch
import torch.nn as nn


class LearnablePattern(nn.Module):
    """2×2 タイリングの学習可能 binary パターン。

    Args:
        depth: 層数 D（= 多焦点画像枚数, 既定 11）
        block: タイリングブロックサイズ（既定 2）
    """

    def __init__(self, depth=11, block=2):
        super().__init__()
        self.depth = depth
        self.block = block
        # logits ∈ R^{D, block, block}
        self.logits = nn.Parameter(torch.randn(depth, block, block))

    def probs(self):
        return torch.sigmoid(self.logits)  # (D, b, b)

    def binary_pattern(self):
        """STE で 2 値化した (D, b, b) パターンを返す（勾配は probs に流れる）。"""
        probs = self.probs()
        hard = (probs > 0.5).float()
        return hard + probs - probs.detach()

    def tiled_masks(self, height, width):
        """(D, b, b) を (D, H, W) にタイリングして返す。"""
        pat = self.binary_pattern()  # (D, b, b)
        reps_h = (height + self.block - 1) // self.block
        reps_w = (width + self.block - 1) // self.block
        tiled = pat.repeat(1, reps_h, reps_w)[:, :height, :width]
        return tiled  # (D, H, W)


def simulate_snapshot(focal_stack, masks):
    """PSF なしの圧縮シミュレーション。

    Args:
        focal_stack: (B, D, H, W) 多焦点画像
        masks: (D, H, W) タイリング済み binary マスク
    Returns:
        snapshot: (B, 1, H, W)
    """
    d = focal_stack.shape[1]
    masked = focal_stack * masks.unsqueeze(0)  # (B, D, H, W)
    snapshot = masked.sum(dim=1, keepdim=True) / d  # (B, 1, H, W)
    return torch.clamp(snapshot, 0.0, 1.0)


def aperture_penalty(probs, target=0.5):
    """開口率を target に近づけるソフト制約。"""
    return (probs.mean() - target) ** 2


def diversity_penalty(probs):
    """各層が全 0 / 全 1 に潰れるのを防ぐ。

    層ごとの開口セル数 open∈[0, b*b] を [0.5, b*b-0.5] に収める。
    """
    block_cells = probs.shape[1] * probs.shape[2]
    open_per_slice = probs.sum(dim=(1, 2))  # (D,)
    too_closed = torch.relu(0.5 - open_per_slice)
    too_open = torch.relu(open_per_slice - (block_cells - 0.5))
    return (too_closed + too_open).mean()
