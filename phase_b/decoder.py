"""Decoder: snapshot 1 枚から多焦点 D 枚を復元する軽量 2D U-Net（B-2）。

M1（最小スパイク）では「2D→3D」フル 3D conv の前段として、
2D U-Net で D チャネル出力（= D 枚の焦点画像）を予測するシンプル版を採用する。
パターン探索の代理器として十分軽量で、loss 低下・pattern 変化の確認に向く。
出力は sigmoid で alpha∈[0,1] を保証。
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SnapshotDecoder(nn.Module):
    """snapshot (B,1,H,W) → 多焦点 (B,D,H,W)。

    Args:
        out_depth: 出力層数 D（= 多焦点画像枚数）
        base: ベースチャネル数
    """

    def __init__(self, out_depth=11, base=32):
        super().__init__()
        self.enc1 = _conv_block(1, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_depth, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))  # (B, D, H, W) in [0,1]
