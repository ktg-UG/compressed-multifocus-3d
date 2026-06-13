"""Phase B: Deep Sensing によるシャッターパターン最適化（方式 A）。

改善案ベース（phase_b_cell_pattern_optimization.md）:
- 多焦点画像 11 枚をそのまま GT として使う（volume / PSF 不要）
- snapshot = Σ_s (mask_s ⊙ focal_s) / n
- Decoder は snapshot → 多焦点 11 枚を復元
- Encoder(学習可能パターン) と Decoder を end-to-end 共最適化
"""
