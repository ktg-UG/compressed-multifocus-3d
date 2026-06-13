# Compressed Multi-focus 3D Reconstruction

> 細胞の3次元構造を「11枚の多焦点画像」ではなく **「1枚の圧縮画像」** から復元する。データ量を約91%削減しても形状類似度（Dice）は **99.6%** 維持。

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.3-76B900?logo=nvidia&logoColor=white)
![Deep Image Prior](https://img.shields.io/badge/Method-Deep_Image_Prior-blue)
![3D U-Net](https://img.shields.io/badge/Model-3D_U--Net-blueviolet)

---

## 📌 プロジェクト概要

光学顕微鏡で細胞の3次元構造を観察するには、焦点位置を変えながら **11枚程度の多焦点画像** を逐次撮影する必要があり、データ量と撮影時間がボトルネックだった。本研究はこの課題に対し、**画素別シャッタによる深度方向の符号化** で全11枚の情報を **1枚の圧縮画像** に畳み込み、深層学習（3D U-Net + Deep Image Prior）で元の3次元構造を復元するパイプラインを提案・実装した。

ベースとなる多焦点復元の実装（Azevedo et al., MICCAI 2022）に **圧縮イメージングモデル `MaskedImagingModel`** を新規追加し、入力データ量を約91%削減しながら、形状の忠実度を示す Dice Score 0.97 をほぼ維持することに成功している。

---

## 🎯 開発背景・動機

研究の出発点は、指導教員と議論する中で見えてきた以下の課題感だった。

| 課題 | 内容 |
|------|------|
| **データ量の増大** | 1サンプルあたり 11枚 × 128×128 で約176kB、高解像度化すれば容易に膨れ上がる |
| **撮影時間の増加** | 焦点位置を都度移動させる必要があり、高速に動く細胞の観察に不向き |
| **スクリーニングの困難さ** | 大量サンプルを処理するには逐次撮影が致命的なボトルネックになる |

一方で、別領域の **ビデオ圧縮センシング（ビデオCS）** では、画素別シャッタで「時間軸」を1枚の画像に符号化し、後段の最適化で高フレームレート動画を復元する技術が確立しつつあった。

> 「時間軸を1枚に圧縮できるなら、**深度軸も1枚に圧縮できるはず**だ」

この仮説を出発点に、ビデオCSの概念を顕微鏡イメージングの深度軸へ拡張し、深層学習で逆問題を解く本研究をスタートさせた。第1段階（卒業論文）でランダムシャッタによる圧縮復元を実証し、現在は第2段階（修士論文）として **シャッタパターン自体を学習で最適化する Deep Sensing** に取り組んでいる。

---

## 🧠 提案手法のコア（研究上の新規貢献）

- **`MaskedImagingModel`（`img_model.py`）の新規実装**
  ベース実装の `ImagingModel`（多焦点画像11枚を生成）に対し、PSF光学モデルと画素別シャッタ符号化を組み合わせた **ハイブリッド順方向モデル** を構築した
- **圧縮画像1枚 → 3D U-Net (DIP) → 復元構造 のパイプライン設計**
  教師データ不要のDeep Image Priorを使い、1サンプルごとに最適化することで、3D GTデータが少ない生体イメージング領域に適合させた
- **ハードウェア制約をネットワーク構造に内包（第2段階）**
  実カメラの仕様（2×2ブロックタイリング、バイナリ制御、全閉/全開禁止）を学習対象に含め、勾配法で最適パターンを獲得する Deep Sensing を実装中
- **手法の比較実験基盤**
  DIP / 反復最適化 / Neural Field の3トレーナーを統一インターフェースで切り替え可能にし、Open-Scivis 13サンプル + 実細胞データで横断評価できる環境を整備

---

## 🧑‍💻 自身の担当工程・実装範囲

研究室の既存ベース実装（Azevedo et al. の多焦点復元コード）を出発点に、**圧縮イメージング化と Deep Sensing 拡張は単独で設計・実装** している。

### 担当工程

- [x] **問題定式化**: ビデオCSの概念を深度軸へ拡張する数理モデルの設計
- [x] **順方向モデル設計**: PSF光学モデル × シャッタ符号化の合成モデル定義
- [x] **実装（深層学習）**: 3D U-Net + DIP の学習スキーム実装、トレーナー基盤整備
- [x] **実装（データ生成）**: ブロックランダムマスクによる合成圧縮画像生成スクリプト群
- [x] **合成・実データ実験**: Open-Scivis 13サンプル + NanoZoomer 実細胞での評価
- [x] **評価・考察**: RMSE / Dice / SSIM の定量比較、定性比較
- [ ] **Phase B（進行中）**: Deep Sensing による学習可能シャッタパターンの最適化
- [ ] **実機検証（未着手）**: 新カメラでの実撮影と復元精度検証

### 具体的な実装ファイル（自身がコード作成・改変したもの）

| ファイル/ディレクトリ | 内容 |
|---|---|
| `img_model.py` の `MaskedImagingModel` | 提案手法の **順方向モデル本体**。PSF + シャッタ符号化の合成 |
| `my-program/create_compressed_inputs/` | ブロックランダムマスク生成、2×2 / 4×4 / 8×8 タイリング、全閉/全開除外 |
| `my-program/snapshot.py` / `random_mask.py` | 圧縮画像・マスク生成のユーティリティ |
| `my-program/forward_only_test.py` / `oracle_reconstruction.py` | 順方向モデル単体検証・上限性能評価 |
| `my-program/visualize/` | 合成データ・実細胞データの 3D 可視化と GIF 生成 |
| `trainers/dip_trainer.py`（部分改修） | 圧縮入力モード追加、保存間隔・PSF マスク対応 |
| `phase_b/` 一式 | Phase B（修論）の **Deep Sensing 学習基盤**。`encoder.py` でシャッタパターン学習層、`decoder.py` で復元、`train_deep_sensing.py` で end-to-end 訓練 |

### 流用元（自身の貢献ではない部分の明示）

- `ImagingModel` の物理光学計算、`models/skipnet3d.py` の 3D U-Net 構造、`utils/lighting.py` の光学パラメータ補助は Azevedo et al. (MICCAI 2022) のベース実装から流用
- `models/network_dncnn.py` は DnCNN の参考実装そのまま

---

## 🛠 使用技術・アーキテクチャ

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.9 |
| 深層学習 | PyTorch 1.13, torchvision |
| モデル | 3D U-Net (Skip Network), Deep Image Prior, Neural Field, DnCNN |
| 数値計算 | NumPy, SciPy |
| 画像処理 | OpenCV, scikit-image, Pillow |
| 可視化・実験管理 | matplotlib, TensorBoard |
| GPU | CUDA 11.3 / NVIDIA RTX A2000 (12GB) |
| 環境管理 | conda (environment.yml) |
| 設定管理 | PyYAML（光学パラメータの外部定義） |

### AIモデル・手法

| 区分 | 内容 |
|------|------|
| 復元ネットワーク | **3D U-Net (Skip Network)**（`models/skipnet3d.py`、Azevedo準拠の構造） |
| 学習スキーム | **Deep Image Prior (DIP)** — 教師データ不要、各サンプルで重み θ を個別最適化 |
| 比較手法 | 反復最適化（HQS / TV正則化）、Neural Field 復元 |
| 事前学習 | 楕円球データ + ガウスノイズで教師ありデノイジング（500 iter） |
| Phase B 提案 | Deep Sensing — 順伝播=バイナリ・逆伝播=連続のスキームでシャッタパターンを勾配最適化 |
| 損失 | L1 ピクセル損失（再構成像 vs 観測圧縮画像） + 3D TV 正則化 |
| 最適化器 | Adam (lr = 1e-3), 20,000 iteration |

### アーキテクチャ（提案手法のデータフロー）

```
[3D GT volume α]            ← 合成データ（評価時のみ参照）
      │
      ▼
┌──────────────────────────────────────────────┐
│  Forward (MaskedImagingModel) ★自身の実装   │
│   1. PSF 畳み込み（光線減衰モデル）           │
│   2. 各深度 z にシャッタマスク S_z を適用    │
│   3. 深度方向に積算 → 1枚の圧縮画像 Î       │
└──────────────────────────────────────────────┘
      │
      ▼  圧縮画像 Î (16 kB)
      ▼
┌──────────────────────────────────────────────┐
│  Reconstruction (DIP)                        │
│   固定ノイズ n → 3D U-Net fθ → α̂           │
│   Loss = || Î - Forward(α̂, S) ||₁ + λ·TV3D   │
│   θ を Adam で 20,000 iter 更新              │
└──────────────────────────────────────────────┘
      │
      ▼
[推定 3D 構造 α̂]  → RMSE / Dice / SSIM で評価

【Phase B（修論）】
  S（シャッタパターン）自体を encoder.py で
  学習可能パラメータ化し、α と同時最適化
```

### 実装方法のポイント

- **物理光学を温存した順方向モデル**: PSF + 光線減衰モデルを `ImagingModel` から流用しつつ、その上にシャッタ符号化を載せる構造にし、実機との整合性とデバッグ容易性を両立
- **教師データ不要のDIP採用**: 細胞 3D GT が大量入手困難な領域に合わせ、各サンプル個別最適化で汎化問題を回避
- **PyTorch 自動微分との整合**: バイナリマスク適用後も勾配が流れるよう、`MaskedImagingModel` の forward を全て微分可能演算で構成
- **conda + YAML 設定**: 光学パラメータ（NA, 解像度, 層数）を `config.yaml` に外出しし、再現性と実験管理を担保

---

## 🔍 工夫した点・技術的チャレンジ

### ① ビデオCSの概念を「深度軸」へ拡張する順方向モデルの定式化

**課題：** ビデオCSは電子シャッタの **線形** 積算で済むが、本手法は **PSF（点拡がり関数）の畳み込み × 光線減衰モデル `l'_j = l_j Π α_i^d_ji` × シャッタ符号化** という非線形パイプラインになる。理論はビデオCSから借りられても、実装時の自由度が大きく、ベース実装の `ImagingModel` を踏襲しながら整合性を保つのは難所だった。

**アプローチ：** Yamaguchi (PSIVT 2020) の物理光学モデルを `ImagingModel` として温存し、その上に **シャッタマスクの深度方向積算** を追加する設計を採用した。物理的に意味のある中間表現（焦点画像）を経由するため、デバッグ・実機検証時に各段階の妥当性を確認できる構造になっている。

**結果：** PyTorchの自動微分と互換性を保ったまま、各深度に独立のバイナリマスクを掛けて積算するモデルを実装。第2段階での Deep Sensing にもそのまま流用できる拡張性を確保した。

### ② 教師データ不要のDIP（Deep Image Prior）採用

**課題：** 細胞の3D GTデータは大量入手が困難で、教師あり学習では汎化が難しい。

**アプローチ：** **固定ランダムノイズ n → 3D U-Net fθ → 推定構造 α̂ = fθ(n)** という DIP の枠組みを採用し、各サンプルごとにネットワーク重み θ のみを更新する設計にした。さらに楕円球データでの事前学習（500イテレーション）を組み合わせ、収束速度と最終精度の両方を改善した。

**結果：** 20000ステップで安定収束。3D GTを持たない実細胞データに対しても、定性的に細胞境界を明瞭に再現できることを確認した。

### ③ ハードウェア制約を学習に組み込む（修士論文部分）

**課題：** 第2段階で最適化したいのは「シャッタパターン」だが、実カメラには **2×2ブロック単位での制御**、**1タップ=バイナリ**、**同一フレーム内タイリング**、**全閉/全開禁止** という制約がある。これらを無視して最適化すると、シミュレーション上は良くても実機に載らない。

**アプローチ：** Yoshida et al. (Sensors 2023) のバイナリ勾配法を参考に、**順伝播ではバイナリ重み、逆伝播では連続重み** を使うスキームを実装。さらに 2×2 タイリング・パターン禁止集合の制約をネットワーク構造側に組み込み、最適化空間を実機制約に閉じ込めた。

**結果：** Phase B として `train_deep_sensing.py` / `encoder.py` / `decoder.py` を整備し、学習可能なシャッタパターンの初期実装を完了。現在、ベースライン（ランダムパターン）に対する優位性を検証中である。

---

## 📊 成果（定量）

合成データ（Open-Scivis 13サンプル、128×128×11）での評価：

| 手法 | 入力 | 平均 RMSE↓ | 平均 Dice↑ | 平均 SSIM↑ |
|------|------|-----------|-----------|-----------|
| 従来手法（多焦点11枚） | 176 kB | 0.0070 | 0.9679 | 0.9896 |
| **提案手法（圧縮1枚）** | **16 kB** | 0.0125 | **0.9664** | 0.9799 |

**入力データ量を約91%削減 (176kB → 16kB) しながら、Dice Score は 99.6%（0.9679 → 0.9664）を維持** している。

代表サンプル別の比較：

| データセット | RMSE（従来） | RMSE（提案） | Dice（従来） | Dice（提案） |
|------------|-------------|-------------|-------------|-------------|
| aneurism | 0.0035 | 0.0117 | 0.9945 | 0.9944 |
| beechnut | 0.0048 | 0.0070 | 0.9788 | 0.9799 |
| chameleon | 0.0070 | 0.0135 | 0.9532 | 0.9574 |
| pancreas | 0.0049 | 0.0093 | 0.9325 | 0.9323 |

実データ（浜松ホトニクス NanoZoomer S60v2 で撮影した細胞の多焦点画像）に対しても、細胞境界の明瞭さ・滑らかな曲面構造を高忠実度で再現できることを定性的に確認した。

---

## 📁 ディレクトリ構成

```
compressed-3d-portfolio/
├── main.py                    # エントリポイント（引数解析・モード切替）
├── img_model.py               # ImagingModel / MaskedImagingModel ★提案手法の核心
├── trainer.py                 # 基底トレーナー
├── total_variation_3d.py      # 3D TV 正則化
├── config.yaml                # 光学パラメータ（NA, 解像度, 層数）
├── environment.yml            # conda 環境定義
├── trainers/                  # DIP / HQS / Iter / Neural Field トレーナー
│   ├── dip_trainer.py         # ★主要：DIP トレーナー
│   ├── hqs_trainer.py
│   ├── iter_trainer.py
│   └── nf_trainer.py
├── models/
│   ├── skipnet3d.py           # 3D U-Net (Skip Network)
│   ├── neural_field.py        # Neural Field モデル
│   ├── network_dncnn.py       # DnCNN デノイザー
│   └── basicblock.py
├── utils/
│   ├── lighting.py            # 光学モデル補助（光線/PSF）
│   ├── loading.py             # データ読み込み
│   ├── plotting.py            # 可視化
│   ├── pretraining.py         # 楕円球データによる事前学習
│   └── denoising.py
├── my-program/                # 自作スクリプト群
│   ├── create_compressed_inputs/   # ブロックランダムマスクで圧縮画像を生成
│   ├── snapshot.py            # 圧縮画像生成
│   ├── random_mask.py         # マスク生成
│   ├── forward_only_test.py   # 順方向モデル単体検証
│   ├── oracle_reconstruction.py
│   └── visualize/             # 3D / GIF 可視化
├── phase_b/                   # 修論部分：Deep Sensing でシャッタ最適化
│   ├── train_deep_sensing.py
│   ├── encoder.py             # シャッタパターン学習層
│   ├── decoder.py             # 復元ネットワーク
│   ├── dataset.py
│   └── make_compressed_eval.py
└── knowledge/
    ├── research_overview.md       # 研究概要（詳細版・推奨）
    └── measurement_model_notes.md # 測定モデル理論ノート
```

---

## 🚀 主要コマンド

```bash
# conda 環境構築
conda env create -f environment.yml
conda activate 3dmfr-env

# 従来手法（多焦点画像 11枚 から復元）
python main.py --model dip --input path/to/multifocus_images/ -v exp_name

# 提案手法（圧縮画像 1枚 から復元）
python main.py --model dip \
  --snapshot snapshot.png \
  --snapshot-masks masks.mat \
  -v exp_name

# 圧縮画像の生成（ブロックランダムマスク）
python my-program/create_compressed_inputs/block_random/snapshot_block_random.py

# Deep Sensing による学習可能パターンの最適化（Phase B）
python phase_b/train_deep_sensing.py
```

---

## 📄 研究情報

- **研究テーマ**: 圧縮多焦点画像を用いた3次元復元（Compressed Multi-focus 3D Reconstruction）
- **所属**: 大阪大学大学院 情報科学研究科 コンピュータサイエンス専攻 M1
- **進捗**:
  - 第1段階（卒業論文・**完了**）: ランダムシャッタによる圧縮画像からの3D復元
  - 第2段階（修士論文・**進行中**）: Deep Sensing によるシャッタパターン最適化
- **詳細**: 研究背景・実験設定・考察の詳細は [`knowledge/research_overview.md`](knowledge/research_overview.md) を参照

---

## 🔬 引用・参考文献

| 文献 | 役割 |
|------|------|
| Azevedo et al., "Deep Volume Reconstruction from Multi-focus Microscopic Images," **MICCAI 2022** | 本実装のベース（多焦点復元の DIP 実装）<br>[caiocj1/multifocus-3d-reconstruction](https://github.com/caiocj1/multifocus-3d-reconstruction) |
| Yamaguchi et al., "3D Image Reconstruction from Multi-focus Microscopic Images," **PSIVT 2020** | イメージングモデル（光線減衰）の元祖 |
| Yoshida et al., "Deep Sensing for Compressive Video Acquisition," **Sensors 2023** | シャッタパターン最適化（第2段階）の参考 |
| Ulyanov et al., "Deep Image Prior," **CVPR 2018** | DIP の原典 |

---

## 👤 開発体制

- **開発形態**: 個人研究（指導教員の指導下、実装・実験はすべて単独）
- **開発期間**: 2024年4月〜継続中
- **担当範囲**: 問題定式化 / 順方向モデル設計・実装 / ネットワーク・学習スキーム実装 / 合成&実データ実験 / 評価・考察

---

## 📝 注記

- 本リポジトリは就活ポートフォリオ向けに、研究室の元リポジトリから **自分の貢献に関わるコードと公開可能な理論ノートのみ** を抜粋して公開している
- 実データ・実験ログ・指導教員向け内部資料・他者著作物の論文PDFは含めていない
- ベース実装 Azevedo et al. のリポジトリは上述の引用表に明示している
