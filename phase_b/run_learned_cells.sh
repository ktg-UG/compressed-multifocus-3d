#!/usr/bin/env bash
# 最適化あり（learned 2×2 パターン）評価: cell 0000-0005
# 疑似GT（inputs/pseudo_gt/cell/<id>）は baseline 実験で生成済みなので再利用。
# 各 cell で learned パターンの圧縮スナップショットを作り、圧縮 DIP を疑似GTに対して実行。
#
# overfit 知見（iter 200〜1400 でピーク）から n_iter は 2000 で十分。peak は tensorboard から抽出。
#
# 使い方: bash phase_b/run_learned_cells.sh [N_ITER] [PATTERN_PT]
set -uo pipefail

PY=python3
N_ITER="${1:-2000}"
PATTERN_PT="${2:-experiments/phase_b_m2/learned_pattern.pt}"
CELLS="0000 0001 0002 0003 0004 0005"
SUMMARY="experiments/phase_b_learned/summary.tsv"
mkdir -p experiments/phase_b_learned
echo -e "cell\tlearned_compressed_final(psnr ssim rmse dice loss)" > "$SUMMARY"

for ID in $CELLS; do
  echo "##################### cell ${ID} (learned) #####################"
  CELL_DIR="inputs/multi_focus_data/cell/${ID}"
  GT_DIR="inputs/pseudo_gt/cell/${ID}"
  CMP_OUT="outputs/cell_cmp_learned_${ID}"

  if [ ! -d "${GT_DIR}" ] || [ "$(ls ${GT_DIR}/*.png 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "  !! 疑似GT ${GT_DIR} が無い。baseline 実験を先に。スキップ"; continue
  fi

  echo "=== [${ID}] learned 圧縮スナップショット生成 ==="
  $PY phase_b/make_compressed_eval.py --cell "${CELL_DIR}" --pattern "${PATTERN_PT}" \
    -o "inputs/compressed_data/learned_phase_b/cell/${ID}"

  echo "=== [${ID}] 圧縮 DIP (learned, 疑似GTに対し各iter PSNR) ==="
  $PY main.py -m dip -p sc \
    -s "inputs/compressed_data/learned_phase_b/cell/${ID}/snapshot.png" \
    -sm "inputs/compressed_data/learned_phase_b/cell/${ID}/masks.mat" \
    -gt "${GT_DIR}" \
    -v "cell_cmp_learned_${ID}" -o "${CMP_OUT}" --n_iter "${N_ITER}"

  CMP_RES=$( [ -f "${CMP_OUT}/result.txt" ] && sed -n '2p' "${CMP_OUT}/result.txt" || echo "NA" )
  echo -e "${ID}\t${CMP_RES}" >> "$SUMMARY"
done

echo "##################### 全完了 #####################"
cat "$SUMMARY"
