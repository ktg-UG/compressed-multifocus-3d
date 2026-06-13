#!/usr/bin/env bash
# Phase B 評価ランナー: cell には volume GT が無いため、
#   1) 従来手法（多焦点 DIP）で疑似 GT volume を作る
#   2) それをクリーン PNG スライス化して -gt に使う
#   3) 学習パターン / ランダムパターンで圧縮スナップショットを生成
#   4) 圧縮 DIP を疑似 GT に対して走らせ、各 iteration の PSNR を記録（learned vs random）
#
# 使い方:
#   bash phase_b/run_eval.sh <CELL_ID> <PATTERN_PT> [MF_ITER] [CMP_ITER]
# 例:
#   bash phase_b/run_eval.sh 0005 experiments/phase_b_m2/learned_pattern.pt 20000 20000
#
# TensorBoard で tb_logs/cell_cmp_learned_<ID> と tb_logs/cell_cmp_random_<ID> の psnr を比較し、
# どの iteration で止めるべきか（最良 PSNR の位置）を判断する。

set -euo pipefail

PY=python3
CELL_ID="${1:?cell id (e.g. 0005)}"
PATTERN_PT="${2:?learned pattern .pt path}"
MF_ITER="${3:-20000}"      # 疑似 GT 用の多焦点 DIP iteration
CMP_ITER="${4:-20000}"     # 圧縮 DIP iteration

CELL_DIR="inputs/multi_focus_data/cell/${CELL_ID}"
GT_DIR="inputs/pseudo_gt/cell/${CELL_ID}"
MF_OUT="outputs/cell_mf_${CELL_ID}"

echo "=== [1/4] 疑似 GT: 多焦点 DIP (${MF_ITER} iter) ==="
if [ ! -f "${MF_OUT}/dip_final_volume.npy" ]; then
  $PY main.py -m dip -p sc --input "${CELL_DIR}" \
    -v "cell_mf_${CELL_ID}" -o "${MF_OUT}" --n_iter "${MF_ITER}"
else
  echo "  既存の ${MF_OUT}/dip_final_volume.npy を再利用"
fi

echo "=== [2/4] 疑似 GT volume -> クリーン PNG スライス ==="
$PY phase_b/volume_npy_to_slices.py "${MF_OUT}/dip_final_volume.npy" -o "${GT_DIR}"

echo "=== [3/4] 圧縮スナップショット生成 (learned / random) ==="
$PY phase_b/make_compressed_eval.py --cell "${CELL_DIR}" --pattern "${PATTERN_PT}" \
  -o "inputs/compressed_data/learned_phase_b/cell/${CELL_ID}"
$PY phase_b/make_compressed_eval.py --cell "${CELL_DIR}" --random --seed 42 \
  -o "inputs/compressed_data/random_baseline/cell/${CELL_ID}"

echo "=== [4/4] 圧縮 DIP (疑似 GT に対し各 iter PSNR を記録) ==="
$PY main.py -m dip -p sc \
  -s "inputs/compressed_data/learned_phase_b/cell/${CELL_ID}/snapshot.png" \
  -sm "inputs/compressed_data/learned_phase_b/cell/${CELL_ID}/masks.mat" \
  -gt "${GT_DIR}" \
  -v "cell_cmp_learned_${CELL_ID}" -o "outputs/cell_cmp_learned_${CELL_ID}" --n_iter "${CMP_ITER}"

$PY main.py -m dip -p sc \
  -s "inputs/compressed_data/random_baseline/cell/${CELL_ID}/snapshot.png" \
  -sm "inputs/compressed_data/random_baseline/cell/${CELL_ID}/masks.mat" \
  -gt "${GT_DIR}" \
  -v "cell_cmp_random_${CELL_ID}" -o "outputs/cell_cmp_random_${CELL_ID}" --n_iter "${CMP_ITER}"

echo "=== 完了 ==="
echo "最終 PSNR:"
echo "  learned: $(head -2 outputs/cell_cmp_learned_${CELL_ID}/result.txt | tail -1)"
echo "  random : $(head -2 outputs/cell_cmp_random_${CELL_ID}/result.txt | tail -1)"
echo "各 iteration の PSNR 推移は TensorBoard で確認:"
echo "  tensorboard --logdir tb_logs"
