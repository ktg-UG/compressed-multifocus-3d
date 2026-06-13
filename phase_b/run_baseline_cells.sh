#!/usr/bin/env bash
# 最適化なし（ランダム2×2）baseline 実験: cell 0000-0005
# 各 cell について:
#   1) 多焦点 DIP -> 疑似 GT volume（最終 iter で固定）
#   2) 疑似 GT volume -> クリーン PNG スライス
#   3) ランダム 2×2 パターンで圧縮スナップショット生成（学習なし baseline）
#   4) 圧縮 DIP を疑似 GT に対して実行（各 iter で PSNR 記録）
#
# 使い方: bash phase_b/run_baseline_cells.sh [N_ITER]
#   N_ITER 省略時 20000

set -uo pipefail

PY=python3
N_ITER="${1:-20000}"
CELLS="0000 0001 0002 0003 0004 0005"
SUMMARY="experiments/phase_b_baseline/summary.tsv"
mkdir -p experiments/phase_b_baseline
echo -e "cell\tmultifocus_final\tcompressed_random_final(psnr ssim rmse dice loss)" > "$SUMMARY"

for ID in $CELLS; do
  echo "##################### cell ${ID} #####################"
  CELL_DIR="inputs/multi_focus_data/cell/${ID}"
  GT_DIR="inputs/pseudo_gt/cell/${ID}"
  MF_OUT="outputs/cell_mf_${ID}"
  CMP_OUT="outputs/cell_cmp_random_${ID}"

  echo "=== [${ID} 1/4] 多焦点 DIP (疑似GT) ==="
  if [ ! -f "${MF_OUT}/dip_final_volume.npy" ]; then
    $PY main.py -m dip -p sc --input "${CELL_DIR}" \
      -v "cell_mf_${ID}" -o "${MF_OUT}" --n_iter "${N_ITER}"
  else
    echo "  既存 ${MF_OUT}/dip_final_volume.npy を再利用"
  fi

  echo "=== [${ID} 2/4] 疑似GT volume -> スライス ==="
  $PY phase_b/volume_npy_to_slices.py "${MF_OUT}/dip_final_volume.npy" -o "${GT_DIR}"

  echo "=== [${ID} 3/4] ランダム2×2 圧縮スナップショット生成 ==="
  $PY phase_b/make_compressed_eval.py --cell "${CELL_DIR}" --random --seed 42 \
    -o "inputs/compressed_data/random_baseline/cell/${ID}"

  echo "=== [${ID} 4/4] 圧縮 DIP (random, 疑似GTに対し各iter PSNR) ==="
  $PY main.py -m dip -p sc \
    -s "inputs/compressed_data/random_baseline/cell/${ID}/snapshot.png" \
    -sm "inputs/compressed_data/random_baseline/cell/${ID}/masks.mat" \
    -gt "${GT_DIR}" \
    -v "cell_cmp_random_${ID}" -o "${CMP_OUT}" --n_iter "${N_ITER}"

  MF_RES=$( [ -f "${MF_OUT}/result.txt" ] && sed -n '2p' "${MF_OUT}/result.txt" || echo "NA" )
  CMP_RES=$( [ -f "${CMP_OUT}/result.txt" ] && sed -n '2p' "${CMP_OUT}/result.txt" || echo "NA" )
  echo -e "${ID}\t${MF_RES}\t${CMP_RES}" >> "$SUMMARY"
  echo "=== cell ${ID} 完了 ==="
done

echo "##################### 全完了 #####################"
echo "サマリ: ${SUMMARY}"
cat "$SUMMARY"
echo "各 iter の PSNR 推移: tensorboard --logdir tb_logs"
