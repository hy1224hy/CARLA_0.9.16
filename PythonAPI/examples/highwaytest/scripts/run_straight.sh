#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
STUDY="${STUDY:-highwaytest/runs/straight_v1}"
STEPS="${STEPS:-200000}"
JOBS="${JOBS:-1}"
read -r -a seed_values <<< "${SEEDS:-42 43 44}"

# 1. Main matrix (18 runs) + per-seed base initialization.
config_root="highwaytest/configs/straight_v1"
if [[ ! -d "$config_root" ]]; then
  "$PYTHON" -m highwaytest.straight plan --output "$config_root" --steps "$STEPS" --seeds "${seed_values[@]}"
fi
"$PYTHON" -m highwaytest.straight check --configs "$config_root" --steps "$STEPS" --seeds "${seed_values[@]}"
for training_seed in "${seed_values[@]}"; do
  if [[ ! -f "$STUDY/base_s${training_seed}/driver.pt" ]]; then
    "$PYTHON" -u -m highwaytest.straight bootstrap --seed "$training_seed" --output "$STUDY/base_s${training_seed}"
  fi
done

# 2. Training blocks: main, beta, reward-ablation (reused runs are not retrained).
"$PYTHON" -u -m highwaytest.straight batch --configs "$config_root" --output "$STUDY/train_v2/main" --jobs "$JOBS" --initialize-from-dir "$STUDY"
"$PYTHON" -u -m highwaytest.straight batch --configs highwaytest/configs/straight_beta_v2 --output "$STUDY/train_v2/beta" --jobs "$JOBS" --initialize-from-dir "$STUDY"
"$PYTHON" -u -m highwaytest.straight batch --configs highwaytest/configs/straight_reward_v2 --output "$STUDY/train_v2/reward" --jobs "$JOBS" --initialize-from-dir "$STUDY"

# 3. Frozen final test: 100 paired cases per run, trajectory + GIF export.
while IFS= read -r run_dir; do
  if [[ -f "$run_dir/selected.pt" && ! -f "$run_dir/test/evaluation.json" ]]; then
    "$PYTHON" -u -m highwaytest.straight evaluate --checkpoint "$run_dir/selected.pt" --split test --cases 100 --gif --output "$run_dir/test"
  fi
done < <(find "$STUDY/train_v2" -mindepth 2 -maxdepth 2 -type d | sort)

# 4. Aggregate comparison (all 30 runs), strategy pool and consistency matrix (main 18 only).
"$PYTHON" -u -m highwaytest.straight report --runs "$STUDY/train_v2/main" "$STUDY/train_v2/beta" "$STUDY/train_v2/reward" --output "$STUDY/comparison_v2"
"$PYTHON" -u -m highwaytest.straight pool --runs "$STUDY/train_v2/main" --output "$STUDY/pool_v2.json"
"$PYTHON" -u -m highwaytest.straight matrix --runs "$STUDY/train_v2/main" --output "$STUDY/matrix_v2.json"
echo "Pipeline finished. Study: $STUDY ; comparison_v2, pool_v2.json, matrix_v2.json written."
