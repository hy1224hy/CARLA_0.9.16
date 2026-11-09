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
  for level in 1 2 3; do
    if [[ ! -f "$STUDY/base_s${training_seed}/attack_L${level}.pt" ]]; then
      "$PYTHON" -u -m highwaytest.straight bootstrap --seed "$training_seed" --attack-level "$level" --output "$STUDY/base_s${training_seed}"
    fi
  done
done

# 2. Training blocks: main, beta, reward-ablation (reused runs are not retrained).
"$PYTHON" -u -m highwaytest.straight batch --configs "$config_root" --output "$STUDY/train/main" --jobs "$JOBS" --initialize-from-dir "$STUDY"
"$PYTHON" -u -m highwaytest.straight batch --configs highwaytest/configs/straight_beta_v2 --output "$STUDY/train/beta" --jobs "$JOBS" --initialize-from-dir "$STUDY"
"$PYTHON" -u -m highwaytest.straight batch --configs highwaytest/configs/straight_reward_v2 --output "$STUDY/train/reward" --jobs "$JOBS" --initialize-from-dir "$STUDY"

# 3. Frozen final test: 100 paired cases per run, trajectory + GIF export.
while IFS= read -r run_dir; do
  if [[ -f "$run_dir/selected.pt" && ! -f "$run_dir/test/evaluation.json" ]]; then
    "$PYTHON" -u -m highwaytest.straight evaluate --checkpoint "$run_dir/selected.pt" --split test --cases 100 --gif --output "$run_dir/test"
  fi
done < <(find "$STUDY/train" -mindepth 2 -maxdepth 2 -type d | sort)

# 4. Aggregate comparison (all 30 runs), strategy pool and consistency matrix (main 18 only).
"$PYTHON" -u -m highwaytest.straight report --runs "$STUDY/train/main" "$STUDY/train/beta" "$STUDY/train/reward" --output "$STUDY/comparison"
"$PYTHON" -u -m highwaytest.straight pool --runs "$STUDY/train/main" --output "$STUDY/pool.json"
"$PYTHON" -u -m highwaytest.straight matrix --runs "$STUDY/train/main" --output "$STUDY/matrix.json"
echo "Pipeline finished. Study: $STUDY ; comparison figures, pool.json, matrix.json written."
