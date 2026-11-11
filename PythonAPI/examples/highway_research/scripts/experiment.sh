#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
examples_directory="$(cd -- "$script_directory/../.." && pwd)"
cd -- "$examples_directory"
study_action="${1:-preflight}"
study_root="${2:-highwaytest/runs/research1_5seeds_20261111_v1}"
study_python="${STUDY_PYTHON:-python}"
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "$study_action" in
  plan) "$study_python" -m highway_research plan --root "$study_root" ;;
  preflight) "$study_python" -m highway_research batch --root "$study_root" --arms main beta mechanism reward --dry-run ;;
  main) "$study_python" -X utf8 -u -m highway_research batch --root "$study_root" --arms main --jobs "${STUDY_JOBS:-4}" ;;
  ablations) "$study_python" -X utf8 -u -m highway_research batch --root "$study_root" --arms beta mechanism reward --jobs "${STUDY_JOBS:-4}" ;;
  audit-dev) "$study_python" -m highway_research audit-study --root "$study_root" --arms main beta mechanism reward --cases 40 ;;
  audit-selected) "$study_python" -m highway_research audit-study --root "$study_root" --arms main --checkpoint selected --cases 40 ;;
  audit-test) "$study_python" -m highway_research audit-study --root "$study_root" --arms main --split test --cases 100 ;;
  audit-robust) "$study_python" -m highway_research audit-study --root "$study_root" --arms main --labels PPO Mut-PPO --conditions nominal idm_short_gap idm_long_gap cautious_lane position_shift --cases 40 ;;
  report) "$study_python" -m highway_research report --root "$study_root" ;;
  tensorboard) "$study_python" -m tensorboard.main --logdir "$study_root/train" --host 127.0.0.1 --port "${STUDY_TB_PORT:-6012}" ;;
  *) printf 'Unknown action: %s\n' "$study_action" >&2; exit 2 ;;
esac
