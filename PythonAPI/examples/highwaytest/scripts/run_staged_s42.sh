#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
study="runs/staged_v1_s42"
if [[ ! -f "highwaytest/${study}/study.json" ]]; then
  python -m highwaytest.staged prepare --output "$study" --seed 42 --normal-steps 100000 --attack-steps 200000
fi
python -u -m highwaytest.staged bootstrap --study "$study"
python -u -m highwaytest.staged normal --study "$study"
python -u -m highwaytest.staged export --study "$study"
python -u -m highwaytest.staged attack --study "$study" --jobs "${JOBS:-2}"
python -m highwaytest.staged status --study "$study"
