#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
study="runs/bottleneck_v4_s42"
if [[ ! -f "highwaytest/${study}/study.json" ]]; then
  python -m highwaytest.bottleneck prepare --output "$study" --seed 42 --normal-steps 100000 --attack-steps 200000
fi
python -u -m highwaytest.bottleneck calibrate --study "$study" --gif
python -u -m highwaytest.bottleneck bootstrap --study "$study"
python -u -m highwaytest.bottleneck normal --study "$study"
python -u -m highwaytest.bottleneck export --study "$study"
python -u -m highwaytest.bottleneck attack --study "$study" --jobs "${JOBS:-2}"
python -m highwaytest.bottleneck status --study "$study"
