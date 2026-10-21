#!/usr/bin/env bash
# Linux launcher for the startup_v1 contract; screen owns both long-lived jobs.
set -Eeuo pipefail

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
PROJECT=$(cd "$(dirname "$SCRIPT")/.." && pwd)
cd "$PROJECT"
CARLA_ROOT=${CARLA_ROOT:-/opt/carla/CARLA_0.9.16}
CARLA_USER=${CARLA_USER:-carla}
CONDA_ENV=${CONDA_ENV:-carla-mappo}
CARLA_SESSION=${CARLA_SESSION:-carla-server}
TRAIN_SESSION=${TRAIN_SESSION:-mappo-startup-v1}
MONITOR_PORT=${MONITOR_PORT:-8080}
MODE=${1:-help}
RUN_ID=${2:-}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "Missing command: $1"; }
screen_exists() { screen -ls 2>/dev/null | awk -v n="$1" '$1 ~ /^[0-9]+\./ {sub(/^[0-9]+\./,"",$1); if ($1==n) found=1} END {exit !found}'; }
run_paths() {
    [[ $RUN_ID =~ ^BD_STARTUP_V1_s42_[A-Za-z0-9_-]+$ ]] || die 'Supply the original BD_STARTUP_V1_s42_... run ID.'
    CONTROL="$PROJECT/diagnostics/server_startup_v1/$RUN_ID"
    PLAN="$CONTROL/session/plan.json"
}
find_conda() {
    CONDA_BIN=${CONDA_BIN:-${CONDA_EXE:-}}
    if [[ -z $CONDA_BIN ]]; then CONDA_BIN=$(type -P conda || true); fi
    [[ -n $CONDA_BIN && -x $CONDA_BIN ]] || die 'Activate conda first, or set CONDA_BIN to its executable path.'
    CONDA_BIN=$(readlink -f "$CONDA_BIN")
}
dispatch_train() {
    screen -dmS "$TRAIN_SESSION" env \
        CONDA_ENV="$CONDA_ENV" CONDA_BIN="$CONDA_BIN" \
        MONITOR_PORT="$MONITOR_PORT" \
        "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
        bash "$SCRIPT" _worker "$RUN_ID" "$1"
    printf 'DISPATCHED (not yet training success): %s\n' "$RUN_ID"
    printf 'Attach: screen -r %s\nLog: %s/launcher.log\n' "$TRAIN_SESSION" "$CONTROL"
    printf 'Status: bash %q status %q\n' "$SCRIPT" "$RUN_ID"
}
start_carla() {
    need screen; need ss
    if screen_exists "$CARLA_SESSION"; then
        printf 'Existing CARLA screen: %s; reusing it.\n' "$CARLA_SESSION"
        return
    fi
    if [[ -n $(ss -H -ltn '( sport = :2000 or sport = :2001 or sport = :2002 )') ]]; then
        printf 'CARLA ports already occupied; not starting a duplicate. Training startup will validate the server.\n'
        return
    fi
    local binary="$CARLA_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    [[ -x $binary ]] || die "CARLA executable missing: $binary"
    id "$CARLA_USER" >/dev/null || die "Unknown CARLA user: $CARLA_USER"
    if [[ $(id -u) == 0 ]]; then
        need runuser
    else
        [[ $(id -un) == "$CARLA_USER" ]] || die "Run as root or $CARLA_USER."
    fi
    mkdir -p "$PROJECT/logs/server_launcher"
    screen -dmS "$CARLA_SESSION" env CARLA_ROOT="$CARLA_ROOT" CARLA_USER="$CARLA_USER" \
        bash "$SCRIPT" _carla
    printf 'CARLA dispatched. Attach: screen -r %s\nLog: %s/logs/server_launcher/carla.log\n' "$CARLA_SESSION" "$PROJECT"
}

case "$MODE" in
help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/run_server_startup_v1.sh COMMAND [RUN_ID]
  all                 Start/reuse CARLA and dispatch a fresh 25->50 update run.
  carla               Start/reuse the CARLA screen only.
  train               Dispatch a fresh run against an existing CARLA server.
  status RUN_ID       Inspect saved state and show launcher log; no CARLA calls.
  resume RUN_ID       Explicitly resume the same finite plan after a failure.
Environment: CONDA_ENV=carla-mappo CARLA_ROOT=/opt/carla/CARLA_0.9.16
             CARLA_USER=carla MONITOR_PORT=8080 CONDA_BIN=/path/to/conda
Screen names: carla-server and mappo-startup-v1. Detach: Ctrl+A, then D.
SSH tunnel/browser setup remains on your local computer.
HELP
    ;;
carla) start_carla ;;
all|train)
    need screen; find_conda
    screen_exists "$TRAIN_SESSION" && die "Training screen already exists: $TRAIN_SESSION"
    [[ $MONITOR_PORT =~ ^[0-9]+$ ]] && ((MONITOR_PORT >= 1 && MONITOR_PORT <= 65535)) || die 'Invalid MONITOR_PORT.'
    [[ -f "$PROJECT/configs/base_driving/base_driving_startup_v1_v1.json" ]] || die 'Run this script from the updated monitor checkout.'
    RUN_ID="BD_STARTUP_V1_s42_$(date +%Y%m%d_%H%M%S)"
    run_paths
    [[ ! -e "$PROJECT/logs/$RUN_ID" ]] || die 'Run already exists.'
    mkdir -p "$(dirname "$CONTROL")"
    mkdir "$CONTROL" # Reserve an exclusive new launch; never overwrite a run.
    printf 'export CONDA_ENV=%q\nexport CONDA_BIN=%q\nexport MONITOR_PORT=%q\nexport TRAIN_SESSION=%q\n' \
        "$CONDA_ENV" "$CONDA_BIN" "$MONITOR_PORT" "$TRAIN_SESSION" > "$CONTROL/launcher.env"
    [[ $MODE != all ]] || start_carla
    dispatch_train fresh
    ;;
status|resume)
    run_paths
    [[ -f "$CONTROL/launcher.env" ]] || die "Launcher configuration missing: $CONTROL/launcher.env"
    source "$CONTROL/launcher.env"
    find_conda
    if [[ $MODE == status ]]; then
        if [[ -f $PLAN ]]; then
            "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
                python -m utils.base_driving_session inspect --plan "$PLAN" \
                </dev/null
        else
            printf 'No prepared plan yet. Inspect the launcher log below.\n'
        fi
        [[ ! -f "$CONTROL/launcher.log" ]] || tail -n 40 "$CONTROL/launcher.log"
    else
        need screen
        screen_exists "$TRAIN_SESSION" && die "Training screen still exists: $TRAIN_SESSION; attach to it instead."
        [[ -f $PLAN ]] || die 'No prepared plan; inspect failure and create a new run.'
        dispatch_train resume
    fi
    ;;
_carla)
    mkdir -p "$PROJECT/logs/server_launcher"
    exec > >(tee -a "$PROJECT/logs/server_launcher/carla.log") 2>&1
    printf '\nCARLA launch: %s\n' "$(date -Is)"
    CARLA_HOME=$(getent passwd "$CARLA_USER" | cut -d: -f6)
    [[ -n $CARLA_HOME ]] || die 'CARLA home directory could not be resolved.'
    RUNTIME="/tmp/xdg-runtime-$CARLA_USER"
    if [[ $(id -u) == 0 ]]; then
        install -d -m 700 -o "$CARLA_USER" -g "$(id -gn "$CARLA_USER")" "$RUNTIME"
        EXEC=(runuser -u "$CARLA_USER" --)
    else
        install -d -m 700 "$RUNTIME"
        EXEC=()
    fi
    cd "$CARLA_ROOT"
    exec "${EXEC[@]}" env -u VK_ICD_FILENAMES HOME="$CARLA_HOME" XDG_RUNTIME_DIR="$RUNTIME" \
        ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
        -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000 -graphicsadapter=0
    ;;
_worker)
    run_paths
    ACTION=${3:?missing worker action}
    [[ $ACTION == fresh || $ACTION == resume ]] || die 'Invalid worker action.'
    exec > >(tee -a "$CONTROL/launcher.log") 2>&1
    trap 'code=$?; printf "Launcher exited with code %s at %s\n" "$code" "$(date -Is)"' EXIT
    cd "$PROJECT"
    printf '\n%s: %s (%s)\n' "$(date -Is)" "$RUN_ID" "$ACTION"
    # This readiness/map operation runs before training, under the shared clock lease.
    python - <<'PY'
import time
import carla
import torch
from utils.mut_mappo_session import CarlaOwner

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
deadline = time.monotonic() + 180
while True:
    try:
        version = client.get_server_version()
        break
    except RuntimeError:
        if time.monotonic() >= deadline:
            raise RuntimeError('CARLA did not become ready within 180 seconds; inspect carla.log')
        time.sleep(3)
if not version.startswith('0.9.16'):
    raise RuntimeError(f'Expected CARLA 0.9.16, got {version}')
with CarlaOwner(purpose='startup_v1 launcher readiness and map check'):
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError('CARLA is already synchronous; check existing/stale trainers before proceeding')
    if world.get_map().name.rsplit('/', 1)[-1] != 'Town05':
        print('Loading Town05...', flush=True)
        world = client.load_world('Town05')
    print('CARLA:', version, 'map:', world.get_map().name, flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)
PY
    if [[ $ACTION == fresh ]]; then
        mkdir -p "$CONTROL/preflight"
        python -m utils.preflight_base_driving_scenarios_v1 \
            --scenario-dir data/training_scenarios_town05_base_vnext_v1 \
            --output "$CONTROL/preflight/train.json" --live
        python -m utils.preflight_base_driving_scenarios_v1 \
            --scenario-dir data/training_scenarios_town05_base_vnext_v1_validation \
            --output "$CONTROL/preflight/validation.json" --live
        python -m utils.base_driving_session prepare \
            --run-dir "logs/$RUN_ID" --control-dir "$CONTROL/session" \
            --contract configs/base_driving/base_driving_startup_v1_v1.json \
            --stop-updates 25 50 --seed 42 \
            --train-geometry "$CONTROL/preflight/train.json" \
            --validation-geometry "$CONTROL/preflight/validation.json" \
            --monitor-mode web --monitor-host 127.0.0.1 --monitor-port "$MONITOR_PORT" \
            --monitor-view overhead --monitor-overhead-height 35 \
            --monitor-smoothing-seconds 0.5 --monitor-width 960 --monitor-fps 5
        python -u -m utils.base_driving_session run --plan "$PLAN"
    else
        python -u -m utils.base_driving_session run --plan "$PLAN" --resume-controller
    fi
    ;;
*) die "Unknown command: $MODE; use --help." ;;
esac
