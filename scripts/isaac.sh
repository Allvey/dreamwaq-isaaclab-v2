#!/usr/bin/env bash
# Run from this checkout using the existing IsaacLab/Conda installation.
set -o pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${DREAMWAQ_CONDA_ROOT:-$HOME/anaconda3}"
source "$conda_root/etc/profile.d/conda.sh" || exit 1
# Conda hooks can contain harmless nonzero commands (e.g. unalias).
# Enable errexit after activation, and reuse an already active environment.
if [[ "${CONDA_PREFIX:-}" != "$conda_root/envs/${DREAMWAQ_CONDA_ENV:-dreamwaq}" ]]; then
    conda activate "${DREAMWAQ_CONDA_ENV:-dreamwaq}" || exit 1
fi
set -eu

lab_dir="${ISAACLAB_PATH:?The Conda environment must configure ISAACLAB_PATH}"
rl_scripts="$lab_dir/scripts/reinforcement_learning/rsl_rl"
# Library preloads for this machine's Isaac Sim installation.
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libX11.so.6:/usr/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export DREAMWAQ_CENET_BETA="${DREAMWAQ_CENET_BETA:-0.5}"
export DREAMWAQ_CENET_LOSS_COEF="${DREAMWAQ_CENET_LOSS_COEF:-1.0}"
export PYTHONUNBUFFERED=1
cd "$project_dir"

command_name="${1:-help}"
if (( $# )); then shift; fi
case "$command_name" in
    soft-train|soft-play|soft-eval|soft-check|soft-physics|soft-rollout)
        exec python "$project_dir/scripts/run_soft.py" "${command_name#soft-}" "$@"
        ;;
    mixed-train|mixed-play|mixed-check)
        exec python "$project_dir/scripts/run_soft.py" "$command_name" "$@"
        ;;
    train)
        exec python "$rl_scripts/train.py" --task Isaac-Velocity-Rough-Go2-DreamWaQ-v0 "$@"
        ;;
    play)
        exec python "$rl_scripts/play.py" --task Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0 "$@"
        ;;
    eval)
        exec python "$project_dir/scripts/rollout_eval.py" --task Isaac-Velocity-Rough-Go2-DreamWaQ-Play-v0 "$@"
        ;;
    *)
        echo "Usage: bash scripts/isaac.sh {train|play|eval|soft-train|soft-play|soft-eval|soft-check|soft-physics|soft-rollout|mixed-train|mixed-play|mixed-check} [arguments]"
        if [[ "$command_name" != help && "$command_name" != --help && "$command_name" != -h ]]; then exit 2; fi
        ;;
esac
