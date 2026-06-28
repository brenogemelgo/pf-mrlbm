#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runStaticDropletSigmaRadiusCommon.inc"

if [[ "${CASE_TWO_FULL_SWEEP:-0}" == "1" ]]; then
    export NSTEPS="${NSTEPS:-120000}"
    export STAMP="${STAMP:-10000}"
    export WIDTH="${WIDTH:-4.0}"
    export MU_L="${MU_L:-1.5e-1}"
    export TAU_PHI="${TAU_PHI:-1.0}"
    export RADIUS_VALUES="${RADIUS_VALUES:-20 24 28 32}"
    run_static_droplet_sigma_radius_case CASE_TWO "${CASE_TWO_RUN_LABEL:-TWO_SAFE}"
else
    export NSTEPS="${NSTEPS:-30000}"
    export STAMP="${STAMP:-10000}"
    export WIDTH="${WIDTH:-5.0}"
    export MU_L="${MU_L:-3.0e-1}"
    export TAU_PHI="${TAU_PHI:-1.0}"
    export SIGMA_VALUES="${SIGMA_VALUES:-0.08}"
    export SIGMA_TAGS="${SIGMA_TAGS:-008}"
    export RADIUS_VALUES="${RADIUS_VALUES:-32}"
    run_static_droplet_sigma_radius_case CASE_TWO "${CASE_TWO_RUN_LABEL:-TWO_SAFE_RETRY}"
fi
