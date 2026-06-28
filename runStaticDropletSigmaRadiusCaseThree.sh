#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runStaticDropletSigmaRadiusCommon.inc"

export NSTEPS="${NSTEPS:-50000}"
export STAMP="${STAMP:-10000}"
export WIDTH="${WIDTH:-6.0}"
export MU_L="${MU_L:-5.0e-1}"
export TAU_PHI="${TAU_PHI:-1.0}"
export RADIUS_VALUES="${RADIUS_VALUES:-28 32 36 40}"

run_static_droplet_sigma_radius_case CASE_THREE "${CASE_THREE_RUN_LABEL:-THREE_SAFE}"
