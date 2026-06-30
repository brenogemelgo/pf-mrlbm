#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runRayleighTaylorCommon.inc"

export RTI_NX="${RTI_NX:-256}"
export RTI_NY="${RTI_NY:-4}"
export RTI_NZ="${RTI_NZ:-512}"
export RTI_NSTEPS="${RTI_NSTEPS:-120000}"
export RTI_STAMP="${RTI_STAMP:-5000}"
export RTI_WIDTH="${RTI_WIDTH:-6.0}"
export RTI_RHO_L="${RTI_RHO_L:-3.0}"
export RTI_RHO_RATIO="${RTI_RHO_RATIO:-3.0}"
export RTI_MU_RATIO="${RTI_MU_RATIO:-1.0}"
export RTI_GRAVITY="${RTI_GRAVITY:-2.0e-7}"
export RTI_U_CHAR="${RTI_U_CHAR:-$(rti_sqrt_gl "${RTI_GRAVITY}" "${RTI_NX}")}"
export RTI_A0="${RTI_A0:-4.0}"
export RTI_TAU_PHI="${RTI_TAU_PHI:-1.0}"
export RTI_QUASI_2D="${RTI_QUASI_2D:-1}"

trajectory_reynolds="${RTI_TRAJECTORY_REYNOLDS:-256}"
trajectory_reynolds_tag="${RTI_TRAJECTORY_REYNOLDS_TAG:-RE256}"
trajectory_weber="${RTI_TRAJECTORY_WEBER:-1000000000}"
run_id="${RTI_TRAJECTORY_LABEL:-RTI_TRAJECTORY_A05}_${trajectory_reynolds_tag}"

rti_run_case CASE_ONE "${run_id}" "${trajectory_reynolds}" "${trajectory_weber}"
