#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runRayleighTaylorCommon.inc"

export RTI_NX="${RTI_NX:-256}"
export RTI_NY="${RTI_NY:-4}"
export RTI_NZ="${RTI_NZ:-512}"
export RTI_NSTEPS="${RTI_NSTEPS:-200000}"
export RTI_STAMP="${RTI_STAMP:-10000}"
export RTI_WIDTH="${RTI_WIDTH:-6.0}"
export RTI_RHO_RATIO="${RTI_RHO_RATIO:-10000.0}"
export RTI_MU_RATIO="${RTI_MU_RATIO:-1000.0}"
export RTI_U_CHAR="${RTI_U_CHAR:-1.0e-2}"
export RTI_GRAVITY="${RTI_GRAVITY:-2.0e-7}"
export RTI_A0="${RTI_A0:-4.0}"
export RTI_TAU_PHI="${RTI_TAU_PHI:-1.0}"
export RTI_QUASI_2D="${RTI_QUASI_2D:-1}"

fixed_weber="${RTI_CASE_TWO_WEBER:-500}"
fixed_weber_tag="${RTI_CASE_TWO_WEBER_TAG:-WE0500}"
reynolds_values=(${RTI_CASE_TWO_REYNOLDS_VALUES:-100 1000 2000})
reynolds_tags=(${RTI_CASE_TWO_REYNOLDS_TAGS:-RE00100 RE01000 RE2000})

if [[ "${#reynolds_values[@]}" != "${#reynolds_tags[@]}" ]]; then
    echo "RTI_CASE_TWO_REYNOLDS_VALUES and RTI_CASE_TWO_REYNOLDS_TAGS must have the same length" >&2
    exit 1
fi

for index in "${!reynolds_values[@]}"; do
    reynolds="${reynolds_values[$index]}"
    reynolds_tag="${reynolds_tags[$index]}"
    run_id="${RTI_CASE_TWO_LABEL:-RTI_TWO}_${reynolds_tag}_${fixed_weber_tag}"
    rti_run_case CASE_TWO "${run_id}" "${reynolds}" "${fixed_weber}"
done
