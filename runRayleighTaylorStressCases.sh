#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runRayleighTaylorCommon.inc"

export RTI_NX="${RTI_NX:-256}"
export RTI_NY="${RTI_NY:-4}"
export RTI_NZ="${RTI_NZ:-512}"
export RTI_NSTEPS="${RTI_NSTEPS:-100000}"
export RTI_STAMP="${RTI_STAMP:-5000}"
export RTI_WIDTH="${RTI_WIDTH:-6.0}"
export RTI_RHO_L="${RTI_RHO_L:-10000.0}"
export RTI_RHO_RATIO="${RTI_RHO_RATIO:-10000.0}"
export RTI_MU_RATIO="${RTI_MU_RATIO:-1000.0}"
export RTI_GRAVITY="${RTI_GRAVITY:-2.0e-7}"
export RTI_U_CHAR="${RTI_U_CHAR:-$(rti_sqrt_gl "${RTI_GRAVITY}" "${RTI_NX}")}"
export RTI_A0="${RTI_A0:-4.0}"
export RTI_TAU_PHI="${RTI_TAU_PHI:-1.0}"
export RTI_QUASI_2D="${RTI_QUASI_2D:-1}"

stress_reynolds="${RTI_STRESS_REYNOLDS:-100}"
stress_reynolds_tag="${RTI_STRESS_REYNOLDS_TAG:-RE0100}"
stress_weber_values=(${RTI_STRESS_WEBER_VALUES:-500})
stress_weber_tags=(${RTI_STRESS_WEBER_TAGS:-WE0500})

if [[ "${#stress_weber_values[@]}" != "${#stress_weber_tags[@]}" ]]; then
    echo "RTI_STRESS_WEBER_VALUES and RTI_STRESS_WEBER_TAGS must have the same length" >&2
    exit 1
fi

for index in "${!stress_weber_values[@]}"; do
    weber="${stress_weber_values[$index]}"
    weber_tag="${stress_weber_tags[$index]}"
    run_id="${RTI_STRESS_LABEL:-RTI_STRESS_RR10000_RM1000}_${stress_reynolds_tag}_${weber_tag}"
    rti_run_case CASE_TWO "${run_id}" "${stress_reynolds}" "${weber}"
done

# Optional stress/demonstration case only. Keep it out of the default
# validation envelope and inspect health diagnostics before using it in slides.
