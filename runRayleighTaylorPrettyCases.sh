#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runRayleighTaylorCommon.inc"

# High-property-contrast RTI presentation defaults.
# These are demonstration sweeps after A=0.5 validation, not literature validation cases.
export RTI_NX="${RTI_NX:-256}"
export RTI_NY="${RTI_NY:-4}"
export RTI_NZ="${RTI_NZ:-512}"
export RTI_NSTEPS="${RTI_NSTEPS:-100000}"
export RTI_STAMP="${RTI_STAMP:-5000}"
export RTI_WIDTH="${RTI_WIDTH:-6.0}"
export RTI_RHO_L="${RTI_RHO_L:-1000.0}"
export RTI_RHO_RATIO="${RTI_RHO_RATIO:-1000.0}"
export RTI_MU_RATIO="${RTI_MU_RATIO:-100.0}"
export RTI_GRAVITY="${RTI_GRAVITY:-2.0e-7}"
export RTI_U_CHAR="${RTI_U_CHAR:-$(rti_sqrt_gl "${RTI_GRAVITY}" "${RTI_NX}")}"
export RTI_A0="${RTI_A0:-4.0}"
export RTI_TAU_PHI="${RTI_TAU_PHI:-1.0}"
export RTI_QUASI_2D="${RTI_QUASI_2D:-1}"

presentation_label="${RTI_PRESENTATION_LABEL:-RTI_PRESENTATION_RR1000_RM100}"

fixed_reynolds="${RTI_PRESENTATION_FIXED_REYNOLDS:-150}"
fixed_reynolds_tag="${RTI_PRESENTATION_FIXED_REYNOLDS_TAG:-RE0150}"
weber_values=(${RTI_PRESENTATION_WEBER_VALUES:-100 500 1000})
weber_tags=(${RTI_PRESENTATION_WEBER_TAGS:-WE0100 WE0500 WE1000})

if [[ "${#weber_values[@]}" != "${#weber_tags[@]}" ]]; then
    echo "RTI_PRESENTATION_WEBER_VALUES and RTI_PRESENTATION_WEBER_TAGS must have the same length" >&2
    exit 1
fi

for index in "${!weber_values[@]}"; do
    weber="${weber_values[$index]}"
    weber_tag="${weber_tags[$index]}"
    run_id="${presentation_label}_${fixed_reynolds_tag}_${weber_tag}"
    rti_run_case CASE_ONE "${run_id}" "${fixed_reynolds}" "${weber}"
done

fixed_weber="${RTI_PRESENTATION_FIXED_WEBER:-500}"
fixed_weber_tag="${RTI_PRESENTATION_FIXED_WEBER_TAG:-WE0500}"
reynolds_values=(${RTI_PRESENTATION_REYNOLDS_VALUES:-50 100 200})
reynolds_tags=(${RTI_PRESENTATION_REYNOLDS_TAGS:-RE0050 RE0100 RE0200})

if [[ "${#reynolds_values[@]}" != "${#reynolds_tags[@]}" ]]; then
    echo "RTI_PRESENTATION_REYNOLDS_VALUES and RTI_PRESENTATION_REYNOLDS_TAGS must have the same length" >&2
    exit 1
fi

for index in "${!reynolds_values[@]}"; do
    reynolds="${reynolds_values[$index]}"
    reynolds_tag="${reynolds_tags[$index]}"
    run_id="${presentation_label}_${reynolds_tag}_${fixed_weber_tag}"
    rti_run_case CASE_TWO "${run_id}" "${reynolds}" "${fixed_weber}"
done
