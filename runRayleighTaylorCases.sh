#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

"${script_dir}/runRayleighTaylorCaseOne.sh"
"${script_dir}/runRayleighTaylorCaseTwo.sh"

if [[ "${RUN_RTI_PRESENTATION_CASES:-0}" == "1" ]]; then
    "${script_dir}/runRayleighTaylorPrettyCases.sh"
fi

if [[ "${RUN_RTI_STRESS_CASES:-0}" == "1" ]]; then
    "${script_dir}/runRayleighTaylorStressCases.sh"
fi
