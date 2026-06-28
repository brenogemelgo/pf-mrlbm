#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

"${script_dir}/runRayleighTaylorCaseOne.sh"
"${script_dir}/runRayleighTaylorCaseTwo.sh"
