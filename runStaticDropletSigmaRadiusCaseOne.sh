#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

source "${script_dir}/runStaticDropletSigmaRadiusCommon.inc"

run_static_droplet_sigma_radius_case CASE_ONE ONE
