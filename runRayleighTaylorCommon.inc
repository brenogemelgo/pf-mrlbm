#!/usr/bin/env bash

rti_expected_checkpoint_bytes() {
    local nx="$1"
    local ny="$2"
    local nz="$3"

    printf '%s\n' "$((11 * nx * ny * nz * 4))"
}

rti_cleanup_dirty_checkpoints() {
    local run_dir="$1"
    local expected_bytes="$2"

    local binaries_dir="${run_dir}/binaries"
    [[ -d "${binaries_dir}" ]] || return 0

    local checkpoint
    while IFS= read -r checkpoint; do
        local bytes
        bytes="$(stat -c '%s' "${checkpoint}")"
        if [[ "${bytes}" != "${expected_bytes}" ]]; then
            echo "removing incomplete checkpoint: ${checkpoint} (${bytes} bytes)"
            rm -f "${checkpoint}"
        fi
    done < <(find "${binaries_dir}" -maxdepth 1 -type f -name 'step_*.bin' | sort)
}

rti_latest_valid_checkpoint() {
    local run_dir="$1"
    local expected_bytes="$2"

    local binaries_dir="${run_dir}/binaries"
    [[ -d "${binaries_dir}" ]] || return 0

    local checkpoint
    while IFS= read -r checkpoint; do
        local bytes
        bytes="$(stat -c '%s' "${checkpoint}")"
        if [[ "${bytes}" == "${expected_bytes}" ]]; then
            printf '%s\n' "${checkpoint}"
        fi
    done < <(find "${binaries_dir}" -maxdepth 1 -type f -name 'step_*.bin' | sort)
}

rti_compile_case() {
    local case_define="$1"
    local reynolds="$2"
    local weber="$3"
    local binary_path="$4"

    nvcc \
        -std=c++20 \
        -O3 \
        --restrict \
        --expt-relaxed-constexpr \
        --fmad=true \
        --extra-device-vectorization \
        --extended-lambda \
        "-arch=${ARCH:-sm_86}" \
        -lineinfo \
        -Xptxas \
        -v \
        -DCASE_RTI \
        "-D${case_define}" \
        "-DRTI_NX=${RTI_NX:-256}" \
        "-DRTI_NY=${RTI_NY:-4}" \
        "-DRTI_NZ=${RTI_NZ:-512}" \
        "-DRTI_NSTEPS=${RTI_NSTEPS:-60000}" \
        "-DRTI_STAMP=${RTI_STAMP:-10000}" \
        "-DRTI_WIDTH=${RTI_WIDTH:-6.0}" \
        "-DRTI_RHO_L=${RTI_RHO_L:-1.0}" \
        "-DRTI_RHO_RATIO=${RTI_RHO_RATIO:-10000.0}" \
        "-DRTI_MU_RATIO=${RTI_MU_RATIO:-1000.0}" \
        "-DRTI_U_CHAR=${RTI_U_CHAR:-1.0e-2}" \
        "-DRTI_REYNOLDS=${reynolds}" \
        "-DRTI_WEBER=${weber}" \
        "-DRTI_GRAVITY=${RTI_GRAVITY:-2.0e-7}" \
        "-DRTI_A0=${RTI_A0:-4.0}" \
        "-DRTI_TAU_PHI=${RTI_TAU_PHI:-1.0}" \
        "-DRTI_QUASI_2D=${RTI_QUASI_2D:-1}" \
        src/main.cu \
        -o "${binary_path}"
}

rti_run_case() {
    local case_define="$1"
    local run_id="$2"
    local reynolds="$3"
    local weber="$4"

    local nx="${RTI_NX:-256}"
    local ny="${RTI_NY:-4}"
    local nz="${RTI_NZ:-512}"
    local nsteps="${RTI_NSTEPS:-60000}"
    local binary_path="${BINARY_PATH:-/tmp/mrlbm_rti_${case_define,,}}"
    local python_bin="${PYTHON_BIN:-python3}"
    local expected_checkpoint_bytes="${EXPECTED_CHECKPOINT_BYTES:-$(rti_expected_checkpoint_bytes "${nx}" "${ny}" "${nz}")}"

    local run_dir="output/rti/${run_id}"
    local final_checkpoint="${run_dir}/binaries/$(printf 'step_%09d.bin' "${nsteps}")"
    local summary_path="${run_dir}/post/rayleigh_taylor_summary.csv"

    echo
    echo "=== ${run_id}: ${case_define}, Re=${reynolds}, We=${weber}, grid=${nx}x${ny}x${nz} ==="

    rti_cleanup_dirty_checkpoints "${run_dir}" "${expected_checkpoint_bytes}"

    if [[ -f "${final_checkpoint}" ]]; then
        if [[ -f "${summary_path}" ]]; then
            echo "complete; skipping ${run_id}"
            return 0
        fi

        echo "final checkpoint exists; post-processing ${run_id}"
        "${python_bin}" postRayleighTaylor.py --runId "${run_id}"
        return 0
    fi

    rti_compile_case "${case_define}" "${reynolds}" "${weber}" "${binary_path}"

    local latest_checkpoint
    latest_checkpoint="$(rti_latest_valid_checkpoint "${run_dir}" "${expected_checkpoint_bytes}" | tail -1)"
    if [[ -n "${latest_checkpoint}" ]]; then
        echo "continuing ${run_id} from ${latest_checkpoint}"
        "${binary_path}" --runId "${run_id}" --continue
    else
        echo "starting ${run_id} from scratch"
        rm -rf "${run_dir}"
        mkdir -p "${run_dir}"
        "${binary_path}" --runId "${run_id}"
    fi

    "${python_bin}" postRayleighTaylor.py --runId "${run_id}"
}
