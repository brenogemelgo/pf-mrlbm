#!/usr/bin/env bash

static_droplet_cleanup_dirty_checkpoints() {
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

static_droplet_latest_valid_checkpoint() {
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

static_droplet_compile_case() {
    local case_define="$1"
    local sigma="$2"
    local radius="$3"
    local arch="$4"
    local nsteps="$5"
    local stamp="$6"
    local width="$7"
    local mu_l="$8"
    local tau_phi="$9"
    local binary_path="${10}"

    nvcc \
        -std=c++20 \
        -O3 \
        --restrict \
        --expt-relaxed-constexpr \
        --fmad=true \
        --extra-device-vectorization \
        --extended-lambda \
        "-arch=${arch}" \
        -lineinfo \
        -Xptxas \
        -v \
        -DCASE_STATIC_DROPLET \
        "-D${case_define}" \
        "-DSTATIC_DROPLET_SIGMA=${sigma}" \
        "-DSTATIC_DROPLET_R_INIT=${radius}" \
        "-DSTATIC_DROPLET_WIDTH=${width}" \
        "-DSTATIC_DROPLET_MU_L=${mu_l}" \
        "-DSTATIC_DROPLET_TAU_PHI=${tau_phi}" \
        "-DSTATIC_DROPLET_NSTEPS=${nsteps}" \
        "-DSTATIC_DROPLET_STAMP=${stamp}" \
        src/main.cu \
        -o "${binary_path}"
}

run_static_droplet_sigma_radius_case() {
    local case_define="$1"
    local case_label="$2"

    local arch="${ARCH:-sm_86}"
    local nsteps="${NSTEPS:-100000}"
    local stamp="${STAMP:-1000}"
    local width="${WIDTH:-3.0}"
    local mu_l="${MU_L:-5.0e-2}"
    local tau_phi="${TAU_PHI:-1.0}"
    local binary_path="${BINARY_PATH:-/tmp/mrlbm_static_droplet_sigma_radius_${case_label,,}}"
    local python_bin="${PYTHON_BIN:-python3}"
    local post_sweep="${POST_SWEEP:-1}"
    local expected_checkpoint_bytes="${EXPECTED_CHECKPOINT_BYTES:-92274688}"

    local sigma_values
    local sigma_tags
    local radius_values
    read -r -a sigma_values <<< "${SIGMA_VALUES:-0.02 0.04 0.06 0.08}"
    read -r -a sigma_tags <<< "${SIGMA_TAGS:-002 004 006 008}"
    read -r -a radius_values <<< "${RADIUS_VALUES:-14 16 18 20}"

    if [[ "${#sigma_values[@]}" != "${#sigma_tags[@]}" ]]; then
        echo "SIGMA_VALUES and SIGMA_TAGS must have the same length" >&2
        return 1
    fi

    for sigma_index in "${!sigma_values[@]}"; do
        local sigma="${sigma_values[$sigma_index]}"
        local sigma_tag="${sigma_tags[$sigma_index]}"

        for radius in "${radius_values[@]}"; do
            local run_id="${case_label}_${sigma_tag}_${radius}"
            local run_dir="output/static_droplet/${run_id}"
            local final_checkpoint="${run_dir}/binaries/$(printf 'step_%09d.bin' "${nsteps}")"
            local summary_path="${run_dir}/post/static_droplet_summary.csv"

            echo
            echo "=== ${run_id}: ${case_define}, SIGMA=${sigma}, R_INIT=${radius} ==="

            static_droplet_cleanup_dirty_checkpoints "${run_dir}" "${expected_checkpoint_bytes}"

            if [[ -f "${final_checkpoint}" ]]; then
                if [[ -f "${summary_path}" ]]; then
                    echo "complete; skipping ${run_id}"
                    continue
                fi

                echo "final checkpoint exists; post-processing ${run_id}"
                "${python_bin}" postStaticDroplet.py --runId "${run_id}"
                continue
            fi

            static_droplet_compile_case \
                "${case_define}" \
                "${sigma}" \
                "${radius}" \
                "${arch}" \
                "${nsteps}" \
                "${stamp}" \
                "${width}" \
                "${mu_l}" \
                "${tau_phi}" \
                "${binary_path}"

            local latest_checkpoint
            latest_checkpoint="$(static_droplet_latest_valid_checkpoint "${run_dir}" "${expected_checkpoint_bytes}" | tail -1)"
            if [[ -n "${latest_checkpoint}" ]]; then
                echo "continuing ${run_id} from ${latest_checkpoint}"
                "${binary_path}" --runId "${run_id}" --continue
            else
                echo "starting ${run_id} from scratch"
                rm -rf "${run_dir}"
                mkdir -p "${run_dir}"
                "${binary_path}" --runId "${run_id}"
            fi

            "${python_bin}" postStaticDroplet.py --runId "${run_id}"
        done
    done

    if [[ "${post_sweep}" == "1" ]]; then
        "${python_bin}" postStaticDropletSigmaSweep.py
    fi
}
