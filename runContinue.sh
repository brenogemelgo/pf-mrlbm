#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${NAME:-}"
runId="${RUN_ID:-}"
extra_args=()

normalize_case_name() {
    local raw_name
    raw_name="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
    case "$raw_name" in
        DROPLET|STATICDROPLET|STATIC_DROPLET|STATIC-DROPLET)
            printf '%s\n' "STATIC_DROPLET"
            ;;
        RTI|RAYLEIGH_TAYLOR|RAYLEIGH-TAYLOR|RAYLEIGHTAYLOR)
            printf '%s\n' "RTI"
            ;;
        *)
            printf '%s\n' "$raw_name"
            ;;
    esac
}

for arg in "$@"; do
    if [[ "$arg" == NAME=* ]]; then
        CASE_NAME="${arg#NAME=}"
    elif [[ "$arg" == RUN_ID=* ]]; then
        runId="${arg#RUN_ID=}"
    elif [[ "$arg" == RUNID=* ]]; then
        runId="${arg#RUNID=}"
    elif [[ "$arg" != --* ]]; then
        ARG_CASE_NAME="$(normalize_case_name "$arg")"
        if [[ -z "$CASE_NAME" && ( "$ARG_CASE_NAME" == "STATIC_DROPLET" || "$ARG_CASE_NAME" == "RTI" ) ]]; then
            CASE_NAME="$ARG_CASE_NAME"
        elif [[ -z "$runId" ]]; then
            runId="$arg"
        else
            extra_args+=("$arg")
        fi
    else
        extra_args+=("$arg")
    fi
done

if [[ -z "$CASE_NAME" ]]; then
    CASE_NAME="RTI"
else
    CASE_NAME="$(normalize_case_name "$CASE_NAME")"
fi

case "$CASE_NAME" in
    STATIC_DROPLET)
        CASE_DEFINE="-DCASE_STATIC_DROPLET"
        ;;
    RTI)
        CASE_DEFINE="-DCASE_RTI"
        ;;
    *)
        echo "Unknown NAME=$CASE_NAME. Use NAME=STATIC_DROPLET or NAME=RTI." >&2
        exit 1
        ;;
esac

case_lower="$(printf '%s' "$CASE_NAME" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$runId" ]]; then
    runId="000"
elif [[ "$runId" =~ ^[0-9]+$ ]]; then
    runId="$(printf '%03d' "$((10#$runId))")"
fi

if [[ -z "$runId" || "$runId" == "." || "$runId" == ".." || "$runId" == *"/"* || "$runId" == *"\\"* ]]; then
    echo "Invalid run id: $runId" >&2
    exit 1
fi

mkdir -p "output/${case_lower}/${runId}"
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86 -lineinfo -Xptxas -v "${CASE_DEFINE}" src/main.cu -o mrlbm
./mrlbm --runId "$runId" "${extra_args[@]}" --continue
