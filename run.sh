#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: ./run.sh CASE_NAME SIM_ID"
    echo "CASE_NAME: jet | staticDroplet | rti"
    exit 1
fi

CASE_NAME="$1"
SIM_ID="$2"

case "$CASE_NAME" in
    jet)
        CASE_DEFINE="-DCASE_JET"
        ;;
    staticDroplet)
        CASE_DEFINE="-DCASE_STATIC_DROPLET"
        ;;
    rti)
        CASE_DEFINE="-DCASE_RTI"
        ;;
    *)
        echo "unknown case: $CASE_NAME"
        echo "CASE_NAME: jet | staticDroplet | rti"
        exit 1
        ;;
esac

rm -rf "output/${CASE_NAME}/${SIM_ID}"
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86 -lineinfo -Xptxas -v "$CASE_DEFINE" main.cu -o mrlbm
./mrlbm "$SIM_ID"
