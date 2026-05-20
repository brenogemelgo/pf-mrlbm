#!/usr/bin/env bash
set -euo pipefail

rm -rf output
mkdir -p output
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86  -lineinfo -Xptxas -v main.cu -o mrlbm
./mrlbm
