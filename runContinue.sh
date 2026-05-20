#!/usr/bin/env bash
set -euo pipefail

mkdir -p output
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict -arch=sm_86 -lineinfo -Xptxas -v main.cu -o mrlbm
./mrlbm --continue
