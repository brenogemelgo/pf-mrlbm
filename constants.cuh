#pragma once

// =================================================================================================== //

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <string>

// =================================================================================================== //

using natural_t = uint32_t;
using real_t = float;
using scalar_t = real_t;
using mask_t = uint8_t;

// =================================================================================================== //

constexpr natural_t NX = 128;
constexpr natural_t NY = 128;
constexpr natural_t NZ = 512;

constexpr natural_t JET_DIAMETER = 30;

constexpr real_t U_CHAR = static_cast<real_t>(0.05);
constexpr real_t REYNOLDS = static_cast<real_t>(1000.0);
constexpr real_t WEBER = static_cast<real_t>(2500.0);

constexpr real_t RHO_RATIO = static_cast<real_t>(1000.0);
constexpr real_t MU_RATIO = static_cast<real_t>(1.0);

constexpr real_t WIDTH = static_cast<real_t>(12.0);

constexpr natural_t NSTEPS = 100000;
constexpr natural_t STAMP = 1000;

// =================================================================================================== //

constexpr natural_t JET_RADIUS = JET_DIAMETER / 2;
constexpr natural_t L_CHAR = JET_DIAMETER;
constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

// =================================================================================================== //

constexpr real_t RHO_L = static_cast<real_t>(1.0);
constexpr real_t RHO_G = RHO_L / RHO_RATIO;
constexpr real_t NU_L = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(L_CHAR)) / static_cast<double>(REYNOLDS));
constexpr real_t MU_L = RHO_L * NU_L;
constexpr real_t MU_G = MU_L / MU_RATIO;
constexpr real_t NU_G = MU_G / RHO_G;
constexpr real_t TAU_L = static_cast<real_t>(3.0) * NU_L + static_cast<real_t>(0.5);
constexpr real_t TAU_G = static_cast<real_t>(3.0) * NU_G + static_cast<real_t>(0.5);
constexpr real_t VISCOSITY = NU_L;

// =================================================================================================== //

constexpr real_t SIGMA = static_cast<real_t>((static_cast<double>(RHO_L) * static_cast<double>(U_CHAR) * static_cast<double>(U_CHAR) * static_cast<double>(L_CHAR)) / static_cast<double>(WEBER));
constexpr real_t BETA_CHEM = static_cast<real_t>(12.0) * SIGMA / WIDTH;
constexpr real_t KAPPA_CHEM = static_cast<real_t>(1.5) * SIGMA * WIDTH;
constexpr real_t TAU_PHI = static_cast<real_t>(1.0);
constexpr real_t DIFF_INT = static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0)) * (TAU_PHI - 0.5);
constexpr real_t KAPPA_INT = static_cast<real_t>(4.0) * DIFF_INT / WIDTH;
constexpr real_t GAMMA = static_cast<real_t>(3.0) * KAPPA_INT;

// =================================================================================================== //

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;

constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //

constexpr natural_t NUM_MOMENTS = 11;
constexpr natural_t NUM_FIELDS = NUM_MOMENTS;

constexpr natural_t PSTAR = 0;
constexpr natural_t UX = 1;
constexpr natural_t UY = 2;
constexpr natural_t UZ = 3;
constexpr natural_t MXX = 4;
constexpr natural_t MYY = 5;
constexpr natural_t MZZ = 6;
constexpr natural_t MXY = 7;
constexpr natural_t MXZ = 8;
constexpr natural_t MYZ = 9;
constexpr natural_t PHI = 10;

// =================================================================================================== //