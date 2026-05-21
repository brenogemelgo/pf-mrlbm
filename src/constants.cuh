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
#include <limits>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <string>

// =================================================================================================== //

using natural_t = uint32_t;
#ifdef USE_DOUBLE_PRECISION
using real_t = double;
#else
using real_t = float;
#endif
using scalar_t = real_t;
using mask_t = uint8_t;

namespace math
{
    __device__ __host__ [[nodiscard]] static __forceinline__ real_t sqrt(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t tanh(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t cos(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }
}

// Conservative phase streaming should emit exactly phi from each source cell.
// Compute the rest phase population as a native real_t residual by default.
// Define PHI_DIRECT_PHASE_RECONSTRUCTION to use the literal equilibrium for q=0 too.
#if !defined(PHI_RESIDUAL_REST) && !defined(PHI_DIRECT_PHASE_RECONSTRUCTION)
#define PHI_RESIDUAL_REST
#endif

// =================================================================================================== //

constexpr mask_t BULK = 0u;
constexpr mask_t WEST = 1u << 0;
constexpr mask_t EAST = 1u << 1;
constexpr mask_t SOUTH = 1u << 2;
constexpr mask_t NORTH = 1u << 3;
constexpr mask_t BACK = 1u << 4;
constexpr mask_t FRONT = 1u << 5;

// face nodes
constexpr mask_t NORTH_FACE = NORTH;
constexpr mask_t SOUTH_FACE = SOUTH;
constexpr mask_t WEST_FACE = WEST;
constexpr mask_t EAST_FACE = EAST;
constexpr mask_t FRONT_FACE = FRONT;
constexpr mask_t BACK_FACE = BACK;

// edge nodes
constexpr mask_t NORTH_WEST = NORTH | WEST;
constexpr mask_t NORTH_EAST = NORTH | EAST;
constexpr mask_t NORTH_FRONT = NORTH | FRONT;
constexpr mask_t NORTH_BACK = NORTH | BACK;
constexpr mask_t SOUTH_WEST = SOUTH | WEST;
constexpr mask_t SOUTH_EAST = SOUTH | EAST;
constexpr mask_t SOUTH_FRONT = SOUTH | FRONT;
constexpr mask_t SOUTH_BACK = SOUTH | BACK;
constexpr mask_t WEST_FRONT = WEST | FRONT;
constexpr mask_t WEST_BACK = WEST | BACK;
constexpr mask_t EAST_FRONT = EAST | FRONT;
constexpr mask_t EAST_BACK = EAST | BACK;

// corner nodes
constexpr mask_t NORTH_WEST_FRONT = NORTH | WEST | FRONT;
constexpr mask_t NORTH_WEST_BACK = NORTH | WEST | BACK;
constexpr mask_t NORTH_EAST_FRONT = NORTH | EAST | FRONT;
constexpr mask_t NORTH_EAST_BACK = NORTH | EAST | BACK;
constexpr mask_t SOUTH_WEST_FRONT = SOUTH | WEST | FRONT;
constexpr mask_t SOUTH_WEST_BACK = SOUTH | WEST | BACK;
constexpr mask_t SOUTH_EAST_FRONT = SOUTH | EAST | FRONT;
constexpr mask_t SOUTH_EAST_BACK = SOUTH | EAST | BACK;

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

#include "../cases/caseSelector.cuh"

using Case = SelectedCase;

constexpr natural_t NX = Case::NX;
constexpr natural_t NY = Case::NY;
constexpr natural_t NZ = Case::NZ;

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

constexpr natural_t NSTEPS = Case::NSTEPS;
constexpr natural_t STAMP = Case::STAMP;

constexpr real_t RHO_L = Case::RHO_L;
constexpr real_t RHO_G = Case::RHO_G;
constexpr real_t MU_L = Case::MU_L;
constexpr real_t MU_G = Case::MU_G;
constexpr real_t WIDTH = Case::WIDTH;
constexpr real_t SIGMA = Case::SIGMA;
constexpr real_t BETA_CHEM = Case::BETA_CHEM;
constexpr real_t KAPPA_CHEM = Case::KAPPA_CHEM;
constexpr real_t TAU_PHI = Case::TAU_PHI;
constexpr real_t GAMMA = Case::GAMMA;
constexpr real_t U_CHAR = Case::U_CHAR;

// =================================================================================================== //

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;

constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //
