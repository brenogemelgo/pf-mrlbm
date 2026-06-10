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
using real_t = float;
using scalar_t = real_t;
using mask_t = uint8_t;

struct D3Q27;

using VelocitySet = D3Q27;
using PhaseVelocitySet = D3Q27;
using GradientVelocitySet = D3Q27;

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

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t log(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
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

constexpr mask_t XMIN_FACE = WEST;
constexpr mask_t XMAX_FACE = EAST;
constexpr mask_t YMIN_FACE = SOUTH;
constexpr mask_t YMAX_FACE = NORTH;
constexpr mask_t ZMIN_FACE = BACK;
constexpr mask_t ZMAX_FACE = FRONT;
constexpr mask_t FACE_MASK = XMIN_FACE | XMAX_FACE | YMIN_FACE | YMAX_FACE | ZMIN_FACE | ZMAX_FACE;

// face nodes
constexpr mask_t NORTH_FACE = YMAX_FACE;
constexpr mask_t SOUTH_FACE = YMIN_FACE;
constexpr mask_t WEST_FACE = XMIN_FACE;
constexpr mask_t EAST_FACE = XMAX_FACE;
constexpr mask_t FRONT_FACE = ZMAX_FACE;
constexpr mask_t BACK_FACE = ZMIN_FACE;

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

#if defined(CASE_STATIC_DROPLET)
#include "../cases/staticDroplet.cuh"
using ActiveCase = StaticDropletCase;
constexpr bool CASE_IS_STATIC_DROPLET = true;
constexpr bool CASE_IS_RTI = false;
#elif defined(CASE_RTI)
#include "../cases/rti.cuh"
using ActiveCase = RTICase;
constexpr bool CASE_IS_STATIC_DROPLET = false;
constexpr bool CASE_IS_RTI = true;
#else
#error "No case selected. Compile with -DCASE_STATIC_DROPLET or -DCASE_RTI."
#endif

constexpr natural_t NX = ActiveCase::NX;
constexpr natural_t NY = ActiveCase::NY;
constexpr natural_t NZ = ActiveCase::NZ;

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

constexpr natural_t NSTEPS = ActiveCase::NSTEPS;
constexpr natural_t STAMP = ActiveCase::STAMP;

constexpr real_t RHO_L = ActiveCase::RHO_L;
constexpr real_t RHO_RATIO = ActiveCase::RHO_RATIO;
constexpr real_t RHO_G = static_cast<real_t>(static_cast<double>(RHO_L) / static_cast<double>(RHO_RATIO));

constexpr real_t MU_RATIO = ActiveCase::MU_RATIO;
constexpr real_t WIDTH = ActiveCase::WIDTH;

#if defined(CASE_RTI)
constexpr real_t U_CHAR = ActiveCase::U_CHAR;
constexpr real_t R_INIT = static_cast<real_t>(0);
constexpr real_t L_CHAR = static_cast<real_t>(NZ);
constexpr real_t REYNOLDS = ActiveCase::REYNOLDS;
constexpr real_t WEBER = ActiveCase::WEBER;
constexpr real_t GRAVITY = ActiveCase::GRAVITY;
constexpr real_t A0 = ActiveCase::A0;
constexpr real_t MU_L =
    static_cast<real_t>((static_cast<double>(RHO_L) *
                         static_cast<double>(U_CHAR) *
                         static_cast<double>(L_CHAR)) /
                        static_cast<double>(REYNOLDS));
constexpr real_t SIGMA =
    static_cast<real_t>((static_cast<double>(RHO_L) *
                         static_cast<double>(U_CHAR) *
                         static_cast<double>(U_CHAR) *
                         static_cast<double>(L_CHAR)) /
                        static_cast<double>(WEBER));
#else
constexpr real_t U_CHAR = static_cast<real_t>(0);
constexpr real_t R_INIT = ActiveCase::R_INIT;
constexpr real_t L_CHAR = R_INIT;
constexpr real_t REYNOLDS = static_cast<real_t>(0);
constexpr real_t WEBER = static_cast<real_t>(0);
constexpr real_t GRAVITY = static_cast<real_t>(0);
constexpr real_t A0 = static_cast<real_t>(0);
constexpr real_t MU_L = ActiveCase::MU_L;
constexpr real_t SIGMA = ActiveCase::SIGMA;
#endif

constexpr real_t MU_G = static_cast<real_t>(static_cast<double>(MU_L) / static_cast<double>(MU_RATIO));
constexpr real_t NU_L = static_cast<real_t>(static_cast<double>(MU_L) / static_cast<double>(RHO_L));
constexpr real_t NU_G = static_cast<real_t>(static_cast<double>(MU_G) / static_cast<double>(RHO_G));

constexpr real_t BETA_CHEM = static_cast<real_t>((static_cast<double>(12.0) * static_cast<double>(SIGMA)) /
                                                 static_cast<double>(WIDTH));
constexpr real_t KAPPA_CHEM = static_cast<real_t>(1.5) * SIGMA * WIDTH;
constexpr real_t TAU_PHI = ActiveCase::TAU_PHI;
constexpr real_t DIFF_INT =
    static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0)) *
    (TAU_PHI - static_cast<real_t>(0.5));
constexpr real_t KAPPA_INT =
    static_cast<real_t>((static_cast<double>(4.0) * static_cast<double>(DIFF_INT)) /
                        static_cast<double>(WIDTH));
constexpr real_t GAMMA = static_cast<real_t>(3.0) * KAPPA_INT;

constexpr real_t ATWOOD =
    static_cast<real_t>((static_cast<double>(RHO_L) - static_cast<double>(RHO_G)) /
                        (static_cast<double>(RHO_L) + static_cast<double>(RHO_G)));
#if defined(CASE_STATIC_DROPLET)
constexpr real_t EXPECTED_DELTA_P =
    static_cast<real_t>((static_cast<double>(2.0) * static_cast<double>(SIGMA)) /
                        static_cast<double>(R_INIT));
#else
constexpr real_t EXPECTED_DELTA_P = static_cast<real_t>(0);
#endif

constexpr bool PERIODIC_X = ActiveCase::PERIODIC_X;
constexpr bool PERIODIC_Y = ActiveCase::PERIODIC_Y;
constexpr bool PERIODIC_Z = ActiveCase::PERIODIC_Z;
#if defined(CASE_STATIC_DROPLET)
constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = ActiveCase::ENABLE_STATIC_DROPLET_DIAGNOSTICS;
#else
constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = false;
#endif

// =================================================================================================== //

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;

constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //
