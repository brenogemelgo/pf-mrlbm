#pragma once

struct RTICase
{
    static constexpr const char *NAME = "RTI";

    // RTI presets are launched by the runRayleighTaylor*.sh scripts.
    // The CUDA case stays generic and is configured through compile-time macros:
    // A=0.5 spike/bubble trajectory validation and high-contrast presentation sweeps.

#if !defined(CASE_ONE) && !defined(CASE_TWO)
#define CASE_ONE
#endif

#if (defined(CASE_ONE) ? 1 : 0) + (defined(CASE_TWO) ? 1 : 0) != 1
#error "Select exactly one of CASE_ONE or CASE_TWO for RTI"
#endif

#ifndef RTI_NX
#define RTI_NX 256
#endif

#ifndef RTI_NY
#define RTI_NY 4
#endif

#ifndef RTI_NZ
#define RTI_NZ 512
#endif

#ifndef RTI_NSTEPS
#define RTI_NSTEPS 120000
#endif

#ifndef RTI_STAMP
#define RTI_STAMP 5000
#endif

#ifndef RTI_WIDTH
#define RTI_WIDTH 6.0
#endif

#ifndef RTI_RHO_L
#define RTI_RHO_L 3.0
#endif

#ifndef RTI_RHO_RATIO
#define RTI_RHO_RATIO 3.0
#endif

#ifndef RTI_MU_RATIO
#define RTI_MU_RATIO 1.0
#endif

#ifndef RTI_U_CHAR
#define RTI_U_CHAR 7.155417527999327e-3
#endif

#ifndef RTI_REYNOLDS
#ifdef CASE_ONE
#define RTI_REYNOLDS 256.0
#else
#define RTI_REYNOLDS 2048.0
#endif
#endif

#ifndef RTI_WEBER
#define RTI_WEBER 1.0e9
#endif

#ifndef RTI_GRAVITY
#define RTI_GRAVITY 2.0e-7
#endif

#ifndef RTI_A0
#define RTI_A0 4.0
#endif

#ifndef RTI_TAU_PHI
#define RTI_TAU_PHI 1.0
#endif

#ifndef RTI_QUASI_2D
#define RTI_QUASI_2D 1
#endif

#ifndef RTI_MASS_TOLERANCE
#define RTI_MASS_TOLERANCE 1.0e-3
#endif

    static constexpr natural_t NX = RTI_NX;
    static constexpr natural_t NY = RTI_NY;
    static constexpr natural_t NZ = RTI_NZ;

    static constexpr natural_t NSTEPS = RTI_NSTEPS;
    static constexpr natural_t STAMP = RTI_STAMP;

    static constexpr real_t WIDTH = static_cast<real_t>(RTI_WIDTH);

    static constexpr real_t RHO_L = static_cast<real_t>(RTI_RHO_L);
    static constexpr real_t RHO_RATIO = static_cast<real_t>(RTI_RHO_RATIO);
    static constexpr real_t MU_RATIO = static_cast<real_t>(RTI_MU_RATIO);

    static constexpr real_t U_CHAR = static_cast<real_t>(RTI_U_CHAR);
    static constexpr real_t REYNOLDS = static_cast<real_t>(RTI_REYNOLDS);
    static constexpr real_t WEBER = static_cast<real_t>(RTI_WEBER);
    static constexpr real_t GRAVITY = static_cast<real_t>(RTI_GRAVITY);

    static constexpr real_t A0 = static_cast<real_t>(RTI_A0);
    static constexpr real_t TAU_PHI = static_cast<real_t>(RTI_TAU_PHI);
    static constexpr real_t MASS_TOLERANCE = static_cast<real_t>(RTI_MASS_TOLERANCE);
    static constexpr bool QUASI_2D = RTI_QUASI_2D != 0;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;
};
