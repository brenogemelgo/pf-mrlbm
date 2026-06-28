#pragma once

struct RTICase
{
    static constexpr const char *NAME = "RTI";

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
#define RTI_NSTEPS 200000
#endif

#ifndef RTI_STAMP
#define RTI_STAMP 10000
#endif

#ifndef RTI_WIDTH
#define RTI_WIDTH 6.0
#endif

#ifndef RTI_RHO_L
#define RTI_RHO_L 1.0
#endif

#ifndef RTI_RHO_RATIO
#define RTI_RHO_RATIO 10000.0
#endif

#ifndef RTI_MU_RATIO
#define RTI_MU_RATIO 1000.0
#endif

#ifndef RTI_U_CHAR
#define RTI_U_CHAR 1.0e-2
#endif

#ifndef RTI_REYNOLDS
#ifdef CASE_ONE
#define RTI_REYNOLDS 100.0
#else
#define RTI_REYNOLDS 100.0
#endif
#endif

#ifndef RTI_WEBER
#ifdef CASE_TWO
#define RTI_WEBER 500.0
#else
#define RTI_WEBER 500.0
#endif
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
    static constexpr bool QUASI_2D = RTI_QUASI_2D != 0;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;
};
