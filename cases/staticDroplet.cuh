#pragma once

struct StaticDropletCase
{
    static constexpr const char *NAME = "STATIC_DROPLET";

#ifdef CASE_FOUR
#error "CASE_FOUR was removed; use CASE_THREE"
#endif

#if !defined(CASE_ONE) && !defined(CASE_TWO) && !defined(CASE_THREE)
#define CASE_ONE
#endif

#if (defined(CASE_ONE) ? 1 : 0) + (defined(CASE_TWO) ? 1 : 0) + (defined(CASE_THREE) ? 1 : 0) != 1
#error "Select exactly one of CASE_ONE, CASE_TWO, or CASE_THREE"
#endif

#ifndef STATIC_DROPLET_NX
#define STATIC_DROPLET_NX 128
#endif

#ifndef STATIC_DROPLET_NY
#define STATIC_DROPLET_NY 128
#endif

#ifndef STATIC_DROPLET_NZ
#define STATIC_DROPLET_NZ 128
#endif

#ifndef STATIC_DROPLET_NSTEPS
#ifdef CASE_TWO
#define STATIC_DROPLET_NSTEPS 120000
#elif defined(CASE_THREE)
#define STATIC_DROPLET_NSTEPS 50000
#else
#define STATIC_DROPLET_NSTEPS 100000
#endif
#endif

#ifndef STATIC_DROPLET_STAMP
#ifdef CASE_TWO
#define STATIC_DROPLET_STAMP 10000
#elif defined(CASE_THREE)
#define STATIC_DROPLET_STAMP 10000
#else
#define STATIC_DROPLET_STAMP 1000
#endif
#endif

#ifndef STATIC_DROPLET_R_INIT
#ifdef CASE_TWO
#define STATIC_DROPLET_R_INIT 24.0
#elif defined(CASE_THREE)
#define STATIC_DROPLET_R_INIT 32.0
#else
#define STATIC_DROPLET_R_INIT 14.0
#endif
#endif

#ifndef STATIC_DROPLET_WIDTH
#ifdef CASE_TWO
#define STATIC_DROPLET_WIDTH 4.0
#elif defined(CASE_THREE)
#define STATIC_DROPLET_WIDTH 6.0
#else
#define STATIC_DROPLET_WIDTH 3.0
#endif
#endif

#ifndef STATIC_DROPLET_RHO_L
#define STATIC_DROPLET_RHO_L 1.0
#endif

#ifndef STATIC_DROPLET_MU_L
#ifdef CASE_TWO
#define STATIC_DROPLET_MU_L 1.5e-1
#elif defined(CASE_THREE)
#define STATIC_DROPLET_MU_L 5.0e-1
#else
#define STATIC_DROPLET_MU_L 5.0e-2
#endif
#endif

#ifndef STATIC_DROPLET_SIGMA
#define STATIC_DROPLET_SIGMA 0.02
#endif

#ifndef STATIC_DROPLET_TAU_PHI
#define STATIC_DROPLET_TAU_PHI 1.0
#endif

#ifdef CASE_ONE
    static constexpr real_t RHO_RATIO = static_cast<real_t>(1.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1.0);
#elif defined(CASE_TWO)
    static constexpr real_t RHO_RATIO = static_cast<real_t>(1000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(100.0);
#elif defined(CASE_THREE)
    static constexpr real_t RHO_RATIO = static_cast<real_t>(10000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1000.0);
#else
    static constexpr real_t RHO_RATIO = static_cast<real_t>(0.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(0.0);
#endif

    static constexpr natural_t NX = STATIC_DROPLET_NX;
    static constexpr natural_t NY = STATIC_DROPLET_NY;
    static constexpr natural_t NZ = STATIC_DROPLET_NZ;

    static constexpr natural_t NSTEPS = STATIC_DROPLET_NSTEPS;
    static constexpr natural_t STAMP = STATIC_DROPLET_STAMP;

    static constexpr real_t R_INIT = static_cast<real_t>(STATIC_DROPLET_R_INIT);
    static constexpr real_t WIDTH = static_cast<real_t>(STATIC_DROPLET_WIDTH);

    static constexpr real_t RHO_L = static_cast<real_t>(STATIC_DROPLET_RHO_L);
    static constexpr real_t MU_L = static_cast<real_t>(STATIC_DROPLET_MU_L);
    static constexpr real_t SIGMA = static_cast<real_t>(STATIC_DROPLET_SIGMA);

    static constexpr real_t TAU_PHI = static_cast<real_t>(STATIC_DROPLET_TAU_PHI);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;

    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = true;
};
