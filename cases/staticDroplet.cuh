#pragma once

struct StaticDropletCase
{
    static constexpr const char *NAME = "STATIC_DROPLET";

#define CASE_ONE

#ifdef CASE_ONE
    static constexpr real_t RHO_RATIO = static_cast<real_t>(1.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1.0);
#elif defined(CASE_TWO)
    static constexpr real_t RHO_RATIO = static_cast<real_t>(100.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(100.0);
#elif defined(CASE_THREE)
    static constexpr real_t RHO_RATIO = static_cast<real_t>(1000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1000.0);
#else
    static constexpr real_t RHO_RATIO = static_cast<real_t>(10000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(10000.0);
#endif

    static constexpr natural_t NX = 128;
    static constexpr natural_t NY = 128;
    static constexpr natural_t NZ = 128;

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t R_INIT = static_cast<real_t>(24.0);
    static constexpr real_t WIDTH = static_cast<real_t>(5.0);

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t MU_L = static_cast<real_t>(5.0e-2);
    static constexpr real_t SIGMA = static_cast<real_t>(0.01);

    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;
    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = true;
};
