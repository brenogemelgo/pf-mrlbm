#pragma once

struct StaticDropletCase
{
    static constexpr const char *NAME = "STATIC_DROPLET";

    static constexpr natural_t NX = 256;
    static constexpr natural_t NY = 256;
    static constexpr natural_t NZ = 256;

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t R_INIT = static_cast<real_t>(48.0);
    static constexpr real_t WIDTH = static_cast<real_t>(4.0);

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t RHO_RATIO = static_cast<real_t>(10000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1000.0);
    static constexpr real_t MU_L = static_cast<real_t>(1.0e-2);
    static constexpr real_t SIGMA = static_cast<real_t>(0.03);

    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;
    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = true;
};
