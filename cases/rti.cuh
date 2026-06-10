#pragma once

struct RTICase
{
    static constexpr const char *NAME = "RTI";

    static constexpr natural_t NX = 128;
    static constexpr natural_t NY = 128;
    static constexpr natural_t NZ = 128;

    static constexpr natural_t NSTEPS = 50000;
    static constexpr natural_t STAMP = 200;

    static constexpr real_t WIDTH = static_cast<real_t>(4.0);

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t RHO_RATIO = static_cast<real_t>(10000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1000.0);

    static constexpr real_t U_CHAR = static_cast<real_t>(2.0e-2);
    static constexpr real_t REYNOLDS = static_cast<real_t>(100.0);
    static constexpr real_t WEBER = static_cast<real_t>(2500.0);
    static constexpr real_t GRAVITY = static_cast<real_t>(1.0e-6);

    static constexpr real_t A0 = static_cast<real_t>(2.0);
    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;
};
