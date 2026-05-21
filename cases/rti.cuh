#pragma once

struct RTICase
{
    static constexpr const char *NAME = "rti";

    // ============================================================================================= //
    // User-selectable RTI parameters
    // ============================================================================================= //

    static constexpr natural_t NX = 128;
    static constexpr natural_t NY = 128;
    static constexpr natural_t NZ = 128;

    static constexpr natural_t NSTEPS = 50000;
    static constexpr natural_t STAMP = 200;

    static constexpr real_t WIDTH = static_cast<real_t>(4.0);
    static constexpr real_t R_INIT = static_cast<real_t>(0);

    // Density ratio: rho_l / rho_g.
    static constexpr real_t DENSITY_RATIO = static_cast<real_t>(10000.0);

    // Dynamic-viscosity ratio: mu_l / mu_g.
    // MU_RATIO = DENSITY_RATIO gives approximately equal kinematic viscosities.
    // MU_RATIO = 1 gives equal dynamic viscosities and high gas kinematic viscosity.
    static constexpr real_t MU_RATIO = static_cast<real_t>(1000.0);

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);

    // Characteristic scales for Re and We.
    static constexpr real_t L_CHAR = static_cast<real_t>(NY);
    static constexpr real_t U_CHAR = static_cast<real_t>(2.0e-2);

    // Increase REYNOLDS for less viscosity.
    // Decrease WEBER for stronger surface tension.
    static constexpr real_t REYNOLDS = static_cast<real_t>(100.0);
    static constexpr real_t WEBER = static_cast<real_t>(2500.0);

    static constexpr real_t GRAVITY = static_cast<real_t>(1.0e-6);

    // Initial perturbation amplitude.
    // A0 = 1 is clean. Increase to 2-4 for faster visible RTI.
    static constexpr real_t A0 = static_cast<real_t>(2.0);

    // ============================================================================================= //
    // Derived parameters consumed by the solver
    // ============================================================================================= //

    static constexpr real_t RHO_G = RHO_L / DENSITY_RATIO;

    static constexpr real_t MU_L = RHO_L * U_CHAR * L_CHAR / REYNOLDS;
    static constexpr real_t MU_G = MU_L / MU_RATIO;

    static constexpr real_t NU_L = MU_L / RHO_L;
    static constexpr real_t NU_G = MU_G / RHO_G;

    static constexpr real_t TAU_L =
        static_cast<real_t>(3.0) * NU_L + static_cast<real_t>(0.5);

    static constexpr real_t TAU_G =
        static_cast<real_t>(3.0) * NU_G + static_cast<real_t>(0.5);

    static constexpr real_t SIGMA =
        RHO_L * U_CHAR * U_CHAR * L_CHAR / WEBER;

    static constexpr real_t BETA_CHEM =
        static_cast<real_t>(12.0) * SIGMA / WIDTH;

    static constexpr real_t KAPPA_CHEM =
        static_cast<real_t>(1.5) * SIGMA * WIDTH;

    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);

    static constexpr real_t DIFF_INT =
        (static_cast<real_t>(1) / static_cast<real_t>(3)) *
        (TAU_PHI - static_cast<real_t>(0.5));

    static constexpr real_t KAPPA_INT =
        static_cast<real_t>(4.0) * DIFF_INT / WIDTH;

    static constexpr real_t GAMMA =
        static_cast<real_t>(3.0) * KAPPA_INT;

    static constexpr real_t ATWOOD =
        (RHO_L - RHO_G) / (RHO_L + RHO_G);

    static constexpr real_t EXPECTED_DELTA_P = static_cast<real_t>(0);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = false;
    static constexpr bool PERIODIC_Z = true;

    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = false;

    // ============================================================================================= //
    // Neighbor policy
    // ============================================================================================= //

    __device__ __host__ [[nodiscard]] static inline int neighborX(const int x) noexcept
    {
        if (x < 0)
        {
            return static_cast<int>(NX) - 1;
        }

        if (x >= static_cast<int>(NX))
        {
            return 0;
        }

        return x;
    }

    __device__ __host__ [[nodiscard]] static inline int neighborY(const int y) noexcept
    {
        if (y < 0)
        {
            return 0;
        }

        if (y >= static_cast<int>(NY))
        {
            return static_cast<int>(NY) - 1;
        }

        return y;
    }

    __device__ __host__ [[nodiscard]] static inline int neighborZ(const int z) noexcept
    {
        if (z < 0)
        {
            return static_cast<int>(NZ) - 1;
        }

        if (z >= static_cast<int>(NZ))
        {
            return 0;
        }

        return z;
    }

    // ============================================================================================= //
    // Boundary policy
    // ============================================================================================= //

    __device__ __host__ [[nodiscard]] static inline mask_t boundaryMask(
        const natural_t,
        const natural_t y,
        const natural_t) noexcept
    {
        mask_t type = BULK;

        if (y == 0)
        {
            type |= SOUTH;
        }

        if (y == NY - static_cast<natural_t>(1))
        {
            type |= NORTH;
        }

        return type;
    }

    __device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirection(
        const unsigned int nodeType,
        const int,
        const int cy,
        const int) noexcept
    {
        return ((nodeType & SOUTH) != 0u && cy > 0) ||
               ((nodeType & NORTH) != 0u && cy < 0);
    }

    template <unsigned int nodeTypeValue>
    __host__ __device__ [[nodiscard]] static inline constexpr bool hasIRBCBoundary() noexcept
    {
        return ((nodeTypeValue & SOUTH) != 0u || (nodeTypeValue & NORTH) != 0u) &&
               !((nodeTypeValue & SOUTH) != 0u && (nodeTypeValue & NORTH) != 0u);
    }

    __device__ [[nodiscard]] static inline bool hasIRBCBoundary(
        const unsigned int nodeType) noexcept
    {
        return ((nodeType & SOUTH) != 0u || (nodeType & NORTH) != 0u) &&
               !((nodeType & SOUTH) != 0u && (nodeType & NORTH) != 0u);
    }

    template <unsigned int>
    __host__ __device__ [[nodiscard]] static inline constexpr bool isCopyOutflowBoundary() noexcept
    {
        return false;
    }

    __device__ [[nodiscard]] static inline bool isCopyOutflowBoundary(
        const unsigned int) noexcept
    {
        return false;
    }

    __device__ __host__ static inline void copyOutflowSource(
        const natural_t x,
        const natural_t y,
        const natural_t z,
        natural_t &sx,
        natural_t &sy,
        natural_t &sz) noexcept
    {
        sx = x;
        sy = y;
        sz = z;
    }

    __device__ __host__ [[nodiscard]] static inline bool boundaryPhiSource(
        const natural_t x,
        const natural_t y,
        const natural_t z,
        const unsigned int nodeType,
        natural_t &sx,
        natural_t &sy,
        natural_t &sz) noexcept
    {
        sx = x;
        sy = y;
        sz = z;

        if ((nodeType & SOUTH) != 0u)
        {
            sy = static_cast<natural_t>(1);
            return true;
        }

        if ((nodeType & NORTH) != 0u)
        {
            sy = NY - static_cast<natural_t>(2);
            return true;
        }

        return false;
    }

    __device__ __host__ static inline void boundaryVelocityPhi(
        const natural_t,
        const natural_t,
        const natural_t,
        const unsigned int,
        const real_t copiedPhi,
        real_t &ubx,
        real_t &uby,
        real_t &ubz,
        real_t &phiB) noexcept
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
        phiB = copiedPhi;
    }

    // ============================================================================================= //
    // Body force
    // ============================================================================================= //

    __device__ __host__ static inline void bodyForce(
        const natural_t,
        const natural_t,
        const natural_t,
        const real_t,
        const real_t rho,
        real_t &,
        real_t &fy,
        real_t &) noexcept
    {
        fy += -rho * GRAVITY;
    }

    // ============================================================================================= //
    // Initial condition helpers
    // ============================================================================================= //

    __device__ __host__ [[nodiscard]] static inline real_t interfaceCenterY() noexcept
    {
        return static_cast<real_t>(0.5) * static_cast<real_t>(NY);
    }

    __device__ __host__ [[nodiscard]] static inline real_t interfaceY(
        const natural_t x,
        const natural_t z) noexcept
    {
        constexpr real_t twoPi = static_cast<real_t>(6.2831853071795864769);

        const real_t kx =
            twoPi * static_cast<real_t>(x) / static_cast<real_t>(NX);

        const real_t kz =
            twoPi * static_cast<real_t>(z) / static_cast<real_t>(NZ);

        return interfaceCenterY() + A0 * math::cos(kx) * math::cos(kz);
    }

    __device__ __host__ [[nodiscard]] static inline real_t interfacePhi(
        const natural_t x,
        const natural_t y,
        const natural_t z) noexcept
    {
        return static_cast<real_t>(0.5) *
               (static_cast<real_t>(1) +
                math::tanh((static_cast<real_t>(y) - interfaceY(x, z)) /
                           (static_cast<real_t>(0.5) * WIDTH)));
    }

    __device__ __host__ [[nodiscard]] static inline real_t flatInterfacePhi(
        const natural_t y) noexcept
    {
        return static_cast<real_t>(0.5) *
               (static_cast<real_t>(1) +
                math::tanh((static_cast<real_t>(y) - interfaceCenterY()) /
                           (static_cast<real_t>(0.5) * WIDTH)));
    }

    __device__ __host__ [[nodiscard]] static inline real_t densityFromPhi(
        const real_t phi) noexcept
    {
        return RHO_G + (RHO_L - RHO_G) * phi;
    }

    // Flat hydrostatic preload avoids artificial x/z-dependent pressure gradients.
    __device__ __host__ [[nodiscard]] static inline real_t hydrostaticPressure(
        const natural_t,
        const natural_t y,
        const natural_t) noexcept
    {
        real_t p = static_cast<real_t>(0);

        for (natural_t yy = y; yy + static_cast<natural_t>(1) < NY; ++yy)
        {
            const real_t phiY = flatInterfacePhi(yy);
            const real_t rhoY = densityFromPhi(phiY);

            p += rhoY * GRAVITY;
        }

        return p;
    }

    // ============================================================================================= //
    // Initial condition
    // ============================================================================================= //

    __device__ __host__ static inline void initialCondition(
        const natural_t x,
        const natural_t y,
        const natural_t z,
        real_t &pstar,
        real_t &ux,
        real_t &uy,
        real_t &uz,
        real_t &mxx,
        real_t &myy,
        real_t &mzz,
        real_t &mxy,
        real_t &mxz,
        real_t &myz,
        real_t &phi) noexcept
    {
        phi = interfacePhi(x, y, z);

        const real_t rho = densityFromPhi(phi);
        const real_t pPhys = hydrostaticPressure(x, y, z);

        // Solver convention: pstar = p / (rho * cs2), with cs2 = 1 / 3.
        pstar = static_cast<real_t>(3.0) * pPhys / rho;

        ux = static_cast<real_t>(0);
        uy = static_cast<real_t>(0);
        uz = static_cast<real_t>(0);

        mxx = static_cast<real_t>(0);
        myy = static_cast<real_t>(0);
        mzz = static_cast<real_t>(0);
        mxy = static_cast<real_t>(0);
        mxz = static_cast<real_t>(0);
        myz = static_cast<real_t>(0);
    }
};
