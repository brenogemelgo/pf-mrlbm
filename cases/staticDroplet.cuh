#pragma once

struct StaticDropletCase
{
    static constexpr const char *NAME = "staticDroplet";

    static constexpr natural_t NX = 256;
    static constexpr natural_t NY = 256;
    static constexpr natural_t NZ = 256;

    static constexpr real_t R_INIT = static_cast<real_t>(48.0);
    static constexpr real_t WIDTH = static_cast<real_t>(6.0);
    static constexpr real_t RHO_RATIO = static_cast<real_t>(1.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1.0);
    static constexpr real_t SIGMA = static_cast<real_t>(0.03);
    static constexpr real_t U_CHAR = static_cast<real_t>(0);

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t RHO_G = RHO_L / RHO_RATIO;
    static constexpr real_t MU_L = static_cast<real_t>(1.0e-2);
    static constexpr real_t MU_G = MU_L / MU_RATIO;

    static constexpr real_t BETA_CHEM = static_cast<real_t>(12.0) * SIGMA / WIDTH;
    static constexpr real_t KAPPA_CHEM = static_cast<real_t>(1.5) * SIGMA * WIDTH;
    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);
    static constexpr real_t DIFF_INT = (static_cast<real_t>(1) / static_cast<real_t>(3)) * (TAU_PHI - static_cast<real_t>(0.5));
    static constexpr real_t KAPPA_INT = static_cast<real_t>(4.0) * DIFF_INT / WIDTH;
    static constexpr real_t GAMMA = static_cast<real_t>(3.0) * KAPPA_INT;
    static constexpr real_t EXPECTED_DELTA_P = static_cast<real_t>(2.0) * SIGMA / R_INIT;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;
    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = true;

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
            return static_cast<int>(NY) - 1;
        }
        if (y >= static_cast<int>(NY))
        {
            return 0;
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

    __device__ __host__ [[nodiscard]] static inline mask_t boundaryMask(
        const natural_t,
        const natural_t,
        const natural_t) noexcept
    {
        return BULK;
    }

    __device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirection(
        const unsigned int,
        const int,
        const int,
        const int) noexcept
    {
        return false;
    }

    template <unsigned int>
    __host__ __device__ [[nodiscard]] static inline constexpr bool hasIRBCBoundary() noexcept
    {
        return false;
    }

    __device__ [[nodiscard]] static inline bool hasIRBCBoundary(const unsigned int) noexcept
    {
        return false;
    }

    template <unsigned int>
    __host__ __device__ [[nodiscard]] static inline constexpr bool isCopyOutflowBoundary() noexcept
    {
        return false;
    }

    __device__ [[nodiscard]] static inline bool isCopyOutflowBoundary(const unsigned int) noexcept
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
        const natural_t,
        const natural_t,
        const natural_t,
        const unsigned int,
        natural_t &,
        natural_t &,
        natural_t &) noexcept
    {
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

    __device__ __host__ static inline void bodyForce(
        const natural_t,
        const natural_t,
        const natural_t,
        const real_t,
        const real_t,
        real_t &,
        real_t &,
        real_t &) noexcept
    {
    }

    __device__ __host__ [[nodiscard]] static inline real_t dropletPhi(
        const natural_t x,
        const natural_t y,
        const natural_t z) noexcept
    {
        const real_t xc = static_cast<real_t>(0.5) * (static_cast<real_t>(NX) - static_cast<real_t>(1));
        const real_t yc = static_cast<real_t>(0.5) * (static_cast<real_t>(NY) - static_cast<real_t>(1));
        const real_t zc = static_cast<real_t>(0.5) * (static_cast<real_t>(NZ) - static_cast<real_t>(1));

        const real_t dx = static_cast<real_t>(x) - xc;
        const real_t dy = static_cast<real_t>(y) - yc;
        const real_t dz = static_cast<real_t>(z) - zc;
        const real_t r = math::sqrt(dx * dx + dy * dy + dz * dz);

        return static_cast<real_t>(0.5) *
               (static_cast<real_t>(1) - math::tanh((r - R_INIT) / (static_cast<real_t>(0.5) * WIDTH)));
    }

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
        phi = dropletPhi(x, y, z);
        const real_t rho = RHO_G + (RHO_L - RHO_G) * phi;
        const real_t pPhys = EXPECTED_DELTA_P * phi;

        // Solver pressure variable is pstar = p_phys / (cs2 * rho), with cs2 = 1/3.
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
