#pragma once

struct JetCase
{
    static constexpr const char *NAME = "jet";

    static constexpr natural_t NX = 128;
    static constexpr natural_t NY = 128;
    static constexpr natural_t NZ = 512;

    static constexpr natural_t JET_DIAMETER = 30;
    static constexpr natural_t JET_RADIUS = JET_DIAMETER / 2;

    static constexpr real_t U_CHAR = static_cast<real_t>(0.05);
    static constexpr real_t REYNOLDS = static_cast<real_t>(1000.0);
    static constexpr real_t WEBER = static_cast<real_t>(2500.0);

    static constexpr real_t RHO_RATIO = static_cast<real_t>(1000.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1.0);

    static constexpr real_t WIDTH = static_cast<real_t>(12.0);
    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t RHO_G = RHO_L / RHO_RATIO;
    static constexpr real_t NU_L = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(JET_DIAMETER)) / static_cast<double>(REYNOLDS));
    static constexpr real_t MU_L = RHO_L * NU_L;
    static constexpr real_t MU_G = MU_L / MU_RATIO;
    static constexpr real_t NU_G = MU_G / RHO_G;

    static constexpr real_t SIGMA = static_cast<real_t>((static_cast<double>(RHO_L) * static_cast<double>(U_CHAR) * static_cast<double>(U_CHAR) * static_cast<double>(JET_DIAMETER)) / static_cast<double>(WEBER));
    static constexpr real_t BETA_CHEM = static_cast<real_t>(12.0) * SIGMA / WIDTH;
    static constexpr real_t KAPPA_CHEM = static_cast<real_t>(1.5) * SIGMA * WIDTH;
    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);
    static constexpr real_t DIFF_INT = static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0)) * (TAU_PHI - static_cast<real_t>(0.5));
    static constexpr real_t KAPPA_INT = static_cast<real_t>(4.0) * DIFF_INT / WIDTH;
    static constexpr real_t GAMMA = static_cast<real_t>(3.0) * KAPPA_INT;
    static constexpr real_t EXPECTED_DELTA_P = static_cast<real_t>(0);

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;
    static constexpr bool ENABLE_STATIC_DROPLET_DIAGNOSTICS = false;

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
            return 0;
        }
        if (z >= static_cast<int>(NZ))
        {
            return static_cast<int>(NZ) - 1;
        }
        return z;
    }

    __device__ __host__ [[nodiscard]] static inline mask_t boundaryMask(
        const natural_t,
        const natural_t,
        const natural_t z) noexcept
    {
        mask_t type = BULK;
        if (z == 0)
        {
            type |= BACK;
        }
        if (z == NZ - 1)
        {
            type |= FRONT;
        }
        return type;
    }

    __device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirection(
        const unsigned int nodeType,
        const int,
        const int,
        const int cz) noexcept
    {
        return ((nodeType & BACK) != 0u && cz > 0) ||
               ((nodeType & FRONT) != 0u && cz < 0);
    }

    template <unsigned int nodeTypeValue>
    __host__ __device__ [[nodiscard]] static inline constexpr bool hasIRBCBoundary() noexcept
    {
        return (nodeTypeValue & BACK) != 0u && (nodeTypeValue & FRONT) == 0u;
    }

    __device__ [[nodiscard]] static inline bool hasIRBCBoundary(const unsigned int nodeType) noexcept
    {
        return (nodeType & BACK) != 0u && (nodeType & FRONT) == 0u;
    }

    template <unsigned int nodeTypeValue>
    __host__ __device__ [[nodiscard]] static inline constexpr bool isCopyOutflowBoundary() noexcept
    {
        return (nodeTypeValue & FRONT) != 0u;
    }

    __device__ [[nodiscard]] static inline bool isCopyOutflowBoundary(const unsigned int nodeType) noexcept
    {
        return (nodeType & FRONT) != 0u;
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
        sz = z > static_cast<natural_t>(0) ? z - static_cast<natural_t>(1) : z;
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

    __device__ __host__ [[nodiscard]] static inline bool isInsideBackJet(
        const natural_t x,
        const natural_t y) noexcept
    {
        const real_t xc = static_cast<real_t>(0.5) *
                          (static_cast<real_t>(NX) - static_cast<real_t>(1));
        const real_t yc = static_cast<real_t>(0.5) *
                          (static_cast<real_t>(NY) - static_cast<real_t>(1));

        const real_t dx = static_cast<real_t>(x) - xc;
        const real_t dy = static_cast<real_t>(y) - yc;
        constexpr real_t r = static_cast<real_t>(JET_RADIUS);

        return dx * dx + dy * dy <= r * r;
    }

    __device__ __host__ static inline void boundaryVelocityPhi(
        const natural_t x,
        const natural_t y,
        const natural_t,
        const unsigned int,
        const real_t,
        real_t &ubx,
        real_t &uby,
        real_t &ubz,
        real_t &phiB) noexcept
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);

        if (isInsideBackJet(x, y))
        {
            ubz = U_CHAR;
            phiB = static_cast<real_t>(1);
        }
        else
        {
            ubz = static_cast<real_t>(0);
            phiB = static_cast<real_t>(0);
        }
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

    __device__ __host__ static inline void initialCondition(
        const natural_t,
        const natural_t,
        const natural_t,
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
        pstar = static_cast<real_t>(0);
        ux = static_cast<real_t>(0);
        uy = static_cast<real_t>(0);
        uz = static_cast<real_t>(0);
        mxx = static_cast<real_t>(0);
        myy = static_cast<real_t>(0);
        mzz = static_cast<real_t>(0);
        mxy = static_cast<real_t>(0);
        mxz = static_cast<real_t>(0);
        myz = static_cast<real_t>(0);
        phi = static_cast<real_t>(0);
    }
};
