#pragma once

struct JetCase
{
    static constexpr const char *NAME = "jet";

    static constexpr natural_t NX = 64;
    static constexpr natural_t NY = 64;
    static constexpr natural_t NZ = 256;

    static constexpr natural_t JET_DIAMETER = 10;
    static constexpr natural_t JET_RADIUS = JET_DIAMETER / 2;

    static constexpr real_t U_CHAR = static_cast<real_t>(0.05);
    static constexpr real_t REYNOLDS = static_cast<real_t>(5000.0);
    static constexpr real_t WEBER = static_cast<real_t>(500.0);

    static constexpr real_t RHO_RATIO = static_cast<real_t>(1.0);
    static constexpr real_t MU_RATIO = static_cast<real_t>(1.0);

    static constexpr real_t WIDTH = static_cast<real_t>(4.0);
    static constexpr real_t R_INIT = static_cast<real_t>(0);
    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr bool JET_INLET_NOISE_ENABLED = true;
    static constexpr real_t JET_INLET_NOISE_SIGMA_U = static_cast<real_t>(0.08);
    static constexpr uint32_t JET_INLET_NOISE_SALT_X = 0xA341316Cu;
    static constexpr uint32_t JET_INLET_NOISE_SALT_Y = 0xC8013EA4u;

    static constexpr real_t RHO_L = static_cast<real_t>(1.0);
    static constexpr real_t RHO_G =
        static_cast<real_t>(static_cast<double>(RHO_L) / static_cast<double>(RHO_RATIO));
    static constexpr real_t NU_L =
        static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(JET_DIAMETER)) /
                            static_cast<double>(REYNOLDS));
    static constexpr real_t MU_L = RHO_L * NU_L;
    static constexpr real_t MU_G =
        static_cast<real_t>(static_cast<double>(MU_L) / static_cast<double>(MU_RATIO));
    static constexpr real_t NU_G =
        static_cast<real_t>(static_cast<double>(MU_G) / static_cast<double>(RHO_G));

    static constexpr real_t SIGMA =
        static_cast<real_t>((static_cast<double>(RHO_L) *
                             static_cast<double>(U_CHAR) *
                             static_cast<double>(U_CHAR) *
                             static_cast<double>(JET_DIAMETER)) /
                            static_cast<double>(WEBER));
    static constexpr real_t BETA_CHEM =
        static_cast<real_t>((static_cast<double>(12.0) * static_cast<double>(SIGMA)) /
                            static_cast<double>(WIDTH));
    static constexpr real_t KAPPA_CHEM = static_cast<real_t>(1.5) * SIGMA * WIDTH;
    static constexpr real_t TAU_PHI = static_cast<real_t>(1.0);
    static constexpr real_t DIFF_INT =
        static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0)) *
        (TAU_PHI - static_cast<real_t>(0.5));
    static constexpr real_t KAPPA_INT =
        static_cast<real_t>((static_cast<double>(4.0) * static_cast<double>(DIFF_INT)) /
                            static_cast<double>(WIDTH));
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

    __device__ __host__ [[nodiscard]] static inline constexpr uint32_t hash32(uint32_t value) noexcept
    {
        value ^= value >> 16;
        value *= 0x7FEB352Du;
        value ^= value >> 15;
        value *= 0x846CA68Bu;
        value ^= value >> 16;

        return value;
    }

    __device__ __host__ [[nodiscard]] static inline real_t uniform01(const uint32_t seed) noexcept
    {
        constexpr real_t inv2_32 = static_cast<real_t>(2.3283064365386963e-10);

        return (static_cast<real_t>(seed) + static_cast<real_t>(0.5)) * inv2_32;
    }

    __device__ __host__ [[nodiscard]] static inline real_t boxMuller(
        real_t rrx,
        const real_t rry) noexcept
    {
        if (rrx < static_cast<real_t>(1.0e-12))
        {
            rrx = static_cast<real_t>(1.0e-12);
        }

        constexpr real_t twoPi = static_cast<real_t>(6.2831853071795864769);
        const real_t radius = math::sqrt(-static_cast<real_t>(2) * math::log(rrx));
        const real_t theta = twoPi * rry;

        return radius * math::cos(theta);
    }

    template <uint32_t SALT>
    __device__ __host__ [[nodiscard]] static inline real_t whiteNoise(
        const natural_t x,
        const natural_t y,
        const natural_t step) noexcept
    {
        const uint32_t base =
            (0x9E3779B9u ^ SALT) ^
            static_cast<uint32_t>(x) ^
            (static_cast<uint32_t>(y) * 0x85EBCA6Bu) ^
            (static_cast<uint32_t>(step) * 0xC2B2AE35u);

        const real_t rrx = uniform01(hash32(base));
        const real_t rry = uniform01(hash32(base ^ 0x68BC21EBu));

        return boxMuller(rrx, rry);
    }

    __device__ __host__ static inline void boundaryVelocityPhi(
        const natural_t x,
        const natural_t y,
        const natural_t,
        const unsigned int,
        const real_t,
        const natural_t step,
        real_t &ubx,
        real_t &uby,
        real_t &ubz,
        real_t &phiB) noexcept
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);

        if (isInsideBackJet(x, y))
        {
            if constexpr (JET_INLET_NOISE_ENABLED)
            {
                ubx = JET_INLET_NOISE_SIGMA_U * U_CHAR * whiteNoise<JET_INLET_NOISE_SALT_X>(x, y, step);
                uby = JET_INLET_NOISE_SIGMA_U * U_CHAR * whiteNoise<JET_INLET_NOISE_SALT_Y>(x, y, step);
            }

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
