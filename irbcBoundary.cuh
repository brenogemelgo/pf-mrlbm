#pragma once

#include "bitmasks.cuh"
#include "deviceFunctions.cuh"

// ===================================================================================================================== //

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_SIZE = 64 * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

// ===================================================================================================================== //

template <natural_t dir>
__device__ [[nodiscard]] static __forceinline__ bool isMissingDirection(const unsigned int nodeType) noexcept
{
    return (((nodeType & WEST) == WEST) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeType & EAST) == EAST) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeType & SOUTH) == SOUTH) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeType & NORTH) == NORTH) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeType & BACK) == BACK) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeType & FRONT) == FRONT) && (VelocitySet::cz<dir>() < 0));
}

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & WEST) == WEST) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeTypeValue & EAST) == EAST) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeTypeValue & SOUTH) == SOUTH) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeTypeValue & NORTH) == NORTH) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeTypeValue & BACK) == BACK) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeTypeValue & FRONT) == FRONT) && (VelocitySet::cz<dir>() < 0));
}

// ===================================================================================================================== //

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t reconstructPopulation(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t src = global3(static_cast<natural_t>(static_cast<int>(x) - cx),
                                  static_cast<natural_t>(static_cast<int>(y) - cy),
                                  static_cast<natural_t>(static_cast<int>(z) - cz));

    const real_t cu = static_cast<real_t>(cx) * moments[midx(src, UX)] +
                      static_cast<real_t>(cy) * moments[midx(src, UY)] +
                      static_cast<real_t>(cz) * moments[midx(src, UZ)];

    const real_t mh = moments[midx(src, MXX)] * VelocitySet::hxx<Q>() +
                      moments[midx(src, MYY)] * VelocitySet::hyy<Q>() +
                      moments[midx(src, MZZ)] * VelocitySet::hzz<Q>() +
                      moments[midx(src, MXY)] * VelocitySet::hxy<Q>() +
                      moments[midx(src, MXZ)] * VelocitySet::hxz<Q>() +
                      moments[midx(src, MYZ)] * VelocitySet::hyz<Q>();

    const real_t wrho = VelocitySet::w<Q>() * moments[midx(src, RHO)];
    return __fmaf_rn(wrho, cu + mh, wrho);
}

// ===================================================================================================================== //

__device__ static __forceinline__ void boundaryVelocity(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if ((nodeType & FRONT) == FRONT)
    {
        ubx = U_CHAR;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void boundaryVelocityConst(
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if constexpr ((nodeTypeValue & FRONT) == FRONT)
    {
        ubx = U_CHAR;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void applyIRBCBoundaryTyped(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    constexpr natural_t tableOffset = static_cast<natural_t>(nodeTypeValue) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityConst<nodeTypeValue>(ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0)};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if constexpr (!isMissingDirectionConst<nodeTypeValue, Q>())
            {
                const real_t f = reconstructPopulation<Q>(moments, x, y, z);

                rhs[0] += f;
                rhs[1] += f * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += f * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += f * VelocitySet::hxy<Q>();
                rhs[4] += f * VelocitySet::hxz<Q>();
                rhs[5] += f * VelocitySet::hyz<Q>();
            }
        });

    rhs[6] = static_cast<real_t>(0);

    real_t solved[IRBC_UNKNOWNS] = {static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0)};

    solved[0] = __fmaf_rn(IRBC_INVERSE[tableOffset + 0], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 1], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 2], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 3], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 4], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 5], rhs[5], IRBC_INVERSE[tableOffset + 6] * rhs[6]))))));
    solved[1] = __fmaf_rn(IRBC_INVERSE[tableOffset + 7], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 8], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 9], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 10], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 11], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 12], rhs[5], IRBC_INVERSE[tableOffset + 13] * rhs[6]))))));
    solved[2] = __fmaf_rn(IRBC_INVERSE[tableOffset + 14], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 15], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 16], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 17], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 18], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 19], rhs[5], IRBC_INVERSE[tableOffset + 20] * rhs[6]))))));
    solved[3] = __fmaf_rn(IRBC_INVERSE[tableOffset + 21], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 22], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 23], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 24], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 25], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 26], rhs[5], IRBC_INVERSE[tableOffset + 27] * rhs[6]))))));
    solved[4] = __fmaf_rn(IRBC_INVERSE[tableOffset + 28], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 29], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 30], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 31], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 32], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 33], rhs[5], IRBC_INVERSE[tableOffset + 34] * rhs[6]))))));
    solved[5] = __fmaf_rn(IRBC_INVERSE[tableOffset + 35], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 36], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 37], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 38], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 39], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 40], rhs[5], IRBC_INVERSE[tableOffset + 41] * rhs[6]))))));
    solved[6] = __fmaf_rn(IRBC_INVERSE[tableOffset + 42], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 43], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 44], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 45], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 46], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 47], rhs[5], IRBC_INVERSE[tableOffset + 48] * rhs[6]))))));

    rho = solved[0];

    const real_t invRho = static_cast<real_t>(1) / rho;

    ux = ubx;
    uy = uby;
    uz = ubz;

    mxx = solved[1] * invRho;
    myy = solved[2] * invRho;
    mzz = solved[3] * invRho;
    mxy = solved[4] * invRho;
    mxz = solved[5] * invRho;
    myz = solved[6] * invRho;
}

__device__ static __forceinline__ void applyIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    const natural_t tableOffset = static_cast<natural_t>(nodeType) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocity(nodeType, ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0)};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if (!isMissingDirection<Q>(nodeType))
            {
                const real_t f = reconstructPopulation<Q>(moments, x, y, z);

                rhs[0] += f;
                rhs[1] += f * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += f * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += f * VelocitySet::hxy<Q>();
                rhs[4] += f * VelocitySet::hxz<Q>();
                rhs[5] += f * VelocitySet::hyz<Q>();
            }
        });

    rhs[6] = static_cast<real_t>(0);

    real_t solved[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    solved[0] = __fmaf_rn(IRBC_INVERSE[tableOffset + 0], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 1], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 2], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 3], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 4], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 5], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 6], rhs[6], solved[0])))))));
    solved[1] = __fmaf_rn(IRBC_INVERSE[tableOffset + 7], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 8], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 9], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 10], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 11], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 12], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 13], rhs[6], solved[1])))))));
    solved[2] = __fmaf_rn(IRBC_INVERSE[tableOffset + 14], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 15], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 16], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 17], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 18], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 19], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 20], rhs[6], solved[2])))))));
    solved[3] = __fmaf_rn(IRBC_INVERSE[tableOffset + 21], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 22], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 23], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 24], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 25], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 26], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 27], rhs[6], solved[3])))))));
    solved[4] = __fmaf_rn(IRBC_INVERSE[tableOffset + 28], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 29], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 30], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 31], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 32], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 33], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 34], rhs[6], solved[4])))))));
    solved[5] = __fmaf_rn(IRBC_INVERSE[tableOffset + 35], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 36], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 37], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 38], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 39], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 40], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 41], rhs[6], solved[5])))))));
    solved[6] = __fmaf_rn(IRBC_INVERSE[tableOffset + 42], rhs[0], __fmaf_rn(IRBC_INVERSE[tableOffset + 43], rhs[1], __fmaf_rn(IRBC_INVERSE[tableOffset + 44], rhs[2], __fmaf_rn(IRBC_INVERSE[tableOffset + 45], rhs[3], __fmaf_rn(IRBC_INVERSE[tableOffset + 46], rhs[4], __fmaf_rn(IRBC_INVERSE[tableOffset + 47], rhs[5], __fmaf_rn(IRBC_INVERSE[tableOffset + 48], rhs[6], solved[6])))))));

    rho = solved[0];

    const real_t invRho = static_cast<real_t>(1) / rho;

    ux = ubx;
    uy = uby;
    uz = ubz;

    mxx = solved[1] * invRho;
    myy = solved[2] * invRho;
    mzz = solved[3] * invRho;
    mxy = solved[4] * invRho;
    mxz = solved[5] * invRho;
    myz = solved[6] * invRho;
}

__device__ static __forceinline__ void dispatchIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    switch (nodeType)
    {
    case WEST_FACE:
        applyIRBCBoundaryTyped<WEST_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_FACE:
        applyIRBCBoundaryTyped<EAST_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_FACE:
        applyIRBCBoundaryTyped<SOUTH_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_FACE:
        applyIRBCBoundaryTyped<NORTH_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case BACK_FACE:
        applyIRBCBoundaryTyped<BACK_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case FRONT_FACE:
        applyIRBCBoundaryTyped<FRONT_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    case NORTH_WEST:
        applyIRBCBoundaryTyped<NORTH_WEST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST:
        applyIRBCBoundaryTyped<NORTH_EAST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_FRONT:
        applyIRBCBoundaryTyped<NORTH_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_BACK:
        applyIRBCBoundaryTyped<NORTH_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    case SOUTH_WEST:
        applyIRBCBoundaryTyped<SOUTH_WEST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST:
        applyIRBCBoundaryTyped<SOUTH_EAST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_FRONT:
        applyIRBCBoundaryTyped<SOUTH_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_BACK:
        applyIRBCBoundaryTyped<SOUTH_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    case WEST_FRONT:
        applyIRBCBoundaryTyped<WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case WEST_BACK:
        applyIRBCBoundaryTyped<WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_FRONT:
        applyIRBCBoundaryTyped<EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_BACK:
        applyIRBCBoundaryTyped<EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    case NORTH_WEST_FRONT:
        applyIRBCBoundaryTyped<NORTH_WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_WEST_BACK:
        applyIRBCBoundaryTyped<NORTH_WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST_FRONT:
        applyIRBCBoundaryTyped<NORTH_EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST_BACK:
        applyIRBCBoundaryTyped<NORTH_EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    case SOUTH_WEST_FRONT:
        applyIRBCBoundaryTyped<SOUTH_WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_WEST_BACK:
        applyIRBCBoundaryTyped<SOUTH_WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST_FRONT:
        applyIRBCBoundaryTyped<SOUTH_EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST_BACK:
        applyIRBCBoundaryTyped<SOUTH_EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;

    default:
        applyIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    }
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__host__ [[nodiscard]] static inline constexpr bool isValidBoundaryTypeConst() noexcept
{
    if constexpr (nodeTypeValue == BULK)
    {
        return false;
    }
    else
    {
        return !(((nodeTypeValue & WEST) != 0u && (nodeTypeValue & EAST) != 0u) ||
                 ((nodeTypeValue & SOUTH) != 0u && (nodeTypeValue & NORTH) != 0u) ||
                 ((nodeTypeValue & BACK) != 0u && (nodeTypeValue & FRONT) != 0u));
    }
}

__host__ static inline void boundaryVelocityHost(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if ((nodeType & FRONT) == FRONT)
    {
        ubx = U_CHAR;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

__host__ static inline void invertIRBCMatrix(
    real_t (&a)[IRBC_UNKNOWNS][IRBC_UNKNOWNS],
    real_t (&inv)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            inv[row][col] = row == col ? static_cast<real_t>(1) : static_cast<real_t>(0);
        }
    }

    for (natural_t pivot = 0; pivot < IRBC_UNKNOWNS; ++pivot)
    {
        natural_t pivotRow = pivot;
        real_t pivotAbs = a[pivot][pivot] < static_cast<real_t>(0)
                              ? -a[pivot][pivot]
                              : a[pivot][pivot];

        for (natural_t row = pivot + 1; row < IRBC_UNKNOWNS; ++row)
        {
            const real_t valueAbs = a[row][pivot] < static_cast<real_t>(0)
                                        ? -a[row][pivot]
                                        : a[row][pivot];

            if (valueAbs > pivotAbs)
            {
                pivotAbs = valueAbs;
                pivotRow = row;
            }
        }

        if (pivotRow != pivot)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                const real_t tmpA = a[pivot][col];
                a[pivot][col] = a[pivotRow][col];
                a[pivotRow][col] = tmpA;

                const real_t tmpInv = inv[pivot][col];
                inv[pivot][col] = inv[pivotRow][col];
                inv[pivotRow][col] = tmpInv;
            }
        }

        const real_t pivotValue = a[pivot][pivot];
        const real_t invPivot = static_cast<real_t>(1) / pivotValue;

        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            a[pivot][col] *= invPivot;
            inv[pivot][col] *= invPivot;
        }

        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const real_t factor = a[row][pivot];

            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
                inv[row][col] -= factor * inv[pivot][col];
            }
        }
    }
}

template <unsigned int nodeTypeValue>
__host__ static inline void assembleIRBCInverse(
    real_t (&invOut)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityHost(nodeTypeValue, ubx, uby, ubz);

    const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

    real_t density[IRBC_UNKNOWNS] = {
        static_cast<real_t>(1),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    real_t momentRows[6][IRBC_UNKNOWNS] = {};

    momentRows[0][1] = static_cast<real_t>(1);
    momentRows[1][2] = static_cast<real_t>(1);
    momentRows[2][3] = static_cast<real_t>(1);
    momentRows[3][4] = static_cast<real_t>(1);
    momentRows[4][5] = static_cast<real_t>(1);
    momentRows[5][6] = static_cast<real_t>(1);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if constexpr (!isMissingDirectionConst<nodeTypeValue, Q>())
            {
                return;
            }

            const real_t cx = static_cast<real_t>(VelocitySet::cx<Q>());
            const real_t cy = static_cast<real_t>(VelocitySet::cy<Q>());
            const real_t cz = static_cast<real_t>(VelocitySet::cz<Q>());

            const real_t h[6] = {
                VelocitySet::hxx<Q>(),
                VelocitySet::hyy<Q>(),
                VelocitySet::hzz<Q>(),
                VelocitySet::hxy<Q>(),
                VelocitySet::hxz<Q>(),
                VelocitySet::hyz<Q>()};

            const real_t cu = ubx * cx + uby * cy + ubz * cz;

            const real_t meqH =
                static_cast<real_t>(0.5) * VelocitySet::as4() *
                    (ubx * ubx * h[0] +
                     uby * uby * h[1] +
                     ubz * ubz * h[2]) +
                VelocitySet::as4() *
                    (ubx * uby * h[3] +
                     ubx * ubz * h[4] +
                     uby * ubz * h[5]);

            const real_t coeff[IRBC_UNKNOWNS] = {
                VelocitySet::w<Q>() *
                    (static_cast<real_t>(1) +
                     VelocitySet::as2() * cu +
                     OMEGA * meqH),

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[0],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[1],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[2],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[3],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[4],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[5]};

#pragma unroll
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                density[col] -= coeff[col];
            }

#pragma unroll
            for (natural_t row = 0; row < 6; ++row)
            {
#pragma unroll
                for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
                {
                    momentRows[row][col] -= h[row] * coeff[col];
                }
            }
        });

    real_t matrix[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};

#pragma unroll
    for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
    {
        matrix[0][col] = density[col];
        matrix[1][col] = momentRows[0][col] - momentRows[2][col];
        matrix[2][col] = momentRows[1][col] - momentRows[2][col];
        matrix[3][col] = momentRows[3][col];
        matrix[4][col] = momentRows[4][col];
        matrix[5][col] = momentRows[5][col];
        matrix[6][col] = static_cast<real_t>(0);
    }

    matrix[6][0] = -ub2;
    matrix[6][1] = static_cast<real_t>(1);
    matrix[6][2] = static_cast<real_t>(1);
    matrix[6][3] = static_cast<real_t>(1);

    invertIRBCMatrix(matrix, invOut);
}

template <unsigned int nodeTypeValue>
__host__ static inline void fillIRBCBoundaryTableEntry(
    real_t (&hostTable)[IRBC_TABLE_SIZE]) noexcept
{
    if constexpr (!isValidBoundaryTypeConst<nodeTypeValue>())
    {
        return;
    }
    else
    {
        real_t inv[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};
        assembleIRBCInverse<nodeTypeValue>(inv);

        constexpr natural_t tableOffset =
            static_cast<natural_t>(nodeTypeValue) * IRBC_TABLE_STRIDE;

#pragma unroll
        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
#pragma unroll
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                hostTable[tableOffset + row * IRBC_UNKNOWNS + col] = inv[row][col];
            }
        }
    }
}

__host__ [[nodiscard]] static inline cudaError_t initIRBCBoundaryTables() noexcept
{
    real_t hostTable[IRBC_TABLE_SIZE] = {};

    constexpr_for<0, 64>(
        [&](const auto nodeTypeConst) noexcept
        {
            fillIRBCBoundaryTableEntry<nodeTypeConst>(hostTable);
        });

    return cudaMemcpyToSymbol(IRBC_INVERSE, hostTable, sizeof(hostTable));
}