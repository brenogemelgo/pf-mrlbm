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
__device__ [[nodiscard]] static __forceinline__ bool isMissingDirection(
    const unsigned int nodeType) noexcept
{
    return Case::isMissingDirection(nodeType,
                                    VelocitySet::cx<dir>(),
                                    VelocitySet::cy<dir>(),
                                    VelocitySet::cz<dir>());
}

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return Case::isMissingDirection(nodeTypeValue,
                                    VelocitySet::cx<dir>(),
                                    VelocitySet::cy<dir>(),
                                    VelocitySet::cz<dir>());
}

// ===================================================================================================================== //

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t reconstructPressureVelocityPopulation(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t src = caseNeighborIndex(static_cast<int>(x) - cx,
                                            static_cast<int>(y) - cy,
                                            static_cast<int>(z) - cz);

    const real_t cu =
        static_cast<real_t>(cx) * moments[midx(src, UX)] +
        static_cast<real_t>(cy) * moments[midx(src, UY)] +
        static_cast<real_t>(cz) * moments[midx(src, UZ)];

    const real_t mh =
        moments[midx(src, MXX)] * VelocitySet::hxx<Q>() +
        moments[midx(src, MYY)] * VelocitySet::hyy<Q>() +
        moments[midx(src, MZZ)] * VelocitySet::hzz<Q>() +
        moments[midx(src, MXY)] * VelocitySet::hxy<Q>() +
        moments[midx(src, MXZ)] * VelocitySet::hxz<Q>() +
        moments[midx(src, MYZ)] * VelocitySet::hyz<Q>();

    return VelocitySet::w<Q>() * (moments[midx(src, PSTAR)] + cu + mh);
}

// ===================================================================================================================== //

__device__ static __forceinline__ void copyCurrentBoundaryMoments(
    const real_t *__restrict__ moments,
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
    const natural_t idx = global3(x, y, z);

    pstar = moments[midx(idx, PSTAR)];
    ux = moments[midx(idx, UX)];
    uy = moments[midx(idx, UY)];
    uz = moments[midx(idx, UZ)];

    mxx = moments[midx(idx, MXX)];
    myy = moments[midx(idx, MYY)];
    mzz = moments[midx(idx, MZZ)];

    mxy = moments[midx(idx, MXY)];
    mxz = moments[midx(idx, MXZ)];
    myz = moments[midx(idx, MYZ)];

    phi = moments[midx(idx, PHI)];
}

__device__ static __forceinline__ void copyCaseOutflowMoments(
    const real_t *__restrict__ moments,
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
    natural_t srcX;
    natural_t srcY;
    natural_t srcZ;
    Case::copyOutflowSource(x, y, z, srcX, srcY, srcZ);

    const natural_t src = global3(srcX, srcY, srcZ);

    pstar = moments[midx(src, PSTAR)];
    ux = moments[midx(src, UX)];
    uy = moments[midx(src, UY)];
    uz = moments[midx(src, UZ)];

    mxx = moments[midx(src, MXX)];
    myy = moments[midx(src, MYY)];
    mzz = moments[midx(src, MZZ)];

    mxy = moments[midx(src, MXY)];
    mxz = moments[midx(src, MXZ)];
    myz = moments[midx(src, MYZ)];

    phi = moments[midx(src, PHI)];
}

// ===================================================================================================================== //

__device__ static __forceinline__ void caseBoundaryState(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz,
    real_t &phiB) noexcept
{
    natural_t srcX;
    natural_t srcY;
    natural_t srcZ;
    real_t copiedPhi = static_cast<real_t>(0);
    if (Case::boundaryPhiSource(x, y, z, nodeType, srcX, srcY, srcZ))
    {
        const natural_t src = global3(srcX, srcY, srcZ);
        copiedPhi = moments[midx(src, PHI)];
    }

    Case::boundaryVelocityPhi(x, y, z, nodeType, copiedPhi, ubx, uby, ubz, phiB);
}

// ===================================================================================================================== //

__device__ static __forceinline__ void solveIRBCSystem(
    const natural_t tableOffset,
    const real_t (&rhs)[IRBC_UNKNOWNS],
    real_t (&solved)[IRBC_UNKNOWNS]) noexcept
{
#pragma unroll
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
        real_t acc = static_cast<real_t>(0);

#pragma unroll
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            acc += IRBC_INVERSE[tableOffset + row * IRBC_UNKNOWNS + col] * rhs[col];
        }

        solved[row] = acc;
    }
}

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void applyIRBCBoundaryTyped(
    const real_t *__restrict__ moments,
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
    constexpr natural_t tableOffset =
        static_cast<natural_t>(nodeTypeValue) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    real_t phiB;

    caseBoundaryState(moments, x, y, z, nodeTypeValue, ubx, uby, ubz, phiB);

    const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

    real_t rhs[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        ub2};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            constexpr int cx = VelocitySet::cx<Q>();
            constexpr int cy = VelocitySet::cy<Q>();
            constexpr int cz = VelocitySet::cz<Q>();

            if constexpr (!isMissingDirectionConst<nodeTypeValue, Q>())
            {
                const real_t f =
                    reconstructPressureVelocityPopulation<Q>(moments, x, y, z);

                rhs[0] += f;
                rhs[1] += f * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += f * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += f * VelocitySet::hxy<Q>();
                rhs[4] += f * VelocitySet::hxz<Q>();
                rhs[5] += f * VelocitySet::hyz<Q>();
            }
            else
            {
                const real_t cuB =
                    static_cast<real_t>(cx) * ubx +
                    static_cast<real_t>(cy) * uby +
                    static_cast<real_t>(cz) * ubz;

                const real_t fB = VelocitySet::w<Q>() * cuB;

                rhs[0] += fB;
                rhs[1] += fB * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += fB * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += fB * VelocitySet::hxy<Q>();
                rhs[4] += fB * VelocitySet::hxz<Q>();
                rhs[5] += fB * VelocitySet::hyz<Q>();
            }
        });

    real_t solved[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    solveIRBCSystem(tableOffset, rhs, solved);

    pstar = solved[0];

    ux = ubx;
    uy = uby;
    uz = ubz;

    mxx = solved[1];
    myy = solved[2];
    mzz = solved[3];

    mxy = solved[4];
    mxz = solved[5];
    myz = solved[6];

    phi = phiB;
}

__device__ static __forceinline__ void applyIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
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
    const natural_t tableOffset =
        static_cast<natural_t>(nodeType) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    real_t phiB;

    caseBoundaryState(moments, x, y, z, nodeType, ubx, uby, ubz, phiB);

    const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

    real_t rhs[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        ub2};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            constexpr int cx = VelocitySet::cx<Q>();
            constexpr int cy = VelocitySet::cy<Q>();
            constexpr int cz = VelocitySet::cz<Q>();

            if (!isMissingDirection<Q>(nodeType))
            {
                const real_t f =
                    reconstructPressureVelocityPopulation<Q>(moments, x, y, z);

                rhs[0] += f;
                rhs[1] += f * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += f * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += f * VelocitySet::hxy<Q>();
                rhs[4] += f * VelocitySet::hxz<Q>();
                rhs[5] += f * VelocitySet::hyz<Q>();
            }
            else
            {
                const real_t cuB =
                    static_cast<real_t>(cx) * ubx +
                    static_cast<real_t>(cy) * uby +
                    static_cast<real_t>(cz) * ubz;

                const real_t fB = VelocitySet::w<Q>() * cuB;

                rhs[0] += fB;
                rhs[1] += fB * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += fB * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += fB * VelocitySet::hxy<Q>();
                rhs[4] += fB * VelocitySet::hxz<Q>();
                rhs[5] += fB * VelocitySet::hyz<Q>();
            }
        });

    real_t solved[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    solveIRBCSystem(tableOffset, rhs, solved);

    pstar = solved[0];

    ux = ubx;
    uy = uby;
    uz = ubz;

    mxx = solved[1];
    myy = solved[2];
    mzz = solved[3];

    mxy = solved[4];
    mxz = solved[5];
    myz = solved[6];

    phi = phiB;
}

// ===================================================================================================================== //

__device__ static __forceinline__ void dispatchIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
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
    if (Case::isCopyOutflowBoundary(nodeType))
    {
        copyCaseOutflowMoments(moments,
                               x,
                               y,
                               z,
                               pstar,
                               ux,
                               uy,
                               uz,
                               mxx,
                               myy,
                               mzz,
                               mxy,
                               mxz,
                               myz,
                               phi);
        return;
    }

    if (!Case::hasIRBCBoundary(nodeType))
    {
        copyCurrentBoundaryMoments(moments,
                                   x,
                                   y,
                                   z,
                                   pstar,
                                   ux,
                                   uy,
                                   uz,
                                   mxx,
                                   myy,
                                   mzz,
                                   mxy,
                                   mxz,
                                   myz,
                                   phi);
        return;
    }

    switch (nodeType)
    {
    case BACK_FACE:
        applyIRBCBoundaryTyped<BACK_FACE>(moments,
                                          x,
                                          y,
                                          z,
                                          pstar,
                                          ux,
                                          uy,
                                          uz,
                                          mxx,
                                          myy,
                                          mzz,
                                          mxy,
                                          mxz,
                                          myz,
                                          phi);
        return;

    case SOUTH_FACE:
        applyIRBCBoundaryTyped<SOUTH_FACE>(moments,
                                           x,
                                           y,
                                           z,
                                           pstar,
                                           ux,
                                           uy,
                                           uz,
                                           mxx,
                                           myy,
                                           mzz,
                                           mxy,
                                           mxz,
                                           myz,
                                           phi);
        return;

    case NORTH_FACE:
        applyIRBCBoundaryTyped<NORTH_FACE>(moments,
                                           x,
                                           y,
                                           z,
                                           pstar,
                                           ux,
                                           uy,
                                           uz,
                                           mxx,
                                           myy,
                                           mzz,
                                           mxy,
                                           mxz,
                                           myz,
                                           phi);
        return;

    case NORTH_BACK:
        applyIRBCBoundaryTyped<NORTH_BACK>(moments,
                                           x,
                                           y,
                                           z,
                                           pstar,
                                           ux,
                                           uy,
                                           uz,
                                           mxx,
                                           myy,
                                           mzz,
                                           mxy,
                                           mxz,
                                           myz,
                                           phi);
        return;

    case SOUTH_BACK:
        applyIRBCBoundaryTyped<SOUTH_BACK>(moments,
                                           x,
                                           y,
                                           z,
                                           pstar,
                                           ux,
                                           uy,
                                           uz,
                                           mxx,
                                           myy,
                                           mzz,
                                           mxy,
                                           mxz,
                                           myz,
                                           phi);
        return;

    case WEST_BACK:
        applyIRBCBoundaryTyped<WEST_BACK>(moments,
                                          x,
                                          y,
                                          z,
                                          pstar,
                                          ux,
                                          uy,
                                          uz,
                                          mxx,
                                          myy,
                                          mzz,
                                          mxy,
                                          mxz,
                                          myz,
                                          phi);
        return;

    case EAST_BACK:
        applyIRBCBoundaryTyped<EAST_BACK>(moments,
                                          x,
                                          y,
                                          z,
                                          pstar,
                                          ux,
                                          uy,
                                          uz,
                                          mxx,
                                          myy,
                                          mzz,
                                          mxy,
                                          mxz,
                                          myz,
                                          phi);
        return;

    case NORTH_WEST_BACK:
        applyIRBCBoundaryTyped<NORTH_WEST_BACK>(moments,
                                                x,
                                                y,
                                                z,
                                                pstar,
                                                ux,
                                                uy,
                                                uz,
                                                mxx,
                                                myy,
                                                mzz,
                                                mxy,
                                                mxz,
                                                myz,
                                                phi);
        return;

    case NORTH_EAST_BACK:
        applyIRBCBoundaryTyped<NORTH_EAST_BACK>(moments,
                                                x,
                                                y,
                                                z,
                                                pstar,
                                                ux,
                                                uy,
                                                uz,
                                                mxx,
                                                myy,
                                                mzz,
                                                mxy,
                                                mxz,
                                                myz,
                                                phi);
        return;

    case SOUTH_WEST_BACK:
        applyIRBCBoundaryTyped<SOUTH_WEST_BACK>(moments,
                                                x,
                                                y,
                                                z,
                                                pstar,
                                                ux,
                                                uy,
                                                uz,
                                                mxx,
                                                myy,
                                                mzz,
                                                mxy,
                                                mxz,
                                                myz,
                                                phi);
        return;

    case SOUTH_EAST_BACK:
        applyIRBCBoundaryTyped<SOUTH_EAST_BACK>(moments,
                                                x,
                                                y,
                                                z,
                                                pstar,
                                                ux,
                                                uy,
                                                uz,
                                                mxx,
                                                myy,
                                                mzz,
                                                mxy,
                                                mxz,
                                                myz,
                                                phi);
        return;

    default:
        applyIRBCBoundary(moments,
                          x,
                          y,
                          z,
                          nodeType,
                          pstar,
                          ux,
                          uy,
                          uz,
                          mxx,
                          myy,
                          mzz,
                          mxy,
                          mxz,
                          myz,
                          phi);
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
    else if constexpr (!Case::hasIRBCBoundary<nodeTypeValue>())
    {
        return false;
    }
    else if constexpr (Case::isCopyOutflowBoundary<nodeTypeValue>())
    {
        return false;
    }
    else
    {
        return !(((nodeTypeValue & WEST) != 0u && (nodeTypeValue & EAST) != 0u) ||
                 ((nodeTypeValue & SOUTH) != 0u && (nodeTypeValue & NORTH) != 0u));
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
    real_t densityRow[IRBC_UNKNOWNS] = {
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

            const real_t h[6] = {
                VelocitySet::hxx<Q>(),
                VelocitySet::hyy<Q>(),
                VelocitySet::hzz<Q>(),
                VelocitySet::hxy<Q>(),
                VelocitySet::hxz<Q>(),
                VelocitySet::hyz<Q>()};

            const real_t coeff[IRBC_UNKNOWNS] = {
                VelocitySet::w<Q>(),
                VelocitySet::w<Q>() * h[0],
                VelocitySet::w<Q>() * h[1],
                VelocitySet::w<Q>() * h[2],
                VelocitySet::w<Q>() * h[3],
                VelocitySet::w<Q>() * h[4],
                VelocitySet::w<Q>() * h[5]};

#pragma unroll
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                densityRow[col] -= coeff[col];
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
        matrix[0][col] = densityRow[col];

        matrix[1][col] = momentRows[0][col] - momentRows[2][col];
        matrix[2][col] = momentRows[1][col] - momentRows[2][col];

        matrix[3][col] = momentRows[3][col];
        matrix[4][col] = momentRows[4][col];
        matrix[5][col] = momentRows[5][col];

        matrix[6][col] = static_cast<real_t>(0);
    }

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
                hostTable[tableOffset + row * IRBC_UNKNOWNS + col] =
                    inv[row][col];
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
