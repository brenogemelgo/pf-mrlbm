#pragma once

#include "helpers.cuh"

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
        static_cast<real_t>(cx) * loadMoment(moments, src, UX) +
        static_cast<real_t>(cy) * loadMoment(moments, src, UY) +
        static_cast<real_t>(cz) * loadMoment(moments, src, UZ);

    const real_t mh =
        loadMoment(moments, src, MXX) * VelocitySet::hxx<Q>() +
        loadMoment(moments, src, MYY) * VelocitySet::hyy<Q>() +
        loadMoment(moments, src, MZZ) * VelocitySet::hzz<Q>() +
        loadMoment(moments, src, MXY) * VelocitySet::hxy<Q>() +
        loadMoment(moments, src, MXZ) * VelocitySet::hxz<Q>() +
        loadMoment(moments, src, MYZ) * VelocitySet::hyz<Q>();

    return VelocitySet::w<Q>() * (loadMoment(moments, src, PSTAR) + cu + mh);
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

    pstar = loadMoment(moments, idx, PSTAR);
    ux = loadMoment(moments, idx, UX);
    uy = loadMoment(moments, idx, UY);
    uz = loadMoment(moments, idx, UZ);

    mxx = loadMoment(moments, idx, MXX);
    myy = loadMoment(moments, idx, MYY);
    mzz = loadMoment(moments, idx, MZZ);

    mxy = loadMoment(moments, idx, MXY);
    mxz = loadMoment(moments, idx, MXZ);
    myz = loadMoment(moments, idx, MYZ);

    phi = loadMoment(moments, idx, PHI);
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
    natural_t srcX = x;
    natural_t srcY = y;
    natural_t srcZ = z;
    srcX = x;
    srcY = y;
    srcZ = z;

    const natural_t src = global3(srcX, srcY, srcZ);

    pstar = loadMoment(moments, src, PSTAR);
    ux = loadMoment(moments, src, UX);
    uy = loadMoment(moments, src, UY);
    uz = loadMoment(moments, src, UZ);

    mxx = loadMoment(moments, src, MXX);
    myy = loadMoment(moments, src, MYY);
    mzz = loadMoment(moments, src, MZZ);

    mxy = loadMoment(moments, src, MXY);
    mxz = loadMoment(moments, src, MXZ);
    myz = loadMoment(moments, src, MYZ);

    phi = loadMoment(moments, src, PHI);
}

// ===================================================================================================================== //

__device__ static __forceinline__ void caseBoundaryState(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    const natural_t step,
    real_t &ubx,
    real_t &uby,
    real_t &ubz,
    real_t &phiB) noexcept
{
    natural_t srcX = x;
    natural_t srcY = y;
    natural_t srcZ = z;
    real_t copiedPhi = static_cast<real_t>(0);
    bool hasPhiSource = nodeType != BULK;
    if ((nodeType & XMIN_FACE) != 0u)
    {
        srcX = static_cast<natural_t>(1);
    }
    if ((nodeType & XMAX_FACE) != 0u)
    {
        srcX = NX - static_cast<natural_t>(2);
    }
    if ((nodeType & YMIN_FACE) != 0u)
    {
        srcY = static_cast<natural_t>(1);
    }
    if ((nodeType & YMAX_FACE) != 0u)
    {
        srcY = NY - static_cast<natural_t>(2);
    }
    if ((nodeType & ZMIN_FACE) != 0u)
    {
        srcZ = static_cast<natural_t>(1);
    }
    if ((nodeType & ZMAX_FACE) != 0u)
    {
        srcZ = NZ - static_cast<natural_t>(2);
    }

    if (hasPhiSource)
    {
        const natural_t src = global3(srcX, srcY, srcZ);
        copiedPhi = loadMoment(moments, src, PHI);
    }

    (void)step;
    ubx = static_cast<real_t>(0);
    uby = static_cast<real_t>(0);
    ubz = static_cast<real_t>(0);
    phiB = copiedPhi;
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
    const natural_t step,
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

    caseBoundaryState(moments, x, y, z, nodeTypeValue, step, ubx, uby, ubz, phiB);

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
    const natural_t step,
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

    caseBoundaryState(moments, x, y, z, nodeType, step, ubx, uby, ubz, phiB);

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

            if (!isMissingDirectionRuntime<Q>(nodeType))
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
    const natural_t step,
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
    if (isCopyOutflowBoundaryRuntime(nodeType))
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

    if (!hasIRBCBoundaryRuntime(nodeType))
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
                                          step,
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
                                           step,
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
                                           step,
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
                                           step,
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
                                           step,
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
                                          step,
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
                                          step,
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
                                                step,
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
                                                step,
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
                                                step,
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
                                                step,
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
                          step,
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
