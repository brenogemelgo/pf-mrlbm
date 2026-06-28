#pragma once

#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"

__device__ static __forceinline__ void applyBodyForce(
    const real_t rho,
    real_t &forceX,
    real_t &forceY,
    real_t &forceZ) noexcept
{
    (void)rho;
    (void)forceX;
    (void)forceY;

    if constexpr (CASE_IS_RTI)
    {
        forceZ += -rho * GRAVITY;
    }
    else
    {
        (void)forceZ;
    }
}

template <typename Set, natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t velocityCu(
    const real_t *__restrict__ moments,
    const natural_t idx) noexcept
{
    constexpr int cx = Set::template cx<Q>();
    constexpr int cy = Set::template cy<Q>();
    constexpr int cz = Set::template cz<Q>();

    real_t cu = static_cast<real_t>(0);
    if constexpr (cx != 0)
    {
        cu += static_cast<real_t>(cx) * loadMoment(moments, idx, UX);
    }
    if constexpr (cy != 0)
    {
        cu += static_cast<real_t>(cy) * loadMoment(moments, idx, UY);
    }
    if constexpr (cz != 0)
    {
        cu += static_cast<real_t>(cz) * loadMoment(moments, idx, UZ);
    }

    return cu;
}

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t phaseVelocityCu(
    const real_t *__restrict__ moments,
    const natural_t idx) noexcept
{
    return velocityCu<VelocitySet, Q>(moments, idx);
}

constexpr natural_t STREAM_PHI_HALO = 2;
constexpr natural_t STREAM_NORMAL_HALO = 1;
constexpr natural_t STREAM_PHI_TILE_NX = BLOCK_NX + 2 * STREAM_PHI_HALO;
constexpr natural_t STREAM_PHI_TILE_NY = BLOCK_NY + 2 * STREAM_PHI_HALO;
constexpr natural_t STREAM_PHI_TILE_NZ = BLOCK_NZ + 2 * STREAM_PHI_HALO;
constexpr natural_t STREAM_PHI_TILE_STRIDE = STREAM_PHI_TILE_NX * STREAM_PHI_TILE_NY;
constexpr natural_t STREAM_PHI_TILE_SIZE = STREAM_PHI_TILE_STRIDE * STREAM_PHI_TILE_NZ;
constexpr natural_t STREAM_NORMAL_TILE_NX = BLOCK_NX + 2 * STREAM_NORMAL_HALO;
constexpr natural_t STREAM_NORMAL_TILE_NY = BLOCK_NY + 2 * STREAM_NORMAL_HALO;
constexpr natural_t STREAM_NORMAL_TILE_NZ = BLOCK_NZ + 2 * STREAM_NORMAL_HALO;
constexpr natural_t STREAM_NORMAL_TILE_STRIDE = STREAM_NORMAL_TILE_NX * STREAM_NORMAL_TILE_NY;
constexpr natural_t STREAM_NORMAL_TILE_SIZE = STREAM_NORMAL_TILE_STRIDE * STREAM_NORMAL_TILE_NZ;
constexpr natural_t STREAM_BLOCK_THREADS = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

__device__ [[nodiscard]] static __forceinline__ natural_t streamSharedPhiIndex(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x + y * STREAM_PHI_TILE_NX + z * STREAM_PHI_TILE_STRIDE;
}

__device__ [[nodiscard]] static __forceinline__ natural_t streamSharedNormalIndex(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x + y * STREAM_NORMAL_TILE_NX + z * STREAM_NORMAL_TILE_STRIDE;
}

__device__ [[nodiscard]] static __forceinline__ natural_t resolveSharedTileLoadCoordinate(
    const int value,
    const int extent,
    const bool periodic) noexcept
{
    if (periodic)
    {
        int wrapped = value % extent;
        if (wrapped < 0)
        {
            wrapped += extent;
        }

        return static_cast<natural_t>(wrapped);
    }

    if (value < 0)
    {
        return static_cast<natural_t>(0);
    }

    if (value >= extent)
    {
        return static_cast<natural_t>(extent - 1);
    }

    return static_cast<natural_t>(value);
}

__device__ [[nodiscard]] static __forceinline__ int streamNormalCenterCoordinate(
    const int value,
    const int extent,
    const bool periodic) noexcept
{
    if (periodic)
    {
        return value;
    }

    if (value < 0)
    {
        return 0;
    }

    if (value >= extent)
    {
        return extent - 1;
    }

    return value;
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateSharedGradientDirection(
    const real_t *__restrict__ sharedPhi,
    const int localCenterX,
    const int localCenterY,
    const int localCenterZ,
    real_t &gradx,
    real_t &grady,
    real_t &gradz) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t localPhi = streamSharedPhiIndex(
            static_cast<natural_t>(localCenterX + cx),
            static_cast<natural_t>(localCenterY + cy),
            static_cast<natural_t>(localCenterZ + cz));
        const real_t weightedPhi = VelocitySet::w<Q>() * sharedPhi[localPhi];

        if constexpr (cx != 0)
        {
            gradx += static_cast<real_t>(cx) * weightedPhi;
        }
        if constexpr (cy != 0)
        {
            grady += static_cast<real_t>(cy) * weightedPhi;
        }
        if constexpr (cz != 0)
        {
            gradz += static_cast<real_t>(cz) * weightedPhi;
        }
    }
}

__device__ static __forceinline__ void computeSharedNormal(
    const real_t *__restrict__ sharedPhi,
    const int baseX,
    const int baseY,
    const int baseZ,
    const natural_t localNormalX,
    const natural_t localNormalY,
    const natural_t localNormalZ,
    real_t &normalX,
    real_t &normalY,
    real_t &normalZ) noexcept
{
    const int rawCenterX = baseX + static_cast<int>(localNormalX) - static_cast<int>(STREAM_NORMAL_HALO);
    const int rawCenterY = baseY + static_cast<int>(localNormalY) - static_cast<int>(STREAM_NORMAL_HALO);
    const int rawCenterZ = baseZ + static_cast<int>(localNormalZ) - static_cast<int>(STREAM_NORMAL_HALO);

    const int centerX = streamNormalCenterCoordinate(rawCenterX, static_cast<int>(NX), PERIODIC_X);
    const int centerY = streamNormalCenterCoordinate(rawCenterY, static_cast<int>(NY), PERIODIC_Y);
    const int centerZ = streamNormalCenterCoordinate(rawCenterZ, static_cast<int>(NZ), PERIODIC_Z);

    const int localCenterX = centerX - baseX + static_cast<int>(STREAM_PHI_HALO);
    const int localCenterY = centerY - baseY + static_cast<int>(STREAM_PHI_HALO);
    const int localCenterZ = centerZ - baseZ + static_cast<int>(STREAM_PHI_HALO);

    real_t gradx = static_cast<real_t>(0);
    real_t grady = static_cast<real_t>(0);
    real_t gradz = static_cast<real_t>(0);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateSharedGradientDirection<Q>(
                sharedPhi, localCenterX, localCenterY, localCenterZ, gradx, grady, gradz);
        });

    gradx *= VelocitySet::as2();
    grady *= VelocitySet::as2();
    gradz *= VelocitySet::as2();

    const real_t gradNorm =
        math::sqrt(gradx * gradx + grady * grady + gradz * gradz) +
        static_cast<real_t>(1.0e-9);

    const real_t invGradNorm = static_cast<real_t>(1) / gradNorm;

    normalX = gradx * invGradNorm;
    normalY = grady * invGradNorm;
    normalZ = gradz * invGradNorm;
}

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t sharedNormalProjection(
    const real_t *__restrict__ sharedNormX,
    const real_t *__restrict__ sharedNormY,
    const real_t *__restrict__ sharedNormZ,
    const natural_t localNormal) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    real_t projection = static_cast<real_t>(0);
    if constexpr (cx != 0)
    {
        projection += static_cast<real_t>(cx) * sharedNormX[localNormal];
    }
    if constexpr (cy != 0)
    {
        projection += static_cast<real_t>(cy) * sharedNormY[localNormal];
    }
    if constexpr (cz != 0)
    {
        projection += static_cast<real_t>(cz) * sharedNormZ[localNormal];
    }

    return projection;
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateGradientDirection(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &gradx,
    real_t &grady,
    real_t &gradz) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const real_t weightedPhi = VelocitySet::w<Q>() * loadMoment(moments, src, PHI);

        if constexpr (cx != 0)
        {
            gradx += static_cast<real_t>(cx) * weightedPhi;
        }
        if constexpr (cy != 0)
        {
            grady += static_cast<real_t>(cy) * weightedPhi;
        }
        if constexpr (cz != 0)
        {
            gradz += static_cast<real_t>(cz) * weightedPhi;
        }
    }
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateHydroStreamDirection(
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
    real_t &myz) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t src = caseNeighborIndex(static_cast<int>(x) - cx,
                                            static_cast<int>(y) - cy,
                                            static_cast<int>(z) - cz);

    const real_t cu = velocityCu<VelocitySet, Q>(moments, src);

    real_t mh = loadMoment(moments, src, MXX) * VelocitySet::hxx<Q>() +
                loadMoment(moments, src, MYY) * VelocitySet::hyy<Q>() +
                loadMoment(moments, src, MZZ) * VelocitySet::hzz<Q>();

    if constexpr (cx * cy != 0)
    {
        mh += loadMoment(moments, src, MXY) * VelocitySet::hxy<Q>();
    }
    if constexpr (cx * cz != 0)
    {
        mh += loadMoment(moments, src, MXZ) * VelocitySet::hxz<Q>();
    }
    if constexpr (cy * cz != 0)
    {
        mh += loadMoment(moments, src, MYZ) * VelocitySet::hyz<Q>();
    }

    const real_t fi = VelocitySet::w<Q>() * (loadMoment(moments, src, PSTAR) + cu + mh);

    pstar += fi;
    if constexpr (cx != 0)
    {
        ux += fi * static_cast<real_t>(cx);
    }
    if constexpr (cy != 0)
    {
        uy += fi * static_cast<real_t>(cy);
    }
    if constexpr (cz != 0)
    {
        uz += fi * static_cast<real_t>(cz);
    }

    mxx += fi * VelocitySet::hxx<Q>();
    myy += fi * VelocitySet::hyy<Q>();
    mzz += fi * VelocitySet::hzz<Q>();

    if constexpr (cx * cy != 0)
    {
        mxy += fi * VelocitySet::hxy<Q>();
    }
    if constexpr (cx * cz != 0)
    {
        mxz += fi * VelocitySet::hxz<Q>();
    }
    if constexpr (cy * cz != 0)
    {
        myz += fi * VelocitySet::hyz<Q>();
    }
}

template <natural_t Q>
__device__ static __forceinline__ void accumulatePhaseStreamDirectionShared(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ sharedNormX,
    const real_t *__restrict__ sharedNormY,
    const real_t *__restrict__ sharedNormZ,
    const int baseX,
    const int baseY,
    const int baseZ,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &phi) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const int srcRawX = static_cast<int>(x) - cx;
    const int srcRawY = static_cast<int>(y) - cy;
    const int srcRawZ = static_cast<int>(z) - cz;

    const natural_t src = caseNeighborIndex(srcRawX, srcRawY, srcRawZ);

    const int normalCenterX = streamNormalCenterCoordinate(srcRawX, static_cast<int>(NX), PERIODIC_X);
    const int normalCenterY = streamNormalCenterCoordinate(srcRawY, static_cast<int>(NY), PERIODIC_Y);
    const int normalCenterZ = streamNormalCenterCoordinate(srcRawZ, static_cast<int>(NZ), PERIODIC_Z);

    const natural_t localNormal = streamSharedNormalIndex(
        static_cast<natural_t>(normalCenterX - baseX + static_cast<int>(STREAM_NORMAL_HALO)),
        static_cast<natural_t>(normalCenterY - baseY + static_cast<int>(STREAM_NORMAL_HALO)),
        static_cast<natural_t>(normalCenterZ - baseZ + static_cast<int>(STREAM_NORMAL_HALO)));

    const real_t phi_src = loadMoment(moments, src, PHI);
    const real_t cu = phaseVelocityCu<Q>(moments, src);
    const real_t projection = sharedNormalProjection<Q>(sharedNormX, sharedNormY, sharedNormZ, localNormal);

    const real_t gi = VelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
                      VelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) * projection;

    phi += gi;
}

template <natural_t Q>
__device__ static __forceinline__ void accumulatePhaseGradientDirection(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &dphix,
    real_t &dphiy,
    real_t &dphiz) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const real_t weightedPhi = VelocitySet::w<Q>() * loadMoment(dbuffer, src, PHI);

        if constexpr (cx != 0)
        {
            dphix += static_cast<real_t>(cx) * weightedPhi;
        }
        if constexpr (cy != 0)
        {
            dphiy += static_cast<real_t>(cy) * weightedPhi;
        }
        if constexpr (cz != 0)
        {
            dphiz += static_cast<real_t>(cz) * weightedPhi;
        }
    }
}

template <natural_t Q>
__device__ static __forceinline__ void accumulatePhaseGradientLaplacianDirection(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const real_t phi,
    real_t &dphix,
    real_t &dphiy,
    real_t &dphiz,
    real_t &lapAcc) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const real_t phi_q = loadMoment(dbuffer, src, PHI);
        const real_t weightedPhi = VelocitySet::w<Q>() * phi_q;

        if constexpr (cx != 0)
        {
            dphix += static_cast<real_t>(cx) * weightedPhi;
        }
        if constexpr (cy != 0)
        {
            dphiy += static_cast<real_t>(cy) * weightedPhi;
        }
        if constexpr (cz != 0)
        {
            dphiz += static_cast<real_t>(cz) * weightedPhi;
        }

        lapAcc += VelocitySet::w<Q>() * (phi_q - phi);
    }
}

__device__ static __forceinline__ void computePhaseGradient(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &dphix,
    real_t &dphiy,
    real_t &dphiz) noexcept
{
    dphix = static_cast<real_t>(0);
    dphiy = static_cast<real_t>(0);
    dphiz = static_cast<real_t>(0);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulatePhaseGradientDirection<Q>(dbuffer, x, y, z, dphix, dphiy, dphiz);
        });

    dphix *= VelocitySet::as2();
    dphiy *= VelocitySet::as2();
    dphiz *= VelocitySet::as2();
}

__device__ static __forceinline__ real_t computePhaseGradientLaplacian(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const real_t phi,
    real_t &dphix,
    real_t &dphiy,
    real_t &dphiz) noexcept
{
    dphix = static_cast<real_t>(0);
    dphiy = static_cast<real_t>(0);
    dphiz = static_cast<real_t>(0);
    real_t lapAcc = static_cast<real_t>(0);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulatePhaseGradientLaplacianDirection<Q>(
                dbuffer, x, y, z, phi, dphix, dphiy, dphiz, lapAcc);
        });

    dphix *= VelocitySet::as2();
    dphiy *= VelocitySet::as2();
    dphiz *= VelocitySet::as2();

    return static_cast<real_t>(2) * lapAcc * VelocitySet::as2();
}

__device__ static __forceinline__ void computePhaseNormalIndicator(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &normx,
    real_t &normy,
    real_t &normz,
    real_t &indicator,
    real_t &dphix,
    real_t &dphiy,
    real_t &dphiz) noexcept
{
    computePhaseGradient(dbuffer, x, y, z, dphix, dphiy, dphiz);

    indicator = math::sqrt(dphix * dphix + dphiy * dphiy + dphiz * dphiz);
    const real_t invIndicator =
        static_cast<real_t>(1) / (indicator + static_cast<real_t>(1.0e-9));

    normx = dphix * invIndicator;
    normy = dphiy * invIndicator;
    normz = dphiz * invIndicator;
}

__device__ static __forceinline__ void computePhaseNormalIndicator(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &normx,
    real_t &normy,
    real_t &normz,
    real_t &indicator) noexcept
{
    real_t dphix;
    real_t dphiy;
    real_t dphiz;
    computePhaseNormalIndicator(
        dbuffer, x, y, z, normx, normy, normz, indicator, dphix, dphiy, dphiz);
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateCurvatureDirection(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &curvatureAcc) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const natural_t sx = src % NX;
        const natural_t sy = (src / NX) % NY;
        const natural_t sz = src / STRIDE;

        real_t normx;
        real_t normy;
        real_t normz;
        real_t indicator;
        computePhaseNormalIndicator(dbuffer, sx, sy, sz, normx, normy, normz, indicator);

        curvatureAcc += VelocitySet::w<Q>() *
                        (static_cast<real_t>(cx) * normx +
                         static_cast<real_t>(cy) * normy +
                         static_cast<real_t>(cz) * normz);
    }
}

__device__ static __forceinline__ real_t computePhaseCurvature(
    const real_t *__restrict__ dbuffer,
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    real_t curvatureAcc = static_cast<real_t>(0);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateCurvatureDirection<Q>(dbuffer, x, y, z, curvatureAcc);
        });

    return VelocitySet::as2() * curvatureAcc;
}

__device__ static __forceinline__ void computeMomentNormal(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &normalX,
    real_t &normalY,
    real_t &normalZ) noexcept
{
    real_t gradx = static_cast<real_t>(0);
    real_t grady = static_cast<real_t>(0);
    real_t gradz = static_cast<real_t>(0);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateGradientDirection<Q>(moments, x, y, z, gradx, grady, gradz);
        });

    gradx *= VelocitySet::as2();
    grady *= VelocitySet::as2();
    gradz *= VelocitySet::as2();

    const real_t gradNorm =
        math::sqrt(gradx * gradx + grady * grady + gradz * gradz) +
        static_cast<real_t>(1.0e-9);

    const real_t invGradNorm = static_cast<real_t>(1) / gradNorm;

    normalX = gradx * invGradNorm;
    normalY = grady * invGradNorm;
    normalZ = gradz * invGradNorm;
}

__global__ void stream(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer,
    const natural_t step)
{
    __shared__ real_t sharedPhi[STREAM_PHI_TILE_SIZE];
    __shared__ real_t sharedNormX[STREAM_NORMAL_TILE_SIZE];
    __shared__ real_t sharedNormY[STREAM_NORMAL_TILE_SIZE];
    __shared__ real_t sharedNormZ[STREAM_NORMAL_TILE_SIZE];

    const int baseX = static_cast<int>(blockIdx.x * BLOCK_NX);
    const int baseY = static_cast<int>(blockIdx.y * BLOCK_NY);
    const int baseZ = static_cast<int>(blockIdx.z * BLOCK_NZ);

    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    const natural_t localThread =
        threadIdx.x + threadIdx.y * BLOCK_NX + threadIdx.z * BLOCK_NX * BLOCK_NY;

    for (natural_t tileIdx = localThread; tileIdx < STREAM_PHI_TILE_SIZE; tileIdx += STREAM_BLOCK_THREADS)
    {
        const natural_t localPhiX = tileIdx % STREAM_PHI_TILE_NX;
        const natural_t localPhiY = (tileIdx / STREAM_PHI_TILE_NX) % STREAM_PHI_TILE_NY;
        const natural_t localPhiZ = tileIdx / STREAM_PHI_TILE_STRIDE;

        const int rawX = baseX + static_cast<int>(localPhiX) - static_cast<int>(STREAM_PHI_HALO);
        const int rawY = baseY + static_cast<int>(localPhiY) - static_cast<int>(STREAM_PHI_HALO);
        const int rawZ = baseZ + static_cast<int>(localPhiZ) - static_cast<int>(STREAM_PHI_HALO);

        const natural_t src = global3(
            resolveSharedTileLoadCoordinate(rawX, static_cast<int>(NX), PERIODIC_X),
            resolveSharedTileLoadCoordinate(rawY, static_cast<int>(NY), PERIODIC_Y),
            resolveSharedTileLoadCoordinate(rawZ, static_cast<int>(NZ), PERIODIC_Z));

        sharedPhi[tileIdx] = loadMoment(moments, src, PHI);
    }

    __syncthreads();

    for (natural_t tileIdx = localThread; tileIdx < STREAM_NORMAL_TILE_SIZE; tileIdx += STREAM_BLOCK_THREADS)
    {
        const natural_t localNormalX = tileIdx % STREAM_NORMAL_TILE_NX;
        const natural_t localNormalY = (tileIdx / STREAM_NORMAL_TILE_NX) % STREAM_NORMAL_TILE_NY;
        const natural_t localNormalZ = tileIdx / STREAM_NORMAL_TILE_STRIDE;

        computeSharedNormal(
            sharedPhi,
            baseX,
            baseY,
            baseZ,
            localNormalX,
            localNormalY,
            localNormalZ,
            sharedNormX[tileIdx],
            sharedNormY[tileIdx],
            sharedNormZ[tileIdx]);
    }

    __syncthreads();

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const uint8_t nodeType = boundaryMask(x, y, z);

    real_t pstar = static_cast<real_t>(0);
    real_t ux = static_cast<real_t>(0);
    real_t uy = static_cast<real_t>(0);
    real_t uz = static_cast<real_t>(0);
    real_t mxx = static_cast<real_t>(0);
    real_t myy = static_cast<real_t>(0);
    real_t mzz = static_cast<real_t>(0);
    real_t mxy = static_cast<real_t>(0);
    real_t mxz = static_cast<real_t>(0);
    real_t myz = static_cast<real_t>(0);
    real_t phi = static_cast<real_t>(0);

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(moments, x, y, z, nodeType, step, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, phi);
    }
    else
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                accumulateHydroStreamDirection<Q>(moments, x, y, z, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
                accumulatePhaseStreamDirectionShared<Q>(
                    moments,
                    sharedNormX,
                    sharedNormY,
                    sharedNormZ,
                    baseX,
                    baseY,
                    baseZ,
                    x,
                    y,
                    z,
                    phi);
            });
    }

    dbuffer[midx(idx, PSTAR)] = pstar;
    dbuffer[midx(idx, UX)] = ux;
    dbuffer[midx(idx, UY)] = uy;
    dbuffer[midx(idx, UZ)] = uz;
    dbuffer[midx(idx, MXX)] = mxx;
    dbuffer[midx(idx, MYY)] = myy;
    dbuffer[midx(idx, MZZ)] = mzz;
    dbuffer[midx(idx, MXY)] = mxy;
    dbuffer[midx(idx, MXZ)] = mxz;
    dbuffer[midx(idx, MYZ)] = myz;
    dbuffer[midx(idx, PHI)] = phi;
}

__global__ void collide(
    real_t *__restrict__ moments,
    const real_t *__restrict__ dbuffer)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const uint8_t nodeType = boundaryMask(x, y, z);

    real_t pstar = loadMoment(dbuffer, idx, PSTAR);
    real_t ux = loadMoment(dbuffer, idx, UX);
    real_t uy = loadMoment(dbuffer, idx, UY);
    real_t uz = loadMoment(dbuffer, idx, UZ);
    real_t mxx = loadMoment(dbuffer, idx, MXX);
    real_t myy = loadMoment(dbuffer, idx, MYY);
    real_t mzz = loadMoment(dbuffer, idx, MZZ);
    real_t mxy = loadMoment(dbuffer, idx, MXY);
    real_t mxz = loadMoment(dbuffer, idx, MXZ);
    real_t myz = loadMoment(dbuffer, idx, MYZ);
    real_t phi = loadMoment(dbuffer, idx, PHI);

    if (nodeType != BULK)
    {
        moments[midx(idx, PSTAR)] = pstar;
        moments[midx(idx, UX)] = ux;
        moments[midx(idx, UY)] = uy;
        moments[midx(idx, UZ)] = uz;
        moments[midx(idx, MXX)] = mxx;
        moments[midx(idx, MYY)] = myy;
        moments[midx(idx, MZZ)] = mzz;
        moments[midx(idx, MXY)] = mxy;
        moments[midx(idx, MXZ)] = mxz;
        moments[midx(idx, MYZ)] = myz;
        moments[midx(idx, PHI)] = phi;

        return;
    }

    const real_t rho = RHO_G + (RHO_L - RHO_G) * phi;
    const real_t mu = MU_G + (MU_L - MU_G) * phi;
    const real_t nu = mu / rho;
    const real_t tau = nu * VelocitySet::as2() + static_cast<real_t>(0.5);

    const real_t invRho = static_cast<real_t>(1) / rho;
    const real_t omega = static_cast<real_t>(1) / tau;
    const real_t tOmega = static_cast<real_t>(1) - omega;
    const real_t omegaD2 = static_cast<real_t>(0.5) * omega;
    const real_t ttOmega = static_cast<real_t>(1) - static_cast<real_t>(0.5) * omega;
    const real_t ttOmegaT3 = static_cast<real_t>(3) * ttOmega;

    real_t forceX = static_cast<real_t>(0);
    real_t forceY = static_cast<real_t>(0);
    real_t forceZ = static_cast<real_t>(0);

    real_t dphix = static_cast<real_t>(0);
    real_t dphiy = static_cast<real_t>(0);
    real_t dphiz = static_cast<real_t>(0);

#if defined(SURFACE_FORCE_CSF)
    real_t normx = static_cast<real_t>(0);
    real_t normy = static_cast<real_t>(0);
    real_t normz = static_cast<real_t>(0);
    real_t indicator = static_cast<real_t>(0);

    computePhaseNormalIndicator(
        dbuffer, x, y, z, normx, normy, normz, indicator, dphix, dphiy, dphiz);

    const real_t curvature = computePhaseCurvature(dbuffer, x, y, z);
    const real_t surfaceForce = -SIGMA * curvature * indicator;

    forceX += surfaceForce * normx;
    forceY += surfaceForce * normy;
    forceZ += surfaceForce * normz;
#elif defined(SURFACE_FORCE_CPF)
    const real_t lapPhi = computePhaseGradientLaplacian(dbuffer, x, y, z, phi, dphix, dphiy, dphiz);
    const real_t muPhi =
        static_cast<real_t>(4) * BETA_CHEM *
            (phi - static_cast<real_t>(1)) * phi * (phi - static_cast<real_t>(0.5)) -
        KAPPA_CHEM * lapPhi;

    forceX += muPhi * dphix;
    forceY += muPhi * dphiy;
    forceZ += muPhi * dphiz;
#else
#error "Select SURFACE_FORCE_CSF or SURFACE_FORCE_CPF"
#endif

    const real_t drhoDphi = RHO_L - RHO_G;
    const real_t drhox = drhoDphi * dphix;
    const real_t drhoy = drhoDphi * dphiy;
    const real_t drhoz = drhoDphi * dphiz;

    forceX += -pstar * VelocitySet::cs2() * drhox;
    forceY += -pstar * VelocitySet::cs2() * drhoy;
    forceZ += -pstar * VelocitySet::cs2() * drhoz;

    const real_t pxx = mxx - ux * ux;
    const real_t pyy = myy - uy * uy;
    const real_t pzz = mzz - uz * uz;
    const real_t pxy = mxy - ux * uy;
    const real_t pxz = mxz - ux * uz;
    const real_t pyz = myz - uy * uz;

    forceX += -ttOmega * (pxx * drhox + pxy * drhoy + pxz * drhoz);
    forceY += -ttOmega * (pxy * drhox + pyy * drhoy + pyz * drhoz);
    forceZ += -ttOmega * (pxz * drhox + pyz * drhoy + pzz * drhoz);

    applyBodyForce(rho, forceX, forceY, forceZ);

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();
    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();

    const real_t uxEq = ux + static_cast<real_t>(1.5) * invRho * forceX;
    const real_t uyEq = uy + static_cast<real_t>(1.5) * invRho * forceY;
    const real_t uzEq = uz + static_cast<real_t>(1.5) * invRho * forceZ;

    moments[midx(idx, PSTAR)] = pstar;
    moments[midx(idx, UX)] = ux + static_cast<real_t>(3) * invRho * forceX;
    moments[midx(idx, UY)] = uy + static_cast<real_t>(3) * invRho * forceY;
    moments[midx(idx, UZ)] = uz + static_cast<real_t>(3) * invRho * forceZ;
    moments[midx(idx, MXX)] = tOmega * mxx + omegaD2 * uxEq * uxEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceX * uxEq + forceX * uxEq);
    moments[midx(idx, MYY)] = tOmega * myy + omegaD2 * uyEq * uyEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceY * uyEq + forceY * uyEq);
    moments[midx(idx, MZZ)] = tOmega * mzz + omegaD2 * uzEq * uzEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceZ * uzEq + forceZ * uzEq);
    moments[midx(idx, MXY)] = tOmega * mxy + omega * uxEq * uyEq + ttOmegaT3 * invRho * (forceX * uyEq + forceY * uxEq);
    moments[midx(idx, MXZ)] = tOmega * mxz + omega * uxEq * uzEq + ttOmegaT3 * invRho * (forceX * uzEq + forceZ * uxEq);
    moments[midx(idx, MYZ)] = tOmega * myz + omega * uyEq * uzEq + ttOmegaT3 * invRho * (forceY * uzEq + forceZ * uyEq);
    moments[midx(idx, PHI)] = phi;
}
