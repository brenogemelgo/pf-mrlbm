#pragma once

#include "boundary/deviceRuntime.cuh"
#include "deviceFunctions.cuh"

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
    return velocityCu<PhaseVelocitySet, Q>(moments, idx);
}

template <typename Set, natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t normalProjection(
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz,
    const natural_t idx) noexcept
{
    constexpr int cx = Set::template cx<Q>();
    constexpr int cy = Set::template cy<Q>();
    constexpr int cz = Set::template cz<Q>();

    real_t projection = static_cast<real_t>(0);
    if constexpr (cx != 0)
    {
        projection += static_cast<real_t>(cx) * __ldg(normx + idx);
    }
    if constexpr (cy != 0)
    {
        projection += static_cast<real_t>(cy) * __ldg(normy + idx);
    }
    if constexpr (cz != 0)
    {
        projection += static_cast<real_t>(cz) * __ldg(normz + idx);
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
    constexpr int cx = GradientVelocitySet::cx<Q>();
    constexpr int cy = GradientVelocitySet::cy<Q>();
    constexpr int cz = GradientVelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const real_t weightedPhi = GradientVelocitySet::w<Q>() * loadMoment(moments, src, PHI);

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
__device__ static __forceinline__ void accumulatePhaseStreamDirection(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &phiAccum,
    real_t &phiCompensation) noexcept
{
    constexpr int cx = PhaseVelocitySet::cx<Q>();
    constexpr int cy = PhaseVelocitySet::cy<Q>();
    constexpr int cz = PhaseVelocitySet::cz<Q>();

    const natural_t src = caseNeighborIndex(static_cast<int>(x) - cx,
                                            static_cast<int>(y) - cy,
                                            static_cast<int>(z) - cz);

    const real_t phi_src = loadMoment(moments, src, PHI);
    const real_t cu = phaseVelocityCu<Q>(moments, src);
    const real_t projection = normalProjection<PhaseVelocitySet, Q>(normx, normy, normz, src);

    const real_t gi =
        PhaseVelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
        PhaseVelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) * projection;

    const real_t yPhi = gi - phiCompensation;
    const real_t tPhi = phiAccum + yPhi;
    phiCompensation = (tPhi - phiAccum) - yPhi;
    phiAccum = tPhi;
}

template <natural_t Q>
__device__ static __forceinline__ void accumulatePhaseRestResidualDirection(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz,
    const natural_t idx,
    const real_t phi_src,
    real_t &nonRest) noexcept
{
    constexpr int cx = PhaseVelocitySet::cx<Q>();
    constexpr int cy = PhaseVelocitySet::cy<Q>();
    constexpr int cz = PhaseVelocitySet::cz<Q>();

    static_assert(cx != 0 || cy != 0 || cz != 0);

    const real_t cu = phaseVelocityCu<Q>(moments, idx);
    const real_t projection = normalProjection<PhaseVelocitySet, Q>(normx, normy, normz, idx);

    nonRest +=
        PhaseVelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
        PhaseVelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) * projection;
}

template <natural_t Q>
__device__ static __forceinline__ void accumulatePhaseGradientLapDirection(
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
    constexpr int cx = GradientVelocitySet::cx<Q>();
    constexpr int cy = GradientVelocitySet::cy<Q>();
    constexpr int cz = GradientVelocitySet::cz<Q>();

    if constexpr (cx != 0 || cy != 0 || cz != 0)
    {
        const natural_t src = caseNeighborIndex(static_cast<int>(x) + cx,
                                                static_cast<int>(y) + cy,
                                                static_cast<int>(z) + cz);
        const real_t phi_q = loadMoment(dbuffer, src, PHI);
        const real_t weightedPhi = GradientVelocitySet::w<Q>() * phi_q;

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

        lapAcc += GradientVelocitySet::w<Q>() * (phi_q - phi);
    }
}

__global__ void computeNormals(
    const real_t *__restrict__ moments,
    real_t *__restrict__ normx,
    real_t *__restrict__ normy,
    real_t *__restrict__ normz)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);

    real_t gradx = static_cast<real_t>(0);
    real_t grady = static_cast<real_t>(0);
    real_t gradz = static_cast<real_t>(0);

    constexpr_for<0, GradientVelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateGradientDirection<Q>(moments, x, y, z, gradx, grady, gradz);
        });

    gradx *= GradientVelocitySet::as2();
    grady *= GradientVelocitySet::as2();
    gradz *= GradientVelocitySet::as2();

    const real_t gradNorm =
        math::sqrt(gradx * gradx + grady * grady + gradz * gradz) +
        static_cast<real_t>(1.0e-9);

    const real_t invGradNorm = static_cast<real_t>(1) / gradNorm;

    normx[idx] = gradx * invGradNorm;
    normy[idx] = grady * invGradNorm;
    normz[idx] = gradz * invGradNorm;
}

__global__ void stream(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz,
    real_t *__restrict__ dbuffer,
    const natural_t step)
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
        real_t phiAccum = static_cast<real_t>(0);
        real_t phiCompensation = static_cast<real_t>(0);

        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                accumulateHydroStreamDirection<Q>(moments, x, y, z, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
            });

        constexpr_for<0, PhaseVelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
#if defined(PHI_RESIDUAL_REST)
                if constexpr (Q == 0)
                {
                    const real_t phi_src = loadMoment(moments, idx, PHI);

                    real_t nonRest = static_cast<real_t>(0);

                    constexpr_for<1, PhaseVelocitySet::Q()>(
                        [&](const auto QR) noexcept
                        {
                            accumulatePhaseRestResidualDirection<QR>(moments, normx, normy, normz, idx, phi_src, nonRest);
                        });

                    const real_t giRest = phi_src - nonRest;
                    const real_t yPhi = giRest - phiCompensation;
                    const real_t tPhi = phiAccum + yPhi;
                    phiCompensation = (tPhi - phiAccum) - yPhi;
                    phiAccum = tPhi;
                    return;
                }
#endif

                accumulatePhaseStreamDirection<Q>(moments, normx, normy, normz, x, y, z, phiAccum, phiCompensation);
            });

        phi = static_cast<real_t>(phiAccum);
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
    real_t lapAcc = static_cast<real_t>(0);

    constexpr_for<0, GradientVelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulatePhaseGradientLapDirection<Q>(dbuffer, x, y, z, phi, dphix, dphiy, dphiz, lapAcc);
        });

    dphix *= GradientVelocitySet::as2();
    dphiy *= GradientVelocitySet::as2();
    dphiz *= GradientVelocitySet::as2();

    const real_t lapPhi = static_cast<real_t>(2) * lapAcc * GradientVelocitySet::as2();
    const real_t muPhi = static_cast<real_t>(4) * BETA_CHEM * (phi - static_cast<real_t>(1)) * phi * (phi - static_cast<real_t>(0.5)) - KAPPA_CHEM * lapPhi;

    forceX += muPhi * dphix;
    forceY += muPhi * dphiy;
    forceZ += muPhi * dphiz;

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
