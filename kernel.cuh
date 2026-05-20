#pragma once

#include "deviceFunctions.cuh"
#include "irbcBoundary.cuh"

__global__ void stream(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer)
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

    real_t normx = static_cast<real_t>(0);
    real_t normy = static_cast<real_t>(0);
    real_t normz = static_cast<real_t>(0);

    if (nodeType == BULK)
    {
        real_t gradx = static_cast<real_t>(0);
        real_t grady = static_cast<real_t>(0);
        real_t gradz = static_cast<real_t>(0);

        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();

                const natural_t src = global3(static_cast<natural_t>(static_cast<int>(x) + cx),
                                              static_cast<natural_t>(static_cast<int>(y) + cy),
                                              static_cast<natural_t>(static_cast<int>(z) + cz));

                const real_t phi_q = moments[midx(src, PHI)];

                gradx += VelocitySet::w<Q>() * static_cast<real_t>(cx) * phi_q;
                grady += VelocitySet::w<Q>() * static_cast<real_t>(cy) * phi_q;
                gradz += VelocitySet::w<Q>() * static_cast<real_t>(cz) * phi_q;
            });

        gradx *= VelocitySet::as2();
        grady *= VelocitySet::as2();
        gradz *= VelocitySet::as2();

        const real_t gradNorm = sqrtf(gradx * gradx + grady * grady + gradz * gradz) + static_cast<real_t>(1.0e-9);
        const real_t invGradNorm = static_cast<scalar_t>(1.0) / gradNorm;

        normx = gradx * invGradNorm;
        normy = grady * invGradNorm;
        normz = gradz * invGradNorm;
    }

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(moments, x, y, z, nodeType, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
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

                const real_t fi = VelocitySet::w<Q>() * (moments[midx(src, PSTAR)] + cu + mh);

                pstar += fi;
                ux += fi * static_cast<real_t>(cx);
                uy += fi * static_cast<real_t>(cy);
                uz += fi * static_cast<real_t>(cz);
                mxx += fi * VelocitySet::hxx<Q>();
                myy += fi * VelocitySet::hyy<Q>();
                mzz += fi * VelocitySet::hzz<Q>();
                mxy += fi * VelocitySet::hxy<Q>();
                mxz += fi * VelocitySet::hxz<Q>();
                myz += fi * VelocitySet::hyz<Q>();
            });

        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
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

                const real_t gi = VelocitySet::w<Q>() * moments[midx(src, PHI)] * (static_cast<real_t>(1.0) + cu + mh) +
                                  VelocitySet::w<Q>() * sharp * (cx * normx + cy * normy + cz * normz);

                phi += gi;
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

    pstar = dbuffer[midx(idx, PSTAR)];
    ux = dbuffer[midx(idx, UX)];
    uy = dbuffer[midx(idx, UY)];
    uz = dbuffer[midx(idx, UZ)];
    mxx = dbuffer[midx(idx, MXX)];
    myy = dbuffer[midx(idx, MYY)];
    mzz = dbuffer[midx(idx, MZZ)];
    mxy = dbuffer[midx(idx, MXY)];
    mxz = dbuffer[midx(idx, MXZ)];
    myz = dbuffer[midx(idx, MYZ)];
    phi = dbuffer[midx(idx, PHI)];

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();
    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();

    real_t forceX = static_cast<real_t>(0);
    real_t forceY = static_cast<real_t>(0);
    real_t forceZ = static_cast<real_t>(0);

    if (nodeType == BULK)
    {
        real_t dphix = static_cast<real_t>(0);
        real_t dphiy = static_cast<real_t>(0);
        real_t dphiz = static_cast<real_t>(0);
        real_t lapAcc = static_cast<real_t>(0);

        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();

                const natural_t src = global3(static_cast<natural_t>(static_cast<int>(x) + cx),
                                              static_cast<natural_t>(static_cast<int>(y) + cy),
                                              static_cast<natural_t>(static_cast<int>(z) + cz));

                const real_t phi_q = dbuffer[midx(src, PHI)];

                dphix += VelocitySet::w<Q>() * static_cast<real_t>(cx) * phi_q;
                dphiy += VelocitySet::w<Q>() * static_cast<real_t>(cy) * phi_q;
                dphiz += VelocitySet::w<Q>() * static_cast<real_t>(cz) * phi_q;
                lapAcc += VelocitySet::w<Q>() * (phi_q - phi);
            });

        dphix *= VelocitySet::as2();
        dphiy *= VelocitySet::as2();
        dphiz *= VelocitySet::as2();

        const real_t lapPhi = static_cast<real_t>(2) * lapAcc / CS2;
        const real_t muPhi = static_cast<real_t>(4) * BETA_CHEM * (phi - static_cast<real_t>(1)) * phi * (phi - static_cast<real_t>(0.5)) - KAPPA_CHEM * lapPhi;

        forceX = muPhi * dphix;
        forceY = muPhi * dphiy;
        forceZ = muPhi * dphiz;
    }

    const real_t rho = RHO_G + (RHO_L - RHO_G) * phi;
    const real_t tau = TAU_G + (TAU_L - TAU_G) * phi;

    const real_t invRho = static_cast<real_t>(1) / rho;
    const real_t omega = static_cast<real_t>(1.0) / tau;
    const real_t tOmega = static_cast<real_t>(1) - omega;
    const real_t omegaD2 = static_cast<real_t>(0.5) * omega;
    const real_t ttOmega = static_cast<real_t>(1) - static_cast<real_t>(0.5) * omega;
    const real_t ttOmegaT3 = static_cast<real_t>(3) * ttOmega;

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