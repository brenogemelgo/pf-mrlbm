#pragma once

#include "deviceFunctions.cuh"
#include "irbcBoundary.cuh"

__global__ void stream(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer,
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz)
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

    const real_t normx_ = normx[idx];
    const real_t normy_ = normy[idx];
    const real_t normz_ = normz[idx];

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
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

                const real_t fi = VelocitySet::w<Q>() * (psrc + cu + mh);

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

        constexpr_for<0, PhaseVelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = PhaseVelocitySet::cx<Q>();
                constexpr int cy = PhaseVelocitySet::cy<Q>();
                constexpr int cz = PhaseVelocitySet::cz<Q>();

                const natural_t src = global3(static_cast<natural_t>(static_cast<int>(x) - cx),
                                              static_cast<natural_t>(static_cast<int>(y) - cy),
                                              static_cast<natural_t>(static_cast<int>(z) - cz));

                const real_t cu = static_cast<real_t>(cx) * moments[midx(src, UX)] +
                                  static_cast<real_t>(cy) * moments[midx(src, UY)] +
                                  static_cast<real_t>(cz) * moments[midx(src, UZ)];

                const real_t gi = PhaseVelocitySet::w<Q>() * moments[midx(src, PHI)] * (static_cast<real_t>(1.0) + cu) +
                                  PhaseVelocitySet::w<Q>() * sharp * (cx * normx_ + cy * normy_ + cz * normz_);

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

__global__ void forces(
    const real_t *__restrict__ dbuffer,
    real_t *__restrict__ normx,
    real_t *__restrict__ normy,
    real_t *__restrict__ normz,
    real_t *__restrict__ fsx,
    real_t *__restrict__ fsy,
    real_t *__restrict__ fsz)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
}

__global__ void collide(
    real_t *__restrict__ moments,
    const real_t *__restrict__ dbuffer,
    const real_t *__restrict__ fsx,
    const real_t *__restrict__ fsy,
    const real_t *__restrict__ fsz)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);

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

    // calculate pressure and viscosity induced forces
    real_t forceX = fsx[idx];
    real_t forceY = fsy[idx];
    real_t forceZ = fsz[idx];

    const real_t rho = MIXTURE LAW USING PHI;
    const real_t invRho = static_cast<real_t>(1) / rho;

    const real_t tau = MIXTURE LAW USING PHI;
    const real_t omega = static_cast<real_t>(1.0) / tau;
    const real_t tOmega = static_cast<real_t>(1) - omega;
    const real_t omegaD2 = static_cast<real_t>(0.5) * omega;
    const real_t ttOmega = static_cast<real_t>(1) - static_cast<real_t>(0.5) * omega;
    const real_t ttOmegaT3 = static_cast<real_t>(3) * ttOmega;

    const real_t uxEq = ux + static_cast<real_t>(1.5) * invRho * forceX;
    const real_t uyEq = uy + static_cast<real_t>(1.5) * invRho * forceY;
    const real_t uzEq = uz + static_cast<real_t>(1.5) * invRho * forceZ;

    moments[midx(idx, UX)] = ux + static_cast<real_t>(3) * invRho * forceX;
    moments[midx(idx, UY)] = uy + static_cast<real_t>(3) * invRho * forceY;
    moments[midx(idx, UZ)] = uz + static_cast<real_t>(3) * invRho * forceZ;
    moments[midx(idx, MXX)] = tOmega * mxx + omegaD2 * uxEq * uxEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceX * uxEq + forceX * uxEq);
    moments[midx(idx, MYY)] = tOmega * myy + omegaD2 * uyEq * uyEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceY * uyEq + forceY * uyEq);
    moments[midx(idx, MZZ)] = tOmega * mzz + omegaD2 * uzEq * uzEq + static_cast<real_t>(1.5) * ttOmega * invRho * (forceZ * uzEq + forceZ * uzEq);
    moments[midx(idx, MXY)] = tOmega * mxy + omega * uxEq * uyEq + ttOmegaT3 * invRho * (forceX * uyEq + forceY * uxEq);
    moments[midx(idx, MXZ)] = tOmega * mxz + omega * uxEq * uzEq + ttOmegaT3 * invRho * (forceX * uzEq + forceZ * uxEq);
    moments[midx(idx, MYZ)] = tOmega * myz + omega * uyEq * uzEq + ttOmegaT3 * invRho * (forceY * uzEq + forceZ * uyEq);
}