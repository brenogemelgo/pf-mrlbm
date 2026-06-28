#pragma once

#include "deviceFunctions.cuh"

__device__ __host__ [[nodiscard]] static __forceinline__ real_t densityFromPhi(
    const real_t phi) noexcept
{
    return RHO_G + (RHO_L - RHO_G) * phi;
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t staticDropletPhi(
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

    return static_cast<real_t>(0.5) * (static_cast<real_t>(1) - math::tanh((r - R_INIT) / (static_cast<real_t>(0.5) * WIDTH)));
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t rtiInterfaceCenterZ() noexcept
{
    return static_cast<real_t>(0.5) * static_cast<real_t>(NZ);
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t rtiInterfaceZ(
    const natural_t x,
    const natural_t y) noexcept
{
    constexpr real_t twoPi = static_cast<real_t>(6.2831853071795864769);

    const real_t kx = twoPi * static_cast<real_t>(x) / static_cast<real_t>(NX);
    const real_t ky = twoPi * static_cast<real_t>(y) / static_cast<real_t>(NY);
    const real_t yMode = RTI_IS_QUASI_2D ? static_cast<real_t>(1) : math::cos(ky);

    return rtiInterfaceCenterZ() + A0 * math::cos(kx) * yMode;
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t rtiInterfacePhi(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return static_cast<real_t>(0.5) * (static_cast<real_t>(1) + math::tanh((static_cast<real_t>(z) - rtiInterfaceZ(x, y)) / (static_cast<real_t>(0.5) * WIDTH)));
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t rtiFlatInterfacePhi(
    const natural_t z) noexcept
{
    return static_cast<real_t>(0.5) * (static_cast<real_t>(1) + math::tanh((static_cast<real_t>(z) - rtiInterfaceCenterZ()) / (static_cast<real_t>(0.5) * WIDTH)));
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t rtiHydrostaticPressure(
    const natural_t z) noexcept
{
    real_t p = static_cast<real_t>(0);

    for (natural_t zz = z; zz + static_cast<natural_t>(1) < NZ; ++zz)
    {
        const real_t phiZ = rtiFlatInterfacePhi(zz);
        const real_t rhoZ = densityFromPhi(phiZ);

        p += rhoZ * GRAVITY;
    }

    return p;
}

__device__ __host__ static __forceinline__ void initialMomentFields(
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
    real_t pPhys = static_cast<real_t>(0);

    if constexpr (CASE_IS_STATIC_DROPLET)
    {
        phi = staticDropletPhi(x, y, z);
        pPhys = EXPECTED_DELTA_P * phi;
    }
    else if constexpr (CASE_IS_RTI)
    {
        phi = rtiInterfacePhi(x, y, z);
        pPhys = rtiHydrostaticPressure(z);
    }
    else
    {
        phi = static_cast<real_t>(0);
    }

    const real_t rho = densityFromPhi(phi);

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

__global__ void initializeCase(
    real_t *moments,
    real_t *dbuffer)
{
    const natural_t x = threadIdx.x + BLOCK_NX * blockIdx.x;
    const natural_t y = threadIdx.y + BLOCK_NY * blockIdx.y;
    const natural_t z = threadIdx.z + BLOCK_NZ * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);

    real_t pstar;
    real_t ux;
    real_t uy;
    real_t uz;
    real_t mxx;
    real_t myy;
    real_t mzz;
    real_t mxy;
    real_t mxz;
    real_t myz;
    real_t phi;

    initialMomentFields(x, y, z, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, phi);

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

    for (natural_t k = 0; k < NUM_MOMENTS; ++k)
    {
        dbuffer[midx(idx, k)] = moments[midx(idx, k)];
    }
}
