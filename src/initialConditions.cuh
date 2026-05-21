#pragma once

#include "deviceFunctions.cuh"

__global__ void caseInit(
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

    Case::initialCondition(x, y, z, pstar, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, phi);

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
