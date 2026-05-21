#pragma once

#include "constants.cuh"

__device__ [[nodiscard]] static inline uint8_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return Case::boundaryMask(x, y, z);
}
