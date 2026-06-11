#pragma once

#include "constants.cuh"

constexpr mask_t BULK = 0u;
constexpr mask_t WEST = 1u << 0;
constexpr mask_t EAST = 1u << 1;
constexpr mask_t SOUTH = 1u << 2;
constexpr mask_t NORTH = 1u << 3;
constexpr mask_t BACK = 1u << 4;
constexpr mask_t FRONT = 1u << 5;

constexpr mask_t XMIN_FACE = WEST;
constexpr mask_t XMAX_FACE = EAST;
constexpr mask_t YMIN_FACE = SOUTH;
constexpr mask_t YMAX_FACE = NORTH;
constexpr mask_t ZMIN_FACE = BACK;
constexpr mask_t ZMAX_FACE = FRONT;
constexpr mask_t FACE_MASK = XMIN_FACE | XMAX_FACE | YMIN_FACE | YMAX_FACE | ZMIN_FACE | ZMAX_FACE;

constexpr mask_t NORTH_FACE = YMAX_FACE;
constexpr mask_t SOUTH_FACE = YMIN_FACE;
constexpr mask_t WEST_FACE = XMIN_FACE;
constexpr mask_t EAST_FACE = XMAX_FACE;
constexpr mask_t FRONT_FACE = ZMAX_FACE;
constexpr mask_t BACK_FACE = ZMIN_FACE;

constexpr mask_t NORTH_WEST = NORTH | WEST;
constexpr mask_t NORTH_EAST = NORTH | EAST;
constexpr mask_t NORTH_FRONT = NORTH | FRONT;
constexpr mask_t NORTH_BACK = NORTH | BACK;
constexpr mask_t SOUTH_WEST = SOUTH | WEST;
constexpr mask_t SOUTH_EAST = SOUTH | EAST;
constexpr mask_t SOUTH_FRONT = SOUTH | FRONT;
constexpr mask_t SOUTH_BACK = SOUTH | BACK;
constexpr mask_t WEST_FRONT = WEST | FRONT;
constexpr mask_t WEST_BACK = WEST | BACK;
constexpr mask_t EAST_FRONT = EAST | FRONT;
constexpr mask_t EAST_BACK = EAST | BACK;

constexpr mask_t NORTH_WEST_FRONT = NORTH | WEST | FRONT;
constexpr mask_t NORTH_WEST_BACK = NORTH | WEST | BACK;
constexpr mask_t NORTH_EAST_FRONT = NORTH | EAST | FRONT;
constexpr mask_t NORTH_EAST_BACK = NORTH | EAST | BACK;
constexpr mask_t SOUTH_WEST_FRONT = SOUTH | WEST | FRONT;
constexpr mask_t SOUTH_WEST_BACK = SOUTH | WEST | BACK;
constexpr mask_t SOUTH_EAST_FRONT = SOUTH | EAST | FRONT;
constexpr mask_t SOUTH_EAST_BACK = SOUTH | EAST | BACK;

__device__ __host__ [[nodiscard]] static __forceinline__ mask_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    mask_t mask = BULK;

    if constexpr (!PERIODIC_X)
    {
        if (x == 0u)
        {
            mask |= XMIN_FACE;
        }
        if (x == NX - 1u)
        {
            mask |= XMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Y)
    {
        if (y == 0u)
        {
            mask |= YMIN_FACE;
        }
        if (y == NY - 1u)
        {
            mask |= YMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Z)
    {
        if (z == 0u)
        {
            mask |= ZMIN_FACE;
        }
        if (z == NZ - 1u)
        {
            mask |= ZMAX_FACE;
        }
    }

    return mask;
}
