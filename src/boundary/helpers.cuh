#pragma once

#include "../deviceFunctions.cuh"

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_SIZE = 64 * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

// ===================================================================================================================== //

__device__ [[nodiscard]] static inline uint8_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    mask_t type = BULK;

    if constexpr (!PERIODIC_X)
    {
        if (x == 0u)
        {
            type |= XMIN_FACE;
        }
        if (x == NX - 1u)
        {
            type |= XMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Y)
    {
        if (y == 0u)
        {
            type |= YMIN_FACE;
        }
        if (y == NY - 1u)
        {
            type |= YMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Z)
    {
        if (z == 0u)
        {
            type |= ZMIN_FACE;
        }
        if (z == NZ - 1u)
        {
            type |= ZMAX_FACE;
        }
    }

    return type;
}

// ===================================================================================================================== //

template <natural_t dir>
__device__ [[nodiscard]] static __forceinline__ bool isMissingDirection(
    const unsigned int nodeType) noexcept
{
    return (((nodeType & XMIN_FACE) != 0u) && VelocitySet::cx<dir>() > 0) ||
           (((nodeType & XMAX_FACE) != 0u) && VelocitySet::cx<dir>() < 0) ||
           (((nodeType & YMIN_FACE) != 0u) && VelocitySet::cy<dir>() > 0) ||
           (((nodeType & YMAX_FACE) != 0u) && VelocitySet::cy<dir>() < 0) ||
           (((nodeType & ZMIN_FACE) != 0u) && VelocitySet::cz<dir>() > 0) ||
           (((nodeType & ZMAX_FACE) != 0u) && VelocitySet::cz<dir>() < 0);
}

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & XMIN_FACE) != 0u) && VelocitySet::cx<dir>() > 0) ||
           (((nodeTypeValue & XMAX_FACE) != 0u) && VelocitySet::cx<dir>() < 0) ||
           (((nodeTypeValue & YMIN_FACE) != 0u) && VelocitySet::cy<dir>() > 0) ||
           (((nodeTypeValue & YMAX_FACE) != 0u) && VelocitySet::cy<dir>() < 0) ||
           (((nodeTypeValue & ZMIN_FACE) != 0u) && VelocitySet::cz<dir>() > 0) ||
           (((nodeTypeValue & ZMAX_FACE) != 0u) && VelocitySet::cz<dir>() < 0);
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__host__ __device__ [[nodiscard]] static inline constexpr bool hasIRBCBoundaryConst() noexcept
{
    if constexpr (nodeTypeValue == BULK)
    {
        return false;
    }
    else
    {
        return !(((nodeTypeValue & XMIN_FACE) != 0u && (nodeTypeValue & XMAX_FACE) != 0u) ||
                 ((nodeTypeValue & YMIN_FACE) != 0u && (nodeTypeValue & YMAX_FACE) != 0u) ||
                 ((nodeTypeValue & ZMIN_FACE) != 0u && (nodeTypeValue & ZMAX_FACE) != 0u));
    }
}

__device__ [[nodiscard]] static inline bool hasIRBCBoundaryRuntime(
    const unsigned int nodeType) noexcept
{
    return nodeType != BULK &&
           !(((nodeType & XMIN_FACE) != 0u && (nodeType & XMAX_FACE) != 0u) ||
             ((nodeType & YMIN_FACE) != 0u && (nodeType & YMAX_FACE) != 0u) ||
             ((nodeType & ZMIN_FACE) != 0u && (nodeType & ZMAX_FACE) != 0u));
}

template <unsigned int>
__host__ __device__ [[nodiscard]] static inline constexpr bool isCopyOutflowBoundaryConst() noexcept
{
    return false;
}

__device__ [[nodiscard]] static inline bool isCopyOutflowBoundaryRuntime(
    const unsigned int) noexcept
{
    return false;
}
