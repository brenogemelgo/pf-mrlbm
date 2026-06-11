#pragma once

#include "../bitmasks.cuh"
#include "../deviceFunctions.cuh"

// ===================================================================================================================== //

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_NODE_TYPES = 64;
constexpr natural_t IRBC_TABLE_SIZE = IRBC_TABLE_NODE_TYPES * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

// ===================================================================================================================== //

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & XMIN_FACE) == XMIN_FACE) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeTypeValue & XMAX_FACE) == XMAX_FACE) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeTypeValue & YMIN_FACE) == YMIN_FACE) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeTypeValue & YMAX_FACE) == YMAX_FACE) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeTypeValue & ZMIN_FACE) == ZMIN_FACE) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeTypeValue & ZMAX_FACE) == ZMAX_FACE) && (VelocitySet::cz<dir>() < 0));
}

template <natural_t dir>
__device__ __host__ [[nodiscard]] static inline bool isMissingDirectionRuntime(
    const unsigned int nodeType) noexcept
{
    return (((nodeType & XMIN_FACE) == XMIN_FACE) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeType & XMAX_FACE) == XMAX_FACE) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeType & YMIN_FACE) == YMIN_FACE) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeType & YMAX_FACE) == YMAX_FACE) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeType & ZMIN_FACE) == ZMIN_FACE) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeType & ZMAX_FACE) == ZMAX_FACE) && (VelocitySet::cz<dir>() < 0));
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
    return nodeType != BULK && !(((nodeType & XMIN_FACE) != 0u && (nodeType & XMAX_FACE) != 0u) ||
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
