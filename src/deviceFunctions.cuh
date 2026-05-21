#pragma once

#include "D3Q27.cuh"

__device__ [[nodiscard]] static __forceinline__ natural_t global3(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x + y * NX + z * STRIDE;
}

__device__ __host__ [[nodiscard]] static __forceinline__ natural_t midx(
    const natural_t idx,
    const natural_t moment) noexcept
{
    return idx + CELLS * moment;
}

__device__ [[nodiscard]] static __forceinline__ natural_t caseNeighborIndex(
    const int x,
    const int y,
    const int z) noexcept
{
    return global3(static_cast<natural_t>(Case::neighborX(x)),
                   static_cast<natural_t>(Case::neighborY(y)),
                   static_cast<natural_t>(Case::neighborZ(z)));
}

template <typename T, T v>
struct IntegralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = IntegralConstant;

    __device__ __host__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ __host__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};

template <const natural_t Start, const natural_t End, typename F>
__device__ __host__ __forceinline__ constexpr void constexpr_for(F &&f) noexcept
{
    if constexpr (Start < End)
    {
        f(IntegralConstant<natural_t, Start>());
        if constexpr (Start + 1 < End)
        {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}
