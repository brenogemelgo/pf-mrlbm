#pragma once

#include "D3Q27.cuh"

namespace math
{
    __device__ __host__ [[nodiscard]] static __forceinline__ real_t sqrt(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::sqrtf(x);
        }
        else
        {
            return ::sqrt(x);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t tanh(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::tanhf(x);
        }
        else
        {
            return ::tanh(x);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t cos(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::cosf(x);
        }
        else
        {
            return ::cos(x);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ real_t log(const real_t x) noexcept
    {
        if constexpr (std::is_same_v<real_t, float>)
        {
            return ::logf(x);
        }
        else
        {
            return ::log(x);
        }
    }
}

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

__device__ static __forceinline__ real_t fma_rn(real_t a, real_t b, real_t c)
{
    if constexpr (std::is_same_v<real_t, float>)
    {
        return __fmaf_rn(a, b, c);
    }
    else
    {
        return __fma_rn(a, b, c);
    }
}

__device__ [[nodiscard]] static __forceinline__ real_t loadMoment(
    const real_t *__restrict__ moments,
    const natural_t idx,
    const natural_t moment) noexcept
{
    return __ldg(moments + midx(idx, moment));
}

__device__ __host__ [[nodiscard]] static __forceinline__ int resolveNeighborCoordinate(
    const int value,
    const int extent,
    const bool periodic) noexcept
{
    if (value < 0)
    {
        return periodic ? extent - 1 : 0;
    }

    if (value >= extent)
    {
        return periodic ? 0 : extent - 1;
    }

    return value;
}

__device__ [[nodiscard]] static __forceinline__ natural_t caseNeighborIndex(
    const int x,
    const int y,
    const int z) noexcept
{
    return global3(static_cast<natural_t>(resolveNeighborCoordinate(x, static_cast<int>(NX), PERIODIC_X)),
                   static_cast<natural_t>(resolveNeighborCoordinate(y, static_cast<int>(NY), PERIODIC_Y)),
                   static_cast<natural_t>(resolveNeighborCoordinate(z, static_cast<int>(NZ), PERIODIC_Z)));
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
