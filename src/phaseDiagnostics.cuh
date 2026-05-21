#pragma once

#include "kernel.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>

#ifndef PHI_DIAG_STEPS
#define PHI_DIAG_STEPS 1000
#endif

#ifndef PHI_DIAG_EVERY
#define PHI_DIAG_EVERY 100
#endif

constexpr natural_t PHASE_DIAG_THREADS = 256;
constexpr natural_t PHASE_DIAG_BLOCKS = (CELLS + PHASE_DIAG_THREADS - 1) / PHASE_DIAG_THREADS;

struct PhaseDefectStats
{
    double sumDefect = 0.0;
    double maxAbsDefect = 0.0;
    double l1Defect = 0.0;
    double l2Defect = 0.0;
};

struct PhaseDiagScratch
{
    double *partialSum = nullptr;
    double *partialMax = nullptr;
    double *partialL1 = nullptr;
    double *partialL2 = nullptr;

    std::vector<double> hostSum;
    std::vector<double> hostMax;
    std::vector<double> hostL1;
    std::vector<double> hostL2;
};

static inline void phaseDiagCheckCuda(const cudaError_t err, const char *call)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA phase diagnostic error: " << call << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void initPhaseDiagScratch(PhaseDiagScratch &scratch)
{
    scratch.hostSum.resize(PHASE_DIAG_BLOCKS);
    scratch.hostMax.resize(PHASE_DIAG_BLOCKS);
    scratch.hostL1.resize(PHASE_DIAG_BLOCKS);
    scratch.hostL2.resize(PHASE_DIAG_BLOCKS);

    phaseDiagCheckCuda(cudaMalloc(reinterpret_cast<void **>(&scratch.partialSum), PHASE_DIAG_BLOCKS * sizeof(double)), "cudaMalloc partialSum");
    phaseDiagCheckCuda(cudaMalloc(reinterpret_cast<void **>(&scratch.partialMax), PHASE_DIAG_BLOCKS * sizeof(double)), "cudaMalloc partialMax");
    phaseDiagCheckCuda(cudaMalloc(reinterpret_cast<void **>(&scratch.partialL1), PHASE_DIAG_BLOCKS * sizeof(double)), "cudaMalloc partialL1");
    phaseDiagCheckCuda(cudaMalloc(reinterpret_cast<void **>(&scratch.partialL2), PHASE_DIAG_BLOCKS * sizeof(double)), "cudaMalloc partialL2");
}

static inline void freePhaseDiagScratch(PhaseDiagScratch &scratch)
{
    if (scratch.partialSum != nullptr)
    {
        phaseDiagCheckCuda(cudaFree(scratch.partialSum), "cudaFree partialSum");
    }
    if (scratch.partialMax != nullptr)
    {
        phaseDiagCheckCuda(cudaFree(scratch.partialMax), "cudaFree partialMax");
    }
    if (scratch.partialL1 != nullptr)
    {
        phaseDiagCheckCuda(cudaFree(scratch.partialL1), "cudaFree partialL1");
    }
    if (scratch.partialL2 != nullptr)
    {
        phaseDiagCheckCuda(cudaFree(scratch.partialL2), "cudaFree partialL2");
    }
}

__global__ void reducePhiKernel(
    const real_t *__restrict__ moments,
    double *__restrict__ partialSum)
{
    __shared__ double shared[PHASE_DIAG_THREADS];

    const natural_t tid = threadIdx.x;
    const natural_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;
    if (idx < CELLS)
    {
        value = static_cast<double>(moments[midx(idx, PHI)]);
    }

    shared[tid] = value;
    __syncthreads();

    for (natural_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialSum[blockIdx.x] = shared[0];
    }
}

__global__ void reduceLocalPhaseDefectKernel(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ normx,
    const real_t *__restrict__ normy,
    const real_t *__restrict__ normz,
    double *__restrict__ partialSum,
    double *__restrict__ partialMax,
    double *__restrict__ partialL1,
    double *__restrict__ partialL2)
{
    __shared__ double sharedSum[PHASE_DIAG_THREADS];
    __shared__ double sharedMax[PHASE_DIAG_THREADS];
    __shared__ double sharedL1[PHASE_DIAG_THREADS];
    __shared__ double sharedL2[PHASE_DIAG_THREADS];

    const natural_t tid = threadIdx.x;
    const natural_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sumValue = 0.0;
    double maxValue = 0.0;
    double l1Value = 0.0;
    double l2Value = 0.0;

    if (idx < CELLS)
    {
        const real_t phi_src = moments[midx(idx, PHI)];

        real_t emission = static_cast<real_t>(0);

#if defined(PHI_RESIDUAL_REST)
        real_t nonRest = static_cast<real_t>(0);

        constexpr_for<1, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();

                const real_t cu = static_cast<real_t>(cx) * moments[midx(idx, UX)] +
                                  static_cast<real_t>(cy) * moments[midx(idx, UY)] +
                                  static_cast<real_t>(cz) * moments[midx(idx, UZ)];

                const real_t gi = VelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
                                  VelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) *
                                      (static_cast<real_t>(cx) * normx[idx] +
                                       static_cast<real_t>(cy) * normy[idx] +
                                       static_cast<real_t>(cz) * normz[idx]);

                nonRest += gi;
            });

        const real_t giRest = phi_src - nonRest;
        emission = nonRest + giRest;
#else
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();

                const real_t cu = static_cast<real_t>(cx) * moments[midx(idx, UX)] +
                                  static_cast<real_t>(cy) * moments[midx(idx, UY)] +
                                  static_cast<real_t>(cz) * moments[midx(idx, UZ)];

                const real_t gi = VelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
                                  VelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) *
                                      (static_cast<real_t>(cx) * normx[idx] +
                                       static_cast<real_t>(cy) * normy[idx] +
                                       static_cast<real_t>(cz) * normz[idx]);

                emission += gi;
            });
#endif

        const double defect = static_cast<double>(emission) - static_cast<double>(phi_src);
        const double absDefect = defect < 0.0 ? -defect : defect;

        sumValue = defect;
        maxValue = absDefect;
        l1Value = absDefect;
        l2Value = defect * defect;
    }

    sharedSum[tid] = sumValue;
    sharedMax[tid] = maxValue;
    sharedL1[tid] = l1Value;
    sharedL2[tid] = l2Value;
    __syncthreads();

    for (natural_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sharedSum[tid] += sharedSum[tid + stride];
            sharedMax[tid] = sharedMax[tid] > sharedMax[tid + stride] ? sharedMax[tid] : sharedMax[tid + stride];
            sharedL1[tid] += sharedL1[tid + stride];
            sharedL2[tid] += sharedL2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialSum[blockIdx.x] = sharedSum[0];
        partialMax[blockIdx.x] = sharedMax[0];
        partialL1[blockIdx.x] = sharedL1[0];
        partialL2[blockIdx.x] = sharedL2[0];
    }
}

static inline double reducePhiSum(
    const real_t *moments,
    PhaseDiagScratch &scratch)
{
    reducePhiKernel<<<PHASE_DIAG_BLOCKS, PHASE_DIAG_THREADS>>>(moments, scratch.partialSum);
    phaseDiagCheckCuda(cudaGetLastError(), "reducePhiKernel launch");
    phaseDiagCheckCuda(cudaMemcpy(scratch.hostSum.data(), scratch.partialSum, PHASE_DIAG_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy phi partialSum");

    double sum = 0.0;
    for (const double value : scratch.hostSum)
    {
        sum += value;
    }
    return sum;
}

static inline PhaseDefectStats reduceLocalPhaseDefect(
    const real_t *moments,
    const real_t *normx,
    const real_t *normy,
    const real_t *normz,
    PhaseDiagScratch &scratch)
{
    reduceLocalPhaseDefectKernel<<<PHASE_DIAG_BLOCKS, PHASE_DIAG_THREADS>>>(
        moments,
        normx,
        normy,
        normz,
        scratch.partialSum,
        scratch.partialMax,
        scratch.partialL1,
        scratch.partialL2);

    phaseDiagCheckCuda(cudaGetLastError(), "reduceLocalPhaseDefectKernel launch");
    phaseDiagCheckCuda(cudaMemcpy(scratch.hostSum.data(), scratch.partialSum, PHASE_DIAG_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy defect partialSum");
    phaseDiagCheckCuda(cudaMemcpy(scratch.hostMax.data(), scratch.partialMax, PHASE_DIAG_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy defect partialMax");
    phaseDiagCheckCuda(cudaMemcpy(scratch.hostL1.data(), scratch.partialL1, PHASE_DIAG_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy defect partialL1");
    phaseDiagCheckCuda(cudaMemcpy(scratch.hostL2.data(), scratch.partialL2, PHASE_DIAG_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy defect partialL2");

    PhaseDefectStats stats;
    double l2Squared = 0.0;
    for (natural_t i = 0; i < PHASE_DIAG_BLOCKS; ++i)
    {
        stats.sumDefect += scratch.hostSum[i];
        stats.maxAbsDefect = std::max(stats.maxAbsDefect, scratch.hostMax[i]);
        stats.l1Defect += scratch.hostL1[i];
        l2Squared += scratch.hostL2[i];
    }

    stats.l2Defect = std::sqrt(l2Squared);
    return stats;
}

static inline void printVelocitySetDiagnostics()
{
    double sumW = 0.0;
    double sumWCx = 0.0;
    double sumWCy = 0.0;
    double sumWCz = 0.0;
    int maxAbsCx = 0;
    int maxAbsCy = 0;
    int maxAbsCz = 0;
    bool hasRest = false;

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            constexpr int cx = VelocitySet::cx<Q>();
            constexpr int cy = VelocitySet::cy<Q>();
            constexpr int cz = VelocitySet::cz<Q>();
            constexpr double w = static_cast<double>(VelocitySet::w<Q>());

            sumW += w;
            sumWCx += w * static_cast<double>(cx);
            sumWCy += w * static_cast<double>(cy);
            sumWCz += w * static_cast<double>(cz);
            maxAbsCx = std::max(maxAbsCx, cx < 0 ? -cx : cx);
            maxAbsCy = std::max(maxAbsCy, cy < 0 ? -cy : cy);
            maxAbsCz = std::max(maxAbsCz, cz < 0 ? -cz : cz);
            hasRest = hasRest || (Q == 0 && cx == 0 && cy == 0 && cz == 0);
        });

    std::cout << std::setprecision(17);
    std::cout << "PHI_DIAG velocity_set"
              << " sum_w=" << sumW
              << " sum_w_cx=" << sumWCx
              << " sum_w_cy=" << sumWCy
              << " sum_w_cz=" << sumWCz
              << " has_q0_rest=" << (hasRest ? 1 : 0)
              << " max_abs_cx=" << maxAbsCx
              << " max_abs_cy=" << maxAbsCy
              << " max_abs_cz=" << maxAbsCz
              << std::endl;

    std::cout << "PHI_DIAG layout"
              << " midx_0_phi=" << midx(0, PHI)
              << " cells_phi=" << (CELLS * PHI)
              << " midx_1_phi=" << midx(1, PHI)
              << " cells_phi_plus_1=" << (CELLS * PHI + 1)
              << std::endl;
}

static inline void runPhaseConservationDiagnostics(
    real_t *moments,
    real_t *dbuffer,
    real_t *normx,
    real_t *normy,
    real_t *normz,
    const dim3 grid,
    const dim3 block,
    const natural_t startStep)
{
    PhaseDiagScratch scratch;
    initPhaseDiagScratch(scratch);

    printVelocitySetDiagnostics();

    const natural_t requestedSteps = static_cast<natural_t>(PHI_DIAG_STEPS);
    const natural_t diagEvery = static_cast<natural_t>(PHI_DIAG_EVERY) > 0
                                     ? static_cast<natural_t>(PHI_DIAG_EVERY)
                                     : static_cast<natural_t>(1);
    const natural_t finalStep = std::min<natural_t>(NSTEPS, startStep + requestedSteps);

    double sumBA = 0.0;
    double sumCB = 0.0;
    double sumCA = 0.0;
    double maxAbsBA = 0.0;
    double maxAbsCB = 0.0;
    double maxAbsCA = 0.0;
    natural_t sampledSteps = 0;

    std::cout << std::setprecision(17);
    std::cout << "PHI_DIAG mode"
              << " case=" << Case::NAME
              << " cells=" << CELLS
              << " real_t=" << (std::is_same_v<real_t, float> ? "float" : "double")
#ifdef PHI_RESIDUAL_REST
              << " residual_rest=1"
#else
              << " residual_rest=0"
#endif
              << " steps=" << requestedSteps
              << " print_every=" << diagEvery
              << std::endl;

    std::cout << "PHI_DIAG_HEADER step,A_before_stream,B_after_stream,C_after_collide,B_minus_A,C_minus_B,C_minus_A,sum_defect,max_abs_defect,l1_defect,l2_defect" << std::endl;

    for (natural_t t = startStep; t < finalStep; ++t)
    {
        const double a = reducePhiSum(moments, scratch);

        computeNormals<<<grid, block>>>(moments, normx, normy, normz);
        phaseDiagCheckCuda(cudaGetLastError(), "computeNormals launch");

        const PhaseDefectStats defect = reduceLocalPhaseDefect(moments, normx, normy, normz, scratch);

        stream<<<grid, block>>>(moments, normx, normy, normz, dbuffer);
        phaseDiagCheckCuda(cudaGetLastError(), "stream launch");

        const double b = reducePhiSum(dbuffer, scratch);

        collide<<<grid, block>>>(moments, dbuffer);
        phaseDiagCheckCuda(cudaGetLastError(), "collide launch");

        const double c = reducePhiSum(moments, scratch);

        const double ba = b - a;
        const double cb = c - b;
        const double ca = c - a;

        sumBA += ba;
        sumCB += cb;
        sumCA += ca;
        maxAbsBA = std::max(maxAbsBA, std::abs(ba));
        maxAbsCB = std::max(maxAbsCB, std::abs(cb));
        maxAbsCA = std::max(maxAbsCA, std::abs(ca));
        ++sampledSteps;

        const natural_t step = t + static_cast<natural_t>(1);
        if (step == startStep + static_cast<natural_t>(1) ||
            step == finalStep ||
            step % diagEvery == 0)
        {
            std::cout << "PHI_DIAG_ROW "
                      << step << ','
                      << a << ','
                      << b << ','
                      << c << ','
                      << ba << ','
                      << cb << ','
                      << ca << ','
                      << defect.sumDefect << ','
                      << defect.maxAbsDefect << ','
                      << defect.l1Defect << ','
                      << defect.l2Defect
                      << std::endl;
        }
    }

    const double invSteps = sampledSteps > 0
                                ? 1.0 / static_cast<double>(sampledSteps)
                                : 0.0;

    std::cout << "PHI_DIAG_SUMMARY"
              << " steps=" << sampledSteps
              << " avg_B_minus_A=" << sumBA * invSteps
              << " avg_C_minus_B=" << sumCB * invSteps
              << " avg_C_minus_A=" << sumCA * invSteps
              << " max_abs_B_minus_A=" << maxAbsBA
              << " max_abs_C_minus_B=" << maxAbsCB
              << " max_abs_C_minus_A=" << maxAbsCA
              << std::endl;

    freePhaseDiagScratch(scratch);
}
