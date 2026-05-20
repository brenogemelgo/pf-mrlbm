#include "initialConditions.cuh"
#include "kernel.cuh"
#include "output.cuh"

// #define BENCHMARK

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        const cudaError_t err = (call);                                          \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (false)

int main(int argc, char **argv)
{
    bool continueFromCheckpoint = false;
    for (int arg = 1; arg < argc; ++arg)
    {
        if (std::strcmp(argv[arg], "--continue") == 0 || std::strcmp(argv[arg], "continue") == 0)
        {
            continueFromCheckpoint = true;
        }
    }

    real_t *moments = nullptr;
    real_t *dbuffer = nullptr;
    constexpr size_t bytes = static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&moments), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dbuffer), bytes));

    real_t *momentsAlloc = moments;
    real_t *dbufferAlloc = dbuffer;

    constexpr dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr dim3 grid(GRID_X, GRID_Y, GRID_Z);

    CUDA_CHECK(initIRBCBoundaryTables());

    natural_t startStep = 0;
    if (continueFromCheckpoint)
    {
        startStep = loadLatestCheckpoint(moments, dbuffer);
    }
    else
    {
        cavityInit<<<grid, block>>>(moments, dbuffer);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

#ifndef BENCHMARK
    writeOutput(moments, startStep);
#endif

    std::cout << std::endl;
    if (continueFromCheckpoint)
    {
        std::cout << "simulation continue from step " << startStep << std::endl;
    }
    else
    {
        std::cout << "simulation start" << std::endl;
    }
    const auto start = std::chrono::high_resolution_clock::now();
#ifndef BENCHMARK
    auto lastStamp = start;
    natural_t lastStampStep = startStep;
#endif

    for (natural_t t = startStep; t < NSTEPS; ++t)
    {
        stream<<<grid, block>>>(moments, dbuffer);
        collide<<<grid, block>>>(moments, dbuffer);

#ifndef BENCHMARK
        if ((t + 1) % STAMP == 0)
        {
            CUDA_CHECK(cudaDeviceSynchronize());

            const auto now = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> stampElapsed = now - lastStamp;

            const natural_t stampSteps = (t + 1) - lastStampStep;
            const double stampMlups = static_cast<double>(CELLS) * static_cast<double>(stampSteps) / stampElapsed.count() / static_cast<double>(1000000);

            std::cout << std::endl;
            std::cout << "step " << (t + 1) << " / " << NSTEPS << std::endl;
            std::cout << "MLUPS: " << stampMlups << std::endl;

            writeOutput(moments, t + 1);

            lastStamp = std::chrono::high_resolution_clock::now();
            lastStampStep = t + 1;
        }
#endif
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const natural_t completedSteps = NSTEPS > startStep ? NSTEPS - startStep : 0;
    const double mlups = static_cast<double>(CELLS) * static_cast<double>(completedSteps) / elapsed.count() / static_cast<double>(1000000);

    std::cout << std::endl;
    std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
    std::cout << "MLUPS: " << mlups << std::endl;

    CUDA_CHECK(cudaFree(momentsAlloc));
    CUDA_CHECK(cudaFree(dbufferAlloc));
    return 0;
}
