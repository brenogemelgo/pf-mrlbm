#pragma once

#include "deviceFunctions.cuh"

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

static inline void outputCheckCuda(
    const cudaError_t err,
    const char *call)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA output error: " << call << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline const char *outputMomentName(const natural_t field)
{
    switch (field)
    {
    case PSTAR:
        return "pstar";
    case UX:
        return "ux";
    case UY:
        return "uy";
    case UZ:
        return "uz";
    case MXX:
        return "mxx";
    case MYY:
        return "myy";
    case MZZ:
        return "mzz";
    case MXY:
        return "mxy";
    case MXZ:
        return "mxz";
    case PHI:
        return "phi";
    default:
        return "myz";
    }
}

static inline const char *outputVtkRealType()
{
    if constexpr (std::is_same_v<real_t, float>)
    {
        return "Float32";
    }
    else
    {
        return "Float64";
    }
}

static inline std::string outputStepName(const natural_t step)
{
    std::ostringstream name;
    name << "step_" << std::setw(9) << std::setfill('0') << step;
    return name.str();
}

static inline std::filesystem::path &outputDirectory()
{
    static std::filesystem::path dir = std::filesystem::path("output") / Case::NAME / "default";
    return dir;
}

static inline void setOutputDirectory(const std::string &simId)
{
    outputDirectory() = std::filesystem::path("output") / Case::NAME / simId;
}

static inline void writeBinary(
    const real_t *deviceMoments,
    const std::filesystem::path &path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cerr << "Could not open binary output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real_t> fieldData(CELLS);

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        outputCheckCuda(
            cudaMemcpy(fieldData.data(), deviceMoments + CELLS * field, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy binary field");

        out.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(CELLS * sizeof(real_t)));
        if (!out)
        {
            std::cerr << "Could not write binary output: " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

static inline bool parseOutputBinaryStep(
    const std::filesystem::path &path,
    natural_t &step)
{
    if (path.extension() != ".bin")
    {
        return false;
    }

    const std::string stem = path.stem().string();
    constexpr const char *prefix = "step_";
    constexpr std::size_t prefixSize = 5;

    if (stem.size() <= prefixSize || stem.compare(0, prefixSize, prefix) != 0)
    {
        return false;
    }

    natural_t value = 0;
    for (std::size_t i = prefixSize; i < stem.size(); ++i)
    {
        const char c = stem[i];
        if (c < '0' || c > '9')
        {
            return false;
        }
        value = value * static_cast<natural_t>(10) + static_cast<natural_t>(c - '0');
    }

    step = value;
    return true;
}

static inline bool findLatestOutputBinary(
    std::filesystem::path &binaryPath,
    natural_t &step)
{
    const std::filesystem::path dir = outputDirectory();
    if (!std::filesystem::exists(dir))
    {
        return false;
    }

    bool found = false;
    natural_t latestStep = 0;
    std::filesystem::path latestPath;

    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        natural_t candidateStep = 0;
        if (!parseOutputBinaryStep(entry.path(), candidateStep))
        {
            continue;
        }

        if (!found || candidateStep > latestStep)
        {
            found = true;
            latestStep = candidateStep;
            latestPath = entry.path();
        }
    }

    if (found)
    {
        step = latestStep;
        binaryPath = latestPath;
    }

    return found;
}

static inline void readBinary(
    real_t *deviceMoments,
    const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "Could not open checkpoint input: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<real_t> fieldData(CELLS);

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        in.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(CELLS * sizeof(real_t)));
        if (!in)
        {
            std::cerr << "Could not read checkpoint field " << outputMomentName(field) << ": " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }

        outputCheckCuda(
            cudaMemcpy(deviceMoments + CELLS * field, fieldData.data(), CELLS * sizeof(real_t), cudaMemcpyHostToDevice),
            "cudaMemcpy checkpoint field");
    }
}

static inline natural_t loadLatestCheckpoint(
    real_t *moments,
    real_t *dbuffer)
{
    std::filesystem::path binaryPath;
    natural_t step = 0;
    if (!findLatestOutputBinary(binaryPath, step))
    {
        std::cerr << "No checkpoint found in " << outputDirectory() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    readBinary(moments, binaryPath);
    outputCheckCuda(
        cudaMemcpy(dbuffer, moments, static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy checkpoint buffer");

    std::cout << "loaded checkpoint " << binaryPath << " at step " << step << std::endl;
    return step;
}

static inline void writeVti(
    const std::filesystem::path &binaryPath,
    const std::filesystem::path &vtiPath)
{
    std::ifstream bin(binaryPath, std::ios::binary);
    if (!bin)
    {
        std::cerr << "Could not open binary input for VTI: " << binaryPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream vti(vtiPath, std::ios::binary);
    if (!vti)
    {
        std::cerr << "Could not open VTI output: " << vtiPath << std::endl;
        std::exit(EXIT_FAILURE);
    }

    constexpr std::uint64_t fieldBytes = static_cast<std::uint64_t>(CELLS) * static_cast<std::uint64_t>(sizeof(real_t));
    std::uint64_t offset = 0;

    vti << "<?xml version=\"1.0\"?>\n";
    vti << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vti << "  <ImageData WholeExtent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    vti << "    <Piece Extent=\"0 " << (NX - 1) << " 0 " << (NY - 1) << " 0 " << (NZ - 1) << "\">\n";
    vti << "      <PointData Scalars=\"rho\">\n";

    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        vti << "        <DataArray type=\"" << outputVtkRealType() << "\" Name=\"" << outputMomentName(field)
            << "\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << offset << "\"/>\n";
        offset += sizeof(std::uint64_t) + fieldBytes;
    }

    vti << "      </PointData>\n";
    vti << "      <CellData/>\n";
    vti << "    </Piece>\n";
    vti << "  </ImageData>\n";
    vti << "  <AppendedData encoding=\"raw\">\n_";

    std::vector<real_t> fieldData(CELLS);
    for (natural_t field = 0; field < NUM_MOMENTS; ++field)
    {
        bin.read(reinterpret_cast<char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
        if (!bin)
        {
            std::cerr << "Could not read binary field for VTI: " << binaryPath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        vti.write(reinterpret_cast<const char *>(&fieldBytes), sizeof(fieldBytes));
        vti.write(reinterpret_cast<const char *>(fieldData.data()), static_cast<std::streamsize>(fieldBytes));
        if (!vti)
        {
            std::cerr << "Could not write VTI output: " << vtiPath << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    vti << "\n  </AppendedData>\n";
    vti << "</VTKFile>\n";
}

static inline void writeCaseDiagnostics(
    const real_t *deviceMoments,
    const natural_t step,
    const std::filesystem::path &dir)
{
    if constexpr (!Case::ENABLE_STATIC_DROPLET_DIAGNOSTICS)
    {
        (void)deviceMoments;
        (void)step;
        (void)dir;
        return;
    }
    else
    {
        std::vector<real_t> phi(CELLS);
        std::vector<real_t> ux(CELLS);
        std::vector<real_t> uy(CELLS);
        std::vector<real_t> uz(CELLS);
        std::vector<real_t> pstar(CELLS);

        outputCheckCuda(
            cudaMemcpy(phi.data(), deviceMoments + CELLS * PHI, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy diagnostics phi");
        outputCheckCuda(
            cudaMemcpy(ux.data(), deviceMoments + CELLS * UX, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy diagnostics ux");
        outputCheckCuda(
            cudaMemcpy(uy.data(), deviceMoments + CELLS * UY, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy diagnostics uy");
        outputCheckCuda(
            cudaMemcpy(uz.data(), deviceMoments + CELLS * UZ, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy diagnostics uz");
        outputCheckCuda(
            cudaMemcpy(pstar.data(), deviceMoments + CELLS * PSTAR, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy diagnostics pstar");

        double mass = 0.0;
        double weightedX = 0.0;
        double weightedY = 0.0;
        double weightedZ = 0.0;
        double maxU = 0.0;
        double pInside = 0.0;
        double pOutside = 0.0;
        natural_t insideCount = 0;
        natural_t outsideCount = 0;
        real_t minPhi = std::numeric_limits<real_t>::max();
        real_t maxPhi = std::numeric_limits<real_t>::lowest();

        for (natural_t z = 0; z < NZ; ++z)
        {
            for (natural_t y = 0; y < NY; ++y)
            {
                for (natural_t x = 0; x < NX; ++x)
                {
                    const natural_t idx = x + y * NX + z * STRIDE;
                    const real_t phiValue = phi[idx];
                    const real_t rho = RHO_G + (RHO_L - RHO_G) * phiValue;
                    const real_t pPhys = pstar[idx] * static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0)) * rho;
                    const double uMag = std::sqrt(static_cast<double>(ux[idx]) * static_cast<double>(ux[idx]) +
                                                  static_cast<double>(uy[idx]) * static_cast<double>(uy[idx]) +
                                                  static_cast<double>(uz[idx]) * static_cast<double>(uz[idx]));

                    mass += static_cast<double>(phiValue);
                    weightedX += static_cast<double>(phiValue) * static_cast<double>(x);
                    weightedY += static_cast<double>(phiValue) * static_cast<double>(y);
                    weightedZ += static_cast<double>(phiValue) * static_cast<double>(z);
                    maxU = std::max(maxU, uMag);
                    minPhi = std::min(minPhi, phiValue);
                    maxPhi = std::max(maxPhi, phiValue);

                    if (phiValue > static_cast<real_t>(0.9))
                    {
                        pInside += static_cast<double>(pPhys);
                        ++insideCount;
                    }
                    else if (phiValue < static_cast<real_t>(0.1))
                    {
                        pOutside += static_cast<double>(pPhys);
                        ++outsideCount;
                    }
                }
            }
        }

        const double invMass = mass > 0.0 ? 1.0 / mass : 0.0;
        const double avgInside = insideCount > 0 ? pInside / static_cast<double>(insideCount) : 0.0;
        const double avgOutside = outsideCount > 0 ? pOutside / static_cast<double>(outsideCount) : 0.0;
        const double deltaP = avgInside - avgOutside;

        const std::filesystem::path diagnosticsPath = dir / "diagnostics.csv";
        const bool writeHeader = !std::filesystem::exists(diagnosticsPath);
        std::ofstream out(diagnosticsPath, std::ios::app);
        if (!out)
        {
            std::cerr << "Could not open diagnostics output: " << diagnosticsPath << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (writeHeader)
        {
            out << "step,mass,phi_min,phi_max,max_u,com_x,com_y,com_z,p_inside_avg,p_outside_avg,delta_p,expected_delta_p\n";
        }

        out << step << ','
            << std::setprecision(10)
            << mass << ','
            << minPhi << ','
            << maxPhi << ','
            << maxU << ','
            << weightedX * invMass << ','
            << weightedY * invMass << ','
            << weightedZ * invMass << ','
            << avgInside << ','
            << avgOutside << ','
            << deltaP << ','
            << Case::EXPECTED_DELTA_P << '\n';
    }
}

static inline void writeOutput(
    const real_t *deviceMoments,
    const natural_t step)
{
    const std::filesystem::path dir = outputDirectory();
    std::filesystem::create_directories(dir);

    const std::string base = outputStepName(step);
    const std::filesystem::path binaryPath = dir / (base + ".bin");
    const std::filesystem::path vtiPath = dir / (base + ".vti");

    writeBinary(deviceMoments, binaryPath);
    writeVti(binaryPath, vtiPath);
    writeCaseDiagnostics(deviceMoments, step, dir);
}
