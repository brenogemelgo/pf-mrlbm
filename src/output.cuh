#pragma once

#include "deviceFunctions.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
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
    case MYZ:
        return "myz";
    case PHI:
        return "phi";
    default:
        return "unknown";
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

static inline std::string outputCaseFolderName()
{
    std::string name(ActiveCase::NAME);
    for (char &c : name)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return name;
}

static inline std::string &simulationRunId()
{
    static std::string runId("000");
    return runId;
}

static inline void setSimulationRunId(
    const std::string &runId)
{
    simulationRunId() = runId;
}

static inline std::filesystem::path getSimulationOutputDirectory()
{
    return std::filesystem::path("output") / outputCaseFolderName() / simulationRunId();
}

static inline std::filesystem::path getBinaryOutputDirectory()
{
    return getSimulationOutputDirectory() / "binaries";
}

static inline std::filesystem::path getVtiOutputDirectory()
{
    return getSimulationOutputDirectory() / "vtis";
}

static inline std::filesystem::path getPostOutputDirectory()
{
    return getSimulationOutputDirectory() / "post";
}

static inline std::filesystem::path getMetadataPath()
{
    return getSimulationOutputDirectory() / "metadata.txt";
}

static inline void createOutputDirectories()
{
    std::filesystem::create_directories(getBinaryOutputDirectory());
    std::filesystem::create_directories(getVtiOutputDirectory());
    std::filesystem::create_directories(getPostOutputDirectory());
}

static inline void writeMetadata()
{
    createOutputDirectories();

    std::ofstream out(getMetadataPath());
    if (!out)
    {
        std::cerr << "Could not open metadata output: " << getMetadataPath() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    out << std::boolalpha << std::setprecision(17);
    out << "caseName = " << ActiveCase::NAME << '\n';
    out << "case_name = " << ActiveCase::NAME << '\n';
    out << "runId = " << simulationRunId() << '\n';
    out << "NX = " << NX << '\n';
    out << "NY = " << NY << '\n';
    out << "NZ = " << NZ << '\n';
    out << "NSTEPS = " << NSTEPS << '\n';
    out << "STAMP = " << STAMP << '\n';
    out << "NUM_MOMENTS = " << NUM_MOMENTS << '\n';
    out << "velocityScaleI = " << static_cast<double>(VelocitySet::scaleI()) << '\n';
    out << "CS2 = " << static_cast<double>(CS2) << '\n';
    out << "AS2 = " << static_cast<double>(AS2) << '\n';
    out << "DX = 1" << '\n';
    out << "DY = 1" << '\n';
    out << "DZ = 1" << '\n';
    out << "DT = 1" << '\n';
    out << "FIELD_NAMES = pstar,ux,uy,uz,mxx,myy,mzz,mxy,mxz,myz,phi" << '\n';
    out << '\n';

    out << "PERIODIC_X = " << PERIODIC_X << '\n';
    out << "PERIODIC_Y = " << PERIODIC_Y << '\n';
    out << "PERIODIC_Z = " << PERIODIC_Z << '\n';
    out << '\n';

    out << "RHO_L = " << static_cast<double>(RHO_L) << '\n';
    out << "RHO_G = " << static_cast<double>(RHO_G) << '\n';
    out << "MU_L = " << static_cast<double>(MU_L) << '\n';
    out << "MU_G = " << static_cast<double>(MU_G) << '\n';
    out << "NU_L = " << static_cast<double>(NU_L) << '\n';
    out << "NU_G = " << static_cast<double>(NU_G) << '\n';
    out << "RHO_RATIO = " << static_cast<double>(RHO_RATIO) << '\n';
    out << "MU_RATIO = " << static_cast<double>(MU_RATIO) << '\n';
    out << "WIDTH = " << static_cast<double>(WIDTH) << '\n';
    out << "SIGMA = " << static_cast<double>(SIGMA) << '\n';
    out << "BETA_CHEM = " << static_cast<double>(BETA_CHEM) << '\n';
    out << "KAPPA_CHEM = " << static_cast<double>(KAPPA_CHEM) << '\n';
    out << "TAU_PHI = " << static_cast<double>(TAU_PHI) << '\n';
    out << "DIFF_INT = " << static_cast<double>(DIFF_INT) << '\n';
    out << "KAPPA_INT = " << static_cast<double>(KAPPA_INT) << '\n';
    out << "GAMMA = " << static_cast<double>(GAMMA) << '\n';
    out << "U_CHAR = " << static_cast<double>(U_CHAR) << '\n';
    out << "R_INIT = " << static_cast<double>(R_INIT) << '\n';
    out << "EXPECTED_DELTA_P = " << static_cast<double>(EXPECTED_DELTA_P) << '\n';
    out << "PHASE_FIELD = phi" << '\n';
    out << "PHI_INTERFACE = 0.5" << '\n';
    out << "PHI_LIQUID = 1" << '\n';
    out << "PHI_GAS = 0" << '\n';
    out << "LIQUID_PHASE_PHI = 1" << '\n';
    out << "GAS_PHASE_PHI = 0" << '\n';
    out << "BULK_LIQUID_PHI_MIN = 0.95" << '\n';
    out << "BULK_GAS_PHI_MAX = 0.05" << '\n';
    out << "DENSITY_INTERPOLATION = RHO_G + (RHO_L - RHO_G) * phi" << '\n';
    out << "VISCOSITY_INTERPOLATION = MU_G + (MU_L - MU_G) * phi" << '\n';
    out << "PRESSURE_FIELD = pstar" << '\n';
    out << "PRESSURE_RECONSTRUCTION = pressure = pstar * CS2 * rho" << '\n';
    out << "SURFACE_FORCE_MODEL = " << SURFACE_FORCE_MODEL << '\n';
    out << '\n';

#if defined(CASE_STATIC_DROPLET)
    out << "ENABLE_STATIC_DROPLET_DIAGNOSTICS = " << ENABLE_STATIC_DROPLET_DIAGNOSTICS << '\n';
    out << "DROPLET_PHASE_PHI = 1" << '\n';
    out << "AMBIENT_PHASE_PHI = 0" << '\n';
#elif defined(CASE_RTI)
    out << "verticalDirection = z" << '\n';
    out << "GRAVITY_X = 0" << '\n';
    out << "GRAVITY_Y = 0" << '\n';
    out << "GRAVITY_Z = " << -static_cast<double>(GRAVITY) << '\n';
    out << "GRAVITY = " << static_cast<double>(GRAVITY) << '\n';
    out << "REYNOLDS = " << static_cast<double>(REYNOLDS) << '\n';
    out << "WEBER = " << static_cast<double>(WEBER) << '\n';
    out << "ATWOOD = " << static_cast<double>(ATWOOD) << '\n';
    out << "A0 = " << static_cast<double>(A0) << '\n';
    out << "L_CHAR = " << static_cast<double>(L_CHAR) << '\n';
    out << "RTI_QUASI_2D = " << RTI_IS_QUASI_2D << '\n';
    out << "INITIAL_INTERFACE_Z = " << static_cast<double>(0.5) * static_cast<double>(NZ) << '\n';
    out << "INITIAL_PERTURBATION_AMPLITUDE = " << static_cast<double>(A0) << '\n';
    out << "PERTURBATION_WAVELENGTH_X = " << static_cast<double>(NX) << '\n';
    out << "PERTURBATION_WAVELENGTH_Y = " << (RTI_IS_QUASI_2D ? static_cast<double>(0) : static_cast<double>(NY)) << '\n';
    out << "RTI_Z_WALL_VELOCITY_BC = no_slip" << '\n';
    out << "RTI_Z_WALL_PHI_BC = neumann_copy" << '\n';
    out << "HEAVY_PHASE_PHI = 1" << '\n';
    out << "LIGHT_PHASE_PHI = 0" << '\n';
#endif
}

static inline void initializeOutputLayout()
{
    createOutputDirectories();
    writeMetadata();
}

static inline real_t outputMomentScale(
    const natural_t field)
{
    switch (field)
    {
    case UX:
    case UY:
    case UZ:
        return VelocitySet::scaleI();
    case MXX:
    case MYY:
    case MZZ:
        return VelocitySet::scaleII();
    case MXY:
    case MXZ:
    case MYZ:
        return VelocitySet::scaleIJ();
    default:
        return static_cast<real_t>(1);
    }
}

static inline void transformMomentFieldForBinary(
    std::vector<real_t> &fieldData,
    const natural_t field,
    const bool toSolverScale)
{
    const real_t scale = outputMomentScale(field);
    if (scale == static_cast<real_t>(1))
    {
        return;
    }

    for (real_t &value : fieldData)
    {
        value = toSolverScale ? value * scale : value / scale;
    }
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

        transformMomentFieldForBinary(fieldData, field, false);

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
    const std::filesystem::path dir = getBinaryOutputDirectory();
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

        transformMomentFieldForBinary(fieldData, field, true);

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
        std::cerr << "No checkpoint found in " << getBinaryOutputDirectory() << std::endl;
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
    vti << "      <PointData Scalars=\"phi\">\n";

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
    if constexpr (!ENABLE_STATIC_DROPLET_DIAGNOSTICS)
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

        outputCheckCuda(cudaMemcpy(phi.data(), deviceMoments + CELLS * PHI, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy diagnostics phi");
        outputCheckCuda(cudaMemcpy(ux.data(), deviceMoments + CELLS * UX, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy diagnostics ux");
        outputCheckCuda(cudaMemcpy(uy.data(), deviceMoments + CELLS * UY, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy diagnostics uy");
        outputCheckCuda(cudaMemcpy(uz.data(), deviceMoments + CELLS * UZ, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy diagnostics uz");
        outputCheckCuda(cudaMemcpy(pstar.data(), deviceMoments + CELLS * PSTAR, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost), "cudaMemcpy diagnostics pstar");

        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double cs2 = static_cast<double>(CS2);
        constexpr double laplaceFactor = 2.0; // sphere: Delta p = 2 sigma / R

        double volumePhi = 0.0;
        double weightedX = 0.0;
        double weightedY = 0.0;
        double weightedZ = 0.0;

        double maxU = 0.0;

        double pInside = 0.0;
        double pOutside = 0.0;

        double rhoInside = 0.0;
        double rhoOutside = 0.0;

        double muInside = 0.0;
        double muOutside = 0.0;

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

                    const real_t rho =
                        RHO_G + (RHO_L - RHO_G) * phiValue;

                    const real_t mu =
                        MU_G + (MU_L - MU_G) * phiValue;

                    const real_t pPhys =
                        pstar[idx] * static_cast<real_t>(cs2) * rho;

                    const double uxValue = static_cast<double>(ux[idx]);
                    const double uyValue = static_cast<double>(uy[idx]);
                    const double uzValue = static_cast<double>(uz[idx]);

                    const double uMag =
                        std::sqrt(uxValue * uxValue +
                                  uyValue * uyValue +
                                  uzValue * uzValue);

                    volumePhi += static_cast<double>(phiValue);

                    weightedX += static_cast<double>(phiValue) * static_cast<double>(x);
                    weightedY += static_cast<double>(phiValue) * static_cast<double>(y);
                    weightedZ += static_cast<double>(phiValue) * static_cast<double>(z);

                    maxU = std::max(maxU, uMag);

                    minPhi = std::min(minPhi, phiValue);
                    maxPhi = std::max(maxPhi, phiValue);

                    if (phiValue > static_cast<real_t>(0.999))
                    {
                        pInside += static_cast<double>(pPhys);
                        rhoInside += static_cast<double>(rho);
                        muInside += static_cast<double>(mu);
                        ++insideCount;
                    }
                    else if (phiValue < static_cast<real_t>(1.0e-7))
                    {
                        pOutside += static_cast<double>(pPhys);
                        rhoOutside += static_cast<double>(rho);
                        muOutside += static_cast<double>(mu);
                        ++outsideCount;
                    }
                }
            }
        }

        const double invVolumePhi = volumePhi > 0.0 ? 1.0 / volumePhi : 0.0;

        const double comX = weightedX * invVolumePhi;
        const double comY = weightedY * invVolumePhi;
        const double comZ = weightedZ * invVolumePhi;

        const double avgInsideP =
            insideCount > 0 ? pInside / static_cast<double>(insideCount) : 0.0;

        const double avgOutsideP =
            outsideCount > 0 ? pOutside / static_cast<double>(outsideCount) : 0.0;

        const double avgInsideRho =
            insideCount > 0 ? rhoInside / static_cast<double>(insideCount) : 0.0;

        const double avgOutsideRho =
            outsideCount > 0 ? rhoOutside / static_cast<double>(outsideCount) : 0.0;

        const double avgInsideMu =
            insideCount > 0 ? muInside / static_cast<double>(insideCount) : 0.0;

        const double avgOutsideMu =
            outsideCount > 0 ? muOutside / static_cast<double>(outsideCount) : 0.0;

        const double deltaP = avgInsideP - avgOutsideP;

        const double effectiveRadius =
            volumePhi > 0.0
                ? std::cbrt((3.0 * volumePhi) / (4.0 * pi))
                : 0.0;

        const double expectedDeltaPInitialRadius =
            static_cast<double>(R_INIT) > 0.0
                ? laplaceFactor * static_cast<double>(SIGMA) / static_cast<double>(R_INIT)
                : 0.0;

        const double expectedDeltaPEffectiveRadius =
            effectiveRadius > 0.0
                ? laplaceFactor * static_cast<double>(SIGMA) / effectiveRadius
                : 0.0;

        const double sigmaRecoveredInitialRadius =
            deltaP * static_cast<double>(R_INIT) / laplaceFactor;

        const double sigmaRecoveredEffectiveRadius =
            deltaP * effectiveRadius / laplaceFactor;

        const double sigmaRecoveryRatioInitialRadius =
            static_cast<double>(SIGMA) != 0.0
                ? sigmaRecoveredInitialRadius / static_cast<double>(SIGMA)
                : 0.0;

        const double sigmaRecoveryRatioEffectiveRadius =
            static_cast<double>(SIGMA) != 0.0
                ? sigmaRecoveredEffectiveRadius / static_cast<double>(SIGMA)
                : 0.0;

        const double rhoRatioRecovered =
            avgOutsideRho != 0.0 ? avgInsideRho / avgOutsideRho : 0.0;

        const double muRatioRecovered =
            avgOutsideMu != 0.0 ? avgInsideMu / avgOutsideMu : 0.0;

        const double rhoRatioExpected =
            static_cast<double>(RHO_L) / static_cast<double>(RHO_G);

        const double muRatioExpected =
            static_cast<double>(MU_L) / static_cast<double>(MU_G);

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
            out << "step,"
                << "volume_phi,"
                << "phi_min,"
                << "phi_max,"
                << "max_u,"
                << "com_x,"
                << "com_y,"
                << "com_z,"
                << "radius_eff,"
                << "rho_inside_avg,"
                << "rho_outside_avg,"
                << "rho_ratio_recovered,"
                << "rho_ratio_expected,"
                << "mu_inside_avg,"
                << "mu_outside_avg,"
                << "mu_ratio_recovered,"
                << "mu_ratio_expected,"
                << "p_inside_avg,"
                << "p_outside_avg,"
                << "delta_p,"
                << "expected_delta_p_r0,"
                << "expected_delta_p_reff,"
                << "sigma_recovered_r0,"
                << "sigma_recovered_reff,"
                << "sigma_recovery_ratio_r0,"
                << "sigma_recovery_ratio_reff\n";
        }

        out << step << ','
            << std::setprecision(10)
            << volumePhi << ','
            << minPhi << ','
            << maxPhi << ','
            << maxU << ','
            << comX << ','
            << comY << ','
            << comZ << ','
            << effectiveRadius << ','
            << avgInsideRho << ','
            << avgOutsideRho << ','
            << rhoRatioRecovered << ','
            << rhoRatioExpected << ','
            << avgInsideMu << ','
            << avgOutsideMu << ','
            << muRatioRecovered << ','
            << muRatioExpected << ','
            << avgInsideP << ','
            << avgOutsideP << ','
            << deltaP << ','
            << expectedDeltaPInitialRadius << ','
            << expectedDeltaPEffectiveRadius << ','
            << sigmaRecoveredInitialRadius << ','
            << sigmaRecoveredEffectiveRadius << ','
            << sigmaRecoveryRatioInitialRadius << ','
            << sigmaRecoveryRatioEffectiveRadius << '\n';
    }
}

#ifdef PHI_CONSERVATION_DIAG
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
        value = static_cast<double>(loadMoment(moments, idx, PHI));
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
        const real_t phi_src = loadMoment(moments, idx, PHI);
        const natural_t x = idx % NX;
        const natural_t y = (idx / NX) % NY;
        const natural_t z = idx / STRIDE;

        real_t normx = static_cast<real_t>(0);
        real_t normy = static_cast<real_t>(0);
        real_t normz = static_cast<real_t>(0);
        computeMomentNormal(moments, x, y, z, normx, normy, normz);

        real_t emission = static_cast<real_t>(0);

#if defined(PHI_RESIDUAL_REST)
        real_t nonRest = static_cast<real_t>(0);

        constexpr_for<1, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();

                const real_t cu = phaseVelocityCu<Q>(moments, idx);

                const real_t gi = VelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
                                  VelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) *
                                      (static_cast<real_t>(cx) * normx +
                                       static_cast<real_t>(cy) * normy +
                                       static_cast<real_t>(cz) * normz);

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

                const real_t cu = phaseVelocityCu<Q>(moments, idx);

                const real_t gi = VelocitySet::w<Q>() * phi_src * (static_cast<real_t>(1.0) + cu) +
                                  VelocitySet::w<Q>() * GAMMA * phi_src * (static_cast<real_t>(1.0) - phi_src) *
                                      (static_cast<real_t>(cx) * normx +
                                       static_cast<real_t>(cy) * normy +
                                       static_cast<real_t>(cz) * normz);

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
    PhaseDiagScratch &scratch)
{
    reduceLocalPhaseDefectKernel<<<PHASE_DIAG_BLOCKS, PHASE_DIAG_THREADS>>>(
        moments,
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

template <typename Set>
static inline void printVelocitySetDiagnosticsFor(const char *label)
{
    double sumW = 0.0;
    double sumWCx = 0.0;
    double sumWCy = 0.0;
    double sumWCz = 0.0;
    int maxAbsCx = 0;
    int maxAbsCy = 0;
    int maxAbsCz = 0;
    bool hasRest = false;

    constexpr_for<0, Set::Q()>(
        [&](const auto Q) noexcept
        {
            constexpr int cx = Set::template cx<Q>();
            constexpr int cy = Set::template cy<Q>();
            constexpr int cz = Set::template cz<Q>();
            constexpr double w = static_cast<double>(Set::template w<Q>());

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
    std::cout << "PHI_DIAG " << label
              << " q=" << Set::Q()
              << " sum_w=" << sumW
              << " sum_w_cx=" << sumWCx
              << " sum_w_cy=" << sumWCy
              << " sum_w_cz=" << sumWCz
              << " has_q0_rest=" << (hasRest ? 1 : 0)
              << " max_abs_cx=" << maxAbsCx
              << " max_abs_cy=" << maxAbsCy
              << " max_abs_cz=" << maxAbsCz
              << std::endl;
}

static inline void printVelocitySetDiagnostics()
{
    printVelocitySetDiagnosticsFor<VelocitySet>("hydro_velocity_set");
    printVelocitySetDiagnosticsFor<VelocitySet>("phase_velocity_set");
    printVelocitySetDiagnosticsFor<VelocitySet>("gradient_velocity_set");

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
              << " case=" << ActiveCase::NAME
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

        const PhaseDefectStats defect = reduceLocalPhaseDefect(moments, scratch);

        stream<<<grid, block>>>(moments, dbuffer, t);
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
#endif

static inline void writeOutput(
    const real_t *deviceMoments,
    const natural_t step)
{
    createOutputDirectories();

    const std::string base = outputStepName(step);
    const std::filesystem::path binaryPath = getBinaryOutputDirectory() / (base + ".bin");
    const std::filesystem::path vtiPath = getVtiOutputDirectory() / (base + ".vti");

    writeBinary(deviceMoments, binaryPath);
    writeVti(binaryPath, vtiPath);
    writeCaseDiagnostics(deviceMoments, step, getPostOutputDirectory());
}
