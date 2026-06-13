caseName = "static_droplet"
runId = "caseOne"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 600
vectorTargetCount = 28

import argparse
import csv

import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import (
    getFloat,
    getGridShape,
    getMetadataValue,
    getPostDir,
    getRunDir,
    listAvailableFields,
    readMetadata,
    readScalarField,
    tryReadStandaloneScalarField,
    writeDatFile,
    writeReport,
)


def applyCommandLineArgs():
    global caseName
    global runId
    global selectedStep
    global outputRoot
    global showPlots
    global figureDpi
    global vectorTargetCount

    parser = argparse.ArgumentParser(description="Post-process a static droplet run")
    parser.add_argument("--caseName", default=caseName)
    parser.add_argument("--runId", default=runId)
    parser.add_argument("--selectedStep", type=int, default=selectedStep)
    parser.add_argument("--outputRoot", default=outputRoot)
    parser.add_argument("--showPlots", action="store_true", default=showPlots)
    parser.add_argument("--figureDpi", type=int, default=figureDpi)
    parser.add_argument("--vectorTargetCount", type=int, default=vectorTargetCount)
    args = parser.parse_args()

    caseName = args.caseName
    runId = args.runId
    selectedStep = args.selectedStep
    outputRoot = args.outputRoot
    showPlots = args.showPlots
    figureDpi = args.figureDpi
    vectorTargetCount = args.vectorTargetCount


def warn(warnings, message):
    if message in warnings:
        return
    warnings.append(message)
    print(f"warning: {message}")


def metadataFloat(metadata, keys, defaultValue=None):
    try:
        return getFloat(metadata, keys)
    except RuntimeError:
        return defaultValue


def metadataText(metadata, keys, defaultValue=None):
    try:
        return str(getMetadataValue(metadata, keys))
    except RuntimeError:
        return defaultValue


def relativeError(value, reference):
    if reference is None or not np.isfinite(reference) or abs(reference) <= 0.0:
        return np.nan
    return (value - reference) / reference


def recoveredSigma(deltaP, radius):
    if not np.isfinite(deltaP) or not np.isfinite(radius) or radius <= 0.0:
        return np.nan
    return 0.5 * deltaP * radius


def sphereRadiusFromVolume(volume):
    if not np.isfinite(volume) or volume <= 0.0:
        return np.nan
    return (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


def domainCenter(shape):
    nz, ny, nx = shape
    return (
        0.5 * (nx - 1),
        0.5 * (ny - 1),
        0.5 * (nz - 1),
    )


def radialDistance(shape, center):
    nz, ny, nx = shape
    centerX, centerY, centerZ = center
    x = np.arange(nx, dtype=np.float64)[None, None, :] - centerX
    y = np.arange(ny, dtype=np.float64)[None, :, None] - centerY
    z = np.arange(nz, dtype=np.float64)[:, None, None] - centerZ
    return np.sqrt(x * x + y * y + z * z)


def estimatePhiIsoRadius(phi, radius, interfacePhi, targetRadius=np.nan):
    binWidth = 0.25
    binIndex = np.floor(radius.ravel() / binWidth).astype(np.int64)
    weights = np.bincount(binIndex, weights=phi.ravel())
    counts = np.bincount(binIndex)
    valid = counts > 0
    if not np.any(valid):
        return np.nan

    radiusBins = (np.arange(counts.size, dtype=np.float64) + 0.5) * binWidth
    phiMean = np.full(counts.size, np.nan, dtype=np.float64)
    phiMean[valid] = weights[valid] / counts[valid]
    diff = phiMean - interfacePhi
    crossing = np.flatnonzero(np.isfinite(diff[:-1]) & np.isfinite(diff[1:]) & (diff[:-1] >= 0.0) & (diff[1:] <= 0.0))
    if crossing.size == 0:
        return np.nan

    if np.isfinite(targetRadius):
        idx = crossing[np.argmin(np.abs(radiusBins[crossing] - targetRadius))]
    else:
        idx = crossing[0]

    r0 = radiusBins[idx]
    r1 = radiusBins[idx + 1]
    p0 = phiMean[idx]
    p1 = phiMean[idx + 1]
    if not np.isfinite(p0) or not np.isfinite(p1) or abs(p1 - p0) <= 0.0:
        return 0.5 * (r0 + r1)
    return r0 + (interfacePhi - p0) * (r1 - r0) / (p1 - p0)


def initialStaticDropletProfile(radius, targetRadius, width):
    if not np.isfinite(targetRadius) or not np.isfinite(width) or width <= 0.0:
        return None
    return 0.5 * (1.0 - np.tanh((radius - targetRadius) / (0.5 * width)))


def readFieldAny(runDir, metadata, aliases, selectedStep=None):
    errors = []
    for alias in aliases:
        try:
            values, step = readScalarField(runDir, metadata, alias, selectedStep)
            return values, step, alias
        except RuntimeError as error:
            errors.append(str(error))
        try:
            values = tryReadStandaloneScalarField(runDir, metadata, alias)
            if values is not None:
                return values, selectedStep, alias
        except RuntimeError as error:
            errors.append(str(error))
    return None, selectedStep, None


def requireFieldAny(runDir, metadata, aliases, selectedStep=None):
    values, step, alias = readFieldAny(runDir, metadata, aliases, selectedStep)
    if values is None:
        availableText = ", ".join(listAvailableFields(runDir))
        aliasText = ", ".join(aliases)
        raise RuntimeError(
            f"Missing required field. Tried aliases: {aliasText}. Available fields: {availableText}"
        )
    return values, step, alias


def phaseMeasure(phi, phasePhi):
    if phasePhi >= 0.5:
        return phi
    return 1.0 - phi


def reconstructDensity(phi, runDir, metadata, step, warnings):
    density, _, alias = readFieldAny(runDir, metadata, ["rho", "density"], step)
    if density is not None:
        return density, f"field:{alias}"

    law = metadataText(metadata, "DENSITY_INTERPOLATION", "")
    rhoLiquid = metadataFloat(metadata, ["RHO_L", "rhoLiquid"])
    rhoGas = metadataFloat(metadata, ["RHO_G", "rhoGas"])

    if rhoLiquid is None or rhoGas is None:
        warn(
            warnings,
            "density field is absent and RHO_L/RHO_G are missing; density-ratio diagnostic skipped",
        )
        return None, None

    if "phi" not in law.lower() and law:
        warn(
            warnings,
            f"density field is absent and interpolation law is not recognized: {law!r}",
        )
        return None, None

    return rhoGas + (rhoLiquid - rhoGas) * phi, "metadata:linear_phi"


def reconstructViscosity(phi, runDir, metadata, step, warnings):
    viscosity, _, alias = readFieldAny(
        runDir,
        metadata,
        ["mu", "viscosity", "dynamicViscosity", "nu"],
        step,
    )
    if viscosity is not None:
        return viscosity, f"field:{alias}"

    law = metadataText(metadata, "VISCOSITY_INTERPOLATION", "")
    muLiquid = metadataFloat(metadata, ["MU_L", "muLiquid"])
    muGas = metadataFloat(metadata, ["MU_G", "muGas"])

    if muLiquid is None or muGas is None:
        warn(
            warnings,
            "viscosity field is absent and MU_L/MU_G are missing; viscosity-ratio diagnostic skipped",
        )
        return None, None

    if "phi" not in law.lower() and law:
        warn(
            warnings,
            f"viscosity field is absent and interpolation law is not recognized: {law!r}",
        )
        return None, None

    return muGas + (muLiquid - muGas) * phi, "metadata:linear_phi"


def validatedPressureCs2(metadata, warnings):
    cs2 = metadataFloat(metadata, ["CS2", "cs2"], 1.0 / 3.0)

    if not np.isfinite(cs2):
        raise RuntimeError("CS2 is not finite")

    if abs(cs2 - 3.0) < 1.0e-6:
        raise RuntimeError(
            "CS2 metadata is 3.0, which looks like inverse sound speed squared AS2. "
            "For D3Q27 lattice pressure reconstruction, CS2 must be 1/3."
        )

    if abs(cs2 - 1.0 / 3.0) > 1.0e-5:
        warn(warnings, f"unexpected CS2={cs2}; expected approximately 1/3")

    return cs2


def reconstructPressure(phi, density, runDir, metadata, step, warnings, cs2):
    pressure, _, alias = readFieldAny(runDir, metadata, ["p", "pressure"], step)
    if pressure is not None:
        return pressure, f"field:{alias}"

    pstar, _, alias = readFieldAny(runDir, metadata, ["pstar", "pressureStar"], step)
    if pstar is None:
        warn(
            warnings,
            "pressure field is absent; Laplace pressure jump diagnostic skipped",
        )
        return None, None

    if density is None:
        warn(
            warnings,
            "pstar is available but density could not be reconstructed; pressure diagnostic skipped",
        )
        return None, None

    return pstar * cs2 * density, f"{alias}:pstar*CS2*rho, CS2={cs2}"


def meanOrNan(values, mask):
    if values is None or not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def boolMetadata(metadata, key, defaultValue=False):
    value = metadataText(metadata, key, None)
    if value is None:
        return defaultValue
    return value.strip().lower() in {"true", "1", "yes", "on"}


def boundaryCounts(metadata):
    nx, ny, nz = getGridShape(metadata)
    periodicX = boolMetadata(metadata, "PERIODIC_X", False)
    periodicY = boolMetadata(metadata, "PERIODIC_Y", False)
    periodicZ = boolMetadata(metadata, "PERIODIC_Z", False)

    xBoundary = np.zeros(nx, dtype=bool)
    yBoundary = np.zeros(ny, dtype=bool)
    zBoundary = np.zeros(nz, dtype=bool)
    if not periodicX:
        xBoundary[0] = True
        xBoundary[-1] = True
    if not periodicY:
        yBoundary[0] = True
        yBoundary[-1] = True
    if not periodicZ:
        zBoundary[0] = True
        zBoundary[-1] = True

    nonBulk = np.count_nonzero(
        xBoundary[None, None, :]
        | yBoundary[None, :, None]
        | zBoundary[:, None, None]
    )
    total = nx * ny * nz
    return total - int(nonBulk), int(nonBulk)


def pressureMaskRow(name, insideMask, outsideMask, pstar, pressure, radiusR0, radiusReff):
    pstarInside = meanOrNan(pstar, insideMask)
    pstarOutside = meanOrNan(pstar, outsideMask)
    pressureInside = meanOrNan(pressure, insideMask)
    pressureOutside = meanOrNan(pressure, outsideMask)
    deltaP = (
        pressureInside - pressureOutside
        if np.isfinite(pressureInside) and np.isfinite(pressureOutside)
        else np.nan
    )
    return {
        "mask": name,
        "inside_cells": int(np.count_nonzero(insideMask)),
        "outside_cells": int(np.count_nonzero(outsideMask)),
        "pstar_inside_raw": pstarInside,
        "pstar_outside_raw": pstarOutside,
        "delta_pstar_raw": pstarInside - pstarOutside
        if np.isfinite(pstarInside) and np.isfinite(pstarOutside)
        else np.nan,
        "pressure_inside": pressureInside,
        "pressure_outside": pressureOutside,
        "delta_p": deltaP,
        "sigma_recovered_r0": recoveredSigma(deltaP, radiusR0),
        "sigma_recovered_reff": recoveredSigma(deltaP, radiusReff),
    }


def writeMaskDiagnostics(outDir, rows):
    path = outDir / "static_droplet_pressure_masks.csv"
    columns = [
        "mask",
        "inside_cells",
        "outside_cells",
        "pstar_inside_raw",
        "pstar_outside_raw",
        "delta_pstar_raw",
        "pressure_inside",
        "pressure_outside",
        "delta_p",
        "sigma_recovered_r0",
        "sigma_recovered_reff",
    ]
    with path.open("w", encoding="utf-8", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def savePhiSlice(outDir, phi, interfacePhi, step):
    nz, ny, nx = phi.shape
    centerY = ny // 2
    phiSlice = phi[:, centerY, :]

    plt.figure(figsize=(6.2, 5.0))
    image = plt.imshow(
        phiSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    plt.contour(
        phiSlice,
        levels=[interfacePhi],
        colors="white",
        linewidths=1.1,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
    )
    plt.colorbar(image, label="phi")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Static droplet phase field, step {step}")
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_slice_phi.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveSpuriousCurrents(outDir, phi, ux, uz, interfacePhi, step):
    nz, ny, nx = phi.shape
    centerY = ny // 2
    phiSlice = phi[:, centerY, :]
    uxSlice = ux[:, centerY, :]
    uzSlice = uz[:, centerY, :]
    speedSlice = np.sqrt(uxSlice * uxSlice + uzSlice * uzSlice)

    stride = max(1, min(nx, nz) // vectorTargetCount)
    xValues = np.arange(nx)
    zValues = np.arange(nz)
    xGrid, zGrid = np.meshgrid(xValues, zValues, indexing="xy")

    plt.figure(figsize=(6.4, 5.2))
    image = plt.imshow(
        speedSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="equal",
        cmap="magma",
    )
    plt.contour(
        phiSlice,
        levels=[interfacePhi],
        colors="cyan",
        linewidths=1.1,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
    )
    plt.quiver(
        xGrid[::stride, ::stride],
        zGrid[::stride, ::stride],
        uxSlice[::stride, ::stride],
        uzSlice[::stride, ::stride],
        color="white",
        pivot="mid",
        scale_units="xy",
        scale=3.0e-5,
        width=0.0026,
    )
    plt.colorbar(image, label="|u|")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Static droplet spurious currents, step {step}")
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_spurious_currents.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveProfile(outDir, phi, pressure, interfacePhi):
    nz, ny, nx = phi.shape
    centerY = ny // 2
    centerZ = nz // 2
    xValues = np.arange(nx, dtype=np.float64)
    xCentered = xValues - 0.5 * (nx - 1)
    phiProfile = phi[centerZ, centerY, :]

    columns = ["x", "xCentered", "phi"]
    values = [xValues, xCentered, phiProfile]
    if pressure is not None:
        columns.append("pressure")
        values.append(pressure[centerZ, centerY, :])

    writeDatFile(outDir / "static_droplet_profile.dat", columns, values)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(xCentered, phiProfile, label="phi")
    plt.axhline(
        interfacePhi, color="k", linestyle="--", linewidth=0.8, label="interface"
    )
    plt.xlabel("x - x_center")
    plt.ylabel("phi")
    if pressure is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(
            xCentered, pressure[centerZ, centerY, :], color="tab:red", label="pressure"
        )
        ax2.set_ylabel("pressure")
    plt.title("Static droplet centerline profile")
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_profile.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def writeSummary(outDir, metrics, warnings):
    csvPath = outDir / "static_droplet_summary.csv"
    with csvPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    reportLines = ["Static droplet validation summary", ""]
    for key, value in metrics.items():
        reportLines.append(f"{key} = {value}")
    if warnings:
        reportLines.extend(["", "Warnings:"])
        reportLines.extend(f"- {message}" for message in warnings)
    writeReport(outDir / "static_droplet_summary.txt", reportLines)


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    outDir = getPostDir(runDir)
    warnings = []

    print(f"run directory: {runDir}")
    print(f"metadata loaded from: {runDir / 'metadata.txt'}")

    phi, step, phiAlias = requireFieldAny(
        runDir, metadata, ["phi", "phase", "phaseField"], selectedStep
    )
    ux, _, uxAlias = requireFieldAny(runDir, metadata, ["ux", "velocityX"], step)
    uy, _, uyAlias = requireFieldAny(runDir, metadata, ["uy", "velocityY"], step)
    uz, _, uzAlias = requireFieldAny(runDir, metadata, ["uz", "velocityZ"], step)
    print(f"selected step: {step}")
    print(f"fields detected: phi={phiAlias}, ux={uxAlias}, uy={uyAlias}, uz={uzAlias}")

    dropletPhasePhi = metadataFloat(metadata, "DROPLET_PHASE_PHI", None)
    if dropletPhasePhi is None:
        dropletPhasePhi = metadataFloat(metadata, "LIQUID_PHASE_PHI", None)
    if dropletPhasePhi is None:
        warn(
            warnings,
            "DROPLET_PHASE_PHI missing; using current StaticDropletCase convention phi=1 inside",
        )
        dropletPhasePhi = 1.0

    interfacePhi = metadataFloat(metadata, "PHI_INTERFACE", 0.5)
    liquidThreshold = metadataFloat(metadata, "BULK_LIQUID_PHI_MIN", 0.999)
    gasThreshold = metadataFloat(metadata, "BULK_GAS_PHI_MAX", 0.001)
    targetRadius = metadataFloat(metadata, ["R_INIT", "RADIUS"], np.nan)
    width = metadataFloat(metadata, "WIDTH", np.nan)
    dx = metadataFloat(metadata, ["DX", "latticeSpacing"], 1.0)
    dy = metadataFloat(metadata, ["DY", "latticeSpacing"], dx)
    dz = metadataFloat(metadata, ["DZ", "latticeSpacing"], dx)
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        raise RuntimeError(
            f"DX, DY, DZ must be positive; got DX={dx}, DY={dy}, DZ={dz}"
        )
    if any(abs(value - 1.0) > 1.0e-12 for value in [dx, dy, dz]):
        warn(
            warnings,
            "DX/DY/DZ are not all 1.0; Laplace pressure and sigma diagnostics use lattice-unit radii, "
            "while physical volume/radius are reported separately.",
        )

    dropletMeasure = phaseMeasure(phi, dropletPhasePhi)
    centerLu = domainCenter(phi.shape)
    radiusLu = radialDistance(phi.shape, centerLu)
    dropletMask = dropletMeasure > liquidThreshold
    ambientMask = dropletMeasure < gasThreshold
    interfaceMask = (dropletMeasure > gasThreshold) & (dropletMeasure < liquidThreshold)

    if not np.any(dropletMask):
        raise RuntimeError(
            "droplet bulk mask is empty; check phase convention and thresholds"
        )
    if not np.any(ambientMask):
        raise RuntimeError(
            "ambient bulk mask is empty; check phase convention and thresholds"
        )
    if not np.any(interfaceMask):
        warn(
            warnings,
            "interface mask is empty; interfacial spurious-current means are NaN",
        )

    phiInitial, initialStep, _ = readFieldAny(
        runDir, metadata, ["phi", "phase", "phaseField"], 0
    )
    initialDropletVolumeLu = np.nan
    radiusPhi05Initial = np.nan
    initialProfileMaxAbsError = np.nan
    initialProfileRmsError = np.nan
    if phiInitial is not None:
        initialDropletMeasure = phaseMeasure(phiInitial, dropletPhasePhi)
        initialDropletVolumeLu = float(np.sum(initialDropletMeasure))
        radiusPhi05Initial = estimatePhiIsoRadius(
            initialDropletMeasure, radiusLu, interfacePhi, targetRadius
        )
        initialProfile = initialStaticDropletProfile(radiusLu, targetRadius, width)
        if initialProfile is not None:
            initialProfileError = phiInitial - initialProfile
            initialProfileMaxAbsError = float(np.max(np.abs(initialProfileError)))
            initialProfileRmsError = float(
                np.sqrt(np.mean(initialProfileError * initialProfileError))
            )

    density, densitySource = reconstructDensity(phi, runDir, metadata, step, warnings)
    viscosity, viscositySource = reconstructViscosity(
        phi, runDir, metadata, step, warnings
    )
    cs2ForPressure = validatedPressureCs2(metadata, warnings)

    pstarRaw, _, pstarAlias = readFieldAny(
        runDir, metadata, ["pstar", "pressureStar"], step
    )

    if pstarRaw is not None:
        pstarIn = meanOrNan(pstarRaw, dropletMask)
        pstarOut = meanOrNan(pstarRaw, ambientMask)
        deltaPstar = pstarIn - pstarOut
    else:
        pstarIn = np.nan
        pstarOut = np.nan
        deltaPstar = np.nan

    pressure, pressureSource = reconstructPressure(
        phi, density, runDir, metadata, step, warnings, cs2ForPressure
    )

    velocityMagnitude = np.sqrt(ux * ux + uy * uy + uz * uz)
    cellVolumePhysical = dx * dy * dz
    dropletVolumeLu = float(np.sum(dropletMeasure))
    dropletVolumePhysical = dropletVolumeLu * cellVolumePhysical
    radiusEffLu = sphereRadiusFromVolume(dropletVolumeLu)
    radiusEffPhysical = sphereRadiusFromVolume(dropletVolumePhysical)
    radiusPhi05 = estimatePhiIsoRadius(dropletMeasure, radiusLu, interfacePhi, targetRadius)
    radiusTargetLu = targetRadius
    radiusTargetPhysical = (
        targetRadius * (cellVolumePhysical ** (1.0 / 3.0))
        if np.isfinite(targetRadius)
        else np.nan
    )
    massChangePhi = (
        dropletVolumeLu - initialDropletVolumeLu
        if np.isfinite(initialDropletVolumeLu)
        else np.nan
    )
    massRelativeChangePhi = relativeError(dropletVolumeLu, initialDropletVolumeLu)
    boundaryBulkNodes, boundaryNonBulkNodes = boundaryCounts(metadata)

    rhoIn = meanOrNan(density, dropletMask)
    rhoOut = meanOrNan(density, ambientMask)
    rhoRatio = rhoIn / rhoOut if np.isfinite(rhoOut) and abs(rhoOut) > 0.0 else np.nan
    targetRhoRatio = metadataFloat(metadata, "RHO_RATIO", np.nan)

    muIn = meanOrNan(viscosity, dropletMask)
    muOut = meanOrNan(viscosity, ambientMask)
    muRatio = muIn / muOut if np.isfinite(muOut) and abs(muOut) > 0.0 else np.nan
    targetMuRatio = metadataFloat(metadata, "MU_RATIO", np.nan)

    pIn = meanOrNan(pressure, dropletMask)
    pOut = meanOrNan(pressure, ambientMask)
    deltaP = pIn - pOut if np.isfinite(pIn) and np.isfinite(pOut) else np.nan
    sigma = metadataFloat(metadata, "SIGMA", np.nan)
    deltaPTheoryReffLu = (
        2.0 * sigma / radiusEffLu
        if np.isfinite(sigma) and np.isfinite(radiusEffLu) and radiusEffLu > 0.0
        else np.nan
    )
    deltaPTheoryR0 = (
        2.0 * sigma / radiusTargetLu
        if np.isfinite(sigma) and np.isfinite(radiusTargetLu) and radiusTargetLu > 0.0
        else np.nan
    )
    sigmaRecoveredReffLu = recoveredSigma(deltaP, radiusEffLu)
    sigmaRecoveredR0Lu = recoveredSigma(deltaP, radiusTargetLu)

    pressureMaskRows = []
    if pressure is not None and pstarRaw is not None:
        pressureMaskRows = [
            pressureMaskRow(
                "phase_phi_gt_0.999_lt_0.001",
                dropletMeasure > 0.999,
                dropletMeasure < 0.001,
                pstarRaw,
                pressure,
                radiusTargetLu,
                radiusEffLu,
            ),
            pressureMaskRow(
                "phase_phi_gt_0.99_lt_0.01",
                dropletMeasure > 0.99,
                dropletMeasure < 0.01,
                pstarRaw,
                pressure,
                radiusTargetLu,
                radiusEffLu,
            ),
        ]
        if np.isfinite(radiusTargetLu) and np.isfinite(width):
            pressureMaskRows.extend(
                [
                    pressureMaskRow(
                        "radial_r_lt_R_minus_2W_gt_R_plus_2W",
                        radiusLu < radiusTargetLu - 2.0 * width,
                        radiusLu > radiusTargetLu + 2.0 * width,
                        pstarRaw,
                        pressure,
                        radiusTargetLu,
                        radiusEffLu,
                    ),
                    pressureMaskRow(
                        "radial_r_lt_R_minus_3W_gt_R_plus_3W",
                        radiusLu < radiusTargetLu - 3.0 * width,
                        radiusLu > radiusTargetLu + 3.0 * width,
                        pstarRaw,
                        pressure,
                        radiusTargetLu,
                        radiusEffLu,
                    ),
                ]
            )

    cs2ForChemistry = cs2ForPressure
    as2ForChemistry = metadataFloat(metadata, ["AS2", "as2"], np.nan)
    as2Source = "metadata"
    if not np.isfinite(as2ForChemistry):
        as2ForChemistry = (
            1.0 / cs2ForChemistry
            if np.isfinite(cs2ForChemistry) and abs(cs2ForChemistry) > 0.0
            else np.nan
        )
        as2Source = "inferred:1/CS2"
    betaChem = metadataFloat(metadata, "BETA_CHEM", np.nan)
    kappaChem = metadataFloat(metadata, "KAPPA_CHEM", np.nan)
    tauPhi = metadataFloat(metadata, "TAU_PHI", np.nan)
    gamma = metadataFloat(metadata, "GAMMA", np.nan)
    betaChemExpected = (
        12.0 * sigma / width
        if np.isfinite(sigma) and np.isfinite(width) and width > 0.0
        else np.nan
    )
    kappaChemExpected = (
        1.5 * sigma * width
        if np.isfinite(sigma) and np.isfinite(width)
        else np.nan
    )
    diffIntExpected = (
        cs2ForChemistry * (tauPhi - 0.5)
        if np.isfinite(cs2ForChemistry) and np.isfinite(tauPhi)
        else np.nan
    )
    kappaIntExpected = (
        4.0 * diffIntExpected / width
        if np.isfinite(diffIntExpected) and np.isfinite(width) and width > 0.0
        else np.nan
    )
    gammaExpected = (
        as2ForChemistry * kappaIntExpected
        if np.isfinite(as2ForChemistry) and np.isfinite(kappaIntExpected)
        else np.nan
    )

    maxVelocity = float(np.max(velocityMagnitude))
    maxInterfaceVelocity = (
        float(np.max(velocityMagnitude[interfaceMask]))
        if np.any(interfaceMask)
        else np.nan
    )
    meanInterfaceVelocity = (
        float(np.mean(velocityMagnitude[interfaceMask]))
        if np.any(interfaceMask)
        else np.nan
    )

    metrics = {
        "step": step,
        "grid": " x ".join(str(v) for v in getGridShape(metadata)),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "center_x_lu": centerLu[0],
        "center_y_lu": centerLu[1],
        "center_z_lu": centerLu[2],
        "boundary_bulk_nodes": boundaryBulkNodes,
        "boundary_nonbulk_nodes": boundaryNonBulkNodes,
        "density_source": densitySource or "skipped",
        "rho_inside": rhoIn,
        "rho_outside": rhoOut,
        "rho_ratio_recovered": rhoRatio,
        "rho_ratio_target": targetRhoRatio,
        "rho_ratio_relative_error": relativeError(rhoRatio, targetRhoRatio),
        "viscosity_source": viscositySource or "skipped",
        "mu_inside": muIn,
        "mu_outside": muOut,
        "mu_ratio_recovered": muRatio,
        "mu_ratio_target": targetMuRatio,
        "mu_ratio_relative_error": relativeError(muRatio, targetMuRatio),
        "droplet_volume": dropletVolumeLu,
        "droplet_volume_lu": dropletVolumeLu,
        "droplet_volume_physical": dropletVolumePhysical,
        "initial_step_for_mass": initialStep,
        "initial_droplet_volume_lu": initialDropletVolumeLu,
        "mass_change_phi": massChangePhi,
        "mass_relative_change_phi": massRelativeChangePhi,
        "radius_effective": radiusEffLu,
        "radius_effective_lu": radiusEffLu,
        "radius_effective_physical": radiusEffPhysical,
        "radius_phi05": radiusPhi05,
        "radius_phi05_lu": radiusPhi05,
        "radius_phi05_initial_lu": radiusPhi05Initial,
        "radius_target": radiusTargetLu,
        "radius_target_lu": radiusTargetLu,
        "radius_target_physical": radiusTargetPhysical,
        "radius_relative_error": relativeError(radiusEffLu, radiusTargetLu),
        "radius_relative_error_lu": relativeError(radiusEffLu, radiusTargetLu),
        "radius_phi05_relative_error_lu": relativeError(radiusPhi05, radiusTargetLu),
        "initial_profile_max_abs_error": initialProfileMaxAbsError,
        "initial_profile_rms_error": initialProfileRmsError,
        "pressure_source": pressureSource or "skipped",
        "pstar_source": pstarAlias or "skipped",
        "pstar_inside_raw": pstarIn,
        "pstar_outside_raw": pstarOut,
        "delta_pstar_raw": deltaPstar,
        "cs2_used_for_pressure": cs2ForPressure,
        "pressure_inside": pIn,
        "pressure_outside": pOut,
        "delta_p_recovered": deltaP,
        "delta_p_theory_2sigma_over_reff": deltaPTheoryReffLu,
        "delta_p_theory_2sigma_over_reff_lu": deltaPTheoryReffLu,
        "delta_p_theory_2sigma_over_r0": deltaPTheoryR0,
        "delta_p_relative_error": relativeError(deltaP, deltaPTheoryReffLu),
        "delta_p_relative_error_reff_lu": relativeError(deltaP, deltaPTheoryReffLu),
        "delta_p_relative_error_r0_lu": relativeError(deltaP, deltaPTheoryR0),
        "sigma_target": sigma,
        "sigma_recovered": sigmaRecoveredReffLu,
        "sigma_recovered_reff": sigmaRecoveredReffLu,
        "sigma_recovered_r0": sigmaRecoveredR0Lu,
        "sigma_recovered_reff_lu": sigmaRecoveredReffLu,
        "sigma_recovered_r0_lu": sigmaRecoveredR0Lu,
        "sigma_relative_error": relativeError(sigmaRecoveredReffLu, sigma),
        "sigma_relative_error_reff": relativeError(sigmaRecoveredReffLu, sigma),
        "sigma_relative_error_r0": relativeError(sigmaRecoveredR0Lu, sigma),
        "sigma_relative_error_reff_lu": relativeError(sigmaRecoveredReffLu, sigma),
        "sigma_relative_error_r0_lu": relativeError(sigmaRecoveredR0Lu, sigma),
        "cs2": cs2ForPressure,
        "as2": as2ForChemistry,
        "as2_source": as2Source,
        "beta_chem": betaChem,
        "beta_chem_expected": betaChemExpected,
        "beta_chem_relative_error": relativeError(betaChem, betaChemExpected),
        "kappa_chem": kappaChem,
        "kappa_chem_expected": kappaChemExpected,
        "kappa_chem_relative_error": relativeError(kappaChem, kappaChemExpected),
        "diff_int_expected": diffIntExpected,
        "kappa_int_expected": kappaIntExpected,
        "gamma": gamma,
        "gamma_expected": gammaExpected,
        "gamma_relative_error": relativeError(gamma, gammaExpected),
        "max_velocity": maxVelocity,
        "max_interface_velocity": maxInterfaceVelocity,
        "mean_interface_velocity": meanInterfaceVelocity,
        "droplet_bulk_cells": int(np.count_nonzero(dropletMask)),
        "ambient_bulk_cells": int(np.count_nonzero(ambientMask)),
        "interface_cells": int(np.count_nonzero(interfaceMask)),
    }

    for row in pressureMaskRows:
        prefix = "mask_" + row["mask"]
        metrics[f"{prefix}_inside_cells"] = row["inside_cells"]
        metrics[f"{prefix}_outside_cells"] = row["outside_cells"]
        metrics[f"{prefix}_delta_p"] = row["delta_p"]
        metrics[f"{prefix}_sigma_recovered_r0"] = row["sigma_recovered_r0"]
        metrics[f"{prefix}_sigma_recovered_reff"] = row["sigma_recovered_reff"]

    if boundaryNonBulkNodes != 0 and metadataText(metadata, "caseName", "") == "STATIC_DROPLET":
        warn(
            warnings,
            f"static droplet case has {boundaryNonBulkNodes} non-bulk boundary nodes",
        )

    print(f"pressure_source: {metrics['pressure_source']}")
    print(f"cs2_used_for_pressure: {metrics['cs2_used_for_pressure']}")
    print(f"delta_pstar_raw: {deltaPstar}")
    print(f"delta_p_recovered: {deltaP}")
    print(f"dx,dy,dz: {dx}, {dy}, {dz}")
    print(f"radius_target: {radiusTargetLu}")
    print(f"radius_effective_lu: {radiusEffLu}")
    print(f"radius_phi05_lu: {radiusPhi05}")
    print(f"radius_effective_physical: {radiusEffPhysical}")
    if pressureMaskRows:
        print("pressure mask sensitivity:")
        for row in pressureMaskRows:
            print(
                f"  {row['mask']}: delta_p={row['delta_p']}, "
                f"sigma_r0={row['sigma_recovered_r0']}, "
                f"sigma_reff={row['sigma_recovered_reff']}"
            )

    savePhiSlice(outDir, phi, interfacePhi, step)
    saveSpuriousCurrents(outDir, phi, ux, uz, interfacePhi, step)
    saveProfile(outDir, phi, pressure, interfacePhi)
    writeMaskDiagnostics(outDir, pressureMaskRows)
    writeSummary(outDir, metrics, warnings)

    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    applyCommandLineArgs()
    main()
