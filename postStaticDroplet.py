#!/usr/bin/env python3

from pathlib import Path
import csv
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import (
    getFloat,
    getGridShape,
    getMetadataValue,
    getRunDir,
    listAvailableFields,
    readMetadata,
    readScalarField,
    tryReadStandaloneScalarField,
    writeDatFile,
    writeReport,
)


DEFAULT_RUN_DIR = getRunDir("static_droplet", "000")
OUTPUT_SUBDIR = "post_static_droplet"
SELECTED_STEP = None
FIGURE_DPI = 600
VECTOR_TARGET_COUNT = 28


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
        warn(warnings, "density field is absent and RHO_L/RHO_G are missing; density-ratio diagnostic skipped")
        return None, None

    if "phi" not in law.lower() and law:
        warn(warnings, f"density field is absent and interpolation law is not recognized: {law!r}")
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
        warn(warnings, "viscosity field is absent and MU_L/MU_G are missing; viscosity-ratio diagnostic skipped")
        return None, None

    if "phi" not in law.lower() and law:
        warn(warnings, f"viscosity field is absent and interpolation law is not recognized: {law!r}")
        return None, None

    return muGas + (muLiquid - muGas) * phi, "metadata:linear_phi"


def reconstructPressure(phi, density, runDir, metadata, step, warnings):
    pressure, _, alias = readFieldAny(runDir, metadata, ["p", "pressure"], step)
    if pressure is not None:
        return pressure, f"field:{alias}"

    pstar, _, alias = readFieldAny(runDir, metadata, ["pstar", "pressureStar"], step)
    if pstar is None:
        warn(warnings, "pressure field is absent; Laplace pressure jump diagnostic skipped")
        return None, None

    reconstruction = metadataText(metadata, "PRESSURE_RECONSTRUCTION", "")
    cs2 = metadataFloat(metadata, "CS2", 1.0 / 3.0)
    if density is None:
        warn(warnings, "pstar is available but density could not be reconstructed; pressure diagnostic skipped")
        return None, None
    if "pstar" not in reconstruction.lower() and reconstruction:
        warn(warnings, f"pressure reconstruction is not recognized: {reconstruction!r}")
        return None, None

    return pstar * cs2 * density, f"{alias}:pstar*CS2*rho"


def meanOrNan(values, mask):
    if values is None or not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


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
    plt.savefig(outDir / "static_droplet_slice_phi.png", dpi=FIGURE_DPI)
    plt.close()


def saveSpuriousCurrents(outDir, phi, ux, uz, interfacePhi, step):
    nz, ny, nx = phi.shape
    centerY = ny // 2
    phiSlice = phi[:, centerY, :]
    uxSlice = ux[:, centerY, :]
    uzSlice = uz[:, centerY, :]
    speedSlice = np.sqrt(uxSlice * uxSlice + uzSlice * uzSlice)

    stride = max(1, min(nx, nz) // VECTOR_TARGET_COUNT)
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
        scale=None,
        width=0.0026,
    )
    plt.colorbar(image, label="|u|")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Static droplet spurious currents, step {step}")
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_spurious_currents.png", dpi=FIGURE_DPI)
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
    plt.axhline(interfacePhi, color="k", linestyle="--", linewidth=0.8, label="interface")
    plt.xlabel("x - x_center")
    plt.ylabel("phi")
    if pressure is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(xCentered, pressure[centerZ, centerY, :], color="tab:red", label="pressure")
        ax2.set_ylabel("pressure")
    plt.title("Static droplet centerline profile")
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_profile.png", dpi=FIGURE_DPI)
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


def resolveRunDir():
    if len(sys.argv) > 2:
        raise SystemExit("usage: python3 postStaticDroplet.py [path/to/run]")
    if len(sys.argv) == 2:
        return Path(sys.argv[1])
    return DEFAULT_RUN_DIR


def main():
    runDir = resolveRunDir()
    metadata = readMetadata(runDir)
    outDir = runDir / OUTPUT_SUBDIR
    outDir.mkdir(parents=True, exist_ok=True)
    warnings = []

    print(f"run directory: {runDir}")
    print(f"metadata loaded from: {runDir / 'metadata.txt'}")

    phi, step, phiAlias = requireFieldAny(runDir, metadata, ["phi", "phase", "phaseField"], SELECTED_STEP)
    ux, _, uxAlias = requireFieldAny(runDir, metadata, ["ux", "velocityX"], step)
    uy, _, uyAlias = requireFieldAny(runDir, metadata, ["uy", "velocityY"], step)
    uz, _, uzAlias = requireFieldAny(runDir, metadata, ["uz", "velocityZ"], step)
    print(f"selected step: {step}")
    print(f"fields detected: phi={phiAlias}, ux={uxAlias}, uy={uyAlias}, uz={uzAlias}")

    dropletPhasePhi = metadataFloat(metadata, "DROPLET_PHASE_PHI", None)
    if dropletPhasePhi is None:
        dropletPhasePhi = metadataFloat(metadata, "LIQUID_PHASE_PHI", None)
    if dropletPhasePhi is None:
        warn(warnings, "DROPLET_PHASE_PHI missing; using current StaticDropletCase convention phi=1 inside")
        dropletPhasePhi = 1.0

    interfacePhi = metadataFloat(metadata, "PHI_INTERFACE", 0.5)
    liquidThreshold = metadataFloat(metadata, "BULK_LIQUID_PHI_MIN", 0.95)
    gasThreshold = metadataFloat(metadata, "BULK_GAS_PHI_MAX", 0.05)
    dx = metadataFloat(metadata, ["DX", "latticeSpacing"], 1.0)
    dy = metadataFloat(metadata, ["DY", "latticeSpacing"], dx)
    dz = metadataFloat(metadata, ["DZ", "latticeSpacing"], dx)

    dropletMeasure = phaseMeasure(phi, dropletPhasePhi)
    dropletMask = dropletMeasure > liquidThreshold
    ambientMask = dropletMeasure < gasThreshold
    interfaceMask = (dropletMeasure > gasThreshold) & (dropletMeasure < liquidThreshold)

    if not np.any(dropletMask):
        raise RuntimeError("droplet bulk mask is empty; check phase convention and thresholds")
    if not np.any(ambientMask):
        raise RuntimeError("ambient bulk mask is empty; check phase convention and thresholds")
    if not np.any(interfaceMask):
        warn(warnings, "interface mask is empty; interfacial spurious-current means are NaN")

    density, densitySource = reconstructDensity(phi, runDir, metadata, step, warnings)
    viscosity, viscositySource = reconstructViscosity(phi, runDir, metadata, step, warnings)
    pressure, pressureSource = reconstructPressure(phi, density, runDir, metadata, step, warnings)

    velocityMagnitude = np.sqrt(ux * ux + uy * uy + uz * uz)
    cellVolume = dx * dy * dz
    dropletVolume = float(np.sum(dropletMeasure) * cellVolume)
    radiusEff = (3.0 * dropletVolume / (4.0 * np.pi)) ** (1.0 / 3.0)
    targetRadius = metadataFloat(metadata, ["R_INIT", "RADIUS"], np.nan)

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
    deltaPTheory = 2.0 * sigma / radiusEff if np.isfinite(sigma) and radiusEff > 0.0 else np.nan

    maxVelocity = float(np.max(velocityMagnitude))
    maxInterfaceVelocity = float(np.max(velocityMagnitude[interfaceMask])) if np.any(interfaceMask) else np.nan
    meanInterfaceVelocity = float(np.mean(velocityMagnitude[interfaceMask])) if np.any(interfaceMask) else np.nan

    metrics = {
        "step": step,
        "grid": " x ".join(str(v) for v in getGridShape(metadata)),
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
        "droplet_volume": dropletVolume,
        "radius_effective": radiusEff,
        "radius_target": targetRadius,
        "radius_relative_error": relativeError(radiusEff, targetRadius),
        "pressure_source": pressureSource or "skipped",
        "pressure_inside": pIn,
        "pressure_outside": pOut,
        "delta_p_recovered": deltaP,
        "delta_p_theory_2sigma_over_reff": deltaPTheory,
        "delta_p_relative_error": relativeError(deltaP, deltaPTheory),
        "max_velocity": maxVelocity,
        "max_interface_velocity": maxInterfaceVelocity,
        "mean_interface_velocity": meanInterfaceVelocity,
        "droplet_bulk_cells": int(np.count_nonzero(dropletMask)),
        "ambient_bulk_cells": int(np.count_nonzero(ambientMask)),
        "interface_cells": int(np.count_nonzero(interfaceMask)),
    }

    savePhiSlice(outDir, phi, interfacePhi, step)
    saveSpuriousCurrents(outDir, phi, ux, uz, interfacePhi, step)
    saveProfile(outDir, phi, pressure, interfacePhi)
    writeSummary(outDir, metrics, warnings)

    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    main()
