caseName = "rti"
runId = "RTI_TRAJECTORY_A05_RE256"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 600
lowMixFraction = 0.05
highMixFraction = 0.95
velocityLimit = None
massTolerance = 1.0e-3
referenceCsv = []

import argparse
import csv
from pathlib import Path

import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import (
    getFloat,
    getInt,
    getMetadataValue,
    getPostDir,
    getRunDir,
    listAvailableFields,
    listAvailableSteps,
    readMetadata,
    readScalarField,
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
    global lowMixFraction
    global highMixFraction
    global velocityLimit
    global massTolerance
    global referenceCsv

    parser = argparse.ArgumentParser(
        description="Post-process Rayleigh-Taylor spike/bubble trajectories"
    )
    parser.add_argument("--caseName", default=caseName)
    parser.add_argument("--runId", default=runId)
    parser.add_argument("--selectedStep", type=int, default=selectedStep)
    parser.add_argument("--outputRoot", default=outputRoot)
    parser.add_argument("--showPlots", action="store_true", default=showPlots)
    parser.add_argument("--figureDpi", type=int, default=figureDpi)
    parser.add_argument("--lowMixFraction", type=float, default=lowMixFraction)
    parser.add_argument("--highMixFraction", type=float, default=highMixFraction)
    parser.add_argument("--velocityLimit", type=float, default=velocityLimit)
    parser.add_argument("--massTolerance", type=float, default=massTolerance)
    parser.add_argument("--referenceCsv", action="append", default=referenceCsv)
    args = parser.parse_args()

    caseName = args.caseName
    runId = args.runId
    selectedStep = args.selectedStep
    outputRoot = args.outputRoot
    showPlots = args.showPlots
    figureDpi = args.figureDpi
    lowMixFraction = args.lowMixFraction
    highMixFraction = args.highMixFraction
    velocityLimit = args.velocityLimit
    massTolerance = args.massTolerance
    referenceCsv = args.referenceCsv or []


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


def readPhi(runDir, metadata, step):
    aliases = ["phi", "phase", "phaseField"]
    for alias in aliases:
        try:
            return readScalarField(runDir, metadata, alias, step)[0], alias
        except RuntimeError:
            pass
    availableText = ", ".join(listAvailableFields(runDir))
    raise RuntimeError(
        f"Missing required phase field. Tried aliases: {', '.join(aliases)}. "
        f"Available fields: {availableText}"
    )


def readOptionalScalar(runDir, metadata, aliases, step):
    for alias in aliases:
        try:
            return readScalarField(runDir, metadata, alias, step)[0], alias
        except RuntimeError:
            pass
    return None, None


def verticalProfile(phi, verticalDirection):
    if verticalDirection == "z":
        return np.mean(phi, axis=(1, 2))
    if verticalDirection == "y":
        return np.mean(phi, axis=(0, 2))
    if verticalDirection == "x":
        return np.mean(phi, axis=(0, 1))
    raise RuntimeError(f"Unsupported verticalDirection={verticalDirection!r}")


def verticalCoordinates(phi, verticalDirection):
    if verticalDirection == "z":
        return np.arange(phi.shape[0], dtype=np.float64)
    if verticalDirection == "y":
        return np.arange(phi.shape[1], dtype=np.float64)
    if verticalDirection == "x":
        return np.arange(phi.shape[2], dtype=np.float64)
    raise RuntimeError(f"Unsupported verticalDirection={verticalDirection!r}")


def phaseMeasure(phi, phasePhi):
    if phasePhi >= 0.5:
        return phi
    return 1.0 - phi


def crossingCoordinates(coord, profile, level):
    crossings = []
    values = profile - level
    for index in range(profile.size - 1):
        left = values[index]
        right = values[index + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            crossings.append(float(coord[index]))
            continue
        if left * right < 0.0 or right == 0.0:
            denom = profile[index + 1] - profile[index]
            if not np.isfinite(denom) or abs(denom) <= 1.0e-30:
                crossings.append(float(coord[index]))
            else:
                weight = (level - profile[index]) / denom
                crossings.append(
                    float(coord[index] + weight * (coord[index + 1] - coord[index]))
                )
    if profile.size > 0 and np.isfinite(values[-1]) and values[-1] == 0.0:
        crossings.append(float(coord[-1]))

    uniqueCrossings = []
    for value in crossings:
        if not uniqueCrossings or not np.isclose(value, uniqueCrossings[-1]):
            uniqueCrossings.append(value)
    return uniqueCrossings


def crossingCoordinate(coord, profile, level):
    crossings = crossingCoordinates(coord, profile, level)
    if not crossings:
        return None
    return crossings[0]


def selectColumnCrossing(coord, profile, level, initialInterface):
    crossings = crossingCoordinates(coord, profile, level)
    if not crossings:
        return np.nan, 0
    selected = min(crossings, key=lambda value: abs(value - initialInterface))
    return selected, len(crossings)


def interfaceHeightField(heavy, verticalDirection, initialInterface, level):
    coord = verticalCoordinates(heavy, verticalDirection)
    nz, ny, nx = heavy.shape
    multipleCrossingColumns = 0

    if verticalDirection == "z":
        eta = np.full((ny, nx), np.nan, dtype=np.float64)
        for y in range(ny):
            for x in range(nx):
                eta[y, x], count = selectColumnCrossing(
                    coord, heavy[:, y, x], level, initialInterface
                )
                multipleCrossingColumns += int(count > 1)
        return eta, multipleCrossingColumns

    if verticalDirection == "y":
        eta = np.full((nz, nx), np.nan, dtype=np.float64)
        for z in range(nz):
            for x in range(nx):
                eta[z, x], count = selectColumnCrossing(
                    coord, heavy[z, :, x], level, initialInterface
                )
                multipleCrossingColumns += int(count > 1)
        return eta, multipleCrossingColumns

    if verticalDirection == "x":
        eta = np.full((nz, ny), np.nan, dtype=np.float64)
        for z in range(nz):
            for y in range(ny):
                eta[z, y], count = selectColumnCrossing(
                    coord, heavy[z, y, :], level, initialInterface
                )
                multipleCrossingColumns += int(count > 1)
        return eta, multipleCrossingColumns

    raise RuntimeError(f"Unsupported verticalDirection={verticalDirection!r}")


def interfaceStats(eta, initialInterface):
    values = eta[np.isfinite(eta)]
    if values.size == 0:
        return {
            "interface_min": np.nan,
            "interface_max": np.nan,
            "bubble_height": np.nan,
            "spike_depth": np.nan,
            "mixing_width": np.nan,
            "interface_mean": np.nan,
            "interface_std": np.nan,
            "interface_valid_columns": 0,
        }

    interfaceMin = float(np.min(values))
    interfaceMax = float(np.max(values))
    bubbleHeight = max(0.0, interfaceMax - initialInterface)
    spikeDepth = max(0.0, initialInterface - interfaceMin)
    return {
        "interface_min": interfaceMin,
        "interface_max": interfaceMax,
        "bubble_height": bubbleHeight,
        "spike_depth": spikeDepth,
        "mixing_width": bubbleHeight + spikeDepth,
        "interface_mean": float(np.mean(values)),
        "interface_std": float(np.std(values)),
        "interface_valid_columns": int(values.size),
    }


def interfaceBounds(coord, heavyProfile, lowLevel, highLevel, initialInterface, warnings):
    lower = crossingCoordinate(coord, heavyProfile, lowLevel)
    upper = crossingCoordinate(coord, heavyProfile, highLevel)

    if lower is None or upper is None:
        mixed = np.where((heavyProfile > lowLevel) & (heavyProfile < highLevel))[0]
        if mixed.size > 0:
            if lower is None:
                lower = float(coord[mixed[0]])
            if upper is None:
                upper = float(coord[mixed[-1]])
            warn(
                warnings, "used threshold fallback for one or more RTI interface bounds"
            )
        else:
            lower = float(initialInterface)
            upper = float(initialInterface)
            warn(
                warnings,
                "no mixed layer found in horizontally averaged profile; "
                "bounds set to initial interface",
            )

    if upper < lower:
        lower, upper = upper, lower

    return lower, upper


def timeScale(metadata, warnings):
    dt = metadataFloat(metadata, ["DT", "timestep", "physicalTimestep"], 1.0)
    atwood = metadataFloat(metadata, ["ATWOOD", "Atwood"], None)
    gravity = metadataFloat(metadata, ["GRAVITY", "gravity"], None)
    gravityZ = metadataFloat(metadata, "GRAVITY_Z", None)
    if gravity is None and gravityZ is not None:
        gravity = abs(gravityZ)
    elif gravity is not None:
        gravity = abs(gravity)
    length = metadataFloat(metadata, ["L_CHAR", "characteristicLength"], None)

    if atwood is None or gravity is None or length is None:
        warn(
            warnings,
            "missing ATWOOD, gravity, or L_CHAR; plotting RTI evolution against lattice step",
        )
        return dt, None
    if atwood <= 0.0 or gravity <= 0.0 or length <= 0.0:
        warn(
            warnings,
            "non-positive ATWOOD, gravity, or L_CHAR; plotting RTI evolution against lattice step",
        )
        return dt, None

    return dt, np.sqrt(atwood * gravity / length)


def inferDensityPair(metadata, warnings):
    rhoHeavy = metadataFloat(
        metadata,
        ["RHO_HEAVY", "RHO_H", "RHO_LIQUID", "RHO_L", "rho_h", "rhoHeavy"],
        None,
    )
    rhoLight = metadataFloat(
        metadata,
        ["RHO_LIGHT", "RHO_GAS", "RHO_G", "rho_l", "rhoLight"],
        None,
    )
    if rhoHeavy is None or rhoLight is None:
        warn(
            warnings,
            "RHO_HEAVY/RHO_LIGHT metadata missing; density diagnostics skipped",
        )
        return None, None
    return rhoHeavy, rhoLight


def loadReferenceCsvs(paths, warnings):
    references = []
    for rawPath in paths:
        path = Path(rawPath)
        if not path.exists():
            warn(warnings, f"reference CSV not found; skipping overlay: {path}")
            continue

        with path.open("r", encoding="utf-8", newline="") as inputFile:
            reader = csv.DictReader(inputFile)
            rows = list(reader)

        if not rows:
            warn(warnings, f"reference CSV is empty; skipping overlay: {path}")
            continue

        try:
            references.append(
                {
                    "path": path,
                    "time_dimensionless": referenceSeries(
                        rows, ["time_dimensionless", "t_star", "tstar", "time"]
                    ),
                    "bubble_height": referenceSeries(
                        rows, ["bubble_height", "bubbleHeight", "h_b", "hb"]
                    ),
                    "spike_depth": referenceSeries(
                        rows, ["spike_depth", "spikeDepth", "h_s", "hs"]
                    ),
                }
            )
        except KeyError as error:
            warn(
                warnings,
                f"reference CSV missing column {error}; skipping overlay: {path}",
            )
    return references


def referenceSeries(rows, keys):
    actualKey = None
    for key in keys:
        if key in rows[0]:
            actualKey = key
            break
    if actualKey is None:
        raise KeyError("/".join(keys))

    values = []
    for row in rows:
        try:
            values.append(float(row[actualKey]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


def saveEvolutionPlots(outDir, rows, xKey, xLabel, references):
    x = np.array([row[xKey] for row in rows], dtype=np.float64)
    bubble = np.array([row["bubble_height"] for row in rows], dtype=np.float64)
    spike = np.array([row["spike_depth"] for row in rows], dtype=np.float64)
    mixing = np.array([row["mixing_width"] for row in rows], dtype=np.float64)

    writeDatFile(
        outDir / "rayleigh_taylor_trajectory.dat",
        [xKey, "bubbleHeight", "spikeDepth"],
        [x, bubble, spike],
    )

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, bubble, label="bubble height")
    plt.plot(x, spike, label="spike depth")
    if xKey in {"t_star", "time_dimensionless"}:
        for reference in references:
            labelBase = reference["path"].stem
            plt.plot(
                reference["time_dimensionless"],
                reference["bubble_height"],
                linestyle=":",
                label=f"{labelBase} bubble ref",
            )
            plt.plot(
                reference["time_dimensionless"],
                reference["spike_depth"],
                linestyle="-.",
                label=f"{labelBase} spike ref",
            )
    plt.xlabel(xLabel)
    plt.ylabel("penetration distance")
    plt.title("Rayleigh-Taylor bubble/spike trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outDir / "rayleigh_taylor_trajectory.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    writeDatFile(
        outDir / "rayleigh_taylor_mixing_width.dat",
        [xKey, "mixingWidth"],
        [x, mixing],
    )

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, mixing, label="mixing width")
    plt.xlabel(xLabel)
    plt.ylabel("width")
    plt.title("Rayleigh-Taylor mixing-layer width")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outDir / "rayleigh_taylor_mixing_width.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveHealthPlot(outDir, rows, xKey, xLabel):
    x = np.array([row[xKey] for row in rows], dtype=np.float64)
    massError = np.array(
        [row["relative_mass_error_phi"] for row in rows], dtype=np.float64
    )
    maxVelocity = np.array([row["max_velocity"] for row in rows], dtype=np.float64)
    minRho = np.array([row["min_rho"] for row in rows], dtype=np.float64)
    maxRho = np.array([row["max_rho"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    axes[0].plot(x, massError)
    axes[0].set_ylabel("relative mass error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, maxVelocity)
    axes[1].set_ylabel("max |u|")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, minRho, label="min rho")
    axes[2].plot(x, maxRho, label="max rho")
    axes[2].set_xlabel(xLabel)
    axes[2].set_ylabel("density")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Rayleigh-Taylor health diagnostics")
    fig.tight_layout()
    fig.savefig(outDir / "rayleigh_taylor_health.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close(fig)


def centralSlice(phi, verticalDirection):
    nz, ny, nx = phi.shape
    if verticalDirection == "z":
        return phi[:, ny // 2, :], "x", "z"
    if verticalDirection == "y":
        return phi[nz // 2, :, :], "x", "y"
    if verticalDirection == "x":
        return phi[:, :, nx // 2], "y", "z"
    raise RuntimeError(f"Unsupported verticalDirection={verticalDirection!r}")


def saveRepresentativeSlices(outDir, runDir, metadata, steps, verticalDirection):
    if len(steps) == 1:
        selected = [steps[0]]
    else:
        selected = [steps[0], steps[len(steps) // 2], steps[-1]]

    fig, axes = plt.subplots(
        1, len(selected), figsize=(4.0 * len(selected), 4.2), squeeze=False
    )
    image = None
    for axis, step in zip(axes[0], selected):
        phi, _ = readPhi(runDir, metadata, step)
        sliceValues, horizontalLabel, verticalLabel = centralSlice(
            phi, verticalDirection
        )
        image = axis.imshow(
            sliceValues,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.contour(sliceValues, levels=[0.5], colors="white", linewidths=0.8)
        axis.set_title(f"step {step}")
        axis.set_xlabel(horizontalLabel)
        axis.set_ylabel(verticalLabel)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="phi", shrink=0.8)
    fig.suptitle("Rayleigh-Taylor phase-field slices")
    fig.savefig(
        outDir / "rayleigh_taylor_slices.png", dpi=figureDpi, bbox_inches="tight"
    )
    if showPlots:
        plt.show()
    plt.close(fig)


def writeCsv(outDir, rows):
    csvPath = outDir / "rayleigh_taylor_summary.csv"
    columns = [
        "step",
        "time",
        "time_lattice",
        "t_star",
        "time_dimensionless",
        "lower_interface",
        "upper_interface",
        "bubble_position",
        "spike_position",
        "bubble_height",
        "spike_depth",
        "mixing_width",
        "interface_min",
        "interface_max",
        "interface_mean",
        "interface_std",
        "interface_valid_columns",
        "interface_multiple_crossing_columns",
        "phi_min",
        "phi_max",
        "phi_mass",
        "mass_phi",
        "phi_mass_relative_change_percent",
        "relative_mass_error_phi",
        "max_velocity_magnitude",
        "max_velocity",
        "rms_velocity_magnitude",
        "rho_min",
        "rho_max",
        "min_rho",
        "max_rho",
        "phi_overshoot",
        "density_bad",
        "mass_error_large",
        "velocity_large",
    ]
    with csvPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def writeTextSummary(outDir, rows, metadata, warnings, xKey):
    final = rows[-1]
    maxVelocitySeries = [
        row["max_velocity"] for row in rows if np.isfinite(row["max_velocity"])
    ]
    maxVelocityOverall = max(maxVelocitySeries) if maxVelocitySeries else np.nan
    anyPhiOvershoot = any(row["phi_overshoot"] for row in rows)
    anyDensityBad = any(row["density_bad"] for row in rows)
    anyMassErrorLarge = any(row["mass_error_large"] for row in rows)
    anyVelocityLarge = any(row["velocity_large"] for row in rows)

    reportLines = [
        "Rayleigh-Taylor trajectory validation summary",
        "",
        f"snapshots = {len(rows)}",
        f"caseName = {metadata.get('caseName', metadata.get('case_name', 'unknown'))}",
        f"runId = {metadata.get('runId', runId)}",
        f"verticalDirection = {metadataText(metadata, 'verticalDirection', 'unknown')}",
        f"gravity_direction = {metadataText(metadata, 'gravity_direction', '-z')}",
        "bubble_position = max phi=0.5 interface coordinate",
        "bubble_height = bubble_position - initial_interface",
        "spike_position = min phi=0.5 interface coordinate",
        "spike_depth = initial_interface - spike_position",
        "mixing_width = bubble_height + spike_depth",
        f"initialInterface = {metadataFloat(metadata, ['INITIAL_INTERFACE_Z', 'initialInterface'], np.nan)}",
        f"ATWOOD = {metadataFloat(metadata, ['ATWOOD', 'Atwood'], np.nan)}",
        f"REYNOLDS = {metadataFloat(metadata, ['REYNOLDS', 'Re'], np.nan)}",
        f"WEBER = {metadataFloat(metadata, ['WEBER', 'We'], np.nan)}",
        f"GRAVITY = {metadataFloat(metadata, ['GRAVITY', 'gravity'], np.nan)}",
        f"L_CHAR = {metadataFloat(metadata, 'L_CHAR', np.nan)}",
        "time_dimensionless = time_lattice * sqrt(Atwood * |gravity| / L_CHAR)",
        "",
        f"final_step = {final['step']}",
        f"final_time_lattice = {final['time_lattice']}",
        f"final_{xKey} = {final[xKey]}",
        f"final_bubble_position = {final['bubble_position']}",
        f"final_spike_position = {final['spike_position']}",
        f"final_bubble_height = {final['bubble_height']}",
        f"final_spike_depth = {final['spike_depth']}",
        f"final_mixing_width = {final['mixing_width']}",
        f"final_phi_min = {final['phi_min']}",
        f"final_phi_max = {final['phi_max']}",
        f"final_mass_phi = {final['mass_phi']}",
        f"final_relative_mass_error_phi = {final['relative_mass_error_phi']}",
        f"final_phi_mass_relative_change_percent = {final['phi_mass_relative_change_percent']}",
        f"final_max_velocity = {final['max_velocity']}",
        f"max_velocity_overall = {maxVelocityOverall}",
        f"final_min_rho = {final['min_rho']}",
        f"final_max_rho = {final['max_rho']}",
        "",
        f"any_phi_overshoot = {anyPhiOvershoot}",
        f"any_density_bad = {anyDensityBad}",
        f"any_mass_error_large = {anyMassErrorLarge}",
        f"any_velocity_large = {anyVelocityLarge}",
    ]
    if warnings:
        reportLines.extend(["", "Warnings:"])
        reportLines.extend(f"- {message}" for message in warnings)
    writeReport(outDir / "rayleigh_taylor_summary.txt", reportLines)


def buildRow(
    runDir,
    metadata,
    step,
    verticalDirection,
    heavyPhasePhi,
    lowLevel,
    highLevel,
    initialInterface,
    dt,
    tStarFactor,
    firstPhiMass,
    rhoHeavy,
    rhoLight,
    warnings,
):
    phi, alias = readPhi(runDir, metadata, step)
    heavy = phaseMeasure(phi, heavyPhasePhi)
    coord = verticalCoordinates(heavy, verticalDirection)
    profile = verticalProfile(heavy, verticalDirection)
    lower, upper = interfaceBounds(
        coord, profile, lowLevel, highLevel, initialInterface, warnings
    )
    eta, multipleCrossingColumns = interfaceHeightField(
        heavy, verticalDirection, initialInterface, 0.5
    )
    stats = interfaceStats(eta, initialInterface)

    phiMin = float(np.nanmin(phi))
    phiMax = float(np.nanmax(phi))
    phiMass = float(np.nansum(phi))
    if firstPhiMass is None or firstPhiMass == 0.0:
        massChange = np.nan
        relativeMassError = np.nan
    else:
        massChange = 100.0 * (phiMass - firstPhiMass) / firstPhiMass
        relativeMassError = abs(phiMass - firstPhiMass) / abs(firstPhiMass)

    ux, uxAlias = readOptionalScalar(
        runDir, metadata, ["ux", "u", "vx", "velocityX"], step
    )
    uy, uyAlias = readOptionalScalar(
        runDir, metadata, ["uy", "v", "vy", "velocityY"], step
    )
    uz, uzAlias = readOptionalScalar(
        runDir, metadata, ["uz", "w", "vz", "velocityZ"], step
    )
    if ux is not None and uy is not None and uz is not None:
        speed = np.sqrt(
            ux.astype(np.float64) ** 2
            + uy.astype(np.float64) ** 2
            + uz.astype(np.float64) ** 2
        )
        maxVelocity = float(np.nanmax(speed))
        rmsVelocity = float(np.sqrt(np.nanmean(speed * speed)))
        velocityAliases = (uxAlias, uyAlias, uzAlias)
    else:
        maxVelocity = np.nan
        rmsVelocity = np.nan
        velocityAliases = None
        warn(warnings, "velocity components missing; velocity diagnostics skipped")

    if rhoHeavy is not None and rhoLight is not None:
        rho = rhoLight + (rhoHeavy - rhoLight) * heavy
        rhoMin = float(np.nanmin(rho))
        rhoMax = float(np.nanmax(rho))
    else:
        rhoMin = np.nan
        rhoMax = np.nan

    time = float(step) * dt
    tStar = time * tStarFactor if tStarFactor is not None else np.nan
    bubblePosition = stats["interface_max"]
    spikePosition = stats["interface_min"]

    return {
        "row": {
            "step": int(step),
            "time": time,
            "time_lattice": time,
            "t_star": tStar,
            "time_dimensionless": tStar,
            "lower_interface": lower,
            "upper_interface": upper,
            "bubble_position": bubblePosition,
            "spike_position": spikePosition,
            "bubble_height": stats["bubble_height"],
            "spike_depth": stats["spike_depth"],
            "mixing_width": stats["mixing_width"],
            "interface_min": stats["interface_min"],
            "interface_max": stats["interface_max"],
            "interface_mean": stats["interface_mean"],
            "interface_std": stats["interface_std"],
            "interface_valid_columns": stats["interface_valid_columns"],
            "interface_multiple_crossing_columns": multipleCrossingColumns,
            "phi_min": phiMin,
            "phi_max": phiMax,
            "phi_mass": phiMass,
            "mass_phi": phiMass,
            "phi_mass_relative_change_percent": massChange,
            "relative_mass_error_phi": relativeMassError,
            "max_velocity_magnitude": maxVelocity,
            "max_velocity": maxVelocity,
            "rms_velocity_magnitude": rmsVelocity,
            "rho_min": rhoMin,
            "rho_max": rhoMax,
            "min_rho": rhoMin,
            "max_rho": rhoMax,
            "phi_overshoot": bool(phiMin < -0.05 or phiMax > 1.05),
            "density_bad": bool(np.isfinite(rhoMin) and rhoMin <= 0.0),
            "mass_error_large": bool(
                np.isfinite(relativeMassError)
                and massTolerance is not None
                and relativeMassError > massTolerance
            ),
            "velocity_large": bool(
                velocityLimit is not None
                and np.isfinite(maxVelocity)
                and maxVelocity > velocityLimit
            ),
        },
        "phiAlias": alias,
        "velocityAliases": velocityAliases,
    }


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    outDir = getPostDir(runDir)
    warnings = []

    steps = listAvailableSteps(runDir)
    if not steps:
        raise RuntimeError(f"No snapshots found in {runDir / 'binaries'}")
    if selectedStep is not None:
        selectedValue = int(selectedStep)
        if selectedValue not in steps:
            availableText = ", ".join(str(value) for value in steps)
            raise RuntimeError(
                f"Selected step {selectedValue} is unavailable. Available steps: {availableText}"
            )
        steps = [selectedValue]

    print(f"run directory: {runDir}")
    print(f"metadata loaded from: {runDir / 'metadata.txt'}")
    print(f"snapshots found: {len(steps)}")

    verticalDirection = metadataText(metadata, "verticalDirection", None)
    if verticalDirection is None:
        verticalDirection = "z"
        warn(
            warnings,
            "verticalDirection missing; using current RTICase convention verticalDirection=z",
        )
    verticalDirection = verticalDirection.strip().lower()

    heavyPhasePhi = metadataFloat(metadata, "HEAVY_PHASE_PHI", None)
    if heavyPhasePhi is None:
        heavyPhasePhi = metadataFloat(metadata, "LIQUID_PHASE_PHI", None)
    if heavyPhasePhi is None:
        heavyPhasePhi = 1.0
        warn(
            warnings,
            "HEAVY_PHASE_PHI missing; using current RTICase convention phi=1 for heavy phase",
        )

    lowLevel = metadataFloat(metadata, "BULK_GAS_PHI_MAX", lowMixFraction)
    highLevel = metadataFloat(metadata, "BULK_LIQUID_PHI_MIN", highMixFraction)
    initialInterface = metadataFloat(
        metadata, ["INITIAL_INTERFACE_Z", "initialInterface"], None
    )
    if initialInterface is None:
        nz = getInt(metadata, "NZ")
        initialInterface = 0.5 * float(nz)
        warn(
            warnings,
            "INITIAL_INTERFACE_Z missing; using current RTICase convention 0.5*NZ",
        )

    dt, tStarFactor = timeScale(metadata, warnings)
    xKey = "t_star" if tStarFactor is not None else "step"
    xLabel = "t*" if tStarFactor is not None else "step"
    rhoHeavy, rhoLight = inferDensityPair(metadata, warnings)

    rows = []
    firstAlias = None
    firstVelocityAliases = None
    firstPhiMass = None
    for step in steps:
        rowData = buildRow(
            runDir,
            metadata,
            step,
            verticalDirection,
            heavyPhasePhi,
            lowLevel,
            highLevel,
            initialInterface,
            dt,
            tStarFactor,
            firstPhiMass,
            rhoHeavy,
            rhoLight,
            warnings,
        )
        row = rowData["row"]
        if firstPhiMass is None:
            firstPhiMass = row["phi_mass"]
            row["phi_mass_relative_change_percent"] = 0.0
            row["relative_mass_error_phi"] = 0.0
            row["mass_error_large"] = False
        if firstAlias is None:
            firstAlias = rowData["phiAlias"]
        if firstVelocityAliases is None and rowData["velocityAliases"] is not None:
            firstVelocityAliases = rowData["velocityAliases"]
        rows.append(row)

    if any(row["interface_multiple_crossing_columns"] > 0 for row in rows):
        warn(
            warnings,
            "one or more columns had multiple phi=0.5 crossings; "
            "selected crossing nearest initial interface",
        )
    if any(row["mass_error_large"] for row in rows):
        warn(warnings, f"relative phase mass error exceeded tolerance {massTolerance}")
    if any(row["density_bad"] for row in rows):
        warn(warnings, "nonphysical density detected: min_rho <= 0")
    references = loadReferenceCsvs(referenceCsv, warnings)

    print(f"field detected: phi={firstAlias}")
    if firstVelocityAliases is not None:
        print(
            "velocity fields detected: "
            f"{firstVelocityAliases[0]}, {firstVelocityAliases[1]}, {firstVelocityAliases[2]}"
        )

    writeCsv(outDir, rows)
    saveEvolutionPlots(outDir, rows, xKey, xLabel, references)
    saveHealthPlot(outDir, rows, xKey, xLabel)
    saveRepresentativeSlices(outDir, runDir, metadata, steps, verticalDirection)
    writeTextSummary(outDir, rows, metadata, warnings, xKey)

    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    applyCommandLineArgs()
    main()
