caseName = "rti"
runId = "000"
selectedStep = None
outputRoot = "output"
outputSubdir = "post_rayleigh_taylor"
showPlots = False
figureDpi = 600
lowMixFraction = 0.05
highMixFraction = 0.95

import csv

import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import (
    getFloat,
    getInt,
    getMetadataValue,
    getRunDir,
    listAvailableFields,
    listAvailableSteps,
    readMetadata,
    readScalarField,
    writeDatFile,
    writeReport,
)


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
    errors = []
    for alias in aliases:
        try:
            return readScalarField(runDir, metadata, alias, step)[0], alias
        except RuntimeError as error:
            errors.append(str(error))
    availableText = ", ".join(listAvailableFields(runDir))
    raise RuntimeError(
        f"Missing required phase field. Tried aliases: {', '.join(aliases)}. Available fields: {availableText}"
    )


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


def crossingCoordinate(coord, profile, level):
    values = profile - level
    for index in range(profile.size - 1):
        left = values[index]
        right = values[index + 1]
        if left == 0.0:
            return float(coord[index])
        if left * right <= 0.0:
            denom = profile[index + 1] - profile[index]
            if abs(denom) <= 1.0e-30:
                return float(coord[index])
            weight = (level - profile[index]) / denom
            return float(coord[index] + weight * (coord[index + 1] - coord[index]))
    if values[-1] == 0.0:
        return float(coord[-1])
    return None


def interfaceBounds(
    coord, heavyProfile, lowLevel, highLevel, initialInterface, warnings
):
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
                "no mixed layer found in horizontally averaged profile; bounds set to initial interface",
            )

    if upper < lower:
        lower, upper = upper, lower

    return lower, upper


def timeScale(metadata, warnings):
    dt = metadataFloat(metadata, ["DT", "timestep", "physicalTimestep"], 1.0)
    atwood = metadataFloat(metadata, ["ATWOOD", "Atwood"], None)
    gravity = metadataFloat(metadata, "GRAVITY", None)
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


def saveEvolutionPlots(outDir, rows, xKey, xLabel):
    x = np.array([row[xKey] for row in rows], dtype=np.float64)
    bubble = np.array([row["bubble_height"] for row in rows], dtype=np.float64)
    spike = np.array([row["spike_depth"] for row in rows], dtype=np.float64)
    mixing = np.array([row["mixing_width"] for row in rows], dtype=np.float64)

    writeDatFile(
        outDir / "rayleigh_taylor_bubble_spike.dat",
        [xKey, "bubbleHeight", "spikeDepth"],
        [x, bubble, spike],
    )

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, bubble, label="bubble height")
    plt.plot(x, spike, label="spike depth")
    plt.xlabel(xLabel)
    plt.ylabel("displacement")
    plt.title("Rayleigh-Taylor bubble and spike growth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outDir / "rayleigh_taylor_bubble_spike.png", dpi=figureDpi)
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
        axis.set_title(f"step {step}")
        axis.set_xlabel(horizontalLabel)
        axis.set_ylabel(verticalLabel)
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
        "t_star",
        "lower_interface",
        "upper_interface",
        "bubble_height",
        "spike_depth",
        "mixing_width",
    ]
    with csvPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def writeTextSummary(outDir, rows, metadata, warnings, xKey):
    final = rows[-1]
    reportLines = [
        "Rayleigh-Taylor validation summary",
        "",
        f"snapshots = {len(rows)}",
        f"caseName = {metadata.get('caseName', metadata.get('case_name', 'unknown'))}",
        f"verticalDirection = {metadataText(metadata, 'verticalDirection', 'unknown')}",
        f"initialInterface = {metadataFloat(metadata, ['INITIAL_INTERFACE_Z', 'initialInterface'], np.nan)}",
        f"ATWOOD = {metadataFloat(metadata, 'ATWOOD', np.nan)}",
        f"GRAVITY = {metadataFloat(metadata, 'GRAVITY', np.nan)}",
        f"L_CHAR = {metadataFloat(metadata, 'L_CHAR', np.nan)}",
        "",
        f"final_step = {final['step']}",
        f"final_time = {final['time']}",
        f"final_{xKey} = {final[xKey]}",
        f"final_bubble_height = {final['bubble_height']}",
        f"final_spike_depth = {final['spike_depth']}",
        f"final_mixing_width = {final['mixing_width']}",
    ]
    if warnings:
        reportLines.extend(["", "Warnings:"])
        reportLines.extend(f"- {message}" for message in warnings)
    writeReport(outDir / "rayleigh_taylor_summary.txt", reportLines)


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    outDir = runDir / outputSubdir
    outDir.mkdir(parents=True, exist_ok=True)
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

    rows = []
    firstAlias = None
    for step in steps:
        phi, alias = readPhi(runDir, metadata, step)
        if firstAlias is None:
            firstAlias = alias
        heavy = phaseMeasure(phi, heavyPhasePhi)
        coord = verticalCoordinates(heavy, verticalDirection)
        profile = verticalProfile(heavy, verticalDirection)
        lower, upper = interfaceBounds(
            coord, profile, lowLevel, highLevel, initialInterface, warnings
        )
        time = float(step) * dt
        tStar = time * tStarFactor if tStarFactor is not None else np.nan
        rows.append(
            {
                "step": int(step),
                "time": time,
                "t_star": tStar,
                "lower_interface": lower,
                "upper_interface": upper,
                "bubble_height": max(0.0, upper - initialInterface),
                "spike_depth": max(0.0, initialInterface - lower),
                "mixing_width": max(0.0, upper - lower),
            }
        )

    print(f"field detected: phi={firstAlias}")
    writeCsv(outDir, rows)
    saveEvolutionPlots(outDir, rows, xKey, xLabel)
    saveRepresentativeSlices(outDir, runDir, metadata, steps, verticalDirection)
    writeTextSummary(outDir, rows, metadata, warnings, xKey)

    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    main()
