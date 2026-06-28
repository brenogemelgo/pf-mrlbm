caseName = "static_droplet"
runIds = [
    # "ONE_002_14",
    # "ONE_002_16",
    # "ONE_002_18",
    # "ONE_002_20",
    # "ONE_004_14",
    # "ONE_004_16",
    # "ONE_004_18",
    # "ONE_004_20",
    # "ONE_006_14",
    # "ONE_006_16",
    # "ONE_006_18",
    # "ONE_006_20",
    # "ONE_008_14",
    # "ONE_008_16",
    # "ONE_008_18",
    # "ONE_008_20",
    # "TWO_SAFE_002_20",
    # "TWO_SAFE_002_24",
    # "TWO_SAFE_002_28",
    # "TWO_SAFE_002_32",
    # "TWO_SAFE_004_20",
    # "TWO_SAFE_004_24",
    # "TWO_SAFE_004_28",
    # "TWO_SAFE_004_32",
    # "TWO_SAFE_006_20",
    # "TWO_SAFE_006_24",
    # "TWO_SAFE_006_28",
    # "TWO_SAFE_006_32",
    # "TWO_SAFE_008_20",
    # "TWO_SAFE_008_24",
    # "TWO_SAFE_008_28",
    # "TWO_SAFE_RETRY_008_32",
    # "TWO_002_14",
    # "TWO_002_16",
    # "TWO_002_18",
    # "TWO_002_20",
    # "TWO_004_14",
    # "TWO_004_16",
    # "TWO_004_18",
    # "TWO_004_20",
    # "TWO_006_14",
    # "TWO_006_16",
    # "TWO_006_18",
    # "TWO_006_20",
    # "TWO_008_14",
    # "TWO_008_16",
    # "TWO_008_18",
    # "TWO_008_20",
    "THREE_SAFE_002_28",
    "THREE_SAFE_002_32",
    "THREE_SAFE_002_36",
    "THREE_SAFE_002_40",
    "THREE_SAFE_004_28",
    "THREE_SAFE_004_32",
    "THREE_SAFE_004_36",
    "THREE_SAFE_004_40",
    "THREE_SAFE_006_28",
    "THREE_SAFE_006_32",
    "THREE_SAFE_006_36",
    "THREE_SAFE_006_40",
    "THREE_SAFE_008_28",
    "THREE_SAFE_008_32",
    "THREE_SAFE_008_36",
    "THREE_SAFE_008_40",
    # "THREE_002_14",
    # "THREE_002_16",
    # "THREE_002_18",
    # "THREE_002_20",
    # "THREE_004_14",
    # "THREE_004_16",
    # "THREE_004_18",
    # "THREE_004_20",
    # "THREE_006_14",
    # "THREE_006_16",
    # "THREE_006_18",
    # "THREE_006_20",
    # "THREE_008_14",
    # "THREE_008_16",
    # "THREE_008_18",
    # "THREE_008_20",
]
comparisonRunId = "sigma_sweep"
outputRoot = "output"
showPlots = False
figureDpi = 600
radiusGroupingTolerance = 1.0e-12

import csv

import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import getFloat, getPostDir, getRunDir, readMetadata, writeDatFile

pointColumns = [
    "runId",
    "sigmaReference",
    "sigmaNumerical",
    "relativeErrorPercent",
    "deltaPRecovered",
    "deltaP0",
    "deltaPCorrected",
    "radiusEffectiveLu",
    "radiusTargetLu",
    "inverseRadiusEffective",
]

fitColumns = [
    "sigmaReference",
    "sigmaNumerical",
    "relativeErrorPercent",
    "slope",
    "deltaP0",
    "rSquared",
    "numberOfRuns",
]


def readSummary(path):
    if not path.exists():
        raise RuntimeError(f"Missing static droplet summary CSV: {path}")

    values = {}
    with path.open("r", encoding="utf-8", newline="") as inputFile:
        reader = csv.DictReader(inputFile)
        for row in reader:
            values[row["metric"]] = row["value"]
    return values


def summaryFloat(summary, key, defaultValue=np.nan):
    try:
        return float(summary[key])
    except (KeyError, TypeError, ValueError):
        return defaultValue


def firstSummaryFloat(summary, keys, defaultValue=np.nan):
    for key in keys:
        value = summaryFloat(summary, key, np.nan)
        if np.isfinite(value):
            return value
    return defaultValue


def warn(message):
    print(f"warning: {message}")


def formatRadius(radius):
    if np.isclose(radius, round(radius)):
        return str(int(round(radius)))
    return f"{radius:g}"


def loadRunRow(runId):
    runDir = getRunDir(caseName, runId, outputRoot)
    postDir = getPostDir(runDir)
    metadata = readMetadata(runDir)
    summary = readSummary(postDir / "static_droplet_summary.csv")

    sigmaReference = getFloat(metadata, "SIGMA")
    deltaPRecovered = summaryFloat(summary, "delta_p_recovered")
    radiusEffectiveLu = firstSummaryFloat(
        summary, ["radius_effective_lu", "radius_effective"]
    )
    radiusTargetLu = firstSummaryFloat(
        summary, ["radius_target_lu", "radius_target"], getFloat(metadata, "R_INIT")
    )

    return {
        "runId": runId,
        "sigmaReference": sigmaReference,
        "deltaPRecovered": deltaPRecovered,
        "radiusEffectiveLu": radiusEffectiveLu,
        "radiusTargetLu": radiusTargetLu,
    }


def rowValidationErrors(row):
    errors = []
    if not np.isfinite(row["deltaPRecovered"]):
        errors.append("nonfinite pressure jump")
    if not np.isfinite(row["radiusEffectiveLu"]) or row["radiusEffectiveLu"] <= 0.0:
        errors.append("nonfinite or nonpositive effective radius")
    if not np.isfinite(row["radiusTargetLu"]) or row["radiusTargetLu"] <= 0.0:
        errors.append("nonfinite or nonpositive target radius")
    if not np.isfinite(row["sigmaReference"]) or row["sigmaReference"] <= 0.0:
        errors.append("nonfinite or nonpositive reference surface tension")
    return errors


def loadValidRows():
    rows = []
    for runId in runIds:
        try:
            row = loadRunRow(runId)
        except RuntimeError as error:
            warn(f"skipping run {runId}: {error}")
            continue

        errors = rowValidationErrors(row)
        if errors:
            warn(f"skipping run {runId}: {', '.join(errors)}")
            continue

        rows.append(row)

    return rows


def areClose(a, b, tolerance):
    return abs(float(a) - float(b)) <= tolerance


def groupRowsBySigma(rows):
    sortedRows = sorted(rows, key=lambda row: row["sigmaReference"])
    groups = []

    for row in sortedRows:
        if groups and areClose(
            row["sigmaReference"], groups[-1]["sigmaReference"], radiusGroupingTolerance
        ):
            groups[-1]["rows"].append(row)
            sigmaValues = [
                groupRow["sigmaReference"] for groupRow in groups[-1]["rows"]
            ]
            groups[-1]["sigmaReference"] = float(np.mean(sigmaValues))
        else:
            groups.append(
                {
                    "sigmaReference": row["sigmaReference"],
                    "rows": [row],
                }
            )

    for group in groups:
        group["rows"].sort(key=lambda row: -row["radiusEffectiveLu"])

    return groups


def distinctRadiusValues(rows):
    distinctValues = []
    duplicateCount = 0
    for row in sorted(rows, key=lambda row: row["radiusEffectiveLu"]):
        radius = row["radiusEffectiveLu"]
        if any(
            areClose(radius, value, radiusGroupingTolerance) for value in distinctValues
        ):
            duplicateCount += 1
        else:
            distinctValues.append(radius)
    return distinctValues, duplicateCount


def fitSamplesByRadius(rows):
    sortedRows = sorted(rows, key=lambda row: row["radiusEffectiveLu"])
    buckets = []
    for row in sortedRows:
        if buckets and areClose(
            row["radiusEffectiveLu"],
            buckets[-1]["radiusEffectiveLu"],
            radiusGroupingTolerance,
        ):
            buckets[-1]["rows"].append(row)
        else:
            buckets.append(
                {
                    "radiusEffectiveLu": row["radiusEffectiveLu"],
                    "rows": [row],
                }
            )

    inverseRadius = []
    deltaPRecovered = []
    for bucket in buckets:
        inverseValues = [1.0 / row["radiusEffectiveLu"] for row in bucket["rows"]]
        pressureValues = [row["deltaPRecovered"] for row in bucket["rows"]]
        inverseRadius.append(float(np.mean(inverseValues)))
        deltaPRecovered.append(float(np.mean(pressureValues)))

    return (
        np.array(inverseRadius, dtype=np.float64),
        np.array(deltaPRecovered, dtype=np.float64),
    )


def fitLaplaceGroup(group):
    rows = group["rows"]
    sigmaReference = group["sigmaReference"]
    distinctRadii, duplicateCount = distinctRadiusValues(rows)

    if duplicateCount > 0:
        warn(
            f"sigma_ref={sigmaReference:.17g}: found {duplicateCount} duplicate "
            "effective radius value(s); all rows are still shown, but duplicates "
            "are not independent radius information"
        )

    if len(distinctRadii) < 2:
        raise RuntimeError(
            f"sigma_ref={sigmaReference:.17g}: at least two distinct effective radii "
            "are required for a linear fit"
        )

    inverseRadiusFit, deltaPRecoveredFit = fitSamplesByRadius(rows)
    inverseRadiusPoint = np.array(
        [1.0 / row["radiusEffectiveLu"] for row in rows], dtype=np.float64
    )
    deltaPRecoveredPoint = np.array(
        [row["deltaPRecovered"] for row in rows], dtype=np.float64
    )

    slope, deltaP0 = np.polyfit(inverseRadiusFit, deltaPRecoveredFit, 1)
    fittedRaw = slope * inverseRadiusFit + deltaP0
    residual = deltaPRecoveredFit - fittedRaw
    ssResidual = float(np.sum(residual * residual))
    centered = deltaPRecoveredFit - np.mean(deltaPRecoveredFit)
    ssTotal = float(np.sum(centered * centered))
    if ssTotal > 0.0:
        rSquared = 1.0 - ssResidual / ssTotal
    else:
        rSquared = 1.0 if ssResidual <= 1.0e-30 else 0.0

    sigmaNumerical = 0.5 * slope
    relativeErrorPercent = 100.0 * abs(sigmaNumerical - sigmaReference) / sigmaReference

    pointRows = []
    for row, inverseValue, pressureValue in zip(
        rows, inverseRadiusPoint, deltaPRecoveredPoint
    ):
        pointRows.append(
            {
                "runId": row["runId"],
                "sigmaReference": sigmaReference,
                "sigmaNumerical": sigmaNumerical,
                "relativeErrorPercent": relativeErrorPercent,
                "deltaPRecovered": pressureValue,
                "deltaP0": deltaP0,
                "deltaPCorrected": pressureValue - deltaP0,
                "radiusEffectiveLu": row["radiusEffectiveLu"],
                "radiusTargetLu": row["radiusTargetLu"],
                "inverseRadiusEffective": inverseValue,
            }
        )

    return {
        "sigmaReference": sigmaReference,
        "sigmaNumerical": sigmaNumerical,
        "relativeErrorPercent": relativeErrorPercent,
        "slope": slope,
        "deltaP0": deltaP0,
        "rSquared": rSquared,
        "numberOfRuns": len(rows),
        "rows": rows,
        "pointRows": pointRows,
        "inverseRadius": inverseRadiusPoint,
        "deltaPCorrected": deltaPRecoveredPoint - deltaP0,
    }


def writeRowsCsv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def writePointDat(path, pointRows):
    numericColumns = [column for column in pointColumns if column != "runId"]
    writeDatFile(
        path,
        numericColumns,
        [[row[column] for row in pointRows] for column in numericColumns],
    )


def paddedUpperLimit(values):
    finiteValues = np.asarray(values, dtype=np.float64)
    finiteValues = finiteValues[np.isfinite(finiteValues)]
    if finiteValues.size == 0:
        return 1.0

    maxValue = float(np.max(finiteValues))
    if maxValue <= 0.0:
        return 1.0
    return maxValue * 1.05


def saveLaplacePlot(outDir, fits):
    allInverseRadius = []
    allRecoveredPressure = []
    allTheoryPressure = []

    plt.figure(figsize=(9.6, 6.0))

    for fit in fits:
        pointRows = fit["pointRows"]
        inverseRadius = np.array(
            [row["inverseRadiusEffective"] for row in pointRows], dtype=np.float64
        )
        recoveredPressure = np.array(
            [row["deltaPRecovered"] for row in pointRows], dtype=np.float64
        )

        lineMax = 1.05 * float(np.max(inverseRadius))
        inverseRadiusLine = np.linspace(0.0, lineMax, 128)
        deltaPTheory = 2.0 * fit["sigmaReference"] * inverseRadiusLine

        label = (
            rf"$\sigma_{{ref}}={fit['sigmaReference']:.2f}$, "
            rf"$\sigma_{{fit}}={fit['sigmaNumerical']:.4f}$, "
            rf"$e={fit['relativeErrorPercent']:.2f}\%$"
        )
        points = plt.plot(inverseRadius, recoveredPressure, "o", label=label)[0]
        color = points.get_color()
        plt.plot(inverseRadiusLine, deltaPTheory, "--", color=color)

        for pointIndex, row in enumerate(pointRows):
            yOffset = 4 if pointIndex % 2 == 0 else -10
            plt.annotate(
                f"R={formatRadius(row['radiusTargetLu'])}",
                (row["inverseRadiusEffective"], row["deltaPRecovered"]),
                textcoords="offset points",
                xytext=(5, yOffset),
                fontsize=8,
            )

        allInverseRadius.extend(inverseRadius.tolist())
        allRecoveredPressure.extend(recoveredPressure.tolist())
        allTheoryPressure.extend(deltaPTheory.tolist())

    xUpper = paddedUpperLimit(allInverseRadius)
    yUpper = paddedUpperLimit(allRecoveredPressure + allTheoryPressure)

    plt.xlabel(r"$1/R_{\mathrm{eff}}$")
    plt.ylabel(r"$\Delta P$")
    plt.title("3D Laplace-law validation")
    plt.xlim(0.0, xUpper)
    plt.ylim(0.0, yUpper)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_laplace.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def verifyFiniteRows(rows, columns, label):
    for rowIndex, row in enumerate(rows):
        for column in columns:
            if column == "runId":
                continue
            if not np.isfinite(float(row[column])):
                raise RuntimeError(
                    f"Nonfinite {label} value at row {rowIndex + 1}, column {column}"
                )


def verifyFitConsistency(pointRows, fitRows):
    for row in fitRows:
        if not np.isclose(row["sigmaNumerical"], 0.5 * row["slope"]):
            raise RuntimeError(
                f"sigmaNumerical is inconsistent with slope for sigma_ref={row['sigmaReference']}"
            )

    for row in pointRows:
        if not np.allclose(
            row["deltaPCorrected"], row["deltaPRecovered"] - row["deltaP0"]
        ):
            raise RuntimeError(
                f"deltaPCorrected is inconsistent with deltaPRecovered - deltaP0 "
                f"for run {row['runId']}"
            )


def main():
    if not runIds:
        raise RuntimeError("Fill runIds with at least one static droplet simulation ID")

    rows = loadValidRows()
    if not rows:
        raise RuntimeError("No valid static droplet runs remain after filtering")

    groups = groupRowsBySigma(rows)
    fits = []
    for group in groups:
        try:
            fits.append(fitLaplaceGroup(group))
        except RuntimeError as error:
            warn(f"skipping group: {error}")

    if not fits:
        raise RuntimeError(
            "No surface-tension group has enough distinct radii for fitting"
        )

    pointRows = []
    fitRows = []
    for fit in fits:
        pointRows.extend(fit["pointRows"])
        fitRows.append({column: fit[column] for column in fitColumns})

    verifyFiniteRows(pointRows, pointColumns, "pointwise")
    verifyFiniteRows(fitRows, fitColumns, "fit")
    verifyFitConsistency(pointRows, fitRows)

    outDir = getPostDir(getRunDir(caseName, comparisonRunId, outputRoot))
    outDir.mkdir(parents=True, exist_ok=True)

    writeRowsCsv(outDir / "static_droplet_laplace_points.csv", pointRows, pointColumns)
    writeRowsCsv(outDir / "static_droplet_laplace_fits.csv", fitRows, fitColumns)
    writePointDat(outDir / "static_droplet_laplace_points.dat", pointRows)
    saveLaplacePlot(outDir, fits)

    for fit in fits:
        print(
            f"sigma_ref={fit['sigmaReference']:.6f}, "
            f"sigma_num={fit['sigmaNumerical']:.6f}, "
            f"error={fit['relativeErrorPercent']:.2f}%, "
            f"deltaP0={fit['deltaP0']:.17g}, "
            f"R2={fit['rSquared']:.6f}"
        )

    print(f"processed run count: {len(pointRows)}")
    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    main()
