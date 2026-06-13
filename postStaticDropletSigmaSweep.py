caseName = "static_droplet"
runIds = [
    "caseOne",
]
comparisonRunId = "sigma_sweep"
outputRoot = "output"
showPlots = False
figureDpi = 600

import csv

import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import getFloat, getPostDir, getRunDir, readMetadata, writeDatFile


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


def recoveredSigma(deltaP, radius):
    if not np.isfinite(deltaP) or not np.isfinite(radius) or radius <= 0.0:
        return np.nan
    return 0.5 * deltaP * radius


def loadRunRow(runId):
    runDir = getRunDir(caseName, runId, outputRoot)
    postDir = getPostDir(runDir)
    metadata = readMetadata(runDir)
    summary = readSummary(postDir / "static_droplet_summary.csv")

    sigmaAnalytical = getFloat(metadata, "SIGMA")
    deltaP = summaryFloat(summary, "delta_p_recovered")
    radiusEff = firstSummaryFloat(summary, ["radius_effective_lu", "radius_effective"])
    radiusTarget = firstSummaryFloat(
        summary, ["radius_target_lu", "radius_target"], getFloat(metadata, "R_INIT")
    )

    sigmaReff = firstSummaryFloat(
        summary, ["sigma_recovered_reff_lu", "sigma_recovered_reff"]
    )
    if not np.isfinite(sigmaReff):
        sigmaReff = recoveredSigma(deltaP, radiusEff)

    sigmaR0 = firstSummaryFloat(
        summary, ["sigma_recovered_r0_lu", "sigma_recovered_r0"]
    )
    if not np.isfinite(sigmaR0):
        sigmaR0 = recoveredSigma(deltaP, radiusTarget)

    return {
        "runId": runId,
        "sigmaAnalytical": sigmaAnalytical,
        "sigmaRecoveredReffLu": sigmaReff,
        "sigmaRecoveredR0Lu": sigmaR0,
        "relativeErrorReffLu": (sigmaReff - sigmaAnalytical) / sigmaAnalytical,
        "relativeErrorR0Lu": (sigmaR0 - sigmaAnalytical) / sigmaAnalytical,
        "deltaPRecovered": deltaP,
        "radiusEffectiveLu": radiusEff,
        "radiusTargetLu": radiusTarget,
    }


def writeRowsCsv(path, rows):
    columns = [
        "runId",
        "sigmaAnalytical",
        "sigmaRecoveredReffLu",
        "sigmaRecoveredR0Lu",
        "relativeErrorReffLu",
        "relativeErrorR0Lu",
        "deltaPRecovered",
        "radiusEffectiveLu",
        "radiusTargetLu",
    ]
    with path.open("w", encoding="utf-8", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def saveSigmaPlot(outDir, rows):
    sigmaAnalytical = np.array([row["sigmaAnalytical"] for row in rows], dtype=np.float64)
    sigmaReff = np.array([row["sigmaRecoveredReffLu"] for row in rows], dtype=np.float64)
    sigmaR0 = np.array([row["sigmaRecoveredR0Lu"] for row in rows], dtype=np.float64)

    writeDatFile(
        outDir / "static_droplet_sigma_sweep.dat",
        ["sigmaAnalytical", "sigmaRecoveredReffLu", "sigmaRecoveredR0Lu"],
        [sigmaAnalytical, sigmaReff, sigmaR0],
    )

    finiteValues = np.concatenate(
        [
            sigmaAnalytical[np.isfinite(sigmaAnalytical)],
            sigmaReff[np.isfinite(sigmaReff)],
            sigmaR0[np.isfinite(sigmaR0)],
        ]
    )
    if finiteValues.size == 0:
        raise RuntimeError("No finite sigma values found for plotting")

    minSigma = float(np.min(finiteValues))
    maxSigma = float(np.max(finiteValues))
    padding = 0.05 * (maxSigma - minSigma) if maxSigma > minSigma else 0.05 * maxSigma
    if padding <= 0.0:
        padding = 1.0e-12

    lineMin = minSigma - padding
    lineMax = maxSigma + padding

    plt.figure(figsize=(6.2, 5.2))
    plt.plot([lineMin, lineMax], [lineMin, lineMax], "k--", linewidth=1.0, label="analytical")
    plt.scatter(sigmaAnalytical, sigmaReff, label="recovered with R_eff", s=48)
    plt.scatter(sigmaAnalytical, sigmaR0, marker="s", label="recovered with R0", s=42)
    for row in rows:
        plt.annotate(
            row["runId"],
            (row["sigmaAnalytical"], row["sigmaRecoveredReffLu"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    plt.xlabel("analytical sigma")
    plt.ylabel("recovered sigma")
    plt.xlim(lineMin, lineMax)
    plt.ylim(lineMin, lineMax)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_sigma_sweep.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveErrorPlot(outDir, rows):
    sigmaAnalytical = np.array([row["sigmaAnalytical"] for row in rows], dtype=np.float64)
    errorReff = np.array([row["relativeErrorReffLu"] for row in rows], dtype=np.float64)
    errorR0 = np.array([row["relativeErrorR0Lu"] for row in rows], dtype=np.float64)

    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.plot(sigmaAnalytical, 100.0 * errorReff, "o-", label="R_eff")
    plt.plot(sigmaAnalytical, 100.0 * errorR0, "s-", label="R0")
    plt.xlabel("analytical sigma")
    plt.ylabel("recovered sigma error (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outDir / "static_droplet_sigma_error.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def main():
    if not runIds:
        raise RuntimeError("Fill runIds with at least one static droplet simulation ID")

    rows = [loadRunRow(runId) for runId in runIds]
    outDir = getPostDir(getRunDir(caseName, comparisonRunId, outputRoot))

    writeRowsCsv(outDir / "static_droplet_sigma_sweep.csv", rows)
    saveSigmaPlot(outDir, rows)
    saveErrorPlot(outDir, rows)

    print(f"processed runs: {', '.join(runIds)}")
    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    main()
