caseName = "rti"
outputRoot = "output"
outputDir = None
figureDpi = 600
comparisonPreset = "presentation"
trajectoryRunIds = [
    "RTI_TRAJECTORY_A05_RE256",
    "RTI_TRAJECTORY_A05_RE2048",
]
validationRunIds = trajectoryRunIds
presentationFixedReRunIds = [
    "RTI_PRESENTATION_RR1000_RM100_RE0150_WE0100",
    "RTI_PRESENTATION_RR1000_RM100_RE0150_WE0500",
    "RTI_PRESENTATION_RR1000_RM100_RE0150_WE1000",
]
presentationFixedWeRunIds = [
    "RTI_PRESENTATION_RR1000_RM100_RE0050_WE0500",
    "RTI_PRESENTATION_RR1000_RM100_RE0100_WE0500",
    "RTI_PRESENTATION_RR1000_RM100_RE0200_WE0500",
]
presentationRunIds = presentationFixedReRunIds + presentationFixedWeRunIds
stressRunIds = [
    "RTI_STRESS_RR10000_RM1000_RE0100_WE0500",
]
defaultRunIds = presentationRunIds

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from postCommon import getFloat, getRunDir, readMetadata


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Compare Rayleigh-Taylor run summaries"
    )
    parser.add_argument("--caseName", default=caseName)
    parser.add_argument("--outputRoot", default=outputRoot)
    parser.add_argument("--outputDir", default=outputDir)
    parser.add_argument("--figureDpi", type=int, default=figureDpi)
    parser.add_argument(
        "--preset",
        choices=[
            "trajectory",
            "validation",
            "presentation",
            "presentation-we",
            "presentation-re",
            "stress",
            "pretty",
            "pretty-we",
            "pretty-re",
        ],
        default=comparisonPreset,
    )
    parser.add_argument("--runIds", nargs="*", default=None)
    return parser.parse_args()


def metadataFloat(metadata, keys, defaultValue=np.nan):
    try:
        return getFloat(metadata, keys)
    except RuntimeError:
        return defaultValue


def readSummary(path):
    with path.open("r", encoding="utf-8", newline="") as inputFile:
        return list(csv.DictReader(inputFile))


def rowFloat(row, key, defaultValue=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return defaultValue


def rowBool(row, key):
    return str(row.get(key, "False")).strip().lower() == "true"


def loadRun(caseNameValue, outputRootValue, runId):
    runDir = getRunDir(caseNameValue, runId, outputRootValue)
    metadata = readMetadata(runDir)
    summaryPath = runDir / "post" / "rayleigh_taylor_summary.csv"
    if not summaryPath.exists():
        raise RuntimeError(f"missing summary CSV for {runId}: {summaryPath}")
    rows = readSummary(summaryPath)
    if not rows:
        raise RuntimeError(f"empty summary CSV for {runId}: {summaryPath}")
    return {
        "runId": runId,
        "metadata": metadata,
        "rows": rows,
    }


def writeAggregate(outDir, runs):
    path = outDir / "rayleigh_taylor_compare_final.csv"
    columns = [
        "runId",
        "reynolds",
        "weber",
        "rhoRatio",
        "muRatio",
        "finalStep",
        "finalBubbleHeight",
        "finalSpikeDepth",
        "finalMixingWidth",
        "finalPhiMin",
        "finalPhiMax",
        "finalPhiMassDriftPercent",
        "finalRelativeMassErrorPhi",
        "maxVelocityOverall",
        "anyPhiOvershoot",
        "anyDensityBad",
        "anyMassErrorLarge",
        "anyVelocityLarge",
    ]
    with path.open("w", encoding="utf-8", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            rows = run["rows"]
            final = rows[-1]
            maxVelocity = max(
                (
                    rowFloat(row, "max_velocity_magnitude")
                    for row in rows
                    if np.isfinite(rowFloat(row, "max_velocity_magnitude"))
                ),
                default=np.nan,
            )
            writer.writerow(
                {
                    "runId": run["runId"],
                    "reynolds": metadataFloat(run["metadata"], ["REYNOLDS", "RE"]),
                    "weber": metadataFloat(run["metadata"], ["WEBER", "WE"]),
                    "rhoRatio": metadataFloat(run["metadata"], ["RHO_RATIO", "R_RHO"]),
                    "muRatio": metadataFloat(run["metadata"], ["MU_RATIO", "R_MU"]),
                    "finalStep": rowFloat(final, "step"),
                    "finalBubbleHeight": rowFloat(final, "bubble_height"),
                    "finalSpikeDepth": rowFloat(final, "spike_depth"),
                    "finalMixingWidth": rowFloat(final, "mixing_width"),
                    "finalPhiMin": rowFloat(final, "phi_min"),
                    "finalPhiMax": rowFloat(final, "phi_max"),
                    "finalPhiMassDriftPercent": rowFloat(
                        final, "phi_mass_relative_change_percent"
                    ),
                    "finalRelativeMassErrorPhi": rowFloat(
                        final, "relative_mass_error_phi"
                    ),
                    "maxVelocityOverall": maxVelocity,
                    "anyPhiOvershoot": any(
                        rowBool(row, "phi_overshoot") for row in rows
                    ),
                    "anyDensityBad": any(rowBool(row, "density_bad") for row in rows),
                    "anyMassErrorLarge": any(
                        rowBool(row, "mass_error_large") for row in rows
                    ),
                    "anyVelocityLarge": any(
                        rowBool(row, "velocity_large") for row in rows
                    ),
                }
            )


def plotBubbleSpike(outDir, runs, dpi, filename):
    plt.figure(figsize=(7.2, 4.8))
    for run in runs:
        rows = run["rows"]
        xKey = "t_star" if np.isfinite(rowFloat(rows[0], "t_star")) else "step"
        x = np.array([rowFloat(row, xKey) for row in rows], dtype=np.float64)
        bubble = np.array([rowFloat(row, "bubble_height") for row in rows])
        spike = np.array([rowFloat(row, "spike_depth") for row in rows])
        labelBase = run["runId"]
        line = plt.plot(x, bubble, label=f"{labelBase} bubble")[0]
        plt.plot(
            x,
            spike,
            linestyle="--",
            color=line.get_color(),
            label=f"{labelBase} spike",
        )
    plt.xlabel(
        "t*" if runs and np.isfinite(rowFloat(runs[0]["rows"][0], "t_star")) else "step"
    )
    plt.ylabel("displacement")
    plt.title("Rayleigh-Taylor bubble/spike comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outDir / filename, dpi=dpi)
    plt.close()


def plotMixingWidth(outDir, runs, dpi, filename):
    plt.figure(figsize=(7.2, 4.8))
    for run in runs:
        rows = run["rows"]
        xKey = "t_star" if np.isfinite(rowFloat(rows[0], "t_star")) else "step"
        x = np.array([rowFloat(row, xKey) for row in rows], dtype=np.float64)
        mixing = np.array([rowFloat(row, "mixing_width") for row in rows])
        plt.plot(x, mixing, label=run["runId"])
    plt.xlabel(
        "t*" if runs and np.isfinite(rowFloat(runs[0]["rows"][0], "t_star")) else "step"
    )
    plt.ylabel("mixing width")
    plt.title("Rayleigh-Taylor mixing-width comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outDir / filename, dpi=dpi)
    plt.close()


def main():
    args = parseArgs()
    outDir = (
        Path(args.outputDir)
        if args.outputDir is not None
        else Path(args.outputRoot) / args.caseName / "comparison" / "post"
    )
    outDir.mkdir(parents=True, exist_ok=True)

    presetRunIds = {
        "trajectory": trajectoryRunIds,
        "validation": validationRunIds,
        "presentation": presentationRunIds,
        "presentation-we": presentationFixedReRunIds,
        "presentation-re": presentationFixedWeRunIds,
        "stress": stressRunIds,
        "pretty": presentationRunIds,
        "pretty-we": presentationFixedReRunIds,
        "pretty-re": presentationFixedWeRunIds,
    }
    selectedRunIds = args.runIds if args.runIds else presetRunIds[args.preset]

    runs = []
    for runId in selectedRunIds:
        try:
            runs.append(loadRun(args.caseName, args.outputRoot, runId))
        except RuntimeError as error:
            print(f"warning: {error}")

    if not runs:
        raise RuntimeError("no readable RTI summaries found for comparison")

    writeAggregate(outDir, runs)
    namePrefix = {
        "presentation-we": "rayleigh_taylor_presentation_weber_sweep",
        "presentation-re": "rayleigh_taylor_presentation_reynolds_sweep",
        "pretty-we": "rayleigh_taylor_presentation_weber_sweep",
        "pretty-re": "rayleigh_taylor_presentation_reynolds_sweep",
    }.get(args.preset, "rayleigh_taylor_compare")
    plotBubbleSpike(outDir, runs, args.figureDpi, f"{namePrefix}_bubble_spike.png")
    plotMixingWidth(outDir, runs, args.figureDpi, f"{namePrefix}_mixing_width.png")
    print(f"compared {len(runs)} runs")
    print(f"outputs written to: {outDir}")


if __name__ == "__main__":
    main()
