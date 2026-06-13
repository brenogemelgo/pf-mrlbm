outputRoot = "output"
defaultSummaryRunId = None
defaultRunIdPrefix = None
defaultSurfaceForce = "csf"
defaultNsteps = 37000
defaultStamp = 37000
arch = "sm_86"
includeWidthSweepByDefault = False

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


caseName = "static_droplet"
binaryPath = Path("/tmp/mrlbm_static_droplet_sweep")
strictMaskName = "radial_r_lt_R_minus_3W_gt_R_plus_3W"


def runCommand(command, cwd):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def readKeyValueCsv(path):
    values = {}
    with path.open("r", encoding="utf-8", newline="") as inputFile:
        reader = csv.DictReader(inputFile)
        for row in reader:
            values[row["metric"]] = row["value"]
    return values


def readRowsCsv(path):
    with path.open("r", encoding="utf-8", newline="") as inputFile:
        return list(csv.DictReader(inputFile))


def asFloat(values, key, defaultValue=np.nan):
    try:
        return float(values[key])
    except (KeyError, TypeError, ValueError):
        return defaultValue


def compileCase(case, nsteps, stamp, repo, surfaceForce):
    command = [
        "nvcc",
        "-std=c++20",
        "-O3",
        "--restrict",
        "--expt-relaxed-constexpr",
        "--fmad=true",
        "--extra-device-vectorization",
        "--extended-lambda",
        f"-arch={arch}",
        "-lineinfo",
        "-Xptxas",
        "-v",
        "-DCASE_STATIC_DROPLET",
        f"-DSURFACE_FORCE_{surfaceForce.upper()}",
        f"-DSTATIC_DROPLET_SIGMA={case['sigma']:.17g}",
        f"-DSTATIC_DROPLET_R_INIT={case['radius']:.17g}",
        f"-DSTATIC_DROPLET_WIDTH={case['width']:.17g}",
        f"-DSTATIC_DROPLET_NSTEPS={nsteps}",
        f"-DSTATIC_DROPLET_STAMP={stamp}",
        "src/main.cu",
        "-o",
        str(binaryPath),
    ]
    runCommand(command, repo)


def runCase(case, repo):
    runCommand([str(binaryPath), "--runId", case["runId"]], repo)


def postCase(case, repo):
    runCommand(["python3", "postStaticDroplet.py", "--runId", case["runId"]], repo)


def collectCase(case, repo):
    runDir = repo / outputRoot / caseName / case["runId"]
    postDir = runDir / "post"
    summary = readKeyValueCsv(postDir / "static_droplet_summary.csv")
    masks = readRowsCsv(postDir / "static_droplet_pressure_masks.csv")
    strictRows = [row for row in masks if row["mask"] == strictMaskName]
    if not strictRows:
        raise RuntimeError(f"Missing strict radial mask row for {case['runId']}")
    strict = strictRows[0]

    sigma = case["sigma"]
    radius = case["radius"]
    deltaPTheory = 2.0 * sigma / radius
    deltaPStrict = float(strict["delta_p"])
    deltaPDefault = asFloat(summary, "delta_p_recovered")
    sigmaRecStrict = float(strict["sigma_recovered_r0"])
    sigmaRecDefault = asFloat(summary, "sigma_recovered_r0_lu")

    return {
        "runId": case["runId"],
        "sweep": case["sweep"],
        "SIGMA": sigma,
        "R_INIT": radius,
        "WIDTH": case["width"],
        "delta_p": deltaPStrict,
        "delta_p_default": deltaPDefault,
        "delta_p_theory": deltaPTheory,
        "sigma_rec_r0": sigmaRecStrict,
        "sigma_rec_r0_default": sigmaRecDefault,
        "relative_error": (sigmaRecStrict - sigma) / sigma,
        "relative_error_default": (sigmaRecDefault - sigma) / sigma,
        "max_velocity": asFloat(summary, "max_velocity"),
        "mass_change_phi": asFloat(summary, "mass_change_phi"),
        "radius_effective_lu": asFloat(summary, "radius_effective_lu"),
        "radius_phi05": asFloat(summary, "radius_phi05_lu"),
    }


def writeRows(path, rows):
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fitWithIntercept(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    yFit = a * x + b
    ssResidual = float(np.sum((y - yFit) * (y - yFit)))
    ssTotal = float(np.sum((y - np.mean(y)) * (y - np.mean(y))))
    r2 = np.nan if ssTotal <= 0.0 else 1.0 - ssResidual / ssTotal
    return float(a), float(b), r2


def plotFit(path, x, y, xlabel, ylabel):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a, b, _ = fitWithIntercept(x, y)
    xLine = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    plt.figure(figsize=(6.2, 4.6))
    plt.scatter(x, y, s=52, label="strict radial mask")
    plt.plot(xLine, a * xLine + b, "k--", linewidth=1.0, label="linear fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plotWidth(path, rows):
    widths = np.array([row["WIDTH"] for row in rows], dtype=np.float64)
    errors = np.array([100.0 * row["relative_error"] for row in rows], dtype=np.float64)
    plt.figure(figsize=(6.2, 4.2))
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.plot(widths, errors, "o-", label="strict radial mask")
    plt.xlabel("WIDTH")
    plt.ylabel("sigma error (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def buildCases(includeWidth, requestedSweep, runIdPrefix):
    cases = [
        {"runId": "sigma005_R24_W5", "sweep": "sigma", "sigma": 0.005, "radius": 24.0, "width": 5.0},
        {"runId": "sigma010_R24_W5", "sweep": "sigma,radius,width", "sigma": 0.01, "radius": 24.0, "width": 5.0},
        {"runId": "sigma020_R24_W5", "sweep": "sigma", "sigma": 0.02, "radius": 24.0, "width": 5.0},
        {"runId": "sigma010_R16_W5", "sweep": "radius", "sigma": 0.01, "radius": 16.0, "width": 5.0},
        {"runId": "sigma010_R32_W5", "sweep": "radius", "sigma": 0.01, "radius": 32.0, "width": 5.0},
    ]
    if includeWidth:
        cases.extend(
            [
                {"runId": "sigma010_R24_W4", "sweep": "width", "sigma": 0.01, "radius": 24.0, "width": 4.0},
                {"runId": "sigma010_R24_W6", "sweep": "width", "sigma": 0.01, "radius": 24.0, "width": 6.0},
                {"runId": "sigma010_R24_W8", "sweep": "width", "sigma": 0.01, "radius": 24.0, "width": 8.0},
            ]
        )
    if requestedSweep != "all":
        cases = [case for case in cases if requestedSweep in case["sweep"].split(",")]
    if runIdPrefix:
        cases = [{**case, "runId": runIdPrefix + case["runId"]} for case in cases]
    return cases


def fitRows(rows, requestedSweep, includeWidth):
    sigmaRows = []
    radiusRows = []
    widthRows = []
    fits = []

    if requestedSweep in ("all", "sigma"):
        sigmaRows = [row for row in rows if row["R_INIT"] == 24.0 and row["WIDTH"] == 5.0]
        if len(sigmaRows) >= 2:
            sigmaA, sigmaB, sigmaR2 = fitWithIntercept(
                [row["SIGMA"] for row in sigmaRows], [row["delta_p"] for row in sigmaRows]
            )
            fits.append(
                {
                    "sweep": "sigma",
                    "slope": sigmaA,
                    "intercept": sigmaB,
                    "theory_slope": 2.0 / 24.0,
                    "slope_ratio": sigmaA / (2.0 / 24.0),
                    "r_squared": sigmaR2,
                }
            )

    if requestedSweep in ("all", "radius"):
        radiusRows = [row for row in rows if row["SIGMA"] == 0.01 and row["WIDTH"] == 5.0]
        if len(radiusRows) >= 2:
            radiusA, radiusB, radiusR2 = fitWithIntercept(
                [1.0 / row["R_INIT"] for row in radiusRows], [row["delta_p"] for row in radiusRows]
            )
            fits.append(
                {
                    "sweep": "radius",
                    "slope": radiusA,
                    "intercept": radiusB,
                    "theory_slope": 0.02,
                    "slope_ratio": radiusA / 0.02,
                    "r_squared": radiusR2,
                }
            )

    if includeWidth and requestedSweep in ("all", "width"):
        widthRows = [row for row in rows if row["SIGMA"] == 0.01 and row["R_INIT"] == 24.0]

    return {
        "sigmaRows": sigmaRows,
        "radiusRows": radiusRows,
        "widthRows": widthRows,
        "fits": fits,
    }


def main():
    parser = argparse.ArgumentParser(description="Run static droplet scaling sweeps")
    parser.add_argument("--nsteps", type=int, default=defaultNsteps)
    parser.add_argument("--stamp", type=int, default=defaultStamp)
    parser.add_argument("--include-width", action="store_true", default=includeWidthSweepByDefault)
    parser.add_argument("--sweeps", choices=("all", "sigma", "radius", "width"), default="all")
    parser.add_argument("--surface-force", choices=("csf", "cpf"), default=defaultSurfaceForce)
    parser.add_argument("--summary-run-id", default=defaultSummaryRunId)
    parser.add_argument("--run-id-prefix", default=defaultRunIdPrefix)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    includeWidth = args.include_width or args.sweeps == "width"
    runIdPrefix = args.run_id_prefix
    if runIdPrefix is None:
        runIdPrefix = f"{args.surface_force}_"
    summaryRunId = args.summary_run_id
    if summaryRunId is None:
        summaryRunId = f"sweep_summary_{args.surface_force}"

    cases = buildCases(includeWidth, args.sweeps, runIdPrefix)
    allRows = []

    for case in cases:
        runDir = repo / outputRoot / caseName / case["runId"]
        if args.reuse and (runDir / "post" / "static_droplet_summary.csv").exists():
            print(f"reusing existing run {case['runId']}", flush=True)
        else:
            if runDir.exists():
                shutil.rmtree(runDir)
            compileCase(case, args.nsteps, args.stamp, repo, args.surface_force)
            runCase(case, repo)
            postCase(case, repo)
        allRows.append(collectCase(case, repo))

    outDir = repo / outputRoot / caseName / summaryRunId / "post"
    outDir.mkdir(parents=True, exist_ok=True)
    fitData = fitRows(allRows, args.sweeps, includeWidth)

    if fitData["sigmaRows"]:
        writeRows(outDir / "static_droplet_sigma_sweep_summary.csv", fitData["sigmaRows"])
    if fitData["radiusRows"]:
        writeRows(outDir / "static_droplet_radius_sweep_summary.csv", fitData["radiusRows"])
    if fitData["widthRows"]:
        writeRows(outDir / "static_droplet_width_sweep_summary.csv", fitData["widthRows"])
    if fitData["fits"]:
        writeRows(outDir / "static_droplet_sweep_fit_summary.csv", fitData["fits"])

    if len(fitData["sigmaRows"]) >= 2:
        plotFit(
            outDir / "delta_p_vs_sigma.png",
            [row["SIGMA"] for row in fitData["sigmaRows"]],
            [row["delta_p"] for row in fitData["sigmaRows"]],
            "SIGMA",
            "delta_p",
        )
    if len(fitData["radiusRows"]) >= 2:
        plotFit(
            outDir / "delta_p_vs_inv_radius.png",
            [1.0 / row["R_INIT"] for row in fitData["radiusRows"]],
            [row["delta_p"] for row in fitData["radiusRows"]],
            "1 / R_INIT",
            "delta_p",
        )
    if len(fitData["widthRows"]) >= 2:
        plotWidth(outDir / "sigma_error_vs_width.png", fitData["widthRows"])

    print(f"summary outputs written to: {outDir}", flush=True)
    for row in fitData["fits"]:
        print(
            f"{row['sweep']}: slope={row['slope']:.17g}, "
            f"intercept={row['intercept']:.17g}, "
            f"slope_ratio={row['slope_ratio']:.17g}, "
            f"r_squared={row['r_squared']:.17g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
