#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from postCommon import writeDatFile


RUN_DIR = Path(".")
OUTPUT_ROOT = RUN_DIR / "output"
CASE_NAME = "static_droplet"
RUN_ID = "000"
RUN_OUTPUT_DIR = OUTPUT_ROOT / CASE_NAME / RUN_ID
POST_DIR = RUN_OUTPUT_DIR / "post"
DIAGNOSTIC_CSV = POST_DIR / "diagnostics.csv"


def rel_error(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)

    out = np.full_like(value, np.nan, dtype=float)
    mask = np.abs(reference) > 0.0
    out[mask] = (value[mask] - reference[mask]) / reference[mask]
    return out


def percent_error(value, reference):
    return 100.0 * rel_error(value, reference)


def alias_columns(df):
    if "volume_phi" not in df.columns and "mass" in df.columns:
        df["volume_phi"] = df["mass"]

    return df


def require(df, names):
    return all(name in df.columns for name in names)


def save_plot(outdir, name):
    path = outdir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_series(df, outdir, ycols, title, ylabel, filename, logy=False):
    if not require(df, ["step", *ycols]):
        return

    writeDatFile(
        outdir / f"{filename}.dat",
        ["step", *ycols],
        [df["step"], *[df[col] for col in ycols]],
    )

    plt.figure(figsize=(8.0, 4.5))

    for col in ycols:
        plt.plot(df["step"], df[col], label=col)

    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)

    if logy:
        plt.yscale("log")

    if len(ycols) > 1:
        plt.legend()

    plt.grid(True, which="both", alpha=0.3)
    save_plot(outdir, filename)


def plot_horizontal_reference(
    df, outdir, ycol, refcol, title, ylabel, filename, logy=False
):
    if not require(df, ["step", ycol, refcol]):
        return

    writeDatFile(
        outdir / f"{filename}.dat",
        ["step", ycol, refcol],
        [df["step"], df[ycol], df[refcol]],
    )

    plt.figure(figsize=(8.0, 4.5))

    plt.plot(df["step"], df[ycol], label=ycol)
    plt.plot(df["step"], df[refcol], linestyle="--", label=refcol)

    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)

    if logy:
        plt.yscale("log")

    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save_plot(outdir, filename)


def plot_derived(df, outdir):
    step = df["step"]

    if require(df, ["volume_phi"]):
        volume0 = float(df["volume_phi"].iloc[0])
        df["volume_phi_rel_drift"] = rel_error(df["volume_phi"], volume0)

        plot_series(
            df,
            outdir,
            ["volume_phi"],
            "Phase-field volume",
            "sum(phi)",
            "volume_phi",
        )

        plot_series(
            df,
            outdir,
            ["volume_phi_rel_drift"],
            "Relative phase-field volume drift",
            "(volume - volume_0) / volume_0",
            "volume_phi_relative_drift",
        )

    if require(df, ["radius_eff"]):
        radius0 = float(df["radius_eff"].iloc[0])
        df["radius_eff_rel_drift"] = rel_error(df["radius_eff"], radius0)

        plot_series(
            df,
            outdir,
            ["radius_eff"],
            "Effective droplet radius",
            "R_eff",
            "radius_eff",
        )

        plot_series(
            df,
            outdir,
            ["radius_eff_rel_drift"],
            "Relative effective-radius drift",
            "(R_eff - R_eff_0) / R_eff_0",
            "radius_eff_relative_drift",
        )

    if require(df, ["com_x", "com_y", "com_z"]):
        x0 = float(df["com_x"].iloc[0])
        y0 = float(df["com_y"].iloc[0])
        z0 = float(df["com_z"].iloc[0])

        df["com_dx"] = df["com_x"] - x0
        df["com_dy"] = df["com_y"] - y0
        df["com_dz"] = df["com_z"] - z0
        df["com_drift"] = np.sqrt(
            df["com_dx"] ** 2 + df["com_dy"] ** 2 + df["com_dz"] ** 2
        )

        plot_series(
            df,
            outdir,
            ["com_dx", "com_dy", "com_dz"],
            "Center-of-mass displacement",
            "lattice units",
            "center_of_mass_displacement_components",
        )

        plot_series(
            df,
            outdir,
            ["com_drift"],
            "Center-of-mass drift magnitude",
            "lattice units",
            "center_of_mass_drift",
        )

    if require(df, ["delta_p", "expected_delta_p_r0"]):
        df["delta_p_error_percent_r0"] = percent_error(
            df["delta_p"], df["expected_delta_p_r0"]
        )

        plot_horizontal_reference(
            df,
            outdir,
            "delta_p",
            "expected_delta_p_r0",
            "Laplace pressure recovery using initial radius",
            "Delta p",
            "delta_p_vs_expected_r0",
        )

        plot_series(
            df,
            outdir,
            ["delta_p_error_percent_r0"],
            "Laplace pressure percent error using initial radius",
            "percent",
            "delta_p_error_percent_r0",
        )

    if require(df, ["delta_p", "expected_delta_p_reff"]):
        df["delta_p_error_percent_reff"] = percent_error(
            df["delta_p"], df["expected_delta_p_reff"]
        )

        plot_horizontal_reference(
            df,
            outdir,
            "delta_p",
            "expected_delta_p_reff",
            "Laplace pressure recovery using effective radius",
            "Delta p",
            "delta_p_vs_expected_reff",
        )

        plot_series(
            df,
            outdir,
            ["delta_p_error_percent_reff"],
            "Laplace pressure percent error using effective radius",
            "percent",
            "delta_p_error_percent_reff",
        )

    if require(df, ["sigma_recovered_r0", "sigma_recovered_reff"]):
        plot_series(
            df,
            outdir,
            ["sigma_recovered_r0", "sigma_recovered_reff"],
            "Recovered surface tension",
            "sigma",
            "sigma_recovered",
        )

    if require(df, ["sigma_recovery_ratio_r0", "sigma_recovery_ratio_reff"]):
        plot_series(
            df,
            outdir,
            ["sigma_recovery_ratio_r0", "sigma_recovery_ratio_reff"],
            "Surface-tension recovery ratio",
            "sigma_recovered / sigma_expected",
            "sigma_recovery_ratio",
        )

        df["sigma_recovery_percent_error_r0"] = 100.0 * (
            df["sigma_recovery_ratio_r0"] - 1.0
        )
        df["sigma_recovery_percent_error_reff"] = 100.0 * (
            df["sigma_recovery_ratio_reff"] - 1.0
        )

        plot_series(
            df,
            outdir,
            ["sigma_recovery_percent_error_r0", "sigma_recovery_percent_error_reff"],
            "Surface-tension recovery percent error",
            "percent",
            "sigma_recovery_percent_error",
        )

    if require(df, ["rho_ratio_recovered", "rho_ratio_expected"]):
        plot_horizontal_reference(
            df,
            outdir,
            "rho_ratio_recovered",
            "rho_ratio_expected",
            "Density-ratio recovery",
            "rho_l / rho_g",
            "rho_ratio_recovery",
            logy=True,
        )

        df["rho_ratio_error_percent"] = percent_error(
            df["rho_ratio_recovered"], df["rho_ratio_expected"]
        )

        plot_series(
            df,
            outdir,
            ["rho_ratio_error_percent"],
            "Density-ratio percent error",
            "percent",
            "rho_ratio_error_percent",
        )

    if require(df, ["mu_ratio_recovered", "mu_ratio_expected"]):
        plot_horizontal_reference(
            df,
            outdir,
            "mu_ratio_recovered",
            "mu_ratio_expected",
            "Dynamic-viscosity-ratio recovery",
            "mu_l / mu_g",
            "mu_ratio_recovery",
            logy=True,
        )

        df["mu_ratio_error_percent"] = percent_error(
            df["mu_ratio_recovered"], df["mu_ratio_expected"]
        )

        plot_series(
            df,
            outdir,
            ["mu_ratio_error_percent"],
            "Dynamic-viscosity-ratio percent error",
            "percent",
            "mu_ratio_error_percent",
        )

    derived_path = outdir / "diagnostics_with_derived_metrics.csv"
    df.to_csv(derived_path, index=False)


def print_summary(df):
    last = df.iloc[-1]

    print("\n=== Static droplet diagnostics summary ===")
    print(f"steps: {df['step'].iloc[0]} -> {df['step'].iloc[-1]}")

    if "volume_phi" in df.columns:
        volume0 = float(df["volume_phi"].iloc[0])
        volumef = float(df["volume_phi"].iloc[-1])
        drift = rel_error([volumef], [volume0])[0]
        print(f"volume_phi: {volume0:.8e} -> {volumef:.8e}  rel_drift={drift:.8e}")

    if "radius_eff" in df.columns:
        radius0 = float(df["radius_eff"].iloc[0])
        radiusf = float(df["radius_eff"].iloc[-1])
        drift = rel_error([radiusf], [radius0])[0]
        print(f"radius_eff: {radius0:.8e} -> {radiusf:.8e}  rel_drift={drift:.8e}")

    if "max_u" in df.columns:
        print(
            f"max_u: {float(df['max_u'].min()):.8e} -> {float(df['max_u'].max()):.8e}  min->max"
        )

    if "delta_p" in df.columns:
        print(f"delta_p final: {float(last['delta_p']):.8e}")

    if "expected_delta_p_reff" in df.columns:
        print(
            f"expected_delta_p_reff final: {float(last['expected_delta_p_reff']):.8e}"
        )

    if "sigma_recovery_ratio_reff" in df.columns:
        print(
            f"sigma_recovery_ratio_reff final: {float(last['sigma_recovery_ratio_reff']):.8e}"
        )

    if "rho_ratio_recovered" in df.columns:
        print(f"rho_ratio_recovered final: {float(last['rho_ratio_recovered']):.8e}")

    if "mu_ratio_recovered" in df.columns:
        print(f"mu_ratio_recovered final: {float(last['mu_ratio_recovered']):.8e}")

    print()


def main():
    if not DIAGNOSTIC_CSV.exists():
        raise RuntimeError(f"Missing static-droplet diagnostics CSV: {DIAGNOSTIC_CSV}")

    POST_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DIAGNOSTIC_CSV)
    df = alias_columns(df)

    if "step" not in df.columns:
        raise RuntimeError("diagnostics file must contain a 'step' column")

    df = df.sort_values("step").reset_index(drop=True)

    print_summary(df)

    plot_series(
        df,
        POST_DIR,
        ["phi_min", "phi_max"],
        "Phase-field extrema",
        "phi",
        "phi_min_max",
    )

    plot_series(
        df,
        POST_DIR,
        ["max_u"],
        "Maximum velocity magnitude",
        "max |u|",
        "max_u",
        logy=True,
    )

    plot_series(
        df,
        POST_DIR,
        ["p_inside_avg", "p_outside_avg"],
        "Average pressure inside and outside droplet",
        "pressure",
        "pressure_inside_outside",
    )

    plot_series(
        df,
        POST_DIR,
        ["delta_p"],
        "Pressure jump",
        "Delta p",
        "delta_p",
    )

    plot_series(
        df,
        POST_DIR,
        ["rho_inside_avg", "rho_outside_avg"],
        "Average density inside and outside droplet",
        "rho",
        "rho_inside_outside",
        logy=True,
    )

    plot_series(
        df,
        POST_DIR,
        ["mu_inside_avg", "mu_outside_avg"],
        "Average dynamic viscosity inside and outside droplet",
        "mu",
        "mu_inside_outside",
        logy=True,
    )

    plot_derived(df, POST_DIR)

    print(f"plots and plot data written to: {POST_DIR}")
    print(
        f"derived CSV written to: {POST_DIR / 'diagnostics_with_derived_metrics.csv'}"
    )


if __name__ == "__main__":
    main()
