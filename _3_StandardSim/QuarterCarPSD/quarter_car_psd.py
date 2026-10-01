"""Two-DOF quarter-car ISO 8608 wheelrate/CPLV screening study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

from _0_Utils.dyn_py.parameters import _combine_mass_properties, _mass_components
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root

G = 9.80665
# ISO 8608:2016 Table C.2, n0=0.1 cycles/m, w=2.
# A and H representatives are simulation values for open-ended classes.
ISO_G0 = {letter: 16e-6 * 4**i for i, letter in enumerate("ABCDEFGH")}


@dataclass(frozen=True)
class Corner:
    axle: str
    sprung_mass_kg: float
    unsprung_mass_kg: float
    tire_stiffness_n_per_m: float
    tire_damping_ns_per_m: float
    mean_load_n: float


def load_corners(path: Path) -> tuple[list[Corner], dict]:
    data = load_yaml(path)
    # Reuse the exact BobSim reduced-model mass ledger, including mirrored links.
    mass, cg, _ = _combine_mass_properties(_mass_components(data))
    xf = float(data["front"]["suspension"]["wheel_center_m"][0])
    xr = float(data["rear"]["suspension"]["wheel_center_m"][0])
    fraction = (cg[0] - xr) / (xf - xr)
    if not 0 < fraction < 1:
        raise ValueError("CG must lie between axles")
    corners = []
    for axle, share in (("front", fraction), ("rear", 1 - fraction)):
        mu = float(data[axle]["masses"]["unsprung"]["mass_kg"])
        ms = mass * share / 2 - mu
        tire = data[axle]["tire"]
        corner = Corner(
            axle,
            ms,
            mu,
            float(tire["vertical_stiffness_n_per_m"]),
            float(tire["vertical_damping_n_s_per_m"]),
            mass * share * G / 2,
        )
        if min(ms, mu, corner.tire_stiffness_n_per_m) <= 0 or corner.tire_damping_ns_per_m < 0:
            raise ValueError("Invalid corner mass or tire properties")
        corners.append(corner)
    return corners, {"total_mass_kg": mass, "front_weight_fraction": fraction, "cg_m": list(cg), "wheelbase_m": xf - xr}


def wheel_damping(corner: Corner, wheelrate: float, zeta: float) -> float:
    """Nominal sprung-mass convention; not an exact coupled-mode damping ratio."""
    if wheelrate <= 0 or zeta <= 0:
        raise ValueError("Wheelrate and damping ratio must be positive")
    return 2 * zeta * math.sqrt(wheelrate * corner.sprung_mass_kg)


def transfer(corner: Corner, wheelrate: float, damping: float, frequency):
    """Road-displacement to dynamic tire-force and suspension-travel FRFs."""
    omega = 2 * np.pi * np.asarray(frequency)
    q = wheelrate + 1j * omega * damping
    t = corner.tire_stiffness_n_per_m + 1j * omega * corner.tire_damping_ns_per_m
    a = q - corner.sprung_mass_kg * omega**2
    d = q + t - corner.unsprung_mass_kg * omega**2
    determinant = a * d - q**2
    zs = q * t / determinant
    zu = a * t / determinant
    return t * (1 - zu), zs - zu


def rms_response(corner, wheelrate, damping, speed, g0, n_min, n_max, points):
    if speed <= 0 or g0 <= 0 or not 0 < n_min < n_max or points < 33:
        raise ValueError("Invalid speed, PSD strength or integration grid")
    n = np.geomspace(n_min, n_max, points)
    # f=v*n and S_z(f)=G_z(f/v)/v; integrate equivalently in spatial frequency.
    force, travel = transfer(corner, wheelrate, damping, speed * n)
    psd = g0 * (n / 0.1) ** -2
    return (float(np.sqrt(simpson(abs(force) ** 2 * psd, x=n))), float(np.sqrt(simpson(abs(travel) ** 2 * psd, x=n))))


def state_matrices(corner, wheelrate, damping):
    """Independent state-space representation, ordered zs, zu, vs, vu."""
    ms, mu = corner.sprung_mass_kg, corner.unsprung_mass_kg
    kt, ct = corner.tire_stiffness_n_per_m, corner.tire_damping_ns_per_m
    a = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-wheelrate / ms, wheelrate / ms, -damping / ms, damping / ms],
            [wheelrate / mu, -(wheelrate + kt) / mu, damping / mu, -(damping + ct) / mu],
        ]
    )
    b = np.array([[0, 0], [0, 0], [0, 0], [kt / mu, ct / mu]])
    return a, b


def run_sweep(config, corners):
    rates = np.linspace(config["wheelrate_min_n_per_m"], config["wheelrate_max_n_per_m"], config["wheelrate_points"])
    if len(rates) < 2 or rates[-1] <= rates[0]:
        raise ValueError("Need an increasing wheelrate sweep with at least two points")
    rows = []
    for corner in corners:
        for rate in rates:
            damping = wheel_damping(corner, rate, config["damping_ratio"])
            args = (
                corner,
                rate,
                damping,
                config["speed_mps"],
                ISO_G0["A"],
                config["spatial_frequency_min_cycles_per_m"],
                config["spatial_frequency_max_cycles_per_m"],
            )
            rms, travel = rms_response(*args, config["frequency_points"])
            coarse, _ = rms_response(*args, (config["frequency_points"] + 1) // 2)
            error = abs(rms - coarse) / rms
            max_pole = float(np.max(np.linalg.eigvals(state_matrices(corner, rate, damping)[0]).real))
            if error > 1e-4 or max_pole >= 0:
                raise RuntimeError("Integration convergence or stability check failed")
            for cls in config["iso_classes"]:
                scale = math.sqrt(ISO_G0[cls] / ISO_G0["A"])
                sigma = rms * scale
                probability = float(ndtr(-corner.mean_load_n / sigma))
                rows.append(
                    {
                        "axle": corner.axle,
                        "iso_class": cls,
                        "wheelrate_n_per_m": float(rate),
                        "wheelrate_lbf_per_in": float(rate / 175.126835),
                        "speed_mps": config["speed_mps"],
                        "nominal_zeta": config["damping_ratio"],
                        "wheel_damping_ns_per_m": damping,
                        "g0_m3": ISO_G0[cls],
                        "mean_load_n": corner.mean_load_n,
                        "rms_load_n": sigma,
                        "cplv_percent": 100 * sigma / corner.mean_load_n,
                        "rms_suspension_travel_mm": 1000 * travel * scale,
                        "gaussian_unloading_probability": probability,
                        "passes_unloading_screen": probability < config["contact_loss_probability_limit"],
                        "integration_relative_error": error,
                        "max_pole_real_per_s": max_pole,
                    }
                )
    return rows


def make_plots(config, corners, rows, output):
    groups = [("a_h", config["iso_classes"])]
    groups.extend((cls.lower(), [cls]) for cls in config["iso_classes"])
    ab = [cls for cls in ("A", "B") if cls in config["iso_classes"]]
    if len(ab) == 2:
        groups.append(("a_b", ab))
    for suffix, classes in groups:
        selected_rows = [r for r in rows if r["iso_class"] in classes]
        _make_plot(config, corners, selected_rows, output, classes, suffix)


def _make_plot(config, corners, rows, output, classes, suffix):
    series, subplots = {}, []
    for corner in corners:
        axle = corner.axle
        series[f"rate_{axle}"] = {}
        series[f"cplv_{axle}"] = {}
        for cls in classes:
            selected = [r for r in rows if r["axle"] == axle and r["iso_class"] == cls]
            series[f"rate_{axle}"][cls] = [r["wheelrate_lbf_per_in"] for r in selected]
            series[f"cplv_{axle}"][cls] = [r["cplv_percent"] for r in selected]
        subplots.append(
            {
                "title": f"{axle.title()} corner | contact-load variation",
                "x": {"key": f"rate_{axle}", "label": "Wheelrate (lb/in)"},
                "y": {"key": f"cplv_{axle}", "label": "CPLV (%)"},
                "yscale": "linear",
                "linewidth": 2.2,
            }
        )
        mean_rows = [r for r in rows if r["axle"] == axle and r["iso_class"] == classes[0]]
        series[f"mean_rate_{axle}"] = [r["wheelrate_lbf_per_in"] for r in mean_rows]
        series[f"mean_load_{axle}"] = [r["mean_load_n"] / 4.4482216152605 for r in mean_rows]
        subplots.append(
            {
                "title": f"{axle.title()} corner | mean contact load",
                "x": {"key": f"mean_rate_{axle}", "label": "Wheelrate (lb/in)"},
                "y": {"key": f"mean_load_{axle}", "label": "Mean contact load (lbf)"},
                "yscale": "linear",
                "linewidth": 2.2,
                "color": "#34495e",
                "label": "Static mean (all shown ISO classes)",
            }
        )
    vehicle_label = Path(config["vehicle"]).stem
    plots = {
        "plots": {
            f"cplv_vs_wheelrate_iso_{suffix}": {
                "layout": "quad",
                "subplots": subplots,
                "title": f"{vehicle_label} quarter-car | ISO {', '.join(classes)} | {config['speed_mps']:g} m/s | "
                f"nominal damping ratio {config['damping_ratio']:.2f}",
            }
        }
    }
    for name, fig in PlotEngine(plots).render({"series": series}):
        fig.set_size_inches(14, 10.5)
        fig.subplots_adjust(left=0.075, right=0.975, top=0.88, bottom=0.20, wspace=0.25, hspace=0.40)
        for ax, mean_ax, corner in zip(fig.axes[::2], fig.axes[1::2], corners):
            ax.set_ylim(0, max(r["cplv_percent"] for r in rows) * 1.25)
            colors = dict(
                zip(
                    "ABCDEFGH", ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
                )
            )
            for line, cls in zip(list(ax.lines), sorted(classes)):
                line.set_color(colors[cls])
                selected = [r for r in rows if r["axle"] == corner.axle and r["iso_class"] == cls]
                bad = np.array([not r["passes_unloading_screen"] for r in selected])
                x, y = line.get_xdata(), line.get_ydata()
                line.set_linestyle("--" if np.any(bad) else "-")
                # Overlay only valid segments; preserve gaps rather than joining across them.
                ax.plot(x, np.where(bad, np.nan, y), color=line.get_color(), lw=2.2)
            ax.legend(title="ISO class", ncol=2 if len(classes) > 4 else 1, fontsize=9, loc="upper left")
            ax.grid(True, which="minor", alpha=0.15)
            mean_ax.set_ylim(0, max(c.mean_load_n for c in corners) / 4.4482216152605 * 1.25)
            mean_ax.legend(loc="upper left", fontsize=9)
            mean_ax.text(
                0.98,
                corner.mean_load_n / (max(c.mean_load_n for c in corners) * 1.25) + 0.04,
                f"{corner.mean_load_n / 4.4482216152605:.2f} lbf ({corner.mean_load_n:.2f} N)",
                transform=mean_ax.transAxes,
                ha="right",
                fontsize=11,
                color="#34495e",
            )
        fig.text(
            0.5,
            0.14,
            "CPLV = 100 × RMS(load fluctuation) / mean load. ISO PSD: w=2, "
            f"{config['spatial_frequency_min_cycles_per_m']:g}–"
            f"{config['spatial_frequency_max_cycles_per_m']:g} cycles/m. "
            f"Tire properties from {vehicle_label}.yml.",
            ha="center",
            fontsize=10,
        )
        limit = 100 * config["contact_loss_probability_limit"]
        fig.text(
            0.5,
            0.105,
            f"Solid: Gaussian unloading probability <{limit:g}%. "
            "Dashed: linear extrapolation beyond this screen. "
            "Screen passing does not verify suspension travel or grip.",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.5,
            0.07,
            "Wheel damping recalculated at every rate (SI units): "
            f"c = 2 × {config['damping_ratio']:.2f} × √(wheelrate × sprung corner mass). "
            "This is a nominal sprung-mass ratio, not an exact coupled-mode ratio.",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.5,
            0.035,
            "Mean load is fixed by static weight. No anti geometry, braking/drive force, or aero response is modeled; "
            "these plots do not predict anti-to-grip effects.",
            ha="center",
            fontsize=10,
        )
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="_3_StandardSim/QuarterCarPSD/quarter_car_psd_config.yml")
    args = parser.parse_args()
    root = repo_root()
    config_path = root / args.config
    config = load_yaml(config_path)
    vehicle_path = root / config["vehicle"]
    corners, mass_summary = load_corners(vehicle_path)
    rows = run_sweep(config, corners)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "wheelrate_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plots(config, corners, rows, output)
    source_paths = [vehicle_path, config_path, Path(__file__), root / "_0_Utils/dyn_py/parameters.py"]
    summary = {
        "config": config,
        "mass_projection": mass_summary,
        "corners": [asdict(c) for c in corners],
        "row_count": len(rows),
        "maximum_integration_relative_error": max(r["integration_relative_error"] for r in rows),
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
        "iso_source": "https://www.iso.org/standard/71202.html",
        "scope": "Linear 2DOF frequency-domain QCM; static gravity mean load; no aero, roll, pitch, "
        "bump stops, travel constraints, tire enveloping or unilateral contact. "
        "Not a grip or full-vehicle optimum prediction.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_vehicle.yml").write_bytes(vehicle_path.read_bytes())
    (output / "input_config.yml").write_bytes(config_path.read_bytes())
    (output / "executed_runner.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
