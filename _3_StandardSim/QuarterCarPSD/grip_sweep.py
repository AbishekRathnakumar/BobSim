"""Finite-record wheelrate versus quasi-steady lateral capacity, strict tire domain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from _0_Utils.mf52_lateral import LateralForce
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, parse_tir, repo_root, tire_template_name, tire_templates_root
from _3_StandardSim.QuarterCarPSD.quarter_car_psd import ISO_G0, load_corners, rms_response, transfer, wheel_damping

LBF = 4.4482216152605
COLORS = dict(zip("ABCDEFGH", ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")))


def road_spectrum(qcm, duration, sample_rate, seed):
    """Periodic random-phase ISO A road; exact integrated PSD energy per bin.

    Fourier coefficients use z(t)=2 Re sum(c_k exp(i 2 pi f_k t)). This gives
    variance=2 sum(abs(c_k)^2). Higher ISO classes use amplitude scaling.
    """
    count = int(round(duration * sample_rate))
    if not np.isclose(count, duration * sample_rate) or count % 2:
        raise ValueError("Need an even integer sample count")
    f = np.fft.rfftfreq(count, 1 / sample_rate)
    fmin = qcm["speed_mps"] * qcm["spatial_frequency_min_cycles_per_m"]
    fmax = qcm["speed_mps"] * qcm["spatial_frequency_max_cycles_per_m"]
    if fmax >= sample_rate / 2 or fmin <= 1 / duration:
        raise ValueError("Road band must resolve above DC and below Nyquist")
    lo = np.maximum(f - 0.5 / duration, fmin)
    hi = np.minimum(f + 0.5 / duration, fmax)
    mask = hi > lo
    variance = np.zeros_like(f)
    variance[mask] = ISO_G0["A"] * 0.1**2 * qcm["speed_mps"] * (1 / lo[mask] - 1 / hi[mask])
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(f))
    coefficient = np.sqrt(variance / 2) * np.exp(1j * phase)
    return f, coefficient, count


def response_record(corner, rate, damping, f, coefficient, count, sample_rate, sample_multiplier=1):
    force, _ = transfer(corner, rate, damping, f)
    output = coefficient * force
    count *= sample_multiplier
    sample_rate *= sample_multiplier
    load = np.fft.irfft(output, n=count) * count
    # Rigorous bound on deviation from piecewise linear interpolation between
    # samples: sup|F''| dt^2/8. Also covers the periodic closing interval.
    omega = 2 * np.pi * f
    curvature_samples = np.fft.irfft(-(omega**2) * output, n=count) * count
    fourth_derivative_bound = 2 * np.sum(abs(output) * omega**4)
    curvature_bound = np.max(abs(curvature_samples)) + fourth_derivative_bound / (8 * sample_rate**2)
    between_sample_margin = float(curvature_bound / (8 * sample_rate**2))
    return load, between_sample_margin, float(np.sqrt(2 * np.sum(abs(output) ** 2)))


def optimize_average(tire, loads, bins=256, grid_points=121):
    """Find a peak on a global slip grid, then refine the full-record mean.

    Histogram bin means only locate promising brackets. Final objective values
    and local refinements evaluate every original load sample without binning.
    """
    exact = LateralForce(tire, loads)
    loads = np.asarray(loads)
    if np.ptp(loads) == 0:
        means, weights = loads[:1], np.ones(1)
    else:
        edges = np.linspace(float(loads.min()), float(loads.max()), bins + 1)
        counts, _ = np.histogram(loads, edges)
        sums, _ = np.histogram(loads, edges, weights=loads)
        active = counts > 0
        means, weights = sums[active] / counts[active], counts[active] / len(loads)
    compressed = LateralForce(tire, means)
    grid = np.linspace(tire["ALPMIN"], tire["ALPMAX"], grid_points)
    values = np.array([float(np.dot(compressed(a), weights)) for a in grid])
    candidates = [(float(np.mean(exact(grid[0]))), grid[0]), (float(np.mean(exact(grid[-1]))), grid[-1])]
    for i in range(1, len(grid) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            result = minimize_scalar(
                lambda a: -float(np.mean(exact(a))),
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-8},
            )
            if not result.success:
                raise RuntimeError("Slip optimization failed")
            candidates.append((-float(result.fun), float(result.x)))
    force, angle = max(candidates)
    force_samples = exact(angle)
    sampling_error = abs(force - float(np.mean(force_samples[::2]))) / max(abs(force), 1)
    return force, angle, sampling_error


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, seed_count):
    groups = {}
    for row in rows:
        key = (row["axle"], row["iso_class"], row["wheelrate_n_per_m"])
        groups.setdefault(key, []).append(row)
    result = []
    for (axle, cls, rate), entries in groups.items():
        valid = all(r["supported"] for r in entries) and len(entries) == seed_count
        force = [r["capacity_n"] for r in entries]
        loss = [r["loss_percent"] for r in entries]
        result.append(
            {
                "axle": axle,
                "iso_class": cls,
                "wheelrate_n_per_m": rate,
                "wheelrate_lbf_per_in": rate / 175.126835,
                "supported": valid,
                "supported_records": sum(r["supported"] for r in entries),
                "total_records": len(entries),
                "smooth_capacity_n": entries[0]["smooth_capacity_n"],
                "capacity_lbf": float(np.mean(force)) / LBF if valid else None,
                "capacity_min_lbf": min(force) / LBF if valid else None,
                "capacity_max_lbf": max(force) / LBF if valid else None,
                "loss_percent": float(np.mean(loss)) if valid else None,
                "loss_min_percent": min(loss) if valid else None,
                "loss_max_percent": max(loss) if valid else None,
                "load_min_bound_n": min(r["load_min_bound_n"] for r in entries),
                "load_max_bound_n": max(r["load_max_bound_n"] for r in entries),
            }
        )
    return result


def make_plots(qcm, config, corners, rows, output):
    groups = [("a_h", qcm["iso_classes"])] + [(c.lower(), [c]) for c in qcm["iso_classes"]]
    if all(c in qcm["iso_classes"] for c in ("A", "B")):
        groups.append(("a_b", ["A", "B"]))
    for suffix, classes in groups:
        series, subplots = {}, []
        for corner in corners:
            for metric, label in (
                ("capacity_lbf", "Effective lateral capacity (lbf)"),
                ("loss_percent", "Grip loss relative to smooth road (%)"),
            ):
                key = corner.axle + "_" + metric
                series[key], series[key + "_rate"] = {}, {}
                for cls in classes:
                    selected = [r for r in rows if r["axle"] == corner.axle and r["iso_class"] == cls]
                    series[key][cls] = [r[metric] if r["supported"] else np.nan for r in selected]
                    series[key + "_rate"][cls] = [r["wheelrate_lbf_per_in"] for r in selected]
                subplots.append(
                    {
                        "title": corner.axle.title() + " corner",
                        "yscale": "linear",
                        "x": {"key": key + "_rate", "label": "Wheelrate (lb/in)"},
                        "y": {"key": key, "label": label},
                        "linewidth": 2.3,
                    }
                )
        plot = {
            "plots": {
                f"grip_vs_wheelrate_iso_{suffix}": {
                    "layout": "quad",
                    "subplots": subplots,
                    "title": f"2027 | ISO {', '.join(classes)} | {qcm['speed_mps']:g} m/s | "
                    f"nominal damping ratio {qcm['damping_ratio']:.2f}",
                }
            }
        }
        for name, fig in PlotEngine(plot).render({"series": series}):
            fig.set_size_inches(14, 10.5)
            fig.subplots_adjust(left=0.08, right=0.975, top=0.88, bottom=0.21, wspace=0.27, hspace=0.4)
            shown = [r for r in rows if r["iso_class"] in classes and r["supported"]]
            loss_top = max([r["loss_max_percent"] for r in shown] + [0.05 if shown else 1.0]) * 1.25
            for ax, sub in zip(fig.axes, subplots):
                key = sub["y"]["key"]
                axle = key.split("_")[0]
                metric = "capacity_lbf" if key.endswith("capacity_lbf") else "loss_percent"
                for line in list(ax.lines):
                    line.remove()
                has_data = False
                for cls in classes:
                    selected = [r for r in rows if r["axle"] == axle and r["iso_class"] == cls]
                    x = np.array(series[key + "_rate"][cls])
                    y = np.array(series[key][cls])
                    valid = np.isfinite(y)
                    has_data = has_data or bool(np.any(valid))
                    label = cls if np.any(valid) else cls + " (unsupported)"
                    ax.plot(
                        x,
                        y,
                        color=COLORS[cls],
                        label=label,
                        lw=2.2,
                        marker="o" if 0 < np.sum(valid) <= 3 else None,
                        markersize=4,
                    )
                    low = "capacity_min_lbf" if metric == "capacity_lbf" else "loss_min_percent"
                    high = "capacity_max_lbf" if metric == "capacity_lbf" else "loss_max_percent"
                    ax.fill_between(
                        x,
                        [r[low] if r["supported"] else np.nan for r in selected],
                        [r[high] if r["supported"] else np.nan for r in selected],
                        color=COLORS[cls],
                        alpha=0.18,
                    )
                if metric == "capacity_lbf":
                    baseline = next(r["smooth_capacity_n"] for r in rows if r["axle"] == axle) / LBF
                    ax.axhline(baseline, color="#555555", ls=":", label=f"Smooth: {baseline:.2f} lbf")
                    available = [r["capacity_min_lbf"] for r in shown if r["axle"] == axle]
                    if available:
                        span = max(baseline - min(available), 0.001 * baseline)
                        ax.set_ylim(min(available) - 0.20 * span, baseline + 0.25 * span)
                    else:
                        ax.set_ylim(0, baseline * 1.15)
                else:
                    ax.set_ylim(0, loss_top)
                ax.set_xlim(qcm["wheelrate_min_n_per_m"] / 175.126835, qcm["wheelrate_max_n_per_m"] / 175.126835)
                ax.legend(loc="upper right", fontsize=8, ncol=2 if len(classes) > 4 else 1)
                if not has_data:
                    ax.text(
                        0.5,
                        0.45,
                        "No supported grip result\nTire-load bounds exceeded",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=12,
                    )
            fig.text(
                0.5,
                0.145,
                "Grip = maximum mean +Fy at one fixed slip angle; zero camber, zero longitudinal slip. "
                "MF5.2 quasi-steady screening, not measured vehicle grip.",
                ha="center",
                fontsize=10,
            )
            fig.text(
                0.5,
                0.105,
                f"{len(config['seeds'])} matched {config['duration_s']:g} s periodic road records; "
                "line = mean of record capacities; shading = record range (not confidence interval).",
                ha="center",
                fontsize=10,
            )
            fig.text(
                0.5,
                0.065,
                "Blank segments: at least one record cannot be bounded inside the tire's "
                "100–1800 N fit. No clipping, extrapolation, or averaging only passing records.",
                ha="center",
                fontsize=10,
            )
            fig.text(
                0.5,
                0.03,
                "ISO band 0.01–10 cycles/m; fixed static mean loads. No anti geometry, aero, "
                "tire relaxation, or travel limits. Capacity axes zoomed separately; all axes linear.",
                ha="center",
                fontsize=10,
            )
            fig.savefig(output / (name + ".png"), dpi=170)
            fig.savefig(output / (name + ".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="_3_StandardSim/QuarterCarPSD/grip_config.yml")
    parser.add_argument("--replot", action="store_true", help="Render saved CSV without recomputing forces")
    args = parser.parse_args()
    root = repo_root()
    path = root / args.config
    config = load_yaml(path)
    qcm = load_yaml(root / config["qcm_config"])
    vehicle_path = root / qcm["vehicle"]
    vehicle = load_yaml(vehicle_path)
    corners, _ = load_corners(vehicle_path)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        prior = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        for source in (path, root / config["qcm_config"], vehicle_path):
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != prior["source_sha256"][source.relative_to(root).as_posix()]
            ):
                raise ValueError("Saved results use different inputs; rerun the study before replotting")
        with (output / "grip_sweep.csv").open(encoding="utf-8") as handle:
            saved = [
                {
                    k: (
                        v
                        if k in ("axle", "iso_class")
                        else v == "True"
                        if k == "supported"
                        else float(v)
                        if v
                        else None
                    )
                    for k, v in row.items()
                }
                for row in csv.DictReader(handle)
            ]
        make_plots(qcm, config, corners, saved, output)
        provenance = {
            "mode": "plot-only refresh; simulation provenance remains in summary.json",
            "plot_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "grip_csv_sha256": hashlib.sha256((output / "grip_sweep.csv").read_bytes()).hexdigest(),
        }
        (output / "plot_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (output / "executed_plot_runner.py").write_bytes(Path(__file__).read_bytes())
        return
    duration, fs = config["duration_s"], config["sample_rate_hz"]
    roads = {seed: road_spectrum(qcm, duration, fs, seed) for seed in config["seeds"]}
    np.savez_compressed(
        output / "road_spectra.npz",
        frequency_hz=next(iter(roads.values()))[0],
        **{f"seed_{seed}": road[1] for seed, road in roads.items()},
    )
    rates = np.linspace(qcm["wheelrate_min_n_per_m"], qcm["wheelrate_max_n_per_m"], qcm["wheelrate_points"])
    rows, tire_paths = [], []
    for corner in corners:
        tire_path = tire_templates_root(vehicle) / (tire_template_name(vehicle, vehicle[corner.axle]) + ".tir")
        tire_paths.append(tire_path)
        tire = {key: float(value) for key, value in parse_tir(tire_path).items() if isinstance(value, (int, float))}
        smooth, smooth_alpha, _ = optimize_average(tire, np.array([corner.mean_load_n]))
        for index, rate in enumerate(rates):
            damping = wheel_damping(corner, rate, qcm["damping_ratio"])
            analytic, _ = rms_response(
                corner,
                rate,
                damping,
                qcm["speed_mps"],
                ISO_G0["A"],
                qcm["spatial_frequency_min_cycles_per_m"],
                qcm["spatial_frequency_max_cycles_per_m"],
                qcm["frequency_points"],
            )
            for seed, (f, spectrum, count) in roads.items():
                dynamic, margin, spectral_rms = response_record(corner, rate, damping, f, spectrum, count, fs)
                psd_error = abs(spectral_rms - analytic) / analytic
                if psd_error > 0.002:
                    raise RuntimeError("Discrete road spectrum does not match analytical load RMS")
                for cls in qcm["iso_classes"]:
                    scale = np.sqrt(ISO_G0[cls] / ISO_G0["A"])
                    loads = corner.mean_load_n + dynamic * scale
                    lower, upper = float(loads.min() - margin * scale), float(loads.max() + margin * scale)
                    used_multiplier, used_margin = 1, margin
                    # Refine only ambiguous bounds, using the SAME Fourier record.
                    # Actual sampled fit departures are never repaired or clipped.
                    for multiplier in (2, 4, 8):
                        if lower >= tire["FZMIN"] and upper <= tire["FZMAX"]:
                            break
                        if loads.min() < tire["FZMIN"] or loads.max() > tire["FZMAX"]:
                            break
                        refined, used_margin, _ = response_record(
                            corner, rate, damping, f, spectrum, count, fs, multiplier
                        )
                        loads = corner.mean_load_n + refined * scale
                        lower = float(loads.min() - used_margin * scale)
                        upper = float(loads.max() + used_margin * scale)
                        used_multiplier = multiplier
                    supported = lower >= tire["FZMIN"] and upper <= tire["FZMAX"]
                    domain_status = (
                        "supported"
                        if supported
                        else (
                            "sampled_departure"
                            if loads.min() < tire["FZMIN"] or loads.max() > tire["FZMAX"]
                            else "bound_unresolved"
                        )
                    )
                    force = angle = loss = error = None
                    if supported:
                        force, angle, error = optimize_average(
                            tire, loads, config["histogram_bins"], config["slip_grid_points"]
                        )
                        if error > 1e-4:
                            raise RuntimeError("Tire-force time-resolution check failed")
                        loss = 100 * (1 - force / smooth)
                    rows.append(
                        {
                            "axle": corner.axle,
                            "iso_class": cls,
                            "seed": seed,
                            "wheelrate_n_per_m": float(rate),
                            "wheelrate_lbf_per_in": float(rate / 175.126835),
                            "wheel_damping_ns_per_m": damping,
                            "mean_load_n": float(loads.mean()),
                            "rms_load_n": float(np.std(loads)),
                            "sampled_min_load_n": float(loads.min()),
                            "sampled_max_load_n": float(loads.max()),
                            "between_sample_margin_n": used_margin * scale,
                            "sample_rate_hz_used": fs * used_multiplier,
                            "domain_status": domain_status,
                            "load_min_bound_n": lower,
                            "load_max_bound_n": upper,
                            "supported": supported,
                            "capacity_n": force,
                            "optimal_slip_rad": angle,
                            "loss_percent": loss,
                            "smooth_capacity_n": smooth,
                            "smooth_optimal_slip_rad": smooth_alpha,
                            "psd_rms_relative_error": psd_error,
                            "force_sampling_relative_error": error,
                        }
                    )
            if index % 30 == 0:
                print(f"{corner.axle}: {index + 1}/{len(rates)} rates", flush=True)
        write_csv(output / "grip_records.csv", rows)
    summary_rows = aggregate(rows, len(roads))
    write_csv(output / "grip_sweep.csv", summary_rows)
    make_plots(qcm, config, corners, summary_rows, output)
    sources = [
        path,
        root / config["qcm_config"],
        vehicle_path,
        Path(__file__),
        root / "_0_Utils/mf52_lateral.py",
        root / "_3_StandardSim/QuarterCarPSD/quarter_car_psd.py",
        root / "_0_Utils/dyn_py/parameters.py",
        *set(tire_paths),
    ]
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for i, source in enumerate(sources):
        (snapshot / f"{i:02}_{source.name}").write_bytes(source.read_bytes())
    summary = {
        "config": config,
        "qcm_config": qcm,
        "records": len(rows),
        "sweep_rows": len(summary_rows),
        "supported_records": sum(r["supported"] for r in rows),
        "supported_sweep_rows": sum(r["supported"] for r in summary_rows),
        "max_psd_rms_relative_error": max(r["psd_rms_relative_error"] for r in rows),
        "max_force_sampling_relative_error": max(r["force_sampling_relative_error"] for r in rows if r["supported"]),
        "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "scope": "Finite periodic records; pure lateral +Fy capacity at one optimized fixed slip angle per record; "
        "zero camber, no tire relaxation, aero or anti. Strict continuous-record load-bound screen. "
        "Unsupported records withheld. Seed range is not a statistical confidence interval.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256", "qcm_config")}, indent=2))


if __name__ == "__main__":
    main()
