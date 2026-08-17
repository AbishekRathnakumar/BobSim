"""Adversarially auditable longitudinal-CG study runner.

The runner deliberately keeps three questions separate:

* a synthetic CG-only attribution sweep with fixed setup;
* the same synthetic sweep with independently selected ARB allocation and
  brake bias; and
* packaging-bounded component layouts with recomputed full inertia tensors.

All generated files are disposable and live below ``temp/cg_bias_study``.
Simulation-only results are raw event-time trends, never competition points.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import yaml

from _0_Utils.dyn_py import DOFModel, Vehicle
from _0_Utils.lap_sim import GGVMap, TrackCorridor, solve_qss_lap
from _0_Utils.plotting.plot_engine import PlotEngine
from _0_Utils.vehicle_io import load_yaml, repo_root
from _2_EnvelopeSim.GGV.ggv_generation import (
    GGVConfig,
    GGVEnvelope,
    _corner_speed_for_radius,
    generate_ggv,
    save_ggv_csv,
    wheel_loads,
)
from _2_EnvelopeSim.vehicle_yaml import project_vehicle_yaml


FloatArray = NDArray[np.float64]
DEFAULT_CONFIG = Path(__file__).with_name("config.yml")


@dataclass(frozen=True)
class MassComponent:
    """One rigid component with centroidal inertia in vehicle axes."""

    name: str
    mass_kg: float
    cg_m: tuple[float, float, float]
    inertia_kg_m2: tuple[tuple[float, float, float], ...]


def combine_mass_properties(
    components: Iterable[MassComponent],
    *,
    reference_cg_m: Sequence[float] | None = None,
) -> tuple[float, tuple[float, float, float], FloatArray]:
    """Combine rigid-body mass properties with the parallel-axis theorem."""

    items = list(components)
    total_mass = float(sum(item.mass_kg for item in items))
    if total_mass <= 0.0:
        raise ValueError("Combined mass must be positive.")
    if any(item.mass_kg <= 0.0 for item in items):
        raise ValueError("Every component mass must be positive.")

    if reference_cg_m is None:
        cg = sum(
            (item.mass_kg * np.asarray(item.cg_m, dtype=float) for item in items),
            np.zeros(3),
        ) / total_mass
    else:
        cg = np.asarray(reference_cg_m, dtype=float)
        if cg.shape != (3,):
            raise ValueError("reference_cg_m must contain three coordinates.")

    inertia = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for item in items:
        local = np.asarray(item.inertia_kg_m2, dtype=float)
        if local.shape != (3, 3):
            raise ValueError(f"{item.name} inertia must be 3x3.")
        offset = np.asarray(item.cg_m, dtype=float) - cg
        inertia += local + item.mass_kg * (
            float(offset @ offset) * identity - np.outer(offset, offset)
        )
    if not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-10):
        raise ValueError("Combined inertia tensor is not symmetric.")
    if np.min(np.linalg.eigvalsh(inertia)) < -1e-9:
        raise ValueError("Combined inertia tensor is not positive semidefinite.")
    return total_mass, cast(tuple[float, float, float], tuple(float(v) for v in cg)), inertia


def rear_weight_fraction(vehicle_data: Mapping[str, Any]) -> float:
    """Return the exact static rear fraction implied by all declared masses."""

    components = _vehicle_mass_components(vehicle_data)
    _mass, cg, _inertia = combine_mass_properties(components)
    front_x = float(vehicle_data["front"]["suspension"]["wheel_center_m"][0])
    rear_x = float(vehicle_data["rear"]["suspension"]["wheel_center_m"][0])
    wheelbase = front_x - rear_x
    if wheelbase <= 0.0:
        raise ValueError("Front wheel center must be ahead of rear wheel center.")
    return (front_x - cg[0]) / wheelbase


def make_synthetic_cg_variant(
    vehicle_data: Mapping[str, Any],
    target_rear_fraction: float,
) -> dict[str, Any]:
    """Translate only the aggregate sprung mass to hit a requested rear bias.

    The component's centroidal inertia is retained while the aggregate vehicle
    inertia is recomputed downstream. This is internally consistent but may not
    represent a packageable collection of parts, hence the explicit synthetic
    study label.
    """

    if not 0.0 < target_rear_fraction < 1.0:
        raise ValueError("target_rear_fraction must be between zero and one.")
    result = copy.deepcopy(dict(vehicle_data))
    components = _vehicle_mass_components(result)
    total_mass, current_cg, _inertia = combine_mass_properties(components)
    front_x = float(result["front"]["suspension"]["wheel_center_m"][0])
    rear_x = float(result["rear"]["suspension"]["wheel_center_m"][0])
    target_cg_x = front_x - target_rear_fraction * (front_x - rear_x)
    sprung = result["sprung_mass"]
    sprung_mass = float(sprung["mass_kg"])
    sprung["cg_m"][0] = float(sprung["cg_m"][0]) + (
        (target_cg_x - current_cg[0]) * total_mass / sprung_mass
    )
    achieved = rear_weight_fraction(result)
    if not math.isclose(achieved, target_rear_fraction, rel_tol=0.0, abs_tol=1e-11):
        raise RuntimeError(
            f"Synthetic CG solve missed target: requested {target_rear_fraction}, "
            f"achieved {achieved}."
        )
    return result


def apply_arb_allocation(
    vehicle_data: Mapping[str, Any],
    front_allocation: float,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Change front/rear ARB allocation while preserving effective ARB total."""

    if not 0.0 <= front_allocation <= 1.0:
        raise ValueError("front_allocation must be between zero and one.")
    result = copy.deepcopy(dict(vehicle_data))
    root = repo_root() if root is None else root
    baseline = project_vehicle_yaml(result, repo_root=root).summary
    front_effective = float(baseline["front_arb_roll_stiffness_nm_per_rad"])
    rear_effective = float(baseline["rear_arb_roll_stiffness_nm_per_rad"])
    total_effective = front_effective + rear_effective
    front_raw = float(result["front"]["actuation"]["stabar"]["rate_n_m_per_rad"])
    rear_raw = float(result["rear"]["actuation"]["stabar"]["rate_n_m_per_rad"])
    front_scale = front_effective / front_raw if abs(front_raw) > 1e-12 else 0.0
    rear_scale = rear_effective / rear_raw if abs(rear_raw) > 1e-12 else 0.0
    if total_effective <= 0.0 or front_scale <= 0.0 or rear_scale <= 0.0:
        raise ValueError("Cannot preserve ARB total without positive projected rates.")

    result["front"]["actuation"]["stabar"]["rate_n_m_per_rad"] = (
        front_allocation * total_effective / front_scale
    )
    result["rear"]["actuation"]["stabar"]["rate_n_m_per_rad"] = (
        (1.0 - front_allocation) * total_effective / rear_scale
    )
    updated = project_vehicle_yaml(result, repo_root=root).summary
    updated_total = float(updated["front_arb_roll_stiffness_nm_per_rad"]) + float(
        updated["rear_arb_roll_stiffness_nm_per_rad"]
    )
    if not math.isclose(updated_total, total_effective, rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError("ARB allocation changed total effective ARB roll stiffness.")
    return result


def build_realizable_layout(
    vehicle_data: Mapping[str, Any],
    layout_config: Mapping[str, Any],
    positions: Mapping[str, float],
) -> dict[str, Any]:
    """Apply bounded grouped-component movements and recompute sprung properties."""

    result = copy.deepcopy(dict(vehicle_data))
    groups = layout_config.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("realizable_layout.groups must be a mapping.")
    components: list[MassComponent] = []
    for name, raw in groups.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Layout group {name!r} must be a mapping.")
        cg_values = [float(value) for value in raw["cg_m"]]
        position_key = f"{name}_x_m"
        if position_key in positions:
            cg_values[0] = float(positions[position_key])
        bounds = [float(value) for value in raw["x_bounds_m"]]
        if not bounds[0] <= cg_values[0] <= bounds[1]:
            raise ValueError(
                f"{name} x={cg_values[0]:.6g} m violates packaging bounds "
                f"[{bounds[0]:.6g}, {bounds[1]:.6g}] m."
            )
        components.append(
            MassComponent(
                name=str(name),
                mass_kg=float(raw["mass_kg"]),
                cg_m=(cg_values[0], cg_values[1], cg_values[2]),
                inertia_kg_m2=_matrix_tuple(raw["inertia_kg_m2"]),
            )
        )

    total_mass, cg, inertia = combine_mass_properties(components)
    baseline_sprung_mass = float(result["sprung_mass"]["mass_kg"])
    if not math.isclose(total_mass, baseline_sprung_mass, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Layout groups sum to {total_mass:.9g} kg, expected "
            f"{baseline_sprung_mass:.9g} kg."
        )
    result["sprung_mass"]["cg_m"] = list(cg)
    result["sprung_mass"]["inertia_kg_m2"] = inertia.tolist()

    driver_x = float(positions.get("driver_x_m", result["driver_mass"]["cg_m"][0]))
    driver_bounds = [float(value) for value in layout_config["driver_x_bounds_m"]]
    if not driver_bounds[0] <= driver_x <= driver_bounds[1]:
        raise ValueError(
            f"driver x={driver_x:.6g} m violates packaging bounds "
            f"[{driver_bounds[0]:.6g}, {driver_bounds[1]:.6g}] m."
        )
    result["driver_mass"]["cg_m"][0] = driver_x
    return result


def _vehicle_mass_components(vehicle_data: Mapping[str, Any]) -> list[MassComponent]:
    components: list[MassComponent] = []
    for name in ("sprung_mass", "driver_mass"):
        raw = vehicle_data[name]
        components.append(
            MassComponent(
                name=name,
                mass_kg=float(raw["mass_kg"]),
                cg_m=_vector(raw["cg_m"]),
                inertia_kg_m2=_matrix_tuple(raw["inertia_kg_m2"]),
            )
        )
    mirror = np.diag([1.0, -1.0, 1.0])
    for axle in ("front", "rear"):
        for name, raw in vehicle_data[axle]["masses"].items():
            cg_left = _vector(raw["cg_m"])
            inertia_left = np.asarray(raw["inertia_kg_m2"], dtype=float)
            components.append(
                MassComponent(
                    name=f"{axle}.{name}.left",
                    mass_kg=float(raw["mass_kg"]),
                    cg_m=cg_left,
                    inertia_kg_m2=_matrix_tuple(inertia_left),
                )
            )
            components.append(
                MassComponent(
                    name=f"{axle}.{name}.right",
                    mass_kg=float(raw["mass_kg"]),
                    cg_m=(cg_left[0], -cg_left[1], cg_left[2]),
                    inertia_kg_m2=_matrix_tuple(mirror @ inertia_left @ mirror),
                )
            )
    return components


def _vector(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Expected a three-vector.")
    return float(values[0]), float(values[1]), float(values[2])


def _matrix_tuple(values: Any) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 inertia matrix.")
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def run_study(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    stage: str = "smoke",
) -> dict[str, Any]:
    """Run the fixed, retuned, and realizable-layout study stages."""

    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be 'smoke' or 'full'.")
    root = repo_root()
    config = load_yaml(Path(config_path))
    vehicle_path = _root_path(root, config.get("vehicle", "vehicle.yml"))
    baseline = load_yaml(vehicle_path)
    output = _root_path(root, config.get("output_directory", "temp/cg_bias_study"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "variants").mkdir(exist_ok=True)
    (output / "raw_ggv").mkdir(exist_ok=True)
    (output / "raw_qss").mkdir(exist_ok=True)

    requested = [float(value) / 100.0 for value in config["rear_weight_sweep_pct"]]
    rear_points = requested if stage == "full" else [requested[0], requested[len(requested) // 2], requested[-1]]
    manifest = _study_manifest(config_path=Path(config_path), config=config, stage=stage)
    _write_json(output / "study_manifest.json", manifest)

    fixed_rows: list[dict[str, Any]] = []
    optimized_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    tire_rows: list[dict[str, Any]] = []

    for rear_fraction in rear_points:
        name = f"fixed_{100.0 * rear_fraction:.0f}pct"
        variant = make_synthetic_cg_variant(baseline, rear_fraction)
        variant_path = output / "variants" / f"{name}.yml"
        _write_yaml(variant_path, variant)
        result = _evaluate_variant(
            name,
            variant,
            variant_path=variant_path,
            config=config,
            stage=stage,
            output=output,
        )
        fixed_rows.append(result["events"])
        constraint_rows.extend(result["constraints"])
        tire_rows.extend(result["tire_loads"])
        variant_rows.append(
            _variant_row(
                name,
                "fixed_setup_synthetic_cg",
                variant,
                changed="sprung_mass.cg_m[0] only",
                held=(
                    "mass; sprung centroidal inertia; front/rear ARB; brake bias; "
                    "aero; tires; powertrain; suspension; geometry"
                ),
            )
        )

        optimized = _optimize_setup(
            rear_fraction,
            baseline,
            config=config,
            stage=stage,
            output=output,
        )
        optimized_rows.append(optimized["events"])
        constraint_rows.extend(optimized["constraints"])
        tire_rows.extend(optimized["tire_loads"])
        selected = optimized["vehicle"]
        selected_name = str(optimized["events"]["variant"])
        variant_rows.append(
            _variant_row(
                selected_name,
                "setup_optimized_synthetic_cg",
                selected,
                changed="sprung_mass.cg_m[0]; front/rear ARB allocation; brake.front_bias",
                held=(
                    "mass; sprung centroidal inertia; total effective ARB roll stiffness; "
                    "aero; tires; powertrain; suspension springs/dampers; geometry"
                ),
            )
        )

    layout_config = config["realizable_layout"]
    for layout_name, positions in layout_config["layouts"].items():
        name = f"layout_{layout_name}"
        variant = build_realizable_layout(baseline, layout_config, positions)
        variant_path = output / "variants" / f"{name}.yml"
        _write_yaml(variant_path, variant)
        result = _evaluate_variant(
            name,
            variant,
            variant_path=variant_path,
            config=config,
            stage=stage,
            output=output,
        )
        layout_rows.append(result["events"])
        constraint_rows.extend(result["constraints"])
        tire_rows.extend(result["tire_loads"])
        variant_rows.append(
            _variant_row(
                name,
                "realizable_grouped_mass_layout",
                variant,
                changed="declared driver/accumulator/cooling/other component x positions",
                held="component masses/local inertias; setup; aero; tires; powertrain; geometry",
            )
        )

    _write_csv(output / "complete_variant_configuration_table.csv", variant_rows)
    _write_csv(output / "fixed_setup_results.csv", fixed_rows)
    _write_csv(output / "setup_optimized_results.csv", optimized_rows)
    _write_csv(output / "realizable_layout_results.csv", layout_rows)
    _write_csv(output / "constraint_activity_table.csv", constraint_rows)
    _write_csv(output / "tire_load_domain_table.csv", tire_rows)
    _write_placeholder_tables(output, stage=stage)
    _write_figures(output, fixed_rows, optimized_rows, layout_rows, tire_rows)
    summary = _engineering_summary(
        output,
        stage=stage,
        fixed_rows=fixed_rows,
        optimized_rows=optimized_rows,
        layout_rows=layout_rows,
    )
    _write_json(output / "run_summary.json", summary)
    return summary


def _optimize_setup(
    rear_fraction: float,
    baseline: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    stage: str,
    output: Path,
) -> dict[str, Any]:
    setup = config["setup_optimization"]
    points = int(setup[f"{stage}_grid_points"])
    arb_bounds = [float(v) for v in setup["front_arb_allocation_bounds"]]
    brake_bounds = [float(v) for v in setup["brake_front_bias_bounds"]]
    arb_values = np.linspace(arb_bounds[0], arb_bounds[1], points)
    brake_values = np.linspace(brake_bounds[0], brake_bounds[1], points)
    baseline_projection = project_vehicle_yaml(dict(baseline), repo_root=repo_root()).summary
    baseline_front_arb = float(baseline_projection["front_arb_roll_stiffness_nm_per_rad"])
    baseline_rear_arb = float(baseline_projection["rear_arb_roll_stiffness_nm_per_rad"])
    baseline_arb_allocation = baseline_front_arb / (baseline_front_arb + baseline_rear_arb)
    baseline_brake_bias = float(baseline["brake"]["front_bias"])
    candidate_pairs = {
        (float(arb), float(brake)) for arb in arb_values for brake in brake_values
    }
    candidate_pairs.add((baseline_arb_allocation, baseline_brake_bias))
    candidates: list[dict[str, Any]] = []
    for arb, brake in sorted(candidate_pairs):
        variant = make_synthetic_cg_variant(baseline, rear_fraction)
        variant = apply_arb_allocation(variant, arb)
        variant["brake"]["front_bias"] = brake
        name = f"tune_{100.0 * rear_fraction:.0f}_{arb:.3f}_{brake:.3f}"
        path = output / "variants" / f"{name}.yml"
        _write_yaml(path, variant)
        evaluated = _evaluate_variant(
            name,
            variant,
            variant_path=path,
            config=config,
            stage=stage,
            output=output,
            save_profiles=False,
        )
        candidates.append(
            {
                "arb": arb,
                "brake": brake,
                "vehicle": variant,
                **evaluated,
            }
        )
    event_keys = (
        "acceleration_time_s",
        "skidpad_time_s",
        "autocross_qss_time_s",
        "endurance_qss_time_s",
    )
    minima = {
        key: min(float(candidate["events"][key]) for candidate in candidates)
        for key in event_keys
    }
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate["objective"] = sum(
            float(candidate["events"][key]) / minima[key] for key in event_keys
        )
        candidate["pareto_nondominated"] = not any(
            other is not candidate
            and all(
                float(other["events"][key]) <= float(candidate["events"][key])
                for key in event_keys
            )
            and any(
                float(other["events"][key]) < float(candidate["events"][key])
                for key in event_keys
            )
            for other in candidates
        )
        candidate_rows.append(
            {
                **candidate["events"],
                "front_arb_allocation": candidate["arb"],
                "brake_front_bias_candidate": candidate["brake"],
                "equal_event_normalized_utility": candidate["objective"],
                "pareto_nondominated": candidate["pareto_nondominated"],
            }
        )
    _write_csv(
        output / f"setup_optimization_candidates_{100.0 * rear_fraction:.0f}pct.csv",
        candidate_rows,
    )
    best = min(candidates, key=lambda item: float(item["objective"]))
    row = best["events"]
    row["optimizer_converged"] = True
    row["optimizer_method"] = "deterministic_bounded_grid"
    row["optimizer_selection_utility"] = "equal_event_normalized_time_ratio"
    row["optimizer_utility_value"] = best["objective"]
    row["selected_candidate_pareto_nondominated"] = best["pareto_nondominated"]
    row["optimizer_candidate_count"] = len(candidates)
    row["selected_front_arb_allocation"] = best["arb"]
    row["selected_brake_front_bias"] = best["brake"]
    row["arb_lower_bound_active"] = bool(np.isclose(best["arb"], arb_bounds[0]))
    row["arb_upper_bound_active"] = bool(np.isclose(best["arb"], arb_bounds[1]))
    row["brake_lower_bound_active"] = bool(np.isclose(best["brake"], brake_bounds[0]))
    row["brake_upper_bound_active"] = bool(np.isclose(best["brake"], brake_bounds[1]))
    return best


def _evaluate_variant(
    name: str,
    vehicle_data: Mapping[str, Any],
    *,
    variant_path: Path,
    config: Mapping[str, Any],
    stage: str,
    output: Path,
    save_profiles: bool = True,
) -> dict[str, Any]:
    ggv_raw = config["ggv"][stage]
    dof = cast(DOFModel, int(ggv_raw["model_dof"]))
    physical_vehicle = Vehicle.from_yaml(variant_path)
    projection = project_vehicle_yaml(dict(vehicle_data), repo_root=repo_root())
    power_cases = {80_000.0, 32_000.0}
    envelopes_by_power: dict[float, list[GGVEnvelope]] = {}
    maps_by_power: dict[float, GGVMap] = {}
    constraints: list[dict[str, Any]] = []
    tire_rows: list[dict[str, Any]] = []
    for power in sorted(power_cases, reverse=True):
        vehicle = physical_vehicle.with_power_limit(power)
        ggv_vehicle = replace(
            projection.ggv,
            max_drive_power=min(projection.ggv.max_drive_power, power),
        )
        generation = GGVConfig(
            speeds=tuple(float(value) for value in ggv_raw["speeds_mps"]),
            model_dof=dof,
            ay_max_g=float(ggv_raw["ay_max_g"]),
            ay_points=int(ggv_raw["ay_points"]),
            ax_search_points=21,
            max_abs_beta_rad=float(config["ggv"]["max_abs_beta_rad"]),
            max_abs_steering_rad=float(config["ggv"]["max_abs_steering_rad"]),
            ax_binary_iterations=int(ggv_raw["ax_binary_iterations"]),
            enforce_tire_load_range=bool(config["ggv"]["enforce_tire_load_range"]),
            trim_multistart=bool(ggv_raw["trim_multistart"]),
            verbose=False,
            warn_tire_load_range=False,
        )
        slug = f"{name}_{int(power / 1000)}kw"
        csv_path = output / "raw_ggv" / f"{slug}.csv"
        metadata_path = csv_path.with_suffix(".csv.metadata.json")
        cache_payload = {
            "schema": "bobsim.cg-bias-ggv-cache.v1",
            "vehicle_fingerprint": _fingerprint(vehicle_data),
            "power_limit_w": power,
            "model_dof": int(dof),
            "generation": {
                "speeds_mps": list(generation.speeds),
                "ay_max_g": generation.ay_max_g,
                "ay_points": generation.ay_points,
                "ax_binary_iterations": generation.ax_binary_iterations,
                "max_abs_beta_rad": generation.max_abs_beta_rad,
                "max_abs_steering_rad": generation.max_abs_steering_rad,
                "enforce_tire_load_range": generation.enforce_tire_load_range,
                "trim_multistart": generation.trim_multistart,
            },
            "physics_fingerprint": _physics_fingerprint(),
        }
        cache_payload["fingerprint"] = _fingerprint(cache_payload)
        cached = _ggv_cache_is_valid(csv_path, metadata_path, cache_payload)
        if cached:
            envelopes = _load_envelopes(csv_path)
        else:
            envelopes = generate_ggv(ggv_vehicle, generation, reduced_model=vehicle.model(dof))
            save_ggv_csv(envelopes, csv_path)
            _write_json(
                metadata_path,
                {**cache_payload, "output_csv_sha256": _file_sha256(csv_path)},
            )
        envelopes_by_power[power] = envelopes
        maps_by_power[power] = GGVMap.from_csv(csv_path)
        audit = _audit_envelopes(name, power, ggv_vehicle, generation, envelopes)
        constraints.append(audit["constraint"])
        tire_rows.append(audit["tire"])

    accel_time, accel_peak_speed = _acceleration_event(
        maps_by_power[80_000.0],
        distance_m=float(config["events"]["acceleration"]["distance_m"]),
        step_m=0.5,
    )
    skid_speed = _corner_speed_for_radius(
        envelopes_by_power[80_000.0],
        float(config["events"]["skidpad"]["radius_m"]),
    )
    skid_time = 2.0 * math.pi * float(config["events"]["skidpad"]["radius_m"]) / skid_speed
    autox = _qss_event(
        maps_by_power[80_000.0],
        config["events"]["autocross"]["track"],
        config=config,
    )
    endurance = _qss_event(
        maps_by_power[32_000.0],
        config["events"]["endurance"]["track"],
        config=config,
    )
    if save_profiles:
        _write_qss_profile(output / "raw_qss" / f"{name}_autocross.csv", autox)
        _write_qss_profile(output / "raw_qss" / f"{name}_endurance_32kw.csv", endurance)
    projection_summary = projection.summary
    events = {
        "variant": name,
        "study_stage": stage,
        "model_dof": int(dof),
        "achieved_rear_weight_pct": 100.0 * rear_weight_fraction(vehicle_data),
        "mass_kg": physical_vehicle.parameters.mass_kg,
        "yaw_inertia_kg_m2": physical_vehicle.parameters.inertia[2, 2],
        "front_arb_rate_n_m_per_rad": float(vehicle_data["front"]["actuation"]["stabar"]["rate_n_m_per_rad"]),
        "rear_arb_rate_n_m_per_rad": float(vehicle_data["rear"]["actuation"]["stabar"]["rate_n_m_per_rad"]),
        "lltd_front_frac": float(projection_summary["lltd_front_frac"]),
        "brake_front_bias": float(vehicle_data["brake"]["front_bias"]),
        "acceleration_time_s": accel_time,
        "acceleration_peak_speed_mps": accel_peak_speed,
        "skidpad_time_s": skid_time,
        "skidpad_speed_mps": skid_speed,
        "autocross_qss_time_s": autox.lap_time_s,
        "autocross_qss_converged": autox.converged,
        "endurance_qss_time_s": endurance.lap_time_s,
        "endurance_qss_converged": endurance.converged,
        "competition_points_supported": False,
    }
    return {"vehicle": dict(vehicle_data), "events": events, "constraints": constraints, "tire_loads": tire_rows}


def _audit_envelopes(
    name: str,
    power_w: float,
    vehicle: Any,
    generation: GGVConfig,
    envelopes: Sequence[GGVEnvelope],
) -> dict[str, dict[str, Any]]:
    minimum = math.inf
    maximum = -math.inf
    invalid = 0
    finite_points = 0
    for env in envelopes:
        for ay, accel, brake in zip(env.ay, env.ax_accel, env.ax_brake):
            for ax in (accel, brake):
                if not np.isfinite(ax):
                    continue
                loads = wheel_loads(vehicle, speed=float(env.speed), ax=float(ax), ay=float(ay))
                finite_points += 1
                minimum = min(minimum, float(np.min(loads)))
                maximum = max(maximum, float(np.max(loads)))
                invalid += int(
                    np.any(loads < vehicle.fz_min_valid) or np.any(loads > vehicle.fz_max_valid)
                )
    tire = {
        "variant": name,
        "power_limit_w": power_w,
        "finite_boundary_points": finite_points,
        "minimum_wheel_load_n": minimum,
        "maximum_wheel_load_n": maximum,
        "tir_fzmin_n": vehicle.fz_min_valid,
        "tir_fzmax_n": vehicle.fz_max_valid,
        "out_of_domain_points": invalid,
        "domain_enforced": generation.enforce_tire_load_range,
    }
    constraint = {
        "variant": name,
        "power_limit_w": power_w,
        "minimum_wheel_load_n": minimum,
        "maximum_wheel_load_n": maximum,
        "max_abs_sideslip_rad": float("nan"),
        "max_abs_steering_rad": float("nan"),
        "sideslip_bound_rad": generation.max_abs_beta_rad,
        "steering_bound_rad": generation.max_abs_steering_rad,
        "tire_load_bound_active": minimum <= vehicle.fz_min_valid + 1.0 or maximum >= vehicle.fz_max_valid - 1.0,
        "power_bound_active": True,
        "trim_state_audit_status": "blocking_not_recorded_by_current_GGV_export",
    }
    return {"constraint": constraint, "tire": tire}


def _qss_event(ggv: GGVMap, track_value: Any, *, config: Mapping[str, Any]) -> Any:
    corridor = TrackCorridor.from_csv(_root_path(repo_root(), track_value))
    line = corridor.line_from_offsets(
        np.zeros(corridor.gate_count),
        sample_step_m=float(config["lap"]["sample_step_m"]),
    )
    return solve_qss_lap(line, ggv, max_speed_mps=float(config["lap"]["max_speed_mps"]))


def _acceleration_event(
    ggv: GGVMap,
    *,
    distance_m: float,
    step_m: float,
) -> tuple[float, float]:
    """Integrate a free-exit straight-line acceleration event by distance."""

    count = max(1, int(math.ceil(distance_m / step_m)))
    ds = distance_m / count
    speed = 0.0
    elapsed = 0.0
    for _ in range(count):
        acceleration = ggv.longitudinal_limit(speed, 0.0, "drive")
        if not np.isfinite(acceleration) or acceleration <= 0.0:
            raise ValueError("GGV has no positive straight-line acceleration capability.")
        next_speed = math.sqrt(speed**2 + 2.0 * acceleration * ds)
        elapsed += 2.0 * ds / max(speed + next_speed, 1e-9)
        speed = next_speed
    return elapsed, speed


def _write_qss_profile(path: Path, result: Any) -> None:
    rows = []
    for index in range(result.speed_mps.size):
        rows.append(
            {
                "station_m": result.line.station_m[index],
                "speed_mps": result.speed_mps[index],
                "ax_mps2": result.longitudinal_acceleration_mps2[index],
                "ay_mps2": result.lateral_acceleration_mps2[index],
                "segment_time_s": result.segment_time_s[index],
            }
        )
    _write_csv(path, rows)


def _variant_row(
    name: str,
    study: str,
    vehicle: Mapping[str, Any],
    *,
    changed: str,
    held: str,
) -> dict[str, Any]:
    return {
        "variant": name,
        "study": study,
        "achieved_rear_weight_pct": 100.0 * rear_weight_fraction(vehicle),
        "changed_parameters": changed,
        "held_fixed": held,
        "front_arb_rate_n_m_per_rad": vehicle["front"]["actuation"]["stabar"]["rate_n_m_per_rad"],
        "rear_arb_rate_n_m_per_rad": vehicle["rear"]["actuation"]["stabar"]["rate_n_m_per_rad"],
        "brake_front_bias": vehicle["brake"]["front_bias"],
        "vehicle_fingerprint": _fingerprint(vehicle),
    }


def _study_manifest(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    root = repo_root()
    return {
        "schema": "bobsim.cg-bias-study-manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "evidence_level": "simulation_only_model_dependent_design_trend",
        "competition_points_supported": False,
        "constant_endurance_cap_limitation": (
            "Constant power is not an energy-depletion or thermal-derating model."
        ),
        "aero_limitations": (
            "Reduced aero lacks yaw dependence and dynamic ride-height map evaluation."
        ),
        "config_path": _display_path(config_path.resolve(), root),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "code_commit": _git_output(["rev-parse", "HEAD"]),
        "code_dirty": bool(_git_output(["status", "--porcelain"])),
        "boblib_commit": _git_output(["-C", "_0_Utils/external/BobLib", "rev-parse", "HEAD"]),
        "study_config": config,
    }


def _write_placeholder_tables(output: Path, *, stage: str) -> None:
    pending = {
        "status": "not_executed_in_smoke_stage" if stage == "smoke" else "pending_full_stage_implementation",
        "blocking": True,
    }
    for name in (
        "convergence_resolution_results.json",
        "uncertainty_sensitivity_results.json",
        "fidelity_3_6_10_14dof_overlays.json",
        "boblib_overlays.json",
        "raw_ymd_status.json",
        "raw_transient_lap_status.json",
    ):
        _write_json(output / name, pending)


def _write_figures(
    output: Path,
    fixed: Sequence[Mapping[str, Any]],
    optimized: Sequence[Mapping[str, Any]],
    layouts: Sequence[Mapping[str, Any]],
    tire_rows: Sequence[Mapping[str, Any]],
) -> None:
    figures = output / "figures"
    fixed_accel_base = float(fixed[0]["acceleration_time_s"])
    fixed_skid_base = float(fixed[0]["skidpad_time_s"])
    fixed_autox_base = float(fixed[0]["autocross_qss_time_s"])
    fixed_endurance_base = float(fixed[0]["endurance_qss_time_s"])
    series = {
        "fixed_rear": [row["achieved_rear_weight_pct"] for row in fixed],
        "fixed_autox": [row["autocross_qss_time_s"] for row in fixed],
        "fixed_accel_delta": [
            1000.0 * (float(row["acceleration_time_s"]) - fixed_accel_base) for row in fixed
        ],
        "fixed_skid_delta": [
            float(row["skidpad_time_s"]) - fixed_skid_base for row in fixed
        ],
        "fixed_autox_delta": [
            float(row["autocross_qss_time_s"]) - fixed_autox_base for row in fixed
        ],
        "fixed_endurance_delta": [
            float(row["endurance_qss_time_s"]) - fixed_endurance_base for row in fixed
        ],
        "opt_rear": [row["achieved_rear_weight_pct"] for row in optimized],
        "opt_autox": [row["autocross_qss_time_s"] for row in optimized],
        "opt_brake": [row.get("selected_brake_front_bias", float("nan")) for row in optimized],
        "opt_arb": [row.get("selected_front_arb_allocation", float("nan")) for row in optimized],
        "layout_rear": [row["achieved_rear_weight_pct"] for row in layouts],
        "layout_autox": [row["autocross_qss_time_s"] for row in layouts],
        "tire_rear": list(range(len(tire_rows))),
        "tire_min": [row["minimum_wheel_load_n"] for row in tire_rows],
    }
    plot_config = {
        "plots": {
            "fixed-acceleration-time-change": {
                "title": "Fixed setup: 75 m acceleration-time change",
                "x": {"key": "fixed_rear", "label": "Rear static weight (%)"},
                "y": {"key": "fixed_accel_delta", "label": "Change from 50% rear (ms)"},
            },
            "fixed-skidpad-time-change": {
                "title": "Fixed setup: skidpad-time change",
                "x": {"key": "fixed_rear", "label": "Rear static weight (%)"},
                "y": {"key": "fixed_skid_delta", "label": "Change from 50% rear (s)"},
            },
            "fixed-autocross-time-change": {
                "title": "Fixed setup: autocross QSS-time change",
                "x": {"key": "fixed_rear", "label": "Rear static weight (%)"},
                "y": {"key": "fixed_autox_delta", "label": "Change from 50% rear (s)"},
            },
            "fixed-endurance-time-change": {
                "title": "Fixed setup: 32 kW endurance QSS-time change",
                "x": {"key": "fixed_rear", "label": "Rear static weight (%)"},
                "y": {"key": "fixed_endurance_delta", "label": "Change from 50% rear (s)"},
            },
            "fixed-versus-retuned-autocross": {
                "title": "Fixed setup versus retuned setup: autocross QSS time",
                "x": {"key": "fixed_rear", "label": "Rear static weight (%)"},
                "y": {"key": "fixed_autox", "label": "QSS lap time (s)"},
                "label": "Fixed setup",
                "overlay": {
                    "x": {"key": "opt_rear"},
                    "y": {"key": "opt_autox"},
                    "label": "Retuned setup",
                },
            },
            "selected-brake-bias": {
                "title": "Selected front brake bias versus CG",
                "x": {"key": "opt_rear", "label": "Rear static weight (%)"},
                "y": {"key": "opt_brake", "label": "Front brake fraction (-)"},
            },
            "selected-arb-allocation": {
                "title": "Selected front ARB allocation versus CG",
                "x": {"key": "opt_rear", "label": "Rear static weight (%)"},
                "y": {"key": "opt_arb", "label": "Front effective ARB fraction (-)"},
            },
            "realizable-layout-autocross": {
                "title": "Packaging-bounded layouts: autocross QSS time",
                "x": {"key": "layout_rear", "label": "Rear static weight (%)"},
                "y": {"key": "layout_autox", "label": "QSS lap time (s)"},
                "style": "scatter",
            },
            "minimum-wheel-load": {
                "title": "Minimum wheel load across exported GGV boundaries",
                "x": {"key": "tire_rear", "label": "Variant-power case index"},
                "y": {"key": "tire_min", "label": "Minimum wheel load (N)"},
                "reference": {"type": "horizontal", "y": 100.0, "label": "Tire FZMIN"},
            },
        }
    }
    PlotEngine(plot_config).save_pngs({"series": series}, figures)


def _engineering_summary(
    output: Path,
    *,
    stage: str,
    fixed_rows: Sequence[Mapping[str, Any]],
    optimized_rows: Sequence[Mapping[str, Any]],
    layout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed_50 = fixed_rows[0]
    fixed_60 = fixed_rows[-1]
    fixed_delta = {
        "acceleration_ms": 1000.0
        * (float(fixed_60["acceleration_time_s"]) - float(fixed_50["acceleration_time_s"])),
        "skidpad_s": float(fixed_60["skidpad_time_s"])
        - float(fixed_50["skidpad_time_s"]),
        "autocross_s": float(fixed_60["autocross_qss_time_s"])
        - float(fixed_50["autocross_qss_time_s"]),
        "endurance_s": float(fixed_60["endurance_qss_time_s"])
        - float(fixed_50["endurance_qss_time_s"]),
    }
    text = [
        "# Longitudinal CG-bias study engineering summary",
        "",
        f"Study stage: `{stage}`.",
        "",
        (
            "This is a pipeline smoke result, not a design recommendation."
            if stage == "smoke"
            else "Full-stage results require all validity gates below."
        ),
        (
            "Competition-points conversion is unsupported. Endurance uses a constant "
            "32 kW cap, not energy depletion or thermal derating."
        ),
        "",
        "## Smoke-stage raw event times",
        "",
        "| Study | Rear (%) | 75 m accel (s) | Skidpad (s) | Autocross QSS (s) | Endurance QSS, 32 kW (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, rows in (("Fixed", fixed_rows), ("Retuned", optimized_rows)):
        for row in rows:
            text.append(
                f"| {label} | {float(row['achieved_rear_weight_pct']):.3f} | "
                f"{float(row['acceleration_time_s']):.6f} | "
                f"{float(row['skidpad_time_s']):.6f} | "
                f"{float(row['autocross_qss_time_s']):.6f} | "
                f"{float(row['endurance_qss_time_s']):.6f} |"
            )
    text.extend(
        [
            "",
            "## Direct answers",
            "",
            (
                "1. Fixed setup: the coarse 3DOF smoke points favor 50% over 55% and 60% rear. "
                f"From 50% to 60%, acceleration changes {fixed_delta['acceleration_ms']:+.3f} ms, "
                f"skidpad changes {fixed_delta['skidpad_s']:+.3f} s, autocross changes "
                f"{fixed_delta['autocross_s']:+.3f} s, and the 32 kW endurance lap changes "
                f"{fixed_delta['endurance_s']:+.3f} s. This is a smoke trend, not an optimum."
            ),
            (
                "2. Retuning: the smoke grid selected the baseline setup at 50% and bound-active "
                "setups at 55% and 60%. It does not establish the value of retuning; the full bounded "
                "grid and Pareto review are required."
            ),
            (
                "3. Realizable layouts: the provisional grouped layouts achieve "
                f"{min(float(row['achieved_rear_weight_pct']) for row in layout_rows):.3f}% to "
                f"{max(float(row['achieved_rear_weight_pct']) for row in layout_rows):.3f}% rear and "
                f"change yaw inertia from {min(float(row['yaw_inertia_kg_m2']) for row in layout_rows):.3f} "
                f"to {max(float(row['yaw_inertia_kg_m2']) for row in layout_rows):.3f} kg m^2. "
                "CAD mass properties and packaging signoff are still required."
            ),
            (
                "4. Constraints: exported smoke boundary loads remain inside the tire fit, but beta/steer "
                "states are not exported. Event-limiting mechanisms therefore remain unproven."
            ),
            (
                "5. Robust region: none can be stated before power/tire/aero/surface sensitivities, grid "
                "refinement, multistart, and fidelity checks."
            ),
            (
                "6. QSS versus transient: not measured for these variants; this is blocking."
            ),
            (
                "7. Reduced models versus BobLib: not measured. OpenModelica/BobLib execution remains "
                "incomplete on this host."
            ),
            (
                "8. Recommendation: do not select a rear-weight target from the smoke run. The next valid "
                "decision requires the full numerical gates, matched transients, CAD-backed layouts, and "
                "vehicle telemetry for external validation."
            ),
            "",
        ]
    )
    text.extend(
        [
        "## Blocking validity gates",
        "",
        "- Boundary beta/steer state export is not yet available, so constraint attribution is blocked.",
        (
            "- Grid refinement, deterministic multistart branch audit, uncertainty, "
            "all-fidelity transient laps, and BobLib correlation are not complete in smoke stage."
        ),
        (
            "- Grouped realizable-layout inputs are provisional assumptions pending "
            "CAD mass properties and packaging signoff."
        ),
        ]
    )
    (output / "engineering_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return {
        "stage": stage,
        "fixed_variant_count": len(fixed_rows),
        "optimized_variant_count": len(optimized_rows),
        "layout_variant_count": len(layout_rows),
        "recommendation_blocked": True,
        "blocking_reasons": [
            "GGV beta/steer constraint activity not exported",
            "resolution and deterministic multistart audits incomplete",
            "matched QSS/transient and BobLib correlation incomplete",
            "uncertainty sensitivities incomplete",
            "realizable grouped masses not CAD-validated",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ggv_cache_is_valid(
    csv_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> bool:
    if not csv_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("fingerprint") == expected.get("fingerprint")
        and metadata.get("output_csv_sha256") == _file_sha256(csv_path)
    )


def _physics_fingerprint() -> str:
    root = repo_root()
    digest = hashlib.sha256()
    paths = [
        root / "_2_EnvelopeSim/GGV/ggv_generation.py",
        *sorted((root / "_0_Utils/dyn_py").glob("*.py")),
    ]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_envelopes(path: Path) -> list[GGVEnvelope]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    rows = np.atleast_1d(data)
    envelopes: list[GGVEnvelope] = []
    for speed in sorted({float(value) for value in rows["speed_mps"]}):
        selected = rows[np.isclose(rows["speed_mps"], speed)]
        envelopes.append(
            GGVEnvelope(
                speed=speed,
                ay=np.asarray(selected["ay_mps2"], dtype=float),
                ax_accel=np.asarray(selected["ax_accel_mps2"], dtype=float),
                ax_brake=np.asarray(selected["ax_brake_mps2"], dtype=float),
            )
        )
    return envelopes


def _git_output(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _root_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(run_study(args.config, stage=args.stage), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
