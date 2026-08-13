"""Run QSS racing-line optimization and/or a dyn_py forward-transient lap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from _0_Utils.dyn_py import DOFModel, create_model, load_reduced_vehicle_parameters
from _0_Utils.lap_sim import (
    GGVMap,
    TrackCorridor,
    optimize_racing_line,
    simulate_transient_lap,
    write_qss_lap_csv,
    write_transient_lap_csv,
)
from _0_Utils.lap_sim.racing_line import LineMode
from _0_Utils.vehicle_io import load_yaml, repo_root


DEFAULT_CONFIG = Path(__file__).with_name("lap_time_eval_config.yml")
Scenario = Literal["qss", "transient", "both"]


def run_lap_time_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    scenario: Scenario = "both",
    model_dof_override: int | None = None,
) -> dict[str, Any]:
    """Run both scenarios on one shared optimized line and return summary metrics."""

    root = repo_root()
    config = load_yaml(Path(config_path))
    track_config = _mapping(config, "track")
    line_config = _mapping(config, "racing_line")
    qss_config = _mapping(config, "qss")
    transient_config = _mapping(config, "transient")
    output_config = _mapping(config, "output")

    model_dof_value = int(
        model_dof_override
        if model_dof_override is not None
        else config.get("model_dof", 3)
    )
    if model_dof_value not in {3, 6, 10, 14}:
        raise ValueError("model_dof must be 3, 6, 10, or 14.")
    model_dof = cast(DOFModel, model_dof_value)
    format_values = {"model_dof": model_dof_value}
    vehicle_path = _root_path(root, config.get("vehicle", "vehicle.yml"))
    track_path = _root_path(root, track_config["boundary_csv"])
    ggv_path = _root_path(
        root,
        str(qss_config["ggv_csv"]).format_map(format_values),
    )
    output_directory = _root_path(
        root,
        str(output_config["directory"]).format_map(format_values),
    )
    if model_dof_override is not None and "{model_dof}" not in str(
        output_config["directory"]
    ):
        output_directory /= f"{model_dof_value}dof"
    output_directory.mkdir(parents=True, exist_ok=True)
    ggv = _load_or_generate_ggv(
        ggv_path,
        vehicle_path=vehicle_path,
        model_dof=model_dof,
        qss_config=qss_config,
    )
    corridor = TrackCorridor.from_csv(track_path)
    line_mode = cast(LineMode, str(line_config.get("mode", "minimum_time_qss")))
    optimized = optimize_racing_line(
        corridor,
        ggv,
        mode=line_mode,
        vehicle_width_m=float(track_config.get("vehicle_width_m", 1.35)),
        safety_margin_m=float(track_config.get("safety_margin_m", 0.1)),
        sample_step_m=float(track_config.get("sample_step_m", 1.0)),
        curvature_max_iterations=int(line_config.get("curvature_max_iterations", 30)),
        lap_time_max_iterations=int(line_config.get("lap_time_max_iterations", 25)),
        max_speed_mps=(
            float(qss_config["max_speed_mps"])
            if qss_config.get("max_speed_mps") is not None
            else None
        ),
    )
    write_qss_lap_csv(output_directory / "qss_lap.csv", optimized.qss_lap)
    summary: dict[str, Any] = {
        "scenario": scenario,
        "model_dof": model_dof,
        "ggv_csv": _display_path(ggv_path, root),
        "line_mode": line_mode,
        "track_length_m": optimized.line.track_length_m,
        "centerline_qss_lap_time_s": optimized.centerline_lap_time_s,
        "minimum_curvature_qss_lap_time_s": optimized.minimum_curvature_lap_time_s,
        "optimized_qss_lap_time_s": optimized.qss_lap.lap_time_s,
        "qss_converged": optimized.qss_lap.converged,
        "line_optimizer_converged": optimized.success,
        "line_optimizer_message": optimized.message,
        "line_optimizer_iterations": optimized.iterations,
    }
    if scenario in {"transient", "both"}:
        parameters = load_reduced_vehicle_parameters(vehicle_path)
        model = create_model(model_dof, parameters)
        transient = simulate_transient_lap(
            model,
            optimized.qss_lap,
            sample_period_s=float(transient_config.get("sample_period_s", 0.02)),
            lookahead_m=float(transient_config.get("lookahead_m", 5.0)),
            steering_trim_step_m=float(
                transient_config.get("steering_trim_step_m", 5.0)
            ),
        )
        write_transient_lap_csv(output_directory / "transient_lap.csv", transient)
        if transient.completed_lap:
            finish_index = int(
                np.flatnonzero(
                    transient.unwrapped_progress_m >= optimized.line.track_length_m
                )[0]
            ) + 1
        else:
            finish_index = transient.transient.time_s.size
        summary.update(
            {
                "transient_success": transient.transient.success,
                "transient_completed_lap": transient.completed_lap,
                "transient_lap_time_s": transient.lap_time_s,
                "transient_lap_delta_s": transient.lap_time_s - optimized.qss_lap.lap_time_s,
                "transient_max_abs_lateral_error_m": float(
                    np.max(np.abs(transient.lateral_error_m[:finish_index]))
                ),
                "transient_max_abs_heading_error_rad": float(
                    np.max(np.abs(transient.heading_error_rad[:finish_index]))
                ),
            }
        )
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_or_generate_ggv(
    path: Path,
    *,
    vehicle_path: Path,
    model_dof: DOFModel,
    qss_config: dict[str, Any],
) -> GGVMap:
    if path.exists():
        return GGVMap.from_csv(path)
    if not bool(qss_config.get("generate_if_missing", True)):
        raise FileNotFoundError(
            f"GGV CSV does not exist: {path}. Run make envelope-ggv or enable generation."
        )

    # This layer may compose EnvelopeSim and dyn_py; neither lower layer imports it.
    from _2_EnvelopeSim.GGV.ggv_generation import GGVConfig, generate_ggv, save_ggv_csv
    from _2_EnvelopeSim.vehicle_yaml import load_vehicle_yaml, project_vehicle_yaml

    parameters = load_reduced_vehicle_parameters(vehicle_path)
    model = create_model(model_dof, parameters)
    vehicle = project_vehicle_yaml(load_vehicle_yaml(vehicle_path)).ggv
    generation = GGVConfig(
        speeds=tuple(float(value) for value in qss_config.get("speeds_mps", (5, 10, 15, 20, 25))),
        model_dof=model_dof,
        ay_max_g=float(qss_config.get("ay_max_g", 4.0)),
        ay_points=int(qss_config.get("ay_points", 21)),
        ax_search_min_g=float(qss_config.get("ax_search_min_g", -3.2)),
        ax_search_max_g=float(qss_config.get("ax_search_max_g", 2.8)),
        ax_search_points=21,
        max_abs_beta_rad=float(qss_config.get("max_abs_beta_rad", 0.25)),
        max_abs_steering_rad=float(qss_config.get("max_abs_steering_rad", 0.5)),
        ax_binary_iterations=int(qss_config.get("ax_binary_iterations", 14)),
        verbose=True,
        progress_every=5,
        warn_tire_load_range=False,
    )
    envelopes = generate_ggv(vehicle, generation, reduced_model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ggv_csv(envelopes, path)
    return GGVMap.from_csv(path)


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be a mapping.")
    return value


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
    parser.add_argument("--scenario", choices=("qss", "transient", "both"), default="both")
    parser.add_argument("--model-dof", type=int, choices=(3, 6, 10, 14))
    parser.add_argument("--all-dof", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.all_dof and args.model_dof is not None:
        raise ValueError("Use either --all-dof or --model-dof, not both.")
    dofs = (3, 6, 10, 14) if args.all_dof else (args.model_dof,)
    summaries = [
        run_lap_time_evaluation(
            args.config,
            scenario=cast(Scenario, args.scenario),
            model_dof_override=dof,
        )
        for dof in dofs
    ]
    output: Any = summaries if args.all_dof else summaries[0]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
