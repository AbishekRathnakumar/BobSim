"""Project ``vehicle.yml`` into the parameters used by ``dyn_py``."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from _0_Utils.vehicle_io import (
    load_yaml,
    parse_tir,
    repo_root,
    tire_template_name,
    tire_templates_root,
    vehicle_yaml_path,
)
from _0_Utils.dyn_py.suspension import (
    DoubleWishboneGeometry,
)


G = 9.80665
CORNERS = ("FL", "FR", "RL", "RR")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TireParameters:
    """Small, smooth combined-slip tire projection from an MF-Tyre file."""

    fz_ref_n: float
    fz_min_n: float
    fz_max_n: float
    pdx1: float
    pdx2: float
    pdy1: float
    pdy2: float
    pkx1: float
    pkx2: float
    pkx3: float
    pky1: float
    pky2: float
    mu_floor: float = 0.8

    def mu_x(self, fz_n: FloatArray) -> FloatArray:
        fz = np.maximum(np.asarray(fz_n, dtype=float), 1.0)
        dfz = (fz - self.fz_ref_n) / self.fz_ref_n
        return np.maximum(self.pdx1 + self.pdx2 * dfz, self.mu_floor)

    def mu_y(self, fz_n: FloatArray) -> FloatArray:
        fz = np.maximum(np.asarray(fz_n, dtype=float), 1.0)
        dfz = (fz - self.fz_ref_n) / self.fz_ref_n
        return np.maximum(np.abs(self.pdy1 + self.pdy2 * dfz), self.mu_floor)

    def longitudinal_stiffness(self, fz_n: FloatArray) -> FloatArray:
        fz = np.maximum(np.asarray(fz_n, dtype=float), 1.0)
        dfz = (fz - self.fz_ref_n) / self.fz_ref_n
        scale = np.exp(self.pkx3 * dfz)
        return np.maximum(fz * (self.pkx1 + self.pkx2 * dfz) * scale, 1.0)

    def cornering_stiffness(self, fz_n: FloatArray) -> FloatArray:
        fz = np.maximum(np.asarray(fz_n, dtype=float), 1.0)
        denominator = max(abs(self.pky2) * self.fz_ref_n, 1.0)
        stiffness = abs(self.pky1) * self.fz_ref_n * np.sin(2.0 * np.arctan(fz / denominator))
        return np.maximum(stiffness, 1.0)


@dataclass(frozen=True)
class PowertrainLimits:
    """Peak and continuous contact-patch limits projected from the EV driveline."""

    peak_power_w: float
    continuous_power_w: float
    hardware_peak_power_w: float
    controller_power_limit_w: float
    peak_motor_torque_nm: float
    continuous_motor_torque_nm: float
    final_drive_ratio: float
    peak_drive_force_n: float
    continuous_drive_force_n: float
    maximum_vehicle_speed_mps: float


@dataclass(frozen=True)
class ReducedVehicleOverrides:
    """Temporary study setup applied after projecting ``vehicle.yml``.

    These values deliberately do not mutate the source vehicle definition.  The
    anti-roll fraction redistributes the existing total anti-roll stiffness
    between axles while leaving the projected wheel rates untouched.  The tire
    mu scale multiplies the MF5.2 longitudinal/lateral peak and load-sensitivity
    coefficients (PDX1/PDX2/PDY1/PDY2) plus this projection's friction floor;
    stiffness and slip-shape terms remain unchanged.  The static rear-weight
    fraction translates the equivalent total CG along the fixed wheelbase and
    preserves the global axle, contact-patch, and aero-hardware locations.
    """

    absolute_cg_height_m: float | None = None
    target_sprung_mass_kg: float | None = None
    static_rear_weight_fraction: float | None = None
    aero_balance_front: float | None = None
    brake_distribution_front: float | None = None
    front_antiroll_stiffness_fraction: float | None = None
    tire_mu_scale: float | None = None


@dataclass(frozen=True)
class ReducedVehicleParameters:
    """Parameters common to every member of the reduced-order model family."""

    mass_kg: float
    sprung_mass_kg: float
    center_of_gravity_m: tuple[float, float, float]
    inertia_kg_m2: tuple[tuple[float, float, float], ...]
    sprung_inertia_kg_m2: tuple[tuple[float, float, float], ...]
    corner_positions_m: tuple[tuple[float, float, float], ...]
    static_wheel_loads_n: tuple[float, float, float, float]
    wheel_radius_m: tuple[float, float, float, float]
    wheel_inertia_kg_m2: tuple[float, float, float, float]
    unsprung_mass_kg: tuple[float, float, float, float]
    suspension_stiffness_n_per_m: tuple[float, float, float, float]
    suspension_damping_n_s_per_m: tuple[float, float, float, float]
    antiroll_stiffness_nm_per_rad: tuple[float, float]
    double_wishbone: DoubleWishboneGeometry
    tire_vertical_stiffness_n_per_m: tuple[float, float, float, float]
    tire_vertical_damping_n_s_per_m: tuple[float, float, float, float]
    tire: TireParameters
    rho_air_kg_m3: float
    cl_area_m2: float
    cd_area_m2: float
    aero_balance_front: float
    aero_cop_m: tuple[float, float, float]
    aero_drag_application_m: tuple[float, float, float]
    peak_drive_power_w: float
    continuous_drive_power_w: float
    peak_drive_force_n: float
    continuous_drive_force_n: float
    maximum_drive_speed_mps: float
    drive_distribution_front: float
    brake_distribution_front: float

    @property
    def inertia(self) -> FloatArray:
        return np.asarray(self.inertia_kg_m2, dtype=float)

    @property
    def sprung_inertia(self) -> FloatArray:
        return np.asarray(self.sprung_inertia_kg_m2, dtype=float)

    @property
    def corner_positions(self) -> FloatArray:
        return np.asarray(self.corner_positions_m, dtype=float)

    @property
    def wheelbase_m(self) -> float:
        positions = self.corner_positions
        return float(np.mean(positions[:2, 0]) - np.mean(positions[2:, 0]))

    @property
    def track_front_m(self) -> float:
        return float(abs(self.corner_positions[0, 1] - self.corner_positions[1, 1]))

    @property
    def track_rear_m(self) -> float:
        return float(abs(self.corner_positions[2, 1] - self.corner_positions[3, 1]))

    @property
    def absolute_cg_height_m(self) -> float:
        """Total-vehicle CG z coordinate in the ``vehicle.yml`` frame."""

        return float(self.center_of_gravity_m[2])

    @property
    def static_front_weight_fraction(self) -> float:
        """Fraction of static total weight carried by the front axle."""

        loads = np.asarray(self.static_wheel_loads_n, dtype=float)
        return float(np.sum(loads[:2]) / np.sum(loads))

    @property
    def static_rear_weight_fraction(self) -> float:
        """Fraction of static total weight carried by the rear axle."""

        return 1.0 - self.static_front_weight_fraction

    @property
    def spring_roll_stiffness_nm_per_rad(self) -> tuple[float, float]:
        """Front/rear spring contribution to elastic roll stiffness."""

        rates = np.asarray(self.suspension_stiffness_n_per_m, dtype=float)
        front = 0.25 * float(np.sum(rates[:2])) * self.track_front_m**2
        rear = 0.25 * float(np.sum(rates[2:])) * self.track_rear_m**2
        return front, rear

    @property
    def elastic_roll_stiffness_nm_per_rad(self) -> tuple[float, float]:
        """Front/rear spring-plus-anti-roll elastic stiffness."""

        spring = self.spring_roll_stiffness_nm_per_rad
        return (
            spring[0] + self.antiroll_stiffness_nm_per_rad[0],
            spring[1] + self.antiroll_stiffness_nm_per_rad[1],
        )

    @property
    def front_roll_stiffness_fraction(self) -> float:
        """Front fraction of total spring-plus-anti-roll stiffness."""

        front, rear = self.elastic_roll_stiffness_nm_per_rad
        total = front + rear
        return front / total if total > 0.0 else 0.5

    @property
    def front_antiroll_stiffness_fraction(self) -> float:
        """Front fraction of the total anti-roll stiffness."""

        front, rear = self.antiroll_stiffness_nm_per_rad
        total = front + rear
        if total <= 0.0:
            return 0.5
        return front / total


def load_reduced_vehicle_parameters(
    path: str | Path | None = None,
    *,
    four_post_metrics_path: str | Path | None = None,
    overrides: ReducedVehicleOverrides | None = None,
) -> ReducedVehicleParameters:
    """Load the active vehicle and project it into the nested N-DOF family.

    ``overrides`` are study-local and are applied only to the returned frozen
    parameter object.  The source YAML and projected tire/powertrain data remain
    unchanged.
    """

    source = Path(path) if path is not None else vehicle_yaml_path()
    data = load_yaml(source)
    root = repo_root()

    components = _mass_components(data)
    total_mass, total_cg, total_inertia = _combine_mass_properties(components)

    unsprung_components = [item for item in components if item[0].endswith(".unsprung")]
    unsprung_mass_total = sum(item[1] for item in unsprung_components)
    sprung_components = [item for item in components if not item[0].endswith(".unsprung")]
    sprung_mass, _sprung_cg, sprung_inertia = _combine_mass_properties(
        sprung_components,
        reference_cg=total_cg,
    )

    front_wc = _vec3(data["front"]["suspension"]["wheel_center_m"])
    rear_wc = _vec3(data["rear"]["suspension"]["wheel_center_m"])
    front_radius = float(data["front"]["wheel"]["radius_m"])
    rear_radius = float(data["rear"]["wheel"]["radius_m"])
    front_contact_z = front_wc[2] - front_radius
    rear_contact_z = rear_wc[2] - rear_radius
    corners_global = (
        front_wc,
        (front_wc[0], -front_wc[1], front_wc[2]),
        rear_wc,
        (rear_wc[0], -rear_wc[1], rear_wc[2]),
    )
    corner_positions = tuple(
        (point[0] - total_cg[0], point[1] - total_cg[1], contact_z - total_cg[2])
        for point, contact_z in zip(
            corners_global,
            (front_contact_z, front_contact_z, rear_contact_z, rear_contact_z),
        )
    )

    wheelbase = abs(front_wc[0] - rear_wc[0])
    front_fraction = min(max((total_cg[0] - rear_wc[0]) / wheelbase, 0.01), 0.99)
    static_loads = (
        0.5 * front_fraction * total_mass * G,
        0.5 * front_fraction * total_mass * G,
        0.5 * (1.0 - front_fraction) * total_mass * G,
        0.5 * (1.0 - front_fraction) * total_mass * G,
    )

    metrics_path = (
        Path(four_post_metrics_path)
        if four_post_metrics_path is not None
        else root / "_3_StandardSim/generated_results/four_post_eval_report_metrics.csv"
    )
    metrics = _load_metrics(metrics_path)
    double_wishbone = DoubleWishboneGeometry.from_vehicle(data)
    front_spring_rate = _wheel_rate(data, "front", metrics, track=2.0 * abs(front_wc[1]))
    rear_spring_rate = _wheel_rate(data, "rear", metrics, track=2.0 * abs(rear_wc[1]))
    front_damping = _wheel_damping(data, "front", metrics)
    rear_damping = _wheel_damping(data, "rear", metrics)

    tire_name = tire_template_name(data, data["front"])
    tire_path = tire_templates_root(data) / f"{tire_name}.tir"
    tire_values = parse_tir(tire_path)
    tire = TireParameters(
        fz_ref_n=_tir_float(tire_values, "FNOMIN"),
        fz_min_n=_tir_float(tire_values, "FZMIN"),
        fz_max_n=_tir_float(tire_values, "FZMAX"),
        pdx1=_tir_float(tire_values, "PDX1"),
        pdx2=_tir_float(tire_values, "PDX2"),
        pdy1=_tir_float(tire_values, "PDY1"),
        pdy2=_tir_float(tire_values, "PDY2"),
        pkx1=_tir_float(tire_values, "PKX1"),
        pkx2=_tir_float(tire_values, "PKX2"),
        pkx3=_tir_float(tire_values, "PKX3"),
        pky1=_tir_float(tire_values, "PKY1"),
        pky2=_tir_float(tire_values, "PKY2"),
    )

    cl_area, cd_area, aero_balance, aero_cop, aero_drag_application = _project_aero(
        data,
        wheelbase,
        total_cg=total_cg,
    )
    powertrain = project_powertrain_limits(data, driven_wheel_radius_m=rear_radius)
    front_unsprung = _unsprung_mass(data, "front")
    rear_unsprung = _unsprung_mass(data, "rear")
    if not math.isclose(
        2.0 * (front_unsprung + rear_unsprung),
        unsprung_mass_total,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise ValueError("Unsprung mass projection is internally inconsistent.")

    parameters = ReducedVehicleParameters(
        mass_kg=total_mass,
        sprung_mass_kg=sprung_mass,
        center_of_gravity_m=total_cg,
        inertia_kg_m2=_matrix_tuple(total_inertia),
        sprung_inertia_kg_m2=_matrix_tuple(sprung_inertia),
        corner_positions_m=corner_positions,
        static_wheel_loads_n=static_loads,
        wheel_radius_m=(front_radius, front_radius, rear_radius, rear_radius),
        wheel_inertia_kg_m2=(
            float(data["front"]["tire"]["wheel_inertia_kg_m2"]),
            float(data["front"]["tire"]["wheel_inertia_kg_m2"]),
            float(data["rear"]["tire"]["wheel_inertia_kg_m2"]),
            float(data["rear"]["tire"]["wheel_inertia_kg_m2"]),
        ),
        unsprung_mass_kg=(front_unsprung, front_unsprung, rear_unsprung, rear_unsprung),
        suspension_stiffness_n_per_m=(
            front_spring_rate,
            front_spring_rate,
            rear_spring_rate,
            rear_spring_rate,
        ),
        suspension_damping_n_s_per_m=(
            front_damping,
            front_damping,
            rear_damping,
            rear_damping,
        ),
        antiroll_stiffness_nm_per_rad=(
            metrics.get("arb_roll_stiffness_front_Nm_per_rad", 0.0),
            metrics.get("arb_roll_stiffness_rear_Nm_per_rad", 0.0),
        ),
        double_wishbone=double_wishbone,
        tire_vertical_stiffness_n_per_m=(
            float(data["front"]["tire"]["vertical_stiffness_n_per_m"]),
            float(data["front"]["tire"]["vertical_stiffness_n_per_m"]),
            float(data["rear"]["tire"]["vertical_stiffness_n_per_m"]),
            float(data["rear"]["tire"]["vertical_stiffness_n_per_m"]),
        ),
        tire_vertical_damping_n_s_per_m=(
            float(data["front"]["tire"]["vertical_damping_n_s_per_m"]),
            float(data["front"]["tire"]["vertical_damping_n_s_per_m"]),
            float(data["rear"]["tire"]["vertical_damping_n_s_per_m"]),
            float(data["rear"]["tire"]["vertical_damping_n_s_per_m"]),
        ),
        tire=tire,
        rho_air_kg_m3=1.225,
        cl_area_m2=cl_area,
        cd_area_m2=cd_area,
        aero_balance_front=aero_balance,
        aero_cop_m=aero_cop,
        aero_drag_application_m=aero_drag_application,
        peak_drive_power_w=powertrain.peak_power_w,
        continuous_drive_power_w=powertrain.continuous_power_w,
        peak_drive_force_n=powertrain.peak_drive_force_n,
        continuous_drive_force_n=powertrain.continuous_drive_force_n,
        maximum_drive_speed_mps=powertrain.maximum_vehicle_speed_mps,
        drive_distribution_front=0.0,
        brake_distribution_front=0.84,
    )
    if overrides is None:
        return parameters
    return apply_reduced_vehicle_overrides(parameters, overrides)


def apply_reduced_vehicle_overrides(
    parameters: ReducedVehicleParameters,
    overrides: ReducedVehicleOverrides,
) -> ReducedVehicleParameters:
    """Return a study-local setup without mutating the baseline parameters."""

    updated = parameters

    if overrides.absolute_cg_height_m is not None:
        target_height = _finite_float(
            overrides.absolute_cg_height_m,
            "absolute_cg_height_m",
        )
        current_height = updated.absolute_cg_height_m
        contact_heights = np.asarray(updated.corner_positions_m, dtype=float)[:, 2] + current_height
        if target_height <= float(np.max(contact_heights)):
            raise ValueError("absolute_cg_height_m must remain above every tire contact patch.")
        height_delta = target_height - current_height
        updated = replace(
            updated,
            center_of_gravity_m=(
                updated.center_of_gravity_m[0],
                updated.center_of_gravity_m[1],
                target_height,
            ),
            corner_positions_m=tuple((x, y, z - height_delta) for x, y, z in updated.corner_positions_m),
            aero_cop_m=(
                updated.aero_cop_m[0],
                updated.aero_cop_m[1],
                updated.aero_cop_m[2] - height_delta,
            ),
            aero_drag_application_m=(
                updated.aero_drag_application_m[0],
                updated.aero_drag_application_m[1],
                updated.aero_drag_application_m[2] - height_delta,
            ),
        )

    if overrides.target_sprung_mass_kg is not None:
        target_sprung_mass_kg = _positive_float(
            overrides.target_sprung_mass_kg,
            "target_sprung_mass_kg",
        )
        if target_sprung_mass_kg != updated.sprung_mass_kg:
            sprung_mass_scale = target_sprung_mass_kg / updated.sprung_mass_kg
            baseline_sprung_inertia = updated.sprung_inertia
            baseline_total_inertia = updated.inertia
            unsprung_inertia = baseline_total_inertia - baseline_sprung_inertia
            scaled_sprung_inertia = sprung_mass_scale * baseline_sprung_inertia
            scaled_total_inertia = unsprung_inertia + scaled_sprung_inertia
            total_mass_kg = target_sprung_mass_kg + sum(updated.unsprung_mass_kg)
            baseline_static_loads = np.asarray(
                updated.static_wheel_loads_n,
                dtype=float,
            )
            front_static_fraction = float(np.sum(baseline_static_loads[:2]) / np.sum(baseline_static_loads))
            updated = replace(
                updated,
                mass_kg=total_mass_kg,
                sprung_mass_kg=target_sprung_mass_kg,
                inertia_kg_m2=_matrix_tuple(scaled_total_inertia),
                sprung_inertia_kg_m2=_matrix_tuple(scaled_sprung_inertia),
                static_wheel_loads_n=(
                    0.5 * front_static_fraction * total_mass_kg * G,
                    0.5 * front_static_fraction * total_mass_kg * G,
                    0.5 * (1.0 - front_static_fraction) * total_mass_kg * G,
                    0.5 * (1.0 - front_static_fraction) * total_mass_kg * G,
                ),
            )

    if overrides.static_rear_weight_fraction is not None:
        rear_fraction = _open_unit_interval(
            overrides.static_rear_weight_fraction,
            "static_rear_weight_fraction",
        )
        if not math.isclose(
            rear_fraction,
            updated.static_rear_weight_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            positions = updated.corner_positions
            front_axle_body_x = float(np.mean(positions[:2, 0]))
            rear_axle_body_x = float(np.mean(positions[2:, 0]))
            wheelbase = front_axle_body_x - rear_axle_body_x
            if wheelbase <= 0.0:
                raise ValueError("static_rear_weight_fraction requires a positive wheelbase.")
            current_cg_x = float(updated.center_of_gravity_m[0])
            rear_axle_global_x = current_cg_x + rear_axle_body_x
            target_cg_x = rear_axle_global_x + (1.0 - rear_fraction) * wheelbase
            cg_x_delta = target_cg_x - current_cg_x
            front_fraction = 1.0 - rear_fraction
            updated = replace(
                updated,
                center_of_gravity_m=(
                    target_cg_x,
                    updated.center_of_gravity_m[1],
                    updated.center_of_gravity_m[2],
                ),
                corner_positions_m=tuple((x - cg_x_delta, y, z) for x, y, z in updated.corner_positions_m),
                aero_cop_m=(
                    updated.aero_cop_m[0] - cg_x_delta,
                    updated.aero_cop_m[1],
                    updated.aero_cop_m[2],
                ),
                aero_drag_application_m=(
                    updated.aero_drag_application_m[0] - cg_x_delta,
                    updated.aero_drag_application_m[1],
                    updated.aero_drag_application_m[2],
                ),
                static_wheel_loads_n=(
                    0.5 * front_fraction * updated.mass_kg * G,
                    0.5 * front_fraction * updated.mass_kg * G,
                    0.5 * rear_fraction * updated.mass_kg * G,
                    0.5 * rear_fraction * updated.mass_kg * G,
                ),
            )

    if overrides.aero_balance_front is not None:
        front_fraction = _unit_interval(
            overrides.aero_balance_front,
            "aero_balance_front",
        )
        positions = updated.corner_positions
        front_x = float(np.mean(positions[:2, 0]))
        rear_x = float(np.mean(positions[2:, 0]))
        cop_x = rear_x + front_fraction * (front_x - rear_x)
        updated = replace(
            updated,
            aero_balance_front=front_fraction,
            aero_cop_m=(cop_x, updated.aero_cop_m[1], updated.aero_cop_m[2]),
        )

    if overrides.brake_distribution_front is not None:
        updated = replace(
            updated,
            brake_distribution_front=_unit_interval(
                overrides.brake_distribution_front,
                "brake_distribution_front",
            ),
        )

    if overrides.front_antiroll_stiffness_fraction is not None:
        target_fraction = _unit_interval(
            overrides.front_antiroll_stiffness_fraction,
            "front_antiroll_stiffness_fraction",
        )
        total_stiffness = sum(updated.antiroll_stiffness_nm_per_rad)
        updated = replace(
            updated,
            antiroll_stiffness_nm_per_rad=(
                target_fraction * total_stiffness,
                (1.0 - target_fraction) * total_stiffness,
            ),
        )

    if overrides.tire_mu_scale is not None:
        tire_mu_scale = _positive_float(
            overrides.tire_mu_scale,
            "tire_mu_scale",
        )
        if tire_mu_scale != 1.0:
            tire = updated.tire
            updated = replace(
                updated,
                tire=replace(
                    tire,
                    pdx1=tire.pdx1 * tire_mu_scale,
                    pdx2=tire.pdx2 * tire_mu_scale,
                    pdy1=tire.pdy1 * tire_mu_scale,
                    pdy2=tire.pdy2 * tire_mu_scale,
                    mu_floor=tire.mu_floor * tire_mu_scale,
                ),
            )

    return updated


def project_powertrain_limits(
    data: Mapping[str, Any],
    *,
    driven_wheel_radius_m: float,
) -> PowertrainLimits:
    """Project motor, inverter, VCU, gearing, and rpm limits to the road."""

    powertrain = data.get("powertrain")
    if not isinstance(powertrain, Mapping):
        return PowertrainLimits(
            peak_power_w=math.inf,
            continuous_power_w=math.inf,
            hardware_peak_power_w=math.inf,
            controller_power_limit_w=math.inf,
            peak_motor_torque_nm=math.inf,
            continuous_motor_torque_nm=math.inf,
            final_drive_ratio=1.0,
            peak_drive_force_n=math.inf,
            continuous_drive_force_n=math.inf,
            maximum_vehicle_speed_mps=math.inf,
        )
    motor = powertrain["pMotor"]
    inverter = powertrain["pInverter"]
    vcu = powertrain["pVCU"]
    driveline = powertrain["pDriveline"]
    ratio = abs(float(driveline["finalDriveRatio"]))
    radius = float(driven_wheel_radius_m)
    if ratio <= 0.0 or radius <= 0.0:
        raise ValueError("Powertrain projection requires positive gearing and wheel radius.")

    peak_torque = min(float(motor["T_peak"]), float(vcu["tau_max"]))
    continuous_torque = min(float(motor["T_cont"]), float(vcu["tau_max"]))
    hardware_peak_power = min(
        float(motor["P_mech_peak"]),
        float(inverter["P_max_mot"]),
    )
    controller_power_limit = float(vcu.get("P_max_mot", math.inf))
    peak_power = min(hardware_peak_power, controller_power_limit)
    continuous_power = min(
        float(motor["P_cont_low"]),
        float(motor["P_cont_high"]),
        float(inverter["P_max_mot"]),
        controller_power_limit,
    )
    maximum_motor_speed_radps = float(motor["rpm_max_peak"]) * 2.0 * math.pi / 60.0
    return PowertrainLimits(
        peak_power_w=peak_power,
        continuous_power_w=continuous_power,
        hardware_peak_power_w=hardware_peak_power,
        controller_power_limit_w=controller_power_limit,
        peak_motor_torque_nm=peak_torque,
        continuous_motor_torque_nm=continuous_torque,
        final_drive_ratio=ratio,
        peak_drive_force_n=peak_torque * ratio / radius,
        continuous_drive_force_n=continuous_torque * ratio / radius,
        maximum_vehicle_speed_mps=maximum_motor_speed_radps * radius / ratio,
    )


MassComponent = tuple[str, float, tuple[float, float, float], FloatArray]


def _mass_components(data: Mapping[str, Any]) -> list[MassComponent]:
    components: list[MassComponent] = []
    for name in ("sprung_mass", "driver_mass"):
        section = data[name]
        components.append((name, float(section["mass_kg"]), _vec3(section["cg_m"]), _matrix(section["inertia_kg_m2"])))

    mirror = np.diag([1.0, -1.0, 1.0])
    for axle in ("front", "rear"):
        for name, section in data[axle]["masses"].items():
            mass = float(section["mass_kg"])
            cg_left = _vec3(section["cg_m"])
            inertia_left = _matrix(section["inertia_kg_m2"])
            cg_right = (cg_left[0], -cg_left[1], cg_left[2])
            inertia_right = mirror @ inertia_left @ mirror
            components.append((f"{axle}.{name}", mass, cg_left, inertia_left))
            components.append((f"{axle}.{name}", mass, cg_right, inertia_right))
    return components


def _combine_mass_properties(
    components: Iterable[MassComponent],
    *,
    reference_cg: Sequence[float] | None = None,
) -> tuple[float, tuple[float, float, float], FloatArray]:
    items = list(components)
    mass = float(sum(item[1] for item in items))
    if mass <= 0.0:
        raise ValueError("Reduced vehicle mass must be positive.")
    if reference_cg is None:
        cg_array = sum((item[1] * np.asarray(item[2]) for item in items), np.zeros(3)) / mass
    else:
        cg_array = np.asarray(reference_cg, dtype=float)

    inertia = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for _name, component_mass, component_cg, component_inertia in items:
        offset = np.asarray(component_cg, dtype=float) - cg_array
        inertia += component_inertia + component_mass * (float(offset @ offset) * identity - np.outer(offset, offset))
    cg = (float(cg_array[0]), float(cg_array[1]), float(cg_array[2]))
    return mass, cg, inertia


def _wheel_rate(
    data: Mapping[str, Any],
    axle: str,
    metrics: Mapping[str, float],
    *,
    track: float,
) -> float:
    metric = metrics.get(f"spring_roll_stiffness_{axle}_Nm_per_rad")
    if metric is not None and metric > 0.0:
        return 2.0 * metric / track**2
    shock_rate = _table_slope(data[axle]["actuation"]["shock"]["spring_table"]["table"])
    motion_ratio = metrics.get(f"static_motion_ratio_{axle}", 1.0)
    return shock_rate / max(motion_ratio**2, 1e-9)


def _wheel_damping(
    data: Mapping[str, Any],
    axle: str,
    metrics: Mapping[str, float],
) -> float:
    shock_damping = _table_slope(data[axle]["actuation"]["shock"]["damper_table"]["table"])
    motion_ratio = metrics.get(f"static_motion_ratio_{axle}", 1.0)
    return shock_damping / max(motion_ratio**2, 1e-9)


def _table_slope(table: Sequence[Sequence[float]]) -> float:
    x0, y0 = (float(value) for value in table[0])
    x1, y1 = (float(value) for value in table[-1])
    if math.isclose(x0, x1):
        raise ValueError("Cannot project a rate from a zero-width table.")
    return (y1 - y0) / (x1 - x0)


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    metrics: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                metrics[str(row["metric"])] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return metrics


def _project_aero(
    data: Mapping[str, Any],
    wheelbase: float,
    *,
    total_cg: tuple[float, float, float],
) -> tuple[
    float,
    float,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    aero = data.get("aero")
    if not isinstance(aero, Mapping):
        origin = (0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.5, origin, origin
    front_grid = [float(value) for value in aero["front_ride_height_grid_m"]]
    rear_grid = [float(value) for value in aero["rear_ride_height_grid_m"]]
    front_height = float(aero.get("nominal_front_ride_height_m", 0.0762))
    rear_height = float(aero.get("nominal_rear_ride_height_m", 0.0762))
    downforce = _bilinear(front_grid, rear_grid, aero["downforce_table_n"], front_height, rear_height)
    drag = _bilinear(front_grid, rear_grid, aero["drag_table_n"], front_height, rear_height)
    pitch_moment = _bilinear(front_grid, rear_grid, aero["my_table_nm"], front_height, rear_height)
    speed = float(aero["reference_speed_m_per_s"])
    dynamic_pressure = 0.5 * 1.225 * speed**2
    cl_area = downforce / dynamic_pressure
    cd_area = drag / dynamic_pressure

    aero_ref = _vec3(aero.get("aero_ref_m", [0.0, 0.0, 0.0]))
    aero_ref_x = aero_ref[0]
    front_x = float(data["front"]["suspension"]["wheel_center_m"][0])
    if abs(downforce) <= 1e-9:
        balance = 0.5
    else:
        # ``my_table_nm`` is a free pitch moment applied at ``aero_ref_m``.
        # A downward force forward of the front axle creates a positive pitch moment in
        # BobLib's x-forward/z-up frame, so first translate the free moment to
        # the front axle and then replace the pair with an equivalent CoP.
        total_pitch_about_front = pitch_moment + (aero_ref_x - front_x) * downforce
        cop_from_front = -total_pitch_about_front / downforce
        balance = 1.0 - cop_from_front / wheelbase
    cop_global_x = front_x - cop_from_front
    cop_body = (
        cop_global_x - total_cg[0],
        aero_ref[1] - total_cg[1],
        aero_ref[2] - total_cg[2],
    )
    aero_ref_body = (
        float(aero_ref[0] - total_cg[0]),
        float(aero_ref[1] - total_cg[1]),
        float(aero_ref[2] - total_cg[2]),
    )
    return (
        cl_area,
        cd_area,
        balance,
        cop_body,
        aero_ref_body,
    )


def _bilinear(
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    table: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> float:
    x0, x1, tx = _bracket(x_grid, x)
    y0, y1, ty = _bracket(y_grid, y)
    values = np.asarray(table, dtype=float)
    return float(
        (1.0 - tx) * (1.0 - ty) * values[x0, y0]
        + tx * (1.0 - ty) * values[x1, y0]
        + (1.0 - tx) * ty * values[x0, y1]
        + tx * ty * values[x1, y1]
    )


def _bracket(grid: Sequence[float], value: float) -> tuple[int, int, float]:
    values = np.asarray(grid, dtype=float)
    if value <= values[0]:
        return 0, 1, 0.0
    if value >= values[-1]:
        return len(values) - 2, len(values) - 1, 1.0
    upper = int(np.searchsorted(values, value))
    lower = upper - 1
    fraction = (value - values[lower]) / (values[upper] - values[lower])
    return lower, upper, float(fraction)


def _unsprung_mass(data: Mapping[str, Any], axle: str) -> float:
    return float(data[axle]["masses"]["unsprung"]["mass_kg"])


def _tir_float(values: Mapping[str, float | str], key: str) -> float:
    try:
        return float(values[key])
    except KeyError as exc:
        raise ValueError(f"Tire file is missing required coefficient {key}.") from exc


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _unit_interval(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")
    return result


def _open_unit_interval(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return result


def _vec3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected a three-vector, got {values!r}.")
    return float(values[0]), float(values[1]), float(values[2])


def _matrix(values: Sequence[Sequence[float]]) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 inertia matrix, got shape {matrix.shape}.")
    return matrix


def _matrix_tuple(matrix: FloatArray) -> tuple[tuple[float, float, float], ...]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )
