from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from _0_Utils.dyn_py import (
    ModelInputs,
    ReducedVehicleOverrides,
    apply_reduced_vehicle_overrides,
    compare_transient_signals,
    create_model,
    load_reduced_vehicle_parameters,
    simulate_transient,
    solve_acceleration_trim,
    solve_moment_state,
    solve_steady_state,
)
from _0_Utils.dyn_py.models import (
    VehicleModel3DOF,
    VehicleModel6DOF,
    VehicleModel10DOF,
    VehicleModel14DOF,
)
from _0_Utils.dyn_py.parameters import G
from _0_Utils.vehicle_io import (
    load_yaml,
    parse_tir,
    tire_template_name,
    tire_templates_root,
    vehicle_yaml_path,
)
from _2_EnvelopeSim.GGV.ggv_generation import (
    GGVConfig,
    _largest_feasible_value,
    generate_ggv,
    solve_ax_limit,
    solve_lateral_limit,
)
from _2_EnvelopeSim.YMD.ymd_generation import YMDConfig, ymd_point
from _2_EnvelopeSim.vehicle_yaml import load_vehicle_yaml, project_vehicle_yaml


@pytest.fixture(scope="module")
def parameters():
    return load_reduced_vehicle_parameters()


@pytest.mark.parametrize(
    ("dof", "state_size", "added_coordinates"),
    (
        (3, 6, ()),
        (6, 12, ("z", "roll", "pitch")),
        (10, 20, ("wheel_angle_fl", "wheel_angle_rr")),
        (14, 28, ("unsprung_z_fl", "unsprung_z_rr")),
    ),
)
def test_nested_model_state_contract(parameters, dof, state_size, added_coordinates):
    model = create_model(dof, parameters)
    assert model.state_size == state_size
    assert len(model.coordinate_names) == dof
    assert len(model.velocity_names) == dof
    for name in added_coordinates:
        assert name in model.coordinate_names

    state = model.initial_state(15.0)
    output = model.evaluate(state)
    assert output.derivative.shape == (state_size,)
    assert output.generalized_acceleration.shape == (dof,)
    assert output.wheel_forces_body_n.shape == (4, 3)
    assert output.geometric_vertical_forces_n.shape == (4,)
    assert output.algebraic_load_transfer_n.shape == (4,)
    assert np.all(np.isfinite(output.derivative))


def test_model_classes_make_the_fidelity_ladder_explicit():
    assert issubclass(VehicleModel6DOF, VehicleModel3DOF)
    assert issubclass(VehicleModel10DOF, VehicleModel6DOF)
    assert issubclass(VehicleModel14DOF, VehicleModel10DOF)

    models = (VehicleModel3DOF, VehicleModel6DOF, VehicleModel10DOF, VehicleModel14DOF)
    assert [model.dof for model in models] == [3, 6, 10, 14]
    assert all(model.added_physics for model in models)


def test_dyn_py_aero_projection_matches_envelope_projection(parameters):
    projection = project_vehicle_yaml(load_vehicle_yaml())

    assert parameters.cl_area_m2 == pytest.approx(projection.ggv.cl_a)
    assert parameters.cd_area_m2 == pytest.approx(projection.ggv.cd_a)
    assert parameters.aero_balance_front == pytest.approx(projection.ggv.aero_balance_front)
    assert np.isfinite(parameters.aero_balance_front)


def test_vehicle_powertrain_is_projected_to_contact_patch(parameters):
    projection = project_vehicle_yaml(load_vehicle_yaml())

    assert parameters.peak_drive_power_w == pytest.approx(80_000.0)
    assert parameters.continuous_drive_power_w == pytest.approx(75_000.0)
    assert parameters.peak_drive_force_n == pytest.approx(220.0 * 3.31 / parameters.wheel_radius_m[2])
    assert parameters.maximum_drive_speed_mps == pytest.approx(
        6500.0 * 2.0 * np.pi / 60.0 * parameters.wheel_radius_m[2] / 3.31
    )
    assert projection.ggv.max_drive_power == pytest.approx(parameters.peak_drive_power_w)
    assert projection.summary["hardware_peak_drive_power_w"] == pytest.approx(124_000.0)
    assert projection.summary["controller_drive_power_limit_w"] == pytest.approx(80_000.0)
    assert projection.ggv.max_drive_force == pytest.approx(parameters.peak_drive_force_n)
    assert projection.ggv.max_drive_speed == pytest.approx(parameters.maximum_drive_speed_mps)


def test_study_overrides_are_scoped_and_preserve_projected_physics(parameters):
    source_bytes = vehicle_yaml_path().read_bytes()
    original_cop_global_z = parameters.absolute_cg_height_m + parameters.aero_cop_m[2]
    original_aero_ref_global_z = parameters.absolute_cg_height_m + parameters.aero_drag_application_m[2]
    original_contact_heights = (
        parameters.absolute_cg_height_m + np.asarray(parameters.corner_positions_m, dtype=float)[:, 2]
    )
    spring_front, spring_rear = parameters.spring_roll_stiffness_nm_per_rad
    total_antiroll = sum(parameters.antiroll_stiffness_nm_per_rad)
    total_elastic = sum(parameters.elastic_roll_stiffness_nm_per_rad)

    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            absolute_cg_height_m=8.0 * 0.0254,
            aero_balance_front=0.5,
            brake_distribution_front=0.67,
            front_antiroll_stiffness_fraction=0.75,
        ),
    )

    assert tuned is not parameters
    assert tuned.absolute_cg_height_m == pytest.approx(8.0 * 0.0254)
    assert parameters.brake_distribution_front == pytest.approx(0.84)
    assert tuned.brake_distribution_front == pytest.approx(0.67)
    assert tuned.aero_balance_front == pytest.approx(0.5)
    assert tuned.aero_cop_m[0] == pytest.approx(
        0.5 * (np.mean(tuned.corner_positions[:2, 0]) + np.mean(tuned.corner_positions[2:, 0]))
    )
    assert tuned.absolute_cg_height_m + tuned.aero_cop_m[2] == pytest.approx(original_cop_global_z)
    assert tuned.absolute_cg_height_m + tuned.aero_drag_application_m[2] == pytest.approx(original_aero_ref_global_z)
    np.testing.assert_allclose(
        tuned.absolute_cg_height_m + np.asarray(tuned.corner_positions_m, dtype=float)[:, 2],
        original_contact_heights,
    )

    assert tuned.suspension_stiffness_n_per_m == parameters.suspension_stiffness_n_per_m
    assert sum(tuned.antiroll_stiffness_nm_per_rad) == pytest.approx(total_antiroll)
    assert tuned.front_antiroll_stiffness_fraction == pytest.approx(0.75)
    expected_elastic_front_fraction = (spring_front + 0.75 * total_antiroll) / total_elastic
    assert tuned.front_roll_stiffness_fraction == pytest.approx(expected_elastic_front_fraction)
    assert tuned.spring_roll_stiffness_nm_per_rad == pytest.approx((spring_front, spring_rear))

    assert tuned.cl_area_m2 == parameters.cl_area_m2
    assert tuned.cd_area_m2 == parameters.cd_area_m2
    assert tuned.tire is parameters.tire
    assert tuned.peak_drive_power_w == parameters.peak_drive_power_w
    assert tuned.peak_drive_force_n == parameters.peak_drive_force_n
    assert tuned.maximum_drive_speed_mps == parameters.maximum_drive_speed_mps
    assert tuned.drive_distribution_front == parameters.drive_distribution_front
    assert vehicle_yaml_path().read_bytes() == source_bytes


def test_reduced_vehicle_loader_keeps_raw_tir_coefficients_with_overrides():
    data = load_yaml(vehicle_yaml_path())
    tire_name = tire_template_name(data, data["front"])
    tire_values = parse_tir(tire_templates_root(data) / f"{tire_name}.tir")

    parameters = load_reduced_vehicle_parameters(overrides=ReducedVehicleOverrides(aero_balance_front=0.5))

    assert parameters.tire.pdx1 == pytest.approx(float(tire_values["PDX1"]))
    assert parameters.tire.pdx2 == pytest.approx(float(tire_values["PDX2"]))
    assert parameters.tire.pdy1 == pytest.approx(float(tire_values["PDY1"]))
    assert parameters.tire.pdy2 == pytest.approx(float(tire_values["PDY2"]))


def test_reduced_vehicle_tire_mu_scale_matches_legacy_semantics(parameters):
    scale = 0.6225437130779028
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(tire_mu_scale=scale),
    )

    assert tuned.tire.pdx1 == pytest.approx(parameters.tire.pdx1 * scale)
    assert tuned.tire.pdx2 == pytest.approx(parameters.tire.pdx2 * scale)
    assert tuned.tire.pdy1 == pytest.approx(parameters.tire.pdy1 * scale)
    assert tuned.tire.pdy2 == pytest.approx(parameters.tire.pdy2 * scale)
    assert tuned.tire.mu_floor == pytest.approx(parameters.tire.mu_floor * scale)

    preserved_fields = (
        "fz_ref_n",
        "fz_min_n",
        "fz_max_n",
        "pkx1",
        "pkx2",
        "pkx3",
        "pky1",
        "pky2",
    )
    for field_name in preserved_fields:
        assert getattr(tuned.tire, field_name) == getattr(parameters.tire, field_name)

    loads_n = np.asarray([100.0, parameters.tire.fz_ref_n, 1800.0])
    np.testing.assert_allclose(
        tuned.tire.mu_x(loads_n),
        scale * parameters.tire.mu_x(loads_n),
    )
    np.testing.assert_allclose(
        tuned.tire.mu_y(loads_n),
        scale * parameters.tire.mu_y(loads_n),
    )
    np.testing.assert_allclose(
        tuned.tire.longitudinal_stiffness(loads_n),
        parameters.tire.longitudinal_stiffness(loads_n),
    )
    np.testing.assert_allclose(
        tuned.tire.cornering_stiffness(loads_n),
        parameters.tire.cornering_stiffness(loads_n),
    )


def test_unit_tire_mu_scale_is_an_identity_override(parameters):
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(tire_mu_scale=1.0),
    )

    assert tuned is parameters
    assert tuned.tire is parameters.tire


def test_sprung_mass_override_preserves_cg_and_unsprung_mass(parameters):
    target_sprung_mass_kg = parameters.sprung_mass_kg + 20.0
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            absolute_cg_height_m=11.5 * 0.0254,
            target_sprung_mass_kg=target_sprung_mass_kg,
        ),
    )

    sprung_scale = target_sprung_mass_kg / parameters.sprung_mass_kg
    baseline_unsprung_inertia = parameters.inertia - parameters.sprung_inertia
    expected_total_inertia = baseline_unsprung_inertia + sprung_scale * parameters.sprung_inertia
    expected_total_mass_kg = target_sprung_mass_kg + sum(parameters.unsprung_mass_kg)
    baseline_front_fraction = sum(parameters.static_wheel_loads_n[:2]) / sum(parameters.static_wheel_loads_n)

    assert tuned.sprung_mass_kg == pytest.approx(target_sprung_mass_kg)
    assert tuned.mass_kg == pytest.approx(expected_total_mass_kg)
    assert tuned.absolute_cg_height_m == pytest.approx(11.5 * 0.0254)
    assert tuned.center_of_gravity_m[:2] == parameters.center_of_gravity_m[:2]
    assert tuned.unsprung_mass_kg == parameters.unsprung_mass_kg
    np.testing.assert_allclose(
        tuned.sprung_inertia,
        sprung_scale * parameters.sprung_inertia,
    )
    np.testing.assert_allclose(tuned.inertia, expected_total_inertia)
    assert sum(tuned.static_wheel_loads_n) == pytest.approx(expected_total_mass_kg * G)
    assert sum(tuned.static_wheel_loads_n[:2]) / sum(tuned.static_wheel_loads_n) == pytest.approx(
        baseline_front_fraction
    )
    assert tuned.suspension_stiffness_n_per_m == parameters.suspension_stiffness_n_per_m
    assert tuned.suspension_damping_n_s_per_m == parameters.suspension_damping_n_s_per_m
    assert tuned.antiroll_stiffness_nm_per_rad == parameters.antiroll_stiffness_nm_per_rad


def test_nominal_sprung_mass_override_is_an_identity(parameters):
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(target_sprung_mass_kg=parameters.sprung_mass_kg),
    )

    assert tuned is parameters


@pytest.mark.parametrize("rear_fraction", (0.45, 0.50, 0.55))
def test_static_rear_weight_override_moves_cg_without_moving_chassis_hardware(parameters, rear_fraction):
    original_axles_global_x = (
        parameters.center_of_gravity_m[0] + np.mean(parameters.corner_positions[:2, 0]),
        parameters.center_of_gravity_m[0] + np.mean(parameters.corner_positions[2:, 0]),
    )
    original_cop_global = np.asarray(parameters.center_of_gravity_m) + np.asarray(parameters.aero_cop_m)
    original_drag_global = np.asarray(parameters.center_of_gravity_m) + np.asarray(parameters.aero_drag_application_m)

    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(static_rear_weight_fraction=rear_fraction),
    )

    tuned_axles_global_x = (
        tuned.center_of_gravity_m[0] + np.mean(tuned.corner_positions[:2, 0]),
        tuned.center_of_gravity_m[0] + np.mean(tuned.corner_positions[2:, 0]),
    )
    assert tuned.static_rear_weight_fraction == pytest.approx(rear_fraction)
    assert tuned.static_front_weight_fraction == pytest.approx(1.0 - rear_fraction)
    assert tuned_axles_global_x == pytest.approx(original_axles_global_x)
    np.testing.assert_allclose(
        np.asarray(tuned.center_of_gravity_m) + np.asarray(tuned.aero_cop_m),
        original_cop_global,
    )
    np.testing.assert_allclose(
        np.asarray(tuned.center_of_gravity_m) + np.asarray(tuned.aero_drag_application_m),
        original_drag_global,
    )
    assert tuned.center_of_gravity_m[1:] == parameters.center_of_gravity_m[1:]
    np.testing.assert_allclose(tuned.inertia, parameters.inertia)
    np.testing.assert_allclose(tuned.sprung_inertia, parameters.sprung_inertia)
    assert tuned.mass_kg == parameters.mass_kg
    assert tuned.sprung_mass_kg == parameters.sprung_mass_kg
    assert sum(tuned.static_wheel_loads_n) == pytest.approx(parameters.mass_kg * G)


def test_nominal_static_rear_weight_override_is_identity(parameters):
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(static_rear_weight_fraction=parameters.static_rear_weight_fraction),
    )

    assert tuned is parameters


@pytest.mark.parametrize(
    "overrides",
    (
        ReducedVehicleOverrides(absolute_cg_height_m=float("nan")),
        ReducedVehicleOverrides(absolute_cg_height_m=-1.0),
        ReducedVehicleOverrides(aero_balance_front=-0.01),
        ReducedVehicleOverrides(brake_distribution_front=1.01),
        ReducedVehicleOverrides(front_antiroll_stiffness_fraction=1.01),
        ReducedVehicleOverrides(target_sprung_mass_kg=0.0),
        ReducedVehicleOverrides(target_sprung_mass_kg=-1.0),
        ReducedVehicleOverrides(target_sprung_mass_kg=float("nan")),
        ReducedVehicleOverrides(target_sprung_mass_kg=float("inf")),
        ReducedVehicleOverrides(static_rear_weight_fraction=0.0),
        ReducedVehicleOverrides(static_rear_weight_fraction=1.0),
        ReducedVehicleOverrides(static_rear_weight_fraction=float("nan")),
        ReducedVehicleOverrides(tire_mu_scale=0.0),
        ReducedVehicleOverrides(tire_mu_scale=-0.01),
        ReducedVehicleOverrides(tire_mu_scale=float("nan")),
        ReducedVehicleOverrides(tire_mu_scale=float("inf")),
    ),
)
def test_reduced_vehicle_overrides_reject_nonphysical_values(parameters, overrides):
    with pytest.raises(ValueError):
        apply_reduced_vehicle_overrides(parameters, overrides)


def test_dyn_py_applies_aero_at_projected_cop(parameters):
    model = create_model(6, parameters)
    projection = project_vehicle_yaml(load_vehicle_yaml())
    speed_mps = float(projection.summary["reference_speed_m_per_s"])
    body_velocities = np.array([speed_mps, 0.0, 0.0, 0.0, 0.0, 0.0])

    force, moment = model._aero_load(body_velocities)

    dynamic_pressure = 0.5 * parameters.rho_air_kg_m3 * speed_mps**2
    drag_force = np.array([-dynamic_pressure * parameters.cd_area_m2, 0.0, 0.0])
    downforce_force = np.array([0.0, 0.0, -dynamic_pressure * parameters.cl_area_m2])
    expected_moment = np.cross(parameters.aero_cop_m, downforce_force)
    expected_moment += np.cross(parameters.aero_drag_application_m, drag_force)

    np.testing.assert_allclose(force, drag_force + downforce_force)
    np.testing.assert_allclose(moment, expected_moment)

    # The CoP representation must preserve the original BobLib convention:
    # force at aero_ref_m plus the tabulated free pitch moment at that point.
    free_moment = np.array([0.0, float(projection.summary["my_nm"]), 0.0])
    source_wrench_moment = (
        np.cross(
            parameters.aero_drag_application_m,
            drag_force + downforce_force,
        )
        + free_moment
    )
    np.testing.assert_allclose(moment, source_wrench_moment)


def test_3dof_and_6dof_initial_planar_response_remains_same_order(parameters):
    no_aero = replace(
        parameters,
        cl_area_m2=0.0,
        cd_area_m2=0.0,
    )
    model_3 = create_model(3, no_aero)
    model_6 = create_model(6, no_aero)
    inputs = ModelInputs(steering_rad=0.025)

    output_3 = model_3.evaluate(model_3.initial_state(15.0), inputs)
    output_6 = model_6.evaluate(model_6.initial_state(15.0), inputs)

    np.testing.assert_allclose(
        output_3.generalized_acceleration,
        output_6.generalized_acceleration[[0, 1, 5]],
        rtol=1e-1,
        atol=1e-3,
    )


def test_3dof_has_algebraic_load_transfer_and_rear_wheel_drive(parameters):
    model = create_model(3, parameters)
    static = solve_acceleration_trim(
        model,
        speed_mps=12.0,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=0.0,
    )
    accelerating = solve_acceleration_trim(
        model,
        speed_mps=12.0,
        longitudinal_acceleration_mps2=G,
        lateral_acceleration_mps2=0.0,
    )
    cornering = solve_acceleration_trim(
        model,
        speed_mps=12.0,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=1.5 * G,
    )

    assert static.success and accelerating.success and cornering.success
    assert np.sum(accelerating.output.normal_loads_n[:2]) < np.sum(static.output.normal_loads_n[:2])
    assert np.sum(accelerating.output.normal_loads_n[2:]) > np.sum(static.output.normal_loads_n[2:])
    assert cornering.output.normal_loads_n[1] > cornering.output.normal_loads_n[0]
    assert cornering.output.normal_loads_n[3] > cornering.output.normal_loads_n[2]
    np.testing.assert_allclose(accelerating.inputs.wheel_torques_nm[:2], 0.0)
    assert np.all(np.asarray(accelerating.inputs.wheel_torques_nm[2:]) > 0.0)


def test_flat_road_qss_loads_converge_across_fidelity_ladder(parameters):
    results = {
        dof: solve_acceleration_trim(
            create_model(dof, parameters),
            speed_mps=12.0,
            longitudinal_acceleration_mps2=0.5 * G,
            lateral_acceleration_mps2=G,
        )
        for dof in (3, 6, 10, 14)
    }
    assert all(result.success for result in results.values())
    np.testing.assert_allclose(
        results[3].output.normal_loads_n,
        results[6].output.normal_loads_n,
        rtol=0.06,
        atol=15.0,
    )
    np.testing.assert_allclose(
        results[6].output.normal_loads_n,
        results[10].output.normal_loads_n,
        rtol=1e-6,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        results[10].output.normal_loads_n,
        results[14].output.normal_loads_n,
        rtol=0.04,
        atol=15.0,
    )


@pytest.mark.parametrize(("dof", "wheel_slice"), ((10, slice(6, 10)), (14, slice(10, 14))))
def test_qss_rotating_wheels_follow_vehicle_acceleration(parameters, dof, wheel_slice):
    result = solve_acceleration_trim(
        create_model(dof, parameters),
        speed_mps=12.0,
        longitudinal_acceleration_mps2=G,
        lateral_acceleration_mps2=0.0,
    )

    assert result.success
    expected = G / np.asarray(parameters.wheel_radius_m)
    np.testing.assert_allclose(
        result.output.generalized_acceleration[wheel_slice],
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_instant_links_are_mirrored_and_close_6dof_upright_force_balance(parameters):
    coefficients = parameters.double_wishbone.instant_links_at(np.zeros(4)).coefficient_matrix
    np.testing.assert_allclose(coefficients[0, 0], coefficients[1, 0], atol=1e-12)
    np.testing.assert_allclose(coefficients[2, 0], coefficients[3, 0], atol=1e-12)
    np.testing.assert_allclose(coefficients[0, 1], -coefficients[1, 1], atol=1e-12)
    np.testing.assert_allclose(coefficients[2, 1], -coefficients[3, 1], atol=1e-12)

    model = create_model(6, parameters)
    output = model.evaluate(model.initial_state(15.0), ModelInputs(steering_rad=0.04))
    assert np.max(np.abs(output.geometric_vertical_forces_n)) > 1.0
    np.testing.assert_allclose(
        output.normal_loads_n,
        output.suspension_forces_n + output.geometric_vertical_forces_n,
        rtol=1e-9,
        atol=1e-8,
    )


def test_instant_links_are_derived_at_current_corner_jounce(parameters):
    nominal = parameters.double_wishbone.instant_links_at(np.zeros(4)).coefficient_matrix
    traveled = parameters.double_wishbone.instant_links_at(np.array([0.02, -0.02, 0.02, -0.02])).coefficient_matrix
    assert not np.allclose(traveled, nominal)


def test_14dof_instant_link_force_has_equal_and_opposite_unsprung_reaction(parameters):
    model = create_model(14, parameters)
    output = model.evaluate(model.initial_state(15.0), ModelInputs(steering_rad=0.04))
    unsprung_accel = output.generalized_acceleration[6:10]
    unsprung_mass = np.asarray(parameters.unsprung_mass_kg)
    np.testing.assert_allclose(
        unsprung_accel,
        -output.geometric_vertical_forces_n / unsprung_mass,
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("dof", (3, 6, 10, 14))
def test_constant_radius_qss_converges_for_every_fidelity(parameters, dof):
    model = create_model(dof, parameters)
    result = solve_steady_state(model, speed_mps=15.0, yaw_rate_radps=0.2)

    assert result.success, result.message
    assert result.residual_norm < 1e-7
    assert np.all(result.output.normal_loads_n > 0.0)
    assert result.unknowns["steering_rad"] > 0.0
    if dof >= 10:
        assert "slip_rr" in result.unknowns
    if dof == 14:
        assert "unsprung_z_rr_m" in result.unknowns


def test_qss_returns_infeasible_result_when_kinematic_steer_guess_exceeds_bound(parameters):
    model = create_model(3, parameters)
    result = solve_acceleration_trim(
        model,
        speed_mps=5.0,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=4.0 * 9.80665,
        max_nfev=20,
    )

    assert not result.success
    assert result.unknowns["steering_rad"] <= 0.7


@pytest.mark.parametrize("dof", (3, 6, 10, 14))
@pytest.mark.parametrize("speed_mps", (6.0, 18.0))
def test_ggv_lateral_trim_is_acceleration_not_implied_corner_radius(
    parameters,
    dof,
    speed_mps,
):
    model = create_model(dof, parameters)
    result = solve_acceleration_trim(
        model,
        speed_mps=speed_mps,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=8.0,
    )

    assert result.success, result.message
    assert result.yaw_rate_radps == pytest.approx(0.0)
    assert result.lateral_acceleration_mps2 == pytest.approx(8.0, abs=1e-6)


def test_ggv_sustainable_lateral_endpoint_is_closed_at_zero_body_ax(parameters):
    projection = project_vehicle_yaml(load_vehicle_yaml())
    ay, ax = solve_lateral_limit(
        projection.ggv,
        speed=12.0,
        ay_upper=2.6 * 9.80665,
        reduced_model=create_model(3, parameters),
        binary_iterations=4,
    )

    assert 1.8 * 9.80665 < ay < 2.6 * 9.80665
    assert ax == pytest.approx(0.0, abs=1e-12)


def test_lateral_endpoint_search_crosses_a_disconnected_feasibility_hole():
    def disconnected_feasibility(value: float) -> bool:
        return value <= 4.0 or 6.0 <= value <= 9.0

    result = _largest_feasible_value(
        disconnected_feasibility,
        lower=0.0,
        upper=10.0,
        scan_step=0.1,
        binary_iterations=10,
    )

    assert result == pytest.approx(9.0, abs=1e-4)


def test_lateral_endpoint_search_accepts_phase_shifted_broad_interval():
    def phase_shifted_interval(value: float) -> bool:
        return value <= 4.0 or 8.96 <= value <= 9.07

    result = _largest_feasible_value(
        phase_shifted_interval,
        lower=0.0,
        upper=10.0,
        scan_step=0.1,
        binary_iterations=10,
    )

    # Only the 9.0 coarse node lies in the upper interval, even though its
    # actual width exceeds the requested 0.1 resolution.
    assert result == pytest.approx(9.07, abs=1e-4)


def test_lateral_endpoint_search_rejects_a_sub_resolution_island():
    def narrow_island(value: float) -> bool:
        return value <= 4.0 or 8.96 <= value <= 9.04

    result = _largest_feasible_value(
        narrow_island,
        lower=0.0,
        upper=10.0,
        scan_step=0.1,
        binary_iterations=10,
    )

    assert result == pytest.approx(4.0, abs=1e-4)


@pytest.mark.parametrize(
    ("rear_fraction", "front_arb_fraction", "front_brake_fraction"),
    (
        (0.54, 0.6285970914538208, 0.6996272063175089),
        (0.55, 0.6657864396429404, 0.6936964883187899),
        (0.56, 0.6729879814212416, 0.6865071463493561),
    ),
)
def test_6dof_lateral_endpoint_recovers_high_ay_branch_after_trim_hole(
    parameters,
    rear_fraction,
    front_arb_fraction,
    front_brake_fraction,
):
    cg_height_m = 11.5 * 0.0254
    projection = project_vehicle_yaml(load_vehicle_yaml(), aero_balance_front=0.5)
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            absolute_cg_height_m=cg_height_m,
            static_rear_weight_fraction=rear_fraction,
            aero_balance_front=0.5,
            brake_distribution_front=front_brake_fraction,
            front_antiroll_stiffness_fraction=front_arb_fraction,
            tire_mu_scale=0.622543713077903,
        ),
    )
    vehicle = replace(
        projection.ggv,
        mass=tuned.mass_kg,
        cg_height=cg_height_m,
        front_static_frac=tuned.static_front_weight_fraction,
        brake_distribution_front=front_brake_fraction,
    )

    ay, ax = solve_lateral_limit(
        vehicle,
        speed=12.0,
        ay_upper=2.2 * G,
        reduced_model=create_model(6, tuned),
        binary_iterations=10,
    )

    assert 14.35 < ay < 14.50
    assert ax == pytest.approx(0.0, abs=1e-12)


def test_rear54_6dof_lateral_endpoint_continues_upper_branch_at_14_mps(parameters):
    cg_height_m = 11.5 * 0.0254
    projection = project_vehicle_yaml(load_vehicle_yaml(), aero_balance_front=0.5)
    front_brake_fraction = 0.6996272063175089
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            absolute_cg_height_m=cg_height_m,
            static_rear_weight_fraction=0.54,
            aero_balance_front=0.5,
            brake_distribution_front=front_brake_fraction,
            front_antiroll_stiffness_fraction=0.6285970914538208,
            tire_mu_scale=0.622543713077903,
        ),
    )
    vehicle = replace(
        projection.ggv,
        mass=tuned.mass_kg,
        cg_height=cg_height_m,
        front_static_frac=tuned.static_front_weight_fraction,
        brake_distribution_front=front_brake_fraction,
    )

    ay, ax = solve_lateral_limit(
        vehicle,
        speed=14.0,
        ay_upper=2.2 * G,
        reduced_model=create_model(6, tuned),
        binary_iterations=10,
    )

    assert ay == pytest.approx(14.597998188354, abs=1e-9)
    assert ax == pytest.approx(0.0, abs=1e-12)


def test_rear55_6dof_lateral_endpoint_respects_tir_load_floor_at_10_mps(
    parameters,
):
    cg_height_m = 11.5 * 0.0254
    projection = project_vehicle_yaml(load_vehicle_yaml(), aero_balance_front=0.5)
    front_brake_fraction = 0.6936964883187899
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            absolute_cg_height_m=cg_height_m,
            static_rear_weight_fraction=0.55,
            aero_balance_front=0.5,
            brake_distribution_front=front_brake_fraction,
            front_antiroll_stiffness_fraction=0.6657864396429404,
            tire_mu_scale=0.622543713077903,
        ),
    )
    vehicle = replace(
        projection.ggv,
        mass=tuned.mass_kg,
        cg_height=cg_height_m,
        front_static_frac=tuned.static_front_weight_fraction,
        brake_distribution_front=front_brake_fraction,
    )

    endpoints = [
        solve_lateral_limit(
            vehicle,
            speed=10.0,
            ay_upper=2.2 * G,
            reduced_model=create_model(6, tuned),
            binary_iterations=10,
        )
        for _ in range(2)
    ]

    assert endpoints[0][0] == pytest.approx(13.51866813793945, abs=1e-9)
    assert endpoints[0][1] == pytest.approx(0.0, abs=1e-12)
    assert endpoints[1] == pytest.approx(endpoints[0], abs=1e-12)


def test_sustainable_lateral_endpoint_obeys_drive_speed_and_force_caps(parameters):
    projection = project_vehicle_yaml(load_vehicle_yaml())
    model = create_model(3, parameters)
    above_cap_speed = projection.ggv.max_drive_speed + 0.1

    ay, _ax = solve_lateral_limit(
        projection.ggv,
        speed=above_cap_speed,
        ay_upper=2.6 * G,
        reduced_model=model,
        binary_iterations=4,
    )
    drive_ax = solve_ax_limit(
        projection.ggv,
        speed=above_cap_speed,
        ay=0.0,
        ax_grid=np.array([0.0, G]),
        mode="drive",
        reduced_model=model,
        ax_binary_iterations=4,
    )
    no_drive_vehicle = replace(
        projection.ggv,
        max_drive_force=0.0,
        max_drive_power=0.0,
    )
    no_drive_ay, _ax = solve_lateral_limit(
        no_drive_vehicle,
        speed=12.0,
        ay_upper=2.6 * G,
        reduced_model=model,
        binary_iterations=4,
    )
    no_drag_parameters = replace(parameters, cd_area_m2=0.0)
    no_drag_no_drive_vehicle = replace(
        no_drive_vehicle,
        cd_a=0.0,
    )
    steering_projection_ay, _ax = solve_lateral_limit(
        no_drag_no_drive_vehicle,
        speed=25.0,
        ay_upper=2.6 * G,
        reduced_model=create_model(3, no_drag_parameters),
        binary_iterations=4,
    )

    assert np.isnan(ay)
    assert np.isnan(drive_ax)
    assert np.isnan(no_drive_ay)
    assert steering_projection_ay == pytest.approx(0.0, abs=1e-12)


def test_ggv_closes_drive_at_exact_sustainable_lateral_edge(parameters):
    projection = project_vehicle_yaml(load_vehicle_yaml(), aero_balance_front=0.5)
    tuned = apply_reduced_vehicle_overrides(
        parameters,
        ReducedVehicleOverrides(
            aero_balance_front=0.5,
            front_antiroll_stiffness_fraction=0.51,
        ),
    )
    envelope = generate_ggv(
        projection.ggv,
        GGVConfig(
            speeds=(11.8,),
            model_dof=3,
            ay_max_g=3.2,
            ay_points=17,
            ax_search_points=3,
            ax_binary_iterations=6,
            verbose=False,
            warn_tire_load_range=False,
            track_relevant_lateral_domain_only=True,
        ),
        reduced_model=create_model(3, tuned),
    )[0]

    positive = envelope.ay >= 0.0
    drive_domain = positive & np.isfinite(envelope.ax_accel)
    brake_domain = positive & np.isfinite(envelope.ax_brake)
    drive_edge_index = int(np.flatnonzero(drive_domain)[-1])
    direct_edge, _ax = solve_lateral_limit(
        projection.ggv,
        speed=11.8,
        ay_upper=3.2 * G,
        reduced_model=create_model(3, tuned),
        binary_iterations=6,
    )

    assert envelope.ay[drive_domain].max() == pytest.approx(direct_edge)
    assert envelope.ay[brake_domain].max() <= direct_edge + 1e-9
    assert envelope.ax_accel[drive_edge_index] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dof", (3, 6))
def test_ggv_brake_search_does_not_require_zero_ax_trim(parameters, dof):
    projection = project_vehicle_yaml(load_vehicle_yaml())
    model = create_model(dof, parameters)

    drive_ax = solve_ax_limit(
        projection.ggv,
        speed=25.0,
        ay=2.0 * G,
        ax_grid=np.array([0.0, G]),
        mode="drive",
        reduced_model=model,
        ax_binary_iterations=6,
    )
    brake_ax = solve_ax_limit(
        projection.ggv,
        speed=25.0,
        ay=2.0 * G,
        ax_grid=np.array([-3.2 * G, 0.0]),
        mode="brake",
        reduced_model=model,
        ax_binary_iterations=6,
    )

    assert np.isnan(drive_ax)
    assert brake_ax < 0.0


def test_prescribed_acceleration_and_moment_qss_use_same_6dof_equations(parameters):
    model = create_model(6, parameters)
    acceleration = solve_acceleration_trim(
        model,
        speed_mps=15.0,
        longitudinal_acceleration_mps2=2.0,
        lateral_acceleration_mps2=4.0,
    )
    assert acceleration.success
    assert acceleration.residual_norm < 1e-7

    left = solve_moment_state(
        model,
        speed_mps=15.0,
        beta_rad=0.02,
        steering_rad=0.04,
    )
    right = solve_moment_state(
        model,
        speed_mps=15.0,
        beta_rad=-0.02,
        steering_rad=-0.04,
    )
    assert left.success and right.success
    assert left.output.body_moment_nm[2] == pytest.approx(-right.output.body_moment_nm[2], rel=2e-3, abs=1.0)


@pytest.mark.parametrize("dof", (3, 6, 10, 14))
def test_transient_result_exposes_boblib_comparison_channels(parameters, dof):
    model = create_model(dof, parameters)
    trim = solve_steady_state(model, speed_mps=10.0)
    assert trim.success
    result = simulate_transient(
        model,
        initial_state=trim.state,
        controls=trim.inputs,
        time_s=np.linspace(0.0, 0.1, 6),
        method="Radau" if dof >= 6 else "RK45",
        rtol=1e-5,
        atol=1e-7,
    )

    assert result.success
    for signal in ("velX", "velY", "yawVel", "sideslip", "accX", "accY", "roll"):
        assert signal in result.signals
    metrics = compare_transient_signals(
        result.time_s,
        {"yawVel": result.signals["yawVel"]},
        result,
    )
    assert metrics["yawVel"]["rmse"] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("dof", (3, 6, 10, 14))
def test_ggv_and_ymd_backends_call_shared_qss_model(parameters, dof):
    projection = project_vehicle_yaml(load_vehicle_yaml())
    model = create_model(dof, parameters)
    ax_limit = solve_ax_limit(
        projection.ggv,
        speed=15.0,
        ay=4.0,
        ax_grid=np.linspace(0.0, 20.0, 21),
        mode="drive",
        reduced_model=model,
    )
    assert np.isfinite(ax_limit)
    assert ax_limit > 0.0

    ay, mz, converged = ymd_point(
        projection.ymd,
        config=YMDConfig(
            speed=15.0,
            model_dof=dof,
            verbose=False,
            warn_tire_load_range=False,
        ),
        beta=0.02,
        hwa=0.04,
        reduced_model=model,
    )
    assert converged
    assert np.isfinite(ay)
    assert np.isfinite(mz)
