"""Physical cross-checks for the frequency-domain quarter-car study."""

import math

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from _3_StandardSim.QuarterCarPSD.quarter_car_psd import (
    Corner,
    ISO_G0,
    load_corners,
    rms_response,
    state_matrices,
    transfer,
    wheel_damping,
)
from _0_Utils.vehicle_io import repo_root


@pytest.fixture
def corner():
    return Corner("front", 55.0, 8.0, 99000.0, 115.0, 63.0 * 9.80665)


def test_frequency_response_matches_independent_state_space(corner):
    k, c = 12000.0, wheel_damping(corner, 12000, 0.7)
    a, b = state_matrices(corner, k, c)
    for f in (0.01, 0.5, 2, 5, 18, 75, 150):
        s = 2j * np.pi * f
        state = np.linalg.solve(s * np.eye(4) - a, b @ [1, s])
        force = corner.tire_stiffness_n_per_m * (1 - state[1]) + corner.tire_damping_ns_per_m * (s - state[3])
        actual_force, actual_travel = transfer(corner, k, c, f)
        assert actual_force == pytest.approx(force, rel=1e-8, abs=1e-8)
        assert actual_travel == pytest.approx(state[0] - state[1], rel=1e-8, abs=1e-8)
    assert transfer(corner, k, c, 0)[0] == pytest.approx(0, abs=1e-9)
    assert np.max(np.linalg.eigvals(a).real) < 0


def test_psd_normalization_against_white_road_velocity_lyapunov():
    # With ct=0, a broad n^-2 road spectrum has finite force variance.
    # Coordinates zs-road, zu-road, vs, vu remove the road-position integrator.
    corner = Corner("front", 55, 8, 99000, 0, 63 * 9.80665)
    k, c, speed, g0 = 12000, 1100, 15, ISO_G0["B"]
    a, _ = state_matrices(corner, k, c)
    b = np.array([-1, -1, 0, 0])
    one_sided_velocity_psd = (2 * np.pi) ** 2 * g0 * 0.1**2 * speed
    covariance = solve_continuous_lyapunov(a, -np.outer(b, b) * one_sided_velocity_psd / 2)
    expected = corner.tire_stiffness_n_per_m * math.sqrt(covariance[1, 1])
    actual, _ = rms_response(corner, k, c, speed, g0, 1e-6, 1e4, 32769)
    assert actual == pytest.approx(expected, rel=2e-4)


def test_class_amplitude_scaling_and_damping(corner):
    for rate in (5000, 20000, 50000):
        c = wheel_damping(corner, rate, 0.7)
        assert c / (2 * math.sqrt(rate * corner.sprung_mass_kg)) == pytest.approx(0.7)
        a, _ = rms_response(corner, rate, c, 15, ISO_G0["A"], 0.01, 10, 4097)
        b, _ = rms_response(corner, rate, c, 15, ISO_G0["B"], 0.01, 10, 4097)
        assert b == pytest.approx(2 * a)


def test_mass_projection_conserves_weight():
    corners, summary = load_corners(repo_root() / "vehicle.yml")
    assert 2 * sum(c.mean_load_n for c in corners) == pytest.approx(summary["total_mass_kg"] * 9.80665)
    assert 2 * sum(c.sprung_mass_kg + c.unsprung_mass_kg for c in corners) == pytest.approx(summary["total_mass_kg"])
