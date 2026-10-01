"""Independent tire, spectral-normalization and unsupported-data checks."""

import numpy as np
import pytest

from _0_Utils.mf52_lateral import LateralForce
from _0_Utils.vehicle_io import parse_tir, repo_root
from _3_StandardSim.QuarterCarPSD.grip_sweep import aggregate, optimize_average, response_record, road_spectrum
from _3_StandardSim.QuarterCarPSD.quarter_car_psd import Corner, ISO_G0, rms_response, wheel_damping
from _5_App.tire_eval import _mf52_fy_pure


@pytest.fixture
def tire():
    return parse_tir(repo_root() / "_0_Utils/tire_templates/16x7p5_10_12psi.tir")


def test_vector_force_matches_existing_scalar_mf52(tire):
    loads = np.array([100.0, 300.0, 650.0, 1200.0, 1800.0])
    model = LateralForce(tire, loads)
    for angle in np.linspace(tire["ALPMIN"], tire["ALPMAX"], 51):
        expected = [_mf52_fy_pure(tire, z, angle, 0) for z in loads]
        np.testing.assert_allclose(model(angle), expected, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("loads", [[99.99, 650], [650, 1800.01], [np.nan], [0], []])
def test_no_out_of_domain_force(tire, loads):
    with pytest.raises(ValueError):
        LateralForce(tire, loads)


def test_optimizer_matches_fine_grid_and_constant_load(tire):
    loads = np.array([250.0, 450.0, 600.0, 700.0, 950.0])
    value, alpha, _ = optimize_average(tire, loads)
    force = LateralForce(tire, loads)
    grid = np.linspace(tire["ALPMIN"], tire["ALPMAX"], 6001)
    brute = max(float(np.mean(force(a))) for a in grid)
    assert value >= brute - 1e-8
    assert value == pytest.approx(brute, rel=1e-6)
    assert value == pytest.approx(np.mean(force(alpha)))
    envelope = np.mean([max(LateralForce(tire, np.array([z]))(a)[0] for a in grid) for z in loads])
    assert value < envelope  # Avoid averaging individually optimized instantaneous peaks.
    smooth, _, _ = optimize_average(tire, np.array([650.0]))
    constant, _, _ = optimize_average(tire, np.full(200, 650.0))
    assert constant == pytest.approx(smooth, rel=1e-12)


def test_road_parseval_and_continuous_load_bound():
    qcm = {"speed_mps": 15.0, "spatial_frequency_min_cycles_per_m": 0.01, "spatial_frequency_max_cycles_per_m": 10.0}
    f, coefficients, count = road_spectrum(qcm, 64, 600, 2027)
    road = np.fft.irfft(coefficients, n=count) * count
    expected = ISO_G0["A"] * 0.1**2 * (1 / 0.01 - 1 / 10)
    assert np.var(road) == pytest.approx(expected, rel=1e-12)
    c = Corner("front", 53.3, 7.7, 98947, 115.844, 61 * 9.80665)
    damping = wheel_damping(c, 15000, 0.7)
    loads, margin, rms = response_record(c, 15000, damping, f, coefficients, count, 600)
    analytic, _ = rms_response(c, 15000, damping, 15, ISO_G0["A"], 0.01, 10, 8193)
    assert np.std(loads) == pytest.approx(rms, rel=1e-12)
    assert rms == pytest.approx(analytic, rel=0.002)
    # Zero pad the SAME spectrum, retaining all phases, for an 8x denser check.
    padded = np.pad(coefficients, (0, 7 * count // 2))
    ff = np.fft.rfftfreq(count * 8, 1 / 4800)
    fine, _, _ = response_record(c, 15000, damping, ff, padded, count * 8, 4800)
    assert fine.min() >= loads.min() - margin
    assert fine.max() <= loads.max() + margin
    np.testing.assert_allclose(fine[::8], loads, atol=1e-10)


def test_aggregation_never_averages_only_passing_records():
    common = {
        "axle": "front",
        "iso_class": "B",
        "wheelrate_n_per_m": 10000,
        "smooth_capacity_n": 1500,
        "load_min_bound_n": 90,
        "load_max_bound_n": 1100,
    }
    rows = [
        {**common, "supported": True, "capacity_n": 1450, "loss_percent": 100 / 30},
        {**common, "supported": False, "capacity_n": None, "loss_percent": None},
    ]
    result = aggregate(rows, 2)[0]
    assert not result["supported"]
    assert result["supported_records"] == 1
    assert result["capacity_lbf"] is None
    assert result["loss_percent"] is None
