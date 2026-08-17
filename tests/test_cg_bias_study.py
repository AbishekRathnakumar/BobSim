from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from _0_Utils.vehicle_io import load_yaml
from _0_Utils.dyn_py import Vehicle
from _0_Utils.lap_sim import GGVMap
from _2_EnvelopeSim.GGV.ggv_generation import GGVConfig, GGVEnvelope
from _2_EnvelopeSim.vehicle_yaml import project_vehicle_yaml
from _4_OptSim.CGBiasStudy.cg_bias_study import (
    MassComponent,
    _acceleration_event,
    _audit_envelopes,
    _fingerprint,
    _file_sha256,
    _ggv_cache_is_valid,
    apply_arb_allocation,
    build_realizable_layout,
    combine_mass_properties,
    make_synthetic_cg_variant,
    rear_weight_fraction,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "_4_OptSim/CGBiasStudy/config.yml")


def test_parallel_axis_recomputes_full_tensor() -> None:
    components = [
        MassComponent("left", 2.0, (0.0, -1.0, 0.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        MassComponent("right", 2.0, (0.0, 1.0, 0.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ]
    mass, cg, inertia = combine_mass_properties(components)
    assert mass == 4.0
    assert cg == (0.0, 0.0, 0.0)
    np.testing.assert_allclose(inertia, np.diag([6.0, 2.0, 6.0]))


@pytest.mark.parametrize("rear_fraction", [0.50, 0.55, 0.60])
def test_synthetic_cg_hits_requested_weight_without_setup_changes(
    rear_fraction: float,
) -> None:
    baseline = load_yaml(ROOT / "vehicle.yml")
    variant = make_synthetic_cg_variant(baseline, rear_fraction)
    assert rear_weight_fraction(variant) == pytest.approx(rear_fraction, abs=1e-11)
    assert variant["brake"] == baseline["brake"]
    assert (
        variant["front"]["actuation"]["stabar"]
        == baseline["front"]["actuation"]["stabar"]
    )
    assert (
        variant["rear"]["actuation"]["stabar"]
        == baseline["rear"]["actuation"]["stabar"]
    )
    assert variant["sprung_mass"]["inertia_kg_m2"] == baseline["sprung_mass"]["inertia_kg_m2"]


def test_arb_allocation_preserves_effective_total_and_other_inputs() -> None:
    baseline = load_yaml(ROOT / "vehicle.yml")
    variant = apply_arb_allocation(baseline, 0.70, root=ROOT)
    assert variant["brake"] == baseline["brake"]
    assert variant["sprung_mass"] == baseline["sprung_mass"]
    assert (
        variant["front"]["actuation"]["stabar"]["rate_n_m_per_rad"]
        != baseline["front"]["actuation"]["stabar"]["rate_n_m_per_rad"]
    )


def test_realizable_baseline_closes_mass_properties_and_rejects_bounds() -> None:
    baseline = load_yaml(ROOT / "vehicle.yml")
    layout = CONFIG["realizable_layout"]
    variant = build_realizable_layout(baseline, layout, layout["layouts"]["baseline"])
    assert variant["sprung_mass"]["mass_kg"] == baseline["sprung_mass"]["mass_kg"]
    np.testing.assert_allclose(
        variant["sprung_mass"]["cg_m"],
        baseline["sprung_mass"]["cg_m"],
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        variant["sprung_mass"]["inertia_kg_m2"],
        baseline["sprung_mass"]["inertia_kg_m2"],
        rtol=0.0,
        atol=1e-6,
    )

    invalid = copy.deepcopy(layout["layouts"]["baseline"])
    invalid["accumulator_x_m"] = -2.0
    with pytest.raises(ValueError, match="packaging bounds"):
        build_realizable_layout(baseline, layout, invalid)


def test_event_power_propagation_and_cache_fingerprint_invalidation() -> None:
    vehicle = Vehicle.from_yaml(ROOT / "vehicle.yml")
    capped = vehicle.with_power_limit(CONFIG["events"]["endurance"]["power_limit_w"])
    assert capped.parameters.peak_drive_power_w == 32_000.0
    assert vehicle.parameters.peak_drive_power_w == 80_000.0

    payload = {"vehicle": "abc", "power_limit_w": 32_000, "resolution": 7}
    assert _fingerprint(payload) != _fingerprint({**payload, "power_limit_w": 36_000})
    assert _fingerprint(payload) != _fingerprint({**payload, "resolution": 9})
    assert _fingerprint(payload) != _fingerprint({**payload, "vehicle": "def"})


def test_constraint_audit_reports_tire_domain() -> None:
    baseline = load_yaml(ROOT / "vehicle.yml")
    projection = project_vehicle_yaml(baseline, repo_root=ROOT)
    envelope = GGVEnvelope(
        speed=12.0,
        ay=np.asarray([0.0]),
        ax_accel=np.asarray([0.0]),
        ax_brake=np.asarray([0.0]),
    )
    generation = GGVConfig(
        speeds=(12.0,),
        ay_points=1,
        enforce_tire_load_range=True,
        verbose=False,
    )
    audit = _audit_envelopes("baseline", 32_000.0, projection.ggv, generation, [envelope])
    assert audit["tire"]["out_of_domain_points"] == 0
    assert audit["tire"]["domain_enforced"] is True
    assert audit["constraint"]["trim_state_audit_status"].startswith("blocking_")


def test_acceleration_event_has_free_exit_speed() -> None:
    ggv = GGVMap.from_arrays(
        speed_mps=[1.0, 30.0],
        ay_mps2=[0.0],
        ax_accel_mps2=[[2.0], [2.0]],
        ax_brake_mps2=[[-3.0], [-3.0]],
    )
    elapsed, exit_speed = _acceleration_event(ggv, distance_m=100.0, step_m=0.5)
    assert elapsed == pytest.approx(10.0, rel=1e-12)
    assert exit_speed == pytest.approx(20.0, rel=1e-12)


def test_cache_rejects_raw_output_overwritten_after_metadata(tmp_path: Path) -> None:
    raw = tmp_path / "ggv.csv"
    metadata = tmp_path / "ggv.csv.metadata.json"
    expected = {"fingerprint": "input-fingerprint"}
    raw.write_text("speed,ax\n6,1\n", encoding="utf-8")
    metadata.write_text(
        "{\n"
        '  "fingerprint": "input-fingerprint",\n'
        f'  "output_csv_sha256": "{_file_sha256(raw)}"\n'
        "}\n",
        encoding="utf-8",
    )
    assert _ggv_cache_is_valid(raw, metadata, expected)

    raw.write_text("speed,ax\n6,2\n", encoding="utf-8")
    assert not _ggv_cache_is_valid(raw, metadata, expected)
    assert not _ggv_cache_is_valid(raw, metadata, {"fingerprint": "changed-input"})
