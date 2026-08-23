from __future__ import annotations

import os
from pathlib import Path

import pytest

from _3_StandardSim.FourPostEval import shark_overlay_report as sor


def test_stale_binary_is_refused_even_when_the_build_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-op `make` must not launder old geometry into a clean-looking report."""
    build_dir = tmp_path / "FourPostSim"
    build_dir.mkdir()
    exe = build_dir / "BobLib.Experiments.Standards.FourPostSim"
    exe.write_bytes(b"stale binary")
    os.utime(exe, (1_000_000, 1_000_000))
    monkeypatch.setattr(sor, "BUILD_DIR", build_dir)

    # Geometry regenerated after the binary was produced.
    with pytest.raises(sor.StaleGeometryError, match="predates the geometry"):
        sor.assert_binary_consumed_geometry({"latest_modified": 2_000_000.0}, "Baseline")

    # Binary newer than the geometry is accepted.
    sor.assert_binary_consumed_geometry({"latest_modified": 500_000.0}, "Baseline")


def test_missing_executable_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sor, "BUILD_DIR", tmp_path / "empty")
    with pytest.raises(sor.StaleGeometryError, match="no four-post executable"):
        sor.assert_binary_consumed_geometry({"latest_modified": 1.0}, "Baseline")


def test_delta_score_reports_absolute_and_relative() -> None:
    """Ratio alone is misleading when the baseline is flat, so both are returned."""
    score = sor._delta_score([0.0, 1.0, 2.0], [0.0, 1.5, 2.0])
    assert score is not None
    assert score["peak"] == pytest.approx(0.5)
    assert score["span"] == pytest.approx(2.0)
    assert score["ratio"] == pytest.approx(0.25)

    flat = sor._delta_score([1.0, 1.0, 1.0], [1.0, 1.2, 1.0])
    assert flat is not None and flat["ratio"] == float("inf")

    identical = sor._delta_score([0.0, 1.0], [0.0, 1.0])
    assert identical is not None and identical["ratio"] == 0.0


def test_a_missing_datum_record_withholds_rather_than_publishes(tmp_path: Path) -> None:
    """Absence of evidence must not read as evidence of a shared datum.

    A vehicle with no sidecar - a fresh clone, a hand-written file - has an unknown
    vertical datum, so the z-dependent curves stay withheld. Failing open here would
    publish exactly the curves the datum question puts in doubt.
    """
    from _0_Utils.shark_import import read_datum_status, write_datum_sidecar

    bare = tmp_path / "vehicle_bare.yml"
    bare.write_text("front: {}\n", encoding="utf-8")
    assert read_datum_status(bare) is None

    # Only a positive "shared_ground_plane" clears the withholding.
    write_datum_sidecar(bare, {"status": "unresolved"})
    assert read_datum_status(bare) == "unresolved"
    write_datum_sidecar(bare, {"status": "shared_ground_plane"})
    assert read_datum_status(bare) == "shared_ground_plane"


def test_interrupted_run_leaves_a_backup_that_blocks_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover backup means vehicle.yml on disk is probably the wrong car."""
    vehicle = tmp_path / "vehicle.yml"
    vehicle.write_text("front: {}\n", encoding="utf-8")
    monkeypatch.setattr(sor, "VEHICLE_YAML", vehicle)
    vehicle.with_suffix(".yml.overlay-backup").write_text("front: {}\n", encoding="utf-8")

    with pytest.raises(sor.StaleGeometryError, match="leftover backup"):
        with sor.installed_vehicle(vehicle):
            pass


def test_z_dependent_curves_are_withheld_but_angles_are_not() -> None:
    """A rigid vertical translation moves heights, not angles or lengths."""
    assert "bump_rc_height_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_swing_arm_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_rc_z_mm" in sor.Z_DEPENDENT_CURVE_IDS
    # Lateral and angular quantities survive the translation, so they still publish.
    assert "bump_rc_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_ic_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_camber_deg" not in sor.Z_DEPENDENT_CURVE_IDS
