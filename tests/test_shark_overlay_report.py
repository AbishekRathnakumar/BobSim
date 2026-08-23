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


def test_z_dependent_curves_are_withheld_but_angles_are_not() -> None:
    """A rigid vertical translation moves heights, not angles or lengths."""
    assert "bump_rc_height_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_swing_arm_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_rc_z_mm" in sor.Z_DEPENDENT_CURVE_IDS
    # Lateral and angular quantities survive the translation, so they still publish.
    assert "bump_rc_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_ic_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_camber_deg" not in sor.Z_DEPENDENT_CURVE_IDS
