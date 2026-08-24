from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

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
    from _0_Utils.shark_import import datum_gate, write_datum_sidecar

    bare = tmp_path / "vehicle_bare.yml"
    bare.write_text("front: {}\n", encoding="utf-8")
    vehicle: dict[str, Any] = {"front": {}}
    assert datum_gate(bare, vehicle)["valid"] is False

    # Only a positive verdict on every recorded axle clears the withholding.
    write_datum_sidecar(bare, "front", {"status": "unresolved"}, vehicle)
    assert datum_gate(bare, vehicle)["valid"] is False
    write_datum_sidecar(bare, "front", {"status": "shared_ground_plane"}, vehicle)
    assert datum_gate(bare, vehicle)["valid"] is True


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


def _fake_build_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A build directory holding a variant binary and its stamp."""
    build_dir = tmp_path / "FourPostSim"
    build_dir.mkdir()
    exe = build_dir / "BobLib.Experiments.Standards.FourPostSim"
    exe.write_bytes(b"binary compiled from the variant")
    stamp = build_dir / ".bobsim_geometry_stamp.json"
    stamp.write_text('{"signature": "variant", "vehicle": "2027"}', encoding="utf-8")
    monkeypatch.setattr(sor, "BUILD_DIR", build_dir)
    monkeypatch.setattr(sor, "GEOMETRY_STAMP", stamp)
    return exe, stamp


@pytest.mark.parametrize("boom", [None, RuntimeError, KeyboardInterrupt])
def test_modelica_stack_is_restored_to_the_baseline_however_the_run_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boom: type[BaseException] | None
) -> None:
    """A comparison must never leave BobLib describing the variant.

    Covers the clean path, a raised error, and an interrupt, because the failure
    mode is identical in all three: the next unrelated build would silently compile
    the imported geometry while vehicle.yml says Orion.
    """
    exe, stamp = _fake_build_dir(tmp_path, monkeypatch)
    regenerated: list[Path] = []
    monkeypatch.setattr(
        sor, "generate_modelica_stack",
        lambda path, root=None: regenerated.append(Path(path)),
    )
    baseline = tmp_path / "vehicle.yml"
    baseline.write_text("front: {}\n", encoding="utf-8")

    if boom is None:
        with sor.orion_modelica_stack(baseline):
            pass
    else:
        with pytest.raises(boom):
            with sor.orion_modelica_stack(baseline):
                raise boom()

    assert regenerated == [baseline], "records must be regenerated from the baseline"
    assert not stamp.exists(), "a stamp asserting the variant build must not survive"
    assert not exe.exists(), "the variant binary must not survive to be reused"


def test_restoration_invalidates_artifacts_even_if_regeneration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed regenerate must not leave a stamped binary claiming to match."""
    exe, stamp = _fake_build_dir(tmp_path, monkeypatch)

    def explode(path: Any, root: Any = None) -> None:
        raise RuntimeError("generator failed")

    monkeypatch.setattr(sor, "generate_modelica_stack", explode)
    baseline = tmp_path / "vehicle.yml"
    baseline.write_text("front: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="generator failed"):
        with sor.orion_modelica_stack(baseline):
            pass

    assert not stamp.exists()
    assert not exe.exists()


def test_modelica_state_report_describes_the_restored_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe, stamp = _fake_build_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sor, "modelica_stack_status_payload",
        lambda vehicle_path, root: {"state": "written", "vehicle_name": "Orion"},
    )
    report = sor.modelica_state_report(tmp_path / "vehicle.yml")
    assert report == {
        "records_match_baseline": True,
        "state": "written",
        "vehicle_name": "Orion",
        "stamp_present": True,
        "executable_present": True,
    }
    sor.invalidate_build_artifacts()
    after = sor.modelica_state_report(tmp_path / "vehicle.yml")
    assert after["stamp_present"] is False and after["executable_present"] is False


def test_post_run_state_is_consistent_across_yaml_records_and_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a comparison, all three must agree that the car is the baseline.

    The three can drift apart independently - the YAML is restored by one context
    manager, the records by another, and the binary is just a file left on disk -
    so the regression asserts them together rather than one at a time.
    """
    exe, stamp = _fake_build_dir(tmp_path, monkeypatch)

    baseline = tmp_path / "vehicle.yml"
    baseline.write_text("name: orion\n", encoding="utf-8")
    variant = tmp_path / "variant.yml"
    variant.write_text("name: variant\n", encoding="utf-8")
    monkeypatch.setattr(sor, "VEHICLE_YAML", baseline)

    regenerated: list[Path] = []
    monkeypatch.setattr(
        sor, "generate_modelica_stack",
        lambda path, root=None: regenerated.append(Path(path)),
    )
    # The variant run fails after the swap, which is the path that used to leave
    # BobLib describing the imported car.
    monkeypatch.setattr(
        sor, "four_post_signature",
        lambda path: (_ for _ in ()).throw(sor.StaleGeometryError("build failed")),
    )

    with sor.orion_modelica_stack(baseline):
        with pytest.raises(sor.StaleGeometryError):
            sor.run_four_post(variant, sor.VARIANT_LABEL, skip_build=False)

    assert baseline.read_text(encoding="utf-8") == "name: orion\n", "YAML restored"
    assert not baseline.with_suffix(".yml.overlay-backup").exists(), "no leftover backup"
    assert regenerated == [baseline], "records regenerated from the baseline"
    assert not stamp.exists() and not exe.exists(), "variant build artifacts invalidated"


def test_foreign_binaries_are_detected_before_they_are_executed(tmp_path: Path) -> None:
    """An ELF simulator on a Windows host cannot be run, and saying so early matters.

    The Modelica stack compiles inside the Linux container, so a Windows host gets a
    binary it cannot exec. Undetected, that surfaces as WinError 193 from deep inside
    the eval runner, after a full Modelica build has already been paid for.
    """
    elf = tmp_path / "sim_elf"
    elf.write_bytes(b"\x7fELF" + b"\x00" * 60)
    pe = tmp_path / "sim_pe"
    pe.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)

    assert sor.host_can_run(elf) is (os.name != "nt")
    assert sor.host_can_run(pe) is (os.name == "nt")
    # An unreadable path is not runnable either.
    assert sor.host_can_run(tmp_path / "absent") is False


def test_build_failure_to_launch_is_a_refusal_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make` being unlaunchable must raise the type that triggers restoration."""
    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError(193, "%1 is not a valid Win32 application")

    monkeypatch.setattr(sor.subprocess, "run", explode)
    with pytest.raises(sor.StaleGeometryError, match="Could not launch"):
        sor.build_four_post()


def test_actuation_differences_separate_force_changes_from_geometry(tmp_path: Path) -> None:
    """A removed ARB is a confound; a moved rocker pivot is the change under test."""
    import yaml

    base: dict[str, Any] = {"rear": {"actuation": {
        "rod_to": "upper",
        "shock": {"spring_table": {"table": [[0, 0]]}, "free_length_m": 0.26},
        "bellcrank": {"pivot_m": [0.0, 0.0, 0.0]},
        "stabar": {"rate_n_m_per_rad": 535.0},
    }}}
    var = copy.deepcopy(base)
    var["rear"]["actuation"].pop("stabar")
    var["rear"]["actuation"]["rod_to"] = "lower"
    var["rear"]["actuation"]["bellcrank"]["pivot_m"] = [0.01, 0.0, 0.0]

    bp, vp = tmp_path / "b.yml", tmp_path / "v.yml"
    bp.write_text(yaml.safe_dump(base), encoding="utf-8")
    vp.write_text(yaml.safe_dump(var), encoding="utf-8")

    found = sor.actuation_differences(bp, vp)
    confounds = {d["field"] for d in found if d["confound"]}
    geometry = {d["field"] for d in found if not d["confound"]}
    assert "stabar (anti-roll bar)" in confounds
    assert {"rod_to", "bellcrank.pivot_m"} <= geometry
    # The bar vanishing is reported once, not again as a missing rate.
    assert "stabar.rate_n_m_per_rad" not in confounds


def test_geometry_only_mode_holds_force_elements_and_reports_what_it_cannot(
    tmp_path: Path,
) -> None:
    """Springs are held; an ARB on a moved pivot is reported as unheld, not faked."""
    import yaml

    base: dict[str, Any] = {"rear": {"actuation": {
        "shock": {"spring_table": {"table": [[0, 100]]}, "damper_table": {"table": [[0, 1]]},
                  "free_length_m": 0.26},
        "bellcrank": {"pivot_m": [0.0, 0.0, 0.0], "pickups_m": {"stabar": [0.0, 0.0, 0.03]}},
        "stabar": {"rate_n_m_per_rad": 535.0},
    }}}
    var: dict[str, Any] = {"rear": {"actuation": {
        "shock": {"spring_table": {"table": [[0, 999]]}, "damper_table": {"table": [[0, 9]]},
                  "free_length_m": 0.30},
        "bellcrank": {"pivot_m": [0.05, 0.0, 0.0], "pickups_m": {}},
    }}}
    bp, vp = tmp_path / "b.yml", tmp_path / "v.yml"
    bp.write_text(yaml.safe_dump(base), encoding="utf-8")
    vp.write_text(yaml.safe_dump(var), encoding="utf-8")

    out, unheld = sor.hold_baseline_actuation(bp, vp, tmp_path / "held.yml")
    held = yaml.safe_load(out.read_text(encoding="utf-8"))

    shock = held["rear"]["actuation"]["shock"]
    assert shock["spring_table"] == base["rear"]["actuation"]["shock"]["spring_table"]
    assert shock["damper_table"] == base["rear"]["actuation"]["shock"]["damper_table"]
    assert shock["free_length_m"] == 0.26
    # Geometry is kept from the variant, which is the whole point of the mode.
    assert held["rear"]["actuation"]["bellcrank"]["pivot_m"] == [0.05, 0.0, 0.0]
    assert any("anti-roll bar" in item for item in unheld)
    assert "stabar" not in held["rear"]["actuation"]

    # With the springs now held, the only confound left is the one we reported.
    remaining = [d for d in sor.actuation_differences(bp, out) if d["confound"]]
    assert {d["field"] for d in remaining} == {"stabar (anti-roll bar)"}


def test_four_post_section_gates_and_flags_confounds() -> None:
    runs = {sor.BASELINE_LABEL: {"summary": {"avg_anti_dive_pct": 1.0}},
            sor.VARIANT_LABEL: {"summary": {"avg_anti_dive_pct": 2.0}}}
    diffs = [{"axle": "rear", "field": "stabar (anti-roll bar)", "kind": "force",
              "confound": True, "baseline": "present", "variant": "absent", "note": "n"}]

    gated = sor.four_post_section(runs, differences=diffs, mode="imported", gated=True)
    assert gated["rows"] == [], "jacking geometry moves with the datum, so it is gated too"
    assert any("WITHHELD" in line for line in gated["lines"])
    assert gated["confounded"] is True

    open_gate = sor.four_post_section(runs, differences=diffs, mode="imported", gated=False)
    assert open_gate["rows"], "with the gate open the numbers are published"
    assert any("CONFOUNDED" in line for line in open_gate["lines"])
    # The label must not read as total roll stiffness.
    labels = " ".join(label for _key, label in sor.SCALAR_METRICS)
    assert "geometric anti-roll" in labels.lower() and "jacking" in labels.lower()


def test_design_position_is_sampled_not_extrapolated() -> None:
    """The app's even grid straddles zero; the report's grid lands on it."""
    from _0_Utils.kin_py.kinematics import DEFAULT_ROLL_DEG, DEFAULT_SWEEP_M

    assert 0.0 not in DEFAULT_SWEEP_M and 0.0 not in DEFAULT_ROLL_DEG
    assert 0.0 in sor.BUMP_SWEEP_M and 0.0 in sor.ROLL_SWEEP_DEG
    # Range is preserved so the curves stay comparable to the app's view.
    assert min(sor.BUMP_SWEEP_M) == pytest.approx(min(DEFAULT_SWEEP_M))
    assert max(sor.BUMP_SWEEP_M) == pytest.approx(max(DEFAULT_SWEEP_M))

    x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert sor._at_design_position(x, [10.0, 11.0, 12.0, 13.0, 14.0]) == 12.0
    assert sor._working_slope(x, [10.0, 11.0, 12.0, 13.0, 14.0]) == pytest.approx(1.0)


def test_ranking_uses_engineering_tolerance_not_baseline_range() -> None:
    """A flat baseline must not turn a negligible change into the top finding."""
    payloads = {
        sor.BASELINE_LABEL: {
            "x_axes": {"jounce_mm": [-1.0, 0.0, 1.0]},
            "axles": {"rear": {"curves": {
                "bump_camber_deg": [-1.0, 0.0, 1.0],
                "bump_caster_deg": [0.0, 0.0, 0.0],
            }}, "front": {"curves": {}}},
        },
        sor.VARIANT_LABEL: {
            "x_axes": {"jounce_mm": [-1.0, 0.0, 1.0]},
            "axles": {"rear": {"curves": {
                "bump_camber_deg": [-1.0, 0.5, 1.0],     # 0.5 deg: 10x tolerance
                "bump_caster_deg": [0.0, 0.001, 0.0],    # 0.001 deg: negligible
            }}, "front": {"curves": {}}},
        },
    }
    rows = sor.curve_metrics(payloads, frozenset(), {"deg": 0.05, "mm": 0.5})
    by_id = {row["meta"]["id"]: row for row in rows}
    assert by_id["bump_camber_deg"]["significance"] == pytest.approx(10.0)
    assert by_id["bump_caster_deg"]["significance"] == pytest.approx(0.02)
    # Ranked by significance, the real change comes first despite the flat baseline
    # scoring an infinite delta-over-range ratio.
    assert by_id["bump_caster_deg"]["ratio"] == float("inf")
    assert rows[0]["meta"]["id"] == "bump_camber_deg"


def test_rear_caster_is_relabelled_as_a_steering_axis_angle() -> None:
    caster = next(m for m in sor.KINEMATIC_CURVE_META if m["id"] == "bump_caster_deg")
    assert sor.display_label(caster, "front") == caster["label"]
    rear = sor.display_label(caster, "rear")
    # Named for the steering axis it actually describes, not "caster", which implies
    # a steered axle. Kept short enough to render without clipping in the PDF table.
    assert "Kingpin side-view inclination" in rear
    assert "caster" not in rear.lower()

    # Only caster is relabelled; every other curve keeps its registry name.
    camber = next(m for m in sor.KINEMATIC_CURVE_META if m["id"] == "bump_camber_deg")
    assert sor.display_label(camber, "rear") == camber["label"]

    # The axis label must move with the title; a plot headed "Kingpin side-view
    # inclination" whose y axis still reads "Caster" is worse than no rename.
    assert sor.display_y_label(caster, "rear") == "Kingpin side-view inclination"
    assert sor.display_y_label(caster, "front") == caster["y_label"]
    assert sor.display_y_label(camber, "rear") == camber["y_label"]


def test_z_dependent_curves_are_withheld_but_angles_are_not() -> None:
    """A rigid vertical translation moves heights, not angles or lengths."""
    assert "bump_rc_height_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_swing_arm_mm" in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_rc_z_mm" in sor.Z_DEPENDENT_CURVE_IDS
    # Lateral and angular quantities survive the translation, so they still publish.
    assert "bump_rc_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_front_ic_y_mm" not in sor.Z_DEPENDENT_CURVE_IDS
    assert "bump_camber_deg" not in sor.Z_DEPENDENT_CURVE_IDS
