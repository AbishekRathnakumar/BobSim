from __future__ import annotations

from pathlib import Path

import pytest

from _0_Utils.shark_import import (
    SharkImportError,
    assess_z_datum,
    build_axle_block,
    check_single_side,
    detect_axle,
    import_shark,
    parse_shark,
    verify_shared_frame,
)
from _0_Utils.vehicle_io import load_yaml, repo_root


SHARK_FIXTURE = Path(__file__).parent / "fixtures" / "rear_pushrod.shk"

pytestmark = pytest.mark.skipif(
    not SHARK_FIXTURE.is_file(), reason="SHARK fixture not present"
)


def _baseline() -> dict:
    return load_yaml(repo_root() / "vehicle.yml")


def test_parse_shark_pairs_names_to_coordinates() -> None:
    points = parse_shark(SHARK_FIXTURE)
    assert points["__template__"] == "Double Wishbone, Push Rod to damper"
    # Paired by name, so a layout change fails loudly rather than mis-mapping.
    assert points["Wheel centre point"] == pytest.approx(
        (-1549.40002441, 606.109985352, 201.088119507)
    )


def test_axle_is_detected_from_coordinates_not_the_section_label() -> None:
    """The SHARK section is labelled FRONT SUSPENSION but the geometry is rear."""
    points = parse_shark(SHARK_FIXTURE)
    axle, offset = detect_axle(points, _baseline())
    assert axle == "rear"
    assert offset < 0.05


def test_shared_frame_is_verified_not_assumed() -> None:
    points = parse_shark(SHARK_FIXTURE)
    offsets = verify_shared_frame(points, _baseline(), "rear")
    assert abs(offsets["dx_mm"]) < 0.05
    assert abs(offsets["dy_mm"]) < 0.05


def test_mismatched_frame_is_refused() -> None:
    points = parse_shark(SHARK_FIXTURE)
    shifted = dict(points)
    x, y, z = shifted["Wheel centre point"]
    shifted["Wheel centre point"] = (x + 50.0, y, z)
    with pytest.raises(SharkImportError, match="shared coordinate frame"):
        verify_shared_frame(shifted, _baseline(), "rear")


def test_two_sided_geometry_is_refused_not_averaged() -> None:
    points = {"a": (0.0, 100.0, 0.0), "b": (0.0, -100.0, 0.0)}
    with pytest.raises(SharkImportError, match="spans both sides"):
        check_single_side(points)


def test_z_datum_is_reported_unresolved_when_the_evidence_conflicts() -> None:
    """The file's own loaded radius contradicts its wheel centre.

    Read as a shared ground-plane datum, the trailing scalar puts the baseline
    contact patch on z = 0 but the SHARK one 1.198 mm above it. That is worse than
    no evidence, so the importer must not take z raw and call it a ride height.
    """
    points = parse_shark(SHARK_FIXTURE)
    datum = assess_z_datum(points, _baseline(), "rear")
    assert datum["status"] == "unresolved"
    assert datum["loaded_radius_mm"] == pytest.approx(199.89, abs=1e-3)
    assert datum["baseline_implied_contact_patch_mm"] == pytest.approx(0.008, abs=1e-3)
    assert datum["shark_implied_contact_patch_mm"] == pytest.approx(1.198, abs=1e-3)
    assert "NOT confirmed as a ride-height change" in datum["reason"]


def test_z_datum_is_resolved_when_both_cars_sit_on_the_ground_plane() -> None:
    """A file whose wheel centre is one loaded radius up needs no caveat."""
    points = dict(parse_shark(SHARK_FIXTURE))
    baseline = _baseline()
    radius = points["__loaded_radius_mm__"]
    x, y, _ = points["Wheel centre point"]
    points["Wheel centre point"] = (x, y, radius)
    baseline["rear"]["suspension"]["wheel_center_m"] = [x / 1000.0, y / 1000.0, radius / 1000.0]
    assert assess_z_datum(points, baseline, "rear")["status"] == "shared_ground_plane"


def test_datum_evidence_is_parsed_from_the_file_not_assumed() -> None:
    points = parse_shark(SHARK_FIXTURE)
    assert points["__loaded_radius_mm__"] == pytest.approx(199.89, abs=1e-3)
    # This file declares zero TITLES entries, so there is no annotation to settle it.
    assert points["__titles__"] == []


def test_import_replaces_only_the_rear_suspension() -> None:
    merged, report = import_shark(SHARK_FIXTURE)
    baseline = _baseline()
    assert report["axle"] == "rear"
    # Front axle and non-suspension data are untouched.
    assert merged["front"] == baseline["front"]
    assert merged["sprung_mass"] == baseline["sprung_mass"]
    assert merged["aero"] == baseline["aero"]
    assert merged["powertrain"] == baseline["powertrain"]
    # Rear hardpoints came from the SHARK file.
    assert merged["rear"]["suspension"]["upper_o_m"] == pytest.approx(
        [-1.578765381, 0.534915771, 0.252956787], abs=1e-6
    )


def test_front_import_merges_into_a_rear_imported_vehicle() -> None:
    """A later front file must merge in without re-running the rear.

    Synthesised by translating the rear corner forward onto the front axle, which
    lands within the frame tolerance because the two axles share a track width. That
    is enough to exercise the merge path: the front block is rebuilt while the
    already-imported rear survives byte-identical.
    """
    rear_merged, _ = import_shark(SHARK_FIXTURE)
    wheelbase_mm = -rear_merged["rear"]["suspension"]["wheel_center_m"][0] * 1000.0

    points = parse_shark(SHARK_FIXTURE)
    shifted: dict = {
        name: (value[0] + wheelbase_mm, value[1], value[2]) if not name.startswith("__") else value
        for name, value in points.items()
    }

    axle, _offset = detect_axle(shifted, rear_merged)
    assert axle == "front"
    verify_shared_frame(shifted, rear_merged, "front")

    front_block, _notes = build_axle_block(shifted, rear_merged["front"], keep_stabar=False)
    merged = {**rear_merged, "front": front_block}

    assert merged["rear"] == rear_merged["rear"]
    assert merged["front"] != rear_merged["front"]


def test_reimport_judges_the_datum_against_orion_not_the_imported_car(tmp_path: Path) -> None:
    """Re-importing must not compare the SHARK file against itself.

    Merging a second axle uses the already-imported car as the merge target. If the
    datum were judged against that same car, dz collapses to zero and the evidence
    becomes self-referential - which can report a shared ground plane that was never
    established and silently un-withhold the z-dependent curves.
    """
    from _0_Utils.shark_import import write_vehicle

    first, first_report = import_shark(SHARK_FIXTURE)
    variant = tmp_path / "vehicle_variant.yml"
    write_vehicle(first, variant)
    assert first_report["datum"]["dz_mm"] == pytest.approx(1.190, abs=1e-3)

    # Merge target is the imported car; datum reference stays the original baseline.
    _second, second_report = import_shark(
        SHARK_FIXTURE,
        baseline_path=variant,
        datum_baseline_path=repo_root() / "vehicle.yml",
    )
    datum = second_report["datum"]
    assert datum["status"] == "unresolved"
    assert datum["dz_mm"] == pytest.approx(1.190, abs=1e-3)
    assert datum["baseline_implied_contact_patch_mm"] == pytest.approx(0.008, abs=1e-3)

    # Without the guard the comparison degenerates: dz vanishes and both contact
    # patches land on the same value, so the evidence says nothing.
    _third, degenerate = import_shark(SHARK_FIXTURE, baseline_path=variant)
    assert degenerate["datum"]["dz_mm"] == pytest.approx(0.0, abs=1e-5)


def test_rod_attachment_is_derived_from_geometry() -> None:
    """This SHARK corner picks the pushrod up on the lower arm.

    Asserted against the reported change rather than the live vehicle.yml: the
    overlay pipeline swaps that file in place while it runs, so reading it here
    makes the suite timing-dependent.
    """
    from _0_Utils.shark_import import _rod_attachment

    points = parse_shark(SHARK_FIXTURE)
    assert _rod_attachment(points) == "lower"


def test_carrying_the_arb_onto_a_moved_bellcrank_is_refused() -> None:
    """The baseline ARB pickup is meaningless once the rocker pivot moves.

    Only reachable via the opt-in --keep-arb path now, but still enforced there:
    transplanting it compiles and then fails to solve part-way through the sweep,
    so the importer must reject it up front rather than after a long build.
    """
    with pytest.raises(SharkImportError, match="Cannot carry the baseline anti-roll bar"):
        import_shark(SHARK_FIXTURE, keep_stabar=True)


def test_coherent_stabar_pickup_is_accepted() -> None:
    """A pickup in scale with the other rocker arms passes the coherence check."""
    from _0_Utils.shark_import import _check_carried_stabar_is_coherent

    _check_carried_stabar_is_coherent(
        {
            "pivot_m": [0.0, 0.0, 0.0],
            "pickups_m": {
                "rod": [0.06, 0.0, 0.0],
                "shock": [0.0, 0.09, 0.0],
                "stabar": [0.0, 0.0, 0.03],
            },
        }
    )


def test_stabar_is_dropped_by_default() -> None:
    """ARB is out of scope: it takes no part in the kinematic solve.

    Dropping it is the default so the ordinary import path never has to make a
    judgement call about a bar the file does not define.
    """
    dropped, _ = import_shark(SHARK_FIXTURE)
    assert "stabar" not in dropped["rear"]["actuation"]
    assert "stabar" not in dropped["rear"]["actuation"]["bellcrank"]["pickups_m"]
    assert dropped["architecture"]["rear"] == "bellcrank"


def test_kinematic_solve_ignores_actuation_entirely() -> None:
    """The claim that ARB is out of scope, asserted rather than assumed.

    If the solver ever starts reading actuation, dropping the bar by default would
    silently change kinematic results and this test should fail loudly.
    """
    from _5_App.kinematics import CornerKinematics

    baseline = _baseline()
    stripped = load_yaml(repo_root() / "vehicle.yml")
    stripped["rear"].pop("actuation", None)

    sweep = (-0.02, 0.0, 0.02)
    with_actuation = CornerKinematics.from_vehicle(baseline, "rear").solve_sweep(sweep, roll_deg=())
    without = CornerKinematics.from_vehicle(stripped, "rear").solve_sweep(sweep, roll_deg=())
    assert with_actuation["curves"] == without["curves"]


def test_datum_sidecar_round_trips(tmp_path: Path) -> None:
    """The datum caveat must survive a later run that does not re-import."""
    from _0_Utils.shark_import import read_datum_status, write_datum_sidecar, write_vehicle

    merged, report = import_shark(SHARK_FIXTURE)
    target = tmp_path / "vehicle_variant.yml"
    write_vehicle(merged, target)
    write_datum_sidecar(target, report["datum"])
    assert read_datum_status(target) == "unresolved"
    # A vehicle with no sidecar carries no claim either way.
    assert read_datum_status(tmp_path / "absent.yml") is None


def test_missing_required_points_fail_loudly() -> None:
    points = parse_shark(SHARK_FIXTURE)
    del points["Upper wishbone outer ball joint"]
    with pytest.raises(SharkImportError, match="pushrod/bellcrank|Upper wishbone"):
        build_axle_block(points, _baseline()["rear"], keep_stabar=True)


def test_missing_required_points_fail_loudly_for_bellcrank() -> None:
    points = parse_shark(SHARK_FIXTURE)
    del points["Rocker axis 1st point"]
    with pytest.raises(SharkImportError, match="pushrod/bellcrank"):
        build_axle_block(points, _baseline()["rear"], keep_stabar=False)
