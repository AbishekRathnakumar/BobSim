"""Overlay a SHARK-imported car against the Orion baseline on the kinematic curves.

The primary product is a *kinematics* overlay: every curve in the app's
`KINEMATIC_CURVE_META` registry, solved straight from the hardpoints, for both
axles and both sweeps. No Modelica build is needed, because the kinematic solver
reads only `suspension`, `steering` and `wheel` - the anti-roll bar, bellcrank and
dampers take no part in it.

The four-post force sim remains available behind `--four-post`. It is secondary and
experimental: it depends on actuation data that this workflow maintains outside
BobSim, so its numbers can conflate a hardpoint change with an ARB or damper change.

Baseline is `vehicle.yml` (Orion) and is never written to. The imported car lives in
a tracked file (`vehicle_2027.yml` by default), not in generated_results/, so a later
front-axle import can merge into the same file instead of re-running the rear.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterator

import numpy as np
import yaml

from _0_Utils.shark_import import (
    Z_DATUM_UNRESOLVED_WARNING,
    Z_DEPENDENT_CURVE_IDS,
    import_shark,
    read_datum_status,
    write_datum_sidecar,
    write_vehicle,
)
from _5_App.kinematics import KINEMATIC_CURVE_META, kinematic_curves_payload
from _5_App.modelica_generator import generate_modelica_stack, modelica_stack_status_payload


ROOT = Path(__file__).resolve().parents[2]
VEHICLE_YAML = ROOT / "vehicle.yml"
DEFAULT_VARIANT_YAML = ROOT / "vehicle_2027.yml"
BUILD_DIR = ROOT / "_3_StandardSim/BuildBobLib/FourPostSim"
GEOMETRY_STAMP = BUILD_DIR / ".bobsim_geometry_stamp.json"
OUT_DIR = ROOT / "_3_StandardSim/generated_results"

BASELINE_LABEL = "Orion"
VARIANT_LABEL = "2027"

# Validated categorical slots 1 and 2 (see dataviz palette).
# node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light -> ALL CHECKS PASS
COLOR_BASELINE = "#2a78d6"
COLOR_VARIANT = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6b68"
GRID = "#e4e4e1"
SURFACE = "#fcfcfb"
WARN = "#a8341a"

# A curve reaches the headline page if it moves by at least this fraction of the
# baseline curve's own range. Relative rather than absolute because the deck mixes
# degrees and millimetres, which have no common scale.
MEANINGFUL_RELATIVE_DELTA = 0.02
HEADLINE_PANEL_LIMIT = 8


class StaleGeometryError(RuntimeError):
    """Raised when the built simulator does not match the current geometry."""


# --------------------------------------------------------------------------
# Kinematics (primary)
# --------------------------------------------------------------------------


def kinematic_payload(vehicle_path: Path) -> dict[str, Any]:
    """Solve the full registry curve deck at the app's default sweep ranges.

    Sweep and roll ranges are deliberately left to `kinematic_curves_payload`
    defaults so this output is directly comparable to anything the app renders.
    """
    vehicle = yaml.safe_load(vehicle_path.read_text(encoding="utf-8"))
    return kinematic_curves_payload(vehicle)


def _curve_series(
    payload: dict[str, Any], axle: str, meta: dict[str, str]
) -> tuple[list[float], list[float | None]] | None:
    axle_payload = payload.get("axles", {}).get(axle) or {}
    curves = axle_payload.get("curves") or {}
    values = curves.get(meta["id"])
    x = payload.get("x_axes", {}).get(meta["x_id"])
    if not values or not x:
        return None
    size = min(len(x), len(values))
    return list(x[:size]), list(values[:size])


def _delta_score(
    base: Sequence[float | None], variant: Sequence[float | None]
) -> dict[str, float] | None:
    """Peak divergence between two curves, in curve units and relative to range.

    Both are reported because neither is sufficient alone. The absolute peak is the
    engineering quantity, but degrees and millimetres cannot be ranked against each
    other; the ratio makes them comparable. The ratio alone is misleading on a rear
    axle, where the baseline caster/trail/scrub curves are nearly flat and any change
    divides by ~zero into a meaningless four-digit percentage.
    """
    pairs = [(b, v) for b, v in zip(base, variant) if b is not None and v is not None]
    if len(pairs) < 2:
        return None
    peak = max(abs(v - b) for b, v in pairs)
    base_values = [b for b, _ in pairs]
    span = max(base_values) - min(base_values)
    if peak <= 1e-9:
        ratio = 0.0
    elif span <= 1e-9:
        ratio = float("inf")  # flat baseline; any movement at all is notable
    else:
        ratio = peak / span
    return {"peak": peak, "span": span, "ratio": ratio}


def rank_curves(
    payloads: dict[str, dict[str, Any]], withheld: frozenset[str]
) -> list[tuple[dict[str, str], str, dict[str, float]]]:
    """Rank (curve, axle) pairs by how much the import moved them."""
    ranked: list[tuple[dict[str, str], str, dict[str, float]]] = []
    for meta in KINEMATIC_CURVE_META:
        if meta["id"] in withheld:
            continue
        for axle in ("front", "rear"):
            base = _curve_series(payloads[BASELINE_LABEL], axle, meta)
            var = _curve_series(payloads[VARIANT_LABEL], axle, meta)
            if base is None or var is None:
                continue
            score = _delta_score(base[1], var[1])
            if score is None:
                continue
            ranked.append((meta, axle, score))
    ranked.sort(key=lambda item: item[2]["ratio"], reverse=True)
    return ranked


# --------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------


def _style_axis(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=9, color=INK, pad=6, loc="left")
    ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)


def _plot_curve(
    ax: Any,
    payloads: dict[str, dict[str, Any]],
    meta: dict[str, str],
    axle: str,
    *,
    show_legend: bool,
) -> None:
    series = {
        label: _curve_series(payload, axle, meta) for label, payload in payloads.items()
    }
    drawn = {label: value for label, value in series.items() if value is not None}
    identical = False
    if len(drawn) == 2:
        base_y = drawn[BASELINE_LABEL][1]
        var_y = drawn[VARIANT_LABEL][1]
        identical = all(
            (b is None and v is None) or (b is not None and v is not None and abs(v - b) <= 1e-9)
            for b, v in zip(base_y, var_y)
        )

    for label, color in ((BASELINE_LABEL, COLOR_BASELINE), (VARIANT_LABEL, COLOR_VARIANT)):
        if label not in drawn:
            continue
        x, y = drawn[label]
        xs = np.asarray(x, dtype=float)
        ys = np.asarray([np.nan if value is None else value for value in y], dtype=float)
        ax.plot(
            xs, ys, color=color, linewidth=1.8, label=label, zorder=3,
            linestyle=(0, (4, 3)) if identical and label == VARIANT_LABEL else "-",
        )

    _style_axis(
        ax,
        f"{meta['x_label']} ({meta['x_unit']})",
        f"{meta['y_label']} ({meta['unit']})",
        f"{axle.title()} - {meta['label']}",
    )
    if identical:
        ax.text(
            0.5, 0.06, "curves coincide - axle unchanged by this import",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="center",
        )
    if show_legend:
        ax.legend(frameon=False, fontsize=7, labelcolor=INK)


def _text_page(pdf: Any, title: str, lines: Sequence[str], *, warn: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.0, 8.5), facecolor=SURFACE)
    fig.text(0.04, 0.94, title, fontsize=15, color=INK, va="top", ha="left")
    body = "\n".join(lines)
    fig.text(
        0.04, 0.88, body, fontsize=9, color=WARN if warn else MUTED,
        va="top", ha="left", wrap=True,
    )
    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def _grid_page(
    pdf: Any,
    panels: Sequence[tuple[dict[str, str], str]],
    payloads: dict[str, dict[str, Any]],
    title: str,
    *,
    ncols: int = 2,
    nrows: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 8.5), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.set_visible(False)

    for index, (meta, axle) in enumerate(panels):
        ax = axes.flat[index]
        ax.set_visible(True)
        _plot_curve(ax, payloads, meta, axle, show_legend=index == 0)

    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    pdf.savefig(fig, facecolor=SURFACE)
    plt.close(fig)


def build_report(
    payloads: dict[str, dict[str, Any]],
    notes: Sequence[str],
    withheld: frozenset[str],
    ranked: Sequence[tuple[dict[str, str], str, dict[str, float]]],
    four_post: dict[str, Any] | None,
    out_path: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (backend must be set first)
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    headline = [
        (meta, axle)
        for meta, axle, score in ranked
        if score["ratio"] >= MEANINGFUL_RELATIVE_DELTA
    ][:HEADLINE_PANEL_LIMIT]

    with PdfPages(out_path) as pdf:
        cover = [f"{BASELINE_LABEL} (vehicle.yml) vs {VARIANT_LABEL} kinematic curves.", ""]
        cover += [f"- {note}" for note in notes]
        if withheld:
            cover += ["", Z_DATUM_UNRESOLVED_WARNING, "", "Withheld curves:"]
            cover += [f"  - {curve_id}" for curve_id in sorted(withheld)]
        _text_page(pdf, "SHARK import overlay", cover, warn=bool(withheld))

        if headline:
            _grid_page(
                pdf, headline[:6], payloads,
                "Headline - curves with a meaningful Orion vs 2027 delta",
            )
            if len(headline) > 6:
                _grid_page(
                    pdf, headline[6:], payloads,
                    "Headline (continued)",
                )
        else:
            _text_page(
                pdf, "Headline",
                ["No curve moved by more than "
                 f"{MEANINGFUL_RELATIVE_DELTA:.0%} of its baseline range."],
            )

        appendix = [
            (meta, axle)
            for meta in KINEMATIC_CURVE_META
            if meta["id"] not in withheld
            for axle in ("front", "rear")
        ]
        for start in range(0, len(appendix), 6):
            page = start // 6 + 1
            total = (len(appendix) + 5) // 6
            _grid_page(
                pdf, appendix[start:start + 6], payloads,
                f"Appendix - full curve deck ({page}/{total})",
            )

        if four_post is not None:
            _text_page(pdf, "Four-post (experimental, secondary)", four_post["lines"], warn=True)

    return out_path


# --------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------


def write_summary_md(
    notes: Sequence[str],
    withheld: frozenset[str],
    ranked: Sequence[tuple[dict[str, str], str, dict[str, float]]],
    four_post: dict[str, Any] | None,
    path: Path,
) -> Path:
    lines = [f"# {VARIANT_LABEL} vs {BASELINE_LABEL} - kinematic overlay", "", "## Notes", ""]
    lines += [f"- {note}" for note in notes]

    if withheld:
        lines += [
            "",
            "## Withheld pending datum resolution",
            "",
            Z_DATUM_UNRESOLVED_WARNING,
            "",
        ]
        lines += [f"- `{curve_id}`" for curve_id in sorted(withheld)]

    meaningful = [item for item in ranked if item[2]["ratio"] >= MEANINGFUL_RELATIVE_DELTA]
    lines += [
        "",
        "## Curves that moved",
        "",
        "Peak delta is the engineering quantity. The baseline range beside it is what that",
        "delta should be read against - several rear curves (caster, trail, scrub) are",
        "nearly flat on Orion, so a small absolute change is a large relative one. Ordering",
        "is by the ratio, which is the only way to rank degrees against millimetres.",
        "",
        "| Curve | Axle | Peak delta | Baseline range | Ratio |",
        "|---|---|---:|---:|---:|",
    ]
    if meaningful:
        for meta, axle, score in meaningful:
            ratio = score["ratio"]
            shown = "baseline flat" if ratio == float("inf") else f"{ratio:.1f}x"
            unit = meta["unit"]
            lines.append(
                f"| {meta['label']} | {axle} | {score['peak']:.3f} {unit} "
                f"| {score['span']:.3f} {unit} | {shown} |"
            )
    else:
        lines.append("| _none above threshold_ | | | | |")

    unchanged = [item for item in ranked if item[2]["ratio"] == 0.0]
    lines += [
        "",
        f"{len(unchanged)} of {len(ranked)} curve/axle pairs are bit-identical between the "
        "two cars (the axle this import does not touch).",
    ]

    if four_post is not None:
        lines += ["", "## Four-post (experimental, secondary)", ""] + list(four_post["lines"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Four-post (opt-in, secondary)
# --------------------------------------------------------------------------


def four_post_signature(vehicle_path: Path) -> tuple[str, dict[str, Any]]:
    """Regenerate the stack and return the four-post content signature."""
    generate_modelica_stack(vehicle_path, root=ROOT)
    status = modelica_stack_status_payload(vehicle_path, ROOT)
    if status["state"] != "written":
        stale = [f["kind"] for f in status["files"] if not f["current"]]
        raise StaleGeometryError(
            f"Modelica stack did not land cleanly (state={status['state']!r}). "
            f"Files not matching the generated content: {', '.join(stale) or 'none'}. "
            "Refusing to run the sim."
        )
    return status["signatures"]["four_post"]["generated"], status


def read_stamp() -> str | None:
    if not GEOMETRY_STAMP.is_file():
        return None
    try:
        return str(json.loads(GEOMETRY_STAMP.read_text(encoding="utf-8"))["signature"])
    except Exception:
        return None


def write_stamp(signature: str, vehicle_name: str) -> None:
    GEOMETRY_STAMP.parent.mkdir(parents=True, exist_ok=True)
    GEOMETRY_STAMP.write_text(
        json.dumps({"signature": signature, "vehicle": vehicle_name}, indent=2),
        encoding="utf-8",
    )


def build_four_post() -> None:
    result = subprocess.run(
        ["make", "standard-build-four-post"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StaleGeometryError(
            "Four-post build failed, so the simulator cannot be proven to match the "
            "current geometry. Refusing to report.\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
        )


def four_post_executable() -> Path | None:
    for name in (
        "BobLib.Experiments.Standards.FourPostSim",
        "BobLib.Experiments.Standards.FourPostSim.exe",
    ):
        candidate = BUILD_DIR / name
        if candidate.is_file():
            return candidate
    return None


def assert_binary_consumed_geometry(status: dict[str, Any], label: str) -> None:
    """Prove the executable was produced *after* the geometry it claims to model.

    Deliberately independent of the makefile: if a dependency is ever missing
    again, `make` reports success without recompiling, and stamping the new
    signature onto that untouched binary would launder stale geometry into a
    report that looks clean. Compare timestamps instead of trusting the build.
    """
    exe = four_post_executable()
    if exe is None:
        raise StaleGeometryError(
            f"[{label}] Build reported success but no four-post executable exists in {BUILD_DIR}."
        )
    newest_geometry = float(status.get("latest_modified") or 0.0)
    built_at = exe.stat().st_mtime
    if newest_geometry and built_at < newest_geometry:
        raise StaleGeometryError(
            f"[{label}] The four-post executable predates the geometry it should model.\n"
            f"  executable mtime : {built_at:.0f}\n"
            f"  geometry mtime   : {newest_geometry:.0f}\n"
            "`make` reported success without recompiling, so the binary still holds the "
            "previous hardpoints. Check the build dependencies for the generated record."
        )


@contextlib.contextmanager
def installed_vehicle(source: Path) -> Iterator[Path]:
    """Temporarily install `source` as the repo vehicle.yml, always restoring it.

    The four-post stack reads the repo vehicle.yml by construction, so the opt-in
    sim path has to swap it. vehicle.yml is restored in a finally block; the
    kinematics path never touches it at all.
    """
    backup = VEHICLE_YAML.with_suffix(".yml.overlay-backup")
    shutil.copy2(VEHICLE_YAML, backup)
    try:
        if source.resolve() != VEHICLE_YAML.resolve():
            shutil.copy2(source, VEHICLE_YAML)
        yield VEHICLE_YAML
    finally:
        shutil.copy2(backup, VEHICLE_YAML)
        backup.unlink(missing_ok=True)


def run_four_post(vehicle_path: Path, label: str, *, skip_build: bool) -> dict[str, Any]:
    """Run the four-post eval against `vehicle_path`, with a stale-geometry guard."""
    from _3_StandardSim.FourPostEval import four_post_eval_sim as fp

    with installed_vehicle(vehicle_path):
        signature, status = four_post_signature(VEHICLE_YAML)
        if not skip_build:
            build_four_post()
            assert_binary_consumed_geometry(status, label)
            write_stamp(signature, status["vehicle_name"])

        stamped = read_stamp()
        if stamped != signature:
            raise StaleGeometryError(
                f"[{label}] The built four-post simulator does not match the current "
                f"geometry.\n  generated signature : {signature}\n"
                f"  built-from signature: {stamped}\n"
                "The executable was compiled from different hardpoints. Rebuild with "
                "`make standard-build-four-post` (do not pass --skip-build)."
            )

        config = fp.load_config(fp.DEFAULT_CONFIG_PATH)
        report_cfg = config.setdefault("report", {})
        report_cfg["enabled"] = False
        # Keep the overlay's metrics out of the canonical four-post CSV: that file is
        # the repo's regression baseline, and it is also read back to seed spring free
        # lengths, so sharing it would let one car's results leak into the other's.
        slug = "baseline" if label == BASELINE_LABEL else "variant"
        report_cfg["metrics_csv_path"] = str(OUT_DIR / f"shark_overlay_metrics_{slug}.csv")
        result = fp.FourPostEvalSim(config).run()
        return {"summary": result["summary"], "series": result["series"]}


SCALAR_METRICS = (
    ("avg_anti_dive_pct", "Anti-dive (%)"),
    ("avg_anti_squat_pct", "Anti-squat (%)"),
    ("avg_anti_roll_front_pct", "Anti-roll front (%)"),
    ("avg_anti_roll_rear_pct", "Anti-roll rear (%)"),
)


def four_post_section(runs: dict[str, Any], *, arb_kept: bool) -> dict[str, Any]:
    """Render the secondary four-post block, with its caveats attached."""
    lines = [
        "EXPERIMENTAL / SECONDARY. The four-post path depends on actuation data - ARB,",
        "bellcrank and damper - that this workflow maintains in SolidWorks and Excel,",
        "outside BobSim. The kinematic curves above do not depend on any of it.",
        "",
    ]
    if not arb_kept:
        lines += [
            "The anti-roll bar was not imported, so any anti-roll or roll-stiffness number",
            "below conflates the hardpoint change with a removed bar. Do not read it as a",
            "geometry result.",
            "",
        ]
    base = (runs.get(BASELINE_LABEL) or {}).get("summary") or {}
    var = (runs.get(VARIANT_LABEL) or {}).get("summary") or {}
    if base and var:
        lines += [f"| Metric | {BASELINE_LABEL} | {VARIANT_LABEL} | Delta |", "|---|---:|---:|---:|"]
        for key, label in SCALAR_METRICS:
            b, v = base.get(key, float("nan")), var.get(key, float("nan"))
            lines.append(f"| {label} | {b:.3f} | {v:.3f} | {v - b:+.3f} |")
    else:
        lines.append("Four-post produced no summary.")
    return {"lines": lines}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overlay a SHARK-imported car against the Orion baseline"
    )
    parser.add_argument(
        "--shark",
        default=None,
        help="Import this .shk into the variant file before overlaying (optional)",
    )
    parser.add_argument("--baseline", default=str(VEHICLE_YAML), help="Baseline vehicle.yml (Orion)")
    parser.add_argument(
        "--variant", default=str(DEFAULT_VARIANT_YAML),
        help="Tracked vehicle.yml holding the imported car",
    )
    parser.add_argument(
        "--import-baseline", default=None,
        help=(
            "Baseline for the import step. Defaults to the variant file when it already "
            "exists, so a later front-axle import merges into it instead of replacing it."
        ),
    )
    parser.add_argument(
        "--keep-arb", action="store_true",
        help="Carry the baseline ARB across (four-post only; irrelevant to kinematics)",
    )
    parser.add_argument(
        "--four-post", action="store_true",
        help="Also run the experimental four-post force sim (needs a Modelica build)",
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not rebuild (guard still enforced)")
    parser.add_argument("--out", default=str(OUT_DIR / "shark_overlay_report.pdf"))
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    variant_path = Path(args.variant)
    notes: list[str] = []
    withheld: frozenset[str] = frozenset()

    if args.shark:
        import_baseline = Path(args.import_baseline) if args.import_baseline else (
            variant_path if variant_path.is_file() else baseline_path
        )
        merged, report = import_shark(
            args.shark,
            baseline_path=import_baseline,
            keep_stabar=args.keep_arb,
            vehicle_name=VARIANT_LABEL,
        )
        write_vehicle(merged, variant_path)
        write_datum_sidecar(variant_path, report["datum"])
        notes += list(report["notes"])
        notes.append(f"Imported {args.shark} onto {import_baseline.name} -> {variant_path.name}")
        if report["datum"]["status"] != "shared_ground_plane":
            withheld = frozenset(Z_DEPENDENT_CURVE_IDS)
    else:
        notes.append(f"No --shark given; overlaying the tracked {variant_path.name} as it stands.")

    if not variant_path.is_file():
        print(f"No variant vehicle at {variant_path}. Pass --shark to create it.", file=sys.stderr)
        return 1

    # The datum caveat is a property of the imported file, not of this invocation, so
    # it must survive a run that does not re-import.
    if not withheld and read_datum_status(variant_path) not in (None, "shared_ground_plane"):
        withheld = frozenset(Z_DEPENDENT_CURVE_IDS)

    payloads = {
        BASELINE_LABEL: kinematic_payload(baseline_path),
        VARIANT_LABEL: kinematic_payload(variant_path),
    }
    for label, payload in payloads.items():
        for warning in payload.get("warnings", []):
            notes.append(f"{label}: {warning}")

    notes.append(
        "Sweep ranges are the app registry defaults, so these curves are directly "
        "comparable to the app's kinematics view."
    )
    notes.append(
        "The anti-roll bar takes no part in the kinematic solve; it is handled outside "
        "this tool."
    )

    ranked = rank_curves(payloads, withheld)

    four_post: dict[str, Any] | None = None
    if args.four_post:
        runs: dict[str, Any] = {}
        for label, path in ((BASELINE_LABEL, baseline_path), (VARIANT_LABEL, variant_path)):
            try:
                runs[label] = run_four_post(path, label, skip_build=args.skip_build)
            except StaleGeometryError as exc:
                print(f"REFUSING TO REPORT\n\n{exc}", file=sys.stderr)
                return 1
        four_post = four_post_section(runs, arb_kept=args.keep_arb)

    out = build_report(payloads, notes, withheld, ranked, four_post, Path(args.out))
    md = write_summary_md(notes, withheld, ranked, four_post, Path(args.out).with_suffix(".md"))
    print(f"Wrote {out}")
    print(f"Wrote {md}")
    if withheld:
        print(f"  WARNING: {Z_DATUM_UNRESOLVED_WARNING}")
    for note in notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
