"""DOE plumbing checks for the StandardSens sweep.

These cover the pure-Python half of the pipeline — config generation, sampling
the baseline BobLib record, and writing variant Modelica — so a broken DOE is
caught without an OpenModelica toolchain. The compile/simulate stages are not
exercised here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
OPTSIM_DIR = ROOT / "_4_OptSim"
ARCHITECTURE_CONFIG = OPTSIM_DIR / "StandardSens/configs/vehicle_architecture.yaml"

# The OptSim workflows are invoked with _4_OptSim on the path (see the opt-*
# make targets), so mirror that here rather than importing via the _4_OptSim
# package prefix.
if str(OPTSIM_DIR) not in sys.path:
    sys.path.insert(0, str(OPTSIM_DIR))

pytest.importorskip("scipy", reason="DOE sampling requires scipy")

from StandardSens.pipeline import generate_configs, generator, search  # noqa: E402


def _localize(doe_config_path: Path) -> Path:
    """Rewrite the config's relative paths as absolute ones.

    `build_doe_config` always emits paths relative to the real configs/
    directory, so a config generated into a tmp dir cannot resolve them. Tests
    write outside the repo to avoid dirtying the checked-in config, so resolve
    the references against their true base here.
    """
    cfg = yaml.safe_load(doe_config_path.read_text())
    config_dir = generate_configs.DOE_CONFIG.parent
    cfg["baseline_mo"] = str((config_dir / cfg["baseline_mo"]).resolve())
    cfg["architecture"]["template"] = str(
        (config_dir.parent / cfg["architecture"]["template"]).resolve()
    )
    doe_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return doe_config_path


def _boblib_record_available(doe_config_path: Path) -> bool:
    cfg = yaml.safe_load(doe_config_path.read_text())
    return Path(cfg["baseline_mo"]).is_file()


@pytest.fixture
def doe_config(tmp_path: Path) -> Path:
    """A freshly generated DOE config written outside the repo tree."""
    out = tmp_path / "_doe_config.yaml"
    generate_configs.refresh_doe_config(
        architecture_config_path=ARCHITECTURE_CONFIG,
        compiler_config_path=generate_configs.COMPILER_CONFIG,
        doe_config_path=out,
    )
    return out


def test_generated_config_uses_posix_separators(doe_config: Path) -> None:
    """Backslashes here are a single opaque filename inside the Linux container."""
    text = doe_config.read_text()
    assert "\\" not in text, "DOE config must not contain native Windows separators"

    cfg = yaml.safe_load(text)
    assert "/" in cfg["baseline_mo"]
    assert "/" in cfg["architecture"]["template"]


def test_checked_in_config_matches_regeneration(doe_config: Path) -> None:
    """The committed _doe_config.yaml should not drift from its source.

    Compares the committed blob rather than the working copy: a local run with
    DOE_SAMPLES/DOE_METHOD overrides legitimately rewrites the file on disk.
    """
    import subprocess

    rel = generate_configs.DOE_CONFIG.relative_to(ROOT).as_posix()
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is unavailable or the config is not committed")

    assert yaml.safe_load(blob) == yaml.safe_load(doe_config.read_text()), (
        "_doe_config.yaml is stale or was hand-edited; it is generated from "
        "configs/vehicle_architecture.yaml"
    )


def test_sample_count_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOBSIM_DOE_METHOD", "lhs")
    monkeypatch.setenv("BOBSIM_DOE_SAMPLES", "3")
    out = tmp_path / "_doe_config.yaml"
    generate_configs.refresh_doe_config(
        architecture_config_path=ARCHITECTURE_CONFIG,
        compiler_config_path=generate_configs.COMPILER_CONFIG,
        doe_config_path=out,
    )
    cfg = yaml.safe_load(out.read_text())
    assert cfg["sampling"]["method"] == "lhs"
    assert cfg["samples"] == 3

    monkeypatch.setenv("BOBSIM_DOE_SAMPLES", "not-a-number")
    with pytest.raises(ValueError, match="BOBSIM_DOE_SAMPLES"):
        generate_configs.refresh_doe_config(
            architecture_config_path=ARCHITECTURE_CONFIG,
            compiler_config_path=generate_configs.COMPILER_CONFIG,
            doe_config_path=out,
        )


def test_search_input_params_match_doe_variables(doe_config: Path) -> None:
    """The reverse lookup must report every parameter the sweep varies."""
    cfg = yaml.safe_load(doe_config.read_text())
    expected = [variable["path"] for variable in cfg["variables"]]
    assert search.load_input_params(doe_config) == expected
    assert len(expected) > 7, "expected the full sweep, not the legacy hardcoded subset"


def test_search_input_params_fall_back_when_config_missing(tmp_path: Path) -> None:
    assert search.load_input_params(tmp_path / "absent.yaml") == search.FALLBACK_INPUT_PARAMS


def test_four_post_metrics_resolve_to_generated_results() -> None:
    """FourPostEval writes to generated_results/; the DOE must look there."""
    primary = generator.FOUR_POST_METRICS_CANDIDATES[0]
    assert primary.parts[-3:-1] == ("_3_StandardSim", "generated_results")

    report_cfg = yaml.safe_load(
        (ROOT / "_3_StandardSim/FourPostEval/four_post_eval_config.yml").read_text()
    )["report"]
    assert Path(report_cfg["metrics_csv_path"]) == primary.relative_to(ROOT)


def test_missing_four_post_metrics_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        generator._load_metrics_csv(tmp_path / "absent.csv")
    assert "standard-eval-four-post" in str(excinfo.value)


@pytest.mark.parametrize("samples", [3])
def test_small_doe_generates_variants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, samples: int
) -> None:
    """End-to-end plumbing: config -> sample the record -> write variant .mo."""
    monkeypatch.setenv("BOBSIM_DOE_METHOD", "lhs")
    monkeypatch.setenv("BOBSIM_DOE_SAMPLES", str(samples))

    doe_config_path = tmp_path / "_doe_config.yaml"
    generate_configs.refresh_doe_config(
        architecture_config_path=ARCHITECTURE_CONFIG,
        compiler_config_path=generate_configs.COMPILER_CONFIG,
        doe_config_path=doe_config_path,
    )

    _localize(doe_config_path)
    if not _boblib_record_available(doe_config_path):
        pytest.skip("BobLib submodule is not checked out; run 'make init'")

    from StandardSens.pipeline.sampler import sample

    variants = sample(doe_config_path)
    assert len(variants) == samples + 1, "LHS returns the baseline plus N samples"

    # Static balance free length needs FourPostEval motion ratios. Stub them so
    # this stays a plumbing test rather than a simulation test.
    metrics_csv = tmp_path / "four_post_eval_report_metrics.csv"
    metrics_csv.write_text(
        "metric,value\n"
        "static_motion_ratio_front,0.62\n"
        "static_motion_ratio_rear,0.71\n"
    )
    monkeypatch.setattr(
        generator, "FOUR_POST_METRICS_CANDIDATES", (metrics_csv,), raising=True
    )

    population = tmp_path / "population"
    generator.generate_variants(doe_config_path, variants, population)

    written = sorted(population.rglob("*.mo"))
    assert len(written) == len(variants)

    for path in written:
        text = path.read_text()
        assert text.lstrip().startswith("within "), f"{path} is not a Modelica record"
        assert "springFreeLength" in text
        # A failed substitution leaves the placeholder or an empty binding.
        assert "{" + "}" not in text

    # The sweep must actually perturb the record away from baseline.
    baseline_text = written[0].read_text()
    assert any(path.read_text() != baseline_text for path in written[1:])


def test_build_template_renders(tmp_path: Path) -> None:
    """The .mos template is filled with str.format.

    Literal Modelica braces such as the MSL version list `{"4.1.0"}` must be
    escaped as `{{...}}` or format() reads them as replacement fields.
    """
    from StandardSens.pipeline import compiler

    rendered = compiler.generate_mos(
        variant_mo=tmp_path / "variant.mo",
        build_dir=tmp_path / "build",
        boblib_path=tmp_path / "BobLib" / "package.mo",
        standard_cfg={"model": "BobLib.Experiments.Standards.VehicleSim"},
        build_options={
            "start_time": 0,
            "stop_time": 1,
            "intervals": 0,
            "tolerance": 1e-6,
            "solver": "dassl",
        },
    )

    # Escaped braces must survive as real Modelica array literals.
    assert 'loadModel(Modelica, {"4.1.0"});' in rendered
    assert 'loadModel(VehicleInterfaces, {"2.0.2"});' in rendered
    assert "BobLib.Experiments.Standards.VehicleSim" in rendered
    assert "{" + "boblib_path" + "}" not in rendered


def test_optsim_entrypoints_import() -> None:
    """Catch import-time breakage in the runners the make targets invoke."""
    import importlib

    for module in (
        "StandardSens.pre_screen_sensitivities",
        "StandardSens.refined_response_surfaces",
        "StandardSens.pipeline.aggregator",
        "StandardSens.pipeline.search",
    ):
        importlib.import_module(module)


