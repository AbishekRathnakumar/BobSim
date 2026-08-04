# Architecture

BobSim is the **workflow layer**. BobLib (a git submodule) is the **physics
layer**. BobSim does not contain vehicle physics; it configures BobLib Modelica
models, builds them with OpenModelica, runs them, and turns the raw output into
metrics, plots, and reports.

```
vehicle.yml ──► _5_App / _0_Utils ──► BobLib Modelica records ──► omc build
                                                                    │
                                    ┌───────────────────────────────┘
                                    ▼
                     _3_StandardSim  (single studies)
                     _2_EnvelopeSim  (GGV / YMD maps)
                     _4_OptSim       (sweeps, sensitivities, DOE)
                                    │
                                    ▼
                     _0_Utils/plotting + reporting ──► CSV metrics, PDF reports
```

## Layers

### `_0_Utils/` — shared foundation
- `vehicle_io.py`: canonical loader/validator for `vehicle.yml` and tire
  templates. `repo_root()` and `vehicle_yaml_path()` are the path helpers other
  layers should use instead of hand-rolling `parents[n]` chains.
- `plotting/`, `reporting/`: the plot engine and report engine that every study
  renders through. New study output should reuse these, not matplotlib directly.
- `vehicle_templates/`, `tire_templates/`: checked-in architecture and tire
  starting points (`.yml`, `.tir`).
- `deploy/`: PyInstaller packaging for the desktop build.
- `external/BobLib/`: **the submodule**. See [boblib-submodule.md](boblib-submodule.md).

### `_1_VisualSim/` — visualization
The rendering engine (`viewer.py`, `run_visual.py`) plus visual templates.
Core model visualization during development still mostly happens in OMEdit;
this layer is for offline/replay visuals and is consumed by `_5_App`.

### `_2_EnvelopeSim/` — performance envelopes
`GGV/` (grip-acceleration envelope) and `YMD/` (yaw moment diagram). These are
quasi-static map generators driven by `*_config.yml` files. `vehicle_loader.py`
and `vehicle_yaml.py` adapt `vehicle.yml` into envelope inputs.

### `_3_StandardSim/` — standard vehicle studies
Four studies, each a directory with a `*_config.yml` and a `*_sim.py`:
`RampSteerEval`, `SteadyStateEval`, `TransientEval`, `FourPostEval`.

Two shared runners sit alongside them:
- `_modelica_runner.py`: drives a compiled OpenModelica executable.
- `_fmu_runner.py`: drives an exported FMU.

Builds are produced by the `.mos` scripts (`build_vehicle_sim.mos`,
`build_four_post_sim.mos`) into `_3_StandardSim/BuildBobLib/`. Reports and metric
CSVs land in `_3_StandardSim/generated_results/`.

### `_4_OptSim/` — sensitivities, response surfaces, DOE
The parameter-sweep layer. `StandardSens/` sweeps StandardSim studies;
`EnvelopeSens/` sweeps envelope outputs. `_shared/` holds the console progress
helpers and the tornado-plot renderer used by both.

This is the layer that supports going *backwards* from target performance to
vehicle parameters — see [doe-reverse-engineering.md](doe-reverse-engineering.md).

### `_5_App/` — local browser app
The primary user entry point (`python -m _5_App.app`, port 8765). It is a
standard-library HTTP shell over everything above: pick/edit a vehicle, write
generated Modelica into BobLib, launch jobs, watch logs, browse results.

Module responsibilities are documented in [`_5_App/README.md`](../_5_App/README.md).
The important boundary: `contracts.py` + `registry.py` declare *what* workflows
exist, `actions.py` dispatches them, `data_services.py` owns the data layer.
Adding a workflow means registering it, not editing the server.

Mutable app state lives under `_5_App/user_data/` (dev) or the user's BobSim
runtime directory (packaged). It is gitignored.

## The vehicle definition

`vehicle.yml` at the repo root is the active vehicle. It carries
`schema: boblib.vehicle.v1` and describes masses, CGs, inertias, suspension
architecture, and paths to BobLib and the tire templates.

Two consumers, and the distinction matters:

1. **BobSim projection, reporting, and sensitivity workflows** read `vehicle.yml`
   directly.
2. **Modelica standard entry points** do *not*. They use checked-in BobLib
   records (`BobLib/Records/VehicleDefn/*.mo`). `_5_App/modelica_generator.py`
   is what writes `vehicle.yml` out into those records.

So `vehicle.yml` and the BobLib record can drift. Keeping them in sync is a
deliberate step (the app's save/generate flow, or the `sync-vehicle` target,
which today just prints the reminder). If a study's numbers disagree with
`vehicle.yml`, suspect an unregenerated record first.
