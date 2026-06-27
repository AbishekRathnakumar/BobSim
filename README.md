# BobSim

BobSim is the BobDyn high-fidelity vehicle analysis workspace. It wraps the
BobLib Modelica vehicle models in a local browser app for configuring vehicles,
writing BobLib-backed Modelica definitions, running standard studies, exploring
results, and keeping generated artifacts organized. The underlying Python and
Modelica workflows are still available as Make targets for automation and
focused development work.

The full documentation lives at:

https://bobdyn.com

Use the BobDyn documentation site as the detailed source of truth. This README is
the quick release guide for getting a clean checkout running and checking that it
is healthy.

## Repository Layout

- `_0_Utils/`: shared Python utilities, plotting/reporting helpers, and the
  BobLib submodule.
- `_1_VisualSim/`: experimental/offline visualization tooling; core model
  visualization currently happens in OMEdit.
- `_2_EnvelopeSim/`: GGV/YMD performance-envelope workflows.
- `_3_StandardSim/`: standard vehicle studies: RampSteerEval, SteadyStateEval,
  TransientEval, and FourPostEval.
- `_4_OptSim/`: sensitivity and response-surface workflows.
- `_5_App/`: local browser app for configuring setups, launching workflows, and
  inspecting generated reports, metrics, configs, and job logs.
- `tests/`: release-polish and workflow regression checks.
- `vehicle.yml`: active vehicle data used by BobSim projection, reporting, and
  sensitivity workflows. The Modelica standard entry points now use checked-in
  `BobLib` records.

## App Quick Start

Initialize the BobLib submodule:

```bash
make init
```

Build the Docker development image:

```bash
make docker-build
```

Show the available targets:

```bash
make help
```

The Docker image is based on OpenModelica and installs the Python dependencies
from `requirements.txt`.

Start the BobSim app:

```bash
make app
```

Then open `http://127.0.0.1:8765`.

The app is the primary entry point for normal vehicle-development work. Use it
to select or edit a vehicle configuration, save the setup, write the generated
Modelica records into BobLib, configure simulation runs, review logs, and browse
saved results. Generated builds, workspaces, and result archives are local
runtime content and are intentionally ignored by git.

The command-line targets below mirror the same workflow pieces for CI,
automation, and focused debugging.

## Release Checks

Run the local fast CI gate:

```bash
make ci
```

This runs:

- `make lint`
- `make typecheck`
- `make test`

GitHub Actions runs the same fast gate inside the BobSim Docker image with the
BobLib submodule checked out recursively. Full StandardSim baseline simulations
are intentionally left out of normal GitHub CI because they can exceed hosted
runner limits.

Run the full StandardSim baseline explicitly when you need to refresh or review
simulation artifacts:

```bash
make regression-baseline
```

## Target Language

BobSim's make targets use a small, intentional vocabulary:

- `docker-*`: build or rebuild the development image.
- `shell-*`: open an interactive shell in a workflow context.
- `standard-*`: build and run standard vehicle evaluations.
- `envelope-*`: generate performance-envelope outputs.
- `opt-*`: run sensitivity and response-surface workflows.
- `clean-*`: remove generated artifacts.

## Standard Simulation Entrypoints

Most standard simulation work should be launched from the app so the vehicle
configuration, generated Modelica, run logs, and saved results stay keyed
together. The Make targets remain useful for direct smoke tests and batch
development.

Run the complete standard baseline:

```bash
make standard-eval-all
```

That target builds missing executables, then runs RampSteerEval,
SteadyStateEval, TransientEval, and FourPostEval.

For focused standard work, build and run individual studies:

```bash
make standard-build
make standard-eval-ramp-steer
make standard-eval-steady-state
make standard-eval-transient

make standard-build-four-post
make standard-eval-four-post
```

Build-only targets are also available:

```bash
make standard-build
make standard-build-four-post
```

Reports and metric CSVs are written under `_3_StandardSim/results/`.

## Cleanup

Remove local caches and user-generated content such as simulation results, app
workspaces, saved app results, and cached Modelica builds:

```bash
make clean
```

Use more specific cleanup targets when you only want to clear part of the
workspace:

```bash
make clean-caches
make clean-standard
make clean-envelope
make clean-opt
make clean-app
```
