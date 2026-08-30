# QSS and transient lap-time simulation

BobSim uses one track geometry, one optimized racing line, and one selected
`dyn_py` fidelity for two related calculations:

1. QSS minimizes lap time subject to the selected model's speed-dependent GGV
   envelope.
2. The transient scenario integrates the same 3/6/10/14DOF equations forward
   while following the QSS line and speed reference.

This makes the QSS result the idealized performance bound and the transient
result the executable check. Their delta includes controller tracking, dynamic
state buildup, and effects that a pointwise equilibrium envelope cannot show.

## Run it

```bash
make lap-eval-qss
make lap-eval-transient
make lap-eval
make lap-eval-all-dof
make lap-validation-visuals
```

Use another config with `LAP_CONFIG=path/to/config.yml`. The default is
`_3_StandardSim/LapTimeEval/lap_time_eval_config.yml`; it chooses a single
top-level `model_dof` for both GGV generation and transient integration. Change
that value to 3, 6, 10, or 14 to walk the same fidelity ladder described in
[reduced-order-dynamics.md](reduced-order-dynamics.md).

If the configured GGV CSV does not exist, LapTimeEval generates it from the
selected reduced model. Racing-envelope trims reject high-sideslip and
countersteer equilibrium roots using the configured `max_abs_beta_rad` and
`max_abs_steering_rad`; YMD remains the appropriate workflow for deliberately
prescribed high-beta states. An existing CSV is treated as a supplied
performance map, so its provenance must match `model_dof` when comparing
fidelities. GGV paths may contain `{model_dof}`; the default uses separate CSVs
so a map produced by one fidelity cannot be silently reused by another.

Generated maps have a sidecar provenance record containing vehicle, physics,
fidelity, power-limit, resolution, trim-bound, tire-domain, and multistart
fingerprints. A cache mismatch regenerates the GGV. A map explicitly supplied
with `generate_if_missing: false` remains usable but is labeled
`supplied_unverified` in `summary.json`.

The default full endurance config applies a 32 kW constant event cap to both
the QSS envelope and forward transient controller. The 80 kW VCU/hardware limit
stays in `vehicle.yml` for acceleration, autocross, skidpad, and uncapped
vehicle characterization. The 32 kW case is an energy-budget proxy only; it
does not model state of charge, lap-to-lap energy allocation, or thermal
derating.

`make lap-validation-visuals` runs a compact, resumable 3/6/10/14DOF acceptance
matrix and writes disposable figures and CSVs under
`temp/lap_time_validation/<dof>dof/`. Each folder contains:

- `envelopes/`: 2D/3D GGV, capability metrics, and three YMD views;
- `qss/`: corridor/racing-line geometry and QSS speed/acceleration profiles;
- `transient/`: path tracking, QSS/transient velocity comparison, yaw/steer/
  acceleration histories, plus body attitude, wheel speed, and unsprung motion
  when those states exist; and
- raw GGV, YMD, QSS lap, transient lap, and summary data.

Once two or more fidelities have completed, `temp/lap_time_validation/overlays/`
also contains shared-axis GGV, QSS speed, transient speed, transient yaw-rate,
and lap-time comparisons. These are the first place to inspect whether an added
state changes a system-level result.

The bundle is ignored by Git. Re-running the target resumes completed
fidelities; use `python -m _3_StandardSim.LapTimeEval.validation_visuals
--force-laps` to refresh laps from cached envelopes, or `--force` to regenerate
everything.

## Track and racing line

Track input is a closed sequence of paired gates:

```csv
left_x_m,left_y_m,right_x_m,right_y_m
...
```

The legal vehicle-center interval at each gate subtracts half the vehicle width
and the safety margin from both sides. Periodic cubic splines turn the gate
offsets into an arc-length-sampled line with heading, curvature, and segment
length.

The checked-in default is
`_3_StandardSim/LapTimeEval/tracks/endurance_michigan_2019.csv`: the 2019
Formula SAE Michigan endurance boundaries retained by Longhorn Racing Electric's
historical `jomama_lapsim`. BobSim converts the original feet to meters and
records the exact provenance in `tracks/README.md`.

The all-DOF acceptance matrix continues to drive the deterministic synthetic
694 m `endurance_reference.csv`, because repeated 10/14DOF GGV and transient
runs on a full 2 km course are too slow for a routine check. Both track outlines
are rendered under `temp/lap_time_validation/reference_tracks/`, so the real
course is always available as a system-level reference rather than being
mistaken for the compact regression case.

Three line modes are available:

- `centerline` uses zero gate offset.
- `minimum_curvature` minimizes integrated squared curvature and is useful as a
  fast, vehicle-independent seed.
- `minimum_time_qss` starts from the minimum-curvature line and minimizes the
  actual propagated QSS lap time. This is the default.

Integrated absolute curvature is deliberately not the final objective. It is
nearly fixed at one revolution for many simple closed tracks and does not price
corner radius, acceleration zones, braking zones, or speed-dependent grip.

## QSS calculation

At every path sample, curvature and the GGV lateral boundary set a local speed
ceiling. Alternating closed-loop forward and backward passes then enforce the
available combined acceleration and braking at each speed and lateral load.
The final segment time uses the two endpoint speeds.

The output `qss_lap.csv` contains station, path geometry, speed, longitudinal
and lateral acceleration, and segment time. It is an equilibrium envelope
calculation: it does not include actuator lag, tire relaxation, or the time
needed for roll, pitch, wheel-speed, and wheel-hop states to settle.

## Forward transient calculation

The transient begins from a steady-state trim at the first QSS path point. If
that interpolated GGV point lies numerically on the feasibility edge, the
initializer searches inward in speed for the nearest valid equilibrium. A
feedforward/feedback driver commands:

- a sparse steady-state roadwheel-steer profile, interpolated along the path,
  plus heading and cross-track correction; and
- QSS longitudinal-acceleration feedforward plus speed-error correction.

Wheel torque is distributed using the vehicle drive and brake fractions. The
same suspension instant links, wheel rates, bar rates, tire forces, body states,
wheel speeds, and unsprung states used elsewhere in `dyn_py` are therefore
active according to the selected fidelity.

`transient_lap.csv` records time, wrapped station, unwrapped progress, tracking
errors, target/actual speed, and yaw rate. `summary.json` reports QSS time,
transient time, their delta, completion, and maximum lateral error.

The summary also records the active model space and validity limits. This is a
single-configuration simulation, so its `swept_parameters` list is empty. DOE
workflows must separately identify the one parameter changed in each variant;
the EnvelopeSens interval-splice table now does that explicitly and writes an
active/swept-space study manifest.

## Interpreting validation

First require the transient run to complete the lap without leaving the legal
corridor. Then compare QSS and transient speed, acceleration, yaw rate, and
attitude histories. Finally replay the same maneuver in BobLib and use the
existing reduced-order signal correlation tools. A smaller QSS/transient delta
does not by itself prove fidelity: agreement with BobLib and measured vehicle
data remains the reference.

The current driver follows a fixed QSS reference; it is not yet a closed-loop
optimal-control solver. Future transient line optimization can replace the
driver/objective without changing the shared track or vehicle equations.

## Verification boundary

The automated tests exercise equation evaluation, constant-radius QSS, a GGV
boundary point, a YMD point, and equilibrium-start transient integration for
all four fidelities. The validation bundle additionally requires every model
to converge its QSS lap, complete the transient lap, and remain inside the
track corridor. Non-converged YMD cells are masked in figures rather than
drawn as valid results.

This establishes internal functionality, not external truth. BobLib and test-
vehicle correlation remain a separate validation phase.

Simulation-only outputs support design trends and raw event-time sensitivities,
not Formula SAE competition points. BobSim does not contain a QSS-to-points
conversion. Such a claim requires controlled correlation, uncertainty, matching
competition telemetry, and demonstrated response-space coverage; absent that
evidence, `summary.json` declares `competition_points_supported: false`.
