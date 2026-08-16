# Reduced-order vehicle dynamics

BobSim has one reduced-order equation set for transient integration and
quasi-steady envelopes. It is intentionally lower fidelity than BobLib: the
point is to make assumptions inspectable, run quickly, and quantify where each
added degree of freedom improves agreement with the Modelica reference.

The implementation is in `_0_Utils/dyn_py/`, alongside the existing
`_0_Utils/kin_py/` convention. EnvelopeSim and
StandardSim both import it; it does not import from either higher layer.

## Fidelity ladder

"DOF" counts generalized coordinates. A second-order mechanical model has two
first-order states per coordinate.

| Model | Generalized coordinates | First-order states | Added physics |
| --- | --- | ---: | --- |
| 3DOF | global x, global y, yaw | 6 | planar motion, algebraic pitch/roll load transfer, fixed drive/brake distribution |
| 6DOF | body x/y/z and roll/pitch/yaw | 12 | heave, roll, pitch, instant-link suspension load transfer, aero pitch |
| 10DOF | 6DOF body + four wheel angles | 20 | individual wheel speed/slip and torque balance |
| 14DOF | 10DOF + four unsprung vertical positions | 28 | tire vertical compliance, wheel hop, and road-height inputs |

Each row is nested in the next. The 3DOF planar equations are therefore the
planar projection of the 6DOF model, and the 10/14DOF models do not carry a
second copy of the body or tire equations.

## Equations and conventions

Axes are x forward, y left, z up. Body translational dynamics retain the
rotating-frame term:

```text
v_dot_body = sum(F_body) / m - omega x v_body
```

Rigid-body angular acceleration uses the projected full inertia tensor:

```text
omega_dot = I^-1 (sum(M_body) - omega x (I omega))
```

Contact-patch velocity is `v_cg + omega x r_corner`. Tire slip is evaluated in
each steered wheel frame. The TIR projection preserves load-dependent peak
friction plus longitudinal and lateral stiffness; a smooth saturation and
combined-slip ellipse close the reduced tire model.

The 6/10DOF normal loads come from sprung heave/roll/pitch, wheel rates,
damping, anti-roll stiffness, static preload, and geometric force transmission
through the double-wishbone instantaneous links. The 14DOF model separates
suspension force from tire vertical force and integrates each unsprung mass.
The 3DOF model closes pitch and roll moments algebraically at every force
evaluation: longitudinal transfer follows contact-patch height and wheelbase,
while lateral transfer follows the front/rear elastic roll-stiffness split.
Positive wheel torque follows `drive_distribution_front` (zero is RWD); brake
torque follows the configured front brake fraction.

Drive capability is projected from the active `vehicle.yml`, not a generic
power constant. The instantaneous GGV and transient controller use the minimum
of motor/VCU peak torque through the final drive, the VCU's FSAE motoring-power
limit, motor/inverter peak power, and the motor-rpm vehicle-speed ceiling. The
80 kW competition limit is therefore distinct from the 124 kW hardware rating.
The projected continuous torque and
power limits are retained separately for a future thermal/endurance derating
model; they do not replace the instantaneous peak GGV boundary.

The 10/14DOF QSS constraints retain wheel inertia. A prescribed vehicle
acceleration therefore gives each wheel the corresponding rolling angular
acceleration rather than incorrectly imposing zero wheel acceleration. In the
14DOF equations, longitudinal and lateral translation use total vehicle mass;
sprung mass is used only for the independently released chassis heave equation.

Aerodynamic downforce is applied at the center of pressure inferred directly
from the nominal `vehicle.yml` downforce and free-pitch-moment maps. The rigid
body receives the equivalent CG wrench
`F_aero, r_CG_to_CoP x F_downforce`; drag remains applied at `aero_ref_m`.
The reported front aero balance is derived from that CoP for the planar axle
load closure. The CoP is not clipped to the wheelbase, because doing so would
change the map's pitch moment.

`vehicle.yml` supplies geometry, component mass/inertia, wheel/tire values,
suspension tables, aero maps, and powertrain layout. When available, the
FourPost metrics CSV supplies measured/projected wheel and anti-roll stiffness;
the YAML shock tables are the fallback.

## Double-wishbone force transmission

The dynamic system receives suspension hardpoints, not hand-entered instant
centers or jacking coefficients. `_0_Utils/dyn_py/suspension.py` solves the
upper-arm, lower-arm, upright, and tie-rod constraints across wheel travel. At
the current corner jounce it constructs reciprocal instantaneous links from the
contact-patch tangent:

```text
Fz_geometric / Fx = -dx_contact / dz_contact
Fz_geometric / Fy = -dy_contact / dz_contact
```

This is equivalent to the side-view and front-view swing-arm line of action,
but remains valid when the 3D wishbone pivot axes are swept and a simple 2D
hardpoint projection is not. Left/right lateral coefficients are mirrored;
longitudinal coefficients retain their axle sign.

Spring/damper and stabilizer-bar forces remain a separate elastic path. Spring
and damper tables are projected through the BobLib/FourPost motion ratio to an
equivalent wheel rate. Bar torsion is projected to axle roll stiffness and
applied as equal-and-opposite corner force. In 6/10DOF the massless-upright
closure is

```text
Fz_tire = Fz_spring/damper/bar + Fz_geometric
```

In 14DOF the geometric force acts upward on the sprung body and with equal and
opposite sign on the explicit unsprung mass.

Repeat the hardpoint-force correlation against a fresh BobLib FourPost report:

```bash
make standard-eval-four-post
make reduced-suspension-correlation
```

For the default vehicle used during implementation, the nominal longitudinal
jacking coefficients differed from BobLib by 2.23% at the front axle and 1.55%
at the rear. Symmetry predicts zero net axle heave from equal left/right lateral
force; BobLib returned residual coefficients below `8.4e-5`.

## QSS is a constraint on the transient model

QSS does not use a parallel force implementation. It evaluates the transient
equations and constrains selected generalized accelerations:

- constant-radius trim: all generalized accelerations are zero in the body
  frame; centripetal acceleration remains through `omega x v`;
- prescribed GGV point: body `ax/ay` are prescribed at zero yaw rate while yaw,
  vertical, rotational, wheel-speed, and unsprung accelerations are
  equilibrated. Curvature and steering-radius feasibility belong to the track
  solver, not the acceleration-capability envelope. The acceleration branch is
  closed at a separately solved sustainable-corner endpoint at zero body
  longitudinal acceleration, with driven-wheel force balancing aero drag. The
  brake branch retains its independently solved feasible sampled domain. This
  preserves valid drive-supported lateral states without inventing brake
  capability there, and refines the track-speed boundary beyond the sampled
  grid. Track-only studies may enable
  `track_relevant_lateral_domain_only` to omit brake-only rows above that
  sustainable boundary because their speed profiles cannot query them;
- YMD point: sideslip and steer are imposed, longitudinal/vertical/wheel states
  are equilibrated, and lateral acceleration plus yaw moment are outputs.

The unknown set grows naturally with fidelity: body sideslip/steer/drive torque,
then heave-roll-pitch, then four wheel slips, then four unsprung positions.

Select the backend in `GGV/ggv_config.yml` or `YMD/ymd_config.yml`:

```yaml
generation:
  model_dof: 6  # 3, 6, 10, or 14
```

## Transient runs and BobLib comparison

Run a reduced-order step steer from a solved straight-line trim:

```bash
make reduced-eval REDUCED_DOF=6
```

Compare common time histories with an existing OpenModelica result CSV:

```bash
make reduced-eval \
  REDUCED_DOF=10 \
  REDUCED_BOBLIB_CSV=_3_StandardSim/BuildBobLib/VehicleSim/results/run_.../BobLib.Experiments.Standards.VehicleSim_res.csv
```

The comparison reports RMSE, range-normalized RMSE, maximum absolute error, and
bias for common channels such as `velX`, `velY`, `yawVel`, `sideslip`, `accX`,
`accY`, and `roll`. Steering inputs must represent the same roadwheel motion;
do not compare handwheel and roadwheel degrees without applying the steering
ratio.

For a real validation pass, run `standard-eval-transient`, retain its result
CSV, then compare all four reduced fidelities against the same case. Use the
error movement between adjacent fidelities to identify whether disagreement is
caused by body attitude, wheel rotation, unsprung motion, or physics that still
belongs only to BobLib.

The repeatable fidelity-discrimination suite makes that comparison directly:

```bash
make reduced-fidelity-suite
make reduced-fidelity-suite REDUCED_MBD_DIR=path/to/boblib_case_csvs
```

It writes common-axis 3/6/10/14DOF overlays under
`temp/fidelity_validation/`. When `REDUCED_MBD_DIR` is supplied, the directory
may contain `step_steer.csv`, `slalom.csv`, `brake_in_turn.csv`, and
`four_wheel_bump.csv`. Each available BobLib CSV is drawn as `MBD (BobLib)`, a
reduced-minus-MBD residual page is added, and RMSE/normalized RMSE/max error/
bias are written to JSON.

The cases are intentionally not redundant:

| Case | Difference it should expose | Lowest useful model |
| --- | --- | --- |
| Step steer | yaw/ay response and sprung roll buildup | 3DOF for planar response; 6DOF for roll |
| 1.2 Hz slalom | gain, phase, and transient load transfer | 6DOF |
| Brake in turn | combined slip, pitch transfer, wheel slip dynamics | 10DOF when wheel-speed transients matter |
| Four-wheel bump | tire vertical compliance, wheel hop, road input | 14DOF |

BobLib is the MBD implementation reference, not automatically physical truth.
Agreement with measured vehicle data remains the final validation layer. This
distinction is kept explicit in the artifact names and metrics.

Before BobLib correlation, run the internal all-fidelity acceptance bundle:

```bash
make lap-validation-visuals
```

This exercises model-specific GGV and YMD calculations and a complete QSS plus
transient lap for 3/6/10/14DOF. Inspect the generated figures under
`temp/lap_time_validation/`; this folder is disposable and gitignored.

## Current reduction limits

- Instant-link slopes vary with corner jounce, but camber/toe tire-force
  effects, compliance, and the detailed individual link loads remain BobLib
  validation targets.
- The tire is a compact MF-derived saturation model, not the full MF52/MF6.2
  equation set, and has no relaxation-length states.
- The powertrain is represented by wheel torque in the transient equations;
  the existing GGV power cap remains an outer feasibility constraint.
- Aero uses the nominal ride-height map projection rather than reevaluating the
  full map during body motion.
- The 14DOF road interface currently supports vertical road height/speed only.

The early Longhorn Racing Electric transient prototypes inspired the model
ladder and state-count convention, but BobSim's implementation is original.
The linked repository has no license file, and its incomplete source was not
copied.
