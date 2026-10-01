# ISO 8608 wheelrate / CPLV study

Run in Docker from BobSim: `docker compose run --rm -T bobsim make quarter-car-psd`.
Verify: `docker compose run --rm -T bobsim make quarter-car-psd-test`.
The equivalent host `make` targets also use Docker. Override `QCM_CONFIG` for another configuration.

Default: **2027.yml, 15 m/s, 5–50 kN/m in 0.25 kN/m steps, ISO A–H**.
The included `2027.yml` is the provisional input used for this study; it is not an
approved final vehicle specification. Select another BobSim vehicle in the config
to run another configuration.

## Definitions and model

Two vertical DOFs per isolated corner: sprung and unsprung displacement. Suspension
spring and viscous damper connect the masses; tire spring and damper connect unsprung
mass to road. This is a Python frequency-domain QCM in BobSim, not the kinematic
`QuarterCar` assembly and not an executed BobLib Modelica full-car simulation.

At every rate, wheel damping is recomputed as `c = 2 * 0.70 * sqrt(k_w * m_s)`.
This is the conventional nominal sprung-mass damping ratio. Finite tire compliance
and unsprung inertia mean the coupled ride-mode pole damping is not exactly 0.70.
Wheelrate and damping are at the wheel: no spring/damper motion-ratio conversion.

`CPLV (%) = 100 * RMS(Fz - mean(Fz)) / mean(Fz)`; raw RMS in N is also saved.
Mean load is the static gravity corner load, without downforce. Use BobSim's shared
mass ledger and longitudinal CG projection; subtract explicit unsprung corner mass
from the corner-supported mass. Small link masses follow the existing reduced-model
classification. The YAML's FrameTorsionEval override block is not used.

With `q = k_w + i*omega*c`, `t = k_t + i*omega*c_t`, solve
`[[q-ms*omega^2, -q], [-q, q+t-mu*omega^2]] [Zs, Zu] = [0, t*Zr]`.
Dynamic contact load is `t*(Zr-Zu)`, including tire viscous force.

ISO 8608:2016 Table C.2 class representatives use `G(n)=G0*(n/0.1)^-2`, with
G0 = 16, 64, 256, 1024, 4096, 16384, 65536, 262144 times 1e-6 m^3 for A–H.
These are representative spectra, not the entire range within a class. A and H have
open bounds; their tabulated representatives are simulation values. Reference:
<https://www.iso.org/standard/71202.html> and ISO 8608:2016 Annex C Table C.2.

The finite spatial band **0.01–10 cycles/m** is a study assumption, not a mandatory
ISO bandwidth. At 15 m/s this is 0.15–150 Hz. Convert with `f=v*n`, `S(f)=G(f/v)/v`;
integrate `|H(v*n)|^2 G(n) dn` using Simpson quadrature on a logarithmic grid.
No sampled road seed is needed for this ensemble PSD response. Adjacent classes
have four times the input PSD, so this linear model has exactly twice the RMS/CPLV.

## Interpretation and checks

All A–H curves use a linear CPLV axis starting at zero. Dashed portions have
Gaussian `P(Fz <= 0) >= 1%` and are only bilateral linear extrapolations. They do not
simulate physical tensile tire force or contact reattachment. The 1% cutoff is a
screening convention; solid curves are not proof of physical validity. RMS travel
is included for later stroke/ride-height checks. No bump stops, tire enveloping,
travel limits, nonlinear damping, aero, roll, pitch, or lateral grip are modeled.
The separate grip workflow below adds a quasi-steady lateral-force calculation;
it does not add those missing vehicle/contact dynamics to the QCM.
Do not infer a full-car optimum or grip-loss percentage from CPLV alone.

Every sweep checks all poles and compares 4097/8193 frequency-point integrals,
requiring RMS relative error below 1e-4. Tests independently compare the closed-form
FRF with a state-space solve and compare the PSD normalization with a continuous
Lyapunov solution for white road velocity (tire damping disabled for that check).

Outputs under `generated_results/quarter_car_psd_2027/`: PNG, PDF, complete CSV,
input snapshots, source hashes, corner parameters, and convergence summary.

Plots include one overview, eight individual class figures (`iso_a` through `iso_h`),
and an A/B-only comparison (`iso_a_b`), each in PNG and PDF. Each figure uses a
linear zero-based CPLV axis scaled to its selected classes, with matching front/rear
limits within that figure. Class colors stay consistent between figures.
Wheelrate is displayed in lb/in (pounds-force per inch); the numerical model and
configuration retain SI units. The selected range is approximately 28.55–285.51 lb/in.
Each figure now uses front/rear rows, with CPLV on the left and mean contact load
on the right. Mean load is in lbf, with N annotated. It is the same for every ISO
class and wheelrate because this model uses fixed static equilibrium loads.
Anti geometry, longitudinal force paths and aero response are absent, so these
figures alone cannot quantify anti-to-grip effects. Both the mean load and the
fluctuations matter when applying a load-sensitive tire force model.

## 2027 run, 2026-09-30

The selected local input projects to 271.2482 kg, 45% front static weight. Per-corner
sprung masses are 53.3198 kg front and 66.8822 kg rear, with 7.7111 kg unsprung each.
Results below are ISO B at 15 m/s, with no aero load:

| Wheelrate (kN/m) | Front CPLV (%) | Rear CPLV (%) |
| --- | --- | --- |
| 5 | 20.886 | 17.045 |
| 10 | 21.721 | 17.997 |
| 20 | 24.098 | 20.216 |
| 50 | 30.446 | 25.775 |

A/B pass the chosen unloading screen across this grid. C passes through 8.75 kN/m
front and 26 kN/m rear; D–H fail throughout. These limits depend on the static mean
load, speed, band, and assumed Gaussian amplitudes. They are not physical contact
loss predictions from a nonlinear tire model.

## Finite-record lateral grip workflow

Run `docker compose run --rm -T bobsim make quarter-car-grip`.
Verify `docker compose run --rm -T bobsim make quarter-car-grip-test`.
Configure `grip_config.yml`, or override `QCM_GRIP_CONFIG`. It references the QCM
config above, so vehicle, speed, rate sweep, damping and ISO band stay consistent.

The metric is **maximum mean positive lateral force at one fixed slip angle per
road record**, with zero camber and zero longitudinal slip. For a record,
`G = max_alpha mean(Fy(alpha, Fz(t)))`; the same calculation at its constant mean
load gives `G_smooth`. Grip loss is `100*(1-G/G_smooth)`. The MF5.2 signed-force
equations are vectorized from BobLib's `MF52/PureSlip/FyPureEval.mo`, and directly
tested against the existing `_5_App/tire_eval.py` scalar implementation. It includes
the full pure-lateral slip curve, shifts and load sensitivity, not just `mu*Fz` at
each sample's individual peak. Positive Fy fixes the direction for any tire-model
asymmetry. There is no mu-floor substitution or constant-friction simplification.

Roads are periodic 64-second random-phase Fourier records with seeds 2027, 2028,
2029, sampled at 2400 Hz. Bin amplitudes match the exact integral of the specified
one-sided ISO PSD over each frequency bin. The same coefficients/phases are used
for both axles and every rate; classes scale the common ISO A road amplitude.
Transfer functions generate the steady periodic load response with no startup
transient. This represents three finite records, not an infinite Gaussian force
expectation or a measured track. Fixed-amplitude random-phase records have exact
specified bin energies; the seed range explores phases, not PSD-amplitude uncertainty.

The tire's fitted loads are strictly 100–1800 N. A row is supported only if the
ENTIRE band-limited record is bounded inside that interval. The sample min/max
is expanded by `sup|Fz''|*dt^2/8`. The curvature bound uses the largest sampled
absolute second derivative plus `sup|Fz''''|*dt^2/8`, with the fourth derivative
bounded by `2*sum(|c_k|*(2*pi*f_k)^4)`. Ambiguous records are interpolated at 2x,
4x and at most 8x sample rate using the SAME Fourier coefficients. Actual observed
fit departures are not refined away. The final sample rate and domain status are
saved per record.
This conservative interpolation bound includes extrema between samples. Rejection
can therefore mean either an observed departure or an inability to certify the
record within the fit; the CSV retains both sample extrema and bounds. No invalid
load is clipped, assigned zero grip, or silently dropped. The class/rate summary
is blank unless all three seed records pass; passing-seed results remain available
only as individual-record diagnostics. A blank result is not zero physical grip.

A 121-point full fitted slip-domain grid, evaluated using weighted means of 256
load bins, locates candidate local peak brackets. Every peak bracket is refined
using ALL original samples; final forces are exact finite-record arithmetic means
at the optimized angle. A separate fine-grid test verifies the optimizer and shows
why the mean of instantaneous peaks is an optimistic upper bound. Halving sample
rate must change the final mean force by less than 1e-4 relative. Every discrete
load RMS must match the analytical PSD result within 0.2%. Seed shading is the
minimum/maximum of three independently optimized record capacities, not a confidence
interval, and not an optimization with a different slip angle at every time sample.

Outputs in `generated_results/quarter_car_grip_2027/` include per-record CSV,
all-seed summary CSV, Fourier road coefficients, input/code snapshots and hashes,
and PNG/PDFs for each class, all classes, and A/B only. Capacity is in lbf and
wheelrate in lb/in; loss is percent. Unsupported figures state the missing result.

Absolute capacity uses the source TIR without track-friction calibration. This
quasi-steady calculation omits tire relaxation, temperature, pressure variation,
nonlinear contact, travel enforcement, camber motion, aero and anti geometry. It
is neither an attained lateral acceleration nor a whole-car optimum. Anti studies
require a longitudinal/lateral force-path and vehicle-attitude model to supply
the appropriate load, camber and slip histories before using this force metric.

### Finite-record results for this configuration

All three records pass the tire-load screen at the following rates. Loss is relative
to the smooth-road capacity at the same mean load, from the first to last supported
rate; these ranges are not universal tire-domain boundaries.

| Class / corner | Supported wheelrate (lb/in) | Grip loss (%) |
| --- | --- | --- |
| A / front | 28.55–285.51 | 0.1425–0.3029 |
| A / rear | 28.55–285.51 | 0.1191–0.2725 |
| B / front | 28.55–29.98 | 0.5703–0.5711 |
| B / rear | 28.55–128.48 | 0.4770–0.7069 |
| C–H / both | None | Withheld |

There are 1,455 supported records and 435 all-seed-supported summary rows.
The other 7,233 records have observed sampled load departures; none remains rejected
solely because the between-sample bound could not certify it. Maximum discrete versus
analytical RMS relative error is 1.185e-6. Thirteen focused tests, targeted Ruff and
mypy pass. The separate default Modelica regression remains at 15 passes and the
same two baseline mismatches documented in the original QCM validation.

All axes are linear. Capacity axes are zoomed separately for each corner; loss axes
share limits within a figure. To refresh plots without repeating the simulation:
`docker compose run --rm -T bobsim python -m _3_StandardSim.QuarterCarPSD.grip_sweep --config _3_StandardSim/QuarterCarPSD/grip_config.yml --replot`.
This checks config and vehicle hashes against the saved run and records the plotting
runner and CSV hashes separately in `plot_manifest.json`; it preserves the executed
simulation sources and their provenance in `summary.json` and `source_snapshot/`.
