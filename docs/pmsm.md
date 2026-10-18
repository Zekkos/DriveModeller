# Brushless machines (PMSM / BLDC)

There are two layers here, and which one you want depends on what you are
trying to find out.

`PMSMSpec` in `drivetrain/pmsm.py` is the machine itself: the dq model, with
MTPA current angles, the voltage ellipse, field weakening, both saturation
flags and the loss terms.  It knows about `i_d`, `i_q`, saliency and reluctance
torque.

`PMSMModel` in the same file is that machine wearing the brushed model's face: a
thin adapter that exposes `kt`, `kv`, `resistance`, `inductance`,
`rotor_inertia`, `bearing_damping`, `static_friction`, `max_current`,
`max_voltage` and `rot_vel` as read-only properties derived from the spec.  A
`PMSMModel` drops into `MotorGroup` and therefore into `DriveSystem`,
`simulate`, `sweep`, `fit`, `presets`, the GUI and the code export without any
of them knowing it is brushless.

```python
from drivetrain import presets, simulate

system = presets.build("bldc_pod")            # 2 x PMSMModel in a MotorGroup
result = simulate.simulate(system, 13.856406, duration=20.0, engine="rk4")
result.final("motor_rpm")                     # 8855.8
```

Everything below is written around `presets.MOTORS["bldc_24v"]`:
`R = 0.12 ohm`, `Ld = 1.5e-4 H`, `Lq = 1.8e-4 H`, `lambda = 1.8e-3 Wb`,
`p = 7` pole pairs, `J = 2e-5 kg*m^2`, `b = 1e-6 N*m/(rad/s)`,
`i_limit = 20 A` peak per machine, `v_limit = 24/sqrt(3) = 13.856 V` peak
phase, `dc_bus_voltage = 24 V`.

## The dq model, and the 1.5 that trips people up

Amplitude-invariant Park, which is the convention most FOC libraries use:

```
v_d = R i_d + Ld di_d/dt - w_e Lq i_q
v_q = R i_q + Lq di_q/dt + w_e (Ld i_d + lambda)
tau = 1.5 p (lambda i_q + (Ld - Lq) i_d i_q)
w_e = p w_mech                      copper loss = 1.5 R (i_d^2 + i_q^2)
```

So `torque_constant = 1.5 p lambda` and `back_emf_constant = p lambda`, and
their ratio is **1.5, by convention, not because the two constants disagree**.
For `bldc_24v` that is `kt = 0.0189` and `kv = 0.0126`.  If you fit a brushless
drive and `kt/kv` comes out 1.5, that is correct; if you were expecting 1, you
have mixed a line-to-line datasheet number with a phase-peak model.  This is
the first thing to check when a fitted PMSM looks self-inconsistent.

Copper loss carries the same 1.5: `PMSMModel.thermal_power(i)` returns
`1.5 i^2 R`, where the brushed model returns `i^2 R`.  At 10 A that is 18 W
against 12 W, and it is not a factor you can wish away by counting machines
differently — it is the three-phase sum in this convention.

## Limits, MTPA and field weakening

`current_pair(torque, rot_vel)` returns the minimum-current `(i_d, i_q)` that
makes a torque at a speed, plus the two saturation flags.  Below base speed it
is the MTPA angle:

```
i_d = (lambda - sqrt(lambda^2 + 8 (Lq - Ld)^2 i_m^2)) / (4 (Lq - Ld))
```

Above base speed the voltage ellipse binds and the answer is the *largest*
feasible negative `i_d` on the ellipse boundary, found by scanning down from
`min(i_limit, characteristic_current)` and bisecting back onto the boundary.
The floor is `max(-i_limit, -characteristic_current)`: the model refuses to
reverse the permanent-magnet field further than that, because demagnetisation is
not modelled.

`torque_limit(rot_vel)` is the capability envelope.  For the default salient
spec it starts at 0.3967 N*m, above `kt * i_limit = 0.378`, because reluctance
torque is already contributing on the MTPA angle.  The envelope is even in
speed, and it is **larger when braking than when driving** above base speed
(0.1776 motoring against 0.2691 regenerative at base speed, 0.0896 against
0.1364 at twice base) because the back-EMF and the resistive drop have opposite
signs.  Pass `regenerative=True` to ask for the braking envelope.

The field-weakening region is close to constant power: 160.48, 161.06 and
161.2 W at 1x, 2x and 4x base speed for a non-salient test machine.  It drifts
slightly *up* with speed because the resistive drop is a smaller share of the
rail as speed rises.

Two honest oddities worth knowing about:

- **Above the rail speed you cannot command zero torque and zero current at
  the same time.**  `current_pair(0.0, 3 * base_speed)` for the default spec
  returns `i_d = -8.04 A`, `i_q = 0`, with the terminal voltage pinned at the
  rail, because the back-EMF alone (`w_e lambda = 20.8 V`) is larger than the
  6.93 V DC link.  A real drive either weakens the field or its clamp absorbs
  the current.  This is the uncontrolled-regeneration hazard, and a
  feed-forward controller that assumes "zero torque demand means zero current"
  will be surprised by it.
- **A machine rated below its characteristic current has no usable
  hyperbola.**  For a test machine with `i_limit = 2 A` and
  `characteristic_current = lambda/Ld = 9 A`, the envelope is flat to 0.5x base
  and then falls to exactly zero from 1.7x base: it cannot demagnetise enough to
  stay inside the ellipse.

## The i_d = 0 collapse, and where it stops being true

`PMSMModel` presents the machine at `i_d = 0`.  That is a one-to-one mapping,
not an approximation: `kt = 1.5 p lambda`, `kv = p lambda`, `R`, `L = Lq`, `J`,
`b`.  `kt` and `kv` cannot drift out of step with the spec because they *are*
the spec.  `Lq` is the inductance that represents the machine, because `Lq` is
the axis carrying the torque current.

The collapse is exact while the machine is current-limited and the voltage
ellipse is slack.  It saturates at `v_limit / kv`, which is base speed exactly,
so **a drivetrain built this way cannot invent a field-weakening region** — it
will happily keep promising full torque up to base speed and then fall off the
rail.

What it gets wrong is subtler than base speed.  A real FOC drive has to spend
part of its voltage budget on the cross-coupling term `w_e Lq i_q` (the
decoupling feed-forward), so the current it can hold at speed is set by the
ellipse, not by the q-axis line.  That bites well below base speed:

| quantity | `bldc_24v` at 20 A |
| --- | --- |
| base speed `v_limit / (p lambda)` | 1099.7 rad/s |
| speed at which the collapse still holds full current | 909.2 rad/s |
| speed at which a dq drive can still hold full current | 447.8 rad/s |
| `collapse_margin()` | 0.508 |

The margin is governed by `Lq i / lambda`, not by saliency: `Lq = 1.8e-4` at
20 A against `lambda = 1.8e-3` is a ratio of 2.0, and that is where the 51%
comes from.  Set `Lq = 1e-6` (ratio 0.01) and the two edges coincide to within
5e-5.  A non-salient machine with a big `Lq i` product is just as optimistic as
a salient one.

These are exposed rather than hidden, because `MotorGroup` and `simulate` apply
`max_current` as a constant clamp with no speed-dependent hook, and making the
collapse itself speed-dependent would mean rewriting the whole pipeline for a
number that only matters near the top of the band.  So:

- `constant_torque_speed()` — the dq edge, closed form: with
  `a = (Lq I)^2 + lambda^2`, `b = 2 lambda R I`, `c = (R I)^2 - V^2`, it is
  `(sqrt(b^2 - 4ac) - b) / (2a) / p`.
- `view_constant_torque_speed()` — the edge the collapse implies,
  `(v_limit - R i_limit) / kv`.
- `collapse_margin()` — their relative gap.
- `current_ceiling(rot_vel)` — the current a dq drive can actually hold there.
- `above_base_speed()`, `voltage_needed()`, `equivalent_note()`.

`equivalent_note()` for `bldc_24v` says it out loud:

```
i_d=0 collapse of a 7-pole-pair PMSM: kt=0.0189, kv=0.0126 (kt/kv=1.5 by
convention), exact below base speed 1099.7 rad/s (10502 rpm), full current only
reachable to 447.8 rad/s rather than the 909.2 rad/s the collapse implies; no
field weakening
```

Use the collapse for system-level questions (does this gearbox match this
motor, how long to accelerate, what does the drive see) and `PMSMSpec` directly
whenever the answer depends on the current angle, the envelope above base speed
or the loss split.

## Voltage and current conventions

`v_limit` is the **peak fundamental phase voltage**, not the battery.  An
SVPWM inverter makes at most `V_bus / sqrt(3)`; plain sinusoidal PWM tops out at
`V_bus / 2`.  `phase_voltage_limit(dc_bus, scheme)` converts.

Feeding the battery number into the brushed pipeline is the single most common
way to get a brushless answer that is wrong by `sqrt(3)`.  For `bldc_pod`,
simulating at 24 V instead of the 13.856 V rail predicts a steady motor speed
of 1474.9 rad/s (14084 rpm) — which is above the machine's own base speed of
1099.7 rad/s, i.e. the model is happily predicting a region it says it cannot
reach.  At the correct rail the simulator and the closed-form steady speed agree
exactly: 927.377 rad/s = 8855.8 rpm, with the current pinned at the 40 A group
limit on the way up.

DC-link current is not phase current:

```
i_bus = 1.5 (R i + kv w) i / V_bus
```

3.92 A at 12 A and 300 rad/s on a 24 V bus, against 12 A of phase current.  It
is not proportional to torque, which is why fitting a brushless drive from a
rail ammeter is a different problem from fitting it from the phase current the
controller reports.  `PMSMModel.dc_link_current()` returns it (and `nan` when no
bus voltage is given), and the fitter warns you when the dataset looks like it
might be one.

## Fitting a brushless machine

The fitter keeps speaking the brushed vocabulary and re-points it at the
constants this system can actually set:

```bash
python3 -m drivetrain fit bench.csv --preset bldc_pod \
    --free "group.motor.kv,group.motor.resistance" \
    --known group.motor.rotor_inertia=2e-5 \
    --known group.motor.bearing_damping=2.5e-5
```

```
group.motor.kv            -> group.motor.spec.flux_linkage
group.motor.kt            -> group.motor.spec.flux_linkage   (same parameter)
group.motor.resistance    -> group.motor.spec.resistance
group.motor.inductance    -> group.motor.spec.inductance_q
group.motor.rotor_inertia -> group.motor.spec.rotor_inertia
...
```

`kt` and `kv` are one flux linkage here, so pinning both is allowed when they
imply the same `lambda` (`kt = 1.5 p lambda`, `kv = p lambda`) and rejected as a
contradiction when they do not:

```
'group.motor.kt' and another brushed name both set
'group.motor.spec.flux_linkage', which is a single parameter on a brushless
machine (kt = 1.5 p lambda, kv = p lambda), and they disagree: 0.0126 against
0.02.  Pin the flux linkage once - or leave it free and let the fit choose it.
```

Warm starts work through the same translation: the back-EMF line of the log
gives a `lambda` start as `kv/p`, the standstill-acceleration line gives one as
`kt/(1.5p)`, and when both are available the start is their geometric mean.

Measured recovery — truth `lambda = 2.1e-3`, `R = 0.09`, `b = 2.5e-5`, log
driven with a 2 V square wave on the 13.856 V rail, sampled at 50 Hz, noise
0.02 A and 5 rpm:

```
group.motor.spec.flux_linkage   0.00210064   (1-sigma 2.9e-05)
group.motor.spec.resistance     0.0900227    (1-sigma 0.0022)
motor_rpm residual rms          13.09        (noise 15)
```

The identifiability answer for the collapse is that six constants never enter
the equations at all, and the fitter says so instead of fitting noise:
`spec.inductance_d` (the collapse has one q-axis current),
`spec.core_loss_coefficient` and `spec.windage_coefficient` (the collapse's only
speed-dependent loss is `bearing_damping`), `spec.flux_temperature_coefficient`
and `spec.reference_temperature` (the collapse reads reference-temperature flux
only) and `spec.dc_bus_voltage` (the collapse drives on the phase rail; the bus
number only converts phase current to rail current, which these channels do not
measure).  Naming one explicitly still fits it, so the door is open if you have
data that can see it.  `spec.pole_pairs` *is* settable, but it only ever enters
as the product `p lambda`, so it is perfectly correlated with `flux_linkage` and
the fitter's indistinguishable-pair warning fires.

Two ways to waste a brushless log:

- **Clip the current.**  Driving 12 V on a 13.856 V rail with a 40 A group limit
  pins the current at 40.14 A for 88% of the log.  The fitter warns
  ("while it clips, back-EMF and winding resistance leave the equations, so
  those samples say nothing about kv or resistance") and the fit comes out
  garbage: `lambda = 0.0107`, `static_friction = 2.11`, normalised rms 421.
  Drive below the limit — 2 V here, a 33 A stall against a 40 A limit — and
  everything recovers.
- **Trust the current channel.**  In the good log above the current residual
  stayed at 4.0 A rms against 0.02 A of noise, because a 50 Hz-sampled square
  wave on a machine with `tau_e = 1.5 ms` leaves the switching edge unresolved.
  The speed channel carried the identification.  Sample current much faster, or
  expect the speed channel to do the work.

### A log in the rotor frame

A drive that runs FOC can log `i_d`, `i_q`, the applied `v_d`/`v_q` and the
electrical rotor angle, and those are channels of their own (`current_d_A`,
`current_q_A`, `voltage_d_V`, `voltage_q_V`, `rotor_angle_rad` and
`rotor_angle_electrical_rad`, plus the three-phase `current_a_A` / `voltage_a_V`
family, Park-transformed at load time).  That changes the identifiability answer
above in exactly one place.  `Ld` and `Lq` live in different voltage equations:

```
v_d = R i_d + Ld di_d/dt - w_e Lq i_q
v_q = R i_q + Lq di_q/dt + w_e (Ld i_d + lambda)
```

so with both dq currents *and* both dq voltages measured `spec.inductance_d`
stops being inert, and the report says so per dataset rather than per model.
Settled dq rows — currents as inputs, voltages as residuals, no integrator
involved — recover all four constants on the truth machine above, with 0.02 A and
20 rpm of injected noise:

```
group.motor.spec.resistance      0.0914739   (1-sigma 5.3e-4,  truth 0.09)
group.motor.spec.flux_linkage    0.00209704  (1-sigma 1.6e-6,  truth 2.1e-3)
group.motor.spec.inductance_d    0.000140043 (1-sigma 2.3e-7,  truth 1.4e-4)
group.motor.spec.inductance_q    0.000189968 (1-sigma 1.5e-7,  truth 1.9e-4)
```

condition 16.5.  The same operating points written as three-phase columns give
the same four numbers to the fourth digit.  Iron loss, windage and
flux-temperature drift stay inert either way: the dq plant's mechanical side has
no such term, so no amount of current-loop data makes them fittable.

A *timeseries* dq log is replayed through `drivetrain.dq_sim` with the logged
`v_d`/`v_q` held across each step, started from the log's own first `i_d`/`i_q`/
speed, and integrated inside the speed-dependent stable step (6.25e-4 s at
standstill, 1.80e-4 s at base speed for `bldc_pod`) — a log too coarse for that is
refused rather than integrated unstably.  Replaying a log with the constants that
generated it leaves exactly zero residual, which is the check worth running before
believing a fitted number.  `docs/fitting.md` has the dataset rules, the worked
recoveries and the sampling sweep that shows logging rate beating excitation shape
for the inductances.

## Catalogue, CLI and GUI

`presets.MOTORS` has `bldc_24v` and `bldc_12v`, and `presets.PRESETS` has
`bldc_pod` (2 machines, 1/6 gearbox, 8 kg vehicle load on 0.10 m wheels).  Any
preset can swap in a brushless machine and vice versa:

```bash
python3 -m drivetrain simulate --preset pod_4x_geared --motor bldc_12v
python3 -m drivetrain sweep --preset bldc_pod --param group.motor.spec.flux_linkage
```

The parameter panel edits `group.motor.spec.*` (labelled "Motor (brushless)")
and refuses `group.motor.kt`, `kv` or `inductance`, which are derived:

```
PMSMModel.kt is derived, not a field, so it cannot be set.  The constants live
under group.motor.spec.* (resistance, inductance_d, inductance_q, flux_linkage,
...).
```

Saved models carry `"kind": "PMSMModel"` next to the motor block and a nested
`spec`, so a brushless drivetrain round-trips through JSON; an unknown kind is
rejected rather than silently loaded as brushed.

### Exporting the dq law itself

Everything above goes through the `i_d = 0` collapse, which is what makes the
machine fit the existing pipeline.  A field-oriented controller does not want the
collapse, so `drivetrain/export_dq.py` is a second generator that keeps the dq
frame:

```bash
python3 -m drivetrain export --dq --preset bldc_pod -o build/model
```

It writes `drive_model_dq.py` (plus bench, manifest and README, schema
`drive-model-dq-1`) whose `current_for_torque(tau, rot_vel)` returns the
`(i_d, i_q, current_saturated, voltage_saturated)` four-tuple — MTPA closed form
below the band edge, fixed-count scan-and-bisect onto the voltage ellipse above
it — with `max_torque`, `torque`, `voltage`, `losses`, `dc_link_current`, a
three-channel `State` and a clocked `FeedForward`.  Ask for `c` and `rust` — the
default — and you also get `drive_model_dq.{h,c,rs}` and a bench per language.
The twins are transliterations of the same fixed-count arithmetic: C returns
guard codes (`dq_step` 0/-1, `dq_integrate` 0/-1/-2) because it has no
exceptions, Rust returns `Result<DqState, DqError>` and is `core`-only with its
own square root, so it drops into a `no_std` crate.  See
`docs/exporting.md`, "The C and Rust twins".

The acceptance test is parity with `PMSMSpec`, not with itself: current magnitudes
agree to 5.7e-13 relative inside the envelope and 2.9e-4 at or above it, and
`max_torque` tracks `torque_limit` to 1.5e-7.  Cross-language parity is measured
too: the generated C and Rust benches print CSVs bit-identical to the Python one
at f64, and a 235-row probe agrees to 1.0e-15 (C) and 8.5e-15 (Rust) relative.
Two of its bugs were found only
because it disagreed with the engine, and both are traps if you write this law by
hand: the MTPA denominator is `Lq - Ld`, the **opposite** sign to the
`RELUCTANCE = Ld - Lq` term in the torque equation (get it wrong and `i_d` has
the wrong sign, costing up to 6% of current), and `i_d` does **not** flip sign
with the torque — only `i_q` carries the direction, because reluctance torque
follows the sign of `i_q`.

The one thing that is genuinely different from the brushed export is the step
limit.  The brushed `max_step_time = 0.5 L/R` is a real, speed-independent pole;
the dq current loop, with the shaft speed held fixed, is a **complex** pair at
`-R(Ld+Lq)/(2 Ld Lq) ± j w_e`, and RK4's absolute-stability region reaches only
`|λh| ≈ 2.78` on the imaginary axis.  So the stable step shrinks with speed — for
`bldc_pod`, 6.25e-4 s at standstill, 1.79753e-4 s at base speed, 9.018e-5 s at
twice base speed.  `step()` raises past `max_step_time(rot_vel)`; `integrate()`
sub-steps, re-sizing each sub-step from the speed it has actually reached.  A
speed trace that goes `nan` a few tens of milliseconds in, after jumping further
than the machine's acceleration can explain, is that bug.

## What is not modelled

Not modelled anywhere: magnetic saturation, iron loss as a function of current
as well as speed, rotor and magnet eddy loss, deadtime and inverter
non-linearity, cogging torque, trapezoidal/harmonic back-EMF, sensorless
rotor-angle estimation, winding thermal mass, demagnetisation events.  Modelled
in `PMSMSpec` but **not** carried by the collapse: reluctance torque, `Ld`, core
and windage loss, magnet temperature drift, and the difference between DC-link
and phase current.  `docs/efficiency.md` bills the losses that *are* modelled,
per operating point and per kind, and keeps the absent kinds visible: the magnet
term is `None` rather than `0.0`, and every report ends with what the map does
not model.

## Tests

- `tests/test_pmsm.py` — 70 tests on the machine: conventions, analytic points,
  equivalence with the brushed model at `i_d = 0`, MTPA, both limits, the
  envelope (flat, hyperbolic, constant power, even in speed, regenerative
  larger than motoring), feed-forward, losses, temperature, validation.  The
  strongest one is `test_the_current_angle_is_optimal`, which brute-forces the
  current angle on a 3001-point grid and asserts `current_pair` uses no more
  current than the best feasible grid point.
- `tests/test_pmsm_system.py` — 45 tests on the seams: the adapter view, the
  band-edge API against an independent ellipse solve, `MotorGroup` collapse
  numbers, JSON round-trip with `kind`, catalogue/panel/sweep behaviour, export
  parity and the machine-specific export notes, and fitting a brushless machine
  through the brushed vocabulary.
- `tests/test_export_dq.py` — 94 tests on the dq generator, all of them parity
  tests rather than smoke tests: constants and derived values against the spec,
  the machine-specific notes, `current_for_torque` / `max_torque` / `mtpa_pair`
  against `PMSMSpec` over a speed × torque grid in both directions, the
  zero-torque corner above the rail speed, the speed-dependent step limit and
  `integrate()`'s re-sized sub-steps, feed-forward, f32 builds, the manifest and
  its `targets` record, a non-salient machine, a machine rated below its
  characteristic current, both bench modes, the `--dq` command line, and the C
  and Rust twins: both compile warning-free, the Rust module builds inside a
  `#![no_std]` crate, a 235-row probe in each language matches the generated
  Python to 1e-12 relative (measured 1.0e-15 and 8.5e-15), and the generated
  benches print CSVs bit-identical to the Python bench in both modes.
- `tests/test_dq_sim.py` — 33 tests on the library dq plant: the stable step
  shrinking with speed and the two mechanisms behind it, `step` refusing an
  unstable `dt`, `integrate`'s re-sized sub-steps matching a fine fixed step,
  fourth-order convergence, the analytic current rise and back-EMF equilibrium,
  energy accounting, the collapse onto the brushed engine, Park conventions (a
  balanced sine must map to a constant vector, zero sequence dropped not folded
  in, amplitude not rms), the channel names, and the feed-forward plant matching
  the generated dq module.
- `tests/test_fit_dq.py` — 33 tests on fitting a rotor-frame log: the dq and phase
  vocabulary, phase→dq conversion at load and its refusals (two phases instead of
  three, phase and dq columns of the same kind, a mechanical angle where the
  transform needs the electrical one), the channel-aware inertness report, settled
  dq rows and three-phase rows recovering resistance, flux, `Ld` and `Lq`, the
  transient replay through the dq integrator and its stable-step guard, and
  `test_a_clean_log_replays_exactly`.
- `tests/test_efficiency.py` — 51 tests on the loss split and the map: books that
close in all four quadrants on both model families, iron loss billed where the
model knows it (`test_efficiency_at_bills_every_loss_the_model_knows`, the bug
that reported 0.874 where the honest number was 0.561), no stiction on a
brushless machine while it is turning, the field-weakening flag meaning weakening
rather than a negative `i_d`, `None` efficiency where no ratio exists, `NaN`
cells outside the envelope and an all-`NaN` magnet matrix, the envelope as a
search rather than a grid walk, the ridge and `best` agreeing, a `kt`/`kv`
disagreement surfacing as `residual_W`, the CSV round-trip leaving the unmodelled
column empty, and the `efficiency` command line.
