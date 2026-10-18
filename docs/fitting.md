# Fitting a model to measured data

`drivetrain/fit.py` adjusts model parameters until the simulation matches a
logged dataset.  The output is an ordinary `DriveSystem`: it runs, sweeps,
validates and loads in the GUI exactly like a built-in preset.

The usual shape of the problem is the one this module is built around: **you
know the parts you built** (wheel radius, gearbox ratio, vehicle mass, rotor
inertia measured on the bench) and **you want the numbers you cannot measure**
(the motor's kv, kt and winding resistance).  So the fit takes an explicit list
of free parameters and an explicit list of known constants, and it tells you
honestly when the data cannot separate two of them.

```bash
python3 -m drivetrain fit bench.csv --preset pod_2x_direct \
    --free electrical \
    --known group.motor.rotor_inertia=2.2e-5 \
    --known group.motor.bearing_damping=3.0e-5 \
    -o fitted.json
```

```
fit: bench.csv (timeseries, 800 residuals)
engine=exact  dt=0.02 s  evaluations=8  iterations=7  status=`ftol` ...

free parameters
  parameter                                start       fitted    1-sigma
  group.motor.kv                           0.018     0.021235    1.9e-05  s/rad
  group.motor.kt                           0.018    0.0211719    1.9e-05  N*m/A
  group.motor.resistance                    0.62     0.440066    4.4e-05  ohm

residual rms per channel
  current_A           0.02766   (n=400, weight 0.0304, max |err| 0.0964, 0.002 of the channel's own spread)
  motor_rpm             9.693   (n=400, weight 9.24, max |err| 27.8, 0.049 of the channel's own spread)
  normalized rms is the residual against the per-channel noise estimate: ~1 = as good as the data allows, >>1 = the model cannot follow the data, <<1 = the weights overestimate the noise

residual rms     0.982 in weight units (1.0 = the fit matches the data as well as the weights allow)
condition number 12.5
  weights are 1-sigma noise per channel -- current_A: high-frequency scatter; motor_rpm: high-frequency scatter

known constants
  group.motor.bearing_damping              3e-05
  group.motor.rotor_inertia              2.2e-05
```

---

## The dataset

A CSV with a header row.  Column names are matched case-insensitively, and the
common shorthand below is accepted:

| canonical | also accepted |
|---|---|
| `time` | `t`, `timestamp`, `seconds`, `sec`, `elapsed`, `time_s` |
| `motor_rpm` | `rpm`, `motor_speed`, `shaft_rpm`, `speed_rpm` |
| `output_rpm` | `gearbox_rpm`, `wheel_rpm`, `output_shaft_rpm` |
| `motor_rad_s` | `rad_s`, `omega`, `angular_velocity`, `motor_omega`, `rotational_speed` |
| `output_rad_s` | `shaft_rad_s`, `output_omega`, `wheel_omega`, `gearbox_omega` |
| `speed_kmh` | `speed`, `kmh`, `km_h`, `ground_speed` |
| `speed_m_s` | `velocity`, `speed_ms`, `m_s`, `mps` |
| `current_A` | `current`, `amps`, `i`, `line_current`, `motor_current` |
| `voltage_V` | `voltage`, `volts`, `v`, `supply`, `bus_voltage` |
| `torque_Nm` | `torque`, `torque_nm` |
| `power_in_W` / `power_out_W` | `power_in`/`pins`/`watts_in`, `power_out`/`pout`/`watts_out` |
| `copper_loss_W` | `copper_loss`, `i2r`, `i_squared_r` |
| `efficiency` | `eff`, `eta` (values above 1.5 are read as percent and rescaled) |
| `distance_m` | `distance`, `odometer` |
| `current_d_A` / `current_q_A` | `id`/`iq`, `i_d`/`i_q`, `current_d`/`current_q`, `d_axis_current`/`quadrature_current`, `direct_axis_current` |
| `voltage_d_V` / `voltage_q_V` | `vd`/`vq`, `v_d`/`v_q`, `voltage_d`/`voltage_q`, `d_axis_voltage`/`q_axis_voltage` |
| `rotor_angle_rad` | `theta`, `theta_mech`, `rotor_angle`, `rotor_position`, `shaft_angle`, `angle` |
| `rotor_angle_electrical_rad` | `theta_e`, `theta_electrical`, `rotor_angle_electrical` |
| `current_a_A` / `current_b_A` / `current_c_A` | `ia`/`ib`/`ic`, `current_a`/…, `phase_a_current`, `line_current_a`, `current_u`/`current_v`/`current_w` |
| `voltage_a_V` / `voltage_b_V` / `voltage_c_V` | `va`/`vb`/`vc`, `voltage_a`/…, `phase_a_voltage`, `terminal_voltage_a`, `voltage_u`/`voltage_v`/`voltage_w` |

The last three rows only mean anything for a brushless machine; see
[A brushless log in its own frame](#a-brushless-log-in-its-own-frame).

Three structural rules:

- **`voltage_V` is the input, not a residual.**  If the log contains the drive
  voltage it is interpolated onto the solver grid and replayed into the
  simulation, so the fit compares like-for-like even when the supply sagged
  mid-run.  It is then excluded from the residuals.  Without that column the fit prescribes the
  input itself: `--excitation step|square|ramp|alternating|coast`,
  `--voltage`, `--period`, `--ramp`.
- **No `time` column means steady state.**  Each row is one settled operating
  point, and `voltage_V` is required — except for a brushless machine measured
  in the rotor frame, where `current_d_A` plus `current_q_A` plus a speed is
  enough and the dq voltages become the residuals.  See
  [A brushless log in its own frame](#a-brushless-log-in-its-own-frame).
  Steady rows are evaluated from the
  closed-form equilibrium (including the current limit), which is the cheapest
  and most robust way to fit `kv` and `resistance` from a datasheet table.
- **Extra experiment column** (`run`, `segment`, `test`, `trial`, `experiment`,
  `block`) splits the file into independent experiments.  Each block is
  simulated separately with its own input; a block whose clock restarts at zero
  is treated as a new run.  Different step counts per block are fine.

Blank and non-numeric cells become gaps and are simply skipped, so a channel
that only exists for part of the log is not a problem.

Angular velocity may be logged in rad/s rather than rpm: a bare `rad_s` or
`omega` column is read as the **motor** shaft (that is where an encoder
normally is), and `shaft_rad_s` / `output_omega` as the output shaft.  If a
speed column tracks the model perfectly but sits 9.5× (= 60/2π) off, the report
says so instead of quietly refitting kv to absorb the unit mix-up.

---

## Free parameters and known constants

`--free` takes dotted parameter paths, or a group name:

| group | paths |
|---|---|
| `motor` | `kv`, `kt`, `resistance`, `rotor_inertia`, `bearing_damping`, `static_friction` (all under `group.motor.`) |
| `electrical` | `group.motor.kv`, `.kt`, `.resistance` |
| `mechanical` | `group.motor.rotor_inertia`, `.bearing_damping`, `.static_friction`, `gearbox.inertia`, `gearbox.damping` |
| `losses` | `group.motor.bearing_damping`, `.static_friction`, `.no_load_friction_current`, `gearbox.efficiency`, `gearbox.damping` |
| `drivetrain` | `gearbox.ratio`, `.efficiency`, `.inertia`, `.damping` |
| `vehicle` | `load.mass`, `load.rolling_resistance`, `load.drag_cd`, `load.air_density`, `load.grade_deg`, `load.extra_rotational_inertia` |
| `rotational` | `load.inertia`, `load.viscous`, `load.constant_torque` |
| `wheel` | `load.wheel.radius`, `load.wheel.mass`, `load.wheel.inertia_factor` |
| `inertia` | every inertia in the system: `group.motor.rotor_inertia`, `gearbox.inertia`, `load.inertia`, `load.extra_rotational_inertia`, `load.wheel.mass` |
| `friction` / `damping` | every loss that is not copper: `group.motor.bearing_damping`, `.static_friction`, `.no_load_friction_current`, `gearbox.damping`, `load.viscous`, `load.constant_torque`, `load.rolling_resistance`, `load.drag_cd` |
| `all` | every parameter this configuration exposes, except the ones you pinned with `--known` and anything without a finite starting value |

Default is `--free motor`.  Anything you pin with `--known path=value` is held
fixed, and a group alias that contains a pinned path just skips that path — so
`--free motor --known group.motor.rotor_inertia=2.2e-5` does what it looks
like.  Naming the same path both free and known is an error rather than a
silent precedence rule.  The two families people actually ask about are
separate groups: `--free inertia --known group.motor.bearing_damping=3e-5` is
"I measured the inertia on the bench and want the friction", and the other way
round is `--free friction --known ...`.

`fit --paths` prints every path the current model exposes, with units and
ranges, plus the group names.

Other knobs:

- `--tie group.motor.kt=group.motor.kv` — one value drives several paths (useful
  for the permanent-magnet identity `kt == kv` in SI units).
- `--bound path=min:max` — widen or narrow a search range.  Catalogue ranges
  are widened ×100 at each end by default, so a starting value that is off by
  two decades still converges; `--bound` overrides that.  Physical hard bounds
  always apply on top: efficiencies are capped at 1.0, and `kv`, `kt`,
  `resistance`, `rotor_inertia`, `gearbox.ratio`, wheel radius and mass are
  bounded away from zero.  A parameter that finishes on a bound is called out.
- Parameters are optimised in log space by default (`--no-log` to disable):
  positive quantities stay positive, and a parameter spanning decades stops
  dominating the conditioning.  The reported 1-sigma is converted back to
  natural units.
- Integer paths (`group.count`, `load.wheels_driven`) are fitted continuously
  and rounded at the end, with a note saying so.
- State fields (`group.motor.current`, `group.motor.rot_vel`) are never
  fittable.  A parameter whose current value is infinite (`max_current`,
  `max_voltage`) cannot be log-fitted either — give it a finite value first
  (`--set group.motor.max_current=20`) and it becomes fit-able.

---

## Reading the report

- **1-sigma** comes from the Jacobian at the optimum (`σ = sqrt(diag(JᵀJ)⁻¹·s²)`),
  so it is the *statistical* uncertainty given the data and its weights, not an
  accuracy claim.  A huge 1-sigma means the data says nothing about that
  parameter.
- **residual rms per channel** is in the channel's own units, with the weight
  used and, in the last figure, the residual as a fraction of that channel's
  own spread (`res.relative_rms`) — the unit-free number, useful when the
  weights are smaller than the signal.
- **Weights are 1-sigma noise estimates, not scatter.**  By default each
  channel's weight is the high-frequency scatter of the log itself (median
  absolute deviation of first and second differences, which ignores the
  signal's own slope), so a channel that is 100 rpm of encoder noise and a
  channel that is 0.03 A of current noise get the same influence.  `--weight
  current_A=0.05` pins a noise level you measured instead; the report says
  which of the two it used.  Steady-state rows have no high-frequency content
  to measure, so there the weight falls back to the channel's own spread.
- **residual rms in weight units** is the single number to judge: ≈1 means the
  fit is as good as the noise allows, ≫1 means the model cannot reach the data
  (wrong structure, or the wrong thing held fixed), ≪1 means the weights
  overestimate the noise and the fit is chasing it.  On a noiseless synthetic
  log the weights fall to their floor, so read the per-channel figures too.
- **condition number** of the same Jacobian.  Below ~1e3 is comfortable; above
  that two or more free parameters are moving together.
- **warnings** name the actual problem and the fix, e.g.
  `group.motor.kt is poorly determined (0.035 on 0.0234); consider fixing it`,
  `group.motor.rotor_inertia ended on its bound (...); widen bounds= if that is wrong`,
  `kv and kt are both free but no electrical channel is measured: consider ties=...`,
  `kt and rotor_inertia trade off when only speed is measured (speed sees kt/J);
  add current_A data or fix one of them`,
  `1 direction(s) of this fit cannot be resolved at all from this data`,
  `indistinguishable in this data: group.motor.bearing_damping / load.viscous (r=+1.000) -- to the fit these are one quantity`,
  `no sensitivity at all: group.motor.inductance (the current loop is algebraic here...)`,
  `the model rides its 40 A current limit for 78% of the fitted trajectory: while the drive is saturated...`,
  `settled operating points contain no transient, so group.motor.rotor_inertia leave no mark on this data at all`,
  `motor_rpm: the model tracks the data exactly but sits 9.5x above it -- that is the 60/2pi factor, so the column is rad/s, not rpm`.

`--json` gives the whole thing (parameters with stderr/bounds, per-channel rms,
condition, warnings, and the fitted model in the same JSON format as
`presets.save`), and `--out model.json` writes just the model.

---

## Fitting a brushless machine

A `PMSMModel` is fitted through the brushed vocabulary.  `--free electrical` on a
`bldc_pod` log works, and the fitter re-points it at the constants a brushless
machine can actually set (`kv` → `group.motor.spec.flux_linkage`, `resistance` →
`spec.resistance`, `inductance` → `spec.inductance_q`, ...), converting the
values on the way: `lambda = kv / p` and `lambda = kt / (1.5 p)`.  Asking for
`kt` and `kv` together — tied, or both pinned to a datasheet pair — is refused,
because on that machine they are one flux linkage seen from two sides.

Two things the report tells you rather than quietly fitting:

- `Ld`, iron loss, windage and flux-temperature drift are **structurally inert**
  under the `i_d = 0` collapse: the model never evaluates them, so a fit would be
  fitting noise.  Naming one explicitly still fits it, on your terms.  `Ld` stops
  being inert as soon as the log is in the rotor frame — see [A brushless log in
  its own frame](#a-brushless-log-in-its-own-frame); iron loss and windage stay
  inert either way, because the dq plant's mechanical side has no such term.
- `pole_pairs` is fittable but perfectly correlated with flux linkage — only the
  product `p * lambda` is ever read.

The hazard that actually bites is **clipping**.  Drive the machine into its
current limit for most of the log and the current channel stops carrying
information about back-EMF and winding resistance; the fit then returns a flux
linkage five times too big with a normalised rms in the hundreds, and says so:

```
the measured current_A sits at 40.14 for 88% of the log: the drive itself is
clipping there
```

Design the experiment instead: a square-wave voltage well inside the linear
region, sampled fast enough to see the electrical pole.  And remember that a rail
ammeter is not a phase-current channel — the DC side reads
`1.5 (R i + kv w) i / V_bus`, which is not proportional to torque.  A worked
recovery (flux 0.00210064 against a truth of 0.0021, sigma 2.9e-5, on a 2 V
square wave on a 13.86 V rail) is in `docs/pmsm.md`.

### A brushless log in its own frame

A drive running FOC reports `i_d`, `i_q`, the voltages it applied on those axes,
and usually a rotor angle rather than a speed.  Those are channels of their own —
`current_d_A`, `current_q_A`, `voltage_d_V`, `voltage_q_V`, `rotor_angle_rad`
(and `rotor_angle_electrical_rad`) — and a log in the rotor frame is a different
experiment, not a brushed log with extra columns.

How such a log is read:

- Three-phase columns are transformed at load time.  `current_a_A`,
  `current_b_A`, `current_c_A` (and the matching `voltage_a_V`/`b`/`c`) become
  `current_d_A`/`current_q_A` through the amplitude-invariant Park transform, so
  a drive that only speaks phase still gets fitted.  All three phases of a kind
  are needed — a two-phase log is refused rather than reconstructed — and phase
  columns of a kind alongside dq columns of the same kind are an error, not a
  redundancy.
- The transform needs the **electrical** angle, `rotor_angle_electrical_rad`.  A
  mechanical angle is refused: converting it needs the pole count, and the pole
  count is one of the numbers a fit may be adjusting.  If both angles are logged,
  the fit reports the median `d(theta_e)/d(theta)` back at you as the pole count
  it found, which is a cheap way to catch a wrong `pole_pairs` before fitting.
- Angles are unwrapped at load, and any zero sequence (`(a + b + c) / 3`) is
  reported and dropped — the dq model has nowhere to put it.

**The two inductances come apart.**  They live in different equations:

```
v_d = R i_d + Ld di_d/dt - w_e Lq i_q
v_q = R i_q + Lq di_q/dt + w_e (Ld i_d + lambda)
```

`Ld` only moves `v_q` and `Lq` only moves `v_d`, so with both currents *and* both
voltages measured the two of them are live parameters instead of inert ones, and
the report stops refusing `inductance_d`.  Without them the brushed answer stands:
one inductance, and it is `Lq`.

**Settled dq rows are a complete electrical experiment.**  A row of
`current_d_A`, `current_q_A` and a speed needs no `voltage_V` and no integrator:
the currents are inputs and the two voltages are the residuals.  Settled rows
still have to spread over speed — back-EMF is proportional to speed, and without
that spread flux linkage and resistance trade off against each other — on the
machine above, with 0.02 A and 20 rpm of injected noise:

| parameter | truth | recovered | 1-sigma | error |
| --- | --- | --- | --- | --- |
| `spec.resistance` | 0.09 | 0.0914739 | 5.3e-4 | 1.6 % |
| `spec.flux_linkage` | 0.0021 | 0.00209704 | 1.6e-6 | 0.14 % |
| `spec.inductance_d` | 1.4e-4 | 0.000140043 | 2.3e-7 | 0.03 % |
| `spec.inductance_q` | 1.9e-4 | 0.000189968 | 1.5e-7 | 0.02 % |

condition 16.5, 7 evaluations, and the dq current residuals are exactly zero
because those currents are the input.  The same operating points written as
three-phase columns give 0.0914466 / 0.00209632 / 1.39807e-4 / 1.90145e-4 at the
same condition number: the transform costs nothing and changes nothing.

**A transient dq log is replayed, not prescribed.**  A timeseries block must log
`voltage_d_V` and `voltage_q_V`: they are what drive the plant, and a prescribed
constant pair constrains resistance and flux and says nothing about the
inductances.  The fit integrates the dq plant (`drivetrain/dq_sim.py`,
`simulate_dq`) on the log's own time base, holds
each voltage across its step (which is what a current loop does between updates),
starts from the log's first `i_d`/`i_q`/speed, and aligns the angle to the first
measured sample.  Replay a log with the constants that generated it and the
residuals are exactly zero — that is the property
`test_a_clean_log_replays_exactly` pins, and it is worth running on your own data
before believing a fitted number.

The stable step of that integrator shrinks with speed, and the fit honours it:
`bldc_pod` allows 6.25e-4 s at standstill, 1.80e-4 s at base speed and 9.0e-5 s
at twice base speed, because the dq current loop has a complex pole pair, not the
real one the brushed model has.  A log too coarse to integrate stably is refused
(the numbers below are the truth machine's, whose `tau_e` is 1.56 ms, not
`bldc_pod`'s):

```
run 1: the dq plant needs at least 103 steps to stay inside the stable step
7.778e-04 s at 24 rad/s (dq pole magnitude 582 1/s), but max_steps=60 caps it
lower.  Raise max_steps, shorten the log, or fit the brushed collapse.
```

**Sampling rate beats excitation shape.**  Each inductance only shows up as the
current transient on its *own* axis, so step both.  But what dominates is how
fast you log.  Four excitations (flat `v_d` with a `v_q` step; `v_d` stepped as
well; both axes stepped twice; a 40 ms square wave), two grids, two noise seeds,
all four constants free — worst parameter error across the two seeds:

| excitation | 2.5e-4 s, 400 samples | 1e-4 s, 800 samples |
| --- | --- | --- |
| flat `v_d`, `v_q` stepped | 12.7 % | 1.5 % |
| `v_d` stepped as well | 7.0 % | 0.7 % |
| both axes stepped twice | 7.3 % | 0.8 % |
| 40 ms square wave | 9.6 % | 1.4 % |

The truth machine's electrical time constant is `Ld / R` = 1.56e-3 s, so the
coarse grid logs about 6 samples across it and the fine grid about 15.  The
normalised rms says this before the parameters do — 6.0 to 7.3 on the coarse grid
against 1.0 to 1.4 on the fine one, where 1.0 means the residuals *are* the
injected noise.  A normalised rms well above 1 on a log that looks clean is a
sampling problem, not a model problem.  The square wave has the best condition
number (7.5 to 8.5 against 19 to 32 for the others) and the worst flux linkage on
the coarse grid (9.6 %), which is the whole trade-off in one row: a
well-conditioned fit of an under-sampled edge still lands in the wrong place.

## What fitting cannot do

Two failure modes are worth internalising, because both produce a *good-looking*
fit.

**Fitting cannot discover structure.**  Generate data from `pod_2x_direct`
(kv = kt = 0.0212, R = 0.44) and fit it starting from `pod_4x_geared` — wrong
gear ratio, wrong motor count — and the fit still lands at a small rms, but at
kv 0.0218, kt 0.0148 (−30 %) and R 0.878 (+99 %).  The free parameters absorb
the structural error.  Get the topology right first: motor count, gearbox
convention and ratio, and whether the load is a vehicle or a rotational mass.
The residual rms in weight units is the tell; the parameter values are not.

**Data decides what is separable.**  Fit `kv`, `kt` and `rotor_inertia` to
speed-only data and the report says so bluntly:

```
condition number 2.7e+11  (degenerate: some free parameters cannot be separated)
  group.motor.rotor_inertia   1.55e-05 -> 3.26e-12   1-sigma unbounded
  1 direction(s) of this fit cannot be resolved at all from this data
  kt and rotor_inertia trade off when only speed is measured (speed sees kt/J)
```

Speed measures `kt/J`, so with no current channel the pair is one parameter
wearing two hats.  Current measures `kt/R` and `kv`; torque measures `kt`.
Inertia and damping only show up in the *shape* of the transient, so a
steady-state datasheet table recovers `kv` and `resistance` exactly (rms 3e-16
on noiseless data) and says nothing at all about inertia, damping or inductance.

The report now names four more ways a dataset can be silent, because all four
still produce a good-looking fit:

- **A saturated drive cannot see its own electrical parameters.**  While the
  current is pinned at `max_current` the back-EMF and resistive terms have left
  the equations, so `kv`, `resistance` and `inductance` are whatever the
  optimiser liked.  If the fitted model rides the limit for more than a fifth
  of the trajectory the fit says so and asks for a run under the limit.
- **A parameter the configuration never evaluates is unfittable, full stop.**
  `group.motor.inductance` on a model with zero inductance (algebraic current
  loop), `load.drag_cd` with no vehicle attached, a wheel path with no wheel:
  the residuals do not move at all when those move, so no dataset can recover
  them.  They are reported as *no sensitivity at all* rather than as a huge
  error bar.
- **Two paths can be the same physical quantity.**  With a 1:1 gearbox, motor
  bearing damping and `load.viscous` are the same torque on the same shaft; the
  Jacobian columns are collinear (r = ±1) and the fit names the pair instead of
  inventing two numbers.  Fit one and pin the other, or `--tie` them.
- **Steady rows are blind to transients.**  Settled operating points contain no
  inertia, inductance or transient damping information at all; friction and
  damping do appear, but only through the equilibrium balance, where they trade
  off against `kv`.  Use a time-series log for those parameters.
- **dq channels do not resurrect a term the model does not have.**  A rotor-frame
  log makes `Ld` live, but iron loss, windage and the flux-temperature
  coefficient stay inert with it: the dq plant's mechanical side models bearing
  damping and the load, and nothing else, so `core_loss_coefficient` and
  `windage_coefficient` move no residual however clean the current-loop data is.
  Those are waveform-and-thermal parameters and this model does not carry them.

A clean sanity check that the machinery works: synthetic square-wave bench data
(true kv = kt = 0.0212, R = 0.44, J = 2.2e-5, b = 3e-5, noise σ = 0.03 A and
10 rpm), fitted with `--free electrical` and J/b pinned, recovers kv to 0.16 %,
kt to 0.13 % and R to 0.015 %, and the weights it estimates from the data come
out at 0.0304 A and 9.24 rpm against the 0.03 A and 10 rpm actually injected.
It converges to the same optimum from starting values of 0.010, 0.018, 0.030
and 0.050 — no local-minimum problem in this parameterisation.

---

## Library API

```python
from drivetrain.fit import Dataset, fit

ds = Dataset.from_csv("bench.csv")          # or Dataset.from_rows(rows, name=...)
res = fit(ds, preset="pod_2x_direct",
          free="electrical",
          known={"group.motor.rotor_inertia": 2.2e-5},
          ties={"group.motor.kt": "group.motor.kv"},
          weights={"current_A": 0.05},
          engine="exact", excitation="square", voltage=6.0, period=2.0)

print(res.report())
res.system                              # fitted DriveSystem
res.by_path["group.motor.kv"].value     # fitted value (.start, .stderr, .unit)
res.parameters                          # list of FitParameter, in fit order
res.rms["current_A"]                    # per-channel residual, channel units
res.relative_rms["current_A"]           # same, as a fraction of the channel's spread
res.weights["current_A"]                # 1-sigma used for that channel
res.condition, res.degenerate, res.warnings
res.save("model.json")                  # loadable with presets.load()
res.predict()                           # measured vs model, per block/channel
```

`fit()` also accepts `system=` instead of `preset=`, `Path`/CSV path instead of
a `Dataset`, and `channels=` to compare only some channels.  `fit_dataset` is a
thin convenience wrapper.  The optimiser is `scipy.optimize.least_squares` when
scipy is installed, and a self-contained damped Gauss-Newton with numerical
Jacobian otherwise — same residuals, same bounds, same report.

---

## GUI

The **Fit dataset** tab does the same thing interactively: browse a CSV, tick
the parameters the fit may move (quick buttons for the groups above), leave
everything else at the value it has in the parameter panel — that is how the
GUI expresses your known constants — press **Fit**, then **Apply to model** to
push the fitted system back into the panel, or **Save fitted model...**.  The
overlay chart shows measured points and the fitted model on the same axes, and
the report pane is the text report above, warnings included.
