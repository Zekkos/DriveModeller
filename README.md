# drive_model

A DC geared drivetrain model: motor group → gearbox → load, with time-domain
solvers, parameter sweeps, design optimisation, a CLI and an interactive GUI.

Rewritten from the original flat scripts (kept in `legacy/`) into a package with
correct parallel-motor physics, current limiting, quadratic road load, and
solvers that can be checked against each other and against the analytic
equilibrium.

---

## Quick start

```bash
python3 run.py                    # open the GUI
python3 -m drivetrain presets     # list built-in presets
python3 -m drivetrain run --preset pod_4x_geared --duration 10
python3 -m drivetrain validate    # cross-check every engine vs the analytic model
```

No third-party install needed beyond `numpy` and `matplotlib` (already present).
`scipy` is used if available and has a pure-numpy fallback. The GUI needs
`qtpy` plus any one Qt binding (PyQt5, PyQt6, PySide2 or PySide6); `QT_API=pyqt6`
etc. selects the binding explicitly. On a PEP 668 "externally managed" system
such as Arch, a local venv that reuses the distro's Qt and matplotlib is the
quickest route:

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python qtpy
.venv/bin/python run.py gui
```

Tests use stdlib `unittest` (no pytest required):

```bash
python3 -m unittest discover -s tests -t .     # ~70 s
```

The suite drives the real Qt widgets on Qt's offscreen platform, and skips
itself when no Qt bindings are installed, so it is safe to run headless.

---

## The model

Everything is referred to the **motor shaft**. State is rotor speed `ω` (rad/s),
plus rotor current `i` when `inductance > 0`.

Electrical:

    V = i·R + kv·ω          (L·di/dt + i·R + kv·ω = V if L > 0)

Mechanical:

    J·dω/dt = kt·i − b·ω − τ_load(ω)

`J` and `b` include the gearbox and load reflected through the ratio.

### Parallel motors — the non-obvious part

N identical motors geared to one shaft are **parallel back-EMF sources**, so
`kt` and `kv` do *not* scale with count. Only the winding resistance divides:

| quantity | scaling with `count` |
|---|---|
| `kt_eff`, `kv_eff` | unchanged |
| `R_eff`, `L_eff` | `/ count` |
| `J_eff` | `× count` |
| stall torque | `× count` |
| no-load speed | unchanged |
| electromechanical damping | `× count` |
| mechanical time constant | unchanged |

Adding motors buys torque and current headroom, **not** top speed. A common
wrong model scales `kt` and `kv` together, which inflates top speed and damping.

### Current limiting

`max_current` is per motor; the group limit is `max_current × count`. When the
algebraic current would exceed it, the drive is treated as a **current source**
(`τ = kt · i_limit`) for that step and at equilibrium — which is what a real ESC
does. Equilibrium speed is reported with a mode: `stalled`, `current_limited`,
`voltage_limited`, or `free`.

### Gearbox ratio convention

The native convention is **speed ratio** = `output_speed / input_speed`, so
`ratio < 1` is a reduction. `--ratio-classical` interprets input as classical
`N:1` reduction instead. Efficiency is applied in the power-flow direction.

### Load

`RotationalLoad` (inertia, viscous, constant torque) or `VehicleLoad` (mass,
wheel radius, rolling resistance, `Cd·A` aero drag, grade, final-drive
efficiency). Reflected load torque is fitted as

    τ_load(ω) = sign(ω) · (static + viscous·|ω| + quadratic·ω²)

which is **exact** for every built-in load model. The fit is verified against the
real model at a speed it was not fitted at; a custom load that does not fit sets
`exact=False` and the engines fall back to evaluating it directly.

---

## CLI

```bash
python3 -m drivetrain <command> [options]
```

| command | what it does |
|---|---|
| `presets` | list presets, motors, wheels |
| `describe` | derived quantities + every editable parameter |
| `run` | time-domain simulation (`--csv`, `--json`, `--excitation`) |
| `sweep` | sweep one parameter, print a metrics table |
| `optimize` | best value of a parameter for an objective |
| `validate` | cross-check engines against each other and the analytic model |
| `fit` | fit parameters to a measured dataset (see `docs/fitting.md`) |
| `export` | write the model as C, Rust and Python for a controller (see `docs/exporting.md`); `--dq` writes the dq/FOC law of a brushless machine |
| `efficiency` | loss split at one operating point, or an efficiency map over the (speed, torque) plane (see `docs/efficiency.md`) |
| `gui` | launch the interactive Qt interface (response / sweep / compare / fit tabs) |

Common options: `--preset NAME`, `--config FILE` (start from a saved model
instead of a preset), `--set path=value` (repeatable), `--json`.

Any nested parameter is addressable by dotted path:

```bash
python3 -m drivetrain run --preset pod_4x_geared \
    --set gearbox.ratio=0.12 --set group.count=6 --set load.drag_cd=0.4
```

Examples:

```bash
# sweep gear ratio, 40 points, all cores
python3 -m drivetrain sweep --preset pod_4x_geared --param gearbox.ratio \
    --min 0.05 --max 1.0 --points 40 --metric steady_kmh -j 0

# best gear ratio for a 5 m/s ground speed
python3 -m drivetrain optimize --preset pod_4x_geared --param gearbox.ratio \
    --min 0.05 --max 1.0 --target 5.0

# export a trajectory
python3 -m drivetrain run --preset winch --duration 5 --csv out.csv

# fit the motor constants to a bench log, holding measured parts fixed
python3 -m drivetrain fit bench.csv --preset pod_2x_direct --free electrical \
    --known group.motor.rotor_inertia=2.2e-5 -o fitted.json

# hand the fitted model to the firmware as C/Rust, sized for a 2 ms control loop
python3 -m drivetrain export --config fitted.json --fit-json fitted.json \
    --sample-time 0.002 --out firmware/model
```

`validate` distinguishes two kinds of disagreement, which used to be conflated:

- **solver disagreement** — engines disagree with each other → real numerical
  failure, fails the check.
- **not settled** — all engines agree but the run is shorter than a few time
  constants → reported as a note, not a failure.

```bash
python3 -m drivetrain validate --duration 30 -j 0              # fast, auto step
python3 -m drivetrain validate --duration 30 --dt 2e-5 -j 0    # tight, convergence stress
```

---

## GUI

```bash
python3 run.py                            # or: python3 -m drivetrain gui
python3 -m drivetrain gui --preset sprinter --theme system
```

The window is built on `qtpy`, so it runs against whichever Qt binding is
installed; matplotlib's `QtAgg` backend resolves the same binding through
`qtpy`, so two bindings can never end up loaded in one process.  Everything is
themed to Gruvbox dark (widget palette, stylesheet and chart rcParams) unless
`--theme system` is passed.

- **Time response** tab — speed, current/torque, power in/out/copper loss,
  efficiency; metrics readout; CSV/PNG export; JSON config save/load.  The
  **Overlay previous runs** spinner on that tab decides how many earlier runs
  stay drawn behind the current one (0 to 6, default 0: the chart shows only the
  run you just asked for).  Older runs come back thinner and fainter and stay out
  of the legend, the history is capped at six runs, and moving the spinner
  re-plots the run already on screen instead of simulating a new one.
- **Parameter sweep** tab — pick a path, min/max/points, pick a metric, plot
  metric vs parameter, sorted table.
- **Compare presets** tab — overlay every preset on one chart.
- **Fit dataset** tab — load a measured CSV, tick the parameters the fit may
  move, and everything else keeps the value it has in the panel; see the
  fitting section below.

File → **Export model code...** writes the model as C, Rust and Python for a
feed-forward controller, asking for the control period first; nothing is
generated automatically.  See the export section below.

Controls are **generated from the model itself** (`presets.field_catalog()`), so
a new parameter in a dataclass appears as a slider with no UI plumbing. Runtime
state (e.g. `motor.current`) is excluded; initial conditions (e.g.
`motor.rot_vel`) stay editable. Simulations run on a worker thread and post
results through a signal, so the window never blocks; a debounced *Live* switch
re-runs on every edit.  The **Motor** and **Wheel** pickers follow the model in
both directions: choosing one rewrites the parameters, editing a parameter
switches the picker to `(custom)`.

Parallel sweeps deliberately start their worker pool with `forkserver` (or
`spawn`) rather than the library default `fork`: forking a live Qt process from
a worker thread leaves the child holding locks owned by threads it did not
inherit, which is a deadlock waiting to happen.

---

## Fitting a model to measured data

`drivetrain/fit.py` adjusts model parameters until the simulation matches a
logged dataset, and the result is an ordinary `DriveSystem` that runs, sweeps,
validates and loads like any preset.  The shape of the problem it is built for:
the parts you built are known constants (wheel radius, gear ratio, vehicle
mass, bench-measured rotor inertia) and the parts you cannot measure are free
(`kv`, `kt`, winding resistance).

```bash
python3 -m drivetrain fit bench.csv --preset pod_2x_direct \
    --free electrical \
    --known group.motor.rotor_inertia=2.2e-5 \
    --known group.motor.bearing_damping=3.0e-5 -o fitted.json
```

A `voltage_V` column is replayed as the measured *input* rather than compared,
held at each sample rather than interpolated, so a square-wave bench log is not
smoothed into something the model cannot be.  A CSV without a `time` column is
read as settled operating points, and a `run`/`segment` column packs several
experiments into one file.  Speed may be logged as rpm or rad/s (`motor_rpm`,
`motor_rad_s`, `omega`, `shaft_rad_s`, ...); a column that is off by the 60/2pi
factor gets called out instead of absorbed.  `--free` takes dotted paths or group
names (`motor`, `electrical`, `mechanical`, `losses`, `drivetrain`, `vehicle`,
`rotational`, `wheel`, `inertia`, `friction`, `damping`, `all`); `fit --paths`
lists them and marks the ones this configuration never evaluates.

The report is the point: fitted values with 1-sigma error bars from the
Jacobian, per-channel residual rms against per-channel weights (which are
1-sigma noise estimates measured from the log's own high-frequency scatter, not
hand-tuned preferences), a condition number, and warnings that name what the
data cannot tell you — `kt` and `rotor_inertia` trade off when only speed is
measured, bearing damping and load viscosity are one quantity at a fixed
operating point, a drive that rides its current limit for most of a log stops
revealing its kv or its resistance, a parameter no equation in this
configuration reads stays forever at its starting value, and a `kt`/`kv` pair
five-fold apart describes an impossible machine rather than a bad fit.  The
report also prints where each starting value came from — the log itself, not the
catalogue — because this residual surface has a minimum for every plausible
pairing of torque constant and damping.  `docs/fitting.md` walks through the
dataset format, how to read the report and what fitting cannot tell you.

---

## Brushless motors (BLDC / PMSM)

`drivetrain/pmsm.py` holds two layers.  `PMSMSpec` is the dq machine model —
MTPA current angles, the voltage ellipse, field weakening, both saturation
flags, reluctance torque, iron and windage loss.  `PMSMModel` is that machine
wearing the brushed model's face: `kt`, `kv`, `resistance`, `inductance`,
`max_current`, `rot_vel` as read-only properties derived from the spec, so it
drops into `MotorGroup` and therefore into `DriveSystem`, `simulate`, `sweep`,
`fit`, the GUI and the export without any of them knowing it is brushless.

```bash
python3 -m drivetrain simulate --preset bldc_pod --duration 20
python3 -m drivetrain simulate --preset pod_4x_geared --motor bldc_12v
```

Three conventions that bite:

- `kt / kv = 1.5`, not 1.  In amplitude-invariant Park, `kt = 1.5 p lambda` and
  `kv = p lambda`.  A fitted brushless drive with that ratio is correct.
- The voltage the pipeline wants is the **peak fundamental phase voltage**
  (`V_bus / sqrt(3)` under SVPWM), not the battery.  Feeding 24 V into a 24 V
  system whose rail is 13.86 V predicts a no-load speed 73% too high — and the
  model will report it without complaint, even above the machine's own base
  speed.
- DC-link current is `1.5 (R i + kv w) i / V_bus`, not phase current, and is not
  proportional to torque.  A rail ammeter and a controller's phase-current
  channel are different measurements.

The collapse to `i_d = 0` is exact while the machine is current-limited and the
voltage ellipse is slack, and it saturates at base speed exactly — so a
brushless drivetrain built this way cannot represent field weakening.  It is
also optimistic about how far full torque reaches, because a real FOC drive
spends part of its voltage on the cross-coupling term `w_e Lq i_q`: for
`bldc_24v` full current is reachable to 447.8 rad/s, not the 909.2 rad/s the
collapse implies.  `constant_torque_speed()`, `current_ceiling(w)` and
`collapse_margin()` report the edge instead of hiding it.  `docs/pmsm.md` covers
the model, the envelope, the fitting story and where the collapse stops being
true.

If you want the dq law itself rather than the collapse — current angle, MTPA,
field weakening — that is a second exporter:

```bash
python3 -m drivetrain export --dq --preset bldc_pod -o build/model
```

It writes `drive_model_dq.py` (plus bench, manifest and README) with
`current_for_torque(tau, rot_vel)` returning the `(i_d, i_q)` pair and both
saturation flags, the three-channel RK4 state, and a clocked feed-forward — and
by default `drive_model_dq.{h,c,rs}` and a bench per language too, transliterated
from the same fixed-count arithmetic: the C and Rust benches print CSVs
bit-identical to the Python one at f64.  The sample-time contract
is different there, and not by a little: a dq current loop is a **complex** pole
pair at `-R(Ld+Lq)/(2 Ld Lq) ± j w_e`, so the stable step shrinks as the shaft
speeds up — 6.25e-4 s at standstill for `bldc_pod` but 1.79753e-4 s at base
speed.  `step()` raises past the limit and `integrate()` sub-steps from the speed
it has actually reached.

The same rotor frame works in reverse, from log to model.
`drivetrain/dq_sim.py` integrates the dq plant as a library, and `fit` reads dq
and three-phase channels — `current_d_A`, `current_q_A`, `voltage_d_V`,
`voltage_q_V`, the electrical rotor angle, or the raw `current_a_A`/`voltage_a_V`
phases, Park-transformed at load time.  That is what makes `Ld` separable from
`Lq` in the first place: they appear in different voltage equations, so a DC-side
log can never see `Ld`, and the identifiability report now answers per dataset
instead of per model.  Settled dq rows recover resistance, flux linkage, `Ld` and
`Lq` to 0.03–1.6 % (condition 16.5), and the same operating points written as
three-phase columns give the same four numbers to the fourth digit.

---

## Exporting the model as controller code

The model is inverted: instead of "given voltage, what does the shaft do", a
feed-forward controller wants "given the speed and acceleration I want, what
voltage do I apply".  `drivetrain/export.py` emits that inversion as plain C, a
`no_std`-friendly Rust module and a Python class, plus a ROS node that wraps the
Python one.  Nothing is ever generated as a side effect of running a
simulation — export is a request, and the output is meant to be checked into a
source tree you control.

```bash
python3 -m drivetrain export --preset pod_4x_geared --sample-time 0.002 -o build/model
python3 -m drivetrain export --config fitted.json --fit-json fitted.json -t c,ros
python3 -m drivetrain export --dq --preset bldc_pod -t c,rust -o build/model
```

The generated code carries no runtime dependency on this repository: one
constants table, one step function, and the feed-forward law
`i = (J·a + b·w + load(w) + friction(w) + τ) / kt`, `v = i·R + kv·w`.  The
acceleration term must be the derivative of your *rate-limited* reference — the
helpers do that, which is why `FeedForward` (Python/Rust) and `dt_ff_update` (C)
exist rather than a bare `feedforward(w, a)` call.  A `drive_model_manifest.json`
records the constants, the sample-time contract (`dt` at or below `max_step_s`),
what the build does *not* model, and a sha256 per file.

Parity is a test, not a claim: the C and Rust builds are compiled and run against
the Python one to 1e-9 relative, and the exported RK4 reproduces
`simulate(engine="rk4")` bit for bit.  An f32 build of `pod_2x_direct` is 2401
bytes of `.text` with no `.data` and no `.bss` (`cc -std=c99 -Os`), allocates
nothing, and recurses nowhere.  `docs/exporting.md` covers the API per language,
the units (motor shaft unless the name says output/ground), the saturation flags
and the ROS node's parameters.

---

## Where the losses go

`efficiency` bills a single operating point, or the whole (speed, torque) plane,
without integrating anything: a 33 × 25 map is a few hundred algebraic solves.

```bash
python3 -m drivetrain efficiency --preset bldc_pod --speed 700 --torque 0.2
# 6684.5 rpm    0.2000 N*m   eff  92.68%   in   150.002 W   out   139.020 W
# losses [W]: copper 10.002  bearing 0.980

python3 -m drivetrain efficiency --preset bldc_pod --map --csv effmap.csv
```

Both axes are motor-side, because the envelope that decides what is possible
belongs to the machine; `--torque-is shaft` rebills the machine's own internal
losses on top if you meant the flange.  Efficiency is always useful over
absorbed, per quadrant, and is `None` — not zero — where there is no ratio to
quote (standstill, or losses larger than the developed power).  The books close
by construction: `power_in − total_losses − power_out ≈ 0` at every point, which
is what the `residual_W` column is for, and on a brushed model it is where a
datasheet `kt`/`kv` mismatch shows up instead of being smoothed over.  Cells
outside the envelope are `NaN`, never interpolated into a neighbour.  Magnet and
rotor loss are not modelled anywhere in the package, so `magnet_W` is `None`
rather than `0.0` and every report ends with a `not modelled on this map:` line.
`docs/efficiency.md` covers the loss columns, the four quadrants, the field-
weakening flag and the ridge.

---

## Performance

Two independent levers: **parallelism** and **per-step cost**.

### Parallelism

Sweeps are embarrassingly parallel. `--jobs 0` uses every core:

```bash
python3 examples/benchmark_parallel.py
```

Measured on 16 cores, 1536 combinations: `workers=1` → 0.85 s, `workers=0` →
0.19 s. Default multiprocessing context is `fork` where available so it also
works from a REPL or notebook.

### Per-step cost

Profiling (not guessing) showed the engines spent nearly all their time on Python
overhead, not arithmetic:

- `scipy.linalg.expm` called **once per step** for a 1×1 system
- `np.clip` called on scalars inside the RK4 stage functions
- every step walked the load object graph through attribute chains

Fixes in `drivetrain/simulate.py`:

- `_FrozenModel` flattens the model to plain floats once per simulation and
  carries the fitted load polynomial.
- First-order steps are closed-form: `exp(a·dt)` plus `_expm1_over_a`, a stable
  `(e^{a·dt} − 1)/a` that uses the series below `|a·dt| < 1e-4` (avoids
  catastrophic cancellation and division by zero for an undamped rotor).
- Second-order (`L > 0`) systems keep the augmented-matrix exponential, but it is
  **cached on the damping value** instead of recomputed every step.

Measured, 50,000 steps (`python3 examples/benchmark_engines.py`):

| engine | before | after | speedup |
|---|---|---|---|
| `exact` | 48,000 steps/s | 2,007,000 steps/s | **41.8×** |
| `rk4` | 31,500 steps/s | 540,000 steps/s | **17.1×** |

`validate` across all presets: **28.9 s → 0.22 s**. Results are bit-identical to
before the optimisation.

### Choosing a step size

Omit `--dt` and the solver resolves the fastest time constant with margin
(`min(τ_mech, τ_elec)/50`). `exact` is unconditionally stable, so for a
first-order system the auto step is both accurate and cheap — a 30 s run of a
1.15 s time-constant drivetrain is ~1,300 steps, not 1.5 million. Pass an
explicit small `--dt` only to stress convergence order.

---

## Layout

```
drivetrain/
  motor.py       MotorModel, MotorGroup (parallel-motor equivalence)
  gearbox.py     Gearbox, RatioConvention
  load.py        Wheel, RotationalLoad, VehicleLoad, LoadBundle
  system.py      DriveSystem, LoadCoefficients (assembly + analytic results)
  simulate.py    engines (exact / rk4 / euler), excitation helpers
  sweep.py       sweeps, grid sweeps, parallel workers, optimisation
  fit.py         dataset parsing, least-squares fitting, identifiability report
  pmsm.py        PMSMSpec (dq machine model), PMSMModel (brushed view of it)
  dq_sim.py      the dq plant as a library integrator (RK4, Park, simulate_dq)
  export.py      feed-forward model as C / Rust / Python + ROS node
  export_dq.py   dq/FOC law for a brushless machine (MTPA, field weakening)
  efficiency.py  loss split per operating point, efficiency map over (speed, torque)
  presets.py     motor/wheel catalogues, presets, JSON, UI metadata
  cli.py         command-line front end
  gui.py         qtpy (Qt) + matplotlib interface, Gruvbox dark theme
legacy/          the original scripts, kept as reference (still runnable)
tests/           unittest suite
docs/            fitting.md, exporting.md, pmsm.md, efficiency.md, roadmap.md
examples/        benchmark_parallel.py, benchmark_engines.py
```

---

## Known limits

- Single-degree-of-freedom: no shaft twist, backlash, or gear mesh compliance.
  `inductance > 0` adds the electrical pole, nothing else.
- `exact` re-linearises the load's viscous term each step, so it is exact for the
  linear part and first-order accurate in the quadratic drag. Use `rk4` when you
  need the nonlinear terms resolved at high precision.
- Coulomb friction is handled with stiction only for **bearing** friction; the
  load's static term does not latch the shaft.
- Presets with very long time constants (`spinning_mass`, settle ≈ 51 s;
  `sprinter`, settle ≈ 69 s) need `--duration` well above the settle time before
  their final value means anything. `validate` says so explicitly rather than
  reporting a mismatch.
- A brushless machine reaches this pipeline through the `i_d = 0` collapse, so
  the system-level model has no field-weakening region and no reluctance torque,
  and it overstates how far full current reaches (see `collapse_margin()`).  Use
  `PMSMSpec` directly when the answer depends on the current angle.
