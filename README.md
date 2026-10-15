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
`tkinter`.

Tests use stdlib `unittest` (no pytest required):

```bash
python3 -m unittest discover -s tests -t .     # ~17 s
```

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
| `gui` | launch the interactive interface |

Common options: `--preset NAME`, `--set path=value` (repeatable), `--json`.

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
```

- **Time response** tab — speed, current/torque, power in/out/copper loss,
  efficiency; metrics readout; CSV/PNG export; JSON config save/load.
- **Parameter sweep** tab — pick a path, min/max/points, pick a metric, plot
  metric vs parameter, sorted table.
- **Compare presets** tab — overlay every preset on one chart.

Controls are **generated from the model itself** (`presets.field_catalog()`), so
a new parameter in a dataclass appears as a slider with no UI plumbing. Runtime
state (e.g. `motor.current`) is excluded; initial conditions (e.g.
`motor.rot_vel`) stay editable. Simulations run on a worker thread and post
results through a queue, so the window never blocks.

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
  presets.py     motor/wheel catalogues, presets, JSON, UI metadata
  cli.py         command-line front end
  gui.py         tkinter + matplotlib interface
legacy/          the original scripts, kept as reference (still runnable)
tests/           unittest suite
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
