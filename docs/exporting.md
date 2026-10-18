# Exporting the model as controller code

The simulator answers "what will this drive do?".  A controller needs the
opposite question, answered fast and without Python in the loop: "what voltage
makes the shaft follow this speed profile?".  `drivetrain export` writes that
answer as standalone C, Rust or Python — constants baked in, no interpreter, no
numpy, no heap.

Nothing is generated automatically.  The export is a deliberate act (`run.py
export`, or `drivetrain.export.export()` in code), because the generated files
are checked into firmware trees and a silently-refreshed model is a silent
behaviour change.

```bash
python run.py export --preset pod_2x_direct --targets c,rust,python --out build/firmware
```

A brushless machine driven field-oriented gets its own generator, described at
the end: `run.py export --dq`.  Same idea, different law.

## Where the model comes from

Three ways in, all of them the same collapse the simulator uses:

| source | flag | note |
| --- | --- | --- |
| a preset | `--preset pod_4x_geared` | plus `--set path.to.param=value` overrides |
| a saved model | `--config model.json` | the JSON `fit --out` writes |
| a fit result | `--fit-json fit.json` | `fit --json` output; its provenance and warnings are carried into every generated file's header comment |

The GUI has the same thing behind **File → Export model code…**: it exports the
model currently in the panel, fitted or not, after asking for the control period
(the dialog is bounded by that model's own stability limit) and a target
directory.  Same generator, same manifest.

The fit chain is the interesting one: measure a drive, fit it, and the fitted
numbers become firmware without an intermediate hand-transcribed file.

```bash
python run.py fit bench.csv --free motor,friction --json > fit.json
python run.py export --fit-json fit.json --targets c --sample-time 0.002 --out firmware/
```

The manifest records what the model came from (`provenance.source`, and for a
fit: dataset name, point count, `normalized_rms`, condition number and the fit's
own warnings).  If the fit warned that two parameters are one quantity, or that
the log rode the current limit for 61% of its length, that warning is now sitting
in the header of the C file that ships — which is where it belongs.

## What lands in the directory

```
drive_model.h / drive_model.c        C99 header + translation unit
drive_model.rs                       Rust module (no_std-friendly, no heap)
drive_model.py                       pure Python, no numpy
drive_model_ros.py                   rclpy node wrapping the Python model
bench_drive_model.*                  step-response benches, one per language
drive_model_manifest.json            constants, derived values, contract, file hashes
README.md                            the same contract, in prose
```

`--name rover` renames every file and symbol; `--prefix rv` renames the C
identifiers (`rv_model`, `rv_step`, `rv_rover`) so two exported models can live
in one binary.  A half-applied prefix is worse than none, so the rename is
all-or-nothing.  `--no-examples` drops the benches.

## The feed-forward law

The model is inverted, not tuned.  At the motor shaft:

```
i = (J·a + b·w + load(w) + friction(w) + tau) / kt          (required current)
v = i·R + kv·w                                              (applied voltage)
```

with `load(w) = sign(w)·(load_static + load_viscous·|w| + load_quadratic·w²)`,
`friction(w) = static_friction·sign(w)`, and both the current and voltage clamps
applied on the way out.  `tau` is a known disturbance you want to cancel:
`feedforward(w, accel, extra_torque)` takes it at the motor shaft,
`feedforward_output(w_out, accel_out, extra_output_torque)` at the output shaft.

The one rule that decides whether this works in practice:

> **Feed the `a` term the derivative of your *limited* reference, not of the
> setpoint you typed.**  A step setpoint implies infinite acceleration, which
> saturates the current and then unwinds as the shaft lags.  Rate-limit the
> reference first, differentiate the limited reference, invert.

The generated `FeedForward` helper does exactly that — it ramps an internal
reference at `accel_limit`, caps it at `speed_limit`, differentiates the ramp,
and inverts.  Call `update(setpoint)` once per control period:

```c
static dt_ff ff;                            /* C: ramp, differentiate, invert */
dt_ff_init(&ff, &dt_drive_model, 0.002f);
dt_command c = dt_ff_update(&ff, setpoint);
if (c.saturated & 2) { /* you asked for more voltage than the rail can give */ }
```

```rust
let mut ff = drive_model::FeedForward::new(drive_model::MODEL, 0.002);
let cmd = ff.update(setpoint);                   // Rust
```

```python
ff = drive_model.FeedForward(drive_model.MODEL, 0.002, speed_limit=60.0,
                             accel_limit=40.0)   # Python
cmd = ff.update(setpoint)
```

`Command` carries the voltage, the current that voltage implies at this speed,
the torque delivered, and two saturation flags.  In C they arrive as one bitmask
(`c.saturated & 1` current, `& 2` voltage); in Rust and Python they are named
fields.  Read them: when the voltage flag is set, the current you asked for is
not what the drive will do, and any outer loop that assumes linearity is fighting
the clamp.  When the voltage clamps first, `current` is reported as what the
*clamped* voltage implies (`v_limit/R` at standstill), not the current you
requested — that is the physical answer, and it is the same in all three
languages.

`dt_ff_init` installs limits derived from the model itself — no-load speed at the
rail, and stall torque over inertia taken to the output shaft — and the struct is
open, so a controller that wants its own ramp sets `ff.accel_limit` afterwards (0
means "do not limit").  The Rust and Python helpers take the same limits as
constructor arguments instead.

## Units

Everything is at the motor shaft unless the name says otherwise:

- `w` rad/s motor, `tau` N·m motor, `i` A at the model's terminals (parallel
  motors already collapsed), `v` V signed — a negative command drives backwards.
- `*_output` variants (`feedforward_output`, `steady_output_speed`,
  `output_torque`, `motor_speed`) speak output-shaft rad/s and N·m.
- `ground_speed(w)` / `motor_speed_for_ground(v)` need `wheel_radius`; a vehicle
  load collapses to shaft quantities, so the wheel is a unit conversion, not a
  state.
- `load_torque(w)` is the resisting torque at the shaft, `current_for_torque(tau)`
  the current that delivers it, `steady_speed(v)` the equilibrium speed of a
  constant voltage (closed form, no iteration).

## Sample-time contract

The manifest and the generated README both state it:

```
integrator: classical RK4, fixed step, matches the Python engine 'rk4'
max_step_s: 0.8856      take dt at or below this
sample_time_s: 0.002    the control period this build was sized for
stiction_eps_rad_s: 1e-6
```

`max_step_s` is derived from the fastest pole in the build — the electrical time
constant when inductance is non-zero, otherwise the mechanical one.  Ask for
`--sample-time` larger than that and the summary prints a warning rather than
pretending.  `dt_mech_time_constant(m, w)` takes a speed because drag makes the
pole speed-dependent; note that `describe()` in the simulator quotes the pole at
the *nominal* speed (where drag counts) while the exported helper defaults to
standstill, so the two numbers legitimately differ by a factor of two on a draggy
rig.

## Float width, and why the languages agree

`--dtype f32` (default) or `f64`.  In an f32 build the C literals carry the `f`
suffix, `sqrtf`/`fabsf` are used instead of the double libm entry points (which
would silently widen every operation), and the Python export rounds its
constants to the same f32 values.  All three languages then agree bit for bit —
which is what makes the parity tests meaningful:

- `test_the_c_model_integrates_like_the_python_one`: 3000 RK4 steps in C vs the
  generated Python, agreeing to 1e-9 relative.
- `test_the_rust_model_integrates_like_the_python_one`: same in Rust.
- `test_the_step_matches_the_repo_rk4_engine`: the generated code vs
  `drivetrain.simulate(engine="rk4")` — bit-identical, first *and* second order.

That last one is the real acceptance test: the firmware and the simulator are the
same model, not two models tuned to look alike.  Two things it caught, worth
knowing if you compare trajectories by hand:

- the repo engine's grid for `duration=3.0, dt=0.001` is 3000 *points* (t = 0 …
  2.999), i.e. 2999 steps;
- with inductance non-zero, the engine starts current at the algebraic value the
  applied voltage implies (`v/R` at standstill), not zero.  Starting it at zero
  costs milliseconds of torque and a permanent ~0.14 rad/s offset — far larger
  than any integrator error.

To check an export against the simulator yourself:

```bash
cc -std=c99 -O2 -Wall -I. bench_drive_model.c drive_model.c -lm -o bench_c && ./bench_c
python -m drivetrain run --preset pod_2x_direct --engine rk4 --dt 0.001 --duration 3
```

The benches print the same CSV in every language (`time_s,motor_rad_s,current_A,
voltage_V`), sized from the model's own time constant, so they can be diffed
against each other and against `run`.

## MCU notes

- C99, no dynamic allocation, no VLAs, no recursion, no exceptions; the model is
  one `const` struct in flash and the state is two scalars.
- Compiled for x86 with `cc -std=c99 -Os`: **2401 bytes of text, 0 data, 0 bss**
  for an f32 build (2670 for f64).  No ARM cross-compiler on this box, so treat
  that as an order of magnitude, not a datasheet number.
- `dt_step` is the forward model (state update); it is there so the same file can
  serve as a plant model, an observer, or a harness — not because a feed-forward
  controller needs it.
- Rust: `pub type Scalar = f32`, plain structs, no `alloc`, no trait objects.  It
  drops into a `no_std` crate as-is.
- Fixed-point is not generated.  The float story is the documented answer for
  now; if a target has no FPU, that is a decision to make deliberately, not
  something to discover during bring-up.

## ROS

`drive_model_ros.py` is a `rclpy` node around the generated Python model.  It
subscribes to `input_topic` (accepts `TwistStamped`, `Twist` or `Float64`) and
publishes `output_topic` as either a `Float64` voltage command or a
`TwistStamped`, on a timer at `sample_time_s`.  Parameters:

```
input_topic (cmd_vel) · input_axis (x) · output_topic (drive_command)
output_type (voltage|cmd_vel) · sample_time_s · kp (0.0) · max_correction_v (0.0)
use_feedback (False) · feedback_topic (wheel_speed)
speed_limit_rad_s (0 = off) · accel_limit_rad_s2 (0 = off)
```

`kp`/`max_correction_v` add a bounded proportional term on top of the
feed-forward — bounded on purpose, so a mismatch between model and plant shows up
as a bounded correction rather than silently re-tuning the model.  It is off by
default: pure feed-forward is the honest starting point.

The node's logic lives in `DriveModelNodeCore`, which imports nothing from ROS.
That is what makes it testable on a machine without a ROS install (the test suite
drives the core directly), and it is also the class to instantiate if you want
the same behaviour inside your own node instead of the generated one.

## The dq/FOC export (brushless machines)

`export()` speaks the brushed interface, so for a PMSM it emits the `i_d = 0`
collapse: one torque constant, one back-EMF constant, one resistance, and the
machine's caveats written into the banner, the manifest and the generated
README.  That is the right model for a DC-side analysis and the wrong one for a
field-oriented controller, which has to choose the current *angle* as well as
its magnitude.  `drivetrain.export_dq.export_dq()` is a second generator, not a
flag on the first:

```bash
python run.py export --dq --preset bldc_pod --targets python --out build/firmware
```

It writes `drive_model_dq.py`, `bench_drive_model_dq.py`,
`drive_model_dq_manifest.json` and `drive_model_dq_README.md` (schema
`drive-model-dq-1`).  Separate names on purpose: a dq build can sit in the same
directory as the brushed one, and you can diff the two.  The GUI has the same
thing under File -> Export dq/FOC code...; on a brushed preset it says so rather
than writing a collapse the user did not ask for.

All three languages.  `--targets python,c,rust` is the default, and `--dq` with
no `--targets` writes `drive_model_dq.{py,h,c,rs}` plus a bench per language.
The manifest still records which targets were asked for and which were emitted:

```json
"targets": {"requested": ["c", "rust", "python"],
            "emitted": ["python", "c", "rust"],
            "not_emitted": []}
```

That bookkeeping stays because a target can be requested before this generator
can render it.  Today nothing is skipped, and asking for a target that is not
implemented at all (`--targets vhdl`) is an error from `ExportOptions`, not a
silent omission.

### What the generated module gives you

`current_for_torque(tau, rot_vel)` returns `(i_d, i_q, current_saturated,
voltage_saturated)` — the same four-tuple the Python engine returns.  Below the
band edge it is the MTPA closed form

```
i_d = (lambda - sqrt(lambda^2 + 8 (Lq - Ld) i_m^2)) / (4 (Lq - Ld))
```

and above it a fixed-count scan-and-bisect onto the voltage ellipse.  Around
that: `max_torque(rot_vel)` for the envelope, `torque(i_d, i_q)` and
`voltage(rot_vel, i_d, i_q)` for the machine, `losses()` split into
copper/iron/windage/bearing/converted/shaft/input, `dc_link_current()` for the
rail current (`1.5 (v_d i_d + v_q i_q) / V_bus`, `nan` when the bus is not a
number), `State` / `derivatives` / `step` / `integrate` / `step_current` for the
plant, and a clocked `FeedForward` as in the brushed export.

Parity with `PMSMSpec` is the acceptance test, not a smoke test: current
magnitudes agree to 5.7e-13 relative inside the envelope and 2.9e-4 at or above
it (the difference is the fixed-count search, not the physics), and
`max_torque(rot_vel)` tracks `torque_limit` to 1.5e-7.  Two bugs that parity
caught, both worth knowing if you write this law by hand:

- the closed form needs `Lq - Ld` in the denominator, which is the *opposite*
  sign to the `RELUCTANCE = Ld - Lq` constant the torque equation uses.  Get it
  wrong and `i_d` has the wrong sign: up to 6% excess current, and a torque a
  little under the limit comes back as unreachable.
- `i_d` does not flip sign with the torque.  Only `i_q` carries the direction,
  because reluctance torque follows the sign of `i_q`; the engine keeps `i_d`
  negative in both directions and so must the generated code.

The saturation flags say which constraint *produced* the answer, not which one
the answer happens to violate by rounding: a pair from the field-weakening branch
always reports `voltage_saturated=True` (it bisects onto the ellipse), and a
demand sitting exactly on the current circle may report either, depending on
where a fixed-count bisection landed.  The current magnitude is the same either
way.

### The sample-time contract is different here

The brushed limit is `0.5 * L/R`: a real pole, speed-independent.  A dq plant is
not that.  With the shaft speed held fixed, the current loop linearises to a
**complex** pole pair

```
-R (Ld + Lq) / (2 Ld Lq)  +/-  j w_e ,      w_e = pole_pairs * rot_vel
```

and RK4's region of absolute stability reaches only `|lambda * h| ~ 2.78` on the
imaginary axis.  So the stable step *shrinks as the shaft speeds up*.  For
`bldc_pod`: 6.25e-4 s at standstill, 1.79753e-4 s at base speed (1099.7 rad/s),
9.018e-5 s at twice base speed.  The generated module carries `POLE_DAMPING`,
`RK4_STABILITY_LIMIT = 1.39`, `pole_magnitude(rot_vel)`,
`max_step_time(rot_vel)` and `sub_steps(dt, rot_vel)`; `step()` raises rather
than integrating past the limit, and `integrate()` re-sizes every sub-step from
the speed it has actually reached — a subdivision fixed from the standstill
number is exactly the bug that makes a high-speed run diverge.

If you meet that before reading this page, the symptom is a speed trace that goes
`nan` a few tens of milliseconds in, and the sample just before it jumps by more
than the machine's maximum acceleration can explain.

`--sample-time` above the standstill limit prints a warning; the GUI status line
quotes the limit at base speed too, because that is the number a control period
has to survive at the top of the speed range.

### The C and Rust twins

The angle search is the same fixed-count scan-and-bisect in every language, and
the RK4 state is three channels (`rot_vel`, `i_d`, `i_q`) instead of two.  The
twins are transliterations of the generated Python, operation for operation, so
the arithmetic order matches too: `dq_step_current` keeps the subtraction order
of `step_current`, and the Rust `update()` calls `voltage_magnitude` twice with
no magnitude guard because the Python does.

C: `drive_model_dq.h` + `drive_model_dq.c`, identifiers prefixed `dq_`/`DQ_`
rather than the brushed `dt_`/`DT_`, so a dq build and a brushed build coexist in
one binary without a symbol collision.  `--c-prefix` rewrites every one of them
or none — a half-renamed C build still collides, so a partial prefix is worse
than no prefix.  Math goes through width-matched macros (`DQ_SQRT` is `sqrtf` in
an f32 build, `DQ_HYPOT` is `hypotf`), because a float build must not drift into
libm's double entry points.  C has no exceptions and no `None`, so the guards
return codes: `dq_step` gives 0 or -1 and leaves the output state untouched on
-1, `dq_integrate` gives 0 / -1 (zero stable step) / -2 (over 100000 sub-steps),
`dq_ff_init` gives 0 / -1 for a non-positive sample time, and `DQ_NAN` stands in
for `None` everywhere an argument is optional.  It compiles clean under
`-Wall -Wextra -Wpedantic -Wconversion -Werror` in both widths.

Rust: `drive_model_dq.rs`, `core` only — no `std`, no allocation, no
floating-point formatting — and a test compiles it inside a `#![no_std]` crate so
that claim stays true.  `core` has no float square root, so the module carries
its own Newton iteration (`dq_sqrt`, seeded from the exponent bits, six
iterations, accurate to 1 ulp but *not* correctly rounded); a firmware crate that
already links libm can swap that one body out.  `step` and `integrate` return
`Result<DqState, DqError>` rather than panicking or silently integrating past an
unstable step, because the speed-dependent stable step is the part of this
contract that gets broken in the field.  `DqError` carries the numbers and
implements `Display`:

```
dt 0.001 s exceeds the stable step 0.000179753 s at 1099.714798 rad/s
(pole magnitude 7732.85 1/s, electrical time constant 0.00125 s); subdivide the step
```

Parity is the acceptance test, and it is measured, not asserted: the generated C
and Rust benches print CSVs **bit-identical** to the Python bench in both modes
at f64, and a 235-row probe over derived constants, the envelope, MTPA, the
current angle, losses, the step guards, a 3000-step plant run and the whole
feed-forward trace agrees to 1.0e-15 relative (C — libm `hypot` is not CPython's
`math.hypot`, one ulp) and 8.5e-15 (Rust — its own square root).  The tests
compare at 1e-12 and say why: bit-identity across languages is not achievable
when one language supplies its own transcendental, and the ill-conditioned
envelope search amplifies one ulp into ~1e-14 legitimately.  At f32 the C and
Rust benches agree exactly with each other (plant settles 184.316879 rad/s,
feed-forward reaches 1272.965698 rad/s against the f64 184.316855 / 1272.967350).

### The bench

`bench_drive_model_dq.py` runs `plant` (default) or `feedforward` (first
argument) and prints `time_s,motor_rad_s,current_A,voltage_V`.  The current
column is the *plant* current `hypot(i_d, i_q)`, not the commanded pair:
feed-forward is open loop, so above the band edge the plant lags the reference
and settles wherever the applied voltage puts it.  That gap is the reason a real
drive closes a current loop around this law rather than trusting it.  The
feed-forward run deliberately ramps its reference to 1.5x base speed — the
default `speed_limit` is base speed, which never exercises field weakening at
all — and prints a note the first time the law saturates.

## What the export does not include

The manifest lists it, and the generated README repeats it, because a control
engineer needs to know what the model will not tell them:

- thermal limits and derating (no temperature state, no resistance drift)
- magnetic saturation, iron losses, cogging, speed-dependent efficiency maps
- ESC dead time, PWM quantisation, current-loop bandwidth, sensor delay
- gear backlash, compliance, windup — the gear is a rigid lossy ratio
- battery impedance and supply sag (`voltage_limit` is a hard rail)
- regenerative braking limits beyond the current and voltage clamps

Two more, specific to the collapse: `current_limit` is *total* at the terminals,
so with n parallel motors a real ESC limiting per motor needs `current_limit / n`
in the exported model; and the model's stiction band is `1e-6 rad/s`, which
catches "is it moving?" but is not a substitute for a backlash model you don't
have.

Brushless machines are modelled (`drivetrain/pmsm.py`), and both exporters reach
them: `export()` emits the DC/brushed-equivalent collapse — one `kv`, one `kt`,
one `R`, one `L` — with the machine's caveats in the banner, and `export --dq`
emits the dq law with MTPA and field weakening.  What neither one models yet is
the inverter in front of the machine: no dead time, no device drops, no current
loop, no angle sensor.  Back-EMF harmonics, cogging and magnetic saturation are
also still out, so a machine with a trapezoidal back-EMF is being described by a
sinusoidal model that is a good approximation and not a measurement.
