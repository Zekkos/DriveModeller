# Roadmap

Status legend: `[done]` shipped and tested · `[wip]` in progress now · `[next]`
queued, not started · `[later]` idea, no design yet.

## Where the project stands

A pure-Python (numpy-only) drivetrain toolkit: `MotorGroup` (n parallel motors,
kv/kt/R/L, rotor inertia, bearing damping, static friction, no-load friction
current, current/voltage limits) → `Gearbox` (ratio + convention, efficiency,
inertia, damping) → `LoadBundle` (rotational inertia/viscous/constant torque, or
vehicle mass/rolling/drag/grade/wheel).  A brushless machine is a `PMSMSpec` -- a
dq model with `Ld`/`Lq`, flux linkage, MTPA and the voltage ellipse -- wearing
`PMSMModel`, a read-only brushed view of it, so it goes through the same
collapse. `DriveSystem` collapses that into a first- or second-order LTI system
with hard limits, and `simulate()` runs it with three interchangeable engines
(analytic, RK4, exact piecewise-linear) that `validate` cross-checks.

Entry points: `run.py` → `drivetrain.cli` (`presets`, `describe`, `run`, `sweep`,
`optimize`, `fit`, `export`, `efficiency`, `validate`, `gui`), `drivetrain.gui`
(Qt interface with a fit tab and "Export model code..." / "Export dq/FOC code..."
/ "Efficiency map (CSV)..." actions), `drivetrain.export` (C / Rust / Python /
ROS codegen for a feed-forward controller) and `drivetrain.export_dq` (dq/FOC
codegen in C, Rust and Python) and `drivetrain.dq_sim` (the same dq plant as a
library integrator, used by the dq/FOC export's tests and by the fitter) and
`drivetrain.efficiency` (loss split per operating point, efficiency map over the
(speed, torque) plane).  421 unit tests, all green
(`python -m unittest discover -s tests`), ~70 s wall clock.

## 1. Fitting measured data `[done]`

Goal (operator, m00001): *"the ability to fit the drive/motor/etc model based on
an input dataset and known values ... many possible configurations and possible
sensory inputs. E.g. only having measurements of voltage, current, and velocity
of the motor, but knowing the inertia of the drive system but maybe not the
friction of the system."*

The machinery is there: `drivetrain/fit.py` (least-squares over a
`Dataset` of time-series blocks or steady-state rows, free parameters by path or
group alias, `known=` constants, per-channel weights, identifiability analysis
via SVD, warnings, `FitResult.report()`).

What was hardened, because the operator's scenario is exactly the case
that fails quietly:

- `[done]` speed channels in rad/s (`motor_rad_s`, `output_rad_s`) so encoder
  logs do not have to be pre-scaled to rpm.
- `[done]` `inertia` / `friction` / `damping` free-parameter groups, and
  `--free all`, so "I know inertia, I do not know friction" is one flag.
- `[done]` noise-aware default weights: per-channel 1-sigma from high-frequency
  scatter instead of raw data std, so a current channel does not drown a speed
  channel.
- `[done]` search bounds widened past the what-if catalogue range (x100 at each
  end, then start-relative), so a badly guessed start does not pin a parameter on
  a bound.  `at_bound` now judges a parameter against its own scale, so a
  1e-6-scale damping is no longer reported as "pinned".
- `[done]` diagnostics for the blind spots: current-limit saturation on either
  side (the logged trace flat-topped, or the model riding its own limit);
  steady rows cannot see inertia, inductance or damping at all; a pair with
  correlation ~1 is *indistinguishable*, not merely correlated; a 9.5x (60/2pi)
  scale mismatch means rpm was logged into a rad/s column; and a parameter that
  no equation in this configuration reads (`no_load_friction_current`, inductance
  when L = 0, drag when there is no vehicle) is reported at its starting value
  with `unbounded` uncertainty instead of wherever the optimiser drifted.
- `[done]` `FitResult.by_path`, `relative_rms` (unit-free: rms as a fraction of
  the channel's own spread), `weights_basis`, `starts`, and the same numbers in
  `as_dict()`; `docs/fitting.md` and the README describe them.
- `[done]` a logged `voltage_V` is replayed zero-order-hold.  Interpolating it
  invented a ramp through every edge of a square wave whenever the solver stepped
  finer than the log, and the fit compromised `kt` by a factor of eight chasing
  that artifact (`normalized_rms` 1036 on noise-free data was the tell).
- `[done]` starting values read off the log rather than off the catalogue: kv
  from drive voltage over top speed, kt from known inertia over standstill
  acceleration, resistance from the voltage left over after back-EMF.  The report
  prints each estimate and where it came from.
- `[done]` one coarse decade rescan when the converged fit still misses the data
  by more than a third of its own variation, skipping parameters this
  configuration does not evaluate (hopping inductance off zero changes what the
  model *is*, which is not a better fit).
- `[done]` a `kt`/`kv` pair more than five-fold apart is physically impossible for
  one machine in SI units, and is now called out as structural error rather than
  left in the table.

Evidence for the operator's own scenario -- voltage, current and speed, known
inertia, unknown friction, `spinning_mass` through a 0.3 gearbox: with the rig
unsaturated the fit returns kv 0.0140, kt 0.0140, R 0.4201 against truths of
0.014 / 0.014 / 0.42, and says plainly that bearing damping and load viscosity
are one quantity in this data (r = -1.000).  With the same rig clipped at 4 A it
returns kv 0.0228 and R 0.685, wrong by 60%, and the warnings say why: the drive
rides its current limit for 61% of the trajectory, the fit is degenerate
(condition 1.5e12) and one direction cannot be resolved at all.

What is left here is polish rather than capability: the GUI fit tab exposes the
new diagnostics only as tooltips, and the rescan is deliberately conservative (it
fires only when the model misses the data badly).

## 2. Export the model as Rust and C for a control system `[done]`

Goal (operator, m00101): *"output a comprehensive rust and c implementation of
the motor model and possible whole drive model for use in a control system. My
primary intention is for use in a feed forward controller."*

Shape of the work (operator answers, m00161): **export only on request** (a
command, not a build-time sync step), **MCU is the primary target**, and a
**Python class usable from a ROS node** is also wanted.

- `[done]` `drivetrain/export.py` (2473 lines) collapses a configured
  `DriveSystem` to the same normalised form the simulator uses and emits
  self-contained source: constants baked in, no dependency on this repository,
  no dependency on each other.
- `[done]` requested, never automatic: `run.py export`, and File → **Export
  model code...** in the GUI (it asks for the control period, bounded by that
  model's own stability limit, then a directory).  Nothing anywhere else writes
  a line of generated code.
- `[done]` three sources: a preset (+ `--set`), a saved model (`--config`), or a
  fit (`--fit-json`) - and the fit's provenance and warnings are copied into
  every generated file's header and into the manifest, so a warning that two
  parameters are one quantity ships inside the C file.
- `[done]` C99 header + translation unit: `dt_scalar`, one `dt_model` struct,
  `extern const dt_model dt_drive_model`, no allocation, no recursion, no VLAs.
  libm calls are width-matched through macros (`sqrtf`/`fabsf`/`INFINITY`)
  because the plain double entry points silently widen an f32 build.
  `--name/--prefix` renames types, functions, macros and the guard together
  (all-or-nothing: a half-applied prefix leaves `dt_step` next to `rv_rover`,
  which is worse than no prefix at all).
- `[done]` Rust as a single module: `pub type Scalar`, `pub const MODEL`,
  `State`, `Command`, `FeedForward`.  No heap, no `std` in the model itself.
- `[done]` Python class with the same method names as the C functions, plus
  `drive_model_ros.py`: a ROS-free `DriveModelNodeCore` (the loop the node runs,
  testable without rclpy) and an rclpy node whose parameters cover topics,
  output type, control period, a bounded P correction and speed/accel limits.
- `[done]` feed-forward is the primary API: `i = (J*a + b*w + load(w) +
  friction(w) + tau)/kt`, `v = i*R + kv*w`, with current and voltage limits and
  stiction applied, and the inverse map for the steady part.  The forward model
  (`step`) ships too, so the same code can be a plant model or observer.
- `[done]` a clocked feed-forward helper in **all three** languages
  (`FeedForward` in Python/Rust, `dt_ff_*` in C) because the useful part is not
  the law, it is that the `a` term must be the derivative of the *rate-limited*
  reference.  The bug this caught: clamping the absolute reference instead of
  the increment pinned the reference at 0.0927 rad/s (0.268 V) for a command of
  129 rad/s.
- `[done]` parity is a test, not a claim: the C and Rust builds are compiled and
  run against the Python one to 1e-9 relative, and the exported RK4 reproduces
  `simulate(engine="rk4")` **bit for bit** in both first- and second-order form.
- `[done]` float width is a build option (`--dtype f32|f64`), rounded **once**
  before rendering so C, Rust, Python, the manifest and the README all print the
  same numbers.  Fixed-point is deliberately not generated (see Decisions).
- `[done]` the sample-time contract is written into the manifest
  (`max_step_s`, `sample_time_s`, stiction band) and into the README: 0.573 s
  for `pod_2x_direct`, 6.43 s for `spinning_mass`.  With no `--sample-time` the
  README says so plainly instead of inventing a period.

Evidence: an f32 `pod_2x_direct` build is **2401 bytes of `.text`, 0 `.data`,
0 `.bss`** (`cc -std=c99 -Os`), and the same source at f64 is 2670 bytes.  gcc
compiles it clean under `-Wall -Wextra -Wpedantic -Wconversion -Werror`; rustc
compiles the generated module in 0.028 s.  Two parity traps found while writing
the tests, both real and both now documented: `simulate(duration=3.0, dt=0.001)`
yields a 3000-point grid, i.e. **2999** steps, and the RK4 engine starts the
current at the algebraic value the applied voltage implies (starting it at zero
costs ~0.14 rad/s of permanent offset once inductance is non-zero).

What is left here is polish: the GUI has no tests at all, so the export action
is verified by an offscreen smoke run rather than a unit test; there is no ARM
cross-compiler on this box, so the footprint numbers are host numbers; and the
generated ROS node has never been run against a real rclpy install (its core is
tested without ROS on purpose).

## 3. Brushless motors (BLDC / PMSM) `[done]`

The machine model exists: `drivetrain/pmsm.py` holds `PMSMSpec`, a dq model in
the amplitude-invariant Park frame (`v_d = R i_d + Ld di_d/dt - w_e Lq i_q`,
`v_q = R i_q + Lq di_q/dt + w_e (Ld i_d + lambda)`,
`T = 1.5 p (lambda i_q + (Ld - Lq) i_d i_q)`) with MTPA, the voltage ellipse,
field weakening, both saturation flags, reluctance torque, iron and windage
loss, and a magnet-temperature coefficient.  It is validated against analytic
no-load and locked-rotor cases, against the brushed model where the two overlap,
against a brute-force search for the minimum-current current angle, and for
four-quadrant symmetry (70 tests in `tests/test_pmsm.py`).

It is wired into the rest of the toolkit through `PMSMModel`, which presents the
brushed interface as read-only properties of the spec, so `MotorGroup`,
`DriveSystem`, `simulate`, `sweep`, `fit`, the GUI and the export all work
unchanged (`tests/test_pmsm_system.py`).  The fitter keeps speaking the brushed
vocabulary and re-points it at the constants a brushless system can set; `Ld`,
iron loss, windage and flux-temperature drift turn out to be **not identifiable
from a DC-side log at all** under that collapse, which answers the question this
section used to leave open.  A log in the rotor frame lifts that ban on `Ld` -- see the three-phase bullet
below.

The pieces, and what is still idea-stage:

- The dq export: `drivetrain/export_dq.py` writes
  `drive_model_dq.py` plus a bench, a manifest and a README (schema
  `drive-model-dq-1`) carrying MTPA, a fixed-count field-weakening search, both
  saturation flags, the three-channel RK4 state and a clocked feed-forward,
  reachable from `run.py export --dq` and the GUI.  It is tested against
  `PMSMSpec`, not against itself: current magnitudes agree to 5.7e-13 relative
  inside the envelope and 2.9e-4 at or above it (the gap is the fixed-count
  search, not the physics), and the bench is deliberately driven to 1.5x base
  speed because the default speed limit never exercises field weakening.  Two of
  its bugs were found by disagreeing with the engine -- the MTPA denominator needs
  `Lq - Ld`, the opposite sign to the `RELUCTANCE` constant the torque equation
  uses, and `i_d` must not flip sign with the torque -- which are exactly the two
  ways this law gets written wrong by hand.
- The same law in C and Rust: `drive_model_dq.{h,c,rs}` and a bench per language,
  transliterated from the generated Python operation for operation, guards and
  all.  C returns codes (`dq_step` 0/-1, `dq_integrate` 0/-1/-2, `dq_ff_init`
  0/-1) and compiles clean under `-Wall -Wextra -Wpedantic -Wconversion -Werror`
  in both widths; Rust is `core`-only with its own square root, so it builds
  inside a `#![no_std]` crate, and its `step`/`integrate` return
  `Result<DqState, DqError>`.  Cross-language parity is measured, not assumed:
  the C and Rust benches print CSVs bit-identical to the Python bench at f64 in
  both modes, and a 235-row probe agrees to 1.0e-15 (C) and 8.5e-15 (Rust)
  relative.  The f32 build is the one that found the bug class -- a tolerance
  macro used inside `dq_max_step_time` promoted to double and tripped
  `-Wconversion`.
- Three-phase datasets in `fit`: `drivetrain/dq_sim.py` is the dq plant as a
  library integrator (state, RK4 with the speed-dependent stable step, Park and
  inverse Park, `simulate_dq`), and `drivetrain/fit.py` reads dq and phase
  channels -- `current_d_A`, `current_q_A`, `voltage_d_V`, `voltage_q_V`,
  `rotor_angle_rad`, and the three-phase `current_a_A` / `voltage_a_V` family,
  Park-transformed at load time from the **electrical** angle.  That is what
  makes `Ld` and `Lq` separable at all: they appear in different voltage
  equations, so the inertness report is channel-aware
  (`structural_inert(system, path, channels=...)`) and calls `inductance_d` live
  only when both dq currents and `voltage_q_V` are measured.  Settled dq rows
  recover resistance, flux linkage, `Ld` and `Lq` to 0.03-1.6 % at condition
  16.5, and the same operating points written as three-phase columns give the
  same answer to the fourth digit.  A transient dq log is replayed through the dq
  integrator and replays a clean log to exactly zero residual; sampling rate
  dominates excitation shape there (6 samples per electrical time constant gives
  12.7 % worst error, 15 samples gives 1.5 %).
- Efficiency as a map over (speed, torque) with the losses split by kind:
  `drivetrain/efficiency.py` bills one operating point or a whole grid without
  integrating anything -- 33 x 25 cells is a few hundred algebraic solves -- and
  cells outside the machine's envelope stay `NaN` rather than being interpolated
  into a neighbour.  Both axes are motor-side, because the envelope that decides
  what is possible belongs to the machine; `torque_is="shaft"` rebills the
  machine's own internal losses on top.  Efficiency is useful-over-absorbed per
  quadrant, and `None` -- not zero -- where no ratio exists (standstill, or
  losses larger than the developed power).  The books close by construction
  (`power_in - total_losses - power_out` is 2.0e-10 on the dq machine and 1.1e-13
  on the brushed collapse), which is what `residual_W` is for: on a brushed model
  it is exactly the `(kv - kt) i w` term, so a datasheet pair that disagrees
  surfaces as a number instead of being smoothed over.  Magnet loss is `None`
  rather than `0.0` because nothing in the package models rotor loss, and every
  report ends with a `not modelled on this map:` line.  Two real bugs fell out of
  building it: `PMSMSpec.efficiency_at` subtracted bearing and windage loss but
  not iron loss (0.874 reported where the honest number was 0.561 on a machine
  with `core_loss_coefficient` switched on), and the field-weakening flag was the
  sign of `i_d`, which on a salient machine is negative at any torque because
  MTPA collects reluctance torque there.  `run.py efficiency`, a GUI "Efficiency
  map (CSV)..." action, 51 tests in `tests/test_efficiency.py`, and
  `docs/efficiency.md`.
- `[later]` Waveform detail the dq model still averages away: trapezoidal
  back-EMF, cogging, deadtime, sensorless estimation.  Secondary to the dq law
  and its feed-forward terms, per the FOC decision below, and still idea-stage --
  no design beyond "the dq model would have to stop averaging".

## Decisions

- Export is **on request only** -- a command the operator runs, not a generated
  file kept in sync at build time. The generated code is a snapshot of the model
  as configured at that moment; keeping it in sync is the operator's call.
- Primary export target is an **MCU** (so: no_std Rust, C99, no allocation,
  fixed sample-time contract). A **Python class for a ROS node** is the second
  target and shares the same generated parameter set.
- BLDC is driven with **FOC** in the real system, so the dq/PMSM formulation is
  the primary model, not trapezoidal BLDC commutation.
- Export emits **float only** (`--dtype f32|f64`), rounded once before rendering
  so every language and the manifest print the same numbers.  Fixed-point is not
  a flag on this generator: the acceptance test is bit-identity with the Python
  engine, and a fixed-point build needs its own error model and its own tests.
  If an FPU-less target becomes the real constraint, that is a separate mode.
- The C symbol prefix is **all or nothing**: `--prefix` renames the types, the
  functions, the macros and the guard, because a build with `rv_rover` next to
  `dt_step` still collides in one binary and looks namespaced while being one.
- Every language gets the **clocked** feed-forward helper, not just the law.
  The law is one line; what goes wrong in the field is feeding the acceleration
  term the derivative of an unlimited reference.
- The ROS node wraps a **ROS-free core** (`DriveModelNodeCore`): the control loop
  is the part worth testing, and it must be testable without an rclpy install.
- A brushless machine reaches the existing pipeline through a **read-only
  brushed view of the dq spec** (`PMSMModel` wrapping `PMSMSpec`), not by
  duck-typing the spec into `MotorGroup`.  The names collide (`torque_at(i_d,
  i_q)` vs `torque_at(voltage, speed)`, `current` vs `current_d/current_q`,
  `inductance` vs `inductance_d`/`inductance_q`) and, worse, `kt`/`kv` would
  become two free numbers that can drift apart; as derived properties they
  cannot.
- The collapse is **`i_d = 0`**, and it is exact while the machine is
  current-limited and the voltage ellipse is slack -- one-to-one, not an
  approximation.  It saturates at base speed exactly, so a brushless drivetrain
  built this way cannot represent field weakening.  It is also optimistic about
  how far full current reaches, because a real drive spends part of its voltage
  budget on the cross-coupling term `w_e Lq i_q`.  The band edge is **reported**
  (`constant_torque_speed()`, `current_ceiling(w)`, `collapse_margin()`, and the
  export notes) rather than hidden, and the collapse is deliberately **not** made
  speed-dependent: `MotorGroup` and `simulate` apply the current limit as a
  constant clamp with no hook to hang a curve on, and a model that quietly
  changes its own limits is harder to debug than one that states where it stops
  being true.
- Fitting a brushless system **keeps the brushed vocabulary** (`kv`, `kt`,
  `resistance`, `inductance`, ...) and re-points it at `group.motor.spec.*`,
  converting the values on the way (`lambda = kv / p`, `lambda = kt / 1.5p`).
  A second vocabulary for the same physical quantities would be a worse
  interface than the translation -- and `kt` and `kv` are one flux linkage there.
- The dq constants the collapse never evaluates (`Ld`, iron loss, windage,
  flux-temperature drift, the DC bus) are reported as **structurally inert**
  rather than fitted to noise.  Naming one explicitly still fits it, on the
  user's terms.
- The dq/FOC export is a **second generator** (`drivetrain/export_dq.py`), not a
  flag on the brushed one.  The two answer different questions -- the brushed
  export inverts a one-constant law, the dq export inverts a current angle -- and
  folding them together would mean a module whose constants mean different things
  depending on a switch.  They share a directory instead: `drive_model.py` and
  `drive_model_dq.py` side by side, so the two can be diffed.
- The dq export emits **all three languages**, and the manifest still records
  `targets: {requested, emitted, not_emitted}`: a target can be requested before
  this generator can render it, and the record is what stops a silent omission
  from reading as a successful build.  Asking for a target that does not exist at
  all (`--targets vhdl`) is an `ExportOptions` error instead.
- The dq C identifiers are `dq_`/`DQ_`, not the brushed `dt_`/`DT_`, so a dq build
  and a brushed build link into one binary without colliding.  `--c-prefix`
  rewrites every one of them or none -- a half-renamed C build still collides on
  the identifiers it missed, which is worse than no prefix at all.  Rust needs no
  such pass: a module is namespaced by the `mod` item that pulls it in.
- The generated Rust carries its **own square root** (`dq_sqrt` in the dq module,
  `sqrt` in the brushed one, both Newton iterations seeded from the exponent bits
  and widening internally so one body serves f32 and f64) because `core` has no
  float square root.  The brushed banner used to claim "`core` only ... drops
  straight into a `#![no_std]` firmware crate" while calling `.sqrt()`: a promise
  that fails at the first firmware build.  Both modules now emit a true banner,
  and each has a test that compiles it inside a `#![no_std]` crate.  A firmware
  crate that already links libm swaps that one body out.
- Cross-language parity is asserted at **1e-12 relative, not bit-identity**.  The
  generated C and Rust benches print CSVs bit-identical to the Python bench at
  f64 -- same arithmetic, same order -- but a wider probe grid differs by 1.0e-15
  (C, where `dq_hypot` is libm's `hypot` and CPython has its own) and 8.5e-15
  (Rust, where it is built from the module's `dq_sqrt`, accurate to 1 ulp but not
  correctly rounded).  The ill-conditioned envelope search turns one ulp of
  square root into 1e-14 of current angle legitimately, so demanding bit-identity
  across languages would mean giving up the fixed-count searches that are the
  whole point of the export.
- The dq **tolerances** (`STICTION_BAND`, `FEASIBILITY_SLACK`, `ENVELOPE_SLACK`,
  `SALIENCE_FLOOR`, `RK4_STABILITY_LIMIT`) print at double precision in all three
  languages and are rounded at the point of use, because they are thresholds in
  comparisons rather than machine constants; an f32 build that rounded them at
  declaration would disagree with the Python about where the envelope sits.
- The dq **stable step shrinks with speed**, and the generated module says so.
  With the shaft held fixed the dq current loop linearises to a complex pair at
  `-R(Ld+Lq)/(2 Ld Lq) +/- j w_e`, and RK4's absolute-stability region reaches
  only `|lambda h| ~ 2.78` on the imaginary axis, so `max_step_time(rot_vel)` is
  the contract -- 6.25e-4 s at standstill for `bldc_pod` but 1.79753e-4 s at base
  speed.  The brushed `0.5 L/R` law is correct there (real, speed-independent
  pole) and wrong here.  `step()` raises past the limit and `integrate()`
  re-sizes every sub-step from the speed it has reached: silently subdividing a
  period using the standstill number is the failure mode this is meant to
  prevent.
- A rotor-frame log is read in the **electrical** angle, and three-phase columns
  are Park-transformed **at load time**.  A mechanical angle is refused rather
  than converted, because converting it multiplies by `pole_pairs` -- one of the
  numbers a fit may be adjusting -- so the log would be its own evidence for a
  parameter the fit is looking for; when both angles are logged the fit reports
  the median `d(theta_e)/d(theta)` as the pole count it observed, which turns the
  refusal into a check the operator can run.  Phase columns become dq channels
  during load and are then consumed, so the residual function speaks one
  vocabulary.  All three phases of a kind are required: reconstructing a missing
  phase from `a + b + c = 0` would assume the balanced machine the model already
  assumes, and the fit could no longer tell a broken sensor from a real neutral
  current.  A measured zero sequence is reported and dropped.
- Identifiability is **channel-dependent, not model-dependent**:
  `structural_inert(system, path, channels=None)`.  Under a DC-side log `Ld` is
  inert; with dq currents and the matching dq voltage it is live.  Omitting
  `channels` keeps the conservative brushed answer, which is what the CLI and GUI
  do today.  Iron loss, windage and flux-temperature drift stay inert under
  either vocabulary, because the dq plant's mechanical side has no such term --
  better data does not make a missing term fittable.
- The dq fit **replays the logged `v_d`/`v_q`** rather than prescribing them, and
  refuses a transient dq block that has neither: a prescribed constant pair
  constrains resistance and flux and says nothing about the inductances.  Each
  voltage is held across its integration step, the same convention
  `dq_sim.simulate_dq` uses, so a log replayed with the constants that generated
  it leaves exactly zero residual -- the property `tests/test_fit_dq.py` pins, and
  the reason a coarse log's large residuals are real parameter error rather than
  an alignment artifact.
- Efficiency is billed **per quadrant, useful side over absorbed side**, and is
  `None` where no ratio exists.  One signed formula either hides
  non-regenerative braking behind a flattering number or turns standstill into a
  division by zero, and a zero there reads as "0 % efficient" rather than "there
  is no output to bill".
- The loss split **closes the books by construction**, with `residual_W` as the
  catch-all, rather than listing only the terms the model happens to model.  The
  identity `power_in - total_losses - power_out = 0` is what makes the split
  checkable, and on a brushed model the residual is exactly `(kv - kt) i w`.
- Magnet and rotor loss are **absent, and marked absent**: `magnet_W` is `None`,
  never `0.0`, and reports end with a `not modelled on this map:` line.  A zero
  would claim the rotor is lossless; `None` says the question was not answered.
- The efficiency map is **steady state and motor-side**.  It is not a simulation:
  no start-up current, no thermal time constants, no controller ripple or PWM
  loss.  Ground-side axes would read better for a vehicle, but the envelope --
  current circle, voltage ellipse, regenerative limit -- is the machine's and
  lives ahead of the gearbox.

## Open questions

- Which MCU class matters (Cortex-M4F with an FPU, or M0+ without)? That decides
  whether f32 is safe for the integrator and how the current-limit arithmetic is
  ordered.
- ~~Does the exported feed-forward need the inverse only, or also a forward
  plant model?~~ Both ship, and the inverse is the deliverable: `feedforward`
  (and the clocked helpers) are the primary API, `step` is there so the same
  code can serve as a plant model or observer on the target.
- For FOC: sensor-based or sensorless? Sensorless changes what the model has to
  predict (back-EMF observability, inductance saturation) and what the export
  needs to expose.
