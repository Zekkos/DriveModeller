# Efficiency: one point, or the whole map

`drivetrain.efficiency` answers a different question from `simulate()`.  A
simulation tells you what the drivetrain does over time; this module tells you
where the power goes at a stated operating point, and then does it again over a
grid of them.  It never integrates, so it is cheap: a 33 × 25 map is a few
hundred algebraic solves, not a time-domain run.

```python
from drivetrain import presets
from drivetrain.efficiency import operating_point, efficiency_map

pod = presets.build("bldc_pod")

point = operating_point(pod, 700.0, 0.2)      # motor rad/s, motor N*m
point.efficiency                              # 0.926791...
point.losses.copper_W                         # 10.0015
point.losses.share()["copper"]                # 0.9107...

emap = efficiency_map(pod, n_speeds=17, n_torques=13)
emap.best("motoring")                         # dict, or None if nothing is feasible
print(emap.report())
```

---

## The two conventions that trip people up

**Both axes are motor-side.**  Speed is rad/s at the motor shaft, torque is N*m
at the motor shaft, and the current is the group current (per-machine figures are
carried alongside).  That is deliberate: the envelope that decides what is
possible is the machine's, and it lives on the motor side of the gearbox.  To
bill the gear stage too, pass `include_gearbox=True`, which adds a `gear_W`
column and lowers `power_out_W` accordingly.

**Torque is the developed torque by default.**  `torque_Nm` is what the machine
produces electromagnetically, before its own bearing, windage and iron losses are
paid for.  Pass `torque_is="shaft"` if you mean the torque measured at the
output flange; the module then bills the machine's internal losses on top of it
and reports the developed value in `developed_W`.

Signs follow the motor, and the four quadrants are treated honestly:

| speed | torque | what it is | `power_in_W` | `power_out_W` |
|---|---|---|---|---|
| + | + | driving forward | + | + |
| − | − | driving in reverse | + | + |
| + | − | braking while moving forward | − (returned) | + |
| − | + | braking while reversing | − | + |

A torque of the same sign as the speed is driving, so `(−700, −0.2)` mirrors
`(700, 0.2)` exactly rather than negating it.  Braking is billed with the shaft
as the absorbed side and the terminals as the useful side, which is why the
braking efficiency is `power_in / power_out` and not its reciprocal.

## Efficiency is useful over absorbed

`_bill` computes it per quadrant, so the number is always the smaller magnitude
over the larger one and can never exceed 1.  It is `None` — not zero, not a
negative number — when there is no ratio to quote:

- at standstill, where the shaft delivers nothing;
- when the machine's own losses eat the whole developed power, so `power_out_W`
  is not positive (`why` says *"the machine's own losses consume the whole
  developed power"*);
- when the electrical side draws no power at a driving point.

Braking that is *not* regenerative gets its own note: if the drive still
supplies power while the shaft is being absorbed, the shaft's power is burned in
the winding rather than returned, and the module says so instead of reporting a
cheerful ratio.

## The books close, by construction

For every point, feasible or not:

```
power_in_W − losses.total_W − power_out_W ≈ 0
```

to 2.0e-10 on the dq machine and 1.1e-13 on the brushed collapse in the test
suite.  That identity is the reason the `residual_W` column exists.  The dq
model closes exactly, because `input_power` is `1.5 (v_d i_d + v_q i_q)` and the
torque constant is `1.5 p λ` while the back-EMF constant is `p λ` — the
amplitude-invariant Park convention, not a loss.  The brushed collapse does not
close on its own whenever `kt ≠ kv`, and the difference `(kv − kt) · i · w` is
billed as a residual loss rather than hidden.  On `pod_4x_geared` that residual
is 0.5% of the loss budget; if it were larger, that would be a sign the datasheet
pairing of `kt` and `kv` is inconsistent, and the map surfaces it instead of
smoothing it over.

## The loss columns

| column | where it comes from |
|---|---|
| `copper_W` | `I²R`, per machine × count (dq: `1.5 R (i_d² + i_q²)`) |
| `iron_W` | `core_loss_coefficient · p² · w²` — **zero on a brushed model**, which has no iron term |
| `windage_W` | `windage_coefficient · w²` — zero on a brushed model, which lumps it into `bearing_damping` |
| `bearing_W` | `bearing_damping_eff · w²` |
| `friction_W` | Coulomb stiction: `static_friction_eff · \|w\|` on the brushed collapse, **zero on a brushless machine while it is turning** |
| `gear_W` | `(1 − η)·\|shaft power\| + damping · w²`, only with `include_gearbox=True` |
| `residual_W` | whatever the model cannot account for, so the books close |
| `magnet_W` | **always `None`** — see below |

The stiction split is a real difference between the two engines, not a
rounding artefact.  `PMSMSpec.loss_torque()` charges static friction only at
standstill, where it is a holding torque and dissipates nothing; the brushed
collapse charges it at every speed, where it is a standing loss.  A brushless
map therefore has an empty friction column while turning, and a brushed one does
not.

`magnet_W` is `None` rather than `0.0` because nothing in this package models
rotor or magnet loss.  A zero would claim the rotor is lossless; `None` says the
question was not answered.  Every report ends with a `not modelled on this map:`
line, which lists magnet loss always and iron loss additionally for a brushed
model.

## Reading a map

```
$ python3 run.py efficiency --preset bldc_pod --map --n-speeds 9 --n-torques 9 --csv /tmp/effmap.csv
efficiency map: PMSMModel x2
axes: motor speed [rad/s] x motor torque [N*m] (developed torque, gear stage not billed)
grid: 9 x 9 = 81 cells, 50 inside the envelope
best motoring: 1099.7148 rad/s (10501.5 rpm) at 0.198374 N*m -> 94.09%  (229.3 W in, 215.7 W out)
    losses: copper 82.1%  bearing 17.9%
best braking: 1099.7148 rad/s (10501.5 rpm) at -0.198374 N*m -> 94.37%  (-208.2 W in, -220.6 W out)
...
```

Cells outside the machine's envelope are `NaN` in every float matrix and `False`
in `feasible`, and they are never interpolated into a neighbour's value.  The
envelope is `count · PMSMSpec.torque_limit(speed)` — the current circle below the
band edge, the voltage ellipse above it, and the wider regenerative envelope when
the torque opposes the motion.

`ridge()` and `ridge_torque()` give the per-speed optimum, which is the shape
worth plotting: on `bldc_pod` the driving ridge runs 84.5% at 2625 rpm, 94.1% at
10501 rpm, then falls back to 87.5% at 21003 rpm as the field-weakening current
takes over.  `envelope()` returns the drive and brake torque limits per speed
with the brake side negative.

When the speed axis is not given, it is derived rather than guessed: a brushless
machine uses twice its base speed (1099.71 rad/s → 2199.43 rad/s), because with
`i_d` allowed down to `−λ/Ld` the voltage requirement stops rising with speed and
a naive "max speed" would be 14949.6 rad/s of unusable axis; a brushed model uses
the collapse's no-load speed at the system voltage limit.  The note in the report
says which was used.

## Field weakening is a flag, not a sign

`field_weakening` is true only when the solution sits on the voltage ellipse
*and* the d axis is negative.  It is not the sign of `i_d`: on a salient machine
MTPA drives the d axis negative at any torque to collect reluctance torque, which
is not weakening.  At 223.9 rad/s on `bldc_pod` a near-limit torque runs with
`i_d = −4.74 A` and `field_weakening False`; at 895.6 rad/s a near-limit torque
runs at `i_d = −7.58 A` and the flag is true.  Above the rail speed (1099.71
rad/s here) even a light torque needs weakening — 5% of the envelope at 2500
rad/s still draws 6.76 A per machine, most of it demagnetising, which is why
weakening costs current rather than saving it.

## Command line

```bash
# one point
python3 run.py efficiency --preset bldc_pod --speed 700 --torque 0.2
# 6684.5 rpm    0.2000 N*m   eff  92.68%   in   150.002 W   out   139.020 W
# losses [W]: copper 10.002  bearing 0.980

# the whole map, with the gear stage billed, written as CSV
python3 run.py efficiency --preset pod_4x_geared --map --include-gearbox \
    --n-speeds 7 --n-torques 7 --csv effmap.csv

# motoring only, torque read as shaft torque, machine JSON
python3 run.py efficiency --preset bldc_pod --map --motoring-only \
    --torque-is shaft --json
```

`--speed`/`--torque` for one point, `--map` for the grid, `--speed-top` and
`--torque-top` to override the auto axes, `--csv` to write it.  An infeasible
point exits 0 and says why (`"driving envelope at this speed is 0.2975 N*m
(current limit 20.000 A, voltage limit 13.856 V)"`) rather than raising: asking
for a point the machine cannot hold is a normal thing to do to a map.  Give
neither `--speed` nor `--torque` without `--map` and the CLI exits 2 with the
usage line.

The CSV has one row per cell — `speed_rad_s, speed_rpm, torque_Nm, feasible,
efficiency, power_in_W, power_out_W, developed_W, shaft_power_W, current_A,
current_per_motor_A, voltage_V, dc_link_current_A`, the seven loss columns,
`magnet_W`, `total_W`, `why` — with empty fields where a value is undefined, so
it loads into a spreadsheet without a `NaN` argument about it.

The GUI has a matching **File → Efficiency map (CSV)...** action, which writes
`efficiency_map.csv` and `efficiency_map.txt` into a directory you pick and
reports the best motoring point in the status bar.

## Library API

- `operating_point(system, speed_rad_s, torque_Nm, *, torque_is="developed", include_gearbox=False)` → `OperatingPoint` with `power_in_W`, `power_out_W`, `developed_W`, `shaft_power_W`, `current_A`, `current_per_motor_A`, `voltage_V`, `dc_link_current_A`, `efficiency`, `losses`, `current_limited`, `voltage_limited`, `field_weakening`, `why`, and `.report()`.
- `efficiency_map(system, *, speeds=None, torques=None, n_speeds=33, n_torques=25, speed_top=None, torque_top=None, include_regen=True, torque_is="developed", include_gearbox=False)` → `EfficiencyMap` with `.shape`, `.matrix(key)`, `.efficiency`, `.feasible`, `.n_feasible()`, `.points_at(i)`, `.best(quadrant)`, `.ridge(quadrant)`, `.ridge_torque(quadrant)`, `.envelope()`, `.rows()`, `.write_csv(path)`, `.as_dict()`, `.report()`.
- `LossSplit(copper_W, iron_W, windage_W, bearing_W, friction_W, gear_W, residual_W, magnet_W=None)` with `.total_W`, `.terms()`, `.share()`, `.as_dict()`.

`as_dict()` renders non-finite cells as `null`, so `--json` output is strict JSON;
the numpy matrices keep `NaN`, because that is what makes a plot show a hole.

## What this does not tell you

The map is steady state.  It says nothing about the current a start-up draws,
nothing about thermal time constants, and nothing about the loss a real
controller adds through commutation ripple, dead time or PWM.  Iron loss is the
`ke · p² · w²` term of the machine model, which is a speed-squared proxy, not an
FEM result.  Magnet and rotor loss are not modelled at all, so a map on a machine
with a big rotor loss is optimistic in exactly one column and the report names
that column.

## Tests

`tests/test_efficiency.py` (51 tests) pins the books-closing identity in every
quadrant for both engines, the `efficiency_at` parity between a map point and
`PMSMSpec.efficiency_at`, the iron-loss term that `efficiency_at` used to omit,
the standstill and envelope refusals, the field-weakening flag against MTPA, the
auto speed axes, the CSV round trip and the CLI paths.
