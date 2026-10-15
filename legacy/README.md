# legacy

The original flat scripts, kept as reference for the `euler` engine and for the
parameter values in `drivetrain/presets.py`. They are not imported by the
package.

Run them from this directory:

```bash
cd legacy && python3 main.py
```

Known issues in these scripts, all fixed in the `drivetrain` package:

- `Motor.dampping` — misspelled attribute (`damping` is the method name).
- `temp.py` uses `rot_intertia`, a typo for a field that does not exist on
  `Motor`; it only works because Python lets you invent attributes on a class.
- `MotorGroup` sums stall torques, damping and inertias, which implicitly scales
  `kt` and `kv` with motor count. That inflates top speed and damping. Correct
  parallel-machine equivalence: `R/n`, `L/n`, `n·J`, `kt`/`kv` unchanged.
- `Gearbox.reflect_inertia` divides by `N**2` and `output_speed` divides by `N`,
  so `ratio > 1` means speed-up, which is the opposite of what the name suggests.
  The package uses an explicit speed-ratio convention instead.
- No current limiting, no load model, no efficiency in the power-flow direction.
