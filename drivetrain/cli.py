"""Command-line interface: ``python3 -m drivetrain <command>``.

Subcommands
-----------
``presets``   list presets, motors and wheels
``describe``  print the derived quantities for a configuration
``run``       time-domain simulation, text summary or CSV
``sweep``     parameter sweep with a metrics table
``optimize``  find the best value of one parameter for an objective
``fit``       fit a model to measured data, keeping known constants fixed
``export``    emit the model as standalone C, Rust and Python for a controller
``gui``       launch the interactive Qt/Matplotlib interface (via qtpy)
``validate``  cross-check the solvers against each other and the analytic model

Common options ``--preset``, ``--config``, ``--set path=value`` and ``--json`` work
everywhere, so any command can be scripted against an arbitrary configuration.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as _mp
import sys

import numpy as np

from . import presets
from .exceptions import DrivetrainError
from .fit import FREE_ALIASES, structural_inert
from .motor import rad_s_to_rpm
from .simulate import EXCITATIONS, simulate, voltage_signal
from .sweep import find_gear_ratio, optimum, sweep, time_to_speed
from .system import DriveSystem


def parse_overrides(items: list[str] | None) -> dict[str, str]:
    """Turn ``--set a.b=2`` strings into a dict."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise DrivetrainError(f"--set expects path=value, got {item!r}")
        path, raw = item.split("=", 1)
        out[path.strip()] = raw.strip()
    return out


def coerce(raw: str):
    """Best-effort string -> number/bool conversion for ``--set`` values."""
    low = raw.lower()
    if low in ("inf", "infinity", "none", "unlimited"):
        return np.inf
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(raw) if raw.lstrip("-").isdigit() else float(raw)
    except ValueError:
        return raw


def build_system(args) -> DriveSystem:
    """Assemble a system from ``--config`` and/or ``--preset`` plus ``--set`` overrides."""
    config = getattr(args, "config", None)
    if config:
        system = presets.load(config)
    else:
        system = presets.build(getattr(args, "preset", "legacy"))
    overrides = parse_overrides(getattr(args, "set", None))
    for path, raw in overrides.items():
        system = _set(system, path, coerce(raw))
    if getattr(args, "ratio_classical", False) and "gearbox.ratio" in overrides:
        # The user typed a classical "N:1 reduction" number; the model's native
        # convention is speed ratio (out/in), so invert it once, here at the edge.
        from dataclasses import replace

        from .gearbox import RatioConvention

        system = replace(system, gearbox=replace(
            system.gearbox,
            ratio=1.0 / system.gearbox.ratio,
            convention=RatioConvention.SPEED))
    return system


def _source_label(args) -> str:
    """Name the model a command started from, for headers."""
    config = getattr(args, "config", None)
    if config:
        return f"model {config}"
    return f"preset {getattr(args, 'preset', 'legacy')}"


def _set(system: DriveSystem, path: str, value):
    from .sweep import set_path

    return set_path(system, path, value)


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------
def cmd_presets(args) -> int:
    print("Presets:")
    for name in presets.preset_names():
        system = presets.build(name)
        d = system.describe()
        print(f"  {name:16s} {d['motors']} motor(s), ratio {system.gearbox.ratio:g}, "
              f"steady {d['steady_rpm_at_limit']:.0f} rpm")
    print("\nMotors:")
    for name, motor in presets.MOTORS.items():
        print(f"  {name:16s} kv={motor.kv:g} kt={motor.kt:g} R={motor.resistance:g} "
              f"J={motor.rotor_inertia:g} limit={motor.max_current:g} A")
    print("\nWheels:")
    for name, wheel in presets.WHEELS.items():
        print(f"  {name:16s} radius={wheel.radius:g} m  mass={wheel.mass:g} kg")
    return 0


def cmd_describe(args) -> int:
    system = build_system(args)
    d = system.describe()
    if args.json:
        print(json.dumps(d, indent=2, default=str))
        return 0
    width = max(len(k) for k in d) + 1
    print(_source_label(args))
    for k, v in d.items():
        if isinstance(v, float) and not np.isfinite(v):
            v = "inf" if v > 0 else "-inf"
        if isinstance(v, float):
            print(f"  {k:<{width}} {v:.6g}")
        else:
            print(f"  {k:<{width}} {v}")
    print("\neditable parameters (--set path=value):")
    for entry in presets.field_catalog(system):
        print(f"  {entry['path']:<40} {entry['value']:<12g} {entry['label']}"
              f"{' [' + entry['unit'] + ']' if entry['unit'] else ''}")
    return 0


def cmd_run(args) -> int:
    system = build_system(args)
    v = voltage_signal(args.excitation, system, args.duration,
                       period=args.period, ramp_time=args.ramp)
    result = simulate(system, v, args.duration, dt=args.dt, engine=args.engine)

    if args.csv:
        ch = result.channels()
        keys = list(ch)
        with open(args.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in zip(*(ch[k] for k in keys)):
                fh.write(",".join(f"{x:.6g}" for x in row) + "\n")
        print(f"wrote {args.csv} ({len(ch['time'])} rows)")
        return 0

    if args.json:
        print(json.dumps({"describe": system.describe(),
                          "metrics": result.metrics(),
                          "peak": result.peak(),
                          "final": result.final()}, indent=2, default=str))
        return 0

    d = system.describe()
    m = result.metrics()
    p = result.peak()
    f = result.final()
    print(f"{_source_label(args)}  [{args.engine}, {args.excitation}, {args.duration:g} s]")
    print(f"  steady speed   {rad_s_to_rpm(system.steady_speed(system.voltage_limit)):9.1f} rpm"
          + (f"   {system.steady_ground_speed(system.voltage_limit) * 3.6:7.2f} km/h"
             if system.wheel else ""))
    print(f"  no-load speed  {rad_s_to_rpm(system.no_load_speed):9.1f} rpm"
          f"   mode {d['steady_mode']}")
    print(f"  stall torque   {system.stall_torque * system.voltage_limit:9.3f} N*m"
          f"   mech tau {d['mech_time_constant']:.4f} s")
    print(f"  rise / settle  {m.get('rise_time_s', float('nan')):9.3f} s"
          f"   {m.get('settle_time_s', float('nan')):.3f} s"
          f"   overshoot {m.get('overshoot_pct', float('nan')):.1f}%")
    print(f"  peak current   {p['peak_current_A']:9.2f} A total"
          f"   {p['peak_current_per_motor_A']:.2f} A each")
    print(f"  peak power in  {p['peak_power_in_W']:9.1f} W"
          f"   peak eff {p['peak_efficiency']:.3f}")
    print(f"  final          {f['motor_rpm']:9.1f} rpm"
          f"   {f['speed_kmh']:.2f} km/h   distance {f['distance_m']:.2f} m")
    return 0


def cmd_sweep(args) -> int:
    system = build_system(args)
    values = np.linspace(args.min, args.max, args.points)
    res = sweep(system, args.param, list(values), voltage=system.voltage_limit,
                duration=args.duration, engine=args.engine, workers=args.jobs)
    cols = [args.param, args.metric, "peak_current_A", "rise_time_s", "steady_kmh"] \
        if "steady_kmh" in res.columns else [args.param, args.metric, "peak_current_A", "rise_time_s"]
    print(res.pretty(cols, max_rows=args.show, sort_by=args.metric,
                     maximize=not args.minimize))
    best = res.best(args.metric, maximize=not args.minimize)
    print(f"\nbest {args.metric}: {float(best[args.metric]):.6g} "
          f"at {args.param} = {float(best[args.param]):.6g}")
    return 0


def cmd_optimize(args) -> int:
    system = build_system(args)

    if args.param == "gearbox.ratio" and (args.target is not None or args.target_rpm is not None):
        ratio = find_gear_ratio(
            system,
            target_ground_speed=args.target,
            target_output_rpm=args.target_rpm,
            voltage=system.voltage_limit,
            bounds=(args.min, args.max),
        )
        from .sweep import set_path

        best = set_path(system, "gearbox.ratio", ratio)
        target = args.target if args.target is not None else 1e9
        t = time_to_speed(best, target, voltage=best.voltage_limit,
                          duration=args.duration, engine=args.engine,
                          on="ground" if best.wheel else "motor", within=0.02)
        print(f"best gear ratio (speed ratio out/in): {ratio:.4f}")
        print(f"  = {1.0 / ratio:.2f}:1 reduction")
        print(f"  steady motor speed {rad_s_to_rpm(best.steady_speed(best.voltage_limit)):.0f} rpm")
        if best.wheel:
            print(f"  steady ground speed {best.steady_ground_speed(best.voltage_limit) * 3.6:.2f} km/h")
        if np.isfinite(t):
            print(f"  time to within 2% of target: {t:.3f} s")
        elif args.target is not None and best.wheel:
            # find_gear_ratio tunes the *steady* speed onto the target, so the
            # target IS the equilibrium and is only approached asymptotically.
            steady = best.steady_ground_speed(best.voltage_limit)
            if abs(steady - args.target) <= 0.02 * abs(args.target):
                print(f"  target equals the steady-state speed: reached only "
                      f"asymptotically (settle ~{best.settle_time:.1f} s)")
            else:
                print(f"  target never reached in {args.duration:g} s "
                      f"(steady {steady * 3.6:.2f} km/h vs target "
                      f"{args.target * 3.6:.2f} km/h)")
        else:
            print(f"  target never reached in {args.duration:g} s")
        return 0

    def cost(candidate: DriveSystem) -> float:
        res = simulate(candidate, voltage_signal("step", candidate, args.duration),
                       args.duration, engine=args.engine)
        if args.metric in res.columns():
            val = res.columns()[args.metric][-1]
        elif args.metric in res.metrics():
            val = res.metrics()[args.metric]
        elif args.metric in res.peak():
            val = res.peak()[args.metric]
        else:
            val = res.final().get(args.metric, np.nan)
        return float(val) if np.isfinite(val) else 1e9

    value, best_cost = optimum(system, cost, args.param, (args.min, args.max))
    print(f"best {args.param} = {value:.6g}   ({args.metric} = {best_cost:.6g})")
    return 0


def _split_list(value):
    """Flatten repeatable/comma-separated CLI lists into a list of tokens."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for item in items:
        out.extend(part.strip() for part in str(item).split(",") if part.strip())
    return out


def _key_value(items, flag, cast=float):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise DrivetrainError(f"{flag} expects path=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = cast(raw)
        except ValueError:
            raise DrivetrainError(f"{flag} expects a number for {key.strip()!r}, got {raw!r}")
    return out


def cmd_fit(args) -> int:
    """Fit a model to measured data, with the constants you already know held fixed."""
    from .fit import Dataset, FREE_ALIASES, fit as run_fit

    if args.paths:
        import math

        from .fit import STATE_PATHS
        from .sweep import get_path
        system = build_system(args)
        catalog = [row for row in presets.field_catalog(system)
                   if row["path"] not in STATE_PATHS]
        groups: dict[str, list[str]] = {}
        for row in catalog:
            groups.setdefault(row["group"], []).append(row["path"])
        print("fittable parameter paths (this configuration):")
        for group in ("Drivetrain", "Motor", "Gearbox", "Load", "Wheel", "Other"):
            for path in groups.get(group, []):
                value = get_path(system, path)
                note = ""
                if isinstance(value, float) and not math.isfinite(value):
                    note = "   (no limit -- give it --known before fitting it)"
                why = structural_inert(system, path)
                if why:
                    note = f"   ({why})"
                print(f"  {path}{note}")
        print("\nshorthand accepted by --free:")
        for alias, paths in FREE_ALIASES.items():
            print(f"  {alias:<12} {', '.join(paths)}")
        return 0

    if not args.dataset:
        raise DrivetrainError("fit needs a dataset CSV (see --help), "
                              "or --paths to list the parameter names)")

    ds = Dataset.from_csv(args.dataset)
    start = presets.load(args.config) if args.config else build_system(args)
    free = _split_list(args.free) or "motor"
    known = _key_value(args.known, "--known")
    ties = _key_value(args.tie, "--tie", cast=str)
    weights = _key_value(args.weight, "--weight")
    bounds = {}
    for item in args.bound or []:
        if "=" not in item:
            raise DrivetrainError(f"--bound expects path=min:max, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            lo, hi = raw.split(":", 1)
            bounds[key.strip()] = (float(lo), float(hi))
        except ValueError:
            raise DrivetrainError(f"--bound expects path=min:max, got {item!r}")

    # In --json mode stdout carries only the document, so chatter goes to stderr.
    say = (lambda *a: None) if args.json else (lambda *a: print(*a))
    say(f"dataset {args.dataset}: {ds.kind}, {len(ds.blocks)} block(s), "
        f"channels {', '.join(ds.channels()) or 'none'}")
    for note in ds.notes:
        say(f"  note: {note}")

    res = run_fit(
        ds, system=start, free=free, known=known, ties=ties,
        channels=_split_list(args.channel) or None, weights=weights or None,
        engine=args.engine, dt=args.dt, voltage=args.voltage,
        excitation=args.excitation, period=args.period, ramp_time=args.ramp,
        log=not args.no_log, bounds=bounds or None, max_nfev=args.max_nfev,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(res.as_dict(), indent=2))
    else:
        print()
        print(res.report())
    if args.out:
        res.save(args.out)
        say(f"\nwrote fitted model to {args.out}")
        say("load it with:  python3 -m drivetrain describe --config "
            f"{args.out}   (or open it in the GUI)")
    return 0 if res.success else 1


def _validate_job(job):
    """Module level so it can be pickled to worker processes."""
    name, engine, duration, dt = job
    system = presets.build(name)
    res = simulate(system, system.voltage_limit, duration, dt=dt, engine=engine)
    return name, engine, float(res.motor_speed[-1])


def cmd_validate(args) -> int:
    """Cross-check the engines against each other and against the analytic model.

    Two different things can disagree, and conflating them hides real bugs:

    * **solver error** -- engines disagree with each other.  This is a genuine
      numerical failure and fails the check.
    * **not settled** -- every engine agrees but none has reached the analytic
      equilibrium yet, because ``duration`` is shorter than a few time constants.
      Reported as a note, not a failure.

    ``--dt`` defaults to *auto*: each engine integrates at its own resolved time
    constant, and the engines are compared at that resolution.  Passing an
    explicit small ``--dt`` is how you stress-test convergence order instead.
    """
    names = args.presets or presets.preset_names()
    jobs = [(name, engine, args.duration, args.dt)
            for name in names for engine in ("exact", "rk4", "euler")]

    if args.jobs and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        ctx = _mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            rows = list(pool.map(_validate_job, jobs))
    else:
        rows = [_validate_job(j) for j in jobs]

    finals: dict[str, dict[str, float]] = {n: {} for n in names}
    for name, engine, w in rows:
        finals[name][engine] = w

    failures = 0
    for name in names:
        system = presets.build(name)
        analytic = system.steady_speed(system.voltage_limit)
        vals = finals[name]
        spread = max(vals.values()) - min(vals.values())
        rel_spread = abs(spread) / max(abs(max(vals.values(), key=abs)), 1e-9)
        err_analytic = abs(vals["exact"] - analytic) / max(abs(analytic), 1e-9)
        settled = args.duration >= system.settle_time

        note = ""
        if rel_spread > args.tol:
            note = f"  <-- SOLVER DISAGREEMENT ({rel_spread * 100:.2f}%)"
            failures += 1
        elif err_analytic > args.tol and not settled:
            note = f"  (not settled: {args.duration:g} s < settle {system.settle_time:.1f} s)"
        elif err_analytic > args.tol:
            note = f"  <-- analytic mismatch ({err_analytic * 100:.1f}%)"
            failures += 1

        print(f"{name:16s} " + "  ".join(f"{e}={rad_s_to_rpm(v):9.2f}" for e, v in vals.items())
              + f"  analytic={rad_s_to_rpm(analytic):9.2f}  spread {rel_spread * 100:.3f}%{note}")

    print(f"\n{len(names) - failures}/{len(names)} presets pass "
          f"(engine agreement + analytic within {args.tol * 100:.2f}% at {args.duration:g} s)")
    return 0 if failures == 0 else 1


def cmd_gui(args) -> int:
    from .gui import main as gui_main

    return gui_main(["--preset", args.preset, "--theme", args.theme])


def cmd_export(args) -> int:
    """Hand the model to the place it is used: C, Rust, Python, ROS 2."""
    from .export import export as write_code, options_from_args, summarise

    fit_doc = None
    if args.fit_json:
        # A fit result JSON carries the whole fitted model, so it can be the
        # source directly: `fit --json > fit.json`, then `export --fit-json`.
        import json
        from pathlib import Path

        from .export import system_from_fit_document

        try:
            fit_doc = json.loads(Path(args.fit_json).read_text(encoding="utf-8"))
        except OSError as exc:
            raise DrivetrainError(f"cannot read --fit-json {args.fit_json}: {exc}")
        system = system_from_fit_document(fit_doc)
        for path, raw in parse_overrides(getattr(args, "set", None)).items():
            system = _set(system, path, coerce(raw))
        source = f"fit result {args.fit_json}"
    else:
        system = build_system(args)
        source = _source_label(args)

    options = options_from_args(args, source, fit_doc=fit_doc)
    if getattr(args, "dq", False):
        # the dq twin: same flags, but the machine keeps its d/q state instead
        # of collapsing to a DC equivalent
        from .export_dq import export_dq, summarise_dq

        paths = export_dq(system, args.out, options)
        if args.json:
            import json

            manifest = next(p for p in paths if p.suffix == ".json")
            print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))
            return 0
        summarise_dq(system, options, paths)
        return 0
    paths = write_code(system, args.out, options)
    if args.json:
        import json

        manifest = next(p for p in paths if p.suffix == ".json")
        print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))
        return 0
    summarise(system, options, paths)
    return 0


def cmd_efficiency(args) -> int:
    """Where the power actually goes, at one point or over a whole map."""
    from .efficiency import efficiency_map, operating_point

    system = build_system(args)
    if args.map:
        emap = efficiency_map(
            system,
            n_speeds=args.n_speeds,
            n_torques=args.n_torques,
            speed_top=args.speed_top,
            torque_top=args.torque_top,
            include_regen=not args.motoring_only,
            torque_is=args.torque_is,
            include_gearbox=args.include_gearbox,
        )
        if args.json:
            import json

            print(json.dumps(emap.as_dict(), indent=2))
            return 0
        print(emap.report())
        if args.csv:
            from pathlib import Path

            path = emap.write_csv(Path(args.csv))
            print(f"\nwrote {emap.shape[0] * emap.shape[1]} rows to {path}")
        return 0

    if args.speed is None or args.torque is None:
        raise DrivetrainError(
            "give --speed and --torque for one operating point, or --map for the "
            "whole map (the speed axis is motor-side rad/s, the torque axis is "
            "motor-side N*m)"
        )
    point = operating_point(
        system, args.speed, args.torque,
        torque_is=args.torque_is, include_gearbox=args.include_gearbox,
    )
    if args.json:
        import json

        print(json.dumps(point.as_dict(), indent=2))
        return 0
    print(point.report())
    return 0


# ----------------------------------------------------------------------
# argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="drivetrain",
        description="DC geared drivetrain simulator: model, sweep, optimise, plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, with_set=True):
        p.add_argument("--preset", default="legacy", choices=presets.preset_names())
        p.add_argument("--config", help="start from a saved model JSON instead of a preset")
        if with_set:
            p.add_argument("--set", "-s", action="append", metavar="PATH=VALUE",
                           help="override any nested parameter, repeatable")
        p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser("presets", help="list presets, motors and wheels")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("describe", help="derived quantities + editable parameters")
    common(p)
    p.add_argument("--ratio-classical", action="store_true",
                   help="interpret --set gearbox.ratio as a classical N:1 reduction")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("run", help="time-domain simulation")
    common(p)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=None, help="solver step (auto if omitted)")
    p.add_argument("--engine", default="exact", choices=("exact", "rk4", "euler"))
    p.add_argument("--excitation", default="step", choices=EXCITATIONS)
    p.add_argument("--period", type=float, default=None, help="square/alternating period")
    p.add_argument("--ramp", type=float, default=None, help="ramp rise time")
    p.add_argument("--csv", help="write the trajectory to this CSV file")
    p.add_argument("--ratio-classical", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="sweep one parameter, print a metrics table")
    common(p)
    p.add_argument("--param", required=True, help="dotted parameter path")
    p.add_argument("--min", type=float, required=True)
    p.add_argument("--max", type=float, required=True)
    p.add_argument("--points", type=int, default=40)
    p.add_argument("--metric", default="steady_rpm")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--engine", default="exact", choices=("exact", "rk4", "euler"))
    p.add_argument("--jobs", "-j", type=int, default=0,
                   help="worker processes (0 = all CPUs, 1 = inline)")
    p.add_argument("--show", type=int, default=20, help="rows to print")
    p.add_argument("--minimize", action="store_true", help="lower metric is better")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("optimize", help="best value of a parameter for an objective")
    common(p)
    p.add_argument("--param", default="gearbox.ratio")
    p.add_argument("--min", type=float, default=0.02)
    p.add_argument("--max", type=float, default=1.0)
    p.add_argument("--metric", default="rise_time_s")
    p.add_argument("--maximize", action="store_true")
    p.add_argument("--target", type=float, default=None,
                   help="target ground speed [m/s] when optimising gearbox.ratio")
    p.add_argument("--target-rpm", type=float, default=None,
                   help="target output-shaft speed [rpm] when optimising gearbox.ratio")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--engine", default="exact", choices=("exact", "rk4", "euler"))
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser(
        "fit", help="fit a model to measured data, keeping known constants fixed")
    p.add_argument("dataset", nargs="?", help="CSV of measured data (see docs/fitting.md)")
    p.add_argument("--preset", default="legacy", choices=presets.preset_names(),
                   help="starting model (default legacy)")
    p.add_argument("--config", help="start from a saved model JSON instead of a preset")
    p.add_argument("--set", "-s", action="append", metavar="PATH=VALUE",
                   help="override any parameter of the starting model, repeatable")
    p.add_argument("--free", "-f", action="append", metavar="PATH|GROUP",
                   help="parameters the fit may move: dotted paths or a group name ("
                        + ", ".join(FREE_ALIASES) + ", all); default motor")
    p.add_argument("--known", "-k", action="append", metavar="PATH=VALUE",
                   help="pin a parameter to a constant you have measured, repeatable")
    p.add_argument("--tie", action="append", metavar="TARGET=SOURCE",
                   help="copy one fitted value onto another path, e.g. "
                        "group.motor.kt=group.motor.kv")
    p.add_argument("--channel", action="append", metavar="NAME",
                   help="restrict the comparison to these data channels, repeatable")
    p.add_argument("--weight", action="append", metavar="CHANNEL=SIGMA",
                   help="1-sigma noise per channel (default: measured from the "
                        "high-frequency scatter of the log itself)")
    p.add_argument("--bound", action="append", metavar="PATH=MIN:MAX",
                   help="override the search range of one parameter (catalogue ranges "
                        "are already widened x100 at each end)")
    p.add_argument("--voltage", type=float, default=None,
                   help="drive level when the dataset has no voltage_V column")
    p.add_argument("--excitation", default="step", choices=EXCITATIONS,
                   help="prescribed input shape when voltage_V is missing")
    p.add_argument("--period", type=float, default=None, help="square/alternating period")
    p.add_argument("--ramp", type=float, default=None, help="ramp rise time")
    p.add_argument("--engine", default="exact", choices=("exact", "rk4", "euler"))
    p.add_argument("--dt", type=float, default=None, help="solver step (auto if omitted)")
    p.add_argument("--max-nfev", type=int, default=400, help="optimiser evaluation budget")
    p.add_argument("--no-log", action="store_true",
                   help="optimise the raw values instead of their logarithms")
    p.add_argument("--out", "-o", help="write the fitted model to this JSON file")
    p.add_argument("--paths", action="store_true",
                   help="list the parameter paths and group names, then exit")
    p.add_argument("--verbose", "-v", action="store_true", help="optimiser progress")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("export", help="emit the model as standalone C / Rust / Python code")
    p.add_argument("--preset", default="legacy", choices=presets.preset_names())
    p.add_argument("--config", help="start from a saved model JSON instead of a preset "
                                    "(e.g. the file written by `fit --out`)")
    p.add_argument("--set", "-s", action="append", metavar="PATH=VALUE",
                   help="override any nested parameter before exporting, repeatable")
    p.add_argument("--fit-json", metavar="FILE",
                   help="export the model inside a `fit --json` document, and carry its "
                        "provenance and warnings into the generated comments")
    p.add_argument("--out", "-o", default="build/export",
                   help="directory for the generated code")
    p.add_argument("--targets", "-t", default="c,rust,python",
                   help="comma-separated: c, rust, python, ros")
    p.add_argument("--dq", action="store_true",
                   help="export the dq/FOC model of a brushless machine (Python only "
                        "today) instead of the i_d=0 DC equivalent")
    p.add_argument("--dtype", default="f32", choices=("f32", "f64"),
                   help="float width for the generated C/Rust")
    p.add_argument("--sample-time", type=float, default=None, metavar="S",
                   help="the control period the code is sized for")
    p.add_argument("--name", default="drive_model",
                   help="module/file name and C symbol prefix base")
    p.add_argument("--prefix", default="dt", help="C symbol prefix")
    p.add_argument("--no-examples", action="store_true",
                   help="skip the runnable step-response benches")
    p.add_argument("--json", action="store_true", help="print the manifest instead of the summary")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "efficiency", help="loss split at an operating point, or an efficiency map")
    common(p)
    p.add_argument("--speed", type=float, default=None,
                   help="motor-shaft speed [rad/s] for a single operating point")
    p.add_argument("--torque", type=float, default=None,
                   help="motor-shaft torque [N*m] for a single operating point")
    p.add_argument("--map", action="store_true",
                   help="bill the whole (speed, torque) plane instead of one point")
    p.add_argument("--n-speeds", type=int, default=33, help="speed rows for --map")
    p.add_argument("--n-torques", type=int, default=25, help="torque columns for --map")
    p.add_argument("--speed-top", type=float, default=None,
                   help="top motor speed [rad/s] for --map (auto by default)")
    p.add_argument("--torque-top", type=float, default=None,
                   help="torque magnitude at the top of the axis [N*m] (auto by default)")
    p.add_argument("--motoring-only", action="store_true",
                   help="drop the braking half of the torque axis from --map")
    p.add_argument("--torque-is", default="developed", choices=("developed", "shaft"),
                   help="whether the torque axis is electromagnetic or delivered")
    p.add_argument("--include-gearbox", action="store_true",
                   help="bill the gear stage's losses in the split")
    p.add_argument("--csv", help="with --map, write the grid to this CSV file")
    p.set_defaults(func=cmd_efficiency)

    p = sub.add_parser("validate", help="cross-check engines vs analytic model")
    p.add_argument("--presets", nargs="*", default=None)
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=None,
                   help="force a step size (default: auto-resolve per system)")
    p.add_argument("--jobs", "-j", type=int, default=0, help="worker processes")
    p.add_argument("--tol", type=float, default=0.01, help="relative tolerance")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("gui", help="launch the interactive interface")
    p.add_argument("--preset", default="legacy", choices=presets.preset_names())
    p.add_argument("--theme", default="gruvbox", choices=("gruvbox", "system"),
                   help="widget and chart colour scheme")
    p.set_defaults(func=cmd_gui)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DrivetrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
