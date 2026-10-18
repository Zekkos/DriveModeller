"""Tests for drivetrain.fit -- fitting a model to measured data."""

from __future__ import annotations

import math
import unittest

import numpy as np

from drivetrain import fit, presets
from drivetrain.exceptions import ConfigError
from drivetrain.simulate import simulate, voltage_signal
from drivetrain.sweep import set_path
from drivetrain.motor import rad_s_to_rpm, rpm_to_rad_s

# a motor that is deliberately not one of the catalogue entries
TRUE = {
    "group.motor.kv": 0.0212,
    "group.motor.kt": 0.0212,
    "group.motor.resistance": 0.44,
    "group.motor.rotor_inertia": 2.2e-5,
    "group.motor.bearing_damping": 3.0e-5,
}
KNOWN = {"group.motor.rotor_inertia": 2.2e-5, "group.motor.bearing_damping": 3.0e-5}
ELECTRICAL = ("group.motor.kv", "group.motor.kt", "group.motor.resistance")


def truth_system() -> object:
    system = presets.build("pod_2x_direct")
    for path, value in TRUE.items():
        system = set_path(system, path, value)
    return system


def bench_rows() -> dict[str, list[float]]:
    """Square-wave drive at 6 V, sampled at 50 Hz with a little noise."""
    system = truth_system()
    dt = 1.0 / 400.0
    v = voltage_signal("square", system, 8.0, dt=dt, value=6.0, period=2.0)
    res = simulate(system, v, 8.0, dt=dt, samples=None)
    ch = res.channels()
    rng = np.random.default_rng(7)
    step = 8
    sel = slice(None, None, step)
    return {
        "time": ch["time"][sel],
        "volts": ch["voltage_V"][sel],
        "current": ch["current_A"][sel] + rng.normal(0, 0.03, size=len(ch["time"][sel])),
        "rpm": ch["motor_rpm"][sel] + rng.normal(0, 10.0, size=len(ch["time"][sel])),
    }


class DatasetTests(unittest.TestCase):
    def test_alias_and_case_insensitive_headers(self) -> None:
        ds = fit.Dataset.from_rows({
            "Time": [0.0, 1.0],
            "CURRENT_A": [1.0, 2.0],
            "Volts": [6.0, 6.0],
            "rpm": [100.0, 200.0],
            "bogus": [1.0, 1.0],
        })
        self.assertEqual(ds.channels(), ["motor_rpm", "current_A", "voltage_V"])
        self.assertEqual(ds.kind, "timeseries")
        self.assertTrue(any("bogus" in n for n in ds.notes))

    def test_run_column_splits_experiments(self) -> None:
        ds = fit.Dataset.from_rows({
            "time": [0.0, 1.0, 0.0, 1.0],
            "run": [1, 1, 2, 2],
            "current_A": [1.0, 2.0, 3.0, 4.0],
            "voltage_V": [6.0, 6.0, 6.0, 6.0],
        })
        self.assertEqual(len(ds.blocks), 2)
        self.assertEqual([b.n for b in ds.blocks], [2, 2])

    def test_time_reset_also_splits(self) -> None:
        ds = fit.Dataset.from_rows({
            "time": [0.0, 1.0, 0.0, 1.0],
            "current_A": [1.0, 2.0, 3.0, 4.0],
            "voltage_V": [6.0, 6.0, 6.0, 6.0],
        })
        self.assertEqual(len(ds.blocks), 2)

    def test_efficiency_percent_is_rescaled(self) -> None:
        ds = fit.Dataset.from_rows({
            "voltage_V": [6.0, 12.0],
            "efficiency": [40.0, 60.0],
        })
        self.assertAlmostEqual(float(ds.blocks[0].columns["efficiency"][0]), 0.4)
        self.assertTrue(any("percent" in n for n in ds.notes))

    def test_steady_rows_need_voltage(self) -> None:
        with self.assertRaises(ConfigError):
            fit.Dataset.from_rows({"motor_rpm": [100.0]})

    def test_csv_round_trip(self) -> None:
        import tempfile
        rows = bench_rows()
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/bench.csv"
            with open(path, "w") as fh:
                fh.write("time,voltage_V,current_A,motor_rpm\n")
                for i in range(len(rows["time"])):
                    fh.write("%.5f,%.4f,%.4f,%.2f\n" % (
                        rows["time"][i], rows["volts"][i],
                        rows["current"][i], rows["rpm"][i]))
            ds = fit.Dataset.from_csv(path)
        self.assertEqual(ds.channels(), ["motor_rpm", "current_A", "voltage_V"])
        self.assertEqual(ds.blocks[0].n, len(rows["time"]))


class FitTests(unittest.TestCase):
    def test_time_domain_recovers_electrical_parameters(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        self.assertTrue(r.success, r.message)
        for p in r.parameters:
            self.assertLess(abs(p.value - TRUE[p.path]) / TRUE[p.path], 0.03, p.path)
            # the quoted uncertainty should be within a few sigma of the real error
            self.assertLess(abs(p.value - TRUE[p.path]), 5 * p.stderr, p.path)
        # weights are now estimated per-channel noise, so normalized rms reads ~1
        # when the fit is as good as the injected noise allows (0.03 A, 10 rpm)
        self.assertLess(r.normalized_rms, 2.0)
        self.assertGreater(r.normalized_rms, 0.3)
        # ...and the unit-free tracking quality is what the old number measured
        self.assertLess(max(r.relative_rms.values()), 0.05)
        self.assertLess(r.condition, 100.0)

    def test_weights_estimate_the_sensor_noise(self) -> None:
        """bench_rows() injects 0.03 A and 10 rpm of noise; the fit should find that."""
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        self.assertLess(r.weights["current_A"], 0.06)
        self.assertGreater(r.weights["current_A"], 0.01)
        self.assertLess(r.weights["motor_rpm"], 20.0)
        self.assertGreater(r.weights["motor_rpm"], 3.0)
        self.assertEqual(set(r.weights_basis), {"motor_rpm", "current_A"})
        # noise-aware weights beat weighting by raw magnitude: the current trace
        # is ~10 A and the speed trace ~1000 rpm, but the current is measured 300x
        # more precisely, so it should carry the larger weight per unit
        self.assertLess(r.weights["current_A"] * 100, r.weights["motor_rpm"])

    def test_known_constants_stay_put(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free="motor", known=KNOWN)
        motor = r.system.group.motor
        self.assertAlmostEqual(motor.rotor_inertia, 2.2e-5)
        self.assertAlmostEqual(motor.bearing_damping, 3.0e-5)
        self.assertNotIn("group.motor.rotor_inertia", {p.path for p in r.parameters})
        self.assertIn("group.motor.rotor_inertia", r.report())

    def test_steady_state_fit(self) -> None:
        system = truth_system()
        volts = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        ds = fit.Dataset.from_rows({
            "voltage_V": volts,
            "motor_rpm": [rad_s_to_rpm(system.steady_speed(v)) for v in volts],
            "current_A": [system.group.current_at(v, system.steady_speed(v)) for v in volts],
        }, name="datasheet")
        self.assertEqual(ds.kind, "steady")
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.kv", "group.motor.resistance"),
                    known=dict(KNOWN, **{"group.motor.kt": TRUE["group.motor.kt"]}))
        self.assertAlmostEqual(r.system.group.motor.kv, TRUE["group.motor.kv"], places=5)
        self.assertAlmostEqual(r.system.group.motor.resistance,
                               TRUE["group.motor.resistance"], places=4)

    def test_prescribed_voltage_without_a_column(self) -> None:
        rows = bench_rows()
        rows.pop("volts")
        ds = fit.Dataset.from_rows(rows)
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN,
                    voltage=6.0, excitation="square", period=2.0)
        self.assertTrue(r.success)
        self.assertLess(abs(r.system.group.motor.kv - TRUE["group.motor.kv"])
                        / TRUE["group.motor.kv"], 0.05)
        self.assertTrue(any("prescribe" in n for n in r.warnings + ds.notes))

    def test_degenerate_speed_only_data_is_flagged(self) -> None:
        rows = bench_rows()
        ds = fit.Dataset.from_arrays(time=rows["time"], motor_rpm=rows["rpm"], name="speed-only")
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.kt", "group.motor.rotor_inertia", "group.motor.kv"),
                    voltage=6.0, excitation="square", period=2.0)
        self.assertTrue(r.degenerate)
        self.assertTrue(any("degenerate" in w for w in r.warnings))
        self.assertTrue(any(p.stderr_text == "unbounded" for p in r.parameters))
        self.assertTrue(any("cannot be resolved at all" in w for w in r.warnings))
        self.assertTrue(any("trade off" in w for w in r.warnings))
        self.assertIn("degenerate", r.report())

    def test_tie_merges_two_parameters(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.kv", "group.motor.resistance"),
                    known=KNOWN, ties={"group.motor.kt": "group.motor.kv"})
        self.assertAlmostEqual(r.system.group.motor.kt, r.system.group.motor.kv, places=12)
        self.assertEqual(len(r.parameters), 2)

    def test_weights_change_the_balance(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        base = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        heavy = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN,
                        weights={"current_A": 0.01})
        self.assertNotAlmostEqual(base.system.group.motor.kv,
                                  heavy.system.group.motor.kv, places=5)

    def test_save_and_predict(self) -> None:
        import tempfile
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        with tempfile.TemporaryDirectory() as d:
            r.save(f"{d}/fitted.json")
            back = presets.load(f"{d}/fitted.json")
        self.assertAlmostEqual(back.group.motor.kv, r.system.group.motor.kv, places=12)
        runs = r.predict()
        self.assertEqual(len(runs), len(ds.blocks))
        self.assertIn("motor_rpm", runs[0]["model"])
        self.assertIn("motor_rpm", runs[0]["data"])
        self.assertEqual(len(runs[0]["model"]["motor_rpm"]), len(runs[0]["data"]["motor_rpm"]))

    def test_report_mentions_every_part(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        text = r.report()
        for fragment in ("free parameters", "residual rms", "known constants", "group.motor.kv"):
            self.assertIn(fragment, text)

    def test_errors(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        start = presets.build("pod_2x_direct")
        with self.assertRaises(ConfigError):
            fit.fit(ds, system=start, free="group.motor.nonsense")
        with self.assertRaises(ConfigError):
            fit.fit(ds, system=start, free="group.motor.kv", known={"group.motor.kv": 0.02})
        with self.assertRaises(ConfigError):
            fit.fit(ds, system=start, free="group.motor.kv", channels="nonsense_channel")
        with self.assertRaises(ConfigError):
            fit.fit(ds, system=start, free="group.motor.rot_vel")

    def test_group_alias_skips_known_constants(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free="motor",
                    known=dict(KNOWN, **{"group.motor.resistance": 0.44}))
        paths = {p.path for p in r.parameters}
        self.assertNotIn("group.motor.resistance", paths)
        self.assertIn("group.motor.kv", paths)
        self.assertAlmostEqual(r.system.group.motor.resistance, 0.44)

    # ---- parameter families ----------------------------------------
    def test_family_groups_and_all_expand(self) -> None:
        system = presets.build("pod_2x_direct")
        every = fit._expand_free("all", system, {"group.motor.kv": 0.02})
        self.assertIn("group.motor.resistance", every)
        self.assertIn("gearbox.ratio", every)
        self.assertIn("load.wheel.radius", every)
        # known constants, state fields and unlimited limits stay out
        self.assertNotIn("group.motor.kv", every)
        self.assertNotIn("group.motor.current", every)
        self.assertNotIn("group.motor.rot_vel", every)
        friction = fit._expand_free("friction", system, {})
        self.assertIn("group.motor.bearing_damping", friction)
        self.assertIn("load.rolling_resistance", friction)
        self.assertNotIn("group.motor.kv", friction)
        inertia = fit._expand_free("inertia", system, {})
        self.assertIn("group.motor.rotor_inertia", inertia)
        self.assertIn("gearbox.inertia", inertia)
        self.assertNotIn("load.drag_cd", inertia)

    def test_unlimited_limits_are_not_free_by_default(self) -> None:
        # a preset with max_current = inf cannot search it, so 'all' skips it
        system = presets.build("pod_2x_direct")
        system = set_path(system, "group.motor.max_current", float("inf"))
        self.assertNotIn("group.motor.max_current", fit._expand_free("all", system, {}))

    def test_by_path_accessor(self) -> None:
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        self.assertEqual(set(r.by_path), {p.path for p in r.parameters})
        self.assertAlmostEqual(r.by_path["group.motor.kv"].value, r.system.group.motor.kv,
                               places=12)


class SensorVocabularyTests(unittest.TestCase):
    """A bench log may report angular velocity in rad/s instead of rpm."""

    def test_rad_per_second_channel_recovers_the_same_motor(self) -> None:
        rows = bench_rows()
        ds = fit.Dataset.from_arrays(
            time=rows["time"], voltage_V=rows["volts"], current_A=rows["current"],
            motor_rad_s=rpm_to_rad_s(rows["rpm"]))
        self.assertEqual(set(ds.channels()), {"voltage_V", "current_A", "motor_rad_s"})
        r = fit.fit(ds, system=presets.build("pod_2x_direct"), free=ELECTRICAL, known=KNOWN)
        self.assertLess(abs(r.system.group.motor.kv - TRUE["group.motor.kv"])
                        / TRUE["group.motor.kv"], 0.03)
        self.assertIn("motor_rad_s", r.rms)

    def test_rad_per_second_headers_are_recognised(self) -> None:
        rows = bench_rows()
        ds = fit.Dataset.from_rows({
            "time": rows["time"], "Volts": rows["volts"],
            "motor_omega": rpm_to_rad_s(rows["rpm"]),
            "shaft_rad_s": rpm_to_rad_s(rows["rpm"]),
        })
        self.assertEqual(set(ds.channels()), {"voltage_V", "motor_rad_s", "output_rad_s"})

    def test_rpm_logged_in_a_rad_per_second_channel_is_called_out(self) -> None:
        rows = bench_rows()
        pinned = {k: v for k, v in TRUE.items() if k != "group.motor.bearing_damping"}
        # everything that sets the speed is known, so a 60x column error cannot
        # be quietly absorbed into kv
        ds = fit.Dataset.from_arrays(time=rows["time"], voltage_V=rows["volts"],
                                     current_A=rows["current"], motor_rad_s=rows["rpm"])
        # dt is quoted because a model this wrong makes the solver work hard: the
        # warning is the same either way, only the wait is not
        r = fit.fit(ds, system=truth_system(), free=("group.motor.inductance",), known=pinned,
                    dt=0.005)
        self.assertTrue(any("rpm in a rad/s channel" in w for w in r.warnings), r.warnings)

    def test_rad_per_second_logged_in_an_rpm_channel_is_called_out(self) -> None:
        rows = bench_rows()
        pinned = {k: v for k, v in TRUE.items() if k != "group.motor.bearing_damping"}
        ds = fit.Dataset.from_arrays(time=rows["time"], voltage_V=rows["volts"],
                                     current_A=rows["current"],
                                     motor_rpm=rpm_to_rad_s(rows["rpm"]))
        r = fit.fit(ds, system=truth_system(), free=("group.motor.inductance",), known=pinned,
                    dt=0.005)
        self.assertTrue(any("column is rad/s, not rpm" in w for w in r.warnings), r.warnings)


class IdentifiabilityTests(unittest.TestCase):
    """The fit has to say what a given dataset cannot tell you."""

    def test_steady_rows_say_which_parameters_are_invisible(self) -> None:
        system = truth_system()
        volts = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        rows = {"voltage_V": volts,
                "motor_rpm": [rad_s_to_rpm(system.steady_speed(v)) for v in volts],
                "current_A": [system.operating_point(v, system.steady_output_speed(v))["current_total_A"]
                              for v in volts]}
        ds = fit.Dataset.from_rows(rows)
        self.assertEqual(ds.kind, "steady")
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.kv", "group.motor.resistance",
                          "group.motor.rotor_inertia"),
                    known={"group.motor.kt": TRUE["group.motor.kt"]})
        self.assertTrue(any("no transient" in w for w in r.warnings), r.warnings)

    def test_speed_only_rows_cannot_separate_friction_from_kv(self) -> None:
        system = truth_system()
        volts = [3.0, 6.0, 9.0, 12.0]
        ds = fit.Dataset.from_rows({
            "voltage_V": volts,
            "motor_rpm": [rad_s_to_rpm(system.steady_speed(v)) for v in volts],
        })
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.kv", "group.motor.bearing_damping"),
                    known={"group.motor.kt": TRUE["group.motor.kt"],
                           "group.motor.resistance": TRUE["group.motor.resistance"]})
        self.assertTrue(any("equilibrium balance" in w for w in r.warnings), r.warnings)

    def test_saturated_drive_cannot_see_the_electrical_parameters(self) -> None:
        truth = set_path(truth_system(), "group.motor.max_current", 1.0)   # 2 A total
        dt = 1.0 / 500.0
        v = voltage_signal("step", truth, 1.0, dt=dt, value=6.0)
        res = simulate(truth, v, 1.0, dt=dt, samples=None)
        ch = res.channels()
        sel = slice(None, None, 5)
        ds = fit.Dataset.from_arrays(
            time=ch["time"][sel], voltage_V=ch["voltage_V"][sel],
            current_A=ch["current_A"][sel], motor_rpm=ch["motor_rpm"][sel])
        start = set_path(presets.build("pod_2x_direct"), "group.motor.max_current", 1.0)
        r = fit.fit(ds, system=start, free=("group.motor.kv", "group.motor.resistance"),
                    known=dict(KNOWN, **{"group.motor.kt": TRUE["group.motor.kt"]}), dt=dt)
        self.assertTrue(any("current limit" in w for w in r.warnings), r.warnings)

    def test_a_parameter_the_configuration_never_uses_is_reported(self) -> None:
        # pod_2x_direct has no inductance, so its current loop is algebraic and
        # inductance is not a thing the model evaluates at all
        ds = fit.Dataset.from_rows(bench_rows())
        r = fit.fit(ds, system=presets.build("pod_2x_direct"),
                    free=("group.motor.inductance", "group.motor.kv"),
                    known=dict(KNOWN, **{"group.motor.kt": TRUE["group.motor.kt"],
                                          "group.motor.resistance": TRUE["group.motor.resistance"]}),
                    dt=0.005)
        self.assertTrue(any("no sensitivity at all" in w and "inductance" in w
                            for w in r.warnings), r.warnings)

    def test_two_paths_that_are_one_quantity_are_named(self) -> None:
        start = presets.build("spinning_mass")
        truth = set_path(start, "group.motor.bearing_damping", 8e-5)
        dt = 1.0 / 500.0
        v = voltage_signal("step", truth, 0.5, dt=dt, value=8.0)
        res = simulate(truth, v, 0.5, dt=dt, samples=None)
        ch = res.channels()
        sel = slice(None, None, 5)
        ds = fit.Dataset.from_arrays(
            time=ch["time"][sel], voltage_V=ch["voltage_V"][sel],
            current_A=ch["current_A"][sel], motor_rpm=ch["motor_rpm"][sel])
        # direct drive: motor bearing damping and load viscous damping are the
        # same torque on the same shaft, so no dataset can tell them apart
        r = fit.fit(ds, system=start,
                    free=("group.motor.bearing_damping", "load.viscous"),
                    known={"group.motor.kv": start.group.motor.kv,
                           "group.motor.kt": start.group.motor.kt,
                           "group.motor.resistance": start.group.motor.resistance,
                           "group.motor.rotor_inertia": start.group.motor.rotor_inertia},
                    dt=dt)
        self.assertTrue(any("indistinguishable" in w for w in r.warnings), r.warnings)


# a bench rig on a gearbox: the drive's own inertia is known from the CAD, the
# friction of the bearings and of the load is not
RIG = {
    "gearbox.ratio": 0.3,
    "group.motor.kv": 0.014,
    "group.motor.kt": 0.014,
    "group.motor.resistance": 0.42,
    "group.motor.rotor_inertia": 2.0e-5,
    "group.motor.bearing_damping": 8.0e-5,
    "load.inertia": 1.2e-3,
    "load.viscous": 2.5e-4,
    "load.constant_torque": 0.010,
    "group.motor.max_voltage": 24.0,
}
RIG_KNOWN = {path: value for path, value in RIG.items()
             if path not in ("group.motor.kv", "group.motor.kt", "group.motor.resistance")}


def rig(ratio: float = 0.3) -> object:
    system = presets.build("spinning_mass")
    for path, value in RIG.items():
        system = set_path(system, path, ratio if path == "gearbox.ratio" else value)
    return system


def rig_log(system: object) -> fit.Dataset:
    """A 50 Hz bench log of a square-wave drive, written on its own grid."""
    dt = 0.02
    v = voltage_signal("square", system, 12.0, dt=dt, value=12.0, period=4.0)
    res = simulate(system, v, 12.0, dt=dt, samples=None)
    ch = res.channels()
    return fit.Dataset.from_arrays(
        time=ch["time"], voltage_V=ch["voltage_V"], current_A=ch["current_A"],
        motor_rpm=ch["motor_rpm"], name="rig")


class StartingPointTests(unittest.TestCase):
    """Where the fit starts, and what it does when that start is in the wrong basin."""

    def test_a_log_coarser_than_the_solver_is_replayed_as_steps(self) -> None:
        # the log holds one voltage per 0.02 s and the solver runs at 0.005 s:
        # replaying the log by interpolation would invent a ramp through every
        # edge of the square wave, and the fit would chase that artifact instead
        # of the motor (it used to compromise kt by a factor of eight doing so)
        ds = rig_log(rig())
        r = fit.fit(ds, system=rig(), free=ELECTRICAL, known=RIG_KNOWN, dt=0.005)
        for path in ELECTRICAL:
            want = RIG[path]
            got = r.by_path[path].value
            self.assertLess(abs(got - want) / want, 0.02,
                            f"{path}: {got:.5g} against {want:.5g}")
        self.assertLess(max(r.relative_rms.values()), 0.02, r.relative_rms)

    def test_the_start_is_read_off_the_log_not_out_of_the_catalogue(self) -> None:
        ds = rig_log(rig())
        r = fit.fit(ds, system=presets.build("spinning_mass"),
                    free=ELECTRICAL,
                    known=RIG_KNOWN, dt=0.005)
        self.assertIn("group.motor.kv", r.starts)
        self.assertIn("group.motor.kt", r.starts)
        for path in ("group.motor.kv", "group.motor.kt"):
            start, why = r.starts[path]
            self.assertTrue(why)
            self.assertLess(start / RIG[path], 3.0, f"{path} start {start:.4g} ({why})")
            self.assertGreater(start / RIG[path], 0.33, f"{path} start {start:.4g} ({why})")

    def test_a_wrong_gear_ratio_shows_up_as_an_impossible_motor(self) -> None:
        # the log was taken through a 0.3 gearbox; the model believes it is direct
        # drive.  The fit still tracks the data, but it has to make the motor do
        # the work the gearbox was really doing, and that leaves kt and kv -- the
        # same number in SI for any motor -- a factor of several apart
        ds = rig_log(rig(0.3))
        start = set_path(presets.build("spinning_mass"), "load.inertia", RIG["load.inertia"])
        start = set_path(start, "group.motor.rotor_inertia", RIG["group.motor.rotor_inertia"])
        start = set_path(start, "group.motor.max_voltage", RIG["group.motor.max_voltage"])
        r = fit.fit(ds, system=start,
                    free=("group.motor.kv", "group.motor.kt", "group.motor.resistance",
                          "group.motor.bearing_damping", "load.viscous",
                          "load.constant_torque"),
                    known={"group.motor.rotor_inertia": RIG["group.motor.rotor_inertia"],
                           "load.inertia": RIG["load.inertia"]},
                    excitation="square", voltage=12.0, period=4.0, dt=0.005)
        self.assertTrue(any("kt and kv inconsistent" in w for w in r.warnings), r.warnings)

    def test_the_coarse_scan_walks_a_bad_start_downhill(self) -> None:
        # one parameter, started two decades out: the scan hops it by factors of
        # ten and keeps whatever lowers the cost
        target = 2.0
        residual = lambda x: np.array([math.exp(x[0]) - target])
        logs = np.array([True])
        lo, hi = np.array([math.log(1e-6)]), np.array([math.log(1e6)])
        x, cost = fit._coarse_scan(residual, np.array([math.log(200.0)]), lo, hi, logs)
        self.assertLess(math.exp(float(x[0])) / target, 3.0)
        self.assertGreater(math.exp(float(x[0])) / target, 0.33)
        self.assertLess(cost, 0.5 * (200.0 - target) ** 2)


if __name__ == "__main__":
    unittest.main()
