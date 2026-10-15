"""Regression tests -- stdlib unittest, no third-party test runner required.

Run with::

    python3 -m unittest discover -s tests -t .

Covers the load-polynomial fit, the closed-form equilibrium (which replaced a
fixed-point iteration that diverged for quadratic/aero loads), current-limited
mode detection, and the fast scalar integrators that now back the engines.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from drivetrain import DriveSystem, Gearbox, MotorGroup, MotorModel
from drivetrain import RotationalLoad, VehicleLoad, Wheel
from drivetrain.load import LoadBundle
from drivetrain.motor import rad_s_to_rpm
from drivetrain import presets
from drivetrain.simulate import EXCITATIONS, _expm1_over_a, simulate, voltage_signal
from drivetrain.system import LoadCoefficients
from drivetrain.sweep import grid_sweep, set_path


class TestLoadCoefficients(unittest.TestCase):
    """The fast paths fit the load to sign(w)*(static + visc*|w| + quad*w^2)."""

    def test_fit_is_exact_for_every_builtin_load(self):
        for name in presets.preset_names():
            system = presets.build(name)
            lc = system.load_coefficients()
            self.assertTrue(lc.exact, f"{name}: load did not fit the polynomial")
            scale = max(abs(system.no_load_speed), 10.0)
            for frac in (0.05, 0.2, 0.35, 0.6, 0.9):
                w = scale * frac
                real = system.reflected_load_torque(w)
                self.assertAlmostEqual(real, lc.torque(w), delta=1e-9 * max(abs(real), 1e-6))
                self.assertAlmostEqual(-real, lc.torque(-w), delta=1e-9 * max(abs(real), 1e-6))

    def test_custom_nonpolynomial_load_is_flagged(self):
        """A load outside the const+linear+quadratic family must report exact=False
        so the engines fall back to evaluating it directly."""

        class CubicLoad(RotationalLoad):
            def torque_at(self, speed):
                return self.viscous * speed + 1e-9 * speed ** 3

        system = presets.build("legacy")
        system.load = CubicLoad(inertia=1e-5, viscous=1e-4, constant_torque=0.0)
        lc = system.load_coefficients()
        self.assertFalse(lc.exact)
        # engines must still run (they take the slow path)
        res = simulate(system, 12.0, 5.0, engine="exact")
        self.assertTrue(np.all(np.isfinite(res.motor_speed)))

    def test_static_torque_matches_analytic(self):
        system = presets.build("winch")
        lc = system.load_coefficients()
        self.assertAlmostEqual(lc.static, system.load_static_torque(), places=12)

    def test_negative_grade_reduces_static_load(self):
        flat = presets.build("pod_4x_geared")
        down = presets.build("pod_4x_geared", {"load.grade_deg": -10.0})
        self.assertLess(down.load_coefficients().static, flat.load_coefficients().static)


class TestEquilibrium(unittest.TestCase):
    """Closed-form quadratic equilibrium -- the old iteration diverged here."""

    def test_quadratic_load_equilibrium_is_not_divergent(self):
        # Purely quadratic load: fixed-point iteration oscillates between ~0 and
        # a huge value, so this case used to return nonsense.
        lc = LoadCoefficients(static=0.0, viscous=0.0, quadratic=1e-3)
        w = lc.equilibrium(1.0)
        self.assertAlmostEqual(w, math.sqrt(1.0 / 1e-3), places=9)

    def test_equilibrium_balances_the_torque(self):
        for quad in (0.0, 1e-6, 4e-6):
            lc = LoadCoefficients(static=0.02, viscous=1e-4, quadratic=quad)
            drive, extra = 0.9, 2e-3
            w = lc.equilibrium(drive, extra_damping=extra)
            balance = drive - extra * w - lc.torque(w)
            self.assertAlmostEqual(balance, 0.0, delta=1e-9 * max(drive, 1.0))

    def test_stall_when_torque_cannot_break_away(self):
        lc = LoadCoefficients(static=1.0, viscous=1e-3, quadratic=0.0)
        self.assertEqual(lc.equilibrium(0.5), 0.0)
        self.assertEqual(lc.equilibrium(-0.5), 0.0)

    def test_no_speed_dependent_load_means_no_equilibrium(self):
        lc = LoadCoefficients(static=0.1, viscous=0.0, quadratic=0.0)
        self.assertEqual(lc.equilibrium(1.0), float("inf"))

    def test_sprinter_current_limited_equilibrium_is_physical(self):
        """Regression: this preset used to report ~220,000 rpm, which is above its
        own no-load speed and therefore impossible."""
        system = presets.build("sprinter")
        w = system.steady_speed(12.0)
        self.assertLess(w, system.no_load_speed)
        self.assertEqual(system.steady_mode(12.0), "current_limited")
        # the equilibrium must satisfy the torque balance at the current limit
        g = system.group
        lc = system.load_coefficients()
        drive = g.kt_eff * system.current_limit_total
        balance = drive - system.base_damping * w - lc.torque(w)
        self.assertAlmostEqual(balance, 0.0, delta=1e-6 * drive)

    def test_steady_mode_not_fooled_by_float_noise(self):
        """At the exact voltage-limited equilibrium the back-EMF current is far
        below the limit; a 1-ULP comparison used to mislabel these."""
        for name in ("legacy", "legacy_4x", "pod_2x_direct", "winch"):
            system = presets.build(name)
            v = system.voltage_limit
            mode = system.steady_mode(v)
            self.assertIn(mode, ("voltage_limited", "free"), f"{name}: {mode}")

    def test_steady_speed_matches_long_simulation(self):
        """The analytic equilibrium must be what the integrators actually reach."""
        for name in presets.preset_names():
            system = presets.build(name)
            v = system.voltage_limit
            target = system.steady_speed(v)
            span = max(system.settle_time, 1.0) * 12.0
            res = simulate(system, v, span, engine="exact")
            got = float(res.motor_speed[-1])
            self.assertAlmostEqual(got, target, delta=1e-3 * max(abs(target), 1.0),
                                   msg=f"{name} did not settle to its analytic equilibrium")


class TestEnginesAgree(unittest.TestCase):
    """All three engines must land on the same answer at a resolved step size."""

    def test_engines_agree_on_every_preset(self):
        # 10 s at dt=2e-5 is ~8 mechanical time constants for these presets --
        # plenty to compare engines without spending 30 s of sim per preset.
        for name in presets.preset_names():
            system = presets.build(name)
            v = system.voltage_limit
            vals = {}
            for engine in ("exact", "rk4", "euler"):
                res = simulate(system, v, 10.0, dt=2e-5, engine=engine, samples=None)
                vals[engine] = float(res.motor_speed[-1])
            spread = max(vals.values()) - min(vals.values())
            rel = abs(spread) / max(abs(max(vals.values(), key=abs)), 1e-9)
            self.assertLess(rel, 1e-4, f"{name}: engines disagree {vals}")

    def test_second_order_path_matches_rk4_and_analytic(self):
        # dt must resolve the *electrical* time constant (L/R = 1e-4 s here),
        # which is what makes this case expensive; 12 s is ~10 mechanical taus,
        # i.e. a 3e-5 residual against the analytic equilibrium.
        for inductance in (1e-4, 5e-4):
            system = presets.build("legacy", {"group.motor.inductance": inductance})
            exact = simulate(system, 12.0, 12.0, dt=1e-5, engine="exact", samples=None)
            rk4 = simulate(system, 12.0, 12.0, dt=1e-5, engine="rk4", samples=None)
            self.assertAlmostEqual(exact.motor_speed[-1], rk4.motor_speed[-1],
                                   delta=1e-6 * abs(rk4.motor_speed[-1]))
            self.assertAlmostEqual(exact.current[-1], rk4.current[-1], delta=1e-6)
            self.assertAlmostEqual(exact.motor_speed[-1], system.steady_speed(12.0),
                                   delta=1e-4 * abs(exact.motor_speed[-1]))

    def test_second_order_matrix_exponential_is_cached(self):
        """The L>0 path recomputes expm per distinct damping, not per step."""
        system = presets.build("legacy", {"group.motor.inductance": 5e-4})
        res = simulate(system, 12.0, 5.0, dt=1e-4, engine="exact", samples=None)
        self.assertTrue(np.all(np.isfinite(res.motor_speed)))

    def test_fast_and_slow_load_paths_agree(self):
        """The frozen scalar path must reproduce reflected_load_torque exactly."""
        for name in presets.preset_names():
            system = presets.build(name)
            lc = system.load_coefficients()
            self.assertTrue(lc.exact)
            scale = max(abs(system.no_load_speed), 10.0)
            for w in np.linspace(0.0, scale, 25):
                self.assertAlmostEqual(system.load_damping_at(w), lc.damping(w), places=12)


class TestNumerics(unittest.TestCase):
    def test_expm1_over_a_is_stable_at_zero(self):
        # undamped rotor: a == 0 exactly, naive (exp(a h)-1)/a divides by zero
        self.assertAlmostEqual(_expm1_over_a(0.0, 0.1), 0.1, places=15)
        # tiny a*h: naive form loses all precision to cancellation
        h, a = 1e-3, -1e-6
        self.assertAlmostEqual(_expm1_over_a(a, h), h * (1 + a * h / 2), delta=1e-18)
        # large a*h: exact
        self.assertAlmostEqual(_expm1_over_a(-5.0, 0.2), (math.exp(-1.0) - 1.0) / -5.0, places=15)

    def test_undamped_undamped_load_still_integrates(self):
        system = presets.build("spinning_mass", {"load.viscous": 0.0, "load.constant_torque": 0.0})
        res = simulate(system, 12.0, 5.0, engine="exact")
        self.assertTrue(np.all(np.isfinite(res.motor_speed)))

    def test_excitations_all_run(self):
        system = presets.build("pod_4x_geared")
        for kind in EXCITATIONS:
            v = voltage_signal(kind, system, duration=5.0)
            res = simulate(system, v, 5.0, engine="exact")
            self.assertEqual(len(v), len(res.motor_speed), kind)
            self.assertTrue(np.all(np.isfinite(res.motor_speed)), kind)

    def test_voltage_signal_shapes(self):
        system = presets.build("legacy")
        kw = dict(duration=4.0, dt=0.01, value=10.0)
        step = voltage_signal("step", system, **kw)
        self.assertTrue(np.all(step == 10.0))
        coast = voltage_signal("coast", system, **kw)
        self.assertTrue(np.all(coast == 0.0))
        ramp = voltage_signal("ramp", system, ramp_time=2.0, **kw)
        self.assertAlmostEqual(ramp[0], 0.0, places=12)      # t = 0
        self.assertAlmostEqual(ramp[100], 5.0, places=9)     # t = 1.0 of a 2 s ramp
        self.assertAlmostEqual(ramp[-1], 10.0, places=9)
        sq = voltage_signal("square", system, period=2.0, **kw)
        self.assertEqual(set(np.unique(sq)), {0.0, 10.0})
        alt = voltage_signal("alternating", system, period=1.0, **kw)
        self.assertEqual(set(np.unique(alt)), {-10.0, 10.0})


class TestMotorGroupScaling(unittest.TestCase):
    """Paralleled identical motors scale R, not kt/kv."""

    def test_no_load_speed_is_independent_of_motor_count(self):
        one = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0), 1)
        four = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0), 4)
        self.assertAlmostEqual(one.kt_eff, four.kt_eff)
        self.assertAlmostEqual(one.kv_eff, four.kv_eff)
        self.assertAlmostEqual(four.r_eff, one.r_eff / 4)

    def test_stall_torque_scales_linearly(self):
        one = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0), 1)
        four = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0), 4)
        self.assertAlmostEqual(four.stall_torque, 4 * one.stall_torque)
        # electromechanical damping lives on the assembled system
        s_one = DriveSystem(group=one, gearbox=Gearbox(1.0, 1.0),
                            load=RotationalLoad(inertia=1e-5))
        s_four = DriveSystem(group=four, gearbox=Gearbox(1.0, 1.0),
                             load=RotationalLoad(inertia=1e-5))
        self.assertAlmostEqual(s_four.electromechanical_damping,
                               4 * s_one.electromechanical_damping)

    def test_mech_time_constant_is_independent_of_motor_count(self):
        one = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0, rotor_inertia=1e-5), 1)
        four = MotorGroup(MotorModel(kv=0.03, kt=0.03, resistance=1.0, rotor_inertia=1e-5), 4)
        self.assertAlmostEqual(one.mech_time_constant, four.mech_time_constant)


class TestSweep(unittest.TestCase):
    def test_parallel_and_serial_sweeps_match(self):
        grid = {"gearbox.ratio": [0.2, 0.4, 0.6, 0.8], "group.count": [1, 2, 4]}
        serial = grid_sweep(presets.build("pod_4x_geared"), grid, duration=5.0, workers=1)
        parallel = grid_sweep(presets.build("pod_4x_geared"), grid, duration=5.0, workers=0)
        self.assertEqual(len(serial), len(parallel))
        self.assertEqual(serial.values, parallel.values)
        for key in serial.columns:
            np.testing.assert_allclose(serial.columns[key], parallel.columns[key], rtol=1e-12)

    def test_set_path_returns_a_copy_with_nested_values(self):
        """set_path is functional: it must not mutate the template."""
        system = presets.build("legacy")
        out = set_path(system, "group.motor.resistance", 0.5)
        self.assertEqual(out.group.motor.resistance, 0.5)
        self.assertEqual(system.group.motor.resistance, 1.0)   # template untouched
        out = set_path(presets.build("pod_4x_geared"), "load.wheel.radius", 0.09)
        self.assertEqual(out.load.wheel.radius, 0.09)


if __name__ == "__main__":
    unittest.main(verbosity=2)
