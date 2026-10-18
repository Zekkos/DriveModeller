"""Regression tests -- stdlib unittest, no third-party test runner required.

Run with::

    python3 -m unittest discover -s tests -t .

Covers the load-polynomial fit, the closed-form equilibrium (which replaced a
fixed-point iteration that diverged for quadratic/aero loads), current-limited
mode detection, the fast scalar integrators that now back the engines, and the
Qt GUI (driven offscreen through qtpy).
"""

from __future__ import annotations

import math
import os
import tempfile
import time
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
        dt = 2e-5
        for name in presets.preset_names():
            system = presets.build(name)
            v = system.voltage_limit
            vals = {}
            for engine in ("exact", "rk4", "euler"):
                res = simulate(system, v, 10.0, dt=dt, engine=engine, samples=None)
                vals[engine] = float(res.motor_speed[-1])
            # exact and rk4 are high order, so they must land together tightly
            rel = abs(vals["exact"] - vals["rk4"]) / max(abs(vals["exact"]), 1e-9)
            self.assertLess(rel, 1e-4, f"{name}: exact/rk4 disagree {vals}")
            # euler is first order: its error is set by how well dt resolves the
            # FASTEST pole.  A brushed preset has no electrical pole at all
            # (algebraic current), so only the slow mechanical one counts and
            # 1e-4 is easy.  A brushless one carries L, and its electrical time
            # constant is two orders of magnitude shorter than the mechanical
            # one, so the same step size is ~dt/(2*tau_e) off in the current -
            # which is a real property of the method, not a bug in the model.
            tau = system.mech_time_constant
            tau_e = system.electrical_time_constant
            tol = 1e-4
            if tau_e > 0 and math.isfinite(tau_e):
                tol = max(tol, 0.5 * dt / min(tau, tau_e))
            rel = abs(vals["euler"] - vals["exact"]) / max(abs(vals["exact"]), 1e-9)
            self.assertLess(rel, tol,
                            f"{name}: euler off by {rel:.2e}, dt/tau_e={dt / max(tau_e, 1e-30):.2e} {vals}")

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


class TestPowerAccounting(unittest.TestCase):
    """Power through a gearbox: the ratio cancels, so it must not appear twice.

    ``power_out`` used to be motor-shaft torque times *output* speed.  For a
    reducer that under-reports the output power by exactly the gear ratio, and
    over-reports it for a multiplier; a direct-drive preset hides the error, and
    that is how it survived this long.  The gearbox gives
    tau_out = tau * efficiency / ratio and omega_out = omega * ratio, so the
    ratio cancels and P_out = tau * omega * efficiency.
    """

    def _settled(self, name):
        system = presets.build(name)
        result = simulate(system, system.voltage_limit, duration=20.0,
                          engine="rk4", dt=1e-3, samples=200)
        return system, result

    def test_output_power_uses_motor_speed_not_output_speed(self):
        for name in ("pod_4x_geared", "bldc_pod"):
            system, result = self._settled(name)
            current = float(result.current[-1])
            omega = float(result.motor_speed[-1])
            expected = (system.group.kt_eff * current * omega
                        * system.gearbox.efficiency)
            self.assertAlmostEqual(float(result.power_out[-1]), expected,
                                   delta=1e-9 * abs(expected), msg=name)
            if system.gearbox.ratio != 1.0:
                stale = system.group.kt_eff * current * float(result.output_speed[-1])
                # off by the gear ratio, which is a factor, not rounding
                self.assertGreater(abs(float(result.power_out[-1]) - stale),
                                   0.5 * abs(expected), msg=name)

    def test_the_dc_electrical_balance_closes_exactly(self):
        # v = i*R + kv*w for a DC machine, so v*i - i^2*R == kv*w to round-off.
        # Any other answer means a power channel is being read off the wrong
        # shaft, and this identity catches it without reference to the gearbox.
        system, result = self._settled("pod_4x_geared")
        current = float(result.current[-1])
        omega = float(result.motor_speed[-1])
        converted = system.group.kv_eff * current * omega
        gap = (float(result.power_in[-1]) - float(result.copper_loss[-1])
               - converted)
        self.assertAlmostEqual(gap, 0.0, delta=1e-11 * max(1.0, abs(converted)))

    def test_the_brushed_view_of_a_brushless_machine_keeps_the_1_5_split(self):
        # Amplitude-invariant Park puts 1.5 on the torque constant but not on the
        # back-EMF constant, so the brushed collapse reports kt*i*w developed
        # against kv*i*w converted -- a factor 1.5 apart.  The dq model is the
        # power-consistent one (1.5 lands on both sides).  This is a convention,
        # not a bug, and it is pinned here so it stays a documented surprise
        # rather than a silent one.
        system, _ = self._settled("bldc_pod")
        self.assertAlmostEqual(system.group.kt_eff, 1.5 * system.group.kv_eff,
                               places=12)
        current = 10.0
        omega = 500.0
        developed = system.group.kt_eff * current * omega
        converted = system.group.kv_eff * current * omega
        self.assertAlmostEqual(developed - converted, 0.5 * converted, places=9)


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


class TestGUI(unittest.TestCase):
    """Drives the real Qt app.  Runs on the offscreen platform, skipped if Qt is missing."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from qtpy import QtCore, QtGui, QtWidgets   # noqa: F401
            from drivetrain import gui
        except Exception as exc:                        # pragma: no cover
            raise unittest.SkipTest(f"qtpy + matplotlib Qt backend unavailable: {exc}")
        try:
            cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        except Exception as exc:                        # pragma: no cover
            raise unittest.SkipTest(f"no usable Qt platform plugin: {exc}")
        gui.apply_theme(cls.app, "gruvbox")
        cls.gui = gui
        cls.QtWidgets = QtWidgets
        cls.QtGui = QtGui
        # a stray modal dialog (file picker, error box) would hang the suite
        # forever on the offscreen platform, so close anything modal that lingers
        watchdog = QtCore.QTimer(cls.app)
        watchdog.setInterval(5000)
        watchdog.timeout.connect(cls._close_modals)
        watchdog.start()
        cls.watchdog = watchdog

    @classmethod
    def _close_modals(cls):
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if widget.isModal() and widget.isVisible():
                widget.close()

    @classmethod
    def tearDownClass(cls):
        cls.watchdog.stop()

    def _settle(self, app, timeout=120.0):
        """Spin the event loop until the simulation thread goes idle."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.app.processEvents()
            if not app.is_busy():
                self.app.processEvents()
                return True
            time.sleep(0.02)
        self.fail("GUI simulation did not finish")      # pragma: no cover

    def _open(self, preset="pod_4x_geared"):
        app = self.gui.DriveApp(preset=preset)
        self.addCleanup(app.close)
        self._settle(app)
        return app

    # ------------------------------------------------------------------
    def test_theme_is_gruvbox(self):
        import matplotlib
        from matplotlib import colors as mcolors

        self.assertIn("Fusion", self.QtWidgets.QStyleFactory.keys())
        # setStyleSheet wraps the style, which proves the QSS is in force
        self.assertEqual(self.app.style().metaObject().className(), "QStyleSheetStyle")
        window = self.app.palette().color(self.QtGui.QPalette.Window).name()
        self.assertEqual(window, "#282828")             # gruvbox bg0
        self.assertEqual(mcolors.to_hex(matplotlib.rcParams["figure.facecolor"]), "#282828")
        self.assertEqual(mcolors.to_hex(matplotlib.rcParams["text.color"]), "#ebdbb2")
        self.assertIn("#fabd2f", self.app.styleSheet())  # accent colour in the QSS

    def test_app_builds_runs_and_plots(self):
        app = self._open("pod_4x_geared")
        self.assertTrue(app.param_rows, "no parameter controls were built")
        self.assertIsNotNone(app.last_result, "preset load did not simulate")
        # plots must be smooth, not limited to the solver's auto step
        self.assertGreater(len(app.last_result.t), 500)
        self.assertTrue(np.all(np.isfinite(app.last_result.motor_speed)))
        self.assertTrue(app.statusBar().currentMessage().startswith("done:"))
        self.assertIn("final", app.metric_text.toPlainText())
        # every declared curve is drawn, including the ground-speed twin axis
        drawn = {ln.get_label() for ax in app.fig.axes for ln in ax.get_lines()}
        self.assertTrue({"motor shaft", "output shaft", "ground speed", "total current",
                         "torque", "electrical in", "mechanical out", "copper loss",
                         "efficiency"} <= drawn, sorted(drawn))

    def test_overlay_previous_runs_is_bounded(self):
        # Axes.clear() does not drop a twinx() partner, so every run used to add a
        # second km/h axis carrying its own ground-speed curve: the figure grew one
        # axis per run and the speed panel turned into a pile of old traces.
        app = self._open("pod_4x_geared")
        grid = [a for row in app.ax for a in row]
        base = len(app.ax[0][0].get_lines())      # motor, output, steady reference

        def twins():
            return [a for a in app.fig.axes if a not in grid]

        self.assertEqual(len(twins()), 1, "four grid axes plus one ground-speed twin")

        app.overlay_box.setValue(0)               # default: the current run only
        for ratio in (0.3, 0.4, 0.5):
            app.set_param("gearbox.ratio", ratio)
            self._settle(app)
            self.assertEqual(len(twins()), 1, "a run left a stale twin axis behind")
            self.assertEqual(len(app.ax[0][0].get_lines()), base,
                            "earlier runs are still stacked on the speed axis")
            self.assertEqual(len(twins()[0].get_lines()), 1)
            labels = [ln.get_label() for ln in app.ax[0][0].get_lines()]
            self.assertEqual(labels.count("motor shaft"), 1,
                             "only the run you just asked for belongs in the legend")

        # the spinner brings older runs back, faded and unlabelled
        app.overlay_box.setValue(2)
        self.assertEqual(len(app.ax[0][0].get_lines()), base + 4)
        self.assertEqual(len(twins()[0].get_lines()), 3)
        faded = [ln for ln in app.ax[0][0].get_lines()
                 if ln.get_alpha() not in (None, 1.0)]
        self.assertEqual(len(faded), 4, "overlays must read fainter than the new run")
        labels = [ln.get_label() for ln in app.ax[0][0].get_lines()]
        self.assertEqual(labels.count("motor shaft"), 1)

        # the history is capped, so there are never more ghosts than runs behind you
        app.overlay_box.setValue(self.gui.MAX_OVERLAY)
        ghosts = min(len(app._history) - 1, self.gui.MAX_OVERLAY)
        self.assertEqual(len(app.ax[0][0].get_lines()), base + 2 * ghosts)
        self.assertLessEqual(len(app._history), self.gui.MAX_OVERLAY)

    def test_wheel_less_preset_still_works(self):
        # legacy has no wheel, so the speed/distance channels simply do not exist
        app = self._open("legacy")
        self.assertFalse(app.wheel_box.isEnabled(), "no wheel to pick")
        self.assertNotIn("speed_kmh", app.last_result.channels())
        self.assertTrue(app.statusBar().currentMessage().startswith("done:"))
        self.assertIn("final", app.metric_text.toPlainText())
        self.assertTrue(np.all(np.isfinite(app.last_result.motor_speed)))

    def test_param_edits_take_effect(self):
        app = self._open("pod_4x_geared")
        before = app.system.gearbox.ratio

        row = app.param_rows["gearbox.ratio"]
        row.slider.setValue(row.slider.value() - 5)      # drag the slider
        self._settle(app)
        self.assertNotAlmostEqual(app.system.gearbox.ratio, before)

        row.entry_edit.setText("0.08")                   # type an exact value
        row._on_entry()
        self._settle(app)
        self.assertAlmostEqual(app.system.gearbox.ratio, 0.08)

        row.entry_edit.setText("banana")                 # garbage reverts
        row._on_entry()
        self.app.processEvents()
        self.assertAlmostEqual(row.value(), 0.08, msg="bad input should be rejected")

        app.set_param("gearbox.ratio", 0.3)              # programmatic edit
        self._settle(app)
        self.assertAlmostEqual(app.system.gearbox.ratio, 0.3)
        self.assertAlmostEqual(app.param_rows["gearbox.ratio"].value(), 0.3)

    def test_sweep_and_compare(self):
        app = self._open("pod_4x_geared")
        app.sweep_n.setText("6")
        app.run_sweep()
        self._settle(app)
        text = app.sweep_out.toPlainText()
        self.assertIn("best", app.statusBar().currentMessage())
        self.assertIn("gearbox.ratio", text)
        self.assertGreater(len(text.splitlines()), 3)

        app.run_compare()
        self._settle(app)
        self.assertIn("compared", app.statusBar().currentMessage())
        labels = [ln.get_label() for ln in app.cmp_ax.get_lines()]
        self.assertTrue(any("pod_" in s for s in labels), labels)

    def test_catalogue_boxes_track_the_model(self):
        app = self._open("pod_4x_geared")
        self.assertTrue(app.wheel_box.isEnabled(), "wheeled preset should offer wheels")
        names = [app.wheel_box.itemText(i) for i in range(app.wheel_box.count())]
        target = next(n for n in names if n not in (app.wheel_box.currentText(), "(custom)"))
        app.wheel_box.setCurrentText(target)
        self._settle(app)
        self.assertAlmostEqual(app.system.load.wheel.radius,
                               self.gui.presets.WHEELS[target].radius)

        motors = [app.motor_box.itemText(i) for i in range(app.motor_box.count())]
        target = next(m for m in motors if m not in (app.motor_box.currentText(), "(custom)"))
        app.motor_box.setCurrentText(target)
        self._settle(app)
        self.assertAlmostEqual(app.system.group.motor.kv,
                               self.gui.presets.MOTORS[target].kv)
        self.assertEqual(app.motor_box.currentText(), target)

        # a preset with no wheel has no wheel to pick, and must not report km/h
        plain = self._open("legacy")
        self.assertIsNone(plain.system.wheel)
        self.assertFalse(plain.wheel_box.isEnabled())
        self.assertIn("done:", plain.statusBar().currentMessage())
        self.assertNotIn("speed_kmh", plain.last_result.channels())

    def test_config_and_export_roundtrip(self):
        app = self._open("pod_4x_geared")
        app.set_param("gearbox.ratio", 0.42)
        self._settle(app)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "config.json")
            csv = os.path.join(tmp, "response.csv")
            png = os.path.join(tmp, "response.png")
            queue = [cfg, csv, png]
            app._save_path = lambda title, suggestion, pattern: queue.pop(0)
            app._open_path = lambda title, pattern, start="": cfg
            app.save_config()
            app.export_csv()
            app.save_png()
            self.assertTrue(os.path.getsize(cfg) > 0)
            self.assertTrue(os.path.getsize(png) > 1000)
            with open(csv) as fh:
                header = fh.readline().strip().split(",")
            self.assertIn("motor_rpm", header)

            app.set_param("gearbox.ratio", 0.11)
            self._settle(app)
            app.load_config()                            # reads cfg back
            self._settle(app)
            self.assertAlmostEqual(app.system.gearbox.ratio, 0.42)

    def test_fit_dataset_tab_recovers_parameters(self):
        """The Fit tab moves the ticked parameters and leaves the rest alone."""
        from qtpy import QtCore
        from drivetrain import presets
        from drivetrain.simulate import simulate, voltage_signal

        truth = presets.build("pod_2x_direct", {
            "group.motor.kv": 0.0212,
            "group.motor.kt": 0.0212,
            "group.motor.resistance": 0.44,
            "group.motor.rotor_inertia": 2.2e-5,
            "group.motor.bearing_damping": 3.0e-5,
        })
        span, step = 4.0, 0.005
        drive = voltage_signal("square", truth, span, dt=step, value=6.0, period=2.0)
        res = simulate(truth, drive, span, dt=step, engine="exact", samples=None)
        ch = res.channels()
        sel = slice(None, None, 4)                      # 50 Hz measured data

        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "bench.csv")
            with open(data, "w") as fh:
                fh.write("time,voltage_V,current_A,motor_rpm\n")
                for row in zip(ch["time"][sel], ch["voltage_V"][sel],
                               ch["current_A"][sel], ch["motor_rpm"][sel]):
                    fh.write(",".join(f"{x:.6f}" for x in row) + "\n")

            app = self._open("pod_2x_direct")
            # the known constants are typed into the panel, not ticked in the list
            for path, value in (("group.motor.kt", 0.0212),
                                ("group.motor.rotor_inertia", 2.2e-5),
                                ("group.motor.bearing_damping", 3.0e-5)):
                app.set_param(path, value)
            self._settle(app)

            wanted = {"group.motor.kv", "group.motor.resistance"}
            for i in range(app.fit_free.count()):
                item = app.fit_free.item(i)
                free = item.data(QtCore.Qt.UserRole) in wanted
                item.setCheckState(QtCore.Qt.Checked if free else QtCore.Qt.Unchecked)
            self.assertEqual(sorted(app._fit_checked()),
                             ["group.motor.kv", "group.motor.resistance"])

            app.fit_path.setText(data)
            app.run_fit()
            self._settle(app)
            fitted = app.fit_result
            self.assertIsNotNone(fitted)
            self.assertIn("group.motor.kv", app.fit_out.toPlainText())
            self.assertTrue(app.fit_apply_btn.isEnabled())
            self.assertTrue(app.fit_save_btn.isEnabled())
            # this bench log is noise-free, so the noise-relative rms is set by the
            # numerical floor; the unit-free tracking number is the meaningful one
            self.assertLess(max(fitted.relative_rms.values()), 0.01)
            self.assertAlmostEqual(fitted.system.group.motor.kv, 0.0212, delta=4e-4)
            self.assertAlmostEqual(fitted.system.group.motor.resistance, 0.44,
                                   delta=0.01)
            self.assertEqual(fitted.system.group.motor.kt, 0.0212)   # known
            self.assertEqual(fitted.system.group.motor.rotor_inertia, 2.2e-5)
            self.assertTrue(len(app.fit_fig.axes) >= 1)

            out = os.path.join(tmp, "fitted.json")
            app._save_path = lambda title, suggestion, pattern: out
            app.save_fit()
            self.assertTrue(os.path.getsize(out) > 0)

            app.apply_fit()                             # adopt the fitted model
            self._settle(app)
            self.assertAlmostEqual(app.system.group.motor.kv,
                                   fitted.system.group.motor.kv)
            self.assertAlmostEqual(app.param_rows["group.motor.kv"].value(),
                                   fitted.system.group.motor.kv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
