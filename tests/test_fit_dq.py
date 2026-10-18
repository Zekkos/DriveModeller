"""Tests for the brushless side of ``drivetrain.fit``: dq logs, three-phase
logs, and what each of them can and cannot identify.

The headline claim these tests pin down is the one the brushed collapse cannot
make: a log of ``i_d``, ``i_q`` and the applied ``v_d``/``v_q`` separates the
d-axis inductance from the q-axis inductance, because the two of them appear in
different voltage equations.  A brushed log -- one current, one voltage -- sees
only the q-axis one, and the fitter says so instead of fitting a number that no
measurement constrains.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from drivetrain import dq_sim, fit, presets
from drivetrain.exceptions import ConfigError
from drivetrain.simulate import simulate
from drivetrain.sweep import set_path

# a machine deliberately offset from the catalogue entry it is fitted from, so a
# recovered value cannot be the starting value by accident
TRUTH = {
    "group.motor.spec.flux_linkage": 2.1e-3,
    "group.motor.spec.resistance": 0.09,
    "group.motor.spec.inductance_d": 1.4e-4,
    "group.motor.spec.inductance_q": 1.9e-4,
    "group.motor.spec.bearing_damping": 2.5e-5,
}
DQ_FREE = [
    "group.motor.spec.resistance",
    "group.motor.spec.flux_linkage",
    "group.motor.spec.inductance_d",
    "group.motor.spec.inductance_q",
]
DQ_CHANNELS = {"current_d_A", "current_q_A", "voltage_d_V", "voltage_q_V", "motor_rad_s"}
BRUSHED_CHANNELS = {"current_A", "motor_rpm", "voltage_V"}


def base_system() -> object:
    """The catalogue machine the fit starts from."""
    return presets.build("bldc_pod")


def truth_system() -> object:
    system = presets.build("bldc_pod")
    for path, value in TRUTH.items():
        system = set_path(system, path, value)
    return system


def settled_dq_rows(seed: int = 7) -> dict[str, list[float]]:
    """Settled dq operating points spread over speed and torque.

    Spread over speed is what makes the inductances separable at all: the term
    that tells Ld from Lq is the back-EMF, and back-EMF is proportional to
    speed.  The currents are the inputs here and the two voltages are what the
    drive reports, so the voltages carry the residuals.
    """
    system = truth_system()
    spec = system.group.motor.spec
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {
        "current_d_A": [], "current_q_A": [], "motor_rpm": [],
        "voltage_d_V": [], "voltage_q_V": [],
    }
    for w in (30.0, 120.0, 400.0, 700.0, 1000.0, 1400.0):
        limit = spec.torque_limit(w)
        for frac in (0.15, 0.45, 0.85):
            i_d, i_q, _, _ = spec.current_pair(frac * limit, w)
            v_d, v_q = spec.voltage_steady(i_d, i_q, w)
            rows["current_d_A"].append(i_d + rng.normal(0, 0.02))
            rows["current_q_A"].append(i_q + rng.normal(0, 0.02))
            rows["motor_rpm"].append(w * 60.0 / (2.0 * math.pi) + rng.normal(0, 20.0))
            rows["voltage_d_V"].append(v_d + rng.normal(0, 0.005))
            rows["voltage_q_V"].append(v_q + rng.normal(0, 0.005))
    return rows


def dq_run_rows(seed: int = 11) -> dict[str, np.ndarray]:
    """A transient dq log with both axes stepped twice.

    Each inductance only shows up as the current transient on its own axis --
    Ld in the d-axis rise, Lq in the q-axis one -- so an excitation that moves
    only one of them leaves the other weakly determined.  Two steps per axis
    also re-excites the electrical pole late enough in the log that the fit is
    not reading one edge."""
    from drivetrain.dq_sim import DqState, simulate_dq

    truth = truth_system()
    t = np.arange(800) * 1.0e-4
    v_d = np.where(t >= 0.05, -0.4, np.where(t >= 0.01, -1.2, 0.0))
    v_q = np.where(t >= 0.02, 3.0, np.where(t >= 0.005, 1.2, 0.0))
    res = simulate_dq(
        truth, v_d=v_d, v_q=v_q, duration=float(t[-1]), t=t, engine="rk4",
        samples=None, initial=DqState(0.0, 0.0, 0.0, 0.0),
    )
    ch = res.channels()
    rng = np.random.default_rng(seed)
    return {
        "time_s": t,
        "voltage_d_V": v_d,
        "voltage_q_V": v_q,
        "current_d_A": ch["current_d_A"] + rng.normal(0, 0.03, t.size),
        "current_q_A": ch["current_q_A"] + rng.normal(0, 0.03, t.size),
        "motor_rad_s": ch["motor_rad_s"] + rng.normal(0, 1.0, t.size),
    }


class VocabularyTests(unittest.TestCase):
    def test_dq_channels_are_model_channels(self) -> None:
        for channel in fit.DQ_CHANNELS:
            self.assertIn(channel, fit.DATA_CHANNELS)
        self.assertIn("rotor_angle_rad", fit.DATA_CHANNELS)

    def test_phase_columns_are_not_model_channels(self) -> None:
        # the Park transform turns them into dq channels at load time, so the
        # model never sees a phase quantity
        for channel in fit.PHASE_CHANNELS:
            self.assertNotIn(channel, fit.DATA_CHANNELS)

    def test_electrical_angle_is_a_coordinate_not_a_channel(self) -> None:
        self.assertNotIn(fit.ELECTRICAL_ANGLE_CHANNEL, fit.DATA_CHANNELS)
        self.assertNotIn(fit.ELECTRICAL_ANGLE_CHANNEL, fit.STEADY_CHANNELS)

    def test_phase_and_angle_headers_are_recognised(self) -> None:
        self.assertEqual(fit._canonical("current_a_A"), "current_a_A")
        self.assertEqual(fit._canonical("CURRENT_B_A"), "current_b_A")
        self.assertEqual(fit._canonical("voltage_c_V"), "voltage_c_V")
        self.assertEqual(fit._canonical("rotor_angle_electrical_rad"),
                         fit.ELECTRICAL_ANGLE_CHANNEL)
        self.assertEqual(fit._canonical("theta_e"), fit.ELECTRICAL_ANGLE_CHANNEL)
        self.assertEqual(fit._canonical("id"), "current_d_A")
        self.assertEqual(fit._canonical("vq"), "voltage_q_V")

    def test_input_channel_tuples(self) -> None:
        self.assertEqual(fit.DQ_INPUT_CHANNELS, ("voltage_d_V", "voltage_q_V"))
        self.assertEqual(fit.DQ_STEADY_INPUT_CHANNELS, ("current_d_A", "current_q_A"))


class PhaseLoadTests(unittest.TestCase):
    def test_three_phase_currents_become_dq(self) -> None:
        theta = np.zeros(4)
        rows = {
            "rotor_angle_electrical_rad": theta,
            "current_a_A": [14.0, 7.0, -7.0, -14.0],
            "current_b_A": [-7.0, -14.0, 14.0, 7.0],
            "current_c_A": [-7.0, 7.0, -7.0, 7.0],
            "motor_rpm": [100.0, 200.0, 300.0, 400.0],
        }
        ds = fit.Dataset.from_rows(rows, name="phases")
        block = ds.blocks[0]
        self.assertIn("current_d_A", block.columns)
        self.assertIn("current_q_A", block.columns)
        # a balanced sine with its peak on phase a at theta_e = 0 sits entirely
        # on the d axis under the amplitude-invariant transform
        self.assertAlmostEqual(float(block.columns["current_d_A"][0]), 14.0, places=9)
        self.assertAlmostEqual(float(block.columns["current_q_A"][0]), 0.0, places=9)
        # the phase columns are consumed rather than left lying around
        for name in fit.PHASE_CHANNELS:
            self.assertNotIn(name, block.columns)
        self.assertTrue(any("Park" in note for note in ds.notes))

    def test_two_phase_log_is_refused(self) -> None:
        rows = {
            "rotor_angle_electrical_rad": [0.0, 0.1],
            "current_a_A": [1.0, 2.0],
            "current_b_A": [-1.0, -2.0],
            "motor_rpm": [100.0, 200.0],
        }
        with self.assertRaises(ConfigError) as caught:
            fit.Dataset.from_rows(rows, name="two-phase")
        self.assertIn("all three phases", str(caught.exception))

    def test_phase_and_dq_columns_of_a_kind_collide(self) -> None:
        rows = {
            "rotor_angle_electrical_rad": [0.0, 0.1],
            "current_a_A": [1.0, 2.0],
            "current_b_A": [-1.0, -2.0],
            "current_c_A": [0.0, 0.0],
            "current_d_A": [1.0, 2.0],
            "current_q_A": [0.0, 0.0],
            "motor_rpm": [100.0, 200.0],
        }
        with self.assertRaises(ConfigError) as caught:
            fit.Dataset.from_rows(rows, name="both")
        self.assertIn("one or the other", str(caught.exception))

    def test_phase_columns_need_the_electrical_angle(self) -> None:
        rows = {
            "rotor_angle_rad": [0.0, 0.1],
            "current_a_A": [1.0, 2.0],
            "current_b_A": [-1.0, -2.0],
            "current_c_A": [0.0, 0.0],
            "motor_rpm": [100.0, 200.0],
        }
        with self.assertRaises(ConfigError) as caught:
            fit.Dataset.from_rows(rows, name="mech-only")
        message = str(caught.exception)
        self.assertIn(fit.ELECTRICAL_ANGLE_CHANNEL, message)
        self.assertIn("mechanical angle is not enough", message)

    def test_zero_sequence_is_reported_not_fitted(self) -> None:
        theta = np.linspace(0.0, 1.0, 6)
        rows = {
            "rotor_angle_electrical_rad": theta,
            "current_a_A": 1.0 + 5.0 * np.cos(theta),
            "current_b_A": 1.0 + 5.0 * np.cos(theta - 2.0 * math.pi / 3.0),
            "current_c_A": 1.0 + 5.0 * np.cos(theta + 2.0 * math.pi / 3.0),
            "motor_rpm": np.linspace(100.0, 700.0, 6),
        }
        ds = fit.Dataset.from_rows(rows, name="zero-seq")
        self.assertTrue(any("zero sequence" in note for note in ds.notes))

    def test_both_angles_report_the_pole_count(self) -> None:
        theta = np.linspace(0.0, 0.4, 40)
        rows = {
            "rotor_angle_rad": theta,
            "rotor_angle_electrical_rad": 7.0 * theta,
            "current_a_A": 5.0 * np.cos(7.0 * theta),
            "current_b_A": 5.0 * np.cos(7.0 * theta - 2.0 * math.pi / 3.0),
            "current_c_A": 5.0 * np.cos(7.0 * theta + 2.0 * math.pi / 3.0),
            "motor_rpm": np.linspace(100.0, 700.0, 40),
        }
        ds = fit.Dataset.from_rows(rows, name="both-angles")
        notes = " ".join(ds.notes)
        self.assertIn("pole count", notes)
        self.assertIn("7", notes)

    def test_angles_are_unwrapped(self) -> None:
        theta = np.array([6.0, 6.2, 0.05, 0.25, 0.45])  # crosses 2 pi
        rows = {
            "rotor_angle_electrical_rad": theta,
            "current_a_A": [5.0] * 5,
            "current_b_A": [-2.5] * 5,
            "current_c_A": [-2.5] * 5,
            "motor_rpm": [100.0] * 5,
        }
        ds = fit.Dataset.from_rows(rows, name="wrap")
        kept = ds.blocks[0].columns[fit.ELECTRICAL_ANGLE_CHANNEL]
        self.assertTrue(np.all(np.diff(kept) > 0.0))


class IdentifiabilityTests(unittest.TestCase):
    def test_inductances_are_inert_under_the_brushed_collapse(self) -> None:
        base = base_system()
        for channels in (None, BRUSHED_CHANNELS, {"current_d_A", "current_q_A"}):
            reason = fit.structural_inert(base, "group.motor.spec.inductance_d", channels)
            self.assertIsNotNone(reason, f"channels={channels}")
        self.assertIsNone(
            fit.structural_inert(base, "group.motor.spec.inductance_d",
                                 DQ_CHANNELS)
        )

    def test_each_inductance_needs_its_own_voltage_axis(self) -> None:
        base = base_system()
        q_only = {"current_d_A", "current_q_A", "voltage_q_V"}
        d_only = {"current_d_A", "current_q_A", "voltage_d_V"}
        # Lq appears only in v_d, Ld only in v_q: each needs the other equation
        self.assertTrue(fit._dq_live(base, "group.motor.spec.inductance_d", q_only))
        self.assertFalse(fit._dq_live(base, "group.motor.spec.inductance_d", d_only))
        self.assertTrue(fit._dq_live(base, "group.motor.spec.inductance_q", d_only))
        self.assertFalse(fit._dq_live(base, "group.motor.spec.inductance_q", q_only))
        self.assertFalse(fit._dq_live(base, "group.motor.spec.inductance_d", set()))

    def test_inductance_q_is_never_reported_inert(self) -> None:
        # inductance_q is the inductance the i_d=0 collapse calls L, so a brushed
        # current rise constrains it; only inductance_d is inert without dq data
        base = base_system()
        for channels in (None, BRUSHED_CHANNELS, {"current_d_A", "current_q_A"},
                         DQ_CHANNELS):
            self.assertIsNone(
                fit.structural_inert(base, "group.motor.spec.inductance_q", channels)
            )

    def test_inductance_d_needs_both_dq_currents(self) -> None:
        base = base_system()
        self.assertIsNotNone(
            fit.structural_inert(base, "group.motor.spec.inductance_d", None)
        )
        self.assertIsNotNone(
            fit.structural_inert(base, "group.motor.spec.inductance_d",
                                 BRUSHED_CHANNELS)
        )
        # currents alone are not enough: they are inputs, and the inductances only
        # appear in the voltage equations
        self.assertIsNotNone(
            fit.structural_inert(base, "group.motor.spec.inductance_d",
                                 {"current_d_A", "current_q_A"})
        )
        self.assertIsNone(
            fit.structural_inert(base, "group.motor.spec.inductance_d",
                                 {"current_d_A", "current_q_A", "voltage_q_V"})
        )

    def test_losses_stay_inert_even_with_dq_channels(self) -> None:
        base = base_system()
        self.assertIsNotNone(
            fit.structural_inert(base, "group.motor.spec.core_loss_coefficient",
                                 DQ_CHANNELS)
        )

    def test_free_all_follows_the_channels(self) -> None:
        base = base_system()
        with_dq = fit._expand_free("all", base, {}, [], DQ_CHANNELS)
        with_brushed = fit._expand_free("all", base, {}, [], BRUSHED_CHANNELS)
        self.assertIn("group.motor.spec.inductance_d", with_dq)
        self.assertIn("group.motor.spec.inductance_q", with_dq)
        self.assertNotIn("group.motor.spec.inductance_d", with_brushed)
        # the brushed collapse's single current sits on the q axis, so a brushed
        # current rise really does constrain Lq
        self.assertIn("group.motor.spec.inductance_q", with_brushed)
        self.assertNotIn("group.motor.spec.core_loss_coefficient", with_dq)

    def test_brushed_inductance_maps_to_the_q_axis(self) -> None:
        mapping = fit.dq_motor_map(base_system())
        self.assertEqual(mapping["group.motor.inductance"], "group.motor.spec.inductance_q")


class SettledDqFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = fit.Dataset.from_rows(settled_dq_rows(), name="dq-settled")
        cls.result = fit.fit(cls.dataset, system=base_system(), free=DQ_FREE,
                             engine="rk4")

    def test_recovers_resistance_flux_and_both_inductances(self) -> None:
        spec = truth_system().group.motor.spec
        fitted = {p.path.split(".")[-1]: p for p in self.result.parameters}
        self.assertEqual(set(fitted), {"resistance", "flux_linkage",
                                       "inductance_d", "inductance_q"})
        for name, tolerance in (("resistance", 0.02), ("flux_linkage", 0.01),
                                ("inductance_d", 0.03), ("inductance_q", 0.03)):
            got = getattr(self.result.system.group.motor.spec, name)
            want = getattr(spec, name)
            self.assertLess(abs(got - want) / abs(want), tolerance,
                            f"{name}: {got:.6g} against {want:.6g}")

    def test_the_two_inductances_come_out_different(self) -> None:
        # this is the whole point of the dq log: one current and one voltage
        # cannot tell Ld from Lq, and this one can
        spec = self.result.system.group.motor.spec
        self.assertLess(spec.inductance_d, spec.inductance_q)
        gap = abs(spec.inductance_q - spec.inductance_d) / spec.inductance_d
        self.assertGreater(gap, 0.25)

    def test_currents_are_inputs_so_their_residuals_are_zero(self) -> None:
        # the dq currents are inputs here, so the model reproduces them exactly;
        # the information lives in the voltages it has to predict
        self.assertEqual(self.result.rms["current_d_A"], 0.0)
        self.assertEqual(self.result.rms["current_q_A"], 0.0)
        for channel in ("voltage_d_V", "voltage_q_V"):
            self.assertIn(channel, self.result.rms)
            self.assertGreater(self.result.rms[channel], 0.0)
        self.assertLess(self.result.rms["motor_rpm"], 1e-9)

    def test_steady_dq_power_out_is_taken_at_motor_speed(self) -> None:
        system = truth_system()
        block = self.dataset.blocks[0]
        channels = fit._dq_steady_channels(system, block)
        w = channels["motor_rad_s"]
        expected = channels["torque_Nm"] * w * system.gearbox.efficiency
        np.testing.assert_allclose(channels["power_out_W"], expected, rtol=1e-12)
        # a reducer cancels out of the ratio: output speed times efficiency would
        # be low by the gear ratio
        wrong = channels["torque_Nm"] * channels["output_rad_s"] * system.gearbox.efficiency
        self.assertGreater(np.max(np.abs(wrong)), 0.0)
        np.testing.assert_allclose(
            channels["power_out_W"] / wrong, 1.0 / system.gearbox.ratio, rtol=1e-9)

    def test_dq_steady_voltage_matches_the_hypot_channel(self) -> None:
        channels = fit._dq_steady_channels(truth_system(), self.dataset.blocks[0])
        np.testing.assert_allclose(
            channels["voltage_V"],
            np.hypot(channels["voltage_d_V"], channels["voltage_q_V"]),
            rtol=1e-12)


class TransientDqFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = fit.Dataset.from_rows(dq_run_rows(), name="dq-run")
        # engine="exact" has no closed form for this plant, and the fit is
        # expected to say so and integrate instead
        cls.result = fit.fit(cls.dataset, system=base_system(), free=DQ_FREE,
                             engine="exact")

    def test_replay_recovers_the_machine(self) -> None:
        spec = truth_system().group.motor.spec
        # tolerances from the sweep in docs/fitting.md: with both axes stepped the
        # four numbers come back inside ~1% on either seed, and the slack here is
        # for the solver step the fit happens to choose, not for the physics
        for name, tolerance in (("resistance", 0.02), ("flux_linkage", 0.03),
                                ("inductance_d", 0.05), ("inductance_q", 0.05)):
            got = getattr(self.result.system.group.motor.spec, name)
            want = getattr(spec, name)
            self.assertLess(abs(got - want) / abs(want), tolerance,
                            f"{name}: {got:.6g} against {want:.6g}")

    def test_a_clean_log_replays_exactly(self) -> None:
        """With the right constants the replay is exact, not merely close.

        The model grid used to stop one step short of the log's last sample, and
        ``np.interp`` does not extrapolate, so that final residual was the model
        clamped at its own end: 1.6e-2 A of error sitting in an otherwise perfect
        replay.  A clean log is the only place that shows up as zero rather than
        as noise, which is why this one carries no noise.
        """
        from drivetrain.dq_sim import DqState, simulate_dq

        truth = truth_system()
        t = np.arange(400) * 2.5e-4
        v_d = np.where(t >= 0.05, -0.4, np.where(t >= 0.01, -1.2, 0.0))
        v_q = np.where(t >= 0.02, 3.0, np.where(t >= 0.005, 1.2, 0.0))
        res = simulate_dq(
            truth, v_d=v_d, v_q=v_q, duration=float(t[-1]), t=t, engine="rk4",
            samples=None, initial=DqState(0.0, 0.0, 0.0, 0.0),
        )
        ch = res.channels()
        dataset = fit.Dataset.from_rows(
            {"time_s": t, "voltage_d_V": v_d, "voltage_q_V": v_q,
             "current_d_A": ch["current_d_A"], "current_q_A": ch["current_q_A"],
             "motor_rad_s": ch["motor_rad_s"]},
            name="clean-run",
        )
        self.assertEqual(len(dataset.blocks), 1)
        block = dataset.blocks[0]
        self.assertEqual(block.time.size, t.size)
        channels, model_t = fit._dq_transient_channels(
            truth, block, dt=None, max_steps=100_000, engine="rk4",
        )
        self.assertAlmostEqual(float(model_t[-1]), float(block.time[-1]), places=12)
        for name in ("current_d_A", "current_q_A", "motor_rad_s"):
            predicted = fit._sample(channels[name], model_t, block.time)
            worst = float(np.max(np.abs(predicted - block.columns[name])))
            self.assertLess(worst, 1e-12, f"{name}: worst error {worst:.3e}")

    def test_the_brushed_exact_engine_falls_back_to_rk4(self) -> None:
        # the dq plant has no closed form, so 'exact' cannot mean 'exact' here;
        # the report still echoes what was asked for, which is why this goes
        # through the prediction path rather than the result
        channels, _ = fit._dq_transient_channels(
            base_system(), self.dataset.blocks[0], dt=None, max_steps=100_000,
            engine="exact",
        )
        self.assertIn("current_d_A", channels)
        self.assertEqual(self.result.engine, "exact")

    def test_solver_step_respects_the_speed_dependent_stability_limit(self) -> None:
        spec = truth_system().group.motor.spec
        top = float(np.max(np.abs(self.dataset.blocks[0].columns["motor_rad_s"])))
        self.assertLessEqual(self.result.dt, dq_sim.max_step_time(spec, top))

    def test_a_log_too_coarse_for_the_stable_step_is_refused(self) -> None:
        dataset = fit.Dataset.from_rows(dq_run_rows(), name="dq-run")
        with self.assertRaises(ConfigError) as caught:
            fit.fit(dataset, system=base_system(), free=["group.motor.spec.resistance"],
                    engine="rk4", max_steps=60)
        message = str(caught.exception)
        self.assertIn("stable step", message)
        self.assertIn("max_steps=60", message)


class DqErrorPathTests(unittest.TestCase):
    def test_settled_dq_rows_need_a_speed(self) -> None:
        rows = {"current_d_A": [-1.0, -2.0], "current_q_A": [5.0, 8.0],
                "voltage_d_V": [0.1, 0.2], "voltage_q_V": [1.0, 2.0]}
        dataset = fit.Dataset.from_rows(rows, name="no-speed")
        with self.assertRaises(ConfigError) as caught:
            fit.fit(dataset, system=base_system(),
                    free=["group.motor.spec.resistance"], engine="rk4")
        message = str(caught.exception)
        self.assertIn("need a speed channel", message)
        self.assertIn("proportional to speed", message)

    def test_dq_timeseries_needs_the_dq_voltage(self) -> None:
        rows = {"time_s": [0.0, 0.1, 0.2], "current_d_A": [1.0, 1.2, 1.1],
                "current_q_A": [2.0, 2.2, 2.1], "motor_rad_s": [10.0, 20.0, 25.0]}
        dataset = fit.Dataset.from_rows(rows, name="no-vdq")
        self.assertTrue(any("voltage_d_V" in note for note in dataset.notes))
        with self.assertRaises(ConfigError) as caught:
            fit.fit(dataset, system=base_system(),
                    free=["group.motor.spec.resistance"], engine="rk4")
        message = str(caught.exception)
        self.assertIn("voltage_d_V and voltage_q_V", message)
        self.assertIn("brushed collapse", message)

    def test_settled_rows_need_a_voltage_or_dq_currents(self) -> None:
        rows = {"voltage_d_V": [0.1, 0.2], "voltage_q_V": [1.0, 2.0],
                "motor_rpm": [100.0, 200.0]}
        with self.assertRaises(ConfigError) as caught:
            fit.Dataset.from_rows(rows, name="no-currents")
        message = str(caught.exception)
        self.assertIn("current_d_A", message)
        self.assertIn("current_q_A", message)


class SteadyPowerOutTests(unittest.TestCase):
    """The fitter's settled-point prediction must account for power the same way
    ``simulate`` does: a reducer cancels out of the ratio, its efficiency does not."""

    def test_steady_channels_match_the_simulated_settled_point(self) -> None:
        for name in ("pod_4x_geared", "pod_2x_direct"):
            system = presets.build(name)
            predicted = fit._steady_channels(system, np.array([12.0]))
            channels = simulate(system, 12.0, 30.0, dt=1.0 / 400.0, samples=2000).channels()
            for channel in ("power_out_W", "power_in_W"):
                got = float(predicted[channel][0])
                want = float(channels[channel][-1])
                self.assertLess(abs(got - want) / abs(want), 1e-4,
                                f"{name} {channel}: {got} against {want}")


if __name__ == "__main__":
    unittest.main()
