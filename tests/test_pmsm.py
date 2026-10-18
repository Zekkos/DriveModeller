"""Tests for the brushless (PMSM / BLDC) machine model.

The tests are grouped by what they establish:

* conventions - the amplitude-invariant dq set, and the modulation schemes
* analytic points - values that are known in closed form
* equivalence - a non-salient machine with ``i_d = 0`` must reproduce the
  brushed DC model that the rest of the repo already uses
* MTPA, limits and the four-quadrant envelope
* field weakening, including the asymmetry between driving and braking
* feed-forward, losses, temperature drift, validation

The strongest test here is ``test_the_current_angle_is_optimal``: it brute
forces the minimum-current angle on a fine grid and checks that the analytic
search matches it.  That test is what caught two real bugs - a golden-section
refinement inside a ``-inf`` valley, and solving the braking quadrant with the
driving ellipse.
"""

from __future__ import annotations

import math
import unittest

from drivetrain.exceptions import ConfigError
from drivetrain.motor import MotorModel
from drivetrain.pmsm import (
    PMSMSpec,
    STICTION_BAND,
    phase_voltage_limit,
)

TWO_PI = 2.0 * math.pi


def _non_salient(**overrides) -> PMSMSpec:
    """A surface-magnet machine: Ld == Lq, so MTPA is exactly i_d = 0."""
    kwargs = dict(inductance_d=2.0e-4, inductance_q=2.0e-4, flux_linkage=1.0e-3,
                  pole_pairs=4, resistance=0.5, current_limit=30.0,
                  voltage_limit=24.0, bearing_damping=0.0, static_friction=0.0)
    kwargs.update(overrides)
    return PMSMSpec(**kwargs)


class ConventionTests(unittest.TestCase):
    """The dq convention is a choice; pin it so a change has to be deliberate."""

    def test_constants_follow_the_amplitude_invariant_set(self):
        m = PMSMSpec()
        self.assertAlmostEqual(m.torque_constant, 1.5 * m.pole_pairs * m.flux_linkage)
        self.assertAlmostEqual(m.back_emf_constant, m.pole_pairs * m.flux_linkage)
        # 3/2, not 1: the amplitude-invariant set puts 1.5 in the torque and
        # power, and leaves back-EMF alone.  A datasheet that quotes kt == kv is
        # using the torque-producing-current convention instead, and mixing the
        # two costs a factor of 1.5 in the current.
        self.assertAlmostEqual(m.torque_constant / m.back_emf_constant, 1.5)

    def test_electrical_speed_and_rpm_use_the_pole_pairs(self):
        m = PMSMSpec(pole_pairs=7)
        self.assertAlmostEqual(m.electrical_speed(10.0), 70.0)
        # rpm() is the shaft, i.e. what a datasheet and a tacho quote; the
        # electrical rate is pole_pairs times that
        self.assertAlmostEqual(m.rpm(10.0), 10.0 * 60 / TWO_PI)
        self.assertAlmostEqual(m.rpm(0.0), 0.0)

    def test_phase_voltage_limit_per_modulation_scheme(self):
        self.assertAlmostEqual(phase_voltage_limit(12.0, "svpwm"), 12.0 / math.sqrt(3.0))
        self.assertAlmostEqual(phase_voltage_limit(12.0, "space_vector"), 12.0 / math.sqrt(3.0))
        self.assertAlmostEqual(phase_voltage_limit(12.0, "thi"), 12.0 / math.sqrt(3.0))
        self.assertAlmostEqual(phase_voltage_limit(12.0, "sinusoidal"), 6.0)
        self.assertAlmostEqual(phase_voltage_limit(12.0), 12.0 / math.sqrt(3.0))  # default

    def test_phase_voltage_limit_rejects_nonsense(self):
        with self.assertRaises(ConfigError):
            phase_voltage_limit(0.0)
        with self.assertRaises(ConfigError):
            phase_voltage_limit(12.0, "six_step")

    def test_base_speed_is_back_emf_against_the_rail(self):
        m = PMSMSpec(flux_linkage=1.8e-3, pole_pairs=7, voltage_limit=6.9282)
        self.assertAlmostEqual(m.base_speed(), 6.9282 / (7 * 1.8e-3))
        self.assertAlmostEqual(m.base_speed(12.0), 12.0 / (7 * 1.8e-3))


class AnalyticPointTests(unittest.TestCase):
    """Closed-form values the model has to reproduce exactly."""

    def setUp(self):
        self.m = PMSMSpec()

    def test_torque_is_linear_in_i_q_when_i_d_is_zero(self):
        self.assertAlmostEqual(self.m.torque_at(0.0, 3.7), self.m.torque_constant * 3.7)

    def test_reluctance_term_is_the_only_deviation(self):
        i_d, i_q = -4.0, 6.0
        expected = (self.m.torque_constant * i_q
                    + 1.5 * self.m.pole_pairs
                    * (self.m.inductance_d - self.m.inductance_q) * i_d * i_q)
        self.assertAlmostEqual(self.m.torque_at(i_d, i_q), expected)
        # Lq > Ld and i_d < 0: the reluctance term adds torque, which is why
        # MTPA for a salient machine biases i_d negative.
        self.assertGreater(self.m.torque_at(i_d, i_q), self.m.torque_constant * i_q)

    def test_no_q_current_means_no_torque(self):
        self.assertAlmostEqual(self.m.torque_at(-5.0, 0.0), 0.0)

    def test_back_emf_appears_only_on_q_at_zero_current(self):
        w = 100.0
        v_d, v_q = self.m.voltage_steady(0.0, 0.0, w)
        self.assertAlmostEqual(v_d, 0.0)
        self.assertAlmostEqual(v_q, self.m.electrical_speed(w) * self.m.flux_linkage)

    def test_at_standstill_only_the_resistive_drop_survives(self):
        v_d, v_q = self.m.voltage_steady(-2.0, 5.0, 0.0)
        self.assertAlmostEqual(v_d, self.m.resistance * -2.0)
        self.assertAlmostEqual(v_q, self.m.resistance * 5.0)

    def test_cross_coupling_terms(self):
        w, i_d, i_q = 80.0, -3.0, 4.0
        we = self.m.electrical_speed(w)
        v_d, v_q = self.m.voltage_steady(i_d, i_q, w)
        self.assertAlmostEqual(v_d, self.m.resistance * i_d - we * self.m.inductance_q * i_q)
        self.assertAlmostEqual(v_q, self.m.resistance * i_q
                               + we * (self.m.inductance_d * i_d + self.m.flux_linkage))

    def test_derivatives_are_the_transient_dq_equations(self):
        m = PMSMSpec(current_d=-1.0, current_q=2.0, rot_vel=50.0)
        we = m.electrical_speed(m.rot_vel)
        v_d, v_q = 3.0, 7.0
        d_id, d_iq, dw = m.derivatives(v_d, v_q)
        self.assertAlmostEqual(d_id, (v_d - m.resistance * m.current_d
                                      + we * m.inductance_q * m.current_q) / m.inductance_d)
        self.assertAlmostEqual(d_iq, (v_q - m.resistance * m.current_q
                                      - we * (m.inductance_d * m.current_d + m.flux_linkage))
                               / m.inductance_q)
        self.assertAlmostEqual(dw, (m.torque_at(m.current_d, m.current_q)
                                    - m.loss_torque(m.rot_vel)) / m.rotor_inertia)

    def test_stiction_holds_the_shaft_at_standstill(self):
        m = PMSMSpec(static_friction=0.05)
        self.assertAlmostEqual(m.acceleration(0.04, 0.0), 0.0)
        self.assertGreater(m.acceleration(0.06, 0.0), 0.0)
        self.assertAlmostEqual(m.loss_torque(0.0), 0.05)
        self.assertLess(STICTION_BAND, 1.0e-3)

    def test_loss_torque_collects_the_loss_mechanisms(self):
        m = PMSMSpec(bearing_damping=1.0e-3, windage_coefficient=1.0e-7,
                     core_loss_coefficient=2.0e-8)
        w = 200.0
        expected = (1.0e-3 * w + 1.0e-7 * w * w
                    + 2.0e-8 * m.pole_pairs ** 2 * w)
        self.assertAlmostEqual(m.loss_torque(w), expected)
        self.assertAlmostEqual(m.loss_torque(-w), expected)  # always opposes motion


class DCModelEquivalenceTests(unittest.TestCase):
    """A non-salient machine driven at i_d = 0 *is* the brushed model."""

    def setUp(self):
        self.pmsm = _non_salient()
        self.kt = 1.5 * self.pmsm.pole_pairs * self.pmsm.flux_linkage
        self.kv = self.pmsm.pole_pairs * self.pmsm.flux_linkage
        self.dc = MotorModel(kv=self.kv, kt=self.kt,
                             resistance=self.pmsm.resistance,
                             max_current=self.pmsm.current_limit,
                             max_voltage=self.pmsm.voltage_limit)

    def test_constants_map_one_to_one(self):
        self.assertAlmostEqual(self.pmsm.torque_constant, self.dc.kt)
        self.assertAlmostEqual(self.pmsm.back_emf_constant, self.dc.kv)
        self.assertAlmostEqual(self.pmsm.resistance, self.dc.resistance)

    def test_current_and_torque_agree_at_every_speed(self):
        for voltage in (6.0, 12.0, 24.0):
            for w in (0.0, 100.0, 500.0, 1000.0):
                i_q = (voltage - self.kv * w) / self.pmsm.resistance
                self.assertAlmostEqual(self.pmsm.torque_at(0.0, i_q),
                                       self.dc.torque_at(voltage, w), places=12)
                self.assertAlmostEqual(i_q, self.dc.current_at(voltage, w), places=12)

    def test_no_load_speed_agrees(self):
        self.assertAlmostEqual(self.pmsm.no_load_speed(12.0), 12.0 / self.kv)
        self.assertAlmostEqual(self.pmsm.no_load_speed(12.0), 12.0 / self.dc.kv)

    def test_mtpa_is_exactly_i_d_zero_when_not_salient(self):
        self.assertFalse(self.pmsm.is_salient)
        kt = self.pmsm.torque_constant
        for tau in (0.25 * kt, 5.0 * kt, 20.0 * kt):
            i_d, i_q = self.pmsm.mtpa_pair(tau)
            self.assertEqual(i_d, 0.0)
            self.assertAlmostEqual(i_q, tau / kt)

    def test_mtpa_current_scales_linearly_with_torque(self):
        _, i1 = self.pmsm.mtpa_pair(3.0)
        _, i2 = self.pmsm.mtpa_pair(9.0)
        self.assertAlmostEqual(i2, 3.0 * i1)


class MTPATests(unittest.TestCase):
    """Minimum torque per ampere, i.e. the smallest current for a torque."""

    def test_salient_machine_biases_i_d_negative(self):
        m = PMSMSpec()
        self.assertTrue(m.is_salient)
        i_d, i_q = m.mtpa_pair(0.2)
        self.assertLess(i_d, 0.0)
        self.assertGreater(i_q, 0.0)
        self.assertAlmostEqual(m.torque_at(i_d, i_q), 0.2)

    def test_mtpa_beats_every_other_angle_at_the_same_current(self):
        m = PMSMSpec()
        i_d, i_q = m.mtpa_pair(0.15)
        i_mag = math.hypot(i_d, i_q)
        best = m.torque_at(i_d, i_q)
        for frac in (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
            other_d = frac * i_mag
            other_q = math.sqrt(max(0.0, i_mag * i_mag - other_d * other_d))
            self.assertLessEqual(m.torque_at(other_d, other_q), best * (1.0 + 1.0e-9))

    def test_mtpa_reports_unreachable_torque_as_infinite_current(self):
        m = _non_salient(current_limit=5.0)
        i_d, i_q = m.mtpa_pair(1000.0)
        self.assertTrue(math.isinf(i_q))
        self.assertEqual(i_d, 0.0)

    def test_zero_torque_needs_zero_current_on_the_mtpa_branch(self):
        self.assertEqual(PMSMSpec().mtpa_pair(0.0), (0.0, 0.0))

    def test_mtpa_is_antisymmetric_in_torque(self):
        # reversing the torque reverses i_q but keeps i_d negative: the
        # reluctance term follows the sign of i_q, so the same field bias helps
        # in both directions of rotation
        m = PMSMSpec()
        i_d, i_q = m.mtpa_pair(0.2)
        self.assertLess(i_d, 0.0)
        other_d, other_q = m.mtpa_pair(-0.2)
        self.assertAlmostEqual(other_d, i_d)
        self.assertAlmostEqual(other_q, -i_q)


class LimitTests(unittest.TestCase):
    """The current circle, the voltage ellipse, and the envelope between them."""

    def setUp(self):
        self.m = PMSMSpec()

    def test_standstill_torque_limit_is_current_times_torque_constant(self):
        m = _non_salient(current_limit=10.0, voltage_limit=math.inf)
        self.assertAlmostEqual(m.torque_limit(0.0), m.torque_constant * 10.0)

    def test_envelope_is_flat_then_decreasing(self):
        # the datasheet shape: full current while the voltage ellipse is slack,
        # then the field-weakening hyperbola.  A surface-magnet machine rated
        # above its characteristic current has both regions.
        m = _non_salient(current_limit=10.0, voltage_limit=24.0)
        self.assertGreater(m.current_limit, m.characteristic_current)
        base = m.base_speed()
        full = m.torque_constant * 10.0
        self.assertAlmostEqual(m.torque_limit(0.0), full)
        self.assertAlmostEqual(m.torque_limit(0.3 * base), full)
        self.assertLess(m.torque_limit(0.5 * base), full)
        self.assertLess(m.torque_limit(2.0 * base), m.torque_limit(base))
        self.assertLess(m.torque_limit(4.0 * base), m.torque_limit(2.0 * base))
        self.assertGreater(m.torque_limit(4.0 * base), 0.0)

    def test_field_weakening_region_is_constant_power(self):
        # the reason the envelope is drawn as a hyperbola: past base speed the
        # voltage limit, not the current limit, sets the operating point, and the
        # capability in that region is nearly constant power (the small excess is
        # the resistive drop, which shrinks as a share of the rail with speed).
        m = _non_salient(current_limit=10.0, voltage_limit=24.0)
        base = m.base_speed()
        power = [m.torque_limit(w) * w for w in (base, 2.0 * base, 4.0 * base)]
        self.assertTrue(all(p > 0.0 for p in power))
        self.assertAlmostEqual(power[0], power[1], delta=0.02 * power[0])
        self.assertAlmostEqual(power[1], power[2], delta=0.02 * power[1])
        # it drifts very slightly UP with speed: the resistive drop is a smaller
        # share of the rail the faster the machine turns, so more of the voltage
        # is available to push current
        self.assertGreater(power[2], power[0])

    def test_salient_machine_starts_above_the_current_circle_torque(self):
        # reluctance torque means the standstill limit is not kt * I: the MTPA
        # angle at the current circle adds (Ld - Lq) * i_d * i_q on top.
        m = PMSMSpec()
        self.assertTrue(m.is_salient)
        base = m.base_speed()
        top = m.torque_limit(0.0)
        self.assertGreater(top, m.torque_constant * m.current_limit)
        self.assertAlmostEqual(m.torque_limit(0.2 * base), top)
        self.assertLess(m.torque_limit(base), top)
        self.assertLess(m.torque_limit(4.0 * base), m.torque_limit(2.0 * base))
        self.assertGreater(m.torque_limit(10.0 * base), 0.0)

    def test_envelope_collapses_without_current_to_weaken_with(self):
        # the hyperbola only exists if the current rating reaches past the
        # characteristic current, the current that cancels the magnet flux.  A
        # machine rated below it cannot demagnetise itself far enough to stay
        # inside the voltage limit for long: the constant-torque region ends at
        # base speed, the torque falls off steeply, and past ~1.7x base no
        # forward torque is possible at all - above there the machine is a
        # generator pushing into the DC link, not a motor.
        m = _non_salient(current_limit=2.0, voltage_limit=24.0)
        self.assertLess(m.current_limit, m.characteristic_current)
        base = m.base_speed()
        full = m.torque_constant * 2.0
        self.assertAlmostEqual(m.torque_limit(0.0), full)
        self.assertAlmostEqual(m.torque_limit(0.5 * base), full)
        reduced = m.torque_limit(1.2 * base)
        self.assertGreater(reduced, 0.0)
        self.assertLess(reduced, full)
        self.assertAlmostEqual(m.torque_limit(1.7 * base), 0.0)
        self.assertAlmostEqual(m.torque_limit(3.0 * base), 0.0)
        # and past that speed the machine cannot even sit at zero current: the
        # back-EMF alone is over the rail, so the model asks for a demagnetising
        # current just to hold the terminal voltage down.  A drive that commands
        # i_d = i_q = 0 up there is fighting the DC link, not idling.
        i_d, i_q, _, voltage_saturated = m.current_pair(0.0, 2.0 * base)
        self.assertLess(i_d, 0.0)
        self.assertAlmostEqual(i_q, 0.0)
        self.assertTrue(voltage_saturated)

    def test_q_axis_inductance_eats_voltage_below_base_speed(self):
        # v_d = -w_e*Lq*i_q: a high-inductance machine runs out of voltage before
        # the back-EMF alone reaches the rail, so its constant-torque region ends
        # early.  This is why Lq matters for a feed-forward controller even
        # though it never appears in a steady-state torque equation.
        slow = _non_salient(current_limit=10.0, voltage_limit=24.0,
                            inductance_d=1.0e-6, inductance_q=1.0e-6)
        fast = _non_salient(current_limit=10.0, voltage_limit=24.0,
                            inductance_d=2.0e-4, inductance_q=2.0e-4)
        w = 0.5 * slow.base_speed()
        self.assertAlmostEqual(slow.torque_limit(w), slow.torque_constant * 10.0)
        self.assertLess(fast.torque_limit(w), fast.torque_constant * 10.0)

    def test_envelope_is_even_in_speed(self):
        for w in (0.0, 100.0, 550.0, 1200.0, 3000.0):
            self.assertAlmostEqual(self.m.torque_limit(w), self.m.torque_limit(-w))

    def test_braking_envelope_is_wider_than_driving_above_base_speed(self):
        base = self.m.base_speed()
        self.assertAlmostEqual(self.m.torque_limit(0.0, regenerative=True),
                               self.m.torque_limit(0.0))
        for w in (base, 2.0 * base, 5.0 * base):
            self.assertGreater(self.m.torque_limit(w, regenerative=True),
                               self.m.torque_limit(w))

    def test_current_pair_never_exceeds_either_limit(self):
        m = self.m
        for w in (0.0, 100.0, 550.0, 1100.0, 3000.0, -550.0):
            for tau in (0.0, 0.05, 0.2, 0.4, -0.05, -0.4):
                i_d, i_q, cur_sat, volt_sat = m.current_pair(tau, w)
                self.assertLessEqual(math.hypot(i_d, i_q),
                                     m.current_limit * (1.0 + 1.0e-6))
                v_d, v_q = m.voltage_steady(i_d, i_q, w)
                self.assertLessEqual(math.hypot(v_d, v_q),
                                     m.voltage_limit * (1.0 + 1.0e-6))

    def test_current_pair_delivers_what_it_can(self):
        m = self.m
        for w in (0.0, 100.0, 300.0):
            for tau in (0.01, 0.05, 0.1):
                i_d, i_q, cur_sat, volt_sat = m.current_pair(tau, w)
                self.assertFalse(cur_sat or volt_sat)
                self.assertAlmostEqual(m.torque_at(i_d, i_q), tau, places=9)

    def test_an_impossible_torque_comes_back_flagged_not_quiet(self):
        m = self.m
        base = m.base_speed()
        i_d, i_q, cur_sat, volt_sat = m.current_pair(0.4, 2.0 * base)
        self.assertTrue(volt_sat)
        self.assertLess(m.torque_at(i_d, i_q), 0.4)
        self.assertLess(i_d, 0.0)  # it weakened the field while trying

    def test_i_d_floor_is_full_demagnetisation(self):
        # i_d = -lambda/Ld is the point where the air-gap flux from the magnets
        # is completely cancelled; the model refuses to go past it rather than
        # pretend a permanent-magnet field can be reversed.
        m = PMSMSpec()
        self.assertAlmostEqual(m.characteristic_current, m.flux_linkage / m.inductance_d)
        for w in (0.0, 1000.0, 5000.0):
            i_d, _, _, _ = m.current_pair(0.0, w)
            self.assertGreaterEqual(i_d, -m.characteristic_current - 1.0e-9)

    def test_four_quadrant_symmetry(self):
        m = self.m
        for tau, w in ((0.05, 1100.0), (0.2, 550.0), (0.02, 3000.0)):
            a = m.current_pair(tau, w)
            b = m.current_pair(-tau, -w)
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], -b[1])
            self.assertEqual(a[2:], b[2:])

    def test_braking_needs_less_current_than_driving(self):
        # v_d = R*i_d - w_e*Lq*i_q and v_q = R*i_q + w_e*(Ld*i_d + lambda): with
        # torque against rotation the resistive drop of i_q partly cancels the
        # back-EMF instead of adding to it, so the same torque costs less of the
        # voltage budget and therefore less current.
        w = 2.0 * self.m.base_speed()
        _, _, _, drive_volt_sat = self.m.current_pair(0.05, w)
        i_d_b, i_q_b, _, _ = self.m.current_pair(-0.05, w)
        i_d_d, i_q_d, _, _ = self.m.current_pair(0.05, w)
        self.assertLess(math.hypot(i_d_b, i_q_b), math.hypot(i_d_d, i_q_d))
        self.assertTrue(drive_volt_sat)

    def test_the_current_angle_is_optimal(self):
        """Brute force the minimum-current angle; the analytic search must match."""
        m = self.m
        grid = [(-30.0 + 38.0 * k / 3000.0) for k in range(3001)]
        for tau, w in ((0.05, 1100.0), (0.2, 550.0), (0.3, 0.0), (0.05, 275.0),
                       (0.1, 660.0), (0.02, 5500.0), (-0.05, 1100.0),
                       (-0.2, 550.0), (-0.02, 5500.0), (0.0, 1650.0)):
            i_d, i_q, cur_sat, volt_sat = m.current_pair(tau, w)
            got = math.hypot(i_d, i_q)
            best = math.inf
            for candidate in grid:
                gain = m.torque_gain_at(candidate)
                if gain == 0.0:
                    continue
                other_q = tau / gain
                if m.current_magnitude(candidate, other_q) > m.current_limit + 1.0e-12:
                    continue
                v_d, v_q = m.voltage_steady(candidate, other_q, w)
                if math.hypot(v_d, v_q) > m.voltage_limit + 1.0e-12:
                    continue
                best = min(best, math.hypot(candidate, other_q))
            if math.isfinite(best):
                # never more current than the best feasible point on the grid;
                # the grid is coarse, so the model may legitimately do better
                self.assertLessEqual(got, best * (1.0 + 1.0e-3))
                self.assertGreater(got, best - 0.05)

    def test_max_speed_is_unbounded_only_without_losses(self):
        self.assertTrue(math.isinf(PMSMSpec(bearing_damping=0.0,
                                            windage_coefficient=0.0,
                                            core_loss_coefficient=0.0).max_speed()))
        # any loss term at all bounds it: the envelope decays towards zero, so
        # the speed where it can no longer pay for the losses is a hard limit
        for losses in (dict(bearing_damping=1.0e-6), dict(windage_coefficient=1.0e-7),
                       dict(core_loss_coefficient=1.0e-8)):
            m = PMSMSpec(**losses)
            self.assertTrue(math.isfinite(m.max_speed()), losses)
            self.assertGreater(m.max_speed(), m.base_speed())
            self.assertLess(m.max_speed(load_torque=0.05), m.max_speed())

    def test_torque_speed_curve_is_monotone_and_starts_at_standstill_torque(self):
        speeds, torques = self.m.torque_speed_curve(n=32)
        self.assertEqual(len(speeds), 32)
        self.assertAlmostEqual(torques[0], self.m.torque_limit(0.0))
        for previous, current in zip(torques, torques[1:]):
            self.assertLessEqual(current, previous + 1.0e-12)


class FieldWeakeningTests(unittest.TestCase):
    def setUp(self):
        self.m = PMSMSpec()
        self.base = self.m.base_speed()

    def test_below_base_speed_no_field_weakening(self):
        tau = 0.05
        self.assertTrue(self.m.mtpa_allows_voltage(tau, 0.5 * self.base))
        i_d, i_q, cur_sat, volt_sat = self.m.current_pair(tau, 0.5 * self.base)
        self.assertFalse(cur_sat or volt_sat)
        mtpa_d, _ = self.m.mtpa_pair(tau)
        self.assertAlmostEqual(i_d, mtpa_d, places=9)
        v_d, v_q = self.m.voltage_steady(i_d, i_q, 0.5 * self.base)
        self.assertLess(math.hypot(v_d, v_q), self.m.voltage_limit)

    def test_above_base_speed_the_pair_sits_on_the_voltage_ellipse(self):
        tau = 0.05
        self.assertFalse(self.m.mtpa_allows_voltage(tau, 2.0 * self.base))
        i_d, i_q, cur_sat, volt_sat = self.m.current_pair(tau, 2.0 * self.base)
        self.assertTrue(volt_sat)
        self.assertLess(i_d, 0.0)
        v_d, v_q = self.m.voltage_steady(i_d, i_q, 2.0 * self.base)
        self.assertAlmostEqual(math.hypot(v_d, v_q), self.m.voltage_limit, places=9)

    def test_zero_torque_above_the_rail_still_draws_current(self):
        # Above the rail speed the back-EMF alone exceeds what the inverter can
        # apply, so "zero torque" is not "zero current": the machine is forcibly
        # weakened and pushes current back towards the supply.  A feed-forward
        # controller that assumes otherwise will fight the drive.
        w = 3.0 * self.base
        i_d, i_q, _, volt_sat = self.m.current_pair(0.0, w)
        self.assertAlmostEqual(i_q, 0.0)
        self.assertLess(i_d, 0.0)
        self.assertTrue(volt_sat)
        v_d, v_q = self.m.voltage_steady(i_d, i_q, w)
        self.assertAlmostEqual(math.hypot(v_d, v_q), self.m.voltage_limit, places=9)

    def test_field_weakening_never_reverses_the_magnet_flux(self):
        for w in (0.0, self.base, 2 * self.base, 10 * self.base, 100 * self.base):
            for tau in (-0.3, 0.0, 0.3):
                i_d, _, _, _ = self.m.current_pair(tau, w)
                self.assertGreaterEqual(self.m.flux_linkage + self.m.inductance_d * i_d, -1.0e-15)


class FeedForwardTests(unittest.TestCase):
    def setUp(self):
        self.m = PMSMSpec(dc_bus_voltage=12.0)

    def test_required_torque_is_the_motion_command(self):
        m = PMSMSpec(rotor_inertia=1.0e-4, bearing_damping=1.0e-3, static_friction=0.0)
        self.assertAlmostEqual(m.required_torque(100.0, 50.0, 0.02),
                               1.0e-4 * 50.0 + 1.0e-3 * 100.0 + 0.02)
        # the load is a magnitude that opposes motion, so it flips with the
        # direction of rotation exactly like the bearing term does
        self.assertAlmostEqual(m.required_torque(-100.0, 0.0, 0.02),
                               -1.0e-3 * 100.0 - 0.02)
        self.assertAlmostEqual(m.required_torque(0.0, -10.0, 0.02),
                               -1.0e-4 * 10.0 - 0.02)  # at rest, opposes the demand

    def test_stiction_in_the_motion_command(self):
        m = PMSMSpec(static_friction=0.05)
        self.assertAlmostEqual(m.required_torque(100.0, 0.0, 0.0), 1.0e-6 * 100.0 + 0.05)
        self.assertAlmostEqual(m.required_torque(-100.0, 0.0, 0.0), -1.0e-6 * 100.0 - 0.05)
        # at standstill there is no motion to oppose, so the direction comes from
        # the acceleration demand; with none at all the model reports the
        # break-away torque in the positive direction
        self.assertAlmostEqual(m.required_torque(0.0, 0.0, 0.02), 0.07)
        self.assertAlmostEqual(m.required_torque(0.0, -1.0e-6, 0.02), -0.07, places=9)
        # a load stronger than the stiction breaks the shaft away by itself
        weak = PMSMSpec(static_friction=0.01)
        self.assertAlmostEqual(weak.required_torque(0.0, 0.0, 0.02), 0.03)

    def test_an_assisting_torque_is_not_a_load(self):
        # a grade or a returning spring pushes WITH the motion, so it is not a
        # load magnitude; pretending otherwise would double-count its sign
        m = PMSMSpec()
        with self.assertRaises(ConfigError):
            m.required_torque(100.0, 0.0, -0.02)
        with self.assertRaises(ConfigError):
            m.feedforward(100.0, 0.0, -0.02)

    def test_feedforward_returns_the_requested_torque_inside_the_envelope(self):
        cmd = self.m.feedforward(100.0, 0.0, 0.01)
        self.assertFalse(cmd.current_saturated or cmd.voltage_saturated)
        self.assertAlmostEqual(cmd.torque, self.m.required_torque(100.0, 0.0, 0.01), places=9)
        self.assertAlmostEqual(self.m.torque_at(cmd.i_d, cmd.i_q), cmd.torque)

    def test_feedforward_reports_the_voltage_it_asks_for(self):
        cmd = self.m.feedforward(200.0, 0.0, 0.0)
        v_d, v_q = self.m.voltage_steady(cmd.i_d, cmd.i_q, 200.0)
        self.assertAlmostEqual(cmd.v_d, v_d)
        self.assertAlmostEqual(cmd.v_q, v_q)
        self.assertLessEqual(cmd.voltage_magnitude, self.m.voltage_limit * (1.0 + 1.0e-9))

    def test_feedforward_saturates_honestly(self):
        cmd = self.m.feedforward(2.0 * self.m.base_speed(), 0.0, 0.5)
        self.assertTrue(cmd.voltage_saturated)
        self.assertLess(cmd.torque, 0.5)
        self.assertLessEqual(cmd.voltage_magnitude, self.m.voltage_limit * (1.0 + 1.0e-9))
        self.assertLessEqual(cmd.current_magnitude, self.m.current_limit * (1.0 + 1.0e-9))

    def test_field_weakening_flag(self):
        low = self.m.feedforward(0.5 * self.m.base_speed(), 0.0, 0.01)
        high = self.m.feedforward(2.0 * self.m.base_speed(), 0.0, 0.01)
        self.assertFalse(low.field_weakening)
        self.assertTrue(high.field_weakening)

    def test_dc_link_current_estimate(self):
        cmd = self.m.feedforward(150.0, 0.0, 0.01)
        self.assertAlmostEqual(cmd.dc_link_current,
                               1.5 * (cmd.v_d * cmd.i_d + cmd.v_q * cmd.i_q) / 12.0)
        self.assertGreater(cmd.dc_link_current, 0.0)

    def test_acceleration_term_enters_the_voltage(self):
        still = self.m.feedforward(0.0, 0.0, 0.0)
        pushing = self.m.feedforward(0.0, 0.0, 0.0, di_d_dt=0.0, di_q_dt=1.0e4)
        self.assertGreater(pushing.v_q, still.v_q)
        self.assertAlmostEqual(pushing.v_q - still.v_q, self.m.inductance_q * 1.0e4)

    def test_feedforward_needs_the_rate_limited_acceleration(self):
        # a step in speed reference would demand an infinite di/dt; the exported
        # controller differentiates the *limited* reference instead
        cmd = self.m.feedforward(0.0, 1.0e4, 0.0)
        self.assertLessEqual(cmd.current_magnitude, self.m.current_limit * (1.0 + 1.0e-9))


class LossTests(unittest.TestCase):
    def test_copper_power(self):
        m = PMSMSpec(resistance=0.5)
        self.assertAlmostEqual(m.copper_power(2.0, 3.0), 1.5 * 0.5 * 13.0)

    def test_core_and_windage_power(self):
        m = PMSMSpec(core_loss_coefficient=1.0e-8, windage_coefficient=1.0e-7)
        self.assertAlmostEqual(m.core_loss_power(100.0),
                               1.0e-8 * (m.pole_pairs * 100.0) ** 2)
        self.assertAlmostEqual(m.windage_power(100.0), 1.0e-7 * 100.0 ** 3)

    def test_input_power_is_the_dq_dot_product(self):
        m = PMSMSpec()
        self.assertAlmostEqual(m.input_power(3.0, 4.0, 1.0, 2.0), 1.5 * (3.0 + 8.0))

    def test_efficiency_is_a_fraction_and_peaks_in_the_middle(self):
        m = PMSMSpec(core_loss_coefficient=1.0e-8, windage_coefficient=1.0e-8)
        low = m.efficiency_at(200.0, 0.05)
        mid = m.efficiency_at(600.0, 0.05)
        self.assertIsNotNone(low)
        self.assertTrue(0.0 < low <= 1.0)
        self.assertTrue(0.0 < mid <= 1.0)
        # copper loss grows with current, iron and windage with speed: efficiency
        # has to rise then fall, so a single monotone curve would be a bug
        self.assertGreater(m.efficiency_at(2000.0, 0.05), 0.0)

    def test_efficiency_is_none_when_nothing_is_being_delivered(self):
        self.assertIsNone(PMSMSpec().efficiency_at(0.0, 0.0))

    def test_developed_power_is_the_dq_dot_product_minus_copper(self):
        # the exact identity the dq model implies: 1.5*(v_d*i_d + v_q*i_q) equals
        # copper loss plus the electromagnetic power developed on the shaft.
        # Getting this wrong by a factor of 1.5 or 3/2 is the classic dq error.
        m = PMSMSpec(core_loss_coefficient=1.0e-8, windage_coefficient=1.0e-8)
        for w, tau in ((300.0, 0.05), (700.0, 0.06), (1500.0, 0.02), (-700.0, -0.06)):
            i_d, i_q, _, _ = m.current_pair(tau, w)
            v_d, v_q = m.voltage_steady(i_d, i_q, w)
            self.assertAlmostEqual(m.input_power(v_d, v_q, i_d, i_q),
                                   m.copper_power(i_d, i_q) + m.torque_at(i_d, i_q) * w,
                                   places=9)

    def test_shaft_torque_is_developed_torque_minus_losses(self):
        m = PMSMSpec(core_loss_coefficient=1.0e-8, windage_coefficient=1.0e-8,
                     bearing_damping=1.0e-5)
        w, tau = 700.0, 0.06
        i_d, i_q, _, _ = m.current_pair(tau, w)
        self.assertAlmostEqual(m.torque_at(i_d, i_q), tau)
        self.assertAlmostEqual(m.loss_torque(w),
                               1.0e-5 * w + 1.0e-8 * w * w
                               + 1.0e-8 * m.pole_pairs ** 2 * w)


class TemperatureTests(unittest.TestCase):
    def test_flux_falls_as_the_magnets_heat_up(self):
        m = PMSMSpec(flux_linkage=1.8e-3, flux_temperature_coefficient=-0.0011,
                     reference_temperature=20.0)
        self.assertAlmostEqual(m.effective_flux(20.0), 1.8e-3)
        self.assertLess(m.effective_flux(120.0), 1.8e-3)
        self.assertAlmostEqual(m.effective_flux(120.0), 1.8e-3 * (1.0 - 0.0011 * 100.0))

    def test_a_hot_machine_spins_faster_on_the_same_voltage(self):
        m = PMSMSpec(flux_temperature_coefficient=-0.0011)
        self.assertGreater(m.no_load_speed(12.0, temperature_c=120.0),
                           m.no_load_speed(12.0))
        self.assertLess(m.torque_constant_at(120.0), m.torque_constant)

    def test_no_coefficient_means_no_drift(self):
        m = PMSMSpec()
        self.assertEqual(m.effective_flux(200.0), m.flux_linkage)
        self.assertEqual(m.no_load_speed(12.0, temperature_c=200.0), m.no_load_speed(12.0))


class ValidationTests(unittest.TestCase):
    def test_nonsense_constants_are_rejected(self):
        bad = dict(resistance=0.0, inductance_d=0.0, inductance_q=-1.0,
                   flux_linkage=0.0, pole_pairs=0, rotor_inertia=0.0,
                   bearing_damping=-1.0, static_friction=-1.0, current_limit=0.0,
                   voltage_limit=0.0, core_loss_coefficient=-1.0,
                   windage_coefficient=-1.0)
        for key, value in bad.items():
            with self.subTest(key=key):
                with self.assertRaises(ConfigError):
                    PMSMSpec(**{key: value})

    def test_zero_inductance_points_at_the_brushed_model(self):
        with self.assertRaises(ConfigError):
            PMSMSpec(inductance_d=0.0)

    def test_copy_is_independent(self):
        m = PMSMSpec()
        c = m.copy()
        c.current_limit = 1.0
        self.assertNotEqual(m.current_limit, c.current_limit)


class StateTests(unittest.TestCase):
    def test_state_fields_are_marked_as_state(self):
        import dataclasses
        marked = {f.name for f in dataclasses.fields(PMSMSpec)
                  if f.metadata.get("state")}
        self.assertEqual(marked, {"current_d", "current_q", "rot_vel"})
        # the dq spec's own speed is solver state: inside a DriveSystem the speed
        # you legitimately sweep is the adapter's group.motor.rot_vel, which the
        # simulator reads.  Leaving this one unmarked would put a second, inert
        # speed slider in the parameter panel.

    def test_command_as_dict_is_json_friendly(self):
        m = PMSMSpec()
        cmd = m.feedforward(300.0, 10.0, 0.01)
        d = cmd.as_dict()
        for key in ("i_d", "i_q", "v_d", "v_q", "torque", "current_magnitude",
                    "voltage_magnitude", "current_saturated", "voltage_saturated",
                    "field_weakening", "dc_link_current", "speed"):
            self.assertIn(key, d)
        import json
        json.dumps(d)


if __name__ == "__main__":
    unittest.main()
