"""Tests for the library dq (field-oriented) simulation.

These are physics tests, not smoke tests.  The dq integrator exists so a fit
can predict a three-phase log, so every test pins a property that a wrong
implementation would break: a known analytic response, an equilibrium that must
not move, an energy identity that must hold exactly, the collapse onto the
brushed ``i_d = 0`` engine, and the step-size limit that keeps RK4 stable on the
dq pole pair.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from drivetrain import dq_sim, presets
from drivetrain.exceptions import ConfigError
from drivetrain.gearbox import Gearbox
from drivetrain.load import RotationalLoad
from drivetrain.motor import MotorGroup
from drivetrain.pmsm import PMSMModel, PMSMSpec
from drivetrain.simulate import simulate_rk4
from drivetrain.system import DriveSystem

# bldc_pod numbers, restated so a test fails loudly if a preset drifts.
R = 0.12
LD = 1.5e-4
LQ = 1.8e-4
FLUX = 1.8e-3
POLES = 7
CURRENT_LIMIT = 20.0
BASE_SPEED = 1099.714798          # rad/s motor side, voltage_limit / (p lambda)


def _pod():
    return presets.build("bldc_pod")


def _bench_spec() -> PMSMSpec:
    """The bldc_pod machine, spelled out from the constants above."""
    return PMSMSpec(resistance=R, inductance_d=LD, inductance_q=LQ, flux_linkage=FLUX,
                    pole_pairs=POLES, rotor_inertia=2.0e-5, bearing_damping=1.0e-6,
                    current_limit=CURRENT_LIMIT, voltage_limit=12.0 / math.sqrt(3.0))


def _lossless_system() -> DriveSystem:
    """One machine, direct drive, no load: only the bearing damping is left.

    An operating point near zero torque cannot be held by the pod's vehicle
    load -- the load would have to *assist* the motion, and `simulate_dq`
    refuses a negative load magnitude because it is a magnitude opposing
    motion.  With nothing but bearing damping the balancing torque is passive.
    """
    return DriveSystem(MotorGroup(PMSMModel(spec=_bench_spec()), 1), Gearbox(1.0, 1.0),
                       RotationalLoad(inertia=0.001, viscous=0.0, constant_torque=0.0))


def _balance_load(mech, spec, i_d: float, i_q: float, w: float) -> float:
    """The passive load magnitude that holds ``(i_d, i_q, w)`` in equilibrium.

    ``dq_sim._Mechanical.load`` adds ``+extra`` for w > 0 and ``-extra`` for
    w < 0, so the whole expression flips sign with the direction of motion.
    A point that needs an *assisting* torque cannot be held by a passive load,
    and that is what this raises for rather than hiding.
    """
    inner = (mech.count * spec.torque_at(i_d, i_q) - mech.damping * w - mech.load(w)
             - (mech.static if w > 0 else -mech.static))
    extra = inner if w > 0 else -inner
    if extra < 0.0:
        if extra > -1.0e-12:
            return 0.0
        raise AssertionError(f"({i_d}, {i_q}, w={w}) needs an assisting torque of "
                             f"{extra:.6g} N*m, which a passive load cannot supply")
    return extra


def _locked_shaft_system() -> DriveSystem:
    """A bench system whose shaft cannot turn, so the dq plant is linear.

    The machine's own ``static_friction`` is set far above anything it can
    produce, so the stiction branch in the mechanical model holds the shaft at
    zero speed forever: the electromagnetic torque is what it is, and it never
    reaches the threshold.  (A load reflected as static friction would work too,
    but it also puts a nonzero magnitude at ``w = 0``, where the direction of a
    passive load is undefined, and that would muddy the comparison.)  With w = 0
    the dq equations decouple into two first-order linear ODEs with a
    closed-form answer, which is what makes this a hard test rather than a
    plausibility one.
    """
    bench = _bench_spec()
    motor = PMSMModel(spec=replace(bench, static_friction=1.0e6))
    return DriveSystem(MotorGroup(motor, 1), Gearbox(1.0, 1.0),
                       RotationalLoad(inertia=0.0, viscous=0.0, constant_torque=0.0))


class StepLimitTests(unittest.TestCase):
    def test_the_stable_step_shrinks_with_speed(self):
        spec = _pod().group.motor.spec
        stall = dq_sim.max_step_time(spec, 0.0)
        base = dq_sim.max_step_time(spec, BASE_SPEED)
        self.assertAlmostEqual(stall, 6.25e-4, places=12)
        # BASE_SPEED here is the preset's rounded flux linkage times pole pairs,
        # so this is the limit for the machine as built, not for the exact
        # analytic base speed; the two differ in the tenth digit.
        self.assertAlmostEqual(base, 1.797525117715873e-4, places=11)
        self.assertLess(base, stall)

    def test_the_two_limits_are_the_two_mechanisms(self):
        # At standstill the accuracy limit (half the electrical time constant)
        # binds; at speed the RK4 stability limit binds.  Both are needed, and
        # a machine with a large Lq/Ld ratio flips which one wins.
        spec = _pod().group.motor.spec
        self.assertAlmostEqual(dq_sim.electrical_time_constant(spec), 1.25e-3, places=15)
        self.assertLess(dq_sim.RK4_STABILITY_LIMIT / dq_sim.pole_magnitude(spec, BASE_SPEED),
                        0.5 * dq_sim.electrical_time_constant(spec))
        self.assertGreater(dq_sim.RK4_STABILITY_LIMIT / dq_sim.pole_magnitude(spec, 0.0),
                           0.5 * dq_sim.electrical_time_constant(spec))

    def test_pole_damping_is_the_real_part_of_the_conjugate_pair(self):
        spec = _pod().group.motor.spec
        expected = R * (LD + LQ) / (2.0 * LD * LQ)
        self.assertAlmostEqual(dq_sim.pole_damping(spec), expected, places=12)
        self.assertAlmostEqual(dq_sim.pole_magnitude(spec, 0.0), expected, places=12)
        self.assertAlmostEqual(
            dq_sim.pole_magnitude(spec, 100.0),
            math.hypot(expected, POLES * 100.0), places=9)

    def test_step_refuses_an_unstable_dt_instead_of_diverging(self):
        system = _pod()
        with self.assertRaises(ConfigError) as ctx:
            dq_sim.step(system, dq_sim.DqState(0.0, 0.0, BASE_SPEED), 0.0, 2.4, 5e-4)
        message = str(ctx.exception)
        self.assertIn("exceeds the stable step", message)
        self.assertIn("subdivide the step", message)
        self.assertIn("1099.71 rad/s", message)

    def test_integrate_subdivides_a_control_period(self):
        system = _pod()
        spec = system.group.motor.spec
        self.assertEqual(dq_sim.sub_steps(spec, 1e-3, 0.0), 2)
        self.assertEqual(dq_sim.sub_steps(spec, 1e-3, BASE_SPEED), 6)
        self.assertEqual(dq_sim.sub_steps(spec, 1e-4, BASE_SPEED), 1)
        state = dq_sim.integrate(system, dq_sim.DqState(), 0.0, 0.5, 1e-3)
        self.assertTrue(math.isfinite(state.i_q))
        self.assertGreater(state.i_q, 0.0)

    def test_the_sub_stepped_period_matches_a_fine_fixed_step(self):
        # integrate() takes a control period and cuts it into stable RK4 steps;
        # a fixed fine step over the same span is the reference.  The start is
        # away from standstill on purpose -- see the standstill test below.
        system = _pod()
        coarse = dq_sim.DqState(0.0, 0.0, 50.0)
        for _ in range(50):
            coarse = dq_sim.integrate(system, coarse, 0.0, 0.5, 1e-3)
        fine = dq_sim.DqState(0.0, 0.0, 50.0)
        for _ in range(2000):
            fine = dq_sim.step(system, fine, 0.0, 0.5, 2.5e-5)
        self.assertAlmostEqual(coarse.rot_vel, fine.rot_vel, delta=abs(fine.rot_vel) * 1e-8)
        self.assertAlmostEqual(coarse.i_q, fine.i_q, delta=abs(fine.i_q) * 1e-8)

    def test_rk4_converges_at_fourth_order_on_a_smooth_run(self):
        # Halving the step must cut the gap between successive answers by about
        # 2**4.  This is the test that says the integrator really is RK4 and not
        # something that merely looks stable.
        system = _pod()
        spans = {dt: int(round(0.05 / dt)) for dt in (2e-4, 1e-4, 5e-5, 2.5e-5)}
        ends = []
        for dt, n in spans.items():
            state = dq_sim.DqState(0.0, 0.0, 50.0)
            for _ in range(n):
                state = dq_sim.step(system, state, 0.0, 0.5, dt)
            ends.append(state.rot_vel)
        gaps = [abs(a - b) for a, b in zip(ends, ends[1:])]
        self.assertTrue(all(b < a for a, b in zip(gaps, gaps[1:])), gaps)
        for a, b in zip(gaps, gaps[1:]):
            self.assertGreater(a / b, 12.0, (a, b))

    def test_a_run_started_exactly_at_standstill_is_not_a_convergence_case(self):
        # A passive load at zero speed has no direction of its own: load(0) is a
        # magnitude, and it is the sign of the speed that decides which way it
        # acts.  The pod has no static friction (static_friction_eff == 0), so a
        # run started at w = 0 sits exactly on that discontinuity, and RK4's
        # four stages disagree about the sign inside the first step.  The result
        # is a first-step difference that does NOT shrink at fourth order, which
        # is why every convergence test above starts away from standstill.
        system = _pod()
        mech = dq_sim._Mechanical.build(system)
        self.assertEqual(system.group.static_friction_eff, 0.0)
        self.assertAlmostEqual(mech.load(-1.0e-9), -mech.load(1.0e-9), places=12)
        self.assertLess(mech.net_torque(0.0, 0.0), 0.0)   # the load starts the shaft backwards

        def end(dt: float) -> float:
            state = dq_sim.DqState()
            for _ in range(int(round(0.05 / dt))):
                state = dq_sim.step(system, state, 0.0, 0.5, dt)
            return state.rot_vel

        coarse = abs(end(1e-4) - end(5e-5))
        fine = abs(end(5e-5) - end(2.5e-5))
        self.assertGreater(fine, 10.0 * coarse)           # fourth order would be fine < coarse


class AnalyticResponseTests(unittest.TestCase):
    """A locked shaft turns the dq plant into two first-order linear ODEs."""

    def test_current_rises_to_v_over_r_with_the_electrical_time_constant(self):
        system = _locked_shaft_system()
        v_d, v_q = 1.0, 2.0
        duration = 0.02
        result = dq_sim.simulate_dq(system, v_d, v_q, duration=duration, dt=1e-6,
                                    samples=None)
        tau_d = LD / R
        tau_q = LQ / R
        for t, i_d, i_q in zip(result.t, result.i_d, result.i_q):
            self.assertAlmostEqual(i_d, (v_d / R) * (1.0 - math.exp(-t / tau_d)), delta=2e-9)
            self.assertAlmostEqual(i_q, (v_q / R) * (1.0 - math.exp(-t / tau_q)), delta=2e-9)
        # 0.02 s is 13 electrical time constants on the q axis, so the current is
        # within 1.6e-6 (relative) of v/R, not closer.  The exact endpoint is the
        # honest target; v/R itself is asserted only to that settling error.
        t_end = float(result.t[-1])
        self.assertAlmostEqual(float(result.i_d[-1]),
                               (v_d / R) * (1.0 - math.exp(-t_end * R / LD)), delta=1e-9)
        self.assertAlmostEqual(float(result.i_q[-1]),
                               (v_q / R) * (1.0 - math.exp(-t_end * R / LQ)), delta=1e-9)
        self.assertLess(abs(float(result.i_q[-1]) - v_q / R), (v_q / R) * 2.0e-6)
        self.assertAlmostEqual(result.motor_speed[-1], 0.0, places=12)

    def test_euler_is_first_order_and_rk4_is_fourth_against_the_exact_answer(self):
        # The locked shaft has a closed-form answer, so each engine's error is
        # measurable rather than inferred from the two agreeing.  Euler must halve
        # when the step halves; RK4 must divide by about 2**4.
        system = _locked_shaft_system()
        v_q = 2.0
        euler_err, rk4_err = [], []
        for dt in (4e-5, 2e-5, 1e-5):
            for engine, sink in (("euler", euler_err), ("rk4", rk4_err)):
                result = dq_sim.simulate_dq(system, 1.0, v_q, duration=0.01, dt=dt,
                                            samples=None, engine=engine)
                t_end = float(result.t[-1])
                exact = (v_q / R) * (1.0 - math.exp(-t_end * R / LQ))
                sink.append(abs(float(result.i_q[-1]) - exact))
        for name, errors in (("euler", euler_err), ("rk4", rk4_err)):
            self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])), (name, errors))
        for a, b in zip(euler_err, euler_err[1:]):
            self.assertLess(b / a, 0.6, (a, b))          # ~1/2: first order
        for a, b in zip(rk4_err, rk4_err[1:]):
            self.assertLess(b / a, 0.09, (a, b))         # ~1/16: fourth order
            self.assertGreater(b / a, 0.02, (a, b))
        self.assertLess(rk4_err[-1] * 1.0e4, euler_err[-1])

    def test_back_emf_makes_the_steady_current_v_minus_kv_w_over_r(self):
        # With the shaft spinning at a fixed speed the q-axis steady state is
        # (v_q - w_e lambda)/R; the locked-shaft load cannot hold a spinning
        # shaft, so drive the current instead and read the balance off the
        # derivative: at the steady current di_q/dt must vanish.
        spec = _pod().group.motor.spec
        w = 300.0
        v_q = 6.0
        i_steady = (v_q - POLES * w * FLUX) / R
        state = dq_sim.DqState(0.0, i_steady, w, 0.0)
        di_d, di_q = dq_sim.electrical_derivatives(spec, state, 0.0, v_q)
        self.assertAlmostEqual(di_q, 0.0, places=12)
        # the d-axis is NOT at rest: with v_d = 0 the cross-coupling term drives
        # di_d/dt = w_e Lq i_q / Ld, which is why a real drive closes a d-axis
        # loop rather than leaving v_d at zero
        self.assertAlmostEqual(di_d, POLES * w * LQ * i_steady / LD, places=12)


class EquilibriumTests(unittest.TestCase):
    def test_a_balanced_operating_point_does_not_move(self):
        system = _pod()
        spec = system.group.motor.spec
        mech = dq_sim._Mechanical.build(system)
        for w, i_d, i_q in ((400.0, -3.0, 8.0), (0.5, 0.0, 1.0), (-250.0, -2.0, -6.0)):
            v_d, v_q = spec.voltage_steady(i_d, i_q, w)
            extra = _balance_load(mech, spec, i_d, i_q, w)
            result = dq_sim.simulate_dq(
                system, v_d, v_q, duration=0.05, dt=1e-5, load_torque=extra,
                initial=dq_sim.DqState(i_d, i_q, w, 0.0))
            self.assertAlmostEqual(float(result.i_d[-1]), i_d, places=12, msg=f"i_d at w={w}")
            self.assertAlmostEqual(float(result.i_q[-1]), i_q, places=12, msg=f"i_q at w={w}")
            self.assertAlmostEqual(float(result.motor_speed[-1]), w, places=12, msg=f"w at w={w}")

    def test_the_rotor_angle_integrates_the_speed(self):
        system = _pod()
        spec = system.group.motor.spec
        mech = dq_sim._Mechanical.build(system)
        w, i_d, i_q = 400.0, -3.0, 8.0
        v_d, v_q = spec.voltage_steady(i_d, i_q, w)
        extra = _balance_load(mech, spec, i_d, i_q, w)
        result = dq_sim.simulate_dq(system, v_d, v_q, duration=0.05, dt=1e-5,
                                    load_torque=extra,
                                    initial=dq_sim.DqState(i_d, i_q, w, 0.0))
        # angle advances by w * t at the last stored sample, exactly
        self.assertAlmostEqual(float(result.angle[-1]), w * float(result.t[-1]), places=9)

    def test_field_weakening_at_zero_torque_is_reachable(self):
        # Above base speed a brushless machine still needs a negative d-axis
        # current even at zero torque, or v_q cannot stay under the rail:
        # i_d = -lambda/Ld cancels the back-EMF term exactly.
        system = _lossless_system()
        spec = system.group.motor.spec
        w = 1500.0
        i_d = -spec.flux_linkage / spec.inductance_d      # -lambda/Ld, the d-axis floor
        v_d, v_q = spec.voltage_steady(i_d, 0.0, w)
        self.assertAlmostEqual(v_q, 0.0, places=15)
        self.assertLess(abs(v_d), spec.voltage_limit)
        _, v_q_plain = spec.voltage_steady(0.0, 0.0, w)
        self.assertGreater(v_q_plain, spec.voltage_limit)   # unreachable with i_d = 0

        # and the integrator holds the weakened point rather than drifting.  The
        # reluctance term still makes torque at i_q > 0, so a passive load soaks
        # up the difference.
        mech = dq_sim._Mechanical.build(system)
        i_q = (mech.damping * w + mech.load(w) + mech.static) / (mech.count * 1.5 * POLES * FLUX)
        extra = _balance_load(mech, spec, i_d, i_q, w)
        v_d, v_q = spec.voltage_steady(i_d, i_q, w)
        result = dq_sim.simulate_dq(system, v_d, v_q, duration=0.05, dt=1e-5,
                                    load_torque=extra,
                                    initial=dq_sim.DqState(i_d, i_q, w, 0.0))
        self.assertAlmostEqual(float(result.i_d[-1]), i_d, places=12)
        self.assertAlmostEqual(float(result.i_q[-1]), i_q, places=12)
        self.assertAlmostEqual(float(result.motor_speed[-1]), w, places=9)


class EnergyTests(unittest.TestCase):
    def test_input_power_equals_converted_plus_copper(self):
        system = _pod()
        spec = system.group.motor.spec
        result = dq_sim.simulate_dq(system, 0.0, 1.0, duration=0.05, dt=1e-5)
        ch = result.channels()
        count = system.group.count
        for i_d, i_q, w, v_d, v_q, converted, copper, drawn in zip(
                ch["current_d_A"], ch["current_q_A"], result.motor_speed,
                ch["voltage_d_V"], ch["voltage_q_V"],
                ch["power_out_W"], ch["copper_loss_W"], ch["power_in_W"]):
            developed = count * spec.torque_at(i_d, i_q) * w
            # The magnetic field is storing energy too: substituting the voltage
            # equations into 1.5 (v_d i_d + v_q i_q) gives copper loss plus
            # developed power plus 1.5 d/dt (Ld i_d^2 + Lq i_q^2) / 2.  At
            # standstill that last term is most of the input, which is why the
            # shorter balance -- input = copper + developed -- is wrong here and
            # not merely imprecise.
            di_d, di_q = dq_sim.electrical_derivatives(
                spec, dq_sim.DqState(i_d, i_q, w, 0.0), v_d, v_q)
            storage = count * 1.5 * (LD * i_d * di_d + LQ * i_q * di_q)
            # developed (electromagnetic) power + copper loss + field storage
            # = electrical input, exactly
            self.assertAlmostEqual(drawn, count * 1.5 * (v_d * i_d + v_q * i_q), places=12)
            self.assertAlmostEqual(count * 1.5 * (v_d * i_d + v_q * i_q),
                                   developed + storage
                                   + count * 1.5 * R * (i_d * i_d + i_q * i_q),
                                   delta=1e-9)
            self.assertAlmostEqual(copper, count * 1.5 * R * (i_d * i_d + i_q * i_q), places=12)
            self.assertAlmostEqual(converted,
                                   developed * system.gearbox.efficiency, places=12)

    def test_efficiency_stays_in_the_zero_to_one_band(self):
        result = dq_sim.simulate_dq(_pod(), 0.0, 2.0, duration=0.2, dt=1e-5)
        eff = result.channels()["efficiency"]
        self.assertTrue(np.all(eff >= 0.0))
        self.assertTrue(np.all(eff <= 1.0))


class BrushedCollapseTests(unittest.TestCase):
    """The dq plant must reproduce the brushed engine when i_d is held at zero."""

    def _run(self, dt: float):
        flat = presets.build("bldc_pod", {"group.motor.spec.inductance_d": LQ})
        spec = flat.group.motor.spec
        mech = dq_sim._Mechanical.build(flat)
        v_q = 0.6
        state = dq_sim.DqState(0.0, v_q / R, 0.0, 0.0)   # brushed starts at the quasi-steady current
        speeds, currents = [], []
        n = int(0.02 / dt) + 1
        for _ in range(n):
            w_e = POLES * state.rot_vel
            v_d = R * state.i_d - w_e * LQ * state.i_q   # the voltage an ideal d-axis loop applies
            speeds.append(state.rot_vel)
            currents.append(mech.count * state.i_q)
            state = dq_sim.step(flat, state, v_d, v_q, dt)
        t = np.arange(n) * dt
        ref = simulate_rk4(flat, v_q, t, dt=dt)
        return (np.array(speeds), np.array(currents), ref, float(np.max(np.abs(ref.motor_speed))))

    def test_it_collapses_onto_the_brushed_engine(self):
        speeds, currents, ref, scale = self._run(1e-5)
        self.assertLess(float(np.max(np.abs(ref.motor_speed - speeds))) / scale, 1e-6)
        self.assertLess(float(np.max(np.abs(ref.current - currents))), 1e-4)

    def test_the_gap_closes_as_the_step_shrinks(self):
        # The residual is the d-axis voltage held constant across a step while
        # i_q moves inside it; it is a discretisation artefact, not a different
        # model, so it must fall with the step.
        speeds_a, _, ref_a, scale_a = self._run(2e-5)
        speeds_b, _, ref_b, scale_b = self._run(5e-6)
        gap_a = float(np.max(np.abs(ref_a.motor_speed - speeds_a))) / scale_a
        gap_b = float(np.max(np.abs(ref_b.motor_speed - speeds_b))) / scale_b
        self.assertLess(gap_b, gap_a)


class ParkTests(unittest.TestCase):
    def test_a_balanced_sine_maps_to_a_constant_vector(self):
        amplitude, w_e = 12.0, 3.7
        for phase in (0.0, 0.3, 1.1, 2.9):
            th = w_e * phase
            a = amplitude * math.cos(th)
            b = amplitude * math.cos(th - 2.0 * math.pi / 3.0)
            c = amplitude * math.cos(th + 2.0 * math.pi / 3.0)
            d, q = dq_sim.park(th, a, b, c)
            self.assertAlmostEqual(d, amplitude, places=12)
            self.assertAlmostEqual(q, 0.0, places=12)
            self.assertAlmostEqual(math.hypot(d, q), amplitude, places=12)

    def test_the_transform_is_a_round_trip(self):
        for th in (0.0, 0.7, 2.5, -1.9):
            a, b, c = 3.0, -4.0, 1.0
            d, q = dq_sim.park(th, a, b, c)
            back = dq_sim.inverse_park(th, d, q)
            if abs(a + b + c) < 1e-12:
                self.assertAlmostEqual(back[0], a, places=12)
                self.assertAlmostEqual(back[1], b, places=12)
                self.assertAlmostEqual(back[2], c, places=12)
            else:   # zero sequence is what the dq frame cannot see
                self.assertAlmostEqual(dq_sim.zero_sequence(*back), 0.0, places=12)
                self.assertAlmostEqual(sum(back), 0.0, places=12)

    def test_zero_sequence_is_dropped_not_folded_in(self):
        d, q = dq_sim.park(0.5, 5.0, 5.0, 5.0)     # pure common-mode
        self.assertAlmostEqual(d, 0.0, places=12)
        self.assertAlmostEqual(q, 0.0, places=12)
        self.assertAlmostEqual(dq_sim.zero_sequence(5.0, 5.0, 5.0), 5.0, places=12)

    def test_the_amplitude_convention_is_not_rms(self):
        # The 2/3 factor is what makes torque 1.5 p lambda i_q; an RMS convention
        # would put sqrt(2) between this module and every model in the library.
        th = 0.0
        d, _ = dq_sim.park(th, 1.0, 0.0, -1.0)     # a = 1, b = 0, c = -1 at th = 0
        self.assertAlmostEqual(d, 1.0, places=12)


class ChannelTests(unittest.TestCase):
    def test_the_channels_are_named_the_way_the_dataset_names_them(self):
        result = dq_sim.simulate_dq(_pod(), 0.0, 1.0, duration=0.02, dt=1e-5)
        ch = result.channels()
        for name in ("time_s", "current_d_A", "current_q_A", "current_A", "voltage_d_V",
                     "voltage_q_V", "voltage_V", "motor_rad_s", "motor_rpm", "output_rad_s",
                     "output_rpm", "torque_Nm", "rotor_angle_rad", "power_in_W",
                     "copper_loss_W", "power_out_W", "efficiency", "speed_kmh", "speed_m_s"):
            self.assertIn(name, ch)
            self.assertTrue(np.all(np.isfinite(ch[name])), name)

    def test_the_magnitude_channels_are_magnitudes(self):
        result = dq_sim.simulate_dq(_pod(), -0.5, -1.0, duration=0.02, dt=1e-5)
        ch = result.channels()
        self.assertTrue(np.all(ch["current_A"] >= 0.0))
        self.assertTrue(np.all(ch["voltage_V"] >= 0.0))
        np.testing.assert_allclose(ch["current_A"], np.hypot(ch["current_d_A"], ch["current_q_A"]))
        np.testing.assert_allclose(ch["voltage_V"], np.hypot(ch["voltage_d_V"], ch["voltage_q_V"]))

    def test_gear_and_wheel_scaling_match_the_brushed_channels(self):
        system = _pod()
        result = dq_sim.simulate_dq(system, 0.0, 1.0, duration=0.02, dt=1e-5)
        ch = result.channels()
        np.testing.assert_allclose(ch["output_rad_s"],
                                   ch["motor_rad_s"] * system.gearbox.ratio)
        np.testing.assert_allclose(ch["speed_m_s"], ch["output_rad_s"] * system.wheel.radius)
        np.testing.assert_allclose(ch["speed_kmh"], ch["speed_m_s"] * 3.6)
        np.testing.assert_allclose(ch["motor_rpm"], ch["motor_rad_s"] * 30.0 / math.pi)

    def test_the_replayed_voltage_is_stored_verbatim(self):
        system = _pod()
        n = 501
        t = np.linspace(0.0, 0.05, n)
        v_d = np.linspace(0.0, 1.0, n)
        v_q = np.linspace(0.5, 2.0, n)
        result = dq_sim.simulate_dq(system, v_d, v_q, t=t)
        np.testing.assert_allclose(result.v_d, v_d)
        np.testing.assert_allclose(result.v_q, v_q)
        self.assertEqual(result.dt, float(t[1] - t[0]))


class ConfigurationTests(unittest.TestCase):
    def test_a_brushed_system_is_refused_with_a_useful_message(self):
        system = presets.build("pod_4x_geared")
        with self.assertRaises(ConfigError) as ctx:
            dq_sim.simulate_dq(system, 0.0, 1.0, duration=0.01)
        self.assertIn("brushless", str(ctx.exception))
        self.assertIn("MotorModel", str(ctx.exception))

    def test_a_non_uniform_log_is_refused(self):
        system = _pod()
        t = np.array([0.0, 0.001, 0.003, 0.004])
        with self.assertRaises(ConfigError) as ctx:
            dq_sim.simulate_dq(system, 0.0, 1.0, t=t)
        self.assertIn("fixed-step", str(ctx.exception))

    def test_a_voltage_log_of_the_wrong_length_is_refused(self):
        system = _pod()
        with self.assertRaises(ConfigError) as ctx:
            dq_sim.simulate_dq(system, np.zeros(7), 1.0, duration=0.05, dt=1e-3)
        self.assertIn("samples", str(ctx.exception))

    def test_a_negative_load_torque_is_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            dq_sim.simulate_dq(_pod(), 0.0, 1.0, duration=0.01, load_torque=-1.0)
        self.assertIn("magnitude that opposes motion", str(ctx.exception))

    def test_an_unknown_engine_is_refused(self):
        with self.assertRaises(ConfigError):
            dq_sim.simulate_dq(_pod(), 0.0, 1.0, duration=0.01, engine="verlet")

    def test_the_default_step_is_sized_for_the_fastest_speed(self):
        system = _pod()
        spec = system.group.motor.spec
        dt = dq_sim.solver_dt(system, spec, None)
        self.assertLessEqual(dt, dq_sim.max_step_time(spec, BASE_SPEED))
        self.assertLessEqual(dt, dq_sim.max_step_time(spec, 2.0 * BASE_SPEED))
        # and it is not so small that a normal run becomes slow
        self.assertGreater(dt, 1e-5)


class FeedForwardParityTests(unittest.TestCase):
    """The library integrator and the generated firmware code must agree."""

    def test_the_plant_matches_the_generated_dq_module(self):
        # The generated module integrates one machine with a prescribed load
        # torque; the library integrates the whole drivetrain.  Match them where
        # they overlap: the electrical equations, with the shaft speed pinned by
        # an equilibrium load in the library and by the load_torque argument in
        # the generated code.
        import importlib.util
        import sys
        import tempfile

        from drivetrain.export import ExportOptions
        from drivetrain.export_dq import export_dq

        system = _pod()
        spec = system.group.motor.spec
        w = 300.0
        i_d, i_q = -2.0, 5.0
        v_d, v_q = spec.voltage_steady(i_d, i_q, w)

        with tempfile.TemporaryDirectory() as tmp:
            written = export_dq(system, tmp, ExportOptions(dtype="f64", targets=("python",),
                                                           sample_time=1e-4))
            path = next(p for p in written if p.suffix == ".py" and "bench" not in p.name)
            spec_obj = importlib.util.spec_from_file_location("generated_dq", path)
            module = importlib.util.module_from_spec(spec_obj)
            sys.modules["generated_dq"] = module
            spec_obj.loader.exec_module(module)

            # the library's own electrical arithmetic, at the same state
            di_d, di_q = dq_sim.electrical_derivatives(spec, dq_sim.DqState(i_d, i_q, w, 0.0),
                                                       v_d, v_q)
            # the generated derivatives returns (di_d, di_q, dw): the third
            # entry is the shaft, which the generated code integrates against a
            # prescribed load torque rather than a drivetrain.  Note the state
            # order there is (rot_vel, i_d, i_q), not the library's.
            gen_state = module.State(w, i_d, i_q)
            gen_di_d, gen_di_q, gen_dw = module.derivatives(gen_state, v_d, v_q)
            self.assertAlmostEqual(di_d, gen_di_d, places=15)
            self.assertAlmostEqual(di_q, gen_di_q, places=15)
            self.assertAlmostEqual(
                gen_dw,
                (spec.torque_at(i_d, i_q) - spec.loss_torque(w)) / spec.rotor_inertia,
                places=12)
            self.assertAlmostEqual(dq_sim.torque_em(spec, dq_sim.DqState(i_d, i_q, w, 0.0)),
                                   module.torque(i_d, i_q), places=15)
            self.assertAlmostEqual(dq_sim.max_step_time(spec, w), module.max_step_time(w),
                                   places=15)
            self.assertEqual(dq_sim.sub_steps(spec, 1e-3, w), module.sub_steps(1e-3, w))
            self.assertEqual(dq_sim.RK4_STABILITY_LIMIT, module.RK4_STABILITY_LIMIT)
            del sys.modules["generated_dq"]


if __name__ == "__main__":
    unittest.main()
