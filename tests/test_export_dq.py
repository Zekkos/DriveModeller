"""Parity tests for the dq/FOC code export (``drivetrain/export_dq.py``).

Same discipline as ``tests/test_export.py``: the generated module is checked
against the Python engine, never against itself.  A generated constant that
disagrees with :class:`~drivetrain.pmsm.PMSMSpec` is a generator bug; a
generated answer that merely differs in the last digits is a rounding story,
so the tolerances below are chosen from what the fixed-iteration searches can
actually resolve, not from machine precision.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from drivetrain import cli, presets
from drivetrain.exceptions import ConfigError
from drivetrain.export import ExportOptions
from drivetrain.export_dq import (
    DqModelConstants,
    dq_constants_from_system,
    dq_machine_notes,
    export_dq,
    round_dq_to_dtype,
)
from drivetrain.gearbox import Gearbox
from drivetrain.load import RotationalLoad
from drivetrain.motor import MotorGroup
from drivetrain.pmsm import PMSMModel, PMSMSpec
from drivetrain.system import DriveSystem

# speeds probed against the engine: standstill, low speed, the constant-torque
# band edge, inside the hyperbola, base speed, and two points past it
SPEEDS = (0.0, 50.0, 200.0, 447.774641, 700.0, 900.0, 1099.714798, 1500.0, 2500.0)
# fraction of the engine's torque_limit at that speed; > 1.0 asks for the
# impossible on purpose, to pin down what the module does then.  0.0 is in the
# grid because zero torque above the rail speed is the interesting corner
FRACS = (0.0, 0.02, 0.05, 0.2, 0.5, 0.9, 0.95, 1.0, 1.05, 1.2)

# Tolerances are measured, not guessed.  Worst case over the grid below on
# bldc_pod: inside the envelope both implementations solve the same equations and
# the fixed-count searches leave 8.7e-6 relative on the current magnitude and
# 1e-11 A on the pair.  Above the envelope each side runs its own max-torque
# search and they discretise differently: 2.9e-4 relative, 9.8e-3 A on the pair.
MAG_TOL = 1.0e-5
PAIR_TOL = 1.0e-6
OUTSIDE_MAG_TOL = 1.0e-3
OUTSIDE_PAIR_TOL = 2.0e-2
LIMIT_TOL = 2.0e-3


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _f32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _system(**spec_overrides):
    """A single-machine brushless drivetrain, for cases the catalogue can't say."""
    machine = PMSMModel(spec=PMSMSpec(**spec_overrides))
    return DriveSystem(
        MotorGroup(machine, 1),
        Gearbox(ratio=0.25, efficiency=0.9, inertia=1e-4, damping=1e-5),
        RotationalLoad(inertia=2e-3, viscous=1e-4, constant_torque=0.02),
    )


# ---------------------------------------------------------------------------
# one f64 export shared by most of the suite
# ---------------------------------------------------------------------------
_TMP: tempfile.TemporaryDirectory | None = None
_GEN = None
_ENGINE_SPEC = None


def setUpModule() -> None:
    global _TMP, _GEN, _ENGINE_SPEC
    _TMP = tempfile.TemporaryDirectory()
    paths = export_dq(
        presets.build("bldc_pod"),
        _TMP.name,
        ExportOptions(dtype="f64", sample_time=1e-3),
    )
    _GEN = _load_module(paths[0], "drive_model_dq_f64")
    _ENGINE_SPEC = presets.build("bldc_pod").group.motor.spec


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


def _export(system, **options):
    """Export *system* into a throwaway directory and load the module."""
    holder = tempfile.TemporaryDirectory()
    options.setdefault("dtype", "f64")
    paths = export_dq(system, holder.name, ExportOptions(**options))
    module = _load_module(paths[0], "dq_generated_" + str(abs(id(paths)) % 10**8))
    return module, paths, holder


class ConstantsTests(unittest.TestCase):
    """The constants handed to the generated module are the engine's."""

    def setUp(self) -> None:
        self.system = presets.build("bldc_pod")
        self.spec = self.system.group.motor.spec
        self.c = dq_constants_from_system(self.system)

    def test_machine_constants_match_the_spec(self) -> None:
        for name in (
            "resistance",
            "inductance_d",
            "inductance_q",
            "flux_linkage",
            "pole_pairs",
            "rotor_inertia",
            "bearing_damping",
            "static_friction",
            "current_limit",
            "voltage_limit",
            "dc_bus_voltage",
            "core_loss_coefficient",
            "windage_coefficient",
            "flux_temperature_coefficient",
            "reference_temperature",
        ):
            self.assertAlmostEqual(getattr(self.c, name), getattr(self.spec, name),
                                   msg=name)

    def test_motor_count_and_the_drivetrain_it_sits_in(self) -> None:
        self.assertEqual(self.c.motor_count, 2)
        self.assertAlmostEqual(self.c.gear_ratio, self.system.gearbox.ratio)
        self.assertAlmostEqual(self.c.gear_efficiency, self.system.gearbox.efficiency)
        self.assertAlmostEqual(self.c.wheel_radius, self.system.wheel.radius)

    def test_derived_constants_match_the_engine(self) -> None:
        s = self.spec
        self.assertAlmostEqual(self.c.torque_constant, s.torque_constant)
        self.assertAlmostEqual(self.c.back_emf_constant, s.back_emf_constant)
        self.assertAlmostEqual(self.c.reluctance_coefficient,
                               s.inductance_d - s.inductance_q)
        self.assertAlmostEqual(self.c.characteristic_current, s.flux_linkage / s.inductance_d)
        self.assertAlmostEqual(self.c.base_speed, s.base_speed())
        self.assertAlmostEqual(self.c.electrical_time_constant,
                               min(s.inductance_d, s.inductance_q) / s.resistance)

    def test_the_ratio_of_torque_constant_to_back_emf_is_the_park_convention(self) -> None:
        # amplitude-invariant Park: kt / kv = 1.5, and that is a convention, not
        # an inconsistency between the two numbers
        self.assertAlmostEqual(self.c.torque_constant / self.c.back_emf_constant, 1.5)

    def test_id_floor_refuses_to_demagnetise(self) -> None:
        s = self.spec
        self.assertAlmostEqual(self.c.id_floor,
                               -min(s.current_limit, s.flux_linkage / s.inductance_d))
        self.assertLessEqual(abs(self.c.id_floor), s.current_limit)

    def test_constant_torque_edge_matches_the_adapter(self) -> None:
        adapter = self.system.group.motor
        self.assertAlmostEqual(self.c.constant_torque_speed(), adapter.constant_torque_speed())
        # ...and it sits below base speed, which is the whole point
        self.assertLess(self.c.constant_torque_speed(), self.c.base_speed)

    def test_rpm_in_the_derived_table_is_mechanical_rpm(self) -> None:
        derived = self.c.derived_dict()
        self.assertAlmostEqual(derived["base_speed_rpm"], self.c.base_speed * 60.0 / (2.0 * math.pi))
        self.assertAlmostEqual(derived["base_speed_rpm"], 10501.50, places=2)

    def test_group_current_is_the_sum_of_the_machines(self) -> None:
        derived = self.c.derived_dict()
        self.assertAlmostEqual(derived["group_current_limit_A"],
                               self.c.current_limit * self.c.motor_count)
        self.assertAlmostEqual(derived["stall_current_A"], self.c.current_limit)

    def test_the_stable_step_shrinks_with_speed(self) -> None:
        # the dq frame rotates the back-EMF coupling at w_e = p * w, so the RK4
        # pole pair moves up the imaginary axis with speed and the stable step
        # gets shorter.  the brushed model's 0.5 * tau_e is not enough here.
        at_rest = self.c.max_step_time()
        at_base = self.c.max_step_time(self.c.base_speed)
        self.assertAlmostEqual(at_rest, 0.5 * self.c.electrical_time_constant)
        self.assertLess(at_base, at_rest)
        self.assertAlmostEqual(at_base, 1.79753e-4, delta=1e-8)

    def test_a_brushed_system_is_refused_with_an_actionable_message(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            dq_constants_from_system(presets.build("pod_2x_direct"))
        message = str(ctx.exception)
        self.assertIn("needs a brushless machine", message)
        self.assertIn("bldc_pod", message)

    def test_an_empty_target_list_is_refused_with_the_targets_that_exist(self) -> None:
        # every target ExportOptions accepts is implemented now, so the only way
        # to ask for something this export cannot render is to ask for nothing
        with self.assertRaises(ConfigError) as ctx:
            export_dq(presets.build("bldc_pod"), tempfile.mkdtemp(),
                      ExportOptions(targets=()))
        message = str(ctx.exception)
        self.assertIn("cannot emit any of the targets", message)
        self.assertIn("python, c, rust", message)

    def test_f32_rounds_every_constant_once(self) -> None:
        from drivetrain.export_dq import round_dq_to_dtype

        rounded = round_dq_to_dtype(self.c, "f32")
        self.assertAlmostEqual(rounded.resistance, _f32_round(self.c.resistance))
        self.assertAlmostEqual(rounded.flux_linkage, _f32_round(self.c.flux_linkage))
        # notes are prose, not numbers, and must survive untouched
        self.assertEqual(rounded.notes, self.c.notes)


class NotesTests(unittest.TestCase):
    """The notes are what stops a brushless export being used wrong."""

    def test_the_voltage_convention_note_names_the_bus_ratio(self) -> None:
        notes = dq_machine_notes(presets.build("bldc_pod"))
        joined = "\n".join(notes)
        self.assertIn("sqrt(3)", joined)
        self.assertIn("phase", joined)

    def test_the_current_note_says_the_limit_is_per_machine(self) -> None:
        joined = "\n".join(dq_machine_notes(presets.build("bldc_pod")))
        self.assertIn("ONE machine", joined)
        self.assertIn("40 A times 1.5", joined)

    def test_the_band_edge_note_gives_both_sides_of_the_gearbox(self) -> None:
        joined = "\n".join(dq_machine_notes(presets.build("bldc_pod")))
        self.assertIn("447.8 rad/s motor side", joined)
        self.assertIn("74.78 rad/s output", joined)
        self.assertIn("1100 rad/s", joined)

    def test_a_rail_that_is_not_a_bus_is_flagged(self) -> None:
        # PMSMSpec's default dc_bus_voltage is inf: the model then has no bus to
        # talk about, and the generated dc_link_current returns nan
        joined = "\n".join(dq_machine_notes(_system()))
        self.assertIn("nan", joined)
        self.assertNotIn("nan", "\n".join(dq_machine_notes(presets.build("bldc_pod"))))

    def test_a_rating_above_the_characteristic_current_is_flagged(self) -> None:
        # 30 A of current circle against a characteristic current of 12 A: the
        # model refuses to reverse the field past lambda/Ld, so the rest of that
        # circle is unreachable (demagnetisation is not modelled)
        joined = "\n".join(dq_machine_notes(_system(current_limit=30.0)))
        self.assertIn("demagnetisation", joined)
        self.assertIn("-12 A", joined)

    def test_a_rating_below_the_characteristic_current_is_not_flagged(self) -> None:
        # 2 A sits well inside lambda/Ld, so the whole circle is usable
        joined = "\n".join(dq_machine_notes(_system(current_limit=2.0)))
        self.assertNotIn("demagnetisation", joined)

    def test_a_non_salient_machine_says_so(self) -> None:
        system = _system(inductance_d=1.5e-4, inductance_q=1.5e-4)
        joined = "\n".join(dq_machine_notes(system))
        self.assertIn("reluctance", joined)

    def test_notes_survive_into_the_manifest_and_the_readme(self) -> None:
        holder = tempfile.TemporaryDirectory()
        paths = export_dq(presets.build("bldc_pod"), holder.name, ExportOptions())
        manifest = json.loads((paths[-2]).read_text())
        self.assertTrue(manifest["notes"])
        readme = paths[-1].read_text()
        self.assertIn("sqrt(3)", readme)
        holder.cleanup()


class StaticParityTests(unittest.TestCase):
    """Algebraic functions, checked point by point against the engine."""

    def setUp(self) -> None:
        self.g = _GEN
        self.s = _ENGINE_SPEC

    def test_torque_matches_the_engine(self) -> None:
        for i_d in (0.0, -3.0, -12.0, 5.0):
            for i_q in (0.0, 1.0, -7.5, 20.0):
                self.assertAlmostEqual(self.g.torque(i_d, i_q), self.s.torque_at(i_d, i_q),
                                       msg=f"torque({i_d}, {i_q})")

    def test_torque_gain_matches_the_engine(self) -> None:
        for i_d in (0.0, -1.0, -12.0, -20.0):
            self.assertAlmostEqual(self.g.torque_gain(i_d), self.s.torque_gain_at(i_d),
                                   msg=f"torque_gain({i_d})")

    def test_voltage_matches_the_engine(self) -> None:
        for w in SPEEDS:
            for i_d, i_q in ((0.0, 10.0), (-5.0, 15.0), (-12.0, 6.0)):
                v_g = self.g.voltage(w, i_d, i_q)
                v_e = self.s.voltage_steady(i_d, i_q, w)
                self.assertAlmostEqual(v_g[0], v_e[0], places=9, msg=f"v_d at {w}")
                self.assertAlmostEqual(v_g[1], v_e[1], places=9, msg=f"v_q at {w}")

    def test_the_transient_terms_are_there(self) -> None:
        # Ld*di_d/dt and Lq*di_q/dt are why the plant needs a small step at all
        steady = self.g.voltage(0.0, 2.0, 5.0)
        ramped = self.g.voltage(0.0, 2.0, 5.0, 1000.0, -2000.0)
        self.assertAlmostEqual(ramped[0] - steady[0], self.s.inductance_d * 1000.0)
        self.assertAlmostEqual(ramped[1] - steady[1], self.s.inductance_q * -2000.0)

    def test_loss_torque_matches_the_engine(self) -> None:
        for w in (0.0, 10.0, -10.0, 500.0, -2500.0):
            self.assertAlmostEqual(self.g.loss_torque(w), self.s.loss_torque(w),
                                   msg=f"loss_torque({w})")

    def test_required_torque_uses_the_opposing_load_convention(self) -> None:
        # load_torque is a magnitude that opposes motion, same as the engine
        self.assertAlmostEqual(self.g.required_torque(0.0, 100.0, 0.02),
                               self.s.required_torque(0.0, 100.0, 0.02))
        self.assertAlmostEqual(self.g.required_torque(300.0, -50.0, 0.02),
                               self.s.required_torque(300.0, -50.0, 0.02))
        with self.assertRaises(ValueError):
            self.g.required_torque(10.0, 0.0, -0.02)

    def test_losses_match_the_engine_at_a_working_point(self) -> None:
        i_d, i_q = -4.0, 14.0
        for w in (0.0, 400.0, 1500.0):
            got = self.g.losses(w, i_d, i_q)
            self.assertAlmostEqual(got["copper_W"], self.s.copper_power(i_d, i_q))
            self.assertAlmostEqual(got["iron_W"], self.s.core_loss_power(w))
            self.assertAlmostEqual(got["windage_W"], self.s.windage_power(w))
            self.assertAlmostEqual(got["bearing_W"], self.s.bearing_damping * w * w)
            self.assertAlmostEqual(got["converted_W"],
                                   self.s.torque_at(i_d, i_q) * w)
            # the three drags together are exactly what the engine calls loss power
            self.assertAlmostEqual(
                got["iron_W"] + got["windage_W"] + got["bearing_W"],
                self.s.loss_torque(w) * abs(w))
            v_d, v_q = self.s.voltage_steady(i_d, i_q, w)
            self.assertAlmostEqual(got["input_W"],
                                   self.s.input_power(v_d, v_q, i_d, i_q))

    def test_the_mechanical_drags_are_not_added_to_the_input(self) -> None:
        # iron, windage and bearing act on the shaft, so they sit inside the
        # converted power; adding them to the input as well counts them twice
        w, i_d, i_q = 400.0, -4.0, 14.0
        got = self.g.losses(w, i_d, i_q)
        self.assertAlmostEqual(got["input_W"],
                               got["copper_W"] + got["converted_W"])
        self.assertAlmostEqual(got["shaft_W"], got["converted_W"]
                               - self.s.loss_torque(w) * abs(w))

    def test_dc_link_current_carries_the_three_phase_factor(self) -> None:
        # a rail ammeter reads 1.5 * (v_d i_d + v_q i_q) / V_bus summed over the
        # machines, not the winding current
        adapter = presets.build("bldc_pod").group.motor
        for w, i in ((0.0, 10.0), (300.0, 12.0), (900.0, 5.0)):
            self.assertAlmostEqual(
                self.g.dc_link_current(w, 0.0, i),
                self.g.MOTOR_COUNT * adapter.dc_link_current(i, w),
                msg=f"dc_link_current({w}, 0, {i})")
        # ...and it is not the winding current
        self.assertNotAlmostEqual(self.g.dc_link_current(300.0, 0.0, 12.0), 12.0)

    def test_dc_link_current_is_nan_without_a_bus(self) -> None:
        module, _paths, holder = _export(_system())
        self.assertTrue(math.isnan(module.dc_link_current(300.0, 0.0, 10.0)))
        holder.cleanup()


if __name__ == "__main__":
    unittest.main()


class CurrentParityTests(unittest.TestCase):
    """current_for_torque() must land where the engine lands.

    The engine solves the same problem with adaptive searches; the generated
    module uses fixed iteration counts so the same code runs in the same time on
    a microcontroller.  Parity is therefore asserted on the current magnitude
    tightly and on the (i_d, i_q) pair loosely: two angles can carry the same
    current and the same torque.
    """

    def setUp(self) -> None:
        self.gen = _GEN
        self.spec = _ENGINE_SPEC

    def _check(self, torque: float, rot_vel: float) -> None:
        e_id, e_iq, e_cur, e_vol = self.spec.current_pair(torque, rot_vel)
        g_id, g_iq, g_cur, g_vol = self.gen.current_for_torque(torque, rot_vel)
        e_mag, g_mag = math.hypot(e_id, e_iq), math.hypot(g_id, g_iq)
        tol = MAG_TOL if abs(torque) <= self.spec.torque_limit(rot_vel) else OUTSIDE_MAG_TOL
        pair_tol = PAIR_TOL if abs(torque) <= self.spec.torque_limit(rot_vel) else OUTSIDE_PAIR_TOL
        self.assertLessEqual(
            abs(g_mag - e_mag) / max(e_mag, 1e-12),
            tol,
            f"current magnitude at tau={torque:.6g}, w={rot_vel:.6g}: "
            f"generated {g_mag:.9g} A against engine {e_mag:.9g} A",
        )
        self.assertLessEqual(
            abs(g_id - e_id),
            pair_tol,
            f"i_d at tau={torque:.6g}, w={rot_vel:.6g}: {g_id:.9g} against {e_id:.9g}",
        )
        self.assertLessEqual(
            abs(g_iq - e_iq),
            pair_tol,
            f"i_q at tau={torque:.6g}, w={rot_vel:.6g}: {g_iq:.9g} against {e_iq:.9g}",
        )
        self.assertEqual(
            (g_cur, g_vol),
            (e_cur, e_vol),
            f"saturation flags at tau={torque:.6g}, w={rot_vel:.6g}: "
            f"generated {(g_cur, g_vol)} against engine {(e_cur, e_vol)}",
        )

    def test_every_speed_and_torque_on_the_grid(self) -> None:
        for rot_vel in SPEEDS:
            limit = self.spec.torque_limit(rot_vel)
            for frac in FRACS:
                for sign in (1.0, -1.0):
                    with self.subTest(rot_vel=rot_vel, frac=frac, sign=sign):
                        self._check(sign * frac * limit, rot_vel)

    def test_the_generated_answer_needs_no_more_current_than_the_engine(self) -> None:
        # the whole point of the search is minimum current; a generator that
        # settles on a worse angle is a bug even if the torque is right
        worse = 0.0
        for rot_vel in SPEEDS:
            limit = self.spec.torque_limit(rot_vel)
            for frac in FRACS:
                torque = frac * limit
                e_mag = math.hypot(*self.spec.current_pair(torque, rot_vel)[:2])
                g_mag = math.hypot(*self.gen.current_for_torque(torque, rot_vel)[:2])
                worse = max(worse, (g_mag - e_mag) / max(e_mag, 1e-12))
        self.assertLess(worse, 5e-4, "the generated search used more current than the engine")

    def test_the_torque_actually_comes_out(self) -> None:
        for rot_vel in SPEEDS:
            for frac in (0.05, 0.5, 1.0):
                torque = frac * self.spec.torque_limit(rot_vel)
                i_d, i_q, _, _ = self.gen.current_for_torque(torque, rot_vel)
                with self.subTest(rot_vel=rot_vel, frac=frac):
                    self.assertAlmostEqual(
                        self.gen.torque(i_d, i_q), torque, delta=1e-6 * max(torque, 1e-3)
                    )

    def test_the_current_angle_stays_inside_the_circle_and_the_ellipse(self) -> None:
        for rot_vel in SPEEDS:
            for frac in FRACS:
                i_d, i_q, _, _ = self.gen.current_for_torque(
                    frac * self.spec.torque_limit(rot_vel), rot_vel
                )
                with self.subTest(rot_vel=rot_vel, frac=frac):
                    self.assertLessEqual(math.hypot(i_d, i_q), self.gen.CURRENT_LIMIT * 1.001)
                    self.assertLessEqual(
                        self.gen.voltage_magnitude(rot_vel, i_d, i_q),
                        self.gen.VOLTAGE_LIMIT * 1.001,
                    )

    def test_the_angle_is_the_same_for_a_torque_of_either_sign(self) -> None:
        # i_d is the field angle, not a function of the torque's sign: the true
        # symmetry is (tau, w) -> (-tau, -w), which flips i_q and leaves i_d
        # alone.  Flipping the torque at a FIXED speed is a different question -
        # that is braking rather than driving, and the braking angle is not the
        # driving angle once the voltage ellipse bites.
        for rot_vel in SPEEDS:
            torque = 0.5 * self.spec.torque_limit(rot_vel)
            up = self.gen.current_for_torque(torque, rot_vel)
            down = self.gen.current_for_torque(-torque, -rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(up[0], down[0], delta=PAIR_TOL)
                self.assertAlmostEqual(up[1], -down[1], delta=PAIR_TOL)
                self.assertEqual(up[2:], down[2:])


class ZeroTorqueTests(unittest.TestCase):
    """Zero torque above the rail speed is the corner that bites a feed-forward drive."""

    def setUp(self) -> None:
        self.gen = _GEN
        self.spec = _ENGINE_SPEC

    def test_zero_torque_still_needs_demagnetising_current_above_the_rail_speed(self) -> None:
        # past w = V/(p*lambda) the back-EMF alone is off the rail, so i_d = 0
        # cannot even be commanded; the module has to weaken the field to keep
        # zero torque controllable, exactly as the engine does
        for rot_vel in (1200.0, 1500.0, 2500.0):
            e_id, e_iq, e_cur, e_vol = self.spec.current_pair(0.0, rot_vel)
            g_id, g_iq, g_cur, g_vol = self.gen.current_for_torque(0.0, rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertLess(g_id, -1.0, "no demagnetising current was found")
                self.assertAlmostEqual(g_id, e_id, delta=PAIR_TOL)
                self.assertAlmostEqual(g_iq, e_iq, delta=PAIR_TOL)
                self.assertEqual((g_cur, g_vol), (e_cur, e_vol))

    def test_the_zero_torque_current_is_the_largest_feasible_i_d(self) -> None:
        # the search bisects onto the ellipse, so the answer sits where |v| is
        # exactly the rail: a hair less demagnetising current and it is infeasible
        for rot_vel in (1200.0, 2500.0):
            i_d, i_q, _, _ = self.gen.current_for_torque(0.0, rot_vel)
            rail = self.gen.VOLTAGE_LIMIT
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(
                    self.gen.voltage_magnitude(rot_vel, i_d, i_q), rail, delta=1e-6 * rail
                )
                self.assertGreater(
                    self.gen.voltage_magnitude(rot_vel, i_d + 0.05, i_q), rail
                )

    def test_zero_torque_below_the_rail_speed_needs_no_current_at_all(self) -> None:
        for rot_vel in (0.0, 200.0, 447.774641, 900.0):
            i_d, i_q, cur, vol = self.gen.current_for_torque(0.0, rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(i_d, 0.0, delta=1e-12)
                self.assertAlmostEqual(i_q, 0.0, delta=1e-12)
                self.assertFalse(cur)
                self.assertFalse(vol)


class MtpaTests(unittest.TestCase):
    """The closed-form MTPA angle must match the engine's iterative one."""

    def setUp(self) -> None:
        self.gen = _GEN
        self.spec = _ENGINE_SPEC

    def test_the_closed_form_matches_the_engine(self) -> None:
        for rot_vel in (0.0, 50.0, 200.0, 447.774641, 700.0, 1099.714798):
            limit = self.spec.torque_limit(rot_vel)
            for frac in (0.02, 0.2, 0.5, 0.9, 1.0):
                for sign in (1.0, -1.0):
                    torque = sign * frac * limit
                    g_id, g_iq = self.gen.mtpa_pair(torque)
                    e_id, e_iq = self.spec.mtpa_pair(torque)
                    g_mag, e_mag = math.hypot(g_id, g_iq), math.hypot(e_id, e_iq)
                    with self.subTest(rot_vel=rot_vel, frac=frac, sign=sign):
                        self.assertLess(
                            abs(g_mag - e_mag) / max(e_mag, 1e-12),
                            1e-9,
                            f"MTPA current at tau={torque:.6g}: {g_mag:.9g} against {e_mag:.9g}",
                        )
                        self.assertAlmostEqual(g_id, e_id, delta=1e-9)
                        self.assertAlmostEqual(g_iq, e_iq, delta=1e-9)

    def test_the_closed_form_uses_lq_minus_ld_not_the_reluctance_constant(self) -> None:
        # RELUCTANCE is Ld - Lq because it multiplies i_d*i_q in the torque; the
        # MTPA denominator is the opposite sign.  Getting this wrong inverts i_d
        # and costs up to 6% of the current.
        self.assertAlmostEqual(self.gen.MTPA_DL, -self.gen.RELUCTANCE)
        self.assertLess(self.gen.mtpa_pair(0.2)[0], 0.0, "MTPA i_d must be demagnetising")

    def test_a_torque_the_current_circle_cannot_reach_returns_no_pair(self) -> None:
        i_d, i_q = self.gen.mtpa_pair(10.0)
        self.assertEqual(i_q, math.inf)
        self.assertEqual(i_d, 0.0)

    def test_a_non_salient_machine_has_no_mtpa_angle(self) -> None:
        gen, _, holder = _export(_system(inductance_d=2.0e-4, inductance_q=2.0e-4))
        try:
            self.assertEqual(gen.MTPA_DL, 0.0)
            i_d, i_q = gen.mtpa_pair(0.1)
            self.assertEqual(i_d, 0.0)
            self.assertAlmostEqual(i_q, 0.1 / gen.TORQUE_CONSTANT, delta=1e-12)
        finally:
            holder.cleanup()


class MaxTorqueTests(unittest.TestCase):
    """max_torque() is the envelope the drive can command at a speed."""

    def setUp(self) -> None:
        self.gen = _GEN
        self.spec = _ENGINE_SPEC

    def test_forward_it_matches_the_motoring_envelope(self) -> None:
        for rot_vel in (0.0, 200.0, 447.774641, 700.0, 1099.714798, 1500.0, 2500.0):
            got = self.gen.max_torque(rot_vel)[0]
            want = self.spec.torque_limit(rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertLess(abs(got - want) / max(want, 1e-12), LIMIT_TOL)

    def test_backward_it_matches_the_regenerative_envelope(self) -> None:
        # max_torque returns the largest POSITIVE torque, which at a negative
        # speed is braking: that envelope is bigger than the motoring one above
        # base speed, and the engine says so with regenerative=True
        for rot_vel in (-700.0, -1099.714798, -1500.0, -2500.0):
            got = self.gen.max_torque(rot_vel)[0]
            want = self.spec.torque_limit(rot_vel, regenerative=True)
            motoring = self.spec.torque_limit(rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertLess(abs(got - want) / max(want, 1e-12), LIMIT_TOL)
                self.assertGreater(got, motoring, "braking should not be smaller than driving")

    def test_below_the_band_edge_the_current_circle_binds_at_the_mtpa_angle(self) -> None:
        for rot_vel in (0.0, 200.0, 447.774641):
            torque, i_d, i_q = self.gen.max_torque(rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(math.hypot(i_d, i_q), self.gen.CURRENT_LIMIT, delta=1e-9)
                self.assertGreater(i_d, self.gen.ID_FLOOR, "the floor is not what binds here")

    def test_above_the_band_edge_the_demagnetisation_floor_binds(self) -> None:
        for rot_vel in (700.0, 1099.714798, 1500.0, 2500.0):
            torque, i_d, i_q = self.gen.max_torque(rot_vel)
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(i_d, self.gen.ID_FLOOR, delta=1e-9)
                self.assertLess(math.hypot(i_d, i_q), self.gen.CURRENT_LIMIT)
                self.assertAlmostEqual(
                    self.gen.voltage_magnitude(rot_vel, i_d, i_q),
                    self.gen.VOLTAGE_LIMIT,
                    delta=1e-6 * self.gen.VOLTAGE_LIMIT,
                )

    def test_the_envelope_falls_off_past_base_speed(self) -> None:
        torques = [self.gen.max_torque(w)[0] for w in (0.0, 447.774641, 1099.714798, 2500.0)]
        self.assertEqual(torques, sorted(torques, reverse=True), torques)


class FeedForwardTests(unittest.TestCase):
    """The clocked current law: what a drive would apply with no current loop."""

    def setUp(self) -> None:
        self.gen = _GEN
        self.spec = _ENGINE_SPEC

    def test_the_defaults_are_the_machine_own_limits(self) -> None:
        ff = self.gen.FeedForward(sample_time=1e-3)
        self.assertEqual(ff.speed_limit, self.gen.BASE_SPEED)
        stall = self.gen.TORQUE_CONSTANT * min(
            self.gen.VOLTAGE_LIMIT / self.gen.RESISTANCE, self.gen.CURRENT_LIMIT
        )
        self.assertAlmostEqual(ff.accel_limit, stall / self.gen.ROTOR_INERTIA, delta=1e-9)
        self.assertEqual(self.gen.FeedForward.stall_torque(), stall)

    def test_a_sample_time_of_zero_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.gen.FeedForward(sample_time=0.0)

    def test_the_ramp_clamps_the_increment_not_the_absolute_value(self) -> None:
        # clamping the absolute value is what pins a step command at a crawl and
        # looks like a broken model rather than a broken limiter
        ff = self.gen.FeedForward(sample_time=1e-3, accel_limit=100.0)
        ff.reset(0.0)
        w_ref, a_ref = ff.ramp(1000.0)
        self.assertAlmostEqual(w_ref, 0.1, places=12)
        self.assertAlmostEqual(a_ref, 100.0, places=9)
        for _ in range(9):
            w_ref, a_ref = ff.ramp(1000.0)
        self.assertAlmostEqual(w_ref, 1.0, places=12)
        self.assertAlmostEqual(a_ref, 100.0, places=9)

    def test_the_reference_never_runs_past_its_own_speed_limit(self) -> None:
        ff = self.gen.FeedForward(sample_time=1e-3, speed_limit=500.0, accel_limit=1e9)
        ff.reset(0.0)
        for _ in range(50):
            w_ref, _ = ff.ramp(5000.0)
        self.assertAlmostEqual(w_ref, 500.0, places=9)
        w_ref, _ = ff.ramp(-5000.0)
        self.assertLessEqual(w_ref, 500.0)

    def test_the_current_matches_the_engine_for_the_same_demand(self) -> None:
        ff = self.gen.FeedForward(sample_time=1e-3, load_torque=0.01)
        for rot_vel in (0.0, 200.0, 447.774641, 700.0, 1099.714798):
            for accel in (0.0, 50.0, -50.0):
                tau = self.gen.required_torque(rot_vel, accel, 0.01)
                want = self.spec.current_pair(tau, rot_vel)[:2]
                got = ff.current(rot_vel, accel)
                with self.subTest(rot_vel=rot_vel, accel=accel):
                    self.assertLess(
                        abs(math.hypot(*got) - math.hypot(*want)) / max(math.hypot(*want), 1e-12),
                        MAG_TOL,
                    )

    def test_update_returns_the_voltage_that_the_law_asks_for(self) -> None:
        ff = self.gen.FeedForward(sample_time=1e-3)
        ff.reset(200.0)
        v_d, v_q, i_d, i_q = ff.update(200.0)
        # steady speed, no load: the only torque to fight is the machine's own
        # drag, so the current is small and i_d is the MTPA angle for that small
        # torque rather than exactly zero
        drag = self.gen.loss_torque(200.0)
        self.assertGreater(i_q, 0.0)
        self.assertLess(i_q, 0.1)
        self.assertLess(i_d, 0.0)
        self.assertGreater(i_d, -1e-3)
        self.assertAlmostEqual(self.gen.torque(i_d, i_q), drag, delta=1e-9)
        # the applied voltage is the back-EMF plus the resistive drop on that
        # current; the d-axis flux term is inside the cross-coupling product, so
        # kv*w alone is short by we*Ld*i_d (small here, but not zero)
        we = self.gen.POLE_PAIRS * 200.0
        self.assertAlmostEqual(
            v_q,
            self.gen.RESISTANCE * i_q + we * (self.gen.INDUCTANCE_D * i_d + self.gen.FLUX_LINKAGE),
            delta=1e-12,
        )
        self.assertAlmostEqual(
            v_d,
            self.gen.RESISTANCE * i_d
            - self.gen.electrical_speed(200.0) * self.gen.INDUCTANCE_Q * i_q,
            delta=1e-12,
        )
        self.assertFalse(ff.current_saturated)
        self.assertFalse(ff.voltage_saturated)

    def test_the_voltage_is_clipped_to_the_rail_and_the_flag_latches(self) -> None:
        # a demand the machine cannot serve: full current at a speed where the
        # back-EMF already eats the rail
        ff = self.gen.FeedForward(sample_time=1e-3, speed_limit=3000.0, accel_limit=1e6)
        ff.reset(1500.0)
        v_d, v_q, i_d, i_q = ff.update(3000.0)
        mag = math.hypot(v_d, v_q)
        self.assertLessEqual(mag, self.gen.VOLTAGE_LIMIT * 1.000001)
        self.assertAlmostEqual(mag, self.gen.VOLTAGE_LIMIT, delta=1e-9)
        self.assertTrue(ff.voltage_saturated)

    def test_the_current_flag_fires_below_the_band_edge_where_the_circle_binds(self) -> None:
        # keep the reference slow so the demand lands at a speed where the
        # current circle, not the voltage ellipse, is what runs out first
        ff = self.gen.FeedForward(sample_time=1e-3, speed_limit=50.0, accel_limit=1e9)
        ff.reset(0.0)
        ff.update(50.0)
        self.assertTrue(ff.current_saturated)
        i_d, i_q = ff.current(50.0, 5e4)
        self.assertAlmostEqual(math.hypot(i_d, i_q), self.gen.CURRENT_LIMIT, delta=1e-9)


class StepTests(unittest.TestCase):
    """The plant: RK4 on (i_d, i_q, w), and the step contract that keeps it stable."""

    def setUp(self) -> None:
        self.gen = _GEN

    def _reference(self, state, v_d, v_q, dt, load):
        """An independent RK4 of the published dq equations, written out here."""
        R, Ld, Lq = self.gen.RESISTANCE, self.gen.INDUCTANCE_D, self.gen.INDUCTANCE_Q
        lam, p = self.gen.FLUX_LINKAGE, self.gen.POLE_PAIRS
        J, B = self.gen.ROTOR_INERTIA, self.gen.BEARING_DAMPING

        def f(w, i_d, i_q):
            we = p * w
            tau = 1.5 * p * (lam * i_q + (Ld - Lq) * i_d * i_q)
            return (
                (tau - B * w - load) / J,
                (v_d - R * i_d + we * Lq * i_q) / Ld,
                (v_q - R * i_q - we * (Ld * i_d + lam)) / Lq,
            )

        w, i_d, i_q = state
        k1 = f(w, i_d, i_q)
        k2 = f(w + 0.5 * dt * k1[0], i_d + 0.5 * dt * k1[1], i_q + 0.5 * dt * k1[2])
        k3 = f(w + 0.5 * dt * k2[0], i_d + 0.5 * dt * k2[1], i_q + 0.5 * dt * k2[2])
        k4 = f(w + dt * k3[0], i_d + dt * k3[1], i_q + dt * k3[2])
        return (
            w + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            i_d + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            i_q + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        )

    def test_the_integrator_matches_an_independent_rk4(self) -> None:
        for w0 in (0.0, 300.0, 1099.714798, 2500.0):
            dt = self.gen.max_step_time(w0) / 8.0
            got = self.gen.State(rot_vel=w0, i_d=0.0, i_q=0.0)
            ref = (w0, 0.0, 0.0)
            for _ in range(int(round(0.004 / dt))):
                got = self.gen.step(got, 0.0, 6.0, dt)
                ref = self._reference(ref, 0.0, 6.0, dt, 0.0)
            with self.subTest(w0=w0):
                self.assertAlmostEqual(got.rot_vel, ref[0], delta=1e-9)
                self.assertAlmostEqual(got.i_d, ref[1], delta=1e-9)
                self.assertAlmostEqual(got.i_q, ref[2], delta=1e-9)

    def test_the_stable_step_shrinks_with_speed(self) -> None:
        # the dq mode is a COMPLEX pair at -R(Ld+Lq)/(2 Ld Lq) +/- j*w_e, so the
        # step RK4 tolerates falls as the frame rotates faster.  The brushed
        # model's 0.5*tau_e is speed-independent and would be wrong here.
        at_rest = self.gen.max_step_time(0.0)
        at_base = self.gen.max_step_time(self.gen.BASE_SPEED)
        self.assertAlmostEqual(at_rest, 0.5 * self.gen.ELECTRICAL_TIME_CONSTANT, delta=1e-15)
        self.assertLess(at_base, at_rest / 2.0)
        self.assertLess(self.gen.max_step_time(2 * self.gen.BASE_SPEED), at_base)
        self.assertAlmostEqual(
            self.gen.max_step_time(self.gen.BASE_SPEED),
            self.gen.RK4_STABILITY_LIMIT / self.gen.pole_magnitude(self.gen.BASE_SPEED),
            delta=1e-15,
        )

    def test_the_guard_refuses_a_step_the_speed_cannot_tolerate(self) -> None:
        dt = 5e-4
        with self.assertRaises(ValueError) as ctx:
            self.gen.step(self.gen.State(rot_vel=self.gen.BASE_SPEED), 0.0, 1.0, dt)
        message = str(ctx.exception)
        self.assertIn("exceeds the stable step", message)
        self.assertIn("subdivide the step", message)
        self.assertIn("pole magnitude", message)

    def test_sub_steps_grows_with_speed(self) -> None:
        self.assertEqual(self.gen.sub_steps(1e-3), 2)
        self.assertEqual(self.gen.sub_steps(1e-3, self.gen.BASE_SPEED), 6)
        self.assertEqual(self.gen.sub_steps(1e-3, 2500.0), 13)

    def test_integrate_resizes_from_the_speed_it_has_reached(self) -> None:
        # a fixed subdivision sized for standstill is exactly the bug that makes
        # a high-speed run diverge
        state = self.gen.State(rot_vel=2500.0, i_d=0.0, i_q=0.0)
        for _ in range(20):
            state = self.gen.integrate(state, 0.0, 1.0, 1e-3)
        self.assertTrue(math.isfinite(state.rot_vel))
        self.assertTrue(math.isfinite(state.i_d) and math.isfinite(state.i_q))
        self.assertLess(abs(state.i_d), 2 * self.gen.CURRENT_LIMIT)

    def test_the_plant_settles_where_the_analytic_model_says(self) -> None:
        # hold the voltage the analytic model needs for a working point, and the
        # RK4 plant should land on that same current pair
        for rot_vel in (447.774641, 1099.714798, 2500.0):
            tau = 0.5 * self.gen.max_torque(rot_vel)[0]
            i_d, i_q, _, _ = self.gen.current_for_torque(tau, rot_vel)
            v_d, v_q = self.gen.voltage(rot_vel, i_d, i_q)
            load = self.gen.torque(i_d, i_q) - self.gen.loss_torque(rot_vel)
            state = self.gen.State(rot_vel=rot_vel, i_d=0.0, i_q=0.0)
            for _ in range(int(round(20 * self.gen.ELECTRICAL_TIME_CONSTANT / 1e-4))):
                state = self.gen.integrate(state, v_d, v_q, 1e-4, load_torque=load)
            with self.subTest(rot_vel=rot_vel):
                self.assertAlmostEqual(state.i_d, i_d, delta=2e-2)
                self.assertAlmostEqual(state.i_q, i_q, delta=2e-2)
                self.assertLess(abs(state.rot_vel - rot_vel) / rot_vel, 2e-3)

    def test_step_current_holds_the_torque_constant_and_drives_only_the_shaft(self) -> None:
        state = self.gen.State(rot_vel=0.0, i_d=0.0, i_q=0.0)
        nxt = self.gen.step_current(state, 0.0, 5.0, 1e-4)
        self.assertAlmostEqual(nxt.i_d, 0.0, delta=1e-12)
        self.assertAlmostEqual(nxt.i_q, 5.0, delta=1e-12, msg="the current loop is assumed perfect")
        self.assertGreater(nxt.rot_vel, 0.0)
        tau = self.gen.torque(0.0, 5.0)
        self.assertAlmostEqual(
            nxt.rot_vel,
            (tau - self.gen.loss_torque(0.0)) * 1e-4 / self.gen.ROTOR_INERTIA,
            delta=1e-5,
            msg="RK4 evaluates the drag mid-step, so a first-order estimate is off by O(dt^2)",
        )

    def test_the_plant_and_the_analytic_model_agree_on_the_steady_state(self) -> None:
        # the plant's equilibrium at a fixed voltage IS the analytic operating
        # point: this is what makes the feed-forward law usable open loop
        for rot_vel in (0.0, 200.0):
            i_d, i_q = 0.0, 3.0
            v_d, v_q = self.gen.voltage(rot_vel, i_d, i_q)
            self.assertAlmostEqual(self.gen.voltage_magnitude(rot_vel, i_d, i_q), math.hypot(v_d, v_q))
            # at those currents and that speed, di_d/dt and di_q/dt are zero by
            # construction; the shaft only stands still if the load balances the
            # torque, which is the operating point the analytic model describes
            state = self.gen.State(rot_vel=rot_vel, i_d=i_d, i_q=i_q)
            balance = self.gen.torque(i_d, i_q) - self.gen.loss_torque(rot_vel)
            d_id, d_iq, d_w = self.gen.derivatives(state, v_d, v_q, load_torque=balance)
            self.assertAlmostEqual(d_id, 0.0, delta=1e-9)
            self.assertAlmostEqual(d_iq, 0.0, delta=1e-9)
            self.assertAlmostEqual(d_w, 0.0, delta=1e-9)


class MachineVariantTests(unittest.TestCase):
    """Machines the catalogue does not carry, because they break different branches."""

    def test_a_rating_below_the_characteristic_current_can_not_fully_demagnetise(self) -> None:
        # 2 A rated against a 12 A characteristic current: i_d can never reach
        # -lambda/Ld, so past the rail speed even zero torque needs the largest
        # i_d that fits, and the module has to say the voltage is what binds
        system = _system(current_limit=2.0, voltage_limit=24.0)
        gen, _, holder = _export(system)
        try:
            spec = system.group.motor.spec
            self.assertEqual(gen.CHARACTERISTIC_CURRENT, 12.0)
            self.assertEqual(gen.ID_FLOOR, -2.0, "the floor follows the rating, not the flux")
            self.assertGreater(gen.BASE_SPEED, gen.constant_torque_speed())
            for rot_vel in (0.0, 500.0, 1000.0, gen.BASE_SPEED, 2000.0):
                got = gen.current_for_torque(0.0, rot_vel)
                want = spec.current_pair(0.0, rot_vel)
                with self.subTest(rot_vel=rot_vel):
                    self.assertAlmostEqual(got[0], want[0], delta=PAIR_TOL)
                    self.assertAlmostEqual(got[1], want[1], delta=PAIR_TOL)
                    self.assertEqual(got[2:], want[2:])
            # past the rail speed the answer is the largest feasible i_d, and it
            # sits exactly on the ellipse
            i_d, i_q, cur, vol = gen.current_for_torque(0.0, 2000.0)
            self.assertLess(i_d, 0.0)
            self.assertFalse(cur)
            self.assertTrue(vol)
            self.assertAlmostEqual(
                gen.voltage_magnitude(2000.0, i_d, i_q), gen.VOLTAGE_LIMIT, delta=1e-9
            )
        finally:
            holder.cleanup()

    def test_the_envelope_of_a_current_limited_machine_matches_the_engine(self) -> None:
        system = _system(current_limit=2.0, voltage_limit=24.0)
        gen, _, holder = _export(system)
        try:
            spec = system.group.motor.spec
            for rot_vel in (0.0, 500.0, 1000.0, gen.BASE_SPEED, 2000.0):
                want = spec.torque_limit(rot_vel)
                got = gen.max_torque(rot_vel)[0]
                with self.subTest(rot_vel=rot_vel):
                    self.assertLess(abs(got - want) / want, LIMIT_TOL)
        finally:
            holder.cleanup()

    def test_a_non_salient_machine_weakens_on_current_alone(self) -> None:
        # Ld == Lq: no reluctance torque, and the MTPA closed form degenerates to
        # i_d = 0; field weakening is then purely demagnetising current.  The
        # cross-coupling term -w_e*Lq*i_q is what eats the rail here, and it
        # binds before the demagnetisation floor does.
        system = _system(inductance_d=2.0e-4, inductance_q=2.0e-4, current_limit=20.0)
        gen, _, holder = _export(system)
        try:
            spec = system.group.motor.spec
            self.assertEqual(gen.MTPA_DL, 0.0)
            self.assertEqual(gen.RELUCTANCE, 0.0)
            self.assertEqual(gen.CHARACTERISTIC_CURRENT, 9.0)
            self.assertEqual(gen.ID_FLOOR, -9.0)
            for rot_vel in (0.0, 400.0, 800.0, 1500.0):
                torque, i_d, i_q = gen.max_torque(rot_vel)
                want = spec.torque_limit(rot_vel)
                with self.subTest(rot_vel=rot_vel):
                    self.assertLess(abs(torque - want) / want, LIMIT_TOL)
                    self.assertLessEqual(math.hypot(i_d, i_q), gen.CURRENT_LIMIT * 1.001)
                    if rot_vel > gen.constant_torque_speed():
                        self.assertLess(i_d, 0.0, "field weakening must kick in")
                        self.assertGreaterEqual(i_d, gen.ID_FLOOR)
                        self.assertAlmostEqual(
                            gen.voltage_magnitude(rot_vel, i_d, i_q),
                            gen.VOLTAGE_LIMIT,
                            delta=1e-6 * gen.VOLTAGE_LIMIT,
                        )
                    else:
                        self.assertAlmostEqual(i_d, 0.0, delta=1e-12, msg="no MTPA angle to find")
                    # the published law, written out here rather than read back
                    # out of the module
                    published = 1.5 * gen.POLE_PAIRS * (
                        gen.FLUX_LINKAGE * i_q + gen.RELUCTANCE * i_d * i_q
                    )
                    self.assertAlmostEqual(published, torque, delta=1e-12)
        finally:
            holder.cleanup()

    def test_the_output_side_helpers_use_the_gearbox(self) -> None:
        system = presets.build("bldc_pod")
        gen, _, holder = _export(system)
        try:
            ratio, eff = system.gearbox.ratio, system.gearbox.efficiency
            self.assertAlmostEqual(gen.output_speed(100.0), 100.0 * ratio, delta=1e-12)
            self.assertAlmostEqual(
                gen.output_torque(100.0, 0.0, 5.0), gen.torque(0.0, 5.0) * ratio * eff, delta=1e-12
            )
            self.assertAlmostEqual(
                gen.vehicle_speed(100.0), 100.0 * ratio * system.wheel.radius, delta=1e-12
            )
        finally:
            holder.cleanup()

    def test_a_system_without_a_wheel_reports_no_vehicle_speed(self) -> None:
        # _system() builds a bare RotationalLoad, so there is no wheel to convert
        # shaft speed into ground speed
        gen, _, holder = _export(_system())
        try:
            self.assertEqual(gen.WHEEL_RADIUS, 0.0)
            self.assertEqual(gen.vehicle_speed(100.0), 0.0)
        finally:
            holder.cleanup()

    def test_a_bus_that_is_not_a_bus_gives_no_rail_current(self) -> None:
        system = _system(voltage_limit=24.0)  # dc_bus_voltage left at its default inf
        gen, _, holder = _export(system)
        try:
            self.assertTrue(math.isinf(gen.DC_BUS_VOLTAGE))
            self.assertTrue(math.isnan(gen.dc_link_current(100.0, 0.0, 5.0)))
        finally:
            holder.cleanup()


class F32Tests(unittest.TestCase):
    """The f32 build: same numbers, rounded once, and still the same answers."""

    def setUp(self) -> None:
        self.system = presets.build("bldc_pod")
        self.f32, self.paths, self.holder = _export(self.system, dtype="f32")
        self.f64 = _GEN

    def tearDown(self) -> None:
        self.holder.cleanup()

    def test_every_constant_is_the_f32_value_of_the_engine_constant(self) -> None:
        c = round_dq_to_dtype(dq_constants_from_system(self.system), "f32")
        for name, entry in c.as_dict().items():
            key = {
                "flux_temperature_coefficient": "FLUX_TEMP_COEFFICIENT",
                "reference_temperature": "REFERENCE_TEMPERATURE",
                "core_loss_coefficient": "CORE_LOSS_COEFFICIENT",
                "windage_coefficient": "WINDAGE_COEFFICIENT",
                "gear_efficiency": "GEAR_EFFICIENCY",
                "gear_ratio": "GEAR_RATIO",
                "wheel_radius": "WHEEL_RADIUS",
                "motor_count": "MOTOR_COUNT",
                "pole_pairs": "POLE_PAIRS",
            }.get(name, name.upper())
            if not isinstance(entry["value"], float) or not math.isfinite(entry["value"]):
                continue
            with self.subTest(name=name):
                self.assertEqual(getattr(self.f32, key), _f32_round(entry["value"]))

    def test_the_f32_module_reproduces_the_f64_answers_to_f32_precision(self) -> None:
        for rot_vel in SPEEDS:
            limit = self.f64.max_torque(rot_vel)[0]
            for frac in (0.05, 0.5, 1.0):
                torque = frac * limit
                a = self.f32.current_for_torque(torque, rot_vel)
                b = self.f64.current_for_torque(torque, rot_vel)
                with self.subTest(rot_vel=rot_vel, frac=frac):
                    self.assertLess(
                        abs(math.hypot(*a[:2]) - math.hypot(*b[:2])) / math.hypot(*b[:2]), 1e-4
                    )

    def test_the_derived_table_is_derived_from_the_rounded_constants(self) -> None:
        # rounding happens once, on the machine constants; every derived number
        # is then computed from those, so the README, the manifest and the module
        # all agree digit for digit
        self.assertAlmostEqual(
            self.f32.BASE_SPEED,
            self.f32.VOLTAGE_LIMIT / self.f32.BACK_EMF_CONSTANT,
            delta=1e-12 * self.f32.BASE_SPEED,
        )
        self.assertAlmostEqual(
            self.f32.ELECTRICAL_TIME_CONSTANT,
            min(self.f32.INDUCTANCE_D, self.f32.INDUCTANCE_Q) / self.f32.RESISTANCE,
            delta=1e-12 * self.f32.ELECTRICAL_TIME_CONSTANT,
        )
        self.assertLess(
            abs(self.f32.BASE_SPEED - self.f64.BASE_SPEED) / self.f64.BASE_SPEED, 1e-6
        )

    def test_the_manifest_records_the_dtype(self) -> None:
        sibling = self.paths[0].with_name("drive_model_dq_manifest.json")
        manifest = json.loads(sibling.read_text())
        self.assertEqual(manifest["dtype"], "f32")


class ManifestTests(unittest.TestCase):
    """The manifest is what a reviewer on a bench compares a build against."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.system = presets.build("bldc_pod")
        self.paths = export_dq(
            self.system,
            self.dir.name,
            ExportOptions(dtype="f64", sample_time=1e-3, provenance={"note": "from the test suite"}),
        )
        self.manifest = json.loads(
            Path(self.dir.name, "drive_model_dq_manifest.json").read_text()
        )

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_the_schema_and_the_law_are_stated(self) -> None:
        self.assertEqual(self.manifest["schema"], "drive-model-dq-1")
        self.assertIn("v_d", self.manifest["law"])
        self.assertIn("1.5*p*(lambda*i_q", self.manifest["law"]["torque"])
        self.assertIn("w_e = p*rot_vel", self.manifest["law"]["integration"])
        self.assertEqual(self.manifest["units"]["i_d"].split("(")[0].strip(), "A")

    def test_it_lists_every_file_it_wrote_with_a_hash(self) -> None:
        names = {entry["name"] for entry in self.manifest["files"]}
        self.assertIn("drive_model_dq.py", names)
        self.assertIn("bench_drive_model_dq.py", names)
        for entry in self.manifest["files"]:
            path = Path(self.dir.name, entry["name"])
            self.assertTrue(path.exists(), entry["name"])
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            self.assertEqual(entry["sha256"], digest[:16])

    def test_it_records_which_targets_were_requested_and_which_were_emitted(self) -> None:
        # the brushed default target list is c, rust, python and the dq export
        # covers all three now; the bookkeeping stays because a future target can
        # be requested before it is implemented
        targets = self.manifest["targets"]
        self.assertEqual(targets["requested"], ["c", "rust", "python"])
        self.assertEqual(targets["emitted"], ["python", "c", "rust"])
        self.assertEqual(targets["not_emitted"], [])

    def test_the_provenance_and_the_notes_survive(self) -> None:
        self.assertEqual(self.manifest["provenance"], {"note": "from the test suite"})
        self.assertTrue(self.manifest["notes"])
        self.assertTrue(
            any("cross-coupling" in note for note in self.manifest["notes"]),
            self.manifest["notes"],
        )

    def test_the_constants_in_the_manifest_are_the_ones_in_the_module(self) -> None:
        module = _load_module(self.paths[0], "dq_manifest_probe")
        for name, entry in self.manifest["constants"].items():
            key = {
                "flux_temperature_coefficient": "FLUX_TEMP_COEFFICIENT",
                "reference_temperature": "REFERENCE_TEMPERATURE",
            }.get(name, name.upper())
            if isinstance(entry["value"], float) and math.isfinite(entry["value"]):
                with self.subTest(name=name):
                    self.assertEqual(getattr(module, key), entry["value"])

    def test_the_readme_carries_the_machine_caveats(self) -> None:
        readme = Path(self.dir.name, "drive_model_dq_README.md").read_text()
        self.assertIn("Caveats", readme)
        for note in self.manifest["notes"]:
            self.assertIn(note.split(":")[0][:40], readme)
        self.assertIn("bench_drive_model_dq.py", readme)

    def test_the_derived_table_is_in_the_manifest(self) -> None:
        derived = self.manifest["derived"]
        self.assertAlmostEqual(derived["base_speed_rad_s"], _GEN.BASE_SPEED, delta=1e-9)
        self.assertIn("max_step_time_at_base_speed_s", derived)
        self.assertIn("constant_torque_speed_rad_s", derived)
# --------------------------------------------------------------------------
# C and Rust: one probe, three languages
# --------------------------------------------------------------------------
#
# The classes above hold the generated Python against PMSMSpec.  These hold the
# generated C and Rust against that same Python module, on the same grid, with a
# driver that prints its rows in the same order in all three languages.
#
# They are not held to bit-identity, and asking for it would be a mistake.  The
# C build calls libm's hypot while CPython calls its own, and the two disagree in
# the last bit; the Rust module carries its own square root (a Newton iteration
# seeded from the exponent bits) because `no_std` has none, so it is one ulp off
# `math.sqrt` over half the number line.  Measured worst case over the grid
# below: 1.0e-15 relative for C against Python, 8.5e-15 for Rust.  CROSS_TOL is
# 1e-12 -- three orders of clear air above that noise, and still four orders
# tighter than the fixed-count searches' own discretisation, so a mistranslated
# coefficient cannot hide inside it.

CROSS_TOL = 1.0e-12

PROBE_SPEEDS = (0.0, 200.0, 447.774641, 700.0, 1099.714798, 2500.0)
PROBE_FRACS = (0.0, 0.2, 0.9, 1.2)
PROBE_DT = 1.0e-3
PROBE_FF_TARGET = 1649.5721976846453  # 1.5 * base speed: deliberately past the edge
PROBE_FF_STEPS = 121
PROBE_PLANT_STEPS = 3000


def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _probe_rows(module) -> list[tuple[str, list[float]]]:
    """Walk the generated Python module through the probe grid.

    The C and Rust drivers below print the same tags in the same order; this is
    the reference they are compared against.  Keep all three in step -- a tag
    that drifts shows up as a tag-list mismatch, which reads far better than a
    row of numbers that happen to disagree.
    """
    rows: list[tuple[str, list[float]]] = []

    def row(tag: str, *values: float) -> None:
        assert len(values) == 6, tag
        rows.append((tag, [float(v) for v in values]))

    def flag(value: object) -> float:
        return 1.0 if value else 0.0

    row("derived", module.TORQUE_CONSTANT, module.BACK_EMF_CONSTANT,
        module.RELUCTANCE, module.MTPA_DL, module.CHARACTERISTIC_CURRENT,
        module.ID_FLOOR)
    row("derived2", module.BASE_SPEED, module.ELECTRICAL_TIME_CONSTANT,
        module.POLE_DAMPING, module.constant_torque_speed(),
        module.max_step_time(0.0), module.max_step_time(module.BASE_SPEED))

    for w in PROBE_SPEEDS:
        tau, i_d_max, i_q_max = module.max_torque(w)
        row("limits", tau, i_d_max, i_q_max, module.loss_torque(w),
            module.max_step_time(w), module.pole_magnitude(w))
        near = module.mtpa_pair(0.5 * tau)
        over = module.mtpa_pair(1.2 * tau)
        row("mtpa", near[0], near[1], over[0], over[1],
            module.torque_gain(near[0]), module.required_torque(w, 1.0, 0.05))
        for frac in PROBE_FRACS:
            i_d, i_q, sat_i, sat_v = module.current_for_torque(frac * tau, w)
            row("current", i_d, i_q, flag(sat_i), flag(sat_v),
                module.voltage_magnitude(w, i_d, i_q), math.hypot(i_d, i_q))
        pw = module.losses(w, -5.0, 15.0)
        row("losses", pw["copper_W"], pw["iron_W"], pw["windage_W"],
            pw["bearing_W"], pw["converted_W"], pw["input_W"])
        row("misc", module.dc_link_current(w, -5.0, 15.0),
            module.torque(-5.0, 15.0), module.output_torque(w, -5.0, 15.0),
            module.vehicle_speed(w), module.output_speed(w),
            module.effective_flux(None))

    # The step guard: C returns -1 and leaves its output argument alone, Rust
    # returns Err, Python raises.  The twin reproduces "output untouched" by
    # handing back the state it was given, so all three print the same numbers.
    start = module.State(1099.714798, 0.0, 0.0)
    try:
        module.step(start, 0.0, 0.0, PROBE_DT)
        rc_big, untouched = 0.0, module.State(0.0, 0.0, 0.0)
    except ValueError:
        rc_big, untouched = -1.0, start
    row("guard", rc_big, untouched.rot_vel, untouched.i_d, untouched.i_q, 0.0, 0.0)
    ok = module.step(start, 0.0, 0.0, 1.0e-4)
    row("step", 0.0, ok.rot_vel, ok.i_d, ok.i_q,
        module.sub_steps(PROBE_DT, 0.0), module.sub_steps(PROBE_DT, module.BASE_SPEED))

    plant = module.State(0.0, 0.0, 0.0)
    for _ in range(PROBE_PLANT_STEPS):
        plant = module.integrate(plant, 2.4, 0.0, PROBE_DT)
    row("plant", 0.0, plant.rot_vel, plant.i_d, plant.i_q,
        module.torque(plant.i_d, plant.i_q), 0.0)

    sc = module.step_current(module.State(0.0, 0.0, 0.0), -5.0, 15.0, 1.0e-4, 0.05)
    row("sc", sc.rot_vel, sc.i_d, sc.i_q, module.torque(sc.i_d, sc.i_q),
        module.loss_torque(sc.rot_vel), 0.0)

    ff = module.FeedForward(PROBE_DT)
    row("ff_init", 0.0, ff.speed_limit, ff.accel_limit, ff.load_torque,
        ff.sample_time, module.FeedForward.stall_torque())
    v_d = v_q = i_d = i_q = 0.0
    for _ in range(PROBE_FF_STEPS):
        v_d, v_q, i_d, i_q = ff.update(PROBE_FF_TARGET)
    row("ff", ff.w_ref, ff.a_ref, v_d, v_q, math.hypot(v_d, v_q),
        flag(ff.current_saturated) + 2.0 * flag(ff.voltage_saturated))
    row("ff_i", i_d, i_q, module.torque(i_d, i_q),
        module.required_torque(ff.w_ref, ff.a_ref, 0.0),
        module.max_step_time(ff.w_ref), 0.0)

    return rows


def _compare_rows(test: unittest.TestCase, got: list[tuple[str, list[float]]],
                  want: list[tuple[str, list[float]]], label: str) -> None:
    """Compare tagged probe rows at CROSS_TOL, naming the first disagreement."""
    test.assertEqual([tag for tag, _ in got], [tag for tag, _ in want],
                     f"{label}: the driver walked a different grid")
    test.assertEqual(len(got), len(want), f"{label}: row count")
    worst = 0.0
    first: tuple[object, ...] | None = None
    for (tag, a), (_, b) in zip(got, want):
        for column, (x, y) in enumerate(zip(a, b)):
            if math.isnan(x) and math.isnan(y):
                continue
            if x == y:
                continue
            error = abs(x - y) / max(abs(x), abs(y), 1.0)
            if error > worst:
                worst, first = error, (tag, column, x, y)
            test.assertLessEqual(error, CROSS_TOL,
                                 f"{label}: {tag} column {column}: {x!r} against {y!r}")
    test.assertLessEqual(worst, CROSS_TOL, f"{label}: worst {worst} at {first}")


def _compare_csv(test: unittest.TestCase, got: str, want: str, label: str) -> None:
    """Compare two bench CSVs header-for-header and number-for-number."""
    got_rows = [line.split(",") for line in got.strip().splitlines()]
    want_rows = [line.split(",") for line in want.strip().splitlines()]
    test.assertEqual(got_rows[0], want_rows[0], f"{label}: CSV header")
    test.assertEqual(len(got_rows), len(want_rows), f"{label}: row count")
    for a, b in zip(got_rows[1:], want_rows[1:]):
        for column, (x, y) in enumerate(zip(a, b)):
            xf, yf = float(x), float(y)
            error = abs(xf - yf) / max(abs(xf), abs(yf), 1.0)
            test.assertLessEqual(
                error, CROSS_TOL,
                f"{label}: row {a[0]} column {column}: {x!r} against {y!r}")


_PROBE_C = r"""/* The same probe as _probe_rows() above, in C.  Keep the tags, the order and
   the grid in step with the Python and Rust twins. */
#include <stdio.h>

#include "drive_model_dq.h"

static const dq_scalar speeds[6] = {0.0, 200.0, 447.774641, 700.0, 1099.714798,
                                    2500.0};
static const dq_scalar fracs[4] = {0.0, 0.2, 0.9, 1.2};

static void row(const char *tag, dq_scalar a, dq_scalar b, dq_scalar c,
                dq_scalar d, dq_scalar e, dq_scalar f) {
    printf("%s %.17g %.17g %.17g %.17g %.17g %.17g\n", tag, a, b, c, d, e, f);
}

int main(void) {
    const dq_model *m = &dq_model_default;
    dq_limits lim;
    dq_pair near, over, v;
    dq_current ci;
    dq_power pw;
    dq_state s, out, next, plant, sc;
    dq_feedforward ff;
    int i, rc;

    row("derived", dq_torque_constant(m), dq_back_emf_constant(m),
        dq_reluctance(m), dq_mtpa_dl(m), dq_characteristic_current(m),
        dq_id_floor(m));
    row("derived2", dq_base_speed(m), dq_electrical_time_constant(m),
        dq_pole_damping(m), dq_constant_torque_speed(m), dq_max_step_time(m, 0.0),
        dq_max_step_time(m, dq_base_speed(m)));

    for (i = 0; i < 6; ++i) {
        const dq_scalar w = speeds[i];
        lim = dq_max_torque(m, w);
        row("limits", lim.torque, lim.i_d, lim.i_q, dq_loss_torque(m, w),
            dq_max_step_time(m, w), dq_pole_magnitude(m, w));
        near = dq_mtpa_pair(m, 0.5 * lim.torque);
        over = dq_mtpa_pair(m, 1.2 * lim.torque);
        row("mtpa", near.d, near.q, over.d, over.q, dq_torque_gain(m, near.d),
            dq_required_torque(m, w, 1.0, 0.05));
        for (int j = 0; j < 4; ++j) {
            ci = dq_current_for_torque(m, fracs[j] * lim.torque, w);
            row("current", ci.i_d, ci.i_q, (dq_scalar) ci.current_saturated,
                (dq_scalar) ci.voltage_saturated,
                dq_voltage_magnitude(m, w, ci.i_d, ci.i_q),
                dq_hypot(ci.i_d, ci.i_q));
        }
        dq_losses(m, w, -5.0, 15.0, &pw);
        row("losses", pw.copper_W, pw.iron_W, pw.windage_W, pw.bearing_W,
            pw.converted_W, pw.input_W);
        row("misc", dq_dc_link_current(m, w, -5.0, 15.0),
            dq_torque(m, -5.0, 15.0, DQ_NAN),
            dq_output_torque(m, w, -5.0, 15.0), dq_vehicle_speed(m, w),
            dq_output_speed(m, w), dq_effective_flux(m, DQ_NAN));
    }

    /* the guard: a rejected step leaves *out exactly as it was found */
    s.rot_vel = 1099.714798;
    s.i_d = 0.0;
    s.i_q = 0.0;
    out = s;
    rc = dq_step(m, s, 0.0, 0.0, 0.001, 0.0, &out);
    row("guard", (dq_scalar) rc, out.rot_vel, out.i_d, out.i_q, 0.0, 0.0);
    next = s;
    rc = dq_step(m, s, 0.0, 0.0, 0.0001, 0.0, &next);
    row("step", (dq_scalar) rc, next.rot_vel, next.i_d, next.i_q,
        (dq_scalar) dq_sub_steps(m, 0.001, 0.0),
        (dq_scalar) dq_sub_steps(m, 0.001, dq_base_speed(m)));

    plant.rot_vel = 0.0;
    plant.i_d = 0.0;
    plant.i_q = 0.0;
    rc = 0;
    for (i = 0; i < 3000; ++i) {
        rc = dq_integrate(m, plant, 2.4, 0.0, 0.001, 0.0, &plant);
        if (rc != 0) {
            break;
        }
    }
    row("plant", (dq_scalar) rc, plant.rot_vel, plant.i_d, plant.i_q,
        dq_torque(m, plant.i_d, plant.i_q, DQ_NAN), 0.0);

    sc.rot_vel = 0.0;
    sc.i_d = 0.0;
    sc.i_q = 0.0;
    sc = dq_step_current(m, sc, -5.0, 15.0, 0.0001, 0.05);
    row("sc", sc.rot_vel, sc.i_d, sc.i_q, dq_torque(m, sc.i_d, sc.i_q, DQ_NAN),
        dq_loss_torque(m, sc.rot_vel), 0.0);

    rc = dq_ff_init(&ff, m, 0.001, DQ_NAN, DQ_NAN, DQ_NAN);
    row("ff_init", (dq_scalar) rc, ff.speed_limit, ff.accel_limit, ff.load_torque,
        ff.sample_time, dq_ff_stall_torque(m));
    for (i = 0; i < 121; ++i) {
        dq_ff_update(&ff, 1649.5721976846453, &v, &ci);
    }
    row("ff", ff.w_ref, ff.a_ref, v.d, v.q, dq_hypot(v.d, v.q),
        (dq_scalar) (ff.current_saturated + 2 * ff.voltage_saturated));
    row("ff_i", ci.i_d, ci.i_q, dq_torque(m, ci.i_d, ci.i_q, DQ_NAN),
        dq_required_torque(m, ff.w_ref, ff.a_ref, 0.0),
        dq_max_step_time(m, ff.w_ref), 0.0);
    return 0;
}
"""


_PROBE_RUST = r"""// The same probe as _probe_rows() above, in Rust.  Keep the tags, the order and
// the grid in step with the Python and C twins.

#[path = "drive_model_dq.rs"]
mod drive_model_dq;

use drive_model_dq::{dq_hypot, DqFeedForward, DqState, Scalar, DQ_MODEL};

const SPEEDS: [Scalar; 6] = [0.0, 200.0, 447.774641, 700.0, 1099.714798, 2500.0];
const FRACS: [Scalar; 4] = [0.0, 0.2, 0.9, 1.2];
const DT: Scalar = 0.001;
const FF_TARGET: Scalar = 1649.5721976846453;

fn row(tag: &str, v: [Scalar; 6]) {
    println!("{} {} {} {} {} {} {}", tag, v[0], v[1], v[2], v[3], v[4], v[5]);
}

fn main() {
    let m = DQ_MODEL;

    row("derived", [
        m.torque_constant(),
        m.back_emf_constant(),
        m.reluctance(),
        m.mtpa_dl(),
        m.characteristic_current(),
        m.id_floor(),
    ]);
    row("derived2", [
        m.base_speed(),
        m.electrical_time_constant(),
        m.pole_damping(),
        m.constant_torque_speed(),
        m.max_step_time(0.0),
        m.max_step_time(m.base_speed()),
    ]);

    for i in 0..6 {
        let w = SPEEDS[i];
        let lim = m.max_torque(w);
        row("limits", [
            lim.torque,
            lim.i_d,
            lim.i_q,
            m.loss_torque(w),
            m.max_step_time(w),
            m.pole_magnitude(w),
        ]);
        let near = m.mtpa_pair(0.5 * lim.torque);
        let over = m.mtpa_pair(1.2 * lim.torque);
        row("mtpa", [
            near.d,
            near.q,
            over.d,
            over.q,
            m.torque_gain(near.d),
            m.required_torque(w, 1.0, 0.05),
        ]);
        for j in 0..4 {
            let c = m.current_for_torque(FRACS[j] * lim.torque, w);
            row("current", [
                c.i_d,
                c.i_q,
                if c.current_saturated { 1.0 } else { 0.0 },
                if c.voltage_saturated { 1.0 } else { 0.0 },
                m.voltage_magnitude(w, c.i_d, c.i_q, 0.0, 0.0),
                dq_hypot(c.i_d, c.i_q),
            ]);
        }
        let p = m.losses(w, -5.0, 15.0);
        row("losses", [
            p.copper_w,
            p.iron_w,
            p.windage_w,
            p.bearing_w,
            p.converted_w,
            p.input_w,
        ]);
        row("misc", [
            m.dc_link_current(w, -5.0, 15.0),
            m.torque(-5.0, 15.0, None),
            m.output_torque(w, -5.0, 15.0),
            m.vehicle_speed(w),
            m.output_speed(w),
            m.effective_flux(None),
        ]);
    }

    // the guard: a rejected step hands back no state at all, so the twin keeps
    // the one it started from -- the same six numbers C prints.
    let s = DqState { rot_vel: 1099.714798, i_d: 0.0, i_q: 0.0 };
    let mut untouched = s;
    let mut rc = 0.0;
    if let Err(_) = m.step(s, 0.0, 0.0, DT, 0.0) {
        rc = -1.0;
    } else {
        untouched = DqState { rot_vel: 0.0, i_d: 0.0, i_q: 0.0 };
    }
    row("guard", [rc, untouched.rot_vel, untouched.i_d, untouched.i_q, 0.0, 0.0]);
    let ok = m.step(s, 0.0, 0.0, 0.0001, 0.0).expect("0.0001 s is inside the guard");
    row("step", [
        0.0,
        ok.rot_vel,
        ok.i_d,
        ok.i_q,
        m.sub_steps(DT, 0.0) as Scalar,
        m.sub_steps(DT, m.base_speed()) as Scalar,
    ]);

    let mut plant = DqState { rot_vel: 0.0, i_d: 0.0, i_q: 0.0 };
    let mut rc3 = 0.0;
    for _ in 0..3000 {
        match m.integrate(plant, 2.4, 0.0, DT, 0.0) {
            Ok(next) => plant = next,
            Err(_) => {
                rc3 = -1.0;
                break;
            }
        }
    }
    row("plant", [
        rc3,
        plant.rot_vel,
        plant.i_d,
        plant.i_q,
        m.torque(plant.i_d, plant.i_q, None),
        0.0,
    ]);

    let sc = m.step_current(DqState { rot_vel: 0.0, i_d: 0.0, i_q: 0.0 }, -5.0, 15.0, 0.0001, 0.05);
    row("sc", [
        sc.rot_vel,
        sc.i_d,
        sc.i_q,
        m.torque(sc.i_d, sc.i_q, None),
        m.loss_torque(sc.rot_vel),
        0.0,
    ]);

    let mut ff = DqFeedForward::new(m, DT).expect("0.001 s is a real control period");
    row("ff_init", [
        0.0,
        ff.speed_limit,
        ff.accel_limit,
        ff.load_torque,
        ff.sample_time,
        m.stall_torque(),
    ]);
    let mut v = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..121 {
        v = ff.update(FF_TARGET);
    }
    let flags = if ff.current_saturated { 1.0 } else { 0.0 }
        + 2.0 * (if ff.voltage_saturated { 1.0 } else { 0.0 });
    row("ff", [ff.w_ref, ff.a_ref, v.0, v.1, dq_hypot(v.0, v.1), flags]);
    row("ff_i", [
        v.2,
        v.3,
        m.torque(v.2, v.3, None),
        m.required_torque(ff.w_ref, ff.a_ref, 0.0),
        m.max_step_time(ff.w_ref),
        0.0,
    ]);
}
"""


@unittest.skipUnless(_which("cc", "gcc", "clang"), "no C compiler on this box")
class CDqExportTests(unittest.TestCase):
    """The generated C, run against the generated Python on the same grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cc = _which("cc", "gcc", "clang")
        cls.dir = tempfile.mkdtemp(prefix="dq-export-c-")
        # python and c from one export, so the two benches share a control
        # period and can be compared row for row
        export_dq(presets.build("bldc_pod"), cls.dir,
                  ExportOptions(dtype="f64", targets=("python", "c"),
                                sample_time=PROBE_DT))
        (Path(cls.dir) / "probe.c").write_text(_PROBE_C, encoding="utf-8")
        (Path(cls.dir) / "header_only.c").write_text(
            '#include "drive_model_dq.h"\nint main(void) { return 0; }\n',
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _build(self, binary: str, *sources: str) -> str:
        done = subprocess.run(
            [self.cc, "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion",
             "-Werror", "-O1", *sources, "-lm", "-o", binary],
            cwd=self.dir, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0,
                         f"C build failed:\n{done.stderr}")
        return done.stderr

    def _run(self, binary: str, *args: str) -> str:
        done = subprocess.run([str(Path(self.dir) / binary), *args], cwd=self.dir,
                              check=True, capture_output=True, text=True)
        return done.stdout

    @staticmethod
    def _rows(stdout: str) -> list[tuple[str, list[float]]]:
        rows = []
        for line in stdout.strip().splitlines():
            parts = line.split()
            rows.append((parts[0], [float(value) for value in parts[1:]]))
        return rows

    def test_the_generated_c_compiles_without_warnings(self) -> None:
        self._build("header_only", "-c", "header_only.c")
        self._build("probe", "probe.c", "drive_model_dq.c")
        self._build("bench_c", "bench_drive_model_dq.c", "drive_model_dq.c")

    def test_the_c_answers_match_the_generated_python(self) -> None:
        self._build("probe", "probe.c", "drive_model_dq.c")
        _compare_rows(self, self._rows(self._run("probe")), _probe_rows(_GEN), "C")

    def test_the_c_bench_matches_the_python_bench(self) -> None:
        self._build("bench_c", "bench_drive_model_dq.c", "drive_model_dq.c")
        for mode in ("plant", "feedforward"):
            c_out = self._run("./bench_c", *([mode] if mode != "plant" else []))
            py_out = subprocess.run(
                [sys.executable, "bench_drive_model_dq.py",
                 *([mode] if mode != "plant" else [])],
                cwd=self.dir, check=True, capture_output=True, text=True).stdout
            _compare_csv(self, c_out, py_out, f"C bench {mode}")

    def test_an_f32_build_runs_and_lands_in_the_right_place(self) -> None:
        # f32 is the width a microcontroller actually uses.  It must still
        # compile clean under -Wconversion, which is where a bare double
        # literal in the template shows up.
        out = tempfile.mkdtemp(prefix="dq-export-c32-")
        try:
            export_dq(presets.build("bldc_pod"), out,
                      ExportOptions(dtype="f32", targets=("c",),
                                    sample_time=PROBE_DT))
            done = subprocess.run(
                [self.cc, "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                 "-Wconversion", "-Werror", "-O1", "bench_drive_model_dq.c",
                 "drive_model_dq.c", "-lm", "-o", "bench32"],
                cwd=out, check=True, capture_output=True, text=True)
            self.assertEqual(done.stderr, "")
            rows = subprocess.run([str(Path(out) / "bench32")], cwd=out, check=True,
                                  capture_output=True, text=True).stdout
            last = rows.strip().splitlines()[-1].split(",")
            # the f64 plant run settles at 184.316855 rad/s; f32 lands at
            # 184.316879, which is where 24 bits of mantissa put it
            self.assertAlmostEqual(float(last[1]), 184.316879, delta=1.0e-3)
        finally:
            shutil.rmtree(out, ignore_errors=True)


@unittest.skipUnless(_which("rustc"), "no rustc on this box")
class RustDqExportTests(unittest.TestCase):
    """The generated Rust, same grid, plus the two claims its banner makes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rustc = _which("rustc")
        cls.dir = tempfile.mkdtemp(prefix="dq-export-rs-")
        export_dq(presets.build("bldc_pod"), cls.dir,
                  ExportOptions(dtype="f64", targets=("python", "rust"),
                                sample_time=PROBE_DT))
        (Path(cls.dir) / "probe.rs").write_text(_PROBE_RUST, encoding="utf-8")
        # the banner promises a module that drops into a #![no_std] crate; this
        # is the test of the promise, not of the prose.  pub mod, because a
        # private module makes every item in it look like dead code.
        (Path(cls.dir) / "nostd.rs").write_text(
            '#![no_std]\n#[path = "drive_model_dq.rs"]\npub mod drive_model_dq;\n',
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _build(self, source: str, output: str, *extra: str) -> str:
        done = subprocess.run(
            [self.rustc, "--edition", "2021", "-O", *extra, source, "-o", output],
            cwd=self.dir, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, f"rustc failed:\n{done.stderr}")
        return done.stderr

    def _run(self, binary: str, *args: str) -> str:
        done = subprocess.run([str(Path(self.dir) / binary), *args], cwd=self.dir,
                              check=True, capture_output=True, text=True)
        return done.stdout

    @staticmethod
    def _rows(stdout: str) -> list[tuple[str, list[float]]]:
        rows = []
        for line in stdout.strip().splitlines():
            parts = line.split()
            rows.append((parts[0], [float(value) for value in parts[1:]]))
        return rows

    def test_the_generated_rust_compiles_without_warnings(self) -> None:
        # a lib crate exports its pub items, so there is no dead code to hide:
        # an empty stderr here is the whole point of the test
        done = subprocess.run(
            [self.rustc, "--crate-type", "lib", "--edition", "2021", "-O",
             "drive_model_dq.rs", "-o", "libdq.rlib"],
            cwd=self.dir, check=True, capture_output=True, text=True)
        self.assertEqual(done.stderr, "")

    def test_the_module_actually_builds_in_a_no_std_crate(self) -> None:
        done = subprocess.run(
            [self.rustc, "--crate-type", "lib", "--edition", "2021", "-O",
             "nostd.rs", "-o", "nostd.rlib"],
            cwd=self.dir, check=True, capture_output=True, text=True)
        self.assertEqual(done.stderr, "")

    def test_the_rust_answers_match_the_generated_python(self) -> None:
        # -A dead_code because a binary crate sees every pub item in the module
        # it pulls in, and the probe only walks part of the surface
        self._build("probe.rs", "probe", "-A", "dead_code")
        _compare_rows(self, self._rows(self._run("probe")), _probe_rows(_GEN), "Rust")

    def test_the_rust_bench_matches_the_python_bench(self) -> None:
        self._build("bench_drive_model_dq.rs", "bench_rs", "-A", "dead_code")
        for mode in ("plant", "feedforward"):
            rs_out = self._run("./bench_rs", *([mode] if mode != "plant" else []))
            py_out = subprocess.run(
                [sys.executable, "bench_drive_model_dq.py",
                 *([mode] if mode != "plant" else [])],
                cwd=self.dir, check=True, capture_output=True, text=True).stdout
            _compare_csv(self, rs_out, py_out, f"Rust bench {mode}")

    def test_the_step_guard_is_a_result_not_a_panic(self) -> None:
        # the stable step shrinks with speed, so the Rust API returns Err where
        # Python raises and C returns -1; a caller on a microcontroller has to
        # be able to see that without unwinding
        driver = r"""
#[path = "drive_model_dq.rs"]
mod drive_model_dq;

use drive_model_dq::{DqState, Scalar, DQ_MODEL};

fn main() {
    let m = DQ_MODEL;
    let s = DqState { rot_vel: 1099.714798, i_d: 0.0, i_q: 0.0 };
    match m.step(s, 0.0, 0.0, 0.001, 0.0) {
        Ok(_) => println!("accepted"),
        Err(e) => println!("{}", e),
    }
    let _ = m.max_step_time(0.0);
}
"""
        (Path(self.dir) / "guard.rs").write_text(driver, encoding="utf-8")
        self._build("guard.rs", "guard", "-A", "dead_code")
        out = self._run("guard").strip()
        self.assertNotEqual(out, "accepted")
        self.assertIn("exceeds the stable step", out)
        self.assertIn("subdivide the step", out)
        # the same numbers the Python ValueError carries, in the same order
        self.assertIn("1099.714798 rad/s", out)




# ----------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------
class CliDqExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dt-cli-dq-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _quiet(self, argv):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_dq_writes_the_dq_files_and_not_the_brushed_ones(self) -> None:
        code, text = self._quiet(["export", "--preset", "bldc_pod", "--dq",
                                  "--dtype", "f64", "--targets", "python",
                                  "--out", self.dir])
        self.assertEqual(code, 0)
        names = {p.name for p in Path(self.dir).iterdir()}
        self.assertEqual(names, {"drive_model_dq.py", "bench_drive_model_dq.py",
                                 "drive_model_dq_manifest.json",
                                 "drive_model_dq_README.md"})
        # the summary has to be the dq one: the brushed max_step_time is half the
        # electrical time constant, which is the wrong law for this plant
        self.assertIn("shrinks as the shaft speeds up", text)
        self.assertIn("constant-torque band edge", text)
        self.assertIn("caveats for this machine", text)

    def test_dq_json_prints_the_manifest(self) -> None:
        code, text = self._quiet(["export", "--preset", "bldc_pod", "--dq",
                                  "--dtype", "f64", "--targets", "python",
                                  "--json", "--out", self.dir])
        self.assertEqual(code, 0)
        manifest = json.loads(text)
        self.assertEqual(manifest["schema"], "drive-model-dq-1")
        self.assertEqual(manifest["targets"]["emitted"], ["python"])
        self.assertTrue(manifest["notes"])

    def test_a_sample_time_above_the_stable_step_warns(self) -> None:
        # 1 ms is the period the brushed export is happy with; here it is twice
        # the standstill limit and five times the limit at base speed
        _, text = self._quiet(["export", "--preset", "bldc_pod", "--dq",
                               "--dtype", "f64", "--targets", "python",
                               "--sample-time", "0.001", "--out", self.dir])
        self.assertIn("WARNING", text)
        self.assertIn("0.001", text)

    def test_dq_on_a_brushed_preset_exits_two_without_writing(self) -> None:
        out = Path(self.dir) / "nope"
        code, _ = self._quiet(["export", "--preset", "legacy", "--dq",
                               "--out", str(out)])
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_the_default_targets_emit_all_three_languages(self) -> None:
        # the brushed default target list is c,rust,python; --dq now renders all
        # three, so nothing is skipped and no apology note is attached
        code, _ = self._quiet(["export", "--preset", "bldc_pod", "--dq",
                               "--dtype", "f64", "--out", self.dir])
        self.assertEqual(code, 0)
        for name in ("drive_model_dq.py", "drive_model_dq.h", "drive_model_dq.c",
                     "drive_model_dq.rs", "bench_drive_model_dq.py",
                     "bench_drive_model_dq.c", "bench_drive_model_dq.rs"):
            self.assertTrue((Path(self.dir) / name).exists(), name)
        manifest = json.loads(
            (Path(self.dir) / "drive_model_dq_manifest.json").read_text())
        self.assertEqual(manifest["targets"]["requested"], ["c", "rust", "python"])
        self.assertEqual(manifest["targets"]["emitted"], ["python", "c", "rust"])
        self.assertEqual(manifest["targets"]["not_emitted"], [])
        self.assertFalse(any("emits Python only" in note
                             for note in manifest["notes"]))
