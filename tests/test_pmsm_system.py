"""Tests for a brushless machine used *through* the rest of the toolkit.

:mod:`tests.test_pmsm` tests the dq machine on its own.  These tests cover the
seams that make it usable in a drivetrain: the ``PMSMModel`` view that presents
a :class:`PMSMSpec` through the brushed interface :class:`MotorGroup` expects,
the ``kind`` discriminator that lets a brushless machine survive a config
round-trip, the preset catalogue and parameter plumbing, the fitter's brushed
vocabulary landing on the constants a brushless system can actually set, and the
export path.

Two numbers matter most here.

* ``kt/kv == 1.5``.  That is the amplitude-invariant Park convention
  (``kt = 1.5*p*lambda``, ``kv = p*lambda``), not an inconsistency, and the
  fitter's consistency check has to be told so.
* :meth:`PMSMModel.collapse_margin`.  The brushed view charges itself only for
  the q-axis voltage, so it believes full current is available roughly twice as
  far up the speed range as a dq drive can actually hold it - 447.8 rad/s
  rather than 909.2 rad/s for the catalogue's ``bldc_24v``.  That is documented
  behaviour with an accessor and a note in the exported README, not a surprise
  waiting in the field.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from drivetrain.exceptions import ConfigError
from drivetrain.export import ExportOptions, ModelConstants, export
from drivetrain.motor import MotorGroup, MotorModel
from drivetrain.pmsm import PMSMModel, PMSMSpec
from drivetrain import fit, presets, simulate, sweep


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AdapterViewTests(unittest.TestCase):
    """The brushed-shaped view onto a dq machine."""

    def setUp(self):
        self.spec = PMSMSpec()
        self.m = PMSMModel(self.spec)

    def test_the_constants_are_derived_from_the_spec(self):
        s = self.spec
        self.assertAlmostEqual(self.m.kt, 1.5 * s.pole_pairs * s.flux_linkage)
        self.assertAlmostEqual(self.m.kv, s.pole_pairs * s.flux_linkage)
        self.assertAlmostEqual(self.m.kt / self.m.kv, 1.5)
        self.assertEqual(self.m.resistance, s.resistance)
        # the axis that carries the torque current sets the electrical dynamics
        self.assertEqual(self.m.inductance, s.inductance_q)
        self.assertEqual(self.m.max_current, s.current_limit)
        self.assertEqual(self.m.max_voltage, s.voltage_limit)
        self.assertEqual(self.m.rotor_inertia, s.rotor_inertia)
        self.assertEqual(self.m.bearing_damping, s.bearing_damping)
        self.assertEqual(self.m.static_friction, s.static_friction)
        # mechanical loss lives in bearing_damping, so there is nothing extra
        # for a fit to chase here
        self.assertEqual(self.m.no_load_friction_current, 0.0)

    def test_the_view_follows_the_spec_it_wraps(self):
        # kt and kv have no setter on purpose: they cannot drift away from the
        # machine they describe
        self.m.spec = replace(self.spec, flux_linkage=2.0e-3)
        self.assertAlmostEqual(self.m.kt, 1.5 * 7 * 2.0e-3)
        self.assertAlmostEqual(self.m.kv, 7 * 2.0e-3)

    def test_it_reproduces_the_brushed_characteristic_formulas(self):
        m, s = self.m, self.spec
        for v, w in ((12.0, 0.0), (12.0, 300.0), (6.0, 900.0)):
            self.assertAlmostEqual(m.torque_at(v, w), m.kt * m.current_at(v, w))
        self.assertAlmostEqual(m.stall_torque_per_volt, m.kt / m.resistance)
        self.assertAlmostEqual(m.stall_current_per_volt, 1.0 / m.resistance)
        self.assertAlmostEqual(m.damping_coefficient,
                               m.bearing_damping + m.kt * m.kv / m.resistance)
        self.assertAlmostEqual(m.mech_time_constant,
                               m.rotor_inertia * m.resistance / (m.kt * m.kv))
        self.assertAlmostEqual(m.electrical_time_constant, s.inductance_q / s.resistance)
        m.rot_vel = m.max_voltage / m.kv          # base speed
        self.assertAlmostEqual(m.ideal_no_load_rpm, m.rpm)
        self.assertAlmostEqual(m.back_emf, m.kv * m.rot_vel)

    def test_copper_loss_carries_the_three_phase_factor(self):
        # a brushed armature dissipates i^2*R; three windings carrying a phase
        # current amplitude i dissipate 1.5*i^2*R in this convention
        i = 10.0
        self.assertAlmostEqual(self.m.thermal_power(i), 1.5 * i * i * self.spec.resistance)
        self.assertAlmostEqual(self.m.thermal_power(i),
                               self.spec.copper_power(i_d=0.0, i_q=i))
        self.assertAlmostEqual(self.m.thermal_power(), self.m.thermal_power(self.m.current))

    def test_dc_link_current_is_the_supply_current_not_the_phase_current(self):
        bus = 24.0
        m = PMSMModel(replace(self.spec, dc_bus_voltage=bus))
        for i, w in ((12.0, 300.0), (20.0, 0.0), (5.0, 900.0)):
            v_d, v_q = m.spec.voltage_steady(0.0, i, w)
            # exact power balance: the bus feeds 1.5*(v . i) at i_d = 0
            self.assertAlmostEqual(m.dc_link_current(i, w),
                                   1.5 * (v_d * 0.0 + v_q * i) / bus)
            self.assertLess(m.dc_link_current(i, w), i,
                            "the supply current must be below the phase current "
                            "any time back-EMF carries most of the voltage")
        self.assertEqual(m.dc_link_current(0.0, 300.0), 0.0)

    def test_dc_link_current_is_unknown_without_a_bus(self):
        self.assertTrue(math.isnan(PMSMModel(replace(self.spec)).dc_link_current(10.0, 0.0)))

    def test_it_refuses_a_foreign_spec(self):
        with self.assertRaises(ConfigError):
            PMSMModel(spec="not a machine")  # type: ignore[arg-type]
        with self.assertRaises(ConfigError):
            PMSMModel(replace(self.spec, resistance=0.0))

    def test_copy_is_independent(self):
        twin = self.m.copy()
        twin.spec = replace(twin.spec, flux_linkage=3.0e-3)
        self.assertEqual(self.spec.flux_linkage, PMSMSpec().flux_linkage)
        self.assertAlmostEqual(twin.kt, 1.5 * 7 * 3.0e-3)


class BandEdgeTests(unittest.TestCase):
    """Where the ``i_d = 0`` collapse stops being exact, and by how much."""

    def setUp(self):
        self.m = presets.MOTORS["bldc_24v"].copy()
        self.s = self.m.spec

    def test_current_ceiling_solves_the_voltage_ellipse(self):
        # an independent quadratic: |(R*I + w_e*lambda, w_e*Lq*I)| = V
        s = self.s
        for w in (0.0, 200.0, 600.0, 1000.0):
            we = s.electrical_speed(w)
            a = (s.resistance ** 2 + (we * s.inductance_q) ** 2)
            b = 2.0 * s.resistance * we * s.flux_linkage
            c = (we * s.flux_linkage) ** 2 - s.voltage_limit ** 2
            disc = b * b - 4.0 * a * c
            want = 0.0 if disc <= 0.0 else (-b + math.sqrt(disc)) / (2.0 * a)
            self.assertAlmostEqual(self.m.current_ceiling(w),
                                   min(s.current_limit, max(0.0, want)), places=9)

    def test_ceiling_and_voltage_agree_at_the_rail(self):
        s = self.s
        edge = self.m.constant_torque_speed()
        self.assertAlmostEqual(self.m.voltage_needed(self.m.current_ceiling(edge), edge),
                               s.voltage_limit, places=9)
        for w in (edge, 1.5 * edge, 2.0 * edge, self.m.base_speed()):
            self.assertAlmostEqual(self.m.voltage_needed(self.m.current_ceiling(w), w),
                                   s.voltage_limit, places=9)
        # below the edge the current limit binds first, so there is voltage to spare
        self.assertLess(self.m.voltage_needed(self.m.current_ceiling(0.0), 0.0),
                        s.voltage_limit)

    def test_the_band_edge_is_the_positive_root_of_the_ellipse(self):
        s = self.s
        i = s.current_limit
        lam = s.flux_linkage
        a = (s.inductance_q * i) ** 2 + lam ** 2
        b = 2.0 * lam * s.resistance * i
        c = (s.resistance * i) ** 2 - s.voltage_limit ** 2
        want = (math.sqrt(b * b - 4.0 * a * c) - b) / (2.0 * a) / s.pole_pairs
        self.assertAlmostEqual(self.m.constant_torque_speed(), want, places=9)
        self.assertLess(self.m.constant_torque_speed(), self.m.base_speed())

    def test_the_collapse_is_optimistic_by_the_decoupling_voltage(self):
        # the collapse charges itself for (V - R*I)/kv; the dq drive can only
        # reach 447.8 rad/s at full current on this machine
        self.assertAlmostEqual(self.m.view_constant_torque_speed(),
                               (self.m.max_voltage - self.m.resistance * self.m.max_current)
                               / self.m.kv, places=9)
        self.assertAlmostEqual(self.m.collapse_margin(), 0.5075279067887125, places=6)
        self.assertGreater(self.m.collapse_margin(), 0.5)

    def test_the_margin_tracks_lq_times_current_over_flux(self):
        # saliency is irrelevant; Lq*I relative to lambda is what sets it
        small = PMSMModel(replace(self.s, inductance_d=1.0e-6, inductance_q=1.0e-6))
        self.assertLess(small.collapse_margin(), 1e-3)
        self.assertAlmostEqual(small.constant_torque_speed(),
                               small.view_constant_torque_speed(),
                               delta=1e-3 * small.view_constant_torque_speed())
        big = PMSMModel(replace(self.s, inductance_q=3.6e-4, inductance_d=3.6e-4))
        self.assertGreater(big.collapse_margin(), self.m.collapse_margin())

    def test_base_speed_is_where_the_back_emf_reaches_the_rail(self):
        s = self.s
        self.assertAlmostEqual(self.m.base_speed(),
                               s.voltage_limit / (s.pole_pairs * s.flux_linkage))
        self.assertFalse(self.m.above_base_speed(0.0))
        self.assertTrue(self.m.above_base_speed(1.01 * self.m.base_speed()))
        # the collapse's own top speed is exactly base speed, which is why it
        # cannot invent a field-weakening region
        self.assertAlmostEqual(self.m.max_voltage / self.m.kv, self.m.base_speed())
        self.assertEqual(self.m.current_ceiling(self.m.base_speed()), 0.0)

    def test_the_note_states_the_band_it_cannot_reach(self):
        note = self.m.equivalent_note()
        self.assertIn("kt/kv=1.5", note)
        self.assertIn("447.8 rad/s", note)
        self.assertIn("909.2 rad/s", note)
        self.assertIn("no field weakening", note)


class GroupCollapseTests(unittest.TestCase):
    """A brushless machine inside ``MotorGroup`` and the simulator."""

    def test_two_machines_collapse_the_way_the_group_says(self):
        group = MotorGroup(presets.MOTORS["bldc_24v"].copy(), 2)
        self.assertAlmostEqual(group.kt_eff, 0.0189)
        self.assertAlmostEqual(group.kv_eff, 0.0126)
        self.assertAlmostEqual(group.r_eff, 0.06)          # R / count
        self.assertAlmostEqual(group.l_eff, 9.0e-05)       # Lq / count
        self.assertAlmostEqual(group.inertia_eff, 4.0e-05)  # count * J
        self.assertAlmostEqual(group.bearing_damping_eff, 2.0e-06)
        self.assertEqual(group.motor.max_current * group.count, 40.0)

    def test_the_group_still_reports_its_state(self):
        group = MotorGroup(presets.MOTORS["bldc_24v"].copy(), 2)
        group.motor.rot_vel = 120.0
        self.assertEqual(group.rot_vel, 120.0)
        group.reset()
        self.assertEqual(group.rot_vel, 0.0)
        self.assertEqual(group.motor.current, 0.0)

    def test_a_brushless_pod_simulates_and_settles_below_base_speed(self):
        from drivetrain import simulate
        system = presets.build("bldc_pod")
        # the supply this pipeline takes is the phase peak, not the battery: the
        # preset's 24 V bus becomes 24/sqrt(3) = 13.86 V of usable phase voltage
        from drivetrain.pmsm import phase_voltage_limit
        self.assertAlmostEqual(system.voltage_limit,
                               phase_voltage_limit(system.group.motor.spec.dc_bus_voltage),
                               places=12)
        self.assertLess(system.voltage_limit, system.group.motor.spec.dc_bus_voltage)

        base = system.group.motor.base_speed()
        result = simulate.simulate(system, voltage=system.voltage_limit,
                                   duration=20.0, engine="rk4", samples=400)
        final = result.channels()["motor_rpm"][-1] * 2.0 * math.pi / 60.0
        self.assertLess(final, base, "the collapse cannot exceed base speed")
        # the simulator and the exported steady-state formula agree exactly
        steady = ModelConstants.from_system(system).steady_speed(system.voltage_limit)
        self.assertAlmostEqual(final, steady, delta=1e-6 * steady)
        # and the current limit is what shapes the start of the run
        self.assertAlmostEqual(result.channels()["current_A"].max(),
                               system.current_limit_total, places=6)

    def test_feeding_the_bus_number_overstates_the_speed(self):
        # the trap: an integrator types the battery voltage into a model whose
        # voltage is the phase rail, and the drive looks sqrt(3) faster than it is
        from drivetrain import simulate
        system = presets.build("bldc_pod")
        bus = system.group.motor.spec.dc_bus_voltage
        fast = simulate.simulate(system, voltage=bus, duration=20.0, engine="rk4",
                                 samples=400).channels()["motor_rpm"][-1]
        honest = simulate.simulate(system, voltage=system.voltage_limit, duration=20.0,
                                   engine="rk4", samples=400).channels()["motor_rpm"][-1]
        self.assertGreater(fast, honest)


class SerialisationTests(unittest.TestCase):
    """The ``kind`` discriminator: presets, configs, ``--fit-json``."""

    def test_a_brushless_system_survives_a_round_trip(self):
        system = presets.build("bldc_pod")
        data = presets.to_dict(system)
        self.assertEqual(data["group"]["motor"]["kind"], "PMSMModel")
        self.assertIn("spec", data["group"]["motor"])
        back = presets.from_json(presets.to_json(system))
        self.assertIsInstance(back.group.motor, PMSMModel)
        self.assertEqual(back.group.count, system.group.count)
        self.assertEqual(back.group.motor.spec, system.group.motor.spec)
        self.assertAlmostEqual(back.group.motor.kt, system.group.motor.kt)
        self.assertAlmostEqual(back.group.motor.kv, system.group.motor.kv)

    def test_an_unconfigured_bus_survives_as_infinity(self):
        system = presets.build("bldc_pod")
        system.group.motor.spec = replace(system.group.motor.spec,
                                          dc_bus_voltage=math.inf)
        back = presets.from_json(presets.to_json(system))
        self.assertTrue(math.isinf(back.group.motor.spec.dc_bus_voltage))

    def test_a_brushed_system_still_round_trips(self):
        system = presets.build("pod_2x_direct")
        data = presets.to_dict(system)
        self.assertEqual(data["group"]["motor"]["kind"], "MotorModel")
        back = presets.from_json(presets.to_json(system))
        self.assertIsInstance(back.group.motor, MotorModel)
        self.assertNotIsInstance(back.group.motor, PMSMModel)

    def test_an_unknown_kind_is_an_error_not_a_silent_substitution(self):
        data = presets.to_dict(presets.build("bldc_pod"))
        data["group"]["motor"]["kind"] = "InductionMotor"
        with self.assertRaises(ConfigError) as ctx:
            presets.from_dict(data)
        self.assertIn("InductionMotor", str(ctx.exception))

    def test_a_brushed_config_without_a_kind_still_loads(self):
        # configs written before the discriminator existed
        data = presets.to_dict(presets.build("pod_2x_direct"))
        self.assertEqual(data["group"]["motor"]["kind"], "MotorModel")
        del data["group"]["motor"]["kind"]
        back = presets.from_dict(data)
        self.assertIsInstance(back.group.motor, MotorModel)


class CatalogueTests(unittest.TestCase):
    """Presets, the parameter panel, and ``--set`` paths."""

    def test_brushless_machines_are_in_the_catalogue(self):
        names = presets.motor_names()
        self.assertIn("bldc_24v", names)
        self.assertIn("bldc_12v", names)
        for name in ("bldc_24v", "bldc_12v"):
            self.assertIsInstance(presets.MOTORS[name], PMSMModel)
        self.assertIn("bldc_pod", presets.preset_names())

    def test_the_brushless_pod_preset_is_a_drive_not_a_toy(self):
        system = presets.build("bldc_pod")
        self.assertIsInstance(system.group.motor, PMSMModel)
        self.assertEqual(system.group.count, 2)
        self.assertTrue(system.wheel is not None)
        self.assertGreater(system.total_inertia, 0.0)
        self.assertGreater(system.current_limit_total, 0.0)

    def test_the_motor_swap_works_both_ways(self):
        brushless = presets.build("pod_4x_geared", {"motor": "bldc_12v"})
        self.assertIsInstance(brushless.group.motor, PMSMModel)
        self.assertEqual(brushless.group.count, 4)
        brushed = presets.build("bldc_pod", {"motor": "c23_l33_w20"})
        self.assertIsInstance(brushed.group.motor, MotorModel)
        self.assertNotIsInstance(brushed.group.motor, PMSMModel)

    def test_the_parameter_panel_lists_the_dq_constants(self):
        paths = {e["path"]: e for e in presets.field_catalog(presets.build("bldc_pod"))}
        for path in ("group.motor.spec.flux_linkage", "group.motor.spec.inductance_d",
                     "group.motor.spec.inductance_q", "group.motor.spec.resistance",
                     "group.motor.spec.current_limit", "group.motor.spec.voltage_limit",
                     "group.motor.spec.dc_bus_voltage"):
            self.assertIn(path, paths)
            self.assertTrue(paths[path]["unit"], f"{path} needs a unit for the widget")
        # pole pairs is a pure number, and the panel should say so rather than
        # invent a unit for it
        self.assertEqual(paths["group.motor.spec.pole_pairs"]["unit"], "")
        # kt/kv are derived here, so the panel must not offer them as inputs
        for path in ("group.motor.kt", "group.motor.kv", "group.motor.inductance"):
            self.assertNotIn(path, paths)
        # ...but the dq spec's own states are solved for, not design inputs, so
        # they stay out of the panel.  The adapter's rot_vel stays in: that is the
        # initial condition the simulator actually reads.
        for path in ("group.motor.spec.current_d", "group.motor.spec.current_q",
                     "group.motor.spec.rot_vel"):
            self.assertNotIn(path, paths)
        self.assertIn("group.motor.rot_vel", paths)

    def test_derived_properties_cannot_be_set_by_path(self):
        system = presets.build("bldc_pod")
        with self.assertRaises(ConfigError) as ctx:
            sweep.set_path(system, "group.motor.kt", 0.02)
        message = str(ctx.exception)
        self.assertIn("derived", message)
        self.assertIn("group.motor.spec", message)

    def test_the_dq_constants_can_be_set_by_path(self):
        system = presets.build("bldc_pod")
        edited = sweep.set_path(system, "group.motor.spec.flux_linkage", 2.0e-3)
        self.assertEqual(edited.group.motor.spec.flux_linkage, 2.0e-3)
        self.assertAlmostEqual(edited.group.motor.kt, 1.5 * 7 * 2.0e-3)
        self.assertAlmostEqual(edited.group.motor.kv, 7 * 2.0e-3)
        # the original is untouched: set_path returns a new system
        self.assertEqual(system.group.motor.spec.flux_linkage,
                         presets.MOTORS["bldc_24v"].spec.flux_linkage)

    def test_pole_pairs_stays_an_integer(self):
        system = presets.build("bldc_pod")
        edited = sweep.set_path(system, "group.motor.spec.pole_pairs", 10)
        self.assertIsInstance(edited.group.motor.spec.pole_pairs, int)
        self.assertEqual(edited.group.motor.spec.pole_pairs, 10)


class ExportTests(unittest.TestCase):
    """The generated controller code has to say what it is not good for."""

    def setUp(self):
        self.system = presets.build("bldc_pod")
        self.c = ModelConstants.from_system(self.system)

    def test_the_constants_come_from_the_machines_view(self):
        m = self.system.group.motor
        self.assertAlmostEqual(self.c.kt, m.kt)
        self.assertAlmostEqual(self.c.kv, m.kv)
        self.assertAlmostEqual(self.c.resistance, self.system.group.r_eff)
        self.assertAlmostEqual(self.c.inductance, self.system.group.l_eff)
        self.assertAlmostEqual(self.c.current_limit, self.system.current_limit_total)
        self.assertAlmostEqual(self.c.voltage_limit, self.system.voltage_limit)
        self.assertEqual(self.c.motor_count, 2)

    def test_the_notes_name_the_limits_of_a_collapsed_machine(self):
        notes = "\n".join(self.c.notes)
        self.assertIn("i_d=0", notes)
        self.assertIn("kt/kv = 1.5", notes)
        self.assertIn("Park convention", notes)
        self.assertIn("no field-weakening region", notes)
        self.assertIn("1099.7 rad/s", notes)          # base speed, motor side
        self.assertIn("447.8 rad/s", notes)           # dq band edge, motor side
        self.assertIn("cross-coupling", notes)
        self.assertIn("rail ammeter", notes)
        # the units trap an integrator is most likely to fall into
        self.assertIn("phase peak", notes)
        self.assertIn("sqrt(3)", notes)
        # the output-side numbers are the ones an integrator actually drives
        self.assertIn("183.7 rad/s output", notes)
        self.assertIn("74.8 rad/s output", notes)

    def test_a_brushed_build_carries_no_machine_notes(self):
        self.assertEqual(ModelConstants.from_system(presets.build("pod_2x_direct")).notes, ())

    def test_the_notes_reach_the_readme_and_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export(self.system, out, ExportOptions(dtype="f32",
                                                   targets=("python", "c"),
                                                   sample_time=0.001))
            readme = (out / "README.md").read_text(encoding="utf-8")
            self.assertIn("Caveats specific to this machine", readme)
            self.assertIn("no field-weakening region", readme)
            manifest = json.loads((out / "drive_model_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["notes"]), len(self.c.notes))
            self.assertIn("Park convention", "\n".join(manifest["notes"]))
            # the notes are also in the source banners, where the integrator
            # reading the header will meet them
            self.assertIn("Park convention", (out / "drive_model.py").read_text(encoding="utf-8"))

    def test_the_exported_model_reproduces_the_simulator(self):
        from drivetrain import simulate
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export(self.system, out, ExportOptions(dtype="f64", targets=("python",)))
            py = _load_module(out / "drive_model.py", "dt_bldc_py")
            result = simulate.simulate(self.system, voltage=self.system.voltage_limit,
                                       duration=2.0, dt=0.001, engine="rk4", samples=None)
            model = py.MODEL
            state = py.State(0.0, model.current_at(self.system.voltage_limit, 0.0))
            for _ in range(len(result.t) - 1):
                state = model.step(state, self.system.voltage_limit, 0.001)
            reference = result.channels()["motor_rpm"][-1] * 2.0 * math.pi / 60.0
            self.assertAlmostEqual(state.w, reference, delta=1e-9 * abs(reference) + 1e-12)

class FitTests(unittest.TestCase):
    """Fitting a brushless system: the free set has to land on settable paths.

    A brushless machine's constants live under ``group.motor.spec.*``, and its
    ``kt`` and ``kv`` are one flux linkage seen from two sides, so the fitter's
    brushed vocabulary (aliases, ``known=``, ties, log-derived starting values)
    is re-pointed rather than refused.  What cannot be re-pointed is reported:
    the collapse never evaluates the d-axis inductance, iron loss, windage, or
    magnet temperature, so those are inert by construction, not badly measured.
    """

    LAMBDA = 2.1e-3
    RESISTANCE = 0.09
    POLE_PAIRS = 7

    @classmethod
    def setUpClass(cls):
        truth = presets.build("bldc_pod")
        truth = sweep.set_path(truth, "group.motor.spec.flux_linkage", cls.LAMBDA)
        truth = sweep.set_path(truth, "group.motor.spec.resistance", cls.RESISTANCE)
        truth = sweep.set_path(truth, "group.motor.spec.bearing_damping", 2.5e-5)
        dt = 1.0 / 400.0
        # 2 V on a 13.86 V rail: the current stays under the 40 A group limit, so
        # the electrical channels still carry information about resistance and flux
        volts = simulate.voltage_signal("square", truth, 2.0, dt=dt, value=2.0, period=2.0)
        result = simulate.simulate(truth, volts, 2.0, dt=dt, samples=None)
        ch = result.channels()
        rng = np.random.default_rng(7)
        sel = slice(None, None, 8)
        cls.rows = {
            "time": ch["time"][sel],
            "volts": ch["voltage_V"][sel],
            "current": ch["current_A"][sel] + rng.normal(0, 0.02, size=len(ch["time"][sel])),
            "rpm": ch["motor_rpm"][sel] + rng.normal(0, 5.0, size=len(ch["time"][sel])),
        }
        cls.known = {
            "group.motor.rotor_inertia": 2.0e-5,
            "group.motor.bearing_damping": 2.5e-5,
        }
        cls.result = fit.fit(
            fit.Dataset.from_rows(cls.rows),
            system=presets.build("bldc_pod"),
            free=["group.motor.kv", "group.motor.resistance"],
            known=cls.known,
            engine="rk4",
        )

    def test_brushed_names_fit_the_spec_paths(self):
        fitted = {p.path: p.value for p in self.result.parameters}
        self.assertEqual(
            set(fitted), {"group.motor.spec.flux_linkage", "group.motor.spec.resistance"}
        )
        self.assertAlmostEqual(fitted["group.motor.spec.flux_linkage"], self.LAMBDA, delta=2e-5)
        self.assertAlmostEqual(fitted["group.motor.spec.resistance"], self.RESISTANCE, delta=2e-3)
        # the fitted machine still reports the convention ratio, because kt and kv
        # were never free: the flux linkage was
        motor = self.result.system.group.motor
        self.assertAlmostEqual(motor.kt / motor.kv, 1.5)
        self.assertAlmostEqual(motor.spec.flux_linkage, fitted["group.motor.spec.flux_linkage"])

    def test_the_re_pointing_is_reported(self):
        note = [w for w in self.result.warnings if "re-pointed" in w]
        self.assertEqual(len(note), 1)
        self.assertIn("group.motor.kv -> group.motor.spec.flux_linkage", note[0])
        self.assertIn("kt and kv are one flux linkage", note[0])

    def test_the_rail_ammeter_is_flagged(self):
        warning = [w for w in self.result.warnings if "DC rail" in w]
        self.assertEqual(len(warning), 1)
        self.assertIn("1.5 (R i + kv w) i", warning[0])

    def test_pinning_kt_and_kv_together_is_one_parameter(self):
        # consistent: kt = 1.5 p lambda and kv = p lambda for the same lambda
        result = fit.fit(
            fit.Dataset.from_rows(self.rows),
            system=presets.build("bldc_pod"),
            free=["group.motor.resistance"],
            known={**self.known,
                   "group.motor.kv": self.POLE_PAIRS * self.LAMBDA,
                   "group.motor.kt": 1.5 * self.POLE_PAIRS * self.LAMBDA},
            engine="rk4",
        )
        self.assertEqual([p.path for p in result.parameters], ["group.motor.spec.resistance"])
        self.assertAlmostEqual(result.system.group.motor.spec.flux_linkage, self.LAMBDA)

    def test_pinning_kt_and_kv_to_disagreeing_values_is_a_contradiction(self):
        with self.assertRaises(ConfigError) as caught:
            fit.fit(
                fit.Dataset.from_rows(self.rows),
                system=presets.build("bldc_pod"),
                free=["group.motor.resistance"],
                known={**self.known,
                       "group.motor.kv": self.POLE_PAIRS * self.LAMBDA,
                       "group.motor.kt": 0.02},
                engine="rk4",
            )
        message = str(caught.exception)
        self.assertIn("single parameter on a brushless machine", message)
        self.assertIn("group.motor.spec.flux_linkage", message)

    def test_tying_kt_to_kv_is_a_no_op_not_a_bug(self):
        result = fit.fit(
            fit.Dataset.from_rows(self.rows),
            system=presets.build("bldc_pod"),
            free=["group.motor.resistance"],
            known=self.known,
            ties={"group.motor.kt": "group.motor.kv"},
            engine="rk4",
        )
        self.assertEqual([p.path for p in result.parameters], ["group.motor.spec.resistance"])
        self.assertTrue([w for w in result.warnings if "one flux linkage" in w])

    def test_the_collapse_cannot_see_the_d_axis_constants(self):
        notes: list[str] = []
        paths = fit._expand_free("all", presets.build("bldc_pod"), {}, notes)
        for path in (
            "group.motor.spec.flux_linkage",
            "group.motor.spec.inductance_q",
            "group.motor.spec.resistance",
            "group.motor.spec.current_limit",
        ):
            self.assertIn(path, paths)
        for path in (
            "group.motor.spec.inductance_d",
            "group.motor.spec.core_loss_coefficient",
            "group.motor.spec.windage_coefficient",
            "group.motor.spec.flux_temperature_coefficient",
            "group.motor.spec.reference_temperature",
            "group.motor.spec.dc_bus_voltage",
        ):
            self.assertNotIn(path, paths)
        self.assertEqual(len(notes), 1)
        for path in ("group.motor.spec.inductance_d", "group.motor.spec.windage_coefficient"):
            self.assertIn(path, notes[0])
        self.assertIn("never evaluates them", notes[0])
        # naming one anyway is allowed: the reason is a note, not a wall
        self.assertEqual(
            fit._expand_free(["group.motor.spec.inductance_d"], presets.build("bldc_pod"), {}),
            ["group.motor.spec.inductance_d"],
        )

    def test_the_dq_state_is_not_a_parameter(self):
        for path in ("group.motor.spec.rot_vel", "group.motor.spec.current_q"):
            with self.assertRaises(ConfigError) as caught:
                fit._expand_free([path], presets.build("bldc_pod"), {})
            self.assertIn("state variable", str(caught.exception))

    def test_a_brushed_system_gets_no_translation(self):
        self.assertEqual(fit.dq_motor_map(presets.build("pod_2x_direct")), {})
        paths = fit._expand_free(["group.motor.kv", "group.motor.resistance"],
                                 presets.build("pod_2x_direct"), {})
        self.assertEqual(paths, ["group.motor.kv", "group.motor.resistance"])
        self.assertFalse([w for w in self.result.warnings if "re-pointed" in w and "legacy" in w])


if __name__ == "__main__":
    unittest.main()
