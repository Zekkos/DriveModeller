"""Tests for ``drivetrain.efficiency``: the loss split, one operating point, and
the map over the whole (speed, torque) plane.

The claim these tests hold is narrower than "the numbers are right" and more
useful than "it runs": the split has to *close*.  At every point, in every
quadrant,

    power_in_W - power_out_W == losses.total_W

must hold to floating-point accuracy, because that identity is the only reason a
loss table can be trusted at all.  A term that is missing -- iron loss on a
brushless machine, stiction billed at a speed where the model charges none, the
(kv - kt) disagreement on a brushed collapse -- breaks the books, and every test
here that could break the books does.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from drivetrain import cli, presets
from drivetrain.efficiency import (
    _LOSS_KEYS,
    _MATRIX_KEYS,
    _machine_loss_torque,
    efficiency_map,
    operating_point,
    LossSplit,
)
from drivetrain.exceptions import ConfigError
from drivetrain.pmsm import PMSMSpec
from drivetrain.sweep import set_path

# a machine with every loss term switched on, so a term that silently vanishes
# shows up as a wrong number rather than as a zero nobody notices
WITH_ALL_LOSSES = {
    "group.motor.spec.core_loss_coefficient": 1e-6,
    "group.motor.spec.windage_coefficient": 1e-8,
    "group.motor.spec.bearing_damping": 1e-5,
    "group.motor.spec.static_friction": 0.002,
    "group.motor.spec.dc_bus_voltage": 24.0,
}


def _all_loss_pod():
    return presets.build("bldc_pod", WITH_ALL_LOSSES)


def _books(point) -> float:
    """How far the split fails to close, in watts."""
    return point.power_in_W - point.losses.total_W - point.power_out_W


def _close_books(test, point, places=8):
    test.assertTrue(
        point.feasible,
        f"({point.speed_rad_s}, {point.torque_Nm}) should be feasible: {point.why}",
    )
    scale = max(abs(point.power_in_W), abs(point.power_out_W), point.losses.total_W, 1.0)
    test.assertAlmostEqual(_books(point) / scale, 0.0, places=places)


class LossSplitTests(unittest.TestCase):
    def test_total_excludes_the_magnet_term_because_it_is_not_modelled(self) -> None:
        split = LossSplit(copper_W=3.0, iron_W=1.0, magnet_W=None)
        self.assertAlmostEqual(split.total_W, 4.0)

    def test_share_sums_to_one_and_is_zero_when_nothing_is_lost(self) -> None:
        split = LossSplit(copper_W=6.0, bearing_W=2.0, gear_W=2.0)
        share = split.share()
        self.assertAlmostEqual(sum(share.values()), 1.0)
        self.assertAlmostEqual(share["copper"], 0.6)
        empty = LossSplit().share()
        self.assertEqual(set(empty), set(share))
        self.assertTrue(all(value == 0.0 for value in empty.values()))

    def test_the_magnet_column_stays_None_rather_than_becoming_zero(self) -> None:
        # 0.0 here would read as "this machine's magnets are lossless", which is a
        # claim the model has no coefficient to support
        split = LossSplit(copper_W=1.0)
        self.assertIsNone(split.as_dict()["magnet_W"])
        self.assertNotIn("magnet_W", split.terms())
        self.assertEqual(split.as_dict()["total_W"], 1.0)


class DqPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = presets.build("bldc_pod")
        self.spec = self.system.group.motor.spec
        self.count = self.system.group.count

    def test_it_agrees_with_the_machine_model_at_one_point(self) -> None:
        point = operating_point(self.system, 700.0, 0.2)
        # efficiency_at is per machine, so the group's torque is divided by the count
        self.assertAlmostEqual(
            point.efficiency, self.spec.efficiency_at(700.0, 0.1), places=12
        )
        self.assertAlmostEqual(point.power_in_W, 150.001504, places=5)
        self.assertAlmostEqual(point.losses.copper_W, 10.001504, places=5)
        self.assertAlmostEqual(point.losses.bearing_W, 0.98, places=6)

    def test_the_books_close_in_every_quadrant(self) -> None:
        for speed in (-1500.0, -700.0, 0.0, 300.0, 700.0, 1500.0):
            for torque in (-0.15, -0.05, 0.0, 0.05, 0.15):
                _close_books(self, operating_point(self.system, speed, torque))

    def test_the_loss_terms_add_up_to_the_total(self) -> None:
        point = operating_point(_all_loss_pod(), 700.0, 0.2)
        terms = point.losses.terms()
        self.assertAlmostEqual(sum(terms.values()), point.losses.total_W, places=12)
        self.assertEqual(set(terms), set(_LOSS_KEYS))

    def test_iron_loss_is_billed_where_the_model_knows_it(self) -> None:
        system = _all_loss_pod()
        spec = system.group.motor.spec
        point = operating_point(system, 700.0, 0.2)
        expected = system.group.count * spec.core_loss_coefficient * spec.pole_pairs**2 * 700.0**2
        self.assertAlmostEqual(point.losses.iron_W, expected, places=6)
        self.assertGreater(point.losses.iron_W, 0.0)
        # and the plain catalogue machine has no iron term switched on at all
        self.assertEqual(operating_point(self.system, 700.0, 0.2).losses.iron_W, 0.0)

    def test_a_brushless_machine_pays_no_stiction_while_it_is_turning(self) -> None:
        # PMSMSpec.loss_torque() charges static_friction only at standstill, where
        # it is a holding torque and dissipates nothing; billing it at every speed
        # invents a loss the integrator never applies
        system = _all_loss_pod()
        point = operating_point(system, 700.0, 0.2)
        self.assertEqual(point.losses.friction_W, 0.0)
        self.assertGreater(system.group.static_friction_eff, 0.0)
        # at standstill it is still in the loss torque, it just cannot dissipate
        self.assertGreater(_machine_loss_torque(system, 0.0, False), 0.0)
        self.assertEqual(operating_point(system, 0.0, 0.1).power_out_W, 0.0)

    def test_a_point_outside_the_envelope_is_refused_not_invented(self) -> None:
        limit = self.count * float(self.spec.torque_limit(700.0))
        point = operating_point(self.system, 700.0, limit + 0.01)
        self.assertFalse(point.feasible)
        self.assertIn("envelope", point.why)
        self.assertTrue(math.isnan(point.power_in_W))

    def test_the_envelope_itself_is_feasible(self) -> None:
        for speed in (0.0, 447.774641, 700.0, 1500.0):
            limit = self.count * float(self.spec.torque_limit(speed))
            point = operating_point(self.system, speed, limit * (1.0 - 1e-9))
            self.assertTrue(point.feasible, point.why)

    def test_braking_bills_the_shaft_as_the_input(self) -> None:
        point = operating_point(self.system, 700.0, -0.2)
        self.assertLess(point.power_in_W, 0.0)  # returned to the source
        self.assertLess(point.power_out_W, 0.0)  # taken by the load
        self.assertLess(point.developed_W, 0.0)
        # useful over absorbed: the shaft is absorbed, the terminals are useful
        self.assertAlmostEqual(
            point.efficiency, abs(point.power_in_W) / abs(point.power_out_W)
        )
        _close_books(self, point)

    def test_the_map_is_symmetric_about_the_origin(self) -> None:
        # a negative speed with a torque of the same sign is still driving: the
        # machine mirrors, it does not invert, so every power keeps its sign
        up = operating_point(self.system, 700.0, 0.2)
        down = operating_point(self.system, -700.0, -0.2)
        self.assertAlmostEqual(up.efficiency, down.efficiency, places=12)
        self.assertAlmostEqual(up.losses.total_W, down.losses.total_W, places=12)
        self.assertAlmostEqual(up.power_in_W, down.power_in_W, places=9)
        self.assertAlmostEqual(up.power_out_W, down.power_out_W, places=9)
        braking = operating_point(self.system, -700.0, 0.2)  # torque against motion
        self.assertLess(braking.power_in_W, 0.0)

    def test_the_field_weakening_flag_means_weakening_not_reluctance(self) -> None:
        # a salient machine runs MTPA with i_d < 0 at ANY torque to collect
        # reluctance torque, so the sign of i_d alone cannot flag field weakening;
        # weakening is the drive pushing the d axis in to stay on the voltage rail
        edge = self.system.group.motor.constant_torque_speed()
        self.assertLess(edge, self.system.group.motor.view_constant_torque_speed())
        speed = 0.5 * edge
        near_limit = 0.9 * self.count * float(self.spec.torque_limit(speed))
        below = operating_point(self.system, speed, near_limit)
        self.assertFalse(below.field_weakening)
        i_d = float(self.spec.current_pair(near_limit / self.count, speed)[0])
        self.assertLess(i_d, 0.0)  # on the d axis, but not weakening
        heavy = 0.9 * self.count * float(self.spec.torque_limit(2.0 * edge))
        self.assertTrue(operating_point(self.system, 2.0 * edge, heavy).field_weakening)
        # above the rail speed even a light torque needs weakening: the back-EMF
        # alone wants more voltage than the drive has
        light = 0.05 * self.count * float(self.spec.torque_limit(2500.0))
        fast = operating_point(self.system, 2500.0, light)
        self.assertTrue(fast.field_weakening)
        # weakening costs current: the same torque far below the rail needs only the
        # torque-producing q axis, here most of the current is demagnetising d axis
        slow = operating_point(self.system, 100.0, light)
        self.assertGreater(fast.current_per_motor_A, 4.0 * slow.current_per_motor_A)

    def test_standstill_developes_power_but_delivers_none(self) -> None:
        point = operating_point(self.system, 0.0, 0.3)
        self.assertTrue(point.feasible)
        self.assertIsNone(point.efficiency)
        self.assertEqual(point.developed_W, 0.0)
        self.assertGreater(point.power_in_W, 0.0)  # all of it copper
        self.assertIn("losses consume", point.why)
        self.assertAlmostEqual(point.power_in_W, point.losses.copper_W, places=9)
        self.assertLess(point.current_A, self.system.current_limit_total)
        # the full standstill envelope does sit on the group current limit
        limit = self.count * float(self.spec.torque_limit(0.0))
        pinned = operating_point(self.system, 0.0, limit)
        self.assertAlmostEqual(pinned.current_A, self.system.current_limit_total, places=6)

    def test_the_dc_link_column_follows_the_bus(self) -> None:
        point = operating_point(self.system, 700.0, 0.2)  # bus is 24 V
        self.assertAlmostEqual(point.dc_link_current_A, point.power_in_W / 24.0, places=9)
        system = presets.build("bldc_pod", {"group.motor.spec.dc_bus_voltage": float("inf")})
        self.assertIsNone(operating_point(system, 700.0, 0.2).dc_link_current_A)

    def test_a_shaft_side_torque_is_referred_back_to_developed(self) -> None:
        point = operating_point(self.system, 700.0, 0.2, torque_is="shaft")
        mech = (
            point.losses.iron_W
            + point.losses.windage_W
            + point.losses.bearing_W
            + point.losses.friction_W
        )
        self.assertAlmostEqual(point.shaft_power_W, 0.2 * 700.0, places=9)
        self.assertAlmostEqual(point.developed_W - mech, point.shaft_power_W, places=9)
        self.assertGreater(point.torque_Nm, 0.2)  # developed has to cover the losses too

    def test_the_gear_stage_is_billed_only_when_asked(self) -> None:
        plain = operating_point(self.system, 700.0, 0.2)
        geared = operating_point(self.system, 700.0, 0.2, include_gearbox=True)
        self.assertEqual(plain.losses.gear_W, 0.0)
        eta = float(self.system.gearbox.efficiency)
        expected = (1.0 - eta) * abs(geared.shaft_power_W) + self.system.gearbox.damping * 700.0**2
        self.assertAlmostEqual(geared.losses.gear_W, expected, places=9)
        self.assertLess(geared.efficiency, plain.efficiency)
        _close_books(self, geared)


class BrushedPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = presets.build("pod_4x_geared")
        self.group = self.system.group

    def test_current_and_voltage_follow_the_collapse(self) -> None:
        point = operating_point(self.system, 100.0, 0.4)
        current = 0.4 / self.group.kt_eff
        self.assertAlmostEqual(point.current_A, current, places=12)
        self.assertAlmostEqual(
            point.voltage_V,
            current * self.group.r_eff + self.group.kv_eff * 100.0,
            places=12,
        )
        _close_books(self, point)

    def test_windage_is_zero_and_stiction_only_when_the_preset_has_it(self) -> None:
        point = operating_point(self.system, 100.0, 0.4)
        self.assertEqual(point.losses.windage_W, 0.0)  # the collapse lumps it in
        self.assertGreater(point.losses.bearing_W, 0.0)
        self.assertEqual(point.losses.friction_W, 0.0)  # this preset has none
        stiction = presets.build(
            "pod_4x_geared", {"group.motor.static_friction": 0.01}
        )
        point = operating_point(stiction, 100.0, 0.4)
        self.assertGreater(point.losses.friction_W, 0.0)  # Coulomb, charged at every speed
        _close_books(self, point)

    def test_a_kt_kv_disagreement_is_billed_as_a_residual(self) -> None:
        kt, kv = self.group.kt_eff, self.group.kv_eff
        self.assertNotAlmostEqual(kt, kv, places=9)
        point = operating_point(self.system, 100.0, 0.4)
        self.assertAlmostEqual(
            point.losses.residual_W, (kv - kt) * point.current_A * 100.0, places=12
        )
        # and a machine whose constants agree has nothing left over
        system = presets.build("pod_4x_geared", {"group.motor.kv": self.group.kt_eff})
        self.assertAlmostEqual(
            operating_point(system, 100.0, 0.4).losses.residual_W, 0.0, places=12
        )

    def test_the_books_close_over_a_sweep(self) -> None:
        for speed in (-300.0, -100.0, 0.0, 100.0, 300.0):
            for torque in (-0.5, -0.1, 0.0, 0.1, 0.5):
                point = operating_point(self.system, speed, torque, include_gearbox=True)
                if point.feasible:
                    _close_books(self, point)

    def test_an_impossible_current_is_refused(self) -> None:
        point = operating_point(self.system, 0.0, self.group.kt_eff * 400.0)
        self.assertFalse(point.feasible)
        self.assertTrue(point.current_limited)
        self.assertIn("current", point.why)


class MapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system = presets.build("bldc_pod")
        cls.map = efficiency_map(cls.system, n_speeds=9, n_torques=9)

    def test_the_grid_shape_and_the_feasible_count(self) -> None:
        self.assertEqual(self.map.shape, (9, 9))
        self.assertEqual(self.map.n_feasible(), int(self.map.feasible.sum()))
        self.assertGreater(self.map.n_feasible(), 0)
        self.assertEqual(self.map.kind, "PMSMModel x2")

    def test_every_matrix_key_the_module_advertises_is_present(self) -> None:
        self.assertEqual(set(self.map.grid), set(_MATRIX_KEYS))

    def test_an_unknown_matrix_names_the_ones_that_exist(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self.map.matrix("efficiency_percent")
        self.assertIn("carries", str(ctx.exception))

    def test_the_magnet_matrix_is_all_nan_because_nothing_is_modelled_there(self) -> None:
        magnet = self.map.matrix("magnet_W")
        self.assertTrue(np.all(np.isnan(magnet)))

    def test_the_envelope_is_the_current_search_not_the_grid(self) -> None:
        env = self.map.envelope()
        count = self.system.group.count
        spec = self.system.group.motor.spec
        for i, speed in enumerate(self.map.speeds):
            self.assertAlmostEqual(
                env["drive_Nm"][i], count * float(spec.torque_limit(float(speed))), places=9
            )
            self.assertAlmostEqual(
                env["brake_Nm"][i],
                -count * float(spec.torque_limit(float(speed), regenerative=True)),
                places=9,
            )

    def test_the_ridge_and_the_ridge_torque_agree(self) -> None:
        eff = self.map.ridge()
        torque = self.map.ridge_torque()
        for i, (best_eff, best_tau) in enumerate(zip(eff, torque)):
            if math.isnan(best_eff):
                self.assertTrue(math.isnan(best_tau))
                continue
            row = self.map.points_at(i)
            winner = [p for p in row if p.torque_Nm == best_tau and p.efficiency is not None]
            self.assertTrue(winner)
            self.assertAlmostEqual(winner[0].efficiency, best_eff, places=12)

    def test_best_is_the_maximum_over_the_grid(self) -> None:
        best = self.map.best("motoring")
        grid = self.map.efficiency.copy()
        grid[:, self.map.torques < 0.0] = np.nan
        self.assertAlmostEqual(best["efficiency"], float(np.nanmax(grid)), places=12)
        self.assertGreater(best["torque_Nm"], 0.0)
        self.assertIn("share", best)
        self.assertAlmostEqual(sum(best["share"].values()), 1.0)
        braking = self.map.best("braking")
        self.assertLess(braking["torque_Nm"], 0.0)
        self.assertLessEqual(braking["efficiency"], 1.0)

    def test_best_rejects_a_quadrant_that_means_nothing(self) -> None:
        for bad in ("drive", "regen", ""):
            with self.assertRaises(ConfigError):
                self.map.best(bad)

    def test_rows_covers_every_cell_including_the_impossible_ones(self) -> None:
        rows = list(self.map.rows())
        self.assertEqual(len(rows), 81)
        self.assertEqual(
            sum(1 for row in rows if not row["feasible"]), 81 - self.map.n_feasible()
        )

    def test_the_csv_round_trips_and_leaves_the_unmodelled_column_empty(self) -> None:
        directory = tempfile.mkdtemp(prefix="dt-eff-csv-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = self.map.write_csv(Path(directory) / "map.csv")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 81)
        self.assertEqual(rows[0]["magnet_W"], "")
        feasible = [row for row in rows if row["feasible"] == "True"]
        self.assertEqual(len(feasible), self.map.n_feasible())
        for row in feasible:
            self.assertAlmostEqual(
                float(row["power_in_W"]) - float(row["power_out_W"]),
                float(row["total_W"]),
                places=6,
            )
            if row["efficiency"] == "":
                # no ratio to quote: standstill delivers nothing, and a light
                # torque at a fast speed can be eaten whole by the machine's losses
                self.assertLessEqual(float(row["power_out_W"]), 0.0)
            else:
                # useful over absorbed, and which side is absorbed flips per quadrant,
                # so the ratio is always the smaller magnitude over the larger one
                self.assertAlmostEqual(
                    float(row["efficiency"]),
                    min(abs(float(row["power_in_W"])), abs(float(row["power_out_W"])))
                    / max(abs(float(row["power_in_W"])), abs(float(row["power_out_W"]))),
                    places=9,
                )

    def test_as_dict_is_strict_json(self) -> None:
        # NaN is what Python prints and what a strict parser refuses; a map that
        # cannot be loaded back is not a machine-readable artifact
        text = json.dumps(self.map.as_dict(), allow_nan=False)
        restored = json.loads(text)
        self.assertEqual(restored["kind"], "PMSMModel x2")
        self.assertEqual(len(restored["grid"]["efficiency"]), 9)
        self.assertIsNone(restored["grid"]["efficiency"][0][0])  # standstill cell

    def test_the_report_shows_the_ridge_and_the_caveats(self) -> None:
        text = self.map.report()
        self.assertIn("efficiency map: PMSMModel x2", text)
        self.assertIn("best motoring", text)
        self.assertIn("ridge", text)
        self.assertIn("not modelled on this map: magnet", text)
        self.assertIn("speed range 0 to", text)

    def test_the_auto_speed_range_is_capped_for_a_dq_machine(self) -> None:
        # with i_d allowed to -lambda/Ld the voltage requirement flattens out, so
        # the machine's own no-load top speed is useless as an axis limit
        notes = " ".join(self.map.notes)
        self.assertIn("twice the base speed", notes)
        base = float(self.system.group.motor.spec.base_speed())
        self.assertAlmostEqual(float(self.map.speeds[-1]), 2.0 * base, places=6)

    def test_the_auto_speed_range_is_the_no_load_speed_for_a_brushed_collapse(self) -> None:
        system = presets.build("pod_4x_geared")
        emap = efficiency_map(system, n_speeds=5, n_torques=5)
        self.assertIn("no-load speed", " ".join(emap.notes))
        self.assertAlmostEqual(
            float(emap.speeds[-1]), float(system.steady_speed(system.voltage_limit)), places=6
        )
        self.assertIn("iron loss", " ".join(emap.not_modelled))

    def test_motoring_only_starts_the_torque_axis_at_zero(self) -> None:
        emap = efficiency_map(self.system, n_speeds=5, n_torques=5, include_regen=False)
        self.assertEqual(float(emap.torques[0]), 0.0)
        self.assertTrue(np.all(emap.torques >= 0.0))
        self.assertIsNone(emap.best("braking"))

    def test_explicit_axes_are_used_verbatim(self) -> None:
        emap = efficiency_map(
            self.system,
            speeds=[0.0, 500.0, 1000.0],
            torques=[-0.2, 0.0, 0.2, 0.4],
        )
        self.assertEqual(emap.shape, (3, 4))
        self.assertEqual(emap.speeds.tolist(), [0.0, 500.0, 1000.0])
        self.assertEqual(emap.torques.tolist(), [-0.2, 0.0, 0.2, 0.4])

    def test_torque_top_names_the_axis(self) -> None:
        emap = efficiency_map(self.system, n_speeds=5, n_torques=5, torque_top=0.25)
        self.assertAlmostEqual(float(emap.torques[0]), -0.25)
        self.assertAlmostEqual(float(emap.torques[-1]), 0.25)

    def test_an_axis_has_to_be_increasing_and_long_enough(self) -> None:
        for kwargs, phrase in (
            ({"speeds": [100.0, 100.0, 200.0]}, "strictly increasing"),
            ({"speeds": [1.0]}, "at least two"),
            ({"speeds": [float("nan"), 1.0, 2.0]}, "finite"),
            ({"torques": [0.5, 0.25, 1.0]}, "strictly increasing"),
            ({"n_speeds": 1}, "at least 2"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ConfigError) as ctx:
                    efficiency_map(self.system, **kwargs)
                self.assertIn(phrase, str(ctx.exception))

    def test_a_bad_torque_convention_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            efficiency_map(self.system, torque_is="wheel")


class IronLossParityTests(unittest.TestCase):
    """``PMSMSpec.efficiency_at`` used to bill bearings and windage but not iron."""

    def test_efficiency_at_bills_every_loss_the_model_knows(self) -> None:
        spec = PMSMSpec(core_loss_coefficient=1e-6)
        speed, torque = 700.0, 0.1
        i_d, i_q, _, _ = spec.current_pair(torque, speed)
        v_d, v_q = spec.voltage_steady(i_d, i_q, speed)
        drawn = spec.input_power(v_d, v_q, i_d, i_q)
        useful = torque * speed - spec.loss_torque(speed) * abs(speed)
        self.assertAlmostEqual(spec.efficiency_at(speed, torque), useful / drawn, places=12)
        self.assertGreater(spec.core_loss_power(speed), 0.0)

    def test_iron_loss_moves_the_number_at_all(self) -> None:
        plain = PMSMSpec().efficiency_at(700.0, 0.1)
        lossy = PMSMSpec(core_loss_coefficient=1e-6).efficiency_at(700.0, 0.1)
        self.assertLess(lossy, plain - 0.2)

    def test_the_map_and_the_closed_form_agree_once_iron_is_on(self) -> None:
        system = _all_loss_pod()
        spec = system.group.motor.spec
        point = operating_point(system, 700.0, 0.2)
        self.assertAlmostEqual(point.efficiency, spec.efficiency_at(700.0, 0.1), places=12)


class CliEfficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="dt-cli-eff-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _quiet(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_one_operating_point(self) -> None:
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "bldc_pod", "--speed", "700", "--torque", "0.2"]
        )
        self.assertEqual(code, 0)
        self.assertIn("92.68%", text)
        self.assertIn("losses [W]: copper", text)

    def test_one_point_as_json(self) -> None:
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "bldc_pod", "--speed", "700", "--torque", "0.2", "--json"]
        )
        self.assertEqual(code, 0)
        point = json.loads(text)
        self.assertAlmostEqual(point["efficiency"], 0.926791, places=6)
        self.assertIsNone(point["losses"]["magnet_W"])

    def test_a_map_writes_the_csv(self) -> None:
        csv_path = Path(self.dir) / "map.csv"
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "bldc_pod", "--map", "--n-speeds", "7",
             "--n-torques", "7", "--csv", str(csv_path)]
        )
        self.assertEqual(code, 0)
        self.assertIn("49 rows to", text)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 49)

    def test_a_map_as_json_is_strict_json(self) -> None:
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "bldc_pod", "--map", "--n-speeds", "5",
             "--n-torques", "5", "--json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(text, parse_constant=lambda bad: (_ for _ in ()).throw(ValueError(bad)))
        self.assertEqual(payload["axes"]["speed_rad_s"][0], 0.0)
        self.assertIs(payload["grid"]["feasible"][0][0], True)

    def test_a_point_needs_both_axes(self) -> None:
        code, _, err = self._quiet(["efficiency", "--preset", "bldc_pod", "--speed", "700"])
        self.assertEqual(code, 2)
        self.assertIn("give --speed and --torque", err)

    def test_a_brushed_preset_maps_too(self) -> None:
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "pod_4x_geared", "--map", "--n-speeds", "5",
             "--n-torques", "5", "--include-gearbox"]
        )
        self.assertEqual(code, 0)
        self.assertIn("MotorModel x4", text)
        self.assertIn("gear", text)
        self.assertIn("iron loss", text)

    def test_an_infeasible_point_is_reported_not_an_error(self) -> None:
        code, text, _ = self._quiet(
            ["efficiency", "--preset", "bldc_pod", "--speed", "700", "--torque", "50"]
        )
        self.assertEqual(code, 0)
        self.assertIn("envelope", text)


if __name__ == "__main__":
    unittest.main()
