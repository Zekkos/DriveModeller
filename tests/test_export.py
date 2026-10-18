"""Tests for the standalone C / Rust / Python export.

The point of ``drivetrain.export`` is that the code it emits is the *same
model* as the simulator, in a language the firmware speaks.  So the tests are
parity tests, not smoke tests:

* the constants handed to the generator are the ones the simulator runs on;
* the generated Python integrates identically to ``engine="rk4"``;
* the generated C and Rust integrate identically to the generated Python;
* the feed-forward law round-trips, i.e. inverting the model and then
  simulating it lands back on the commanded speed.

The C and Rust cases compile with whatever toolchain is on the box and skip
themselves when there is none.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from drivetrain import cli, presets, simulate
from drivetrain.export import (ConfigError, ExportOptions, ModelConstants,
                               constants_from_system, export, manifest_name)
from drivetrain.sweep import set_path


def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _f32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rpm_to_rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


# ----------------------------------------------------------------------
# the constants
# ----------------------------------------------------------------------
class ConstantsTests(unittest.TestCase):
    """What leaves the building must be what the simulator runs on."""

    def test_matches_the_simulation_kernel(self):
        from drivetrain.simulate import _FrozenModel

        for name in ("legacy", "legacy_4x", "pod_2x_direct", "pod_4x_geared",
                     "spinning_mass", "sprinter", "winch"):
            system = presets.build(name)
            c = constants_from_system(system)
            kernel = _FrozenModel.build(system)
            with self.subTest(preset=name):
                self.assertAlmostEqual(c.kt, kernel.kt, places=12)
                self.assertAlmostEqual(c.kv, kernel.kv, places=12)
                self.assertAlmostEqual(c.resistance, kernel.R, places=12)
                self.assertAlmostEqual(c.inductance, kernel.L, places=12)
                self.assertAlmostEqual(c.inertia, kernel.J, places=12)
                self.assertAlmostEqual(c.damping, kernel.b, places=12)
                self.assertAlmostEqual(c.static_friction, kernel.static, places=12)
                self.assertAlmostEqual(c.load_static, kernel.ts, places=12)
                self.assertAlmostEqual(c.load_viscous, kernel.bv, places=12)
                self.assertAlmostEqual(c.load_quadratic, kernel.ca, places=12)
                self.assertAlmostEqual(c.current_limit, kernel.limit, places=9)
                self.assertEqual(c.is_second_order, not kernel.first_order)

    def test_derived_quantities_agree_with_the_system(self):
        system = presets.build("pod_2x_direct")
        c = constants_from_system(system)
        self.assertAlmostEqual(c.electromechanical_damping,
                               system.electromechanical_damping, places=12)
        # the exported pole is the small-signal one at a chosen speed - tangent
        # drag, 2*ca*|w| - while describe() quotes the secant pole at the
        # nominal speed.  They differ by design; the exported one is what sizes
        # the control period, so check it against its own definition.
        b0 = c.damping + c.load_viscous + c.electromechanical_damping
        self.assertAlmostEqual(c.mech_time_constant(0.0), c.inertia / b0, places=9)
        self.assertLess(c.mech_time_constant(200.0), c.mech_time_constant(0.0))
        self.assertAlmostEqual(c.no_load_speed(12.0), 12.0 / c.kv, places=6)
        self.assertEqual(c.motor_count, 2)
        self.assertAlmostEqual(c.per_motor_current_limit, c.current_limit / 2.0)

    def test_inductance_makes_the_model_second_order(self):
        system = set_path(presets.build("pod_2x_direct"), "group.motor.inductance", 0.002)
        c = constants_from_system(system)
        self.assertTrue(c.is_second_order)
        # two motors in parallel halve the terminal inductance
        self.assertAlmostEqual(c.electrical_time_constant, (0.002 / 2.0) / c.resistance,
                               places=9)
        self.assertLess(c.max_step_time(), 0.01)

    def test_steady_speed_matches_the_system_solver(self):
        system = presets.build("pod_2x_direct")
        c = constants_from_system(system)
        # the exported root is the voltage-limited answer; this rig settles
        # well below its current limit, so the two must agree
        self.assertLess(abs((12.0 - c.kv * c.steady_speed(12.0)) / c.resistance),
                        c.current_limit)
        self.assertAlmostEqual(c.steady_speed(12.0), system.steady_speed(12.0), places=6)
        self.assertAlmostEqual(c.steady_output_speed(12.0),
                               system.steady_output_speed(12.0), places=6)
        self.assertAlmostEqual(c.steady_speed(-12.0), -c.steady_speed(12.0), places=9)

    def test_every_preset_can_be_exported(self):
        # a grade or a payload is a constant torque, which the collapsed form
        # absorbs, so every shipped preset must be exportable
        for name in presets.preset_names():
            with self.subTest(preset=name):
                ModelConstants.from_system(presets.build(name))


# ----------------------------------------------------------------------
# the files
# ----------------------------------------------------------------------
class ExportFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dt-export-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.system = presets.build("pod_2x_direct")

    def test_every_target_is_written_with_a_manifest(self):
        options = ExportOptions(targets=("c", "rust", "python", "ros"), sample_time=0.002)
        paths = export(self.system, self.dir, options)
        names = {p.name for p in paths}
        self.assertEqual(names, {
            "drive_model.h", "drive_model.c", "bench_drive_model.c",
            "drive_model.rs", "bench_drive_model.rs",
            "drive_model.py", "bench_drive_model.py",
            "drive_model_ros.py", "bench_drive_model_ros.py",
            "drive_model_manifest.json", "README.md"})
        manifest = json.loads((Path(self.dir) / f"{manifest_name(options)}.json").read_text())
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["dtype"], "f32")
        self.assertEqual(manifest["sample_time_s"], 0.002)
        self.assertIn("kv", manifest["constants"])
        self.assertIn("mech_time_constant_s", manifest["derived"])
        listed = {f["name"] for f in manifest["files"]}
        skip = {"README.md", "drive_model_manifest.json"}
        self.assertEqual(listed - skip, names - skip)

    def test_targets_are_respected(self):
        paths = export(self.system, self.dir, ExportOptions(targets=("c",)))
        names = {p.name for p in paths}
        self.assertIn("drive_model.c", names)
        self.assertFalse(any(n.endswith((".rs", ".py")) for n in names))

    def test_module_name_and_prefix_rename_everything(self):
        options = ExportOptions(targets=("c", "rust", "python"), module_name="rover",
                                header_prefix="rv")
        names = {p.name for p in export(self.system, self.dir, options)}
        self.assertIn("rover.c", names)
        self.assertIn("rover.h", names)
        self.assertIn("rover.rs", names)
        text = (Path(self.dir) / "rover.h").read_text()
        self.assertIn("rv_rover_STICTION_BAND", text)
        self.assertIn("extern const rv_model rv_rover;", text)

    def test_f32_builds_carry_f32_values_everywhere(self):
        export(self.system, self.dir, ExportOptions(targets=("c", "python")))
        source = (Path(self.dir) / "drive_model.c").read_text()
        py = _load_module(Path(self.dir) / "drive_model.py", "dt_f32_model")
        literal = re.search(r"/\*kv\*/ ([0-9.e+-]+)f,", source)
        self.assertIsNotNone(literal, "the f32 build must emit f-suffixed literals")
        self.assertAlmostEqual(float(literal.group(1)), py.MODEL.kv, delta=1e-15)
        self.assertEqual(py.MODEL.kv, _f32_round(py.MODEL.kv))
        self.assertNotEqual(py.MODEL.kv, constants_from_system(self.system).kv)

    def test_f64_builds_keep_full_precision(self):
        export(self.system, self.dir, ExportOptions(dtype="f64", targets=("python",)))
        py = _load_module(Path(self.dir) / "drive_model.py", "dt_f64_model")
        self.assertEqual(py.MODEL.kv, constants_from_system(self.system).kv)

    def test_bad_options_are_rejected(self):
        for bad in (dict(dtype="f16"), dict(targets=("verilog",)),
                    dict(sample_time=0.0), dict(module_name="drive-model")):
            with self.subTest(**bad), self.assertRaises(ConfigError):
                ExportOptions(**bad)

    def test_the_readme_documents_what_was_written(self):
        export(self.system, self.dir, ExportOptions(targets=("c", "python")))
        readme = (Path(self.dir) / "README.md").read_text()
        self.assertIn("drive_model.c", readme)
        self.assertIn("feed-forward", readme.lower())


# ----------------------------------------------------------------------
# the generated Python: parity with the simulator
# ----------------------------------------------------------------------
class PythonModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="dt-py-")
        cls.system = presets.build("pod_2x_direct")
        export(cls.system, cls.dir, ExportOptions(dtype="f64",
                                                  targets=("python", "ros"),
                                                  sample_time=0.001))
        sys.path.insert(0, cls.dir)
        cls.py = _load_module(Path(cls.dir) / "drive_model.py", "dt_gen_model")
        cls.ros = _load_module(Path(cls.dir) / "drive_model_ros.py", "dt_gen_ros")
        cls.model = cls.py.MODEL

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.dir)
        for name in ("dt_gen_model", "dt_gen_ros", "drive_model"):
            sys.modules.pop(name, None)
        shutil.rmtree(cls.dir, True)

    def _integrate(self, voltage: float, steps: int, dt: float = 0.001):
        """Step the exported model exactly as many times as the engine did.

        ``simulate`` returns a grid of ``duration/dt`` points, i.e. one fewer
        integration step than that.  Off by one here is a real error on a rig
        that is still accelerating, so the step count comes from the result.
        """
        state = self.py.State(0.0, 0.0)
        for _ in range(steps):
            state = self.model.step(state, voltage, dt)
        return state

    def test_the_step_matches_the_repo_rk4_engine(self):
        result = simulate.simulate(self.system, voltage=12.0, duration=3.0,
                                   dt=0.001, engine="rk4", samples=None)
        reference = _rpm_to_rad_s(result.channels()["motor_rpm"][-1])
        state = self._integrate(12.0, len(result.t) - 1)
        self.assertAlmostEqual(state.w, reference, delta=1e-6 * abs(reference))
        # the terminal current at that speed is the algebraic one
        self.assertAlmostEqual(state.i, self.model.current_at(12.0, state.w),
                               delta=1e-6)

    def test_the_second_order_step_matches_the_repo_rk4_engine(self):
        system = set_path(self.system, "group.motor.inductance", 0.002)
        export(system, self.dir, ExportOptions(dtype="f64", targets=("python",),
                                               module_name="second_order"))
        second = _load_module(Path(self.dir) / "second_order.py", "dt_second_order")
        result = simulate.simulate(system, voltage=12.0, duration=1.0, dt=0.0005,
                                   engine="rk4", samples=None)
        reference = _rpm_to_rad_s(result.channels()["motor_rpm"][-1])
        # with inductance, current is a state, and the engine starts it at the
        # algebraic value the applied voltage implies at standstill.  Starting
        # it at zero instead costs a few milliseconds of torque, which is worth
        # ~0.14 rad/s of permanent offset - far more than integrator error.
        state = second.State(0.0, second.MODEL.current_at(12.0, 0.0))
        for _ in range(len(result.t) - 1):
            state = second.MODEL.step(state, 12.0, 0.0005)
        self.assertTrue(second.MODEL.inductance > 0.0)
        self.assertAlmostEqual(state.w, reference, delta=1e-9 * abs(reference))
        self.assertAlmostEqual(state.i, result.channels()["current_A"][-1],
                               delta=1e-9 * abs(state.i) + 1e-12)

    def test_the_feed_forward_law_round_trips(self):
        """Invert the model, simulate the result, and the speed holds."""
        for speed in (0.0, 20.0, 60.0, 120.0, 200.0):
            with self.subTest(speed=speed):
                voltage = self.model.steady_voltage(speed, 0.0)
                self.assertLessEqual(abs(voltage), self.model.voltage_limit)
                state = self.py.State(speed, self.model.current_at(voltage, speed))
                for _ in range(2000):
                    state = self.model.step(state, voltage, 0.0005)
                self.assertAlmostEqual(state.w, speed, delta=1e-3 * max(1.0, speed))

    def test_the_feed_forward_output_side_helper_matches_the_motor_side_one(self):
        ratio = self.model.gear_ratio
        command = self.model.feedforward_output(60.0, 5.0, 2.0)
        direct = self.model.feedforward(60.0 / ratio, 5.0 / ratio, 2.0 / ratio)
        self.assertAlmostEqual(command.voltage, direct.voltage, places=9)

    def test_the_ramp_is_acceleration_limited(self):
        """The limiter clamps the *change*, so a step reference still arrives."""
        ff = self.py.FeedForward(self.model, 0.002, accel_limit=20.0)
        ff.update(100.0)
        self.assertAlmostEqual(ff.reference, 20.0 * 0.002, places=12)
        for _ in range(99):
            ff.update(100.0)
        # 100 ticks at 0.002 s with a 20 rad/s^2 limit is 2.0 rad/s of travel
        self.assertAlmostEqual(ff.reference, 100.0 * 0.002 * 20.0, places=9)
        for _ in range(5000):
            ff.update(100.0)
        self.assertAlmostEqual(ff.reference, 100.0, places=9)

    def test_the_speed_limit_caps_the_command(self):
        ff = self.py.FeedForward(self.model, 0.002, speed_limit=50.0)
        for _ in range(20000):
            command = ff.update(1e9)
        self.assertAlmostEqual(ff.reference, 50.0, places=6)
        self.assertLessEqual(abs(command.voltage), self.model.voltage_limit)

    def test_saturation_is_reported(self):
        command = self.model.feedforward(0.0, 0.0, 1.0e6)
        self.assertTrue(command.current_saturated)
        self.assertTrue(command.voltage_saturated)
        # the requested torque wants far more than the current limit, and the
        # current limit wants more than the voltage can push at standstill: the
        # voltage clamps first, and `current` is what that voltage implies
        self.assertGreater(command.current, self.model.current_limit * 0.5)
        self.assertAlmostEqual(command.voltage, self.model.voltage_limit, places=9)
        self.assertAlmostEqual(command.current,
                               self.model.voltage_limit / self.model.resistance,
                               places=6)
        clean = self.model.feedforward(60.0, 0.0, 0.0)
        self.assertFalse(clean.current_saturated)
        self.assertFalse(clean.voltage_saturated)

    def test_stiction_holds_the_shaft_below_stall(self):
        m = self.py.DriveModel(kv=self.model.kv, kt=self.model.kt,
                               resistance=self.model.resistance, inductance=0.0,
                               inertia=self.model.inertia, damping=0.0,
                               static_friction=0.05, load_static=0.02,
                               load_viscous=0.0, load_quadratic=0.0,
                               current_limit=100.0, voltage_limit=24.0,
                               gear_ratio=1.0, gear_efficiency=1.0, wheel_radius=0.0)
        # stall voltage: the point where the motor torque equals the two static
        # torques; half of it cannot break away, twice it can
        stall = (0.05 + 0.02) * m.resistance / m.kt
        hold = 0.5 * stall
        self.assertAlmostEqual(m.acceleration(0.0, hold), 0.0, places=12)
        state = self.py.State(0.0, 0.0)
        for _ in range(2000):
            state = m.step(state, hold, 0.001)
        self.assertEqual(state.w, 0.0)
        for _ in range(2000):
            state = m.step(state, 2.0 * stall, 0.001)
        self.assertGreater(state.w, 1.0)

    def test_unit_conversions(self):
        m = self.model
        self.assertAlmostEqual(m.output_speed(100.0), 100.0 * m.gear_ratio, places=12)
        self.assertAlmostEqual(m.motor_speed(100.0 * m.gear_ratio), 100.0, places=9)
        self.assertAlmostEqual(m.ground_speed(100.0),
                               100.0 * m.gear_ratio * m.wheel_radius, places=12)
        self.assertAlmostEqual(m.motor_speed_for_ground(m.ground_speed(100.0)),
                               100.0, places=9)

    def test_the_ros_core_drives_the_model_without_ros(self):
        core = self.ros.DriveModelNodeCore(model=self.model, sample_time=0.002)
        for _ in range(5000):
            voltage = core.tick(60.0)
        self.assertLessEqual(abs(voltage), self.model.voltage_limit)
        self.assertAlmostEqual(core.ff.reference, 60.0, places=6)
        # the node carries its own prediction of the shaft speed; the plant has
        # a time constant of seconds, so after 10 s it is close but not there
        target = 60.0 / self.model.gear_ratio
        self.assertGreater(core.state.w, 0.8 * target)
        self.assertLessEqual(core.state.w, target + 1.0)

    def test_the_ros_core_correction_is_bounded(self):
        corrected = self.ros.DriveModelNodeCore(model=self.model, sample_time=0.002,
                                                kp=1.0, max_correction=0.5)
        plain = self.ros.DriveModelNodeCore(model=self.model, sample_time=0.002,
                                            kp=0.0)
        # both cores run identically with no feedback, so at the comparison tick
        # the only difference between them is the P term.  A slow setpoint keeps
        # the feed-forward well clear of the voltage limit, where a saturated
        # output would hide the correction completely.
        for _ in range(300):
            corrected.tick(5.0)
            plain.tick(5.0)
        v_corrected = corrected.tick(5.0, measured=4.0)
        v_plain = plain.tick(5.0, measured=4.0)
        self.assertLess(abs(v_corrected - v_plain), self.model.voltage_limit * 0.1)
        self.assertAlmostEqual(v_corrected - v_plain, 0.5, places=9)
        # the correction is bounded, and it pushes towards the setpoint
        self.assertLessEqual(abs(v_corrected - v_plain), 0.5 + 1e-9)
        self.assertGreater(v_corrected, v_plain)


# ----------------------------------------------------------------------
# the generated C
# ----------------------------------------------------------------------
@unittest.skipUnless(_which("cc", "gcc", "clang"), "no C compiler on this box")
class CExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = _which("cc", "gcc", "clang")
        cls.dir = tempfile.mkdtemp(prefix="dt-c-")
        cls.system = presets.build("pod_2x_direct")
        export(cls.system, cls.dir, ExportOptions(dtype="f64", targets=("c",),
                                                  sample_time=0.001))
        (Path(cls.dir) / "driver.c").write_text("""
#include "drive_model.h"
#include <stdio.h>

int main(void) {
    dt_state s = {0.0, 0.0};
    const dt_scalar dt = (dt_scalar) 0.001;
    for (int k = 0; k < 3000; ++k) {
        s = dt_step(&dt_drive_model, s, (dt_scalar) 12.0, dt);
    }
    printf("%.12g %.12g %.12g\\n", (double) s.w, (double) s.i,
           (double) dt_steady_speed(&dt_drive_model, 12.0));
    return 0;
}
""", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def _build(self, source: str, binary: str, *extra: str) -> None:
        subprocess.run([self.cc, "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                        "-Wconversion", "-Werror", "-O1", *extra,
                        source, "-lm", "-o", binary],
                       cwd=self.dir, check=True, capture_output=True, text=True)

    def test_the_generated_code_compiles_without_warnings(self):
        self._build("drive_model.c", "lib_check.o", "-c")
        self._build("bench_drive_model.c", "bench", "drive_model.c")

    def test_the_c_model_integrates_like_the_python_one(self):
        self._build("driver.c", "driver", "drive_model.c")
        out = subprocess.run(["./driver"], cwd=self.dir, check=True,
                             capture_output=True, text=True).stdout.split()
        w_c, i_c, steady_c = (float(value) for value in out)
        # the reference is a fresh f64 python build of the same system
        ref_dir = tempfile.mkdtemp(prefix="dt-c-ref-")
        try:
            export(self.system, ref_dir, ExportOptions(dtype="f64", targets=("python",)))
            ref = _load_module(Path(ref_dir) / "drive_model.py", "dt_c_py_ref2")
            state = ref.State(0.0, 0.0)
            for _ in range(3000):
                state = ref.MODEL.step(state, 12.0, 0.001)
        finally:
            sys.modules.pop("dt_c_py_ref2", None)
            shutil.rmtree(ref_dir, True)
        self.assertAlmostEqual(w_c, state.w, delta=1e-9 * abs(state.w))
        self.assertAlmostEqual(i_c, state.i, delta=1e-9 * abs(state.i) + 1e-12)
        self.assertAlmostEqual(steady_c, ref.MODEL.steady_speed(12.0), delta=1e-9)

    def test_the_c_feedforward_ramps_like_the_python_one(self):
        (Path(self.dir) / "ff.c").write_text("""
#include "drive_model.h"
#include <stdio.h>

int main(void) {
    dt_ff ff;
    dt_ff_init(&ff, &dt_drive_model, (dt_scalar) 0.001);
    printf("%.12g %.12g\\n", (double) ff.speed_limit, (double) ff.accel_limit);
    for (int k = 0; k < 200; ++k) {
        dt_command c = dt_ff_update(&ff, (dt_scalar) 20.0);
        if (k == 199) {
            printf("%.12g %.12g %.12g %d\\n", (double) dt_ff_reference(&ff),
                   (double) c.voltage_V, (double) c.current_A, (int) c.saturated);
        }
    }
    dt_ff_reset(&ff, (dt_scalar) 5.0);
    printf("%.12g\\n", (double) dt_ff_reference(&ff));
    return 0;
}
""", encoding="utf-8")
        self._build("ff.c", "ff", "drive_model.c")
        out = [float(value) for value in subprocess.run(
            ["./ff"], cwd=self.dir, check=True, capture_output=True, text=True).stdout.split()]
        ref_dir = tempfile.mkdtemp(prefix="dt-ff-ref-")
        try:
            export(self.system, ref_dir, ExportOptions(dtype="f64", targets=("python",)))
            ref = _load_module(Path(ref_dir) / "drive_model.py", "dt_ff_py_ref")
            ff = ref.FeedForward(ref.MODEL, 0.001)
            command = None
            for _ in range(200):
                command = ff.update(20.0)
        finally:
            sys.modules.pop("dt_ff_py_ref", None)
            shutil.rmtree(ref_dir, True)
        # the model-derived defaults agree, and so does the ramp after 200 ticks
        self.assertAlmostEqual(out[0], ff.speed_limit, delta=1e-9 * ff.speed_limit)
        self.assertAlmostEqual(out[1], ff.accel_limit, delta=1e-9 * ff.accel_limit)
        self.assertAlmostEqual(out[2], ff.reference, delta=1e-9 * ff.reference)
        self.assertAlmostEqual(out[3], command.voltage, delta=1e-9 * command.voltage)
        self.assertAlmostEqual(out[4], command.current, delta=1e-9 * command.current)
        # the ramp is still asking for the rail at 20 rad/s, because the
        # acceleration term dominates: both agree it is saturated
        self.assertEqual(int(out[5]) & 2, 2)
        self.assertTrue(command.voltage_saturated)
        ff.reset(5.0)
        self.assertAlmostEqual(out[6], 5.0, places=9)

    def test_an_f32_build_compiles_and_runs(self):
        f32 = tempfile.mkdtemp(prefix="dt-c32-")
        try:
            export(self.system, f32, ExportOptions(dtype="f32", targets=("c",)))
            shutil.copy(Path(self.dir) / "driver.c", Path(f32) / "driver.c")
            subprocess.run([self.cc, "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                            "-Wconversion", "-Werror", "-O1", "drive_model.c",
                            "driver.c", "-lm", "-o", "driver"],
                           cwd=f32, check=True, capture_output=True, text=True)
            out = subprocess.run(["./driver"], cwd=f32, check=True,
                                 capture_output=True, text=True).stdout.split()
            self.assertAlmostEqual(float(out[2]), 214.995712, delta=1e-3)
        finally:
            shutil.rmtree(f32, True)


# ----------------------------------------------------------------------
# the generated Rust
# ----------------------------------------------------------------------
@unittest.skipUnless(_which("rustc"), "no rustc on this box")
class RustExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rustc = _which("rustc")
        cls.dir = tempfile.mkdtemp(prefix="dt-rs-")
        cls.system = presets.build("pod_2x_direct")
        export(cls.system, cls.dir, ExportOptions(dtype="f64", targets=("rust",),
                                                  sample_time=0.001))
        (Path(cls.dir) / "driver.rs").write_text("""
mod drive_model;

use drive_model::{MODEL, State};

fn main() {
    let mut s = State { w: 0.0, i: 0.0 };
    let dt: f64 = 0.001;
    for _ in 0..3000 {
        s = MODEL.step(s, 12.0, dt);
    }
    println!("{:.12} {:.12} {:.12}", s.w, s.i, MODEL.steady_speed(12.0));
}
""", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, True)

    def test_the_module_actually_builds_in_a_no_std_crate(self) -> None:
        # the banner claims `core` only, and the module carries its own square
        # root to make that claim true; this tests the claim, not the prose.
        # pub mod, because a private module makes every item look like dead code.
        (Path(self.dir) / "nostd.rs").write_text(
            '#![no_std]\n#[path = "drive_model.rs"]\npub mod drive_model;\n',
            encoding="utf-8")
        done = subprocess.run([self.rustc, "--crate-type", "lib", "--edition",
                               "2021", "-O", "nostd.rs", "-o", "nostd.rlib"],
                              cwd=self.dir, check=True, capture_output=True,
                              text=True)
        self.assertEqual(done.stderr, "")

    def test_the_rust_model_integrates_like_the_python_one(self):
        subprocess.run([self.rustc, "--edition", "2021", "-O", "-A", "dead_code",
                        "driver.rs", "-o", "driver"],
                       cwd=self.dir, check=True, capture_output=True, text=True)
        out = subprocess.run(["./driver"], cwd=self.dir, check=True,
                             capture_output=True, text=True).stdout.split()
        w_rs, i_rs, steady_rs = (float(value) for value in out)
        ref_dir = tempfile.mkdtemp(prefix="dt-rs-ref-")
        try:
            export(self.system, ref_dir, ExportOptions(dtype="f64", targets=("python",)))
            ref = _load_module(Path(ref_dir) / "drive_model.py", "dt_rs_py_ref")
            state = ref.State(0.0, 0.0)
            for _ in range(3000):
                state = ref.MODEL.step(state, 12.0, 0.001)
        finally:
            sys.modules.pop("dt_rs_py_ref", None)
            shutil.rmtree(ref_dir, True)
        self.assertAlmostEqual(w_rs, state.w, delta=1e-9 * abs(state.w))
        self.assertAlmostEqual(i_rs, state.i, delta=1e-9 * abs(state.i) + 1e-12)
        self.assertAlmostEqual(steady_rs, ref.MODEL.steady_speed(12.0), delta=1e-9)

    def test_the_generated_bench_runs(self):
        subprocess.run([self.rustc, "--edition", "2021", "-O", "-A", "dead_code",
                        "bench_drive_model.rs", "-o", "rsbench"],
                       cwd=self.dir, check=True, capture_output=True, text=True)
        out = subprocess.run(["./rsbench"], cwd=self.dir, check=True,
                             capture_output=True, text=True).stdout
        self.assertIn("motor_rad_s", out.splitlines()[0])
        self.assertIn("steady_output", out)


# ----------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------
class CliExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dt-cli-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _quiet(self, argv):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def test_export_writes_the_requested_targets(self):
        code, text = self._quiet(["export", "--preset", "spinning_mass",
                                  "--targets", "python,c", "--out", self.dir])
        self.assertEqual(code, 0)
        names = {p.name for p in Path(self.dir).iterdir()}
        self.assertIn("drive_model.py", names)
        self.assertIn("drive_model.c", names)
        self.assertNotIn("drive_model.rs", names)
        self.assertIn("drive_model_manifest.json", names)
        self.assertIn("spinning_mass", text)

    def test_export_accepts_a_fit_document(self):
        system = presets.build("pod_4x_geared")
        fit_path = Path(self.dir) / "fit.json"
        fit_path.write_text(json.dumps({
            "model": json.loads(presets.to_json(system)),
            "dataset": "bench.csv",
            "normalized_rms": 0.98,
            "warnings": ["bearing damping and load viscosity are one quantity"],
        }), encoding="utf-8")
        out = Path(self.dir) / "firmware"
        code, _ = self._quiet(["export", "--fit-json", str(fit_path),
                               "--targets", "python", "--out", str(out)])
        self.assertEqual(code, 0)
        manifest = json.loads((out / "drive_model_manifest.json").read_text())
        self.assertTrue(manifest["provenance"]["source"].endswith("fit.json"))
        self.assertEqual(manifest["provenance"]["fit"]["dataset"], "bench.csv")
        self.assertIn("one quantity", json.dumps(manifest))
        py = _load_module(out / "drive_model.py", "dt_cli_fit_model")
        self.assertAlmostEqual(py.MODEL.gear_ratio, 0.25, places=9)
        sys.modules.pop("dt_cli_fit_model", None)

    def test_set_overrides_apply_before_export(self):
        out = Path(self.dir) / "geared"
        code, _ = self._quiet(["export", "--preset", "pod_2x_direct", "--targets",
                               "python", "--out", str(out), "-s",
                               "group.motor.inductance=0.004"])
        self.assertEqual(code, 0)
        py = _load_module(out / "drive_model.py", "dt_cli_set_model")
        self.assertAlmostEqual(py.MODEL.inductance, 0.002, places=9)  # /2 motors, f32 build
        sys.modules.pop("dt_cli_set_model", None)

    def test_an_unknown_target_exits_two(self):
        code, _ = self._quiet(["export", "--preset", "legacy", "--targets", "vhdl",
                               "--out", self.dir])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
