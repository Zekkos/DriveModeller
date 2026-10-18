"""Emit the fitted drive model as standalone C, Rust and Python source.

The Python package simulates; the point of this module is to hand the *same*
model to the place where it is actually used, which for this project means an
MCU running a feed-forward controller.  The generated code has no dependency on
this repository, no dynamic allocation, no filesystem, and (for Rust) no
standard library -- it is a struct of constants and a handful of pure
functions.

What gets exported is the collapsed single-terminal representation the
simulator itself uses internally (``drivetrain.simulate._FrozenModel``):

    electrical   v = i*R + kv*w            (L = 0 -> current is algebraic)
    mechanical   J*dw/dt = kt*i - b*w - load(w) - friction(w)
    load         load(w) = sign(w)*(ts + bv*|w| + ca*w**2)   [reflected to motor]

with every quantity referred to the **motor shaft**, plus the gearbox and wheel
geometry so the caller can work in output-side or ground-side units instead.
The current limit is the total at the model's terminals; ``motor_count`` and the
per-motor limit are exported too, because a real ESC may enforce its own limit
per winding.

The feed-forward law is the reason for the exercise::

    i_ff = (J*a + b*w + load(w) + friction(w) + tau_extra) / kt
    v_ff = i_ff*R + kv*w            (+ L*di/dt when inductance is non-zero)

which is the model inverted, not tuned.  The forward integrator is exported
alongside it so the same constants can be used for observers, for
simulation-in-the-loop, and for checking that the inversion actually round-trips.

Usage::

    from drivetrain import presets
    from drivetrain.export import export, ExportOptions

    export(presets.build("pod_2x_direct"), "build/firmware",
           ExportOptions(dtype="f32", sample_time=0.001))

or from the command line::

    python -m drivetrain export --config fitted.json --out build/firmware
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Sequence

from .exceptions import ConfigError, DrivetrainError
from .pmsm import PMSMModel
from .system import DriveSystem

__all__ = ["ModelConstants", "ExportOptions", "export", "constants_from_system",
           "system_from_fit_document", "options_from_args", "TARGETS", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

TARGETS: tuple[str, ...] = ("c", "rust", "python", "ros")


# ----------------------------------------------------------------------
# the constants that leave the building
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConstants:
    """Everything the exported code needs, as plain floats.

    Units are SI throughout, referred to the motor shaft unless the name says
    otherwise.  This is deliberately a flat bag of scalars: it is what a C
    struct, a Rust struct and a Python class all map onto without ceremony.
    """

    # electrical, at the model terminals (parallel motors already collapsed)
    kv: float                 # back-EMF constant [V/(rad/s)] = 1/no-load-speed
    kt: float                 # torque constant [N*m/A]
    resistance: float         # winding resistance [ohm]
    inductance: float         # winding inductance [H]; 0 -> algebraic current

    # mechanical, referred to the motor shaft
    inertia: float            # rotors + gear + reflected load [kg*m^2]
    damping: float            # bearings + gears, viscous [N*m/(rad/s)]
    static_friction: float    # bearing/gear Coulomb torque [N*m]

    # reflected load torque at the motor shaft: sign(w)*(ts + bv|w| + ca*w^2)
    load_static: float        # [N*m]
    load_viscous: float       # [N*m/(rad/s)]
    load_quadratic: float     # [N*m/(rad/s)^2]

    # limits
    current_limit: float      # total at the terminals [A]; inf = unlimited
    voltage_limit: float      # [V]; inf = unlimited

    # geometry, for output-side and ground-side units
    gear_ratio: float         # output_speed / motor_speed
    gear_efficiency: float    # 0..1
    wheel_radius: float       # [m]; 0.0 when there is no wheel
    motor_count: int = 1

    # provenance, not used by the generated code
    notes: tuple[str, ...] = ()

    # -- derived ---------------------------------------------------------
    @property
    def electromechanical_damping(self) -> float:
        """Back-EMF feedback expressed as a viscous load [N*m/(rad/s)]."""
        return self.kt * self.kv / self.resistance if self.resistance else 0.0

    @property
    def is_second_order(self) -> bool:
        return self.inductance > 0.0

    @property
    def per_motor_current_limit(self) -> float:
        return self.current_limit / self.motor_count if math.isfinite(self.current_limit) else math.inf

    @property
    def electrical_time_constant(self) -> float:
        return self.inductance / self.resistance if self.inductance and self.resistance else 0.0

    def total_damping_at(self, w: float) -> float:
        """Small-signal damping about speed ``w``, back-EMF included."""
        return (self.damping + self.electromechanical_damping
                + self.load_viscous + 2.0 * self.load_quadratic * abs(w))

    def mech_time_constant(self, w: float = 0.0) -> float:
        b = self.total_damping_at(w)
        return self.inertia / b if b > 0.0 else math.inf

    def no_load_speed(self, voltage: float) -> float:
        return voltage / self.kv if self.kv else math.inf

    def stall_torque(self, voltage: float) -> float:
        """Locked-rotor torque at this voltage, current limit respected."""
        if not self.kt:
            return 0.0
        tau = self.kt * voltage / self.resistance if self.resistance else math.inf
        if math.isfinite(self.current_limit):
            tau = min(tau, self.kt * self.current_limit)
        return tau

    def steady_speed(self, voltage: float) -> float:
        """Speed the drive settles at under a constant voltage (0 if it cannot
        move).  Voltage-limited: it does not honour ``current_limit``, exactly
        as the generated C/Rust/Python ``steady_speed`` does."""
        sign = 1.0 if voltage >= 0.0 else -1.0
        av = abs(voltage)
        if not self.resistance > 0.0 or not self.inertia > 0.0:
            return 0.0
        drive = self.kt * av / self.resistance - self.load_static
        if drive <= 0.0:
            return 0.0
        b = self.damping + self.load_viscous + self.electromechanical_damping
        if self.load_quadratic > 0.0:
            w = 2.0 * drive / (b + math.sqrt(b * b + 4.0 * self.load_quadratic * drive))
        else:
            w = drive / b if b > 0.0 else 0.0
        return sign * w

    def steady_output_speed(self, voltage: float) -> float:
        return self.steady_speed(voltage) * self.gear_ratio

    def max_step_time(self) -> float:
        """Largest stable fixed step for the exported RK4 integrator [s].

        RK4 is stable for |pole*dt| < ~2.78.  The fastest pole is the current
        loop when inductance is non-zero, otherwise the speed loop at the top
        speed, where aerodynamic load adds the most damping.  Half of that is
        taken as the contract, so the exported model is also accurate and not
        merely stable.
        """
        if self.is_second_order:
            tau_e = self.electrical_time_constant
            if tau_e > 0.0:
                return 0.5 * tau_e
        w_ref = self.no_load_speed(self.voltage_limit if math.isfinite(self.voltage_limit)
                                   else 10.0 * self.kv * self.inertia)
        pole = self.total_damping_at(w_ref) / self.inertia if self.inertia else 0.0
        return 0.5 / pole if pole > 0.0 else math.inf

    # -- construction ----------------------------------------------------
    @classmethod
    def from_system(cls, system: DriveSystem) -> "ModelConstants":
        """Collapse a :class:`DriveSystem` the way the simulator does."""
        g = system.group
        lc = system.load_coefficients()
        if not lc.exact:
            # The polynomial is exact for every load in this package.  A custom
            # load that the fit could not represent would silently export the
            # wrong torque curve, so say so instead of hiding it.
            raise DrivetrainError(
                "this load model does not reduce to the exported torque polynomial "
                "(sign(w)*(ts + bv|w| + ca*w^2)); export a load this package can represent"
            )
        wheel = system.wheel
        return cls(
            kv=float(g.kv_eff), kt=float(g.kt_eff),
            resistance=float(g.r_eff), inductance=float(g.l_eff),
            inertia=float(system.total_inertia),
            damping=float(system.base_damping),
            static_friction=float(g.static_friction_eff),
            load_static=float(lc.static), load_viscous=float(lc.viscous),
            load_quadratic=float(lc.quadratic),
            current_limit=float(system.current_limit_total),
            voltage_limit=float(system.voltage_limit),
            gear_ratio=float(system.gearbox.ratio),
            gear_efficiency=float(system.gearbox.efficiency),
            wheel_radius=float(wheel.radius) if wheel is not None else 0.0,
            motor_count=int(g.count),
            notes=_machine_notes(system),
        )

    def as_dict(self) -> dict[str, Any]:
        """The constants plus their units, for the manifest and the docs."""
        return {
            "kv": {"value": self.kv, "unit": "V/(rad/s)"},
            "kt": {"value": self.kt, "unit": "N*m/A"},
            "resistance": {"value": self.resistance, "unit": "ohm"},
            "inductance": {"value": self.inductance, "unit": "H"},
            "inertia": {"value": self.inertia, "unit": "kg*m^2 at the motor shaft"},
            "damping": {"value": self.damping, "unit": "N*m/(rad/s)"},
            "static_friction": {"value": self.static_friction, "unit": "N*m"},
            "load_static": {"value": self.load_static, "unit": "N*m at the motor shaft"},
            "load_viscous": {"value": self.load_viscous, "unit": "N*m/(rad/s) at the motor shaft"},
            "load_quadratic": {"value": self.load_quadratic, "unit": "N*m/(rad/s)^2"},
            "current_limit": {"value": self.current_limit, "unit": "A total at the terminals"},
            "voltage_limit": {"value": self.voltage_limit, "unit": "V"},
            "gear_ratio": {"value": self.gear_ratio, "unit": "output speed / motor speed"},
            "gear_efficiency": {"value": self.gear_efficiency, "unit": "0..1"},
            "wheel_radius": {"value": self.wheel_radius, "unit": "m (0 = no wheel)"},
            "motor_count": {"value": self.motor_count, "unit": "parallel motors"},
        }

    def derived_dict(self) -> dict[str, float]:
        return {
            "electromechanical_damping_Nm_per_rad_s": self.electromechanical_damping,
            "electrical_time_constant_s": self.electrical_time_constant,
            "mech_time_constant_s": self.mech_time_constant(),
            "no_load_speed_rad_s": self.no_load_speed(self.voltage_limit),
            "no_load_speed_rpm": math.degrees(self.no_load_speed(self.voltage_limit)) * 60.0
            / (2.0 * math.pi) if math.isfinite(self.voltage_limit) else math.inf,
            "stall_torque_Nm": self.stall_torque(self.voltage_limit),
            "stall_current_A": self.current_limit if math.isfinite(self.current_limit)
            else (self.voltage_limit / self.resistance if self.resistance else math.inf),
            "max_step_time_s": self.max_step_time(),
            "output_no_load_speed_rad_s": self.no_load_speed(self.voltage_limit) * self.gear_ratio,
            "output_stall_torque_Nm": self.stall_torque(self.voltage_limit)
            * self.gear_ratio * self.gear_efficiency,
        }


def _machine_notes(system: DriveSystem) -> tuple[str, ...]:
    """Limits of the exported model that the integrator has to be told about.

    A brushless machine reaches this export through the ``i_d = 0`` collapse, and
    the collapse has two edges worth naming in the generated README: the speed
    where a dq drive can no longer hold full current (the inverter starts paying
    for the cross-coupling voltage), and base speed, above which a real drive
    weakens the field and this model simply stops.  Silence here reads as "the
    model is valid everywhere".
    """
    motor = system.group.motor
    if not isinstance(motor, PMSMModel):
        return ()
    ratio = abs(float(system.gearbox.ratio))
    spec = motor.spec
    notes = [
        f"brushless machine ({system.group.count} x {spec.pole_pairs}-pole-pair PMSM) "
        f"collapsed to its i_d=0 DC equivalent; kt/kv = 1.5 is the amplitude-invariant "
        f"Park convention, not an inconsistency between the two constants",
    ]
    base = motor.base_speed()
    notes.append(
        f"no field-weakening region: this model tops out at base speed {base:.1f} rad/s "
        f"motor side ({base * ratio:.1f} rad/s output, {spec.rpm(base):.0f} rpm); a real "
        f"FOC drive keeps making torque above that by demagnetising current this model "
        f"cannot represent"
    )
    if motor.collapse_margin() > 0.02:
        edge = motor.constant_torque_speed()
        notes.append(
            f"full current only reachable to {edge:.1f} rad/s motor side "
            f"({edge * ratio:.1f} rad/s output): above it a dq drive holds less current "
            f"than this model assumes, because it must spend part of its voltage budget "
            f"on the cross-coupling term w_e*Lq*i_q ({motor.collapse_margin():.0%} "
            f"optimistic band edge)"
        )
    if math.isfinite(spec.dc_bus_voltage):
        notes.append(
            f"voltage_V in this code is the phase peak ({spec.voltage_limit:.2f} V), not the "
            f"{spec.dc_bus_voltage:.1f} V bus: an SVPWM drive can only make bus/sqrt(3) of "
            f"fundamental phase voltage, so feeding the battery number into this model "
            f"predicts sqrt(3) too much no-load speed"
        )
        notes.append(
            f"DC-link current is 1.5*(R*i + kv*w)*i / {spec.dc_bus_voltage:.1f} V, not the "
            f"winding current: a rail ammeter reads a different number from the phase "
            f"current that makes torque"
        )
    return tuple(notes)


def constants_from_system(system: DriveSystem) -> ModelConstants:
    """Shorthand for :meth:`ModelConstants.from_system`."""
    return ModelConstants.from_system(system)


def _f32(value: float) -> float:
    """Nearest float32 to *value*; infinities pass through untouched."""
    if not math.isfinite(value):
        return float(value)
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, ValueError):
        return float(value)


def round_to_dtype(constants: ModelConstants, dtype: str) -> ModelConstants:
    """Quantise every constant to the target's float width, once, up front.

    Rounding here rather than in each literal formatter is what makes the C,
    Rust, Python and manifest agree: they are all reading the same numbers, and
    derived quantities (time constants, no-load speed, the stable step) are
    computed from the values the firmware will actually hold.
    """
    if dtype != "f32":
        return constants
    rounded = {f.name: _f32(getattr(constants, f.name))
               for f in fields(constants)
               if isinstance(getattr(constants, f.name), float)}
    return replace(constants, **rounded)


@dataclass(frozen=True)
class ExportOptions:
    """What kind of code to emit and how to shape it."""

    dtype: str = "f32"                    # "f32" or "f64"
    sample_time: float | None = None      # the control period the code is sized for
    targets: tuple[str, ...] = ("c", "rust", "python")
    module_name: str = "drive_model"
    header_prefix: str = "dt"             # C symbol prefix
    include_examples: bool = True         # a runnable step-response bench per language
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in ("f32", "f64"):
            raise ConfigError("dtype must be 'f32' or 'f64'")
        targets = tuple(self.targets)
        for t in targets:
            if t not in TARGETS:
                raise ConfigError(f"unknown export target {t!r}; expected one of {', '.join(TARGETS)}")
        object.__setattr__(self, "targets", targets)
        if self.sample_time is not None and self.sample_time <= 0:
            raise ConfigError("sample_time must be > 0")
        if not self.module_name.replace("_", "").isalnum():
            raise ConfigError("module_name must be alphanumeric/underscore")


# ----------------------------------------------------------------------
# literal formatting
# ----------------------------------------------------------------------
def _py_num(v: float, dtype: str = "f64") -> str:
    """A Python literal for ``v``, rounded to ``dtype`` like the other targets.

    Python floats are always doubles, but a build declared ``f32`` must carry
    the same *values* as the C and Rust builds: otherwise the three disagree in
    the last digits and a bench comparison looks like a bug in the generator.
    """
    if math.isinf(v):
        return 'math.inf' if v > 0 else '-math.inf'
    if math.isnan(v):
        return 'math.nan'
    value = float(v)
    if dtype == "f32":
        try:
            value = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, ValueError):
            pass
    return repr(value)


def _c_num(v: float, dtype: str) -> str:
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return ("(-INFINITY)" if v < 0 else "INFINITY")
    if dtype == "f32":
        # print the value the compiled float will actually hold, so the source
        # and the binary agree to the bit
        try:
            v = struct.unpack("f", struct.pack("f", v))[0]
        except OverflowError:      # outside float range: print it as-is
            pass
    s = repr(float(v))
    if "e" in s or "E" in s:
        mant, sep, exp = s.partition("e") if "e" in s else s.partition("E")
        if "." not in mant:
            mant += ".0"
        s = f"{mant}{sep}{exp}"
    elif "." not in s:
        s += ".0"
    return s + ("f" if dtype == "f32" else "")


def _rust_num(v: float, dtype: str) -> str:
    if math.isnan(v):
        return f"{dtype}::NAN"
    if math.isinf(v):
        return (f"-{dtype}::INFINITY" if v < 0 else f"{dtype}::INFINITY")
    text = f"{v:.17g}" if dtype == "f64" else f"{v:.9g}"
    if "." not in text and "e" not in text and "E" not in text and "inf" not in text:
        text += ".0"
    return text


def _banner(options: ExportOptions, constants: ModelConstants, style: str = "/*") -> str:
    """The provenance header every generated file carries."""
    p = dict(options.provenance or {})
    lines = [
        "Generated by drivetrain/export.py -- do not edit by hand.",
        f"source: {p.get('source', 'a DriveSystem built in code')}",
        f"float type: {options.dtype}",
    ]
    if options.sample_time:
        lines.append(f"control sample time: {options.sample_time:g} s")
    lines.append(f"stable fixed step: <= {constants.max_step_time():g} s")
    if p.get("fit"):
        fit = p["fit"]
        lines.append(f"fitted from: {fit.get('dataset', 'a dataset')} "
                     f"({fit.get('points', '?')} points, normalized rms {fit.get('normalized_rms', '?')})")
        for w in list(fit.get("warnings", ()))[:6]:
            lines.append(f"fit warning: {w}")
    for note in constants.notes:
        lines.append(f"note: {note}")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines.append(f"generated: {stamp}")
    if style == "//":
        return "\n".join(f"// {line}" for line in lines)
    if style == "#":
        return "\n".join(f"# {line}" for line in lines)
    return "/*\n" + "\n".join(f" * {line}" for line in lines) + "\n */"


# ----------------------------------------------------------------------
# the shared math, spelled out once per language
# ----------------------------------------------------------------------
# The three renderers below emit the *same* functions with the same names and
# the same order, so a control engineer can read one and recognise the others.
# Keeping them as templates rather than generating an AST keeps the output
# readable enough to audit by eye, which matters when the code goes on a
# microcontroller that a human has to sign off.

_C_HEADER = Template("""\
$banner

#ifndef ${guard}
#define ${guard}

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Width-matched helpers: a float build must not drift into libm's double
 * entry points, and INFINITY has to arrive as a dt_scalar, not a double. */
#define ${name}_SQRT ${sqrt}
#define ${name}_FABS ${fabs}
#define ${name}_INF ${inf}
/* Below this speed the Coulomb term is undefined and the stiction branch
 * holds the shaft instead; keep it well above the encoder's resolution. */
#define ${name}_STICTION_BAND ${stiction_eps}

/* ---------------------------------------------------------------------
 * Types
 * ------------------------------------------------------------------ */

typedef ${ctype} dt_scalar;

/* Every constant of the plant, referred to the motor shaft. */
typedef struct {
    /* electrical, at the terminals of the model (parallel motors collapsed) */
    dt_scalar kv;                 /* back-EMF constant  [V/(rad/s)] */
    dt_scalar kt;                 /* torque constant    [N*m/A]     */
    dt_scalar resistance;         /* winding resistance [ohm]       */
    dt_scalar inductance;         /* winding inductance [H]; 0 -> algebraic current */

    /* mechanical, at the motor shaft */
    dt_scalar inertia;            /* rotors + gear + reflected load [kg*m^2] */
    dt_scalar damping;            /* bearings + gears, viscous      [N*m/(rad/s)] */
    dt_scalar static_friction;    /* bearing/gear Coulomb torque    [N*m] */

    /* reflected load torque at the motor shaft:
       load(w) = sign(w) * (load_static + load_viscous*|w| + load_quadratic*w*w) */
    dt_scalar load_static;        /* [N*m]          */
    dt_scalar load_viscous;       /* [N*m/(rad/s)]  */
    dt_scalar load_quadratic;     /* [N*m/(rad/s)^2] */

    /* limits; INFINITY means the model does not enforce one */
    dt_scalar current_limit;      /* total at the terminals [A] */
    dt_scalar voltage_limit;      /* [V] */

    /* geometry, for output-side and ground-side units */
    dt_scalar gear_ratio;         /* output_speed / motor_speed */
    dt_scalar gear_efficiency;    /* 0..1 */
    dt_scalar wheel_radius;       /* [m]; 0 when there is no wheel */
    int motor_count;              /* parallel motors behind the terminals */
} dt_model;

/* State of the forward model.  $second_order_note */
typedef struct {
    dt_scalar w;   /* motor angular velocity [rad/s] */
    dt_scalar i;   /* terminal current [A]; only a state when inductance > 0 */
} dt_state;

/* A feed-forward result: what to apply, and what held it back. */
typedef struct {
    dt_scalar voltage_V;
    dt_scalar current_A;      /* current the voltage implies at this speed */
    dt_scalar torque_Nm;      /* motor-shaft torque actually commanded */
    int saturated;            /* bit 1: current limit, bit 2: voltage limit */
} dt_command;

/* The constants for this build. */
extern const dt_model ${name};
extern const dt_scalar ${name}_sample_time_s;   /* the control period this was sized for */
extern const dt_scalar ${name}_max_step_s;      /* largest stable fixed step */

/* ---------------------------------------------------------------------
 * Motor-side primitives
 * ------------------------------------------------------------------ */

/* Reflected load torque at motor speed w; always opposes motion. */
dt_scalar dt_load_torque(const dt_model *m, dt_scalar w);

/* Torque/current conversions (no back-EMF, no losses beyond kt). */
dt_scalar dt_torque_for_current(const dt_model *m, dt_scalar i);
dt_scalar dt_current_for_torque(const dt_model *m, dt_scalar tau);

/* Terminal current the model would draw at voltage v, speed w (clamped). */
dt_scalar dt_current_at(const dt_model *m, dt_scalar v, dt_scalar w);

/* dw/dt at speed w under applied voltage v, including stiction. */
dt_scalar dt_acceleration(const dt_model *m, dt_scalar w, dt_scalar v);

/* Voltage that produces speed w with acceleration accel, plus an optional
   extra motor-shaft torque (a known disturbance, e.g. a grade or a payload).
   This is the model inverted: i = (J*a + b*w + load(w) + friction(w) + tau) / kt,
   then v = i*R + kv*w (+ L*di/dt when inductance is non-zero). */
dt_command dt_feedforward(const dt_model *m, dt_scalar w, dt_scalar accel,
                          dt_scalar extra_torque);

/* Voltage that holds speed w indefinitely (accel = 0). */
dt_scalar dt_steady_voltage(const dt_model *m, dt_scalar w, dt_scalar extra_torque);

/* Speed the drive settles at under a constant voltage (0 if it cannot move).
   Voltage-limited answer: it does not honour current_limit, so on a stiff rig
   the real settling speed can be lower. */
dt_scalar dt_steady_speed(const dt_model *m, dt_scalar v);

/* Advance the model one fixed step under applied voltage v.
   Classical RK4; dt must be <= ${name}_max_step_s. */
dt_state dt_step(const dt_model *m, dt_state s, dt_scalar v, dt_scalar dt);

/* Voltage that would be needed to hold the current state still (useful for a
   hover/hold command); same as dt_steady_voltage at the current speed. */
static inline dt_scalar dt_hold_voltage(const dt_model *m, dt_state s) {
    return dt_steady_voltage(m, s.w, (${zero}));
}

/* ---------------------------------------------------------------------
 * Output-side and ground-side helpers
 * ------------------------------------------------------------------ */

static inline dt_scalar dt_output_speed(const dt_model *m, dt_scalar w) {
    return w * m->gear_ratio;
}

static inline dt_scalar dt_motor_speed(const dt_model *m, dt_scalar w_out) {
    return w_out / m->gear_ratio;
}

/* Motor torque -> output torque, with gear losses in the driving direction. */
static inline dt_scalar dt_output_torque(const dt_model *m, dt_scalar tau_motor) {
    if (tau_motor >= ${zero}) {
        return tau_motor * m->gear_ratio * m->gear_efficiency;
    }
    return tau_motor * m->gear_ratio / m->gear_efficiency;
}

/* Output torque demand -> motor torque, accounting for gear losses. */
static inline dt_scalar dt_motor_torque_for_output(const dt_model *m, dt_scalar tau_out) {
    if (tau_out >= ${zero}) {
        return tau_out / (m->gear_ratio * m->gear_efficiency);
    }
    return tau_out * m->gear_efficiency / m->gear_ratio;
}

/* Feed-forward in output-side units: command an output speed and
   acceleration, optionally an extra output torque, get back a voltage. */
dt_command dt_feedforward_output(const dt_model *m, dt_scalar w_out, dt_scalar accel_out,
                                 dt_scalar extra_output_torque);

/* Steady output speed at a constant voltage, and the voltage for a steady
   output speed. */
dt_scalar dt_steady_output_speed(const dt_model *m, dt_scalar v);
dt_scalar dt_steady_output_voltage(const dt_model *m, dt_scalar w_out);

/* ---------------------------------------------------------------------
 * Clocked feed-forward: rate-limit, differentiate, invert
 * ------------------------------------------------------------------ */

/* The part that is easy to get wrong, done once.  Rate-limit the commanded
   output speed, differentiate the *limited* reference (so the acceleration the
   model is asked for is the one the plant is actually given, not the one the
   caller typed), then invert.  Call dt_ff_update once per control period.
   The struct is deliberately open: override speed_limit, accel_limit or
   extra_output_torque after dt_ff_init if the model-derived defaults are not
   what your controller wants. */
typedef struct {
    const dt_model *model;
    dt_scalar sample_time;            /* control period [s] */
    dt_scalar speed_limit;            /* output rad/s; 0 = no limit */
    dt_scalar accel_limit;            /* output rad/s^2; 0 = no limit */
    dt_scalar extra_output_torque;    /* assumed disturbance [N*m at output] */
    dt_scalar reference;              /* the limited reference [output rad/s] */
} dt_ff;

/* Installs model-derived limits: the no-load speed at the voltage rail, and
   stall torque over inertia taken to the output shaft.  reference starts at 0. */
void dt_ff_init(dt_ff *ff, const dt_model *m, dt_scalar sample_time);

dt_scalar dt_ff_reference(const dt_ff *ff);
void dt_ff_reset(dt_ff *ff, dt_scalar reference);
dt_command dt_ff_update(dt_ff *ff, dt_scalar setpoint);

/* Ground speed [m/s] from motor speed; 0 when the model has no wheel. */
static inline dt_scalar dt_ground_speed(const dt_model *m, dt_scalar w) {
    return w * m->gear_ratio * m->wheel_radius;
}

/* Inverse: wheel speed -> motor speed. */
static inline dt_scalar dt_motor_speed_for_ground(const dt_model *m, dt_scalar v_ground) {
    return (m->wheel_radius > ${zero}) ? v_ground / (m->gear_ratio * m->wheel_radius) : (${zero});
}

/* ---------------------------------------------------------------------
 * Diagnostics
 * ------------------------------------------------------------------ */

/* Ideal no-load speed at voltage v [rad/s at the motor]. */
static inline dt_scalar dt_no_load_speed(const dt_model *m, dt_scalar v) {
    return (m->kv > ${zero}) ? v / m->kv : ${name}_INF;
}

/* Locked-rotor torque at voltage v, current limit respected [N*m at motor]. */
dt_scalar dt_stall_torque(const dt_model *m, dt_scalar v);

/* Small-signal damping about speed w, back-EMF feedback included. */
dt_scalar dt_total_damping(const dt_model *m, dt_scalar w);

/* Speed-loop time constant about speed w [s]. */
dt_scalar dt_mech_time_constant(const dt_model *m, dt_scalar w);

/* Largest fixed step the exported integrator can take stably [s]. */
dt_scalar dt_max_step_time(const dt_model *m);

#ifdef __cplusplus
}
#endif

#endif /* ${guard} */
""")

_C_SOURCE = Template("""\
$banner

#include "${header}"
#include <math.h>

const dt_model ${name} = {
    /* electrical */
    /*kv*/ ${kv}, /*kt*/ ${kt}, /*R*/ ${resistance}, /*L*/ ${inductance},
    /* mechanical */
    /*J*/ ${inertia}, /*b*/ ${damping}, /*static*/ ${static_friction},
    /* reflected load torque */
    /*ts*/ ${load_static}, /*bv*/ ${load_viscous}, /*ca*/ ${load_quadratic},
    /* limits */
    /*i_max*/ ${current_limit}, /*v_max*/ ${voltage_limit},
    /* geometry */
    /*ratio*/ ${gear_ratio}, /*eta*/ ${gear_efficiency}, /*wheel*/ ${wheel_radius},
    /*motors*/ ${motor_count}
};

const dt_scalar ${name}_sample_time_s = ${sample_time};
const dt_scalar ${name}_max_step_s = ${max_step};

/* Clamp to +/- limit.  Written out rather than fmin/fmax so it inlines to a
   couple of instructions on a Cortex-M with no libm in the hot path. */
static dt_scalar clamp_sym(dt_scalar x, dt_scalar limit) {
    if (!(limit > ${zero})) {          /* NaN or infinite limit: no clamping */
        return x;
    }
    if (x > limit) {
        return limit;
    }
    if (x < -limit) {
        return -limit;
    }
    return x;
}

dt_scalar dt_load_torque(const dt_model *m, dt_scalar w) {
    dt_scalar a = (w >= ${zero}) ? w : -w;
    dt_scalar tau = m->load_static + m->load_viscous * a + m->load_quadratic * a * a;
    return (w >= ${zero}) ? tau : -tau;
}

dt_scalar dt_torque_for_current(const dt_model *m, dt_scalar i) {
    return m->kt * i;
}

dt_scalar dt_current_for_torque(const dt_model *m, dt_scalar tau) {
    return (m->kt > ${zero}) ? tau / m->kt : ${zero};
}

dt_scalar dt_current_at(const dt_model *m, dt_scalar v, dt_scalar w) {
    if (!(m->resistance > ${zero})) {
        return ${name}_INF;
    }
    return clamp_sym((v - m->kv * w) / m->resistance, m->current_limit);
}

/* Net motor-shaft torque at speed w with current i, Coulomb friction and
   stiction applied the way the simulator applies them. */
static dt_scalar net_torque(const dt_model *m, dt_scalar w, dt_scalar i) {
    dt_scalar tau = m->kt * i - m->damping * w - dt_load_torque(m, w);
    if (w > ${name}_STICTION_BAND) {
        return tau - m->static_friction;
    }
    if (w < -${name}_STICTION_BAND) {
        return tau + m->static_friction;
    }
    /* near standstill: stiction can hold the shaft */
    if (tau >= -m->static_friction && tau <= m->static_friction) {
        return ${zero};
    }
    return (tau > ${zero}) ? tau - m->static_friction : tau + m->static_friction;
}

dt_scalar dt_acceleration(const dt_model *m, dt_scalar w, dt_scalar v) {
    if (!(m->inertia > ${zero})) {
        return ${zero};
    }
    return net_torque(m, w, dt_current_at(m, v, w)) / m->inertia;
}

dt_command dt_feedforward(const dt_model *m, dt_scalar w, dt_scalar accel,
                          dt_scalar extra_torque) {
    dt_command c;
    dt_scalar i = ${zero};
    if (m->kt > ${zero}) {
        dt_scalar a = (w >= ${zero}) ? w : -w;
        dt_scalar tau_load = m->load_static + m->load_viscous * a + m->load_quadratic * a * a;
        tau_load = (w >= ${zero}) ? tau_load : -tau_load;
        dt_scalar fric = ${zero};
        if (w > ${name}_STICTION_BAND) {
            fric = m->static_friction;
        } else if (w < -${name}_STICTION_BAND) {
            fric = -m->static_friction;
        }
        i = (m->inertia * accel + m->damping * w + tau_load + fric + extra_torque) / m->kt;
    }
    c.saturated = ${zero};
    if (m->current_limit > ${zero} && (i > m->current_limit || i < -m->current_limit)) {
        i = clamp_sym(i, m->current_limit);
        c.saturated |= ${sat_current};
    }
    c.current_A = i;
    c.torque_Nm = m->kt * i;
    c.voltage_V = i * m->resistance + m->kv * w;
    if (m->voltage_limit > ${zero}
            && (c.voltage_V > m->voltage_limit || c.voltage_V < -m->voltage_limit)) {
        c.voltage_V = clamp_sym(c.voltage_V, m->voltage_limit);
        c.saturated |= ${sat_voltage};
        /* the voltage that is actually reachable draws a different current */
        c.current_A = dt_current_at(m, c.voltage_V, w);
        c.torque_Nm = m->kt * c.current_A;
    }
    return c;
}

dt_scalar dt_steady_voltage(const dt_model *m, dt_scalar w, dt_scalar extra_torque) {
    dt_command c = dt_feedforward(m, w, ${zero}, extra_torque);
    return c.voltage_V;
}

dt_scalar dt_steady_speed(const dt_model *m, dt_scalar v) {
    /* kt*(v - kv*w)/R = b*w + load(w), with load odd in w, so solve for |v|
       and give the answer the sign of v. */
    dt_scalar sign = (v >= ${zero}) ? ${one} : -${one};
    dt_scalar av = (v >= ${zero}) ? v : -v;
    if (!(m->resistance > ${zero}) || !(m->inertia > ${zero})) {
        return ${zero};
    }
    dt_scalar drive = m->kt * av / m->resistance - m->load_static;
    if (drive <= ${zero}) {
        return ${zero};      /* not enough voltage to break away */
    }
    dt_scalar b = m->damping + m->load_viscous + m->kt * m->kv / m->resistance;
    dt_scalar ca = m->load_quadratic;
    dt_scalar w;
    if (ca > ${zero}) {
        /* stable form of the positive quadratic root */
        w = ${two} * drive / (b + ${name}_SQRT(b * b + ${four} * ca * drive));
    } else {
        w = (b > ${zero}) ? drive / b : ${zero};
    }
    return sign * w;
}

/* Derivatives for the second-order case, spelled out so the integrator below
   can be read against the two state equations in the header. */
static dt_scalar di_dt(const dt_model *m, dt_scalar i, dt_scalar w, dt_scalar v) {
    return (v - i * m->resistance - m->kv * w) / m->inductance;
}

static dt_scalar dw_dt(const dt_model *m, dt_scalar i, dt_scalar w) {
    return net_torque(m, w, i) / m->inertia;
}

dt_state dt_step(const dt_model *m, dt_state s, dt_scalar v, dt_scalar dt) {
    if (!(dt > ${zero})) {
        return s;
    }
    if (m->inductance > ${zero}) {
        /* second order: state is (i, w) */
        dt_scalar half = ${half} * dt;
        dt_scalar i0 = s.i, w0 = s.w;
        dt_scalar k1i = di_dt(m, i0, w0, v);
        dt_scalar k1w = dw_dt(m, i0, w0);
        dt_scalar i2 = i0 + half * k1i, w2 = w0 + half * k1w;
        dt_scalar k2i = di_dt(m, i2, w2, v);
        dt_scalar k2w = dw_dt(m, i2, w2);
        dt_scalar i3 = i0 + half * k2i, w3 = w0 + half * k2w;
        dt_scalar k3i = di_dt(m, i3, w3, v);
        dt_scalar k3w = dw_dt(m, i3, w3);
        dt_scalar i4 = i0 + dt * k3i, w4 = w0 + dt * k3w;
        dt_scalar k4i = di_dt(m, i4, w4, v);
        dt_scalar k4w = dw_dt(m, i4, w4);
        s.i = i0 + (dt / ${six}) * (k1i + ${two} * k2i + ${two} * k3i + k4i);
        s.w = w0 + (dt / ${six}) * (k1w + ${two} * k2w + ${two} * k3w + k4w);
        s.i = clamp_sym(s.i, m->current_limit);
        return s;
    }
    {
        dt_scalar half = ${half} * dt;
        dt_scalar k1 = dt_acceleration(m, s.w, v);
        dt_scalar k2 = dt_acceleration(m, s.w + half * k1, v);
        dt_scalar k3 = dt_acceleration(m, s.w + half * k2, v);
        dt_scalar k4 = dt_acceleration(m, s.w + dt * k3, v);
        s.w = s.w + (dt / ${six}) * (k1 + ${two} * k2 + ${two} * k3 + k4);
        s.i = dt_current_at(m, v, s.w);
        return s;
    }
}

dt_command dt_feedforward_output(const dt_model *m, dt_scalar w_out, dt_scalar accel_out,
                                 dt_scalar extra_output_torque) {
    dt_scalar w = dt_motor_speed(m, w_out);
    dt_scalar a = dt_motor_speed(m, accel_out);
    dt_scalar tau = dt_motor_torque_for_output(m, extra_output_torque);
    return dt_feedforward(m, w, a, tau);
}

dt_scalar dt_steady_output_speed(const dt_model *m, dt_scalar v) {
    return dt_steady_speed(m, v) * m->gear_ratio;
}

dt_scalar dt_steady_output_voltage(const dt_model *m, dt_scalar w_out) {
    return dt_steady_voltage(m, dt_motor_speed(m, w_out), ${zero});
}

static dt_scalar dt_ff_clamp(dt_scalar x, dt_scalar limit) {
    if (!(limit > ${zero})) {
        return x;                /* a limit of 0 means "do not limit" */
    }
    if (x > limit) {
        return limit;
    }
    if (x < -limit) {
        return -limit;
    }
    return x;
}

void dt_ff_init(dt_ff *ff, const dt_model *m, dt_scalar sample_time) {
    ff->model = m;
    ff->sample_time = sample_time;
    ff->speed_limit = dt_no_load_speed(m, m->voltage_limit) * m->gear_ratio;
    ff->accel_limit = (m->inertia > ${zero})
        ? dt_stall_torque(m, m->voltage_limit) * m->gear_ratio / m->inertia
        : ${name}_INF;
    ff->extra_output_torque = ${zero};
    ff->reference = ${zero};
}

dt_scalar dt_ff_reference(const dt_ff *ff) {
    return ff->reference;
}

void dt_ff_reset(dt_ff *ff, dt_scalar reference) {
    ff->reference = reference;
}

dt_command dt_ff_update(dt_ff *ff, dt_scalar setpoint) {
    dt_scalar target = dt_ff_clamp(setpoint, ff->speed_limit);
    dt_scalar max_step = ff->accel_limit * ff->sample_time;
    /* clamp the *change*, not the absolute reference: clamping the reference
       itself would pin the whole command to one sample's worth of motion */
    dt_scalar next = ff->reference + dt_ff_clamp(target - ff->reference, max_step);
    dt_scalar step = next - ff->reference;
    dt_scalar accel = (ff->sample_time > ${zero}) ? step / ff->sample_time : ${zero};
    ff->reference = next;
    return dt_feedforward_output(ff->model, next, accel, ff->extra_output_torque);
}

dt_scalar dt_stall_torque(const dt_model *m, dt_scalar v) {
    dt_scalar tau = (m->resistance > ${zero}) ? m->kt * v / m->resistance : ${name}_INF;
    if (m->current_limit > ${zero} && m->kt * m->current_limit < tau) {
        tau = m->kt * m->current_limit;
    }
    return tau;
}

dt_scalar dt_total_damping(const dt_model *m, dt_scalar w) {
    dt_scalar a = (w >= ${zero}) ? w : -w;
    return m->damping + m->load_viscous + ${two} * m->load_quadratic * a
        + m->kt * m->kv / m->resistance;
}

dt_scalar dt_mech_time_constant(const dt_model *m, dt_scalar w) {
    dt_scalar b = dt_total_damping(m, w);
    return (b > ${zero}) ? m->inertia / b : ${name}_INF;
}

dt_scalar dt_max_step_time(const dt_model *m) {
    (void)m;   /* the bound is a property of the constants, and they are fixed here */
    return ${max_step};
}
""")

_C_EXAMPLE = Template("""\
$banner

/* Step response of the exported model, printed as CSV.
 *
 *   cc -std=c11 -Wall -Wextra -I. ${file} ${source_file} -lm -o bench && ./bench
 *
 * The first column is time, then motor speed [rad/s], current [A] and applied
 * voltage [V].  It is here to be diffed against the Python model:
 *
 *   python -m drivetrain run --config <source> --engine rk4 --dt ${sample_hint} ...
 */
#include "${header}"
#include <math.h>
#include <stdio.h>

int main(void) {
    const dt_scalar dt = ${bench_dt};
    const dt_scalar duration = ${bench_duration};
    const dt_scalar v_on = ${bench_voltage};
    dt_state s;
    dt_scalar t;
    int n = 0;

    s.w = ${zero};
    s.i = ${zero};
    printf("time_s,motor_rad_s,current_A,voltage_V\\n");
    for (t = ${zero}; t <= duration + ${tiny}; t += dt) {
        dt_scalar v = (t < ${bench_step_at}) ? ${zero} : v_on;
        printf("%.6f,%.9g,%.9g,%.9g\\n", (double)t, (double)s.w, (double)s.i, (double)v);
        s = dt_step(&${name}, s, v, dt);
        if (++n > ${bench_max_steps}) {
            break;
        }
    }
    (void)dt_steady_output_speed(&${name}, v_on);
    return 0;
}
""")

_RUST = Template("""\
$banner

//! Plant model for a feed-forward controller: a struct of constants and pure
//! functions.  Uses `core` only -- no `std`, no allocation, no floating-point
//! formatting -- so it drops straight into a `#![no_std]` firmware crate.  The
//! square root is the one thing `core` does not carry, so the module brings its
//! own; see `sqrt` below.
//!
//! All quantities are SI and referred to the **motor shaft** unless the name
//! says otherwise.  Sign convention: everything is signed, and a negative
//! command drives the shaft the other way.

pub type Scalar = $dtype;

/// Saturation flags for [`Command`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Saturated {
    pub current: bool,
    pub voltage: bool,
}

/// Every constant of the plant, referred to the motor shaft.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveModel {
    /// back-EMF constant [V/(rad/s)]; no-load speed is `v / kv`
    pub kv: Scalar,
    /// torque constant [N*m/A]
    pub kt: Scalar,
    /// winding resistance [ohm], at the terminals of the model
    pub resistance: Scalar,
    /// winding inductance [H]; 0 means current is algebraic
    pub inductance: Scalar,

    /// rotors + gear + reflected load inertia [kg*m^2] at the motor shaft
    pub inertia: Scalar,
    /// bearings + gears, viscous [N*m/(rad/s)]
    pub damping: Scalar,
    /// bearing/gear Coulomb torque [N*m]
    pub static_friction: Scalar,

    /// reflected load torque at the motor shaft:
    /// `sign(w) * (load_static + load_viscous*|w| + load_quadratic*w*w)`
    pub load_static: Scalar,
    pub load_viscous: Scalar,
    pub load_quadratic: Scalar,

    /// limits; `Scalar::INFINITY` means the model does not enforce one
    pub current_limit: Scalar,
    pub voltage_limit: Scalar,

    /// output_speed / motor_speed
    pub gear_ratio: Scalar,
    /// 0..1
    pub gear_efficiency: Scalar,
    /// [m]; 0 when there is no wheel
    pub wheel_radius: Scalar,
    /// parallel motors behind the terminals
    pub motor_count: u32,
}

/// State of the forward model. $second_order_note
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct State {
    /// motor angular velocity [rad/s]
    pub w: Scalar,
    /// terminal current [A]; only a state when inductance > 0
    pub i: Scalar,
}

/// A feed-forward result: what to apply, and what held it back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Command {
    pub voltage: Scalar,
    /// current the applied voltage implies at this speed
    pub current: Scalar,
    /// motor-shaft torque actually commanded
    pub torque: Scalar,
    pub saturated: Saturated,
}

$constants

$impl

/// A clocked feed-forward controller in output-side units.
///
/// It rate-limits the commanded output speed, differentiates the limited
/// reference (so the acceleration term is the one the plant is actually asked
/// for, not the one the operator typed), and emits a voltage each tick.
#[derive(Clone, Copy, Debug)]
pub struct FeedForward {
    pub model: DriveModel,
    pub sample_time: Scalar,
    /// max commanded output speed [rad/s at the output shaft]
    pub speed_limit: Scalar,
    /// max output acceleration [rad/s^2 at the output shaft]
    pub accel_limit: Scalar,
    /// extra output torque assumed by the feed-forward [N*m]
    pub load_torque: Scalar,
    reference: Scalar,
}

impl FeedForward {
    pub fn new(model: DriveModel, sample_time: Scalar) -> Self {
        let max_a = if model.inertia > 0.0 {
            model.stall_torque(model.voltage_limit) * model.gear_ratio / model.inertia
        } else {
            Scalar::INFINITY
        };
        FeedForward {
            model,
            sample_time,
            speed_limit: model.no_load_speed(model.voltage_limit) * model.gear_ratio,
            accel_limit: max_a,
            load_torque: 0.0,
            reference: 0.0,
        }
    }

    /// The limited reference the controller is currently tracking.
    pub fn reference(&self) -> Scalar {
        self.reference
    }

    /// One control tick: command an output speed, get a voltage.
    pub fn update(&mut self, setpoint: Scalar) -> Command {
        let target = clamp(setpoint, self.speed_limit);
        let max_step = self.accel_limit * self.sample_time;
        // clamp the *change*, not the absolute reference: clamping the
        // reference itself would pin the whole command to one sample's worth
        let next = self.reference + clamp(target - self.reference, max_step);
        let step = next - self.reference;
        self.reference = next;
        let accel = if self.sample_time > 0.0 { step / self.sample_time } else { 0.0 };
        self.model.feedforward_output(next, accel, self.load_torque)
    }

    /// Reset the internal reference (e.g. after a fault or a mode change).
    pub fn reset(&mut self, reference: Scalar) {
        self.reference = reference;
    }
}

fn clamp(x: Scalar, limit: Scalar) -> Scalar {
    if !(limit > 0.0) {
        return x;
    }
    if x > limit {
        return limit;
    }
    if x < -limit {
        return -limit;
    }
    x
}

// The square root is the only thing in this module that `core` does not have:
// `sqrt` lives in `std` (or libm) precisely because a bare-metal target has to
// decide where it comes from.  Rather than make the whole module depend on a
// math library for one call, the module carries a Newton iteration seeded from
// the exponent bits.
const SQRT_SEED: u64 = 0x1FF8_0000_0000_0000;
const SQRT_ITERATIONS: u32 = 6;

/// Square root of a non-negative number.
///
/// Four iterations already reach the last bit; six are there for margin.  The
/// iteration runs in `f64` whatever `Scalar` is, so an f32 build gets an
/// f32-accurate answer from a wider intermediate and one implementation serves
/// both widths.  It is accurate to 1 ulp but it is *not* correctly rounded --
/// measured over 20k samples it differs from `f64::sqrt` in the last bit about
/// half the time -- so a firmware crate that already links libm may swap this
/// one body out and get a slightly different answer.
fn sqrt(x: Scalar) -> Scalar {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    let wide = x as f64;
    let mut y = f64::from_bits((wide.to_bits() >> 1) + SQRT_SEED);
    let mut k = 0;
    while k < SQRT_ITERATIONS {
        y = 0.5 * (y + wide / y);
        k += 1;
    }
    y as Scalar
}
""")

_RUST_IMPL = Template("""\
impl DriveModel {
    /// Reflected load torque at motor speed `w`; always opposes motion.
    pub fn load_torque(&self, w: Scalar) -> Scalar {
        let a = w.abs();
        let tau = self.load_static + self.load_viscous * a + self.load_quadratic * a * a;
        if w >= 0.0 { tau } else { -tau }
    }

    pub fn torque_for_current(&self, i: Scalar) -> Scalar {
        self.kt * i
    }

    pub fn current_for_torque(&self, tau: Scalar) -> Scalar {
        if self.kt > 0.0 { tau / self.kt } else { 0.0 }
    }

    /// Terminal current the model would draw at `v`, `w` (clamped).
    pub fn current_at(&self, v: Scalar, w: Scalar) -> Scalar {
        if !(self.resistance > 0.0) {
            return Scalar::INFINITY;
        }
        clamp((v - self.kv * w) / self.resistance, self.current_limit)
    }

    /// Net motor-shaft torque at speed `w` with current `i`.
    pub fn net_torque(&self, w: Scalar, i: Scalar) -> Scalar {
        let tau = self.kt * i - self.damping * w - self.load_torque(w);
        if w > STICTION_EPS {
            return tau - self.static_friction;
        }
        if w < -STICTION_EPS {
            return tau + self.static_friction;
        }
        if tau >= -self.static_friction && tau <= self.static_friction {
            return 0.0;
        }
        if tau > 0.0 { tau - self.static_friction } else { tau + self.static_friction }
    }

    /// `dw/dt` at speed `w` under applied voltage `v`.
    pub fn acceleration(&self, w: Scalar, v: Scalar) -> Scalar {
        if !(self.inertia > 0.0) {
            return 0.0;
        }
        self.net_torque(w, self.current_at(v, w)) / self.inertia
    }

    /// Voltage that produces speed `w` with acceleration `accel`, plus an
    /// optional extra motor-shaft torque.  This is the model inverted:
    /// `i = (J*a + b*w + load(w) + friction(w) + tau) / kt`, then
    /// `v = i*R + kv*w`.
    pub fn feedforward(&self, w: Scalar, accel: Scalar, extra_torque: Scalar) -> Command {
        let mut i: Scalar = 0.0;
        if self.kt > 0.0 {
            let a = w.abs();
            let tl = self.load_static + self.load_viscous * a + self.load_quadratic * a * a;
            let tau_load = if w >= 0.0 { tl } else { -tl };
            let fric = if w > STICTION_EPS {
                self.static_friction
            } else if w < -STICTION_EPS {
                -self.static_friction
            } else {
                0.0
            };
            i = (self.inertia * accel + self.damping * w + tau_load + fric + extra_torque) / self.kt;
        }
        let mut saturated = Saturated::default();
        if self.current_limit > 0.0 && (i > self.current_limit || i < -self.current_limit) {
            i = clamp(i, self.current_limit);
            saturated.current = true;
        }
        let mut voltage = i * self.resistance + self.kv * w;
        let mut current = i;
        let mut torque = self.kt * i;
        if self.voltage_limit > 0.0
            && (voltage > self.voltage_limit || voltage < -self.voltage_limit)
        {
            voltage = clamp(voltage, self.voltage_limit);
            saturated.voltage = true;
            current = self.current_at(voltage, w);
            torque = self.kt * current;
        }
        Command { voltage, current, torque, saturated }
    }

    /// Voltage that holds speed `w` indefinitely.
    pub fn steady_voltage(&self, w: Scalar, extra_torque: Scalar) -> Scalar {
        self.feedforward(w, 0.0, extra_torque).voltage
    }

    /// Speed the drive settles at under a constant voltage (0 if it cannot move).
    /// Voltage-limited: `current_limit` is not honoured here.
    pub fn steady_speed(&self, v: Scalar) -> Scalar {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let av = v.abs();
        if !(self.resistance > 0.0) || !(self.inertia > 0.0) {
            return 0.0;
        }
        let drive = self.kt * av / self.resistance - self.load_static;
        if drive <= 0.0 {
            return 0.0;
        }
        let b = self.damping + self.load_viscous + self.kt * self.kv / self.resistance;
        let ca = self.load_quadratic;
        let w = if ca > 0.0 {
            2.0 * drive / (b + sqrt(b * b + 4.0 * ca * drive))
        } else if b > 0.0 {
            drive / b
        } else {
            0.0
        };
        sign * w
    }

    /// Advance the model one fixed step under applied voltage `v` (RK4).
    pub fn step(&self, s: State, v: Scalar, dt: Scalar) -> State {
        if !(dt > 0.0) {
            return s;
        }
        if self.inductance > 0.0 {
            let half = 0.5 * dt;
            let (i0, w0) = (s.i, s.w);
            let mut di = [0.0; 4];
            let mut dw = [0.0; 4];
            for k in 0..4 {
                let (i, w) = match k {
                    0 => (i0, w0),
                    1 | 2 => (i0 + half * di[k - 1], w0 + half * dw[k - 1]),
                    _ => (i0 + dt * di[2], w0 + dt * dw[2]),
                };
                di[k] = (v - i * self.resistance - self.kv * w) / self.inductance;
                dw[k] = self.net_torque(w, i) / self.inertia;
            }
            let s6 = dt / 6.0;
            State {
                i: clamp(i0 + s6 * (di[0] + 2.0 * di[1] + 2.0 * di[2] + di[3]), self.current_limit),
                w: w0 + s6 * (dw[0] + 2.0 * dw[1] + 2.0 * dw[2] + dw[3]),
            }
        } else {
            let half = 0.5 * dt;
            let k1 = self.acceleration(s.w, v);
            let k2 = self.acceleration(s.w + half * k1, v);
            let k3 = self.acceleration(s.w + half * k2, v);
            let k4 = self.acceleration(s.w + dt * k3, v);
            let w = s.w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            State { w, i: self.current_at(v, w) }
        }
    }

    // -- output side ---------------------------------------------------

    pub fn output_speed(&self, w: Scalar) -> Scalar {
        w * self.gear_ratio
    }

    pub fn motor_speed(&self, w_out: Scalar) -> Scalar {
        w_out / self.gear_ratio
    }

    /// Motor torque -> output torque, gear losses in the driving direction.
    pub fn output_torque(&self, tau_motor: Scalar) -> Scalar {
        if tau_motor >= 0.0 {
            tau_motor * self.gear_ratio * self.gear_efficiency
        } else {
            tau_motor * self.gear_ratio / self.gear_efficiency
        }
    }

    /// Output torque demand -> motor torque, accounting for gear losses.
    pub fn motor_torque_for_output(&self, tau_out: Scalar) -> Scalar {
        if tau_out >= 0.0 {
            tau_out / (self.gear_ratio * self.gear_efficiency)
        } else {
            tau_out * self.gear_efficiency / self.gear_ratio
        }
    }

    /// Feed-forward in output-side units.
    pub fn feedforward_output(&self, w_out: Scalar, accel_out: Scalar,
                              extra_output_torque: Scalar) -> Command {
        self.feedforward(
            self.motor_speed(w_out),
            self.motor_speed(accel_out),
            self.motor_torque_for_output(extra_output_torque),
        )
    }

    pub fn steady_output_speed(&self, v: Scalar) -> Scalar {
        self.steady_speed(v) * self.gear_ratio
    }

    pub fn steady_output_voltage(&self, w_out: Scalar) -> Scalar {
        self.steady_voltage(self.motor_speed(w_out), 0.0)
    }

    /// Ground speed [m/s]; 0 when the model has no wheel.
    pub fn ground_speed(&self, w: Scalar) -> Scalar {
        w * self.gear_ratio * self.wheel_radius
    }

    pub fn motor_speed_for_ground(&self, v_ground: Scalar) -> Scalar {
        if self.wheel_radius > 0.0 {
            v_ground / (self.gear_ratio * self.wheel_radius)
        } else {
            0.0
        }
    }

    // -- diagnostics ---------------------------------------------------

    pub fn no_load_speed(&self, v: Scalar) -> Scalar {
        if self.kv > 0.0 { v / self.kv } else { Scalar::INFINITY }
    }

    pub fn stall_torque(&self, v: Scalar) -> Scalar {
        let mut tau = if self.resistance > 0.0 {
            self.kt * v / self.resistance
        } else {
            Scalar::INFINITY
        };
        if self.current_limit > 0.0 && self.kt * self.current_limit < tau {
            tau = self.kt * self.current_limit;
        }
        tau
    }

    pub fn total_damping(&self, w: Scalar) -> Scalar {
        self.damping + self.load_viscous + 2.0 * self.load_quadratic * w.abs()
            + self.kt * self.kv / self.resistance
    }

    pub fn mech_time_constant(&self, w: Scalar) -> Scalar {
        let b = self.total_damping(w);
        if b > 0.0 { self.inertia / b } else { Scalar::INFINITY }
    }

    pub fn max_step_time(&self) -> Scalar {
        ${max_step}
    }
}

/// Speed below which Coulomb stiction can hold the shaft [rad/s].
pub const STICTION_EPS: Scalar = ${stiction_eps};
""")

_RUST_BENCH = Template("""\
$banner

//! Step response of the exported model, printed as CSV.
//!
//!   rustc -O bench.rs -o bench && ./bench
//!
//! `drive_model.rs` must sit next to this file.  Columns: time, motor speed
//! [rad/s], current [A], applied voltage [V].

use core::fmt::Write;
use std::io::{self, Write as _};

mod drive_model;
use drive_model::{DriveModel, State, MODEL, SAMPLE_TIME_S, MAX_STEP_S};

fn main() {
    let dt: f64 = ${bench_dt};
    let duration: f64 = ${bench_duration};
    let v_on: f64 = ${bench_voltage};
    let step_at: f64 = ${bench_step_at};
    let mut s = State { w: 0.0, i: 0.0 };
    let model: DriveModel = MODEL;
    let mut out = String::new();
    let mut t = 0.0f64;
    let mut n = 0usize;
    out.push_str("time_s,motor_rad_s,current_A,voltage_V\\n");
    while t <= duration + 1e-12 && n < ${bench_max_steps} {
        let v = if t < step_at { 0.0 } else { v_on };
        let _ = write!(out, "{:.6},{:.9},{:.9},{:.9}\\n", t, s.w, s.i, v);
        s = model.step(s, v as ${dtype}, dt as ${dtype});
        t += dt;
        n += 1;
    }
    let _ = io::stdout().write_all(out.as_bytes());
    let _ = write!(io::stdout(), "# sample_time {}s max_step {}s steady_output {:.6} rad/s\\n",
                   SAMPLE_TIME_S, MAX_STEP_S, model.steady_output_speed(v_on as ${dtype}));
}
""")

_PYTHON = Template('''\
$banner

"""Standalone drive model for a feed-forward controller.

Generated from the fitted/selected configuration; no dependency on the
``drivetrain`` package, no numpy, no allocation in the hot path.  All
quantities are SI and referred to the motor shaft unless the name says
otherwise.  The same functions exist in the generated C and Rust, with the
same names.

    m = MODEL
    m.feedforward(w=120.0, accel=40.0)        -> Command(voltage=..., ...)
    m.step(state, voltage, dt)                -> new state
"""

import math

STICTION_EPS = ${stiction_eps}   # rad/s: below this, stiction can hold the shaft


class Command:
    """A feed-forward result: what to apply, and what held it back."""

    __slots__ = ("voltage", "current", "torque", "current_saturated", "voltage_saturated")

    def __init__(self, voltage, current, torque, current_saturated=False,
                 voltage_saturated=False):
        self.voltage = voltage
        self.current = current
        self.torque = torque
        self.current_saturated = current_saturated
        self.voltage_saturated = voltage_saturated

    def __repr__(self):
        flags = "".join(("c" if self.current_saturated else "")
                        + ("v" if self.voltage_saturated else ""))
        return (f"Command(voltage={self.voltage:.6g} V, current={self.current:.6g} A, "
                f"torque={self.torque:.6g} N*m{', saturated ' + flags if flags else ''})")


class State:
    """Forward-model state. $second_order_note"""

    __slots__ = ("w", "i")

    def __init__(self, w=0.0, i=0.0):
        self.w = w
        self.i = i

    def __repr__(self):
        return f"State(w={self.w:.6g}, i={self.i:.6g})"


def _clamp(x, limit):
    if not limit > 0.0:          # NaN or infinite limit: no clamping
        return x
    if x > limit:
        return limit
    if x < -limit:
        return -limit
    return x


class DriveModel:
    """The plant, as constants and pure functions."""

    __slots__ = ("kv", "kt", "resistance", "inductance", "inertia", "damping",
                 "static_friction", "load_static", "load_viscous", "load_quadratic",
                 "current_limit", "voltage_limit", "gear_ratio", "gear_efficiency",
                 "wheel_radius", "motor_count", "sample_time", "max_step")

    def __init__(self, kv, kt, resistance, inductance, inertia, damping,
                 static_friction, load_static, load_viscous, load_quadratic,
                 current_limit, voltage_limit, gear_ratio, gear_efficiency,
                 wheel_radius, motor_count=1, sample_time=None, max_step=None):
        self.kv = kv
        self.kt = kt
        self.resistance = resistance
        self.inductance = inductance
        self.inertia = inertia
        self.damping = damping
        self.static_friction = static_friction
        self.load_static = load_static
        self.load_viscous = load_viscous
        self.load_quadratic = load_quadratic
        self.current_limit = current_limit
        self.voltage_limit = voltage_limit
        self.gear_ratio = gear_ratio
        self.gear_efficiency = gear_efficiency
        self.wheel_radius = wheel_radius
        self.motor_count = motor_count
        self.sample_time = sample_time
        self.max_step = max_step

    # -- motor side ----------------------------------------------------
    def load_torque(self, w):
        """Reflected load torque at motor speed ``w``; always opposes motion."""
        a = abs(w)
        tau = self.load_static + self.load_viscous * a + self.load_quadratic * a * a
        return tau if w >= 0.0 else -tau

    def torque_for_current(self, i):
        return self.kt * i

    def current_for_torque(self, tau):
        return tau / self.kt if self.kt > 0.0 else 0.0

    def current_at(self, v, w):
        """Terminal current the model would draw at ``v``, ``w`` (clamped)."""
        if not self.resistance > 0.0:
            return math.inf
        return _clamp((v - self.kv * w) / self.resistance, self.current_limit)

    def net_torque(self, w, i):
        """Motor-shaft torque at speed ``w`` with current ``i``, friction included."""
        tau = self.kt * i - self.damping * w - self.load_torque(w)
        if w > STICTION_EPS:
            return tau - self.static_friction
        if w < -STICTION_EPS:
            return tau + self.static_friction
        if -self.static_friction <= tau <= self.static_friction:
            return 0.0
        return tau - self.static_friction if tau > 0.0 else tau + self.static_friction

    def acceleration(self, w, v):
        """``dw/dt`` at speed ``w`` under applied voltage ``v``."""
        if not self.inertia > 0.0:
            return 0.0
        return self.net_torque(w, self.current_at(v, w)) / self.inertia

    def feedforward(self, w, accel=0.0, extra_torque=0.0):
        """Voltage that produces speed ``w`` with acceleration ``accel``.

        The model inverted: ``i = (J*a + b*w + load(w) + friction(w) + tau) / kt``,
        then ``v = i*R + kv*w``.  ``extra_torque`` is a known disturbance
        referred to the motor shaft.
        """
        i = 0.0
        if self.kt > 0.0:
            a = abs(w)
            tl = self.load_static + self.load_viscous * a + self.load_quadratic * a * a
            tau_load = tl if w >= 0.0 else -tl
            if w > STICTION_EPS:
                fric = self.static_friction
            elif w < -STICTION_EPS:
                fric = -self.static_friction
            else:
                fric = 0.0
            i = (self.inertia * accel + self.damping * w + tau_load + fric
                 + extra_torque) / self.kt
        current_saturated = False
        if self.current_limit > 0.0 and not (-self.current_limit <= i <= self.current_limit):
            i = _clamp(i, self.current_limit)
            current_saturated = True
        voltage = i * self.resistance + self.kv * w
        current, torque = i, self.kt * i
        voltage_saturated = False
        if self.voltage_limit > 0.0 and not (-self.voltage_limit <= voltage <= self.voltage_limit):
            voltage = _clamp(voltage, self.voltage_limit)
            voltage_saturated = True
            current = self.current_at(voltage, w)
            torque = self.kt * current
        return Command(voltage, current, torque, current_saturated, voltage_saturated)

    def steady_voltage(self, w, extra_torque=0.0):
        return self.feedforward(w, 0.0, extra_torque).voltage

    def steady_speed(self, v):
        """Speed the drive settles at under a constant voltage (0 if it cannot
        move).  Voltage-limited: current_limit is not honoured here."""
        sign = 1.0 if v >= 0.0 else -1.0
        av = abs(v)
        if not self.resistance > 0.0 or not self.inertia > 0.0:
            return 0.0
        drive = self.kt * av / self.resistance - self.load_static
        if drive <= 0.0:
            return 0.0
        b = self.damping + self.load_viscous + self.kt * self.kv / self.resistance
        ca = self.load_quadratic
        if ca > 0.0:
            w = 2.0 * drive / (b + math.sqrt(b * b + 4.0 * ca * drive))
        else:
            w = drive / b if b > 0.0 else 0.0
        return sign * w

    def step(self, state, v, dt):
        """Advance the model one fixed step under applied voltage ``v`` (RK4)."""
        if not dt > 0.0:
            return State(state.w, state.i)
        if self.inductance > 0.0:
            half, i0, w0 = 0.5 * dt, state.i, state.w
            di, dw = [0.0] * 4, [0.0] * 4
            for k in range(4):
                if k == 0:
                    i, w = i0, w0
                elif k in (1, 2):
                    i, w = i0 + half * di[k - 1], w0 + half * dw[k - 1]
                else:
                    i, w = i0 + dt * di[2], w0 + dt * dw[2]
                di[k] = (v - i * self.resistance - self.kv * w) / self.inductance
                dw[k] = self.net_torque(w, i) / self.inertia
            s6 = dt / 6.0
            return State(
                w0 + s6 * (dw[0] + 2.0 * dw[1] + 2.0 * dw[2] + dw[3]),
                _clamp(i0 + s6 * (di[0] + 2.0 * di[1] + 2.0 * di[2] + di[3]), self.current_limit),
            )
        half = 0.5 * dt
        k1 = self.acceleration(state.w, v)
        k2 = self.acceleration(state.w + half * k1, v)
        k3 = self.acceleration(state.w + half * k2, v)
        k4 = self.acceleration(state.w + dt * k3, v)
        w = state.w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return State(w, self.current_at(v, w))

    # -- output side ---------------------------------------------------
    def output_speed(self, w):
        return w * self.gear_ratio

    def motor_speed(self, w_out):
        return w_out / self.gear_ratio

    def output_torque(self, tau_motor):
        """Motor torque -> output torque, gear losses in the driving direction."""
        if tau_motor >= 0.0:
            return tau_motor * self.gear_ratio * self.gear_efficiency
        return tau_motor * self.gear_ratio / self.gear_efficiency

    def motor_torque_for_output(self, tau_out):
        if tau_out >= 0.0:
            return tau_out / (self.gear_ratio * self.gear_efficiency)
        return tau_out * self.gear_efficiency / self.gear_ratio

    def feedforward_output(self, w_out, accel_out=0.0, extra_output_torque=0.0):
        """Feed-forward in output-side units."""
        return self.feedforward(self.motor_speed(w_out), self.motor_speed(accel_out),
                                self.motor_torque_for_output(extra_output_torque))

    def steady_output_speed(self, v):
        return self.steady_speed(v) * self.gear_ratio

    def steady_output_voltage(self, w_out):
        return self.steady_voltage(self.motor_speed(w_out))

    def ground_speed(self, w):
        return w * self.gear_ratio * self.wheel_radius

    def motor_speed_for_ground(self, v_ground):
        if self.wheel_radius > 0.0:
            return v_ground / (self.gear_ratio * self.wheel_radius)
        return 0.0

    # -- diagnostics ---------------------------------------------------
    def no_load_speed(self, v):
        return v / self.kv if self.kv > 0.0 else math.inf

    def stall_torque(self, v):
        tau = self.kt * v / self.resistance if self.resistance > 0.0 else math.inf
        if self.current_limit > 0.0 and self.kt * self.current_limit < tau:
            tau = self.kt * self.current_limit
        return tau

    def total_damping(self, w):
        return (self.damping + self.load_viscous + 2.0 * self.load_quadratic * abs(w)
                + self.kt * self.kv / self.resistance)

    def mech_time_constant(self, w=0.0):
        b = self.total_damping(w)
        return self.inertia / b if b > 0.0 else math.inf

    def max_step_time(self):
        return ${max_step}

    def as_dict(self):
        """The constants as a plain dict (round-trips through ``from_dict``)."""
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        return cls(**{name: d[name] for name in cls.__slots__ if name in d})


MODEL = DriveModel(
    kv=${kv}, kt=${kt}, resistance=${resistance}, inductance=${inductance},
    inertia=${inertia}, damping=${damping}, static_friction=${static_friction},
    load_static=${load_static}, load_viscous=${load_viscous},
    load_quadratic=${load_quadratic},
    current_limit=${current_limit}, voltage_limit=${voltage_limit},
    gear_ratio=${gear_ratio}, gear_efficiency=${gear_efficiency},
    wheel_radius=${wheel_radius}, motor_count=${motor_count},
    sample_time=${sample_time}, max_step=${max_step},
)

SAMPLE_TIME_S = ${sample_time}
MAX_STEP_S = ${max_step}


class FeedForward:
    """A clocked feed-forward controller in output-side units.

    Rate-limits the commanded output speed, differentiates the *limited*
    reference (so the acceleration term is what the plant is actually asked
    for, not what the operator typed), and emits a voltage each tick.
    """

    __slots__ = ("model", "sample_time", "speed_limit", "accel_limit",
                 "load_torque", "_reference")

    def __init__(self, model, sample_time, speed_limit=None, accel_limit=None,
                 load_torque=0.0):
        self.model = model
        self.sample_time = sample_time
        self.speed_limit = (model.no_load_speed(model.voltage_limit) * model.gear_ratio
                            if speed_limit is None else speed_limit)
        self.accel_limit = (model.stall_torque(model.voltage_limit) * model.gear_ratio
                            / model.inertia if accel_limit is None else accel_limit)
        self.load_torque = load_torque
        self._reference = 0.0

    @property
    def reference(self):
        """The limited reference the controller is currently tracking."""
        return self._reference

    def reset(self, reference=0.0):
        self._reference = reference

    def update(self, setpoint):
        """One control tick: command an output speed [rad/s], get a voltage."""
        target = _clamp(setpoint, self.speed_limit)
        max_step = self.accel_limit * self.sample_time
        # clamp the *change*, not the absolute reference: clamping the
        # reference itself would pin the whole command to one sample's worth
        nxt = self._reference + _clamp(target - self._reference, max_step)
        step = nxt - self._reference
        self._reference = nxt
        accel = step / self.sample_time if self.sample_time > 0.0 else 0.0
        return self.model.feedforward_output(nxt, accel, self.load_torque)
''')

_PYTHON_BENCH = Template('''\
$banner

"""Step response of the exported model, printed as CSV.

    python3 ${file}

Columns: time, motor speed [rad/s], current [A], applied voltage [V].
"""

from drive_model import MODEL, State

DT = ${bench_dt}
DURATION = ${bench_duration}
V_ON = ${bench_voltage}
STEP_AT = ${bench_step_at}


def main():
    state = State(0.0, 0.0)
    print("time_s,motor_rad_s,current_A,voltage_V")
    t = 0.0
    n = 0
    while t <= DURATION + 1e-12 and n < ${bench_max_steps}:
        v = 0.0 if t < STEP_AT else V_ON
        print(f"{t:.6f},{state.w:.9g},{state.i:.9g},{v:.9g}")
        state = MODEL.step(state, v, DT)
        t += DT
        n += 1
    print(f"# steady output speed at {V_ON:g} V: "
          f"{MODEL.steady_output_speed(V_ON):.6f} rad/s")


if __name__ == "__main__":
    main()
''')

_ROS = Template('''\
$banner

"""ROS 2 node that turns a velocity command into a voltage using the model.

The control law lives in ``drive_model.FeedForward`` and has no ROS
dependency, so it can be unit-tested and reused on a bare MCU.  This file is
only the I/O glue: it subscribes to a velocity setpoint and publishes a
command, with an optional proportional correction on measured speed.

rclpy is imported lazily, so importing this module without ROS still gives
you ``DriveModelNodeCore`` (the same logic, clock-driven, no ROS).

    ros2 run <pkg> drive_model --ros-args -p input_topic:=cmd_vel_out
"""

import math

from drive_model import MODEL, FeedForward, State

try:                                            # ROS is optional
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, TwistStamped
    from std_msgs.msg import Float64, String
    _HAS_ROS = True
except ImportError:                             # pragma: no cover
    rclpy = None
    Node = object
    _HAS_ROS = False


class DriveModelNodeCore:
    """ROS-free control loop: setpoint [rad/s at the output] -> voltage [V].

    Drive it yourself with ``tick(setpoint, measured, dt)`` if you have your
    own scheduler; the ROS node below is a thin wrapper around this.
    """

    def __init__(self, model=MODEL, sample_time=${node_sample_time},
                 kp=${node_kp}, max_correction=${node_max_correction},
                 use_feedback=True, wheel_radius=None):
        self.model = model
        self.ff = FeedForward(model, sample_time)
        self.kp = kp
        self.max_correction = max_correction
        self.use_feedback = use_feedback
        self.state = State(0.0, 0.0)
        self.measured = 0.0
        self.wheel_radius = wheel_radius if wheel_radius is not None else model.wheel_radius
        self.last_voltage = 0.0

    def set_load_torque(self, tau_output_Nm):
        """Tell the feed-forward about a known load at the output shaft."""
        self.ff.load_torque = tau_output_Nm

    def tick(self, setpoint, measured=None):
        """One sample.  ``setpoint`` is output speed [rad/s] (or ground speed
        [m/s] when the model has a wheel and ``setpoint_is_ground_speed``)."""
        cmd = self.ff.update(setpoint)
        correction = 0.0
        if self.use_feedback and measured is not None and self.kp != 0.0:
            self.measured = measured
            error = self.ff.reference - measured
            correction = self.kp * error
            if self.max_correction > 0.0:
                correction = max(-self.max_correction, min(self.max_correction, correction))
        voltage = cmd.voltage + correction
        limit = self.model.voltage_limit
        if limit > 0.0:
            voltage = max(-limit, min(limit, voltage))
        self.last_voltage = voltage
        # integrate the model alongside, so an observer has a prediction to compare to
        self.state = self.model.step(self.state, voltage, self.ff.sample_time)
        return voltage


$node_class
''')

_ROS_NODE = Template('''\
class DriveModelNode(Node):
    """Thin rclpy wrapper around :class:`DriveModelNodeCore`.

    Parameters (``--ros-args -p name:=value``):

      input_topic        velocity setpoint topic (TwistStamped, Twist, or Float64)
      input_axis         'x' or 'yaw' when the input is a Twist
      output_topic       command topic published by this node
      output_type        'voltage' (Float64) or 'cmd_vel' (TwistStamped)
      sample_time_s      control period
      kp                 proportional speed correction gain (0 disables it)
      max_correction_v   cap on that correction
      use_feedback       subscribe to ``feedback_topic`` and close the loop
      speed_limit_rad_s  clamp on the commanded output speed (0 = model limit)
      accel_limit_rad_s2 clamp on output acceleration (0 = model limit)
    """

    def __init__(self):
        super().__init__("drive_model")
        self.declare_parameter("input_topic", "cmd_vel")
        self.declare_parameter("input_axis", "x")
        self.declare_parameter("output_topic", "drive_command")
        self.declare_parameter("output_type", "voltage")
        self.declare_parameter("sample_time_s", ${node_sample_time})
        self.declare_parameter("kp", ${node_kp})
        self.declare_parameter("max_correction_v", ${node_max_correction})
        self.declare_parameter("use_feedback", False)
        self.declare_parameter("feedback_topic", "wheel_speed")
        self.declare_parameter("speed_limit_rad_s", 0.0)
        self.declare_parameter("accel_limit_rad_s2", 0.0)

        p = lambda name: self.get_parameter(name).value
        self.core = DriveModelNodeCore(
            sample_time=float(p("sample_time_s")), kp=float(p("kp")),
            max_correction=float(p("max_correction_v")), use_feedback=bool(p("use_feedback")),
        )
        if p("speed_limit_rad_s"):
            self.core.ff.speed_limit = float(p("speed_limit_rad_s"))
        if p("accel_limit_rad_s2"):
            self.core.ff.accel_limit = float(p("accel_limit_rad_s2"))

        self._setpoint = 0.0
        sub = p("input_topic")
        self.create_subscription(TwistStamped, sub, self._on_twist_stamped, 10)
        self.create_subscription(Twist, sub, self._on_twist, 10)
        self.create_subscription(Float64, sub, self._on_float, 10)
        if self.core.use_feedback:
            self.create_subscription(Float64, p("feedback_topic"), self._on_feedback, 10)

        if p("output_type") == "voltage":
            self._pub = self.create_publisher(Float64, p("output_topic"), 10)
            self._publish = self._publish_voltage
        else:
            self._pub = self.create_publisher(TwistStamped, p("output_topic"), 10)
            self._publish = self._publish_twist

        self.create_timer(self.core.ff.sample_time, self._tick)
        self.get_logger().info(
            f"drive model: kv={MODEL.kv:.6g} kt={MODEL.kt:.6g} R={MODEL.resistance:.6g} "
            f"J={MODEL.inertia:.6g} b={MODEL.damping:.6g} ratio={MODEL.gear_ratio:.6g}; "
            f"{sub} -> {p('output_topic')} every {self.core.ff.sample_time*1000:.1f} ms"
        )

    # -- input ---------------------------------------------------------
    def _to_output_speed(self, linear_x, angular_z):
        axis = self.get_parameter("input_axis").value
        if axis == "yaw":
            return angular_z / MODEL.gear_ratio if MODEL.gear_ratio else 0.0
        if MODEL.wheel_radius > 0.0:
            return linear_x / MODEL.wheel_radius      # ground speed -> wheel speed
        return linear_x

    def _on_twist_stamped(self, msg):
        self._setpoint = self._to_output_speed(msg.twist.linear.x, msg.twist.angular.z)

    def _on_twist(self, msg):
        self._setpoint = self._to_output_speed(msg.linear.x, msg.angular.z)

    def _on_float(self, msg):
        self._setpoint = float(msg.data)

    def _on_feedback(self, msg):
        self.core.measured = float(msg.data)

    # -- output --------------------------------------------------------
    def _tick(self):
        measured = self.core.measured if self.core.use_feedback else None
        v = self.core.tick(self._setpoint, measured)
        if self._publish is self._publish_voltage:
            out = Float64()
            out.data = float(v)
        else:
            out = TwistStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.twist.linear.x = float(v)
        self._pub.publish(out)

    def _publish_voltage(self, msg):
        self._pub.publish(msg)

    def _publish_twist(self, msg):
        self._pub.publish(msg)


def main(args=None):
    if not _HAS_ROS:
        raise SystemExit("rclpy is not available; use DriveModelNodeCore directly, "
                         "or run this under ROS 2")
    rclpy.init(args=args)
    node = DriveModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
''')


_ROS_BENCH = Template('''\
$banner

"""Drive the ROS node core without ROS: a ramped setpoint through the
feed-forward, printed as CSV.

    python3 bench_${module_name}_ros.py

The node core is deliberately ROS-free (``rclpy`` is optional in
${node_module}.py), so this runs anywhere and doubles as the reference for
what the node should publish.  Columns: time, setpoint, published voltage,
predicted output speed [rad/s], predicted current [A].
"""

from ${node_module} import DriveModelNodeCore


def main():
    core = DriveModelNodeCore(sample_time=${node_sample_time})
    dt = ${node_sample_time}
    duration = ${bench_duration}
    step_at = ${bench_step_at}
    target = ${steady_speed}          # rad/s at the output shaft
    t = 0.0
    print("time_s,setpoint_rad_s,voltage_V,predicted_output_rad_s,current_A")
    while t <= duration:
        setpoint = target if t >= step_at else 0.0
        v = core.tick(setpoint)
        w_out = core.state.w * core.model.gear_ratio
        print(f"{t:.6f},{setpoint:.6f},{v:.6f},{w_out:.6f},{core.state.i:.6f}")
        t += dt
    print(f"# steady output speed at {target:.4g} rad/s command: "
          f"{w_out:.4f} rad/s, {core.state.i:.4f} A")


if __name__ == "__main__":
    main()
''')

_ROS_STUB = '''\
# rclpy is not installed in this environment, so the ROS node class is not
# generated.  DriveModelNodeCore above is the same control loop without ROS;
# re-run the export under a ROS 2 environment to get the node wrapper.
'''

_MANIFEST = Template("""\
{
  "schema": ${schema},
  "generated_utc": "${stamp}",
  "generator": "drivetrain/export.py",
  "dtype": "${dtype}",
  "sample_time_s": ${sample_time},
  "max_step_s": ${max_step},
  "provenance": ${provenance},
  "constants": ${constants},
  "derived": ${derived},
  "notes": ${notes},
  "units": {
    "speed": "rad/s at the motor shaft unless the name says output or ground",
    "torque": "N*m at the motor shaft unless the name says output",
    "current": "A at the terminals of the model (parallel motors collapsed)",
    "voltage": "V, signed; a negative command drives the shaft the other way",
    "inertia": "kg*m^2 referred to the motor shaft",
    "load_torque": "sign(w)*(load_static + load_viscous*|w| + load_quadratic*w^2)"
  },
  "contract": {
    "integrator": "classical RK4, fixed step, matches the Python engine 'rk4'",
    "max_step_s": "take dt at or below this; the model is stable and accurate there",
    "sample_time_s": "the control period this build was sized for, if one was given",
    "stiction_eps_rad_s": ${stiction_eps},
    "limits": "current_limit is total at the terminals; a real ESC may limit per motor",
    "extra_torque": "feedforward(..., extra_torque) is a known disturbance at the motor shaft; feedforward_output(..., extra_output_torque) is at the output shaft"
  },
  "not_modelled": [
    "thermal limits and derating (no temperature state, no resistance drift)",
    "magnetic saturation, iron losses, cogging, and speed-dependent efficiency maps",
    "ESC dead time, PWM quantisation, current-loop bandwidth and sensor delay",
    "gear backlash, compliance and windup; the gear is a rigid lossy ratio",
    "battery impedance and supply sag (voltage_limit is a hard rail)",
    "regenerative braking limits beyond the current and voltage clamps"
  ],
  "files": ${files}
}
""")

_README = Template("""\
# Exported drive model

Generated from **${source}**.

| file | what it is |
|---|---|
${file_table}

## The constants

All SI, referred to the **motor shaft**.  Parallel motors are already collapsed
into one terminal pair, so `current_limit` is the total at those terminals.

| constant | value | unit |
|---|---|---|
${constant_table}

Derived quantities (informational, not in the struct):

| quantity | value |
|---|---|
${derived_table}

## Feed-forward

The point of the export: the model inverted, not tuned.

```c
dt_command c = dt_feedforward_output(&${name}, w_out, accel_out, tau_extra_out);
pwm = c.voltage_V / BUS_VOLTAGE;      /* and respect c.saturated */
```

```c
dt_ff ff;                             /* clocked: ramp, differentiate, invert */
dt_ff_init(&ff, &${name}, ${sample_time_lit});
for (;;) { c = dt_ff_update(&ff, setpoint); }
```

```rust
let c = MODEL.feedforward_output(w_out, accel_out, tau_extra_out);
if !c.saturated.voltage { apply(c.voltage); }
```

```python
cmd = MODEL.feedforward_output(w_out, accel_out, tau_extra_out)
apply(cmd.voltage)
```

`w_out` and `accel_out` are at the **output shaft** (rad/s and rad/s^2);
`tau_extra_out` is a known disturbance there (a grade, a payload, a known
external torque).  For motor-side units use `feedforward(w, accel, extra_torque)`.

The acceleration term is the one the plant is asked for, so feed it the
derivative of your *limited* reference, not of the setpoint you typed.  The
generated `FeedForward` helper does exactly that: rate-limit, differentiate,
invert.

## Sample-time contract

`dt` for `dt_step` / `step` must be at or below **${max_step} s** (the largest
step the exported RK4 stays stable and accurate at).  ${sample_time_note}

${machine_note}## What this model does not include

${not_modelled}

## Checking it against the simulator

Each language ships a step-response bench that prints the same CSV:

    cc -std=c99 -O2 -Wall -I. ${c_example} ${name}.c -lm -o bench_c && ./bench_c
    rustc -O ${rust_example} -o bench_rust && ./bench_rust
    python3 ${py_example}

They agree with each other to the last printed digit, and with
`python -m drivetrain run --engine rk4 --dt ${bench_dt}` from the source
configuration to within the integrator's own error.
${fit_note}
""")


# ----------------------------------------------------------------------
# renderers
# ----------------------------------------------------------------------
def _second_order_note(c: ModelConstants) -> str:
    if c.is_second_order:
        return "second order here: inductance is non-zero, so current is a state too"
    return "first order here: inductance is zero, so `i` is diagnostic, not a state"


def _c_mapping(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    dtype = o.dtype
    ctype = "float" if dtype == "f32" else "double"
    m = {
        "banner": _banner(o, c),
        "guard": f"{o.header_prefix.upper()}_{o.module_name.upper()}_H",
        "name": f"{o.header_prefix}_{o.module_name}",
        "ctype": ctype,
        "second_order_note": _second_order_note(c),
        "max_step": _c_num(c.max_step_time(), dtype),
        "sample_time": _c_num(o.sample_time if o.sample_time else math.inf, dtype),
        "stiction_eps": _c_num(1e-6, dtype),
        "zero": "0.0" if dtype == "f64" else "0.0f",
        "one": "1.0" if dtype == "f64" else "1.0f",
        "two": "2.0" if dtype == "f64" else "2.0f",
        "four": "4.0" if dtype == "f64" else "4.0f",
        "half": "0.5" if dtype == "f64" else "0.5f",
        "six": "6.0" if dtype == "f64" else "6.0f",
        "sat_current": "1",
        "sat_voltage": "2",
        # libm's double entry points would silently widen every operation in a
        # float build, so the generated code calls the width-matched ones
        "sqrt": "sqrtf" if dtype == "f32" else "sqrt",
        "fabs": "fabsf" if dtype == "f32" else "fabs",
        "inf": "((dt_scalar)INFINITY)",
        "dtype": dtype,
    }
    for k, v in (("kv", c.kv), ("kt", c.kt), ("resistance", c.resistance),
                 ("inductance", c.inductance), ("inertia", c.inertia),
                 ("damping", c.damping), ("static_friction", c.static_friction),
                 ("load_static", c.load_static), ("load_viscous", c.load_viscous),
                 ("load_quadratic", c.load_quadratic),
                 ("current_limit", c.current_limit), ("voltage_limit", c.voltage_limit),
                 ("gear_ratio", c.gear_ratio), ("gear_efficiency", c.gear_efficiency),
                 ("wheel_radius", c.wheel_radius)):
        m[k] = _c_num(v, dtype)
    m["motor_count"] = str(int(c.motor_count))
    return m


def _bench_values(o: ExportOptions, c: ModelConstants,
                  dtype: str | None = None) -> dict[str, str]:
    """A step that shows something interesting in every generated bench.

    ``dtype`` is passed for the C bench, which needs width-correct literals to
    compile clean under -Wconversion; Rust and Python infer the width.
    """
    v = c.voltage_limit if math.isfinite(c.voltage_limit) else 12.0
    tau = c.mech_time_constant()
    duration = 8.0 * tau if math.isfinite(tau) else 5.0
    dt = min(0.001, (c.max_step_time() or 0.001) / 4.0)
    num = (lambda x: _c_num(x, dtype)) if dtype else (lambda x: f"{x:.6g}")
    return {
        "bench_dt": num(dt),
        "bench_duration": num(duration),
        "bench_voltage": num(v),
        "bench_step_at": num(0.1 * duration),
        "bench_max_steps": str(int(duration / dt) + 2),
        "sample_hint": f"{dt:.6g}",
        "tiny": _c_num(1e-12, dtype) if dtype else "1e-12",
    }


def _c_prefixed(text: str, prefix: str) -> str:
    """Rewrite the default ``dt_`` identifier prefix through the requested one.

    The templates are written against ``dt_`` because that is what 99% of builds
    use, but a prefix exists so two exported models can live in one binary.  A
    half-applied prefix is worse than none: the instance would be ``rv_rover``
    while the type stayed ``dt_model`` and the functions stayed ``dt_step``, so
    two builds would still collide.  Rewriting after substitution keeps the
    templates readable and makes the rename complete.
    """
    if prefix == "dt":
        return text
    text = re.sub(r"\bdt_(?=[a-z])", f"{prefix}_", text)
    return re.sub(r"\bDT_(?=[A-Z])", f"{prefix.upper()}_", text)


def _render_c(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    m = _c_mapping(c, o)
    header = f"{o.module_name}.h"
    files = {header: _C_HEADER.substitute(**m, header=header,
                                          file=f"{o.module_name}.c")}
    files[f"{o.module_name}.c"] = _C_SOURCE.substitute(**m, header=header)
    if o.include_examples:
        files[f"bench_{o.module_name}.c"] = _C_EXAMPLE.substitute(
            **m, **_bench_values(o, c, o.dtype), file=f"bench_{o.module_name}.c",
            header=header, source_file=f"{o.module_name}.c")
    if o.header_prefix != "dt":
        files = {k: _c_prefixed(v, o.header_prefix) for k, v in files.items()}
    return files


def _rust_mapping(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    dtype = o.dtype
    m = {
        "banner": _banner(o, c, style="//"),
        "dtype": dtype,
        "second_order_note": _second_order_note(c),
        "max_step": _rust_num(c.max_step_time(), dtype),
        "stiction_eps": _rust_num(1e-6, dtype),
        "dtype_upper": dtype,
    }
    lines = ["/// The constants for this build.",
             f"pub const MODEL: DriveModel = DriveModel {{"]
    for k, v in (("kv", c.kv), ("kt", c.kt), ("resistance", c.resistance),
                 ("inductance", c.inductance), ("inertia", c.inertia),
                 ("damping", c.damping), ("static_friction", c.static_friction),
                 ("load_static", c.load_static), ("load_viscous", c.load_viscous),
                 ("load_quadratic", c.load_quadratic),
                 ("current_limit", c.current_limit), ("voltage_limit", c.voltage_limit),
                 ("gear_ratio", c.gear_ratio), ("gear_efficiency", c.gear_efficiency),
                 ("wheel_radius", c.wheel_radius)):
        lines.append(f"    {k}: {_rust_num(v, dtype)},")
    lines.append(f"    motor_count: {int(c.motor_count)}u32,")
    lines.append("};")
    lines.append("")
    lines.append(f"/// The control period this build was sized for [{dtype}].")
    lines.append(f"pub const SAMPLE_TIME_S: {dtype} = "
                 f"{_rust_num(o.sample_time if o.sample_time else 0.0, dtype)};")
    lines.append(f"/// Largest stable fixed step for [`DriveModel::step`] [{dtype}].")
    lines.append(f"pub const MAX_STEP_S: {dtype} = {_rust_num(c.max_step_time(), dtype)};")
    m["constants"] = "\n".join(lines)
    m["impl"] = _RUST_IMPL.substitute(max_step=m["max_step"],
                                      stiction_eps=m["stiction_eps"])
    return m


def _render_rust(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    m = _rust_mapping(c, o)
    files = {f"{o.module_name}.rs": _RUST.substitute(**m)}
    if o.include_examples:
        files[f"bench_{o.module_name}.rs"] = _RUST_BENCH.substitute(
            **m, **_bench_values(o, c, "f64"), file=f"bench_{o.module_name}.rs")
    return files


def _render_python(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    m = {
        "banner": _banner(o, c, style="#"),
        "second_order_note": _second_order_note(c),
        "stiction_eps": "1e-6",
        "max_step": _py_num(c.max_step_time(), o.dtype),
        "sample_time": _py_num(o.sample_time, o.dtype) if o.sample_time else "None",
    }
    for k, v in (("kv", c.kv), ("kt", c.kt), ("resistance", c.resistance),
                 ("inductance", c.inductance), ("inertia", c.inertia),
                 ("damping", c.damping), ("static_friction", c.static_friction),
                 ("load_static", c.load_static), ("load_viscous", c.load_viscous),
                 ("load_quadratic", c.load_quadratic),
                 ("current_limit", c.current_limit), ("voltage_limit", c.voltage_limit),
                 ("gear_ratio", c.gear_ratio), ("gear_efficiency", c.gear_efficiency),
                 ("wheel_radius", c.wheel_radius)):
        m[k] = _py_num(v, o.dtype)
    m["motor_count"] = str(int(c.motor_count))
    files = {f"{o.module_name}.py": _PYTHON.substitute(**m)}
    if o.include_examples:
        files[f"bench_{o.module_name}.py"] = _PYTHON_BENCH.substitute(
            **m, **_bench_values(o, c), file=f"bench_{o.module_name}.py")
    return files


def _render_ros(c: ModelConstants, o: ExportOptions) -> dict[str, str]:
    sample = o.sample_time or min(0.01, c.max_step_time())
    m = {
        "banner": _banner(o, c, style="#"),
        "node_sample_time": _py_num(sample, o.dtype),
        "node_kp": _py_num(0.0, o.dtype),
        "node_max_correction": _py_num(0.0, o.dtype),
        "module_name": o.module_name,
        "steady_speed": _py_num(0.6 * c.steady_output_speed(
            c.voltage_limit if math.isfinite(c.voltage_limit) else 12.0), o.dtype),
    }
    files = {f"{o.module_name}_ros.py": _ROS.substitute(
        **m, node_class=_ROS_NODE.substitute(**m))}
    if o.include_examples:
        # the core loop is ROS-free on purpose, so the example runs anywhere:
        # it is the reference the ROS node and the parity test are checked
        # against
        files[f"bench_{o.module_name}_ros.py"] = _ROS_BENCH.substitute(
            **m, **_bench_values(o, c), node_module=f"{o.module_name}_ros")
    return files


def _table(rows: Iterable[tuple[str, str, str]]) -> str:
    return "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)


def _render_readme(c: ModelConstants, o: ExportOptions, files: Sequence[str]) -> str:
    p = dict(o.provenance or {})
    bench = _bench_values(o, c)
    constants_rows = [(k, _fmt(v["value"]), v["unit"])
                      for k, v in c.as_dict().items()]
    derived_rows = [(k, _fmt(v), "") for k, v in c.derived_dict().items()]
    file_rows = [(f"`{f}`", _file_kind(f), "") for f in files]
    not_modelled = "\n".join(f"- {item}" for item in _NOT_MODELLED)
    fit_note = ""
    if p.get("fit"):
        fit = p["fit"]
        warns = list(fit.get("warnings", ()))
        if warns:
            fit_note = ("\n### Caveats carried over from the fit\n\n"
                        + "\n".join(f"- {w}" for w in warns))
    machine_note = ""
    if c.notes:
        machine_note = (
            "## Caveats specific to this machine\n\n"
            + "\n".join(f"- {n}" for n in c.notes) + "\n\n"
        )
    return _README.substitute(
        source=p.get("source", "a DriveSystem built in code"),
        name=f"{o.header_prefix}_{o.module_name}",
        file_table=_table(file_rows),
        constant_table=_table(constants_rows),
        derived_table=_table([(a, b, "—") for a, b, _ in derived_rows]),
        max_step=f"{c.max_step_time():g} s",
        sample_time_lit=(f"{o.sample_time:g}" if o.sample_time else "0.002"),
        sample_time_note=(
            f"This build was sized for a control period of **{o.sample_time:g} s**."
            if o.sample_time else
            "No control period was given, so this build is not sized for a fixed "
            "rate - re-export with `--sample-time` if your controller runs on a "
            "timer, and the feed-forward helpers will take that period as their "
            "default."),
        not_modelled=not_modelled,
        machine_note=machine_note,
        c_example=f"bench_{o.module_name}.c",
        rust_example=f"bench_{o.module_name}.rs",
        py_example=f"bench_{o.module_name}.py",
        bench_dt=bench["bench_dt"],
        fit_note=fit_note,
    )


_NOT_MODELLED = [
    "thermal limits and derating: no temperature state, no resistance drift",
    "magnetic saturation, iron losses, cogging, speed-dependent efficiency maps",
    "ESC dead time, PWM quantisation, current-loop bandwidth, sensor delay",
    "gear backlash and compliance: the gear is a rigid, lossy ratio",
    "battery impedance and supply sag: voltage_limit is a hard rail",
    "regenerative braking limits beyond the current and voltage clamps",
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isinf(v):
            return "unlimited"
        return f"{v:.6g}"
    return str(v)


def _file_kind(name: str) -> str:
    if name.endswith(".h"):
        return "types, prototypes, doc comments"
    if name.endswith(".c"):
        return f"constants + implementation{' (bench)' if name.startswith('bench_') else ''}"
    if name.endswith(".rs"):
        return f"no_std Rust module{' (bench)' if name.startswith('bench_') else ''}"
    if name.endswith("_ros.py"):
        return "ROS 2 node + ROS-free control core"
    if name.endswith(".py"):
        return f"pure-Python model{' (bench)' if name.startswith('bench_') else ''}"
    if name.endswith(".md"):
        return "constants, feed-forward recipe, sample-time contract"
    return "generated artifact"


# ----------------------------------------------------------------------
# the entry point
# ----------------------------------------------------------------------
def export(system: DriveSystem, out_dir: str | Path,
           options: ExportOptions | None = None, *,
           constants: ModelConstants | None = None) -> list[Path]:
    """Write the standalone model into *out_dir*; returns the file paths.

    The directory is created if needed.  Existing generated files are
    overwritten; anything else in the directory is left alone.
    """
    options = options or ExportOptions()
    c = round_to_dtype(constants or constants_from_system(system), options.dtype)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    if "c" in options.targets:
        written.update(_render_c(c, options))
    if "rust" in options.targets:
        written.update(_render_rust(c, options))
    if "python" in options.targets:
        written.update(_render_python(c, options))
    if "ros" in options.targets:
        written.update(_render_ros(c, options))

    paths: list[Path] = []
    for name, text in written.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)

    files = [{"name": p.name, "sha256": hashlib.sha256(
        p.read_bytes()).hexdigest()[:16]} for p in paths]
    manifest = _MANIFEST.substitute(
        schema=SCHEMA_VERSION,
        stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dtype=options.dtype,
        sample_time=(_py_num(options.sample_time) if options.sample_time else "null"),
        max_step=_py_num(c.max_step_time()),
        provenance=json.dumps(dict(options.provenance or {}), indent=2, default=str)
        .replace("\n", "\n  "),
        constants=json.dumps(c.as_dict(), indent=2, default=str).replace("\n", "\n  "),
        derived=json.dumps(c.derived_dict(), indent=2, default=str).replace("\n", "\n  "),
        notes=json.dumps(list(c.notes), indent=2, default=str).replace("\n", "\n  "),
        stiction_eps=_py_num(1e-6),
        files=json.dumps(files, indent=2).replace("\n", "\n  "),
    )
    manifest_path = out / f"{manifest_name(options)}.json"
    manifest_path.write_text(manifest, encoding="utf-8")
    paths.append(manifest_path)

    readme = _render_readme(c, options, [p.name for p in paths])
    (out / "README.md").write_text(readme, encoding="utf-8")
    paths.append(out / "README.md")
    return paths


def manifest_name(options: ExportOptions) -> str:
    """File name (no suffix) of the JSON manifest that accompanies the code."""
    return f"{options.module_name}_manifest"


# ----------------------------------------------------------------------
# CLI helpers (the argument parser itself lives in cli.py)
# ----------------------------------------------------------------------
def system_from_fit_document(doc: Mapping[str, Any]) -> DriveSystem:
    """Rebuild the fitted system from a ``fit --json`` document.

    ``FitResult.as_dict()`` embeds the whole model under ``"model"``, so a fit
    becomes firmware without an intermediate file: ``fit --json`` -> this ->
    :func:`export`.
    """
    from . import presets

    model = doc.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("this JSON has no \"model\" block; it is not a fit result "
                          "(re-run `fit --json`, or pass the model file to --config)")
    return presets.from_dict(dict(model))


def fit_provenance(doc: Mapping[str, Any], path: Any = None) -> dict[str, Any]:
    """The part of a fit result worth carrying into the generated comments."""
    return {
        "dataset": doc.get("dataset"),
        "kind": doc.get("kind"),
        "points": doc.get("points"),
        "normalized_rms": doc.get("normalized_rms"),
        "condition": doc.get("condition"),
        "warnings": list(doc.get("warnings", ())),
        "file": str(path) if path else None,
    }


def options_from_args(args: Any, source: str, *,
                      fit_doc: Mapping[str, Any] | None = None) -> ExportOptions:
    """Build :class:`ExportOptions` from an ``export`` argparse namespace."""
    targets = tuple(t.strip() for t in str(args.targets).split(",") if t.strip())
    provenance: dict[str, Any] = {"source": source}
    fit_json = getattr(args, "fit_json", None)
    if fit_doc is None and fit_json:
        try:
            fit_doc = json.loads(Path(fit_json).read_text(encoding="utf-8"))
        except OSError as exc:
            raise DrivetrainError(f"cannot read --fit-json {fit_json}: {exc}")
    if fit_doc is not None:
        provenance["fit"] = fit_provenance(fit_doc, fit_json)
    return ExportOptions(
        dtype=args.dtype, sample_time=args.sample_time, targets=targets,
        module_name=args.name, header_prefix=args.prefix,
        include_examples=not args.no_examples, provenance=provenance,
    )


def summarise(system: DriveSystem, options: ExportOptions,
              paths: Sequence[Path] | None = None) -> None:
    """Print the sample-time contract and the feed-forward law for a human."""
    c = constants_from_system(system)
    source = (options.provenance or {}).get("source")
    if source:
        print(f"source: {source}")
    if paths:
        print(f"wrote {len(paths)} files to {paths[0].parent} "
              f"({options.dtype}, {', '.join(options.targets)})")
        for p in paths:
            print(f"  {p}")
    print(f"\nsample-time contract: dt <= {c.max_step_time():g} s "
          f"(mechanical time constant {c.mech_time_constant():g} s"
          + (f", electrical {c.electrical_time_constant:g} s" if c.is_second_order else "")
          + ")")
    if options.sample_time and options.sample_time > c.max_step_time():
        print(f"WARNING: --sample-time {options.sample_time:g} s exceeds that; the "
              "exported integrator will be inaccurate or unstable there")
    w0 = c.no_load_speed(c.voltage_limit)
    print(f"no-load speed: {w0:.1f} rad/s at the motor, {w0 * c.gear_ratio:.1f} rad/s "
          f"at the output ({c.gear_ratio:.3g}:1 ratio)")
    print(f"stall torque:  {c.stall_torque(c.voltage_limit):.2f} N*m at the motor, "
          f"{c.stall_torque(c.voltage_limit) * c.gear_ratio * c.gear_efficiency:.2f} N*m "
          "at the output")
    print(f"current limit: {c.current_limit:g} A total across {c.motor_count} motor(s) "
          f"= {c.per_motor_current_limit:g} A each")
    print("\nfeed-forward (the model inverted, not tuned):")
    print("  i = (J*a + b*w + load(w) + friction(w) + tau) / kt;   v = i*R + kv*w")
    print("  feed the a term the derivative of your LIMITED reference, not of the setpoint")
    if (options.provenance or {}).get("fit", {}).get("warnings"):
        print("\ncarried from the fit:")
        for w in options.provenance["fit"]["warnings"]:
            print(f"  ! {w}")
