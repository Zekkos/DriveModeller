"""Fit a drivetrain model to measured data, holding the constants you already know.

System identification splits naturally into three parts:

* a **dataset** -- measured channels from a bench run, logged as CSV;
* **known constants** -- quantities measured elsewhere (wheel radius, vehicle
  mass, gear ratio, motor count ...) get pinned to their measured values so the
  fit cannot "explain" them away;
* **free parameters** -- the handful of paths the data can actually see, usually
  on the motor side, which the optimiser adjusts.

The fit runs in log space for positive parameters, so quantities spanning orders
of magnitude (rotor inertia ~1e-5 kg*m^2, winding resistance ~1 ohm) are equally
conditioned and stay positive.  It reports a linearised standard error per
parameter plus an identifiability warning when the data cannot separate two of
them -- ``kt`` and rotor inertia are the classic pair, because speed alone sees
only their ratio.

Dataset formats
---------------
Delimited text (comma / semicolon / tab / whitespace) with a header row.
Column names are matched case-insensitively against the model's own channel
vocabulary (``motor_rpm``, ``current_A``, ``voltage_V``, ``speed_kmh`` ...) with
common aliases accepted (``rpm``, ``amps``, ``volts`` ...).  Unknown columns are
ignored and listed in :attr:`Dataset.notes`.

* **Time series with a measured voltage column** -- the measured voltage drives
  the model, which is the honest way to identify a drivetrain: current, speed
  and back-EMF then all carry independent information.
* **Time series without voltage** -- pass ``voltage=`` and ``excitation=``
  (``step``, ``square``, ``ramp``, ``alternating``, ``coast``) to prescribe the
  input instead.
* **Steady-state rows** -- no ``time`` column: each row is one settled operating
  point at a given ``voltage_V``.  Predictions use the closed-form equilibrium,
  so this is fast and ideal for datasheet points (no-load rpm, stall current).

Several experiments can share one file: add a ``run`` / ``segment`` column and
each label becomes its own time base and input signal.  Time-series blocks are
assumed to start from rest, matching the engines' initial condition.

Examples
--------
    >>> from drivetrain import fit
    >>> r = fit.fit("bench_07.csv", preset="pod_2x_direct",
    ...             known={"load.mass": 12.5, "load.wheel.radius": 0.0625},
    ...             free=("group.motor.kv", "group.motor.resistance"))
    >>> print(r.report())
    >>> r.save("fitted.json")        # loadable by the CLI and the GUI

Scipy is used when available; a built-in Levenberg-Marquardt keeps fitting
working without it.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import presets
from .dq_sim import (
    DQ_ENGINES,
    DqState,
    machine_of,
    max_step_time,
    park_arrays,
    pole_magnitude,
    simulate_dq,
)
from .exceptions import ConfigError
from .load import RotationalLoad, VehicleLoad
from .motor import rad_s_to_rpm
from .pmsm import PMSMModel
from .simulate import simulate, voltage_signal
from .sweep import get_path, set_path
from .system import DriveSystem

try:  # optional: scipy's trust-region solver, with LM as fallback
    from scipy.optimize import least_squares as _scipy_least_squares
except ImportError:  # pragma: no cover - exercised on scipy-free installs
    _scipy_least_squares = None

# ----------------------------------------------------------------------
# vocabulary
# ----------------------------------------------------------------------
#: canonical measured quantities the model can also predict
DATA_CHANNELS: tuple[str, ...] = (
    "motor_rpm", "motor_rad_s", "output_rpm", "output_rad_s",
    "speed_kmh", "speed_m_s", "current_A", "torque_Nm",
    "voltage_V", "power_in_W", "power_out_W", "copper_loss_W", "efficiency",
    "distance_m",
    "current_d_A", "current_q_A", "voltage_d_V", "voltage_q_V", "rotor_angle_rad",
)

#: the brushless machine's own coordinates (see docs/pmsm.md)
DQ_CHANNELS: tuple[str, ...] = (
    "current_d_A", "current_q_A", "voltage_d_V", "voltage_q_V", "rotor_angle_rad",
)

#: raw inverter-side quantities; a Park transform turns these into dq channels
PHASE_CHANNELS: tuple[str, ...] = (
    "current_a_A", "current_b_A", "current_c_A",
    "voltage_a_V", "voltage_b_V", "voltage_c_V",
)

#: The angle that transform needs.  Deliberately *not* in DATA_CHANNELS: the
#: model's electrical angle follows from p * theta, so it is a coordinate the
#: data is written in, not something the residuals should chase.
ELECTRICAL_ANGLE_CHANNEL = "rotor_angle_electrical_rad"

#: channels that can drive the model for time-series data
INPUT_CHANNELS: tuple[str, ...] = ("voltage_V",)

#: What a brushless transient logs instead of a drive-voltage magnitude: the
#: vector, not its length.  A magnitude says how hard the inverter pushed, not in
#: which direction, and the dq equations are written in that direction.
DQ_INPUT_CHANNELS: tuple[str, ...] = ("voltage_d_V", "voltage_q_V")

#: What a brushless settled point supplies instead of a drive voltage: the
#: operating point, from which the voltages are algebraic.
DQ_STEADY_INPUT_CHANNELS: tuple[str, ...] = ("current_d_A", "current_q_A")

#: channels predictable from settled operating points (no trajectory -> no distance)
STEADY_CHANNELS: tuple[str, ...] = tuple(
    ch for ch in DATA_CHANNELS if ch != "distance_m"
)

TIME_COLUMNS = ("time", "t", "timestamp", "seconds", "sec", "elapsed")
GROUP_COLUMNS = ("run", "segment", "test", "trial", "experiment", "block")

_LOWER_CHANNEL = {ch.lower(): ch for ch in DATA_CHANNELS}
#: a three-phase log names its columns current_a_A, current_b_A and so on.  Those are
#: not model channels -- the Park transform turns them into dq channels at load time,
#: which is why DATA_CHANNELS leaves them out -- so the table above has to be extended
#: by hand or every phase column would be dropped as unrecognised.  Both spellings are
#: registered: a header is matched on its canonical form, and the canonical form of
#: ``voltage_a_A`` is ``voltage_a_a``, not the channel's own lower-cased name
#: ``voltage_a_v``.
_LOWER_CHANNEL.update({ch.lower(): ch for ch in PHASE_CHANNELS})
_LOWER_TIME = {c.lower(): c for c in TIME_COLUMNS}
_LOWER_GROUP = {c.lower(): c for c in GROUP_COLUMNS}

#: loose column names -> canonical channel
ALIASES: dict[str, str] = {
    "t": "time", "timestamp": "time", "seconds": "time", "sec": "time",
    "elapsed": "time", "time_s": "time",
    "rpm": "motor_rpm", "motor_speed": "motor_rpm", "shaft_rpm": "motor_rpm",
    "motor_rpm_measured": "motor_rpm", "speed_rpm": "motor_rpm",
    "gearbox_rpm": "output_rpm", "wheel_rpm": "output_rpm", "output_shaft_rpm": "output_rpm",
    # angular velocity logged in rad/s rather than rpm: the motor side is what an
    # encoder on the shaft reports, so a bare rad/s column means the motor side
    "rad_s": "motor_rad_s", "rads": "motor_rad_s", "rad_per_s": "motor_rad_s",
    "rad_sec": "motor_rad_s", "radps": "motor_rad_s", "omega": "motor_rad_s",
    "angular_velocity": "motor_rad_s", "angular_speed": "motor_rad_s",
    "motor_omega": "motor_rad_s", "motor_rad_s": "motor_rad_s",
    "motor_speed_rad_s": "motor_rad_s", "rotational_speed": "motor_rad_s",
    "shaft_rad_s": "output_rad_s", "output_omega": "output_rad_s",
    "output_rad_s": "output_rad_s", "wheel_omega": "output_rad_s",
    "gearbox_omega": "output_rad_s", "load_omega": "output_rad_s",
    "speed": "speed_kmh", "kmh": "speed_kmh", "km_h": "speed_kmh",
    "ground_speed": "speed_kmh", "velocity": "speed_m_s", "speed_ms": "speed_m_s",
    "m_s": "speed_m_s", "mps": "speed_m_s",
    "current": "current_A", "amps": "current_A", "i": "current_A",
    "line_current": "current_A", "motor_current": "current_A",
    "voltage": "voltage_V", "volts": "voltage_V", "v": "voltage_V",
    "supply": "voltage_V", "bus_voltage": "voltage_V",
    "torque": "torque_Nm", "torque_nm": "torque_Nm",
    "power_in": "power_in_W", "pins": "power_in_W", "watts_in": "power_in_W",
    "power_out": "power_out_W", "pout": "power_out_W", "watts_out": "power_out_W",
    "copper_loss": "copper_loss_W", "i2r": "copper_loss_W", "i_squared_r": "copper_loss_W",
    "eff": "efficiency", "eta": "efficiency",
    "distance": "distance_m", "odometer": "distance_m",
    # brushless dq coordinates
    "id": "current_d_A", "i_d": "current_d_A", "id_a": "current_d_A",
    "i_d_a": "current_d_A", "current_d": "current_d_A", "d_current": "current_d_A",
    "d_current_a": "current_d_A", "daxis_current": "current_d_A",
    "d_axis_current": "current_d_A", "direct_axis_current": "current_d_A",
    "iq": "current_q_A", "i_q": "current_q_A", "iq_a": "current_q_A",
    "i_q_a": "current_q_A", "current_q": "current_q_A", "q_current": "current_q_A",
    "q_current_a": "current_q_A", "qaxis_current": "current_q_A",
    "q_axis_current": "current_q_A", "quadrature_current": "current_q_A",
    "vd": "voltage_d_V", "v_d": "voltage_d_V", "vd_v": "voltage_d_V",
    "v_d_v": "voltage_d_V", "voltage_d": "voltage_d_V", "d_voltage": "voltage_d_V",
    "daxis_voltage": "voltage_d_V", "d_axis_voltage": "voltage_d_V",
    "vq": "voltage_q_V", "v_q": "voltage_q_V", "vq_v": "voltage_q_V",
    "v_q_v": "voltage_q_V", "voltage_q": "voltage_q_V", "q_voltage": "voltage_q_V",
    "qaxis_voltage": "voltage_q_V", "q_axis_voltage": "voltage_q_V",
    "theta": "rotor_angle_rad", "theta_m": "rotor_angle_rad",
    "theta_mech": "rotor_angle_rad", "mech_angle": "rotor_angle_rad",
    "mechanical_angle": "rotor_angle_rad", "rotor_angle": "rotor_angle_rad",
    "rotor_position": "rotor_angle_rad", "shaft_angle": "rotor_angle_rad",
    "mech_position": "rotor_angle_rad", "angle": "rotor_angle_rad",
    # the electrical angle: a coordinate, so not a channel of its own
    "theta_e": ELECTRICAL_ANGLE_CHANNEL,
    "theta_electrical": ELECTRICAL_ANGLE_CHANNEL,
    "electrical_angle": ELECTRICAL_ANGLE_CHANNEL,
    "electrical_angle_rad": ELECTRICAL_ANGLE_CHANNEL,
    "angle_e": ELECTRICAL_ANGLE_CHANNEL, "angle_electrical": ELECTRICAL_ANGLE_CHANNEL,
    "rotor_angle_electrical": ELECTRICAL_ANGLE_CHANNEL,
    "rotor_angle_electrical_rad": ELECTRICAL_ANGLE_CHANNEL,
    "electrical_position": ELECTRICAL_ANGLE_CHANNEL,
    "commutation_angle": ELECTRICAL_ANGLE_CHANNEL,
    # three-phase logs; the Park transform below turns these into dq channels
    "ia": "current_a_A", "i_a": "current_a_A", "ia_a": "current_a_A",
    "current_a": "current_a_A", "phase_a_current": "current_a_A",
    "phase_current_a": "current_a_A", "a_phase_current": "current_a_A",
    "line_current_a": "current_a_A", "current_u": "current_a_A",
    "ib": "current_b_A", "i_b": "current_b_A", "ib_a": "current_b_A",
    "current_b": "current_b_A", "phase_b_current": "current_b_A",
    "phase_current_b": "current_b_A", "b_phase_current": "current_b_A",
    "line_current_b": "current_b_A", "current_v": "current_b_A",
    "ic": "current_c_A", "i_c": "current_c_A", "ic_a": "current_c_A",
    "current_c": "current_c_A", "phase_c_current": "current_c_A",
    "phase_current_c": "current_c_A", "c_phase_current": "current_c_A",
    "line_current_c": "current_c_A", "current_w": "current_c_A",
    "va": "voltage_a_V", "v_a": "voltage_a_V", "va_v": "voltage_a_V",
    "voltage_a": "voltage_a_V", "phase_a_voltage": "voltage_a_V",
    "phase_voltage_a": "voltage_a_V", "a_phase_voltage": "voltage_a_V",
    "terminal_voltage_a": "voltage_a_V", "voltage_u": "voltage_a_V",
    "vb": "voltage_b_V", "v_b": "voltage_b_V", "vb_v": "voltage_b_V",
    "voltage_b": "voltage_b_V", "phase_b_voltage": "voltage_b_V",
    "phase_voltage_b": "voltage_b_V", "b_phase_voltage": "voltage_b_V",
    "terminal_voltage_b": "voltage_b_V", "voltage_v": "voltage_b_V",
    "vc": "voltage_c_V", "v_c": "voltage_c_V", "vc_v": "voltage_c_V",
    "voltage_c": "voltage_c_V", "phase_c_voltage": "voltage_c_V",
    "phase_voltage_c": "voltage_c_V", "c_phase_voltage": "voltage_c_V",
    "terminal_voltage_c": "voltage_c_V", "voltage_w": "voltage_c_V",
}

#: convenience names for groups of parameter paths
FREE_ALIASES: dict[str, tuple[str, ...]] = {
    "motor": (
        "group.motor.kv", "group.motor.kt", "group.motor.resistance",
        "group.motor.rotor_inertia", "group.motor.bearing_damping",
        "group.motor.static_friction",
    ),
    "electrical": ("group.motor.kv", "group.motor.kt", "group.motor.resistance"),
    "mechanical": (
        "group.motor.rotor_inertia", "group.motor.bearing_damping",
        "group.motor.static_friction", "gearbox.inertia", "gearbox.damping",
    ),
    "losses": (
        "group.motor.bearing_damping", "group.motor.static_friction",
        "gearbox.efficiency", "gearbox.damping",
    ),
    "drivetrain": ("gearbox.ratio", "gearbox.efficiency", "gearbox.inertia", "gearbox.damping"),
    "vehicle": (
        "load.mass", "load.rolling_resistance", "load.drag_cd", "load.air_density",
        "load.grade_deg", "load.extra_rotational_inertia",
    ),
    "rotational": ("load.inertia", "load.viscous", "load.constant_torque"),
    "wheel": ("load.wheel.radius", "load.wheel.mass", "load.wheel.inertia_factor"),
    # the two families the operator usually knows or does not know separately:
    # "I measured the inertia on the bench but I have no idea about friction"
    "inertia": (
        "group.motor.rotor_inertia", "gearbox.inertia", "load.inertia",
        "load.extra_rotational_inertia", "load.wheel.mass",
    ),
    "friction": (
        "group.motor.bearing_damping", "group.motor.static_friction",
        "gearbox.damping", "load.viscous",
        "load.constant_torque", "load.rolling_resistance", "load.drag_cd",
    ),
    "damping": (
        "group.motor.bearing_damping", "group.motor.static_friction",
        "gearbox.damping", "load.viscous",
        "load.constant_torque", "load.rolling_resistance", "load.drag_cd",
    ),
}

#: ``--free all`` means every parameter this configuration exposes
FREE_ALL = "all"

#: parameters that only ever show up in the *shape* of a transient, never in a
#: settled operating point
TRANSIENT_ONLY_PATHS = (
    "group.motor.rotor_inertia", "group.motor.inductance", "gearbox.inertia",
    "load.inertia", "load.extra_rotational_inertia", "load.wheel.mass",
    "load.wheel.inertia_factor",
)

#: parameters the electrical channels say nothing about on their own
MECHANICAL_PATHS = frozenset({
    "group.motor.rotor_inertia", "group.motor.bearing_damping",
    "group.motor.static_friction", "gearbox.inertia", "gearbox.damping",
    "load.inertia", "load.viscous", "load.constant_torque",
    "load.rolling_resistance", "load.drag_cd", "load.extra_rotational_inertia",
})

#: channels that carry electrical *response* information (voltage_V is the input,
#: so measuring it says nothing the model does not already know)
ELECTRICAL_CHANNELS = frozenset({"current_A", "torque_Nm", "power_in_W", "copper_loss_W"})

#: channels that measure speed rather than electrical quantities
MOTION_CHANNELS = frozenset(
    {"motor_rpm", "motor_rad_s", "output_rpm", "output_rad_s",
     "speed_kmh", "speed_m_s", "distance_m"}
)

#: parameter families that only exist in some configurations: paths, why, test
CONFIG_FAMILIES: tuple[tuple[tuple[str, ...], str, Callable[[Any], bool]], ...] = (
    (
        ("load.mass", "load.rolling_resistance", "load.drag_cd", "load.air_density",
         "load.grade_deg", "load.gravity", "load.final_drive_efficiency",
         "load.wheels_driven"),
        "this system has no ground vehicle attached",
        lambda s: isinstance(s.load, VehicleLoad),
    ),
    (
        ("load.viscous", "load.constant_torque", "load.inertia"),
        "this system has no load bolted to the output shaft",
        lambda s: isinstance(s.load, RotationalLoad),
    ),
    (
        ("load.wheel.radius", "load.wheel.mass", "load.wheel.inertia_factor"),
        "there is no wheel in this configuration",
        lambda s: s.wheel is not None,
    ),
    (
        ("group.motor.inductance",),
        "the current loop is algebraic here, since inductance is zero, so it never enters the"
        " equations",
        lambda s: s.is_second_order,
    ),
)


#: Brushed parameter names -> the paths a brushless system can actually set.
#: A PMSM carried by a drivetrain runs through its ``i_d = 0`` collapse, so the
#: brushed vocabulary still names the right physics; only the addresses move, and
#: ``kt`` and ``kv`` collapse onto the single flux linkage they are both read from
#: (``kt = 1.5 p lambda``, ``kv = p lambda``).  Empty for a brushed machine, so
#: callers can apply it unconditionally.
DQ_MOTOR_PATHS: dict[str, str] = {
    "group.motor.kv": "group.motor.spec.flux_linkage",
    "group.motor.kt": "group.motor.spec.flux_linkage",
    "group.motor.resistance": "group.motor.spec.resistance",
    "group.motor.inductance": "group.motor.spec.inductance_q",
    "group.motor.rotor_inertia": "group.motor.spec.rotor_inertia",
    "group.motor.bearing_damping": "group.motor.spec.bearing_damping",
    "group.motor.static_friction": "group.motor.spec.static_friction",
    "group.motor.max_current": "group.motor.spec.current_limit",
    "group.motor.max_voltage": "group.motor.spec.voltage_limit",
}

#: dq constants the drivetrain collapse never evaluates.  This is the honest
#: answer to the identifiability question: a log of a drive running ``i_d = 0``
#: cannot separate these from the parameters they masquerade as, however clean it
#: is.  Fitting them recovers the noise, not the machine.
DQ_COLLAPSE_INERT: dict[str, str] = {
    "group.motor.spec.inductance_d":
        "the drivetrain carries this machine through its i_d=0 collapse, whose single "
        "current sits on the q axis, so the d-axis inductance never enters these equations",
    "group.motor.spec.core_loss_coefficient":
        "the collapse's only speed-dependent loss is bearing_damping, so an iron loss "
        "would be indistinguishable from it",
    "group.motor.spec.windage_coefficient":
        "the collapse's only speed-dependent loss is bearing_damping, so a windage loss "
        "would be indistinguishable from it",
    "group.motor.spec.flux_temperature_coefficient":
        "the collapse reads the reference-temperature flux linkage, and nothing in this "
        "model varies magnet temperature, so a hot machine looks identical to a cold one",
    "group.motor.spec.reference_temperature":
        "only the flux linkage at the reference temperature is ever read, so the reference "
        "itself has no effect",
    "group.motor.spec.dc_bus_voltage":
        "the collapse drives on the phase rail, not the bus: the bus number only converts "
        "phase current into rail current, which these channels do not measure",
}


def _dq_live(system: DriveSystem, path: str, channels: set[str]) -> bool:
    """True when what a dataset measures puts a collapse-inert parameter to work.

    Inertness is a property of the *question asked*, not of the parameter.  Under
    the i_d=0 collapse there is one current, it sits on the q axis, and the only
    inductance it ever sees is the one the collapse calls ``L`` (which is why
    ``inductance_q`` is not in :data:`DQ_COLLAPSE_INERT` at all: a brushed transient
    does constrain it, through the current rise time).  A log that carries the two
    axes separately changes the equations:

        v_d = R i_d - w_e Lq i_q
        v_q = R i_q + w_e (Ld i_d + lambda)

    Lq appears only in v_d and Ld only in v_q, so each inductance becomes fittable
    exactly when the log carries both currents *and* the voltage of the axis whose
    equation contains it.  Above a few rad/s those cross-coupling terms dwarf the
    R i terms, which is the whole reason saliency is measurable at all.
    """
    if not ("current_d_A" in channels and "current_q_A" in channels):
        return False
    if path == "group.motor.spec.inductance_d":
        return "voltage_q_V" in channels
    if path == "group.motor.spec.inductance_q":
        return "voltage_d_V" in channels
    return False


def dq_motor_map(system: DriveSystem) -> dict[str, str]:
    """The brushed-name -> settable-path mapping for this system's motor."""
    return dict(DQ_MOTOR_PATHS) if isinstance(system.group.motor, PMSMModel) else {}


def _dq_known_value(path: str, value: float, system: DriveSystem) -> float:
    """Convert a pinned brushed value onto the constant a brushless system sets.

    ``known={"group.motor.kv": 0.0126}`` pins a back-EMF constant, but a brushless
    machine has no back-EMF constant to set - it has a flux linkage, and
    ``kv = p lambda``.  The number has to be divided by the pole pairs on the way
    through, or pinning ``kv`` would pin a flux linkage ``p`` times too large.
    """
    spec = getattr(system.group.motor, "spec", None)
    poles = float(getattr(spec, "pole_pairs", 1)) or 1.0
    if path == "group.motor.kv":
        return value / poles
    if path == "group.motor.kt":
        return value / (1.5 * poles)
    return value


def _translate_known(
    known: Mapping[str, float], mapping: Mapping[str, str], system: DriveSystem
) -> dict[str, float]:
    """Re-point brushed names in ``known=`` at the paths a brushless system sets.

    Pinning both ``kt`` and ``kv`` is the interesting case: on a brushless machine
    they are one flux linkage read from opposite ends, so two values that imply
    different flux linkages are a contradiction rather than two measurements.  Two
    that imply the same one - which is what a datasheet gives you - are accepted.
    """
    out: dict[str, float] = {}
    given: dict[str, list[str]] = {}
    for path, value in known.items():
        real = mapping.get(path, path)
        converted = _dq_known_value(path, value, system) if real != path else value
        if real in out:
            scale = max(abs(converted), abs(out[real]), 1e-12)
            if abs(out[real] - converted) > 1e-9 * scale:
                raise ConfigError(
                    f"{given[real][0]} and {path}={value!r} both set {real!r}, which is a single "
                    f"parameter on a brushless machine (kt = 1.5 p lambda, kv = p lambda), and "
                    f"they imply different values: {out[real]!r} against {converted!r}. Pin "
                    f"{real} once - or leave it free and let the fit choose it."
                )
        out[real] = converted
        given.setdefault(real, []).append(f"{path}={value!r}")
    return out


def _translate_warm(
    warm: Mapping[str, tuple[float, str]], mapping: Mapping[str, str], system: DriveSystem
) -> dict[str, tuple[float, str]]:
    """Convert log-derived starting values onto the paths a brushless system sets.

    The log states the brushed quantities directly - the back-EMF line gives ``kv``,
    standstill acceleration gives ``kt`` - and both are reads of the *same* flux
    linkage from opposite ends of the run.  Where both exist, take the geometric
    mean: they should agree once the 1.5 convention factor is applied, and the
    average of two independent reads beats either one alone.
    """
    out = dict(warm)
    flux = "group.motor.spec.flux_linkage"
    if flux in mapping.values():
        p = float(getattr(system.group.motor, "pole_pairs", 1)) or 1.0
        reads: list[tuple[float, str]] = []
        if "group.motor.kv" in warm:
            value, how = warm["group.motor.kv"]
            if value > 0:
                reads.append((value / p, f"{how} / pole pairs"))
        if "group.motor.kt" in warm:
            value, how = warm["group.motor.kt"]
            if value > 0:
                reads.append((value / (1.5 * p), f"{how} / (1.5 x pole pairs)"))
        if len(reads) == 1:
            out[flux] = reads[0]
        elif len(reads) > 1:
            product = 1.0
            for value, _ in reads:
                product *= value
            out[flux] = (
                product ** (1.0 / len(reads)),
                "geometric mean of the back-EMF and standstill-torque reads of one flux linkage",
            )
    for source, (value, how) in warm.items():
        real = mapping.get(source)
        if real and real != flux:
            out.setdefault(real, (value, how))
    return out


#: parameters that may legitimately come out negative: a torque that helps the
#: motion along rather than resisting it, or a downhill grade.  Everything else
#: in the catalogue is a magnitude, and a negative inertia, mass, damping or
#: resistance is not a thing, however much the optimiser would like one.
SIGNED_PATHS = frozenset({"load.constant_torque", "load.grade_deg"})

#: channels whose flat-topped trace gives the game away about current limiting
CLIPPABLE_CHANNELS = frozenset(
    {"current_A", "current_d_A", "current_q_A", "torque_Nm", "power_in_W", "power_out_W",
     "copper_loss_W"}
)

#: a fit that misses the data by this fraction of the data's own variation is
#: assumed to be sitting in the wrong basin, and gets one coarse decade rescan
#: before anyone believes it
RESCAN_MISS_FRACTION = 0.35

#: catalogue fields that exist for the datasheet but that no equation reads
NEVER_MODELLED: dict[str, str] = {
    "group.motor.no_load_friction_current":
        "no equation in the model reads it, it is a datasheet label rather than a "
        "model parameter",
}


def structural_inert(system: DriveSystem, path: str,
                     channels: set[str] | None = None) -> str | None:
    """Why a parameter takes no part in this configuration, or ``None``.

    A parameter the configuration never evaluates cannot be recovered from any
    dataset, however good: the residuals simply do not move when it moves.

    ``channels`` is what the dataset measures.  Some parameters are inert only
    because of how the model is being driven: the two dq inductances are invisible
    to the i_d=0 collapse but live the moment a log carries both axes (see
    :func:`_dq_live`), so the same path can be inert for one dataset and fittable
    for the next.  Omit ``channels`` to ask the question for a brushed log, which is
    the conservative answer.
    """
    if path in NEVER_MODELLED:
        return NEVER_MODELLED[path]
    if (path in DQ_COLLAPSE_INERT and isinstance(system.group.motor, PMSMModel)
            and not (channels and _dq_live(system, path, channels))):
        return DQ_COLLAPSE_INERT[path]
    for paths, why, present in CONFIG_FAMILIES:
        if path in paths and not present(system):
            return why
    return None

#: parameters with a hard physical range, regardless of what the optimiser wants
HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "gearbox.efficiency": (1e-3, 1.0),
    "load.final_drive_efficiency": (1e-3, 1.0),
    "group.motor.kv": (1e-5, math.inf),
    "group.motor.kt": (1e-5, math.inf),
    "group.motor.resistance": (1e-6, math.inf),
    "group.motor.rotor_inertia": (1e-12, math.inf),
    "gearbox.ratio": (1e-6, math.inf),
    "load.wheel.radius": (1e-6, math.inf),
    "load.mass": (1e-6, math.inf),
    # the brushless machine's own constants, for a fit that works on the dq model
    # rather than on its brushed collapse
    "group.motor.spec.resistance": (1e-6, math.inf),
    "group.motor.spec.inductance_d": (1e-8, 1.0),
    "group.motor.spec.inductance_q": (1e-8, 1.0),
    "group.motor.spec.flux_linkage": (1e-6, math.inf),
    "group.motor.spec.rotor_inertia": (1e-12, math.inf),
    "group.motor.spec.bearing_damping": (0.0, math.inf),
}

#: state fields, never fittable
STATE_PATHS = (
    "group.motor.current", "group.motor.rot_vel",
    "group.motor.spec.current_d", "group.motor.spec.current_q",
    "group.motor.spec.rot_vel",
)


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------
def _dq_from_phase(keep: dict[str, np.ndarray], notes: list[str]) -> None:
    """Park-transform logged three-phase columns into dq channels, in place.

    The transform happens at load time rather than inside the fit because it needs
    no model: the inverter's own commutation angle is enough.  Which is also why
    that angle has to be the *electrical* one -- a shaft-encoder angle would have
    to be multiplied by the pole count, and the pole count is one of the numbers a
    fit may be adjusting.
    """
    trios = (
        ("current", ("current_a_A", "current_b_A", "current_c_A"),
         "current_d_A", "current_q_A"),
        ("voltage", ("voltage_a_V", "voltage_b_V", "voltage_c_V"),
         "voltage_d_V", "voltage_q_V"),
    )
    asked = [t for t in trios if any(ch in keep for ch in t[1])]
    if not asked:
        return
    for kind, phases, d_name, q_name in asked:
        got = [ch for ch in phases if ch in keep]
        if len(got) != 3:
            raise ConfigError(
                f"phase {kind} columns need all three phases (got " + ", ".join(got)
                + "); a two-phase log cannot be transformed without knowing where the "
                "return current goes"
            )
        if d_name in keep or q_name in keep:
            raise ConfigError(
                f"the dataset carries both phase {kind} columns and dq {d_name}/"
                f"{q_name} columns; log one or the other, not both"
            )
    theta = keep.get(ELECTRICAL_ANGLE_CHANNEL)
    if theta is None:
        raise ConfigError(
            "three-phase columns need an electrical angle: log "
            f"{ELECTRICAL_ANGLE_CHANNEL} (theta_e, the angle the drive commutates on). "
            "A mechanical angle is not enough here, because the transform needs the "
            "pole count and the pole count is one of the numbers a fit may be "
            "adjusting; log the dq currents directly if only a shaft encoder exists."
        )
    for kind, phases, d_name, q_name in asked:
        d, q = park_arrays(theta, keep[phases[0]], keep[phases[1]], keep[phases[2]])
        zero = (keep[phases[0]] + keep[phases[1]] + keep[phases[2]]) / 3.0
        peak = float(np.nanmax(np.abs(zero))) if zero.size else 0.0
        ref = float(np.nanmax(np.hypot(d, q))) if d.size else 0.0
        for name in phases:
            del keep[name]
        keep[d_name] = d
        keep[q_name] = q
        notes.append(
            f"Park-transformed {', '.join(phases)} into {d_name}, {q_name} on "
            f"{ELECTRICAL_ANGLE_CHANNEL} (amplitude-invariant, so the dq values are "
            "phase amplitudes rather than RMS)"
        )
        if ref > 0.0 and peak > 1e-3 * ref:
            notes.append(
                f"the logged {kind} phases carry a zero sequence of {peak:.4g} "
                f"({100.0 * peak / ref:.1f}% of the vector); the dq transform drops it, "
                "as a three-wire machine does anyway"
            )


def _angle_housekeeping(keep: dict[str, np.ndarray], notes: list[str]) -> None:
    """Unwrap wrapped angles, and read the pole count off two angles if both are logged."""
    for ch in (ELECTRICAL_ANGLE_CHANNEL, "rotor_angle_rad"):
        if ch not in keep:
            continue
        arr = keep[ch]
        fixed = np.unwrap(arr) if arr.size > 1 else arr
        if not np.array_equal(fixed, arr):
            keep[ch] = fixed
            notes.append(f"unwrapped {ch}, which the sensor had wrapped past +/- pi")
    te = keep.get(ELECTRICAL_ANGLE_CHANNEL)
    tm = keep.get("rotor_angle_rad")
    if te is None or tm is None:
        return
    n = min(te.size, tm.size)
    if n < 6:
        return
    dte = np.diff(te[:n])
    dtm = np.diff(tm[:n])
    good = np.abs(dtm) > 1e-6
    if int(good.sum()) < 5:
        return
    est = float(np.median(dte[good] / dtm[good]))
    if not np.isfinite(est) or est <= 0.0:
        return
    nearest = round(est)
    note = (
        "the log carries both an electrical and a mechanical angle, so the pole count "
        f"is arithmetic rather than fittable: d(theta_e)/d(theta) = {est:.3f}"
    )
    if nearest >= 1 and abs(est - float(nearest)) <= 0.05 * float(nearest):
        note += (
            f", i.e. {int(nearest)} pole pairs -- set group.motor.spec.pole_pairs to "
            "that rather than fitting it, because a fit that sees only one of the two "
            "angles can never separate p from the flux linkage"
        )
    notes.append(note)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _read_table(text: str) -> tuple[list[str], list[list[float]]]:
    """Split delimited text into a header and numeric rows."""
    lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", "//", ";", "%"))
    ]
    if not lines:
        raise ConfigError("The dataset contains no data.")

    try:
        delim = csv.Sniffer().sniff(lines[0], delimiters=",;\t ").delimiter
    except csv.Error:
        delim = None

    def cells(line: str) -> list[str]:
        if delim:
            return [c.strip() for c in next(csv.reader([line], delimiter=delim))]
        return [c for c in re.split(r"[,\s;]+", line.strip()) if c]

    head = [c.strip().strip("\"'") for c in cells(lines[0])]
    body = lines
    if not _is_number(head[0]):
        body = lines[1:]
    else:
        head = [f"col{i + 1}" for i in range(len(head))]

    width = len(head)
    rows: list[list[float]] = []
    for line in body:
        parts = cells(line)
        if len(parts) < 2:
            continue
        vals = [float(p) if _is_number(p) else math.nan for p in parts[:width]]
        vals += [math.nan] * (width - len(vals))
        rows.append(vals)
    if not rows:
        raise ConfigError("No numeric rows found in the dataset.")
    return head, rows


def _canonical(name: str, extra: Mapping[str, str] | None = None) -> str | None:
    """Map a column header onto a canonical channel name (or ``None``).

    Headers are matched case-insensitively, so ``current_A``, ``CURRENT A`` and
    ``current_a`` all land on the same channel.
    """
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    key = re.sub(r"_+$", "", key)
    if extra and key in extra:
        key = extra[key]
    key = key.lower()
    if key in _LOWER_CHANNEL:
        return _LOWER_CHANNEL[key]
    if key in _LOWER_TIME:
        return _LOWER_TIME[key]
    if key in _LOWER_GROUP:
        return _LOWER_GROUP[key]
    return ALIASES.get(key)


@dataclass
class Block:
    """One experiment: a time base (or settled points) plus measured channels."""

    label: str
    time: np.ndarray | None
    columns: dict[str, np.ndarray]

    @property
    def kind(self) -> str:
        return "steady" if self.time is None else "timeseries"

    @property
    def n(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def channels(self) -> list[str]:
        return [ch for ch in DATA_CHANNELS if ch in self.columns]

    def duration(self) -> float:
        if self.time is None or len(self.time) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def describe(self) -> str:
        if self.time is None:
            return f"{self.label}: {self.n} steady-state points"
        return (
            f"{self.label}: {self.n} samples over {self.duration():.3f} s "
            f"({', '.join(self.channels())})"
        )


@dataclass
class Dataset:
    """Measured channels, split into one :class:`Block` per experiment."""

    name: str = "dataset"
    blocks: list[Block] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- introspection -------------------------------------------------
    @property
    def kind(self) -> str:
        return self.blocks[0].kind if self.blocks else "empty"

    @property
    def n(self) -> int:
        return sum(b.n for b in self.blocks)

    def channels(self) -> list[str]:
        seen = {ch for b in self.blocks for ch in b.columns}
        return [ch for ch in DATA_CHANNELS if ch in seen]

    def has(self, channel: str) -> bool:
        return any(channel in b.columns for b in self.blocks)

    def describe(self) -> str:
        kinds = {b.kind for b in self.blocks}
        kind = "steady-state" if kinds == {"steady"} else "time series"
        return (
            f"{self.name}: {kind}, {len(self.blocks)} block(s), {self.n} rows; "
            f"channels {', '.join(self.channels()) or 'none'}"
        )

    def __len__(self) -> int:
        return self.n

    # -- construction --------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path, *, aliases: Mapping[str, str] | None = None) -> "Dataset":
        """Read a delimited file into a :class:`Dataset`."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Dataset {p} does not exist.")
        head, rows = _read_table(p.read_text())
        return cls.from_rows(
            {h: [r[i] for r in rows] for i, h in enumerate(head)},
            name=p.name,
            aliases=aliases,
        )

    @classmethod
    def from_rows(
        cls,
        columns: Mapping[str, Sequence[float]],
        *,
        name: str = "dataset",
        aliases: Mapping[str, str] | None = None,
    ) -> "Dataset":
        """Build a dataset from a mapping of column name -> values."""
        notes: list[str] = []
        raw: dict[str, np.ndarray] = {}
        time: np.ndarray | None = None
        group: np.ndarray | None = None

        for key, values in columns.items():
            arr = np.asarray(values, dtype=float)
            canon = _canonical(str(key), aliases)
            if canon is None:
                notes.append(f"ignored column {str(key).strip()!r} (not a model channel)")
                continue
            if canon in TIME_COLUMNS:
                if time is not None:
                    notes.append(f"ignored extra time column {str(key).strip()!r}")
                    continue
                time = arr
            elif canon in GROUP_COLUMNS:
                group = arr
            elif canon in raw:
                notes.append(f"ignored duplicate column {str(key).strip()!r}")
            else:
                raw[canon] = arr

        if not raw:
            raise ConfigError(
                "Dataset has no recognised channels. Known channels: " + ", ".join(DATA_CHANNELS)
            )

        keep = {ch: arr for ch, arr in raw.items() if np.isfinite(arr).any()}
        for ch in raw:
            if ch not in keep:
                notes.append(f"column {ch} is entirely empty")

        # efficiency logged in percent is a very common accident
        if "efficiency" in keep:
            peak = float(np.nanmax(np.abs(keep["efficiency"])))
            if peak > 1.5:
                keep["efficiency"] = keep["efficiency"] / 100.0
                notes.append("rescaled efficiency from percent to a 0..1 fraction")

        _dq_from_phase(keep, notes)
        _angle_housekeeping(keep, notes)

        labels: np.ndarray | None = None
        if group is not None:
            labels = group
        elif time is not None:
            # a reset time base marks a new experiment
            jumps = np.concatenate(([True], np.diff(time) <= 0))
            if jumps.sum() > 1:
                labels = np.cumsum(jumps) - 1
                notes.append(f"split into {int(jumps.sum())} experiments on a time-base reset")

        blocks: list[Block] = []
        if time is None:
            rows = np.all(np.isfinite(np.vstack([keep[c] for c in keep])), axis=0)
            cols = {ch: arr[rows] for ch, arr in keep.items()}
            if not cols:
                raise ConfigError("Dataset has no usable rows.")
            blocks.append(Block("steady", None, cols))
        else:
            if labels is None:
                labels = np.zeros(len(time), dtype=int)
            order = np.argsort(labels, kind="stable")
            for gid in dict.fromkeys(labels[order]):
                sel = order[labels[order] == gid]
                sel = sel[np.isfinite(time[sel])]
                sel = sel[np.argsort(time[sel], kind="stable")]
                cols = {ch: arr[sel][np.isfinite(arr[sel])] for ch, arr in keep.items()}
                cols = {ch: arr for ch, arr in cols.items() if arr.size}
                t = time[sel]
                if t.size and cols:
                    label = f"run {int(gid) + 1}" if labels.max() > 0 else "run 1"
                    blocks.append(Block(label, t, cols))

        for b in blocks:
            if b.time is None:
                if "voltage_V" not in b.columns and not _dq_steady_inputs(b):
                    raise ConfigError(
                        "Steady-state rows need a voltage_V column (the drive "
                        "voltage each row was measured at), or -- for a brushless "
                        "machine -- current_d_A and current_q_A plus a speed."
                    )
            elif "voltage_V" not in b.columns and not _dq_transient_inputs(b):
                if any(ch in b.columns for ch in DQ_CHANNELS):
                    notes.append(
                        f"{b.label}: no {' or '.join(DQ_INPUT_CHANNELS)} column -- the fit "
                        "will prescribe the dq input, and a prescribed dq input is a "
                        "constant v_d/v_q pair, which constrains resistance and flux but "
                        "hardly anything else"
                    )
                else:
                    notes.append(f"{b.label}: no voltage_V column -- the fit will prescribe the input")

        return cls(name=name, blocks=blocks, notes=notes)

    @classmethod
    def from_arrays(
        cls,
        *,
        time: Sequence[float] | None = None,
        name: str = "dataset",
        **channels: Sequence[float],
    ) -> "Dataset":
        """Convenience constructor: ``from_arrays(time=t, motor_rpm=y, ...)``."""
        cols: dict[str, Sequence[float]] = {}
        if time is not None:
            cols["time"] = time
        cols.update(channels)
        return cls.from_rows(cols, name=name)


# ----------------------------------------------------------------------
# model predictions
# ----------------------------------------------------------------------
def _steady_channels(system: DriveSystem, voltage: np.ndarray) -> dict[str, np.ndarray]:
    """Closed-form equilibrium channels for a vector of drive voltages.

    Mirrors :func:`drivetrain.simulate._enrich` so steady and time-domain
    predictions use exactly the same definitions.
    """
    v = np.asarray(voltage, dtype=float)
    w = np.array([system.steady_speed(float(x)) for x in v])          # rad/s, motor shaft
    i = np.array([system.group.current_at(float(x), float(wi)) for x, wi in zip(v, w)])
    torque = np.array([system.group.torque_at(float(x), float(wi)) for x, wi in zip(v, w)])

    gb = system.gearbox
    out_speed = w * gb.ratio
    # `torque` is motor-shaft torque, so the power is taken at the motor speed:
    # the gearbox trades torque for speed and the ratio cancels
    # (tau_out = tau * efficiency / ratio, omega_out = omega * ratio), which leaves
    # tau * omega * efficiency.  Multiplying motor torque by output speed
    # under-reports a reducer by its ratio and over-reports a multiplier.
    power_out = torque * w * gb.efficiency
    power_in = v * i
    r = system.group.motor.resistance / system.group.count
    copper = i**2 * r
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(np.abs(power_in) > 1e-12, power_out / power_in, 0.0)

    out: dict[str, np.ndarray] = {
        "voltage_V": v,
        "motor_rpm": rad_s_to_rpm(w),
        "motor_rad_s": w,
        "output_rpm": rad_s_to_rpm(out_speed),
        "output_rad_s": out_speed,
        "current_A": i,
        "torque_Nm": torque,
        "power_in_W": power_in,
        "power_out_W": power_out,
        "copper_loss_W": copper,
        "efficiency": np.clip(eff, 0.0, 1.0),
    }
    wheel = system.wheel
    if wheel is not None:
        ground = out_speed * wheel.radius
        out["speed_m_s"] = ground
        out["speed_kmh"] = ground * 3.6
    return out


# -- brushless (dq) prediction ----------------------------------------------

def _dq_steady_inputs(block: Block) -> bool:
    """True when settled rows give the operating point rather than a drive voltage."""
    return all(ch in block.columns for ch in DQ_STEADY_INPUT_CHANNELS)


def _dq_transient_inputs(block: Block) -> bool:
    """True when a timeseries block logs the voltages the inverter actually applied.

    Settled rows do not count even when they carry the two columns: a settled row has
    no applied voltage of its own, and there v_d/v_q are the *prediction* the
    inductances show up in, not an input to replay.
    """
    return block.time is not None and all(ch in block.columns for ch in DQ_INPUT_CHANNELS)


def _dq_speed(system: DriveSystem, block: Block) -> np.ndarray | None:
    """Motor-shaft speed in rad/s from whichever motion channel the rows carry."""
    if "motor_rad_s" in block.columns:
        return np.asarray(block.columns["motor_rad_s"], dtype=float)
    if "motor_rpm" in block.columns:
        return np.asarray(block.columns["motor_rpm"], dtype=float) * (math.tau / 60.0)
    gb = system.gearbox
    if abs(gb.ratio) < 1e-12:
        raise ConfigError(
            f"{block.label}: the gearbox ratio is zero, so no output speed can be "
            "referred back to the motor shaft."
        )
    if "output_rad_s" in block.columns:
        return np.asarray(block.columns["output_rad_s"], dtype=float) / gb.ratio
    if "output_rpm" in block.columns:
        return np.asarray(block.columns["output_rpm"], dtype=float) * (math.tau / 60.0) / gb.ratio
    wheel = system.wheel
    if wheel is not None:
        if "speed_m_s" in block.columns:
            ground = np.asarray(block.columns["speed_m_s"], dtype=float)
        elif "speed_kmh" in block.columns:
            ground = np.asarray(block.columns["speed_kmh"], dtype=float) / 3.6
        else:
            return None
        if abs(wheel.radius) < 1e-12:
            raise ConfigError(
                f"{block.label}: the wheel radius is zero, so ground speed cannot be "
                "referred back to the motor shaft."
            )
        return ground / (wheel.radius * gb.ratio)
    return None


def _dq_block_error(system: DriveSystem, block: Block) -> str | None:
    """Why this block's dq channels cannot be replayed, or ``None`` when they can.

    Checked before the first probe so the answer names the missing column instead
    of the generic "the model cannot produce ...", which reads like a bug in the
    model when it is a gap in the log.
    """
    logged = [ch for ch in DQ_CHANNELS if ch in block.columns]
    if not logged:
        return None
    if block.time is None:
        if _dq_steady_inputs(block):
            if _dq_speed(system, block) is not None:
                return None
            return (
                f"{block.label}: settled brushless rows need a speed channel as well as "
                f"{', '.join(DQ_STEADY_INPUT_CHANNELS)}. The term that separates Ld from "
                "Lq is the back-EMF, which is proportional to speed: without a speed "
                "measurement the two inductances are one unknown again."
            )
        return (
            f"{block.label}: settled brushless rows need {', '.join(DQ_STEADY_INPUT_CHANNELS)} "
            "-- the currents the drive is holding at that operating point -- plus a speed "
            "channel. Model channels: " + ", ".join(sorted(STEADY_CHANNELS))
        )
    if _dq_transient_inputs(block):
        return None
    return (
        f"{block.label}: logs {', '.join(logged)}, which only the dq plant can predict, and "
        "the dq plant needs the applied " + " and ".join(DQ_INPUT_CHANNELS) + ". Every FOC "
        "logger reports those two; log them, or drop the dq channels and fit the brushed "
        "collapse instead."
    )


def _dq_steady_channels(system: DriveSystem, block: Block) -> dict[str, np.ndarray]:
    """Predictions for settled brushless rows: currents and speed measured, voltages algebraic.

    This is the measurement a brushed log cannot make.  With the two axes logged
    separately they stop being one current:

        v_d = R i_d - w_e Lq i_q
        v_q = R i_q + w_e (Ld i_d + lambda)

    so Lq appears only in v_d and Ld only in v_q, and above a few rad/s the
    cross-coupling terms dwarf the R i terms.  That is what makes the two
    inductances identifiable at all; under the i_d=0 collapse they are not
    (docs/fitting.md, "Fitting a brushless machine").

    ``current_A`` here is one machine's phase current, ``hypot(i_d, i_q)``, which
    is what a drive reports for itself.  The brushed path reports the total across
    the motor group, so on a two-motor system the two conventions differ by the
    motor count; a log that carries both is read as two different instruments.
    """
    spec = machine_of(system).spec
    count = system.group.count
    i_d = np.asarray(block.columns["current_d_A"], dtype=float)
    i_q = np.asarray(block.columns["current_q_A"], dtype=float)
    w = _dq_speed(system, block)
    if w is None:
        raise ConfigError(f"{block.label}: no speed channel to evaluate the back-EMF at.")
    n = int(w.size)
    v_d = np.empty(n)
    v_q = np.empty(n)
    tau = np.empty(n)
    for k in range(n):
        v_d[k], v_q[k] = spec.voltage_steady(float(i_d[k]), float(i_q[k]), float(w[k]))
        tau[k] = count * spec.torque_at(float(i_d[k]), float(i_q[k]))

    gb = system.gearbox
    out_speed = w * gb.ratio
    # tau is motor-shaft torque, so the power is taken at motor speed: see
    # _steady_channels for why the gear ratio cancels and the efficiency does not
    power_out = tau * w * gb.efficiency
    # 1.5 because the Park transform here is amplitude-invariant: the dq pair
    # describes three windings, and one machine's air-gap power is
    # 1.5 (v_d i_d + v_q i_q).  Same factor as dq_sim.DqSimulationResult.channels().
    power_in = count * 1.5 * (v_d * i_d + v_q * i_q)
    copper = count * 1.5 * spec.resistance * (i_d ** 2 + i_q ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(np.abs(power_in) > 1e-12, power_out / power_in, 0.0)

    out: dict[str, np.ndarray] = {
        "current_d_A": i_d,
        "current_q_A": i_q,
        "voltage_d_V": v_d,
        "voltage_q_V": v_q,
        "voltage_V": np.hypot(v_d, v_q),
        "current_A": np.hypot(i_d, i_q),
        "motor_rad_s": w,
        "motor_rpm": rad_s_to_rpm(w),
        "output_rad_s": out_speed,
        "output_rpm": rad_s_to_rpm(out_speed),
        "torque_Nm": tau,
        "power_in_W": power_in,
        "power_out_W": power_out,
        "copper_loss_W": copper,
        "efficiency": np.clip(eff, 0.0, 1.0),
    }
    wheel = system.wheel
    if wheel is not None:
        ground = out_speed * wheel.radius
        out["speed_m_s"] = ground
        out["speed_kmh"] = ground * 3.6
    return out


def _dq_solver_dt(system: DriveSystem, block: Block, t: np.ndarray, max_steps: int) -> float:
    """Solver step for the dq plant, set by the block's own top speed.

    The dq current loop has a complex pole pair whose imaginary part is the
    electrical speed, so RK4's stable step shrinks as the machine speeds up: on
    ``bldc_pod`` it is 6.25e-4 s at standstill but 1.80e-4 s at base speed and
    9.0e-5 s at twice base.  A step chosen from the electrical time constant
    alone -- which is what the brushed path uses -- is unstable up there.
    """
    spec = machine_of(system).spec
    w = _dq_speed(system, block)
    if w is not None and w.size:
        top = float(np.nanmax(np.abs(w)))
    else:
        top = float(system.no_load_speed())
    if not np.isfinite(top):
        top = 0.0
    limit = max_step_time(spec, top)
    spacing = float(np.median(np.diff(t))) if len(t) > 1 else limit
    step = min(0.5 * limit, spacing) if spacing > 0 else 0.5 * limit
    span = float(t[-1] - t[0])
    if span > 0 and span / step > max_steps:
        step = span / float(max_steps)
        if step > limit:
            raise ConfigError(
                f"{block.label}: the dq plant needs at least {int(np.ceil(span / limit))} "
                f"steps to stay inside the stable step {limit:.3e} s at {top:.0f} rad/s "
                f"(dq pole magnitude {pole_magnitude(spec, top):.0f} 1/s), but max_steps="
                f"{max_steps} caps it lower.  Raise max_steps, shorten the log, or fit "
                "the brushed collapse."
            )
    return step


def _dq_transient_channels(
    system: DriveSystem,
    block: Block,
    *,
    dt: float | None,
    max_steps: int,
    engine: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Replay the logged dq voltages through the dq plant and predict every channel.

    The state starts where the log starts: a block that logs i_d, i_q and speed is
    initialised from its own first samples, which is what a real drive is doing at
    the instant logging begins.  A block that logs none of them starts from rest.
    The gearbox, its efficiency and the system's own load model are all on the
    other side of the shaft, so a vehicle log decelerates the way the vehicle
    decelerates.
    """
    t = np.asarray(block.time, dtype=float)
    t = t - float(t[0])
    span = float(t[-1])
    if span <= 0:
        raise ConfigError(f"{block.label}: time base spans {span} s, need at least two samples.")

    step = dt if dt is not None else _dq_solver_dt(system, block, t, max_steps)
    # n points means n - 1 steps, and the grid has to reach the log's last sample:
    # stop one step short and that final residual is the model clamped at its own
    # end (np.interp does not extrapolate) rather than a prediction
    n = int(max(2, min(max_steps + 1, round(span / step) + 1)))
    step = span / (n - 1)
    grid = np.arange(n) * step

    held: dict[str, np.ndarray] = {}
    missing = [ch for ch in DQ_INPUT_CHANNELS if ch not in block.columns]
    if missing:
        raise ConfigError(
            f"{block.label}: a dq timeseries needs the applied "
            f"{' and '.join(DQ_INPUT_CHANNELS)} (missing " + ", ".join(missing)
            + ").  Those two voltages are what drive the dq plant; without them the fit "
            "can only prescribe a constant pair, and a constant pair says nothing about "
            "the inductances.  Log them, or fit the brushed collapse with voltage_V."
        )
    for ch in DQ_INPUT_CHANNELS:
        vv = np.asarray(block.columns[ch], dtype=float)
        # zero-order hold, as for the brushed path: a current loop holds its last
        # command between updates, and a fit at a finer solver step must not
        # invent edges the drive never made
        idx = np.searchsorted(t, grid, side="right") - 1
        np.clip(idx, 0, vv.size - 1, out=idx)
        held[ch] = vv[idx]

    w = _dq_speed(system, block)
    w0 = float(w[0]) if w is not None and w.size else 0.0
    i0 = float(block.columns["current_d_A"][0]) if "current_d_A" in block.columns else 0.0
    q0 = float(block.columns["current_q_A"][0]) if "current_q_A" in block.columns else 0.0

    # the dq plant has no closed form, so the brushed 'exact' engine does not exist here
    dq_engine = engine if engine in DQ_ENGINES else "rk4"
    res = simulate_dq(
        system, held["voltage_d_V"], held["voltage_q_V"], t=grid,
        engine=dq_engine, samples=None,
        initial=DqState(i_d=i0, i_q=q0, rot_vel=w0, angle=0.0),
    )
    ch = res.channels()
    ch.pop("time_s", None)
    if "rotor_angle_rad" in block.columns:
        measured = np.asarray(block.columns["rotor_angle_rad"], dtype=float)
        # an encoder's zero is arbitrary: only the *change* in angle is comparable,
        # so the model's angle is shifted to meet the log's first sample.  Both are
        # unwrapped (at load time for the data, continuously for the model), so the
        # residual never sees a 2*pi jump.
        ch["rotor_angle_rad"] = (ch["rotor_angle_rad"] - float(ch["rotor_angle_rad"][0])
                                 + float(measured[0]))
    return ch, res.t


def _solver_dt(system: DriveSystem, t: np.ndarray, max_steps: int) -> float:
    """Solver step: accurate enough, never finer than the data, capped in count."""
    taus = [system.mech_time_constant, system.electrical_time_constant]
    taus = [x for x in taus if np.isfinite(x) and x > 0]
    auto = max(min(taus) / 50.0, 1e-7) if taus else 1e-3
    spacing = float(np.median(np.diff(t))) if len(t) > 1 else auto
    step = min(auto, spacing) if spacing > 0 else auto
    span = float(t[-1] - t[0])
    if span > 0 and span / step > max_steps:
        step = span / float(max_steps)
    return step


def _model_channels(
    system: DriveSystem,
    block: Block,
    *,
    engine: str,
    dt: float | None,
    excitation: str,
    voltage: float | None,
    max_steps: int,
    period: float | None = None,
    ramp_time: float | None = None,
    voltage_hold: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Predict every channel on the model's own grid.

    Returns the channel dict plus the model time base -- ``None`` for
    steady-state rows, whose predictions line up one-for-one with the data.
    """
    if block.time is None:
        if _dq_steady_inputs(block):
            return _dq_steady_channels(system, block), None
        return _steady_channels(system, block.columns["voltage_V"]), None

    if _dq_transient_inputs(block):
        return _dq_transient_channels(system, block, dt=dt, max_steps=max_steps, engine=engine)

    t = np.asarray(block.time, dtype=float)
    t = t - t[0]
    span = float(t[-1])
    if span <= 0:
        raise ConfigError(f"{block.label}: time base spans {span} s, need at least two samples.")

    step = dt if dt is not None else _solver_dt(system, t, max_steps)
    n = int(max(2, min(max_steps, round(span / step))))
    step = span / n
    grid = np.arange(n) * step

    if "voltage_V" in block.columns:
        vv = np.asarray(block.columns["voltage_V"], dtype=float)
        if voltage_hold:
            # the engines apply v[i] across step i, so a logged drive voltage is
            # held between samples, not ramped into them: a square wave logged on
            # its own grid then replays exactly, and a fit at a finer solver step
            # does not invent edges the drive never made
            idx = np.searchsorted(t, grid, side="right") - 1
            np.clip(idx, 0, vv.size - 1, out=idx)
            v = vv[idx]
        else:
            v = np.interp(grid, t, vv)
    else:
        if voltage is None:
            raise ConfigError(
                f"{block.label} has no voltage_V column, so the input must be given "
                "with voltage= (a drive level in volts)."
            )
        v = voltage_signal(
            excitation, system, span, dt=step, value=voltage,
            period=period, ramp_time=ramp_time,
        )
        if len(v) != n:  # voltage_signal builds its own grid; make them agree
            v = np.interp(grid, np.arange(len(v)) * (span / (len(v) - 1)), v)

    res = simulate(system, v, n * step, dt=step, engine=engine, samples=None)
    ch = res.channels()
    # the same speeds in rad/s, for logs that come off an encoder rather than a
    # tachometer; derived here so simulate()'s own channel list stays unchanged
    ch["motor_rad_s"] = res.motor_speed
    if res.output_speed is not None:
        ch["output_rad_s"] = res.output_speed
    return ch, res.t


# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------
@dataclass
class FitParameter:
    """One parameter the optimiser may move."""

    path: str
    label: str
    unit: str
    start: float
    value: float
    log: bool
    bounds: tuple[float, float]
    integer: bool = False
    stderr: float = 0.0

    @property
    def relative_stderr(self) -> float:
        scale = abs(self.value) or abs(self.start)
        if not scale:
            return math.inf
        return abs(self.stderr / scale)

    @property
    def stderr_text(self) -> str:
        """1-sigma as text: ``unbounded`` when the data cannot pin the value down."""
        if not math.isfinite(self.stderr) or self.relative_stderr > 20.0:
            return "unbounded"
        return f"{self.stderr:.2g}"

    def at_bound(self, rel: float = 1e-4) -> bool:
        """True when the fitted value sits on a *finite* bound.

        The tolerance is relative to the parameter's own magnitude: a damping of
        1e-6 has bounds down at 1e-12, and an absolute 1e-4 tolerance would call
        every value in that range "pinned".  An open-ended bound is not a bound
        either -- comparing against it yields infinities on both sides, and
        ``inf <= inf`` is true.
        """
        lo, hi = self.bounds
        base = max(abs(self.value), 1e-12)
        if math.isfinite(lo) and abs(self.value - lo) <= rel * max(abs(lo), base):
            return True
        return math.isfinite(hi) and abs(self.value - hi) <= rel * max(abs(hi), base)


def _read_value(system: DriveSystem, path: str) -> Any:
    try:
        value = get_path(system, path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a user-facing error
        raise ConfigError(f"Unknown parameter path {path!r}: {exc}") from exc
    return value


def _meta(path: str, value: Any) -> tuple[str, str]:
    meta = presets.param_meta(path, value)
    return str(meta.get("label", path)), str(meta.get("unit", ""))


def _param_bounds(
    path: str, start: float, explicit: Mapping[str, tuple[float, float]] | None
) -> tuple[float, float]:
    """Plausible search range: explicit, then physical, then catalogue, then start-relative.

    The catalogue range is a what-if slider range, not a physical one, so it is
    widened by two orders of magnitude: a fitted value that lands just outside
    the slider range must not be pinned there by the search box, and a start
    value that is wrong by a factor of 50 still has to be able to reach the data.
    """
    if explicit and path in explicit:
        lo, hi = (float(explicit[path][0]), float(explicit[path][1]))
        return (min(lo, hi), max(lo, hi))

    hard = HARD_BOUNDS.get(path)
    signed = path in SIGNED_PATHS
    meta = presets.PARAM_META.get(path)
    if meta is not None:
        lo_c = float(meta.get("min", -math.inf))
        hi_c = float(meta.get("max", math.inf))
    else:
        span = abs(start) * 50.0
        lo_c, hi_c = (start - span, start + span) if start else (-1.0, 1.0)

    # the slider range, widened by two orders of magnitude on each side
    lo = lo_c / 100.0 if lo_c > 0 else (lo_c * 100.0 if lo_c < 0 else 0.0)
    hi = hi_c * 100.0 if hi_c > 0 else (hi_c / 100.0 if hi_c < 0 else math.inf)
    if signed:
        lo = min(lo, -hi if math.isfinite(hi) else -math.inf)
    else:
        lo = max(lo, 0.0)

    # whatever the catalogue says, a parameter may be wrong by orders of magnitude
    if start > 0:
        lo, hi = min(lo, start / 1e4), max(hi, start * 1e4)
    elif start < 0:
        lo, hi = min(lo, start * 1e4), max(hi, start / 1e4)
    elif signed and math.isfinite(hi) and hi > 0:
        # a fitted-from-nothing signed quantity (no friction fitted yet) still
        # needs room either way
        lo, hi = min(lo, -1e-3 * hi), max(hi, 1e-3 * hi)

    if hard is not None:
        lo = max(lo, hard[0])
        hi = min(hi, hard[1])
    # a magnitude searched on a log scale needs a strictly positive floor, and
    # six decades below the start is room enough for a wildly wrong guess
    if start > 0 and not signed and not lo > 0:
        lo = start / 1e6
    # always let the optimiser start where it is standing
    eps = max(1e-12, 1e-6 * abs(start))
    if start > 0 or signed:
        lo = min(lo, start - eps)
        hi = max(hi, start + eps)
    if not (hi > lo):
        raise ConfigError(f"Cannot build a search range for {path!r} (start {start}).")
    return lo, hi


def _expand_free(
    free: Sequence[str] | str,
    system: DriveSystem,
    known: Mapping[str, float],
    notes: list[str] | None = None,
    channels: set[str] | None = None,
) -> list[str]:
    """Expand alias names, drop paths this system does not have, keep order.

    ``all`` means every parameter this configuration exposes; like a group
    alias, it silently skips paths the system has not got, ones that are not
    numeric, ones already pinned as known constants, and ones this
    configuration never evaluates (a group member with no effect on the
    residuals is only ever noise).  Naming such a path explicitly still fits
    it, and the report then says it has no sensitivity.
    """
    tokens = (free,) if isinstance(free, str) else tuple(free)
    tokens = [t for t in (str(tok).strip() for tok in tokens) if t]

    blanket: set[str] = set()
    if any(t.lower() == FREE_ALL for t in tokens):
        tokens = [t for t in tokens if t.lower() != FREE_ALL]
        blanket = {
            str(entry["path"]) for entry in presets.field_catalog(system)
            if str(entry["path"]) not in STATE_PATHS
        }
        tokens = tokens + sorted(blanket)

    out: list[str] = []
    skipped_inert: list[str] = []
    motor_map = dq_motor_map(system)
    renamed: dict[str, str] = {}
    for token in tokens:
        named = token.lower() in FREE_ALIASES or token in blanket
        paths = FREE_ALIASES.get(token.lower(), (token,))
        for path in paths:
            source = path
            if path in motor_map:
                # a brushless machine keeps its constants under spec.*, and its
                # kt and kv are one flux linkage seen from two sides
                path = motor_map[path]
                renamed[source] = path
            if path in STATE_PATHS:
                raise ConfigError(f"{path!r} is a state variable, not a fittable parameter.")
            if path in known:
                if named:      # a group alias simply skips known constants
                    continue
                raise ConfigError(
                    f"{path!r} is listed both as free and as a known constant; pick one."
                )
            try:
                value = _read_value(system, path)
            except ConfigError:
                if named:      # an alias covers paths this system does not have
                    continue
                raise
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                if named:
                    continue
                raise ConfigError(f"{path!r} is not a numeric parameter (got {type(value).__name__}).")
            if named and not math.isfinite(float(value)):
                # an unlimited limit (max_current = inf) cannot be searched; only
                # an explicitly named path should raise on that
                continue
            if named:
                why = structural_inert(system, path, channels)
                if why:
                    if notes is not None:
                        skipped_inert.append(f"{path} ({why})")
                    continue
            if path not in out:
                out.append(path)
    if notes is not None and renamed:
        notes.append(
            "brushless machine, so the free set was re-pointed at the constants this "
            "system can set: " + ", ".join(sorted({f"{a} -> {b}" for a, b in renamed.items()}))
            + ". kt and kv are one flux linkage here (kt = 1.5 p lambda, kv = p lambda), so "
              "both names set the same parameter."
        )
    if notes is not None and skipped_inert:
        notes.append(
            "left out of the free set by the group: " + ", ".join(skipped_inert)
            + " -- this configuration never evaluates them, so they could only ever be noise."
              " Name a path explicitly to fit one anyway."
        )
    if not out:
        raise ConfigError(
            "Nothing to fit. Pass free= with parameter paths, or one of: "
            + ", ".join(sorted(FREE_ALIASES))
        )
    return out


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------
@dataclass
class FitResult:
    """A fitted model plus the statistics needed to judge the fit."""

    system: DriveSystem
    parameters: list[FitParameter]
    rms: dict[str, float]
    max_error: dict[str, float]
    points: dict[str, int]
    weights: dict[str, float]
    cost: float
    normalized_rms: float
    condition: float
    correlations: np.ndarray
    success: bool
    message: str
    evaluations: int
    iterations: int
    engine: str
    dt: float
    dataset: str
    kind: str
    warnings: list[str] = field(default_factory=list)
    degenerate: bool = False
    ties: dict[str, str] = field(default_factory=dict)
    known: dict[str, float] = field(default_factory=dict)
    weights_basis: dict[str, str] = field(default_factory=dict)
    relative_rms: dict[str, float] = field(default_factory=dict)
    starts: dict[str, tuple[float, str]] = field(default_factory=dict)
    _blocks: list[Block] = field(default_factory=list, repr=False)
    _opts: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- reporting -----------------------------------------------------
    def parameter_table(self) -> str:
        rows = []
        for p in self.parameters:
            rows.append(
                f"  {p.path:<34}{p.start:>12.6g}{p.value:>13.6g}{p.stderr_text:>11}  {p.unit}"
            )
        head = f"  {'parameter':<34}{'start':>12}{'fitted':>13}{'1-sigma':>11}"
        return "\n".join([head] + rows)

    def report(self) -> str:
        lines = [
            f"fit: {self.dataset} ({self.kind}, {sum(self.points.values())} residuals)",
            f"engine={self.engine}  dt={self.dt:.3g} s  evaluations={self.evaluations}"
            f"  iterations={self.iterations}  status={self.message}",
            "",
            "free parameters",
            self.parameter_table(),
            "",
            "residual rms per channel",
        ]
        for ch, value in sorted(self.rms.items()):
            rel = self.relative_rms.get(ch)
            lines.append(
                f"  {ch:<16}{value:>11.4g}   (n={self.points[ch]}, "
                f"weight {self.weights[ch]:.3g}, max |err| {self.max_error[ch]:.3g}"
                + (f", {rel:.2g} of the channel's own spread" if rel is not None else "")
                + ")"
            )
        lines.append(
            "  normalized rms is the residual against the per-channel noise estimate: "
            "~1 = as good as the data allows, >>1 = the model cannot follow the data, "
            "<<1 = the weights overestimate the noise"
        )
        lines += [
            "",
            f"residual rms     {self.normalized_rms:.3g} in weight units "
            "(1.0 = the fit matches the data as well as the weights allow)",
            f"condition number {self.condition:.3g}"
            + ("  (degenerate: some free parameters cannot be separated)" if self.degenerate else ""),
        ]
        if self.weights_basis:
            lines.append(
                "  weights are 1-sigma noise per channel -- "
                + "; ".join(f"{ch}: {how}" for ch, how in sorted(self.weights_basis.items()))
            )
        if self.starts:
            lines += ["", "starting values read off the log, not off the catalogue"]
            for path, (value, why) in sorted(self.starts.items()):
                lines.append(f"  {path:<34}{value:>12.6g}  {why}")
        pairs = self.strong_correlations()
        if pairs:
            lines += ["", "parameter correlations (|r| > 0.9)"]
            lines += [f"  {a} <-> {b}  {c:+.3f}" for a, b, c in pairs]
        if self.known:
            lines += ["", "known constants"]
            for path, value in sorted(self.known.items()):
                lines.append(f"  {path:<34}{value:>12.6g}")
        if self.ties:
            lines += ["", "tied parameters"]
            for target, source in sorted(self.ties.items()):
                lines.append(f"  {target:<34}= {source}")
        if self.warnings:
            lines += ["", "warnings"]
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    # -- outputs -------------------------------------------------------
    @property
    def by_path(self) -> dict[str, FitParameter]:
        """Fitted parameters keyed by path, for scripting a result."""
        return {p.path: p for p in self.parameters}

    def strong_correlations(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Parameter pairs the data cannot tell apart (|correlation| > *threshold*)."""
        names = [p.path for p in self.parameters]
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                try:
                    value = float(self.correlations[i, j])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(value) and abs(value) > threshold:
                    pairs.append((names[i], names[j], value))
        return sorted(pairs, key=lambda item: -abs(item[2]))

    def save(self, path: str | Path) -> None:
        """Write the fitted model as a config the CLI and GUI can load."""
        presets.save(str(path), self.system)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary of the fit (model config included)."""
        return {
            "dataset": self.dataset,
            "kind": self.kind,
            "engine": self.engine,
            "dt": self.dt,
            "cost": self.cost,
            "normalized_rms": self.normalized_rms,
            "condition": self.condition,
            "success": self.success,
            "message": self.message,
            "evaluations": self.evaluations,
            "rms": self.rms,
            "relative_rms": self.relative_rms,
            "weights": self.weights,
            "weights_basis": self.weights_basis,
            "starts": {p: {"value": v, "basis": why} for p, (v, why) in self.starts.items()},
            "known": self.known,
            "ties": self.ties,
            "warnings": self.warnings,
            "parameters": [
                {
                    "path": p.path, "start": p.start, "value": p.value,
                    "stderr": p.stderr, "unit": p.unit, "log": p.log,
                    "bounds": list(p.bounds),
                }
                for p in self.parameters
            ],
            # to_json() is the round-trip-safe form (it stringifies infinities
            # and the gear-ratio convention), so the result is plain JSON.
            "model": json.loads(presets.to_json(self.system)),
        }

    def predict(self) -> list[dict[str, Any]]:
        """Measured vs modelled traces on each block's own time base."""
        out = []
        for block in self._blocks:
            model, model_t = _model_channels(self.system, block, **self._opts)
            entry: dict[str, Any] = {"label": block.label, "kind": block.kind}
            t: np.ndarray | None = None
            if block.time is not None:
                t = np.asarray(block.time, float)
                t = t - float(t[0])
                entry["time"] = t
            for ch in block.channels():
                entry.setdefault("data", {})[ch] = block.columns[ch]
                if ch in model:
                    entry.setdefault("model", {})[ch] = _sample(model[ch], model_t, t)
            out.append(entry)
        return out


def _sample(values: np.ndarray, model_t: np.ndarray | None, data_t: np.ndarray | None) -> np.ndarray:
    """Resample a model channel onto the measured points."""
    arr = np.asarray(values, dtype=float)
    if model_t is None or data_t is None:
        return arr
    return np.interp(data_t, model_t, arr)


def _motor_omega(block: Block) -> np.ndarray | None:
    """Motor-shaft angular velocity in rad/s from whichever speed channel is logged."""
    if "motor_rad_s" in block.columns:
        return np.asarray(block.columns["motor_rad_s"], dtype=float)
    if "motor_rpm" in block.columns:
        return np.asarray(block.columns["motor_rpm"], dtype=float) * (math.tau / 60.0)
    return None


def _damping_gain(system: DriveSystem, path: str) -> float:
    """How much total motor-side damping one unit of ``path`` buys.

    Measured, not derived: gear ratios, efficiencies and motor count all sit
    between a parameter and the damping the equations actually use.
    """
    base = float(_read_value(system, path))
    step = max(abs(base) * 0.05, 1e-12)
    low = max(base - step, 0.0)
    try:
        up = set_path(system, path, base + step).total_damping
        down = set_path(system, path, low).total_damping
    except Exception:  # noqa: BLE001 - not a damping parameter, no gain
        return 0.0
    gain = (up - down) / ((base + step) - low)
    return gain if math.isfinite(gain) and gain > 0 else 0.0


DAMPING_PATHS = ("group.motor.bearing_damping", "gearbox.damping", "load.viscous")


def _warm_starts(
    system: DriveSystem, blocks: Sequence[Block], free: set[str]
) -> dict[str, tuple[float, str]]:
    """Guess the free parameters from the log itself, the way a person would.

    A catalogue preset is a different machine than the one on the bench, and the
    residual surface of a drive model is full of local minima, so where the data
    states a value directly it is worth starting there:

    * ``kv``          -- drive voltage over motor speed at the fast end
    * ``resistance``  -- the voltage left over for the current
    * ``kt``          -- inertia times standstill acceleration, over current
    * damping paths   -- the current needed to hold a settled speed
    * ``gearbox.ratio`` / ``load.wheel.radius`` -- two speed channels that differ
      by exactly the ratio or the rolling radius

    Every estimate is a median over the samples where its assumption holds, and
    anything the log cannot support is simply left at the preset value.
    """
    out: dict[str, tuple[float, str]] = {}
    rows = [b for b in blocks if b.time is not None]
    if not rows:
        return out

    def pooled(name: str) -> list[np.ndarray]:
        return [np.asarray(b.columns[name], dtype=float) for b in rows if name in b.columns]

    omegas = [_motor_omega(b) for b in rows]
    volts = pooled("voltage_V")
    currents = pooled("current_A")
    have_speed = all(o is not None for o in omegas)
    have_voltage = len(volts) == len(rows)
    have_current = len(currents) == len(rows)
    limit = system.current_limit_total

    def per_block(arrays: list[np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(a, float) for a in arrays]

    # ---- kv: the back-emf line, read off the fast end ----------------
    kv_used = system.group.kv_eff
    if "group.motor.kv" in free and have_speed and have_voltage:
        ratios: list[float] = []
        for w, v in zip(omegas, per_block(volts)):
            top = float(np.nanmax(np.abs(w)))
            if not math.isfinite(top) or top <= 0:
                continue
            keep = np.abs(w) >= 0.3 * top
            ratios.extend(float(a / b) for a, b in zip(v[keep], w[keep])
                          if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9
                          and a * b > 0)
        if len(ratios) >= 8:
            est = float(np.median(ratios))
            if math.isfinite(est) and est > 0:
                out["group.motor.kv"] = (est, "drive voltage over motor speed at the fast end")
                kv_used = est

    # ---- resistance: the voltage the back-emf did not use -------------
    if "group.motor.resistance" in free and have_speed and have_voltage and have_current:
        parts: list[float] = []
        for w, v, i in zip(omegas, per_block(volts), per_block(currents)):
            top = float(np.nanmax(np.abs(i)))
            if not math.isfinite(top) or top <= 0:
                continue
            keep = np.abs(i) >= 0.1 * top
            if math.isfinite(limit):
                keep &= np.abs(i) < 0.98 * limit      # a clipped current says nothing about R
            parts.extend(float((a - kv_used * b) / c) for a, b, c in zip(v[keep], w[keep], i[keep])
                         if math.isfinite(a) and math.isfinite(b) and math.isfinite(c))
        if len(parts) >= 8:
            est = float(np.median(parts))
            if math.isfinite(est) and est > 0:
                out["group.motor.resistance"] = (est, "the voltage left over once back-emf is subtracted")

    # ---- kt: torque = inertia x acceleration, read at standstill ------
    inertia = system.total_inertia
    if "group.motor.kt" in free and have_speed and have_current and math.isfinite(inertia) and inertia > 0:
        parts = []
        for b, w in zip(rows, omegas):
            t = np.asarray(b.time, float)
            i = np.asarray(b.columns["current_A"], float)
            dw = np.gradient(w, t)
            wtop, itop, dtop = (float(np.nanmax(np.abs(a))) for a in (w, i, dw))
            if not all(math.isfinite(x) and x > 0 for x in (wtop, itop, dtop)):
                continue
            keep = (np.abs(w) <= 0.15 * wtop) & (np.abs(i) >= 0.5 * itop) & (np.abs(dw) >= 0.2 * dtop)
            parts.extend(float(inertia * a / c) for a, c in zip(dw[keep], i[keep])
                         if math.isfinite(a) and math.isfinite(c) and a * c > 0)
        if len(parts) >= 8:
            est = float(np.median(parts))
            if math.isfinite(est) and est > 0:
                out["group.motor.kt"] = (
                    est, f"{inertia:g} kg m^2 of inertia x standstill acceleration over current")

    # ---- damping: the current that holds a settled speed --------------
    free_damping = [p for p in DAMPING_PATHS if p in free]
    if free_damping and have_speed and have_current:
        kt_used = out.get("group.motor.kt", (system.group.kt_eff,))[0]
        bal: list[float] = []
        for b, w in zip(rows, omegas):
            t = np.asarray(b.time, float)
            i = np.asarray(b.columns["current_A"], float)
            dw = np.gradient(w, t)
            wtop, dtop = float(np.nanmax(np.abs(w))), float(np.nanmax(np.abs(dw)))
            if not (math.isfinite(wtop) and wtop > 0 and math.isfinite(dtop) and dtop > 0):
                continue
            keep = (np.abs(dw) <= 0.1 * dtop) & (np.abs(w) >= 0.3 * wtop)
            for a, c, o in zip(dw[keep], i[keep], w[keep]):
                if not (math.isfinite(a) and math.isfinite(c) and math.isfinite(o)) or abs(o) < 1e-9:
                    continue
                try:
                    held = system.reflected_load_torque(o)
                except Exception:  # noqa: BLE001
                    held = 0.0
                bal.append(float((kt_used * c - held) / o))
        if len(bal) >= 8:
            est = float(np.median(bal))
            if math.isfinite(est) and est > 0:
                delta = est - system.total_damping
                if abs(delta) > 1e-15:
                    share = delta / len(free_damping)
                    for path in free_damping:
                        gain = _damping_gain(system, path)
                        if gain <= 0:
                            continue
                        current = float(_read_value(system, path))
                        value = max(current + share / gain, 1e-12)
                        out[path] = (value, f"the current that holds a settled speed "
                                            f"(total damping {est:.3g} N m s)")

    # ---- geometry: two channels that differ by exactly a ratio --------
    if "gearbox.ratio" in free:
        outs = pooled("output_rad_s") or [
            w * (math.tau / 60.0) for w in pooled("output_rpm")]
        if len(outs) == len(rows) and have_speed:
            rr = [float(a / b) for a, b in zip(np.concatenate(outs), np.concatenate(omegas))
                  if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9 and a * b > 0]
            if len(rr) >= 8:
                est = float(np.median(rr))
                if math.isfinite(est) and est > 0:
                    out["gearbox.ratio"] = (est, "output speed over motor speed")

    if "load.wheel.radius" in free:
        ground = pooled("speed_m_s") or [g / 3.6 for g in pooled("speed_kmh")]
        outs = pooled("output_rad_s") or [w * (math.tau / 60.0) for w in pooled("output_rpm")]
        if len(ground) == len(rows) and len(outs) == len(rows):
            rr = [float(a / abs(b)) for a, b in zip(np.concatenate(ground), np.concatenate(outs))
                  if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9]
            if len(rr) >= 8:
                est = float(np.median(rr))
                if math.isfinite(est) and est > 0:
                    out["load.wheel.radius"] = (est, "ground speed over wheel angular speed")

    return out


def _coarse_scan(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    logs: np.ndarray,
    *,
    factors: Sequence[float] = (0.03, 0.1, 0.3, 3.0, 10.0, 30.0),
    rounds: int = 1,
    skip: Sequence[bool] | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy decade hops, one parameter at a time.

    A drive model's residual surface has a minimum for every plausible combination
    of torque constant and damping, and a gradient method that starts on the wrong
    one stays there.  This walks each parameter out by factors of ten and keeps any
    move that actually lowers the cost, which is enough to fall into the right basin
    before the local method refines it.

    Parameters the starting configuration never evaluates (``skip``) are left where
    they are: hopping inductance off zero switches the model from an algebraic
    current loop to a second-order one, which lowers the cost by changing what the
    model *is* rather than by fitting the data better.
    """
    x = np.array(x, dtype=float)
    r = fun(x)
    best = 0.5 * float(r @ r)
    for _ in range(rounds):
        improved = False
        for j in range(len(x)):
            if skip is not None and skip[j]:
                continue
            value = math.exp(x[j]) if logs[j] else float(x[j])
            for factor in factors:
                trial = x.copy()
                nxt = value * factor
                trial[j] = math.log(nxt) if logs[j] else nxt
                trial = np.clip(trial, lo, hi)
                r = fun(trial)
                cost = 0.5 * float(r @ r)
                if cost < best * (1.0 - 1e-9):
                    best, x, improved = cost, trial, True
        if not improved:
            break
    return x, best


# ----------------------------------------------------------------------
# optimisation
# ----------------------------------------------------------------------
def _jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
              lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    f0 = fun(x)
    jac = np.empty((len(f0), len(x)))
    for j in range(len(x)):
        h = 1e-5 * max(1.0, abs(x[j]))
        xp = x.copy()
        xm = x.copy()
        xp[j] = min(x[j] + h, hi[j])
        xm[j] = max(x[j] - h, lo[j])
        if xp[j] == xm[j]:
            jac[:, j] = 0.0
            continue
        jac[:, j] = (fun(xp) - fun(xm)) / (xp[j] - xm[j])
    return jac


def _levenberg_marquardt(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_nfev: int,
    tol: float = 1e-12,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, int, str]:
    """Damped Gauss-Newton with bound clipping; no scipy required."""
    x = np.clip(np.asarray(x0, float), lo, hi)
    r = fun(x)
    cost = 0.5 * float(r @ r)
    nfev = 1
    lam = 1e-3
    iterations = 0
    message = "max iterations"
    for iterations in range(1, 201):
        jac = _jacobian(fun, x, lo, hi)
        nfev += 2 * len(x)
        g = jac.T @ r
        h = jac.T @ jac
        improved = False
        for _ in range(40):
            a = h + lam * np.diag(np.maximum(np.diag(h), 1e-12))
            try:
                step = np.linalg.solve(a, -g)
            except np.linalg.LinAlgError:
                step = -g * 1e-3
            xn = np.clip(x + step, lo, hi)
            rn = fun(xn)
            nfev += 1
            cn = 0.5 * float(rn @ rn)
            if cn < cost:
                gain = cost - cn
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10.0
            if nfev > max_nfev:
                break
        if verbose:
            print(f"  it {iterations:>3}  cost {cost:.6g}  lambda {lam:.1e}")
        if nfev >= max_nfev:
            message = "evaluation limit reached"
            break
        if not improved:
            message = "stopped: no improvement inside the bounds"
            break
        if gain < tol * max(cost, 1.0):
            message = "converged"
            break
    return x, r, cost, nfev, iterations, message


def _noise_sigma(values: np.ndarray) -> float:
    """High-frequency scatter of one channel: the sensor noise the fit should respect.

    Successive differences of order 1 and 2 are both signal *plus* noise, so each
    gives an upper bound on sigma; the tighter of the two wins.  A perfectly smooth
    synthetic trace returns ~0, which would let the fit chase solver round-off, so
    the estimate is floored a little below the channel's own range and capped at
    its standard deviation (never worse than weighting by raw scatter).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 0.0
    span = float(x.max() - x.min())
    if span <= 0:
        return 0.0
    estimates = []
    for order, norm in ((1, math.sqrt(2.0)), (2, math.sqrt(6.0))):
        d = np.diff(x, n=order)
        mad = float(np.median(np.abs(d - np.median(d))))
        if mad > 0:
            estimates.append(mad / (0.6745 * norm))
    if not estimates:
        return 0.0
    return min(max(min(estimates), 1e-9, 1e-6 * span), float(np.std(x)))


# ----------------------------------------------------------------------
# the fit itself
# ----------------------------------------------------------------------
def fit(
    dataset: Dataset | str | Path,
    *,
    preset: str | None = None,
    system: DriveSystem | None = None,
    free: Sequence[str] | str = "motor",
    known: Mapping[str, float] | None = None,
    ties: Mapping[str, str] | None = None,
    channels: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    engine: str = "exact",
    dt: float | None = None,
    voltage: float | None = None,
    excitation: str = "step",
    period: float | None = None,
    ramp_time: float | None = None,
    voltage_hold: bool = True,
    log: bool = True,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    max_nfev: int = 400,
    max_steps: int = 100_000,
    verbose: bool = False,
) -> FitResult:
    """Fit ``system`` (or ``preset``) to measured data.

    ``known`` pins paths to measured values -- the constants you already have.
    ``free`` selects what the optimiser may move, as paths or a
    :data:`FREE_ALIASES` group.  ``ties`` copies one fitted value onto another
    path afterwards (e.g. ``{"group.motor.kt": "group.motor.kv"}`` for an SI
    motor where the two constants are equal).

    Returns a :class:`FitResult` holding the fitted model, standard errors,
    per-channel residuals and identifiability warnings.

    A logged ``voltage_V`` column is replayed as a held input (zero-order hold),
    which is how the engines consume it: ``v[i]`` is applied across step ``i``.
    Pass ``voltage_hold=False`` to interpolate instead -- only right for a
    genuinely analog supply logged far faster than it changes.
    """
    ds = dataset if isinstance(dataset, Dataset) else Dataset.from_csv(dataset)
    if not ds.blocks:
        raise ConfigError("Dataset has no usable blocks.")

    base = system if system is not None else presets.build(preset or "legacy")
    dq = dq_motor_map(base)
    known_map = {str(k): float(v) for k, v in (known or {}).items()}
    if dq:
        known_map = _translate_known(known_map, dq, base)
    for path, value in known_map.items():
        _read_value(base, path)                      # validates the path
        base = set_path(base, path, value)

    notes: list[str] = []
    # what the log actually carries decides whether a dq-collapse-inert parameter is
    # inert at all, so the dataset has to be in the room when that is judged
    measured = set(ds.channels())
    paths = _expand_free(free, base, known_map, notes, measured)
    tie_map = {str(k): str(v) for k, v in (ties or {}).items()}
    if dq:
        tying = [f"{target!r}~{source!r}" for target, source in tie_map.items()
                 if dq.get(target, target) == dq.get(source, source)]
        if tying:
            notes.append(
                "dropped the tie(s) " + ", ".join(tying) + ": on a brushless machine kt and kv "
                "are one flux linkage (kt = 1.5 p lambda, kv = p lambda), so tying them "
                "together asks for a number that is already fixed"
            )
        tie_map = {dq.get(target, target): dq.get(source, source)
                   for target, source in tie_map.items()
                   if dq.get(target, target) != dq.get(source, source)}
    for target, source in tie_map.items():
        _read_value(base, target)
        _read_value(base, source)

    # ---- which channels to compare ---------------------------------
    steady = ds.kind == "steady"
    usable = STEADY_CHANNELS if steady else DATA_CHANNELS
    if isinstance(channels, str):
        channels = (channels,)
    wanted = list(channels) if channels else [ch for ch in ds.channels() if ch in usable]
    wanted = [ch for ch in wanted if ch not in INPUT_CHANNELS]

    # Which channels a block *drives on* rather than measures.  A brushed block is
    # driven by voltage_V; a dq timeseries is driven by the v_d/v_q the inverter
    # applied, so replaying it against itself would score the hold, not the model.
    # A settled dq row has no input of its own -- there the same two columns are the
    # prediction the inductances show up in -- so a channel is dropped only when it
    # is an input for every block that carries it.
    def _is_input(block: Block, ch: str) -> bool:
        return ch in INPUT_CHANNELS or (
            ch in DQ_INPUT_CHANNELS and _dq_transient_inputs(block)
        )

    wanted = [ch for ch in wanted
              if any(ch in b.columns and not _is_input(b, ch) for b in ds.blocks)]
    if channels:
        missing = [ch for ch in wanted if not ds.has(ch)]
        if missing:
            raise ConfigError(
                "Dataset has no data for: " + ", ".join(missing)
                + ". Available: " + ", ".join(ds.channels())
            )
    if not wanted:
        raise ConfigError(
            "No comparable channels. The dataset needs at least one of: "
            + ", ".join(usable)
        )

    # A brushless drive is usually instrumented on the DC rail, and that current
    # is not the current that makes torque.  Say so before anyone fits a rail
    # ammeter to a phase-current model and blames the resistance value that comes out.
    if dq and (ELECTRICAL_CHANNELS & set(wanted)):
        bus = float(getattr(base.group.motor, "spec").dc_bus_voltage)
        if math.isfinite(bus) and bus > 0:
            notes.append(
                "this machine is brushless: current_A here is the phase current that makes "
                "torque, while an ammeter on the DC rail reads 1.5 (R i + kv w) i / "
                f"{bus:.1f} V, which is not proportional to torque. If the logged current is a "
                "rail current, fit the phase current the controller reports (or convert it "
                "first), otherwise resistance and flux linkage come out wrong together."
            )

    probes: dict[str, list[str]] = {}
    opts = dict(engine=engine, dt=dt, excitation=excitation, voltage=voltage,
                max_steps=max_steps, period=period, ramp_time=ramp_time,
                voltage_hold=voltage_hold)
    for block in ds.blocks:
        blocked = _dq_block_error(base, block)
        if blocked:
            raise ConfigError(blocked)
        probe, _ = _model_channels(base, block, **opts)
        probes[block.label] = sorted(k for k in probe if k != "time")
        missing = [ch for ch in wanted
                   if ch in block.columns and not _is_input(block, ch) and ch not in probe]
        if missing:
            raise ConfigError(
                f"The model cannot produce {', '.join(missing)} for {block.label} "
                "(does the preset have a wheel, or is the channel measured at all?). "
                "Model channels: " + ", ".join(probes[block.label])
            )

    # ---- weights (1-sigma noise per channel) ------------------------
    # A weight is a noise estimate, not a preference: it says how much of a
    # channel's residual is the sensor rather than the model.  Estimated from the
    # data's own high-frequency scatter, so a 200 A current trace and a 200 rpm
    # speed trace both contribute in proportion to how badly they are measured
    # instead of in proportion to how big their units are.
    sigmas: dict[str, float] = {}
    basis: dict[str, str] = {}
    spread_of: dict[str, float] = {}
    for ch in wanted:
        data = np.concatenate([b.columns[ch] for b in ds.blocks
                               if ch in b.columns and not _is_input(b, ch)])
        spread_of[ch] = float(np.nanstd(data[np.isfinite(data)])) \
            if np.any(np.isfinite(data)) else 0.0
        if weights and ch in weights:
            sigma, how = float(weights[ch]), "given"
        else:
            estimates = [
                _noise_sigma(b.columns[ch])
                for b in ds.blocks
                if ch in b.columns and b.time is not None and not _is_input(b, ch)
            ]
            estimates = [e for e in estimates if e > 0]
            if estimates and not steady:
                sigma, how = float(np.median(estimates)), "high-frequency scatter"
            else:
                sigma, how = float(np.nanstd(data)), "data scatter (no time base)"
        if not np.isfinite(sigma) or sigma <= 0:
            sigma, how = max(float(np.nanmean(np.abs(data))), 1e-9), "data scatter (no time base)"
        sigmas[ch] = max(sigma, 1e-12)
        basis[ch] = how

    # ---- starting values -------------------------------------------
    # The catalogue preset describes a different machine than the one on the
    # bench, and this residual surface has a minimum for every plausible pairing
    # of torque constant and damping, so start from whatever the log states
    # directly instead of hoping the gradient walk finds its way across.
    warm_paths = set(paths) | ({b for b, d in dq.items() if d in paths} if dq else set())
    warm = _warm_starts(base, ds.blocks, warm_paths)
    if dq:
        # the log states the brushed quantities directly (the back-EMF line, the
        # voltage left over after it, standstill acceleration), so ask for those
        # and convert them onto the paths this system can set
        warm = _translate_warm(warm, dq, base)

    # ---- parameter vector ------------------------------------------
    params: list[FitParameter] = []
    for path in paths:
        start = float(warm[path][0]) if path in warm else float(_read_value(base, path))
        if not math.isfinite(start):
            raise ConfigError(
                f"{path!r} starts at {start}; a free parameter must be finite "
                "(give it a known value first)."
            )
        lo, hi = _param_bounds(path, start, bounds)
        use_log = bool(log) and start > 0 and lo > 0
        params.append(FitParameter(
            path=path, label=_meta(path, start)[0], unit=_meta(path, start)[1],
            start=start, value=start, log=use_log, bounds=(lo, hi),
            integer=isinstance(_read_value(base, path), int) and not isinstance(
                _read_value(base, path), bool),
        ))

    def to_fit(value: float, p: FitParameter) -> float:
        return math.log(value) if p.log else value

    def from_fit(x: float, p: FitParameter) -> float:
        return math.exp(x) if p.log else x

    x0 = np.array([to_fit(p.start, p) for p in params])
    lo = np.array([to_fit(p.bounds[0], p) for p in params])
    hi = np.array([to_fit(p.bounds[1], p) for p in params])
    x0 = np.clip(x0, lo, hi)

    def build(x: np.ndarray) -> DriveSystem:
        sys_ = base
        for p, xi in zip(params, x):
            value = from_fit(float(xi), p)
            if p.integer:
                value = int(round(value))
            sys_ = set_path(sys_, p.path, value)
        return sys_

    blocks = ds.blocks
    n_res = sum(int(np.count_nonzero(np.isfinite(b.columns[ch])))
                for b in blocks for ch in wanted if ch in b.columns)
    if n_res < len(params) + 1:
        raise ConfigError(
            f"Only {n_res} data points for {len(params)} free parameters; "
            "collect more data or fit fewer parameters."
        )

    # a measured quantity that sits at one value for a long stretch is the drive's
    # own current loop clipping, not the motor's behaviour
    clipped: list[tuple[str, float, float]] = []
    for ch in wanted:
        if ch not in CLIPPABLE_CHANNELS:
            continue
        parts = [b.columns[ch] for b in blocks
                 if ch in b.columns and b.columns[ch] is not None]
        if not parts:
            continue
        vals = np.concatenate(parts)
        vals = vals[np.isfinite(vals)]
        top = float(np.max(np.abs(vals))) if vals.size else 0.0
        if vals.size < 20 or not math.isfinite(top) or top <= 0:
            continue
        frac = float(np.mean(np.abs(np.abs(vals) - top) <= max(4.0 * _noise_sigma(vals), 1e-3 * top)))
        if frac > 0.2:
            clipped.append((ch, top, frac))

    failures = {"count": 0}

    def residual(x: np.ndarray) -> np.ndarray:
        try:
            sys_ = build(x)
        except Exception:  # noqa: BLE001 - bad step, let the optimiser back off
            failures["count"] += 1
            return np.full(n_res, 1e3)
        try:
            out: list[np.ndarray] = []
            for block in blocks:
                model, model_t = _model_channels(sys_, block, **opts)
                t: np.ndarray | None = None
                if block.time is not None:
                    t = np.asarray(block.time, float)
                    t = t - float(t[0])
                for ch in wanted:
                    if ch not in block.columns or _is_input(block, ch):
                        continue
                    measured = block.columns[ch]
                    pred = _sample(model[ch], model_t, t)
                    out.append((pred - measured) / sigmas[ch])
            return np.concatenate(out) if out else np.full(n_res, 1e3)
        except Exception:  # noqa: BLE001 - solver blew up on this step
            failures["count"] += 1
            return np.full(n_res, 1e3)

    # ---- optimise ---------------------------------------------------
    def solve_from(x_start: np.ndarray):
        if _scipy_least_squares is not None:
            sol = _scipy_least_squares(
                residual, x_start, bounds=(lo, hi), method="trf", max_nfev=max_nfev,
                x_scale="jac", diff_step=1e-6, ftol=1e-12, xtol=1e-12, gtol=1e-12,
                verbose=2 if verbose else 0,
            )
            return (sol.x, sol.fun, float(sol.cost), int(sol.nfev), int(sol.njev),
                    str(sol.message), bool(sol.success))
        x_, r_, cost_, nfev_, njev_, msg_ = _levenberg_marquardt(
            residual, x_start, lo, hi, max_nfev=max_nfev, verbose=verbose,
        )
        return x_, r_, cost_, nfev_, njev_, msg_, not msg_.startswith("stopped")

    x, r, cost, evaluations, iterations, message, success = solve_from(x0)

    # ---- rescue a fit that landed in the wrong basin ----------------
    # A gradient method only ever walks downhill from where it started.  If the
    # result still misses the data by a large fraction of the data's own
    # variation, hop each parameter out by decades, keep whatever lowers the cost,
    # and re-optimise.  The yardstick is the signal, not the noise weights: a
    # quiet bench log makes the normalised rms enormous even when the model is
    # right, and a noisy one makes it small even when the model is wrong.
    rescanned = False
    limit = sum(
        int(np.count_nonzero(np.isfinite(b.columns[ch])))
        * (RESCAN_MISS_FRACTION * spread_of[ch] / sigmas[ch]) ** 2
        for b in blocks for ch in wanted
    )
    if len(params) <= 24 and limit > 0.0 and 2.0 * cost > limit:
        logs = np.array([p.log for p in params], dtype=bool)
        skip = [structural_inert(base, p.path, measured) is not None for p in params]
        x_try, cost_try = _coarse_scan(residual, x, lo, hi, logs, skip=skip)
        if cost_try < 0.5 * cost:
            x2, r2, cost2, nfev2, njev2, msg2, ok2 = solve_from(x_try)
            if cost2 < 0.5 * cost:
                rescanned = True
                x, r, cost = x2, r2, cost2
                evaluations, iterations = evaluations + nfev2, iterations + njev2
                message = msg2 + " (after a coarse decade rescan)"
                success = ok2

    x = np.clip(x, lo, hi)
    fitted = build(x)
    for target, source in tie_map.items():
        fitted = set_path(fitted, target, _read_value(fitted, source))

    # ---- per-channel error -----------------------------------------
    rms: dict[str, float] = {}
    max_err: dict[str, float] = {}
    points: dict[str, int] = {}
    # how big the model and the data each are, and how well they track: that is
    # what exposes a column logged in the wrong unit (see the warnings below)
    scale: dict[str, tuple[float, float, float]] = {}
    relative: dict[str, float] = {}
    limit = fitted.current_limit_total
    saturated = [0, 0]                      # samples at the current limit, total
    for ch in wanted:
        diffs: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        for block in blocks:
            if ch not in block.columns:
                continue
            model, model_t = _model_channels(fitted, block, **opts)
            t: np.ndarray | None = None
            if block.time is not None:
                t = np.asarray(block.time, float) - float(block.time[0])
            pred = np.asarray(_sample(model[ch], model_t, t), float)
            d = pred - block.columns[ch]
            diffs.append(np.asarray(d, float))
            preds.append(pred)
            acts.append(np.asarray(block.columns[ch], float))
            if ch == "current_A" and math.isfinite(limit) and limit > 0:
                saturated[0] += int(np.count_nonzero(np.abs(pred) >= 0.999 * limit))
                saturated[1] += int(pred.size)
        d = np.concatenate(diffs)
        points[ch] = int(d.size)
        rms[ch] = float(np.sqrt(np.mean(d**2))) if d.size else 0.0
        max_err[ch] = float(np.max(np.abs(d))) if d.size else 0.0
        p_all, a_all = np.concatenate(preds), np.concatenate(acts)
        shape = 0.0
        if p_all.size > 2 and np.std(p_all) > 0 and np.std(a_all) > 0:
            shape = float(np.corrcoef(p_all, a_all)[0, 1])
        scale[ch] = (
            float(np.median(np.abs(p_all))) if p_all.size else 0.0,
            float(np.median(np.abs(a_all))) if a_all.size else 0.0,
            shape if math.isfinite(shape) else 0.0,
        )
        # unit-free tracking quality: the old yardstick, kept alongside the
        # noise-relative one so a noiseless bench log still reads sensibly
        spread = float(np.std(a_all)) if a_all.size else 0.0
        relative[ch] = rms[ch] / spread if spread > 0 else (
            0.0 if rms[ch] == 0.0 else math.inf
        )
    normalized = float(np.sqrt(np.mean(r * r))) if (r := np.asarray(r, float)).size else 0.0

    # ---- statistics -------------------------------------------------
    jac = _jacobian(residual, x, lo, hi)
    dof = max(jac.shape[0] - len(x), 1)
    s2 = 2.0 * cost / dof
    # The covariance comes from the Jacobian's singular values.  Directions the
    # data cannot resolve at all (singular values down at the finite-difference
    # noise floor) are floored instead of truncated: truncating them makes the
    # quoted 1-sigma look precise for a parameter that is really unconstrained.
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    unresolvable = 0
    if not singular.size or singular[0] <= 0.0:
        # a Jacobian that is zero everywhere: nothing in this data moves, so no
        # spread can be quoted for anything
        cov = np.diag([math.inf] * len(x))
        unresolvable = len(x)
    else:
        floor = singular[0] * 1e-10
        unresolvable = int(np.count_nonzero(singular <= floor))
        kept = np.maximum(singular, floor)
        cov = (vh.T * (s2 / kept**2)) @ vh
    # a parameter this configuration never evaluates has a zero Jacobian column:
    # it cannot be fitted, and no amount of data will change that
    inert = [params[i].path for i in range(min(len(params), jac.shape[1]))
             if jac.shape[0] and not np.any(jac[:, i])]
    structural: list[str] = []
    dead: list[str] = []
    inert_paths: set[str] = set()
    for p in params:
        # judged against the configuration as specified, not as fitted: hopping a
        # zero inductance off zero switches the model from an algebraic current
        # loop to a second-order one, and that trick must not launder an
        # unidentifiable parameter into a "fitted" one
        why = structural_inert(base, p.path, measured)
        if why:
            structural.append(f"{p.path} ({why})")
            inert_paths.add(p.path)
        elif p.path in inert:
            dead.append(p.path)
            inert_paths.add(p.path)
    for p, xi, sd in zip(params, x, np.sqrt(np.abs(np.diag(cov)))):
        if p.path in inert_paths:
            # the optimiser can put a parameter with no effect on the residuals
            # anywhere it likes, usually on a bound: report the value it started
            # at instead of pretending that drift means something
            p.value, p.stderr = p.start, math.inf
            continue
        p.value = from_fit(float(xi), p)
        if p.integer:
            p.value = float(int(round(p.value)))
        p.stderr = abs(sd * (abs(p.value) if p.log else 1.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(np.abs(np.diag(cov)))
        corr = cov / np.outer(d, d)
    condition = float(singular[0] / max(singular[-1], 1e-300)) if singular.size else math.inf

    # ---- warnings ---------------------------------------------------
    warnings: list[str] = []
    if structural:
        warnings.append(
            "no sensitivity at all: " + ", ".join(structural)
            + " leave the residuals unchanged wherever they are put, because this configuration"
            " never evaluates them. No dataset can recover a parameter the model does not use;"
            " they are reported at their starting values."
        )
    if dead:
        warnings.append(
            "no sensitivity at all: " + ", ".join(dead)
            + " leave the residuals unchanged wherever they are put (inductance is inert when"
            " the current loop is algebraic, Coulomb stiction when the shaft is never near"
            " standstill, drag when the rig never moves, a wheel channel when there is no"
            " wheel). Drop them from free= or change the configuration; they are reported at"
            " their starting values."
        )
    if failures["count"]:
        warnings.append(
            f"{failures['count']} trial step(s) were rejected (invalid model or solver failure)"
        )
    if not success:
        warnings.append(f"optimiser stopped early: {message}")
    if unresolvable:
        warnings.append(
            f"{unresolvable} direction(s) of this fit cannot be resolved at all from this "
            "data: the 1-sigma for the parameters involved is unbounded"
        )
    if condition > 1e6:
        warnings.append(
            f"the fit is degenerate (condition number {condition:.2g}): the data cannot "
            "separate every free parameter"
        )
        # name the parameters that move together
        try:
            worst = vh[-1]
            culprits = [p.path for p, w in zip(params, worst)
                        if abs(w) > 0.3 and p.path not in inert_paths]
            if len(culprits) >= 2:
                warnings.append(
                    "least observable direction: " + ", ".join(culprits)
                    + " -- fix one of them, or measure another channel"
                )
        except (IndexError, ValueError):  # pragma: no cover
            pass
    for p in params:
        if p.path in inert_paths:
            continue          # already reported as having no sensitivity at all
        if p.at_bound():
            warnings.append(f"{p.path} ended on its bound ({p.value:.6g}); widen bounds= if that is wrong")
        elif p.relative_stderr > 0.5:
            if p.log and p.relative_stderr > 3.0:
                how = (f"its log-space 1-sigma is {p.relative_stderr:.2g}, so the data cannot"
                       " pin it down over decades")
            elif p.log:
                how = f"a factor of {math.exp(p.relative_stderr):.2g} either way"
            else:
                how = f"{p.stderr:.2g} on {p.value:.4g}"
            warnings.append(
                f"{p.path} is poorly determined ({how}); consider fixing it"
            )
    names = {p.path for p in params}
    measured = set(wanted)
    if {"group.motor.kv", "group.motor.kt"} <= names and not (
        {"current_A", "power_in_W", "copper_loss_W", "torque_Nm"} & measured
    ):
        warnings.append(
            "kv and kt are both free but no electrical channel is measured: consider "
            "ties={'group.motor.kt': 'group.motor.kv'}"
        )
    if {"group.motor.kt", "group.motor.rotor_inertia"} <= names and "current_A" not in measured:
        warnings.append(
            "kt and rotor_inertia trade off when only speed is measured (speed sees kt/J); "
            "add current_A data or fix one of them"
        )
    # ---- does the answer describe a motor at all? -------------------
    # A motor that develops kt N*m per ampere also returns kv*omega volts of
    # back-emf, and in SI those two numbers are the same number: kv*omega*i is
    # the electrical power handed to the shaft, kt*i*omega is what comes off it.
    # A fit that leaves them a factor of several apart is not a slightly wrong
    # motor, it is a model that is compensating for something else -- most
    # often a gear ratio that was pinned at a value the log does not have.
    try:
        kv_out = float(get_path(fitted, "group.motor.kv"))
        kt_out = float(get_path(fitted, "group.motor.kt"))
    except (AttributeError, KeyError, TypeError):  # pragma: no cover
        kv_out = kt_out = 0.0
    if kv_out > 0.0 and kt_out > 0.0:
        mismatch = kt_out / kv_out
        if not 0.2 <= mismatch <= 5.0:
            which = "the fit leaves" if {"group.motor.kv", "group.motor.kt"} & names else "this model has"
            warnings.append(
                f"{which} kt and kv inconsistent: kt = {kt_out:.4g} N*m/A but "
                f"kv = {kv_out:.4g} V/(rad/s), a factor of {mismatch:.3g} apart, where energy"
                " conservation wants one number. A fit this lopsided is not a slightly wrong"
                " motor, it is the model absorbing something else: check the gear ratio you"
                " pinned against the log, the units of the speed column, or whether one of the"
                " two constants is quoted through a gear ratio."
            )
    if any(p.integer for p in params):
        warnings.append("integer parameters were fitted continuously then rounded")
    # ---- what this particular data cannot tell you --------------------
    if steady:
        blind = [p.path for p in params if p.path in TRANSIENT_ONLY_PATHS]
        if blind:
            warnings.append(
                "settled operating points contain no transient, so " + ", ".join(blind)
                + " leave no mark on this data at all: recover them from a time-series log,"
                " or pin them with known="
            )
        if MECHANICAL_PATHS & names and not ELECTRICAL_CHANNELS & measured:
            warnings.append(
                "voltage/speed operating points alone cannot separate friction and damping from"
                " kv -- they only ever appear as the equilibrium balance. Measure current, or pin"
                " one of them with known="
            )
    for ch, top, frac in clipped:
        warnings.append(
            f"the measured {ch} sits at {top:.4g} for {100 * frac:.0f}% of the log: the drive"
            " itself is clipping there (its current loop, a torque limit, or the sensor range)."
            " While it clips, back-EMF and winding resistance leave the equations, so those"
            " samples say nothing about kv or resistance: log a run that stays under the"
            " limit, raise max_current, or pin the electrical parameters with known=."
        )
    if (
        saturated[1]
        and saturated[0] / saturated[1] > 0.2
        and names & {"group.motor.kv", "group.motor.resistance", "group.motor.inductance"}
    ):
        warnings.append(
            f"the model rides its {limit:g} A current limit for "
            f"{100 * saturated[0] / saturated[1]:.0f}% of the fitted trajectory: while the drive"
            " is saturated the back-EMF and resistive terms leave the equations, so kv,"
            " resistance and inductance are whatever the optimiser liked rather than what the"
            " data says. Log a run that stays under the limit, or raise max_current."
        )
    if {"group.motor.kv", "group.motor.resistance"} <= names and not ELECTRICAL_CHANNELS & measured:
        warnings.append(
            "kv and resistance are both free with no electrical channel measured: speed alone"
            " pins kv and the total damping, leaving resistance free to absorb the difference"
            " -- measure current, or pin it with known={'group.motor.resistance': ...}"
        )
    merged = []
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            try:
                c = float(corr[i, j])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(c) and abs(c) >= 0.999:
                merged.append(f"{params[i].path} / {params[j].path} (r={c:+.3f})")
    if merged:
        warnings.append(
            "indistinguishable in this data: " + "; ".join(merged[:5])
            + " -- to the fit these are one quantity. Fit one and pin the other with known=,"
            " or tie them."
        )
    for ch, (model_med, data_med, shape) in scale.items():
        if shape < 0.99 or not (model_med > 0 and data_med > 0):
            continue
        ratio = model_med / data_med
        # 9.549 = 60/2pi is the factor between rpm and rad/s: a column that
        # tracks the model perfectly but sits that far off is a unit mix-up,
        # not a wrong parameter (and 3.6 for km/h vs m/s is outside this band)
        if 6.0 <= ratio <= 15.0:
            warnings.append(
                f"{ch}: the model tracks the data exactly but sits {ratio:.1f}x above it -- that"
                " is the 60/2pi factor, so the column is rad/s, not rpm. Log it as motor_rad_s /"
                " output_rad_s, or multiply the column by 60/(2pi)."
            )
        elif 1.0 / 15.0 <= ratio <= 1.0 / 6.0:
            warnings.append(
                f"{ch}: the data tracks the model exactly but sits {1.0 / ratio:.1f}x above it"
                " -- that is the 60/2pi factor, so the column looks like rpm in a rad/s channel."
                " Divide it by 60/(2pi), or name it *_rpm."
            )
    if rescanned:
        warnings.append(
            "the first optimum was a local minimum: a coarse decade rescan of every free"
            " parameter found a basin at least twice as good, and the values below come from"
            " that. If they still look wrong the log does not constrain this combination --"
            " fit fewer parameters at once, or excite the drive harder."
        )
    for note in notes:
        warnings.append(note)
    for note in ds.notes:
        warnings.append(f"dataset: {note}")

    dt_used = dt
    if dt_used is None and ds.kind != "steady":
        dt_used = min(
            _solver_dt(fitted, b.time, max_steps) for b in blocks if b.time is not None
        ) if any(b.time is not None for b in blocks) else 0.0

    return FitResult(
        system=fitted,
        parameters=params,
        rms=rms,
        max_error=max_err,
        points=points,
        weights=sigmas,
        weights_basis=basis,
        relative_rms=relative,
        cost=cost,
        normalized_rms=normalized,
        condition=condition,
        correlations=corr,
        success=success,
        message=message,
        evaluations=evaluations,
        iterations=iterations,
        engine=engine,
        dt=float(dt_used or 0.0),
        dataset=ds.name,
        kind=ds.kind,
        warnings=warnings,
        degenerate=bool(unresolvable) or condition > 1e6,
        ties=tie_map,
        known=known_map,
        starts={p: (v, why) for p, (v, why) in warm.items()},
        _blocks=blocks,
        _opts=opts,
    )


def fit_dataset(*args: Any, **kwargs: Any) -> FitResult:
    """Alias for :func:`fit` (reads better in scripts: ``fit_dataset(csv_path)``)."""
    return fit(*args, **kwargs)
