"""Export a brushless machine as standalone controller code.

The brushed exporter (:mod:`drivetrain.export`) emits a one-port DC model.  That
is the right shape for a brushed drive and the wrong shape for a brushless one:
a three-phase machine under FOC has two current states, a voltage *ellipse*
rather than a scalar voltage limit, and a current *angle* the controller has to
choose.  This module emits that model instead: ``drive_model_dq.py`` carries the
dq laws, a minimum-current command (MTPA with field weakening on top), a clocked
feed-forward helper and an RK4 plant model, with nothing beyond the standard
library.

Conventions, inherited from :mod:`drivetrain.pmsm`:

* amplitude-invariant Park, so ``torque_constant = 1.5 * p * lambda`` and
  ``back_emf_constant = p * lambda``.  The factor of 1.5 between them is the
  convention, not a mistake in the constants;
* ``voltage_limit`` is the peak fundamental phase voltage -- ``V_bus / sqrt(3)``
  for SVPWM, ``V_bus / 2`` for sinusoidal -- not the DC bus;
* ``current_limit`` is per machine, and the generated module commands
  per-machine phase current.  ``motor_count`` machines draw ``motor_count``
  times that from the DC link, scaled again by 1.5 (see ``dc_link_current``).

The generated module is dependency-free and allocation-free on purpose: it is
meant to sit next to a current loop on a controller, not to be a simulation
library.  Where it has to search (the field-weakening angle) it searches with a
fixed iteration count, so its worst-case cost is known before it ships.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

from .exceptions import ConfigError
from .dq_sim import RK4_STABILITY_LIMIT
from .export import (ExportOptions, _banner, _c_num, _py_num,
                       _rust_num)
from .pmsm import PMSMModel
from .system import DriveSystem

__all__ = ["DqModelConstants", "dq_constants_from_system", "dq_machine_notes",
           "export_dq", "summarise_dq"]

SCHEMA_VERSION = "drive-model-dq-1"

# The field-weakening search is a scan plus a bisection, both with a fixed trip
# count.  24 scan steps over the allowed i_d range plus 40 bisection steps
# lands well inside f32 resolution for any machine this package models, and it
# keeps the generated code free of while-True loops.
WEAKENING_SCAN_STEPS = 24
WEAKENING_BISECT_STEPS = 40
MTPA_BISECT_STEPS = 60

# Algorithm constants, emitted into every language so a firmware engineer
# cannot get two different answers out of one manifest.  They are printed at
# full double precision whatever the build's float type: they are tolerances,
# not machine constants, and a 1e-12 relative slack rounded to f32 is not a
# slack at all, which would make the f32 and f64 feasibility tests differ.
#
# RK4_STABILITY_LIMIT is imported from .dq_sim instead of restated here.  It is
# the one number that decides whether a control period is legal, and a library
# that accepts a step the generated code refuses -- or the other way round -- is
# a disagreement no test catches as cleanly as a shared constant prevents.
STICTION_BAND = 1.0e-6        # rad/s; below this static friction is not engaged
FEASIBILITY_SLACK = 1.0e-12   # relative: one rounding inside a limit is inside
ENVELOPE_SLACK = 1.0e-3       # relative: matched to the envelope scan, not to eps
SALIENCE_FLOOR = 1.0e-15      # |Ld - Lq| at or below this is treated as non-salient


# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DqModelConstants:
    """Per-machine dq constants, plus the drivetrain ratios around them.

    Everything here is *per machine* on the electrical side: the generator
    emits one machine's constants and the module's ``motor_count`` tells the
    caller how many of them the DC link is feeding.  That is how a real FOC
    drive is wired -- one controller per machine, or one controller with the
    current limit split -- and it avoids the trap of exporting a parallel
    combination of machines whose per-unit constants no longer match any
    datasheet.
    """

    resistance: float                 # phase resistance per machine [ohm]
    inductance_d: float               # [H]
    inductance_q: float               # [H]
    flux_linkage: float               # magnet flux linkage lambda_ [Wb]
    pole_pairs: int
    rotor_inertia: float              # kg*m^2 at the motor shaft
    bearing_damping: float            # N*m/(rad/s)
    static_friction: float            # N*m
    current_limit: float              # phase current amplitude, per machine [A]
    voltage_limit: float              # peak fundamental phase voltage [V]
    dc_bus_voltage: float             # V (math.inf = not modelled)
    core_loss_coefficient: float      # iron loss ~ k * w_e^2 [W/(rad/s)^2 elec]
    windage_coefficient: float        # windage ~ k * w^3 [W/(rad/s)^3]
    flux_temperature_coefficient: float   # fraction/K, negative for NdFeB
    reference_temperature: float      # C the flux figure was measured at
    motor_count: int
    gear_ratio: float                 # output speed / motor speed
    gear_efficiency: float
    wheel_radius: float               # m (0.0 = no wheel in this system)
    notes: tuple[str, ...] = ()

    # -- derived ---------------------------------------------------------
    @property
    def torque_constant(self) -> float:
        """``1.5 * p * lambda`` [N*m/A] -- the amplitude-invariant convention."""
        return 1.5 * self.pole_pairs * self.flux_linkage

    @property
    def back_emf_constant(self) -> float:
        """``p * lambda`` [V/(rad/s)] -- 1.5x smaller than ``torque_constant``."""
        return self.pole_pairs * self.flux_linkage

    @property
    def reluctance_coefficient(self) -> float:
        """``Ld - Lq`` [H]; multiplies ``i_d * i_q`` in the torque law."""
        return self.inductance_d - self.inductance_q

    @property
    def is_salient(self) -> bool:
        return abs(self.inductance_q - self.inductance_d) > 1.0e-15

    @property
    def characteristic_current(self) -> float:
        """``lambda / Ld``: the ``i_d`` that cancels the magnet flux [A]."""
        return self.flux_linkage / self.inductance_d

    @property
    def id_floor(self) -> float:
        """Most negative ``i_d`` the module will command [A].

        The model has no demagnetisation term, so weakening past the
        characteristic current is refused even if the current circle allows it.
        """
        return -min(self.current_limit, self.characteristic_current) \
            if math.isfinite(self.current_limit) else -self.characteristic_current

    @property
    def base_speed(self) -> float:
        """Motor speed where back-EMF alone reaches the voltage rail [rad/s]."""
        return self.voltage_limit / self.back_emf_constant

    @property
    def electrical_time_constant(self) -> float:
        """The fastest pole in the plant: ``min(Ld, Lq) / R`` [s]."""
        return min(self.inductance_d, self.inductance_q) / self.resistance

    @property
    def mech_time_constant(self) -> float:
        """``J * R / (kt * kv)`` [s] at no load."""
        return self.rotor_inertia * self.resistance \
            / (self.torque_constant * self.back_emf_constant)

    def constant_torque_speed(self) -> float:
        """Speed up to which full ``current_limit`` still fits the voltage rail.

        ``|v|^2 <= V^2`` at ``i_d = 0``, ``i_q = I`` is quadratic in the
        electrical speed: ``a*w_e^2 + b*w_e + c = 0`` with ``a = (Lq*I)^2 +
        lambda_^2``, ``b = 2*lambda_*R*I``, ``c = (R*I)^2 - V^2``.  This is the
        edge of the constant-torque band, and it is *below* base speed whenever
        ``Lq*I`` is comparable to ``lambda_``, because the drive has to spend
        part of its voltage budget on the cross-coupling term ``-w_e*Lq*i_q``.
        """
        i = self.current_limit
        if not math.isfinite(i) or not math.isfinite(self.voltage_limit):
            return self.base_speed
        a = (self.inductance_q * i) ** 2 + self.flux_linkage ** 2
        b = 2.0 * self.flux_linkage * self.resistance * i
        c = (self.resistance * i) ** 2 - self.voltage_limit ** 2
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return 0.0
        return (math.sqrt(disc) - b) / (2.0 * a) / self.pole_pairs

    def no_load_speed(self, voltage: float) -> float:
        """Speed at which the machine draws no current at this voltage [rad/s]."""
        return float(voltage) / self.back_emf_constant

    def stall_torque(self, voltage: float) -> float:
        """Torque at standstill, current limit honoured [N*m]."""
        v = float(voltage)
        if not math.isfinite(v):
            return math.inf
        i = v / self.resistance
        if math.isfinite(self.current_limit):
            i = min(i, self.current_limit)
        return self.torque_constant * i

    def pole_magnitude(self, rot_vel: float = 0.0) -> float:
        """Distance of the dq electrical pole pair from the origin [1/s].

        Freezing the shaft speed, the current loop linearises to a complex pair
        at ``-pole_damping +/- j*p*rot_vel``.  The imaginary part is the
        ELECTRICAL speed, which is why a dq simulation's stable step shrinks as
        the machine speeds up.
        """
        return math.hypot(self.pole_damping, self.pole_pairs * float(rot_vel))

    @property
    def pole_damping(self) -> float:
        """Real part of the dq electrical pole pair [1/s]."""
        return (self.resistance * (self.inductance_d + self.inductance_q)
                / (2.0 * self.inductance_d * self.inductance_q))

    def max_step_time(self, rot_vel: float = 0.0) -> float:
        """Largest stable RK4 step for the exported plant model at *rot_vel* [s].

        RK4 is stable for ``|pole*dt| < ~2.78``; the contract takes half of
        that, so it is accurate as well as stable.  The pole magnitude grows
        with speed (see pole_magnitude), so this is NOT one number for a
        machine: it is set by R/L at standstill and by p*rot_vel at speed.
        """
        return min(0.5 * self.electrical_time_constant,
                   1.39 / self.pole_magnitude(rot_vel))

    # -- reporting -------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """The constants plus their units, for the manifest and the docs."""
        return {
            "resistance": {"value": self.resistance, "unit": "ohm per phase"},
            "inductance_d": {"value": self.inductance_d, "unit": "H"},
            "inductance_q": {"value": self.inductance_q, "unit": "H"},
            "flux_linkage": {"value": self.flux_linkage, "unit": "Wb"},
            "pole_pairs": {"value": self.pole_pairs, "unit": "pairs"},
            "rotor_inertia": {"value": self.rotor_inertia, "unit": "kg*m^2 at the motor shaft"},
            "bearing_damping": {"value": self.bearing_damping, "unit": "N*m/(rad/s)"},
            "static_friction": {"value": self.static_friction, "unit": "N*m"},
            "current_limit": {"value": self.current_limit, "unit": "A phase amplitude, per machine"},
            "voltage_limit": {"value": self.voltage_limit, "unit": "V peak fundamental phase"},
            "dc_bus_voltage": {"value": self.dc_bus_voltage, "unit": "V (inf = not modelled)"},
            "core_loss_coefficient": {"value": self.core_loss_coefficient,
                                      "unit": "W/(rad/s)^2 electrical"},
            "windage_coefficient": {"value": self.windage_coefficient, "unit": "W/(rad/s)^3"},
            "flux_temperature_coefficient": {"value": self.flux_temperature_coefficient,
                                             "unit": "fraction per K"},
            "reference_temperature": {"value": self.reference_temperature, "unit": "C"},
            "motor_count": {"value": self.motor_count, "unit": "machines on this link"},
            "gear_ratio": {"value": self.gear_ratio, "unit": "output speed / motor speed"},
            "gear_efficiency": {"value": self.gear_efficiency, "unit": "0..1"},
            "wheel_radius": {"value": self.wheel_radius, "unit": "m (0 = no wheel)"},
        }

    def derived_dict(self) -> dict[str, float]:
        kt, kv = self.torque_constant, self.back_emf_constant
        return {
            "torque_constant_Nm_per_A": kt,
            "back_emf_constant_V_per_rad_s": kv,
            "torque_over_back_emf": kt / kv if kv else math.nan,
            "reluctance_coefficient_H": self.reluctance_coefficient,
            "characteristic_current_A": self.characteristic_current,
            "id_floor_A": self.id_floor,
            "base_speed_rad_s": self.base_speed,
            "base_speed_rpm": self.base_speed * 60.0 / (2.0 * math.pi),
            "constant_torque_speed_rad_s": self.constant_torque_speed(),
            "electrical_time_constant_s": self.electrical_time_constant,
            "mech_time_constant_s": self.mech_time_constant,
            "no_load_speed_rad_s": self.no_load_speed(self.voltage_limit),
            "stall_torque_Nm": self.stall_torque(self.voltage_limit),
            "stall_current_A": self.current_limit if math.isfinite(self.current_limit)
            else self.voltage_limit / self.resistance,
            "group_current_limit_A": self.current_limit * self.motor_count
            if math.isfinite(self.current_limit) else math.inf,
            "max_step_time_s": self.max_step_time(),
            "max_step_time_at_base_speed_s": self.max_step_time(self.base_speed),
            "output_constant_torque_speed_rad_s": self.constant_torque_speed() * self.gear_ratio,
            "output_stall_torque_Nm": self.stall_torque(self.voltage_limit)
            * self.gear_ratio * self.gear_efficiency,
        }

    # -- construction ----------------------------------------------------
    @classmethod
    def from_system(cls, system: DriveSystem) -> "DqModelConstants":
        """Read the dq constants out of a system that carries a brushless machine."""
        motor = system.group.motor
        if not isinstance(motor, PMSMModel):
            raise ConfigError(
                "the dq export needs a brushless machine; this system has a "
                f"{type(motor).__name__}. Export the brushed model with export(). "
                "build('bldc_pod') or build(name, {'motor': 'bldc_24v'}) gives you "
                "a system this works on."
            )
        s = motor.spec
        wheel = system.wheel
        c = cls(
            resistance=float(s.resistance),
            inductance_d=float(s.inductance_d),
            inductance_q=float(s.inductance_q),
            flux_linkage=float(s.flux_linkage),
            pole_pairs=int(s.pole_pairs),
            rotor_inertia=float(s.rotor_inertia),
            bearing_damping=float(s.bearing_damping),
            static_friction=float(s.static_friction),
            current_limit=float(s.current_limit),
            voltage_limit=float(s.voltage_limit),
            dc_bus_voltage=float(s.dc_bus_voltage),
            core_loss_coefficient=float(s.core_loss_coefficient),
            windage_coefficient=float(s.windage_coefficient),
            flux_temperature_coefficient=float(s.flux_temperature_coefficient),
            reference_temperature=float(s.reference_temperature),
            motor_count=int(system.group.count),
            gear_ratio=float(system.gearbox.ratio),
            gear_efficiency=float(system.gearbox.efficiency),
            wheel_radius=float(wheel.radius) if wheel is not None else 0.0,
            notes=(),
        )
        # notes come from the constants themselves, so the manifest, the README
        # and the source banner all carry the same wording
        return replace(c, notes=_notes_for(c))


def dq_constants_from_system(system: DriveSystem) -> DqModelConstants:
    """The dq constants of *system*, unrounded."""
    return DqModelConstants.from_system(system)


def _f32(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, ValueError):
        return float(value)


def round_dq_to_dtype(c: DqModelConstants, dtype: str) -> DqModelConstants:
    """Round every constant to *dtype* once, up front.

    The module, the manifest and the README all read these same rounded values,
    so a reader can copy a number out of the README into the C header and get
    bit-identical behaviour.  Rounding at print time instead would make the
    three disagree in the last digits and look like a generator bug.
    """
    if dtype != "f32":
        return c
    out = {}
    for f in fields(c):
        v = getattr(c, f.name)
        if f.name == "notes":
            continue
        out[f.name] = _f32(v) if isinstance(v, float) else v
    return replace(c, **out)


# ----------------------------------------------------------------------
# notes
# ----------------------------------------------------------------------
def dq_machine_notes(system: DriveSystem) -> tuple[str, ...]:
    """The caveats for *system*'s machine, in that machine's own numbers."""
    return _notes_for(DqModelConstants.from_system(system))


def _notes_for(c: DqModelConstants) -> tuple[str, ...]:
    """What the generated dq module does not model, in this machine's numbers.

    Deliberately machine-specific: a wall of generic caveats gets skimmed, and
    the one that matters is the one that says "your machine, your rail".
    """
    notes: list[str] = []
    if math.isfinite(c.dc_bus_voltage):
        ratio = c.dc_bus_voltage / c.voltage_limit
        notes.append(
            f"voltage_limit {c.voltage_limit:.4g} V is the peak fundamental phase "
            f"voltage, not the {c.dc_bus_voltage:.4g} V bus: the ratio {ratio:.3f} is "
            "the SVPWM linear-range figure (bus/sqrt(3)); a sinusoidal drive only "
            "reaches bus/2, so re-derive the limit if the modulator is not SVPWM"
        )
    else:
        notes.append(
            "dc_bus_voltage is not finite, so dc_link_current() returns nan: the "
            "module knows the phase side but not what the battery sees"
        )
    notes.append(
        f"current_limit {c.current_limit:.4g} A is the phase amplitude of ONE machine; "
        f"this link feeds {c.motor_count}, so the DC side sees "
        f"{c.current_limit * c.motor_count:.4g} A times 1.5 (see dc_link_current)"
    )
    if c.characteristic_current < c.current_limit:
        notes.append(
            f"the current circle allows {c.current_limit:.4g} A but the model refuses "
            f"i_d below {c.id_floor:.4g} A (the characteristic current lambda/Ld): "
            "demagnetisation is not modelled, so the field cannot be weakened far "
            "enough to use the rest of that circle"
        )
    edge, base = c.constant_torque_speed(), c.base_speed
    if edge < 0.98 * base:
        notes.append(
            f"full current only fits the rail to {edge:.4g} rad/s motor side "
            f"({edge * c.gear_ratio:.4g} rad/s output), not to base speed "
            f"{base:.4g} rad/s ({base * c.gear_ratio:.4g} output): "
            f"Lq*i = {c.inductance_q * c.current_limit:.4g} against "
            f"lambda = {c.flux_linkage:.4g}, so the drive spends part of its "
            "voltage budget on the cross-coupling term -w_e*Lq*i_q"
        )
    if not c.is_salient:
        notes.append("Ld == Lq, so there is no reluctance torque and MTPA is i_d = 0")
    if c.core_loss_coefficient == 0.0 and c.windage_coefficient == 0.0:
        notes.append(
            "iron and windage loss are zero in this model: losses() reports copper "
            "plus bearing damping only, which under-reads a real machine at speed"
        )
    if c.flux_temperature_coefficient == 0.0:
        notes.append(
            "magnet flux is temperature-independent here; a NdFeB machine loses on the "
            "order of 0.1%/K, so a hot machine quietly loses torque_constant with it"
        )
    notes.append(
        "no inverter non-linearity: deadtime, device drops, current-sensor offset and "
        "PWM quantisation all show up as torque ripple this model cannot predict"
    )
    return tuple(notes)


# ----------------------------------------------------------------------
# renderers
# ----------------------------------------------------------------------
def _dq_bench_numbers(o: ExportOptions, c: DqModelConstants) -> dict[str, float]:
    """Numbers for the step-response bench, sized from this machine's own time
    constants rather than hard-coded, so the bench exercises the model it was
    generated from.  Raw floats: each language prints them its own way."""
    dt = o.sample_time or min(1.0e-4, 0.25 * c.electrical_time_constant)
    duration = 12.0 * max(c.mech_time_constant, 4.0 * c.electrical_time_constant)
    step_at = 0.1 * duration
    # a raw rail-voltage step on an unlimited plant would ask for V/R amps, which
    # is not a trajectory anyone ships; drive the machine's own current limit at
    # standstill instead, so the run stays inside the ratings these constants say
    v = c.voltage_limit if math.isfinite(c.voltage_limit) else 12.0
    if math.isfinite(c.current_limit):
        v = min(v, c.resistance * c.current_limit)
    return {"dt": dt, "duration": duration, "step_at": step_at, "voltage": v,
            "target": 1.5 * c.base_speed}


def _dq_bench_values(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    n = _dq_bench_numbers(o, c)
    return {
        "bench_dt": _py_num(n["dt"], o.dtype),
        "bench_duration": _py_num(n["duration"], o.dtype),
        "bench_step_at": _py_num(n["step_at"], o.dtype),
        "bench_voltage": _py_num(n["voltage"], o.dtype),
        "bench_target": _py_num(n["target"], o.dtype),
    }


def _dq_constants_map(c: DqModelConstants, dtype: str) -> dict[str, str]:
    """Every constant as a literal, for the templates."""
    m: dict[str, str] = {}
    for f in fields(c):
        if f.name == "notes":
            continue
        v = getattr(c, f.name)
        m[f.name] = str(int(v)) if f.name in ("pole_pairs", "motor_count") \
            else _py_num(v, dtype)
    return m


# ----------------------------------------------------------------------
# the generated module
# ----------------------------------------------------------------------
_DQ_PYTHON = Template('''${banner}

"""dq/FOC model of a brushless drive, generated by drive_model.

Machine: ${pole_pairs} pole pairs, ${flux_linkage} Wb magnet flux, ${resistance} ohm,
${inductance_d} H / ${inductance_q} H, ${current_limit} A per machine, ${voltage_limit} V
peak phase.  ${motor_count} machine(s) on the link.

Conventions baked into every number here:

* amplitude-invariant Park transform, so torque_constant = 1.5*p*lambda =
  ${torque_constant} N*m/A while back_emf_constant = p*lambda = ${back_emf_constant}
  V/(rad/s).  The 1.5 between them is the convention, not a bug;
* VOLTAGE_LIMIT is the peak fundamental phase voltage the inverter can apply, not
  the DC bus (${dc_bus} V);
* CURRENT_LIMIT is the phase current amplitude of ONE machine.

Currents are dq amplitudes at the machine's own electrical frequency; the frame is
oriented so that flux_linkage sits entirely on the d axis (i_d = 0 is the
non-salient optimum).  Nothing in here models the inverter's deadtime, device
drops or sensor offsets.
"""

from __future__ import annotations

import math

# --- constants ---------------------------------------------------------
RESISTANCE = ${resistance}                    # ohm per phase
INDUCTANCE_D = ${inductance_d}               # H
INDUCTANCE_Q = ${inductance_q}               # H
FLUX_LINKAGE = ${flux_linkage}               # Wb, magnet flux at ${reference_temperature} C
POLE_PAIRS = ${pole_pairs}
ROTOR_INERTIA = ${rotor_inertia}             # kg*m^2 at the motor shaft
BEARING_DAMPING = ${bearing_damping}         # N*m/(rad/s)
STATIC_FRICTION = ${static_friction}         # N*m break-away
CURRENT_LIMIT = ${current_limit}             # A phase amplitude, per machine
VOLTAGE_LIMIT = ${voltage_limit}             # V peak fundamental phase
DC_BUS_VOLTAGE = ${dc_bus}                   # V (math.inf = not modelled)
CORE_LOSS_COEFFICIENT = ${core_loss_coefficient}   # W/(rad/s)^2 electrical
WINDAGE_COEFFICIENT = ${windage_coefficient}       # W/(rad/s)^3
FLUX_TEMP_COEFFICIENT = ${flux_temperature_coefficient}  # fraction/K
REFERENCE_TEMPERATURE = ${reference_temperature}   # C
MOTOR_COUNT = ${motor_count}
GEAR_RATIO = ${gear_ratio}                   # output speed / motor speed
GEAR_EFFICIENCY = ${gear_efficiency}
WHEEL_RADIUS = ${wheel_radius}               # m (0.0 = no wheel in this system)

# --- derived -----------------------------------------------------------
TORQUE_CONSTANT = 1.5 * POLE_PAIRS * FLUX_LINKAGE          # N*m/A
BACK_EMF_CONSTANT = POLE_PAIRS * FLUX_LINKAGE              # V/(rad/s)
RELUCTANCE = INDUCTANCE_D - INDUCTANCE_Q                   # H; multiplies i_d*i_q
MTPA_DL = INDUCTANCE_Q - INDUCTANCE_D                      # H; the MTPA angle needs THIS sign
CHARACTERISTIC_CURRENT = FLUX_LINKAGE / INDUCTANCE_D       # A; i_d that cancels the magnet
ID_FLOOR = -min(CURRENT_LIMIT, CHARACTERISTIC_CURRENT)     # A; no demagnetisation modelled
BASE_SPEED = VOLTAGE_LIMIT / BACK_EMF_CONSTANT             # rad/s motor, back-EMF hits the rail
ELECTRICAL_TIME_CONSTANT = min(INDUCTANCE_D, INDUCTANCE_Q) / RESISTANCE
# Freezing the speed, the current loop linearises to a complex pole pair at
# -POLE_DAMPING +/- j*p*rot_vel: the dq frame rotates the back-EMF coupling at
# the ELECTRICAL speed, which is fast, and RK4 has to resolve it.
POLE_DAMPING = RESISTANCE * (INDUCTANCE_D + INDUCTANCE_Q) / (2.0 * INDUCTANCE_D * INDUCTANCE_Q)
RK4_STABILITY_LIMIT = ${rk4_limit}  # half of RK4's 2.78 imaginary-axis boundary
MAX_STEP_TIME = min(0.5 * ELECTRICAL_TIME_CONSTANT,
                    RK4_STABILITY_LIMIT / POLE_DAMPING)   # s at standstill;
# max_step_time(rot_vel) below is the speed-aware version of that contract.
STICTION_BAND = ${stiction}                                       # rad/s; below this, static friction
WEAKENING_SCAN_STEPS = ${scan_steps}
WEAKENING_BISECT_STEPS = ${bisect_steps}
MTPA_BISECT_STEPS = ${mtpa_steps}


def effective_flux(temperature_c: float | None = None) -> float:
    """Magnet flux at *temperature_c*; pass None for the reference temperature."""
    if temperature_c is None:
        return FLUX_LINKAGE
    return FLUX_LINKAGE * (1.0 + FLUX_TEMP_COEFFICIENT * (temperature_c - REFERENCE_TEMPERATURE))


def electrical_speed(rot_vel: float) -> float:
    """Electrical angular velocity [rad/s] from mechanical [rad/s]."""
    return POLE_PAIRS * rot_vel


def pole_magnitude(rot_vel: float = 0.0) -> float:
    """Distance of the dq electrical pole pair from the origin [1/s]."""
    return math.hypot(POLE_DAMPING, electrical_speed(rot_vel))


def max_step_time(rot_vel: float = 0.0) -> float:
    """Largest stable RK4 step at *rot_vel* [s].

    This SHRINKS WITH SPEED, and that is not a detail: at standstill the
    electrical poles sit at -R/Ld and -R/Lq, but every rad/s of shaft speed
    pushes the pair out along the imaginary axis by p rad/s, so a step chosen
    once at standstill goes unstable at speed.  For this machine the contract
    is ${max_step} s at standstill and ${max_step_base} s at base speed.
    """
    return min(0.5 * ELECTRICAL_TIME_CONSTANT,
               RK4_STABILITY_LIMIT / pole_magnitude(rot_vel))


def torque(i_d: float, i_q: float, flux: float | None = None) -> float:
    """Shaft-side developed torque [N*m] from dq currents.

    The reluctance term (Ld - Lq)*i_d*i_q is only present in a salient machine;
    with Ld == Lq this reduces to TORQUE_CONSTANT*i_q.

    ``flux`` overrides the flux linkage [Wb] - pass ``effective_flux(85.0)`` for
    a hot machine.  It is a flux, not a temperature: the compensation has to be
    asked for explicitly, because a fitted ``FLUX_TEMP_COEFFICIENT`` of 0 makes a
    silent temperature argument look like working code.
    """
    lam = flux if isinstance(flux, float) else FLUX_LINKAGE
    return 1.5 * POLE_PAIRS * (lam * i_q + RELUCTANCE * i_d * i_q)


def torque_gain(i_d: float) -> float:
    """dTorque/di_q at this d-axis current [N*m/A] -- falls as you weaken the field."""
    return 1.5 * POLE_PAIRS * (FLUX_LINKAGE + RELUCTANCE * i_d)


def voltage(rot_vel: float, i_d: float, i_q: float,
            di_d_dt: float = 0.0, di_q_dt: float = 0.0):
    """Steady-state (or commanded) phase voltages (v_d, v_q) [V].

    Pass di_d_dt / di_q_dt when the current is moving; the L*di/dt terms are what
    a current loop has to fight during a torque step, and ignoring them is the
    usual cause of a feed-forward controller that lags.
    """
    we = electrical_speed(rot_vel)
    v_d = RESISTANCE * i_d - we * INDUCTANCE_Q * i_q + INDUCTANCE_D * di_d_dt
    v_q = RESISTANCE * i_q + we * (INDUCTANCE_D * i_d + FLUX_LINKAGE) + INDUCTANCE_Q * di_q_dt
    return v_d, v_q


def voltage_magnitude(rot_vel: float, i_d: float, i_q: float,
                      di_d_dt: float = 0.0, di_q_dt: float = 0.0) -> float:
    """|v| [V]; must stay at or under VOLTAGE_LIMIT or the inverter saturates."""
    v_d, v_q = voltage(rot_vel, i_d, i_q, di_d_dt, di_q_dt)
    return math.hypot(v_d, v_q)


def _iq_from_voltage(i_d: float, rot_vel: float, v_limit: float | None = None):
    """Largest i_q that still fits the voltage budget at this i_d and speed [A].

    |v|^2 <= V^2 is quadratic in i_q:
    a*i_q^2 + b*i_q + c <= 0 with a = R^2 + (w_e*Lq)^2,
    b = 2*R*w_e*(lambda + (Ld - Lq)*i_d), c = (R*i_d)^2 + (w_e*(Ld*i_d + lambda))^2 - V^2.
    Returns None when no i_q fits at all -- the back-EMF alone exceeds the rail.
    """
    v = VOLTAGE_LIMIT if v_limit is None else v_limit
    if not math.isfinite(v):
        return math.inf
    we = electrical_speed(rot_vel)
    a = RESISTANCE * RESISTANCE + (we * INDUCTANCE_Q) ** 2
    b = 2.0 * RESISTANCE * we * (FLUX_LINKAGE + RELUCTANCE * i_d)
    c = (RESISTANCE * i_d) ** 2 + (we * (INDUCTANCE_D * i_d + FLUX_LINKAGE)) ** 2 - v * v
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    return (-b + math.sqrt(disc)) / (2.0 * a)


def _fits_current(i_d: float, i_q: float) -> bool:
    if not math.isfinite(CURRENT_LIMIT):
        return True
    return i_d * i_d + i_q * i_q <= CURRENT_LIMIT * CURRENT_LIMIT * (1.0 + ${slack})


def _feasible(i_d: float, i_q: float, rot_vel: float) -> bool:
    if not _fits_current(i_d, i_q):
        return False
    return voltage_magnitude(rot_vel, i_d, i_q) <= VOLTAGE_LIMIT * (1.0 + ${slack})


def mtpa_pair(tau: float) -> tuple[float, float]:
    """Minimum-current (i_d, i_q) for torque *tau* ignoring the voltage limit.

    Closed form for a salient machine: with dl = Lq - Ld and i_m the current
    magnitude, i_d = (lambda - sqrt(lambda^2 + 8*dl^2*i_m^2)) / (4*dl).  The
    sign of i_q follows the sign of *tau*; a negative i_d is the same angle for
    either direction of rotation.  Returns (i_d, math.inf) when no current
    within CURRENT_LIMIT can produce *tau*.
    """
    sign = 1.0 if tau >= 0.0 else -1.0
    tau_mag = abs(tau)
    if abs(RELUCTANCE) <= ${salience_floor}:
        if not math.isfinite(CURRENT_LIMIT):
            return 0.0, sign * (tau_mag / TORQUE_CONSTANT)
        if TORQUE_CONSTANT * CURRENT_LIMIT < tau_mag:
            return 0.0, math.inf
        return 0.0, sign * (tau_mag / TORQUE_CONSTANT)

    def torque_at(i_m: float) -> float:
        i_d = (FLUX_LINKAGE - math.sqrt(
            FLUX_LINKAGE * FLUX_LINKAGE + 8.0 * MTPA_DL * MTPA_DL * i_m * i_m
        )) / (4.0 * MTPA_DL)
        i_q_sq = i_m * i_m - i_d * i_d
        if i_q_sq <= 0.0:
            return 0.0
        return torque(i_d, math.sqrt(i_q_sq))

    ceiling = CURRENT_LIMIT if math.isfinite(CURRENT_LIMIT) else math.inf
    # the same 1e-12 relative slack the feasibility tests use: without it, a
    # torque exactly ON the current circle looks unreachable by a hair
    if math.isfinite(ceiling) and torque_at(ceiling) < tau_mag * (1.0 - ${slack}):
        return 0.0, math.inf
    hi = ceiling if math.isfinite(ceiling) else tau_mag / TORQUE_CONSTANT * 4.0
    lo = 0.0
    for _ in range(MTPA_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if torque_at(mid) < tau_mag:
            lo = mid
        else:
            hi = mid
    i_m = 0.5 * (lo + hi)
    i_d = (FLUX_LINKAGE - math.sqrt(
        FLUX_LINKAGE * FLUX_LINKAGE + 8.0 * MTPA_DL * MTPA_DL * i_m * i_m
    )) / (4.0 * MTPA_DL)
    i_q_sq = i_m * i_m - i_d * i_d
    i_q = math.sqrt(i_q_sq) if i_q_sq > 0.0 else 0.0
    return i_d, sign * i_q


def max_torque(rot_vel: float) -> tuple[float, float, float]:
    """(torque, i_d, i_q) at the largest *positive* torque this speed allows.

    The binding constraint is the current circle at low speed and the voltage
    ellipse above base speed.  The search is a fixed-count scan plus bisection so
    its cost is known before it ships; it is exact to within the scan resolution.
    """
    we = electrical_speed(rot_vel)
    best_t = 0.0
    best = (0.0, 0.0)
    floor = ID_FLOOR
    steps = WEAKENING_SCAN_STEPS
    step = (0.0 - floor) / steps if steps else 0.0
    for k in range(steps + 1):
        i_d = -k * step
        i_q_c = math.sqrt(max(0.0, CURRENT_LIMIT * CURRENT_LIMIT - i_d * i_d)) \\
            if math.isfinite(CURRENT_LIMIT) else math.inf
        i_q_v = _iq_from_voltage(i_d, rot_vel)
        if i_q_v is None:
            continue
        i_q = min(i_q_c, i_q_v)
        t = torque(i_d, i_q)
        if t > best_t:
            best_t, best = t, (i_d, i_q)
    # refine around the best i_d with a finer scan: bounded, and it cannot
    # wander the way a bisection on a non-monotone objective does
    if step > 0.0:
        centre = best[0]
        fine = step / 16.0
        for k in range(-16, 17):
            i_d = centre + k * fine
            if i_d < floor or i_d > 0.0:
                continue
            i_q_c = math.sqrt(max(0.0, CURRENT_LIMIT * CURRENT_LIMIT - i_d * i_d)) \\
                if math.isfinite(CURRENT_LIMIT) else math.inf
            i_q_v = _iq_from_voltage(i_d, rot_vel)
            if i_q_v is None:
                continue
            i_q = min(i_q_c, i_q_v)
            t = torque(i_d, i_q)
            if t > best_t:
                best_t, best = t, (i_d, i_q)
    return best_t, best[0], best[1]


def current_for_torque(tau: float, rot_vel: float):
    """(i_d, i_q, current_saturated, voltage_saturated) for torque *tau* [N*m].

    Minimum current at that torque: MTPA when the voltage ellipse allows it,
    otherwise the field-weakening angle.  Work is done in the frame where the
    commanded torque is positive and the result is reflected back, because
    flipping BOTH the signs of (i_q, w_e) leaves |v| unchanged while flipping one
    of them does not -- braking is not the same problem as driving.  Only i_q
    carries the torque's sign on the way back: i_d stays negative either way,
    because the reluctance term follows the sign of i_q, not of the torque.

    A zero torque demand is not a special case: above the rail speed the
    back-EMF alone exceeds VOLTAGE_LIMIT, so holding zero current is not an
    option either.  The weakening branch then answers with the demagnetising
    d-axis current that cancels enough flux to fit inside the rail, which is
    what a real drive does there.  When even that is impossible - a machine
    rated below its characteristic current lambda/Ld can not weaken far enough -
    no current fits at all, and the module returns the max-torque angle with the
    flags naming what bounded the search.  That is the uncontrolled-regeneration
    flags naming what bounded the search.  That is the uncontrolled-regeneration
    corner, and it is a property of the rating, not of the search.

    The flags say which constraint produced the answer, not which one the answer
    happens to violate by rounding: a pair from the field-weakening branch always
    reports voltage_saturated True, and a torque sitting exactly on the current
    circle may report saturated True or False depending on where the fixed-count
    bisection landed.  The current magnitude is the same either way.
    """
    sign = 1.0 if tau >= 0.0 else -1.0
    w = rot_vel if sign > 0.0 else -rot_vel
    tau_mag = abs(tau)

    i_d, i_q = mtpa_pair(tau_mag)
    if math.isfinite(i_q) and _feasible(i_d, i_q, w):
        return i_d, sign * i_q, False, False

    # field weakening: for i_d < 0 the current magnitude falls monotonically as
    # i_d rises toward zero, so the answer is the LARGEST feasible i_d
    floor = ID_FLOOR
    steps = WEAKENING_SCAN_STEPS
    step = (0.0 - floor) / steps if steps else 0.0
    found = None
    for k in range(steps + 1):
        i_d_try = -k * step
        gain = torque_gain(i_d_try)
        if gain <= 0.0:
            continue
        i_q_try = tau_mag / gain
        if _feasible(i_d_try, i_q_try, w):
            found = i_d_try
            break
    if found is None:
        # the request is outside the envelope at this speed: hand back the best
        # this speed can do and name the constraint that bounded it, the way the
        # engine's _which_limit_bound does, so the flags mean the same thing in
        # both implementations
        _, i_d_max, i_q_max = max_torque(w)
        # the tolerance matches max_torque()'s fixed-count scan (a few 1e-4), not
        # machine precision: at the corner where the current circle and the
        # voltage ellipse cross, the scan lands just inside the circle
        on_current = math.isfinite(CURRENT_LIMIT) and (
            math.hypot(i_d_max, i_q_max) >= CURRENT_LIMIT * (1.0 - ${envelope_slack}))
        on_voltage = (voltage_magnitude(w, i_d_max, i_q_max)
                      >= VOLTAGE_LIMIT * (1.0 - ${envelope_slack}))
        return i_d_max, sign * i_q_max, on_current, on_voltage
    bad = found + step
    for _ in range(WEAKENING_BISECT_STEPS):
        mid = 0.5 * (found + bad)
        gain = torque_gain(mid)
        if gain > 0.0 and _feasible(mid, tau_mag / gain, w):
            found = mid
        else:
            bad = mid
    gain = torque_gain(found)
    i_q = tau_mag / gain if gain > 0.0 else 0.0
    over_current = not _fits_current(found, i_q)
    # this branch exists only because the MTPA angle did not fit, and it
    # bisects ONTO the voltage ellipse, so the drive is on the rail by
    # construction: comparing the magnitude with a strict > would report False
    # for a point that sits one rounding inside the rail
    return found, sign * i_q, over_current, True


def loss_torque(rot_vel: float) -> float:
    """Torque needed to overcome this model's mechanical and iron losses [N*m].

    Iron and windage loss are specified as *powers*, so dividing by speed makes
    the iron term look linear in mechanical speed even though it is quadratic in
    electrical speed.
    """
    w = abs(rot_vel)
    if w <= 0.0:
        return 0.0
    we = electrical_speed(w)
    return (BEARING_DAMPING * w + WINDAGE_COEFFICIENT * w * w
            + CORE_LOSS_COEFFICIENT * we * we / w)


def required_torque(rot_vel: float, accel: float = 0.0, load_torque: float = 0.0) -> float:
    """Developed torque needed to follow *accel* at *rot_vel* against friction and load.

    *load_torque* is a non-negative magnitude that opposes motion (same
    convention as the rest of drive_model).  A torque that actively assists the
    motion -- a grade, a spring -- has to be added to this as a signed torque,
    not passed as a negative magnitude, or it gets counted twice.
    """
    if load_torque < 0.0:
        raise ValueError("load_torque is a magnitude that opposes motion and must be >= 0")
    w = rot_vel
    tau = ROTOR_INERTIA * accel + BEARING_DAMPING * w
    if w > STICTION_BAND:
        tau += STATIC_FRICTION + load_torque
    elif w < -STICTION_BAND:
        tau -= STATIC_FRICTION + load_torque
    else:
        tau += STATIC_FRICTION + load_torque
    return tau


def losses(rot_vel: float, i_d: float, i_q: float) -> dict:
    """Power balance at an operating point [W].

    ``input_W = copper_W + converted_W`` exactly -- that is the whole of the
    electrical input in this model.  iron, windage and bearing are drags acting
    on the shaft, so they sit INSIDE the converted power:
    ``shaft_W = converted_W - iron_W - windage_W - bearing_W``.  Adding them to
    the input as well would count them twice.
    """
    we = electrical_speed(abs(rot_vel))
    copper = 1.5 * RESISTANCE * (i_d * i_d + i_q * i_q)
    iron = CORE_LOSS_COEFFICIENT * we * we
    wind = WINDAGE_COEFFICIENT * abs(rot_vel) ** 3
    bearing = BEARING_DAMPING * rot_vel * rot_vel
    converted = torque(i_d, i_q) * rot_vel        # air-gap (developed) power
    return {
        "copper_W": copper,
        "iron_W": iron,
        "windage_W": wind,
        "bearing_W": bearing,
        "converted_W": converted,
        "shaft_W": converted - iron - wind - bearing,
        "input_W": copper + converted,
    }


def dc_link_current(rot_vel: float, i_d: float, i_q: float) -> float:
    """Current drawn from the DC link [A] -- what a battery ammeter reads.

    Not the winding current: the 1.5 is the three-phase sum under the
    amplitude-invariant convention, and the whole point of this function is that
    a drive fitted from rail current alone is fitting a different quantity.
    Returns math.nan when DC_BUS_VOLTAGE is not finite.
    """
    if not math.isfinite(DC_BUS_VOLTAGE):
        return math.nan
    v_d, v_q = voltage(rot_vel, i_d, i_q)
    return MOTOR_COUNT * 1.5 * (v_d * i_d + v_q * i_q) / DC_BUS_VOLTAGE


# --- plant model -------------------------------------------------------
class State:
    """Mechanical speed and dq currents -- the three states of this plant."""

    __slots__ = ("rot_vel", "i_d", "i_q")

    def __init__(self, rot_vel: float = 0.0, i_d: float = 0.0, i_q: float = 0.0) -> None:
        self.rot_vel = rot_vel
        self.i_d = i_d
        self.i_q = i_q

    def as_tuple(self) -> tuple[float, float, float]:
        return self.rot_vel, self.i_d, self.i_q


def derivatives(state: State, v_d: float, v_q: float, load_torque: float = 0.0):
    """(di_d/dt, di_q/dt, dw/dt) at this state under applied phase voltages."""
    we = electrical_speed(state.rot_vel)
    di_d = (v_d - RESISTANCE * state.i_d + we * INDUCTANCE_Q * state.i_q) / INDUCTANCE_D
    di_q = (v_q - RESISTANCE * state.i_q - we * (INDUCTANCE_D * state.i_d + FLUX_LINKAGE)) / INDUCTANCE_Q
    tau_load = load_torque + loss_torque(state.rot_vel)
    dw = (torque(state.i_d, state.i_q) - tau_load) / ROTOR_INERTIA
    return di_d, di_q, dw


def step(state: State, v_d: float, v_q: float, dt: float, load_torque: float = 0.0) -> State:
    """Advance the plant one fixed step with classic RK4.

    dt must be <= max_step_time(state.rot_vel), which tightens as the machine
    speeds up.  A control period is normally LONGER than that even at
    standstill, so a 1 ms loop cannot call this directly at dt = 1e-3: use
    integrate(), which sub-divides.  Refusing here is deliberate -- a silently
    unstable step is the kind of error that only shows up as a simulation that
    diverges in the field.
    """
    limit = max_step_time(state.rot_vel)
    if dt > limit:
        raise ValueError(
            f"dt {dt:g} s exceeds the stable step {limit:g} s at "
            f"{state.rot_vel:g} rad/s (pole magnitude "
            f"{pole_magnitude(state.rot_vel):g} 1/s, electrical time constant "
            f"{ELECTRICAL_TIME_CONSTANT:g} s); subdivide the step"
        )
    a1 = derivatives(state, v_d, v_q, load_torque)
    s2 = State(state.rot_vel + 0.5 * dt * a1[2], state.i_d + 0.5 * dt * a1[0],
               state.i_q + 0.5 * dt * a1[1])
    a2 = derivatives(s2, v_d, v_q, load_torque)
    s3 = State(state.rot_vel + 0.5 * dt * a2[2], state.i_d + 0.5 * dt * a2[0],
               state.i_q + 0.5 * dt * a2[1])
    a3 = derivatives(s3, v_d, v_q, load_torque)
    s4 = State(state.rot_vel + dt * a3[2], state.i_d + dt * a3[0], state.i_q + dt * a3[1])
    a4 = derivatives(s4, v_d, v_q, load_torque)
    return State(
        state.rot_vel + dt / 6.0 * (a1[2] + 2.0 * a2[2] + 2.0 * a3[2] + a4[2]),
        state.i_d + dt / 6.0 * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0]),
        state.i_q + dt / 6.0 * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1]),
    )


def sub_steps(dt: float, rot_vel: float = 0.0) -> int:
    """RK4 sub-steps a control period of *dt* needs at *rot_vel* (>= 1)."""
    limit = max_step_time(rot_vel)
    if limit <= 0.0 or dt <= limit:
        return 1
    return int(math.ceil(dt / limit))


def integrate(state: State, v_d: float, v_q: float, dt: float,
              load_torque: float = 0.0) -> State:
    """Advance one control period, sub-divided until every RK4 step is stable.

    The voltage is held constant across the period, which is what a sampled
    current loop does between updates.  The sub-step size is recomputed from
    the speed reached so far, because the stable step shrinks as the machine
    speeds up: a fixed subdivision sized for standstill is exactly the bug that
    makes a high-speed run diverge.  Cost is the number of RK4 steps, which for
    a 1 ms period on this machine is 2 at standstill and 9 at base speed.
    """
    remaining = dt
    guard = 0
    while remaining > 0.0:
        h = min(remaining, max_step_time(state.rot_vel))
        if h <= 0.0:
            raise ValueError("this machine has a zero stable step; check the inductances")
        state = step(state, v_d, v_q, h, load_torque)
        remaining -= h
        guard += 1
        if guard > 100000:
            raise RuntimeError("integrate() took 100000 sub-steps for one period; "
                               "the step contract and the speed are inconsistent")
    return state


def step_current(state: State, i_d_cmd: float, i_q_cmd: float, dt: float,
                 load_torque: float = 0.0) -> State:
    """Advance one step assuming the current loop already holds the command.

    The usual case when the current loop is much faster than the mechanical
    dynamics: currents jump to the command, and only the shaft integrates.  The
    torque is held constant across the step, so keep dt small against the
    mechanical time constant.
    """
    tau = torque(i_d_cmd, i_q_cmd)
    w = state.rot_vel
    k1 = (tau - load_torque - loss_torque(w)) / ROTOR_INERTIA
    k2 = (tau - load_torque - loss_torque(w + 0.5 * dt * k1)) / ROTOR_INERTIA
    k3 = (tau - load_torque - loss_torque(w + 0.5 * dt * k2)) / ROTOR_INERTIA
    k4 = (tau - load_torque - loss_torque(w + dt * k3)) / ROTOR_INERTIA
    return State(w + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), i_d_cmd, i_q_cmd)


# --- feed-forward ------------------------------------------------------
class FeedForward:
    """Clocked feed-forward: a speed target in, a phase-voltage pair out.

    The law is the whole point of this module::

        (i_d, i_q) = current_for_torque(J*a + b*w + load + friction, w)
        (v_d, v_q) = voltage(w, i_d, i_q)

    and the failure mode in the field is feeding it an *a* taken from the
    derivative of an unlimited reference.  So the reference is rate limited here,
    on the INCREMENT, not on the absolute value -- clamping the absolute value is
    what pins a 129 rad/s command at 0.09 rad/s and looks like a broken model.

    speed_limit defaults to base speed and accel_limit to the stall torque of
    this machine divided by its rotor inertia; both are the model's own limits,
    not the vehicle's.  Set them to what the machine can actually do, or the
    helper will happily command a trajectory the current loop cannot follow.
    """

    def __init__(self, sample_time: float, speed_limit: float | None = None,
                 accel_limit: float | None = None, load_torque: float = 0.0) -> None:
        if sample_time <= 0.0:
            raise ValueError("sample_time must be > 0")
        self.sample_time = float(sample_time)
        self.speed_limit = BASE_SPEED if speed_limit is None else float(speed_limit)
        self.accel_limit = (self.stall_torque() / ROTOR_INERTIA if accel_limit is None
                            else float(accel_limit))
        self.load_torque = float(load_torque)
        self.w_ref = 0.0
        self.a_ref = 0.0
        self.current_saturated = False
        self.voltage_saturated = False

    @staticmethod
    def stall_torque() -> float:
        """Torque at standstill with the current limit honoured [N*m]."""
        i = VOLTAGE_LIMIT / RESISTANCE
        if math.isfinite(CURRENT_LIMIT):
            i = min(i, CURRENT_LIMIT)
        return TORQUE_CONSTANT * i

    def reset(self, rot_vel: float = 0.0) -> None:
        """Restart the reference at *rot_vel* with zero acceleration."""
        self.w_ref = max(-self.speed_limit, min(self.speed_limit, float(rot_vel)))
        self.a_ref = 0.0

    def ramp(self, target: float) -> tuple[float, float]:
        """Advance the internal speed reference one sample; returns (w_ref, a_ref).

        The clamp is on the increment (accel_limit * sample_time), which is what
        makes the resulting a_ref the real acceleration the machine will see.
        """
        limit = self.speed_limit
        target = max(-limit, min(limit, float(target)))
        max_step = self.accel_limit * self.sample_time
        nxt = self.w_ref + max(-max_step, min(max_step, target - self.w_ref))
        self.a_ref = (nxt - self.w_ref) / self.sample_time
        self.w_ref = nxt
        return self.w_ref, self.a_ref

    def current(self, rot_vel: float | None = None, accel: float | None = None,
                load_torque: float | None = None):
        """(i_d, i_q) that the model needs at this operating point, with flags."""
        w = self.w_ref if rot_vel is None else float(rot_vel)
        a = self.a_ref if accel is None else float(accel)
        load = self.load_torque if load_torque is None else float(load_torque)
        tau = required_torque(w, a, load)
        i_d, i_q, sat_i, sat_v = current_for_torque(tau, w)
        self.current_saturated = sat_i
        self.voltage_saturated = sat_v
        return i_d, i_q

    def update(self, target: float):
        """One control period: returns (v_d, v_q, i_d, i_q)."""
        w, a = self.ramp(target)
        tau = required_torque(w, a, self.load_torque)
        i_d, i_q, sat_i, sat_v = current_for_torque(tau, w)
        self.current_saturated = sat_i
        self.voltage_saturated = sat_v
        v_d, v_q = voltage(w, i_d, i_q)
        if voltage_magnitude(w, i_d, i_q) > VOLTAGE_LIMIT:
            scale = VOLTAGE_LIMIT / voltage_magnitude(w, i_d, i_q)
            v_d, v_q = v_d * scale, v_q * scale
            self.voltage_saturated = True
        return v_d, v_q, i_d, i_q


# --- output side -------------------------------------------------------
def output_speed(motor_rad_s: float) -> float:
    """Output-shaft speed [rad/s] for a motor speed [rad/s]."""
    return motor_rad_s * GEAR_RATIO


def output_torque(motor_rad_s: float, i_d: float, i_q: float) -> float:
    """Torque delivered to the output shaft [N*m], after the gearbox losses."""
    return torque(i_d, i_q) * GEAR_RATIO * GEAR_EFFICIENCY


def vehicle_speed(motor_rad_s: float) -> float:
    """Ground speed [m/s]; 0.0 when this system has no wheel defined."""
    if WHEEL_RADIUS <= 0.0:
        return 0.0
    return output_speed(motor_rad_s) * WHEEL_RADIUS


def summary() -> dict:
    """The constants and their derived values, for logging or a config dump."""
    return {
        "resistance_ohm": RESISTANCE, "inductance_d_H": INDUCTANCE_D,
        "inductance_q_H": INDUCTANCE_Q, "flux_linkage_Wb": FLUX_LINKAGE,
        "pole_pairs": POLE_PAIRS, "motor_count": MOTOR_COUNT,
        "torque_constant_Nm_per_A": TORQUE_CONSTANT,
        "back_emf_constant_V_per_rad_s": BACK_EMF_CONSTANT,
        "characteristic_current_A": CHARACTERISTIC_CURRENT, "id_floor_A": ID_FLOOR,
        "current_limit_A": CURRENT_LIMIT, "voltage_limit_V": VOLTAGE_LIMIT,
        "dc_bus_voltage_V": DC_BUS_VOLTAGE, "base_speed_rad_s": BASE_SPEED,
        "constant_torque_speed_rad_s": constant_torque_speed(),
        "electrical_time_constant_s": ELECTRICAL_TIME_CONSTANT,
        "max_step_time_s": MAX_STEP_TIME,
        "stall_torque_Nm": FeedForward.stall_torque(),
        "gear_ratio": GEAR_RATIO, "gear_efficiency": GEAR_EFFICIENCY,
        "wheel_radius_m": WHEEL_RADIUS,
    }


def constant_torque_speed() -> float:
    """Motor speed up to which full CURRENT_LIMIT still fits the voltage rail [rad/s].

    |v|^2 = V^2 at i_d = 0, i_q = I is quadratic in w_e: a*w_e^2 + b*w_e + c = 0
    with a = (Lq*I)^2 + lambda^2, b = 2*lambda*R*I, c = (R*I)^2 - V^2.  This is
    below base speed whenever Lq*I is comparable to lambda, because the drive has
    to spend part of its voltage budget on the cross-coupling term -w_e*Lq*i_q.
    """
    if not math.isfinite(CURRENT_LIMIT):
        return BASE_SPEED
    i = CURRENT_LIMIT
    a = (INDUCTANCE_Q * i) ** 2 + FLUX_LINKAGE ** 2
    b = 2.0 * FLUX_LINKAGE * RESISTANCE * i
    c = (RESISTANCE * i) ** 2 - VOLTAGE_LIMIT ** 2
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0
    return (math.sqrt(disc) - b) / (2.0 * a) / POLE_PAIRS
''')


# ----------------------------------------------------------------------
# bench, manifest, README
# ----------------------------------------------------------------------
_BENCH_PYTHON = Template('''\
"""Step response of the generated dq model, printed as CSV.

    python ${bench_name}
    python ${bench_name} feedforward

Both print `time_s,motor_rad_s,current_A,voltage_V`, so a run can be diffed
against another build of the same constants, or against the Python engine in
this repo.  `plant` is a raw q-axis voltage step with no controller in it;
`feedforward` ramps a speed target through the clocked feed-forward law and
drives the plant with the voltages it asks for.

Generated values: dt = ${bench_dt} s, duration = ${bench_duration} s, step at
t = ${bench_step_at} s, applied voltage = ${bench_voltage} V (the inverter's
phase limit), feed-forward target = ${bench_target} rad/s motor side.
"""
from __future__ import annotations

import math
import sys

from ${module_name}_dq import (
    BASE_SPEED,
    DC_BUS_VOLTAGE,
    MAX_STEP_TIME,
    FeedForward,
    State,
    constant_torque_speed,
    integrate,
    losses,
    max_torque,
    step,
    sub_steps,
    summary,
    max_step_time,
)

DT = ${bench_dt}
DURATION = ${bench_duration}
STEP_AT = ${bench_step_at}
VOLTAGE = ${bench_voltage}
TARGET = ${bench_target}


def plant_run() -> list:
    """A raw voltage step on the q axis: pure plant, no controller.

    There is no current limit here -- that is the drive's job, not the plant's.
    The current column therefore runs past CURRENT_LIMIT, which is exactly what
    a raw voltage step does to a machine with a 1 ms electrical time constant.
    Use the feedforward mode to see a drive that respects its limits.
    """
    state = State()
    rows = []
    t = 0.0
    while t < DURATION - 0.5 * DT:
        v_d, v_q = (0.0, 0.0) if t < STEP_AT else (0.0, VOLTAGE)
        state = integrate(state, v_d, v_q, DT)
        t += DT
        rows.append((t, state.rot_vel, math.hypot(state.i_d, state.i_q),
                     math.hypot(v_d, v_q)))
    return rows


def feedforward_run() -> list:
    """The feed-forward law ramps a target and drives the plant with it.

    The law picks the current angle (MTPA, or field weakening above the band
    edge) and never asks for more than CURRENT_LIMIT, so this is the run that
    resembles a real drive.  The reference is deliberately allowed to run past
    base speed -- that is where the interesting behaviour is.  The current
    column is the PLANT's current, not the command: feed-forward is open loop,
    so above the band edge the plant lags the reference and settles where the
    applied voltage puts it, which is not where the reference asked for it to
    be.  That gap is the reason a real drive closes a current loop around this
    law rather than trusting it.
    """
    ff = FeedForward(sample_time=DT, speed_limit=TARGET)
    state = State()
    rows = []
    t = 0.0
    target = 0.0
    first_sat = None
    while t < DURATION - 0.5 * DT:
        if t >= STEP_AT:
            target = TARGET
        v_d, v_q, i_d, i_q = ff.update(target)
        if (ff.current_saturated or ff.voltage_saturated) and first_sat is None:
            first_sat = t
        state = integrate(state, v_d, v_q, DT)
        t += DT
        rows.append((t, state.rot_vel, math.hypot(state.i_d, state.i_q),
                     math.hypot(v_d, v_q)))
    if first_sat is not None:
        print("feed-forward hit its current/voltage limit at t = %.4f s: above the "
              "field-weakening band edge (%.1f rad/s motor side) this machine can "
              "not follow a reference ramped at its stall acceleration"
              % (first_sat, constant_torque_speed()), file=sys.stderr)
    return rows


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "plant"
    rows = plant_run() if mode == "plant" else feedforward_run()
    print("time_s,motor_rad_s,current_A,voltage_V")
    print("control period %g s -> %d RK4 sub-steps at standstill (stable step %g s), "
          "%d at base speed (stable step %g s)"
          % (DT, sub_steps(DT), MAX_STEP_TIME,
             sub_steps(DT, BASE_SPEED), max_step_time(BASE_SPEED)), file=sys.stderr)
    for row in rows:
        print("%.6f,%.6f,%.6f,%.6f" % row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')


_MANIFEST_DQ = Template('''\
{
  "schema": "${schema}",
  "generated_utc": "${stamp}",
  "generator": "drive_model dq export",
  "dtype": "${dtype}",
  "targets": ${targets},
  "sample_time_s": ${sample_time},
  "max_step_s": ${max_step},
  "provenance": ${provenance},
  "constants": ${constants},
  "derived": ${derived},
  "notes": ${notes},
  "law": {
    "v_d": "R*i_d + Ld*di_d/dt - w_e*Lq*i_q",
    "v_q": "R*i_q + Lq*di_q/dt + w_e*(Ld*i_d + lambda)",
    "torque": "1.5*p*(lambda*i_q + (Ld-Lq)*i_d*i_q)",
    "current_selection": "MTPA closed form, then the largest feasible i_d on the voltage ellipse (field weakening)",
    "integration": "RK4 on (i_d, i_q, w); dt must stay under max_step_time(rot_vel), which shrinks with speed because the dq frame rotates the back-EMF coupling at w_e = p*rot_vel"
  },
  "units": {
    "i_d": "A (phase, amplitude-invariant Park, per machine)",
    "i_q": "A (phase, amplitude-invariant Park, per machine)",
    "v_d": "V (phase peak fundamental, not the DC bus)",
    "v_q": "V (phase peak fundamental, not the DC bus)",
    "rot_vel": "rad/s (motor side)",
    "torque": "N m (per machine)",
    "time": "s"
  },
  "not_modelled": [
    "inverter deadtime, device drops, current-sensor offset, PWM quantisation",
    "magnet temperature drift unless flux_temperature_coefficient is set",
    "iron loss beyond the linear core_loss_coefficient term",
    "saturation of Ld/Lq with current",
    "current-loop dynamics (the feed-forward law assumes the loop tracks)",
    "back-EMF harmonics, cogging, unbalance, mutual coupling between machines"
  ],
  "files": ${files}
}
''')


_README_DQ = Template('''\
# ${module_name}_dq: brushless drive model, generated code

Generated by `drive_model` on ${stamp}.  Everything here is derived from one
drivetrain description, and every number in it is the value the model was
built with - nothing is hand-tuned per language.

## Files

${file_list}

## What this is

The dq (field-oriented) model of the machine, not the collapsed DC
equivalent.  It is the model to use when the controller really is an FOC
drive: it knows about the d axis, the voltage ellipse, MTPA and field
weakening, which the brushed collapse cannot represent.

```
v_d = R*i_d + Ld*di_d/dt - w_e*Lq*i_q
v_q = R*i_q + Lq*di_q/dt + w_e*(Ld*i_d + lambda)
tau = 1.5*p*(lambda*i_q + (Ld - Lq)*i_d*i_q)
w_e = p * w
```

The current angle is chosen by `current_for_torque()`: MTPA first
(`i_d = (lambda - sqrt(lambda^2 + 8*(Lq-Ld)^2*i^2)) / (4*(Lq-Ld))`), and when
that angle cannot be fed inside the voltage limit, the **largest** feasible
`i_d` on the ellipse boundary, found by a fixed-count scan and bisection.  No
iteration count depends on the input, so the runtime is bounded.

## Constants

| name | value | unit | meaning |
| --- | --- | --- | --- |
${constant_rows}

## Derived

${derived_rows}

## Conventions that bite

- `v_d`/`v_q` are **phase** peak fundamental voltages.  On a three-phase
  bridge the DC bus is `sqrt(3)` times the SVPWM phase limit.  Feeding the bus
  number into `voltage_limit` reports a machine that can spin faster than it
  can produce torque.
- `current_limit` is **per machine**.  A group of ${motor_count} draws
  ${group_current} A total, and the DC link current is
  `1.5*(v_d*i_d + v_q*i_q)/V_bus`, not the winding current.
- `torque_constant / back_emf_constant = 1.5` is the amplitude-invariant Park
  convention, not a datasheet inconsistency.
- Above `base_speed = ${base_speed} rad/s` the machine can be spinning while
  `current_for_torque(0.0, w)` still needs current: the back-EMF alone
  exceeds the rail.  That is the uncontrolled-regeneration case, and a
  feed-forward controller has to answer it with a brake, not a current
  command.

## Caveats specific to this machine

${note_block}

## What this model does not include

- inverter deadtime, device voltage drops, current-sensor offset, PWM
  quantisation
- magnet temperature drift (unless `flux_temperature_coefficient` is set)
- iron loss beyond the linear `core_loss_coefficient` term
- Ld/Lq saturation with current
- current-loop dynamics: the feed-forward law assumes the loop tracks the
  current it asks for
- back-EMF harmonics, cogging, phase unbalance, coupling between machines

## Running the bench

```
python ${bench_name}               # open-loop q-axis voltage step
python ${bench_name} feedforward   # clocked feed-forward driving the plant
```

Both print `time_s,motor_rad_s,current_A,voltage_V`.  The integrator is RK4 and
refuses `dt` above `max_step_time(rot_vel)`, which is `${max_step} s` at
standstill and `${max_step_base} s` at base speed: the dq frame rotates the
back-EMF coupling at the electrical speed, so the stable step shrinks as the
machine speeds up.  `integrate(state, v_d, v_q, dt)` re-sizes its sub-steps for
the speed it has reached, which is what a 1 ms control period needs.

## Parity

The generated module is the same arithmetic as `drivetrain/pmsm.py` in this
repo.  `tests/test_export_dq.py` runs the exported code against the engine and
checks that the current it asks for is never more than the engine's optimum.
''')


# ----------------------------------------------------------------------
# renderers
# ----------------------------------------------------------------------
def _markdown_rows(mapping: Mapping[str, Any], dtype: str) -> str:
    """``as_dict()`` rows as a markdown table: value and unit side by side."""
    lines = []
    for name, entry in mapping.items():
        if isinstance(entry, Mapping):
            value, unit = entry.get("value"), entry.get("unit", "")
        else:
            value, unit = entry, ""
        text = _py_num(value, dtype) if isinstance(value, (int, float)) else str(value)
        lines.append(f"| `{name}` | {text} | {unit} |")
    return "\n".join(lines)


def _render_dq_python(options: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    mapping = _dq_constants_map(c, options.dtype)
    mapping["dc_bus"] = mapping.pop("dc_bus_voltage")
    mapping["scan_steps"] = str(WEAKENING_SCAN_STEPS)
    mapping["bisect_steps"] = str(WEAKENING_BISECT_STEPS)
    mapping["mtpa_steps"] = str(MTPA_BISECT_STEPS)
    mapping["torque_constant"] = _py_num(c.torque_constant, options.dtype)
    mapping["back_emf_constant"] = _py_num(c.back_emf_constant, options.dtype)
    mapping["banner"] = _banner(options, c, "#")
    mapping["max_step"] = _py_num(c.max_step_time(), options.dtype)
    mapping["max_step_base"] = _py_num(c.max_step_time(c.base_speed), options.dtype)
    # the tolerances are printed at double precision whatever the build's float
    # type, so the feasibility tests mean the same thing in Python, C and Rust
    for key, value in (("rk4_limit", RK4_STABILITY_LIMIT),
                       ("stiction", STICTION_BAND),
                       ("slack", FEASIBILITY_SLACK),
                       ("envelope_slack", ENVELOPE_SLACK),
                       ("salience_floor", SALIENCE_FLOOR)):
        mapping[key] = _py_num(value, "f64")
    name = f"{options.module_name}_dq.py"
    files = {name: _DQ_PYTHON.substitute(mapping)}
    if options.include_examples:
        # each target owns its own bench, so -t python still gets a runnable one
        files.update(_render_dq_bench(options, c))
    return files


def _render_dq_bench(options: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    values = dict(_dq_bench_values(options, c))
    values["module_name"] = options.module_name
    values["bench_name"] = f"bench_{options.module_name}_dq.py"
    name = values["bench_name"]
    return {name: _BENCH_PYTHON.substitute(values)}


# ----------------------------------------------------------------------
# C
# ----------------------------------------------------------------------

_DQ_C_HEADER = Template(r'''${banner}

#ifndef ${guard}
#define ${guard}

#include <math.h>

/* Width-matched math: a float build must not drift into libm's double entry
   points, and INFINITY has to arrive as a dq_scalar, not a double. */
#define DQ_SQRT ${sqrt}
#define DQ_FABS ${fabs}
#define DQ_HYPOT ${hypot}
#define DQ_FMIN ${fmin}
#define DQ_FMAX ${fmax}
#define DQ_CEIL ${ceil}
#define DQ_ISNAN ${isnan}
#define DQ_ISFINITE ${isfinite}
#define DQ_INF ((dq_scalar)INFINITY)
#define DQ_NAN ((dq_scalar)NAN)

/* Iteration counts for the current search: fixed counts, so the cost of
   dq_current_for_torque() is the same every control period. */
#define DQ_WEAKENING_SCAN_STEPS ${scan_steps}
#define DQ_WEAKENING_BISECT_STEPS ${bisect_steps}
#define DQ_MTPA_BISECT_STEPS ${mtpa_steps}

/* Half of RK4's absolute-stability reach on the imaginary axis (2.78). */
#define DQ_RK4_STABILITY_LIMIT ${rk4_limit}
/* Below this speed static friction is not engaged, matching the engine. */
#define DQ_STICTION_BAND ${stiction}
/* Relative slack in the feasibility tests: a point one rounding inside a limit
   counts as inside. */
#define DQ_FEASIBILITY_SLACK ${slack}
/* Relative tolerance for deciding which limit bound the envelope, matched to
   the discretisation of the envelope scan rather than to machine precision. */
#define DQ_ENVELOPE_SLACK ${envelope_slack}
/* |Ld - Lq| at or below this is treated as non-salient, matching the engine. */
#define DQ_SALIENCE_FLOOR ${salience_floor}

typedef ${ctype} dq_scalar;

static inline dq_scalar dq_sqrt(dq_scalar x) { return DQ_SQRT(x); }
static inline dq_scalar dq_fabs(dq_scalar x) { return DQ_FABS(x); }
static inline dq_scalar dq_hypot(dq_scalar x, dq_scalar y) { return DQ_HYPOT(x, y); }
static inline dq_scalar dq_fmin(dq_scalar x, dq_scalar y) { return DQ_FMIN(x, y); }
static inline dq_scalar dq_fmax(dq_scalar x, dq_scalar y) { return DQ_FMAX(x, y); }
static inline dq_scalar dq_ceil(dq_scalar x) { return DQ_CEIL(x); }

/* Every constant of the machine, referred to the motor shaft.  A value of
   DQ_INF in a limit field means the model does not enforce one, exactly as
   math.inf does in the Python engine. */
typedef struct {
    dq_scalar resistance;                    /* winding resistance per phase [ohm] */
    dq_scalar inductance_d;                  /* d-axis inductance [H] */
    dq_scalar inductance_q;                  /* q-axis inductance [H] */
    dq_scalar flux_linkage;                  /* magnet flux at reference_temperature [Wb] */
    int pole_pairs;                          /* pole pairs (dimensionless) */
    dq_scalar rotor_inertia;                 /* rotor inertia at the motor shaft [kg*m^2] */
    dq_scalar bearing_damping;               /* bearing viscous damping [N*m/(rad/s)] */
    dq_scalar static_friction;               /* bearing/gear Coulomb torque [N*m] */
    dq_scalar current_limit;                 /* phase current amplitude limit [A], per machine */
    dq_scalar voltage_limit;                 /* fundamental phase voltage limit [V peak]; the inverter rail, not the DC bus */
    dq_scalar dc_bus_voltage;                /* DC bus voltage [V]; DQ_INF means not modelled */
    dq_scalar core_loss_coefficient;         /* iron loss [W/(rad/s)^2 electrical] */
    dq_scalar windage_coefficient;           /* windage [W/(rad/s)^3] */
    dq_scalar flux_temp_coefficient;         /* flux temperature coefficient [fraction/K] */
    dq_scalar reference_temperature;         /* temperature flux_linkage was measured at [C] */
    int motor_count;                         /* machines running in parallel behind the same gearbox */
    dq_scalar gear_ratio;                    /* output speed / motor speed */
    dq_scalar gear_efficiency;               /* 0..1 */
    dq_scalar wheel_radius;                  /* [m]; 0 when there is no wheel in this system */
} dq_model;

/* A d/q pair: a voltage, or a current. */
typedef struct { dq_scalar d, q; } dq_pair;

/* State of the dq plant: rotor speed and the two current components. */
typedef struct { dq_scalar rot_vel, i_d, i_q; } dq_state;

/* Rates of the state, in the same order as dq_derivatives(). */
typedef struct { dq_scalar di_d, di_q, dw; } dq_rates;

/* The largest torque reachable at a speed, and the pair that produces it. */
typedef struct { dq_scalar torque, i_d, i_q; } dq_limits;

/* Power breakdown [W].  converted_W is the air-gap power -- the electrical
   power converted to mechanical -- and the drags sit below it, so
   input_W = copper_W + converted_W exactly. */
typedef struct {
    dq_scalar copper_W, iron_W, windage_W, bearing_W;
    dq_scalar converted_W, shaft_W, input_W;
} dq_power;

/* A current command and which constraint produced it.
   dq_current_for_torque() documents the flag contract. */
typedef struct {
    dq_scalar i_d, i_q;
    int current_saturated, voltage_saturated;
} dq_current;

extern const dq_model dq_model_default;
extern const dq_scalar dq_sample_time_s;   /* the control period this build was sized for */

#ifdef __cplusplus
extern "C" {
#endif

/* --- derived constants ------------------------------------------------- */

/* Magnet flux at a temperature [Wb]; pass DQ_NAN for the reference value. */
dq_scalar dq_effective_flux(const dq_model *m, dq_scalar temperature_c);

/* Torque per ampere of q current [N*m/A]: 1.5 * p * lambda.  The 1.5 is the
   amplitude-invariant Park convention, not a fudge: it is why torque_constant
   and back_emf_constant differ by exactly 1.5 here and in the Python engine. */
static inline dq_scalar dq_torque_constant(const dq_model *m) {
    return (dq_scalar)1.5 * (dq_scalar)m->pole_pairs * m->flux_linkage;
}

/* Back-EMF per mechanical rad/s [V/(rad/s)]: p * lambda. */
static inline dq_scalar dq_back_emf_constant(const dq_model *m) {
    return (dq_scalar)m->pole_pairs * m->flux_linkage;
}

/* Ld - Lq [H]; multiplies i_d * i_q in the torque law. */
static inline dq_scalar dq_reluctance(const dq_model *m) {
    return m->inductance_d - m->inductance_q;
}

/* Lq - Ld [H]: the sign the MTPA closed form needs.  Confusing this with
   dq_reluctance() inverts i_d and costs up to 6% of the current on a salient
   machine. */
static inline dq_scalar dq_mtpa_dl(const dq_model *m) {
    return m->inductance_q - m->inductance_d;
}

/* The demagnetising current that cancels the magnet flux [A]: lambda / Ld. */
static inline dq_scalar dq_characteristic_current(const dq_model *m) {
    return m->flux_linkage / m->inductance_d;
}

/* The most negative i_d this model will use [A].  No demagnetisation is
   modelled, so the search never asks for a current that would drive the magnet
   past its knee. */
static inline dq_scalar dq_id_floor(const dq_model *m) {
    return -dq_fmin(m->current_limit, dq_characteristic_current(m));
}

/* Speed at which the back-EMF alone reaches the rail [rad/s motor]: V / kv. */
static inline dq_scalar dq_base_speed(const dq_model *m) {
    return m->voltage_limit / dq_back_emf_constant(m);
}

/* Small-signal current time constant [s]. */
static inline dq_scalar dq_electrical_time_constant(const dq_model *m) {
    return dq_fmin(m->inductance_d, m->inductance_q) / m->resistance;
}

/* Real part of the dq current-loop pole pair [1/s].  With the shaft speed
   frozen the current loop linearises to a COMPLEX conjugate pair at
   -R*(Ld+Lq)/(2*Ld*Lq) +/- j*we.  RK4's absolute-stability region only reaches
   |lambda*h| ~ 2.78 on the imaginary axis, so the stable step of the plant
   shrinks as the machine speeds up. */
static inline dq_scalar dq_pole_damping(const dq_model *m) {
    return m->resistance * (m->inductance_d + m->inductance_q)
           / ((dq_scalar)2.0 * m->inductance_d * m->inductance_q);
}

/* Electrical angular velocity [rad/s electrical]: p * rot_vel. */
static inline dq_scalar dq_electrical_speed(const dq_model *m, dq_scalar rot_vel) {
    return (dq_scalar)m->pole_pairs * rot_vel;
}

/* Magnitude of the current-loop pole pair at a shaft speed [1/s]. */
static inline dq_scalar dq_pole_magnitude(const dq_model *m, dq_scalar rot_vel) {
    return dq_hypot(dq_pole_damping(m), dq_electrical_speed(m, rot_vel));
}

/* Largest stable fixed step for dq_step() [s], AT THE SPEED GIVEN.  This is NOT
   a constant: it falls roughly as 1/rot_vel once the electrical speed dominates
   the pole.  A brushed model's step limit is speed-independent because its pole
   is real; that limit is wrong here. */
static inline dq_scalar dq_max_step_time(const dq_model *m, dq_scalar rot_vel) {
    /* The cast is not decoration: DQ_RK4_STABILITY_LIMIT is printed at double
       precision so every build quotes the same number the Python module uses,
       and an f32 build would otherwise widen this division to double and drift
       from the f32 arithmetic the rest of the file does. */
    return dq_fmin((dq_scalar)0.5 * dq_electrical_time_constant(m),
                   (dq_scalar) DQ_RK4_STABILITY_LIMIT
                   / dq_pole_magnitude(m, rot_vel));
}

/* dTorque/di_q at this d-axis current [N*m/A] -- falls as you weaken the field. */
static inline dq_scalar dq_torque_gain(const dq_model *m, dq_scalar i_d) {
    return (dq_scalar)1.5 * (dq_scalar)m->pole_pairs
           * (m->flux_linkage + dq_reluctance(m) * i_d);
}

/* Shaft-side developed torque [N*m] from dq currents.  The reluctance term
   (Ld-Lq)*i_d*i_q is only present in a salient machine; with Ld == Lq this
   reduces to dq_torque_constant()*i_q.  `flux` is a flux LINKAGE in Wb (pass
   dq_effective_flux(m, 85.0) for a hot machine), NOT a temperature; DQ_NAN means
   "use the model's own flux_linkage". */
static inline dq_scalar dq_torque(const dq_model *m, dq_scalar i_d, dq_scalar i_q,
                                  dq_scalar flux) {
    dq_scalar lam = DQ_ISNAN(flux) ? m->flux_linkage : flux;
    return (dq_scalar)1.5 * (dq_scalar)m->pole_pairs
           * (lam * i_q + dq_reluctance(m) * i_d * i_q);
}

/* Output shaft speed from motor speed. */
static inline dq_scalar dq_output_speed(const dq_model *m, dq_scalar motor_rad_s) {
    return motor_rad_s * m->gear_ratio;
}

/* Output shaft torque from a motor current pair [N*m], after gearbox losses.
   motor_rad_s is unused: this gearbox has a constant ratio and a constant
   efficiency, so there is no speed-dependent loss to ask about.  The argument
   is kept so the call reads like dq_output_speed above. */
static inline dq_scalar dq_output_torque(const dq_model *m, dq_scalar motor_rad_s,
                                         dq_scalar i_d, dq_scalar i_q) {
    (void) motor_rad_s;
    return dq_torque(m, i_d, i_q, DQ_NAN) * m->gear_ratio * m->gear_efficiency;
}

/* Vehicle speed from motor speed [m/s]; 0 when there is no wheel. */
static inline dq_scalar dq_vehicle_speed(const dq_model *m, dq_scalar motor_rad_s) {
    if (m->wheel_radius <= (dq_scalar)0.0) {
        return (dq_scalar)0.0;
    }
    return dq_output_speed(m, motor_rad_s) * m->wheel_radius;
}

/* --- machine ------------------------------------------------------------ */

/* Steady-state dq voltages for a current at a speed (di/dt = 0). */
dq_pair dq_voltage_steady(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                          dq_scalar i_q);

/* dq voltages for a current at a speed, including the L*di/dt terms. */
dq_pair dq_voltage(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d, dq_scalar i_q,
                   dq_scalar di_d_dt, dq_scalar di_q_dt);

/* Magnitude of the steady-state voltage vector [V peak]. */
dq_scalar dq_voltage_magnitude(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                               dq_scalar i_q);

/* The largest i_q that keeps the voltage vector inside a circle of radius
   v_limit at a given i_d.  Returns DQ_NAN when no i_q fits at all, and DQ_INF
   when v_limit is not finite.  Solving |v(i_d,i_q)| = v is a quadratic in i_q,
   so this is exact: it is the shape of the voltage limit, an ellipse in the dq
   plane centred at i_d = -lambda/Ld and rotating with speed.  Pass DQ_NAN for
   v_limit to use the model's own. */
dq_scalar dq_iq_from_voltage(const dq_model *m, dq_scalar i_d, dq_scalar rot_vel,
                             dq_scalar v_limit);

/* Does a current pair sit inside the current circle (with slack)? */
int dq_fits_current(const dq_model *m, dq_scalar i_d, dq_scalar i_q);

/* Does a current pair sit inside the voltage ellipse (with slack)? */
int dq_feasible(const dq_model *m, dq_scalar i_d, dq_scalar i_q, dq_scalar rot_vel);

/* Minimum-current (i_d, i_q) for a torque ignoring the voltage limit.  Closed
   form for a salient machine: with dl = Lq - Ld and i_m the current magnitude,
   i_d = (lambda - sqrt(lambda^2 + 8*dl^2*i_m^2)) / (4*dl).  The sign of i_q
   follows the sign of tau; a negative i_d is the same angle for either
   direction of rotation.  q = DQ_INF when no current within current_limit can
   produce tau. */
dq_pair dq_mtpa_pair(const dq_model *m, dq_scalar tau);

/* The largest torque reachable at a speed, and the pair that produces it.
   Above base speed this falls off the current circle onto the voltage ellipse,
   and the pair sits at the id floor.  At a NEGATIVE speed the answer is the
   regenerative envelope, which is larger than the motoring one above base
   speed: the back-EMF and the applied voltage add instead of fighting. */
dq_limits dq_max_torque(const dq_model *m, dq_scalar rot_vel);

/* The current pair that produces a torque demand at a speed.  This is the
   search a drive's current loop runs: try MTPA, and when the voltage ellipse
   rejects it, walk i_d negative until the demand fits.  Fixed iteration
   counts, so the cost is the same every control period.

   A demand outside the envelope returns the best pair that IS reachable, with
   the flags set -- it does not saturate a MTPA answer and call that the
   torque.  Zero torque above the rail speed still needs current: the weakening
   angle that makes v_q zero is i_d = -lambda/Ld.

   The flags say which constraint PRODUCED the answer, not which one the answer
   happens to violate by rounding: a pair from the field-weakening branch always
   reports voltage_saturated 1, and a torque sitting exactly on the current
   circle may report current_saturated 1 or 0 depending on where the
   fixed-count bisection landed.  The current magnitude is the same either way. */
dq_current dq_current_for_torque(const dq_model *m, dq_scalar tau, dq_scalar rot_vel);

/* Coulomb + viscous + iron drag torque [N*m]; always opposes motion.  Iron and
   windage are specified as POWERS, so dividing by speed makes the iron term
   look linear in mechanical speed even though it is quadratic in electrical
   speed. */
dq_scalar dq_loss_torque(const dq_model *m, dq_scalar rot_vel);

/* Developed torque needed to follow accel at rot_vel against friction and
   load.  load_torque is a NON-NEGATIVE magnitude that opposes motion; a torque
   that actively assists the motion (a grade, a spring) has to be added to this
   as a signed torque, not passed as a negative magnitude, or it gets counted
   twice.  C has no exception: a negative load_torque returns DQ_NAN instead of
   raising, so check for NaN if that mistake is possible in your caller. */
dq_scalar dq_required_torque(const dq_model *m, dq_scalar rot_vel, dq_scalar accel,
                             dq_scalar load_torque);

/* Power balance at an operating point [W].  input_W = copper_W + converted_W
   exactly -- that is the whole of the electrical input in this model.  iron,
   windage and bearing are drags acting on the shaft, so they sit INSIDE the
   converted power: shaft_W = converted_W - iron_W - windage_W - bearing_W.
   Adding them to the input as well would count them twice. */
void dq_losses(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d, dq_scalar i_q,
               dq_power *out);

/* Current drawn from the DC link [A] -- what a battery ammeter reads.  Not the
   winding current: the 1.5 is the three-phase sum under the amplitude-invariant
   convention, and the whole point of this function is that a drive fitted from
   rail current alone is fitting a different quantity.  Returns DQ_NAN when the
   bus voltage is not finite. */
dq_scalar dq_dc_link_current(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                             dq_scalar i_q);

/* Motor speed up to which full current_limit still fits the voltage rail
   [rad/s].  |v|^2 = V^2 at i_d = 0, i_q = I is quadratic in w_e with
   a = (Lq*I)^2 + lambda^2, b = 2*lambda*R*I, c = (R*I)^2 - V^2.  This is below
   base speed whenever Lq*I is comparable to lambda, because the drive has to
   spend part of its voltage budget on the cross-coupling term -w_e*Lq*i_q. */
dq_scalar dq_constant_torque_speed(const dq_model *m);

/* --- plant -------------------------------------------------------------- */

/* State rates for the plant equations: (di_d/dt, di_q/dt, dw/dt). */
void dq_derivatives(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
                    dq_scalar load_torque, dq_rates *out);

/* Advance the plant one FIXED step with classic RK4.  Returns 0, or -1 when dt
   exceeds dq_max_step_time() at the speed the state is at: the dq frame rotates
   the back-EMF coupling at we = p*w, so the pole pair moves up the imaginary
   axis with speed and RK4 goes unstable.  The failure is silent divergence, so
   the guard is strict and dq_integrate() is the way to run a normal control
   period. */
int dq_step(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
            dq_scalar dt, dq_scalar load_torque, dq_state *out);

/* RK4 sub-steps a control period of dt needs at rot_vel (>= 1). */
int dq_sub_steps(const dq_model *m, dq_scalar dt, dq_scalar rot_vel);

/* Advance one control period, sub-divided until every RK4 step is stable at
   the speed REACHED, re-sizing every pass.  A fixed subdivision sized for
   standstill is exactly the bug this avoids.  Returns 0, -1 for a machine with
   a zero stable step, -2 if the sub-step guard trips (the step contract and the
   speed are inconsistent). */
int dq_integrate(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
                 dq_scalar dt, dq_scalar load_torque, dq_state *out);

/* Advance one step assuming the current loop already holds the command: the
   currents jump to the command and only the shaft integrates, with the torque
   held constant across the step.  Keep dt small against the mechanical time
   constant. */
dq_state dq_step_current(const dq_model *m, dq_state state, dq_scalar i_d_cmd,
                         dq_scalar i_q_cmd, dq_scalar dt, dq_scalar load_torque);

/* --- feed-forward -------------------------------------------------------- */

/* A clocked feed-forward law: a speed target in, a phase-voltage pair out.

     (i_d, i_q) = dq_current_for_torque(J*a + b*w + load + friction, w)
     (v_d, v_q) = dq_voltage(w, i_d, i_q)

   The failure mode in the field is feeding it an a taken from the derivative of
   an unlimited reference, so the reference is rate limited HERE, on the
   INCREMENT, not on the absolute value -- clamping the absolute value is what
   pins a 129 rad/s command at 0.09 rad/s and looks like a broken model.

   speed_limit/accel_limit/load_torque pass DQ_NAN to take the model's own
   defaults (base speed, stall torque over rotor inertia, zero); those defaults
   are the MODEL's limits, not the vehicle's.  Set them to what the machine can
   actually do, or the law commands a trajectory the current loop cannot follow. */
typedef struct {
    const dq_model *model;
    dq_scalar sample_time;
    dq_scalar speed_limit;
    dq_scalar accel_limit;
    dq_scalar load_torque;
    dq_scalar w_ref;   /* the rate-limited speed reference the law is following */
    dq_scalar a_ref;   /* its derivative, which is what the torque law is fed */
    int current_saturated;
    int voltage_saturated;
} dq_feedforward;

/* Stall torque of the machine at the current/voltage limits [N*m]. */
static inline dq_scalar dq_ff_stall_torque(const dq_model *m) {
    dq_scalar i = m->voltage_limit / m->resistance;
    if (DQ_ISFINITE(m->current_limit)) {
        i = dq_fmin(i, m->current_limit);
    }
    return dq_torque_constant(m) * i;
}

/* Returns 0, or -1 when sample_time is not positive (a zero period makes the
   rate limiter a no-op and the law meaningless). */
int dq_ff_init(dq_feedforward *ff, const dq_model *m, dq_scalar sample_time,
               dq_scalar speed_limit, dq_scalar accel_limit, dq_scalar load_torque);

/* Restart the reference at rot_vel (clamped to the speed limit) with zero
   acceleration. */
void dq_ff_reset(dq_feedforward *ff, dq_scalar rot_vel);

/* Advance the internal speed reference one sample; returns (w_ref, a_ref).
   The clamp is on the increment (accel_limit * sample_time), which is what
   makes a_ref the real acceleration the machine will see. */
dq_pair dq_ff_ramp(dq_feedforward *ff, dq_scalar target);

/* (i_d, i_q) the model needs at this operating point, with the flags updated.
   Pass DQ_NAN for rot_vel/accel/load_torque to use the internal reference and
   the configured load. */
dq_pair dq_ff_current(dq_feedforward *ff, dq_scalar rot_vel, dq_scalar accel,
                      dq_scalar load_torque);

/* One control period: ramps the reference, then returns the voltages in *out_v
   and the current in *out_i. */
void dq_ff_update(dq_feedforward *ff, dq_scalar target, dq_pair *out_v,
                  dq_current *out_i);

#ifdef __cplusplus
}
#endif

#endif /* ${guard} */
''')

_DQ_C_SOURCE = Template(r'''${banner}

#include "${header}"

/* Every constant of the machine, in the order the header declares them.  The
   values are the ones the Python engine holds, rounded once to this build's
   float type, so the C, the Rust and the Python all read the same numbers. */
const dq_model dq_model_default = {
    /*resistance*/ ${resistance},
    /*inductance_d*/ ${inductance_d},
    /*inductance_q*/ ${inductance_q},
    /*flux_linkage*/ ${flux_linkage},
    /*pole_pairs*/ ${pole_pairs},
    /*rotor_inertia*/ ${rotor_inertia},
    /*bearing_damping*/ ${bearing_damping},
    /*static_friction*/ ${static_friction},
    /*current_limit*/ ${current_limit},
    /*voltage_limit*/ ${voltage_limit},
    /*dc_bus_voltage*/ ${dc_bus_voltage},
    /*core_loss_coefficient*/ ${core_loss_coefficient},
    /*windage_coefficient*/ ${windage_coefficient},
    /*flux_temp_coefficient*/ ${flux_temp_coefficient},
    /*reference_temperature*/ ${reference_temperature},
    /*motor_count*/ ${motor_count},
    /*gear_ratio*/ ${gear_ratio},
    /*gear_efficiency*/ ${gear_efficiency},
    /*wheel_radius*/ ${wheel_radius},
};

const dq_scalar dq_sample_time_s = ${sample_time};

/* --- machine ------------------------------------------------------------ */

dq_scalar dq_effective_flux(const dq_model *m, dq_scalar temperature_c) {
    if (DQ_ISNAN(temperature_c)) {
        return m->flux_linkage;
    }
    return m->flux_linkage
           * ((dq_scalar)1.0 + m->flux_temp_coefficient
              * (temperature_c - m->reference_temperature));
}

dq_pair dq_voltage_steady(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                          dq_scalar i_q) {
    return dq_voltage(m, rot_vel, i_d, i_q, (dq_scalar)0.0, (dq_scalar)0.0);
}

dq_pair dq_voltage(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d, dq_scalar i_q,
                   dq_scalar di_d_dt, dq_scalar di_q_dt) {
    dq_scalar we = dq_electrical_speed(m, rot_vel);
    dq_pair v;
    v.d = m->resistance * i_d - we * m->inductance_q * i_q
          + m->inductance_d * di_d_dt;
    v.q = m->resistance * i_q + we * (m->inductance_d * i_d + m->flux_linkage)
          + m->inductance_q * di_q_dt;
    return v;
}

dq_scalar dq_voltage_magnitude(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                               dq_scalar i_q) {
    dq_pair v = dq_voltage_steady(m, rot_vel, i_d, i_q);
    return dq_hypot(v.d, v.q);
}

dq_scalar dq_iq_from_voltage(const dq_model *m, dq_scalar i_d, dq_scalar rot_vel,
                             dq_scalar v_limit) {
    dq_scalar v = DQ_ISNAN(v_limit) ? m->voltage_limit : v_limit;
    if (!DQ_ISFINITE(v)) {
        return DQ_INF;
    }
    dq_scalar we = dq_electrical_speed(m, rot_vel);
    dq_scalar a = m->resistance * m->resistance
                  + (we * m->inductance_q) * (we * m->inductance_q);
    dq_scalar b = (dq_scalar)2.0 * m->resistance * we
                  * (m->flux_linkage + dq_reluctance(m) * i_d);
    dq_scalar r = m->resistance * i_d;
    dq_scalar e = we * (m->inductance_d * i_d + m->flux_linkage);
    dq_scalar c = r * r + e * e - v * v;
    dq_scalar disc = b * b - (dq_scalar)4.0 * a * c;
    if (disc < (dq_scalar)0.0) {
        return DQ_NAN;   /* no i_q fits at all: the back-EMF alone exceeds the rail */
    }
    return (-b + dq_sqrt(disc)) / ((dq_scalar)2.0 * a);
}

int dq_fits_current(const dq_model *m, dq_scalar i_d, dq_scalar i_q) {
    if (!DQ_ISFINITE(m->current_limit)) {
        return 1;
    }
    return i_d * i_d + i_q * i_q
           <= m->current_limit * m->current_limit * ((dq_scalar)1.0 + DQ_FEASIBILITY_SLACK);
}

int dq_feasible(const dq_model *m, dq_scalar i_d, dq_scalar i_q, dq_scalar rot_vel) {
    if (!dq_fits_current(m, i_d, i_q)) {
        return 0;
    }
    return dq_voltage_magnitude(m, rot_vel, i_d, i_q)
           <= m->voltage_limit * ((dq_scalar)1.0 + DQ_FEASIBILITY_SLACK);
}

/* Torque on the MTPA locus at current magnitude i_m, for the bisection. */
static dq_scalar dq_mtpa_torque_at(const dq_model *m, dq_scalar i_m) {
    dq_scalar dl = dq_mtpa_dl(m);
    dq_scalar lam = m->flux_linkage;
    dq_scalar i_d = (lam - dq_sqrt(lam * lam + (dq_scalar)8.0 * dl * dl * i_m * i_m))
                    / ((dq_scalar)4.0 * dl);
    dq_scalar i_q_sq = i_m * i_m - i_d * i_d;
    if (i_q_sq <= (dq_scalar)0.0) {
        return (dq_scalar)0.0;
    }
    return dq_torque(m, i_d, dq_sqrt(i_q_sq), DQ_NAN);
}

dq_pair dq_mtpa_pair(const dq_model *m, dq_scalar tau) {
    dq_scalar sign = tau >= (dq_scalar)0.0 ? (dq_scalar)1.0 : (dq_scalar)-1.0;
    dq_scalar tau_mag = dq_fabs(tau);
    dq_pair out;

    if (dq_fabs(dq_reluctance(m)) <= DQ_SALIENCE_FLOOR) {
        /* Not salient: MTPA degenerates to i_d = 0, and any i_d costs current
           for no torque. */
        out.d = (dq_scalar)0.0;
        if (!DQ_ISFINITE(m->current_limit)) {
            out.q = sign * (tau_mag / dq_torque_constant(m));
            return out;
        }
        if (dq_torque_constant(m) * m->current_limit < tau_mag) {
            out.q = DQ_INF;
            return out;
        }
        out.q = sign * (tau_mag / dq_torque_constant(m));
        return out;
    }

    dq_scalar ceiling = DQ_ISFINITE(m->current_limit) ? m->current_limit : DQ_INF;
    if (DQ_ISFINITE(ceiling)
        && dq_mtpa_torque_at(m, ceiling) < tau_mag * ((dq_scalar)1.0 - DQ_FEASIBILITY_SLACK)) {
        /* No current inside the circle produces this torque: report it the way
           the caller expects, with an infinite q rather than a plausible pair
           that quietly saturates. */
        out.d = (dq_scalar)0.0;
        out.q = DQ_INF;
        return out;
    }
    dq_scalar hi = DQ_ISFINITE(ceiling) ? ceiling
                   : tau_mag / dq_torque_constant(m) * (dq_scalar)4.0;
    dq_scalar lo = (dq_scalar)0.0;
    for (int k = 0; k < DQ_MTPA_BISECT_STEPS; ++k) {
        dq_scalar mid = (dq_scalar)0.5 * (lo + hi);
        if (dq_mtpa_torque_at(m, mid) < tau_mag) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    dq_scalar i_m = (dq_scalar)0.5 * (lo + hi);
    dq_scalar dl = dq_mtpa_dl(m);
    dq_scalar lam = m->flux_linkage;
    dq_scalar i_d = (lam - dq_sqrt(lam * lam + (dq_scalar)8.0 * dl * dl * i_m * i_m))
                    / ((dq_scalar)4.0 * dl);
    dq_scalar i_q_sq = i_m * i_m - i_d * i_d;
    out.d = i_d;
    out.q = sign * (i_q_sq > (dq_scalar)0.0 ? dq_sqrt(i_q_sq) : (dq_scalar)0.0);
    return out;
}

dq_limits dq_max_torque(const dq_model *m, dq_scalar rot_vel) {
    dq_limits best;
    best.torque = (dq_scalar)0.0;
    best.i_d = (dq_scalar)0.0;
    best.i_q = (dq_scalar)0.0;
    dq_scalar floor = dq_id_floor(m);
    dq_scalar step = ((dq_scalar)0.0 - floor) / (dq_scalar)DQ_WEAKENING_SCAN_STEPS;

    for (int k = 0; k <= DQ_WEAKENING_SCAN_STEPS; ++k) {
        dq_scalar i_d = -(dq_scalar)k * step;
        dq_scalar i_q_c = DQ_ISFINITE(m->current_limit)
            ? dq_sqrt(dq_fmax((dq_scalar)0.0,
                              m->current_limit * m->current_limit - i_d * i_d))
            : DQ_INF;
        dq_scalar i_q_v = dq_iq_from_voltage(m, i_d, rot_vel, DQ_NAN);
        if (DQ_ISNAN(i_q_v)) {
            continue;
        }
        dq_scalar i_q = dq_fmin(i_q_c, i_q_v);
        dq_scalar t = dq_torque(m, i_d, i_q, DQ_NAN);
        if (t > best.torque) {
            best.torque = t;
            best.i_d = i_d;
            best.i_q = i_q;
        }
    }
    /* Refine around the best i_d with a finer scan: bounded, and it cannot
       wander the way a bisection on a non-monotone objective does. */
    if (step > (dq_scalar)0.0) {
        dq_scalar centre = best.i_d;
        dq_scalar fine = step / (dq_scalar)16.0;
        for (int k = -16; k <= 16; ++k) {
            dq_scalar i_d = centre + (dq_scalar)k * fine;
            if (i_d < floor || i_d > (dq_scalar)0.0) {
                continue;
            }
            dq_scalar i_q_c = DQ_ISFINITE(m->current_limit)
                ? dq_sqrt(dq_fmax((dq_scalar)0.0,
                                  m->current_limit * m->current_limit - i_d * i_d))
                : DQ_INF;
            dq_scalar i_q_v = dq_iq_from_voltage(m, i_d, rot_vel, DQ_NAN);
            if (DQ_ISNAN(i_q_v)) {
                continue;
            }
            dq_scalar i_q = dq_fmin(i_q_c, i_q_v);
            dq_scalar t = dq_torque(m, i_d, i_q, DQ_NAN);
            if (t > best.torque) {
                best.torque = t;
                best.i_d = i_d;
                best.i_q = i_q;
            }
        }
    }
    return best;
}

dq_current dq_current_for_torque(const dq_model *m, dq_scalar tau, dq_scalar rot_vel) {
    dq_current out;
    dq_scalar sign = tau >= (dq_scalar)0.0 ? (dq_scalar)1.0 : (dq_scalar)-1.0;
    dq_scalar w = sign > (dq_scalar)0.0 ? rot_vel : -rot_vel;
    dq_scalar tau_mag = dq_fabs(tau);

    dq_pair mtpa = dq_mtpa_pair(m, tau_mag);
    if (DQ_ISFINITE(mtpa.q) && dq_feasible(m, mtpa.d, mtpa.q, w)) {
        out.i_d = mtpa.d;
        out.i_q = sign * mtpa.q;
        out.current_saturated = 0;
        out.voltage_saturated = 0;
        return out;
    }

    /* Field weakening: for i_d < 0 the current magnitude falls monotonically as
       i_d rises toward zero, so the answer is the LARGEST feasible i_d. */
    dq_scalar floor = dq_id_floor(m);
    dq_scalar step = ((dq_scalar)0.0 - floor) / (dq_scalar)DQ_WEAKENING_SCAN_STEPS;
    dq_scalar found = DQ_NAN;
    for (int k = 0; k <= DQ_WEAKENING_SCAN_STEPS; ++k) {
        dq_scalar i_d_try = -(dq_scalar)k * step;
        dq_scalar gain = dq_torque_gain(m, i_d_try);
        if (gain <= (dq_scalar)0.0) {
            continue;
        }
        if (dq_feasible(m, i_d_try, tau_mag / gain, w)) {
            found = i_d_try;
            break;
        }
    }
    if (DQ_ISNAN(found)) {
        /* The request is outside the envelope at this speed: hand back the best
           this speed can do and name the constraint that bounded it, the way the
           engine's _which_limit_bound does, so the flags mean the same thing in
           both implementations.  The tolerance matches max_torque()'s fixed-count
           scan (a few 1e-4), not machine precision. */
        dq_limits lim = dq_max_torque(m, w);
        int on_current = DQ_ISFINITE(m->current_limit)
            && dq_hypot(lim.i_d, lim.i_q)
               >= m->current_limit * ((dq_scalar)1.0 - DQ_ENVELOPE_SLACK);
        int on_voltage = dq_voltage_magnitude(m, w, lim.i_d, lim.i_q)
            >= m->voltage_limit * ((dq_scalar)1.0 - DQ_ENVELOPE_SLACK);
        out.i_d = lim.i_d;
        out.i_q = sign * lim.i_q;
        out.current_saturated = on_current;
        out.voltage_saturated = on_voltage;
        return out;
    }
    dq_scalar bad = found + step;
    for (int k = 0; k < DQ_WEAKENING_BISECT_STEPS; ++k) {
        dq_scalar mid = (dq_scalar)0.5 * (found + bad);
        dq_scalar gain = dq_torque_gain(m, mid);
        if (gain > (dq_scalar)0.0 && dq_feasible(m, mid, tau_mag / gain, w)) {
            found = mid;
        } else {
            bad = mid;
        }
    }
    dq_scalar gain = dq_torque_gain(m, found);
    dq_scalar i_q = gain > (dq_scalar)0.0 ? tau_mag / gain : (dq_scalar)0.0;
    out.i_d = found;
    out.i_q = sign * i_q;
    out.current_saturated = !dq_fits_current(m, found, i_q);
    /* This branch exists only because the MTPA angle did not fit, and it
       bisects ONTO the voltage ellipse, so the drive is on the rail by
       construction: comparing the magnitude with a strict > would report 0 for
       a point that sits one rounding inside the rail. */
    out.voltage_saturated = 1;
    return out;
}

dq_scalar dq_loss_torque(const dq_model *m, dq_scalar rot_vel) {
    dq_scalar w = dq_fabs(rot_vel);
    if (w <= (dq_scalar)0.0) {
        return (dq_scalar)0.0;
    }
    dq_scalar we = dq_electrical_speed(m, w);
    return m->bearing_damping * w + m->windage_coefficient * w * w
           + m->core_loss_coefficient * we * we / w;
}

dq_scalar dq_required_torque(const dq_model *m, dq_scalar rot_vel, dq_scalar accel,
                             dq_scalar load_torque) {
    if (load_torque < (dq_scalar)0.0) {
        /* C has no exception to raise: a negative load is a caller error, and
           NaN propagates into whatever the caller computes next rather than
           silently counting the assisting torque twice. */
        return DQ_NAN;
    }
    dq_scalar w = rot_vel;
    dq_scalar tau = m->rotor_inertia * accel + m->bearing_damping * w;
    if (w > DQ_STICTION_BAND) {
        tau += m->static_friction + load_torque;
    } else if (w < -DQ_STICTION_BAND) {
        tau -= m->static_friction + load_torque;
    } else {
        tau += m->static_friction + load_torque;
    }
    return tau;
}

void dq_losses(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d, dq_scalar i_q,
               dq_power *out) {
    dq_scalar we = dq_electrical_speed(m, rot_vel);
    dq_scalar w = dq_fabs(rot_vel);
    out->copper_W = (dq_scalar)1.5 * m->resistance * (i_d * i_d + i_q * i_q);
    out->iron_W = m->core_loss_coefficient * we * we;
    out->windage_W = m->windage_coefficient * w * w * w;
    out->bearing_W = m->bearing_damping * rot_vel * rot_vel;
    out->converted_W = dq_torque(m, i_d, i_q, DQ_NAN) * rot_vel;
    out->shaft_W = out->converted_W - out->iron_W - out->windage_W - out->bearing_W;
    out->input_W = out->copper_W + out->converted_W;
}

dq_scalar dq_dc_link_current(const dq_model *m, dq_scalar rot_vel, dq_scalar i_d,
                             dq_scalar i_q) {
    if (!DQ_ISFINITE(m->dc_bus_voltage)) {
        return DQ_NAN;
    }
    dq_pair v = dq_voltage_steady(m, rot_vel, i_d, i_q);
    return (dq_scalar)m->motor_count * (dq_scalar)1.5 * (v.d * i_d + v.q * i_q)
           / m->dc_bus_voltage;
}

dq_scalar dq_constant_torque_speed(const dq_model *m) {
    if (!DQ_ISFINITE(m->current_limit)) {
        return dq_base_speed(m);
    }
    dq_scalar i = m->current_limit;
    dq_scalar lam = m->flux_linkage;
    dq_scalar r = m->resistance;
    dq_scalar a = (m->inductance_q * i) * (m->inductance_q * i) + lam * lam;
    dq_scalar b = (dq_scalar)2.0 * lam * r * i;
    dq_scalar c = (r * i) * (r * i) - m->voltage_limit * m->voltage_limit;
    dq_scalar disc = b * b - (dq_scalar)4.0 * a * c;
    if (disc <= (dq_scalar)0.0) {
        return (dq_scalar)0.0;
    }
    return (dq_sqrt(disc) - b) / ((dq_scalar)2.0 * a)
           / (dq_scalar)m->pole_pairs;
}

/* --- plant -------------------------------------------------------------- */

void dq_derivatives(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
                    dq_scalar load_torque, dq_rates *out) {
    dq_scalar we = dq_electrical_speed(m, state.rot_vel);
    out->di_d = (v_d - m->resistance * state.i_d + we * m->inductance_q * state.i_q)
                / m->inductance_d;
    out->di_q = (v_q - m->resistance * state.i_q
                 - we * (m->inductance_d * state.i_d + m->flux_linkage))
                / m->inductance_q;
    dq_scalar tau = dq_torque(m, state.i_d, state.i_q, DQ_NAN);
    dq_scalar tau_load = load_torque + dq_loss_torque(m, state.rot_vel);
    out->dw = (tau - tau_load) / m->rotor_inertia;
}

int dq_step(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
            dq_scalar dt, dq_scalar load_torque, dq_state *out) {
    dq_scalar limit = dq_max_step_time(m, state.rot_vel);
    if (dt > limit) {
        return -1;
    }
    dq_rates a1, a2, a3, a4;
    dq_state s2, s3, s4;
    dq_derivatives(m, state, v_d, v_q, load_torque, &a1);
    s2.rot_vel = state.rot_vel + (dq_scalar)0.5 * dt * a1.dw;
    s2.i_d = state.i_d + (dq_scalar)0.5 * dt * a1.di_d;
    s2.i_q = state.i_q + (dq_scalar)0.5 * dt * a1.di_q;
    dq_derivatives(m, s2, v_d, v_q, load_torque, &a2);
    s3.rot_vel = state.rot_vel + (dq_scalar)0.5 * dt * a2.dw;
    s3.i_d = state.i_d + (dq_scalar)0.5 * dt * a2.di_d;
    s3.i_q = state.i_q + (dq_scalar)0.5 * dt * a2.di_q;
    dq_derivatives(m, s3, v_d, v_q, load_torque, &a3);
    s4.rot_vel = state.rot_vel + dt * a3.dw;
    s4.i_d = state.i_d + dt * a3.di_d;
    s4.i_q = state.i_q + dt * a3.di_q;
    dq_derivatives(m, s4, v_d, v_q, load_torque, &a4);
    out->rot_vel = state.rot_vel + dt / (dq_scalar)6.0
                   * (a1.dw + (dq_scalar)2.0 * a2.dw + (dq_scalar)2.0 * a3.dw + a4.dw);
    out->i_d = state.i_d + dt / (dq_scalar)6.0
               * (a1.di_d + (dq_scalar)2.0 * a2.di_d + (dq_scalar)2.0 * a3.di_d + a4.di_d);
    out->i_q = state.i_q + dt / (dq_scalar)6.0
               * (a1.di_q + (dq_scalar)2.0 * a2.di_q + (dq_scalar)2.0 * a3.di_q + a4.di_q);
    return 0;
}

int dq_sub_steps(const dq_model *m, dq_scalar dt, dq_scalar rot_vel) {
    dq_scalar limit = dq_max_step_time(m, rot_vel);
    if (limit <= (dq_scalar)0.0 || dt <= limit) {
        return 1;
    }
    return (int)dq_ceil(dt / limit);
}

int dq_integrate(const dq_model *m, dq_state state, dq_scalar v_d, dq_scalar v_q,
                 dq_scalar dt, dq_scalar load_torque, dq_state *out) {
    dq_scalar remaining = dt;
    dq_state s = state;
    int n = 0;
    while (remaining > (dq_scalar)0.0) {
        dq_scalar h = dq_fmin(remaining, dq_max_step_time(m, s.rot_vel));
        if (h <= (dq_scalar)0.0) {
            return -1;   /* this machine has a zero stable step: check the inductances */
        }
        dq_state next;
        if (dq_step(m, s, v_d, v_q, h, load_torque, &next) != 0) {
            return -2;
        }
        s = next;
        remaining -= h;
        if (++n > 100000) {
            return -2;   /* the step contract and the speed are inconsistent */
        }
    }
    *out = s;
    return 0;
}

dq_state dq_step_current(const dq_model *m, dq_state state, dq_scalar i_d_cmd,
                         dq_scalar i_q_cmd, dq_scalar dt, dq_scalar load_torque) {
    /* The current loop is assumed fast enough to hold the command: the currents
       jump to it and only the shaft integrates, with the torque held constant
       across the RK4 stages. */
    dq_scalar tau = dq_torque(m, i_d_cmd, i_q_cmd, DQ_NAN);
    dq_scalar w = state.rot_vel;
    dq_scalar k1 = (tau - load_torque - dq_loss_torque(m, w)) / m->rotor_inertia;
    dq_scalar k2 = (tau - load_torque
                    - dq_loss_torque(m, w + (dq_scalar)0.5 * dt * k1)) / m->rotor_inertia;
    dq_scalar k3 = (tau - load_torque
                    - dq_loss_torque(m, w + (dq_scalar)0.5 * dt * k2)) / m->rotor_inertia;
    dq_scalar k4 = (tau - load_torque
                    - dq_loss_torque(m, w + dt * k3)) / m->rotor_inertia;
    dq_state s;
    s.rot_vel = w + dt / (dq_scalar)6.0
                * (k1 + (dq_scalar)2.0 * k2 + (dq_scalar)2.0 * k3 + k4);
    s.i_d = i_d_cmd;
    s.i_q = i_q_cmd;
    return s;
}

/* --- feed-forward -------------------------------------------------------- */

int dq_ff_init(dq_feedforward *ff, const dq_model *m, dq_scalar sample_time,
               dq_scalar speed_limit, dq_scalar accel_limit, dq_scalar load_torque) {
    if (!(sample_time > (dq_scalar)0.0)) {
        return -1;   /* a zero period makes the rate limiter a no-op */
    }
    ff->model = m;
    ff->sample_time = sample_time;
    ff->speed_limit = DQ_ISNAN(speed_limit) ? dq_base_speed(m) : speed_limit;
    ff->accel_limit = DQ_ISNAN(accel_limit)
        ? dq_ff_stall_torque(m) / m->rotor_inertia : accel_limit;
    ff->load_torque = DQ_ISNAN(load_torque) ? (dq_scalar)0.0 : load_torque;
    ff->w_ref = (dq_scalar)0.0;
    ff->a_ref = (dq_scalar)0.0;
    ff->current_saturated = 0;
    ff->voltage_saturated = 0;
    return 0;
}

void dq_ff_reset(dq_feedforward *ff, dq_scalar rot_vel) {
    dq_scalar limit = ff->speed_limit;
    dq_scalar w = rot_vel;
    if (w > limit) {
        w = limit;
    } else if (w < -limit) {
        w = -limit;
    }
    ff->w_ref = w;
    ff->a_ref = (dq_scalar)0.0;
}

dq_pair dq_ff_ramp(dq_feedforward *ff, dq_scalar target) {
    dq_scalar limit = ff->speed_limit;
    dq_pair out;
    if (target > limit) {
        target = limit;
    } else if (target < -limit) {
        target = -limit;
    }
    dq_scalar max_step = ff->accel_limit * ff->sample_time;
    dq_scalar delta = target - ff->w_ref;
    if (delta > max_step) {
        delta = max_step;
    } else if (delta < -max_step) {
        delta = -max_step;
    }
    dq_scalar nxt = ff->w_ref + delta;
    ff->a_ref = (nxt - ff->w_ref) / ff->sample_time;
    ff->w_ref = nxt;
    out.d = ff->w_ref;
    out.q = ff->a_ref;
    return out;
}

dq_pair dq_ff_current(dq_feedforward *ff, dq_scalar rot_vel, dq_scalar accel,
                      dq_scalar load_torque) {
    dq_model *m = (dq_model *)ff->model;
    dq_scalar w = DQ_ISNAN(rot_vel) ? ff->w_ref : rot_vel;
    dq_scalar a = DQ_ISNAN(accel) ? ff->a_ref : accel;
    dq_scalar load = DQ_ISNAN(load_torque) ? ff->load_torque : load_torque;
    dq_scalar tau = dq_required_torque(m, w, a, load);
    dq_current c = dq_current_for_torque(m, tau, w);
    ff->current_saturated = c.current_saturated;
    ff->voltage_saturated = c.voltage_saturated;
    dq_pair out;
    out.d = c.i_d;
    out.q = c.i_q;
    return out;
}

void dq_ff_update(dq_feedforward *ff, dq_scalar target, dq_pair *out_v,
                  dq_current *out_i) {
    dq_model *m = (dq_model *)ff->model;
    dq_pair ref = dq_ff_ramp(ff, target);
    dq_scalar w = ref.d;
    dq_scalar a = ref.q;
    dq_scalar tau = dq_required_torque(m, w, a, ff->load_torque);
    dq_current c = dq_current_for_torque(m, tau, w);
    ff->current_saturated = c.current_saturated;
    ff->voltage_saturated = c.voltage_saturated;
    dq_pair v = dq_voltage_steady(m, w, c.i_d, c.i_q);
    if (dq_voltage_magnitude(m, w, c.i_d, c.i_q) > m->voltage_limit) {
        dq_scalar scale = m->voltage_limit
                          / dq_voltage_magnitude(m, w, c.i_d, c.i_q);
        v.d = v.d * scale;
        v.q = v.q * scale;
        ff->voltage_saturated = 1;
    }
    c.voltage_saturated = ff->voltage_saturated;
    *out_v = v;
    *out_i = c;
}
''')

_DQ_BENCH_C = Template(r'''${banner}
/*
 * Step-response bench for the generated dq model.
 *
 * Build and run:
 *
 *   cc -std=c11 -Wall -Wextra -pedantic -I. ${file} ${source_file} -lm -o bench_dq
 *   ./bench_dq            # open-loop plant, one voltage step
 *   ./bench_dq feedforward  # the feed-forward law driving the same plant
 *
 * Generated values: dt = ${dt_text} s, duration = ${duration_text} s, step at
 * t = ${step_at_text} s, applied q-axis voltage = ${voltage_text} V, feed-forward
 * target = ${target_text} rad/s motor side (1.5x base speed, so the run crosses
 * the field-weakening band edge).
 *
 * stdout is CSV: time_s,motor_rad_s,current_A,voltage_V.  The current column is
 * the PLANT's |i| = hypot(i_d, i_q), not the command, and the voltage column is
 * the magnitude actually applied.  stderr carries the sub-step commentary, so
 * the CSV stays pipeable.
 */
#include <stdio.h>
#include <string.h>

#include "${header}"

static void print_row(dq_scalar t, const dq_state *s, dq_scalar v_d, dq_scalar v_q)
{
    printf("%.6f,%.6f,%.6f,%.6f\n", (double) t, (double) s->rot_vel,
           (double) dq_hypot(s->i_d, s->i_q), (double) dq_hypot(v_d, v_q));
}

static int plant_run(dq_scalar dt, dq_scalar duration, dq_scalar step_at,
                     dq_scalar voltage)
{
    const dq_model *m = &dq_model_default;
    dq_state s, next;
    dq_scalar t = (dq_scalar)0.0;
    long n = 0;

    s.rot_vel = (dq_scalar)0.0;
    s.i_d = (dq_scalar)0.0;
    s.i_q = (dq_scalar)0.0;
    while (t < duration - (dq_scalar)0.5 * dt) {
        dq_scalar v_d, v_q;
        if (++n > ${n_steps}) {
            break;
        }
        v_d = (dq_scalar)0.0;
        v_q = (t < step_at) ? (dq_scalar)0.0 : voltage;
        if (dq_integrate(m, s, v_d, v_q, dt, (dq_scalar)0.0, &next) != 0) {
            fprintf(stderr, "dq_integrate refused this step\n");
            return 1;
        }
        s = next;
        t += dt;
        print_row(t, &s, v_d, v_q);
    }
    return 0;
}

static int feedforward_run(dq_scalar dt, dq_scalar duration, dq_scalar step_at,
                           dq_scalar target)
{
    const dq_model *m = &dq_model_default;
    dq_feedforward ff;
    dq_state s, next;
    dq_scalar t = (dq_scalar)0.0;
    dq_scalar reference = (dq_scalar)0.0;
    dq_scalar first_sat = (dq_scalar)-1.0;
    long n = 0;

    /* speed_limit = the target, so the ramp is limited by acceleration alone and
     * the current limit is what does the limiting: that is the corner this run
     * exists to show. */
    if (dq_ff_init(&ff, m, dt, target, DQ_NAN, (dq_scalar)0.0) != 0) {
        fprintf(stderr, "dq_ff_init rejected the sample time\n");
        return 1;
    }
    s.rot_vel = (dq_scalar)0.0;
    s.i_d = (dq_scalar)0.0;
    s.i_q = (dq_scalar)0.0;
    while (t < duration - (dq_scalar)0.5 * dt) {
        dq_pair v;
        dq_current c;
        if (++n > ${n_steps}) {
            break;
        }
        if (t >= step_at) {
            reference = target;
        }
        dq_ff_update(&ff, reference, &v, &c);
        if ((c.current_saturated || c.voltage_saturated) && first_sat < (dq_scalar)0.0) {
            first_sat = t;
        }
        if (dq_integrate(m, s, v.d, v.q, dt, (dq_scalar)0.0, &next) != 0) {
            fprintf(stderr, "dq_integrate refused this step\n");
            return 1;
        }
        s = next;
        t += dt;
        print_row(t, &s, v.d, v.q);
    }
    if (first_sat >= (dq_scalar)0.0) {
        fprintf(stderr,
                "feed-forward hit its current/voltage limit at t = %.4f s: above "
                "the field-weakening band edge (%.1f rad/s motor side) this "
                "machine can not follow a reference ramped at its stall "
                "acceleration\n",
                (double) first_sat, (double) dq_constant_torque_speed(m));
    }
    return 0;
}

int main(int argc, char **argv)
{
    const dq_model *m = &dq_model_default;
    const dq_scalar dt = ${bench_dt};
    const dq_scalar duration = ${bench_duration};
    const dq_scalar step_at = ${bench_step_at};
    const dq_scalar voltage = ${bench_voltage};
    const dq_scalar target = ${bench_target};
    const int ff_mode = (argc > 1 && strcmp(argv[1], "feedforward") == 0);

    printf("time_s,motor_rad_s,current_A,voltage_V\n");
    fprintf(stderr,
            "control period %g s -> %d RK4 sub-steps at standstill (stable step "
            "%g s), %d at base speed (stable step %g s)\n",
            (double) dt, dq_sub_steps(m, dt, (dq_scalar)0.0),
            (double) dq_max_step_time(m, (dq_scalar)0.0),
            dq_sub_steps(m, dt, dq_base_speed(m)),
            (double) dq_max_step_time(m, dq_base_speed(m)));
    return ff_mode ? feedforward_run(dt, duration, step_at, target)
                   : plant_run(dt, duration, step_at, voltage);
}
''')

# ----------------------------------------------------------------------
# C
# ----------------------------------------------------------------------
# The dq plant in C is a transliteration of the generated Python module, not a
# re-derivation: the current search has to run the same fixed number of
# iterations, in the same order, with the same slacks, or the two builds give a
# firmware engineer two different answers.  Identifiers are dq_/DQ_ rather than
# the brushed dt_/DT_ so a brushed export and a dq export of the same machine
# can sit in one binary without colliding.

_C_MATH = {
    # a float build must not drift into libm's double entry points
    "f32": {"sqrt": "sqrtf", "fabs": "fabsf", "hypot": "hypotf", "fmin": "fminf",
            "fmax": "fmaxf", "ceil": "ceilf", "isnan": "isnan",
            "isfinite": "isfinite"},
    "f64": {"sqrt": "sqrt", "fabs": "fabs", "hypot": "hypot", "fmin": "fmin",
            "fmax": "fmax", "ceil": "ceil", "isnan": "isnan",
            "isfinite": "isfinite"},
}

# the struct member names, which are shorter than the dataclass fields
_C_FIELD = {"flux_temperature_coefficient": "flux_temp_coefficient"}


def _dq_c_numbers(c: DqModelConstants, dtype: str) -> dict[str, str]:
    m = {}
    for f in fields(c):
        if f.name in ("notes", "pole_pairs", "motor_count"):
            continue
        m[_C_FIELD.get(f.name, f.name)] = _c_num(getattr(c, f.name), dtype)
    m["pole_pairs"] = str(int(c.pole_pairs))
    m["motor_count"] = str(int(c.motor_count))
    return m


def _dq_c_mapping(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    dtype = o.dtype
    m = {
        "banner": _banner(o, c),
        # the prefix is added by _dq_prefixed, so it is not baked in here
        "guard": f"DQ_{o.module_name.upper()}_H",
        "ctype": "float" if dtype == "f32" else "double",
        "sample_time": _c_num(o.sample_time if o.sample_time else math.inf,
                              dtype),
        "scan_steps": str(WEAKENING_SCAN_STEPS),
        "bisect_steps": str(WEAKENING_BISECT_STEPS),
        "mtpa_steps": str(MTPA_BISECT_STEPS),
        # tolerances at double precision in every language: an f32 build rounds
        # them at the comparison, which is what an f32 drive actually does
        "rk4_limit": _c_num(RK4_STABILITY_LIMIT, "f64"),
        "stiction": _c_num(STICTION_BAND, "f64"),
        "slack": _c_num(FEASIBILITY_SLACK, "f64"),
        "envelope_slack": _c_num(ENVELOPE_SLACK, "f64"),
        "salience_floor": _c_num(SALIENCE_FLOOR, "f64"),
    }
    m.update(_C_MATH[dtype])
    m.update(_dq_c_numbers(c, dtype))
    return m


def _dq_bench_c_values(o: ExportOptions, c: DqModelConstants,
                       dtype: str | None = None) -> dict[str, str]:
    """The same bench numbers as the Python bench, printed the way C wants them.

    ``bench_*`` are literals for the generated code; ``*_text`` are the %g forms
    quoted in the file's own comment, and ``n_steps`` is the iteration guard that
    keeps a broken build from spinning forever.
    """
    dtype = dtype or o.dtype
    n = _dq_bench_numbers(o, c)
    dt, duration = n["dt"], n["duration"]
    return {
        "bench_dt": _c_num(dt, dtype),
        "bench_duration": _c_num(duration, dtype),
        "bench_step_at": _c_num(n["step_at"], dtype),
        "bench_voltage": _c_num(n["voltage"], dtype),
        "bench_target": _c_num(n["target"], dtype),
        "dt_text": f"{dt:g}",
        "duration_text": f"{duration:g}",
        "step_at_text": f"{n['step_at']:g}",
        "voltage_text": f"{n['voltage']:g}",
        "target_text": f"{n['target']:g}",
        "n_steps": str(int(math.ceil((duration - 0.5 * dt) / dt))),
    }


def _dq_prefixed(text: str, prefix: str) -> str:
    """Rewrite the default ``dq_``/``DQ_`` identifiers through a custom prefix.

    Mirrors :func:`drivetrain.export._c_prefixed`: the rename is all-or-nothing,
    because a half-renamed build still collides with a second export.  The dq
    templates are written against ``dq_`` (not ``dt_``) so a brushed export and
    a dq export of the same machine can be linked together as they are.
    """
    if prefix == "dt":
        return text
    text = re.sub(r"\bdq_(?=[a-z])", f"{prefix}_dq_", text)
    return re.sub(r"\bDQ_(?=[A-Z])", f"{prefix.upper()}_DQ_", text)


def _render_dq_c(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    """The dq plant in C: header, source, and the bench when asked for."""
    m = _dq_c_mapping(o, c)
    header = f"{o.module_name}_dq.h"
    source = f"{o.module_name}_dq.c"
    files = {header: _DQ_C_HEADER.substitute(**m, header=header,
                                             file=source),
             source: _DQ_C_SOURCE.substitute(**m, header=header)}
    if o.include_examples:
        bench = f"bench_{o.module_name}_dq.c"
        files[bench] = _DQ_BENCH_C.substitute(
            **m, **_dq_bench_c_values(o, c, o.dtype), file=bench,
            header=header, source_file=source)
    if o.header_prefix != "dt":
        files = {k: _dq_prefixed(v, o.header_prefix) for k, v in files.items()}
    return files


# ----------------------------------------------------------------------
# the dq plant in Rust
#
# The same algorithm again, spelled for a `no_std` crate: no allocation, no
# formatting inside the module, and no `std` float methods, because `core` does
# not provide `sqrt`, `hypot` or `ceil` -- the module supplies its own (see
# `dq_sqrt`), which is also the single seam a firmware crate swaps for a
# hardware square root.  `step` returns a `Result` rather than silently
# integrating an unstable step: the stable step here depends on speed, so a
# caller that ignores it gets a wrong answer that looks plausible.
# ----------------------------------------------------------------------

_DQ_RUST = Template(r'''${banner}

//! dq (field-oriented-control) plant model of a brushless machine.
//!
//! This is the machine itself in the rotor frame, not a brushed equivalent of
//! it: three states (shaft speed, d-axis current, q-axis current), a torque
//! that depends on the current ANGLE and not only its magnitude, and a voltage
//! limit that becomes a rotating ellipse as speed rises.  Everything here is
//! the same arithmetic as the Python engine in this repo, transliterated
//! operation for operation, so a run of this code and a run of the Python one
//! agree to the last bit in an f64 build.
//!
//! The conventions that bite, in one place:
//!
//! * Amplitude-invariant Park.  Torque is `1.5 * pole_pairs * (flux * i_q +
//!   (Ld - Lq) * i_d * i_q)`, so `torque_constant / back_emf_constant` is
//!   exactly 1.5.  A datasheet quoted in the other convention differs by that
//!   factor, and it is the first thing to check when a fitted machine looks
//!   inconsistent.
//! * `voltage_limit` is the PHASE voltage peak the inverter can apply, not the
//!   DC link.  For a sinusoidal drive it is bus/2, for space-vector PWM
//!   bus/sqrt(3).  `dc_link_current` converts back to the battery side.
//! * `current_limit` is the phase-current magnitude, i.e. the radius of the
//!   circle in the dq plane, not a d or q component.
//!
//! The current-loop search (which current pair produces a torque, and at which
//! angle) is fixed-iteration arithmetic: a scan and a bisection with counts
//! baked into the constants.  There is no dynamic allocation, no recursion, no
//! system calls and no formatting in the model itself, so it drops into a
//! `#![no_std]` firmware crate.  The one seam is `dq_sqrt`, `dq_hypot` and
//! `dq_ceil` at the bottom of this file, which use `core`'s float methods;
//! point those three at libm and the crate is freestanding with no soft-float
//! library at all.
//!
//! # Stepping
//!
//! The stable step SHRINKS WITH SPEED here.  With the shaft speed held, the dq
//! current loop linearises to a complex pole pair at
//! `-R(Ld+Lq)/(2 Ld Lq) +/- j w_e`, and RK4's absolute-stability region only
//! reaches about 2.78 of imaginary-axis distance.  A control period that is
//! comfortable at standstill is unstable at speed: `max_step_time(rot_vel)`
//! takes that into account, `sub_steps(dt, rot_vel)` says how many pieces a
//! control period needs at a given speed, and `integrate` re-sizes every
//! sub-step from the speed it has actually reached.  `step` refuses a `dt`
//! longer than the stable step at the state it is given rather than quietly
//! subdividing it, because a drive that silently changes its own rate is worse
//! than one that returns an error.
//!
//! # Example
//!
//! ```no_run
//! include!("drive_model_dq.rs");
//!
//! let m = DQ_MODEL;
//! let mut s = DqState::default();
//! // one control period, sub-stepped as the machine needs
//! s = m.integrate(s, 0.0, 2.4, 0.001, 0.0).unwrap();
//! let c = m.current_for_torque(0.2, s.rot_vel);
//! let v = m.voltage_steady(s.rot_vel, c.i_d, c.i_q);
//! ```

pub type Scalar = ${dtype};

/// Why an operation was refused.  Every variant carries the numbers that
/// explain it, so a caller can log or branch without parsing text.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DqError {
    /// `dt` is longer than the stable step at the state's speed.  The pole
    /// magnitude is what RK4 has to stay inside; the stable step is
    /// `RK4_STABILITY_LIMIT / pole_magnitude(rot_vel)` unless the electrical
    /// time constant is the tighter of the two.
    StepTooLarge {
        dt: Scalar,
        limit: Scalar,
        rot_vel: Scalar,
        pole: Scalar,
        electrical_time_constant: Scalar,
    },
    /// `max_step_time` came out zero or negative: check the inductances and
    /// the stator resistance.
    ZeroStableStep,
    /// `integrate` hit its sub-step guard: the step contract and the speed
    /// disagree, usually because a control period was passed at a speed far
    /// above the one it was designed for.
    TooManySubSteps { limit: u32 },
    /// a feed-forward sample time must be greater than zero
    BadSampleTime(Scalar),
}

impl core::fmt::Display for DqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DqError::StepTooLarge {
                dt,
                limit,
                rot_vel,
                pole,
                electrical_time_constant,
            } => write!(
                f,
                "dt {} s exceeds the stable step {} s at {} rad/s (pole magnitude \
                 {} 1/s, electrical time constant {} s); subdivide the step",
                dt, limit, rot_vel, pole, electrical_time_constant
            ),
            DqError::ZeroStableStep => {
                write!(f, "this machine has a zero stable step; check the inductances")
            }
            DqError::TooManySubSteps { limit } => write!(
                f,
                "integrate() took {} sub-steps for one period; the step contract and \
                 the speed are inconsistent",
                limit
            ),
            DqError::BadSampleTime(sample_time) => {
                write!(f, "sample_time must be > 0, got {}", sample_time)
            }
        }
    }
}

/// The machine, the transmission and the limits, as one flat set of constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DqModel {
    /// stator resistance per phase [ohm]
    pub resistance: Scalar,
    /// d-axis inductance [H]
    pub inductance_d: Scalar,
    /// q-axis inductance [H]
    pub inductance_q: Scalar,
    /// rotor permanent-flux linkage [Wb]
    pub flux_linkage: Scalar,
    /// electrical pole pairs
    pub pole_pairs: u32,
    /// rotor inertia referred to the motor shaft [kg m^2]
    pub rotor_inertia: Scalar,
    /// viscous bearing damping at the motor shaft [N m s]
    pub bearing_damping: Scalar,
    /// Coulomb (static) friction at the motor shaft [N m]
    pub static_friction: Scalar,
    /// peak phase-current limit [A]
    pub current_limit: Scalar,
    /// peak phase-voltage limit -- the inverter's phase output, not the DC link [V]
    pub voltage_limit: Scalar,
    /// DC link voltage [V]; `INFINITY` means the bus is not modelled
    pub dc_bus_voltage: Scalar,
    /// iron-loss coefficient [W / (rad/s electrical)^2]
    pub core_loss_coefficient: Scalar,
    /// windage coefficient [W / (rad/s)^2]
    pub windage_coefficient: Scalar,
    /// flux linkage temperature coefficient [1/K]
    pub flux_temperature_coefficient: Scalar,
    /// temperature the flux linkage was measured at [degC]
    pub reference_temperature: Scalar,
    /// motors sharing the shaft
    pub motor_count: u32,
    /// output/input speed of the gearbox
    pub gear_ratio: Scalar,
    /// gearbox efficiency, 0..1
    pub gear_efficiency: Scalar,
    /// wheel radius [m]
    pub wheel_radius: Scalar,
}

/// A dq pair: `d` is the d-axis value, `q` the q-axis value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqPair {
    pub d: Scalar,
    pub q: Scalar,
}

/// The plant's derivative vector: `di_d`, `di_q` [A/s] and `dw` [rad/s^2].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqRates {
    pub di_d: Scalar,
    pub di_q: Scalar,
    pub dw: Scalar,
}

/// The three states of this plant: shaft speed [rad/s] and the two currents [A].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqState {
    pub rot_vel: Scalar,
    pub i_d: Scalar,
    pub i_q: Scalar,
}

/// Power balance at an operating point [W].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqPower {
    pub copper_w: Scalar,
    pub iron_w: Scalar,
    pub windage_w: Scalar,
    pub bearing_w: Scalar,
    pub converted_w: Scalar,
    pub shaft_w: Scalar,
    pub input_w: Scalar,
}

/// The largest torque a speed can produce, and the current angle that produces
/// it.  Above the band edge this sits on the flux-weakening ellipse, not on the
/// current circle.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqLimits {
    pub torque: Scalar,
    pub i_d: Scalar,
    pub i_q: Scalar,
}

/// A current pair that produces a torque, plus the flags naming which
/// constraint produced the answer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DqCurrent {
    pub i_d: Scalar,
    pub i_q: Scalar,
    pub current_saturated: bool,
    pub voltage_saturated: bool,
}

/// The machine as exported: `motor_count` motor(s), one gearbox, one wheel size.
pub const DQ_MODEL: DqModel = DqModel {
${model_block}};

/// Control period this model was exported for [s]; `INFINITY` means none was
/// given, in which case the caller picks the period and `max_step_time` is the
/// contract.
pub const DQ_SAMPLE_TIME_S: Scalar = ${sample_time};

/// Stable step at standstill [s] -- the same number as `max_step_time(0.0)`,
/// printed so a firmware build can size its control loop without a call.
pub const DQ_MAX_STEP_S: Scalar = ${max_step};

/// RK4's absolute-stability distance on the imaginary axis, halved for margin.
/// The dq current loop's pole pair sits there once speed dominates.
pub const RK4_STABILITY_LIMIT: Scalar = ${rk4_limit};

/// Below this speed the stiction branch of `required_torque` applies [rad/s].
pub const STICTION_BAND: Scalar = ${stiction};

/// Relative slack used by the feasibility tests, so a pair sitting exactly on
/// the current circle or the voltage ellipse counts as feasible.
pub const FEASIBILITY_SLACK: Scalar = ${slack};

/// Relative slack used when reporting which constraint bound a torque that is
/// outside the envelope; matched to the fixed-count scan that finds it.
pub const ENVELOPE_SLACK: Scalar = ${envelope_slack};

/// Below this, saliency is treated as zero and MTPA is `i_d = 0`.  A machine
/// with `Ld == Lq` has no reluctance torque to chase, and the closed form for
/// the optimal `i_d` divides by `Lq - Ld`.
pub const SALIENCE_FLOOR: Scalar = ${salience_floor};

/// Scan steps for the field-weakening search over `i_d`.
pub const WEAKENING_SCAN_STEPS: u32 = ${scan_steps};

/// Bisection steps refining that `i_d` onto the voltage ellipse.
pub const WEAKENING_BISECT_STEPS: u32 = ${bisect_steps};

/// Bisection steps for the MTPA current magnitude.
pub const MTPA_BISECT_STEPS: u32 = ${mtpa_steps};

/// Bit pattern that halves a `Scalar`'s biased exponent, i.e. a square-root
/// seed good to a factor of ~1.42 for any normal float (0x1FC00000 at f32,
/// 0x1FF8000000000000 at f64).
pub const SQRT_SEED: ${bits_type} = ${sqrt_seed};

/// Newton steps that finish the width of `Scalar` from that seed: 5 at f32,
/// 6 at f64.  Fixed count, so the cost is known before it ships.
pub const SQRT_ITERATIONS: u32 = ${sqrt_iterations};

impl DqModel {
    /// Flux linkage at a winding temperature [Wb].  Pass `None` for the
    /// reference temperature, which is what every internal call does.
    pub fn effective_flux(&self, temperature_c: Option<Scalar>) -> Scalar {
        match temperature_c {
            None => self.flux_linkage,
            Some(t) => self.flux_linkage
                * (1.0 + self.flux_temperature_coefficient * (t - self.reference_temperature)),
        }
    }

    /// Electrical (pole-frame) speed for a mechanical shaft speed [rad/s].
    pub fn electrical_speed(&self, rot_vel: Scalar) -> Scalar {
        self.pole_pairs as Scalar * rot_vel
    }

    /// `Ld - Lq` [H]: the term that multiplies `i_d * i_q` in the torque.
    pub fn reluctance(&self) -> Scalar {
        self.inductance_d - self.inductance_q
    }

    /// `Lq - Ld` [H]: the MTPA closed form divides by THIS sign, not by the
    /// reluctance above.  Confusing the two inverts the optimal `i_d`.
    pub fn mtpa_dl(&self) -> Scalar {
        self.inductance_q - self.inductance_d
    }

    /// `lambda / Ld` [A]: the demagnetising `i_d` that cancels the magnet.
    pub fn characteristic_current(&self) -> Scalar {
        self.flux_linkage / self.inductance_d
    }

    /// The most negative `i_d` this model will use [A]: the current circle or
    /// the characteristic current, whichever binds first.  No demagnetisation
    /// is modelled, which is why the floor exists at all.
    pub fn id_floor(&self) -> Scalar {
        -(self.current_limit.min(self.characteristic_current()))
    }

    /// Torque constant `1.5 p lambda` [N m/A].
    pub fn torque_constant(&self) -> Scalar {
        1.5 * self.pole_pairs as Scalar * self.flux_linkage
    }

    /// Back-EMF constant `p lambda` [V / (rad/s mechanical)].  The ratio to the
    /// torque constant is exactly 1.5, and that is the amplitude-invariant Park
    /// convention, not a bug.
    pub fn back_emf_constant(&self) -> Scalar {
        self.pole_pairs as Scalar * self.flux_linkage
    }

    /// No-load (back-EMF) speed for the phase limit [rad/s motor side].
    pub fn base_speed(&self) -> Scalar {
        self.voltage_limit / self.back_emf_constant()
    }

    /// Electrical time constant at the smaller inductance [s].
    pub fn electrical_time_constant(&self) -> Scalar {
        self.inductance_d.min(self.inductance_q) / self.resistance
    }

    /// Real part of the linearised dq pole pair [1/s].
    pub fn pole_damping(&self) -> Scalar {
        self.resistance * (self.inductance_d + self.inductance_q)
            / (2.0 * self.inductance_d * self.inductance_q)
    }

    /// Distance of the linearised dq pole from the origin [1/s].  The real part
    /// is fixed by resistance and inductances; the imaginary part grows with
    /// speed, which is why the stable step shrinks as the machine spins up.
    pub fn pole_magnitude(&self, rot_vel: Scalar) -> Scalar {
        dq_hypot(self.pole_damping(), self.electrical_speed(rot_vel))
    }

    /// Largest RK4 step that is stable at this speed [s].
    pub fn max_step_time(&self, rot_vel: Scalar) -> Scalar {
        (0.5 * self.electrical_time_constant()).min(RK4_STABILITY_LIMIT / self.pole_magnitude(rot_vel))
    }

    /// Developed (electromagnetic) torque [N m].  `flux` is a flux linkage in
    /// Wb -- pass `Some(self.effective_flux(85.0))` for a hot machine, `None`
    /// for the reference value.  It is NOT a temperature.
    pub fn torque(&self, i_d: Scalar, i_q: Scalar, flux: Option<Scalar>) -> Scalar {
        let lam = match flux {
            None => self.flux_linkage,
            Some(lam) => lam,
        };
        1.5 * self.pole_pairs as Scalar
            * (lam * i_q + self.reluctance() * i_d * i_q)
    }

    /// `dT/di_q` at an operating point [N m/A] -- the incremental torque gain,
    /// which is not the torque constant once `i_d` is non-zero.
    pub fn torque_gain(&self, i_d: Scalar) -> Scalar {
        1.5 * self.pole_pairs as Scalar * (self.flux_linkage + self.reluctance() * i_d)
    }

    /// Steady-state phase voltages for a current pair at a speed [V].
    pub fn voltage_steady(&self, rot_vel: Scalar, i_d: Scalar, i_q: Scalar) -> DqPair {
        self.voltage(rot_vel, i_d, i_q, 0.0, 0.0)
    }

    /// Phase voltages for a current pair, its rate of change and a speed [V].
    ///
    /// The cross-coupling terms are what make a dq current loop hard to tune:
    /// at speed a step on `q` shows up on `d` and back.
    pub fn voltage(
        &self,
        rot_vel: Scalar,
        i_d: Scalar,
        i_q: Scalar,
        di_d_dt: Scalar,
        di_q_dt: Scalar,
    ) -> DqPair {
        let we = self.electrical_speed(rot_vel);
        DqPair {
            d: self.resistance * i_d + self.inductance_d * di_d_dt - we * self.inductance_q * i_q,
            q: self.resistance * i_q + self.inductance_q * di_q_dt
                + we * (self.inductance_d * i_d + self.flux_linkage),
        }
    }

    /// Voltage magnitude at an operating point [V] -- the radius the inverter
    /// limit applies to.
    pub fn voltage_magnitude(
        &self,
        rot_vel: Scalar,
        i_d: Scalar,
        i_q: Scalar,
        di_d_dt: Scalar,
        di_q_dt: Scalar,
    ) -> Scalar {
        let v = self.voltage(rot_vel, i_d, i_q, di_d_dt, di_q_dt);
        dq_hypot(v.d, v.q)
    }

    /// Largest `i_q` that still fits the voltage budget at this `i_d` and
    /// speed [A], or `None` when no `i_q` fits at all -- the back-EMF alone
    /// exceeds the rail.
    ///
    /// `|v|^2 <= V^2` is quadratic in `i_q`: `a*i_q^2 + b*i_q + c <= 0` with
    /// `a = R^2 + (w_e*Lq)^2`, `b = 2 R w_e (lambda + (Ld - Lq) i_d)`,
    /// `c = (R i_d)^2 + (w_e (Ld i_d + lambda))^2 - V^2`.
    fn iq_from_voltage(
        &self,
        i_d: Scalar,
        rot_vel: Scalar,
        v_limit: Option<Scalar>,
    ) -> Option<Scalar> {
        let v = match v_limit {
            None => self.voltage_limit,
            Some(v) => v,
        };
        if !v.is_finite() {
            return Some(Scalar::INFINITY);
        }
        let we = self.electrical_speed(rot_vel);
        let a = self.resistance * self.resistance + { let t = we * self.inductance_q; t * t };
        let b = 2.0 * self.resistance * we * (self.flux_linkage + self.reluctance() * i_d);
        let c = {
            let t1 = self.resistance * i_d;
            let t2 = we * (self.inductance_d * i_d + self.flux_linkage);
            t1 * t1 + t2 * t2 - v * v
        };
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        Some((-b + dq_sqrt(disc)) / (2.0 * a))
    }

    /// Is a current pair inside the current circle?
    pub fn fits_current(&self, i_d: Scalar, i_q: Scalar) -> bool {
        if !self.current_limit.is_finite() {
            return true;
        }
        i_d * i_d + i_q * i_q
            <= self.current_limit * self.current_limit * (1.0 + FEASIBILITY_SLACK)
    }

    /// Is a current pair inside BOTH the current circle and the voltage
    /// ellipse at this speed?
    pub fn feasible(&self, i_d: Scalar, i_q: Scalar, rot_vel: Scalar) -> bool {
        self.fits_current(i_d, i_q)
            && self.voltage_magnitude(rot_vel, i_d, i_q, 0.0, 0.0)
                <= self.voltage_limit * (1.0 + FEASIBILITY_SLACK)
    }

    /// Minimum-current `(i_d, i_q)` for a torque, ignoring the voltage limit.
    ///
    /// Closed form for a salient machine: with `dl = Lq - Ld` and `i_m` the
    /// current magnitude, `i_d = (lambda - sqrt(lambda^2 + 8 dl^2 i_m^2)) /
    /// (4 dl)`.  The sign of `i_q` follows the sign of the torque; a negative
    /// `i_d` is the same angle for either direction of rotation.  Returns
    /// `(0.0, INFINITY)` when no current within `current_limit` can produce it.
    pub fn mtpa_pair(&self, tau: Scalar) -> DqPair {
        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let tau_mag = dq_abs(tau);
        if dq_abs(self.reluctance()) <= SALIENCE_FLOOR {
            if !self.current_limit.is_finite() {
                return DqPair { d: 0.0, q: sign * (tau_mag / self.torque_constant()) };
            }
            if self.torque_constant() * self.current_limit < tau_mag {
                return DqPair { d: 0.0, q: Scalar::INFINITY };
            }
            return DqPair { d: 0.0, q: sign * (tau_mag / self.torque_constant()) };
        }
        // torque_at(i_m): the best this circle radius can do
        let dl = self.mtpa_dl();
        let torque_at = |i_m: Scalar| -> Scalar {
            let i_d = (self.flux_linkage
                - dq_sqrt(self.flux_linkage * self.flux_linkage
                    + 8.0 * dl * dl * i_m * i_m))
                / (4.0 * dl);
            let i_q_sq = i_m * i_m - i_d * i_d;
            if i_q_sq <= 0.0 {
                return 0.0;
            }
            self.torque(i_d, dq_sqrt(i_q_sq), None)
        };
        let ceiling = if self.current_limit.is_finite() {
            self.current_limit
        } else {
            Scalar::INFINITY
        };
        // the same relative slack the feasibility tests use: without it, a
        // torque exactly ON the current circle looks unreachable by a hair
        if ceiling.is_finite() && torque_at(ceiling) < tau_mag * (1.0 - FEASIBILITY_SLACK) {
            return DqPair { d: 0.0, q: Scalar::INFINITY };
        }
        let mut hi = if ceiling.is_finite() {
            ceiling
        } else {
            tau_mag / self.torque_constant() * 4.0
        };
        let mut lo = 0.0;
        let mut k = 0u32;
        while k < MTPA_BISECT_STEPS {
            let mid = 0.5 * (lo + hi);
            if torque_at(mid) < tau_mag {
                lo = mid;
            } else {
                hi = mid;
            }
            k += 1;
        }
        let i_m = 0.5 * (lo + hi);
        let i_d = (self.flux_linkage
            - dq_sqrt(self.flux_linkage * self.flux_linkage + 8.0 * dl * dl * i_m * i_m))
            / (4.0 * dl);
        let i_q_sq = i_m * i_m - i_d * i_d;
        let i_q = if i_q_sq > 0.0 { dq_sqrt(i_q_sq) } else { 0.0 };
        DqPair { d: i_d, q: sign * i_q }
    }

    /// `(torque, i_d, i_q)` at the largest POSITIVE torque this speed allows.
    ///
    /// The binding constraint is the current circle at low speed and the
    /// voltage ellipse above base speed.  The search is a fixed-count scan plus
    /// a finer scan around its best point, so its cost is known before it
    /// ships; it is exact to within the scan resolution.
    pub fn max_torque(&self, rot_vel: Scalar) -> DqLimits {
        let mut best_t = 0.0;
        let mut best = DqPair { d: 0.0, q: 0.0 };
        let floor = self.id_floor();
        let steps = WEAKENING_SCAN_STEPS;
        let step = if steps != 0 {
            (0.0 - floor) / steps as Scalar
        } else {
            0.0
        };
        let mut k = 0u32;
        while k <= steps {
            let i_d = -(k as Scalar) * step;
            let i_q_c = if self.current_limit.is_finite() {
                dq_sqrt((self.current_limit * self.current_limit - i_d * i_d).max(0.0))
            } else {
                Scalar::INFINITY
            };
            let i_q_v = match self.iq_from_voltage(i_d, rot_vel, None) {
                Some(v) => v,
                None => {
                    k += 1;
                    continue;
                }
            };
            let i_q = i_q_c.min(i_q_v);
            let t = self.torque(i_d, i_q, None);
            if t > best_t {
                best_t = t;
                best = DqPair { d: i_d, q: i_q };
            }
            k += 1;
        }
        // refine around the best i_d with a finer scan: bounded, and it cannot
        // wander the way a bisection on a non-monotone objective does
        if step > 0.0 {
            let centre = best.d;
            let fine = step / 16.0;
            let mut j = -16i32;
            while j <= 16 {
                let i_d = centre + j as Scalar * fine;
                if i_d < floor || i_d > 0.0 {
                    j += 1;
                    continue;
                }
                let i_q_c = if self.current_limit.is_finite() {
                    dq_sqrt((self.current_limit * self.current_limit - i_d * i_d).max(0.0))
                } else {
                    Scalar::INFINITY
                };
                let i_q_v = match self.iq_from_voltage(i_d, rot_vel, None) {
                    Some(v) => v,
                    None => {
                        j += 1;
                        continue;
                    }
                };
                let i_q = i_q_c.min(i_q_v);
                let t = self.torque(i_d, i_q, None);
                if t > best_t {
                    best_t = t;
                    best = DqPair { d: i_d, q: i_q };
                }
                j += 1;
            }
        }
        DqLimits { torque: best_t, i_d: best.d, i_q: best.q }
    }

    /// `(i_d, i_q, current_saturated, voltage_saturated)` for a torque [N m].
    ///
    /// Minimum current at that torque: MTPA when the voltage ellipse allows it,
    /// otherwise the field-weakening angle.  Work is done in the frame where the
    /// commanded torque is positive and the result is reflected back, because
    /// flipping BOTH the signs of `(i_q, w_e)` leaves `|v|` unchanged while
    /// flipping one of them does not -- braking is not the same problem as
    /// driving.  Only `i_q` carries the torque's sign on the way back: `i_d`
    /// stays negative either way, because the reluctance term follows the sign
    /// of `i_q`, not of the torque.
    ///
    /// A zero torque demand is not a special case: above the rail speed the
    /// back-EMF alone exceeds the phase limit, so holding zero current is not
    /// an option either.  The weakening branch then answers with the
    /// demagnetising d-axis current that cancels enough flux to fit inside the
    /// rail, which is what a real drive does there.  When even that is
    /// impossible -- a machine rated below its characteristic current `lambda/
    /// Ld` cannot weaken far enough -- no current fits at all, and the answer
    /// is the max-torque angle with the flags naming what bounded the search.
    /// That is the uncontrolled-regeneration corner, and it is a property of
    /// the rating, not of the search.
    ///
    /// The flags say which constraint PRODUCED the answer, not which one the
    /// answer happens to violate by rounding: a pair from the field-weakening
    /// branch always reports `voltage_saturated` true, and a torque sitting
    /// exactly on the current circle may report `current_saturated` true or
    /// false depending on where the fixed-count bisection landed.  The current
    /// magnitude is the same either way.
    pub fn current_for_torque(&self, tau: Scalar, rot_vel: Scalar) -> DqCurrent {
        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let w = if sign > 0.0 { rot_vel } else { -rot_vel };
        let tau_mag = dq_abs(tau);

        let mtpa = self.mtpa_pair(tau_mag);
        if mtpa.q.is_finite() && self.feasible(mtpa.d, mtpa.q, w) {
            return DqCurrent {
                i_d: mtpa.d,
                i_q: sign * mtpa.q,
                current_saturated: false,
                voltage_saturated: false,
            };
        }

        // field weakening: for i_d < 0 the current magnitude falls
        // monotonically as i_d rises toward zero, so the answer is the LARGEST
        // feasible i_d
        let floor = self.id_floor();
        let steps = WEAKENING_SCAN_STEPS;
        let step = if steps != 0 {
            (0.0 - floor) / steps as Scalar
        } else {
            0.0
        };
        let mut found: Option<Scalar> = None;
        let mut k = 0u32;
        while k <= steps {
            let i_d_try = -(k as Scalar) * step;
            let gain = self.torque_gain(i_d_try);
            if gain <= 0.0 {
                k += 1;
                continue;
            }
            if self.feasible(i_d_try, tau_mag / gain, w) {
                found = Some(i_d_try);
                break;
            }
            k += 1;
        }
        let mut found = match found {
            None => {
                // outside the envelope at this speed: hand back the best this
                // speed can do and name the constraint that bounded it, at the
                // tolerance of max_torque's fixed-count scan (a few 1e-4)
                // rather than machine precision -- at the corner where the
                // circle and the ellipse cross the scan lands just inside
                let lim = self.max_torque(w);
                let on_current = self.current_limit.is_finite()
                    && dq_hypot(lim.i_d, lim.i_q)
                        >= self.current_limit * (1.0 - ENVELOPE_SLACK);
                let on_voltage = self.voltage_magnitude(w, lim.i_d, lim.i_q, 0.0, 0.0)
                    >= self.voltage_limit * (1.0 - ENVELOPE_SLACK);
                return DqCurrent {
                    i_d: lim.i_d,
                    i_q: sign * lim.i_q,
                    current_saturated: on_current,
                    voltage_saturated: on_voltage,
                };
            }
            Some(i_d) => i_d,
        };
        let mut bad = found + step;
        let mut j = 0u32;
        while j < WEAKENING_BISECT_STEPS {
            let mid = 0.5 * (found + bad);
            let gain = self.torque_gain(mid);
            if gain > 0.0 && self.feasible(mid, tau_mag / gain, w) {
                found = mid;
            } else {
                bad = mid;
            }
            j += 1;
        }
        let gain = self.torque_gain(found);
        let i_q = if gain > 0.0 { tau_mag / gain } else { 0.0 };
        let over_current = !self.fits_current(found, i_q);
        // this branch exists only because the MTPA angle did not fit, and it
        // bisects ONTO the voltage ellipse, so the drive is on the rail by
        // construction: comparing the magnitude with a strict > would report
        // false for a point that sits one rounding inside the rail
        DqCurrent {
            i_d: found,
            i_q: sign * i_q,
            current_saturated: over_current,
            voltage_saturated: true,
        }
    }

    /// Torque needed to overcome this model's mechanical and iron losses
    /// [N m], as a positive magnitude.  Iron and windage loss are specified as
    /// POWERS, so dividing by speed makes the iron term look linear in
    /// mechanical speed even though it is quadratic in electrical speed.
    /// `static_friction` is not here: it is a Coulomb term handled by
    /// `required_torque`.
    pub fn loss_torque(&self, rot_vel: Scalar) -> Scalar {
        let w = dq_abs(rot_vel);
        if w <= 0.0 {
            return 0.0;
        }
        let we = self.electrical_speed(w);
        self.bearing_damping * w + self.windage_coefficient * w * w
            + self.core_loss_coefficient * we * we / w
    }

    /// Developed torque needed to follow `accel` at `rot_vel` against friction
    /// and load [N m].
    ///
    /// `load_torque` is a non-negative magnitude that opposes motion, the
    /// convention used everywhere in this repo.  A torque that actively assists
    /// the motion -- a grade, a spring -- has to be added to the result as a
    /// signed torque, not passed as a negative magnitude, or it gets counted
    /// twice.  A negative `load_torque` returns NaN here rather than guessing
    /// what was meant; in Python it raises.
    pub fn required_torque(&self, rot_vel: Scalar, accel: Scalar, load_torque: Scalar) -> Scalar {
        if load_torque < 0.0 {
            return Scalar::NAN;
        }
        let w = rot_vel;
        let mut tau = self.rotor_inertia * accel + self.bearing_damping * w;
        if w > STICTION_BAND {
            tau += self.static_friction + load_torque;
        } else if w < -STICTION_BAND {
            tau -= self.static_friction + load_torque;
        } else {
            tau += self.static_friction + load_torque;
        }
        tau
    }

    /// Power balance at an operating point [W].
    ///
    /// `input_w = copper_w + converted_w` exactly -- that is the whole of the
    /// electrical input in this model.  iron, windage and bearing are drags
    /// acting on the shaft, so they sit INSIDE the converted power:
    /// `shaft_w = converted_w - iron_w - windage_w - bearing_w`.  Adding them
    /// to the input as well would count them twice.  Nothing here is
    /// multiplied by `motor_count`: these are per machine, like the constants.
    pub fn losses(&self, rot_vel: Scalar, i_d: Scalar, i_q: Scalar) -> DqPower {
        let we = self.electrical_speed(dq_abs(rot_vel));
        let copper = 1.5 * self.resistance * (i_d * i_d + i_q * i_q);
        let iron = self.core_loss_coefficient * we * we;
        let w = dq_abs(rot_vel);
        let wind = self.windage_coefficient * w * w * w;
        let bearing = self.bearing_damping * rot_vel * rot_vel;
        let converted = self.torque(i_d, i_q, None) * rot_vel; // air-gap power
        DqPower {
            copper_w: copper,
            iron_w: iron,
            windage_w: wind,
            bearing_w: bearing,
            converted_w: converted,
            shaft_w: converted - iron - wind - bearing,
            input_w: copper + converted,
        }
    }

    /// Current drawn from the DC link [A] -- what a battery ammeter reads.
    ///
    /// Not the winding current: the 1.5 is the three-phase sum under the
    /// amplitude-invariant convention, and the whole point of this function is
    /// that a drive fitted from rail current alone is fitting a different
    /// quantity.  This is also the one place `motor_count` multiplies: a
    /// battery sees every machine on the shaft.  NaN when the bus is not
    /// modelled, because a rail current with no rail is not a number.
    pub fn dc_link_current(&self, rot_vel: Scalar, i_d: Scalar, i_q: Scalar) -> Scalar {
        if !self.dc_bus_voltage.is_finite() {
            return Scalar::NAN;
        }
        let v = self.voltage_steady(rot_vel, i_d, i_q);
        self.motor_count as Scalar * 1.5 * (v.d * i_d + v.q * i_q) / self.dc_bus_voltage
    }

    /// `(di_d/dt, di_q/dt, dw/dt)` at this state under applied phase voltages.
    pub fn derivatives(
        &self,
        state: DqState,
        v_d: Scalar,
        v_q: Scalar,
        load_torque: Scalar,
    ) -> DqRates {
        let we = self.electrical_speed(state.rot_vel);
        let di_d = (v_d - self.resistance * state.i_d + we * self.inductance_q * state.i_q)
            / self.inductance_d;
        let di_q = (v_q - self.resistance * state.i_q
            - we * (self.inductance_d * state.i_d + self.flux_linkage))
            / self.inductance_q;
        let tau_load = load_torque + self.loss_torque(state.rot_vel);
        let dw = (self.torque(state.i_d, state.i_q, None) - tau_load) / self.rotor_inertia;
        DqRates { di_d, di_q, dw }
    }

    /// Advance the plant one fixed step with classic RK4.
    ///
    /// `dt` must be `<= max_step_time(state.rot_vel)`, which tightens as the
    /// machine speeds up.  A control period is normally LONGER than that even
    /// at standstill, so a 1 ms loop cannot call this directly at `dt = 1e-3`:
    /// use `integrate`, which sub-divides.  Refusing here is deliberate -- a
    /// silently unstable step is the kind of error that only shows up as a
    /// simulation that diverges in the field.
    pub fn step(
        &self,
        state: DqState,
        v_d: Scalar,
        v_q: Scalar,
        dt: Scalar,
        load_torque: Scalar,
    ) -> Result<DqState, DqError> {
        let limit = self.max_step_time(state.rot_vel);
        if dt > limit {
            return Err(DqError::StepTooLarge {
                dt,
                limit,
                rot_vel: state.rot_vel,
                pole: self.pole_magnitude(state.rot_vel),
                electrical_time_constant: self.electrical_time_constant(),
            });
        }
        let a1 = self.derivatives(state, v_d, v_q, load_torque);
        let s2 = DqState {
            rot_vel: state.rot_vel + 0.5 * dt * a1.dw,
            i_d: state.i_d + 0.5 * dt * a1.di_d,
            i_q: state.i_q + 0.5 * dt * a1.di_q,
        };
        let a2 = self.derivatives(s2, v_d, v_q, load_torque);
        let s3 = DqState {
            rot_vel: state.rot_vel + 0.5 * dt * a2.dw,
            i_d: state.i_d + 0.5 * dt * a2.di_d,
            i_q: state.i_q + 0.5 * dt * a2.di_q,
        };
        let a3 = self.derivatives(s3, v_d, v_q, load_torque);
        let s4 = DqState {
            rot_vel: state.rot_vel + dt * a3.dw,
            i_d: state.i_d + dt * a3.di_d,
            i_q: state.i_q + dt * a3.di_q,
        };
        let a4 = self.derivatives(s4, v_d, v_q, load_torque);
        Ok(DqState {
            rot_vel: state.rot_vel
                + dt / 6.0 * (a1.dw + 2.0 * a2.dw + 2.0 * a3.dw + a4.dw),
            i_d: state.i_d
                + dt / 6.0 * (a1.di_d + 2.0 * a2.di_d + 2.0 * a3.di_d + a4.di_d),
            i_q: state.i_q
                + dt / 6.0 * (a1.di_q + 2.0 * a2.di_q + 2.0 * a3.di_q + a4.di_q),
        })
    }

    /// RK4 sub-steps a control period of `dt` needs at `rot_vel` (>= 1).
    pub fn sub_steps(&self, dt: Scalar, rot_vel: Scalar) -> u32 {
        let limit = self.max_step_time(rot_vel);
        if limit <= 0.0 || dt <= limit {
            return 1;
        }
        let n = dq_ceil(dt / limit);
        if n >= 4294967295.0 {
            u32::MAX
        } else {
            n as u32
        }
    }

    /// Advance one control period, sub-divided until every RK4 step is stable.
    ///
    /// The voltage is held constant across the period, which is what a sampled
    /// current loop does between updates.  The sub-step size is recomputed from
    /// the speed reached so far, because the stable step shrinks as the machine
    /// speeds up: a fixed subdivision sized for standstill is exactly the bug
    /// that makes a high-speed run diverge.  Cost is the number of RK4 steps,
    /// which for a 1 ms period on this machine is 2 at standstill and 9 at
    /// base speed.
    pub fn integrate(
        &self,
        state: DqState,
        v_d: Scalar,
        v_q: Scalar,
        dt: Scalar,
        load_torque: Scalar,
    ) -> Result<DqState, DqError> {
        let mut state = state;
        let mut remaining = dt;
        let mut guard = 0u32;
        while remaining > 0.0 {
            let h = remaining.min(self.max_step_time(state.rot_vel));
            if h <= 0.0 {
                return Err(DqError::ZeroStableStep);
            }
            state = self.step(state, v_d, v_q, h, load_torque)?;
            remaining -= h;
            guard += 1;
            if guard > 100000 {
                return Err(DqError::TooManySubSteps { limit: 100000 });
            }
        }
        Ok(state)
    }

    /// Advance one step assuming the current loop already holds the command.
    ///
    /// The usual case when the current loop is much faster than the mechanical
    /// dynamics: currents jump to the command, and only the shaft integrates.
    /// The torque is held constant across the step -- RK4 on the shaft alone,
    /// with the currents pinned -- so keep `dt` small against the mechanical
    /// time constant.
    pub fn step_current(
        &self,
        state: DqState,
        i_d_cmd: Scalar,
        i_q_cmd: Scalar,
        dt: Scalar,
        load_torque: Scalar,
    ) -> DqState {
        let tau = self.torque(i_d_cmd, i_q_cmd, None);
        let w = state.rot_vel;
        let k1 = (tau - load_torque - self.loss_torque(w)) / self.rotor_inertia;
        let k2 =
            (tau - load_torque - self.loss_torque(w + 0.5 * dt * k1)) / self.rotor_inertia;
        let k3 =
            (tau - load_torque - self.loss_torque(w + 0.5 * dt * k2)) / self.rotor_inertia;
        let k4 = (tau - load_torque - self.loss_torque(w + dt * k3)) / self.rotor_inertia;
        DqState {
            rot_vel: w + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            i_d: i_d_cmd,
            i_q: i_q_cmd,
        }
    }

    /// Output-shaft speed [rad/s] for a motor speed [rad/s].
    pub fn output_speed(&self, motor_rad_s: Scalar) -> Scalar {
        motor_rad_s * self.gear_ratio
    }

    /// Torque delivered to the output shaft [N m], after the gearbox losses.
    ///
    /// `motor_rad_s` is unused: this gearbox has a constant ratio and a
    /// constant efficiency, so there is no speed-dependent loss to ask about.
    /// The argument is kept so the call reads like `output_speed`.
    pub fn output_torque(&self, motor_rad_s: Scalar, i_d: Scalar, i_q: Scalar) -> Scalar {
        let _ = motor_rad_s;
        self.torque(i_d, i_q, None) * self.gear_ratio * self.gear_efficiency
    }

    /// Ground speed [m/s]; 0.0 when this system has no wheel defined.
    pub fn vehicle_speed(&self, motor_rad_s: Scalar) -> Scalar {
        if self.wheel_radius <= 0.0 {
            return 0.0;
        }
        self.output_speed(motor_rad_s) * self.wheel_radius
    }

    /// Torque at standstill with the current limit honoured [N m]: the default
    /// acceleration limit for a feed-forward ramp is this over `rotor_inertia`.
    pub fn stall_torque(&self) -> Scalar {
        let mut i = self.voltage_limit / self.resistance;
        if self.current_limit.is_finite() {
            i = i.min(self.current_limit);
        }
        self.torque_constant() * i
    }

    /// Motor speed up to which full `current_limit` still fits the voltage rail
    /// [rad/s].
    ///
    /// `|v|^2 = V^2` at `i_d = 0, i_q = I` is quadratic in `w_e`:
    /// `a*w_e^2 + b*w_e + c = 0` with `a = (Lq*I)^2 + lambda^2`,
    /// `b = 2*lambda*R*I`, `c = (R*I)^2 - V^2`.  This sits below base speed
    /// whenever `Lq*I` is comparable to `lambda`, because the drive has to spend
    /// part of its voltage budget on the cross-coupling term `-w_e*Lq*i_q`.
    pub fn constant_torque_speed(&self) -> Scalar {
        if !self.current_limit.is_finite() {
            return self.base_speed();
        }
        let i = self.current_limit;
        let a = (self.inductance_q * i) * (self.inductance_q * i) + self.flux_linkage * self.flux_linkage;
        let b = 2.0 * self.flux_linkage * self.resistance * i;
        let c = (self.resistance * i) * (self.resistance * i) - self.voltage_limit * self.voltage_limit;
        let disc = b * b - 4.0 * a * c;
        if disc <= 0.0 {
            return 0.0;
        }
        (dq_sqrt(disc) - b) / (2.0 * a) / self.pole_pairs as Scalar
    }
}
// --- free helpers -------------------------------------------------------

// The float math methods are inherent to `std`, not to `core`, so a `no_std`
// crate has nothing to call here.  These four are the whole of what the module
// needs beyond `core`.

pub fn dq_abs(x: Scalar) -> Scalar {
    x.abs()
}

/// Square root by exponent-halved seed plus Newton.
///
/// `sqrt` is one of the float methods `core` does not provide, so the module
/// supplies its own.  Halving the biased exponent puts the seed within ~3% of
/// the answer for any NORMAL float, and every Newton step doubles the correct
/// digits: measured against `std`'s `sqrt` over 20 000 samples spanning 24
/// decades, four iterations already land within 1 ulp at f64, and six are run
/// for margin.  Newton is not correctly rounded, so roughly a quarter of the
/// results differ from libm in the last bit; that is why this module matches
/// the C and Python models to ~1e-14 relative rather than bit for bit.
/// Subnormal inputs would need more iterations than are run, and the arguments
/// here are discriminants, current squares and hypot parts -- magnitudes the
/// model itself produces, never subnormal.  A firmware crate with a hardware
/// square root replaces this one body and nothing else.
///
/// The iteration runs at f64 width in every build.  An f32 build widens, takes
/// the square root where the iteration has room to converge, and rounds back
/// down: one seed constant and one bit path instead of two, and a result that
/// is correctly rounded at f32 for the price.
pub fn dq_sqrt(x: Scalar) -> Scalar {
    let xd = x as f64;
    if xd <= 0.0 {
        return (if xd < 0.0 { f64::NAN } else { xd }) as Scalar;
    }
    if !xd.is_finite() {
        return xd as Scalar;
    }
    let bits = xd.to_bits();
    let mut y = f64::from_bits((bits >> 1) + SQRT_SEED);
    let mut k = 0u32;
    while k < SQRT_ITERATIONS {
        y = 0.5 * (y + xd / y);
        k += 1;
    }
    y as Scalar
}

/// Hypotenuse, scaled so the square cannot overflow or underflow.  Matches
/// `math.hypot` on the corners that matter: an infinite argument gives
/// infinity even when the other is NaN, and NaN propagates otherwise.
pub fn dq_hypot(a: Scalar, b: Scalar) -> Scalar {
    let a = dq_abs(a);
    let b = dq_abs(b);
    let hi = a.max(b);
    let lo = a.min(b);
    if !hi.is_finite() {
        return hi;
    }
    if hi == 0.0 {
        return 0.0;
    }
    let r = lo / hi;
    hi * dq_sqrt(1.0 + r * r)
}

/// Smallest integer not less than `x`, as a float.
pub fn dq_ceil(x: Scalar) -> Scalar {
    let n = x as i64; // `as` saturates in Rust; it does not wrap
    let f = n as Scalar;
    if (f as i64) != n {
        x // outside the integer range at this width: already a whole number
    } else if f < x {
        f + 1.0
    } else {
        f
    }
}

// --- feed-forward -------------------------------------------------------

/// A clocked feed-forward law: a speed target in, a phase-voltage pair out.
///
/// The law is the whole point of this module::
///
/// ```text
/// (i_d, i_q) = current_for_torque(J*a + b*w + load + friction, w)
/// (v_d, v_q) = voltage(w, i_d, i_q)
/// ```
///
/// and the failure mode in the field is feeding it an `a` taken from the
/// derivative of an unlimited reference.  So the reference is rate limited
/// HERE, on the INCREMENT, not on the absolute value -- clamping the absolute
/// value is what pins a 129 rad/s command at 0.09 rad/s and looks like a broken
/// model.
///
/// `speed_limit` defaults to base speed and `accel_limit` to this machine's
/// stall torque divided by its rotor inertia; both are the MODEL's limits, not
/// the vehicle's.  Set them to what the machine can actually do, or the law
/// commands a trajectory the current loop cannot follow.  The model is held by
/// value (`DqModel` is `Copy`), so a feed-forward instance has no lifetime and
/// lives on the stack next to the control loop.
#[derive(Clone, Copy, Debug)]
pub struct DqFeedForward {
    pub model: DqModel,
    pub sample_time: Scalar,
    pub speed_limit: Scalar,
    pub accel_limit: Scalar,
    pub load_torque: Scalar,
    /// the rate-limited speed reference the law is following [rad/s motor side]
    pub w_ref: Scalar,
    /// its derivative, which is what the torque law is fed [rad/s^2]
    pub a_ref: Scalar,
    pub current_saturated: bool,
    pub voltage_saturated: bool,
}

impl DqFeedForward {
    /// Feed-forward at the model's own limits: base speed, stall torque over
    /// rotor inertia, no load torque.
    pub fn new(model: DqModel, sample_time: Scalar) -> Result<Self, DqError> {
        Self::with_limits(model, sample_time, None, None, None)
    }

    /// Feed-forward with the limits stated.  `None` takes the model's default
    /// for that limit.  A sample time that is not greater than zero is an
    /// error rather than a NaN reference: the ramp would be meaningless.
    pub fn with_limits(
        model: DqModel,
        sample_time: Scalar,
        speed_limit: Option<Scalar>,
        accel_limit: Option<Scalar>,
        load_torque: Option<Scalar>,
    ) -> Result<Self, DqError> {
        if !(sample_time > 0.0) {
            return Err(DqError::BadSampleTime(sample_time));
        }
        Ok(DqFeedForward {
            model,
            sample_time,
            speed_limit: speed_limit.unwrap_or_else(|| model.base_speed()),
            accel_limit: accel_limit
                .unwrap_or_else(|| model.stall_torque() / model.rotor_inertia),
            load_torque: load_torque.unwrap_or(0.0),
            w_ref: 0.0,
            a_ref: 0.0,
            current_saturated: false,
            voltage_saturated: false,
        })
    }

    /// Torque at standstill with the current limit honoured [N m].
    pub fn stall_torque(&self) -> Scalar {
        self.model.stall_torque()
    }

    /// Restart the reference at `rot_vel` with zero acceleration.  The
    /// saturation flags are left alone: they describe the last `current` or
    /// `update` call, not the reference.
    pub fn reset(&mut self, rot_vel: Scalar) {
        self.w_ref = dq_clamp(rot_vel, -self.speed_limit, self.speed_limit);
        self.a_ref = 0.0;
    }

    /// Advance the internal speed reference one sample; returns
    /// `(w_ref, a_ref)`.
    ///
    /// The clamp is on the increment (`accel_limit * sample_time`), which is
    /// what makes the resulting `a_ref` the real acceleration the machine will
    /// see.
    pub fn ramp(&mut self, target: Scalar) -> (Scalar, Scalar) {
        let limit = self.speed_limit;
        let target = dq_clamp(target, -limit, limit);
        let max_step = self.accel_limit * self.sample_time;
        let nxt = self.w_ref + dq_clamp(target - self.w_ref, -max_step, max_step);
        self.a_ref = (nxt - self.w_ref) / self.sample_time;
        self.w_ref = nxt;
        (self.w_ref, self.a_ref)
    }

    /// `(i_d, i_q)` the model needs at an operating point, with the saturation
    /// flags left on `self`.  `None` arguments fall back to the internal
    /// reference, which is the usual case.
    pub fn current(
        &mut self,
        rot_vel: Option<Scalar>,
        accel: Option<Scalar>,
        load_torque: Option<Scalar>,
    ) -> DqPair {
        let w = rot_vel.unwrap_or(self.w_ref);
        let a = accel.unwrap_or(self.a_ref);
        let load = load_torque.unwrap_or(self.load_torque);
        let tau = self.model.required_torque(w, a, load);
        let got = self.model.current_for_torque(tau, w);
        self.current_saturated = got.current_saturated;
        self.voltage_saturated = got.voltage_saturated;
        DqPair { d: got.i_d, q: got.i_q }
    }

    /// One control period: returns `(v_d, v_q, i_d, i_q)`.
    ///
    /// The voltage is scaled back onto the rail when the pair does not fit --
    /// a feed-forward law that hands the inverter a vector longer than the rail
    /// is simply asking for something it cannot do, and the scaled pair is at
    /// least the right ANGLE.  `voltage_magnitude` is evaluated twice here, on
    /// purpose: the same two calls the Python module makes, so the two agree
    /// bit for bit rather than by luck.
    pub fn update(&mut self, target: Scalar) -> (Scalar, Scalar, Scalar, Scalar) {
        let (w, a) = self.ramp(target);
        let tau = self.model.required_torque(w, a, self.load_torque);
        let got = self.model.current_for_torque(tau, w);
        self.current_saturated = got.current_saturated;
        self.voltage_saturated = got.voltage_saturated;
        let mut v = self.model.voltage(w, got.i_d, got.i_q, 0.0, 0.0);
        if self.model.voltage_magnitude(w, got.i_d, got.i_q, 0.0, 0.0)
            > self.model.voltage_limit
        {
            let scale = self.model.voltage_limit
                / self.model.voltage_magnitude(w, got.i_d, got.i_q, 0.0, 0.0);
            v = DqPair { d: v.d * scale, q: v.q * scale };
            self.voltage_saturated = true;
        }
        (v.d, v.q, got.i_d, got.i_q)
    }
}

pub fn dq_clamp(x: Scalar, lo: Scalar, hi: Scalar) -> Scalar {
    x.clamp(lo, hi)
}
''')


_DQ_BENCH_RUST = Template(r'''${banner}
// Step-response bench for the generated dq model.
//
// Build and run:
//
//   rustc -O --edition 2021 ${file} -o bench_dq
//   ./bench_dq              # open-loop plant, one voltage step
//   ./bench_dq feedforward  # the feed-forward law driving the same plant
//
// `${source_file}` has to sit next to this file: the model is a plain module,
// so there is no crate, no Cargo.toml and no build step.  This file is the
// parity check, not a library user.
//
// Generated values: dt = ${dt_text} s, duration = ${duration_text} s, step at
// t = ${step_at_text} s, applied q-axis voltage = ${voltage_text} V, feed-forward
// target = ${target_text} rad/s motor side (1.5x base speed, so the run crosses
// the field-weakening band edge).  ${n_steps} iterations is the hard guard that
// keeps a broken build from spinning forever.
//
// stdout is CSV: time_s,motor_rad_s,current_A,voltage_V.  The current column is
// the PLANT's |i| = hypot(i_d, i_q), not the command, and the voltage column is
// the magnitude actually applied.  stderr carries the sub-step commentary, so
// the CSV stays pipeable.

mod ${module_mod};

use ${module_mod}::{dq_hypot, DqFeedForward, DqState, Scalar, DQ_MODEL};

const DT: Scalar = ${bench_dt};
const DURATION: Scalar = ${bench_duration};
const STEP_AT: Scalar = ${bench_step_at};
const VOLTAGE: Scalar = ${bench_voltage};
const TARGET: Scalar = ${bench_target};
const N_STEPS: u32 = ${n_steps};

fn print_row(t: Scalar, s: &DqState, v_d: Scalar, v_q: Scalar) {
    println!(
        "{:.6},{:.6},{:.6},{:.6}",
        t,
        s.rot_vel,
        dq_hypot(s.i_d, s.i_q),
        dq_hypot(v_d, v_q)
    );
}

fn plant_run() -> i32 {
    let mut s = DqState { rot_vel: 0.0, i_d: 0.0, i_q: 0.0 };
    let mut t: Scalar = 0.0;
    let mut n: u32 = 0;
    while t < DURATION - 0.5 * DT {
        n += 1;
        if n > N_STEPS {
            break;
        }
        let v_d: Scalar = 0.0;
        let v_q: Scalar = if t < STEP_AT { 0.0 } else { VOLTAGE };
        match DQ_MODEL.integrate(s, v_d, v_q, DT, 0.0) {
            Ok(next) => s = next,
            Err(e) => {
                eprintln!("integrate refused this step: {}", e);
                return 1;
            }
        }
        t += DT;
        print_row(t, &s, v_d, v_q);
    }
    0
}

fn feedforward_run() -> i32 {
    // speed_limit = the target, so the ramp is limited by acceleration alone and
    // the current limit is what does the limiting: that is the corner this run
    // exists to show.
    let mut ff = match DqFeedForward::with_limits(
        DQ_MODEL, DT, Some(TARGET), None, Some(0.0))
    {
        Ok(ff) => ff,
        Err(e) => {
            eprintln!("feed-forward rejected the sample time: {}", e);
            return 1;
        }
    };
    let mut s = DqState { rot_vel: 0.0, i_d: 0.0, i_q: 0.0 };
    let mut t: Scalar = 0.0;
    let mut reference: Scalar = 0.0;
    let mut first_sat: Scalar = -1.0;
    let mut n: u32 = 0;
    while t < DURATION - 0.5 * DT {
        n += 1;
        if n > N_STEPS {
            break;
        }
        if t >= STEP_AT {
            reference = TARGET;
        }
        let (v_d, v_q, _i_d, _i_q) = ff.update(reference);
        if (ff.current_saturated || ff.voltage_saturated) && first_sat < 0.0 {
            first_sat = t;
        }
        match DQ_MODEL.integrate(s, v_d, v_q, DT, 0.0) {
            Ok(next) => s = next,
            Err(e) => {
                eprintln!("integrate refused this step: {}", e);
                return 1;
            }
        }
        t += DT;
        print_row(t, &s, v_d, v_q);
    }
    if first_sat >= 0.0 {
        eprintln!(
            "feed-forward hit its current/voltage limit at t = {:.4} s: above the \
             field-weakening band edge ({:.1} rad/s motor side) this machine can \
             not follow a reference ramped at its stall acceleration",
            first_sat,
            DQ_MODEL.constant_torque_speed()
        );
    }
    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ff_mode = args.len() > 1 && args[1] == "feedforward";
    println!("time_s,motor_rad_s,current_A,voltage_V");
    eprintln!(
        "control period {} s -> {} RK4 sub-steps at standstill (stable step {} s), \
         {} at base speed (stable step {} s)",
        DT,
        DQ_MODEL.sub_steps(DT, 0.0),
        DQ_MODEL.max_step_time(0.0),
        DQ_MODEL.sub_steps(DT, DQ_MODEL.base_speed()),
        DQ_MODEL.max_step_time(DQ_MODEL.base_speed()),
    );
    std::process::exit(if ff_mode { feedforward_run() } else { plant_run() });
}
''')


# Struct field order, which is also the order the C struct literal and the
# manifest constants are printed in.  Rust spelling, not the C shortening.
_DQ_RUST_FIELDS = (
    "resistance", "inductance_d", "inductance_q", "flux_linkage", "pole_pairs",
    "rotor_inertia", "bearing_damping", "static_friction", "current_limit",
    "voltage_limit", "dc_bus_voltage", "core_loss_coefficient",
    "windage_coefficient", "flux_temperature_coefficient",
    "reference_temperature", "motor_count", "gear_ratio", "gear_efficiency",
    "wheel_radius",
)

# The square-root seed is the exponent-halving bias for IEEE binary64: adding
# it to the halved bit pattern lands within ~3% of the root, and Newton
# converges to full width in four iterations.  The iteration runs at f64 in
# every build, so an f32 export widens rather than carrying a second seed.
_DQ_SQRT_SEED = "0x1FF8_0000_0000_0000"
_DQ_SQRT_ITERATIONS = 6


def _dq_rust_numbers(c: DqModelConstants, dtype: str) -> dict[str, str]:
    """Every machine constant in the struct literal, at the requested width."""
    return {
        "resistance": _rust_num(c.resistance, dtype),
        "inductance_d": _rust_num(c.inductance_d, dtype),
        "inductance_q": _rust_num(c.inductance_q, dtype),
        "flux_linkage": _rust_num(c.flux_linkage, dtype),
        "pole_pairs": str(int(c.pole_pairs)),
        "rotor_inertia": _rust_num(c.rotor_inertia, dtype),
        "bearing_damping": _rust_num(c.bearing_damping, dtype),
        "static_friction": _rust_num(c.static_friction, dtype),
        "current_limit": _rust_num(c.current_limit, dtype),
        "voltage_limit": _rust_num(c.voltage_limit, dtype),
        "dc_bus_voltage": _rust_num(c.dc_bus_voltage, dtype),
        "core_loss_coefficient": _rust_num(c.core_loss_coefficient, dtype),
        "windage_coefficient": _rust_num(c.windage_coefficient, dtype),
        "flux_temperature_coefficient": _rust_num(
            c.flux_temperature_coefficient, dtype),
        "reference_temperature": _rust_num(c.reference_temperature, dtype),
        "motor_count": str(int(c.motor_count)),
        "gear_ratio": _rust_num(c.gear_ratio, dtype),
        "gear_efficiency": _rust_num(c.gear_efficiency, dtype),
        "wheel_radius": _rust_num(c.wheel_radius, dtype),
    }


def _dq_rust_mapping(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    """Substitutions for the Rust template."""
    numbers = _dq_rust_numbers(c, o.dtype)
    return {
        "banner": _banner(o, c, "//"),
        "dtype": o.dtype,
        "model_block": "\n".join(
            f"    {name}: {numbers[name]}," for name in _DQ_RUST_FIELDS),
        "sample_time": _rust_num(o.sample_time or math.inf, o.dtype),
        "max_step": _rust_num(c.max_step_time(), o.dtype),
        # Tolerances print at double precision in every language, so all three
        # models quote the same numbers; an f32 build rounds them at the point
        # of use, which is what the comparisons do anyway.
        "rk4_limit": _rust_num(RK4_STABILITY_LIMIT, "f64"),
        "stiction": _rust_num(STICTION_BAND, "f64"),
        "slack": _rust_num(FEASIBILITY_SLACK, "f64"),
        "envelope_slack": _rust_num(ENVELOPE_SLACK, "f64"),
        "salience_floor": _rust_num(SALIENCE_FLOOR, "f64"),
        "scan_steps": str(WEAKENING_SCAN_STEPS),
        "bisect_steps": str(WEAKENING_BISECT_STEPS),
        "mtpa_steps": str(MTPA_BISECT_STEPS),
        "bits_type": "u64",
        "sqrt_seed": _DQ_SQRT_SEED,
        "sqrt_iterations": str(_DQ_SQRT_ITERATIONS),
    }


def _dq_bench_rust_values(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    """The bench numbers as Rust literals, plus the %g forms quoted in its comment."""
    n = _dq_bench_numbers(o, c)
    dt, duration = n["dt"], n["duration"]
    return {
        "bench_dt": _rust_num(dt, o.dtype),
        "bench_duration": _rust_num(duration, o.dtype),
        "bench_step_at": _rust_num(n["step_at"], o.dtype),
        "bench_voltage": _rust_num(n["voltage"], o.dtype),
        "bench_target": _rust_num(n["target"], o.dtype),
        "dt_text": f"{dt:g}",
        "duration_text": f"{duration:g}",
        "step_at_text": f"{n['step_at']:g}",
        "voltage_text": f"{n['voltage']:g}",
        "target_text": f"{n['target']:g}",
        "n_steps": str(int(math.ceil((duration - 0.5 * dt) / dt))),
    }


def _render_dq_rust(o: ExportOptions, c: DqModelConstants) -> dict[str, str]:
    """The dq plant in Rust, plus the bench when examples are on.

    There is no prefix pass here, unlike the C renderer: a Rust module is
    namespaced by the `mod` item that pulls it in, so a brushed export and a dq
    export of the same machine can sit in one crate without stepping on each
    other's names.
    """
    m = _dq_rust_mapping(o, c)
    source = f"{o.module_name}_dq.rs"
    files = {source: _DQ_RUST.substitute(**m)}
    if o.include_examples:
        bench = f"bench_{o.module_name}_dq.rs"
        files[bench] = _DQ_BENCH_RUST.substitute(
            **m, **_dq_bench_rust_values(o, c), file=bench, source_file=source,
            module_mod=f"{o.module_name}_dq")
    return files


# Which languages the dq export renders, in the order the manifest reports them.
# Registering a renderer here is the whole of what it takes to offer a target.
_DQ_RENDERERS: dict[str, Any] = {"python": _render_dq_python, "c": _render_dq_c,
                                "rust": _render_dq_rust}
_DQ_TARGET_ORDER = ("python", "c", "rust")


def dq_manifest_name(options: ExportOptions) -> str:
    return f"{options.module_name}_dq_manifest.json"


def _render_dq_readme(options: ExportOptions, c: DqModelConstants,
                      names: Sequence[str]) -> str:
    bench = _dq_bench_values(options, c)
    derived = c.derived_dict()
    return _README_DQ.substitute(
        module_name=options.module_name,
        stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        file_list="\n".join(f"- `{n}`" for n in names),
        constant_rows=_markdown_rows(c.as_dict(), options.dtype),
        derived_rows="\n".join(
            f"- `{k}`: {_py_num(v, options.dtype)}" for k, v in derived.items()),
        motor_count=c.motor_count,
        group_current=_py_num(c.motor_count * c.current_limit, options.dtype),
        base_speed=_py_num(c.base_speed, options.dtype),
        note_block="\n".join(f"- {n}" for n in c.notes) or "- none",
        bench_name=f"bench_{options.module_name}_dq.py",
        max_step=_py_num(c.max_step_time(), options.dtype),
        max_step_base=_py_num(c.max_step_time(c.base_speed), options.dtype),
    )


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------
def export_dq(system: DriveSystem, out_dir: str | Path,
              options: ExportOptions | None = None, *,
              constants: DqModelConstants | None = None) -> list[Path]:
    """Write the dq/FOC model of *system* into *out_dir*; returns the paths.

    *system* must carry a brushless machine (`bldc_pod`, or any preset with
    `--motor bldc_24v`).  The dq targets are python, c and rust: a target list
    that names none of them raises rather than quietly picking one, and a target
    this export cannot render yet is reported in the manifest and in the notes
    instead of being dropped.
    """
    options = options or ExportOptions()
    wanted = [t for t in _DQ_TARGET_ORDER if t in options.targets]
    renderable = [t for t in wanted if t in _DQ_RENDERERS]
    skipped = ([t for t in options.targets if t not in wanted]
               + [t for t in wanted if t not in _DQ_RENDERERS])
    if not renderable:
        raise ConfigError(
            "the dq export cannot emit any of the targets you asked for "
            f"({', '.join(options.targets) or 'none'}); the dq targets are "
            + ", ".join(_DQ_TARGET_ORDER) + "."
        )
    c = round_dq_to_dtype(
        constants or dq_constants_from_system(system), options.dtype)
    if skipped:
        # the brushed ExportOptions default asks for c, rust and python; say so
        # in the same place the other caveats live rather than failing quietly
        c = replace(c, notes=c.notes + (
            "targets " + ", ".join(skipped) + " were requested, but this dq "
            "export does not render them today; the brushed export "
            "(drivetrain.export.export) covers them for the i_d=0 collapse of "
            "this same machine",
        ))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for target in renderable:
        written.update(_DQ_RENDERERS[target](options, c))

    paths: list[Path] = []
    for name, text in written.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)

    files = [{"name": p.name,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
             for p in paths]
    manifest = _MANIFEST_DQ.substitute(
        schema=SCHEMA_VERSION,
        stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dtype=options.dtype,
        targets=json.dumps({"requested": list(options.targets),
                            "emitted": renderable,
                            "not_emitted": skipped}, indent=2)
        .replace("\n", "\n  "),
        sample_time=(_py_num(options.sample_time) if options.sample_time else "null"),
        max_step=_py_num(c.max_step_time()),
        provenance=json.dumps(dict(options.provenance or {}), indent=2,
                               default=str).replace("\n", "\n  "),
        constants=json.dumps(c.as_dict(), indent=2, default=str)
        .replace("\n", "\n  "),
        derived=json.dumps(c.derived_dict(), indent=2, default=str)
        .replace("\n", "\n  "),
        notes=json.dumps(list(c.notes), indent=2, default=str)
        .replace("\n", "\n  "),
        files=json.dumps(files, indent=2).replace("\n", "\n  "),
    )
    manifest_path = out / dq_manifest_name(options)
    manifest_path.write_text(manifest, encoding="utf-8")
    paths.append(manifest_path)

    readme = _render_dq_readme(options, c, [p.name for p in paths])
    readme_path = out / f"{options.module_name}_dq_README.md"
    readme_path.write_text(readme, encoding="utf-8")
    paths.append(readme_path)
    return paths


def summarise_dq(system: DriveSystem, options: ExportOptions,
                 paths: Sequence[Path] | None = None) -> None:
    """Print the dq sample-time contract and the current law for a human.

    The brushed summary in :mod:`drivetrain.export` cannot stand in for this
    one: its ``max_step_time`` is half the electrical time constant, which is
    correct for a real pole at ``-R/L`` and wrong for the dq frame, where the
    pole pair is complex and rotates with the shaft.
    """
    c = round_dq_to_dtype(dq_constants_from_system(system), options.dtype)
    source = (options.provenance or {}).get("source")
    if source:
        print(f"source: {source}")
    if paths:
        emitted = [t.name for t in paths]
        # The file suffixes are the honest record of what was rendered: a .h and
        # its .c are one target, and there is no side channel from here back to
        # the target list export_dq() settled on.
        by_suffix = {".py": "python", ".c": "c", ".h": "c", ".rs": "rust"}
        targets = sorted({by_suffix[p.suffix] for p in paths
                          if p.suffix in by_suffix})
        print(f"wrote {len(emitted)} files to {paths[0].parent} "
              f"({options.dtype}, {', '.join(targets)})")
        for p in paths:
            print(f"  {p}")

    dt_stop = c.max_step_time()
    dt_base = c.max_step_time(c.base_speed)
    print(f"\nsample-time contract: dt <= {dt_stop:g} s at standstill, "
          f"{dt_base:g} s at base speed {c.base_speed:.1f} rad/s")
    print("  the dq current loop linearises to a complex pole pair at "
          "-R(Ld+Lq)/(2 Ld Lq) +/- j*w_e, so the stable step shrinks as the shaft speeds up")
    if options.sample_time:
        if options.sample_time > dt_stop:
            print(f"WARNING: --sample-time {options.sample_time:g} s exceeds even the "
                  "standstill limit; step() will raise and integrate() will sub-step "
                  "every period")
        elif options.sample_time > dt_base:
            print(f"note: --sample-time {options.sample_time:g} s is above the stable "
                  "step at speed; step() raises there, integrate() sub-steps")

    edge = c.constant_torque_speed()
    print(f"\nconstant-torque band edge: {edge:.1f} rad/s motor side "
          f"({edge * c.gear_ratio:.2f} rad/s output); base speed "
          f"{c.base_speed:.1f} rad/s")
    if edge < c.base_speed and math.isfinite(edge) and math.isfinite(c.base_speed):
        print(f"  the rail is reached at {100.0 * edge / c.base_speed:.0f}% of base "
              "speed: above that, Lq*i eats voltage the back-EMF also needs, so full "
              "current needs field weakening")
    print(f"current limit: {c.current_limit:g} A phase amplitude per machine, "
          f"{c.current_limit * c.motor_count:g} A across {c.motor_count} machine(s)")
    print(f"i_d floor: {c.id_floor:g} A (characteristic current lambda/Ld = "
          f"{c.characteristic_current:g} A); demagnetisation is not modelled")
    if math.isfinite(c.dc_bus_voltage):
        print(f"DC link: {c.dc_bus_voltage:g} V; the rail ammeter reads "
              "1.5*(v_d*i_d + v_q*i_q)/V_bus, not the phase current")

    print("\ncurrent law (the model inverted, not tuned):")
    print("  tau = 1.5*p*(lambda*i_q + (Ld - Lq)*i_d*i_q);  i_d <= 0, from MTPA "
          "below the band edge and from field weakening above it")
    print("  v_d = R*i_d - w_e*Lq*i_q;  v_q = R*i_q + w_e*(Ld*i_d + lambda)")
    print("  feed the a term the derivative of your LIMITED reference, not of the setpoint")

    if c.notes:
        print("\ncaveats for this machine:")
        for note in c.notes:
            print(f"  ! {note}")

    fit_warnings = (options.provenance or {}).get("fit", {}).get("warnings")
    if fit_warnings:
        print("\ncarried from the fit:")
        for warning in fit_warnings:
            print(f"  ! {warning}")
