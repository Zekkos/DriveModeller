"""Efficiency as a map over (speed, torque), with the losses split by where the heat lands.

The closed-form answers the model gives elsewhere -- ``steady_speed(voltage)``,
``torque_speed_curve()``, ``operating_point(voltage, output_speed)`` -- are all
one-dimensional slices: you pick a voltage and the model tells you where you
land.  That is enough to size a drivetrain but not enough to answer the question
a designer actually asks, which is *how good is this machine at the operating
points it will spend its life in*.  A vehicle idles at low torque and accelerates
at high; a conveyor sits at one point all day.  The answer is a surface.

This module evaluates the machine itself, not the vehicle, so the axes are
**motor-side**:

* ``speed_rad_s``  -- rotor speed at the motor shaft [rad/s]
* ``torque_Nm``    -- torque developed by the whole motor group [N*m]

The gearbox and the load are deliberately off the surface: a reducer rescales
both axes linearly, so a map drawn at the wheels is the same map stretched, and
the interesting structure (where copper loss dominates, where field weakening
starts) gets hidden by the ratio.  Pass ``include_gearbox=True`` to bill the
gear stage as a loss term instead, and read :attr:`EfficiencyMap.notes` for what
that does and does not model.

Sign convention, and why the books close
----------------------------------------
Speed is non-negative on this surface; torque may be either sign.  A negative
torque at a positive speed is braking, and the power flows backwards, so the
map defines input and output per quadrant:

* **motoring** (``torque * speed >= 0``): input is the electrical power drawn,
  output is the shaft power delivered after the machine's own losses.
* **braking** (``torque * speed < 0``): input is the mechanical power the shaft
  has to be *given* to hold that operating point, output is the electrical power
  that comes back out of the machine.

Either way ``power_in_W - power_out_W`` equals the sum of the loss terms exactly
-- that identity is what the tests check, and it is the reason the split is
worth having: a loss table that does not close the books is a guess.

What is *not* modelled
----------------------
There is no magnet-loss term anywhere in this package, so
:attr:`LossSplit.magnet_W` is ``None`` rather than ``0.0``: the machine model has
flux linkage and a flux-temperature coefficient, but no rotor eddy-current loss
and no magnet temperature dynamics.  A brushed collapse has no iron-loss term
either (its ``bearing_damping`` field is documented as "bearing/windage", i.e.
one lumped mechanical term).  Both are reported in
:attr:`EfficiencyMap.not_modelled` instead of being quietly zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .exceptions import ConfigError
from .pmsm import PMSMModel
from .system import DriveSystem

__all__ = [
    "LossSplit",
    "OperatingPoint",
    "EfficiencyMap",
    "operating_point",
    "efficiency_map",
]

# Relative slack on the envelope tests: the dq envelope is found by a fixed-count
# bisection, so a point sitting exactly on it lands a rounding step outside.
_ENVELOPE_SLACK = 1e-9

# Terms the model can bill, in the order the report prints them.
_MODELLED_TERMS = ("copper", "iron", "windage", "bearing", "friction", "gear", "residual")


# ---------------------------------------------------------------------------
# the loss split
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LossSplit:
    """Where the power that did not make it to the shaft went [W].

    Every term is a positive magnitude at a motoring point.  ``magnet_W`` is
    ``None`` because the machine model has no rotor-loss term at all -- see the
    module docstring -- and reporting ``0.0`` there would read as "this machine
    has an ideal magnet", which is a claim nobody has evidence for.
    """

    copper_W: float = 0.0
    iron_W: float = 0.0
    windage_W: float = 0.0
    bearing_W: float = 0.0
    friction_W: float = 0.0
    gear_W: float = 0.0
    residual_W: float = 0.0
    magnet_W: float | None = None

    @property
    def total_W(self) -> float:
        """Sum of the modelled terms.  ``magnet_W`` is excluded: it is not modelled."""
        return (
            self.copper_W
            + self.iron_W
            + self.windage_W
            + self.bearing_W
            + self.friction_W
            + self.gear_W
            + self.residual_W
        )

    def terms(self) -> dict[str, float]:
        """The modelled terms by name, zeros included (they keep a plot aligned)."""
        return {f"{name}_W": getattr(self, f"{name}_W") for name in _MODELLED_TERMS}

    def share(self) -> dict[str, float]:
        """Fraction of the total loss per term; empty when nothing is lost."""
        total = self.total_W
        if total <= 0.0:
            return {name: 0.0 for name in _MODELLED_TERMS}
        return {name: getattr(self, f"{name}_W") / total for name in _MODELLED_TERMS}

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.terms())
        out["magnet_W"] = self.magnet_W
        out["total_W"] = self.total_W
        return out


# ---------------------------------------------------------------------------
# one operating point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatingPoint:
    """A single (speed, torque) point on the machine's own map.

    ``torque_Nm`` is the torque the *group develops* (the axis value);
    ``shaft_power_W`` is what reaches the machine's shaft after the machine's own
    bearings, stiction and iron loss, and ``power_out_W`` is what reaches the
    boundary the map is billed at -- the shaft, or the far side of the gear stage
    when ``include_gearbox`` was set.  Powers are signed: negative developed
    power means the machine is being driven.
    """

    speed_rad_s: float
    torque_Nm: float
    feasible: bool
    power_in_W: float = float("nan")
    power_out_W: float = float("nan")
    developed_W: float = float("nan")
    shaft_power_W: float = float("nan")
    current_A: float = float("nan")
    current_per_motor_A: float = float("nan")
    voltage_V: float = float("nan")
    dc_link_current_A: float | None = None
    efficiency: float | None = None
    losses: LossSplit = field(default_factory=LossSplit)
    current_limited: bool = False
    voltage_limited: bool = False
    field_weakening: bool = False
    why: str = ""

    @property
    def speed_rpm(self) -> float:
        return self.speed_rad_s * 30.0 / math.pi

    def as_dict(self) -> dict[str, Any]:
        return {
            "speed_rad_s": self.speed_rad_s,
            "speed_rpm": self.speed_rpm,
            "torque_Nm": self.torque_Nm,
            "feasible": self.feasible,
            "power_in_W": self.power_in_W,
            "shaft_power_W": self.shaft_power_W,
            "power_out_W": self.power_out_W,
            "developed_W": self.developed_W,
            "current_A": self.current_A,
            "current_per_motor_A": self.current_per_motor_A,
            "voltage_V": self.voltage_V,
            "dc_link_current_A": self.dc_link_current_A,
            "efficiency": self.efficiency,
            "losses": self.losses.as_dict(),
            "current_limited": self.current_limited,
            "voltage_limited": self.voltage_limited,
            "field_weakening": self.field_weakening,
            "why": self.why,
        }

    def report(self) -> str:
        if not self.feasible:
            return f"{self.speed_rpm:8.1f} rpm  {self.torque_Nm:8.4f} N*m   infeasible: {self.why}"
        eff = "  n/a" if self.efficiency is None else f"{100.0 * self.efficiency:6.2f}%"
        lines = [
            f"{self.speed_rpm:8.1f} rpm  {self.torque_Nm:8.4f} N*m   "
            f"eff {eff}   in {self.power_in_W:9.3f} W   out {self.power_out_W:9.3f} W",
        ]
        parts = [_loss_label(name) + f" {val:.3f}"
                 for name, val in self.losses.terms().items() if val > 0.0]
        if parts:
            lines.append("      losses [W]: " + "  ".join(parts))
        if self.why:
            lines.append(f"      note: {self.why}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# machine-side plumbing: one code path per machine kind
# ---------------------------------------------------------------------------
def _loss_label(name: str) -> str:
    """`copper_W` -> `copper`: the report already says what the units are."""
    return name[:-2] if name.endswith("_W") else name


def _dq_motor(system: DriveSystem) -> PMSMModel | None:
    """The brushless machine in *system*, or ``None`` for a brushed collapse.

    Deliberately not :func:`dq_sim.machine_of`: that raises, and this module has
    to answer for both machine kinds.
    """
    motor = system.group.motor
    return motor if isinstance(motor, PMSMModel) else None


def _machine_loss_torque(system: DriveSystem, speed: float, include_gearbox: bool) -> float:
    """Torque the machine burns just to turn at this speed [N*m, opposes motion].

    Mirrors what the integrators charge: bearings and stiction are in the
    mechanical equation of both engines, iron loss is a loss term on the dq side
    only, and the gear stage's viscous damping joins when it is being billed.
    """
    w = abs(speed)
    damping = system.group.bearing_damping_eff
    if include_gearbox:
        damping += system.gearbox.damping
    tau = damping * w
    motor = _dq_motor(system)
    static = system.group.static_friction_eff
    if motor is None or w == 0.0:
        # The brushed engines charge Coulomb stiction at every speed, so there it
        # is a standing loss.  PMSMSpec.loss_torque() charges it only at
        # standstill, where it is a holding torque and dissipates nothing, so a
        # brushless machine must not be billed for it while it is turning.
        tau += static
    if motor is not None:
        spec = motor.spec
        tau += system.group.count * spec.core_loss_coefficient * spec.pole_pairs**2 * w
    return tau


def _brushed_point(
    system: DriveSystem, speed: float, torque: float, include_gearbox: bool
) -> OperatingPoint:
    """Solve the brushed collapse for the (speed, torque) point.

    The brushed machine is a current source in the torque axis, so the required
    winding current follows from ``torque = kt_eff * i`` and the required
    terminal voltage from ``v = R i + kv w``.  Feasibility is then simply whether
    either limit is exceeded.
    """
    group = system.group
    kt, kv, r = float(group.kt_eff), float(group.kv_eff), float(group.r_eff)
    if kt <= 0.0:
        raise ConfigError("the efficiency map needs a motor with a positive torque constant.")
    current = torque / kt
    voltage = current * r + kv * speed

    i_limit = system.current_limit_total
    v_limit = system.voltage_limit
    current_limited = math.isfinite(i_limit) and abs(current) > i_limit * (1.0 + _ENVELOPE_SLACK)
    voltage_limited = math.isfinite(v_limit) and abs(voltage) > v_limit * (1.0 + _ENVELOPE_SLACK)
    if current_limited or voltage_limited:
        which = []
        if current_limited:
            which.append(f"current {abs(current):.3f} A over the {i_limit:.3f} A limit")
        if voltage_limited:
            which.append(f"voltage {abs(voltage):.3f} V over the {v_limit:.3f} V limit")
        return OperatingPoint(speed, torque, False, current_limited=current_limited,
                              voltage_limited=voltage_limited, why="; ".join(which))

    losses = LossSplit(
        copper_W=current * current * r,
        bearing_W=group.bearing_damping_eff * speed * speed,
        friction_W=group.static_friction_eff * abs(speed),
        # (kv - kt) * i * w: the model lets kt and kv be independent datasheet
        # numbers, and in SI an ideal machine has them equal.  Any difference is
        # power that the electrical side and the mechanical side disagree about,
        # so bill it as a term rather than fudging one of the two.
        residual_W=(kv - kt) * current * speed,
    )
    return _bill(system, speed, torque, voltage * current, abs(current),
                 abs(current) / group.count, abs(voltage), losses, include_gearbox,
         current_limited=abs(current) >= i_limit if math.isfinite(i_limit) else False,
         voltage_limited=abs(voltage) >= v_limit if math.isfinite(v_limit) else False)


def _dq_point(system: DriveSystem, speed: float, torque: float, include_gearbox: bool) -> OperatingPoint:
    """Solve the dq machine for the (speed, torque) point.

    The current pair comes from :meth:`PMSMSpec.current_pair`, i.e. MTPA below the
    band edge and the field-weakening ellipse above it, so the map inherits the
    same envelope as ``torque_limit()``.  Feasibility is checked against that
    envelope explicitly: ``current_pair`` answers an infeasible request with the
    best pair it can find rather than with a refusal, and a map cell built on that
    would look real.
    """
    motor = _dq_motor(system)
    assert motor is not None  # the caller picked this path
    spec = motor.spec
    count = system.group.count
    per_machine = torque / count

    tau_limit = count * spec.torque_limit(speed, regenerative=per_machine * speed < 0.0)
    if abs(torque) > abs(tau_limit) * (1.0 + _ENVELOPE_SLACK):
        kind = "braking" if per_machine * speed < 0.0 else "driving"
        return OperatingPoint(
            speed, torque, False,
            why=f"{kind} envelope at this speed is {abs(tau_limit):.4f} N*m "
                f"(current limit {spec.current_limit:.3f} A, voltage limit {spec.voltage_limit:.3f} V)",
        )

    i_d, i_q, cur_sat, volt_sat = spec.current_pair(per_machine, speed)
    v_d, v_q = spec.voltage_steady(i_d, i_q, speed)
    phase_current = math.hypot(i_d, i_q)
    losses = LossSplit(
        copper_W=count * spec.copper_power(i_d, i_q),
        iron_W=count * spec.core_loss_power(speed),
        windage_W=count * spec.windage_power(speed),
        bearing_W=system.group.bearing_damping_eff * speed * speed,
        # A Coulomb term lives only at standstill, where it dissipates nothing,
        # so a brushless machine has no friction column while it is turning: that
        # is how PMSMSpec.loss_torque() treats it.  The brushed collapse charges
        # stiction at every speed, and there it is a standing loss.
        friction_W=0.0,
    )
    power_in = count * spec.input_power(v_d, v_q, i_d, i_q)
    bus = spec.dc_bus_voltage
    dc_link = power_in / bus if math.isfinite(bus) and abs(bus) > 0.0 else None
    point = _bill(system, speed, torque, power_in, count * phase_current, phase_current,
                  math.hypot(v_d, v_q), losses, include_gearbox,
                  current_limited=bool(cur_sat), voltage_limited=bool(volt_sat),
                  # not merely i_d < 0: on a salient machine MTPA drives the d axis
                  # negative at any torque to collect reluctance torque, which is not
                  # field weakening.  Weakening is when the d axis is pushed further
                  # in to stay on the voltage rail, i.e. the solution sits on the
                  # ellipse rather than on MTPA or on the current circle.
                  field_weakening=bool(volt_sat) and i_d < 0.0)
    point = replace(point, dc_link_current_A=dc_link)
    return point


def _bill(
    system: DriveSystem,
    speed: float,
    torque: float,
    electrical_W: float,
    current_A: float,
    per_motor_A: float,
    voltage_V: float,
    losses: LossSplit,
    include_gearbox: bool,
    *,
    current_limited: bool = False,
    voltage_limited: bool = False,
    field_weakening: bool = False,
) -> OperatingPoint:
    """Turn the raw electrical/mechanical numbers into a billed operating point.

    ``electrical_W`` is the power at the machine's electrical terminals, signed by
    the machine's own convention (positive = drawn from the source).  The two
    reported powers keep that sign convention instead of being re-labelled per
    quadrant: ``power_in_W`` is what the source supplies (negative = the machine
    returns power to it) and ``power_out_W`` is what the shaft delivers (negative =
    the load drives the machine).  With that convention

        power_in_W - power_out_W == losses.total_W

    holds in every quadrant, which is the whole reason the split carries a
    residual term: it is what makes the identity exact for a collapse whose ``kt``
    and ``kv`` disagree.  ``efficiency`` is useful-over-absorbed, and which side is
    useful depends on the quadrant.
    """
    developed = torque * speed
    mech = losses.iron_W + losses.windage_W + losses.bearing_W + losses.friction_W
    shaft = developed - mech
    gear_W = 0.0
    if include_gearbox:
        eta = float(system.gearbox.efficiency)
        gear_W = (1.0 - eta) * abs(shaft) + system.gearbox.damping * speed * speed
    losses = replace(losses, gear_W=gear_W)

    power_in = electrical_W
    power_out = shaft - gear_W

    notes = []
    if developed >= 0.0:
        # Motoring: the source is the input, the shaft is the output.
        if power_in > 0.0 and power_out > 0.0:
            efficiency = min(1.0, max(0.0, power_out / power_in))
        else:
            efficiency = None
            if power_in > 0.0:
                notes.append("the machine's own losses consume the whole developed power")
            else:
                notes.append("the electrical side draws no power at this point")
    else:
        # Braking: the shaft is the input and the terminals are the output, so the
        # ratio inverts.  A positive electrical power here means the drive is still
        # pushing against the source -- the shaft's power is being burned in the
        # winding (plugging) rather than returned to the bus.
        if power_in < 0.0 and power_out < 0.0:
            efficiency = min(1.0, max(0.0, power_in / power_out))
        else:
            efficiency = None
            notes.append(
                "braking here is not regenerative: the drive still supplies power, so "
                "the shaft's power is burned in the winding instead of returned"
            )

    return OperatingPoint(
        speed_rad_s=speed,
        torque_Nm=torque,
        feasible=True,
        power_in_W=power_in,
        power_out_W=power_out,
        developed_W=developed,
        shaft_power_W=shaft,
        current_A=current_A,
        current_per_motor_A=per_motor_A,
        voltage_V=voltage_V,
        efficiency=efficiency,
        losses=losses,
        current_limited=current_limited,
        voltage_limited=voltage_limited,
        field_weakening=field_weakening,
        why="; ".join(notes),
    )


def operating_point(
    system: DriveSystem,
    speed_rad_s: float,
    torque_Nm: float,
    *,
    torque_is: str = "developed",
    include_gearbox: bool = False,
) -> OperatingPoint:
    """The efficiency and loss split at one (speed, torque) point on the machine.

    ``torque_is`` selects which torque the axis means:

    * ``"developed"`` (default) -- the electromagnetic torque the machine
      produces, which is the same quantity ``torque_limit()`` and
      ``current_pair()`` describe, so a point on the envelope is on the envelope.
    * ``"shaft"`` -- the torque actually delivered at the machine's shaft, after
      the machine's own bearings, stiction and iron loss.  The developed torque is
      recovered by adding that loss back, which is what a duty cycle asks for.

    ``include_gearbox=True`` bills the gear stage -- its ``efficiency`` as a
    fraction of the shaft power plus its ``damping`` as a viscous loss -- and
    moves the map's output boundary to the far side of it.
    """
    speed = float(speed_rad_s)
    torque = float(torque_Nm)
    if torque_is not in ("developed", "shaft"):
        raise ConfigError(
            f"torque_is must be 'developed' or 'shaft', got {torque_is!r}."
        )
    if torque_is == "shaft" and torque:
        loss = _machine_loss_torque(system, speed, include_gearbox)
        torque = torque + math.copysign(loss, torque)
    if _dq_motor(system) is not None:
        return _dq_point(system, speed, torque, include_gearbox)
    return _brushed_point(system, speed, torque, include_gearbox)


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------

#: Every matrix an :class:`EfficiencyMap` carries, in report order.
_MATRIX_KEYS = (
    "feasible",
    "efficiency",
    "power_in_W",
    "power_out_W",
    "developed_W",
    "shaft_power_W",
    "current_A",
    "current_per_motor_A",
    "voltage_V",
    "dc_link_current_A",
    "copper_W",
    "iron_W",
    "windage_W",
    "bearing_W",
    "friction_W",
    "gear_W",
    "magnet_W",
    "residual_W",
    "total_W",
)

#: The loss columns that actually carry a number on this model.
_LOSS_KEYS = (
    "copper_W",
    "iron_W",
    "windage_W",
    "bearing_W",
    "friction_W",
    "gear_W",
    "residual_W",
)


def _axis(values, n, top, name: str, unit: str) -> np.ndarray:
    """One axis of the grid: an explicit list, or ``n`` points from 0 to *top*."""
    if values is not None:
        grid = np.asarray(values, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ConfigError(
                f"{name} must be a 1-D sequence of at least two values; got shape "
                f"{grid.shape}."
            )
        if np.any(~np.isfinite(grid)):
            raise ConfigError(f"{name} must be finite throughout.")
        if np.any(np.diff(grid) <= 0.0):
            raise ConfigError(f"{name} must be strictly increasing.")
        return grid
    if int(n) < 2:
        raise ConfigError(f"n_{name} must be at least 2, got {n}.")
    if not math.isfinite(top) or top <= 0.0:
        raise ConfigError(
            f"the {name} range needs a positive finite upper bound; got {top!r} {unit}."
        )
    return np.linspace(0.0, float(top), int(n))


def _avail_current(voltage: float, resistance: float, limit: float) -> float:
    """Current a drive with authority over *voltage* can push through *resistance*."""
    if voltage <= 0.0:
        return 0.0 if math.isfinite(voltage) else float("inf")
    available = voltage / resistance if resistance > 0.0 else float("inf")
    return min(available, limit) if math.isfinite(limit) else available


def _brushed_envelope(system: DriveSystem, speed: float) -> tuple[float, float]:
    """(driving, braking) torque magnitudes the brushed collapse can hold at *speed*.

    The braking number is the larger one, and for the same reason the dq model's
    ``regenerative=True`` envelope is: at speed the back-EMF and the resistive drop
    partly cancel, so a drive that can reverse its terminal voltage can hold more
    braking current than driving current.
    """
    group = system.group
    kt = float(group.kt_eff)
    kv = float(group.kv_eff)
    r = float(group.r_eff)
    i_limit = float(system.current_limit_total)
    v_limit = float(system.voltage_limit)
    back_emf = kv * abs(float(speed))
    drive_v = (v_limit - back_emf) if math.isfinite(v_limit) else float("inf")
    brake_v = (v_limit + back_emf) if math.isfinite(v_limit) else float("inf")
    return (
        kt * _avail_current(drive_v, r, i_limit),
        kt * _avail_current(brake_v, r, i_limit),
    )


def _envelope_pair(system: DriveSystem, speed: float) -> tuple[float, float]:
    """(driving, braking) torque magnitudes at *speed*, for either machine kind."""
    motor = _dq_motor(system)
    if motor is not None:
        count = system.group.count
        return (
            count * float(motor.spec.torque_limit(float(speed))),
            count * float(motor.spec.torque_limit(float(speed), regenerative=True)),
        )
    return _brushed_envelope(system, speed)


def _speed_top(system: DriveSystem, speed_top, notes: list) -> float:
    """The top of the speed axis when the caller did not name one."""
    if speed_top is not None:
        return float(speed_top)
    motor = _dq_motor(system)
    if motor is not None:
        top = float(motor.spec.max_speed(0.0))
        source = "the machine's own top speed at no load, set by its voltage limit"
        # With i_d allowed down to -lambda/Ld the dq voltage requirement stops
        # growing with speed, so the no-load top speed can be an order of magnitude
        # above the region anyone would map.  Cap the axis and say why.
        base = float(motor.spec.base_speed())
        if math.isfinite(base) and base > 0.0 and (
            not math.isfinite(top) or top > 2.0 * base
        ):
            top = 2.0 * base
            source = (
                f"twice the base speed ({base:.6g} rad/s): with i_d allowed to "
                f"-lambda/Ld this model's voltage requirement stops rising with speed, so "
                f"the machine's own no-load top speed sits far above the region a map is "
                f"useful for -- pass speed_top=... to see further"
            )
    else:
        top = float(system.steady_speed(system.voltage_limit))
        source = "the collapse's no-load speed at the system voltage limit"
    if not math.isfinite(top) or top <= 0.0:
        top = float(system.no_load_speed)
        source = "the ideal no-load speed (voltage limit / back-EMF constant)"
    if not math.isfinite(top) or top <= 0.0:
        raise ConfigError(
            "this system has no natural top speed -- the voltage limit is unbounded and "
            "nothing else limits the speed -- so the map cannot pick a speed range. Pass "
            "speed_top=... in motor-side rad/s."
        )
    notes.append(f"speed range 0 to {top:.6g} rad/s taken from {source}")
    return top


def _json_matrix(matrix: np.ndarray) -> list:
    """A nested list with every non-finite cell written as ``None``.

    NaN is what numpy prints and what a strict JSON parser refuses, so a map
    dumped with ``allow_nan=False`` -- which is what a plotting script or a
    spreadsheet importer asks for -- would not survive the round trip.  ``None``
    is also the honest spelling of "this cell sat outside the envelope": it is a
    missing measurement, not a number that happens to be odd.
    """
    if matrix.dtype.kind == "b":
        return [[bool(value) for value in row] for row in matrix]
    return [
        [None if not math.isfinite(float(value)) else float(value) for value in row]
        for row in matrix
    ]


def _json_vector(values) -> list:
    """One axis or ridge as a list, non-finite entries as ``None``."""
    return [None if not math.isfinite(float(value)) else float(value) for value in values]


@dataclass(frozen=True)
class EfficiencyMap:
    """A rectangular grid of :class:`OperatingPoint` over (motor speed, motor torque).

    ``speeds`` indexes axis 0 and ``torques`` axis 1 of every matrix in ``grid``.
    A cell outside the machine's envelope is ``NaN`` (and ``False`` in
    ``grid["feasible"]``) and is never filled in: a map that interpolates across an
    envelope boundary is the quickest way to size a drivetrain for a torque the
    machine cannot make.  Call :meth:`envelope` for the boundary itself, which is
    computed exactly rather than read off the grid.
    """

    speeds: np.ndarray
    torques: np.ndarray
    kind: str
    points: tuple
    grid: dict
    system: DriveSystem
    notes: tuple = ()
    not_modelled: tuple = ()
    include_gearbox: bool = False
    torque_is: str = "developed"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.speeds.size, self.torques.size)

    def matrix(self, key: str) -> np.ndarray:
        try:
            return self.grid[key]
        except KeyError:
            raise ConfigError(
                f"the map has no {key!r} matrix; it carries {sorted(self.grid)}."
            ) from None

    @property
    def efficiency(self) -> np.ndarray:
        return self.matrix("efficiency")

    @property
    def feasible(self) -> np.ndarray:
        return self.matrix("feasible")

    def n_feasible(self) -> int:
        return int(self.feasible.sum())

    def points_at(self, speed_index: int) -> list[OperatingPoint]:
        return list(self.points[speed_index])

    def best(self, quadrant: str = "motoring") -> dict | None:
        """The most efficient feasible point, and where it sits.

        ``quadrant`` is ``"motoring"`` (torque >= 0), ``"braking"`` (torque < 0) or
        ``"any"``.  The answer is a dict rather than a bare number because "the peak
        efficiency" is useless without the speed and torque that produced it, and
        because the loss share at that point is what tells you what to fix.
        """
        if quadrant not in ("motoring", "braking", "any"):
            raise ConfigError(
                f"quadrant must be 'motoring', 'braking' or 'any', got {quadrant!r}."
            )
        best_point = None
        for row in self.points:
            for point in row:
                if not point.feasible or point.efficiency is None:
                    continue
                if quadrant == "motoring" and point.torque_Nm < 0.0:
                    continue
                if quadrant == "braking" and point.torque_Nm >= 0.0:
                    continue
                if best_point is None or point.efficiency > best_point.efficiency:
                    best_point = point
        if best_point is None:
            return None
        return {
            "speed_rad_s": best_point.speed_rad_s,
            "speed_rpm": best_point.speed_rpm,
            "torque_Nm": best_point.torque_Nm,
            "efficiency": best_point.efficiency,
            "power_in_W": best_point.power_in_W,
            "power_out_W": best_point.power_out_W,
            "losses": best_point.losses.as_dict(),
            "share": best_point.losses.share(),
        }

    def _best_per_row(self, quadrant: str) -> list[OperatingPoint | None]:
        """The most efficient feasible point in each speed row, quadrant-filtered.

        One scan serves :meth:`ridge`, :meth:`ridge_torque` and the report table, so
        they can never disagree about which point won.
        """
        if quadrant not in ("motoring", "braking", "any"):
            raise ConfigError(
                f"quadrant must be 'motoring', 'braking' or 'any', got {quadrant!r}."
            )
        out: list[OperatingPoint | None] = []
        for row in self.points:
            winner = None
            for point in row:
                if point.efficiency is None:
                    continue
                if quadrant == "motoring" and point.torque_Nm < 0.0:
                    continue
                if quadrant == "braking" and point.torque_Nm >= 0.0:
                    continue
                if winner is None or point.efficiency > winner.efficiency:
                    winner = point
            out.append(winner)
        return out

    def ridge(self, quadrant: str = "motoring") -> np.ndarray:
        """Best efficiency found at each speed, ``NaN`` where nothing is feasible.

        The ridge of the surface is the curve a controller should be following; the
        peak of the whole map is a single point on it.  The default is the driving
        ridge: asking for the best of either quadrant at a given speed answers a
        question nobody asks, because the two have different inputs and different
        outputs (see :meth:`_bill`), so ``quadrant="any"`` exists only to compare
        them.
        """
        out = np.full(self.speeds.size, np.nan)
        for i, winner in enumerate(self._best_per_row(quadrant)):
            if winner is not None:
                out[i] = winner.efficiency
        return out

    def ridge_torque(self, quadrant: str = "motoring") -> np.ndarray:
        """The torque that produces :meth:`ridge` at each speed."""
        out = np.full(self.speeds.size, np.nan)
        for i, winner in enumerate(self._best_per_row(quadrant)):
            if winner is not None:
                out[i] = winner.torque_Nm
        return out

    def envelope(self) -> dict[str, np.ndarray]:
        """The exact torque envelope at each speed, not the grid's approximation.

        ``drive_Nm`` is the largest driving torque and ``brake_Nm`` the largest
        braking torque (a negative number) at that speed.  For a brushless machine
        these are ``torque_limit()`` and ``torque_limit(regenerative=True)`` times
        the motor count, so they are the same envelope the current search uses.
        """
        drive = np.empty(self.speeds.size)
        brake = np.empty(self.speeds.size)
        for i, w in enumerate(self.speeds):
            up, down = _envelope_pair(self.system, float(w))
            drive[i] = up
            brake[i] = -down
        return {"speed_rad_s": self.speeds.copy(), "drive_Nm": drive, "brake_Nm": brake}

    def rows(self) -> Iterable[dict]:
        """Every cell as a flat dict, for a spreadsheet or a plotting script."""
        for j, w in enumerate(self.speeds):
            for k, tau in enumerate(self.torques):
                point = self.points[j][k]
                row = {
                    "speed_rad_s": w,
                    "speed_rpm": point.speed_rpm,
                    "torque_Nm": tau,
                    "feasible": point.feasible,
                    "efficiency": point.efficiency,
                    "power_in_W": point.power_in_W,
                    "power_out_W": point.power_out_W,
                    "developed_W": point.developed_W,
                    "shaft_power_W": point.shaft_power_W,
                    "current_A": point.current_A,
                    "current_per_motor_A": point.current_per_motor_A,
                    "voltage_V": point.voltage_V,
                    "dc_link_current_A": point.dc_link_current_A,
                }
                row.update(point.losses.as_dict())
                row["why"] = point.why
                yield row

    def write_csv(self, path) -> Path:
        """Write the whole map, one row per cell, as CSV.  Returns the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "speed_rad_s", "speed_rpm", "torque_Nm", "feasible", "efficiency",
            "power_in_W", "power_out_W", "developed_W", "shaft_power_W",
            "current_A", "current_per_motor_A", "voltage_V", "dc_link_current_A",
            *_LOSS_KEYS, "magnet_W", "total_W", "why",
        ]
        with destination.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in self.rows():
                writer.writerow(
                    ["" if row[key] is None else row[key] for key in header]
                )
        return destination

    def as_dict(self) -> dict:
        """JSON-ready: the axes, every matrix, the peak point, notes and caveats."""
        return {
            "kind": self.kind,
            "axes": {
                "speed_rad_s": self.speeds.tolist(),
                "torque_Nm": self.torques.tolist(),
                "torque_is": self.torque_is,
                "include_gearbox": self.include_gearbox,
            },
            "grid": {
                key: _json_matrix(matrix) for key, matrix in self.grid.items()
            },
            "best_motoring": self.best("motoring"),
            "best_braking": self.best("braking"),
            "ridge_efficiency": _json_vector(self.ridge()),
            "ridge_torque": _json_vector(self.ridge_torque()),
            "envelope": {k: _json_vector(v) for k, v in self.envelope().items()},
            "not_modelled": list(self.not_modelled),
            "notes": list(self.notes),
        }

    def report(self) -> str:
        lines = [
            f"efficiency map: {self.kind}",
            f"axes: motor speed [rad/s] x motor torque [N*m] ({self.torque_is} torque"
            f"{', gear stage billed' if self.include_gearbox else ', gear stage not billed'})",
            f"grid: {self.speeds.size} x {self.torques.size} = "
            f"{self.speeds.size * self.torques.size} cells, {self.n_feasible()} inside "
            f"the envelope",
        ]
        for quadrant, label in (("motoring", "best motoring"), ("braking", "best braking")):
            best = self.best(quadrant)
            if best is None:
                lines.append(f"{label}: nothing feasible in that quadrant")
                continue
            lines.append(
                f"{label}: {best['speed_rad_s']:.4f} rad/s "
                f"({best['speed_rpm']:.1f} rpm) at {best['torque_Nm']:.6g} N*m "
                f"-> {100.0 * best['efficiency']:.2f}% "
                f"({best['power_in_W']:.4g} W in, {best['power_out_W']:.4g} W out)"
            )
            share = {k: v for k, v in best["share"].items() if v > 0.0}
            if share:
                lines.append(
                    "  loss share: "
                    + ", ".join(f"{_loss_label(k)} {100.0 * v:.1f}%" for k, v in share.items())
                )
        ridge_row = self._best_per_row("motoring")
        step = max(1, math.ceil(self.speeds.size / 12))
        lines.append("driving ridge (best efficiency at each speed):")
        lines.append("    speed_rpm   torque_Nm   efficiency   power_in_W   power_out_W")
        for i in range(0, self.speeds.size, step):
            point = ridge_row[i]
            if point is None:
                lines.append(
                    f"    {self.speeds[i] * 60.0 / (2.0 * math.pi):10.1f}   "
                    f"no cell here delivers output power to bill"
                )
                continue
            lines.append(
                f"    {self.speeds[i] * 60.0 / (2.0 * math.pi):10.1f}   "
                f"{point.torque_Nm:9.5g}   {100.0 * point.efficiency:10.2f}%   "
                f"{point.power_in_W:11.5g}   {point.power_out_W:11.5g}"
            )
        if self.not_modelled:
            lines.append("not modelled on this map: " + ", ".join(self.not_modelled))
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def efficiency_map(
    system: DriveSystem,
    *,
    speeds: Sequence[float] | None = None,
    torques: Sequence[float] | None = None,
    n_speeds: int = 33,
    n_torques: int = 25,
    speed_top: float | None = None,
    torque_top: float | None = None,
    include_regen: bool = True,
    torque_is: str = "developed",
    include_gearbox: bool = False,
) -> EfficiencyMap:
    """Evaluate the whole (speed, torque) plane for *system*.

    The axes are motor-side, because that is where the machine's own losses live:
    reflecting them to the wheel turns the map into the same shape scaled by the
    gear ratio, which hides the reducer rather than showing it.  Pass
    ``include_gearbox=True`` to bill the gear stage as well.  The far side of the
    gearbox is one call away (``system.gearbox.output_torque(tau)`` and
    ``system.gearbox.output_speed(w)``); the map deliberately carries no wheel-side
    columns, because they are the motor-side columns rescaled.

    With no ``speeds``/``torques`` given, the grid spans 0 to the machine's own top
    speed and ``-brake_top`` to ``+drive_top`` (``include_regen=False`` starts the
    torque axis at zero).  The default is deliberately generous: the interesting
    part of a brushless map is the field-weakening corner, and a grid that stops at
    the base speed never shows it.
    """
    if torque_is not in ("developed", "shaft"):
        raise ConfigError(
            f"torque_is must be 'developed' or 'shaft', got {torque_is!r}."
        )
    notes: list[str] = []
    motor = _dq_motor(system)
    count = system.group.count

    w_axis = _axis(speeds, n_speeds, _speed_top(system, speed_top, notes), "speeds", "rad/s")

    if torques is None:
        if torque_top is not None:
            drive_top = brake_top = float(torque_top)
        else:
            drive_top, brake_top = _envelope_pair(system, 0.0)
            if not math.isfinite(drive_top) or drive_top <= 0.0:
                raise ConfigError(
                    "this system has no torque limit at standstill -- neither a current "
                    "limit nor a voltage limit -- so the map cannot pick a torque range. "
                    "Pass torque_top=... in motor-side N*m."
                )
            brake_top = brake_top if include_regen else 0.0
        if include_regen:
            tau_axis = np.linspace(-float(brake_top), float(drive_top), int(n_torques))
        else:
            tau_axis = np.linspace(0.0, float(drive_top), int(n_torques))
        if torque_top is None and brake_top > drive_top:
            notes.append(
                f"torque range -{brake_top:.6g} to +{drive_top:.6g} N*m: the braking "
                f"envelope at standstill is the same as the driving one, and above the "
                f"base speed it is the larger of the two"
            )
    else:
        tau_axis = _axis(torques, n_torques, 0.0, "torques", "N*m")

    points: list[list[OperatingPoint]] = []
    for w in w_axis:
        points.append(
            [operating_point(system, w, tau, torque_is=torque_is,
                             include_gearbox=include_gearbox) for tau in tau_axis]
        )

    grid: dict[str, np.ndarray] = {}
    for key in _MATRIX_KEYS:
        if key == "feasible":
            grid[key] = np.zeros((w_axis.size, tau_axis.size), dtype=bool)
        else:
            grid[key] = np.full((w_axis.size, tau_axis.size), np.nan)
    for j, row in enumerate(points):
        for k, point in enumerate(row):
            if not point.feasible:
                continue
            grid["feasible"][j, k] = True
            for key in _MATRIX_KEYS:
                if key == "feasible":
                    continue
                value = getattr(point, key, None)
                if value is None:
                    continue
                grid[key][j, k] = value
            for key in _LOSS_KEYS:
                grid[key][j, k] = getattr(point.losses, key)
            grid["total_W"][j, k] = point.losses.total_W

    kind = f"{type(system.group.motor).__name__} x{count}"
    not_modelled: list[str] = []
    if motor is None:
        not_modelled.append("iron loss (the brushed collapse has no iron term)")
    not_modelled.append(
        "magnet (rotor) loss -- no coefficient exists in this model, so the split "
        "reports it as unmodelled rather than as zero"
    )
    if motor is None:
        notes.append(
            "the brushed collapse lumps bearing and windage into bearing_damping, so "
            "the windage column stays at zero by construction"
        )
    if motor is None:
        if abs(system.group.kv_eff - system.group.kt_eff) > 1e-9 * max(
            abs(system.group.kv_eff), abs(system.group.kt_eff), 1.0
        ):
            notes.append(
                f"kt ({system.group.kt_eff:g}) and kv ({system.group.kv_eff:g}) differ, so "
                f"the (kv - kt) * i * w term is billed as a residual loss to keep the books "
                f"closed; a real machine's kt and kv are the same number in SI"
            )
    else:
        notes.append(
            "the dq books close exactly: kt = 1.5 p lambda and kv = p lambda differ by the "
            "Park convention, not by a loss, so the residual column is zero by construction"
        )
    if include_gearbox:
        notes.append(
            f"the gear stage is billed as {1.0 - system.gearbox.efficiency:.3g} of the "
            f"shaft power plus {system.gearbox.damping:g} N*m/(rad/s) of gear-cage viscous "
            f"drag; the same efficiency is applied in both directions, which a worm "
            f"stage with a different return efficiency would not be"
        )
    if torque_is == "shaft":
        notes.append(
            "the torque axis is the shaft-delivered torque, so the machine's own loss "
            "torque is added back to find the developed torque that produces it"
        )
    notes.append(
        "the grid covers non-negative speed; (-speed, -torque) mirrors (speed, torque), "
        "which is why the torque axis carries the braking half"
    )

    return EfficiencyMap(
        speeds=w_axis,
        torques=tau_axis,
        kind=kind,
        points=tuple(tuple(row) for row in points),
        grid=grid,
        system=system,
        notes=tuple(notes),
        not_modelled=tuple(not_modelled),
        include_gearbox=include_gearbox,
        torque_is=torque_is,
    )
