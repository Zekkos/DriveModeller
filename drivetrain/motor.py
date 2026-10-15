"""Brushed DC motor model.

Two levels of fidelity are provided:

``MotorModel``
    Full second-order electro-mechanical model including winding inductance::

        L di/dt     = V - i*R - kv*omega
        J domega/dt = kt*i - b*omega - tau_load

``MotorModel`` with ``inductance == 0`` (the default)
    Quasi-electro-static reduction of the above, which is what the original
    ``motor.py`` implemented.  Electrical dynamics are assumed instantaneous, so
    torque is a linear function of speed::

        tau(V, omega) = tau_stall - (kt*kv/R) * omega

    Valid whenever the electrical time constant ``L/R`` is much shorter than the
    mechanical time constant ``J*R/(kt*kv)`` -- true for most small gear motors.

Parameter source for the default values:
    https://eprints.polsri.ac.id/4591/14/7%20Motor%20DC%208p.pdf
    (C23-L33 gearmotor, winding code 20)

Unit convention (SI throughout, angles in radians):
    voltage  V          current  A
    torque   N*m        speed    rad/s
    inertia  kg*m^2     damping  N*m/(rad/s)
    kv       V/(rad/s)  kt       N*m/A
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .exceptions import ConfigError

TWO_PI = 2.0 * np.pi


def rpm_to_rad_s(rpm) -> np.ndarray | float:
    """Convert revolutions per minute to radians per second (scalar or array)."""
    return np.asarray(rpm, float) * TWO_PI / 60.0


def rad_s_to_rpm(rad_s) -> np.ndarray | float:
    """Convert radians per second to revolutions per minute (scalar or array)."""
    return np.asarray(rad_s, float) * 60.0 / TWO_PI


@dataclass
class MotorModel:
    """Brushed DC motor (second order when ``inductance > 0``, else first order).

    Sign convention: ``rot_vel`` and produced torque are positive in the driving
    direction.  A passive load always opposes motion and is applied as a
    magnitude by :class:`~drivetrain.system.DriveSystem`.
    """

    # -- electromechanical constants -------------------------------------
    kv: float = 0.0301                  # back-EMF constant      [V/(rad/s)]
    kt: float = 0.0300                  # torque constant        [N*m/A]
    resistance: float = 1.0             # winding resistance     [Ohm]
    inductance: float = 0.0             # winding inductance     [H]  (0 => quasi-static)
    rotor_inertia: float = 1.554e-5     # rotor + armature inertia [kg*m^2]
    bearing_damping: float = 1.0e-6     # mechanical bearing/windage damping [N*m/(rad/s)]
    static_friction: float = 0.0        # Coulomb friction torque [N*m]
    no_load_friction_current: float = 0.0  # datasheet no-load current [A]

    # -- limits ----------------------------------------------------------
    max_current: float = np.inf         # continuous current limit [A]
    max_voltage: float = np.inf         # applied voltage limit    [V]

    # -- state -----------------------------------------------------------
    # ``current`` is runtime state, not a design input: it is solved for at every
    # step, so the UI must not offer it as a knob.  ``rot_vel`` *is* editable
    # because it is an initial condition you legitimately want to sweep.
    current: float = field(default=0.0, metadata={"state": True})  # winding current [A]
    rot_vel: float = 0.0                # shaft speed     [rad/s]

    # -- derived quantities ----------------------------------------------
    @property
    def rpm(self) -> float:
        """Shaft speed in revolutions per minute."""
        return rad_s_to_rpm(self.rot_vel)

    @property
    def back_emf(self) -> float:
        """Counter EMF at the current speed [V]."""
        return self.kv * self.rot_vel

    @property
    def damping_coefficient(self) -> float:
        """Total small-signal viscous damping [N*m/(rad/s)].

        Includes the electromechanical term ``kt*kv/R``, which is present even
        with lossless bearings because the shorted winding converts kinetic
        energy into copper loss (dynamic braking).
        """
        return self.bearing_damping + (self.kt * self.kv) / self.resistance

    @property
    def no_load_speed_per_volt(self) -> float:
        """Ideal no-load speed per volt of applied voltage [rad/s/V]."""
        return 1.0 / self.kv if self.kv else np.inf

    @property
    def stall_torque_per_volt(self) -> float:
        """Locked-rotor torque per volt [N*m/V]."""
        return self.kt / self.resistance

    @property
    def stall_current_per_volt(self) -> float:
        """Locked-rotor current per volt [A/V]."""
        return 1.0 / self.resistance

    @property
    def mech_time_constant(self) -> float:
        """Mechanical time constant ``J*R/(kt*kv)`` [s]."""
        denom = self.kt * self.kv
        return self.rotor_inertia * self.resistance / denom if denom else np.inf

    @property
    def electrical_time_constant(self) -> float:
        """Electrical time constant ``L/R`` [s]."""
        return self.inductance / self.resistance if self.resistance else 0.0

    @property
    def ideal_no_load_rpm(self) -> float:
        """No-load speed at the configured voltage limit [rpm]."""
        v = 0.0 if np.isinf(self.max_voltage) else self.max_voltage
        return rad_s_to_rpm(v / self.kv) if self.kv and v else np.inf

    # -- torque / current characteristics --------------------------------
    def torque_at(self, voltage: float, speed: float) -> float:
        """Quasi-static shaft torque at a given voltage and speed [N*m]."""
        return self.stall_torque_per_volt * (voltage - self.kv * speed)

    def current_at(self, voltage: float, speed: float) -> float:
        """Quasi-static winding current at a given voltage and speed [A]."""
        return (voltage - self.kv * speed) / self.resistance

    def thermal_power(self, current: float | None = None) -> float:
        """Instantaneous copper loss ``I^2 R`` [W]."""
        i = self.current if current is None else current
        return i * i * self.resistance

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        if self.resistance <= 0:
            raise ConfigError("Motor winding resistance must be > 0.")
        if self.kt <= 0 or self.kv <= 0:
            raise ConfigError("Motor kt and kv must be > 0.")
        if self.rotor_inertia <= 0:
            raise ConfigError("Motor rotor inertia must be > 0.")
        if self.inductance < 0:
            raise ConfigError("Motor inductance must be >= 0.")
        if self.bearing_damping < 0:
            raise ConfigError("Motor bearing damping must be >= 0.")
        if self.max_current <= 0:
            raise ConfigError("Motor current limit must be > 0 (use math.inf for unlimited).")

    def copy(self) -> MotorModel:
        return replace(self)


@dataclass
class MotorGroup:
    """``count`` identical motors bolted to a common shaft / drivetrain.

    Because the motors are identical and share one shaft, the parallel
    combination is exactly equivalent to a single machine with::

        kt_eff = kt    kv_eff = kv    R_eff = R/n    J_eff = n*J

    kt and kv do NOT scale: the windings are parallel back-EMF sources, and
    torque per *total* ampere is unchanged.  Only resistance drops, so::

        stall torque  = n * kt * V / R     (grows linearly)
        no-load speed = V / kv             (unchanged)
        em damping    = n * kt * kv / R    (grows linearly)
        mech tau      = J * R / (kt * kv)  (unchanged)

    Adding motors therefore buys torque and current headroom and cuts the
    per-motor current, but does not raise top speed.  The closed form is used by
    the fast solvers; per-motor state is still reported so that current sharing
    and per-motor limits can be checked.
    """

    motor: MotorModel = field(default_factory=MotorModel)
    count: int = 1

    def __post_init__(self) -> None:
        self.count = int(self.count)
        if self.count < 1:
            raise ConfigError("Motor group needs at least one motor.")
        self.motor.validate()

    # -- equivalent single-machine view ----------------------------------
    #
    # Identical motors on one shaft are parallel voltage sources, so kt and kv
    # do NOT scale with count -- only the winding resistance does.
    @property
    def kt_eff(self) -> float:
        """Torque per *total* ampere -- unchanged by paralleling."""
        return self.motor.kt

    @property
    def kv_eff(self) -> float:
        """Back-EMF per speed -- unchanged (parallel identical voltage sources)."""
        return self.motor.kv

    @property
    def r_eff(self) -> float:
        return self.motor.resistance / self.count

    @property
    def l_eff(self) -> float:
        """Parallel windings see the same back-EMF, so ``L_eff = L / n``."""
        return self.motor.inductance / self.count

    @property
    def inertia_eff(self) -> float:
        return self.count * self.motor.rotor_inertia

    @property
    def bearing_damping_eff(self) -> float:
        return self.count * self.motor.bearing_damping

    @property
    def static_friction_eff(self) -> float:
        return self.count * self.motor.static_friction

    @property
    def damping_total(self) -> float:
        """Bearing damping plus the electromechanical damping term ``kt*kv/R_eff``."""
        return self.bearing_damping_eff + (self.kt_eff * self.kv_eff) / self.r_eff

    @property
    def stall_torque(self) -> float:
        """Group locked-rotor torque per volt [N*m/V]."""
        return self.kt_eff / self.r_eff

    @property
    def no_load_speed_per_volt(self) -> float:
        """No-load speed per volt -- unchanged by paralleling identical motors."""
        return 1.0 / self.motor.kv if self.motor.kv else np.inf

    @property
    def mech_time_constant(self) -> float:
        denom = self.kt_eff * self.kv_eff
        return self.inertia_eff * self.r_eff / denom if denom else np.inf

    # -- state -----------------------------------------------------------
    @property
    def rot_vel(self) -> float:
        return self.motor.rot_vel

    @rot_vel.setter
    def rot_vel(self, value: float) -> None:
        self.motor.rot_vel = float(value)

    @property
    def rpm(self) -> float:
        return rad_s_to_rpm(self.rot_vel)

    @property
    def total_current(self) -> float:
        """Summed current drawn by all windings [A]."""
        return self.count * self.motor.current

    @property
    def per_motor_current(self) -> float:
        return self.motor.current

    @property
    def copper_power(self) -> float:
        """Total I^2R loss in all windings [W]."""
        return self.count * self.motor.thermal_power()

    def reset(self) -> None:
        self.motor.current = 0.0
        self.motor.rot_vel = 0.0

    def validate(self) -> None:
        self.motor.validate()

    # -- characteristics -------------------------------------------------
    def torque_at(self, voltage: float, speed: float) -> float:
        """Group shaft torque at a given voltage and speed (quasi-static) [N*m]."""
        return self.stall_torque * (voltage - self.kv_eff * speed)

    def current_at(self, voltage: float, speed: float) -> float:
        """Total group current at a given voltage and speed (quasi-static) [A]."""
        return (voltage - self.kv_eff * speed) / self.r_eff

    def limited_current(self, current: float) -> float:
        """Clamp current to the configured per-motor limit (scaled to the group)."""
        limit = self.motor.max_current
        if not np.isfinite(limit):
            return float(current)
        total_limit = limit * self.count
        return float(np.clip(current, -total_limit, total_limit))

    def torque_speed_curve(
        self, voltage: float, n: int = 64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(speed_rad_s, torque_Nm, current_A)`` at one applied voltage.

        Torque is clamped at the configured current limit, which caps available
        torque below the ideal linear stall value.
        """
        v_max = self.motor.max_voltage
        v = float(np.clip(voltage, -v_max, v_max))
        omega_max = v / self.kv_eff if self.kv_eff else 0.0
        omega = np.linspace(0.0, omega_max, n)
        current = (v - self.kv_eff * omega) / self.r_eff
        torque = self.kt_eff * current
        limit = self.motor.max_current
        if np.isfinite(limit):
            current = np.clip(current, -limit, limit)
            torque = self.kt_eff * current
        return omega, torque, current
