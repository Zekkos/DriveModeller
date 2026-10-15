"""Wheel and vehicle load models.

All quantities are defined at the *wheel / axle* side (the gearbox output) and
are referred back to the motor by :class:`~drivetrain.gearbox.Gearbox`.

Two load flavours:

``RotationalLoad``
    Inertia disk + viscous drag + constant drag -- the model the original
    scripts used (``J = 1/2 m r^2``, ``k = 2e-4``, ``tau = 0``).

``VehicleLoad``
    A drive pod on the ground.  Translational vehicle mass becomes an equivalent
    rotational inertia at the wheel (``J = m r^2``) and road load is built from
    rolling resistance, aerodynamic drag and grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exceptions import ConfigError


@dataclass
class Wheel:
    """A single wheel / output drum."""

    radius: float = 0.05          # [m]
    mass: float = 1.0             # [kg]
    inertia_factor: float = 0.5   # 0.5 solid disk, 1.0 thin rim, 0 massless

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ConfigError("Wheel radius must be > 0.")
        if self.mass < 0:
            raise ConfigError("Wheel mass must be >= 0.")
        if self.inertia_factor < 0:
            raise ConfigError("Wheel inertia factor must be >= 0.")

    @property
    def inertia(self) -> float:
        """Rotational inertia about the axle [kg*m^2]."""
        return self.inertia_factor * self.mass * self.radius**2

    @property
    def circumference(self) -> float:
        return 2.0 * np.pi * self.radius

    def linear_speed(self, angular_speed: float) -> float:
        """Ground speed for a wheel angular speed: [rad/s] -> [m/s]."""
        return float(angular_speed) * self.radius

    def angular_speed(self, linear_speed: float) -> float:
        """Wheel angular speed needed for a ground speed: [m/s] -> [rad/s]."""
        return float(linear_speed) / self.radius

    def rpm_for_speed(self, linear_speed: float) -> float:
        """Wheel rpm for a ground speed [m/s]."""
        return self.angular_speed(linear_speed) * 60.0 / (2.0 * np.pi)


@dataclass
class RotationalLoad:
    """Pure inertia + viscous + constant-torque load (no ground contact model)."""

    inertia: float = 1.25e-3      # [kg*m^2]  (== 0.5 * 1.0 * 0.05**2)
    viscous: float = 2.0e-4       # [N*m/(rad/s)]
    constant_torque: float = 0.0  # [N*m], always opposes motion

    def inertia_at(self, speed: float = 0.0) -> float:
        return self.inertia

    def torque_at(self, speed: float) -> float:
        """Opposing load torque at a load-side speed (sign opposes motion)."""
        return self.viscous * float(speed) + self.constant_torque * np.sign(speed)

    def with_wheel(self, wheel: Wheel) -> Wheel:
        """Attach a wheel purely so the UI can report ground speed."""
        return wheel


@dataclass
class VehicleLoad:
    """Drive pod carrying a vehicle mass on the ground.

    The vehicle mass appears as an equivalent rotational inertia at the wheel
    (``m r^2`` per driven wheel), which is why acceleration capability scales
    with wheel radius and why that term is gearbox-independent once reflected.
    """

    mass: float = 8.0                    # total mass carried [kg]
    wheel: Wheel = field(default_factory=Wheel)
    wheels_driven: int = 2               # wheels contributing drive torque
    rolling_resistance: float = 0.02     # coefficient (dimensionless)
    drag_cd: float = 0.7                 # Cd * A [m^2]
    air_density: float = 1.2             # [kg/m^3]
    grade_deg: float = 0.0               # slope, positive = uphill
    gravity: float = 9.81
    final_drive_efficiency: float = 1.0  # extra loss in bearings / final drive
    extra_rotational_inertia: float = 0.0  # additional spinning parts [kg*m^2]

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ConfigError("Vehicle mass must be > 0.")
        if self.wheels_driven < 1:
            raise ConfigError("Vehicle needs at least one driven wheel.")
        if not 0.0 < self.final_drive_efficiency <= 1.0:
            raise ConfigError("Final drive efficiency must be in (0, 1].")
        self.wheel.__post_init__()

    # ------------------------------------------------------------------
    @property
    def equivalent_inertia(self) -> float:
        """Translational mass referred to one driven axle + spinning parts."""
        return (
            self.mass * self.wheel.radius**2 * self.wheels_driven
            + self.extra_rotational_inertia
        )

    def inertia_at(self, speed: float = 0.0) -> float:
        return self.equivalent_inertia

    @property
    def grade_rad(self) -> float:
        return np.deg2rad(self.grade_deg)

    def resistance_force(self, linear_speed: float) -> float:
        """Rolling + aero + grade force opposing motion [N] (always >= 0 signed
        with the resistance, may be negative on a steep downhill)."""
        v = abs(float(linear_speed))
        rolling = self.rolling_resistance * self.mass * self.gravity * np.cos(self.grade_rad)
        aero = 0.5 * self.air_density * self.drag_cd * v * v
        grade = self.mass * self.gravity * np.sin(self.grade_rad)
        return rolling + aero + grade

    def torque_at(self, speed: float) -> float:
        """Opposing torque at one driven axle (sign opposes motion) [N*m]."""
        v = self.wheel.linear_speed(speed)
        tau = self.resistance_force(v) * self.wheel.radius / self.wheels_driven
        tau /= max(self.final_drive_efficiency, 1.0e-9)
        return tau * np.sign(speed) if speed != 0.0 else tau

    def linear_speed(self, axle_speed: float) -> float:
        return self.wheel.linear_speed(axle_speed)

    def required_torque_for(self, linear_speed: float, accel: float) -> float:
        """Torque at ONE driven axle to track a ground speed with an acceleration."""
        force = self.resistance_force(linear_speed) + self.mass * accel
        return force * self.wheel.radius / self.wheels_driven


@dataclass
class LoadBundle:
    """Aggregated load at the gearbox output: inertia, viscous, constant."""

    inertia: float = 0.0
    viscous: float = 0.0
    constant_torque: float = 0.0

    def torque_at(self, speed: float) -> float:
        return self.viscous * float(speed) + self.constant_torque * np.sign(speed)

    def inertia_at(self, speed: float = 0.0) -> float:
        return self.inertia
