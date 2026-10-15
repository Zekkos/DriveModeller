"""Gear stage with two interchangeable ratio conventions.

``ratio`` is stored in **speed-ratio** form: ``output_speed / input_speed``
(equivalently teeth-on-driven / teeth-on-driver):

    ratio < 1   ->  speed reduction, torque multiplication  ("gear down")
    ratio == 1  ->  direct drive
    ratio > 1   ->  speed multiplier, torque reduction      ("gear up")

This is the reciprocal of the convention used by the original ``gearbox.py``
(which divided output speed by ``N``).  Both are accepted: pass
``convention=RatioConvention.GEAR_RATIO`` and the number is normalised on
construction, so old parameter sets keep working.

Efficiency is always applied so that it *reduces* the useful quantity, in the
power-flow direction:

    driving (motor -> load):  output torque = input torque * ratio * eta
    braking (load -> motor):  output torque = input torque * ratio / eta

Reflected-to-input (motor side) formulas used by the solvers:

    J_reflected   = J_load   * ratio**2                  (energy storage, lossless)
    b_reflected   = b_load   * ratio**2 / eta            (dissipative)
    tau_reflected = tau_load * ratio / eta               (dissipative)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .exceptions import ConfigError


class RatioConvention(Enum):
    """How a user-supplied ratio number should be interpreted."""

    SPEED_RATIO = "speed_ratio"   # output/input  -- this module's native form
    GEAR_RATIO = "gear_ratio"     # input/output  -- classical / legacy code


@dataclass
class Gearbox:
    """A single gear stage (or a compound train collapsed to one equivalent stage)."""

    ratio: float = 1.0
    efficiency: float = 1.0
    inertia: float = 0.0        # gear stack inertia referred to the *input* side [kg*m^2]
    damping: float = 0.0        # gear viscous loss referred to the input side    [N*m/(rad/s)]
    convention: RatioConvention = RatioConvention.SPEED_RATIO

    def __post_init__(self) -> None:
        self.ratio = self._normalise_ratio(self.ratio, self.convention)
        # Normalise the stored convention so that dataclasses.replace() (used by
        # the sweep machinery) does not invert the ratio a second time.
        self.convention = RatioConvention.SPEED_RATIO
        self.efficiency = float(self.efficiency)
        self.inertia = float(self.inertia)
        self.damping = float(self.damping)
        self.validate()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_ratio(ratio: float, convention: RatioConvention) -> float:
        ratio = float(ratio)
        if ratio <= 0:
            raise ConfigError("Gear ratio must be > 0 (use ratio=1 for direct drive).")
        if convention is RatioConvention.GEAR_RATIO:
            return 1.0 / ratio
        return ratio

    def validate(self) -> None:
        if not 0.0 < self.efficiency <= 1.0:
            raise ConfigError("Gearbox efficiency must be in (0, 1].")
        if self.inertia < 0:
            raise ConfigError("Gearbox inertia must be >= 0.")
        if self.damping < 0:
            raise ConfigError("Gearbox damping must be >= 0.")

    # ------------------------------------------------------------------
    @property
    def is_reduction(self) -> bool:
        """True when the stage trades speed for torque."""
        return self.ratio < 1.0

    @property
    def gear_ratio(self) -> float:
        """The same stage in classical ``input/output`` form."""
        return 1.0 / self.ratio

    # ------------------------------------------------------------------
    # forward transformations (motor -> load)
    # ------------------------------------------------------------------
    def output_speed(self, input_speed: float) -> float:
        """Load-side speed for a given motor-side speed."""
        return float(input_speed) * self.ratio

    def output_torque(self, input_torque: float) -> float:
        """Torque delivered to the load side for a given motor-side torque."""
        if input_torque >= 0:
            return input_torque * self.ratio * self.efficiency
        # Reverse power flow (coasting / braking): losses oppose motion again.
        return input_torque * self.ratio / self.efficiency

    def input_torque_for_output(self, output_torque: float) -> float:
        """Motor-side torque required to hold a given load-side torque."""
        if output_torque >= 0:
            return output_torque / (self.ratio * self.efficiency)
        return output_torque * self.efficiency / self.ratio

    # ------------------------------------------------------------------
    # reflected transformations (load -> motor)
    # ------------------------------------------------------------------
    def reflect_inertia(self, inertia: float) -> float:
        """Load-side inertia seen from the motor side."""
        return float(inertia) * self.ratio**2

    def reflect_viscous(self, damping: float) -> float:
        """Load-side viscous drag seen from the motor side (losses included).

        A damper dissipates rather than stores energy, so efficiency applies
        here just as it does for a constant load torque.
        """
        return float(damping) * self.ratio**2 / self.efficiency

    def reflect_torque(self, torque: float) -> float:
        """Constant load-side torque seen from the motor side (losses included)."""
        return float(torque) * self.ratio / self.efficiency

    def reflect_inertia_to_output(self, inertia: float) -> float:
        """Motor-side inertia seen from the load side."""
        return float(inertia) / self.ratio**2

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, float]:
        return {
            "speed_ratio": self.ratio,
            "gear_ratio": self.gear_ratio,
            "efficiency": self.efficiency,
            "input_inertia": self.inertia,
            "input_damping": self.damping,
        }
