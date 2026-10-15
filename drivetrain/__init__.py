"""drivetrain -- motor / gearbox / load simulation toolkit."""

from .exceptions import ConfigError, DrivetrainError, SimulationError
from .gearbox import Gearbox, RatioConvention
from .load import LoadBundle, RotationalLoad, VehicleLoad, Wheel
from .motor import MotorGroup, MotorModel, rad_s_to_rpm, rpm_to_rad_s
from .system import DriveSystem

__all__ = [
    "ConfigError", "DrivetrainError", "SimulationError",
    "Gearbox", "RatioConvention",
    "LoadBundle", "RotationalLoad", "VehicleLoad", "Wheel",
    "MotorGroup", "MotorModel", "rad_s_to_rpm", "rpm_to_rad_s",
    "DriveSystem",
]
