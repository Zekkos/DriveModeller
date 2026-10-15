"""Configuration / validation errors raised by the drivetrain package."""


class DrivetrainError(ValueError):
    """Base class for all drivetrain configuration errors."""


class ConfigError(DrivetrainError):
    """A parameter set is physically impossible or self-inconsistent."""


class SimulationError(DrivetrainError):
    """The integrator could not complete (instability, divergence, ...)."""
