"""Assembly of motor group -> gearbox -> load into one shaft-level system.

:class:`DriveSystem` reduces the whole drivetrain to the motor (input) shaft.
Damping is deliberately split in two so nothing is counted twice:

``base_damping``
    Motor bearings + the electromechanical term ``kt*kv/R`` + gear-stage loss.
    Independent of the load.

``total_damping``
    ``base_damping`` plus the load's viscous contribution linearised about the
    nominal speed.  Used for analytic reporting (time constants, no-load speed).

The solvers instead evaluate the load directly with
:meth:`DriveSystem.reflected_load_torque` at every sample, which is what lets
quadratic aerodynamic drag and Coulomb friction work at all speeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .exceptions import ConfigError
from .gearbox import Gearbox
from .load import LoadBundle, RotationalLoad, VehicleLoad, Wheel
from .motor import MotorGroup, MotorModel, rad_s_to_rpm

LoadType = RotationalLoad | VehicleLoad


@dataclass(frozen=True)
class LoadCoefficients:
    """Reflected load torque fitted to a polynomial in motor speed.

        tau_load(w) = sign(w) * (static + viscous*|w| + quadratic*w**2)

    This is *exact* for every load model in this package: constant Coulomb /
    rolling / grade terms, viscous damping, and quadratic aerodynamic drag.  The
    fit is verified against the real model at a speed it was not fitted at, so a
    custom load that does not fit is reported via :attr:`exact` and callers can
    fall back to evaluating the model directly.

    Having the load in this form is what makes the fast paths possible: the
    integrators do scalar arithmetic per step instead of walking the load object
    graph, and the equilibrium speed becomes a closed-form quadratic instead of
    an iteration that can diverge.
    """

    static: float
    viscous: float
    quadratic: float
    exact: bool = True

    def torque(self, w: float) -> float:
        """Opposing load torque at motor speed ``w`` [N*m]."""
        if w > 0.0:
            return self.static + self.viscous * w + self.quadratic * w * w
        if w < 0.0:
            return -(self.static - self.viscous * w + self.quadratic * w * w)
        return self.static

    def damping(self, w: float) -> float:
        """Secant viscous coefficient at ``w`` (static term excluded)."""
        return self.viscous + self.quadratic * abs(w)

    def equilibrium(self, drive_torque: float, extra_damping: float = 0.0) -> float:
        """Speed where ``drive_torque`` balances this load.  Always >= 0.

        ``extra_damping`` adds speed-proportional damping that is not part of the
        load itself -- electromechanical damping for a voltage-driven motor, or
        the gearbox/shaft viscous terms.

        Solved in closed form.  A fixed-point iteration on this equation does not
        converge when the load is mostly quadratic (the classic aero case), which
        silently returned nonsense equilibrium speeds for any preset with drag.
        """
        a = self.quadratic
        b = self.viscous + float(extra_damping)
        c = self.static - float(drive_torque)
        if c >= 0.0:
            # the load cannot even be started by this torque
            return 0.0
        if a > 0.0:
            disc = b * b - 4.0 * a * c          # c < 0, so disc > 0 always
            # numerically stable form of (-b + sqrt(disc)) / (2a)
            return float(-2.0 * c / (b + math.sqrt(disc)))
        if b > 0.0:
            return float(-c / b)
        # nothing speed-dependent opposes the torque: no finite equilibrium
        return float("inf")


@dataclass
class DriveSystem:
    """motor group + gearbox + load, normalised to the motor (input) shaft."""

    group: MotorGroup = field(default_factory=lambda: MotorGroup(MotorModel()))
    gearbox: Gearbox = field(default_factory=lambda: Gearbox(1.0, 1.0))
    load: LoadType = field(default_factory=RotationalLoad)

    def __post_init__(self) -> None:
        self.group.validate()
        self.gearbox.validate()

    # ------------------------------------------------------------------
    # reflected quantities (all at the motor shaft)
    # ------------------------------------------------------------------
    @property
    def electromechanical_damping(self) -> float:
        """Damping produced by the shorted winding: ``kt*kv/R`` [N*m/(rad/s)].

        This is *not* a bearing loss -- it appears because back-EMF drives a
        braking current through the winding resistance.  The solvers get it for
        free from the ``kt * i`` term, so it must NOT be included in
        :attr:`base_damping` (that would count it twice).
        """
        return self.group.kt_eff * self.group.kv_eff / self.group.r_eff

    @property
    def base_damping(self) -> float:
        """Mechanical damping only: motor bearings + gear stage, load excluded."""
        return self.group.bearing_damping_eff + self.gearbox.damping

    def reflected_load_torque(self, motor_speed: float) -> float:
        """Load torque opposing motion, referred to the motor shaft [N*m].

        Evaluated at the true load-side speed, so quadratic drag and Coulomb
        friction are handled exactly rather than linearised.
        """
        output_speed = motor_speed * self.gearbox.ratio
        return self.gearbox.reflect_torque(self.load.torque_at(output_speed))

    def load_coefficients(self) -> LoadCoefficients:
        """Fit the reflected load torque as a polynomial in speed.

        The static term comes from :meth:`load_static_torque` (exact); the viscous
        and quadratic terms are recovered from two speed samples, which determines
        them uniquely for a ``const + linear + quadratic`` load.
        """
        f = self.reflected_load_torque
        static = float(self.load_static_torque())
        scale = max(abs(self.no_load_speed), 10.0)
        w1, w2 = scale * 0.1, scale * 0.4
        g1, g2 = float(f(w1)) - static, float(f(w2)) - static
        quadratic = (g2 - g1 * w2 / w1) / (w2 * w2 - w1 * w2)
        viscous = g1 / w1 - quadratic * w1
        if quadratic < 0.0:
            quadratic = 0.0

        w3 = scale * 0.25                       # held-out verification speed
        pred = static + viscous * w3 + quadratic * w3 * w3
        actual = float(f(w3))
        exact = abs(pred - actual) <= 1e-6 * max(abs(actual), abs(static), 1e-12)
        return LoadCoefficients(
            static=static, viscous=max(viscous, 0.0), quadratic=quadratic, exact=exact
        )

    def load_damping_at(self, motor_speed: float) -> float:
        """Secant viscous coefficient of the load at a motor speed [N*m/(rad/s)].

        The speed-independent part (rolling resistance, grade, constant drag) is
        removed first, so this is purely the speed-proportional remainder.  For
        quadratic aerodynamic drag the coefficient grows linearly with speed.
        """
        lc = self.load_coefficients()
        if lc.exact:
            return float(lc.damping(motor_speed))
        w = max(abs(motor_speed), 1e-9)
        tau = self.reflected_load_torque(w) - self.load_static_torque()
        coeff = tau / w
        return float(coeff) if np.isfinite(coeff) and coeff > 0 else 0.0

    def load_static_torque(self) -> float:
        """Speed-independent part of the reflected load torque [N*m]."""
        load = self.load
        if isinstance(load, RotationalLoad):
            return self.gearbox.reflect_torque(load.constant_torque)
        # VehicleLoad: rolling resistance + grade are speed independent.
        rolling = load.rolling_resistance * load.mass * load.gravity * np.cos(load.grade_rad)
        grade = load.mass * load.gravity * np.sin(load.grade_rad)
        tau = (rolling + grade) * load.wheel.radius / load.wheels_driven
        return self.gearbox.reflect_torque(tau / max(load.final_drive_efficiency, 1e-9))

    def bundle(self) -> LoadBundle:
        """Load expressed at the motor shaft as (inertia, viscous, constant)."""
        nominal = self._nominal_speed()
        return LoadBundle(
            inertia=self.gearbox.reflect_inertia(self.load.inertia_at(0.0)),
            viscous=self.load_damping_at(nominal),
            constant_torque=self.load_static_torque(),
        )

    def _nominal_speed(self) -> float:
        """Linearisation point for reporting: half of the loaded steady speed."""
        v = self.voltage_limit
        if not np.isfinite(v):
            return 100.0
        w = self.steady_speed(v)
        return max(w * 0.5, 1.0) if np.isfinite(w) else 100.0

    @property
    def total_inertia(self) -> float:
        """Motor rotors + gear stage + reflected load inertia [kg*m^2]."""
        return (
            self.group.inertia_eff
            + self.gearbox.inertia
            + self.gearbox.reflect_inertia(self.load.inertia_at(0.0))
        )

    @property
    def settle_time(self) -> float:
        """Rough 4-tau settling time [s] -- how long to simulate for a steady value."""
        return 4.0 * self.mech_time_constant

    @property
    def total_damping(self) -> float:
        """Bearings + electromechanical + load viscous, at the nominal speed."""
        return (
            self.base_damping
            + self.electromechanical_damping
            + self.load_damping_at(self._nominal_speed())
        )

    @property
    def stall_torque(self) -> float:
        """Locked-rotor torque at the motor shaft per volt [N*m/V]."""
        return self.group.stall_torque

    @property
    def voltage_limit(self) -> float:
        return self.group.motor.max_voltage

    @property
    def no_load_speed(self) -> float:
        """Ideal (damping-free) no-load speed at the voltage limit [rad/s]."""
        v = self.voltage_limit
        return v * self.group.no_load_speed_per_volt if np.isfinite(v) else np.inf

    @property
    def mech_time_constant(self) -> float:
        """``J_total / b_total`` [s] -- the speed-loop pole."""
        b = self.total_damping
        return self.total_inertia / b if b > 0 else np.inf

    @property
    def electrical_time_constant(self) -> float:
        return self.group.l_eff / self.group.r_eff if self.group.r_eff else 0.0

    @property
    def is_second_order(self) -> bool:
        return self.group.motor.inductance > 0

    @property
    def algebraic_current(self) -> bool:
        """True when winding inductance is zero, i.e. current is algebraic."""
        return self.group.motor.inductance <= 0

    @property
    def wheel(self) -> Wheel | None:
        """Output wheel, when the load model has one (used for ground speed)."""
        return getattr(self.load, "wheel", None)

    # ------------------------------------------------------------------
    # steady state
    # ------------------------------------------------------------------
    def steady_speed(self, voltage: float) -> float:
        """Motor-shaft equilibrium speed for a voltage [rad/s].

        Solved in closed form from the fitted load polynomial.  The balance is

            quadratic*w^2 + (viscous + extra_damping)*w + static = drive_torque

        which is a quadratic with exactly one non-negative root, so there is no
        iteration to fail to converge.

        The current limit is honoured: if the unconstrained equilibrium would draw
        more than ``max_current``, the drive is a current source at equilibrium and
        the speed settles where ``kt * i_limit`` balances the load instead (there
        is no back-EMF damping in that mode, so ``extra_damping`` drops to just the
        mechanical base damping).
        """
        g = self.group
        v = float(voltage)
        lc = self.load_coefficients()
        kt, kv, R = g.kt_eff, g.kv_eff, g.r_eff
        em = kt * kv / R

        # --- unconstrained (voltage-driven) equilibrium --------------------
        w = lc.equilibrium(kt * v / R, extra_damping=em + self.base_damping)

        limit = self.current_limit_total
        if not np.isfinite(limit) or not np.isfinite(w):
            return float(math.copysign(w, v) if v else w)

        # --- is that equilibrium reachable inside the current limit? -------
        if abs((v - kv * w) / R) <= limit:
            return float(math.copysign(w, v) if v else w)

        # --- current-limited equilibrium ----------------------------------
        drive = kt * (limit if v >= 0 else -limit)
        w = lc.equilibrium(drive, extra_damping=self.base_damping)
        return float(math.copysign(w, v) if v else w)

    @property
    def current_limit_total(self) -> float:
        """Group-level current ceiling [A] (``max_current`` is per motor)."""
        per = self.group.motor.max_current
        return np.inf if not np.isfinite(per) else per * self.group.count

    def steady_mode(self, voltage: float) -> str:
        """``"stalled"`, ``"current_limited"``, ``"voltage_limited"`` or ``"free"``."""
        if not np.isfinite(self.current_limit_total):
            return "free"
        kt, kv, R = self.group.kt_eff, self.group.kv_eff, self.group.r_eff
        w = abs(self.steady_speed(voltage))
        if w == 0.0:
            return "stalled"
        if not np.isfinite(w):
            return "free"
        if abs((voltage - kv * w) / R) > self.current_limit_total * 0.999:
            return "current_limited"
        return "voltage_limited"

    def steady_output_speed(self, voltage: float) -> float:
        return self.gearbox.output_speed(self.steady_speed(voltage))

    def steady_ground_speed(self, voltage: float) -> float:
        wheel = self.wheel
        if wheel is None:
            return np.nan
        return wheel.linear_speed(self.steady_output_speed(voltage))

    def steady_output_torque(self, voltage: float) -> float:
        """Gross torque available at the gearbox output at stall [N*m]."""
        return self.gearbox.output_torque(self.stall_torque * voltage)

    def operating_point(self, voltage: float, output_speed: float) -> dict[str, float]:
        """Torque / current / power triple at a commanded output speed."""
        in_speed = output_speed / self.gearbox.ratio
        torque = self.group.torque_at(voltage, in_speed)
        current = self.group.current_at(voltage, in_speed)
        limit = self.group.motor.max_current * self.group.count
        if np.isfinite(limit):
            current = float(np.clip(current, -limit, limit))
            torque = self.group.kt_eff * current
        p_in = voltage * current
        p_out = torque * output_speed * self.gearbox.efficiency
        return {
            "motor_speed_rad_s": in_speed,
            "motor_speed_rpm": rad_s_to_rpm(in_speed),
            "output_speed_rad_s": output_speed,
            "output_speed_rpm": rad_s_to_rpm(output_speed),
            "torque_motor_Nm": torque,
            "torque_output_Nm": self.gearbox.output_torque(torque),
            "current_total_A": current,
            "current_per_motor_A": current / self.group.count,
            "power_in_W": p_in,
            "power_out_W": p_out,
            "copper_loss_W": current**2 * self.group.r_eff,
            "efficiency": p_out / p_in if abs(p_in) > 1e-12 else 0.0,
        }

    # ------------------------------------------------------------------
    # state-space form
    # ------------------------------------------------------------------
    def state_matrices(
        self, damping: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``A``/``B`` for ``dx/dt = A x + B u``.

        With ``inductance == 0`` the system is first order: the state is
        ``x = [speed]`` and ``u = [voltage, tau_static]``, obtained by
        substituting the algebraic current ``i = (V - kv*omega)/R`` into the
        mechanical equation.

        With ``inductance > 0`` the state is ``x = [current, speed]`` (second
        order) and ``u = [voltage, tau_static]`` as well.

        ``damping`` overrides :attr:`total_damping`; the solvers pass
        ``base_damping`` plus a freshly evaluated load coefficient.
        """
        g = self.group
        J = self.total_inertia
        b = self.total_damping if damping is None else damping

        if self.algebraic_current:
            A = np.array([[-(g.kt_eff * g.kv_eff / g.r_eff + b) / J]])
            B = np.array([[g.kt_eff / g.r_eff / J, -1.0 / J]])
            return A, B

        L, R = g.l_eff, g.r_eff
        A = np.array([
            [-R / L, -g.kv_eff / L],
            [g.kt_eff / J, -b / J],
        ])
        B = np.array([
            [1.0 / L, 0.0],
            [0.0, -1.0 / J],
        ])
        return A, B

    def initial_state(self) -> np.ndarray:
        """State vector matching the current object state."""
        if self.algebraic_current:
            return np.array([self.group.rot_vel], float)
        return np.array([self.group.total_current, self.group.rot_vel], float)

    def copy(self) -> DriveSystem:
        return replace(
            self,
            group=replace(self.group, motor=replace(self.group.motor)),
            gearbox=replace(self.gearbox),
            load=replace(self.load),
        )

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, float]:
        b = self.bundle()
        return {
            "motors": self.group.count,
            "stall_torque_per_volt": self.stall_torque,
            "stall_torque_at_limit": self.stall_torque * (
                self.voltage_limit if np.isfinite(self.voltage_limit) else 1.0
            ),
            "no_load_rpm": rad_s_to_rpm(self.no_load_speed),
            "steady_rpm_at_limit": rad_s_to_rpm(self.steady_speed(
                self.voltage_limit if np.isfinite(self.voltage_limit) else 0.0)),
            "total_inertia": self.total_inertia,
            "base_damping": self.base_damping,
            "electromechanical_damping": self.electromechanical_damping,
            "total_damping": self.total_damping,
            "reflected_inertia": b.inertia,
            "reflected_viscous": b.viscous,
            "reflected_static_torque": b.constant_torque,
            "mech_time_constant": self.mech_time_constant,
            "settle_time": self.settle_time,
            "steady_mode": self.steady_mode(
                self.voltage_limit if np.isfinite(self.voltage_limit) else 0.0),
            "current_limit_total": self.current_limit_total,
            "electrical_time_constant": self.electrical_time_constant,
            "speed_ratio": self.gearbox.ratio,
            "efficiency": self.gearbox.efficiency,
        }
