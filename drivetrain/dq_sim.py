"""Time-domain simulation of the dq (field-oriented) plant.

``drivetrain.simulate`` integrates the *brushed* view: one current, one voltage,
back-EMF against torque.  That is the right state for a brushed drive, and for a
three-phase machine run at ``i_d = 0``, but it cannot represent what a field
oriented drive actually does, because a FOC drive commands two currents and the
current *angle* carries information the single-axis model throws away: the
d-axis inductance, the reluctance torque, and the voltage a d-axis current costs
or saves.

This module integrates the same three states the generated dq code integrates
(``i_d``, ``i_q``, shaft speed) plus the rotor angle, against the *drivetrain*
mechanical model -- gearbox, wheels, rolling resistance, drag -- so a dq
simulation and a brushed simulation of the same vehicle differ only in the
machine model, never in the drivetrain.  The mechanical side reuses
``_FrozenModel`` from ``drivetrain.simulate`` rather than reimplementing it:
inertia, reflected load, bearing damping and shaft stiction come from exactly
the code the brushed engines use.

The electrical arithmetic is a transliteration of the generated dq module
(``drivetrain.export_dq``), operation for operation, so the library and the
firmware code agree; ``tests/test_dq_sim.py`` pins that.

What is NOT modelled is the same list as in ``docs/pmsm.md``: no iron
saturation, no demagnetisation, no inverter non-linearity, no sensorless
estimation.  The current limit is deliberately NOT applied to the state either:
a logged voltage vector is what the inverter actually applied, and the current
that follows is whatever the machine allows.  If a drive current-limits, its log
already shows the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .exceptions import ConfigError
from .motor import rad_s_to_rpm
from .pmsm import PMSMModel
from .simulate import _FrozenModel, _slow_load
from .system import DriveSystem

# RK4's absolute-stability region reaches |lambda*h| ~ 2.78 on the imaginary
# axis; half of that is the working limit.  The dq export imports this same
# number, so the library and the generated code cannot drift apart on the one
# constant that decides whether a control period is legal.
RK4_STABILITY_LIMIT = 1.39

# The band inside which the shaft counts as stopped, for stiction.  Matches the
# 1e-12 that simulate_rk4.net_torque uses, so the dq and brushed engines agree
# about when the shaft is standing still.
_STOP_BAND = 1.0e-12

DQ_ENGINES = ("euler", "rk4")


@dataclass(frozen=True)
class DqState:
    """The four states of the dq plant.

    ``angle`` is the *mechanical* rotor angle in radians, integrated from the
    speed and left unwrapped.  The dynamics do not depend on it -- the rotor
    frame is what makes the dq equations steady -- but a logged encoder does,
    and comparing the two is how a pole-pair count gets checked.
    """

    i_d: float = 0.0
    i_q: float = 0.0
    rot_vel: float = 0.0
    angle: float = 0.0


@dataclass(frozen=True)
class DqRates:
    """Time derivatives of :class:`DqState`."""

    di_d: float = 0.0
    di_q: float = 0.0
    dw: float = 0.0
    dtheta: float = 0.0


@dataclass(frozen=True)
class _Mechanical:
    """The drivetrain as the dq plant sees it: inertia, damping, stiction, load.

    Built from ``_FrozenModel`` so the numbers are the same ones the brushed
    engines use.  ``count`` machines each produce their own electromagnetic
    torque, so the group torque is ``count * tau_em``.
    """

    system: DriveSystem
    inertia: float
    damping: float
    static: float
    count: int
    frozen: _FrozenModel
    extra_load: float = 0.0

    @classmethod
    def build(cls, system: DriveSystem, extra_load: float = 0.0) -> "_Mechanical":
        return cls(system=system,
                   inertia=float(system.total_inertia),
                   damping=float(system.base_damping),
                   static=float(system.group.static_friction_eff),
                   count=int(system.group.count),
                   frozen=_FrozenModel.build(system),
                   extra_load=float(extra_load))

    def load(self, rot_vel: float) -> float:
        """Reflected load torque at the motor shaft, always opposing motion."""
        base = (self.frozen.load(rot_vel) if self.frozen.exact_fit
                else _slow_load(self.system, rot_vel))
        if not self.extra_load:
            return base
        if rot_vel > 0.0:
            return base + self.extra_load
        if rot_vel < 0.0:
            return base - self.extra_load
        return base

    def net_torque(self, tau_em_total: float, rot_vel: float) -> float:
        """Shaft torque after bearings, stiction and load -- as simulate.py does."""
        tau = tau_em_total - self.damping * rot_vel - self.load(rot_vel)
        if rot_vel > _STOP_BAND or rot_vel < -_STOP_BAND:
            return tau - (self.static if rot_vel > 0 else -self.static)
        if abs(tau) <= abs(self.static):          # stiction can hold the shaft
            return 0.0
        return tau - (self.static if tau > 0 else -self.static)


# -- machine-side helpers ----------------------------------------------------

def machine_of(system: DriveSystem) -> PMSMModel:
    """The brushless machine in *system*, or a ConfigError naming the brushed one."""
    motor = system.group.motor
    if not isinstance(motor, PMSMModel):
        raise ConfigError(
            f"the dq simulation needs a brushless machine; this system has a "
            f"{type(motor).__name__}. Simulate the brushed model with "
            f"drivetrain.simulate(), or get a brushless system with "
            f"presets.build('bldc_pod')."
        )
    return motor


def electrical_time_constant(spec: PMSMModel) -> float:
    """``min(Ld, Lq) / R`` -- the fast pole of the current loop."""
    return min(spec.inductance_d, spec.inductance_q) / spec.resistance


def electrical_speed(spec: PMSMModel, rot_vel: float) -> float:
    return spec.pole_pairs * rot_vel


def pole_damping(spec: PMSMModel) -> float:
    """Real part of the dq current-loop pole with the shaft speed frozen.

    With ``w`` held constant the dq equations linearise to a complex conjugate
    pair at ``-R(Ld+Lq)/(2 Ld Lq) +/- j w_e``.  The real part is what RK4's
    damping tolerance is about and the imaginary part grows with speed, which is
    why the stable step shrinks as the machine speeds up.
    """
    ld, lq, r = spec.inductance_d, spec.inductance_q, spec.resistance
    return r * (ld + lq) / (2.0 * ld * lq)


def pole_magnitude(spec: PMSMModel, rot_vel: float) -> float:
    """|pole| at *rot_vel*; RK4 needs ``|pole| * dt`` inside its stability region."""
    return math.hypot(pole_damping(spec), electrical_speed(spec, rot_vel))


def max_step_time(spec: PMSMModel, rot_vel: float = 0.0) -> float:
    """Largest stable RK4 step at *rot_vel*.

    Two limits, whichever binds: half the electrical time constant (accuracy on
    the fast exponential) and the RK4 stability limit over the pole magnitude
    (which tightens with speed).  On ``bldc_pod`` the first binds at standstill
    (6.25e-4 s) and the second everywhere above ~250 rad/s (1.80e-4 s at base
    speed, 9.0e-5 s at twice base).
    """
    return min(0.5 * electrical_time_constant(spec),
               RK4_STABILITY_LIMIT / pole_magnitude(spec, rot_vel))


def sub_steps(spec: PMSMModel, dt: float, rot_vel: float = 0.0) -> int:
    """RK4 sub-steps a control period of *dt* needs at *rot_vel* (>= 1)."""
    limit = max_step_time(spec, rot_vel)
    if limit <= 0.0 or dt <= limit:
        return 1
    return int(math.ceil(dt / limit))


def electrical_derivatives(spec: PMSMModel, state: DqState,
                           v_d: float, v_q: float) -> tuple[float, float]:
    """(di_d/dt, di_q/dt) under applied phase voltages ``v_d``, ``v_q``.

    Written in the same order as the generated dq module, because these are the
    equations a fitted machine model is judged against:

        v_d = R i_d + Ld di_d/dt - w_e Lq i_q
        v_q = R i_q + Lq di_q/dt + w_e (Ld i_d + lambda)
    """
    we = electrical_speed(spec, state.rot_vel)
    di_d = (v_d - spec.resistance * state.i_d
            + we * spec.inductance_q * state.i_q) / spec.inductance_d
    di_q = (v_q - spec.resistance * state.i_q
            - we * (spec.inductance_d * state.i_d + spec.flux_linkage)) / spec.inductance_q
    return di_d, di_q


def torque_em(spec: PMSMModel, state: DqState) -> float:
    """Electromagnetic torque of ONE machine for this current pair."""
    return spec.torque_at(state.i_d, state.i_q)


def derivatives(system: DriveSystem, mech: _Mechanical, state: DqState,
                v_d: float, v_q: float) -> DqRates:
    """Full state derivatives: machine equations plus the drivetrain."""
    spec = machine_of(system).spec
    di_d, di_q = electrical_derivatives(spec, state, v_d, v_q)
    dw = mech.net_torque(mech.count * torque_em(spec, state), state.rot_vel) / mech.inertia
    return DqRates(di_d, di_q, dw, state.rot_vel)


# -- integration -------------------------------------------------------------

def step(system: DriveSystem, state: DqState, v_d: float, v_q: float, dt: float,
         mech: _Mechanical | None = None, engine: str = "rk4") -> DqState:
    """Advance one step.  Raises past the stable step rather than diverging.

    ``dt`` must be <= ``max_step_time(spec, state.rot_vel)``, which shrinks as
    the machine speeds up.  A control period is normally longer than that even
    at standstill, so use :func:`integrate` for a sampled loop.  Refusing here
    is deliberate: a silently unstable step shows up as a simulation that
    diverges in the field, far from the mistake that caused it.
    """
    if engine not in DQ_ENGINES:
        raise ConfigError(f"unknown dq engine {engine!r}; choose from {list(DQ_ENGINES)}")
    spec = machine_of(system).spec
    mech = mech if mech is not None else _Mechanical.build(system)
    limit = max_step_time(spec, state.rot_vel)
    if dt > limit:
        raise ConfigError(
            f"dt {dt:g} s exceeds the stable step {limit:g} s at "
            f"{state.rot_vel:g} rad/s (pole magnitude "
            f"{pole_magnitude(spec, state.rot_vel):g} 1/s, electrical time "
            f"constant {electrical_time_constant(spec):g} s); subdivide the step"
        )
    if engine == "euler":
        r = derivatives(system, mech, state, v_d, v_q)
        return DqState(state.i_d + dt * r.di_d, state.i_q + dt * r.di_q,
                       state.rot_vel + dt * r.dw, state.angle + dt * r.dtheta)
    r1 = derivatives(system, mech, state, v_d, v_q)
    s2 = DqState(state.i_d + 0.5 * dt * r1.di_d, state.i_q + 0.5 * dt * r1.di_q,
                 state.rot_vel + 0.5 * dt * r1.dw, state.angle + 0.5 * dt * r1.dtheta)
    r2 = derivatives(system, mech, s2, v_d, v_q)
    s3 = DqState(state.i_d + 0.5 * dt * r2.di_d, state.i_q + 0.5 * dt * r2.di_q,
                 state.rot_vel + 0.5 * dt * r2.dw, state.angle + 0.5 * dt * r2.dtheta)
    r3 = derivatives(system, mech, s3, v_d, v_q)
    s4 = DqState(state.i_d + dt * r3.di_d, state.i_q + dt * r3.di_q,
                 state.rot_vel + dt * r3.dw, state.angle + dt * r3.dtheta)
    r4 = derivatives(system, mech, s4, v_d, v_q)
    sixth = dt / 6.0
    return DqState(
        state.i_d + sixth * (r1.di_d + 2.0 * r2.di_d + 2.0 * r3.di_d + r4.di_d),
        state.i_q + sixth * (r1.di_q + 2.0 * r2.di_q + 2.0 * r3.di_q + r4.di_q),
        state.rot_vel + sixth * (r1.dw + 2.0 * r2.dw + 2.0 * r3.dw + r4.dw),
        state.angle + sixth * (r1.dtheta + 2.0 * r2.dtheta + 2.0 * r3.dtheta + r4.dtheta),
    )


def integrate(system: DriveSystem, state: DqState, v_d: float, v_q: float,
              dt: float, mech: _Mechanical | None = None) -> DqState:
    """Advance one control period, sub-divided until every RK4 step is stable.

    The voltage is held across the period, which is what a sampled current loop
    does between updates.  The sub-step is recomputed from the speed reached,
    because the stable step shrinks with speed: a fixed subdivision sized for
    standstill is exactly the bug that makes a high-speed run diverge.
    """
    spec = machine_of(system).spec
    mech = mech if mech is not None else _Mechanical.build(system)
    remaining = dt
    guard = 0
    while remaining > 0.0:
        h = min(remaining, max_step_time(spec, state.rot_vel))
        if h <= 0.0:
            raise ConfigError("this machine has a zero stable step; check the inductances")
        state = step(system, state, v_d, v_q, h, mech=mech)
        remaining -= h
        guard += 1
        if guard > 100000:
            raise ConfigError("integrate() took 100000 sub-steps for one period; "
                              "the step contract and the speed are inconsistent")
    return state


# -- Park transform ---------------------------------------------------------

_TWO_PI_OVER_3 = 2.0 * math.pi / 3.0


def park(angle_electrical: float, a: float, b: float, c: float) -> tuple[float, float]:
    """Amplitude-invariant Park transform of three phase currents or voltages.

    ``d = (2/3)(a cos th + b cos(th - 2pi/3) + c cos(th + 2pi/3))`` and the
    matching quadrature, with the 2/3 chosen so that a balanced sine of
    amplitude ``I`` at angle ``th`` maps to ``i_d = I, i_q = 0`` -- the dq
    quantities are phase *amplitudes*, not RMS, which is the convention every dq
    model here assumes (torque is ``1.5 p lambda i_q`` for the same reason).

    A drive that reports RMS is off by ``sqrt(2)``; the fit says so when its
    residuals look like that.
    """
    cos_b = math.cos(angle_electrical - _TWO_PI_OVER_3)
    cos_c = math.cos(angle_electrical + _TWO_PI_OVER_3)
    sin_b = math.sin(angle_electrical - _TWO_PI_OVER_3)
    sin_c = math.sin(angle_electrical + _TWO_PI_OVER_3)
    d = (2.0 / 3.0) * (a * math.cos(angle_electrical) + b * cos_b + c * cos_c)
    q = -(2.0 / 3.0) * (a * math.sin(angle_electrical) + b * sin_b + c * sin_c)
    return d, q


def park_arrays(
    angle_electrical,
    a,
    b,
    c,
):
    """Vectorised :func:`park` over numpy arrays of the same length.

    Same arithmetic, same 2/3, same sign on the quadrature as :func:`park`, so a
    log transformed this way and a trajectory computed with the scalar version sit
    on the same convention.  NaNs pass through.
    """
    th = np.asarray(angle_electrical, dtype=float)
    ia = np.asarray(a, dtype=float)
    ib = np.asarray(b, dtype=float)
    ic = np.asarray(c, dtype=float)
    d = (2.0 / 3.0) * (
        ia * np.cos(th) + ib * np.cos(th - _TWO_PI_OVER_3) + ic * np.cos(th + _TWO_PI_OVER_3)
    )
    q = -(2.0 / 3.0) * (
        ia * np.sin(th) + ib * np.sin(th - _TWO_PI_OVER_3) + ic * np.sin(th + _TWO_PI_OVER_3)
    )
    return d, q


def inverse_park(angle_electrical: float, d: float, q: float) -> tuple[float, float, float]:
    """Inverse of :func:`park` for a balanced three-phase set (zero sequence 0)."""
    a = d * math.cos(angle_electrical) - q * math.sin(angle_electrical)
    b = d * math.cos(angle_electrical - _TWO_PI_OVER_3) \
        - q * math.sin(angle_electrical - _TWO_PI_OVER_3)
    c = d * math.cos(angle_electrical + _TWO_PI_OVER_3) \
        - q * math.sin(angle_electrical + _TWO_PI_OVER_3)
    return a, b, c


def zero_sequence(a: float, b: float, c: float) -> float:
    """ ``(a + b + c) / 3`` -- what the Park transform throws away."""
    return (a + b + c) / 3.0


# -- simulation result ------------------------------------------------------

@dataclass
class DqSimulationResult:
    """Sampled dq trajectory, shaped like ``simulate.SimulationResult``."""

    t: np.ndarray
    i_d: np.ndarray
    i_q: np.ndarray
    motor_speed: np.ndarray
    torque: np.ndarray
    v_d: np.ndarray
    v_q: np.ndarray
    angle: np.ndarray
    system: DriveSystem
    engine: str
    dt: float
    load_torque: float = 0.0

    @property
    def current(self) -> np.ndarray:
        """Phase current amplitude, ``hypot(i_d, i_q)`` -- what a drive reports."""
        return np.hypot(self.i_d, self.i_q)

    @property
    def voltage(self) -> np.ndarray:
        """Applied phase-voltage amplitude, ``hypot(v_d, v_q)``."""
        return np.hypot(self.v_d, self.v_q)

    def channels(self) -> dict[str, np.ndarray]:
        """Every channel a dataset can name, under the names fit.py accepts."""
        gb = self.system.gearbox
        spec = machine_of(self.system).spec
        count = self.system.group.count
        out_speed = self.motor_speed * gb.ratio
        copper = count * 1.5 * spec.resistance * (self.i_d ** 2 + self.i_q ** 2)
        # Motor-shaft torque at motor-shaft speed: the gear ratio cancels across
        # an ideal reducer, so this is the mechanical power at the output shaft
        # after the gearbox loss.  See drivetrain.simulate._enrich.
        converted = self.torque * self.motor_speed * gb.efficiency
        with np.errstate(divide="ignore", invalid="ignore"):
            drawn = count * 1.5 * (self.v_d * self.i_d + self.v_q * self.i_q)
            eff = np.where(np.abs(drawn) > 1e-12, converted / drawn, 0.0)
        data = {
            "time_s": self.t,
            "current_d_A": self.i_d,
            "current_q_A": self.i_q,
            "current_A": self.current,
            "voltage_d_V": self.v_d,
            "voltage_q_V": self.v_q,
            "voltage_V": self.voltage,
            "motor_rad_s": self.motor_speed,
            "motor_rpm": rad_s_to_rpm(self.motor_speed),
            "output_rad_s": out_speed,
            "output_rpm": rad_s_to_rpm(out_speed),
            "torque_Nm": self.torque,
            "rotor_angle_rad": self.angle,
            "power_in_W": drawn,
            "copper_loss_W": copper,
            "power_out_W": converted,
            "efficiency": np.clip(eff, 0.0, 1.0),
        }
        wheel = self.system.wheel
        if wheel is not None:
            data["speed_m_s"] = out_speed * wheel.radius
            data["speed_kmh"] = out_speed * wheel.radius * 3.6
        return data

    def final(self, name: str) -> float:
        return float(self.channels()[name][-1])

    def peak(self, name: str) -> float:
        return float(np.max(np.abs(self.channels()[name])))

    def metrics(self, channel: str = "motor_rad_s", target: float | None = None,
                tol: float = 0.02) -> dict[str, float]:
        """Settling metrics, like ``SimulationResult.metrics`` but by channel name."""
        ch = self.channels()[channel]
        out = {f"final_{channel}": float(ch[-1]),
               f"peak_{channel}": float(np.max(np.abs(ch)))}
        if target is not None:
            band = abs(target) * tol
            inside = np.abs(ch - target) <= band
            out["settling_time_s"] = 0.0 if bool(inside.all()) else \
                float(self.t[int(np.argmax(~inside[::-1]))])
            out["steady_state_error"] = float(ch[-1] - target)
        return out


# -- the entry point --------------------------------------------------------

def solver_dt(system: DriveSystem, spec: PMSMModel, dt: float | None,
              top_speed: float | None = None) -> float:
    """A default integration step that is still stable at the *fastest* speed.

    Sizing the step from the electrical time constant alone is the classic way
    to write a run that diverges once the machine speeds up: on ``bldc_pod`` the
    stable step is 6.25e-4 s at standstill and 1.80e-4 s at base speed.  So the
    default is sized against a speed the machine might reach, not the one it
    starts at, and with a factor of two of margin on top.
    """
    if dt is not None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ConfigError("the dq integration step must be a positive, finite number")
        return float(dt)
    if top_speed is None:
        w_ref = float(system.no_load_speed)
    else:
        w_ref = abs(float(top_speed))
    if not math.isfinite(w_ref):
        w_ref = spec.voltage_limit / (spec.pole_pairs * spec.flux_linkage)
    return min(0.25 * electrical_time_constant(spec),
               0.5 * RK4_STABILITY_LIMIT / pole_magnitude(spec, w_ref))


def simulate_dq(
    system: DriveSystem,
    v_d: float | Sequence[float] | np.ndarray = 0.0,
    v_q: float | Sequence[float] | np.ndarray = 0.0,
    duration: float = 1.0,
    dt: float | None = None,
    engine: str = "rk4",
    samples: int | None = 2000,
    initial: DqState | None = None,
    load_torque: float = 0.0,
    t: Sequence[float] | np.ndarray | None = None,
) -> DqSimulationResult:
    """Integrate the dq plant against the drivetrain, with applied voltages in.

    ``v_d`` and ``v_q`` are the applied phase voltages in the rotor frame: either
    scalars, or one value per stored sample so a drive log replays exactly.  The
    voltage is held across each integration step, which is what a current loop
    does between updates.

    Pass ``t`` to integrate on a grid you already have (a log's timestamps);
    ``duration``/``dt`` are then derived from it.  ``samples`` re-bins the
    *stored* trajectory without changing the solver step.

    ``load_torque`` is an *extra* torque at the motor shaft on top of the
    system's own load model, a magnitude that opposes motion (the
    ``RotationalLoad`` convention).  It exists so a test can hold the shaft at
    an equilibrium, and so a grade or a winch can be added without rebuilding
    the system.
    """
    if engine not in DQ_ENGINES:
        raise ConfigError(f"unknown dq engine {engine!r}; choose from {list(DQ_ENGINES)}")
    motor = machine_of(system)
    spec = motor.spec
    if load_torque < 0.0:
        raise ConfigError("load_torque is a magnitude that opposes motion and must be "
                          ">= 0; an active torque that assists the motion (a grade, a "
                          "spring) has to be applied through the system's load model")

    if t is not None:
        grid = np.asarray(t, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ConfigError("the dq time grid needs at least two increasing samples")
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
            raise ConfigError("the dq time grid must be finite and strictly increasing")
        steps = np.diff(grid)
        if steps.max() - steps.min() > 1e-9 * max(1.0, steps.max()):
            raise ConfigError("the dq integrator is fixed-step: the time grid has "
                              f"spacing from {steps.min():g} s to {steps.max():g} s; "
                              "resample the log onto a uniform grid first")
        dt = float(steps[0])
        n = grid.size
    else:
        if not math.isfinite(duration) or duration <= 0.0:
            raise ConfigError("the dq simulation needs a positive, finite duration")
        dt = solver_dt(system, spec, dt)
        n = int(max(2, round(duration / dt)))
        grid = np.arange(n) * dt

    arrays = {}
    for name, value in (("v_d", v_d), ("v_q", v_q)):
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size not in (1, n):
            raise ConfigError(
                f"{name} has {arr.size} samples but the simulation grid has {n}; "
                f"pass a scalar, or one value per grid point"
            )
        if not np.all(np.isfinite(arr)):
            raise ConfigError(f"{name} has a non-finite sample; a logged NaN cannot be integrated")
        arrays[name] = arr

    mech = _Mechanical.build(system, extra_load=load_torque)
    state = initial if initial is not None else DqState(
        0.0, 0.0, float(system.group.rot_vel), 0.0)

    i_d = np.empty(n)
    i_q = np.empty(n)
    speed = np.empty(n)
    torque = np.empty(n)
    angle = np.empty(n)
    applied_d = np.empty(n)
    applied_q = np.empty(n)

    for k in range(n):
        vd = float(arrays["v_d"][0] if arrays["v_d"].size == 1 else arrays["v_d"][k])
        vq = float(arrays["v_q"][0] if arrays["v_q"].size == 1 else arrays["v_q"][k])
        i_d[k] = state.i_d
        i_q[k] = state.i_q
        speed[k] = state.rot_vel
        angle[k] = state.angle
        applied_d[k] = vd
        applied_q[k] = vq
        torque[k] = mech.count * torque_em(spec, state)
        if k == n - 1:
            break
        state = step(system, state, vd, vq, dt, mech=mech, engine=engine)

    result = DqSimulationResult(
        t=grid, i_d=i_d, i_q=i_q, motor_speed=speed, torque=torque,
        v_d=applied_d, v_q=applied_q, angle=angle, system=system,
        engine=engine, dt=dt, load_torque=load_torque,
    )
    if samples and samples < n:
        idx = np.linspace(0, n - 1, samples).astype(int)
        result = DqSimulationResult(
            t=result.t[idx], i_d=result.i_d[idx], i_q=result.i_q[idx],
            motor_speed=result.motor_speed[idx], torque=result.torque[idx],
            v_d=result.v_d[idx], v_q=result.v_q[idx], angle=result.angle[idx],
            system=system, engine=engine, dt=dt, load_torque=load_torque,
        )
    return result
