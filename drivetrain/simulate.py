"""Time-domain solvers for :class:`~drivetrain.system.DriveSystem`.

Three engines, selectable by name:

``exact``  (default)
    Matrix-exponential integration of the state-space model with the load's
    viscous term re-linearised each sample.  Unconditionally stable and exact for
    the linear part, so it is the right choice for sweeps and optimisation.

``rk4``
    Fixed-step classical Runge-Kutta on the nonlinear state equations.  Handles
    Coulomb friction, current limiting, reverse power flow and quadratic road
    load.  Needs ``dt`` well below the fastest time constant.

``euler``
    Legacy explicit Euler, kept only so results can be compared against the
    original scripts in ``legacy/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .exceptions import SimulationError
from .motor import rad_s_to_rpm
from .system import DriveSystem

try:  # scipy is optional; a Pade-free fallback keeps things working without it
    from scipy.linalg import expm as _scipy_expm
except Exception:  # pragma: no cover
    _scipy_expm = None


# ----------------------------------------------------------------------
# results container
# ----------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Sampled state trajectory plus derived output / power channels."""

    t: np.ndarray
    motor_speed: np.ndarray                        # rad/s at the motor shaft
    current: np.ndarray                            # A, summed over all motors
    torque: np.ndarray                             # N*m gross motor-shaft torque
    voltage: np.ndarray                            # V applied
    output_speed: np.ndarray | None = None         # rad/s at the gearbox output
    ground_speed: np.ndarray | None = None         # m/s
    distance: np.ndarray | None = None             # m
    distance_revs: np.ndarray | None = None        # output-shaft revolutions
    power_out: np.ndarray | None = None            # W mechanical, gearbox output
    power_in: np.ndarray | None = None             # W electrical
    copper_loss: np.ndarray | None = None          # W
    efficiency: np.ndarray | None = None           # mechanical out / electrical in
    system: DriveSystem | None = None
    engine: str = ""
    dt: float = 0.0

    # -- convenience views ---------------------------------------------
    @property
    def motor_rpm(self) -> np.ndarray:
        return rad_s_to_rpm(self.motor_speed)

    @property
    def output_rpm(self) -> np.ndarray:
        return rad_s_to_rpm(self.output_speed)

    @property
    def speed_kmh(self) -> np.ndarray:
        return self.ground_speed * 3.6

    def channels(self) -> dict[str, np.ndarray]:
        """All plotted channels by name (used by the UI and CSV export)."""
        out = {
            "time": self.t,
            "motor_rpm": self.motor_rpm,
            "output_rpm": self.output_rpm,
            "current_A": self.current,
            "torque_Nm": self.torque,
            "voltage_V": self.voltage,
            "power_out_W": self.power_out,
            "power_in_W": self.power_in,
            "copper_loss_W": self.copper_loss,
            "efficiency": self.efficiency,
        }
        if self.system is not None and self.system.wheel is not None:
            out["speed_m_s"] = self.ground_speed
            out["speed_kmh"] = self.speed_kmh
            out["distance_m"] = self.distance
        else:
            out["output_revs"] = self.distance_revs
        return {k: v for k, v in out.items() if v is not None}

    def final(self) -> dict[str, float]:
        """Last sample of every channel."""
        out = {
            "motor_rpm": float(self.motor_rpm[-1]),
            "output_rpm": float(self.output_rpm[-1]),
            "current_A": float(self.current[-1]),
            "torque_Nm": float(self.torque[-1]),
            "efficiency": float(self.efficiency[-1]),
        }
        if self.system is not None and self.system.wheel is not None:
            out["speed_m_s"] = float(self.ground_speed[-1])
            out["speed_kmh"] = float(self.speed_kmh[-1])
            out["distance_m"] = float(self.distance[-1])
        return out

    def peak(self) -> dict[str, float]:
        """Peak values -- the numbers that size a motor controller."""
        return {
            "peak_current_A": float(np.max(np.abs(self.current))),
            "peak_current_per_motor_A": float(np.max(np.abs(self.current)) / (
                self.system.group.count if self.system else 1
            )),
            "peak_torque_Nm": float(np.max(np.abs(self.torque))),
            "peak_power_in_W": float(np.max(self.power_in)),
            "peak_copper_loss_W": float(np.max(self.copper_loss)),
            "peak_efficiency": float(np.max(self.efficiency)),
            "max_motor_rpm": float(np.max(self.motor_rpm)),
        }

    def metrics(self, target: float | None = None, tol: float = 0.02) -> dict[str, float]:
        """Rise time, settling time, overshoot and DC error of a step response.

        ``target`` defaults to the final value; pass the commanded speed to
        measure tracking against a setpoint the drivetrain cannot reach.
        """
        y = self.motor_speed
        target = float(y[-1]) if target is None else float(target)
        if abs(target) < 1e-12:
            return {}
        norm = y / target
        t = self.t
        rise = np.argmax(norm >= 0.9) if (norm >= 0.9).any() else None
        outside = np.abs(norm - 1.0) > tol
        settle = t[len(t) - np.argmax(outside[::-1]) - 1] if outside.any() else t[0]
        return {
            "steady_state": target,
            "rise_time_s": float(t[rise]) if rise is not None else np.nan,
            "settle_time_s": float(settle),
            "overshoot_pct": float(max(0.0, (norm.max() - 1.0) * 100.0)),
            "final_value": float(y[-1]),
            "dc_error": float(1.0 - y[-1] / target),
        }


# ----------------------------------------------------------------------
# waveform helpers -- anything 1-D over the sample times works as an input
# ----------------------------------------------------------------------
def step_voltage(n: int, value: float, step_at: float = 0.0, t: np.ndarray | None = None) -> np.ndarray:
    """Constant voltage ``value`` from ``step_at`` onwards."""
    if t is None:
        return np.full(n, value)
    return np.where(t >= step_at, value, 0.0)


def square_voltage(t: np.ndarray, value: float, period: float, duty: float = 0.5) -> np.ndarray:
    """PWM-like square wave between ``value`` and 0."""
    phase = np.mod(t, period) / period
    return np.where(phase < duty, value, 0.0)


def ramp_voltage(t: np.ndarray, value: float, ramp_time: float) -> np.ndarray:
    """Linear ramp 0 -> ``value`` over ``ramp_time``, then hold."""
    return value * np.clip(t / max(ramp_time, 1e-9), 0.0, 1.0)


def alternating_voltage(t: np.ndarray, value: float, period: float) -> np.ndarray:
    """Sign-alternating drive -- exercises reverse power flow and braking."""
    return value * np.where(np.mod(t, 2.0 * period) < period, 1.0, -1.0)


# ----------------------------------------------------------------------
# engine internals
# ----------------------------------------------------------------------
def _expm(matrix: np.ndarray) -> np.ndarray:
    if _scipy_expm is not None:
        return _scipy_expm(matrix)
    norm = np.linalg.norm(matrix, ord=np.inf)
    if norm == 0:
        return np.eye(matrix.shape[0])
    s = int(np.ceil(np.log2(max(norm, 1e-16)))) + 1
    a = matrix / (2.0**s)
    phi = np.eye(matrix.shape[0]) + a + 0.5 * a @ a + (a @ a @ a) / 6.0
    for _ in range(s):
        phi = phi @ phi
    return phi


def _derive_dt(system: DriveSystem, dt: float | None) -> float:
    """Auto step: resolve the fastest time constant with margin."""
    if dt is not None:
        return float(dt)
    taus = [system.mech_time_constant, system.electrical_time_constant]
    taus = [x for x in taus if np.isfinite(x) and x > 0]
    if not taus:
        return 1e-3
    return float(max(min(taus) / 50.0, 1e-7))


def _step_matrices(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Phi, Gamma)`` for ``x(k+1) = Phi x(k) + Gamma u(k)``.

    ``Phi = expm(A dt)`` and ``Gamma = integral_0^dt expm(A s) B ds``, computed
    together via the augmented-matrix exponential so no inverse of ``A`` is
    needed -- ``A`` can be singular (undamped free-spinning rotor).
    """
    m = A.shape[0]
    p = B.shape[1]
    aug = np.zeros((m + p, m + p))
    aug[:m, :m] = A * dt
    aug[:m, m:m + p] = B * dt
    E = _expm(aug)
    return E[:m, :m], E[:m, m:m + p]


def _linearised(
    system: DriveSystem, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """State matrices with the load's damping re-evaluated about ``state``.

    ``base_damping`` (motor + gears) is combined with the load's secant viscous
    coefficient at the current speed, so quadratic aerodynamic drag is tracked
    sample by sample instead of being frozen at one operating point.  The static
    (Coulomb / rolling / grade) part enters as an input.
    """
    speed = state[-1]
    damping = system.base_damping + system.load_damping_at(speed)
    A, B = system.state_matrices(damping=damping)
    return A, B, system.load_static_torque()


def _expm1_over_a(a: float, h: float) -> float:
    """``(exp(a*h) - 1) / a`` evaluated stably, including at ``a -> 0``.

    This is the input gain of an exact first-order step.  The naive form loses
    all precision for small ``a*h`` (and divides by zero for an undamped rotor),
    so the series is used below the cancellation threshold.
    """
    ah = a * h
    if -1e-4 < ah < 1e-4:
        # (e^x - 1)/a = h * (1 + x/2 + x^2/6 + ...)
        return h * (1.0 + ah * 0.5 + ah * ah / 6.0)
    return (math.exp(ah) - 1.0) / a


@dataclass(frozen=True)
class _FrozenModel:
    """Every per-step quantity precomputed once, so the integrators are tight loops.

    Profiling showed the engines spend nearly all their time on Python overhead
    (attribute chains, ``np.clip`` on scalars, ``scipy.linalg.expm`` called once
    per step for a 1x1 system), not on arithmetic.  This struct removes that by
    flattening the model into plain floats and fitting the reflected load torque
    to the polynomial

        tau_load(w) = sign(w) * (ts + bv*|w| + ca*w**2)

    which is *exact* for every load model in this package: constant Coulomb /
    rolling / grade terms, viscous damping, and quadratic aerodynamic drag.  The
    fit is verified against the real model at a speed it was not fitted at; if a
    custom load does not match, :attr:`exact_fit` is False and the engines fall
    back to calling the model directly.
    """

    kt: float
    kv: float
    R: float
    L: float
    J: float
    b: float          # mechanical damping (bearings + gears), load excluded
    em: float         # electromechanical damping kt*kv/R
    limit: float      # total current limit [A], inf if unlimited
    static: float     # bearing Coulomb friction [N*m]
    first_order: bool
    ts: float = 0.0   # speed-independent load torque at the motor shaft
    bv: float = 0.0   # viscous load coefficient
    ca: float = 0.0   # quadratic (aero) load coefficient
    exact_fit: bool = True

    @classmethod
    def build(cls, system: DriveSystem) -> "_FrozenModel":
        g = system.group
        model = cls(
            kt=float(g.kt_eff), kv=float(g.kv_eff), R=float(g.r_eff),
            L=float(g.l_eff), J=float(system.total_inertia),
            b=float(system.base_damping),
            em=float(g.kt_eff * g.kv_eff / g.r_eff),
            limit=float(g.motor.max_current) * g.count,
            static=float(g.static_friction_eff),
            first_order=bool(system.algebraic_current),
        )
        return model._fit_load(system)

    def _fit_load(self, system: DriveSystem) -> "_FrozenModel":
        lc = system.load_coefficients()
        return replace(self, ts=lc.static, bv=lc.viscous, ca=lc.quadratic,
                       exact_fit=lc.exact)

    # -- hot-path helpers ------------------------------------------------
    def load(self, w: float) -> float:
        """Reflected load torque at motor speed ``w`` (always opposes motion)."""
        if w > 0.0:
            return self.ts + self.bv * w + self.ca * w * w
        if w < 0.0:
            return -(self.ts - self.bv * w + self.ca * w * w)
        return self.ts

    def load_damping(self, w: float) -> float:
        """Secant viscous coefficient of the load at ``w`` [N*m/(rad/s)]."""
        return self.bv + self.ca * abs(w)

    def clamped_current(self, i: float) -> float:
        limit = self.limit
        if i > limit:
            return limit
        if i < -limit:
            return -limit
        return i


def _slow_load(system: DriveSystem, w: float) -> float:
    """Fallback path for loads the polynomial fit could not represent."""
    return float(system.reflected_load_torque(w))


def _enrich(result: SimulationResult, system: DriveSystem) -> SimulationResult:
    """Fill in output-side, power and efficiency channels."""
    gb = system.gearbox
    dt = float(result.t[1] - result.t[0]) if len(result.t) > 1 else 0.0
    result.output_speed = result.motor_speed * gb.ratio
    # Torque here is motor-shaft torque, so the power has to be taken at the
    # motor speed: the gearbox trades torque for speed and the ratio cancels
    # (tau_out = tau * efficiency / ratio, omega_out = omega * ratio), which
    # leaves tau * omega * efficiency.  Multiplying motor torque by output speed
    # under-reports a reducer by its ratio and over-reports a multiplier.
    result.power_out = result.torque * result.motor_speed * gb.efficiency
    result.power_in = result.voltage * result.current
    r = system.group.motor.resistance / system.group.count
    result.copper_loss = result.current**2 * r
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(np.abs(result.power_in) > 1e-12, result.power_out / result.power_in, 0.0)
    result.efficiency = np.clip(eff, 0.0, 1.0)
    wheel = system.wheel
    if wheel is not None:
        result.ground_speed = result.output_speed * wheel.radius
        result.distance = np.cumsum(result.ground_speed) * dt
    else:
        result.distance_revs = np.cumsum(result.output_speed) * dt / (2.0 * np.pi)
    return result


# ----------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------
def simulate_exact(
    system: DriveSystem, voltage: np.ndarray | float, t: np.ndarray, dt: float | None = None
) -> SimulationResult:
    """Exponential-fitting integration with current-limit mode switching.

    Below the current limit the model is linear and the step is exact; when the
    algebraic current would exceed the limit the drive is treated as a current
    source (constant torque ``kt * i_limit``) for that step, which is what a real
    ESC does.

    For the usual quasi-static motor (``inductance == 0``) the step is done in
    closed form with scalars -- one ``exp`` and a couple of multiplies.  Calling
    ``scipy.linalg.expm`` once per sample for a 1x1 system was the single largest
    cost in the whole package.  Second-order (``L > 0``) systems keep the
    augmented-matrix exponential, but it is cached on the damping value instead
    of being recomputed every step.
    """
    dt = _derive_dt(system, dt)
    n = len(t)
    v = np.full(n, float(voltage)) if np.isscalar(voltage) else np.asarray(voltage, float)
    if len(v) != n:
        raise SimulationError("voltage array length must match time array length.")

    m = _FrozenModel.build(system)
    kt, kv, R, J = m.kt, m.kv, m.R, m.J
    limited = math.isfinite(m.limit)
    limit = m.limit
    ts, bv, ca = m.ts, m.bv, m.ca
    fitted = m.exact_fit

    speed = np.empty(n)
    current = np.empty(n)
    torque = np.empty(n)

    if m.first_order:
        w = float(system.group.rot_vel)
        b_mech = m.b
        em = m.em
        b0 = kt / R / J          # voltage -> acceleration
        dth = dt                 # local alias
        for k in range(n):
            vk = float(v[k])
            i = (vk - kv * w) / R
            if limited and i > limit:
                i = limit
                saturated = True
            elif limited and i < -limit:
                i = -limit
                saturated = True
            else:
                saturated = False
            speed[k] = w
            current[k] = i
            torque[k] = kt * i
            if k == n - 1:
                break

            aw = abs(w)
            load_lin = (bv + ca * aw) if fitted else 0.0
            if saturated:
                # current source: no back-EMF feedback, so no electromechanical damping
                a = -(b_mech + load_lin) / J
                c = (kt * i - (ts if fitted else _slow_load(system, w))) / J
            else:
                a = -(em + b_mech + load_lin) / J
                c = b0 * vk - (ts if fitted else _slow_load(system, w) + load_lin * w) / J
            gain = _expm1_over_a(a, dth)
            w = math.exp(a * dth) * w + gain * c
        
    else:
        # second order: cache the matrix exponential per distinct damping value
        x = system.initial_state()
        cache: dict[float, tuple[tuple, tuple]] = {}
        for k in range(n):
            vk = float(v[k])
            i = float(x[0])
            if limited:
                i = m.clamped_current(i)
            w = float(x[1])
            speed[k] = w
            current[k] = i
            torque[k] = kt * i
            if k == n - 1:
                break
            damping = m.b + (m.load_damping(w) if fitted else system.load_damping_at(w))
            key = round(damping, 18) if damping else 0.0
            hit = cache.get(key)
            if hit is None:
                A, B, tau0 = _linearised(system, x)
                Phi, Gamma = _step_matrices(A, B, dt)
                hit = ((Phi[0, 0], Phi[0, 1], Phi[1, 0], Phi[1, 1],
                        Gamma[0, 0], Gamma[0, 1], Gamma[1, 0], Gamma[1, 1]), float(tau0))
                cache[key] = hit
            (p00, p01, p10, p11, g00, g01, g10, g11), tau0 = hit
            i_new = p00 * i + p01 * w + g00 * vk + g01 * tau0
            w_new = p10 * i + p11 * w + g10 * vk + g11 * tau0
            if limited:
                i_new = m.clamped_current(i_new)
            x = np.array([i_new, w_new])

    result = SimulationResult(
        t=t, motor_speed=speed, current=current, torque=torque, voltage=v,
        system=system, engine="exact", dt=dt,
    )
    return _enrich(result, system)


def simulate_rk4(
    system: DriveSystem, voltage: np.ndarray | float, t: np.ndarray, dt: float | None = None
) -> SimulationResult:
    """Fixed-step classical RK4 on the nonlinear state equations (L=0 or L>0).

    Written with plain floats and precomputed load coefficients: the previous
    version allocated numpy arrays and called ``np.clip`` inside the stage
    functions, which cost more than the arithmetic itself.
    """
    dt = _derive_dt(system, dt)
    n = len(t)
    v = np.full(n, float(voltage)) if np.isscalar(voltage) else np.asarray(voltage, float)

    m = _FrozenModel.build(system)
    kt, kv, R, L, J, b = m.kt, m.kv, m.R, m.L, m.J, m.b
    limit, static = m.limit, m.static
    limited = math.isfinite(limit)
    fitted = m.exact_fit

    def load(w: float) -> float:
        return m.load(w) if fitted else _slow_load(system, w)

    def current_of(vk: float, w: float) -> float:
        i = (vk - kv * w) / R
        if limited:
            if i > limit:
                return limit
            if i < -limit:
                return -limit
        return i

    def net_torque(vk: float, w: float, i: float) -> float:
        tau = kt * i - b * w - load(w)
        if w > 1e-12 or w < -1e-12:
            return tau - (static if w > 0 else -static)
        if abs(tau) <= abs(static):        # stiction can hold the shaft
            return 0.0
        return tau - (static if tau > 0 else -static)

    speed = np.empty(n)
    current = np.empty(n)
    torque = np.empty(n)

    w = float(system.group.rot_vel)
    i_state = current_of(float(v[0]), w)
    half = 0.5 * dt
    sixth = dt / 6.0

    for k in range(n):
        vk = float(v[k])
        if L <= 0.0:
            i_state = current_of(vk, w)
        speed[k] = w
        current[k] = i_state
        torque[k] = kt * i_state
        if k == n - 1:
            break

        if L > 0.0:
            i = i_state

            def f(i: float, wv: float) -> tuple[float, float]:
                return ((vk - i * R - kv * wv) / L, net_torque(vk, wv, i) / J)

            di1, dw1 = f(i, w)
            di2, dw2 = f(i + half * di1, w + half * dw1)
            di3, dw3 = f(i + half * di2, w + half * dw2)
            di4, dw4 = f(i + dt * di3, w + dt * dw3)
            i_state = i + sixth * (di1 + 2.0 * di2 + 2.0 * di3 + di4)
            w = w + sixth * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4)
            if limited:
                i_state = m.clamped_current(i_state)
        else:
            def fw(wv: float) -> float:
                return net_torque(vk, wv, current_of(vk, wv)) / J

            k1 = fw(w)
            k2 = fw(w + half * k1)
            k3 = fw(w + half * k2)
            k4 = fw(w + dt * k3)
            w += sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    result = SimulationResult(
        t=t, motor_speed=speed, current=current, torque=torque, voltage=v,
        system=system, engine="rk4", dt=dt,
    )
    return _enrich(result, system)


def simulate_euler(
    system: DriveSystem, voltage: np.ndarray | float, t: np.ndarray, dt: float | None = None
) -> SimulationResult:
    """Explicit Euler -- legacy behaviour, kept for comparison only."""
    dt = _derive_dt(system, dt)
    n = len(t)
    v = np.full(n, float(voltage)) if np.isscalar(voltage) else np.asarray(voltage, float)

    m = _FrozenModel.build(system)
    kt, kv, R, J, b = m.kt, m.kv, m.R, m.J, m.b
    limit = m.limit
    limited = math.isfinite(limit)
    fitted = m.exact_fit

    speed = np.empty(n)
    current = np.empty(n)
    torque = np.empty(n)
    w = float(system.group.rot_vel)
    dt_over_J = dt / J

    for k in range(n):
        i = (float(v[k]) - kv * w) / R
        if limited:
            if i > limit:
                i = limit
            elif i < -limit:
                i = -limit
        tau_i = kt * i
        tau = tau_i - b * w - (m.load(w) if fitted else _slow_load(system, w))
        speed[k] = w
        current[k] = i
        torque[k] = tau_i
        if k == n - 1:
            break
        w += tau * dt_over_J

    result = SimulationResult(
        t=t, motor_speed=speed, current=current, torque=torque, voltage=v,
        system=system, engine="euler", dt=dt,
    )
    return _enrich(result, system)


ENGINES = {"exact": simulate_exact, "rk4": simulate_rk4, "euler": simulate_euler}


def voltage_signal(
    kind: str,
    system: DriveSystem,
    duration: float,
    dt: float | None = None,
    value: float | None = None,
    period: float | None = None,
    ramp_time: float | None = None,
) -> np.ndarray:
    """Build an excitation waveform on the *solver's own* time grid.

    The engines require the voltage array to line up sample-for-sample with the
    internal time vector, which is easy to get wrong by hand (it depends on
    ``dt``, which is itself derived from the system's fastest time constant).
    Building the signal here keeps callers honest::

        v = voltage_signal("square", system, duration=5.0)
        simulate(system, v, 5.0)

    ``kind`` is one of ``step``, ``square``, ``ramp``, ``alternating``, ``coast``.
    ``value`` defaults to the system's own voltage limit.
    """
    dt = _derive_dt(system, dt)
    n = int(max(2, round(duration / dt)))
    t = np.arange(n) * dt
    v = value if value is not None else (
        system.voltage_limit if np.isfinite(system.voltage_limit) else 12.0)

    kind = kind.strip().lower()
    if kind in ("step", "const", "dc", "on"):
        return np.full(n, float(v))
    if kind == "square":
        return square_voltage(t, v, period if period else duration / 3.0)
    if kind == "ramp":
        return ramp_voltage(t, v, ramp_time if ramp_time else duration / 2.0)
    if kind == "alternating":
        return alternating_voltage(t, v, period if period else duration / 4.0)
    if kind in ("coast", "off", "zero", "brake"):
        return np.zeros(n)
    raise SimulationError(f"Unknown excitation {kind!r}; choose from {EXCITATIONS}.")


EXCITATIONS = ("step", "square", "ramp", "alternating", "coast")


def simulate(
    system: DriveSystem,
    voltage: float | np.ndarray = 12.0,
    duration: float = 5.0,
    dt: float | None = None,
    engine: str = "exact",
    samples: int | None = 2000,
) -> SimulationResult:
    """Run a time-domain simulation.

    ``samples`` re-bins the *stored* trajectory (plots do not need solver
    resolution) while the internal step stays at ``dt``.
    """
    if engine not in ENGINES:
        raise SimulationError(f"Unknown engine {engine!r}; choose from {sorted(ENGINES)}.")
    dt = _derive_dt(system, dt)
    n_internal = int(max(2, round(duration / dt)))
    t_internal = np.arange(n_internal) * dt
    result = ENGINES[engine](system, voltage, t_internal, dt)
    if samples and samples < n_internal:
        idx = np.linspace(0, n_internal - 1, samples).astype(int)
        result = SimulationResult(
            t=result.t[idx],
            motor_speed=result.motor_speed[idx],
            current=result.current[idx],
            torque=result.torque[idx],
            voltage=result.voltage[idx],
            system=system,
            engine=engine,
            dt=dt,
        )
        result = _enrich(result, system)
    return result
