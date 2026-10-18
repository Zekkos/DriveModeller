"""Brushless PMSM in the rotor dq frame -- the machine an FOC drive commutates.

This is a *different machine*, not another parameter on
:class:`~drivetrain.motor.MotorModel`.  The brushed model above has one
electrical constant, one resistance and one mechanical port; a permanent-magnet
synchronous machine has two inductances, a rotor flux linkage, a pole count, and
a current that splits into two components the drive controls independently.  All
of that is invisible to a ``kv``/``kt`` pair, and so is the thing FOC actually
buys: the drive can inject a *negative* d-axis current to weaken the rotor flux
and keep spinning above the speed where the back-EMF alone would exceed the
available voltage.

Equations (SI, rotor frame, motor convention -- everything is a peak, not an rms)::

    v_d = R*i_d + Ld*d(i_d)/dt - w_e*Lq*i_q
    v_q = R*i_q + Lq*d(i_q)/dt + w_e*(Ld*i_d + lambda_)
    tau = 1.5*p*(lambda_*i_q + (Ld - Lq)*i_d*i_q)
    J*d(w)/dt = tau - b*w - tau_load - tau_friction
    w_e = p*w

``1.5`` comes from the amplitude-invariant Park transform, and it is the single
most common source of arguments about motor constants: with that convention the
torque constant is ``1.5*p*lambda_`` N*m/A, copper loss is ``1.5*R*(i_d**2 +
i_q**2)`` W, and input power is ``1.5*(v_d*i_d + v_q*i_q)`` W.  Some datasheets
and some textbooks use the power-invariant transform instead, in which case the
``1.5`` disappears from power and appears in the torque constant as ``p*lambda_``
with currents scaled by ``sqrt(1.5)``.  This module is amplitude-invariant
throughout; if your numbers came from a power-invariant source, multiply the
quoted current by ``sqrt(1.5)`` and the quoted inductance/flux linkage by
``1/sqrt(1.5)``... no -- the safe conversion is in the notes at the bottom of
this docstring.

Why the dq frame is the right level of detail here
--------------------------------------------------
The mechanical port is unchanged: whatever the machine is, it delivers a torque
into a gearbox and a load, and everything downstream of
:func:`drivetrain.system.DriveSystem` stays valid.  Only the electrical side
changes, and in the dq frame it changes into something just as small: two first
order electrical states instead of one, one algebraic torque law with a
reluctance term, and two circular limits (a current circle and a voltage
ellipse) instead of one current clamp.

The interesting consequences, all of which this module computes:

* **Base speed is not maximum speed.**  With ``i_d = 0`` the back-EMF is
  ``p*lambda_*w``, so the machine stops at ``V/(p*lambda_)``.  A brushed motor
  stops there too and that is the end of the story.  A PMSM with an inverter that
  can still push current can run far past it by driving ``i_d`` negative until
  ``Ld*i_d + lambda_`` nearly cancels -- field weakening.
* **Saliency is free torque.**  When ``Lq > Ld`` (interior magnets) the
  reluctance term ``1.5*p*(Ld - Lq)*i_d*i_q`` is positive for ``i_d < 0``, so
  the same current circle produces more torque than ``1.5*p*lambda_*i_q`` alone.
  Maximum torque per ampere (MTPA) is the angle that trades those two off.
* **The DC-link ammeter is not the phase ammeter.**  A shunt on the supply rail
  reads ``1.5*(v_d*i_d + v_q*i_q)/V_bus`` for a lossless inverter, which at low
  speed is nothing like the phase current.  Fitting a PMSM from a single rail
  current has to say so.

What is deliberately *not* here
-------------------------------
Cogging torque, trapezoidal back-EMF and commutation ripple, inverter deadtime,
current-loop bandwidth and sensor delay, iron loss as a dq current component,
magnet temperature *dynamics* (only a steady coefficient), magnetic saturation of
``Ld``/``Lq`` with current, and breakdown of the linear magnetic circuit.  Iron
loss is included as a loss term only (see :meth:`PMSM.core_loss_power`), which is
honest for efficiency and feed-forward but does not feed back into the torque
law; that approximation is stated where it is used, not hidden.

Conversion notes for power-invariant numbers
--------------------------------------------
If a source defines ``tau = p*lambda_*i_q`` and ``P = v_d*i_d + v_q*i_q`` (the
power-invariant convention), convert into this module's convention with::

    i  -> i * sqrt(1.5)          (currents, and therefore current limits)
    L  -> L / sqrt(1.5)          (inductances)
    R  -> R / 1.5                (resistance)
    lambda_ -> lambda_ / 1.5     (flux linkage)
    v  -> v / sqrt(1.5)          (voltages, and therefore voltage limits)

Torque, mechanical quantities and ``w_e = p*w`` are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .exceptions import ConfigError
from .motor import rad_s_to_rpm, rpm_to_rad_s

# Speed below which "moving" and "stuck" are indistinguishable, mirroring the
# stiction band the RK4 engine uses.  See DriveSystem for the same constant.
STICTION_BAND = 1.0e-6

_SQRT_1_5 = math.sqrt(1.5)


def phase_voltage_limit(dc_bus: float, scheme: str = "svpwm") -> float:
    """Peak fundamental phase voltage an inverter can make from a DC bus.

    ``svpwm`` (and third-harmonic injection) use the full bus: the linear region
    ends at ``V_bus / sqrt(3)`` phase peak.  Plain sinusoidal PWM tops out at
    ``V_bus / 2``.  Over-modulation buys a little more and costs waveform
    distortion; it is not modelled.
    """
    if dc_bus <= 0:
        raise ConfigError("DC bus voltage must be > 0.")
    scheme = str(scheme).strip().lower()
    if scheme in ("svpwm", "space_vector", "5segment", "thi"):
        return float(dc_bus) / math.sqrt(3.0)
    if scheme in ("sinusoidal", "spwm", "2segment"):
        return float(dc_bus) / 2.0
    raise ConfigError(f"unknown modulation scheme {scheme!r}; expected svpwm or sinusoidal")


# ---------------------------------------------------------------------------
# small deterministic 1-D optimisers
#
# Written with plain ``math`` rather than numpy on purpose: the same shape has to
# compile into C and Rust for the export, where a root finder with a while-loop
# and a tolerance is easier to reason about than a vectorised search.
# ---------------------------------------------------------------------------

def _maximise(fn, lo: float, hi: float, grid: int = 48, refine: int = 48) -> float:
    """Maximise ``fn`` on ``[lo, hi]``: coarse grid, then golden-section.

    The objective is piecewise smooth (it switches between the current circle and
    the voltage ellipse) and unimodal in practice, which is exactly the case a
    grid seed protects against: golden-section alone can settle on the wrong
    smooth piece.
    """
    if not (hi > lo):
        return lo
    step = (hi - lo) / grid
    best_x, best_f = lo, fn(lo)
    for k in range(1, grid + 1):
        x = lo + k * step
        f = fn(x)
        if f > best_f:
            best_x, best_f = x, f
    a = max(lo, best_x - step)
    b = min(hi, best_x + step)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(refine):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = fn(d)
        if (b - a) <= 1.0e-12 * max(1.0, abs(a), abs(b)):
            break
    x = 0.5 * (a + b)
    return x if fn(x) >= best_f else best_x


def _clamp(value: float, limit: float) -> float:
    """Symmetric clamp; ``limit`` of ``inf`` (or a non-positive limit) is open."""
    if not math.isfinite(limit) or limit <= 0.0:
        return float(value)
    return float(min(max(float(value), -limit), limit))


@dataclass
class DqCommand:
    """One inverter frame worth of feed-forward: what to apply, and what it costs.

    ``voltage_saturated`` means the requested torque could not be met at this
    speed without exceeding the inverter's voltage; ``current_saturated`` means
    the current circle bound it instead.  Both can be true at once at the corner
    where the two limits meet.
    """

    i_d: float
    i_q: float
    v_d: float
    v_q: float
    torque: float                     # electromagnetic torque actually commanded [N*m]
    current_saturated: bool = False
    voltage_saturated: bool = False
    field_weakening: bool = False
    dc_link_current: float = 0.0      # supply-rail current estimate [A]
    speed: float = 0.0                # mechanical speed it was computed at [rad/s]

    @property
    def current_magnitude(self) -> float:
        """Phase current amplitude ``sqrt(i_d**2 + i_q**2)`` [A]."""
        return math.hypot(self.i_d, self.i_q)

    @property
    def voltage_magnitude(self) -> float:
        """Phase voltage amplitude ``sqrt(v_d**2 + v_q**2)`` [V]."""
        return math.hypot(self.v_d, self.v_q)

    def as_dict(self) -> dict:
        return {
            "i_d": self.i_d, "i_q": self.i_q,
            "v_d": self.v_d, "v_q": self.v_q,
            "torque": self.torque,
            "current_magnitude": self.current_magnitude,
            "voltage_magnitude": self.voltage_magnitude,
            "current_saturated": self.current_saturated,
            "voltage_saturated": self.voltage_saturated,
            "field_weakening": self.field_weakening,
            "dc_link_current": self.dc_link_current,
            "speed": self.speed,
        }


@dataclass
class PMSMSpec:
    """A permanent-magnet synchronous machine, sinusoidal/FOC, dq frame.

    Sign convention matches the brushed model: ``rot_vel`` and produced torque
    are positive in the driving direction, and a passive load opposes motion and
    is applied elsewhere as a magnitude.

    ``voltage_limit`` is the **peak fundamental phase voltage** the inverter can
    produce (use :func:`phase_voltage_limit` to get it from a DC bus), not the
    bus itself.  ``current_limit`` is the peak phase current amplitude -- the
    radius of the current circle, which is what an FOC drive limits on, not the
    DC-link average.
    """

    # -- electromagnetic constants ---------------------------------------
    resistance: float = 0.12              # phase resistance (per phase, wye) [Ohm]
    inductance_d: float = 1.5e-4          # d-axis inductance            [H]
    inductance_q: float = 1.8e-4          # q-axis inductance            [H]
    flux_linkage: float = 1.8e-3          # PM flux linking a phase      [Wb]
    pole_pairs: int = 7                   # rotor pole pairs             [-]

    # -- mechanical ------------------------------------------------------
    rotor_inertia: float = 2.0e-5         # rotor inertia at the shaft   [kg*m^2]
    bearing_damping: float = 1.0e-6       # viscous bearing/windage      [N*m/(rad/s)]
    static_friction: float = 0.0          # Coulomb friction torque      [N*m]

    # -- limits ----------------------------------------------------------
    current_limit: float = 20.0           # peak phase current amplitude [A]
    voltage_limit: float = 12.0 / math.sqrt(3.0)  # peak phase voltage    [V]
    dc_bus_voltage: float = math.inf      # supply rail, for the rail-current estimate only

    # -- temperature -----------------------------------------------------
    flux_temperature_coefficient: float = 0.0  # d(lambda)/lambda per degC [1/degC]
    reference_temperature: float = 20.0        # temperature the constants above are quoted at

    # -- loss terms (loss only; they do not change the torque law) -------
    core_loss_coefficient: float = 0.0    # iron loss ~ k_e * w_e**2      [W/(rad/s)^2]
    windage_coefficient: float = 0.0      # mechanical loss ~ k * w**3    [W/(rad/s)^3]

    # -- state -----------------------------------------------------------
    current_d: float = field(default=0.0, metadata={"state": True})
    current_q: float = field(default=0.0, metadata={"state": True})
    rot_vel: float = field(default=0.0, metadata={"state": True})

    def __post_init__(self) -> None:
        self.pole_pairs = int(self.pole_pairs)
        self.validate()

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        if self.resistance <= 0:
            raise ConfigError("PMSM phase resistance must be > 0.")
        if self.inductance_d <= 0 or self.inductance_q <= 0:
            raise ConfigError("PMSM inductances must be > 0 (use the brushed model for L=0).")
        if self.flux_linkage <= 0:
            raise ConfigError("PMSM flux linkage must be > 0 (a synchronous machine needs magnets).")
        if self.pole_pairs < 1:
            raise ConfigError("PMSM pole_pairs must be >= 1.")
        if self.rotor_inertia <= 0:
            raise ConfigError("PMSM rotor inertia must be > 0.")
        if self.bearing_damping < 0 or self.static_friction < 0:
            raise ConfigError("PMSM mechanical losses must be >= 0.")
        if self.current_limit <= 0:
            raise ConfigError("PMSM current limit must be > 0 (use math.inf for unlimited).")
        if self.voltage_limit <= 0:
            raise ConfigError("PMSM voltage limit must be > 0 (use math.inf for unlimited).")
        if self.core_loss_coefficient < 0 or self.windage_coefficient < 0:
            raise ConfigError("PMSM loss coefficients must be >= 0.")

    def copy(self) -> PMSMSpec:
        return replace(self)

    # -- the machine's identity ------------------------------------------
    @property
    def is_salient(self) -> bool:
        """True when ``Ld != Lq``, i.e. an interior-magnet machine with reluctance torque."""
        return abs(self.inductance_q - self.inductance_d) > 1.0e-15

    @property
    def torque_constant(self) -> float:
        """``d(tau)/d(i_q)`` at ``i_d = 0``: ``1.5*p*lambda_`` [N*m/A].

        This is the number that plays the role of ``kt`` in the brushed model,
        and for a surface-magnet machine it is the whole torque law.
        """
        return 1.5 * self.pole_pairs * self.flux_linkage

    @property
    def back_emf_constant(self) -> float:
        """``q``-axis back-EMF per mechanical rad/s: ``p*lambda_`` [V/(rad/s)].

        Plays the role of ``kv``.  Note ``torque_constant / back_emf_constant``
        is ``1.5``, not ``1`` -- that ratio is the amplitude-invariant Park
        convention, not an inconsistency, and it is the first thing to check when
        a fitted PMSM looks like it has mismatched kt and kv.
        """
        return self.pole_pairs * self.flux_linkage

    @property
    def reluctance_coefficient(self) -> float:
        """``1.5*p*(Ld - Lq)``, multiplying ``i_d*i_q`` in the torque law [N*m/A^2]."""
        return 1.5 * self.pole_pairs * (self.inductance_d - self.inductance_q)

    @property
    def electrical_time_constant(self) -> float:
        """``Lq/R`` [s] -- the slower of the two axes, so the one that matters."""
        return self.inductance_q / self.resistance

    @property
    def characteristic_current(self) -> float:
        """``lambda_/Ld`` [A]: the d-current that cancels the magnet flux.

        Field weakening past this point reverses the air-gap flux; the model
        treats it as the practical limit, which is also roughly where a real
        machine starts to saturate and where the demagnetising risk begins.
        """
        return self.flux_linkage / self.inductance_d

    def effective_flux(self, temperature_c: float | None = None) -> float:
        """Flux linkage at a magnet temperature (steady coefficient only)."""
        if temperature_c is None or not self.flux_temperature_coefficient:
            return self.flux_linkage
        dk = 1.0 + self.flux_temperature_coefficient * (float(temperature_c) - self.reference_temperature)
        return self.flux_linkage * max(dk, 0.0)

    def electrical_speed(self, rot_vel: float | None = None) -> float:
        """Electrical angular velocity ``w_e = p*w`` [rad/s electrical]."""
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        return self.pole_pairs * w

    def rpm(self, rot_vel: float | None = None) -> float:
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        return float(rad_s_to_rpm(w))

    # -- torque and voltage at a current pair ----------------------------
    def back_emf_constant_at(self, temperature_c: float | None = None) -> float:
        """Back-EMF constant at a magnet temperature [V/(rad/s mechanical)]."""
        return self.pole_pairs * self.effective_flux(temperature_c)

    def torque_constant_at(self, temperature_c: float | None = None) -> float:
        """Torque constant at a magnet temperature [N*m/A].

        Heat demagnetises the magnets, so a hot machine needs more current for
        the same torque and runs faster on the same voltage.  Only a steady
        coefficient is modelled - no demagnetisation knee, no irreversible loss.
        """
        return 1.5 * self.pole_pairs * self.effective_flux(temperature_c)

    def no_load_speed(self, voltage: float | None = None,
                      *, temperature_c: float | None = None) -> float:
        """Speed at which the back-EMF equals the applied voltage [rad/s mechanical].

        ``v / (p * lambda)``: the ideal lossless no-load speed, so a real machine
        settles a little below it.  Above base speed the inverter can only hold
        the shaft here by field-weakening, which costs current even at zero
        torque - treat this as a bound, not an operating point.
        """
        v = self.voltage_limit if voltage is None else abs(float(voltage))
        k = self.back_emf_constant_at(temperature_c)
        return v / k if k > 0.0 else math.inf

    def torque_at(self, i_d: float, i_q: float) -> float:
        """Electromagnetic torque for a current pair [N*m]."""
        return 1.5 * self.pole_pairs * (
            self.flux_linkage * i_q + (self.inductance_d - self.inductance_q) * i_d * i_q
        )

    def torque_gain_at(self, i_d: float) -> float:
        """``d(tau)/d(i_q)`` at this ``i_d``: ``1.5*p*(lambda_ + (Ld-Lq)*i_d)`` [N*m/A].

        Positive throughout the operating range the model allows; it going
        negative would mean the reluctance term had taken over, which needs
        ``i_d`` past :attr:`characteristic_current`.
        """
        return 1.5 * self.pole_pairs * (
            self.flux_linkage + (self.inductance_d - self.inductance_q) * float(i_d)
        )

    def current_magnitude(self, i_d: float, i_q: float) -> float:
        return math.hypot(float(i_d), float(i_q))

    def voltage_steady(self, i_d: float, i_q: float, rot_vel: float | None = None) -> tuple[float, float]:
        """Steady-state (``di/dt = 0``) voltage pair for a current pair at a speed."""
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        we = self.electrical_speed(w)
        v_d = self.resistance * i_d - we * self.inductance_q * i_q
        v_q = self.resistance * i_q + we * (self.inductance_d * i_d + self.flux_linkage)
        return v_d, v_q

    def voltage_for(self, i_d: float, i_q: float, rot_vel: float | None = None,
                    di_d_dt: float = 0.0, di_q_dt: float = 0.0) -> tuple[float, float]:
        """Full voltage pair, including the ``L di/dt`` terms."""
        v_d, v_q = self.voltage_steady(i_d, i_q, rot_vel)
        return (v_d + self.inductance_d * float(di_d_dt),
                v_q + self.inductance_q * float(di_q_dt))

    def derivatives(self, v_d: float, v_q: float, rot_vel: float | None = None,
                    torque_load: float = 0.0) -> tuple[float, float, float]:
        """State derivatives ``(di_d/dt, di_q/dt, dw/dt)`` for a applied voltage.

        ``torque_load`` is everything on the mechanical port that is not the
        machine's own bearing damping or static friction, applied as a torque
        (passive loads oppose motion, so pass the *magnitude* through
        ``-sign(w)*tau_load`` yourself or use :meth:`acceleration`).
        """
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        we = self.electrical_speed(w)
        di_d = (v_d - self.resistance * self.current_d + we * self.inductance_q * self.current_q) \
            / self.inductance_d
        di_q = (v_q - self.resistance * self.current_q - we * (self.inductance_d * self.current_d
                + self.flux_linkage)) / self.inductance_q
        return di_d, di_q, self.acceleration(self.torque_at(self.current_d, self.current_q), w, torque_load)

    def acceleration(self, torque: float, rot_vel: float | None = None,
                     load_torque: float = 0.0) -> float:
        """Mechanical acceleration from a given electromagnetic torque [rad/s^2]."""
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        tau = float(torque) - self.bearing_damping * w - float(load_torque)
        if self.static_friction and abs(w) > STICTION_BAND:
            tau -= self.static_friction * (1.0 if w > 0 else -1.0)
        elif self.static_friction and abs(tau) <= self.static_friction:
            return 0.0
        return tau / self.rotor_inertia

    # -- the two limits ---------------------------------------------------
    def _id_floor(self, i_limit: float) -> float:
        """Most negative ``i_d`` the model allows: the current circle, or the
        point where the magnet flux is fully cancelled, whichever is nearer zero."""
        return max(-i_limit, -self.characteristic_current) if math.isfinite(i_limit) \
            else -self.characteristic_current

    def _iq_from_voltage(self, i_d: float, we: float, v_limit: float) -> float | None:
        """Largest ``i_q >= 0`` the voltage ellipse allows at this ``i_d``.

        ``|v|^2 <= V^2`` is quadratic in ``i_q``::

            A*i_q^2 + B*i_q + C - V^2 <= 0
            A = R^2 + (w_e*Lq)^2
            B = 2*R*w_e*(lambda_ + (Ld - Lq)*i_d)
            C = (R*i_d)^2 + (w_e*(Ld*i_d + lambda_))^2

        so the answer is the positive root.  ``None`` means no ``i_q`` at this
        ``i_d`` satisfies the voltage limit at all -- the back-EMF alone is too
        big, and the machine has to be weakened further (more negative ``i_d``)
        before any current can flow at all.
        """
        if not math.isfinite(v_limit):
            return math.inf
        lam, ld, lq, r = self.flux_linkage, self.inductance_d, self.inductance_q, self.resistance
        a = r * r + (we * lq) ** 2
        b = 2.0 * r * we * (lam + (ld - lq) * i_d)
        c = (r * i_d) ** 2 + (we * (ld * i_d + lam)) ** 2 - v_limit * v_limit
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        root = (-b + math.sqrt(disc)) / (2.0 * a)
        return root if root >= 0.0 else 0.0

    def _iq_ceiling(self, i_d: float, we: float, i_limit: float, v_limit: float) -> float:
        """The current circle and the voltage ellipse, whichever binds first."""
        circle = math.sqrt(max(0.0, i_limit * i_limit - i_d * i_d)) if math.isfinite(i_limit) \
            else math.inf
        ellipse = self._iq_from_voltage(i_d, we, v_limit)
        if ellipse is None:
            return -1.0          # this i_d is infeasible at any current
        return min(circle, ellipse)

    def torque_limit(self, rot_vel: float, i_limit: float | None = None,
                     voltage_limit: float | None = None, *,
                     regenerative: bool = False) -> float:
        """Largest *magnitude* of torque the drive can command at this speed [N*m].

        This is the MTPA / current-circle / voltage-ellipse envelope of the
        classic two-circle diagram, and it is the number a feed-forward
        controller has to respect before it asks for an acceleration.  It is even
        in speed: the map ``(i_d, i_q, w_e) -> (i_d, -i_q, -w_e)`` leaves ``|v|``
        unchanged and flips the sign of both torque and speed, so forward
        motoring at ``+w`` and reverse motoring at ``-w`` have the same envelope.

        ``regenerative=True`` asks the other question - how much *braking* torque
        is available, with torque against the direction of rotation.  That
        envelope is larger, because the back-EMF term and the resistive drop of
        ``i_q`` partly cancel instead of adding, so more of the voltage is left
        for current.  Pass it explicitly; the default stays motoring because that
        is what a capability curve quotes.
        """
        i_lim = self.current_limit if i_limit is None else float(i_limit)
        v_lim = self.voltage_limit if voltage_limit is None else float(voltage_limit)
        we = abs(self.electrical_speed(rot_vel))
        if regenerative:
            we = -we
        best_i_d = self._max_torque_id(we, i_lim, v_lim)
        iq = self._iq_ceiling(best_i_d, we, i_lim, v_lim)
        return max(0.0, self.torque_at(best_i_d, max(0.0, iq)))

    def loss_torque(self, rot_vel: float) -> float:
        """Torque the machine spends on its own losses at a speed [N*m].

        Bearing damping is modelled as a torque; windage and iron loss are
        modelled as *powers*, so they are divided by speed to become an
        equivalent torque.  Iron loss ``k_e*w_e^2`` therefore shows up as
        ``k_e*p^2*w``, which is the same linear-in-speed behaviour a
        eddy-current loss has, and is the reason it is a loss term and not a
        dq current component in this model.
        """
        w = abs(float(rot_vel))
        if w == 0.0:
            return self.static_friction
        tau = self.bearing_damping * w + self.windage_coefficient * w * w
        tau += self.core_loss_coefficient * self.pole_pairs ** 2 * w
        return tau

    def max_speed(self, load_torque: float = 0.0, tol: float = 1.0e-9) -> float:
        """Highest mechanical speed at which the drive can still hold ``load_torque``.

        In the ideal model the field-weakening envelope decays smoothly towards
        zero rather than hitting it, so a machine with no losses has no maximum
        speed at all -- the honest answer for ``load_torque = 0`` and zero loss
        coefficients is ``inf``.  Set :attr:`windage_coefficient` or
        :attr:`core_loss_coefficient` (or pass an external ``load_torque``) and
        the answer becomes finite and useful: it is where the envelope crosses
        the torque the drive has to spend just to keep spinning.
        """
        lo, hi = 0.0, max(1.0, self.base_speed())

        def deficit(w: float) -> float:
            return self.torque_limit(w) - self.loss_torque(w) - abs(float(load_torque))

        if deficit(hi) <= 0.0:
            return hi
        while deficit(hi) > 0.0:
            hi *= 2.0
            if hi > 1.0e12:
                return math.inf
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if deficit(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= tol * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def base_speed(self, voltage_limit: float | None = None) -> float:
        """Speed at which the back-EMF alone consumes the available voltage.

        At ``i_d = 0`` this is a hard wall; with field weakening it is only where
        constant-torque operation ends.  ``V / (p * lambda_)``.
        """
        v = self.voltage_limit if voltage_limit is None else float(voltage_limit)
        return v / self.back_emf_constant

    # -- current selection -------------------------------------------------
    def mtpa_pair(self, torque: float, i_limit: float | None = None) -> tuple[float, float]:
        """Minimum-current ``(i_d, i_q)`` for a torque, ignoring the voltage limit.

        Closed form for a salient machine (with ``dl = Lq - Ld``)::

            i_d = (lambda_ - sqrt(lambda_^2 + 8*dl^2*i_m^2)) / (4*dl)

        which tends to ``0`` as ``dl -> 0``, so the same expression covers
        surface-magnet machines too.  It comes from setting ``d(tau)/d(i_q) = 0``
        on the current circle, which is ``2*dl*x^2 + lambda_*x - dl*i_m^2 = 0`` in
        ``x = -i_d``.  ``i_m`` is then solved from the torque by bisection, since
        torque is quadratic in the current angle but monotone in its magnitude.
        """
        i_lim = self.current_limit if i_limit is None else float(i_limit)
        tau = abs(float(torque))
        sign = 1.0 if float(torque) >= 0 else -1.0
        if tau == 0.0:
            return 0.0, 0.0
        kt = self.torque_constant
        if not self.is_salient:
            i_q = tau / kt
            if math.isfinite(i_lim) and i_q > i_lim:
                return 0.0, math.inf      # not achievable within the current circle
            return 0.0, sign * i_q
        dl = self.inductance_q - self.inductance_d
        lam = self.flux_linkage

        def mtpa_d(i_m: float) -> float:
            return (lam - math.sqrt(lam * lam + 8.0 * dl * dl * i_m * i_m)) / (4.0 * dl)

        def torque_at_mag(i_m: float) -> float:
            i_d = mtpa_d(i_m)
            i_q = math.sqrt(max(0.0, i_m * i_m - i_d * i_d))
            return abs(self.torque_at(i_d, i_q))

        lo, hi = 0.0, tau / kt
        while torque_at_mag(hi) < tau and hi < 1.0e9:
            hi *= 2.0
        if torque_at_mag(hi) < tau:
            return 0.0, math.inf
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if torque_at_mag(mid) >= tau:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1.0e-12 * max(1.0, hi):
                break
        i_m = 0.5 * (lo + hi)
        i_d = mtpa_d(i_m)
        i_q = math.sqrt(max(0.0, i_m * i_m - i_d * i_d))
        if math.isfinite(i_lim) and i_m > i_lim + 1.0e-9:
            return 0.0, math.inf
        return i_d, sign * i_q

    def current_pair(self, torque: float, rot_vel: float) -> tuple[float, float, bool, bool]:
        """Best ``(i_d, i_q)`` for a torque at a speed, with both limits honoured.

        Returns ``(i_d, i_q, current_saturated, voltage_saturated)``.  "Best"
        means minimum current magnitude, which is MTPA when the voltage allows it
        and a field-weakening angle when it does not.  If the torque cannot be
        delivered at all at this speed, the pair returned is the one that gets
        closest (maximum available torque) with the flag that bound it set.
        """
        tau = float(torque)
        w = float(rot_vel)
        tau_mag = abs(tau)
        sign = 1.0 if tau >= 0 else -1.0
        i_lim, v_lim = self.current_limit, self.voltage_limit
        # Work in the frame where the commanded torque is positive.  The voltage
        # ellipse depends on the *relative* sign of torque and back-EMF, which is
        # what separates motoring from regenerative braking, and flipping both
        # signs leaves |v| unchanged - so one search covers all four quadrants.
        # Flipping only one of them does not: braking at +w is not the same
        # problem as driving at +w, and using the driving ellipse for braking
        # asks for more current than the machine actually needs.
        w_frame = sign * w
        we_frame = self.electrical_speed(w_frame)
        braking = w_frame < 0.0

        limit = self.torque_limit(w_frame, regenerative=braking)
        if tau_mag > limit + 1.0e-12:
            # Not achievable: give the angle that makes the most torque at this
            # speed and let the caller see which limit was responsible.
            i_d, iq = self._max_torque_pair(we_frame, i_lim, v_lim)
            which = self._which_limit_bound(i_d, iq, we_frame, i_lim, v_lim)
            return i_d, sign * max(iq, 0.0), which in ("current", "both"), \
                which in ("voltage", "both")

        i_d0, i_q0 = self.mtpa_pair(tau)
        if not math.isinf(i_q0) and self.mtpa_allows_voltage(tau, w):
            return i_d0, i_q0, False, False

        # Field weakening: the minimum-current pair is not in the interior of
        # the voltage ellipse, it sits on it.  Search for the boundary.
        best = self._weakened_pair_id(tau_mag, w_frame, i_lim, v_lim)
        if best is None:
            # No angle can deliver this torque inside the voltage limit, which
            # means torque_limit() and this search disagree; fall back to the
            # best-effort pair rather than return something silently infeasible.
            i_d, iq = self._max_torque_pair(we_frame, i_lim, v_lim)
            return i_d, sign * iq, True, True
        gain = self.torque_gain_at(best)
        iq = tau_mag / gain if gain > 0 else math.inf
        saturated = math.isfinite(i_lim) and math.hypot(best, iq) >= i_lim * (1.0 - 1.0e-9)
        at_voltage = False
        if math.isfinite(v_lim):
            v_d, v_q = self.voltage_steady(best, iq, w_frame)
            at_voltage = math.hypot(v_d, v_q) >= v_lim * (1.0 - 1.0e-9)
        return best, sign * iq, saturated, at_voltage

    def _pair_feasible(self, i_d: float, i_q: float, rot_vel: float,
                       i_limit: float, v_limit: float) -> bool:
        """Does this pair sit inside both the current circle and the voltage ellipse?"""
        if math.isfinite(i_limit) and self.current_magnitude(i_d, i_q) > i_limit * (1.0 + 1.0e-12):
            return False
        if math.isfinite(v_limit):
            v_d, v_q = self.voltage_steady(i_d, i_q, rot_vel)
            if math.hypot(v_d, v_q) > v_limit * (1.0 + 1.0e-12):
                return False
        return True

    def _weakened_pair_id(self, tau_mag: float, rot_vel: float, i_limit: float,
                          v_limit: float) -> float | None:
        """The ``i_d`` that makes ``tau_mag`` at the smallest current once MTPA is out of reach.

        Above base speed the minimum-current pair is never in the interior of the
        voltage ellipse - it sits *on* it.  For ``i_d < 0`` the current magnitude
        ``sqrt(i_d^2 + (tau/gain(i_d))^2)`` falls monotonically as ``i_d`` rises
        towards zero, so the answer is the largest ``i_d`` that still fits both
        limits: scan down from the top of the allowed range for the first
        feasible point, then bisect back up onto the boundary.  That is cheaper
        and a lot more reliable than an interior optimiser, which will happily
        return a point a few grid steps inside the ellipse and call it converged
        (measured: 8.33 A returned where 7.81 A was available).
        """
        lo = self._id_floor(i_limit)
        hi = min(i_limit, self.characteristic_current) if math.isfinite(i_limit) \
            else self.characteristic_current
        if not hi > lo:
            return None

        def pair(i_d: float) -> tuple[float, float] | None:
            gain = self.torque_gain_at(i_d)
            if gain <= 0.0:
                return None
            return i_d, tau_mag / gain

        step = (hi - lo) / 64.0
        upper = None
        x = hi
        for _ in range(65):
            p = pair(x)
            if p is not None and self._pair_feasible(p[0], p[1], rot_vel, i_limit, v_limit):
                upper = x
                break
            x -= step
        if upper is None:
            return None
        if upper >= hi:
            return hi
        lo_b, hi_b = upper, upper + step        # hi_b is infeasible, or off the range
        for _ in range(80):
            mid = 0.5 * (lo_b + hi_b)
            p = pair(mid)
            if p is not None and self._pair_feasible(p[0], p[1], rot_vel, i_limit, v_limit):
                lo_b = mid
            else:
                hi_b = mid
            if hi_b - lo_b <= 1.0e-12 * max(1.0, abs(lo_b)):
                break
        return lo_b

    def mtpa_allows_voltage(self, torque: float, rot_vel: float) -> bool:
        """True when the MTPA angle for a torque fits inside the voltage limit at
        this speed.  False is the definition of *field weakening required*: the
        drive has to push ``i_d`` past the minimum-current angle to keep the
        back-EMF term small enough to make current flow at all."""
        i_d, i_q = self.mtpa_pair(torque)
        if math.isinf(i_q):
            return False
        if not math.isfinite(self.voltage_limit):
            return True
        v_d, v_q = self.voltage_steady(i_d, i_q, rot_vel)
        return math.hypot(v_d, v_q) <= self.voltage_limit * (1.0 + 1.0e-9)

    def _max_torque_id(self, we: float, i_lim: float, v_lim: float) -> float:
        """Best ``i_d`` for maximum torque.  ``we`` is *signed*: positive is the
        motoring quadrant, negative the regenerative one."""
        lo = self._id_floor(i_lim)
        hi = i_lim if math.isfinite(i_lim) else self.characteristic_current
        if not hi > lo:
            return 0.0

        def tau_of(i_d: float) -> float:
            iq = self._iq_ceiling(i_d, we, i_lim, v_lim)
            if iq < 0.0:
                return -1.0
            return self.torque_gain_at(i_d) * iq

        return _maximise(tau_of, lo, hi)

    def _max_torque_pair(self, we: float, i_lim: float, v_lim: float) -> tuple[float, float]:
        best = self._max_torque_id(we, i_lim, v_lim)
        iq = self._iq_ceiling(best, we, i_lim, v_lim)
        return best, max(0.0, iq if iq is not None and iq >= 0.0 else 0.0)

    def _which_limit_bound(self, i_d: float, i_q: float, we: float,
                           i_lim: float, v_lim: float) -> str:
        circle = math.isfinite(i_lim) and math.hypot(i_d, i_q) >= i_lim * (1.0 - 1.0e-9)
        at_voltage = False
        if math.isfinite(v_lim):
            v_d, v_q = self.voltage_steady(i_d, i_q, we / self.pole_pairs)
            at_voltage = math.hypot(v_d, v_q) >= v_lim * (1.0 - 1.0e-9)
        if circle and at_voltage:
            return "both"
        return "voltage" if at_voltage else "current" if circle else "neither"

    # -- feed-forward ------------------------------------------------------
    @staticmethod
    def _opposes_motion(magnitude: float, rot_vel: float, fallback: float = 0.0) -> float:
        """The torque needed to overcome a passive load of this magnitude.

        A passive load opposes the motion, so the machine has to push with it in
        the direction it is already moving (or, at standstill, in the direction
        it is about to move - ``fallback``, normally the acceleration demand or
        the rest of the torque demand).  ``magnitude`` must be >= 0: an active
        torque that *assists* the motion - a grade, a returning spring - is not a
        load, and the caller adds that one to the result as a signed torque.
        """
        magnitude = float(magnitude)
        if magnitude < 0.0:
            raise ConfigError(
                "load torque is a magnitude that opposes motion and must be >= 0; "
                "an active torque that assists the motion (a grade, a spring) has "
                "to be added to required_torque() as a signed torque"
            )
        direction = rot_vel if abs(rot_vel) > STICTION_BAND else fallback
        return magnitude if direction >= 0.0 else -magnitude

    def required_torque(self, rot_vel: float, accel: float = 0.0,
                        load_torque: float = 0.0) -> float:
        """Electromagnetic torque the machine must make for a motion command [N*m].

        ``load_torque`` is everything on the shaft that is not the rotor's own
        inertia, bearing damping or stiction - gearbox losses, a winch drum,
        aerodynamic drag - given as a *magnitude that opposes motion*, the same
        convention :class:`~drivetrain.load.LoadModel` uses.  Bearing damping is
        already a signed term here, so it is not doubled.  At standstill, with no
        acceleration demand either, the result is the torque needed to break away
        in the positive direction; ask with an acceleration of the sign you mean
        to move and you get the break-away torque the other way.
        """
        w = float(rot_vel)
        acceleration = float(accel)
        tau = self.rotor_inertia * acceleration + self.bearing_damping * w
        tau += self._opposes_motion(float(load_torque), w, acceleration)
        if self.static_friction:
            tau += self._opposes_motion(self.static_friction, w, tau)
        return tau

    def feedforward(self, rot_vel: float, accel: float = 0.0, load_torque: float = 0.0,
                    di_d_dt: float = 0.0, di_q_dt: float = 0.0) -> DqCommand:
        """Voltage vector for a motion command -- the FOC feed-forward term.

        The acceleration term has to be the derivative of the *rate-limited*
        reference, exactly as in the brushed export: asking for an instantaneous
        speed change asks for infinite current, and the answer comes back
        clamped rather than correct.
        """
        w = float(rot_vel)
        tau_req = self.required_torque(w, accel, load_torque)
        i_d, i_q, cur_sat, volt_sat = self.current_pair(tau_req, w)
        tau = self.torque_at(i_d, i_q)
        v_d, v_q = self.voltage_for(i_d, i_q, w, di_d_dt, di_q_dt)
        if math.isfinite(self.voltage_limit):
            mag = math.hypot(v_d, v_q)
            if mag > self.voltage_limit:
                scale = self.voltage_limit / mag
                v_d, v_q = v_d * scale, v_q * scale
                volt_sat = True
        rail = 0.0
        if math.isfinite(self.dc_bus_voltage) and self.dc_bus_voltage > 0:
            rail = 1.5 * (v_d * i_d + v_q * i_q) / self.dc_bus_voltage
        return DqCommand(
            i_d=i_d, i_q=i_q, v_d=v_d, v_q=v_q, torque=tau,
            current_saturated=cur_sat, voltage_saturated=volt_sat,
            field_weakening=self.mtpa_allows_voltage(tau_req, w) is False,
            dc_link_current=rail, speed=w,
        )

    # -- losses and capability -------------------------------------------
    def copper_power(self, i_d: float | None = None, i_q: float | None = None) -> float:
        i_d = self.current_d if i_d is None else float(i_d)
        i_q = self.current_q if i_q is None else float(i_q)
        return 1.5 * self.resistance * (i_d * i_d + i_q * i_q)

    def core_loss_power(self, rot_vel: float | None = None) -> float:
        """Iron loss, modelled as ``k_e * w_e**2`` (loss only, no dq current component)."""
        we = abs(self.electrical_speed(self.rot_vel if rot_vel is None else rot_vel))
        return self.core_loss_coefficient * we * we

    def windage_power(self, rot_vel: float | None = None) -> float:
        w = abs(float(self.rot_vel if rot_vel is None else rot_vel))
        return self.windage_coefficient * w ** 3

    def input_power(self, v_d: float, v_q: float, i_d: float, i_q: float) -> float:
        """DC-link power for a lossless inverter: ``1.5*(v_d*i_d + v_q*i_q)`` [W]."""
        return 1.5 * (v_d * i_d + v_q * i_q)

    def efficiency_at(self, rot_vel: float, torque: float) -> float | None:
        """Output / input power at a (speed, torque) point, or None if idle.

        Copper loss is exact; iron loss is the simple ``k_e*w_e^2`` term, so this
        is only as good as that term is honest.
        """
        w = float(rot_vel)
        i_d, i_q, _, _ = self.current_pair(torque, w)
        v_d, v_q = self.voltage_steady(i_d, i_q, w)
        pin = self.input_power(v_d, v_q, i_d, i_q)
        if pin <= 0:
            return None
        # loss_torque() already charges bearing, windage and iron, and it is
        # signed with the direction of motion, so the power those losses take is
        # its magnitude times the speed.  Charging only bearing and windage here
        # over-reported efficiency on any machine with a non-zero
        # core_loss_coefficient: the iron term was invisible.
        useful = torque * w - self.loss_torque(w) * abs(w)
        return max(0.0, min(1.0, useful / pin))

    def torque_speed_curve(self, n: int = 64, top_speed: float | None = None):
        """``(speed_rad_s, torque_Nm)`` capability envelope at the machine's limits.

        The shape is the one every PMSM datasheet draws: flat until base speed
        (constant current), then the field-weakening hyperbola, then whatever is
        left after the losses.  ``top_speed`` defaults to :meth:`max_speed`,
        which needs a loss term to be finite; when the machine is lossless the
        curve stops at three times base speed and says so by being truncated.
        """
        import numpy as np

        top = top_speed if top_speed is not None else self.max_speed()
        if not math.isfinite(top):
            top = 3.0 * self.base_speed()
        w = np.linspace(0.0, float(top), int(n))
        tau = np.array([self.torque_limit(float(x)) for x in w])
        return w, tau


# ---------------------------------------------------------------------------
# the machine wearing the brushed-motor interface
# ---------------------------------------------------------------------------

@dataclass
class PMSMModel:
    """A brushless machine that speaks the :class:`~drivetrain.motor.MotorModel`
    interface, so the rest of the toolkit (gearbox, load, wheel, vehicle,
    simulation, sweeps, export) works on a brushless drivetrain unchanged.

    Under FOC with ``i_d = 0`` -- which is what a drive does from standstill up
    to base speed, and where a feed-forward controller spends almost all of its
    time -- the dq equations collapse onto exactly the brushed equations this
    toolkit is built around::

        v_q = R*i_q + Lq*di_q/dt + p*lambda*w        tau = 1.5*p*lambda*i_q

        v = R*i + L*di/dt + kv*w              (brushed)
        tau = kt*i

    so the mapping below is one-to-one, not an approximation:

    ===============  ==========================================
    brushed quantity  brushless origin
    ===============  ==========================================
    ``kt``           ``1.5 * p * lambda``
    ``kv``           ``p * lambda``
    ``resistance``   phase resistance (per machine)
    ``inductance``   ``Lq`` (the axis that carries the torque current)
    ``rotor_inertia``/``bearing_damping``/``static_friction``  as configured
    ``max_current``  phase-current limit (per machine, peak)
    ``max_voltage``  phase-voltage limit (peak fundamental, not the DC bus)
    ===============  ==========================================

    ``kt / kv == 1.5`` is the amplitude-invariant Park convention (see the module
    docstring), not a mismatch between the two constants.

    The collapse is exact while the drive is current-limited and the voltage
    ellipse is slack, and it stops being valid above base speed, where a real FOC
    drive demagnetises the field to keep the current inside the inverter.  What
    makes this safe rather than silently wrong: the brushed model's no-load speed
    is ``max_voltage / kv``, which under the mapping above is
    ``voltage_limit / (p*lambda)`` -- the base speed, exactly.  A drivetrain
    assembled this way therefore saturates at base speed instead of inventing a
    field-weakening region it does not model.  Check an operating point with
    :meth:`above_base_speed`, and use :class:`PMSMSpec` directly when field
    weakening matters.

    Not carried across: reluctance torque (an ``i_d = 0`` drive gets none of it),
    iron and windage loss beyond ``bearing_damping``, magnet temperature drift,
    and the difference between phase current and DC-link current (see
    :meth:`dc_link_current`).
    """

    spec: PMSMSpec = field(default_factory=PMSMSpec)
    current: float = 0.0      # phase current amplitude [A] (state)
    rot_vel: float = 0.0      # mechanical shaft speed [rad/s] (state)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PMSMSpec):
            raise ConfigError("PMSMModel needs a PMSMSpec.")
        self.spec.validate()

    # -- the brushed view -------------------------------------------------
    @property
    def kt(self) -> float:
        """Torque per ampere of q-axis current ``1.5*p*lambda`` [N*m/A]."""
        return self.spec.torque_constant

    @property
    def kv(self) -> float:
        """Back-EMF per mechanical rad/s, ``p*lambda`` [V/(rad/s)]."""
        return self.spec.back_emf_constant

    @property
    def resistance(self) -> float:
        """Phase resistance per machine [Ohm]."""
        return self.spec.resistance

    @property
    def inductance(self) -> float:
        """``Lq`` [H]: the axis carrying the torque current sets the dynamics."""
        return self.spec.inductance_q

    @property
    def rotor_inertia(self) -> float:
        return self.spec.rotor_inertia

    @property
    def bearing_damping(self) -> float:
        return self.spec.bearing_damping

    @property
    def static_friction(self) -> float:
        return self.spec.static_friction

    @property
    def max_current(self) -> float:
        """Phase-current limit per machine, peak [A]."""
        return self.spec.current_limit

    @property
    def max_voltage(self) -> float:
        """Phase-voltage limit, peak fundamental [V] -- not the DC bus."""
        return self.spec.voltage_limit

    @property
    def no_load_friction_current(self) -> float:
        """Always 0: this model carries mechanical loss as damping, not as a
        datasheet no-load current, so there is nothing to fit here."""
        return 0.0

    # -- characteristics (same shapes as MotorModel) ----------------------
    @property
    def rpm(self) -> float:
        return rad_s_to_rpm(self.rot_vel)

    @property
    def back_emf(self) -> float:
        """Line-to-neutral back-EMF at the current speed [V peak]."""
        return self.kv * self.rot_vel

    @property
    def damping_coefficient(self) -> float:
        """Bearing damping plus the electromechanical term ``kt*kv/R``."""
        return self.bearing_damping + (self.kt * self.kv) / self.resistance

    @property
    def no_load_speed_per_volt(self) -> float:
        return 1.0 / self.kv if self.kv else math.inf

    @property
    def stall_torque_per_volt(self) -> float:
        return self.kt / self.resistance

    @property
    def stall_current_per_volt(self) -> float:
        return 1.0 / self.resistance

    @property
    def mech_time_constant(self) -> float:
        """``J*R/(kt*kv)`` [s] -- the same number the dq model implies at i_d=0."""
        denom = self.kt * self.kv
        return self.rotor_inertia * self.resistance / denom if denom else math.inf

    @property
    def electrical_time_constant(self) -> float:
        """``Lq/R`` [s]."""
        return self.inductance / self.resistance if self.resistance else 0.0

    @property
    def ideal_no_load_rpm(self) -> float:
        """No-load speed at the phase-voltage limit [rpm] -- i.e. base speed."""
        v = 0.0 if math.isinf(self.max_voltage) else self.max_voltage
        return rad_s_to_rpm(v / self.kv) if self.kv and v else math.inf

    def torque_at(self, voltage: float, speed: float) -> float:
        """Quasi-static shaft torque at a phase voltage and speed [N*m]."""
        return self.stall_torque_per_volt * (voltage - self.kv * speed)

    def current_at(self, voltage: float, speed: float) -> float:
        """Quasi-static phase-current amplitude at a voltage and speed [A]."""
        return (voltage - self.kv * speed) / self.resistance

    def thermal_power(self, current: float | None = None) -> float:
        """Copper loss per machine: ``1.5 * I^2 * R`` [W].

        The 1.5 is the three-phase sum in the amplitude-invariant convention
        (``1.5 * R * (i_d^2 + i_q^2)`` with ``i_d = 0``); a brushed winding's
        ``I^2 R`` has no such factor.
        """
        i = self.current if current is None else current
        return 1.5 * i * i * self.resistance

    # -- dq helpers worth having on this view -----------------------------
    def base_speed(self) -> float:
        """Mechanical speed where the back-EMF reaches the phase-voltage limit."""
        return self.spec.base_speed()

    def above_base_speed(self, rot_vel: float | None = None) -> bool:
        """True when this operating point needs field weakening, i.e. when the
        brushed view of the machine is no longer valid."""
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        return abs(w) > self.base_speed()

    def dc_link_current(self, current: float | None = None,
                        rot_vel: float | None = None) -> float:
        """Current drawn from the DC supply [A], which is *not* the phase current.

        Power balance at ``i_d = 0``: the bus supplies the copper loss
        (``1.5*R*i**2``) plus the converted mechanical power
        (``1.5*p*lambda*w*i``), so ``I_bus = 1.5*(R*i + p*lambda*w) * i / V_bus``.
        The 1.5 is the same amplitude-invariant Park factor that puts 1.5 in
        ``kt``: the phase current that makes the torque is *not* the current the
        windings draw from the supply.  Returns ``nan`` when
        ``spec.dc_bus_voltage`` is not configured, because there is then no bus
        to talk about.  An ammeter on the supply rail of a FOC drive reads this
        number, not the winding current that makes the torque -- which is why
        fitting a brushless drive from DC-link current alone is a different (and
        harder) problem than fitting a brushed one.
        """
        i = self.current if current is None else float(current)
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        bus = self.spec.dc_bus_voltage
        if not math.isfinite(bus) or bus <= 0:
            return math.nan
        return 1.5 * (self.resistance * i + self.kv * w) * i / bus

    # -- where the collapse stops being exact ---------------------------
    def current_ceiling(self, rot_vel: float | None = None) -> float:
        """Phase current a dq drive can still hold at this speed with ``i_d = 0``.

        The inverter spends its voltage on both axes: holding ``i_q`` costs the
        resistive drop and the back-EMF on the q-axis *and* ``w_e*Lq*i_q`` on the
        d-axis, and both come out of the same budget.  The brushed collapse only
        charges itself for the q-axis, so above this speed it believes the drive
        can push more current than it can.
        """
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        we = self.spec.electrical_speed(w)
        iq = self.spec._iq_from_voltage(0.0, we, self.spec.voltage_limit)
        if iq is None or iq <= 0.0:
            return 0.0
        return min(self.spec.current_limit, float(iq))

    def voltage_needed(self, current: float | None = None,
                       rot_vel: float | None = None) -> float:
        """Phase voltage magnitude needed to hold ``current`` at ``rot_vel``."""
        i = self.current if current is None else float(current)
        w = self.rot_vel if rot_vel is None else float(rot_vel)
        v_d, v_q = self.spec.voltage_steady(0.0, i, w)
        return math.hypot(v_d, v_q)

    def constant_torque_speed(self) -> float:
        """Speed at which :meth:`current_ceiling` first drops below ``current_limit``.

        ``|(R*I + w_e*lambda, w_e*Lq*I)| = voltage_limit`` is a quadratic in
        ``w_e``: ``a*w_e^2 + b*w_e + c = 0`` with ``a = (Lq*I)^2 + lambda^2``,
        ``b = 2*lambda*R*I``, ``c = (R*I)^2 - V^2``.  The positive root is the edge
        of the constant-torque band.
        """
        s = self.spec
        i = s.current_limit
        lam = s.effective_flux()
        a = (s.inductance_q * i) ** 2 + lam ** 2
        b = 2.0 * lam * s.resistance * i
        c = (s.resistance * i) ** 2 - s.voltage_limit ** 2
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return 0.0
        return (math.sqrt(disc) - b) / (2.0 * a) / s.pole_pairs

    def view_constant_torque_speed(self) -> float:
        """The same band edge as the brushed collapse computes it: ``(V - R*I)/kv``."""
        return max(0.0, (self.max_voltage - self.resistance * self.max_current) / self.kv)

    def collapse_margin(self) -> float:
        """How much too wide the collapse's constant-torque band is, as a fraction.

        ``0.0`` means the two agree.  The margin is set by how large the
        decoupling voltage ``w_e*Lq*I`` is next to the back-EMF
        ``w_e*lambda`` - i.e. by the ratio ``Lq*I/lambda``, plus the resistive
        drop ``R*I`` on top of it - and *not* by saliency: a non-salient machine
        with ``Lq = Ld = 2e-4``, ``lambda = 1.8e-3`` and ``I = 10 A`` has
        ``Lq*I/lambda = 1.1`` and a 32% margin, while the same machine with
        ``Lq = 1e-6`` (``Lq*I/lambda = 0.01``) has none at all.  It is the number
        to look at before trusting a brushed-model simulation of a brushless
        drive near the top of its speed range.  It says nothing about the region
        above base speed, which the collapse cannot represent at all.
        """
        dq = self.constant_torque_speed()
        view = self.view_constant_torque_speed()
        if view <= 0.0:
            return 0.0
        return (view - dq) / view

    def equivalent_note(self) -> str:
        """One-line statement of what this view is and is not good for."""
        dq = self.constant_torque_speed()
        view = self.view_constant_torque_speed()
        band = ""
        if view > 0.0 and dq < 0.98 * view:
            band = (f", full current only reachable to {dq:.1f} rad/s rather than the "
                    f"{view:.1f} rad/s the collapse implies")
        return (f"i_d=0 collapse of a {self.spec.pole_pairs}-pole-pair PMSM: "
                f"kt={self.kt:.6g}, kv={self.kv:.6g} (kt/kv=1.5 by convention), "
                f"exact below base speed {self.base_speed():.1f} rad/s "
                f"({rad_s_to_rpm(self.base_speed()):.0f} rpm){band}; no field weakening")

    # -- plumbing ---------------------------------------------------------
    def validate(self) -> None:
        self.spec.validate()

    def copy(self) -> PMSMModel:
        return replace(self)
