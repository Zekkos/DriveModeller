"""Named starting configurations, and (de)serialisation of a whole system.

Use :func:`build` from the CLI/GUI to turn a preset name plus a flat dict of
overrides into a :class:`~drivetrain.system.DriveSystem`.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .exceptions import ConfigError
from .gearbox import Gearbox, RatioConvention
from .load import RotationalLoad, VehicleLoad, Wheel
from .motor import MotorGroup, MotorModel
from .pmsm import PMSMModel, PMSMSpec
from .sweep import set_path
from .system import DriveSystem

# ----------------------------------------------------------------------
# motor catalogue
# ----------------------------------------------------------------------
MOTORS: dict[str, MotorModel] = {
    # Default: C23-L33 gearmotor, winding code 20 (see motor.py for the source).
    "c23_l33_w20": MotorModel(
        kv=0.0301, kt=0.0300, resistance=1.0, rotor_inertia=1.554e-5,
        bearing_damping=1e-6, max_voltage=12.0, max_current=12.0,
    ),
    # Same family, different winding: lower kv => faster, less torque.
    "c23_l33_w12": MotorModel(
        kv=0.0180, kt=0.0180, resistance=0.62, rotor_inertia=1.554e-5,
        bearing_damping=1e-6, max_voltage=12.0, max_current=20.0,
    ),
    # Generic 12 V brushed "FAULHAUER-ish" core, small and low inertia.
    "core_12v": MotorModel(
        kv=0.0060, kt=0.0060, resistance=0.35, rotor_inertia=2.0e-7,
        bearing_damping=5e-8, max_voltage=12.0, max_current=8.0,
    ),
    # Bigger 24 V motor for heavier pods.
    "big_24v": MotorModel(
        kv=0.0120, kt=0.0120, resistance=0.15, rotor_inertia=2.0e-5,
        bearing_damping=5e-7, max_voltage=24.0, max_current=40.0,
    ),
    # Idealised lossless-ish motor, handy for sanity checks.
    "ideal": MotorModel(
        kv=0.05, kt=0.05, resistance=0.5, rotor_inertia=1e-5,
        bearing_damping=0.0, max_voltage=12.0, max_current=float("inf"),
    ),
    # Brushless machines.  These reach the brushed pipeline through the i_d=0
    # collapse (kt = 1.5*p*lambda, kv = p*lambda, L = Lq), which is exact below
    # the constant-torque band edge and cannot represent field weakening above
    # it.  Use PMSMSpec directly, or export the dq model, past that edge.
    "bldc_24v": PMSMModel(PMSMSpec(
        resistance=0.12, inductance_d=1.5e-4, inductance_q=1.8e-4, flux_linkage=1.8e-3,
        pole_pairs=7, rotor_inertia=2.0e-5, bearing_damping=1.0e-6,
        current_limit=20.0, voltage_limit=24.0 / math.sqrt(3.0), dc_bus_voltage=24.0,
    )),
    "bldc_12v": PMSMModel(PMSMSpec(
        resistance=0.35, inductance_d=6.0e-5, inductance_q=8.0e-5, flux_linkage=9.0e-4,
        pole_pairs=10, rotor_inertia=4.0e-6, bearing_damping=2.0e-7,
        current_limit=8.0, voltage_limit=12.0 / math.sqrt(3.0), dc_bus_voltage=12.0,
    )),
}

WHEELS: dict[str, Wheel] = {
    "tiny": Wheel(radius=0.025, mass=0.05),
    "small": Wheel(radius=0.05, mass=0.10),
    "medium": Wheel(radius=0.0825, mass=0.25),
    "large": Wheel(radius=0.10, mass=0.50),
}


@dataclass
class Preset:
    """A named, ready-to-simulate configuration."""

    name: str
    description: str
    system: DriveSystem
    tags: tuple[str, ...] = ()

    def copy(self) -> DriveSystem:
        return self.system.copy()


def _rotational_pod(motors: int, ratio: float, wheel: Wheel, mass_disk: float) -> DriveSystem:
    """Legacy-style pod: motor -> gearbox -> spinning disk (no ground contact)."""
    return DriveSystem(
        group=MotorGroup(MOTORS["c23_l33_w20"].copy(), motors),
        gearbox=Gearbox(ratio=ratio, efficiency=0.9, inertia=0.0),
        load=RotationalLoad(
            inertia=0.5 * mass_disk * wheel.radius**2,
            viscous=2.0e-4,
            constant_torque=0.0,
        ),
    )


PRESETS: dict[str, Preset] = {
    "legacy": Preset(
        name="legacy",
        description="Original script: 1 motor, direct drive, 1 kg 50 mm disk, 12 V.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["c23_l33_w20"].copy(), 1),
            gearbox=Gearbox(ratio=1.0, efficiency=1.0),
            load=RotationalLoad(inertia=1.25e-3, viscous=2.0e-4, constant_torque=0.0),
        ),
        tags=("legacy", "rotational"),
    ),
    "legacy_4x": Preset(
        name="legacy_4x",
        description="Legacy load driven by 4 parallel motors (the main.py comparison).",
        system=_rotational_pod(4, 1.0, WHEELS["small"], 1.0),
        tags=("legacy", "rotational", "multi-motor"),
    ),
    "pod_4x_geared": Preset(
        name="pod_4x_geared",
        description="4 C23-L33 motors, 4:1 reduction, 8 kg pod on 100 mm wheels.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["c23_l33_w20"].copy(), 4),
            gearbox=Gearbox(ratio=0.25, efficiency=0.9, inertia=2e-5),
            load=VehicleLoad(
                mass=8.0, wheel=WHEELS["large"], wheels_driven=2,
                rolling_resistance=0.02, drag_cd=0.5,
            ),
        ),
        tags=("vehicle", "pod"),
    ),
    "pod_2x_direct": Preset(
        name="pod_2x_direct",
        description="2 motors direct-drive on a light 3 kg pod, 50 mm wheels.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["c23_l33_w12"].copy(), 2),
            gearbox=Gearbox(ratio=1.0, efficiency=0.95),
            load=VehicleLoad(mass=3.0, wheel=WHEELS["small"], wheels_driven=2, drag_cd=0.25),
        ),
        tags=("vehicle", "pod", "direct"),
    ),
    "sprinter": Preset(
        name="sprinter",
        description="High-speed 24 V pod: big motor, 6:1 up... reduction, 165 mm wheels.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["big_24v"].copy(), 2),
            gearbox=Gearbox(ratio=0.30, efficiency=0.88, inertia=6e-5),
            load=VehicleLoad(
                mass=14.0, wheel=WHEELS["large"], wheels_driven=4,
                rolling_resistance=0.015, drag_cd=0.9,
            ),
        ),
        tags=("vehicle", "fast"),
    ),
    "winch": Preset(
        name="winch",
        description="High reduction hoist/winch: torque out, speed irrelevant.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["c23_l33_w20"].copy(), 2),
            gearbox=Gearbox(ratio=0.02, efficiency=0.75, inertia=1e-4, damping=2e-4),
            load=RotationalLoad(inertia=5e-3, viscous=1e-3, constant_torque=1.5),
        ),
        tags=("rotational", "torque"),
    ),
    "spinning_mass": Preset(
        name="spinning_mass",
        description="Reaction-wheel / flywheel spin-up test.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["core_12v"].copy(), 1),
            gearbox=Gearbox(ratio=1.0, efficiency=0.95),
            load=RotationalLoad(inertia=2e-3, viscous=5e-5, constant_torque=0.0),
        ),
        tags=("rotational", "flywheel"),
    ),
    "bldc_pod": Preset(
        name="bldc_pod",
        description="Brushless pod: 2 PMSMs, 6:1 reduction, 8 kg on 100 mm wheels, 24 V bus.",
        system=DriveSystem(
            group=MotorGroup(MOTORS["bldc_24v"].copy(), 2),
            gearbox=Gearbox(ratio=0.167, efficiency=0.9, inertia=2e-5),
            load=VehicleLoad(
                mass=8.0, wheel=WHEELS["large"], wheels_driven=2,
                rolling_resistance=0.02, drag_cd=0.5,
            ),
        ),
        tags=("vehicle", "pod", "brushless"),
    ),
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def motor_names() -> list[str]:
    return sorted(MOTORS)


def wheel_names() -> list[str]:
    return sorted(WHEELS)


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ConfigError(f"Unknown preset {name!r}; choose from {preset_names()}.")
    return PRESETS[name]


def build(name: str = "legacy", overrides: dict[str, Any] | None = None) -> DriveSystem:
    """Build a system from a preset plus dotted-path overrides.

    ``overrides`` keys are dotted paths (``"gearbox.ratio"``, ``"load.mass"``,
    ``"group.motor.resistance"`` ...).  A special ``"motor"`` key swaps in a
    catalogue motor by name, and ``"wheel"`` swaps in a catalogue wheel.
    """
    system = get_preset(name).copy()
    overrides = dict(overrides or {})
    if "motor" in overrides:
        key = overrides.pop("motor")
        if key not in MOTORS:
            raise ConfigError(f"Unknown motor {key!r}; choose from {motor_names()}.")
        count = system.group.count
        system = replace(system, group=MotorGroup(MOTORS[key].copy(), count))
    if "wheel" in overrides:
        key = overrides.pop("wheel")
        if key not in WHEELS:
            raise ConfigError(f"Unknown wheel {key!r}; choose from {wheel_names()}.")
        if not hasattr(system.load, "wheel"):
            raise ConfigError("The 'wheel' shortcut only applies to vehicle loads.")
        system = set_path(system, "load.wheel", WHEELS[key].copy())
    for path, value in overrides.items():
        system = set_path(system, path, value)
    system.__post_init__()
    return system


# ----------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------
def to_dict(system: DriveSystem) -> dict[str, Any]:
    """Plain-dict form of a system (JSON-safe, round-trips through :func:`from_dict`)."""
    return {
        "group": {
            "count": system.group.count,
            "motor": {
                "kind": type(system.group.motor).__name__,
                **asdict(system.group.motor),
            },
        },
        "gearbox": asdict(system.gearbox),
        "load": {
            "kind": type(system.load).__name__,
            **asdict(system.load),
        },
    }


def _inf_safe(obj: Any) -> Any:
    """Make a nested structure JSON-safe (enums, NaN, +/-inf)."""
    if isinstance(obj, RatioConvention):
        return obj.value
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: _inf_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_inf_safe(v) for v in obj]
    return obj


def to_json(system: DriveSystem, indent: int = 2) -> str:
    return json.dumps(_inf_safe(to_dict(system)), indent=indent)


def _clean(mapping: dict[str, Any]) -> dict[str, Any]:
    """Undo the JSON encoding of ``None``/``inf``/``nan`` from :func:`_inf_safe`."""
    out = {}
    for k, v in mapping.items():
        if v is None:
            continue
        if v == "inf":
            v = float("inf")
        elif v == "-inf":
            v = float("-inf")
        elif v == "nan":
            v = float("nan")
        out[k] = v
    return out


def _motor_from_dict(data: dict[str, Any]) -> Any:
    """Rebuild a motor from its serialised form, dispatching on ``kind``.

    A brushless machine is stored as ``{"kind": "PMSMModel", "spec": {...}}``:
    the dq specification is the truth, and the brushed-looking ``kt``/``kv`` the
    rest of the toolkit runs on are derived from it, so they are not stored and
    cannot drift out of step with it.
    """
    kind = data.pop("kind", "MotorModel")
    if kind == "MotorModel":
        return MotorModel(**data)
    if kind == "PMSMModel":
        spec_data = _clean(data.pop("spec", {}))
        return PMSMModel(spec=PMSMSpec(**spec_data), **data)
    raise ConfigError(f"unknown motor kind {kind!r}; expected MotorModel or PMSMModel")


def from_dict(data: dict[str, Any]) -> DriveSystem:
    """Rebuild a system from :func:`to_dict` output."""
    clean = _clean

    motor_data = clean(data["group"]["motor"])
    motor = _motor_from_dict(motor_data)
    group = MotorGroup(motor, int(data["group"]["count"]))
    gb_data = clean(data["gearbox"])
    gb_data["convention"] = RatioConvention(gb_data.get("convention", "speed_ratio"))
    gearbox = Gearbox(**gb_data)

    load_data = clean(data["load"])
    kind = load_data.pop("kind", "RotationalLoad")
    if load_data.get("wheel") is not None:
        load_data["wheel"] = Wheel(**load_data["wheel"])
    cls = {"RotationalLoad": RotationalLoad, "VehicleLoad": VehicleLoad}[kind]
    load = cls(**load_data)
    return DriveSystem(group=group, gearbox=gearbox, load=load)


def from_json(text: str) -> DriveSystem:
    return from_dict(json.loads(text))


def save(path: str, system: DriveSystem) -> None:
    with open(path, "w") as fh:
        fh.write(to_json(system))


def load(path: str) -> DriveSystem:
    with open(path) as fh:
        return from_json(fh.read())


# ----------------------------------------------------------------------
# parameter metadata for the UI
# ----------------------------------------------------------------------
# Slider ranges are deliberately generous: the point of the UI is exploring
# what-if designs, not just nudging around the preset default.  Anything not
# listed here gets an auto range in :func:`param_meta`.
PARAM_META: dict[str, dict[str, Any]] = {
    # -- motor ---------------------------------------------------------
    "group.motor.kv":            {"label": "Velocity constant kv", "unit": "s/rad",  "min": 0.002,  "max": 0.30,   "step": 0.0005, "group": "Motor"},
    "group.motor.kt":            {"label": "Torque constant kt",   "unit": "N*m/A",  "min": 0.001,  "max": 0.30,   "step": 0.0005, "group": "Motor"},
    "group.motor.resistance":    {"label": "Winding resistance",   "unit": "ohm",    "min": 0.01,   "max": 5.0,    "step": 0.01,   "group": "Motor"},
    "group.motor.inductance":    {"label": "Winding inductance",   "unit": "H",      "min": 0.0,    "max": 0.001,  "step": 1e-6,   "group": "Motor"},
    "group.motor.rotor_inertia": {"label": "Rotor inertia",        "unit": "kg*m^2", "min": 0.0,    "max": 0.001,  "step": 1e-6,   "group": "Motor"},
    "group.motor.bearing_damping":{"label": "Bearing damping",     "unit": "N*m*s",  "min": 0.0,    "max": 0.001,  "step": 1e-6,   "group": "Motor"},
    "group.motor.static_friction":{"label": "Static friction",     "unit": "N*m",    "min": 0.0,    "max": 0.05,   "step": 0.001,  "group": "Motor"},
    "group.motor.max_voltage":   {"label": "Max drive voltage",    "unit": "V",      "min": 1.0,    "max": 60.0,   "step": 0.5,    "group": "Motor"},
    "group.motor.max_current":   {"label": "Current limit (each)", "unit": "A",      "min": 0.5,    "max": 200.0,  "step": 0.5,    "group": "Motor"},
    "group.motor.no_load_friction_current":{"label": "No-load friction current","unit": "A", "min": 0.0, "max": 2.0, "step": 0.01, "group": "Motor"},
    "group.motor.rot_vel":       {"label": "Initial speed",        "unit": "rad/s",  "min": -500.0, "max": 1500.0, "step": 5.0,    "group": "Motor"},
    # brushless (dq spec) parameters - listed only when the motor is a PMSMModel
    "group.motor.spec.resistance": {"label": "Phase resistance", "unit": "ohm", "min": 0.005, "max": 5.0, "step": 0.005, "group": "Motor (brushless)"},
    "group.motor.spec.inductance_d": {"label": "d-axis inductance", "unit": "H", "min": 0.0, "max": 0.002, "step": 1e-6, "group": "Motor (brushless)"},
    "group.motor.spec.inductance_q": {"label": "q-axis inductance", "unit": "H", "min": 0.0, "max": 0.002, "step": 1e-6, "group": "Motor (brushless)"},
    "group.motor.spec.flux_linkage": {"label": "Magnet flux linkage", "unit": "Wb", "min": 0.0, "max": 0.02, "step": 1e-5, "group": "Motor (brushless)"},
    "group.motor.spec.pole_pairs": {"label": "Pole pairs", "unit": "", "min": 1, "max": 30, "step": 1, "group": "Motor (brushless)"},
    "group.motor.spec.current_limit": {"label": "Current limit (peak, each)", "unit": "A", "min": 0.5, "max": 200.0, "step": 0.5, "group": "Motor (brushless)"},
    "group.motor.spec.voltage_limit": {"label": "Phase voltage limit (peak)", "unit": "V", "min": 1.0, "max": 100.0, "step": 0.5, "group": "Motor (brushless)"},
    "group.motor.spec.dc_bus_voltage": {"label": "DC bus voltage", "unit": "V", "min": 1.0, "max": 100.0, "step": 0.5, "group": "Motor (brushless)"},
    "group.motor.spec.rotor_inertia": {"label": "Rotor inertia", "unit": "kg*m^2", "min": 0.0, "max": 0.001, "step": 1e-6, "group": "Motor (brushless)"},
    "group.motor.spec.bearing_damping": {"label": "Bearing damping", "unit": "N*m*s", "min": 0.0, "max": 0.001, "step": 1e-6, "group": "Motor (brushless)"},
    "group.motor.spec.static_friction": {"label": "Static friction", "unit": "N*m", "min": 0.0, "max": 0.05, "step": 0.001, "group": "Motor (brushless)"},
    "group.motor.spec.core_loss_coefficient": {"label": "Iron loss coefficient", "unit": "W/(rad/s elec)^2", "min": 0.0, "max": 1e-6, "step": 1e-10, "group": "Motor (brushless)"},
    "group.motor.spec.windage_coefficient": {"label": "Windage coefficient", "unit": "W/(rad/s)^3", "min": 0.0, "max": 1e-6, "step": 1e-10, "group": "Motor (brushless)"},
    "group.motor.spec.flux_temperature_coefficient": {"label": "Flux temp coefficient", "unit": "1/K", "min": -0.005, "max": 0.0, "step": 1e-5, "group": "Motor (brushless)"},
    # -- gearbox -------------------------------------------------------
    "gearbox.ratio":             {"label": "Gear ratio",           "unit": "out/in", "min": 0.02,   "max": 2.0,    "step": 0.005,  "group": "Gearbox"},
    "gearbox.efficiency":        {"label": "Gear efficiency",      "unit": "",       "min": 0.30,   "max": 1.0,    "step": 0.005,  "group": "Gearbox"},
    "gearbox.inertia":           {"label": "Gear inertia",         "unit": "kg*m^2", "min": 0.0,    "max": 0.001,  "step": 1e-6,   "group": "Gearbox"},
    "gearbox.damping":           {"label": "Gear loss damping",    "unit": "N*m*s",  "min": 0.0,    "max": 0.001,  "step": 1e-6,   "group": "Gearbox"},
    # -- load ----------------------------------------------------------
    "load.inertia":              {"label": "Load inertia",         "unit": "kg*m^2", "min": 0.0,    "max": 0.05,   "step": 0.0001, "group": "Load"},
    "load.viscous":              {"label": "Viscous drag",         "unit": "N*m*s",  "min": 0.0,    "max": 0.01,   "step": 1e-5,   "group": "Load"},
    "load.constant_torque":      {"label": "Constant load torque", "unit": "N*m",    "min": 0.0,    "max": 1.0,    "step": 0.005,  "group": "Load"},
    "load.mass":                 {"label": "Vehicle mass",         "unit": "kg",     "min": 0.1,    "max": 60.0,   "step": 0.1,    "group": "Load"},
    "load.wheels_driven":        {"label": "Wheels driven",        "unit": "",       "min": 1,      "max": 8,      "step": 1,      "group": "Load"},
    "load.rolling_resistance":   {"label": "Rolling resistance",   "unit": "",       "min": 0.0,    "max": 0.20,   "step": 0.001,  "group": "Load"},
    "load.drag_cd":              {"label": "Aero drag Cd*A",       "unit": "m^2",    "min": 0.0,    "max": 1.0,    "step": 0.005,  "group": "Load"},
    "load.air_density":          {"label": "Air density",          "unit": "kg/m^3", "min": 0.5,    "max": 1.5,    "step": 0.01,   "group": "Load"},
    "load.grade_deg":            {"label": "Grade",                "unit": "deg",    "min": -30.0,  "max": 30.0,   "step": 0.5,    "group": "Load"},
    "load.extra_rotational_inertia":{"label": "Extra rotating inertia","unit":"kg*m^2","min": 0.0,   "max": 0.05,   "step": 0.0001, "group": "Load"},
    "load.gravity":              {"label": "Gravity",              "unit": "m/s^2",  "min": 1.6,    "max": 24.8,   "step": 0.1,    "group": "Load"},
    "load.final_drive_efficiency":{"label": "Final drive eff.",    "unit": "",       "min": 0.5,    "max": 1.0,    "step": 0.005,  "group": "Load"},
    "load.wheel.radius":         {"label": "Wheel radius",         "unit": "m",      "min": 0.005,  "max": 0.25,   "step": 0.001,  "group": "Wheel"},
    "load.wheel.mass":           {"label": "Wheel mass",           "unit": "kg",     "min": 0.0,    "max": 2.0,    "step": 0.005,  "group": "Wheel"},
    "load.wheel.inertia_factor": {"label": "Wheel inertia factor", "unit": "",       "min": 0.0,    "max": 1.0,    "step": 0.01,   "group": "Wheel"},
    "load.wheel.width":          {"label": "Wheel width",          "unit": "m",      "min": 0.001,  "max": 0.2,    "step": 0.001,  "group": "Wheel"},
    # -- group ---------------------------------------------------------
    "group.count":               {"label": "Motor count",          "unit": "",       "min": 1,      "max": 12,     "step": 1,      "group": "Drivetrain"},
}


def param_meta(path: str, value: Any = None) -> dict[str, Any]:
    """Label, unit, slider range and group for a parameter path.

    Unknown paths get an auto range bracketing ``value`` so the UI still works
    for parameters added later.  Zero-valued unknowns get a small symmetric
    range, which is the useful default for "add this loss" style terms.
    """
    meta = PARAM_META.get(path)
    if meta:
        return dict(meta)

    name = path.rsplit(".", 1)[-1].replace("_", " ")
    v = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
    if v == 0.0:
        lo, hi, step = -1.0, 1.0, 0.01
    elif abs(v) < 1.0:
        lo, hi = 0.0, round(abs(v) * 4.0, 10)
        step = max((hi - lo) / 200.0, 1e-9)
    else:
        lo, hi = 0.0, abs(v) * 4.0
        step = (hi - lo) / 200.0
    return {"label": name, "unit": "", "min": lo, "max": hi, "step": step, "group": "Other"}


def field_catalog(system: DriveSystem) -> list[dict[str, Any]]:
    """Flat list of every editable parameter -- drives the UI parameter panel.

    Each entry carries enough metadata for a widget: dotted ``path``, current
    ``value``, human ``label``, ``unit``, slider ``min``/``max``/``step`` and a
    ``group`` name for section headers.  Ranges come from :data:`PARAM_META` when
    known, otherwise they are derived from the current value so the slider always
    brackets something useful.
    """
    out: list[dict[str, Any]] = []

    def add(prefix: str, obj: Any) -> None:
        for f in dataclasses.fields(obj):
            if not f.repr or f.name in ("convention",):
                continue
            if (f.metadata or {}).get("state"):
                continue  # runtime state, solved for -- not a design input
            value = getattr(obj, f.name)
            if dataclasses.is_dataclass(value):
                continue
            if isinstance(value, (int, float, bool)) and f.name not in ("count",):
                path = f"{prefix}.{f.name}"
                entry = dict(param_meta(path, value))
                entry["path"] = path
                entry["value"] = value
                entry["type"] = type(value).__name__
                entry["help"] = (f.metadata or {}).get("help", "")
                out.append(entry)

    add("group.motor", system.group.motor)
    spec = getattr(system.group.motor, "spec", None)
    if dataclasses.is_dataclass(spec):
        # a brushless machine keeps its constants in a nested dq spec, which the
        # dataclass filter below would otherwise skip entirely
        add("group.motor.spec", spec)
    add("gearbox", system.gearbox)
    add("load", system.load)
    if hasattr(system.load, "wheel"):
        add("load.wheel", system.load.wheel)
    entry = dict(param_meta("group.count", system.group.count))
    entry["path"] = "group.count"
    entry["value"] = system.group.count
    entry["type"] = "int"
    out.append(entry)
    return out
