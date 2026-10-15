"""Parameter sweeps, design-space exploration and small optimisers.

Everything here works on a *template* :class:`~drivetrain.system.DriveSystem`
plus a list of ``(path, value)`` overrides, so any nested parameter can be swept
without writing plumbing code::

    sweep(system, "gearbox.ratio", [0.1, 0.2, 0.3])
    grid_sweep(system, {"gearbox.ratio": [0.1, 0.2], "group.count": [2, 4]})

Paths are dotted attribute names resolved with :func:`dataclasses.replace`, so
the template is never mutated.

**Parallelism.**  Sweeps are embarrassingly parallel and dominate wall-clock time
for design-space exploration, so :func:`sweep` and :func:`grid_sweep` accept
``workers``.  ``workers=1`` runs inline (fastest for small jobs, and required if
``extra_metrics`` contains lambdas); ``workers>1`` or ``workers=0`` (meaning
``os.cpu_count()``) uses a :class:`~concurrent.futures.ProcessPoolExecutor`,
which sidesteps the GIL.  Because work items are pickled, ``extra_metrics``
values must be module-level functions, not lambdas, in parallel mode.
"""

from __future__ import annotations

import itertools
import multiprocessing as _mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .exceptions import ConfigError
from .motor import rad_s_to_rpm
from .simulate import SimulationResult, simulate
from .system import DriveSystem


# ----------------------------------------------------------------------
# dotted-path override machinery
# ----------------------------------------------------------------------
def _split_path(path: str) -> tuple[str, ...]:
    return tuple(p for p in path.split(".") if p)


def set_path(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of ``obj`` with the dotted ``path`` set to ``value``.

    Supports dataclass fields and list indices, e.g. ``"load.wheel.radius"`` or
    ``"motors.0.kv"``.
    """
    parts = _split_path(path)
    if not parts:
        raise ConfigError(f"Empty parameter path {path!r}.")

    def walk(node: Any, remaining: tuple[str, ...]) -> Any:
        head, *rest = remaining
        if not rest:
            if isinstance(node, (list, tuple)):
                items = list(node)
                items[int(head)] = value
                return type(node)(items) if isinstance(node, tuple) else items
            if not hasattr(node, head):
                raise ConfigError(f"{type(node).__name__} has no field {head!r}.")
            return replace(node, **{head: value})
        child = node[head] if isinstance(node, (list, tuple)) else getattr(node, head)
        new_child = walk(child, tuple(rest))
        if isinstance(node, (list, tuple)):
            items = list(node)
            items[int(head)] = new_child
            return type(node)(items) if isinstance(node, tuple) else items
        return replace(node, **{head: new_child})

    return walk(obj, parts)


def get_path(obj: Any, path: str) -> Any:
    """Read a dotted path (see :func:`set_path`)."""
    node = obj
    for part in _split_path(path):
        node = node[part] if isinstance(node, (list, tuple)) else getattr(node, part)
    return node


# ----------------------------------------------------------------------
# sweep results
# ----------------------------------------------------------------------
@dataclass
class SweepResult:
    """Tabular sweep output, addressable by column name."""

    params: list[str]
    values: list[list[Any]]
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    systems: list[DriveSystem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    @property
    def table(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, col in zip(self.params, np.array(self.values, dtype=object).T):
            out[name] = col
        out.update(self.columns)
        return out

    def argbest(self, column: str, maximize: bool = True) -> int:
        """Row index of the best value in ``column``."""
        col = np.asarray(self.columns[column], dtype=float)
        col = np.where(np.isfinite(col), col, -np.inf if maximize else np.inf)
        return int(np.argmax(col) if maximize else np.argmin(col))

    def best(self, column: str, maximize: bool = True) -> dict[str, Any]:
        """Row dictionary for the best value in ``column``."""
        return self.table_row(self.argbest(column, maximize))

    def table_row(self, i: int) -> dict[str, Any]:
        """One row as a plain dict (scalars, not arrays)."""
        return {k: (v[i] if np.ndim(v) else v) for k, v in self.table.items()}

    def to_rows(self) -> list[dict[str, Any]]:
        n = len(self)
        keys = list(self.table)
        return [{k: self.table[k][i] for k in keys} for i in range(n)]

    def sorted_rows(self, column: str, maximize: bool = True) -> list[int]:
        """Row indices ordered by ``column``, best first."""
        col = np.asarray(self.columns[column], dtype=float)
        col = np.where(np.isfinite(col), col, -np.inf if maximize else np.inf)
        return list(np.argsort(col)[::-1] if maximize else np.argsort(col))

    def pretty(
        self,
        columns: Sequence[str] | None = None,
        max_rows: int = 40,
        sort_by: str | None = None,
        maximize: bool = True,
    ) -> str:
        """Fixed-width table, optionally sorted by a metric column."""
        keys = list(columns) if columns else list(self.table)
        order = self.sorted_rows(sort_by, maximize) if sort_by else list(range(len(self)))
        rows = [self.table_row(i) for i in order[:max_rows]]
        widths = {k: max(len(k), 12) for k in keys}
        for r in rows:
            for k in keys:
                widths[k] = max(widths[k], len(_fmt(r.get(k))))
        head = "  ".join(k.rjust(widths[k]) for k in keys)
        lines = [head, "  ".join("-" * widths[k] for k in keys)]
        for r in rows:
            lines.append("  ".join(_fmt(r.get(k)).rjust(widths[k]) for k in keys))
        if sort_by:
            lines.append(f"  (sorted by {sort_by}, {'highest' if maximize else 'lowest'} first)")
        if len(self) > max_rows:
            lines.append(f"... {len(self) - max_rows} more rows")
        return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "inf" if v > 0 else "-"
        a = abs(v)
        if a >= 1000 or (a and a < 0.01):
            return f"{v:.3g}"
        return f"{v:.3f}"
    return str(v)


# ----------------------------------------------------------------------
# parallel evaluation machinery
# ----------------------------------------------------------------------
@dataclass
class _Job:
    """One picklable unit of sweep work."""

    template: DriveSystem
    params: tuple[str, ...]
    combo: tuple[Any, ...]
    voltage: float
    duration: float
    engine: str
    dt: float | None
    run: bool


def _apply(template: DriveSystem, params: Sequence[str], combo: Sequence[Any]) -> DriveSystem:
    system = template
    for path, value in zip(params, combo):
        system = set_path(system, path, value)
    return system


def _run_job(job: _Job) -> dict[str, float]:
    """Evaluate one combination.  Module level so it can be pickled to workers."""
    system = _apply(job.template, job.params, job.combo)
    res = (
        simulate(system, job.voltage, job.duration, dt=job.dt, engine=job.engine)
        if job.run else None
    )
    return _metric_dict(system, res)


def resolve_workers(workers: int | None) -> int:
    """Normalise a ``workers`` request: ``None``/1 -> inline, 0 -> all CPUs."""
    if workers is None or workers == 1:
        return 1
    if workers <= 0:
        return max(1, os.cpu_count() or 1)
    return int(workers)


# Start method for the worker pool.  ``fork`` is the default where available:
# it needs no re-import of the __main__ module (so it works from REPLs and
# notebooks) and it copies the parent's memory cheaply.  On Windows, or when the
# parent already has live threads that could hold locks across fork(), set
# ``drivetrain.sweep.MP_CONTEXT = "forkserver"`` -- scripts must then be guarded
# with ``if __name__ == "__main__":``.
MP_CONTEXT = "fork" if "fork" in _mp.get_all_start_methods() else "forkserver"


def set_mp_context(name: str) -> str:
    """Choose the multiprocessing start method ("fork", "forkserver", "spawn")."""
    global MP_CONTEXT
    if name not in _mp.get_all_start_methods():
        raise ConfigError(
            f"Start method {name!r} unavailable; options: {_mp.get_all_start_methods()}."
        )
    MP_CONTEXT = name
    return MP_CONTEXT


def _evaluate_many(
    template: DriveSystem,
    params: Sequence[str],
    combos: Sequence[tuple[Any, ...]],
    *,
    voltage: float,
    duration: float,
    engine: str,
    dt: float | None,
    run: bool,
    workers: int | None,
    extra_metrics: dict[str, Callable[[DriveSystem, SimulationResult | None], float]] | None,
    progress: Callable[[int, int], None] | None = None,
    mp_context: str | None = None,
) -> tuple[list[DriveSystem], list[dict[str, float]]]:
    """Run every combination, inline or across processes.

    Results are returned in input order regardless of completion order, so sweeps
    stay reproducible.
    """
    jobs = [
        _Job(template, tuple(params), tuple(c), voltage, duration, engine, dt, run)
        for c in combos
    ]
    n = len(jobs)
    systems = [_apply(template, params, c) for c in combos]

    # Inline path: cheap, and the only one that supports closures/lambdas.
    if resolve_workers(workers) == 1 or n == 1:
        out: list[dict[str, float]] = []
        for i, job in enumerate(jobs):
            metrics = _run_job(job)
            if extra_metrics:
                for name, fn in extra_metrics.items():
                    metrics[name] = float(fn(systems[i], _maybe_simulate(job)))
            out.append(metrics)
            if progress:
                progress(i + 1, n)
        return systems, out

    # Parallel path: extra_metrics must be picklable (module-level functions).
    if extra_metrics:
        import pickle

        for name, fn in extra_metrics.items():
            try:
                pickle.dumps(fn)
            except Exception as exc:  # pragma: no cover - user error path
                raise ConfigError(
                    f"extra_metrics[{name!r}] is not picklable ({exc}); use a "
                    "module-level function or workers=1."
                ) from exc

    results: list[dict[str, float] | None] = [None] * n
    n_workers = min(resolve_workers(workers), n)
    ctx = _mp.get_context(mp_context or MP_CONTEXT)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_job, job): i for i, job in enumerate(jobs)}
        done = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if progress:
                progress(done, n)
    return systems, [r for r in results if r is not None]


def _maybe_simulate(job: _Job) -> SimulationResult | None:
    if not job.run:
        return None
    system = _apply(job.template, job.params, job.combo)
    return simulate(system, job.voltage, job.duration, dt=job.dt, engine=job.engine)


# ----------------------------------------------------------------------
# sweep engines
# ----------------------------------------------------------------------
def _metric_dict(system: DriveSystem, result: SimulationResult | None) -> dict[str, float]:
    out = {
        "steady_rpm": rad_s_to_rpm(system.steady_speed(system.voltage_limit)),
        "steady_output_rpm": rad_s_to_rpm(system.steady_output_speed(system.voltage_limit)),
        "mech_tau_s": system.mech_time_constant,
        "stall_torque_Nm": system.steady_output_torque(system.voltage_limit),
        "total_inertia": system.total_inertia,
    }
    if system.wheel is not None:
        out["steady_kmh"] = system.steady_ground_speed(system.voltage_limit) * 3.6
    if result is not None:
        out.update(result.final())
        out.update(result.peak())
        m = result.metrics()
        out["rise_time_s"] = m.get("rise_time_s", np.nan)
        out["settle_time_s"] = m.get("settle_time_s", np.nan)
        out["overshoot_pct"] = m.get("overshoot_pct", np.nan)
    return out


def sweep(
    template: DriveSystem,
    param: str | Sequence[str],
    values: Sequence[Any] | Sequence[Sequence[Any]],
    *,
    voltage: float = 12.0,
    duration: float = 5.0,
    engine: str = "exact",
    dt: float | None = None,
    run: bool = True,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    mp_context: str | None = None,
    extra_metrics: dict[str, Callable[[DriveSystem, SimulationResult | None], float]] | None = None,
) -> SweepResult:
    """Sweep one or more parameters and record metrics for every combination.

    ``param``/``values`` may be a single path/list, or parallel lists of paths
    swept pairwise.  For a full Cartesian product use :func:`grid_sweep`.

    ``workers`` > 1 spreads combinations over processes (see the module docstring);
    ``workers=0`` means all CPUs.
    """
    params = [param] if isinstance(param, str) else list(param)
    if len(params) == 1:
        combos = [(v,) for v in values]
    else:
        combos = [tuple(c) for c in zip(*(list(v) for v in values))]

    systems, metric_rows = _evaluate_many(
        template, params, combos,
        voltage=voltage, duration=duration, engine=engine, dt=dt, run=run,
        workers=workers, extra_metrics=extra_metrics, progress=progress,
        mp_context=mp_context,
    )

    result = SweepResult(params=params, values=[list(c) for c in combos], systems=systems)
    cols: dict[str, list[float]] = {}
    for metrics in metric_rows:
        for k, v in metrics.items():
            cols.setdefault(k, []).append(v)
    result.columns = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return result


def grid_sweep(
    template: DriveSystem,
    grid: dict[str, Sequence[Any]],
    *,
    voltage: float = 12.0,
    duration: float = 5.0,
    engine: str = "exact",
    dt: float | None = None,
    run: bool = True,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    mp_context: str | None = None,
    extra_metrics: dict[str, Callable[[DriveSystem, SimulationResult | None], float]] | None = None,
) -> SweepResult:
    """Full Cartesian product over ``grid`` (dict of path -> candidate values)."""
    params = list(grid)
    combos = list(itertools.product(*(grid[p] for p in params)))

    systems, metric_rows = _evaluate_many(
        template, params, combos,
        voltage=voltage, duration=duration, engine=engine, dt=dt, run=run,
        workers=workers, extra_metrics=extra_metrics, progress=progress,
        mp_context=mp_context,
    )

    result = SweepResult(params=params, values=[list(c) for c in combos], systems=systems)
    cols: dict[str, list[float]] = {}
    for metrics in metric_rows:
        for k, v in metrics.items():
            cols.setdefault(k, []).append(v)
    result.columns = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return result


# ----------------------------------------------------------------------
# design questions
# ----------------------------------------------------------------------
def find_gear_ratio(
    template: DriveSystem,
    target_ground_speed: float | None = None,
    target_output_rpm: float | None = None,
    voltage: float = 12.0,
    bounds: tuple[float, float] = (0.02, 4.0),
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Gear speed-ratio that puts the steady no-load speed on a target.

    Uses bisection on the monotonic steady-speed curve, so it is robust to the
    nonlinearity introduced by aerodynamic drag.
    """
    if target_ground_speed is None and target_output_rpm is None:
        raise ConfigError("Provide target_ground_speed [m/s] or target_output_rpm.")

    def error(ratio: float) -> float:
        s = set_path(template, "gearbox.ratio", ratio)
        if target_output_rpm is not None:
            return s.steady_output_speed(voltage) - target_output_rpm * 2.0 * np.pi / 60.0
        return s.steady_ground_speed(voltage) - target_ground_speed

    lo, hi = float(bounds[0]), float(bounds[1])
    f_lo, f_hi = error(lo), error(hi)
    if f_lo * f_hi > 0:
        raise ConfigError(
            f"No gear ratio in [{lo}, {hi}] reaches the target "
            f"(errors {f_lo:.4g} .. {f_hi:.4g}). Widen the bounds or lower the target."
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = error(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def time_to_speed(
    system: DriveSystem, target: float, voltage: float = 12.0,
    duration: float = 30.0, engine: str = "exact", on: str = "ground",
    within: float = 0.0,
) -> float:
    """Time [s] for the drivetrain to reach ``target``.

    ``on`` selects the quantity: ``ground`` (m/s), ``motor`` (rad/s),
    ``output`` (rad/s) or ``kmh`` (km/h).

    ``within`` is the acceptance band as a fraction of ``target``: the target
    counts as reached once the speed is at least ``target * (1 - within)``.  This
    matters because an equilibrium is only approached asymptotically -- when a
    design is tuned so the steady speed *equals* the target, a strict crossing
    never happens and the honest answer is "infinitely long", which is useless as
    a metric.  Use ``within=0.02`` for a 2 % band.
    """
    res = simulate(system, voltage, duration, engine=engine, samples=None)
    if on == "ground":
        y = res.ground_speed
    elif on == "kmh":
        y = res.speed_kmh
    elif on == "motor":
        y = res.motor_speed
    else:
        y = res.output_speed
    level = float(target) * (1.0 - float(within))
    hit = np.argmax(y >= level) if (y >= level).any() else None
    return float(res.t[hit]) if hit is not None else np.inf


def distance_to_speed(
    system: DriveSystem, target: float, voltage: float = 12.0,
    duration: float = 30.0, engine: str = "exact",
) -> float:
    """Distance [m] travelled before reaching ground speed ``target`` [m/s]."""
    res = simulate(system, voltage, duration, engine=engine, samples=None)
    hit = np.argmax(res.ground_speed >= target) if (res.ground_speed >= target).any() else None
    return float(res.distance[hit]) if hit is not None else np.inf


def optimum(
    template: DriveSystem,
    cost: Callable[[DriveSystem], float],
    param: str,
    bounds: tuple[float, float],
    tol: float = 1e-5,
    max_iter: int = 60,
) -> tuple[float, float]:
    """Minimise ``cost(system)`` over one parameter with golden-section search.

    Returns ``(best_value, best_cost)``.  Assumes a unimodal cost, which holds
    for the usual "minimise time-to-speed" / "minimise current" objectives.
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(bounds[0]), float(bounds[1])
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc = cost(set_path(template, param, c))
    fd = cost(set_path(template, param, d))
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = cost(set_path(template, param, c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = cost(set_path(template, param, d))
    x = 0.5 * (a + b)
    return x, cost(set_path(template, param, x))


def time_to_speed_cost(
    target: float, voltage: float = 12.0, duration: float = 30.0, on: str = "ground"
) -> Callable[[DriveSystem], float]:
    """Cost function for :func:`optimum`: time to reach ``target`` (penalised if never)."""
    def cost(system: DriveSystem) -> float:
        t = time_to_speed(system, target, voltage=voltage, duration=duration, on=on)
        return t if np.isfinite(t) else duration * 10.0
    return cost
