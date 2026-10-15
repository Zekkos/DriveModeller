#!/usr/bin/env python3
"""Engine benchmark -- measures the per-step cost of each integrator.

Run with::

    python3 examples/benchmark_engines.py

Prints steps/second for each engine so regressions in the hot loops are easy to
spot.  The engines are dominated by Python overhead rather than arithmetic, so
they regress easily: a stray ``np.clip`` on a scalar or a per-step ``scipy
.expm`` call costs an order of magnitude.  See ``drivetrain/simulate.py``.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from drivetrain import presets  # noqa: E402
from drivetrain.simulate import simulate  # noqa: E402

DURATION = 1.0
DT = 2e-5          # 50,000 steps -- enough to time reliably
ENGINES = ("exact", "rk4", "euler")

# Measured before the scalar fast paths landed, from cProfile over the same
# 50,000 steps (1 s at dt=2e-5).  Only exact and rk4 were profiled before the
# change, so euler has no baseline to compare against.
#   exact  1.041 s/run  -- scipy.linalg.expm called once per step for a 1x1 system
#   rk4    1.585 s/run  -- np.clip on scalars + array allocation inside the stages
BASELINE = {"exact": 50_000.0 / 1.041, "rk4": 50_000.0 / 1.585}


def main() -> int:
    steps = int(round(DURATION / DT))
    print(f"{steps:,} steps per run (duration={DURATION}s, dt={DT:g})")
    print(f"cpu count: {os.cpu_count()}\n")

    header = f"{'preset':16s}" + "".join(f"{e:>12s}" for e in ENGINES)
    print(header)
    print("-" * len(header))

    best: dict[str, float] = {e: 0.0 for e in ENGINES}
    for name in presets.preset_names():
        system = presets.build(name)
        cells = []
        for engine in ENGINES:
            # warm up once (import/JIT-free but caches and first-touch pages)
            simulate(system, 12.0, 0.05, dt=DT, engine=engine, samples=None)
            t0 = time.perf_counter()
            simulate(system, 12.0, DURATION, dt=DT, engine=engine, samples=None)
            dt_meas = time.perf_counter() - t0
            rate = steps / dt_meas
            best[engine] = max(best[engine], rate)
            cells.append(f"{rate:12,.0f}")
        print(f"{name:16s}" + "".join(cells))

    print("\nbest rate per engine, vs the pre-fast-path baseline:")
    for engine in ENGINES:
        if engine in BASELINE:
            speedup = best[engine] / BASELINE[engine]
            print(f"  {engine:6s} {best[engine]:12,.0f} steps/s   ({speedup:5.1f}x baseline)")
        else:
            print(f"  {engine:6s} {best[engine]:12,.0f} steps/s   (no baseline profiled)")

    print("\nnote: 'exact' is unconditionally stable -- it is not slower than "
          "euler, it is\nthe accurate one.  For wall-clock throughput on many "
          "combinations, use the\nparallel sweep (examples/benchmark_parallel.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
