"""Benchmark serial vs parallel sweeps.  Run as a script (not from stdin).

    python3 examples/benchmark_parallel.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from drivetrain import presets as P
from drivetrain.sweep import MP_CONTEXT, grid_sweep, resolve_workers


def main() -> None:
    tpl = P.build("pod_4x_geared")
    grid = {
        "gearbox.ratio": [round(0.12 + 0.02 * i, 3) for i in range(24)],
        "group.count": [1, 2, 3, 4, 6, 8, 10, 12],
        "load.mass": [3.0, 5.0, 7.0, 9.0, 11.0, 14.0, 18.0, 24.0],
    }
    n = 24 * 8 * 8
    print(f"{n} combinations, start method = {MP_CONTEXT!r}, "
          f"{resolve_workers(0)} worker slots available\n")

    for workers in (1, 0):
        t0 = time.perf_counter()
        res = grid_sweep(tpl, grid, voltage=12.0, duration=8.0, workers=workers)
        dt = time.perf_counter() - t0
        print(f"  workers={workers:<3} -> {dt:6.2f} s   ({n / dt:8.0f} combos/s)")

    best = res.best("steady_kmh", maximize=True)
    print("\nfastest drivetrain found:", {k: best[k] for k in
          ("gearbox.ratio", "group.count", "load.mass", "steady_kmh")})
    print()
    print(res.pretty(["gearbox.ratio", "group.count", "load.mass",
                      "steady_kmh", "rise_time_s", "peak_current_A"],
                     max_rows=8, sort_by="steady_kmh", maximize=True))
    print()
    print("quickest to 60% of steady speed:")
    print(res.pretty(["gearbox.ratio", "group.count", "load.mass",
                      "steady_kmh", "rise_time_s", "stall_torque_Nm"],
                     max_rows=8, sort_by="rise_time_s", maximize=False))


if __name__ == "__main__":
    main()
