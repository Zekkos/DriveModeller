#!/usr/bin/env python3
"""Convenience launcher: ``python3 run.py [command ...]``.

``python3 run.py`` with no arguments opens the GUI; otherwise it forwards to the
CLI (``python3 run.py sweep --param gearbox.ratio --min 0.05 --max 1 --points 40``).
"""
import sys

from drivetrain.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["gui"]))
