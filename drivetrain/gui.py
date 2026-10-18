"""Interactive Qt (qtpy) + Matplotlib interface for the drivetrain model.

Run it with::

    python3 run.py                 # or:  python3 -m drivetrain gui
    python3 run.py gui --preset pod_4x_geared

The Qt binding itself is chosen by ``qtpy`` -- PyQt5, PyQt6, PySide2 or PySide6,
whichever is installed -- so this file never names one of them.  Matplotlib's
``QtAgg`` backend resolves its binding through qtpy as well, which matters: a
process must not load two different bindings at once (that is a hard crash, not
a warning), and going through qtpy twice guarantees a single consistent one.

Layout: parameter sliders on the left (auto-generated from the model's own field
catalogue, so new parameters appear in the UI with no UI changes), tabbed plots
on the right, and a metrics read-out underneath.

Styling
-------
The window is themed to the Gruvbox dark palette (``GRUVBOX``): a Fusion-based
Qt palette + stylesheet, and matching matplotlib rcParams, so plots and widgets
share one colour language.  ``--theme system`` opts back out to the desktop's
own look.  Matplotlib's Qt toolbar icons are painted from the widget palette, so
they follow the theme for free.

Design notes
------------
* Simulations run on a ``QThread`` worker and post results back through a Qt
  signal.  The connection is automatically queued, so the handler runs on the
  GUI thread and no widget is ever touched from the worker -- that is the whole
  reason for the signal instead of a shared result object.
* Edits are stored as *overrides* applied to the selected preset.  That keeps the
  model itself the single source of truth and makes every UI state reproducible:
  the override dict is exactly what save/load JSON writes.
* The sweep tab fans combinations out over processes (``drivetrain.sweep``),
  which is where the wall-clock win actually lives.
* Live edits are debounced (~80 ms), so dragging a slider queues one simulation
  per pause rather than one per pixel.
* The response panel keeps exactly one ground-speed twin axis, not one per run:
  ``Axes.clear()`` does not drop a ``twinx()`` partner, so a re-render deletes
  every axis that is not part of the 2x2 grid first (``_clear_figure``).  Older
  runs are drawn behind the current one only up to the overlay spinner's count,
  so the panel cannot accumulate an unbounded pile of previous traces.
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import queue
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

from . import export, fit, presets
from . import efficiency as eff
from . import export_dq as dq_export
from .exceptions import DrivetrainError
from .motor import rad_s_to_rpm
from .simulate import (
    EXCITATIONS,
    SimulationResult,
    simulate,
    voltage_signal,
)
from .sweep import set_path, sweep
from .system import DriveSystem

try:  # Qt is optional at import time so the library stays usable headless
    from qtpy import QtCore, QtGui, QtWidgets

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
except ImportError as exc:  # pragma: no cover - depends on installed bindings
    raise DrivetrainError(
        "No Qt bindings available. Install qtpy plus one of PyQt5, PyQt6, "
        "PySide2 or PySide6, e.g. uv pip install qtpy pyqt5"
    ) from exc

import numpy as np
from matplotlib.figure import Figure


ENGINES = ("exact", "rk4", "euler")
# combo entry for "the model is not a catalogue part", so the selection never lies
CUSTOM = "(custom)"
# channel names as exposed by SimulationResult.channels()
PLOT_CHANNELS = ("motor_rpm", "output_rpm", "speed_kmh", "current_A", "torque_Nm",
                 "voltage_V", "power_in_W", "power_out_W", "copper_loss_W",
                 "efficiency", "distance_m")
# metric names produced by drivetrain.sweep._metric_dict()
SWEEP_METRICS = ("steady_rpm", "steady_output_rpm", "steady_kmh", "stall_torque_Nm",
                 "rise_time_s", "settle_time_s", "overshoot_pct", "peak_current_A",
                 "peak_current_per_motor_A", "peak_torque_Nm", "peak_power_in_W",
                 "peak_copper_loss_W", "peak_efficiency", "max_motor_rpm",
                 "mech_tau_s", "total_inertia", "motor_rpm", "speed_kmh",
                 "efficiency", "distance_m")

# slider resolution guard: a step tiny enough to need more than this many
# detents is unusable by hand, so the slider coarsens and the entry stays exact
_MAX_STEPS = 100_000

# A Qt process must not fork() from a worker thread: the child inherits only
# that thread, so a lock held by another thread stays held forever.  Parallel
# sweeps therefore start their pool from a clean interpreter instead, which is
# a few tens of milliseconds slower but cannot deadlock the GUI.
_PARALLEL_START = ("forkserver" if "forkserver" in multiprocessing.get_all_start_methods()
                   else "spawn")

# ----------------------------------------------------------------------
# Gruvbox dark (the "hard" variant of the official palette)
# ----------------------------------------------------------------------
GRUVBOX: dict[str, str] = {
    "bg_hard": "#1d2021", "bg0": "#282828", "bg_soft": "#32302f", "bg1": "#3c3836",
    "bg2": "#504945", "bg3": "#665c54", "bg4": "#7c6f64",
    "fg": "#ebdbb2", "fg_bright": "#fbf4c1", "fg_dim": "#a89984", "gray": "#928374",
    "red": "#fb4934", "green": "#b8bb26", "yellow": "#fabd2f", "blue": "#83a598",
    "purple": "#d3869b", "aqua": "#8ec07c", "orange": "#fe8019",
}
# line colours for the plot series, so curves read as part of the theme
SERIES = {
    "speed": GRUVBOX["yellow"],
    "output": GRUVBOX["aqua"],
    "ground": GRUVBOX["green"],
    "current": GRUVBOX["red"],
    "torque": GRUVBOX["blue"],
    "power_in": GRUVBOX["orange"],
    "power_out": GRUVBOX["yellow"],
    "loss": GRUVBOX["purple"],
    "efficiency": GRUVBOX["green"],
    "sweep": GRUVBOX["aqua"],
    "reference": GRUVBOX["gray"],
}
CYCLE = (GRUVBOX["yellow"], GRUVBOX["aqua"], GRUVBOX["red"], GRUVBOX["blue"],
         GRUVBOX["purple"], GRUVBOX["orange"], GRUVBOX["green"], GRUVBOX["gray"])


def _color(name: str) -> QtGui.QColor:
    return QtGui.QColor(GRUVBOX[name])


def gruvbox_palette() -> QtGui.QPalette:
    """Palette form of the theme.

    Needed in addition to the stylesheet: matplotlib's Qt navigation icons are
    painted from ``palette().color(foregroundRole())``, so a stylesheet alone
    would leave them dark-on-dark.
    """
    p = QtGui.QPalette()
    role = QtGui.QPalette
    p.setColor(role.Window, _color("bg0"))
    p.setColor(role.WindowText, _color("fg"))
    p.setColor(role.Base, _color("bg_soft"))
    p.setColor(role.AlternateBase, _color("bg1"))
    p.setColor(role.Text, _color("fg"))
    p.setColor(role.Button, _color("bg2"))
    p.setColor(role.ButtonText, _color("fg"))
    p.setColor(role.Highlight, _color("bg4"))
    p.setColor(role.HighlightedText, _color("fg_bright"))
    p.setColor(role.ToolTipBase, _color("bg1"))
    p.setColor(role.ToolTipText, _color("fg"))
    p.setColor(role.PlaceholderText, _color("bg4"))
    p.setColor(role.Link, _color("blue"))
    for group in (role.Active, role.Inactive):
        p.setColor(group, role.WindowText, _color("fg"))
        p.setColor(group, role.ButtonText, _color("fg"))
        p.setColor(group, role.Text, _color("fg"))
    for role_name in ("WindowText", "ButtonText", "Text", "Base"):
        p.setColor(role.Disabled, getattr(role, role_name), _color("bg4"))
    return p


def gruvbox_stylesheet() -> str:
    """Qt stylesheet for the Fusion style (native styles ignore most of this)."""
    c = GRUVBOX
    return f"""
    QWidget {{
        background-color: {c['bg0']};
        color: {c['fg']};
        selection-background-color: {c['bg4']};
        selection-color: {c['fg_bright']};
    }}
    QMainWindow, QToolTip, QMenuBar, QMenu, QStatusBar, QToolBar,
    QScrollArea, QTabWidget > QStackedWidget, QWidget#paramHost {{
        background-color: {c['bg0']};
    }}
    QToolTip {{
        color: {c['fg']};
        background-color: {c['bg1']};
        border: 1px solid {c['bg3']};
        padding: 3px;
    }}
    QLabel {{ background: transparent; }}
    QLabel[dim="1"] {{ color: {c['fg_dim']}; }}

    QMenuBar {{ border-bottom: 1px solid {c['bg1']}; }}
    QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {c['bg2']}; color: {c['fg_bright']}; }}
    QMenu {{ border: 1px solid {c['bg2']}; padding: 4px; }}
    QMenu::item {{ padding: 5px 22px 5px 12px; }}
    QMenu::item:selected {{ background: {c['bg2']}; color: {c['fg_bright']}; }}
    QMenu::separator {{ height: 1px; background: {c['bg2']}; margin: 4px 6px; }}

    QToolBar {{ border: none; border-bottom: 1px solid {c['bg1']}; padding: 5px; spacing: 6px; }}
    QToolBar QLabel {{ color: {c['fg_dim']}; padding-left: 6px; }}
    QToolBar::separator {{ width: 1px; background: {c['bg2']}; margin: 4px 8px; }}

    QPushButton {{
        background: {c['bg2']}; border: 1px solid {c['bg3']}; border-radius: 4px;
        padding: 5px 12px; color: {c['fg']};
    }}
    QPushButton:hover {{ background: {c['bg3']}; border-color: {c['bg4']}; color: {c['fg_bright']}; }}
    QPushButton:pressed {{ background: {c['bg1']}; }}
    QPushButton:disabled {{ color: {c['bg4']}; background: {c['bg1']}; border-color: {c['bg2']}; }}
    QPushButton#runButton {{ border: 1px solid {c['yellow']}; color: {c['yellow']}; font-weight: 600; }}
    QPushButton#runButton:hover {{ background: {c['bg3']}; color: {c['fg_bright']}; }}

    QToolButton {{ background: transparent; border: none; border-radius: 4px; padding: 3px; }}
    QToolButton:hover {{ background: {c['bg2']}; }}
    QToolButton:checked {{ background: {c['bg2']}; border: 1px solid {c['bg3']}; }}
    QToolButton#groupToggle {{
        background: {c['bg1']}; border: 1px solid {c['bg2']}; color: {c['fg_dim']};
        font-weight: 700; padding: 0;
    }}
    QToolButton#groupToggle:hover {{ background: {c['bg2']}; color: {c['fg_bright']}; }}

    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QAbstractItemView {{
        background: {c['bg_soft']}; color: {c['fg']};
        border: 1px solid {c['bg2']}; border-radius: 4px; padding: 3px 6px;
        selection-background-color: {c['bg4']}; selection-color: {c['fg_bright']};
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {c['yellow']}; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
                           width: 18px; border: none; }}
    QComboBox QAbstractItemView {{
        border: 1px solid {c['bg2']}; background: {c['bg1']};
        selection-background-color: {c['bg3']}; selection-color: {c['fg_bright']};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 16px; background: {c['bg2']}; border: none;
    }}
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {c['bg3']}; }}
    QPlainTextEdit#metrics, QPlainTextEdit#sweepOut, QPlainTextEdit#fitOut {{
        background: {c['bg_hard']}; border-color: {c['bg1']};
    }}

    QCheckBox {{ background: transparent; spacing: 6px; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px; border: 1px solid {c['bg3']};
        border-radius: 3px; background: {c['bg_soft']};
    }}
    QCheckBox::indicator:checked {{ background: {c['green']}; border-color: {c['green']}; }}
    QCheckBox::indicator:hover {{ border-color: {c['bg4']}; }}

    QTabWidget::pane {{ border: 1px solid {c['bg1']}; border-radius: 4px; }}
    QTabBar::tab {{
        background: {c['bg1']}; color: {c['fg_dim']};
        border: 1px solid {c['bg1']}; border-bottom: none;
        padding: 6px 14px; margin-right: 2px;
        border-top-left-radius: 4px; border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{ background: {c['bg0']}; color: {c['yellow']}; }}
    QTabBar::tab:hover {{ color: {c['fg']}; }}

    QSlider::groove:horizontal {{
        height: 4px; background: {c['bg1']}; border: 1px solid {c['bg2']}; border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {c['yellow']}; border-radius: 2px; }}
    QSlider::add-page:horizontal {{ background: {c['bg1']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {c['fg']}; width: 13px; margin: -6px 0; border-radius: 6px;
    }}
    QSlider::handle:horizontal:hover {{ background: {c['fg_bright']}; }}
    QSlider::handle:horizontal:pressed {{ background: {c['yellow']}; }}
    QSlider:disabled::sub-page:horizontal {{ background: {c['bg2']}; }}

    QScrollBar:vertical {{ background: {c['bg0']}; width: 12px; margin: 0; border: none; }}
    QScrollBar::handle:vertical {{ background: {c['bg2']}; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['bg3']}; }}
    QScrollBar:horizontal {{ background: {c['bg0']}; height: 12px; margin: 0; border: none; }}
    QScrollBar::handle:horizontal {{ background: {c['bg2']}; border-radius: 5px; min-width: 28px; }}
    QScrollBar::handle:horizontal:hover {{ background: {c['bg3']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QScrollArea {{ border: none; }}
    QSplitter::handle {{ background: {c['bg1']}; }}
    QSplitter::handle:hover {{ background: {c['bg3']}; }}
    QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {c['bg2']}; }}

    QStatusBar {{ border-top: 1px solid {c['bg1']}; color: {c['fg_dim']}; padding: 3px 6px; }}
    QStatusBar::item {{ border: none; }}
    QMessageBox QLabel {{ background: transparent; color: {c['fg']}; }}
    """


def apply_gruvbox_mpl() -> None:
    """Match matplotlib to the theme, globally.

    ``Axes.clear()`` re-reads rcParams, so styling rcParams (rather than the
    individual axes) is what keeps the plots themed after every re-plot.
    """
    c = GRUVBOX
    wanted = {
        "figure.facecolor": c["bg0"], "figure.edgecolor": c["bg0"],
        "savefig.facecolor": c["bg0"], "savefig.edgecolor": c["bg0"],
        "axes.facecolor": c["bg0"], "axes.edgecolor": c["bg4"],
        "axes.labelcolor": c["fg"], "axes.titlecolor": c["fg_bright"],
        "text.color": c["fg"],
        "xtick.color": c["fg_dim"], "ytick.color": c["fg_dim"],
        "grid.color": c["bg2"],
        "legend.facecolor": c["bg1"], "legend.edgecolor": c["bg3"],
        "legend.labelcolor": c["fg"], "legend.framealpha": 0.9,
        "lines.linewidth": 1.5,
    }
    try:  # cycler is a matplotlib dependency, but stay defensive
        from cycler import cycler
        wanted["axes.prop_cycle"] = cycler(color=list(CYCLE))
    except ImportError:  # pragma: no cover
        pass
    for key, value in wanted.items():
        if key in matplotlib.rcParams:
            matplotlib.rcParams[key] = value


def apply_theme(app: QtWidgets.QApplication, theme: str = "gruvbox") -> None:
    """Install the theme on the whole application."""
    if theme != "gruvbox":
        return
    app.setStyle("Fusion")
    app.setPalette(gruvbox_palette())
    app.setStyleSheet(gruvbox_stylesheet())
    apply_gruvbox_mpl()


def _drive_voltage(system: DriveSystem) -> float:
    """Voltage to drive open-loop sweeps/comparisons with.

    ``voltage_limit`` is infinite when the motor has no configured ceiling, which
    is meaningless as a drive level, so fall back to a nominal 12 V.
    """
    v = system.voltage_limit
    return float(v) if np.isfinite(v) else 12.0


class _Worker(QtCore.QThread):
    """Runs jobs on a background thread; each job carries its own completion callback.

    ``result_ready`` is emitted from the worker thread but connected to an object
    that lives on the GUI thread, so Qt queues it and the callback runs where
    widget mutation is legal.
    """

    result_ready = QtCore.Signal(object, object)   # (callback, ("ok"|"error", payload))

    def __init__(self) -> None:
        super().__init__()
        self.jobs: queue.Queue = queue.Queue()
        self.busy = False
        self.setObjectName("drivetrain-worker")

    def submit(self, on_result, fn, *args, **kwargs) -> None:
        self.jobs.put((on_result, fn, args, kwargs))

    def stop(self) -> None:
        """Finish the current job, then exit the loop cleanly."""
        self.requestInterruption()
        self.jobs.put(None)

    def run(self) -> None:  # noqa: D102 - Thread override
        while True:
            job = self.jobs.get()
            if job is None:
                return
            on_result, fn, args, kwargs = job
            try:
                self.result_ready.emit(on_result, ("ok", fn(*args, **kwargs)))
            except Exception as exc:  # surface model/config errors in the UI
                traceback.print_exc()
                self.result_ready.emit(
                    on_result, ("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.jobs.task_done()


class _EnterEdit(QtWidgets.QLineEdit):
    """Line edit with its own ``entered`` signal.

    Deliberately *not* ``returnPressed``: the built-in signal is also used by
    Qt's default-button machinery, and we do not want Enter in a sweep field to
    fire a plain simulation run as well.
    """

    entered = QtCore.Signal()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.entered.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _fixed_font(point: float = 9.0) -> QtGui.QFont:
    """The platform's monospace face (portable across Qt5/Qt6 bindings)."""
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    font.setPointSizeF(point)
    return font


def _placeholder(ax, text: str) -> None:
    """Show a hint on an unused chart instead of misleading 0..1 default axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color=GRUVBOX["fg_dim"], fontsize=11)


MAX_OVERLAY = 6      # earlier runs the time-response tab is allowed to keep


def _clear_figure(fig, grid) -> None:
    """Clear a subplot grid *and* drop the twin axes an earlier run attached to it.

    ``Axes.clear()`` does not touch a ``twinx()`` partner, so clearing only the
    2x2 grid left every previous run's ground-speed curve sitting on the figure,
    each run adding another km/h scale down the right edge and another set of
    curves nobody asked to keep.  Anything in the figure that is not part of the
    grid is a leftover, so it is removed rather than merely cleared.
    """
    keep = [a for row in grid for a in np.atleast_1d(row)]
    for a in list(fig.axes):
        if a not in keep:
            fig.delaxes(a)
    for a in keep:
        a.clear()


def _plot_dt(duration: float, points: int = 1500) -> float:
    """Step size for GUI plots.

    The solver's auto step resolves the fastest *time constant*, which is the
    right choice for accuracy but can be coarse for drawing: a 5 s window on a
    2.5 s system is only ~100 samples.  The engines run at ~2M steps/s, so
    asking for a fixed ~1500 points costs under a millisecond and makes the
    curves smooth regardless of how slow the drivetrain is.
    """
    return max(float(duration) / float(points), 1e-7)


def _number_or_none(text: str) -> float | None:
    """Parse an optional numeric field; blank or junk means "unset"."""
    text = text.strip()
    if not text or text in {"-", "--"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


class _ParamRow(QtWidgets.QWidget):
    """One parameter: label, slider and an exact numeric entry.

    Qt sliders are integer-only, so the slider indexes ``lo + n * step``.  The
    entry is the precise value -- it accepts anything, the slider only snaps to
    the metadata step.
    """

    value_changed = QtCore.Signal(str, float)

    def __init__(self, entry: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.path = entry["path"]
        self.default = float(entry["value"])
        value = self.default
        lo, hi, step = float(entry["min"]), float(entry["max"]), float(entry["step"])
        self.is_int = entry.get("type") == "int"
        if step <= 0:
            step = 1e-6
        # keep the current value inside the slider range
        if not (lo <= value <= hi):
            lo, hi = min(lo, value), max(hi, value)
        # some limits are legitimately infinite (max_current defaults to "no
        # limit"), and int(round(inf)) raises -- so an unbounded parameter gets a
        # finite slider span and the entry stays the authority on its value
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = -1e9, 1e9
        if not np.isfinite(step) or step <= 0:
            step = max((hi - lo) / 200.0, 1e-9)
        self.lo, self.hi, self.step = lo, hi, step
        self.n_steps = int(min(_MAX_STEPS, max(1, round((hi - lo) / step))))

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(6)

        label = entry["label"] + (f"  [{entry['unit']}]" if entry["unit"] else "")
        cap = QtWidgets.QLabel(label, self)
        self.cap = cap
        cap.setMinimumWidth(150)
        tip = entry["path"] + (f"\n{entry['help']}" if entry.get("help") else "")
        cap.setToolTip(tip)
        row.addWidget(cap)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, self.n_steps)
        self.slider.setPageStep(max(1, self.n_steps // 10))
        self.slider.setValue(self._to_slider(value))
        self.slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.slider, 1)

        self.entry_edit = QtWidgets.QLineEdit(_fmt_num(value), self)
        self.entry_edit.setFixedWidth(82)
        self.entry_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.entry_edit.returnPressed.connect(self._on_entry)
        self.entry_edit.editingFinished.connect(self._on_entry)
        row.addWidget(self.entry_edit)

        self._value = self._clamp(value)
        self._syncing = False

    # -- mapping ---------------------------------------------------------
    def _to_slider(self, value: float) -> int:
        if not np.isfinite(value):        # inf == "unlimited": park at the far end
            return self.n_steps if value > 0 else 0
        return int(max(0, min(self.n_steps, round((value - self.lo) / self.step))))

    def _from_slider(self, index: int) -> float:
        value = self.lo + index * self.step
        return float(round(value)) if self.is_int else float(value)

    def _clamp(self, value: float) -> float:
        if not np.isfinite(value):
            return float(value)          # inf is meaningful here ("no limit"), keep it
        value = round(value) if self.is_int else value
        return float(min(self.hi, max(self.lo, value)))

    # -- access ----------------------------------------------------------
    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Programmatic set: moves slider + entry, does not emit."""
        self._value = self._clamp(float(value))
        self._syncing = True
        self.slider.setValue(self._to_slider(self._value))
        self.entry_edit.setText(_fmt_num(self._value))
        self._syncing = False

    # -- handlers --------------------------------------------------------
    def _on_slider(self, index: int) -> None:
        if self._syncing:
            return
        self._value = self._from_slider(index)
        self._syncing = True
        self.entry_edit.setText(_fmt_num(self._value))
        self._syncing = False
        self.value_changed.emit(self.path, self._value)

    def _on_entry(self) -> None:
        if self._syncing:
            return
        raw = self.entry_edit.text().strip()
        try:
            value = float(raw)
        except ValueError:
            self.entry_edit.setText(_fmt_num(self._value))   # revert, visibly
            self.entry_edit.selectAll()
            return
        self.set_value(value)
        self.value_changed.emit(self.path, self._value)


class _ParamGroup(QtWidgets.QWidget):
    """A collapsible section of the parameter panel."""

    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.rows: dict[str, _ParamRow] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QtWidgets.QWidget(self)
        head_row = QtWidgets.QHBoxLayout(head)
        head_row.setContentsMargins(0, 5, 0, 2)
        head_row.setSpacing(6)
        self.toggle = QtWidgets.QToolButton(head)
        self.toggle.setObjectName("groupToggle")
        self.toggle.setText("-")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setFixedSize(20, 18)
        self.toggle.setToolTip("collapse / expand")
        self.toggle.toggled.connect(self._set_open)
        head_row.addWidget(self.toggle)
        title = QtWidgets.QLabel(f"<b>{name}</b>", head)
        head_row.addWidget(title)
        line = QtWidgets.QFrame(head)
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        head_row.addWidget(line, 1)
        outer.addWidget(head)

        self.body = QtWidgets.QWidget(self)
        self.grid = QtWidgets.QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(1)
        outer.addWidget(self.body)

    def add(self, entry: dict[str, Any]) -> _ParamRow:
        row = _ParamRow(entry, self.body)
        self.rows[entry["path"]] = row
        self.grid.addWidget(row, len(self.rows) - 1, 0)
        return row

    def _set_open(self, open_: bool) -> None:
        self.body.setVisible(open_)
        self.toggle.setText("-" if open_ else "+")

    def set_open(self, open_: bool) -> None:
        self.toggle.setChecked(open_)


class DriveApp(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self, parent=None, preset: str = "legacy") -> None:
        super().__init__(parent)
        self.worker = _Worker()
        self.worker.result_ready.connect(self._deliver)
        self.worker.start()

        self.preset_name = preset
        # base_system is what pending overrides are applied to: the selected
        # preset, or a config file once one has been loaded.
        self.base_system: DriveSystem = presets.build(preset)
        self.system: DriveSystem = self.base_system
        self.overrides: dict[str, float] = {}
        self.wheel_name: str | None = None
        self.param_rows: dict[str, _ParamRow] = {}
        self.groups: dict[str, _ParamGroup] = {}
        self.last_result: SimulationResult | None = None
        # runs drawn behind the current one when the overlay spinner is raised.
        # Bounded by MAX_OVERLAY, so the chart can never accumulate without limit.
        self._history: list[tuple[DriveSystem, SimulationResult]] = []
        self._last_payload: tuple | None = None
        self.fit_result: fit.FitResult | None = None

        self.setWindowTitle("Drivetrain Model Explorer")
        self.resize(1420, 880)
        self.setMinimumSize(1100, 640)

        # debounce for "live" re-simulation while a slider is being dragged
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(80)
        self._live_timer.timeout.connect(self.run)

        self._build_menus()
        self._build_toolbar()
        self._build_body()
        self.statusBar().showMessage("ready")

        self.load_preset(preset)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction("Save config...", self.save_config)
        file_menu.addAction("Load config...", self.load_config)
        file_menu.addSeparator()
        file_menu.addAction("Export CSV...", self.export_csv)
        file_menu.addAction("Save PNG...", self.save_png)
        file_menu.addSeparator()
        code = file_menu.addAction("Export model code...", self.export_model_code)
        dq_code = file_menu.addAction("Export dq/FOC code...", self.export_dq_code)
        code.setToolTip(
            "Write the model as C, Rust and Python for a feed-forward "
            "controller.  Nothing is generated automatically - this is the "
            "request."
        )
        dq_code.setToolTip(
            "Write the dq/FOC model of a brushless machine as Python: MTPA and "
            "field weakening instead of the i_d=0 DC equivalent.  Needs a "
            "preset with a brushless motor, e.g. bldc_pod."
        )
        loss_map = file_menu.addAction("Efficiency map (CSV)...", self.efficiency_map_csv)
        loss_map.setToolTip(
            "Bill the losses over the whole motor (speed, torque) plane and write "
            "the grid as CSV plus a text report: copper, iron, windage, bearings, "
            "stiction and gears, with the best point in each quadrant."
        )
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        help_menu = bar.addMenu("&Help")
        help_menu.addAction("About", self._about)

    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar("Main", self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.addToolBar(QtCore.Qt.TopToolBarArea, bar)

        def add_label(text: str) -> QtWidgets.QLabel:
            lab = QtWidgets.QLabel(text, bar)
            lab.setProperty("dim", "1")
            bar.addWidget(lab)
            return lab

        add_label("Preset")
        self.preset_box = QtWidgets.QComboBox(bar)
        self.preset_box.addItems(presets.preset_names())
        self.preset_box.setCurrentText(self.preset_name)
        self.preset_box.setMinimumWidth(150)
        self.preset_box.currentTextChanged.connect(self.load_preset)
        bar.addWidget(self.preset_box)

        add_label("Motor")
        self.motor_box = QtWidgets.QComboBox(bar)
        self.motor_box.addItem(CUSTOM)
        self.motor_box.addItems(presets.motor_names())
        self.motor_box.setMinimumWidth(140)
        self.motor_box.currentTextChanged.connect(self._on_motor_choice)
        bar.addWidget(self.motor_box)

        add_label("Wheel")
        self.wheel_box = QtWidgets.QComboBox(bar)
        self.wheel_box.addItem(CUSTOM)
        self.wheel_box.addItems(presets.wheel_names())
        self.wheel_box.setMinimumWidth(120)
        self.wheel_box.setToolTip("wheel catalogue (vehicle presets only)")
        self.wheel_box.currentTextChanged.connect(self._on_wheel_choice)
        bar.addWidget(self.wheel_box)

        add_label("Engine")
        self.engine_box = QtWidgets.QComboBox(bar)
        self.engine_box.addItems(ENGINES)
        self.engine_box.setCurrentText("exact")
        self.engine_box.setMinimumWidth(80)
        bar.addWidget(self.engine_box)

        add_label("Excitation")
        self.excite_box = QtWidgets.QComboBox(bar)
        self.excite_box.addItems(EXCITATIONS)
        self.excite_box.setCurrentText("step")
        self.excite_box.setMinimumWidth(110)
        bar.addWidget(self.excite_box)

        add_label("Duration (s)")
        self.duration_box = QtWidgets.QDoubleSpinBox(bar)
        self.duration_box.setRange(0.01, 3600.0)
        self.duration_box.setDecimals(3)
        self.duration_box.setSingleStep(0.5)
        self.duration_box.setValue(5.0)
        bar.addWidget(self.duration_box)

        bar.addSeparator()
        self.run_btn = QtWidgets.QPushButton("Run", bar)
        self.run_btn.setObjectName("runButton")
        self.run_btn.setToolTip("simulate (Enter in the parameter panel, or Ctrl+Enter)")
        self.run_btn.clicked.connect(self.run)
        bar.addWidget(self.run_btn)
        bar.addAction("Reset params", self.reset_overrides)
        bar.addAction("Save config", self.save_config)
        bar.addAction("Load config", self.load_config)

        self.live_check = QtWidgets.QCheckBox("Live", bar)
        self.live_check.setChecked(True)
        self.live_check.setToolTip("re-run on every parameter edit")
        bar.addWidget(self.live_check)

        # Ctrl+Enter runs from anywhere.  Plain Enter is scoped to the parameter
        # panel (see _build_body) so it never double-fires with the sweep fields.
        self._shortcuts = []
        for sequence in ("Ctrl+Return", "Ctrl+Enter"):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.activated.connect(self.run)
            self._shortcuts.append(shortcut)

    def _build_body(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        self.setCentralWidget(splitter)

        # ---- left: scrollable parameter panel -------------------------
        panel = QtWidgets.QWidget(splitter)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 2, 2)
        layout.setSpacing(4)

        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel("<b>Parameters</b>", panel))
        hdr.addStretch(1)
        collapse_btn = QtWidgets.QPushButton("collapse all", panel)
        expand_btn = QtWidgets.QPushButton("expand all", panel)
        collapse_btn.clicked.connect(lambda: self._toggle_sections(False))
        expand_btn.clicked.connect(lambda: self._toggle_sections(True))
        hdr.addWidget(collapse_btn)
        hdr.addWidget(expand_btn)
        layout.addLayout(hdr)

        self.param_scroll = QtWidgets.QScrollArea(panel)
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.param_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.param_host = QtWidgets.QWidget()
        self.param_host.setObjectName("paramHost")
        self.param_host_layout = QtWidgets.QVBoxLayout(self.param_host)
        self.param_host_layout.setContentsMargins(0, 0, 6, 0)
        self.param_host_layout.setSpacing(2)
        self.param_scroll.setWidget(self.param_host)
        layout.addWidget(self.param_scroll, 1)
        splitter.addWidget(panel)

        # Enter inside the parameter panel = run (local context: it does not
        # steal Enter from the sweep/compare fields)
        self._param_enter = QtWidgets.QShortcut(QtGui.QKeySequence("Return"), self.param_host)
        self._param_enter.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._param_enter.activated.connect(self.run)

        # ---- right: tabs of plots -------------------------------------
        self.notebook = QtWidgets.QTabWidget(splitter)
        self.notebook.setDocumentMode(True)
        splitter.addWidget(self.notebook)

        self._build_response_tab()
        self._build_sweep_tab()
        self._build_compare_tab()
        self._build_fit_tab()

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        panel.setMinimumWidth(430)
        self.notebook.setMinimumWidth(560)
        splitter.setSizes([470, 950])

    def _build_response_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.notebook.addTab(tab, "Time response")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        ctl = QtWidgets.QWidget(tab)
        row = QtWidgets.QHBoxLayout(ctl)
        row.setContentsMargins(4, 2, 4, 0)
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Overlay previous runs", ctl))
        self.overlay_box = QtWidgets.QSpinBox(ctl)
        self.overlay_box.setRange(0, MAX_OVERLAY)
        self.overlay_box.setValue(0)
        self.overlay_box.setToolTip(
            "How many earlier runs stay on the chart, oldest drawn faintest.\n"
            "0 shows only the run you just asked for."
        )
        self.overlay_box.valueChanged.connect(self._redraw_response)
        row.addWidget(self.overlay_box)
        row.addStretch(1)
        layout.addWidget(ctl)

        self.fig = Figure(figsize=(9.2, 5.6), dpi=100, constrained_layout=True)
        self.ax = self.fig.subplots(2, 2)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumSize(520, 320)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(NavigationToolbar2QT(self.canvas, tab))

        bottom = QtWidgets.QHBoxLayout()
        self.metric_text = QtWidgets.QPlainTextEdit(tab)
        self.metric_text.setObjectName("metrics")
        self.metric_text.setReadOnly(True)
        self.metric_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.metric_text.setFont(_fixed_font(9))
        self.metric_text.setFixedHeight(196)
        bottom.addWidget(self.metric_text, 1)

        buttons = QtWidgets.QVBoxLayout()
        png_btn = QtWidgets.QPushButton("Save PNG", tab)
        csv_btn = QtWidgets.QPushButton("Export CSV", tab)
        png_btn.clicked.connect(self.save_png)
        csv_btn.clicked.connect(self.export_csv)
        buttons.addWidget(png_btn)
        buttons.addWidget(csv_btn)
        buttons.addStretch(1)
        bottom.addLayout(buttons)
        layout.addLayout(bottom)

    def _build_sweep_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.notebook.addTab(tab, "Parameter sweep")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        ctl = QtWidgets.QWidget(tab)
        row1 = QtWidgets.QHBoxLayout(ctl)
        row1.setContentsMargins(4, 4, 4, 0)
        row1.setSpacing(6)
        row1.addWidget(QtWidgets.QLabel("Parameter", ctl))
        self.sweep_param_box = QtWidgets.QComboBox(ctl)
        self.sweep_param_box.setMinimumWidth(240)
        row1.addWidget(self.sweep_param_box, 1)
        row1.addWidget(QtWidgets.QLabel("from", ctl))
        self.sweep_lo = _EnterEdit("0.05", ctl)
        row1.addWidget(self.sweep_lo)
        row1.addWidget(QtWidgets.QLabel("to", ctl))
        self.sweep_hi = _EnterEdit("1.0", ctl)
        row1.addWidget(self.sweep_hi)
        row1.addWidget(QtWidgets.QLabel("points", ctl))
        self.sweep_n = _EnterEdit("40", ctl)
        row1.addWidget(self.sweep_n)
        for edit, width in ((self.sweep_lo, 84), (self.sweep_hi, 84), (self.sweep_n, 56)):
            edit.setFixedWidth(width)
            edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            edit.entered.connect(self.run_sweep)
        layout.addWidget(ctl)

        ctl2 = QtWidgets.QWidget(tab)
        row2 = QtWidgets.QHBoxLayout(ctl2)
        row2.setContentsMargins(4, 2, 4, 2)
        row2.setSpacing(6)
        row2.addWidget(QtWidgets.QLabel("Metric", ctl2))
        self.sweep_metric_box = QtWidgets.QComboBox(ctl2)
        self.sweep_metric_box.addItems(SWEEP_METRICS)
        self.sweep_metric_box.setCurrentText("steady_rpm")
        self.sweep_metric_box.setMinimumWidth(180)
        row2.addWidget(self.sweep_metric_box)
        self.sweep_par_check = QtWidgets.QCheckBox("parallel", ctl2)
        self.sweep_par_check.setChecked(True)
        row2.addWidget(self.sweep_par_check)
        self.sweep_btn = QtWidgets.QPushButton("Run sweep", ctl2)
        self.sweep_btn.clicked.connect(self.run_sweep)
        row2.addWidget(self.sweep_btn)
        row2.addStretch(1)
        layout.addWidget(ctl2)

        self.sweep_fig = Figure(figsize=(9.2, 5.0), dpi=100, constrained_layout=True)
        self.sweep_ax = self.sweep_fig.subplots(1, 1)
        _placeholder(self.sweep_ax, "pick a parameter and press Run sweep")
        self.sweep_canvas = FigureCanvasQTAgg(self.sweep_fig)
        self.sweep_canvas.setMinimumSize(520, 260)
        layout.addWidget(self.sweep_canvas, 1)
        layout.addWidget(NavigationToolbar2QT(self.sweep_canvas, tab))

        self.sweep_out = QtWidgets.QPlainTextEdit(tab)
        self.sweep_out.setObjectName("sweepOut")
        self.sweep_out.setReadOnly(True)
        self.sweep_out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.sweep_out.setFont(_fixed_font(9))
        self.sweep_out.setFixedHeight(160)
        layout.addWidget(self.sweep_out)

    def _build_compare_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.notebook.addTab(tab, "Compare presets")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        ctl = QtWidgets.QWidget(tab)
        row = QtWidgets.QHBoxLayout(ctl)
        row.setContentsMargins(4, 4, 4, 0)
        row.setSpacing(6)
        note = QtWidgets.QLabel("Overlay every preset's response on one chart", ctl)
        note.setProperty("dim", "1")
        row.addWidget(note)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("quantity", ctl))
        self.compare_qty_box = QtWidgets.QComboBox(ctl)
        self.compare_qty_box.addItems(PLOT_CHANNELS)
        self.compare_qty_box.setCurrentText("motor_rpm")
        self.compare_qty_box.setMinimumWidth(150)
        row.addWidget(self.compare_qty_box)
        self.compare_btn = QtWidgets.QPushButton("Compare", ctl)
        self.compare_btn.clicked.connect(self.run_compare)
        row.addWidget(self.compare_btn)
        layout.addWidget(ctl)

        self.cmp_fig = Figure(figsize=(9.2, 5.6), dpi=100, constrained_layout=True)
        self.cmp_ax = self.cmp_fig.subplots(1, 1)
        _placeholder(self.cmp_ax, "pick a quantity and press Compare")
        self.cmp_canvas = FigureCanvasQTAgg(self.cmp_fig)
        self.cmp_canvas.setMinimumSize(520, 320)
        layout.addWidget(self.cmp_canvas, 1)
        layout.addWidget(NavigationToolbar2QT(self.cmp_canvas, tab))

    def _build_fit_tab(self) -> None:
        tab = QtWidgets.QWidget()
        self.notebook.addTab(tab, "Fit dataset")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        ctl = QtWidgets.QWidget(tab)
        row = QtWidgets.QHBoxLayout(ctl)
        row.setContentsMargins(4, 4, 4, 0)
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Dataset", ctl))
        self.fit_path = QtWidgets.QLineEdit(ctl)
        self.fit_path.setPlaceholderText(
            "measured CSV: time, voltage_V, current_A, motor_rpm, ...")
        self.fit_path.editingFinished.connect(self.describe_dataset)
        row.addWidget(self.fit_path, 1)
        browse = QtWidgets.QPushButton("Browse...", ctl)
        browse.clicked.connect(self.browse_dataset)
        row.addWidget(browse)
        self.fit_btn = QtWidgets.QPushButton("Fit", ctl)
        self.fit_btn.setObjectName("runButton")
        self.fit_btn.clicked.connect(self.run_fit)
        row.addWidget(self.fit_btn)
        layout.addWidget(ctl)

        note = QtWidgets.QLabel(
            "Tick what the fit may move; everything else keeps the value it has in the "
            "panel, so measure your known constants and type them in first.  A "
            "voltage_V column is used as the measured input; without one the "
            "prescribed input below drives the model.", tab)
        note.setProperty("dim", "1")
        note.setWordWrap(True)
        layout.addWidget(note)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, tab)
        split.setChildrenCollapsible(False)

        left = QtWidgets.QWidget(split)
        left.setMinimumWidth(280)
        llay = QtWidgets.QVBoxLayout(left)
        llay.setContentsMargins(0, 2, 0, 0)
        llay.setSpacing(4)
        quick = QtWidgets.QWidget(left)
        qgrid = QtWidgets.QGridLayout(quick)
        qgrid.setContentsMargins(0, 0, 0, 0)
        qgrid.setSpacing(3)
        for index, alias in enumerate(list(fit.FREE_ALIASES) + ["all", "none"]):
            btn = QtWidgets.QPushButton(alias, quick)
            btn.setToolTip("tick exactly: " + (
                "every fittable parameter" if alias == "all"
                else "nothing" if alias == "none"
                else ", ".join(fit.FREE_ALIASES[alias])))
            btn.clicked.connect(lambda _=False, name=alias: self._fit_select(name))
            qgrid.addWidget(btn, index // 3, index % 3)
        llay.addWidget(quick)

        self.fit_free = QtWidgets.QListWidget(left)
        self.fit_free.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.fit_free.setWordWrap(True)
        llay.addWidget(self.fit_free, 1)

        pin = QtWidgets.QWidget(left)
        pgrid = QtWidgets.QGridLayout(pin)
        pgrid.setContentsMargins(0, 4, 0, 0)
        pgrid.setSpacing(4)
        phint = QtWidgets.QLabel("prescribed input (no voltage_V column)", pin)
        phint.setProperty("dim", "1")
        pgrid.addWidget(phint, 0, 0, 1, 3)
        for col, text in enumerate(("excitation", "voltage", "period")):
            pgrid.addWidget(QtWidgets.QLabel(text, pin), 1, col)
        self.fit_excite_box = QtWidgets.QComboBox(pin)
        self.fit_excite_box.addItems(EXCITATIONS)
        self.fit_excite_box.setToolTip(
            "used only when the dataset has no voltage_V column")
        pgrid.addWidget(self.fit_excite_box, 2, 0)
        self.fit_voltage = QtWidgets.QDoubleSpinBox(pin)
        self.fit_voltage.setRange(0.0, 240.0)
        self.fit_voltage.setDecimals(2)
        self.fit_voltage.setSingleStep(0.5)
        self.fit_voltage.setValue(12.0)
        self.fit_voltage.setSuffix(" V")
        pgrid.addWidget(self.fit_voltage, 2, 1)
        self.fit_period = _EnterEdit("", pin)
        self.fit_period.setPlaceholderText("-")
        self.fit_period.setFixedWidth(56)
        self.fit_period.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.fit_period.setToolTip(
            "square/alternating period in s; blank lets the model choose")
        pgrid.addWidget(self.fit_period, 2, 2)
        llay.addWidget(pin)
        split.addWidget(left)

        right = QtWidgets.QWidget(split)
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(2)
        self.fit_fig = Figure(figsize=(9.2, 4.4), dpi=100, constrained_layout=True)
        self.fit_ax = self.fit_fig.subplots(1, 1)
        self.fit_axes = [self.fit_ax]
        _placeholder(self.fit_ax, "load a CSV, tick parameters, press Fit")
        self.fit_canvas = FigureCanvasQTAgg(self.fit_fig)
        self.fit_canvas.setMinimumSize(480, 240)
        rlay.addWidget(self.fit_canvas, 1)
        rlay.addWidget(NavigationToolbar2QT(self.fit_canvas, right))
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 930])
        layout.addWidget(split, 1)

        self.fit_out = QtWidgets.QPlainTextEdit(tab)
        self.fit_out.setObjectName("fitOut")
        self.fit_out.setReadOnly(True)
        self.fit_out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.fit_out.setFont(_fixed_font(9))
        self.fit_out.setFixedHeight(260)
        layout.addWidget(self.fit_out)

        btns = QtWidgets.QWidget(tab)
        brow = QtWidgets.QHBoxLayout(btns)
        brow.setContentsMargins(4, 2, 4, 0)
        brow.setSpacing(6)
        self.fit_apply_btn = QtWidgets.QPushButton("Apply to model", btns)
        self.fit_apply_btn.setEnabled(False)
        self.fit_apply_btn.clicked.connect(self.apply_fit)
        brow.addWidget(self.fit_apply_btn)
        self.fit_save_btn = QtWidgets.QPushButton("Save fitted model...", btns)
        self.fit_save_btn.setEnabled(False)
        self.fit_save_btn.clicked.connect(self.save_fit)
        brow.addWidget(self.fit_save_btn)
        brow.addStretch(1)
        layout.addWidget(btns)

    # ------------------------------------------------------------------
    # parameter panel plumbing
    # ------------------------------------------------------------------
    def rebuild_params(self) -> None:
        """Rebuild the slider panel from the current model's field catalogue."""
        while self.param_host_layout.count():
            item = self.param_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.param_rows.clear()
        self.groups.clear()

        catalog = presets.field_catalog(self.system)
        sweep_values = []
        groups: dict[str, list[dict]] = {}
        for entry in catalog:
            groups.setdefault(entry["group"], []).append(entry)
            sweep_values.append(entry["path"])

        order = ["Drivetrain", "Motor", "Gearbox", "Load", "Wheel", "Other"]
        for name in [g for g in order if g in groups] + [g for g in groups if g not in order]:
            section = _ParamGroup(name, self.param_host)
            for entry in groups[name]:
                row = section.add(entry)
                row.value_changed.connect(self._on_param_change)
                self.param_rows[entry["path"]] = row
            self.groups[name] = section
            self.param_host_layout.addWidget(section)
        if self.param_rows:
            # one shared label column: sliders line up across every group
            width = max(row.cap.sizeHint().width() for row in self.param_rows.values()) + 6
            for row in self.param_rows.values():
                row.cap.setFixedWidth(width)
        self.param_host_layout.addStretch(1)

        current = self.sweep_param_box.currentText()
        self.sweep_param_box.clear()
        self.sweep_param_box.addItems(sweep_values)
        if current in sweep_values:
            self.sweep_param_box.setCurrentText(current)
        elif "gearbox.ratio" in sweep_values:
            self.sweep_param_box.setCurrentText("gearbox.ratio")
        elif sweep_values:
            self.sweep_param_box.setCurrentText(sweep_values[0])

        for path, value in self.overrides.items():
            if path in self.param_rows:
                self.param_rows[path].set_value(value)
        self._sync_ranges()
        self._rebuild_fit_list()

    def _toggle_sections(self, open_: bool) -> None:
        for section in self.groups.values():
            section.set_open(open_)

    def _on_param_change(self, path: str, value: float) -> None:
        self.overrides[path] = value
        self.set_status(f"{path} = {_fmt_num(value)}")
        self._sync_ranges()
        if path.startswith("group.motor.") or path.startswith("load.wheel."):
            self._sync_catalogue_boxes()   # the model may have left the catalogue
        if self.live_check.isChecked():
            self._live_timer.start()      # debounce: one run per pause, not per pixel

    def _sync_ranges(self) -> None:
        """Grey out / clamp nothing silently -- just report out-of-limit params."""
        bad = []
        for path, value in self.overrides.items():
            if path in ("group.motor.max_current", "group.motor.max_voltage"):
                continue
            if value <= 0 and path.endswith(("resistance", "kv", "kt", "radius")):
                bad.append(path)
        if bad:
            self.set_status("warning: non-positive " + ", ".join(bad))

    # ------------------------------------------------------------------
    # model state
    # ------------------------------------------------------------------
    def load_preset(self, name: str) -> None:
        if not name or name not in presets.PRESETS:
            return
        self.preset_name = name
        if self.preset_box.currentText() != name:
            self.preset_box.setCurrentText(name)
        self.overrides.clear()
        self.wheel_name = None
        self.base_system = self.system = presets.build(name)
        self._history.clear()          # a different machine: old curves are meaningless
        self._sync_catalogue_boxes()
        self.rebuild_params()
        self.run()

    def _sync_catalogue_boxes(self) -> None:
        """Show which catalogue motor/wheel the current model actually uses.

        A preset's motor is not necessarily a catalogue entry, and editing motor
        or wheel parameters by hand moves it off-catalogue -- saying so in the
        combo beats leaving a stale (wrong) selection there.
        """
        system = self._apply_overrides()
        for box, match in ((self.motor_box, _match_motor(system.group.motor)),
                           (self.wheel_box, _match_wheel(getattr(system.load, "wheel", None)))):
            box.blockSignals(True)
            box.setCurrentText(match or CUSTOM)
            box.blockSignals(False)
        has_wheel = getattr(system.load, "wheel", None) is not None
        self.wheel_box.setEnabled(has_wheel)
        if not has_wheel:
            self.wheel_box.blockSignals(True)
            self.wheel_box.setCurrentText(CUSTOM)
            self.wheel_box.blockSignals(False)

    def _on_motor_choice(self, name: str) -> None:
        if name not in presets.MOTORS:
            return
        motor = presets.MOTORS[name]
        self.overrides.update({
            "group.motor.kv": motor.kv,
            "group.motor.kt": motor.kt,
            "group.motor.resistance": motor.resistance,
            "group.motor.rotor_inertia": motor.rotor_inertia,
            "group.motor.max_current": motor.max_current,
            "group.motor.max_voltage": motor.max_voltage,
        })
        self.rebuild_params()
        self._sync_catalogue_boxes()
        self.run()

    def _on_wheel_choice(self, name: str) -> None:
        if name not in presets.WHEELS or getattr(self.system.load, "wheel", None) is None:
            return
        self.wheel_name = name
        # drop stale per-wheel numbers, otherwise they would be re-applied on top
        # of the newly selected wheel and silently undo the choice
        self.overrides = {k: v for k, v in self.overrides.items()
                          if not k.startswith("load.wheel.")}
        self.rebuild_params()
        self._sync_catalogue_boxes()
        self.run()

    def _apply_overrides(self) -> DriveSystem:
        """Rebuild the model from the base config plus the current override dict."""
        system = self.base_system
        if self.wheel_name in presets.WHEELS and getattr(system.load, "wheel", None) is not None:
            system = set_path(system, "load.wheel", presets.WHEELS[self.wheel_name].copy())
        for path, value in self.overrides.items():
            try:
                system = set_path(system, path, value)
            except Exception as exc:
                self.set_status(f"ignored {path}: {exc}")
        return system

    def set_param(self, path: str, value: float) -> None:
        """Set a parameter by dotted path (goes through the row if there is one)."""
        row = self.param_rows.get(path)
        if row is not None:
            row.set_value(value)      # widgets only -- set_value() never emits
        self.overrides[path] = float(value)
        self._sync_ranges()
        if path.startswith("group.motor.") or path.startswith("load.wheel."):
            self._sync_catalogue_boxes()
        self.run()

    def reset_overrides(self) -> None:
        self.overrides.clear()
        self.wheel_name = None
        self.base_system = self.system = presets.build(self.preset_name)
        self._history.clear()
        self._sync_catalogue_boxes()
        self.rebuild_params()
        self.run()

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._live_timer.stop()
        system = self._apply_overrides()
        self.system = system
        duration = float(self.duration_box.value())
        engine = self.engine_box.currentText()
        kind = self.excite_box.currentText()

        def job():
            v = voltage_signal(kind, system, duration, dt=_plot_dt(duration))
            res = simulate(system, v, duration, dt=_plot_dt(duration), engine=engine)
            return system, res, kind

        self._start(f"simulating {self.preset_name} [{engine}] for {duration:g} s ...",
                    job, self._show_response)

    def _start(self, status: str, job, on_result) -> None:
        self.worker.busy = True
        self.run_btn.setEnabled(False)
        self.set_status(status)
        self.worker.submit(on_result, job)

    def _deliver(self, on_result, payload) -> None:
        """Runs on the GUI thread: the queued cross-thread signal is what makes this safe."""
        self.worker.busy = False
        self.run_btn.setEnabled(True)
        kind, body = payload
        if kind == "error":
            self.set_status(body)
            QtWidgets.QMessageBox.critical(self, "Computation failed", body)
            return
        try:
            on_result(body)
        except Exception as exc:      # never let a plotting bug kill the process:
            traceback.print_exc()     # PyQt aborts on exceptions raised in a slot
            self.set_status(f"display error: {type(exc).__name__}: {exc}")

    def _show_response(self, payload) -> None:
        system, result, kind = payload
        self.last_result = result
        ch = result.channels()
        # Re-rendering the payload already on screen (the overlay spinner moved)
        # is not a new run, so it must not push a second copy into the history.
        if payload is not self._last_payload:
            self._history.append((system, result))
            del self._history[:-MAX_OVERLAY]
            self._last_payload = payload
        axes = self.ax
        _clear_figure(self.fig, axes)

        # One km/h partner for the whole panel, not one per run: the unit is the
        # same every time, so an earlier run's ground speed belongs on this run's
        # scale instead of stacking another axis down the right edge.
        ground = None
        if system.wheel is not None:
            ground = axes[0][0].twinx()
            ground.set_ylabel("km/h", color=SERIES["ground"])
            ground.tick_params(axis="y", colors=SERIES["ground"])

        keep = self.overlay_box.value()
        earlier = self._history[-(keep + 1):-1] if keep else []
        for n, (old_system, old_result) in enumerate(earlier):
            # oldest faintest, so the run you just asked for stays the boldest line
            alpha = 0.15 + 0.45 * (n + 1) / len(earlier)
            self._plot_traces(axes, ground, old_result.channels(),
                              old_system.wheel is not None, alpha=alpha, labelled=False)

        v = system.voltage_limit if np.isfinite(system.voltage_limit) else 12.0
        steady = rad_s_to_rpm(system.steady_speed(v)) if kind == "step" else None
        self._plot_traces(axes, ground, result.channels(), system.wheel is not None,
                          alpha=1.0, labelled=True)

        if steady is not None and np.isfinite(steady):
            axes[0][0].axhline(steady, color=SERIES["reference"], ls=":", lw=1,
                               label=f"steady {steady:.0f} rpm")
        axes[0][0].set_ylabel("speed [rpm]")
        faded = f", {len(earlier)} earlier run{'s' if len(earlier) != 1 else ''} faded" if earlier else ""
        axes[0][0].set_title(
            f"{self.preset_name}  [{self.engine_box.currentText()}, {kind}{faded}]")
        axes[0][0].legend(fontsize=7, loc="best")

        limit = system.current_limit_total
        if np.isfinite(limit):
            axes[0][1].axhline(limit, color=SERIES["reference"], ls=":", lw=1,
                               label=f"limit {limit:g} A")
            axes[0][1].axhline(-limit, color=SERIES["reference"], ls=":", lw=1)
        axes[0][1].set_ylabel("current [A] / torque [N*m]")
        axes[0][1].legend(fontsize=7, loc="best")

        peak_eff = result.peak()["peak_efficiency"]
        if np.isfinite(peak_eff):
            axes[1][1].axhline(peak_eff, color=SERIES["reference"], ls=":", lw=1,
                               label=f"peak {peak_eff:.3f}")
        axes[1][1].set_ylabel("efficiency")
        axes[1][1].legend(fontsize=7, loc="best")

        for row in axes:
            for a in row:
                a.grid(alpha=0.35)
                a.set_xlabel("time [s]")
        self.canvas.draw_idle()

        self.metric_text.setPlainText(
            _format_metrics(system, result, v, kind) + "\n" + _format_describe(system))
        peak_kmh = ch["speed_kmh"][-1] if "speed_kmh" in ch else None
        self.set_status(
            f"done: {ch['motor_rpm'][-1]:.0f} rpm motor"
            + (f", {peak_kmh:.2f} km/h" if peak_kmh is not None else "")
            + f", peak current {result.peak()['peak_current_A']:.1f} A"
            + (f" | {len(earlier)} earlier run{'s' if len(earlier) != 1 else ''} overlaid"
               if earlier else "")
        )

    def _plot_traces(self, axes, ground, ch, has_wheel: bool, *, alpha: float,
                     labelled: bool) -> None:
        """Draw one run's curves on the four panels.

        Earlier runs come through thinner, fainter and unlabelled, so the legend
        still describes the run you just asked for and an overlay of six stays
        readable instead of becoming the mess it replaced.
        """
        t = ch["time"]
        w = 0.7 if alpha < 1.0 else 1.0
        axes[0][0].plot(t, ch["motor_rpm"], lw=1.6 * w, alpha=alpha,
                        color=SERIES["speed"],
                        label="motor shaft" if labelled else None)
        axes[0][0].plot(t, ch["output_rpm"], lw=1.4 * w, ls="--", alpha=alpha,
                        color=SERIES["output"],
                        label="output shaft" if labelled else None)
        if ground is not None and has_wheel:
            ground.plot(t, ch["speed_kmh"], lw=1.3 * w, ls=":", alpha=alpha,
                        color=SERIES["ground"],
                        label="ground speed" if labelled else None)
        axes[0][1].plot(t, ch["current_A"], lw=1.6 * w, alpha=alpha,
                        color=SERIES["current"],
                        label="total current" if labelled else None)
        axes[0][1].plot(t, ch["torque_Nm"], lw=1.4 * w, ls="--", alpha=alpha,
                        color=SERIES["torque"],
                        label="torque" if labelled else None)
        axes[1][0].plot(t, ch["power_in_W"], lw=1.5 * w, alpha=alpha,
                        color=SERIES["power_in"],
                        label="electrical in" if labelled else None)
        axes[1][0].plot(t, ch["power_out_W"], lw=1.5 * w, alpha=alpha,
                        color=SERIES["power_out"],
                        label="mechanical out" if labelled else None)
        axes[1][0].plot(t, ch["copper_loss_W"], lw=1.3 * w, ls="--", alpha=alpha,
                        color=SERIES["loss"],
                        label="copper loss" if labelled else None)
        axes[1][1].plot(t, ch["efficiency"], lw=1.6 * w, alpha=alpha,
                        color=SERIES["efficiency"],
                        label="efficiency" if labelled else None)

    def _redraw_response(self, *_ignored) -> None:
        """Re-plot the run already on screen when the overlay count changes."""
        if self._last_payload is not None:
            self._show_response(self._last_payload)

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    def run_sweep(self) -> None:
        try:
            path = self.sweep_param_box.currentText()
            lo, hi = float(self.sweep_lo.text()), float(self.sweep_hi.text())
            n = max(2, int(float(self.sweep_n.text())))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Sweep", f"bad sweep range: {exc}")
            return
        metric = self.sweep_metric_box.currentText()
        system = self._apply_overrides()
        workers = 0 if self.sweep_par_check.isChecked() else 1
        duration = float(self.duration_box.value())
        engine = self.engine_box.currentText()
        v = _drive_voltage(system)

        def job():
            values = np.linspace(lo, hi, n)
            if workers > 1:
                try:
                    return sweep(system, path, list(values), voltage=v, duration=duration,
                                 engine=engine, workers=workers, mp_context=_PARALLEL_START)
                except (RuntimeError, OSError):
                    # no fresh interpreter could be launched (for example when
                    # the GUI was started with ``python -c``): sweep in one
                    # process instead, which gives the same answers
                    pass
            return sweep(system, path, list(values), voltage=v, duration=duration,
                         engine=engine, workers=1)

        self._start(
            f"sweeping {path} over {n} points "
            f"({'parallel' if workers else 'inline'}) ...",
            job, lambda res: self._show_sweep(path, metric, res))

    def _show_sweep(self, path: str, metric: str, res) -> None:
        self.sweep_ax.clear()
        if metric not in res.columns:
            # presets without a wheel have no ground-speed metrics, and a
            # wheel-less run has no step-response metrics either
            self.sweep_ax.set_title(f"{metric} is not available for {self.preset_name}")
            self.sweep_out.setPlainText(
                "available metrics: " + ", ".join(sorted(res.columns)))
            self.sweep_canvas.draw_idle()
            self.set_status(f"sweep done, but {metric} is unavailable here")
            return
        x = np.asarray([v[0] for v in res.values], dtype=float)
        y = np.asarray(res.columns[metric], dtype=float)
        self.sweep_ax.plot(x, y, marker="o", ms=3, lw=1.4, color=SERIES["sweep"])
        self.sweep_ax.set_xlabel(path)
        self.sweep_ax.set_ylabel(metric)
        self.sweep_ax.grid(alpha=0.35)
        best = res.best(metric, maximize=True)
        self.sweep_ax.set_title(
            f"{metric} vs {path}   "
            f"(best {float(best[metric]):.4g} at {path} = {float(best[path]):.4g})")
        self.sweep_canvas.draw_idle()

        self.sweep_out.setPlainText(res.pretty(
            [path, metric, "peak_current_A", "rise_time_s"],
            max_rows=12, sort_by=metric, maximize=True))
        self.set_status(
            f"sweep done: {len(res)} points, best {metric} = {float(best[metric]):.4g}")

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    def run_compare(self) -> None:
        qty = self.compare_qty_box.currentText()
        duration = float(self.duration_box.value())
        engine = self.engine_box.currentText()
        names = presets.preset_names()

        def job():
            out = []
            for name in names:
                sys_ = presets.build(name)
                out.append((name, simulate(sys_, _drive_voltage(sys_), duration,
                                           dt=_plot_dt(duration), engine=engine)))
            return out

        self._start(f"comparing {len(names)} presets ...", job,
                    lambda res: self._show_compare(qty, res))

    def _show_compare(self, qty: str, rows) -> None:
        self.cmp_ax.clear()
        plotted = 0
        for name, res in rows:
            ch = res.channels()
            if qty not in ch:
                continue          # e.g. ground speed for presets without wheels
            self.cmp_ax.plot(ch["time"], ch[qty], lw=1.3, label=name)
            plotted += 1
        self.cmp_ax.set_ylabel(qty)
        self.cmp_ax.set_xlabel("time [s]")
        self.cmp_ax.grid(alpha=0.35)
        if plotted:
            self.cmp_ax.legend(fontsize=8, ncol=2)
        else:
            self.cmp_ax.set_title(f"no preset produces {qty}")
        self.cmp_canvas.draw_idle()
        self.set_status(f"compared {plotted} of {len(rows)} presets on {qty}")

    # ------------------------------------------------------------------
    # dataset fitting
    # ------------------------------------------------------------------
    def _rebuild_fit_list(self) -> None:
        """Refresh the fittable-parameter checklist, keeping current ticks."""
        if not hasattr(self, "fit_free"):
            return
        checked = set(self._fit_checked())
        system = self._apply_overrides()
        self.fit_free.blockSignals(True)
        self.fit_free.clear()
        for entry in presets.field_catalog(system):
            path = entry["path"]
            if path in fit.STATE_PATHS:
                continue
            value = entry["value"]
            unit = entry.get("unit") or ""
            item = QtWidgets.QListWidgetItem(f"{path}   {entry['label']}")
            item.setData(QtCore.Qt.UserRole, path)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if path in checked
                               else QtCore.Qt.Unchecked)
            tip = f"{path} = {value:g} {unit}".rstrip()
            if isinstance(value, float) and not math.isfinite(value):
                tip += "\nno limit: set a finite value in the panel before fitting"
            why = fit.structural_inert(system, path)
            if why:
                tip += f"\nno sensitivity in this configuration: {why}"
            item.setToolTip(tip)
            self.fit_free.addItem(item)
        self.fit_free.blockSignals(False)

    def _fit_checked(self) -> list[str]:
        paths = []
        for i in range(self.fit_free.count()):
            item = self.fit_free.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                path = item.data(QtCore.Qt.UserRole)
                if path:
                    paths.append(path)
        return paths

    def _fit_select(self, alias: str) -> None:
        wanted: set[str] | None
        if alias == "all":
            wanted = None
        elif alias == "none":
            wanted = set()
        else:
            wanted = set(fit.FREE_ALIASES.get(alias, ()))
        self.fit_free.blockSignals(True)
        for i in range(self.fit_free.count()):
            item = self.fit_free.item(i)
            path = item.data(QtCore.Qt.UserRole)
            on = wanted is None or path in wanted
            item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.fit_free.blockSignals(False)
        self.set_status(f"{len(self._fit_checked())} parameters free")

    def browse_dataset(self) -> None:
        path = self._open_path("Load dataset", "CSV (*.csv);;All files (*)")
        if not path:
            return
        self.fit_path.setText(path)
        self.describe_dataset()

    def describe_dataset(self) -> None:
        path = self.fit_path.text().strip()
        if not path:
            return
        try:
            ds = fit.Dataset.from_csv(path)
        except Exception as exc:
            self.set_status(f"dataset error: {exc}")
            QtWidgets.QMessageBox.warning(self, "Dataset", str(exc))
            return
        extra = f"  [{'; '.join(ds.notes)}]" if ds.notes else ""
        self.set_status(f"dataset {ds.describe()}{extra}")

    def run_fit(self) -> None:
        path = self.fit_path.text().strip()
        if not path:
            self.set_status("pick a dataset CSV first")
            QtWidgets.QMessageBox.information(
                self, "Fit", "Choose a measured-data CSV first.")
            return
        free = self._fit_checked()
        if not free:
            self.set_status("tick at least one parameter to fit")
            QtWidgets.QMessageBox.information(
                self, "Fit",
                "Tick at least one parameter for the fit to move.  Unticked ones are "
                "kept as known constants at their panel values.")
            return
        system = self._apply_overrides()
        engine = self.engine_box.currentText()
        voltage = float(self.fit_voltage.value())
        excitation = self.fit_excite_box.currentText()
        period = _number_or_none(self.fit_period.text())
        name = Path(path).name

        def job():
            ds = fit.Dataset.from_csv(path)
            return fit.fit(ds, system=system, free=tuple(free), engine=engine,
                           voltage=voltage, excitation=excitation, period=period)

        self._start(f"fitting {len(free)} parameters to {name} ...", job,
                    self._show_fit)

    def _show_fit(self, res: fit.FitResult) -> None:
        self.fit_result = res
        self.fit_out.setPlainText(res.report())
        self._plot_fit(res)
        self.fit_apply_btn.setEnabled(True)
        self.fit_save_btn.setEnabled(True)
        extra = f", {len(res.warnings)} warning(s)" if res.warnings else ""
        self.set_status(
            f"fit: rms {res.normalized_rms:.3g} weight units, "
            f"condition {_fmt_num(res.condition)}{extra}")

    def _plot_fit(self, res: fit.FitResult) -> None:
        channels = list(res.rms)[:3]          # worst residuals first
        self.fit_fig.clear()
        # Figure.clear() drops the layout engine, which is what hides the tick
        # labels and leaves the axes bunched to one side, so re-assert it here
        self.fit_fig.set_layout_engine("constrained")
        self.fit_axes = list(np.atleast_1d(self.fit_fig.subplots(1, len(channels) or 1)))
        self.fit_ax = self.fit_axes[0]
        axes = self.fit_axes[:len(channels)]
        if not channels:
            _placeholder(self.fit_ax, "nothing to overlay")
            self.fit_canvas.draw_idle()
            return
        try:
            predictions = res.predict()
        except Exception as exc:      # the overlay is a bonus, not a requirement
            self.set_status(f"overlay skipped: {type(exc).__name__}: {exc}")
            predictions = []
        steady = any(p.get("kind") == "steady" for p in predictions)
        for ax, ch in zip(axes, channels):
            first = True
            for pred in predictions:
                x = pred["data"].get("voltage_V") if steady else pred.get("time")
                if x is None:
                    continue
                data = pred["data"].get(ch)
                model = pred["model"].get(ch)
                if data is None:
                    continue
                ax.plot(x, data, color=GRUVBOX["gray"], lw=1.0, marker="o", ms=3.5,
                        label="measured" if first else None)
                if model is not None:
                    ax.plot(x, model, color=SERIES["sweep"], lw=1.5,
                            label="model" if first else None)
                first = False
            ax.set_title(f"{ch}  rms {res.rms[ch]:.3g}", fontsize=9)
            ax.set_xlabel("drive voltage [V]" if steady else "time [s]")
            ax.grid(alpha=0.35)
        if any(ax.get_legend_handles_labels()[1] for ax in axes):
            axes[0].legend(fontsize=8, ncol=2)
        self.fit_canvas.draw_idle()

    def apply_fit(self) -> None:
        if self.fit_result is None:
            return
        self._adopt_model(self.fit_result.system,
                          f"applied the fit to {self.fit_result.dataset}")

    def save_fit(self) -> None:
        if self.fit_result is None:
            QtWidgets.QMessageBox.information(self, "Save", "Run a fit first.")
            return
        path = self._save_path("Save fitted model", "fitted_model.json",
                               "JSON (*.json)")
        if not path:
            return
        try:
            self.fit_result.save(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save", str(exc))
            return
        self.set_status(f"wrote {path}")

    def _adopt_model(self, system: DriveSystem, status: str) -> None:
        """Make ``system`` the working model and resync every control."""
        self.overrides.clear()
        self.wheel_name = None
        self.base_system = self.system = system
        matched = _match_preset(system)
        if matched is not None and matched != self.preset_name:
            self.preset_name = matched
            self.preset_box.blockSignals(True)
            self.preset_box.setCurrentText(matched)
            self.preset_box.blockSignals(False)
        self._sync_catalogue_boxes()
        self.rebuild_params()
        self.set_status(status)
        self.run()

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_csv(self) -> None:
        if self.last_result is None:
            QtWidgets.QMessageBox.information(self, "Export", "Run a simulation first.")
            return
        path = self._save_path("Export CSV", "response.csv", "CSV (*.csv)")
        if not path:
            return
        ch = self.last_result.channels()
        keys = list(ch)
        with open(path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in zip(*(ch[k] for k in keys)):
                fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
        self.set_status(f"wrote {path}")

    def save_png(self) -> None:
        path = self._save_path("Save PNG", "response.png", "PNG (*.png)")
        if not path:
            return
        self.fig.savefig(path, dpi=150)
        self.set_status(f"wrote {path}")

    def save_config(self) -> None:
        path = self._save_path("Save config", "config.json", "JSON (*.json)")
        if not path:
            return
        presets.save(path, self._apply_overrides())
        self.set_status(f"wrote {path}")

    def load_config(self) -> None:
        path = self._open_path("Load config", "JSON (*.json)")
        if not path:
            return
        try:
            system = presets.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load", str(exc))
            return
        self._adopt_model(system, f"loaded {path}")

    def export_model_code(self) -> None:
        """Write the model out as C, Rust and Python for a controller.

        The model is something you inspect; the code is something you ask for,
        once, into a source tree you control.  The control period is asked for
        rather than assumed: it is a property of the controller, not of the
        motor, and the generated code is sized around it.
        """
        system = self._apply_overrides()
        try:
            constants = export.ModelConstants.from_system(system)
        except DrivetrainError as exc:
            QtWidgets.QMessageBox.critical(self, "Export model code", str(exc))
            return
        limit = constants.max_step_time()
        seconds = self._ask_sample_time(0.002, limit)
        if seconds is None:
            return
        directory = self._open_directory("Export model code", "drive_model_export")
        if not directory:
            return
        options = export.ExportOptions(
            sample_time=seconds,
            provenance={"source": f"GUI ({self.preset_name})"},
        )
        try:
            files = export.export(system, directory, options)
        except (DrivetrainError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Export model code", str(exc))
            return
        self.set_status(
            f"wrote {len(files)} files to {directory} "
            f"({options.dtype}, {', '.join(options.targets)}, "
            f"control period {seconds:g} s, max step {limit:g} s)"
        )

    def export_dq_code(self) -> None:
        """Write the dq/FOC model of a brushless machine out as Python.

        Same dialog shape as the brushed export, because the question asked is
        the same one: what control period does your controller run at.  The
        bound is different though, and that difference is the point: a dq plant
        has a complex pole pair that rotates with the shaft, so the stable step
        at base speed can be several times smaller than at standstill.
        """
        system = self._apply_overrides()
        try:
            constants = dq_export.DqModelConstants.from_system(system)
        except DrivetrainError as exc:
            QtWidgets.QMessageBox.critical(self, "Export dq/FOC code", str(exc))
            return
        limit = constants.max_step_time()
        at_base = constants.max_step_time(constants.base_speed)
        seconds = self._ask_sample_time(0.0005, limit)
        if seconds is None:
            return
        directory = self._open_directory("Export dq/FOC code", "drive_model_dq_export")
        if not directory:
            return
        options = export.ExportOptions(
            targets=("python",),
            sample_time=seconds,
            provenance={"source": f"GUI ({self.preset_name})"},
        )
        try:
            files = dq_export.export_dq(system, directory, options)
        except (DrivetrainError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Export dq/FOC code", str(exc))
            return
        status = (
            f"wrote {len(files)} files to {directory} "
            f"({options.dtype}, python, control period {seconds:g} s, "
            f"stable step {limit:g} s at standstill / {at_base:g} s at base speed)"
        )
        if seconds > at_base:
            status += (
                " - above the stable step at speed: use integrate(), which "
                "sub-steps; step() raises there"
            )
        self.set_status(status)

    def efficiency_map_csv(self) -> None:
        """Write the loss map of the current model out as CSV plus a text report.

        The map answers where the watts go over the whole (speed, torque) plane,
        which is the question a single closed-form operating point cannot answer
        for a brushless machine: the dq envelope is not a straight line, and the
        best point sits inside it rather than on it.
        """
        system = self._apply_overrides()
        directory = self._open_directory("Efficiency map", "drive_model_efficiency")
        if not directory:
            return
        try:
            emap = eff.efficiency_map(system, include_gearbox=True)
            csv_path = emap.write_csv(Path(directory) / "efficiency_map.csv")
            report_path = Path(directory) / "efficiency_map.txt"
            report_path.write_text(emap.report() + "\n", encoding="utf-8")
        except (DrivetrainError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Efficiency map", str(exc))
            return
        status = (
            f"efficiency map {emap.shape[0]}x{emap.shape[1]}, "
            f"{emap.n_feasible()} feasible cells -> {csv_path} and {report_path}"
        )
        best = emap.best("motoring")
        if best is not None:
            status += (
                f"; best motoring {best['speed_rpm']:.0f} rpm at "
                f"{best['torque_Nm']:.4g} N*m -> {100.0 * best['efficiency']:.1f}%"
            )
        self.set_status(status)

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, "About",
            "Drivetrain Model Explorer<br><br>Qt GUI via qtpy "
            f"({_qt_api_name()}), themed to Gruvbox dark.")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def _save_path(self, title: str, suggestion: str, pattern: str) -> str:
        """Ask for a output file name ('' means the user cancelled).

        Kept as a thin seam so the dialogs can be replaced in tests.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, suggestion, pattern)
        return path

    def _open_path(self, title: str, pattern: str, start: str = "") -> str:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, pattern)
        return path

    def _open_directory(self, title: str, suggestion: str) -> str:
        """Ask for a directory to write generated code into ('' = cancelled)."""
        return QtWidgets.QFileDialog.getExistingDirectory(self, title, suggestion)

    def _ask_sample_time(self, default: float, maximum: float) -> float | None:
        """Ask for the control period ('' None means the user cancelled).

        Bounded by the model's own stability limit, so a typo cannot produce a
        build whose contract the integrator cannot keep.
        """
        upper = max(default, maximum)
        value, ok = QtWidgets.QInputDialog.getDouble(
            self, "Export model code", "Control period [s]:", default, 1e-6, upper, 6)
        return value if ok else None

    def set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def is_busy(self) -> bool:
        """True while a simulation is queued, debounced or running."""
        return bool(self.worker.busy or not self.worker.jobs.empty()
                    or self._live_timer.isActive())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.worker.stop()
        self.worker.wait(3000)
        super().closeEvent(event)


def _qt_api_name() -> str:
    try:  # pragma: no cover - cosmetic
        import qtpy
        return f"{qtpy.API_NAME}, Qt {qtpy.QT_VERSION}"
    except Exception:
        return QtCore.qVersion()


def _match_preset(system: DriveSystem) -> str | None:
    """Which preset a loaded config is identical to, if any."""
    try:
        payload = presets.to_dict(system)
    except Exception:
        return None
    for name, cand in presets.PRESETS.items():
        if presets.to_dict(cand.system) == payload:
            return name
    return None


def _match_motor(motor) -> str | None:
    """Which catalogue motor the model currently uses, if any."""
    for name, cand in presets.MOTORS.items():
        if (cand.kv, cand.kt, cand.resistance, cand.rotor_inertia,
                cand.max_current, cand.max_voltage) == (
                motor.kv, motor.kt, motor.resistance, motor.rotor_inertia,
                motor.max_current, motor.max_voltage):
            return name
    return None


def _match_wheel(wheel) -> str | None:
    """Which catalogue wheel the model currently uses, if any."""
    if wheel is None:
        return None
    for name, cand in presets.WHEELS.items():
        if dataclasses.asdict(cand) == dataclasses.asdict(wheel):
            return name
    return None


def _fmt_num(v: float) -> str:
    if v is None:
        return ""
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e6:
        return f"{v:.3e}"
    return f"{v:.5g}"


def _format_metrics(system: DriveSystem, res: SimulationResult, v: float, kind: str) -> str:
    """Human-readable metric block shown under the response plots."""
    m = res.metrics()
    p = res.peak()
    f = res.final()

    def g(d, key, fmt="{:g}"):
        val = d.get(key, float("nan"))
        return fmt.format(val) if val is not None and np.isfinite(val) else "-"

    ground = (f"   {system.steady_ground_speed(v) * 3.6:8.2f} km/h"
              if system.wheel is not None else "")
    return "\n".join([
        f"steady speed      {rad_s_to_rpm(system.steady_speed(v)):>10.1f} rpm motor{ground}",
        f"no-load speed     {rad_s_to_rpm(system.no_load_speed):>10.1f} rpm"
        f"   equilibrium mode: {system.steady_mode(v)}",
        f"stall torque      {system.stall_torque * v:>10.3f} N*m at motor shaft"
        f"   ({system.stall_torque:.4g} N*m per volt)",
        f"mech time const   {system.mech_time_constant:>10.4f} s"
        f"   settle ~{system.settle_time:.2f} s   electrical tau "
        f"{system.electrical_time_constant:.2e} s",
        f"rise / settle     {g(m, 'rise_time_s', '{:9.3f}')} s"
        f"   {g(m, 'settle_time_s', '{:8.3f}')} s   overshoot {g(m, 'overshoot_pct', '{:6.1f}')}%",
        f"peak current      {p['peak_current_A']:>10.2f} A total"
        f"   {p['peak_current_per_motor_A']:.2f} A each   torque {p['peak_torque_Nm']:.3f} N*m",
        f"power in / loss   {p['peak_power_in_W']:>10.1f} W"
        f"   copper {p['peak_copper_loss_W']:.1f} W   peak eff {p['peak_efficiency']:.3f}",
        f"final ({kind})     {f['motor_rpm']:>10.1f} rpm"
        f"   {g(f, 'speed_kmh', '{:9.2f}')} km/h   distance {g(f, 'distance_m', '{:7.2f}')} m",
    ]) + "\n"


def _format_describe(system: DriveSystem) -> str:
    d = system.describe()
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
            parts.append(f"{k}={v:.4g}")
    return "system: " + "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the GUI (also reachable via ``python3 run.py gui``)."""
    import argparse

    ap = argparse.ArgumentParser(prog="drivetrain gui")
    ap.add_argument("--preset", default="legacy", choices=presets.preset_names())
    ap.add_argument("--theme", default="gruvbox", choices=("gruvbox", "system"),
                    help="gruvbox dark palette, or the desktop's own style")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Drivetrain Model Explorer")
    apply_theme(app, args.theme)
    window = DriveApp(preset=args.preset)
    window.show()
    run = getattr(app, "exec", None) or app.exec_   # Qt6 dropped exec_
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
