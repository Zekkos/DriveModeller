"""Interactive Tkinter + Matplotlib interface for the drivetrain model.

Run it with::

    python3 run.py            # or:  python3 -m drivetrain gui
    python3 run.py gui --preset pod_4x_geared

Layout: parameter sliders on the left (auto-generated from the model's own field
catalogue, so new parameters appear in the UI with no UI changes), tabbed plots
on the right, and a metrics read-out underneath.

Design notes
------------
* Simulations run on a worker thread and post results through a queue, so long
  sweeps never freeze the window.  All widget mutation happens on the Tk thread
  via ``root.after`` polling -- Tkinter is not thread-safe and must not be
  touched from the worker.
* Edits are stored as *overrides* applied to the selected preset.  That keeps the
  model itself the single source of truth and makes every UI state reproducible:
  the override dict is exactly what save/load JSON writes.
* The sweep tab fans combinations out over processes (``drivetrain.sweep``),
  which is where the wall-clock win actually lives.
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import presets
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

try:  # Tkinter is optional at import time so the library stays usable headless
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on the distro's python-tk
    raise DrivetrainError(
        "Tkinter is not available. On Debian/Ubuntu: sudo apt install python3-tk"
    ) from exc


ENGINES = ("exact", "rk4", "euler")
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


class _Worker(threading.Thread):
    """Runs jobs on a background thread; each job carries its own completion callback.

    Callbacks run on the Tk thread (the poll loop invokes them), so widget updates
    stay on the thread that owns them.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.jobs: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.busy = False

    def submit(self, on_result, fn, *args, **kwargs) -> None:
        self.jobs.put((on_result, fn, args, kwargs))

    def run(self) -> None:  # noqa: D102 - Thread override
        while True:
            on_result, fn, args, kwargs = self.jobs.get()
            try:
                self.results.put((on_result, ("ok", fn(*args, **kwargs))))
            except Exception as exc:  # surface model/config errors in the UI
                traceback.print_exc()
                self.results.put((on_result, ("error", f"{type(exc).__name__}: {exc}")))
            finally:
                self.jobs.task_done()


class DriveApp:
    """Main application window."""

    def __init__(self, root: tk.Tk, preset: str = "legacy") -> None:
        self.root = root
        self.worker = _Worker()
        self.worker.start()

        self.preset_name = preset
        self.system: DriveSystem = presets.build(preset)
        self.overrides: dict[str, float] = {}
        self.sliders: dict[str, dict[str, Any]] = {}
        self.last_result = None

        root.title("Drivetrain Model Explorer")
        root.geometry("1420x880")
        root.minsize(1100, 640)

        self._build_toolbar()
        self._build_body()
        self._build_status()

        self.load_preset(preset)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_results()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Preset").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value=self.preset_name)
        self.preset_box = ttk.Combobox(
            bar, textvariable=self.preset_var, values=presets.preset_names(),
            width=16, state="readonly")
        self.preset_box.pack(side=tk.LEFT, padx=(4, 12))
        self.preset_box.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset_var.get()))

        ttk.Label(bar, text="Motor").pack(side=tk.LEFT)
        self.motor_var = tk.StringVar()
        self.motor_box = ttk.Combobox(bar, textvariable=self.motor_var,
                                      values=presets.motor_names(), width=14, state="readonly")
        self.motor_box.pack(side=tk.LEFT, padx=(4, 12))
        self.motor_box.bind("<<ComboboxSelected>>", self._on_motor_choice)

        ttk.Label(bar, text="Engine").pack(side=tk.LEFT)
        self.engine_var = tk.StringVar(value="exact")
        ttk.Combobox(bar, textvariable=self.engine_var, values=ENGINES,
                     width=7, state="readonly").pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(bar, text="Excitation").pack(side=tk.LEFT)
        self.excite_var = tk.StringVar(value="step")
        ttk.Combobox(bar, textvariable=self.excite_var, values=EXCITATIONS,
                     width=12, state="readonly").pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(bar, text="Duration (s)").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="5")
        ttk.Entry(bar, textvariable=self.duration_var, width=6).pack(side=tk.LEFT, padx=(4, 12))

        self.run_btn = ttk.Button(bar, text="Run  (Enter)", command=self.run)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Reset params", command=self.reset_overrides).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Save config", command=self.save_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Load config", command=self.load_config).pack(side=tk.LEFT, padx=4)

        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Live", variable=self.live_var).pack(side=tk.LEFT, padx=(12, 0))
        self.root.bind("<Return>", lambda e: self.run())

    def _build_body(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # ---- left: scrollable parameter panel -------------------------
        left = ttk.Frame(paned, width=390)
        paned.add(left, weight=0)

        hdr = ttk.Frame(left)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Parameters", font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="collapse all", width=12,
                   command=lambda: self._toggle_sections(False)).pack(side=tk.RIGHT)
        ttk.Button(hdr, text="expand all", width=11,
                   command=lambda: self._toggle_sections(True)).pack(side=tk.RIGHT)

        self.param_canvas = tk.Canvas(left, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.param_canvas.yview)
        self.param_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.param_frame = ttk.Frame(self.param_canvas)
        self._param_window = self.param_canvas.create_window(
            (0, 0), window=self.param_frame, anchor="nw")
        self.param_frame.bind("<Configure>", self._on_param_configure)
        self.param_canvas.bind("<Configure>", self._on_canvas_configure)
        # mouse wheel scrolling, including over child widgets
        self.param_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.param_canvas.bind_all("<Button-4>", lambda e: self.param_canvas.yview_scroll(-3, "units"))
        self.param_canvas.bind_all("<Button-5>", lambda e: self.param_canvas.yview_scroll(3, "units"))

        # ---- right: tabs of plots -------------------------------------
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_response_tab()
        self._build_sweep_tab()
        self._build_compare_tab()

    def _build_response_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Time response")

        self.fig = Figure(figsize=(9.2, 5.6), dpi=100, constrained_layout=True)
        self.ax = self.fig.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, tab)

        bottom = ttk.Frame(tab)
        bottom.pack(fill=tk.X)
        self.metric_text = tk.Text(bottom, height=7, width=110, font=("TkFixedFont", 9),
                                   background="#f7f7f4")
        self.metric_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.metric_text.configure(state="disabled")
        ttk.Button(bottom, text="Export CSV", command=self.export_csv).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom, text="Save PNG", command=self.save_png).pack(side=tk.RIGHT)

    def _build_sweep_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Parameter sweep")

        ctl = ttk.Frame(tab, padding=6)
        ctl.pack(fill=tk.X)
        ttk.Label(ctl, text="Parameter").pack(side=tk.LEFT)
        self.sweep_param = tk.StringVar(value="gearbox.ratio")
        self.sweep_param_box = ttk.Combobox(ctl, textvariable=self.sweep_param,
                                            width=30, values=[])
        self.sweep_param_box.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctl, text="from").pack(side=tk.LEFT)
        self.sweep_lo = tk.StringVar(value="0.05")
        ttk.Entry(ctl, textvariable=self.sweep_lo, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctl, text="to").pack(side=tk.LEFT)
        self.sweep_hi = tk.StringVar(value="1.0")
        ttk.Entry(ctl, textvariable=self.sweep_hi, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctl, text="points").pack(side=tk.LEFT)
        self.sweep_n = tk.StringVar(value="40")
        ttk.Entry(ctl, textvariable=self.sweep_n, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctl, text="Metric").pack(side=tk.LEFT, padx=(10, 0))
        self.sweep_metric = tk.StringVar(value="steady_rpm")
        ttk.Combobox(ctl, textvariable=self.sweep_metric, values=SWEEP_METRICS,
                     width=18, state="readonly").pack(side=tk.LEFT, padx=4)

        self.sweep_par = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="parallel", variable=self.sweep_par).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctl, text="Run sweep", command=self.run_sweep).pack(side=tk.LEFT, padx=6)

        self.sweep_fig = Figure(figsize=(9.2, 5.0), dpi=100, constrained_layout=True)
        self.sweep_ax = self.sweep_fig.subplots(1, 1)
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=tab)
        self.sweep_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.sweep_canvas, tab)

        self.sweep_out = tk.Text(tab, height=8, font=("TkFixedFont", 9), background="#f7f7f4")
        self.sweep_out.pack(fill=tk.X)

    def _build_compare_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Compare presets")

        ctl = ttk.Frame(tab, padding=6)
        ctl.pack(fill=tk.X)
        ttk.Label(ctl, text="Overlay every preset's response on one chart").pack(side=tk.LEFT)
        ttk.Button(ctl, text="Compare", command=self.run_compare).pack(side=tk.RIGHT, padx=4)
        ttk.Label(ctl, text="quantity").pack(side=tk.RIGHT)
        self.compare_qty = tk.StringVar(value="motor_rpm")
        ttk.Combobox(ctl, textvariable=self.compare_qty, values=PLOT_CHANNELS,
                     width=16, state="readonly").pack(side=tk.RIGHT, padx=4)

        self.cmp_fig = Figure(figsize=(9.2, 5.6), dpi=100, constrained_layout=True)
        self.cmp_ax = self.cmp_fig.subplots(1, 1)
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, master=tab)
        self.cmp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.cmp_canvas, tab)

    def _build_status(self) -> None:
        self.status = ttk.Label(self.root, text="ready", relief=tk.SUNKEN, anchor=tk.W,
                                padding=(6, 3))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # parameter panel plumbing
    # ------------------------------------------------------------------
    def _on_param_configure(self, _event) -> None:
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all") or (0, 0, 0, 0))

    def _on_canvas_configure(self, event) -> None:
        self.param_canvas.itemconfigure(self._param_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self.param_canvas.yview_scroll(int(-event.delta / 60) if event.delta else 0, "units")

    def _toggle_sections(self, open_: bool) -> None:
        for state in self.sliders.values():
            if "body" in state:
                state["body"].pack_forget() if not open_ else state["body"].pack(fill=tk.X)
                state["open"] = open_
                state["toggle"].configure(text="-" if open_ else "+")
        self.param_frame.event_generate("<Configure>")

    def rebuild_params(self) -> None:
        """Rebuild the slider panel from the current model's field catalogue."""
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.sliders.clear()

        catalog = presets.field_catalog(self.system)
        sweep_values = []
        groups: dict[str, list[dict]] = {}
        for entry in catalog:
            groups.setdefault(entry["group"], []).append(entry)
            sweep_values.append(entry["path"])

        order = ["Drivetrain", "Motor", "Gearbox", "Load", "Wheel", "Other"]
        for group in [g for g in order if g in groups] + [g for g in groups if g not in order]:
            self._build_group(group, groups[group])

        self.sweep_param_box.configure(values=sweep_values)
        if self.sweep_param.get() not in sweep_values and sweep_values:
            self.sweep_param.set(sweep_values[0])
        self._sync_ranges()
        self.param_frame.event_generate("<Configure>")

    def _build_group(self, name: str, entries: list[dict]) -> None:
        head = ttk.Frame(self.param_frame)
        head.pack(fill=tk.X, pady=(8, 2))
        toggle = ttk.Button(head, text="-", width=2,
                            command=lambda n=name: self._toggle_group(n))
        toggle.pack(side=tk.LEFT)
        ttk.Label(head, text=name, font=("", 10, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Separator(head, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        body = ttk.Frame(self.param_frame)
        body.pack(fill=tk.X)
        self.sliders.setdefault("__sections__", {})[name] = {"body": body, "open": True,
                                                             "toggle": toggle}
        for entry in entries:
            self._build_slider(body, entry)

    def _toggle_group(self, name: str) -> None:
        state = self.sliders["__sections__"][name]
        state["open"] = not state["open"]
        if state["open"]:
            state["body"].pack(fill=tk.X)
            state["toggle"].configure(text="-")
        else:
            state["body"].pack_forget()
            state["toggle"].configure(text="+")
        self.param_frame.event_generate("<Configure>")

    def _build_slider(self, parent: ttk.Frame, entry: dict) -> None:
        path = entry["path"]
        value = float(entry["value"])
        lo, hi, step = float(entry["min"]), float(entry["max"]), float(entry["step"])
        is_int = entry.get("type") == "int"

        # keep the current value inside the slider range
        if not (lo <= value <= hi):
            lo, hi = min(lo, value), max(hi, value)

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=2, pady=1)

        label = entry["label"] + (f"  [{entry['unit']}]" if entry["unit"] else "")
        ttk.Label(row, text=label, width=26, anchor="w").pack(side=tk.LEFT)

        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                          command=lambda _v, p=path: self._on_slider(p))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ent = ttk.Entry(row, width=10)
        ent.insert(0, _fmt_num(value))
        ent.pack(side=tk.LEFT, padx=(4, 0))
        ent.bind("<Return>", lambda e, p=path: self._on_entry(p))
        ent.bind("<FocusOut>", lambda e, p=path: self._on_entry(p))

        self.sliders[path] = {
            "var": var, "scale": scale, "entry": ent, "lo": lo, "hi": hi,
            "step": step, "int": is_int, "label": entry["label"], "unit": entry["unit"],
            "default": value,
        }
        var.trace_add("write", lambda *_: self._on_slider(path))

    def _on_slider(self, path: str) -> None:
        state = self.sliders[path]
        value = state["var"].get()
        if state["int"]:
            value = round(value)
        if state["entry"] and state["entry"].get() != _fmt_num(value):
            state["entry"].delete(0, tk.END)
            state["entry"].insert(0, _fmt_num(value))
        self.overrides[path] = value
        self.set_status(f"{path} = {_fmt_num(value)}")
        if self.live_var.get():
            self.run()

    def _on_entry(self, path: str) -> None:
        state = self.sliders[path]
        raw = state["entry"].get().strip()
        try:
            value = float(raw)
        except ValueError:
            self.set_status(f"{path}: not a number, ignored")
            state["entry"].delete(0, tk.END)
            state["entry"].insert(0, _fmt_num(state["var"].get()))
            return
        value = round(value) if state["int"] else value
        state["var"].set(value)
        self.overrides[path] = value
        if self.live_var.get():
            self.run()

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
        self.preset_name = name
        self.preset_var.set(name)
        self.overrides.clear()
        self.system = presets.build(name)
        self.motor_var.set(_guess_motor(name))
        self.rebuild_params()
        self.run()

    def _on_motor_choice(self, _event=None) -> None:
        name = self.motor_var.get()
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
        self.run()

    def _apply_overrides(self) -> DriveSystem:
        """Rebuild the model from the preset plus the current override dict."""
        system = presets.build(self.preset_name)
        for path, value in self.overrides.items():
            try:
                system = set_path(system, path, value)
            except Exception as exc:
                self.set_status(f"ignored {path}: {exc}")
        return system

    def reset_overrides(self) -> None:
        self.overrides.clear()
        self.system = presets.build(self.preset_name)
        self.rebuild_params()
        self.run()

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            duration = 5.0
        system = self._apply_overrides()
        self.system = system
        engine = self.engine_var.get()
        kind = self.excite_var.get()

        def job():
            v = voltage_signal(kind, system, duration)
            res = simulate(system, v, duration, engine=engine)
            return system, res, kind

        self._start(f"simulating {self.preset_name} [{engine}] for {duration:g} s ...",
                    job, self._show_response)

    def _start(self, status: str, job, on_result) -> None:
        self.worker.busy = True
        self.run_btn.configure(state="disabled")
        self.set_status(status)
        self.worker.submit(on_result, job)

    def _poll_results(self) -> None:
        try:
            on_result, (kind, payload) = self.worker.results.get_nowait()
            self.worker.busy = False
            self.run_btn.configure(state="normal")
            if kind == "error":
                self.set_status(payload)
                messagebox.showerror("Simulation failed", payload)
            else:
                on_result(payload)
        except queue.Empty:
            pass
        self.root.after(40, self._poll_results)

    def _show_response(self, payload) -> None:
        system, result, kind = payload
        self.last_result = result
        ch = result.channels()
        t = ch["time"]
        axes = self.ax
        for row in axes:
            for a in row:
                a.clear()

        v = system.voltage_limit if np.isfinite(system.voltage_limit) else 12.0
        steady = rad_s_to_rpm(system.steady_speed(v)) if kind == "step" else None

        axes[0][0].plot(t, ch["motor_rpm"], lw=1.6, label="motor shaft")
        axes[0][0].plot(t, ch["output_rpm"], lw=1.4, ls="--", label="output shaft")
        if system.wheel is not None:
            ax = axes[0][0].twinx()
            ax.plot(t, ch["speed_kmh"], lw=1.3, ls=":", color="tab:green", label="ground speed")
            ax.set_ylabel("km/h", color="tab:green")
        if steady is not None and np.isfinite(steady):
            axes[0][0].axhline(steady, color="0.5", ls=":", lw=1,
                               label=f"steady {steady:.0f} rpm")
        axes[0][0].set_ylabel("speed [rpm]")
        axes[0][0].set_title(f"{self.preset_name}  [{self.engine_var.get()}, {kind}]")
        axes[0][0].legend(fontsize=7, loc="best")

        axes[0][1].plot(t, ch["current_A"], lw=1.6, color="tab:red", label="total current")
        axes[0][1].plot(t, ch["torque_Nm"], lw=1.4, color="tab:blue", ls="--", label="torque")
        limit = system.current_limit_total
        if np.isfinite(limit):
            axes[0][1].axhline(limit, color="k", ls=":", lw=1, label=f"limit {limit:g} A")
            axes[0][1].axhline(-limit, color="k", ls=":", lw=1)
        axes[0][1].set_ylabel("current [A] / torque [N*m]")
        axes[0][1].legend(fontsize=7, loc="best")

        axes[1][0].plot(t, ch["power_in_W"], lw=1.5, label="electrical in")
        axes[1][0].plot(t, ch["power_out_W"], lw=1.5, label="mechanical out")
        axes[1][0].plot(t, ch["copper_loss_W"], lw=1.3, ls="--", label="copper loss")
        axes[1][0].set_ylabel("power [W]")
        axes[1][0].legend(fontsize=7, loc="best")

        axes[1][1].plot(t, ch["efficiency"], lw=1.6, color="tab:green", label="efficiency")
        peak_eff = result.peak()["peak_efficiency"]
        if np.isfinite(peak_eff):
            axes[1][1].axhline(peak_eff, color="0.5", ls=":", lw=1, label=f"peak {peak_eff:.3f}")
        axes[1][1].set_ylabel("efficiency")
        axes[1][1].legend(fontsize=7, loc="best")

        for row in axes:
            for a in row:
                a.grid(alpha=0.3)
                a.set_xlabel("time [s]")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.metric_text.configure(state="normal")
        self.metric_text.delete("1.0", tk.END)
        self.metric_text.insert(tk.END, _format_metrics(system, result, v, kind))
        self.metric_text.insert(tk.END, "\n" + _format_describe(system))
        self.metric_text.configure(state="disabled")
        self.set_status(
            f"done: {ch['motor_rpm'][-1]:.0f} rpm motor, {ch['speed_kmh'][-1]:.2f} km/h, "
            f"peak current {result.peak()['peak_current_A']:.1f} A"
        )

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    def run_sweep(self) -> None:
        try:
            path = self.sweep_param.get()
            lo, hi = float(self.sweep_lo.get()), float(self.sweep_hi.get())
            n = max(2, int(float(self.sweep_n.get())))
        except Exception as exc:
            messagebox.showerror("Sweep", f"bad sweep range: {exc}")
            return
        metric = self.sweep_metric.get()
        system = self._apply_overrides()
        workers = 0 if self.sweep_par.get() else 1
        duration = float(self.duration_var.get() or 5)
        engine = self.engine_var.get()
        v = system.voltage_limit

        def job():
            values = np.linspace(lo, hi, n)
            return sweep(system, path, list(values), voltage=v, duration=duration,
                         engine=engine, workers=workers)

        self._start(
            f"sweeping {path} over {n} points "
            f"({'parallel' if workers else 'inline'}) ...",
            job, lambda res: self._show_sweep(path, metric, res))

    def _show_sweep(self, path: str, metric: str, res) -> None:
        self.sweep_ax.clear()
        x = np.asarray([v[0] for v in res.values], dtype=float)
        y = np.asarray(res.columns[metric], dtype=float)
        self.sweep_ax.plot(x, y, marker="o", ms=3, lw=1.4)
        self.sweep_ax.set_xlabel(path)
        self.sweep_ax.set_ylabel(metric)
        self.sweep_ax.grid(alpha=0.3)
        best = res.best(metric, maximize=True)
        self.sweep_ax.set_title(
            f"{metric} vs {path}   "
            f"(best {float(best[metric]):.4g} at {path} = {float(best[path]):.4g})")
        self.sweep_fig.tight_layout()
        self.sweep_canvas.draw_idle()

        self.sweep_out.configure(state="normal")
        self.sweep_out.delete("1.0", tk.END)
        self.sweep_out.insert(tk.END, res.pretty([path, metric, "peak_current_A", "rise_time_s"],
                                                 max_rows=12, sort_by=metric, maximize=True))
        self.sweep_out.configure(state="disabled")
        self.set_status(f"sweep done: {len(res)} points, best {metric} = {float(best[metric]):.4g}")

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    def run_compare(self) -> None:
        qty = self.compare_qty.get()
        duration = float(self.duration_var.get() or 5)
        engine = self.engine_var.get()
        names = presets.preset_names()

        def job():
            out = []
            for name in names:
                sys_ = presets.build(name)
                out.append((name, simulate(sys_, sys_.voltage_limit, duration, engine=engine)))
            return out

        self._start(f"comparing {len(names)} presets ...", job,
                    lambda res: self._show_compare(qty, res))

    def _show_compare(self, qty: str, rows) -> None:
        self.cmp_ax.clear()
        for name, res in rows:
            self.cmp_ax.plot(res.channels()["time"], res.channels()[qty], lw=1.3, label=name)
        self.cmp_ax.set_ylabel(qty)
        self.cmp_ax.set_xlabel("time [s]")
        self.cmp_ax.grid(alpha=0.3)
        self.cmp_ax.legend(fontsize=8, ncol=2)
        self.cmp_fig.tight_layout()
        self.cmp_canvas.draw_idle()
        self.set_status(f"compared {len(rows)} presets on {qty}")

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_csv(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("Export", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
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
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if not path:
            return
        self.fig.savefig(path, dpi=150)
        self.set_status(f"wrote {path}")

    def save_config(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        presets.save(self._apply_overrides(), path)
        self.set_status(f"wrote {path}")

    def load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            system = presets.load(path)
        except Exception as exc:
            messagebox.showerror("Load", str(exc))
            return
        self.overrides.clear()
        self.system = system
        self.rebuild_params()
        self.run()

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _on_close(self) -> None:
        self.root.destroy()
def _guess_motor(preset: str) -> str:
    """Best-effort motor-catalogue match for a preset name."""
    for name in presets.MOTORS:
        if name in preset:
            return name
    return next(iter(presets.MOTORS))


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
        f"   {f['speed_kmh']:.2f} km/h   distance {f['distance_m']:.2f} m",
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
    args = ap.parse_args(argv)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    DriveApp(root, preset=args.preset)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
