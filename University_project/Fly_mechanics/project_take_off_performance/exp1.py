"""
Flight Performance Calculator - Dynamic Tkinter App
Replicates the UI from the screenshot with full interactivity.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import math
import json

# ── Colour palette ──────────────────────────────────────────────────────────
BG_DARK    = "#0f1923"
BG_PANEL   = "#1a2535"
BG_INPUT   = "#111c2a"
BG_SELECT  = "#223044"
ACCENT     = "#3a7fc1"
ACCENT2    = "#4db8ff"
TEXT_WHITE = "#e8f0fe"
TEXT_GREY  = "#8aa0bb"
TEXT_GOLD  = "#f0c040"
BORDER     = "#2a3f58"
ROW_ALT    = "#192030"

# ── Default aircraft database ────────────────────────────────────────────────
DEFAULT_AIRCRAFT = {
    "Cessna 172": {
        "mtow": 1111, "wing_area": 16.2, "thrust": 5.4,
        "v1": 55, "vr": 60, "v2": 70,
        "cl": 1.63, "cd": 0.037,
        "altitude": 0, "isa_dev": 0,
        "emoji": "✈️"
    },
    "Piper PA-28": {
        "mtow": 1089, "wing_area": 15.8, "thrust": 4.9,
        "v1": 52, "vr": 57, "v2": 65,
        "cl": 1.58, "cd": 0.041,
        "altitude": 0, "isa_dev": 0,
        "emoji": "🛩️"
    },
    "Airbus A320": {
        "mtow": 73500, "wing_area": 122.6, "thrust": 240.0,
        "v1": 145, "vr": 155, "v2": 165,
        "cl": 2.20, "cd": 0.025,
        "altitude": 0, "isa_dev": 0,
        "emoji": "🛫"
    },
    "Boeing 737": {
        "mtow": 79016, "wing_area": 125.4, "thrust": 260.0,
        "v1": 148, "vr": 158, "v2": 170,
        "cl": 2.10, "cd": 0.026,
        "altitude": 0, "isa_dev": 0,
        "emoji": "🛬"
    },
    "Embraer E190": {
        "mtow": 51800, "wing_area": 92.5, "thrust": 145.0,
        "v1": 135, "vr": 142, "v2": 152,
        "cl": 1.95, "cd": 0.028,
        "altitude": 0, "isa_dev": 0,
        "emoji": "✈️"
    },
}

METHODS = [
    "Method 1: Basic Equations",
    "Method 2: Takeoff Performance Data",
    "Method 3: Manufacturer Charts",
    "Method 4: Aerodynamic Simulation",
]

# ── Physics helpers ──────────────────────────────────────────────────────────
RHO_SL = 1.225          # kg/m³  sea-level density
G      = 9.80665        # m/s²

def _rho(alt_m: float, isa_dev: float = 0.0) -> float:
    """ISA density at altitude with optional temperature deviation."""
    T0 = 288.15 + isa_dev
    T  = T0 - 0.0065 * alt_m
    T  = max(T, 216.65)
    p_ratio = (T / T0) ** 5.2561
    rho = RHO_SL * p_ratio * (T0 / T)
    return rho

def _knots_to_ms(v_kt: float) -> float:
    return v_kt * 0.514444

def calc_method1(ac: dict) -> dict:
    """Basic kinematic equations."""
    rho  = _rho(ac["altitude"], ac["isa_dev"])
    mtow = ac["mtow"]
    S    = ac["wing_area"]
    T    = ac["thrust"] * 1000        # kN → N
    cl   = ac["cl"]
    cd   = ac["cd"]
    vr   = _knots_to_ms(ac["vr"])
    v2   = _knots_to_ms(ac["v2"])

    W = mtow * G
    mu = 0.02                         # rolling friction

    # Ground roll  s = v² / (2 * a)
    D_avg = 0.5 * rho * (vr/2)**2 * S * cd
    L_avg = 0.5 * rho * (vr/2)**2 * S * cl
    F_net = T - mu*(W - L_avg) - D_avg
    F_net = max(F_net, 1)
    a_gnd = F_net / (mtow)
    s_gnd = max(1, vr**2 / (2 * a_gnd))

    # Rotation (15 m airborne transition)
    s_rot = 15.0 * (vr / 60)

    # Climb  (obstacle clearance to 35 ft = 10.67 m)
    gamma = math.asin(min(0.99, (T - W*0.05) / W))
    gamma = max(gamma, 0.02)
    h_ob  = 10.67
    s_clb = h_ob / math.tan(gamma) if gamma > 0.001 else 3000

    kn = cl
    kw = cd
    twr = T / W

    return {
        "ground_roll": round(s_gnd),
        "rotation":    round(s_rot),
        "climb":       round(s_clb),
        "kn": round(kn, 3),
        "kw": round(kw, 4),
        "twr": round(twr, 3),
    }

def calc_method2(ac: dict) -> dict:
    """Takeoff performance data (empirical multipliers)."""
    base = calc_method1(ac)
    return {
        "ground_roll": round(base["ground_roll"] * 1.08),
        "rotation":    round(base["rotation"]    * 1.05),
        "climb":       round(base["climb"]       * 1.12),
        "kn":  base["kn"],  "kw": base["kw"],  "twr": base["twr"],
    }

def calc_method3(ac: dict) -> dict:
    """Manufacturer chart (conservative book values)."""
    base = calc_method1(ac)
    return {
        "ground_roll": round(base["ground_roll"] * 1.15),
        "rotation":    round(base["rotation"]    * 1.10),
        "climb":       round(base["climb"]       * 1.20),
        "kn":  base["kn"],  "kw": base["kw"],  "twr": base["twr"],
    }

def calc_method4(ac: dict) -> dict:
    """Aerodynamic simulation (CFD-like correction)."""
    rho   = _rho(ac["altitude"], ac["isa_dev"])
    ratio = rho / RHO_SL
    base  = calc_method1(ac)
    return {
        "ground_roll": round(base["ground_roll"] / ratio),
        "rotation":    round(base["rotation"]    / math.sqrt(ratio)),
        "climb":       round(base["climb"]       / ratio),
        "kn":  base["kn"],  "kw": base["kw"],  "twr": base["twr"],
    }

CALCULATORS = [calc_method1, calc_method2, calc_method3, calc_method4]

# ── Main Application ─────────────────────────────────────────────────────────
class FlightCalculatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flight Performance Calculator")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(860, 700)

        # State
        self.aircraft_db   = {k: dict(v) for k, v in DEFAULT_AIRCRAFT.items()}
        self.selected_ac   = tk.StringVar(value="Cessna 172")
        self.selected_meth = tk.IntVar(value=0)
        self.show_detail   = tk.BooleanVar(value=False)

        # Input vars (populated from aircraft data)
        self.var_mtow    = tk.DoubleVar(value=1111)
        self.var_wing    = tk.DoubleVar(value=16.2)
        self.var_thrust  = tk.DoubleVar(value=5.4)
        self.var_v1      = tk.DoubleVar(value=55)
        self.var_vr      = tk.DoubleVar(value=60)
        self.var_v2      = tk.DoubleVar(value=70)
        self.var_cl      = tk.DoubleVar(value=1.63)
        self.var_cd      = tk.DoubleVar(value=0.037)
        self.var_alt     = tk.DoubleVar(value=0)
        self.var_isa     = tk.DoubleVar(value=0)

        # Result labels (created later)
        self.lbl_ground  = None
        self.lbl_rot     = None
        self.lbl_climb   = None
        self.lbl_kn      = None
        self.lbl_kw      = None
        self.lbl_twr     = None

        self._build_ui()
        self._load_aircraft_to_inputs()
        self._calculate()

        # Trace changes in input vars → auto recalculate
        for v in (self.var_mtow, self.var_wing, self.var_thrust,
                  self.var_v1, self.var_vr, self.var_v2,
                  self.var_cl, self.var_cd, self.var_alt, self.var_isa):
            v.trace_add("write", lambda *_: self._calculate())

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = tk.Frame(self, bg=BG_DARK, padx=18, pady=18)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # Title
        tk.Label(outer, text="✈  FLIGHT PERFORMANCE CALCULATOR",
                 font=("Courier New", 18, "bold"), bg=BG_DARK, fg=TEXT_WHITE,
                 pady=10).grid(row=0, column=0, sticky="w")

        # Subtitle banner
        banner = tk.Frame(outer, bg=BG_PANEL, bd=0, relief="flat",
                          highlightthickness=1, highlightbackground=BORDER)
        banner.grid(row=1, column=0, sticky="ew", pady=(0,12))
        tk.Label(banner,
                 text="We have ≠ (multiple) types of aircraft — choose one.\n"
                      "Choose the method (mth) you want to calculate the distance.",
                 font=("Courier New", 10), bg=BG_PANEL, fg=TEXT_GREY,
                 justify="left", padx=14, pady=8).pack(anchor="w")

        # Middle row: aircraft list | image | method list
        mid = tk.Frame(outer, bg=BG_DARK)
        mid.grid(row=2, column=0, sticky="ew", pady=(0,12))
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)
        mid.columnconfigure(2, weight=3)

        self._build_aircraft_panel(mid)
        self._build_aircraft_display(mid)
        self._build_method_panel(mid)

        # Results table
        self._build_results_table(outer)

        # Detail toggle
        detail_row = tk.Frame(outer, bg=BG_DARK)
        detail_row.grid(row=4, column=0, sticky="ew", pady=(4,0))
        tk.Label(detail_row, text="If you want more detailed calculation outputs →",
                 font=("Courier New", 9), bg=BG_DARK, fg=TEXT_GREY).pack(side="left")
        tk.Button(detail_row, text="[ CLICK HERE ]",
                  font=("Courier New", 9, "bold"), bg=BG_DARK, fg=ACCENT2,
                  bd=0, cursor="hand2",
                  command=self._toggle_detail).pack(side="left", padx=4)

        # Output section
        self._build_output_section(outer)

        # Detailed input panel
        self._build_input_panel(outer)

    def _build_aircraft_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0,6))

        tk.Label(frame, text="AIRCRAFT SELECTION",
                 font=("Courier New", 9, "bold"), bg=BG_PANEL, fg=TEXT_GREY,
                 padx=10, pady=6).pack(anchor="w")

        self.ac_listbox = tk.Listbox(
            frame, bg=BG_INPUT, fg=TEXT_WHITE,
            selectbackground=ACCENT, selectforeground=TEXT_WHITE,
            font=("Courier New", 10), bd=0, highlightthickness=0,
            activestyle="none", height=5
        )
        self.ac_listbox.pack(fill="both", expand=True, padx=8, pady=(0,6))
        self._refresh_ac_listbox()
        self.ac_listbox.bind("<<ListboxSelect>>", self._on_ac_select)

        btn_add = tk.Button(frame, text="+ ADD NEW AIRCRAFT",
                            font=("Courier New", 9, "bold"),
                            bg=BG_SELECT, fg=ACCENT2, bd=0,
                            activebackground=ACCENT, activeforeground=TEXT_WHITE,
                            padx=10, pady=5, cursor="hand2",
                            command=self._add_aircraft)
        btn_add.pack(fill="x", padx=8, pady=(0,8))

    def _build_aircraft_display(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        frame.grid(row=0, column=1, sticky="nsew", padx=3)

        self.ac_emoji_lbl = tk.Label(frame, text="✈️",
                                     font=("Segoe UI Emoji", 48),
                                     bg=BG_PANEL, fg=TEXT_WHITE)
        self.ac_emoji_lbl.pack(expand=True, pady=(14,4))

        tk.Label(frame, text="AIRCRAFT IMAGE",
                 font=("Courier New", 8, "bold"), bg=BG_PANEL,
                 fg=TEXT_GREY).pack()

        self.ac_name_lbl = tk.Label(frame, text="Cessna 172",
                                    font=("Courier New", 9), bg=BG_PANEL,
                                    fg=TEXT_GREY, wraplength=120, justify="center")
        self.ac_name_lbl.pack(padx=6, pady=(2,14))

    def _build_method_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        frame.grid(row=0, column=2, sticky="nsew", padx=(6,0))

        tk.Label(frame, text="CALCULATION METHODS",
                 font=("Courier New", 9, "bold"), bg=BG_PANEL, fg=TEXT_GREY,
                 padx=10, pady=6).pack(anchor="w")

        for i, m in enumerate(METHODS):
            rb = tk.Radiobutton(frame, text=m, variable=self.selected_meth,
                                value=i, command=self._calculate,
                                font=("Courier New", 9), bg=BG_PANEL,
                                fg=TEXT_WHITE, selectcolor=ACCENT,
                                activebackground=BG_PANEL,
                                activeforeground=ACCENT2,
                                wraplength=180, justify="left",
                                padx=8, pady=3)
            rb.pack(anchor="w", fill="x")

        tk.Button(frame, text="[ Edit Parameters ]",
                  font=("Courier New", 9), bg=BG_SELECT, fg=ACCENT2,
                  bd=0, padx=8, pady=4, cursor="hand2",
                  command=self._edit_parameters).pack(
                      fill="x", padx=8, pady=8, side="bottom")

    def _build_results_table(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        frame.grid(row=3, column=0, sticky="ew", pady=(0,4))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Header
        for col, (text, anchor) in enumerate([
            ("Performance Metric", "w"), ("Calculated Distance (m)", "e")
        ]):
            tk.Label(frame, text=text, font=("Courier New", 10, "bold"),
                     bg=BG_SELECT, fg=TEXT_WHITE,
                     padx=14, pady=6, anchor=anchor
                     ).grid(row=0, column=col, sticky="ew")

        rows = [
            ("• Ground roll distance", "lbl_ground"),
            ("• Rotation distance",    "lbl_rot"),
            ("• Climb distance",       "lbl_climb"),
        ]
        for r, (metric, attr) in enumerate(rows, start=1):
            bg = BG_PANEL if r % 2 else ROW_ALT
            tk.Label(frame, text=metric, font=("Courier New", 10),
                     bg=bg, fg=TEXT_WHITE, padx=14, pady=5, anchor="w"
                     ).grid(row=r, column=0, sticky="ew")
            lbl = tk.Label(frame, text="—", font=("Courier New", 10, "bold"),
                           bg=bg, fg=TEXT_GOLD, padx=14, pady=5, anchor="e")
            lbl.grid(row=r, column=1, sticky="ew")
            setattr(self, attr, lbl)

    def _build_output_section(self, parent):
        self.out_frame = tk.Frame(parent, bg=BG_DARK)
        self.out_frame.grid(row=5, column=0, sticky="ew", pady=(6,0))

        tk.Label(self.out_frame, text="[ OUTPUT ]",
                 font=("Courier New", 12, "bold"), bg=BG_DARK,
                 fg=TEXT_WHITE).grid(row=0, column=0, sticky="w", pady=(4,2))

        items = [
            ("Kₙ (Coefficient of Lift):", "lbl_kn"),
            ("Kw (Coefficient of Drag):", "lbl_kw"),
            ("Thrust-to-Weight Ratio: ", "lbl_twr"),
        ]
        for r, (label, attr) in enumerate(items, start=1):
            tk.Label(self.out_frame, text=label,
                     font=("Courier New", 10), bg=BG_DARK,
                     fg=TEXT_GREY).grid(row=r, column=0, sticky="w", padx=14)
            lbl = tk.Label(self.out_frame, text="—",
                           font=("Courier New", 10, "bold"),
                           bg=BG_DARK, fg=TEXT_GOLD)
            lbl.grid(row=r, column=1, sticky="w", padx=6)
            setattr(self, attr, lbl)

        # Detail panel (hidden by default)
        self.detail_frame = tk.Frame(parent, bg=BG_PANEL, bd=0,
                                     highlightthickness=1, highlightbackground=BORDER)
        self.detail_frame.grid(row=6, column=0, sticky="ew", pady=(6,0))
        self.detail_frame.grid_remove()

        tk.Label(self.detail_frame, text="DETAILED OUTPUTS",
                 font=("Courier New", 10, "bold"), bg=BG_PANEL, fg=ACCENT2,
                 padx=12, pady=6).pack(anchor="w")

        self.detail_text = tk.Text(self.detail_frame, height=6,
                                   font=("Courier New", 9), bg=BG_INPUT,
                                   fg=TEXT_WHITE, bd=0, padx=10, pady=8,
                                   state="disabled")
        self.detail_text.pack(fill="x", padx=8, pady=(0,8))

    def _build_input_panel(self, parent):
        outer = tk.Frame(parent, bg=BG_PANEL, bd=0,
                         highlightthickness=1, highlightbackground=BORDER)
        outer.grid(row=7, column=0, sticky="ew", pady=(10,0))
        outer.columnconfigure((0,1,2,3), weight=1)

        tk.Label(outer,
                 text="DETAILED AIRCRAFT DATA INPUT PANEL  "
                      "(Edit values — calculations update live)",
                 font=("Courier New", 8, "bold"), bg=BG_PANEL, fg=TEXT_GREY,
                 padx=12, pady=6).grid(row=0, column=0, columnspan=4, sticky="w")

        fields = [
            # (label, var, unit, row, col)
            ("Max Takeoff Weight (kg)", self.var_mtow, "kg",  1, 0),
            ("Wing Area (m²)",          self.var_wing, "m²",  1, 1),
            ("Thrust Rating (kN)",      self.var_thrust,"kN", 1, 2),
            ("V1 (kt)",                 self.var_v1,   "kt",  2, 0),
            ("VR (kt)",                 self.var_vr,   "kt",  2, 1),
            ("V2 (kt)",                 self.var_v2,   "kt",  2, 2),
            ("CL (Lift coeff.)",        self.var_cl,   "",    3, 0),
            ("CD (Drag coeff.)",        self.var_cd,   "",    3, 1),
            ("Altitude (m)",            self.var_alt,  "m",   3, 2),
            ("ISA deviation (°C)",      self.var_isa,  "°C",  4, 0),
        ]

        for (label, var, unit, row, col) in fields:
            cell = tk.Frame(outer, bg=BG_PANEL)
            cell.grid(row=row, column=col, sticky="ew", padx=8, pady=4)
            tk.Label(cell, text=label, font=("Courier New", 8),
                     bg=BG_PANEL, fg=TEXT_GREY).pack(anchor="w")
            row_f = tk.Frame(cell, bg=BG_INPUT,
                             highlightthickness=1, highlightbackground=BORDER)
            row_f.pack(fill="x")
            ent = tk.Entry(row_f, textvariable=var, bg=BG_INPUT, fg=TEXT_WHITE,
                           font=("Courier New", 10), bd=0, insertbackground=ACCENT2,
                           width=10)
            ent.pack(side="left", padx=6, pady=4, fill="x", expand=True)
            if unit:
                tk.Label(row_f, text=unit, font=("Courier New", 9),
                         bg=BG_INPUT, fg=TEXT_GREY, padx=4).pack(side="right")

        # Save/Reset buttons
        btn_row = tk.Frame(outer, bg=BG_PANEL)
        btn_row.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=8)

        tk.Button(btn_row, text="💾  Save to Aircraft",
                  font=("Courier New", 9, "bold"), bg=ACCENT, fg=TEXT_WHITE,
                  bd=0, padx=12, pady=5, cursor="hand2",
                  command=self._save_inputs_to_aircraft).pack(side="left", padx=(0,8))

        tk.Button(btn_row, text="↺  Reset to Default",
                  font=("Courier New", 9), bg=BG_SELECT, fg=TEXT_GREY,
                  bd=0, padx=12, pady=5, cursor="hand2",
                  command=self._reset_to_default).pack(side="left")

        tk.Button(btn_row, text="🗑  Delete Aircraft",
                  font=("Courier New", 9), bg="#3d1a1a", fg="#ff6b6b",
                  bd=0, padx=12, pady=5, cursor="hand2",
                  command=self._delete_aircraft).pack(side="right")

    # ── Data helpers ─────────────────────────────────────────────────────────
    def _refresh_ac_listbox(self):
        self.ac_listbox.delete(0, "end")
        for name in self.aircraft_db:
            em = self.aircraft_db[name].get("emoji", "✈️")
            self.ac_listbox.insert("end", f"  {em}  {name}")
        # Reselect current
        for i, name in enumerate(self.aircraft_db):
            if name == self.selected_ac.get():
                self.ac_listbox.selection_set(i)
                self.ac_listbox.see(i)
                break

    def _on_ac_select(self, _=None):
        sel = self.ac_listbox.curselection()
        if not sel:
            return
        name = list(self.aircraft_db.keys())[sel[0]]
        self.selected_ac.set(name)
        self._load_aircraft_to_inputs()
        self._update_display()
        self._calculate()

    def _load_aircraft_to_inputs(self):
        ac = self.aircraft_db.get(self.selected_ac.get(), {})
        if not ac:
            return
        self.var_mtow.set(ac["mtow"])
        self.var_wing.set(ac["wing_area"])
        self.var_thrust.set(ac["thrust"])
        self.var_v1.set(ac["v1"])
        self.var_vr.set(ac["vr"])
        self.var_v2.set(ac["v2"])
        self.var_cl.set(ac["cl"])
        self.var_cd.set(ac["cd"])
        self.var_alt.set(ac.get("altitude", 0))
        self.var_isa.set(ac.get("isa_dev", 0))

    def _update_display(self):
        name = self.selected_ac.get()
        ac   = self.aircraft_db.get(name, {})
        em   = ac.get("emoji", "✈️")
        self.ac_emoji_lbl.config(text=em)
        self.ac_name_lbl.config(text=name)

    def _get_current_ac_dict(self) -> dict:
        return {
            "mtow":      self.var_mtow.get(),
            "wing_area": self.var_wing.get(),
            "thrust":    self.var_thrust.get(),
            "v1":        self.var_v1.get(),
            "vr":        self.var_vr.get(),
            "v2":        self.var_v2.get(),
            "cl":        self.var_cl.get(),
            "cd":        self.var_cd.get(),
            "altitude":  self.var_alt.get(),
            "isa_dev":   self.var_isa.get(),
            "emoji":     self.aircraft_db.get(self.selected_ac.get(), {}).get("emoji", "✈️"),
        }

    # ── Calculation ──────────────────────────────────────────────────────────
    def _calculate(self, *_):
        try:
            ac   = self._get_current_ac_dict()
            meth = self.selected_meth.get()
            res  = CALCULATORS[meth](ac)
        except Exception as e:
            print(f"Calc error: {e}")
            return

        # Update result labels
        if self.lbl_ground:
            self.lbl_ground.config(text=f"{res['ground_roll']:,} m")
        if self.lbl_rot:
            self.lbl_rot.config(text=f"{res['rotation']:,} m")
        if self.lbl_climb:
            self.lbl_climb.config(text=f"{res['climb']:,} m")
        if self.lbl_kn:
            self.lbl_kn.config(text=f"{res['kn']:.4f}")
        if self.lbl_kw:
            self.lbl_kw.config(text=f"{res['kw']:.5f}")
        if self.lbl_twr:
            self.lbl_twr.config(text=f"{res['twr']:.4f}")

        self._update_detail_text(ac, res, meth)

    def _update_detail_text(self, ac: dict, res: dict, meth: int):
        rho = _rho(ac["altitude"], ac["isa_dev"])
        lines = [
            f"Method     : {METHODS[meth]}",
            f"Air density: {rho:.4f} kg/m³",
            f"MTOW       : {ac['mtow']:.1f} kg   Wing: {ac['wing_area']:.2f} m²",
            f"Thrust     : {ac['thrust']:.2f} kN  V1/VR/V2: {ac['v1']:.0f}/{ac['vr']:.0f}/{ac['v2']:.0f} kt",
            f"Ground roll: {res['ground_roll']:,} m",
            f"Rotation   : {res['rotation']:,} m",
            f"Climb dist : {res['climb']:,} m",
            f"Total TO   : {res['ground_roll']+res['rotation']+res['climb']:,} m",
            f"CL={res['kn']:.4f}  CD={res['kw']:.5f}  T/W={res['twr']:.4f}",
        ]
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", "\n".join(lines))
        self.detail_text.config(state="disabled")

    # ── Button actions ───────────────────────────────────────────────────────
    def _toggle_detail(self):
        self.show_detail.set(not self.show_detail.get())
        if self.show_detail.get():
            self.detail_frame.grid()
        else:
            self.detail_frame.grid_remove()

    def _save_inputs_to_aircraft(self):
        name = self.selected_ac.get()
        if name not in self.aircraft_db:
            return
        self.aircraft_db[name].update({
            "mtow":      self.var_mtow.get(),
            "wing_area": self.var_wing.get(),
            "thrust":    self.var_thrust.get(),
            "v1":        self.var_v1.get(),
            "vr":        self.var_vr.get(),
            "v2":        self.var_v2.get(),
            "cl":        self.var_cl.get(),
            "cd":        self.var_cd.get(),
            "altitude":  self.var_alt.get(),
            "isa_dev":   self.var_isa.get(),
        })
        messagebox.showinfo("Saved", f"Parameters saved for {name}.", parent=self)

    def _reset_to_default(self):
        name = self.selected_ac.get()
        if name in DEFAULT_AIRCRAFT:
            self.aircraft_db[name] = dict(DEFAULT_AIRCRAFT[name])
            self._load_aircraft_to_inputs()
            messagebox.showinfo("Reset", f"Parameters reset for {name}.", parent=self)

    def _add_aircraft(self):
        dlg = AddAircraftDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            name, data = dlg.result
            self.aircraft_db[name] = data
            self._refresh_ac_listbox()
            self.selected_ac.set(name)
            self._load_aircraft_to_inputs()
            self._update_display()
            self._calculate()

    def _delete_aircraft(self):
        name = self.selected_ac.get()
        if len(self.aircraft_db) <= 1:
            messagebox.showwarning("Cannot Delete", "At least one aircraft is required.", parent=self)
            return
        if messagebox.askyesno("Delete", f"Delete '{name}'?", parent=self):
            del self.aircraft_db[name]
            self.selected_ac.set(next(iter(self.aircraft_db)))
            self._refresh_ac_listbox()
            self._load_aircraft_to_inputs()
            self._update_display()
            self._calculate()

    def _edit_parameters(self):
        """Open a parameter editor popup."""
        EditParametersDialog(self)


# ── Add Aircraft Dialog ───────────────────────────────────────────────────────
class AddAircraftDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Add New Aircraft")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.grab_set()
        self.result = None
        self._build()

    def _build(self):
        tk.Label(self, text="NEW AIRCRAFT", font=("Courier New", 13, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE, pady=12).grid(row=0, column=0,
                 columnspan=2, padx=20)

        fields = [
            ("Aircraft Name",   "name",    "Cessna 152"),
            ("MTOW (kg)",       "mtow",    "757"),
            ("Wing Area (m²)",  "wing",    "14.6"),
            ("Thrust (kN)",     "thrust",  "3.7"),
            ("V1 (kt)",         "v1",      "50"),
            ("VR (kt)",         "vr",      "55"),
            ("V2 (kt)",         "v2",      "62"),
            ("CL",              "cl",      "1.55"),
            ("CD",              "cd",      "0.038"),
            ("Emoji",           "emoji",   "🛩️"),
        ]
        self.vars = {}
        for r, (label, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=label, font=("Courier New", 9),
                     bg=BG_DARK, fg=TEXT_GREY, anchor="e", width=16
                     ).grid(row=r, column=0, padx=(20,6), pady=3, sticky="e")
            var = tk.StringVar(value=default)
            self.vars[key] = var
            tk.Entry(self, textvariable=var, bg=BG_INPUT, fg=TEXT_WHITE,
                     font=("Courier New", 10), bd=0,
                     insertbackground=ACCENT2, width=18
                     ).grid(row=r, column=1, padx=(0,20), pady=3)

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.grid(row=len(fields)+2, column=0, columnspan=2, pady=14)
        tk.Button(btn_row, text="✔  Add Aircraft",
                  font=("Courier New", 10, "bold"), bg=ACCENT, fg=TEXT_WHITE,
                  bd=0, padx=12, pady=5, cursor="hand2",
                  command=self._submit).pack(side="left", padx=8)
        tk.Button(btn_row, text="✘  Cancel",
                  font=("Courier New", 10), bg=BG_SELECT, fg=TEXT_GREY,
                  bd=0, padx=12, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")

    def _submit(self):
        try:
            name = self.vars["name"].get().strip()
            if not name:
                raise ValueError("Name is required.")
            data = {
                "mtow":      float(self.vars["mtow"].get()),
                "wing_area": float(self.vars["wing"].get()),
                "thrust":    float(self.vars["thrust"].get()),
                "v1":        float(self.vars["v1"].get()),
                "vr":        float(self.vars["vr"].get()),
                "v2":        float(self.vars["v2"].get()),
                "cl":        float(self.vars["cl"].get()),
                "cd":        float(self.vars["cd"].get()),
                "altitude":  0,
                "isa_dev":   0,
                "emoji":     self.vars["emoji"].get().strip() or "✈️",
            }
            self.result = (name, data)
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e), parent=self)


# ── Edit Parameters Dialog ────────────────────────────────────────────────────
class EditParametersDialog(tk.Toplevel):
    def __init__(self, parent: FlightCalculatorApp):
        super().__init__(parent)
        self.app = parent
        self.title("Edit Method Parameters")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.grab_set()
        self._build()

    def _build(self):
        tk.Label(self, text="METHOD CORRECTION FACTORS",
                 font=("Courier New", 12, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE, pady=12
                 ).grid(row=0, column=0, columnspan=3, padx=20)

        headers = ["Method", "Ground Roll ×", "Rotation ×", "Climb ×"]
        for c, h in enumerate(headers):
            tk.Label(self, text=h, font=("Courier New", 9, "bold"),
                     bg=BG_SELECT, fg=TEXT_WHITE, padx=10, pady=4, width=14
                     ).grid(row=1, column=c, padx=2, pady=(0,4))

        info = [
            ("Basic Equations",    "1.00", "1.00", "1.00"),
            ("Takeoff Perf. Data", "1.08", "1.05", "1.12"),
            ("Manufacturer Charts","1.15", "1.10", "1.20"),
            ("Aero Simulation",    "÷ρ/ρ₀","÷√(ρ/ρ₀)","÷ρ/ρ₀"),
        ]
        for r, (m, g, ro, cl) in enumerate(info, start=2):
            for c, val in enumerate([m, g, ro, cl]):
                tk.Label(self, text=val, font=("Courier New", 9),
                         bg=BG_PANEL if r % 2 == 0 else ROW_ALT,
                         fg=TEXT_WHITE, padx=10, pady=5, width=14, anchor="center"
                         ).grid(row=r, column=c, padx=2, pady=1)

        tk.Label(self,
                 text="Method 4 scales results by air density ratio (ISA + altitude).",
                 font=("Courier New", 8), bg=BG_DARK, fg=TEXT_GREY,
                 pady=8).grid(row=6, column=0, columnspan=4, padx=20)

        tk.Button(self, text="Close", font=("Courier New", 10),
                  bg=ACCENT, fg=TEXT_WHITE, bd=0, padx=16, pady=5,
                  cursor="hand2", command=self.destroy
                  ).grid(row=7, column=0, columnspan=4, pady=12)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FlightCalculatorApp()
    app.mainloop()