import tkinter as tk
from tkinter import ttk


class AircraftApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Aircraft Performance Calculator")
        self.root.geometry("900x650")

        # --- Main Layout Grid Configuration ---
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Style configurations
        self.style = ttk.Style()
        self.style.configure("TLabelframe", padding=10)
        self.style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))

        # Track dynamically controlled widgets for Section 1 and 2
        self.section_widgets = []

        self.create_widgets()
        self.setup_logic()

    def create_widgets(self):
        
        # =====================================================================
        # TOP LEFT: SECTION 1 (Engine & Name)
        # =====================================================================
        self.frame_sec1 = ttk.LabelFrame(self.root, text="① Custom Specifications")
        self.frame_sec1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Engine sub-frame
        engine_frame = ttk.LabelFrame(self.frame_sec1, text="Engine")
        engine_frame.pack(fill="x", padx=5, pady=5)

        # --------------------------------------------------------------------
        self.engine_var = tk.StringVar(value="None")

        engine_buttons = [
            ("Jet", "r_jet"),
            ("Piston", "r_piston"),
        ]

        for text, attr_name in engine_buttons:
            rb = ttk.Radiobutton( engine_frame, text=text, variable=self.engine_var,
                value=text
            )
            rb.pack(anchor="w", padx=5, pady=2)

            setattr(self, attr_name, rb)
            self.section_widgets.append(rb)

        # --------------------------------------------------------------------
        # Conservation de la variable existante self.r_Unit
        unit_frame = ttk.LabelFrame(self.frame_sec1, text="Unit")
        unit_frame.pack(fill="x", padx=5, pady=5)

        self.unit_var = tk.StringVar(value="SI")

        self.r_Unit = ttk.Radiobutton( unit_frame, text="SI", variable=self.unit_var,
            value="SI"
        )
        self.r_Unit.pack(anchor="w", padx=5, pady=2)

        self.r_US = ttk.Radiobutton( unit_frame, text="US", variable=self.unit_var,
            value="US"
        )
        self.r_US.pack(anchor="w", padx=5, pady=2)

        self.section_widgets.extend([self.r_Unit, self.r_US])


        # Name entry sub-frame
        name_frame = ttk.LabelFrame(self.frame_sec1, text="Name")
        name_frame.pack(fill="x", padx=5, pady=5)

        self.entry_name = ttk.Entry(name_frame)
        self.entry_name.pack(fill="x", padx=5, pady=5)
        self.section_widgets.append(self.entry_name)





        # =====================================================================
        # BOTTOM LEFT: SECTION 2 (Aerodynamic & Geometry)
        # =====================================================================
        self.frame_sec2 = ttk.LabelFrame(self.root, text="② Parameters")
        self.frame_sec2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_sec2.columnconfigure(0, weight=1)
        self.frame_sec2.columnconfigure(1, weight=1)
        self.frame_sec2.columnconfigure(2, weight=1)
        self.frame_sec2.columnconfigure(3, weight=1)

        """
        # Aerodynamic column
        aero_frame = ttk.LabelFrame(self.frame_sec2, text="Aerodynamic")
        aero_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        ttk.Label(aero_frame, text="Cd:").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.entry_cd = ttk.Entry(aero_frame, width=8)
        self.entry_cd.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(aero_frame, text="Cl:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.entry_cl = ttk.Entry(aero_frame, width=8)
        self.entry_cl.grid(row=1, column=1, padx=5, pady=2)
        self.section_widgets.extend([self.entry_cd, self.entry_cl])
            """

        # Aerodynamic column
        aero_frame = ttk.LabelFrame(self.frame_sec2, text="Aerodynamic")
        aero_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        aero_labels = ["Cd", "Cdo", "Cl", "Clmax" , "g", "mu", "rho", ]
        self.aero_entries = {}                                                                               # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
        for i, label in enumerate(aero_labels):
            ttk.Label(aero_frame, text=f"{label}:").grid(
                row=i, column=0, sticky="w", padx=5
            )
            entry_aero = ttk.Entry(aero_frame, width=8)
            entry_aero.grid(row=i, column=1, padx=5, pady=2)
            self.aero_entries[label] = entry_aero                                                                # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
            self.section_widgets.append(entry_aero)


        # Geometry column
        geom_frame = ttk.LabelFrame(self.frame_sec2, text="Geometry")
        geom_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        geom_labels = ["W" "Sw", "bw", "hw", "Ra", ]
        self.geom_entries = {}                                                                              # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
        for i, label in enumerate(geom_labels):
            ttk.Label(geom_frame, text=f"{label}:").grid(
                row=i, column=1, sticky="w", padx=5
            )
            entry = ttk.Entry(geom_frame, width=8)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.geom_entries[label] = entry                                                                 # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
            self.section_widgets.append(entry)

        #Propulsion column
        prop_frame = ttk.LabelFrame(self.frame_sec2, text="Propulsives_forces")
        prop_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        prop_labels = ["T" "P", "T0", "T2", "A1", "A2", "A3", "A4", ]
        self.prop_entries = {}                                                                              # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
        for i, label in enumerate(prop_labels):
            ttk.Label(prop_frame, text=f"{label}:").grid(
                row=i, column=0, sticky="w", padx=5
            )
            entry_prop = ttk.Entry(prop_frame, width=8)
            entry_prop.grid(row=i, column=2, padx=5, pady=2)
            self.prop_entries[label] = entry                                                                 # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
            self.section_widgets.append(entry_prop)

        #Constante column
        const_frame = ttk.LabelFrame(self.frame_sec2, text="Constantes")
        const_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        const_labels = ["e" , "g", "mu", "rho", "Vi", "Vhw", "tr", "hoc", "n", ]
        self.const_entries = {}                                                                              # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
        for i, label in enumerate(const_labels):
            ttk.Label(const_frame, text=f"{label}:").grid(
                row=i, column=0, sticky="w", padx=5
            )
            entry_const = ttk.Entry(const_frame, width=8)
            entry_const.grid(row=i, column=2, padx=5, pady=2)
            self.const_entries[label] = entry                                                                 # REALLY IMPORTANT AN OUTPUT FOR NEW AICRAFT
            self.section_widgets.append(entry_const)

        self.param = self.aero_entries | self.geom_entries |self.prop_entries | self.const_entries
        self.param["alpha"] = 0
        self.param["alpha_0"] = 0

        # =====================================================================
        # CENTER: IMAGE DISPLAY AREA
        # =====================================================================
        self.frame_image = ttk.LabelFrame(self.root, text="Visualizer")
        self.frame_image.grid(
            row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew"
        )

        # Canvas acts as the placeholder placeholder for drawings or image objects
        self.canvas = tk.Canvas(self.frame_image, bg="#f0f0f0", bd=2, relief="ridge")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas_text = self.canvas.create_text(
            150,
            200,
            text="[ Image Area ]\nSelect options & Launch",
            justify="center",
            fill="gray",
        )

        # =====================================================================
        # TOP RIGHT: SELECTION & CONTROLS (Aircraft, Model, Actions)
        # =====================================================================
        self.frame_control = ttk.LabelFrame(self.root, text="Selection Panel")
        self.frame_control.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.frame_control.columnconfigure(0, weight=1)
        self.frame_control.columnconfigure(1, weight=1)

        # Aircraft Selection List
        ttk.Label(self.frame_control, text="Aircraft:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.list_aircraft = tk.Listbox(self.frame_control, height=5, exportselection=False)
        for item in ["Cessna", "Beluga", "C-130", "New aircraft"]:
            self.list_aircraft.insert(tk.END, item)
        self.list_aircraft.grid(
            row=1, column=0, padx=5, pady=5, sticky="nsew", columnspan=2
        )

        # Model Selection List
        ttk.Label(self.frame_control, text="Model:").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        self.list_model = tk.Listbox(self.frame_control, height=4, exportselection=False)
        for item in ["M 1", "M 2", "M 3","M 4"]:
            self.list_model.insert(tk.END, item)
        self.list_model.grid(
            row=3, column=0, padx=5, pady=5, sticky="nsew", columnspan=2
        )

        # Action Buttons
        self.btn_launch = tk.Button(
            self.frame_control,
            text="Launch",
            fg="green",
            font=("Arial", 10, "bold"),
            command=self.on_launch,
        )
        self.btn_launch.grid(row=4, column=0, padx=5, pady=10, sticky="ew")

        self.btn_reset = tk.Button(
            self.frame_control,
            text="Reset",
            fg="red",
            font=("Arial", 10, "bold"),
            command=self.on_reset,
        )
        self.btn_reset.grid(row=4, column=1, padx=5, pady=10, sticky="ew")

        # =====================================================================
        # BOTTOM RIGHT: OUTPUT DISPLAY
        # =====================================================================
        self.frame_output = ttk.LabelFrame(self.root, text="Output")
        self.frame_output.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

        self.lbl_result1 = ttk.Label(
            self.frame_output, text="command / label (result 1) : --", font=("Arial", 10)
        )
        self.lbl_result1.pack(anchor="w", padx=10, pady=10)

        self.lbl_label1 = ttk.Label(
            self.frame_output, text="command (label 1) : --", font=("Arial", 10)
        )
        self.lbl_label1.pack(anchor="w", padx=10, pady=10)

    # =====================================================================
    # LOGIC & INTERACTION HANDLING
    # =====================================================================
    def setup_logic(self):
        # Bind Selection change event to check if "New aircraft" is picked
        self.list_aircraft.bind("<<ListboxSelect>>", self.toggle_input_fields)
        # Initialize custom sections to disabled state
        self.set_sections_enabled(False)

    def toggle_input_fields(self, event=None):
        try:
            selected_index = self.list_aircraft.curselection()[0]
            selected_aircraft = self.list_aircraft.get(selected_index)

            if selected_aircraft == "New aircraft":
                self.set_sections_enabled(True)
            else:
                self.set_sections_enabled(False)
        except IndexError:
            pass  # Nothing selected

    def set_sections_enabled(self, enable):
        state = "normal" if enable else "disabled"
        for widget in self.section_widgets:
            widget.config(state=state)

    def on_launch(self):
        try:
            ac_index = self.list_aircraft.curselection()[0]
            aircraft = self.list_aircraft.get(ac_index)
        except IndexError:
            aircraft = "None Selected"

        try:
            model_index = self.list_model.curselection()[0]
            model = self.list_model.get(model_index)
        except IndexError:
            model = "None Selected"

        # Update text/image canvas area dynamically based on configuration
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            20, 20, 260, 360, outline="blue", fill="#e6f2ff", width=2
        )
        self.canvas.create_text(
            140,
            180,
            text=f"Rendering Map / Spec Matrix\nfor\n{aircraft}\n({model})",
            justify="center",
            font=("Arial", 11, "bold"),
        )

        # Process Outputs
        if aircraft == "New aircraft":
            custom_name = self.entry_name.get() or "Unnamed Custom"
            engine = self.engine_var.get()
            self.lbl_result1.config(
                text=f"command / label (result 1) : {custom_name} [{engine}]"
            )
            self.lbl_label1.config(
                text=f"command (label 1) : CD={self.entry_cd.get() or 0} | SW={self.geom_entries['Sw'].get() or 0}"
            )
        else:
            self.lbl_result1.config(
                text=f"command / label (result 1) : {aircraft} Simulation Successful."
            )
            self.lbl_label1.config(
                text=f"command (label 1) : Loaded Preset standard {model}"
            )

    def on_reset(self):
        # Clear Selections
        self.list_aircraft.selection_clear(0, tk.END)
        self.list_model.selection_clear(0, tk.END)

        # Clear text Entry fields
        self.entry_name.delete(0, tk.END)
        self.entry_cd.delete(0, tk.END)
        self.entry_cl.delete(0, tk.END)
        for entry in self.geom_entries.values():
            entry.delete(0, tk.END)

        # Reset radio selection
        self.engine_var.set("None")

        # Clear outputs and Canvas layout
        self.lbl_result1.config(text="command / label (result 1) : --")
        self.lbl_label1.config(text="command (label 1) : --")
        self.canvas.delete("all")
        self.canvas.create_text(
            150,
            200,
            text="[ Image Area ]\nSelect options & Launch",
            justify="center",
            fill="gray",
        )

        # Turn sections back to disabled
        self.set_sections_enabled(False)


if __name__ == "__main__":
    root = tk.Tk()
    app = AircraftApp(root)
    root.mainloop()