import tkinter as tk
from tkinter import ttk

class Aircraft_new:

    def __init__(self, root):

        self.root = root
        self.root.title("Aircraft Parameters")

        self.entries = {}

        parameters = [
            "Unit", "Name", "Engine",
            "Weight", "Wing Area", "bw", "hw", "Ra",
            "CLmax", "CL", "CD", "CD0", "e",
            "g", "mu", "rho",
            "P", "T", "T0", "T1", "T2",
            "Efficiency",
            "A1", "A2", "A3", "A4",
            "n"
        ]

        # Création des labels et entrées
        for i, param in enumerate(parameters):

            row = i

            tk.Label(root, text=param, width=15, anchor="w").grid(row=row, column=0, padx=5, pady=2)

            entry = tk.Entry(root, width=20)
            entry.grid(row=row, column=1, padx=5, pady=2)

            self.entries[param] = entry

        ttk.Button(  root, text="Get Parameters", command=self.get_values  ).grid( row=len(parameters),
                    column=0, columnspan=2, pady=10 )

        self.result_label= tk.Label(root, text="")
        self.result_label.grid( row=len(parameters)+1, column=0, columnspan=2 )

    def get_values(self):
        aircraft_data = {}
        for name, entry in self.entries.items():

            value = entry.get()
            try:
                aircraft_data[name] = float(value)
            except ValueError:
                aircraft_data[name] = value

        print("\nAircraft Parameters")
        print("-------------------")

        for key, value in aircraft_data.items():
            print(f"{key:12s} : {value}")

        self.result_label.config(
            text="Parameters successfully loaded!"
        )

        return aircraft_data






"""
class Aircraft_choice:

    def __init__(self, root):

        self.root = root
        self.root.title("Aircraft Parameters")

        self.entries = {}

        parameters = [
            "Cessna 172", "Beluga XL", "Dassault Rafale", "Airbus A320", 
            "Lockheed Martin C-130J Super Hercules", "Executive business jet", "Cirrus SR22"
            "Executive business jet", "data1", "data2"
        ]

        # Création des labels et entrées
        for i, param in enumerate(parameters):

            row = i
            tk.Button(root, text=param, width=15, anchor ="w", command=lambda: choose_aircraft(param) 
                        ).grid(row=row, column=0, padx=5, pady=2)

        
"""
 

root = tk.Tk()
app = Aircraft_choice(root)
root.mainloop()