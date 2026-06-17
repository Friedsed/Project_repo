from tkinter import *


class Aircraft_new:
    def __init__(self, root_parent):

        self.root = Toplevel(root_parent)
        self.root.title("Aircraft Parameters")

        self.entries = {}
        self.data = None

        parameters = [
            "Unit", "name", "engine",
            "W", "Sw", "bw", "hw", "Ra",
            "Clmax", "Cl", "Cd", "Cdo","Cdol", "e",
            "g", "mu", "rho",
            "P", "T", "T0", "T1", "T2", "efficiency",
            "Vi", "Vhw", "tr", "hoc", 
            "alpha", "alpha_0",
            "A1", "A2", "A3", "A4",
            "n"
        ]

        # Création des labels et entrées
        for i, param in enumerate(parameters):
            row = i
            Label(self.root, text=param, width=15, anchor="w").grid(row=row, column=0, padx=5, pady=2)
            entry =  Entry(self.root, width=20)
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.entries[param] = entry

        #Button(  root, text="Get Parameters", command=self.get_values  ).grid( row=len(parameters),
         #           column=10, columnspan=2, pady=10 )
        
        Button(self.root, text="Valider", command=self.validate).grid(row=len(parameters), column=1)


    def validate(self):
        self.data = {}
        for name, entry in self.entries.items():
            value = entry.get()
            try:
                self.data[name] = float(value)
            except ValueError:
                self.data[name] = value
        self.root.destroy()
        







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
             Button(root, text=param, width=15, anchor ="w", command=lambda: choose_aircraft(param) 
                        ).grid(row=row, column=0, padx=5, pady=2)

root =  Tk()
app = Aircraft_new(root)
root.mainloop()
        
"""
 
