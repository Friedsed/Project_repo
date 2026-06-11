from tkinter import *
from data import *
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
from exp2 import *
import pandas as pd
from Model1 import *
from Model2 import *
from Model3 import *
from Model4 import *
from Model5 import *

aircraft_dict = None
model_num = None
displayed_widgets = []          # ← track the widgets displayed by launch()



def choose_aircraft(root, name, data):
    """
    Function that makes the choice and decides which data to load.
    """
    image_choice(root, name)
    global aircraft_dict

    if name == "Cessna 172":
        aircraft_dict = data10

    elif name == "Airbus A320":
        aircraft_dict = data9

    elif name == "Beluga XL":
        aircraft_dict = data5

    elif name == "Cirrus SR22":
        aircraft_dict = data3

    elif name == "Executive business jet":
        aircraft_dict = data4

    elif name == "Lockheed Martin C-130J":
        aircraft_dict = data7

    elif name == "Dassault Rafale":
        aircraft_dict = data8

    elif name == "data1":
        aircraft_dict = data1

    elif name == "data2":
        aircraft_dict = data2

    elif name == "New Aircraft":
       aircraft_dict = data

    print("Aircraft selected:", name)


def choose_model(n):
    global model_num
    model_num = n


def launch(root, r):
    """
    Launch allows the computation once the aircraft and the model are chosen.
    If we click on Launch, the results will be displayed below the Launch button.
    """
    global aircraft_dict, model_num

    if aircraft_dict is None or model_num is None:
        print("Choisis un avion et un modèle")
        return

    def display(root, r, title, book, Unit, labels, results):
        """
        Function used to display the output for each choice.

        The Tkinter window is divided into rows and columns,
        which is why each Label or Button is assigned a specific position.
        """
        global displayed_widgets

        # Labels are used to display text, and grid() is used to position them.
        w = Label(root, text=title, font=("Arial", 20))
        w.grid(row=r + 1, column=0, columnspan=3)
        displayed_widgets.append(w)

        w = Label(root, text=book, font=("Arial", 20))
        w.grid(row=r + 2, column=0, columnspan=3)
        displayed_widgets.append(w)

        w = Label(root, text=Unit, font=("Arial", 20), bg = "#4169E1")
        w.grid(row=r + 3, column=0, columnspan=3)
        displayed_widgets.append(w)

        w = Label(root, text="", font=("Arial", 20))
        w.grid(row=r + 4, column=0, columnspan=3)
        displayed_widgets.append(w)
        Label(root, text="            ").grid(row = r + 5 , column=1, columnspan=3)


        start_row = r + 6

        for i, key in enumerate(labels):
            w = Label(root, text=key, width=30, anchor="w", font=("Arial", 20))
            w.grid(row=start_row + i, column=0, padx=5, pady=2)
            displayed_widgets.append(w)

            w = Label(root, text=results.get(key, "N/A"), width=30, anchor="w", font=("Arial", 20))
            w.grid(row=start_row + i, column=4, padx=5, pady=2)
            displayed_widgets.append(w)

    # Output obtained when choosing an aircraft and a model.
    # Example: Cessna 172 + Model1

    if model_num == 1:
        params = get_model_params("Model1", aircraft_dict)
        plane = AircraftTakeoff1(params)
        results = plane.set_result()
        plane.plot_distance()
        plane.plot_acceleration()
        plane.plot_forces()

        labels = [
        
            "Runaway distance Sa is",
            "The Rotation distance Sr is",
            "The Climb distance Sc is",
            "The total distance S for takeoff is ",
            "Lift-off speed Vlo is"
        ]

        display(
            root, r,
            "Ground run estimation using the numerical integration method",
            "B1: Book used: Mechanics of Flight by Warren",
            "SI, << meter ==> length >>   << Newton ==> forces >>   << Watts ==> Power >>",
            labels,
            results
        )

    elif model_num == 2:
        params = get_model_params("Model2", aircraft_dict)
        plane = AircraftTakeoff2(params)
        results = plane.set_result()

        labels = [
            "Ground run distance Sa is",
            "Lift-off speed Vlo is"
        ]

        display(
            root, r,
            "General ground run estimation using average acceleration",
            "B2: General Aviation Aircraft Design - Snorri Gudmundsson",
            "SI, << meter ==> length >>  << Newton ==> forces >>   << Watts ==> Power >>",
            labels,
            results
        )

    elif model_num == 4:
        params = get_model_params("Model4", aircraft_dict)
        plane = AircraftTakeoff4(params)
        results = plane.set_result()
        plane.plot_speed()
        plane.plot_acceleration()
        plane.plot_forces()

        labels = [
            "Ground run distance Sa is",
            "The rotation speed in m/s is :" ,
            "The climb distance Sc is",
            "The total distance S for takeoff is ",
            "Lift-off speed Vlo is"
        ]

        display(
            root, r,
            "Ground run estimation using the numerical integration method",
            "B2: General Aviation Aircraft Design - Snorri Gudmundsson",
            "SI, << meter ==> length >>  << Newton ==> forces >>  << Watts ==> Power >>",
            labels,
            results
        )

    elif model_num == 5:
        params = get_model_params("Model5", aircraft_dict)
        plane = AircraftTakeoff5(params)
        results = plane.set_result()
        plane.plot_distance()
        plane.plot_acceleration()
        plane.plot_forces()

        labels = [
            "Ground run distance Sa is",
            "The rotation distance Sr, also called the transition distance, is",
            "The climb distance Sc is"
        ]

        display(
            root, r,
            "Ground run estimation using the numerical integration method",
            "B2: General Aviation Aircraft Design - Snorri Gudmundsson",
            "SI, << meter ==> length >>  << Newton ==> forces >>   << Watts ==> Power >>",
            labels,
            results
        )


def In_between_Function(root):
    """
    Use for new Aicraft , because it allows me to save the data once the customer 
    clic on Valider then get the result in the code systeme for computing with model.
    """
    plane = Aircraft_new(root)
    root.wait_window(plane.root)  # attend la fermeture
    data15 = plane.data
    print(data15)
    choose_aircraft(root, "New Aircraft", data15)

    
def Aircraft_choice(root, n):
    """
    Function used to choose the aircraft and update the picture.
    Works with choose_aircraft().
    """
    global param
    parameters = [
        "Cessna 172", "Beluga XL", "Dassault Rafale", "Airbus A320",
        "Lockheed Martin C-130J", "Executive business jet",
        "Cirrus SR22", "data1", "data2", "New Aircraft"
    ]
    for i in range(n, n + len(parameters)):
        param = parameters[i - n]
        if param == "New Aircraft":
            row = i
            Button( root, text=param, width=20, anchor="w",
                command=lambda p=param: In_between_Function(root)
            ).grid(row=row, column=0, padx=5, pady=2)
        else :
            row = i
            Button( root, text=param, width=20, anchor="w",
                command=lambda p=param: choose_aircraft(root, p, data=None)
            ).grid(row=row, column=0, padx=5, pady=2)


def image_choice(root, name):
    """
    Function used to display the image corresponding to the selected aircraft.

    This function depends on choose_aircraft(root, p).
    """
    dico = {
        "Cessna 172": r"Images/Cessna172.png",
        "Beluga XL": r"Images/begura.png",
        "Dassault Rafale": r"Images/Dassault_rafale.png",
        "Airbus A320": r"Images/A320.png",
        "Lockheed Martin C-130J": r"Images/C-130J_super_hercules.png",
        "Executive business jet": r"Images/Executive_jet.png",
        "Cirrus SR22": r"Images/Cirrus.png",
        "data1": r"Images/Executive_jet.png",
        "data2": r"Images/Executive_jet.png",
        "New Aircraft": r"Images/Executive_jet.png",
    }
    can1 = Canvas(root, width=600, height=400, bg="white")
    can1.grid(row=5, column=2, rowspan=3, padx=10, pady=5)
    photo = PhotoImage(file=dico[name])
    can1.create_image(300, 200, image=photo)
    can1.image = photo


def Model_choice(root, n):
    """
    Function that handles the model selection.
    This function is executed once a model is chosen.
    """
    parameter = [
        "Model1", "Model2", "Model3", "Model4", "Model5"
    ]
    Label(root, text="Choose model").grid(row=1, column=3, columnspan=3)
    for i in range(n, n + len(parameter)):
        param = parameter[i - n]
        row = i
        Button(root, text=param, width=15, anchor="w", bg = "yellow", command=lambda k=i - n + 1: choose_model(k)
        ).grid(row=row, column=3, padx=5, pady=2)


def reset():
    """    Button( root, text=param, width=20, anchor="w",
                command=lambda p=param: Aircraft_new().get_values
            ).gr
    After launching several aircraft/model combinations and displaying results,
    this function clears the window and resets the selections.
    """
    global displayed_widgets, aircraft_dict, model_num
    for w in displayed_widgets:
        w.destroy()
    displayed_widgets = []
    aircraft_dict = None
    model_num = None
    print("Reset effectué — tu peux relancer un nouveau calcul")


def comande():
    """
    Main function that calls all the other functions to run the program.
    """
    fen = Tk()
    fen.title("Takeoff Simulation")
    Label(fen, text="Welcome to our aircraft takeoff distance simulation",font=("Arial", 20), bg="#FF1493"
    ).grid(row=0, column=0, columnspan=3)
    Aircraft_choice(fen, 2)
    Label(fen, text="").grid(row=1, column=1, columnspan=3)
    Model_choice(fen, 2)
    Button(fen, text="Launch",  bg="green", command=lambda: launch(fen, 26)
    ).grid(row=14, column=0, columnspan=3)
    Button(fen, text="Reset", bg="red", fg="white", command=reset
    ).grid(row=14, column=1, columnspan=3)
    Label(fen, text="            ").grid(row=15, column=1, columnspan=3)

    fen.mainloop()


comande()

