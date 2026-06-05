from tkinter import *
from data import *
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from conversion import *
from experince import *
import pandas as pd 
from Model1 import *
from Model2 import *
from Model3 import *
from Model4 import *
from Model5 import *

aircraft_dict = None
model_num = None

def choose_aircraft(root,name):

    image_choice(root, name)
    global aircraft_dict
    if name == "Cessna 172":
        aircraft_dict = data6

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

    print("Aircraft selected :", name)
    

def choose_model(n):
    global model_num
    model_num = n

"""
def launch(root, r):
    global aircraft_dict, model_num

    if aircraft_dict is None or model_num is None:
        print("Choisis un avion et un modèle")
        return

    if model_num == 1:

        params = get_model_params("Model1", aircraft_dict)
        plane = AircraftTakeoff1(params)
        results = plane.set_result()   
        list=["Cd", "K0", "K1" , "K2", "Kr", "Vlo", "Kw" , "Kt", "Runaway distance is" 
                    , "The Rotation distance is", "The Climb distance is"]

        Label( root, text="Ground run estimation using numerical Integration method"
                ).grid(row=r + 1, column=0, columnspan=3)
        Label( root, text="B1: Book used Mechanics of Flight by Warren"
                ).grid(row=r + 2, column=0, columnspan=3)
        Label(root, text="").grid(row=r + 3, column=0, columnspan=3)
        for i in range( r, r + len(list) ):
            row = i
            Label( root,text=list[i-r], width=20,  anchor="w" ).grid(row=row, column=0, padx=5, pady=2)
            Label( root,text=results[list[i-r]], width=20,  anchor="w" ).grid(row=row, column=4, padx=5, pady=2)

    if model_num == 2:

        params = get_model_params("Model2", aircraft_dict)
        plane = AircraftTakeoff2(params)
        results = plane.set_result()   
        list=[ "Runaway distance is :",   " The lift off speed is" ]

        Label( root, text=" General ground run estimation using Average Acceleration for tricycle propeller aicraft ONLY ; page 797 " 
                    ).grid(row=r + 1, column=0, columnspan=3)
        Label( root, text=" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON "
                ).grid(row=r + 2, column=0, columnspan=3)
        Label(root, text="").grid(row=r + 3, column=0, columnspan=3)
        for i in range( r, r + len(list) ):
            row = i
            Label( root,text=list[i-r], width=20,  anchor="w" ).grid(row=row, column=0, padx=5, pady=2)
            Label( root,text=results[list[i-r]], width=20,  anchor="w" ).grid(row=row, column=4, padx=5, pady=2)

    if model_num == 4:

        params = get_model_params("Model4", aircraft_dict)
        plane = AircraftTakeoff4(params)
        results = plane.set_result()   
        list=[ "Runaway distance is",  " The lift off speed", "The drag during the lift off is", 
            "The thrust during the lift off is", "The rotating distance also called the transition is"
            , "The climbing distance is"  ]

        Label( root, text=" Ground run estimation using numerical Integration method"
                    ).grid(row=r + 1, column=0, columnspan=3)
        Label( root, text=" B2: Book used General Aviation Aicraft  Design ; Applied Methods and Procedures ; SNORRI GUDMUNDSSON "
                ).grid(row=r + 2, column=0, columnspan=3)
        Label(root, text="").grid(row=r + 3, column=0, columnspan=3)
        for i in range( r, r + len(list) ):
            row = i
            Label( root,text=list[i-r], width=20,  anchor="w" ).grid(row=row, column=0, padx=5, pady=2)
            Label( root,text=results[list[i-r]], width=20,  anchor="w" ).grid(row=row, column=4, padx=5, pady=2)


  """

def launch(root, r):
    global aircraft_dict, model_num

    if aircraft_dict is None or model_num is None:
        print("Choisis un avion et un modèle")
        return

    def display(root, r, title, book, labels, results):
        Label(root, text=title).grid(row=r + 1, column=0, columnspan=3)
        Label(root, text=book).grid(row=r + 2, column=0, columnspan=3)
        Label(root, text="").grid(row=r + 3, column=0, columnspan=3)

        start_row = r + 4

        for i, key in enumerate(labels):
            Label(root, text=key, width=30, anchor="w").grid(
                row=start_row + i, column=0, padx=5, pady=2
            )

            Label(root, text=results.get(key, "N/A"), width=30, anchor="w").grid(
                row=start_row + i, column=4, padx=5, pady=2
            )

    # ---------------- MODEL 1 ----------------
    if model_num == 1:
        params = get_model_params("Model1", aircraft_dict)
        plane = AircraftTakeoff1(params)
        results = plane.set_result()

        labels = [
            "Cd", "K0", "K1", "K2", "Kr", "Vlo", "Kw", "Kt",
            "Runaway distance is",
            "The Rotation distance is",
            "The Climb distance is"
        ]

        display(
            root, r,
            "Ground run estimation using numerical Integration method",
            "B1: Book used Mechanics of Flight by Warren",
            labels,
            results
        )

    # ---------------- MODEL 2 ----------------
    elif model_num == 2:
        params = get_model_params("Model2", aircraft_dict)
        plane = AircraftTakeoff2(params)
        results = plane.set_result()

        labels = [
            "Runaway distance is",
            "The lift off speed"
        ]

        display(
            root, r,
            "General ground run estimation using Average Acceleration (tricycle propeller aircraft only)",
            "B2: General Aviation Aircraft Design - Snorri Gudmundsson",
            labels,
            results
        )

    # ---------------- MODEL 4 ----------------
    elif model_num == 4:
        params = get_model_params("Model4", aircraft_dict)
        plane = AircraftTakeoff4(params)
        results = plane.set_result()

        labels = [
            "Runaway distance is",
            "The lift off speed",
            "The drag during the lift off is",
            "The thrust during the lift off is",
            "The rotating distance also called the transition is",
            "The climbing distance is"
        ]

        display(
            root, r,
            "Ground run estimation using numerical Integration method",
            "B2: General Aviation Aircraft Design - Snorri Gudmundsson",
            labels,
            results
        )   

 




#Function to make the choice of the aicraft and allow to chnage the picture 
def Aircraft_choice(root, n):
    global param
    parameters = [
        "Cessna 172", "Beluga XL", "Dassault Rafale", "Airbus A320",
        "Lockheed Martin C-130J", "Executive business jet",
        "Cirrus SR22", "data1", "data2"
    ]
    for i in range(n, n + len(parameters)):
        param = parameters[i - n]
        row = i
        Button( root, text=param,  width=20,  anchor="w", command=lambda p=param: choose_aircraft(root, p)
            ).grid(row=row, column=0, padx=5, pady=2)


# Funtion to make the choice of the image depending on the aicraft the user choose
def image_choice(root,name):

    dico = {
        "Cessna 172": r"Images/Cessna172.png",    "Beluga XL": r"Images/begura.png",
        "Dassault Rafale": r"Images/Dassault_rafale.png",
        "Airbus A320": r"Images/A320.png", "Lockheed Martin C-130J": r"Images/C-130J_super_hercules.png",
        "Executive business jet": "",  "Cirrus SR22": "", "data1": "",  "data2": ""
    }
    can1 = Canvas(root, width=500, height=300, bg="white")
    can1.grid(row=5, column=2, rowspan=3, padx=10, pady=5)
    photo = PhotoImage(file=dico[name])
    can1.create_image(250, 150, image=photo)
    can1.image = photo

# Function that handle the choose of the model depending one what is choosed 
def Model_choice(root, n):

    parameter = [
        "Model1", "Model2", "Model3", "Model4", "Model5"
    ]
    Label(root, text="Choose model").grid(row=1, column=3, columnspan=3)
    for i in range(n, n + len(parameter)):
        param = parameter[i - n]
        row = i
        Button( root, text=param, width=15, anchor="w", command=lambda k=i - n + 1: choose_model(k)
        ).grid(row=row, column=3, padx=5, pady=2)















def start():
    launch(fen, 14)


def comande():
    fen = Tk()
    fen.title("Simulation takeoff")
    Label(fen, text="Welcome in our simulation of aircraft takeoff distance").grid(row=0, column=0, columnspan=3)
    Aircraft_choice(fen, 2)
    Label(fen, text="Choose model").grid(row=1, column=1, columnspan=3)
    Model_choice(fen,2)
    Button( fen,  text="Lancer", command=lambda: launch(fen, 26)).grid(row=14, column=0, columnspan=3)
    fen.mainloop()

comande()

    