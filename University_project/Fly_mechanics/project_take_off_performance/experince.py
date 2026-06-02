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

def choose_aircraft(name):
    global aircraft_dict
    if name == "Cessna 172":
        aircraft_dict = data6
    elif name == "Airbus A320":
        aircraft_dict = data5
    elif name == "Belugar Xl":
        aircraft_dict = data8
    elif name == "Cirrus SR22":
        aircraft_dict = data3
    elif name == "Executive business jet":
        aircraft_dict = data4
    elif name == "Lockheed Martin C-130J Super Hercules":
        aircraft_dict = data7
    elif name == "Dassault Rafale":
        aircraft_dict = data8

def choose_model(n):
    global model_num
    model_num = n

def launch():
    global aircraft_dict, model_num

    if aircraft_dict is None or model_num is None:
        print("Choisis un avion et un modèle")
        return

    if model_num == 1:
        
        param = get_model_params("Model1", aircraft_dict)
        plane = AircraftTakeoff1(param)
        plane.summary()

    elif model_num == 2:
        
        param = get_model_params("Model2", aircraft_dict)
        plane = AircraftTakeoff2(param)
        plane.summary()

    elif model_num == 3:
        
        param = get_model_params("Model3", aircraft_dict)
        plane = AircraftTakeoff3(param)
        plane.summary()

    elif model_num == 4:
        
        param = get_model_params("Model4", aircraft_dict)
        plane = AircraftTakeoff4(param)
        plane.summary()

    elif model_num == 5:
       
        param = get_model_params("Model5", aircraft_dict)
        plane = AircraftTakeoff5(param)
        plane.summary()

    elif model_num == 6:
        
        param = get_model_params("Model6", aircraft_dict)
        plane = AircraftTakeoff6(param)
        plane.summary()

def comande():
    fen = Tk()
    fen.title("Simulation takeoff")

    Label(fen, text="Welcome in our simulation of aircraft takeoff distance").grid(row=0, column=0, columnspan=3)
    Label(fen, text="Choose aircraft").grid(row=1, column=0, columnspan=3)

    Button(fen, text="Cessna 172", command=lambda: choose_aircraft("Cessna 172")).grid(row=2, column=0)
    Button(fen, text="Airbus A320", command=lambda: choose_aircraft("Airbus A320")).grid(row=3, column=0)
    Button(fen, text="Belugar Xl", command=lambda: choose_aircraft("Belugar Xl")).grid(row=4, column=0)
    Button(fen, text="Lockheed Martin C-130J Super Hercules", command=lambda: choose_aircraft("Lockheed Martin C-130J Super Hercules")).grid(row=5, column=0)
    Button(fen, text="Dassault Rafale", command=lambda: choose_aircraft("Dassault Rafale")).grid(row=6, column=0)
    Button(fen, text="Executive business jet", command=lambda: choose_aircraft("Executive business jet")).grid(row=7, column=0)
    Button(fen, text="Cirrus SR22", command=lambda: choose_aircraft("Cirrus SR22")).grid(row=8, column=0)

    Label(fen, text="Choose model").grid(row=1, column=1, columnspan=3)

    Button(fen, text="Model1", command=lambda: choose_model(1)).grid(row=2, column=1)
    Button(fen, text="Model2", command=lambda: choose_model(2)).grid(row=2, column=2)
    Button(fen, text="Model3", command=lambda: choose_model(3)).grid(row=2, column=3)
    Button(fen, text="Model4", command=lambda: choose_model(4)).grid(row=3, column=1)
    Button(fen, text="Model5", command=lambda: choose_model(5)).grid(row=3, column=2)
    Button(fen, text="Model6", command=lambda: choose_model(6)).grid(row=3, column=3)

    Button(fen, text="Lancer", command=launch).grid(row=10, column=0, columnspan=3)

    fen.mainloop()

comande()