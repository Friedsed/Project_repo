
"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 14 juin 2026

General code of the project to compile the code  ( Don't execute the code used for comparing the
 results obtained from the different models, which is in  
 :  Comparaison.py.  )
"""



from tkinter import *
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

import pandas as pd

from Conversion import *
from data import *
from Model_1 import *
from Model_2 import *
#from Model_3 import *
from Model_4 import *
from Model_5 import *
from Function import *





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
