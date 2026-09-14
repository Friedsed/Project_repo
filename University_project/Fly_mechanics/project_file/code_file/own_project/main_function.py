
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



def comande ():
    fen = Tk()
    fen.title(" Learning software")
    Label(fen, text="Welcome to Friedly WOLI software" , font=("Arial", 20 ), bg = "#FF1493" ).grid(row=0, column=0, columnspan = 3 )
    Model_choice(fen, 2)

    fen.mainloop()
















