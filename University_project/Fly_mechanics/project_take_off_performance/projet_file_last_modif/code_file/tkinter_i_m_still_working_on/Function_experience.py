"""
Author: Friedly WOLI
Date: 27 April 2026

Student in third year in Mechanics and Energetics
University of Toulouse

Modified: 14 juin 2026

Code Développer tous seul en me bassant sur le code proposé par l'IA pour le programme : Comparaision.py 



"""

import tkinter as tk


def Aircraft_choice(root):
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
    for i in range(0, len(parameters)):
        param = parameters[i ]
        if param == "New Aircraft":
            row = i
            tk.Button( root, text=param, width=20, anchor="w"
               
            ).pack( padx=5, pady=2)
        else :
            row = i
            tk.Button( root, text=param, width=20, anchor="w"
                
            ).pack( padx=5, pady=2)



def Model_choice(root):
    """
    Function that handles the model selection.
    This function is executed once a model is chosen.
    """
    parameter = [
        "Model1", "Model2", "Model4", "Model5"
    ]
    tk.Label(root, text="Choose model").pack()
    for i in range(0, len(parameter)):
        param = parameter[i ]
        row = i
        tk.Button(root, text=param, width=15, anchor="w", bg = "yellow").pack( padx=5, pady=2)





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