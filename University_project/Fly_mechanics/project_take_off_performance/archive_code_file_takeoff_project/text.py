import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv







Lecture = csv.DictReader(open("sim_data.csv"))
Lp = list(Lecture)

avions = []

data = {
    "Sa": {"Model1": [], "Model2": [], "Model3": [], "Model4": []},
    "Sr": {"Model1": [], "Model2": [], "Model3": [], "Model4": []},
    "Sc": {"Model1": [], "Model2": [], "Model3": [], "Model4": []},
    "S":  {"Model1": [], "Model2": [], "Model3": [], "Model4": []},
    "Vlo":{"Model1": [], "Model2": [], "Model3": [], "Model4": []},
}

for row in Lp:
    avions.append(row["Avions"])

    data["Sa"]["Model1"].append(float(row["Sa1"]))
    data["Sa"]["Model2"].append(float(row["Sa2"]))
    data["Sa"]["Model3"].append(float(row["Sa3"]))
    data["Sa"]["Model4"].append(float(row["Sa4"]))

    data["Sr"]["Model1"].append(float(row["Sr1"]))
    data["Sr"]["Model2"].append(float(row["Sr2"]))
    data["Sr"]["Model3"].append(float(row["Sr3"]))
    data["Sr"]["Model4"].append(float(row["Sr4"]))

    data["Sc"]["Model1"].append(float(row["Sc1"]))
    data["Sc"]["Model2"].append(float(row["Sc2"]))
    data["Sc"]["Model3"].append(float(row["Sc3"]))
    data["Sc"]["Model4"].append(float(row["Sc4"]))

    data["S"]["Model1"].append(float(row["S1"]))
    data["S"]["Model2"].append(float(row["S2"]))
    data["S"]["Model3"].append(float(row["S3"]))
    data["S"]["Model4"].append(float(row["S4"]))

    data["Vlo"]["Model1"].append(float(row["Vlo1"]))
    data["Vlo"]["Model2"].append(float(row["Vlo2"]))
    data["Vlo"]["Model3"].append(float(row["Vlo3"]))
    data["Vlo"]["Model4"].append(float(row["Vlo4"]))



i=7


if i==1:
    idx = avions.index("Executive Jet ")
elif i==2:
    idx = avions.index("Cessna 172")
elif i==3:
    idx = avions.index("Belugar XL")
elif i==4 :
    idx = avions.index("Airbus A30")
elif i==5 :
    idx = avions.index("C – 130")
elif i==6 :
    idx = avions.index("Data1")
elif i==7 :
    idx = avions.index("Data2")





variables = ["Sa", "Sr", "Sc", "S", "Vlo"]
models = ["Model1", "Model2", "Model3", "Model4"]

for var in variables:
    plt.figure(figsize=(7, 4))
    values = [data[var][m][idx] for m in models]
    plt.bar(models, values)
    plt.title(f"{var} pour {avions[idx]}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(var)
    plt.grid(axis="y")
    plt.show()














































# ============================================================/////////////////////////////////////////////////////////////////////////////////////////
# Management of the running data                                                                                                                             |
# ============================================================////////////////////////////////////////////////////////////////////////////////////////

"""
print ( Lp[4])     =====    {'Avions': 'C – 130', '': '', 'Sa1': '664.46', 'Sr1': '186.09',
 'S1': '913.8', 'Vlo1': '63.25', 'Sa2': '659.31', 'Sr2': '', 'Sc2': '', 'S2': '', 'Vlo2': '63.2', 
 'Sa3': '676.39', 'Sr3': '63.25', 'Sc3': '223.28', 'S3': '962.92', 'Vlo3': '63.81', 'Sa4': '652.91',
  'Sr4': '63.25', 'Sc4': '186.09', 'S4': '', 'Vlo4': ''}


avions.append( dict['Avions'])
print (avions )  =====   ['', 'Cessna 172', 'Belugar XL', 'Dassault Rafale', 'Airbus A30', 'C – 130', 'Executive Jet ', 'Ciruss ', 'Data1', 'Data2']

"""


"""

Lecture = csv.DictReader( open("sim_data.csv"))
Lp =list(Lecture) 

avions = []
Sa1, Sr1, Sc1, S1, Vlo1 = [], [], [], [], []
Sa2, Sr2, Sc2, S2, Vlo2 = [], [], [], [], []
Sa3, Sr3, Sc3, S3, Vlo3 = [], [], [], [], []
Sa4, Sr4, Sc4, S4, Vlo4 = [], [], [], [], []







for i in range (0, len(Lp) ) :
    dict = Lp[i]   # des element comme la ligne 12 ou 14 du code
    avions.append( dict['Avions'])

    Sa1.append(dict['Sa1'])
    Sa2.append(dict['Sa2'])
    Sa3.append(dict['Sa3'])
    Sa4.append(dict['Sa4'])

    Sr1.append(dict['Sr1'])
    Sr2.append(dict['Sr2'])
    Sr3.append(dict['Sr3'])
    Sr4.append(dict['Sr4'])

    Sc1.append(dict['Sc1'])
    Sc2.append(dict['Sc2'])
    Sc3.append(dict['Sc3'])
    Sc4.append(dict['Sc4'])

    S1.append(dict['S1'])
    S2.append(dict['S2'])
    S3.append(dict['S3'])
    S4.append(dict['S4'])

    Vlo1.append(dict['Vlo1'])
    Vlo2.append(dict['Vlo2'])
    Vlo3.append(dict['Vlo3'])
    Vlo4.append(dict['Vlo4'])


M1, M2, M3, M4 = {}, {}, {}, {}
Var = { "Sa", "Sr", "Sc", "S", "Vlo"}

for var in Var :
    if var == "Sa":
        M1[var] = Sa1
        M2[var] = Sa2
        M3[var] = Sa3
        M4[var] = Sa4
    if var == "Sr":
        M1[var] = Sr1
        M2[var] = Sr2
        M3[var] = Sr3
        M4[var] = Sr4
    if var == "Sc":
        M1[var] = Sc1
        M2[var] = Sc2
        M3[var] = Sc3
        M4[var] = Sc4
    if var == "S":
        M1[var] = S1
        M2[var] = S2
        M3[var] = S3
        M4[var] = S4
    if var == "Vlo":
        M1[var] = Vlo1
        M2[var] = Vlo2
        M3[var] = Vlo3
        M4[var] = Vlo4


    
    

    
modeles = {"Model1": M1 , "Model2": M2 , "Model3": M3 , "Model4": M4  }
    
    


"""
































