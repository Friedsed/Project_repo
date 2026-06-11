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



i=3


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









