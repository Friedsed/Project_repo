
import numpy as np
import matplotlib.pyplot as plt

import csv


fichier = csv.DictReader(open("jeux.csv"))

list_jeux=list(fichier)

print(list_jeux [0])
print(list_jeux [0].keys())
print(" The type of the element list_jeux is :",type(list_jeux))
print(" The type of the element list_jeux[0] is :",type(list_jeux[0]))
print("The type of the keys in list_jeux[0] is :",type(list_jeux[0]['Year']))

yearm=[]
for i in range (len(list_jeux)):
    yearm.append(list_jeux[i]['Year'])
    #year.append(float(list_jeux[i]['Year']))

for i  in range (len(yearm)):
    if yearm[i]=='' or yearm[i]=='N/A':
        yearm[i]=yearm[i-1]
    else:
        yearm[i]=int(yearm[i])

print(max(yearm))
print(min(yearm))
