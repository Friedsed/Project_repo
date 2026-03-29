import numpy as np
#import sympy as sp



#######################################################DECLARATION DE VARIABLES#######################################################

n = 3 # taille de la matrice
k1=10
#matrice = [[0 for _ in range(n)] for _ in range(n)]
liste=[[0 for _ in range(n)] for _ in range(k1) ]
listeJ=[[0 for _ in range(n)] for _ in range(k1) ]
#b = [0 for _ in range(n)]

matrice = [
    [3, 1, -1],
    [1, 5, 2],
    [2, -1, -6]
]
b=[2,17,-18]
liste[0]=[0,0,0]
listeJ[0]=[0,0,0]

#######################################################Codage #######################################################


for k in range(n-1):
    cmpt=0
    k=k+1
    for i in range(n):
        alpha=0
        for j in range(n):
            if j<i:
                alpha=matrice[i][j]* liste[k][j] + alpha
                cmpt+=1
            elif j>i:
                alpha=matrice[i][j]* liste[k-1][j] + alpha
                cmpt+=1
            else:
                alpha=0+alpha
                cmpt+=1
        liste[k][i]= (b[i]-alpha)/matrice[i][i]



print("La solution est :{liste}, et le nbre d'itérations est :{cmpt}")





for k in range(n-1):
    k=k+1
    cmpt=0
    for i in range(n):
        alphaJ=0
        for j in range(n):
            if j<i:
                alphaJ=matrice[i][j]* listeJ[k][j] + alphaJ
            elif j>i:
                alphaJ=matrice[i][j]* listeJ[k][j] + alphaJ
            else:
                alphaJ=0+alphaJ
        listeJ[k][i]= (b[i]-alphaJ)/matrice[i][i]



print(f"La solution est : {listeJ}, le nbre d'itérations est : {cmpt}")