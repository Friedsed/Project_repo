import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def comparer_modeles_avions(
    avions,
    modeles,
    variables=("Sa", "Sr", "Sc", "S", "Vlo"),
    reference=None,
    tracer=True
):
    """
    Paramètres
    ----------
    avions : liste de str
        Noms des avions.
    modeles : dict
        Dictionnaire de la forme :
        {
            "Model1": {
                "Sa": [...],
                "Sr": [...],
                "Sc": [...],
                "S":  [...],
                "Vlo":[...]
            },
            "Model2": {...},
            "Model4": {...}
        }
    variables : tuple
        Variables à comparer.
    reference : dict ou None
        Même structure que modeles, ou bien un modèle de référence.
    tracer : bool
        Si True, affiche les graphes.

    Retour
    ------
    df_long : DataFrame
        Données au format long.
    resume : DataFrame
        Résumé des erreurs si référence fournie.
    """

    lignes = []

    for modele, data_modele in modeles.items():
        for var in variables:
            valeurs = data_modele[var]
            for avion, valeur in zip(avions, valeurs):
                lignes.append({
                    "Avion": avion,
                    "Modele": modele,
                    "Variable": var,
                    "Valeur": valeur
                })

    df = pd.DataFrame(lignes)

    resume = None

    if reference is not None:
        ref_lignes = []
        for var in variables:
            ref_vals = reference[var]
            for avion, ref_val in zip(avions, ref_vals):
                ref_lignes.append({
                    "Avion": avion,
                    "Variable": var,
                    "Reference": ref_val
                })

        df_ref = pd.DataFrame(ref_lignes)
        df = df.merge(df_ref, on=["Avion", "Variable"], how="left")

        df["Erreur_abs"] = (df["Valeur"] - df["Reference"]).abs()
        df["Erreur_rel"] = df["Erreur_abs"] / df["Reference"].replace(0, np.nan)

        resume = df.groupby(["Modele", "Variable"]).agg(
            MAE=("Erreur_abs", "mean"),
            MedAE=("Erreur_abs", "median"),
            MAPE=("Erreur_rel", "mean")
        ).reset_index()

    if tracer:
        sns.set_theme(style="whitegrid")

        for var in variables:
            sub = df[df["Variable"] == var].copy()

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=sub, x="Avion", y="Valeur", hue="Modele", marker="o")
            plt.title(f"Comparaison des modèles pour {var}")
            plt.xlabel("Avion")
            plt.ylabel(var)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.show()

        if "Erreur_abs" in df.columns:
            for var in variables:
                sub = df[df["Variable"] == var].copy()

                plt.figure(figsize=(12, 6))
                sns.barplot(data=sub, x="Avion", y="Erreur_abs", hue="Modele")
                plt.title(f"Erreur absolue pour {var}")
                plt.xlabel("Avion")
                plt.ylabel("Erreur absolue")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.show()

    return df, resume



avions = [
    "Cessna",
    "Beluga XL",
    "Dassault Rafale",
    "Airbus A320",
    "Lockheed Martin C-130J",
    "Executive business jet",
    "Cirrus SR22",
    "data1",
    "data2"
]

modeles = {
    "Model1": {
        "Sa":  [1082.92, 4153.78, 1626.29, 3179.92, 2179.98, 2235, 1093.75, 327.02, 599.68],
        "Sr":  [99.74,   87.2,    249.21,  250.21,  207.52,  199.43, 118.87, 104.44, 104.44],
        "Sc":  [747.57, 1411.31, 467.23, 835.76, 610.54, 601.44, 290.07, 185.69, 185.69],
        "S":   [1930.22, 5652.29, 2342.74, 4265.89, 2998.04, 3035.86, 1502.69, 617.16, 889.82],
        "Vlo": [99.74, 87.2, 249.21, 250.21, 207.52, 199.43, 118.87, 104.44, 104.44]
    },
    "Model2": {
        "Sa":  [626.93, 4044.11, 1616.19, 3176.44, 2163.08, 2207.89, 875.71, 445.07, 426.13],
        "Sr":  [99.74, 87.2, 249.21, 250.21, 207.52, 199.43, 118.87, 104.44, 104.44],
        "Sc":  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "S":   [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "Vlo": [99.74, 87.2, 249.21, 250.21, 207.52, 199.43, 118.87, 104.44, 104.44]
    },
    "Model4": {
        "Sa":  [920.9, 8401.96, 2615.2, 1743.43, 1497.95, 149.26, 1111.04, 1365.25, 1365.25],
        "Sr":  [289.1, 0, 497, 298.4, 268.61, 637.62, 273.81, 340.1, 340.1],
        "Sc":  [220.2, 472.06, -208.78, 26.47, 82.91, 637.62, 157.25, 84.33, 84.33],
        "S":   [1430.01, 8874.02, 2903.42, 2068.29, 1849.47, 1424.83, 1542.11, 1789.68, 1789.68],
        "Vlo": [99, 570, 252, 179, 155, 13, 116, 139, 139]
    }
}

df_long, resume = comparer_modeles_avions(avions, modeles, tracer=True)
print(df_long.head())
print(resume)

reference = {
    "Sa":  [650, 4900, 5000, 2000, 2000, 2200, 1100, 350, 600],
    "Sr":  [805, 1500, 1500, 800, 800, 200, 120, 100, 100],
    "Sc":  [1455, 6400, 6500, 2800, 2800, 600, 300, 180, 180],
    "S":   [1800, 5600, 2300, 4200, 3000, 3000, 1500, 600, 900],
    "Vlo": [100, 90, 250, 250, 210, 200, 120, 100, 100]
}

df_long, resume = comparer_modeles_avions(avions, modeles, reference=reference, tracer=True)