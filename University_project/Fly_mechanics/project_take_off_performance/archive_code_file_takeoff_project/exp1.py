
import tkinter as tk
from tkinter import messagebox


# ---------------------------------------------------------
# FONCTION DE TRAITEMENT EN ARRIÈRE-PLAN
# ---------------------------------------------------------
def traiter_donnees():
    """Cette fonction récupère les données stockées en arrière-plan

    et simule un traitement automatique.
    """
    # Récupération des valeurs actuelles des variables d'arrière-plan
    choix_lettre = variable_choix.get()
    valeur_1 = variable_val1.get()
    valeur_2 = variable_val2.get()
    valeur_3 = variable_val3.get()

    # Simulation du traitement de données
    resultat = (
        f"--- Traitement en arrière-plan ---\n"
        f"Option choisie : {choix_lettre}\n"
        f"Valeur 1 : {valeur_1}\n"
        f"Valeur 2 : {valeur_2}\n"
        f"Valeur 3 : {valeur_3}\n"
        f"Statut : Données synchronisées et traitées !"
    )

    # Affichage du résultat dans le Bloc 4 (Zone de log)
    texte_log.delete("1.0", tk.END)
    texte_log.insert(tk.END, resultat)


# ---------------------------------------------------------
# INTERFACE GRAPHIQUE PRINCIPALE
# ---------------------------------------------------------
fenetre = tk.Tk()
fenetre.title("Aircraft simulation")
fenetre.geometry("650x500")

# Configuration de la grille principale (2x2)
fenetre.rowconfigure((0, 1), weight=1)
fenetre.columnconfigure((0, 1), weight=1)

# ---------------------------------------------------------
# VARIABLES D'ARRIÈRE-PLAN (Conservation des données)
# ---------------------------------------------------------
# Ces variables conservent l'état réel de tes données à tout moment.
variable_choix = tk.StringVar(value="A")  # Choix par défaut : A
variable_val1 = tk.StringVar(value="100")  # Valeur par défaut
variable_val2 = tk.StringVar(value="200")
variable_val3 = tk.StringVar(value="300")

# ---
# BLOC 1 : LE CHOIX (A, B ou C)
# ---
bloc1 = tk.LabelFrame(
    fenetre, text=" 1. Choisir une option ", padx=15, pady=15, fg="blue"
)
bloc1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# On lie les 3 Radiobuttons à la MÊME variable 'variable_choix'
radio_A = tk.Radiobutton(
    bloc1, text="Option A", variable=variable_choix, value="A"
)
radio_B = tk.Radiobutton(
    bloc1, text="Option B", variable=variable_choix, value="B"
)
radio_C = tk.Radiobutton(
    bloc1, text="Option C", variable=variable_choix, value="C"
)

radio_A.pack(anchor="w", pady=2)
radio_B.pack(anchor="w", pady=2)
radio_C.pack(anchor="w", pady=2)


# ---
# BLOC 2 : LES VALEURS (Saisies / Entrées)
# ---
bloc2 = tk.LabelFrame(
    fenetre, text=" 2. Saisie des Valeurs ", padx=15, pady=15, fg="blue"
)
bloc2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# On lie directement chaque Entry à sa variable d'arrière-plan via 'textvariable'
tk.Label(bloc2, text="Valeur X :").pack(anchor="w")
entry1 = tk.Entry(bloc2, textvariable=variable_val1)
entry1.pack(fill="x", pady=2)

tk.Label(bloc2, text="Valeur Y :").pack(anchor="w")
entry2 = tk.Entry(bloc2, textvariable=variable_val2)
entry2.pack(fill="x", pady=2)

tk.Label(bloc2, text="Valeur Z :").pack(anchor="w")
entry3 = tk.Entry(bloc2, textvariable=variable_val3)
entry3.pack(fill="x", pady=2)


# ---
# BLOC 3 : LE PANEL DE CONTRÔLE (Déclencheur)
# ---
bloc3 = tk.LabelFrame(
    fenetre, text=" 3. Actions ", padx=15, pady=15, fg="blue"
)
bloc3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

label_info = tk.Label(
    bloc3,
    text="Modifie une valeur à droite\nou change d'option, puis valide.",
    justify="left",
)
label_info.pack(pady=10)

# Ce bouton appelle la fonction qui va lire les variables d'arrière-plan
bouton_calcul = tk.Button(
    bloc3,
    text="Lancer le traitement",
    command=traiter_donnees,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 10, "bold"),
)
bouton_calcul.pack(fill="x", pady=5)


# ---
# BLOC 4 : CONSOLE DE TRAITEMENT (Aperçu de l'arrière-plan)
# ---
bloc4 = tk.LabelFrame(
    fenetre, text=" 4. Console de traitement (Arrière-plan) ", fg="green"
)
bloc4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

texte_log = tk.Text(bloc4, bg="#F5F5F5", font=("Courier", 9))
texte_log.pack(fill="both", expand=True, padx=5, pady=5)
texte_log.insert(
    "1.0", "En attente du premier traitement..."
)  # Message initial

# Lancement de la boucle
fenetre.mainloop()