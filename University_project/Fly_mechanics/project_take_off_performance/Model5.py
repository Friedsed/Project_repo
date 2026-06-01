import numpy as np
import matplotlib.pyplot as plt


class AircraftTakeoff:

    def __init__(self, params):

        self.p = params

        self.p.setdefault("g", 9.80665)
        self.p.setdefault("mu", 0.04)
        self.p.setdefault("pente", 0.0)
        self.p.setdefault("phi", 1.0)
        self.p.setdefault("k", 0.0)
        self.p.setdefault("altitude", 0)

        self.p["W"] = self.p["mass_kg"] * self.p["g"]

        rho0 = 1.225

        self.p["rho"] = (
            rho0
            * (1 - 2.25577e-5 * self.p["altitude"]) ** 5.2559
        )

    # =====================================================
    # Vitesse de décollage
    # =====================================================

    def Vlo(self):

        Vi = np.sqrt(
            (2 * self.p["W"])
            /
            (
                self.p["rho"]
                * self.p["Sw"]
                * self.p["Clmax"]
            )
        )

        return 1.1 * Vi

    # =====================================================
    # Coefficient de traînée
    # =====================================================

    def Cd(self):

        Cl = 0.75 * self.p["Clmax"]

        return (
            self.p["Cdo"]
            +
            self.p["phi"]
            *
            (
                Cl**2
                /
                (
                    np.pi
                    * self.p["Ra"]
                    * 0.82
                )
            )
        )

    # =====================================================
    # Constantes du modèle
    # =====================================================

    def constantes(self, modele=1):

        rho0 = 1.225

        Vlo = self.Vlo()

        poussee = (
            self.p["T0"]
            *
            (
                self.p["rho"]
                / rho0
            )
        )

        A = self.p["g"] * (
            (poussee / self.p["W"])
            - self.p["mu"]
            - self.p["pente"]
        )

        Cl = 0.75 * self.p["Clmax"]

        Cd = self.Cd()

        if modele == 1:

            B = (
                self.p["rho"]
                * self.p["Sw"]
                * (
                    Cd
                    - self.p["mu"] * Cl
                )
            ) / (
                2 * self.p["mass_kg"]
            )

        elif modele == 2:

            B = (
                self.p["g"]
                / self.p["W"]
            ) * (
                0.5
                * self.p["rho"]
                * self.p["Sw"]
                * (
                    Cd
                    - self.p["mu"] * Cl
                )
                +
                self.p["k"]
            )

        else:

            raise ValueError(
                "modele doit être 1 ou 2"
            )

        return Vlo, A, B

    # =====================================================
    # Distance de décollage
    # =====================================================

    def distance(self, modele=1):

        Vlo, A, B = self.constantes(modele)

        if A <= B * Vlo**2:
            return None

        s_accel = (
            -(1 / (2 * B))
            *
            np.log(
                1
                -
                (B / A)
                * Vlo**2
            )
        )

        s_rotation = Vlo

        return s_accel + s_rotation

    # =====================================================
    # Temps jusqu'à VLO
    # =====================================================

    def temps(self, modele=2):

        Vlo, A, B = self.constantes(modele)

        if A <= B * Vlo**2:
            return None

        return (
            1
            /
            np.sqrt(A * B)
        ) * np.arctanh(
            np.sqrt(B / A)
            * Vlo
        )

    # =====================================================
    # Trajectoire
    # =====================================================

    def trajectoire(
        self,
        modele=1,
        n_points=200
    ):

        Vlo, A, B = self.constantes(modele)

        vitesses = np.linspace(
            0,
            Vlo,
            n_points
        )

        distances = np.where(
            vitesses > 0,
            -(1 / (2 * B))
            *
            np.log(
                1
                -
                (B / A)
                * vitesses**2
            ),
            0
        )

        return distances, vitesses

    # =====================================================
    # Distance selon la pente
    # =====================================================

    def distance_par_pentes(
        self,
        slopes,
        modele=1
    ):

        distances = []

        for pente in slopes:

            p = self.p.copy()

            p["pente"] = pente

            sim = AircraftTakeoff(p)

            distances.append(
                sim.distance(modele)
            )

        return distances

    # =====================================================
    # Approximation vitesse moyenne
    # =====================================================

    def approx_distance_v_moyenne(self):

        rho = self.p["rho"]

        g = self.p["g"]

        W = self.p["W"]

        Vavg = 0.7 * self.Vlo()

        Cl = 0.75 * self.p["Clmax"]

        Cd = self.Cd()

        D = (
            0.5
            * rho
            * self.p["Sw"]
            * Cd
            * Vavg**2
        )

        L = (
            0.5
            * rho
            * self.p["Sw"]
            * Cl
            * Vavg**2
        )

        T = (
            self.p["T0"]
            *
            (
                rho / 1.225
            )
        )

        denom = (
            g
            * rho
            * self.p["Sw"]
            * self.p["Clmax"]
            *
            (
                T
                -
                (
                    D
                    +
                    self.p["mu"]
                    * (W - L)
                )
            )
        )

        if denom <= 0:
            return None

        return 1.44 * W**2 / denom

    # =====================================================
    # Approximation haute performance
    # =====================================================

    def approx_distance_haute_performance(self):

        rho = self.p["rho"]

        g = self.p["g"]

        W = self.p["W"]

        T = (
            self.p["T0"]
            *
            (
                rho / 1.225
            )
        )

        denom = (
            g
            * rho
            * self.p["Sw"]
            * self.p["Clmax"]
            * T
        )

        if denom <= 0:
            return None

        return 1.44 * W**2 / denom

    # =====================================================
    # Graphique vitesse-distance
    # =====================================================

    def plot_velocity_distance(
        self,
        modeles=(1, 2),
        n_points=200
    ):

        plt.figure(figsize=(10, 6))

        for modele in modeles:

            S, V = self.trajectoire(
                modele,
                n_points
            )

            plt.plot(
                S,
                V,
                label=f"Modèle {modele}"
            )

        plt.xlabel("Distance (m)")
        plt.ylabel("Vitesse (m/s)")
        plt.title(
            "Vitesse en fonction de la distance"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    # =====================================================
    # Graphique pente-distance
    # =====================================================

    def plot_distance_vs_slope(
        self,
        slopes=(0, 0.005, 0.01, 0.02),
        modele=1
    ):

        distances = self.distance_par_pentes(
            slopes,
            modele
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            [100 * p for p in slopes],
            distances,
            marker="o"
        )

        plt.xlabel("Pente (%)")
        plt.ylabel("Distance (m)")
        plt.title(
            f"Distance de décollage - modèle {modele}"
        )

        plt.grid(True)
        plt.show()

    # =====================================================
    # Résumé
    # =====================================================

    def summary(self):

        results = {

            "Aircraft":
                self.p.get("name", "Unknown"),

            "Altitude":
                self.p["altitude"],

            "Density":
                self.p["rho"],

            "VLO":
                self.Vlo(),

            "Distance modèle 1":
                self.distance(1),

            "Distance modèle 2":
                self.distance(2),

            "Temps modèle 2":
                self.temps(2),

            "Approx moyenne":
                self.approx_distance_v_moyenne(),

            "Approx haute performance":
                self.approx_distance_haute_performance()
        }

        print("=" * 120)

        for k, v in results.items():
            print(f"{k} : {v}")

        print("=" * 120)

        return results