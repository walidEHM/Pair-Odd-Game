import numpy as np
import random

class MarkovBrain:
    """
    Cerveau IA basé sur la Théorie Pure des Jeux à Somme Nulle.

    Jeu : Pair-Impair
    - Joueur IA  : veut que la somme soit IMPAIR  → gain = +1
    - Joueur Hum : veut que la somme soit PAIR    → gain = +1

    Matrice des gains (du point de vue de l'IA) :
                Humain joue PAIR (0)   Humain joue IMPAIR (1)
    IA joue PAIR (0)       -1                    +1
    IA joue IMPAIR (1)     +1                    -1

    Stratégie Mixte Optimale (Von Neumann / Minimax) :
    - Δ = (a11 + a22) - (a12 + a21) = (-1 + -1) - (1 + 1) = -4
    - p*_IA    = (a22 - a21) / Δ = (-1 - 1) / -4 = 0.5  → jouer PAIR 50% du temps
    - q*_Humain= (a22 - a12) / Δ = (-1 - 1) / -4 = 0.5
    - Valeur du jeu V = (a11*a22 - a12*a21) / Δ = (1 - 1) / -4 = 0

    → Le jeu est équitable (V=0), et la stratégie optimale est 50/50.
    """

    # --- Matrice des gains (point de vue IA) ---
    #         Hum_PAIR  Hum_IMPAIR
    # IA_PAIR    -1        +1
    # IA_IMPAIR  +1        -1
    GAIN_MATRIX = np.array([
        [-1,  1],   # IA joue PAIR
        [ 1, -1]    # IA joue IMPAIR
    ])

    def __init__(self):
        # --- Calcul automatique de la stratégie mixte optimale (Minimax) ---
        a11 = self.GAIN_MATRIX[0, 0]  # -1
        a12 = self.GAIN_MATRIX[0, 1]  # +1
        a21 = self.GAIN_MATRIX[1, 0]  # +1
        a22 = self.GAIN_MATRIX[1, 1]  # -1

        delta = (a11 + a22) - (a12 + a21)  # Facteur Delta

        if delta != 0:
            # Probabilité optimale pour l'IA de jouer PAIR (ligne 0)
            self.p_pair = (a22 - a21) / delta
            # Probabilité optimale pour l'IA de jouer IMPAIR (ligne 1)
            self.p_impair = 1 - self.p_pair
            # Valeur théorique du jeu
            self.game_value = (a11 * a22 - a12 * a21) / delta
        else:
            # Cas dégénéré : 50/50 par défaut
            self.p_pair = 0.5
            self.p_impair = 0.5
            self.game_value = 0.0

        self.options_pair   = [0, 2, 4]
        self.options_impair = [1, 3, 5]

        # Suivi des scores (somme nulle)
        self.net_gain = 0
        self.rounds   = 0

        # Historique pour affichage
        self.last_ia_type  = None   # 0=PAIR, 1=IMPAIR
        self.last_hum_type = None   # 0=PAIR, 1=IMPAIR

    # ------------------------------------------------------------------
    def get_move(self, _pred=None):
        """
        Théorie Pure : ignore toute prédiction sur l'humain.
        L'IA tire aléatoirement selon sa stratégie mixte optimale (p*, 1-p*).
        C'est la stratégie Minimax — elle rend l'IA inexploitable.
        """
        if random.random() < self.p_pair:
            self.last_ia_type = 0
            return random.choice(self.options_pair)
        else:
            self.last_ia_type = 1
            return random.choice(self.options_impair)

    # ------------------------------------------------------------------
    def predict(self):
        """
        Dans la théorie pure, l'IA n'a pas besoin de prédire l'adversaire.
        Méthode conservée pour compatibilité avec main.py.
        """
        return None

    # ------------------------------------------------------------------
    def update(self, hum_val, win_ia):
        """
        Met à jour le gain cumulé (somme nulle).
        La matrice de Markov n'est plus utilisée : l'IA ne cherche plus
        à exploiter les habitudes humaines, elle joue de façon optimale.
        """
        self.last_hum_type = 0 if hum_val % 2 == 0 else 1
        self.rounds += 1

        if win_ia:
            self.net_gain += 1
        else:
            self.net_gain -= 1

    # ------------------------------------------------------------------
    def get_probabilities(self):
        """
        Retourne les probabilités THÉORIQUES optimales de l'IA (p*, 1-p*).
        Contrairement à l'ancienne version, ce ne sont plus des fréquences
        observées mais les valeurs calculées par le Minimax.
        """
        return round(self.p_pair * 100), round(self.p_impair * 100)

    # ------------------------------------------------------------------
    def get_game_value(self):
        """Valeur théorique du jeu V (calculée par Minimax)."""
        return self.game_value

    # ------------------------------------------------------------------
    def get_expected_gain(self):
        """
        Gain moyen espéré sur le nombre de rounds joués.
        Devrait tendre vers V * rounds = 0 sur le long terme.
        """
        if self.rounds == 0:
            return 0.0
        return round(self.net_gain / self.rounds, 3)