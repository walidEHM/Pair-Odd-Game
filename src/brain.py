
import numpy as np
import random

class MarkovBrain:
    """
    Cerveau IA basé sur la Théorie des Jeux.
    Possède deux modes :
    - mode minimax (use_learning=False) : stratégie mixte optimale (50/50) inexploitable.
    - mode learning (use_learning=True) : utilise une chaîne de Markov d'ordre 1 pour modéliser
      les transitions des coups de l'humain et joue la meilleure réponse, avec une supervision
      du maximin (retour à la stratégie 50/50 en cas d'incertitude).
    """

    GAIN_MATRIX = np.array([[-1, 1], [1, -1]])

    def __init__(self, seuil=0.6):
        # --- Calcul de la stratégie mixte optimale (Minimax) ---
        a11, a12, a21, a22 = -1, 1, 1, -1
        delta = (a11 + a22) - (a12 + a21)

        if delta != 0:
            self.p_pair = (a22 - a21) / delta  # = 0.5
            self.p_impair = 1 - self.p_pair
            self.game_value = (a11 * a22 - a12 * a21) / delta  # = 0
        else:
            self.p_pair = 0.5
            self.p_impair = 0.5
            self.game_value = 0.0

        self.options_pair = [0, 2, 4]
        self.options_impair = [1, 3, 5]

        # Suivi des scores cumulés (somme nulle)
        self.net_gain = 0
        self.rounds = 0

        # Historique des derniers types de coups (pour affichage)
        self.last_ia_type = None
        self.last_hum_type = None

        # Paramètre de mode : False = minimax pur, True = apprentissage markovien
        self.use_learning = False
        self.seuil = seuil  # seuil de confiance pour l'utilisation de la prédiction

        # Pour l'apprentissage markovien
        self.transition_matrix = np.zeros((2, 2), dtype=int)  # [état précédent][état suivant]
        self.last_hum_state = None  # dernier état humain observé (0 = PAIR, 1 = IMPAIR)

    def set_learning(self, enabled):
        """Active ou désactive le mode apprentissage."""
        self.use_learning = enabled

    def get_move(self, _pred=None):
        """
        Retourne le nombre de doigts joué par l'IA.
        Si use_learning=True, utilise la chaîne de Markov pour prédire le prochain coup humain
        et joue la meilleure réponse, avec retour au minimax en cas de faible confiance.
        Sinon, utilise la stratégie mixte optimale (50/50).
        """
        if self.use_learning:
            # ---------- MODE APPRENTISSAGE AVEC CHAÎNE DE MARKOV ----------
            # Calcul de la probabilité que l'humain joue PAIR sachant le dernier état
            prob_pair = 0.5  # valeur par défaut
            if self.last_hum_state is not None:
                total = np.sum(self.transition_matrix[self.last_hum_state])
                if total > 0:
                    prob_pair = self.transition_matrix[self.last_hum_state][0] / total

            # Décision avec supervision du maximin
            if prob_pair > self.seuil:
                # Humain très probablement PAIR → l'IA joue IMPAIR pour gagner
                self.last_ia_type = 1
                return random.choice(self.options_impair)
            elif prob_pair < 1 - self.seuil:
                # Humain très probablement IMPAIR → l'IA joue PAIR
                self.last_ia_type = 0
                return random.choice(self.options_pair)
            else:
                # Incertitude : on revient à la stratégie minimax (50/50)
                if random.random() < 0.5:
                    self.last_ia_type = 0
                    return random.choice(self.options_pair)
                else:
                    self.last_ia_type = 1
                    return random.choice(self.options_impair)
        else:
            # ---------- MODE MINIMAX PUR ----------
            if random.random() < self.p_pair:
                self.last_ia_type = 0
                return random.choice(self.options_pair)
            else:
                self.last_ia_type = 1
                return random.choice(self.options_impair)

    def predict(self):
        """Méthode conservée pour compatibilité, non utilisée."""
        return None

    def update(self, hum_val, win_ia):
        """
        Met à jour les statistiques après un coup.
        Enregistre le type de coup humain et met à jour la matrice de transition markovienne.
        """
        hum_type = 0 if hum_val % 2 == 0 else 1
        self.last_hum_type = hum_type
        self.rounds += 1

        # Mise à jour de la matrice de transition
        if self.last_hum_state is not None:
            self.transition_matrix[self.last_hum_state][hum_type] += 1
        self.last_hum_state = hum_type

        # Mise à jour du gain net (somme nulle)
        if win_ia:
            self.net_gain += 1
        else:
            self.net_gain -= 1

    def get_probabilities(self):
        """
        Retourne les probabilités affichées dans l'interface.
        On affiche toujours les probabilités théoriques du minimax (50/50)
        pour que l'utilisateur voie la référence, quel que soit le mode.
        """
        return round(self.p_pair * 100), round(self.p_impair * 100)

    def get_game_value(self):
        return self.game_value

    def get_expected_gain(self):
        if self.rounds == 0:
            return 0.0
        return round(self.net_gain / self.rounds, 3)