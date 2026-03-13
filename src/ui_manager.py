import cv2
import numpy as np
import os

class UIManager:
    def __init__(self):
        self.width, self.height = 900, 720
        self.bg_color = (45, 50, 60)
        self.cyan = (255, 255, 0)
        self.white = (255, 255, 255)
        self.orange = (0, 165, 255)
        self.hand_images = {}
        self.load_assets()

    def image_resize_aspect(self, img, target_size=300):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
        offset_y = (target_size - new_h) // 2
        offset_x = (target_size - new_w) // 2
        if resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        return canvas

    def load_assets(self):
        names = {0:"zero.png", 1:"one.png", 2:"two.png", 3:"three.png", 4:"four.png", 5:"five.png"}
        for val, name in names.items():
            path = os.path.join("assets", name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.hand_images[val] = self.image_resize_aspect(img)

    def draw_skeleton(self, score_ia, score_humain, probas, log,
                      game_value=0.0, expected_gain=0.0,
                      last_ia=None, last_hum=None, last_sum=None,
                      rounds=0, winrate_ia=0.0, current_fingers=None):
        """
        Affiche l'interface complète avec :
        - scores cumules (somme nulle)
        - dernier coup joue
        - statistiques (rounds, winrate IA, gain moyen observe)
        - main detectee en direct
        - rappel theorique (strategie optimale et valeur du jeu)
        - message de log
        """
        frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- Lignes de structure ---
        cv2.line(frame, (450, 20), (450, 480), (200, 200, 200), 1)
        cv2.line(frame, (50, 520), (850, 520), (200, 200, 200), 2)

        # --- SCORES (Somme Nulle) ---
        color_ia  = (0, 255, 0) if score_ia  >= 0 else (0, 0, 255)
        color_hum = (0, 255, 0) if score_humain >= 0 else (0, 0, 255)

        cv2.putText(frame, f"GAIN: {score_ia:+}",     (50,  505), font, 1.2, color_ia,  3)
        cv2.putText(frame, f"GAIN: {score_humain:+}", (600, 505), font, 1.2, color_hum, 3)

        # --- Titres des joueurs ---
        cv2.putText(frame, "JOUEUR 1 (IA)",     (130, 45), font, 0.8, self.white, 2)
        cv2.putText(frame, "JOUEUR 2 (VOUS)",   (570, 45), font, 0.8, self.white, 2)

        # --- Informations dynamiques ---
        start_y = 540
        line_height = 25

        # Dernier coup
        if last_ia is not None and last_hum is not None and last_sum is not None:
            dernier = (f"Dernier coup: IA={last_ia} ({'PAIR' if last_ia%2==0 else 'IMPAIR'}), "
                       f"Vous={last_hum} ({'PAIR' if last_hum%2==0 else 'IMPAIR'}) -> "
                       f"Somme={last_sum} -> {'IA' if last_sum%2!=0 else 'VOUS'} gagne")
        else:
            dernier = "Dernier coup: ---"
        cv2.putText(frame, dernier, (50, start_y), font, 0.55, self.white, 1)

        # Statistiques
        stats = f"Rounds: {rounds}  |  Winrate IA: {winrate_ia:.1f}%  |  Gain moyen observe: {expected_gain:+.3f}"
        cv2.putText(frame, stats, (50, start_y + line_height), font, 0.55, self.white, 1)

        # Main detectee en direct
        if current_fingers is not None:
            main_txt = f"Doigt detectee: {current_fingers} doigts ({'PAIR' if current_fingers%2==0 else 'IMPAIR'})"
        else:
            main_txt = "Main detectee: ---"
        cv2.putText(frame, main_txt, (50, start_y + 2*line_height), font, 0.55, self.white, 1)

        # Rappel theorique (strategie optimale et valeur du jeu)
        theory = f"Strategie optimale IA: PAIR {probas[0]}% / IMPAIR {probas[1]}%  |  Valeur jeu V={game_value:+.2f}"
        cv2.putText(frame, theory, (50, start_y + 3*line_height + 5), font, 0.5, self.orange, 1)

        # --- Log (message d'information) ---
        cv2.putText(frame, f"> {log}", (50, 680), font, 0.5, self.cyan, 1)

        return frame

    def draw_ia_hand(self, frame, val):
        """Dessine la main de l'IA à partir de l'image correspondant au nombre 'val'."""
        if val in self.hand_images:
            hand = self.hand_images[val]
            y, x = 100, 75
            alpha = hand[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+300, x:x+300, c] = (
                    alpha * hand[:, :, c] + (1 - alpha) * frame[y:y+300, x:x+300, c]
                )