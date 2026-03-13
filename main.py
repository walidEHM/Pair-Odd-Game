import cv2
import time
from src.vision import VisionManager
from src.brain import MarkovBrain
from src.ui_manager import UIManager
import numpy as np

vision = VisionManager()
brain = MarkovBrain()
ui = UIManager()
cap = cv2.VideoCapture(0)

etat = "ATTENTE"
score_ia = 0
score_humain = 0

choix_ia, main_visuelle = 0, 0
timer_start = 0
log = "ESPACE POUR JOUER"

last_ia_val = None
last_hum_val = None
last_total = None

def resize_with_letterbox(frame, target_w, target_h):
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Créer un canevas de la taille cible, rempli de noir
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    nb_fingers, frame_v = vision.detect_fingers(frame)

    key = cv2.waitKey(1) & 0xFF

    # --- Récupération des données Minimax à chaque frame ---
    probas        = brain.get_probabilities()   # (50, 50) — stratégie mixte optimale
    game_value    = brain.get_game_value()      # V = 0.0  — valeur théorique du jeu
    expected_gain = brain.get_expected_gain()   # gain moyen observé → tend vers 0

    # ----------------------------------------------------------
    if etat == "ATTENTE":
        main_visuelle = 0
        if key == ord(' '):
            etat, timer_start = "CHRONO", time.time()
            choix_ia = brain.get_move()

    elif etat == "CHRONO":
        elapsed = time.time() - timer_start
        main_visuelle = int((elapsed * 20) % 5) + 1

        if elapsed > 3:
            hum_val = nb_fingers
            total = choix_ia + hum_val
            win_ia = (total % 2 != 0)

            # Sauvegarde pour affichage
            last_ia_val = choix_ia
            last_hum_val = hum_val
            last_total = total

            # Mise à jour du score (somme nulle)
            brain.update(hum_val, win_ia)

            # --- LOGIQUE DE SOMME NULLE ---
            if win_ia:
                score_ia += 1
                score_humain -= 1
                log = f"IA GAGNE (+1) | VOUS PERDEZ (-1)  ->  {total} est IMPAIR"
            else:
                score_ia -= 1
                score_humain += 1
                log = f"VOUS GAGNEZ (+1) | IA PERD (-1)  ->  {total} est PAIR"

            # Rafraîchissement après update
            expected_gain = brain.get_expected_gain()

            main_visuelle = choix_ia
            etat, timer_start = "RESULTAT", time.time()

    elif etat == "RESULTAT" and time.time() - timer_start > 3:
        etat = "ATTENTE"

    # ----------------------------------------------------------
    # Calcul du winrate de l'IA
    rounds = brain.rounds
    if rounds > 0:
        # wins = (net_gain + rounds) / 2  car net_gain = wins - pertes
        winrate_ia = (brain.net_gain + rounds) / (2 * rounds) * 100
    else:
        winrate_ia = 0.0

    # Affichage complet avec toutes les données dynamiques
    layout = ui.draw_skeleton(
        score_ia,
        score_humain,
        probas,
        log,
        game_value=game_value,
        expected_gain=expected_gain,
        last_ia=last_ia_val,
        last_hum=last_hum_val,
        last_sum=last_total,
        rounds=rounds,
        winrate_ia=winrate_ia,
        current_fingers=nb_fingers
    )
    ui.draw_ia_hand(layout, main_visuelle)

    webcam_crop = resize_with_letterbox(frame_v, 400, 300)
    layout[100:400, 475:875] = webcam_crop

    cv2.imshow("AIOIT - Zero-Sum Game", layout)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()