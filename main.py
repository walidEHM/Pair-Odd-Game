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

etat = "CHOIX"                 # <-- Changé : maintenant on commence par le choix
regle_joueur = None             # True = joueur gagne sur pair, False = sur impair
score_ia = 0
score_humain = 0

choix_ia, main_visuelle = 0, 0
timer_start = 0
log = ""

last_ia_val = None
last_hum_val = None
last_total = None

def resize_with_letterbox(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

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

    # --- État CHOIX : écran de sélection de la règle ---
    if etat == "CHOIX":
        # Création de l'écran de choix avec overlay sombre
        choice_frame = np.full((ui.height, ui.width, 3), ui.bg_color, dtype=np.uint8)
        overlay = choice_frame.copy()
        cv2.rectangle(overlay, (0, 0), (ui.width, ui.height), (0, 0, 0), -1)
        choice_frame = cv2.addWeighted(overlay, 0.6, choice_frame, 0.4, 0)

        # Instructions
        cv2.putText(choice_frame, "Choisissez votre regle de gain :", (50, 420),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui.cyan, 2)
        cv2.putText(choice_frame, "Appuyez sur P : vous gagnez si la somme est PAIRE", (50, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ui.white, 2)
        cv2.putText(choice_frame, "Appuyez sur I : vous gagnez si la somme est IMPAIRE", (50, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ui.white, 2)
        cv2.putText(choice_frame, "L'IA adoptera la regle opposee.", (50, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)

        # Affichage du flux webcam dans le coin droit (optionnel)
        webcam_crop = resize_with_letterbox(frame_v, 400, 300)
        choice_frame[100:400, 475:875] = webcam_crop

        cv2.imshow("AIOIT - Zero-Sum Game", choice_frame)

        # Gestion des touches de choix
        if key == ord('p'):
            regle_joueur = True
            etat = "ATTENTE"
            log = "Regle choisie : Vous gagnez sur PAIR. L'IA gagne sur IMPAIR."
        elif key == ord('i'):
            regle_joueur = False
            etat = "ATTENTE"
            log = "Regle choisie : Vous gagnez sur IMPAIR. L'IA gagne sur PAIR."
        elif key == ord('q'):
            break

        continue  # On saute le reste de la boucle tant qu'on est dans le choix

    # --- États suivants (ATTENTE, CHRONO, RESULTAT) ---
    # Récupération des données Minimax
    probas        = brain.get_probabilities()   # (50, 50) — stratégie mixte optimale
    game_value    = brain.get_game_value()      # V = 0.0  — valeur théorique du jeu
    expected_gain = brain.get_expected_gain()   # gain moyen observé → tend vers 0

    # Détermination du texte de stratégie
    strategy_text = "Minimax" if not brain.use_learning else "Maximin (Markov)"

    if etat == "ATTENTE":
        main_visuelle = 0
        if key == ord(' '):
            etat, timer_start = "CHRONO", time.time()
            choix_ia = brain.get_move()
        # Option: basculer le mode avec la touche 'l'
        if key == ord('l'):
            brain.set_learning(not brain.use_learning)
            log = f"Mode: {'Minimax' if not brain.use_learning else 'Maximin (Markov)'}"

    elif etat == "CHRONO":
        elapsed = time.time() - timer_start
        main_visuelle = int((elapsed * 20) % 5) + 1

        if elapsed > 3:
            hum_val = nb_fingers
            total = choix_ia + hum_val

            # Détermination du gagnant en fonction de la règle choisie
            if regle_joueur:          # joueur gagne sur pair
                win_ia = (total % 2 != 0)   # IA gagne si impair
            else:                      # joueur gagne sur impair
                win_ia = (total % 2 == 0)   # IA gagne si pair

            # Sauvegarde pour affichage
            last_ia_val = choix_ia
            last_hum_val = hum_val
            last_total = total

            # Mise à jour du score (somme nulle)
            brain.update(hum_val, win_ia)

            # --- LOGIQUE DE SOMME NULLE (adaptée à la règle) ---
            if win_ia:
                score_ia += 1
                score_humain -= 1
                if regle_joueur:
                    log = f"IA GAGNE (+1) | VOUS PERDEZ (-1)  ->  {total} est IMPAIR (vous vouliez pair)"
                else:
                    log = f"IA GAGNE (+1) | VOUS PERDEZ (-1)  ->  {total} est PAIR (vous vouliez impair)"
            else:
                score_ia -= 1
                score_humain += 1
                if regle_joueur:
                    log = f"VOUS GAGNEZ (+1) | IA PERD (-1)  ->  {total} est PAIR"
                else:
                    log = f"VOUS GAGNEZ (+1) | IA PERD (-1)  ->  {total} est IMPAIR"

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
        winrate_ia = (brain.net_gain + rounds) / (2 * rounds) * 100
    else:
        winrate_ia = 0.0

    # Affichage complet (avec ajout de regle_joueur et ia_strategy)
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
        current_fingers=nb_fingers,
        regle_joueur=regle_joueur,
        ia_strategy=strategy_text   # <-- AJOUT
    )

    ui.draw_ia_hand(layout, main_visuelle)

    webcam_crop = resize_with_letterbox(frame_v, 400, 300)
    layout[100:400, 475:875] = webcam_crop

    cv2.imshow("AIOIT - Zero-Sum Game", layout)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 