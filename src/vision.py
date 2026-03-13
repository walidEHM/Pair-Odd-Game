import cv2
import mediapipe as mp
import math


class VisionManager():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def _angle(self, a, b, c):
        """
        Calcule l'angle ABC en degrés
        a, b, c = landmarks Mediapipe
        """
        ab = (a.x - b.x, a.y - b.y, a.z - b.z)
        cb = (c.x - b.x, c.y - b.y, c.z - b.z)

        dot = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2]
        norm_ab = math.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
        norm_cb = math.sqrt(cb[0]**2 + cb[1]**2 + cb[2]**2)

        if norm_ab * norm_cb == 0:
            return 0

        cos_angle = dot / (norm_ab * norm_cb)
        cos_angle = max(-1, min(1, cos_angle))

        return math.degrees(math.acos(cos_angle))

    def detect_fingers(self, frame):
        """
        Détecte le nombre de doigts levés (0 à 5)
        Méthode robuste basée sur l'angle des phalanges.
        Compatible paume, dos, inclinaison et doigts pliés.
        Retourne (finger_count, frame_annoté)
        """

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Dessin des points
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4),  # points
                    self.mp_draw.DrawingSpec(thickness=10)                    # lignes
                )

                lm = hand_landmarks.landmark

                # ===== DOIGTS (index, majeur, annulaire, auriculaire) =====
                fingers = [
                    (5, 6, 8),     # index
                    (9, 10, 12),   # majeur
                    (13, 14, 16),  # annulaire
                    (17, 18, 20)   # auriculaire
                ]

                for mcp, pip, tip in fingers:
                    ang = self._angle(lm[mcp], lm[pip], lm[tip])

                    if ang > 155:   # seuil doigt droit
                        finger_count += 1

                # ===== POUCE =====
                thumb_angle = self._angle(lm[1], lm[2], lm[4])

                if thumb_angle > 145:
                    finger_count += 1

        return finger_count, frame