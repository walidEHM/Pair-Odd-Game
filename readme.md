# 🎮 Jeu Pair–Impair avec IA (Vision par Ordinateur + Théorie des Jeux)

Ce projet implémente un **jeu Pair/Impair basé sur l'IA** où un joueur humain affronte une IA.  
Le système utilise la **vision par ordinateur pour détecter le nombre de doigts montrés par le joueur via une webcam**, tandis que l'IA choisit son coup en utilisant les principes de la **théorie des jeux**.

L'objectif du projet est d'explorer comment les **stratégies mathématiques et la perception visuelle peuvent interagir dans un système d'IA**.

---

## 🚀 Fonctionnalités

- 👁️ Détection en temps réel des doigts via une webcam (MediaPipe)
- 🤖 Deux stratégies d'IA sélectionnables :
  - **Minimax (pur)** : stratégie mixte optimale (50/50) – inexploitable
  - **Maximin avec apprentissage markovien** : apprend les schémas de l'humain et joue la meilleure réponse, avec retour au minimax en cas d'incertitude
- 🎯 Choix de la règle au début : le joueur décide s'il gagne sur **somme paire** ou **impaire** ; l'IA adopte la règle opposée
- 📊 Statistiques en direct :
  - Taux de victoire
  - Nombre de manches
  - Gain cumulé (somme nulle : +1 pour le gagnant, –1 pour le perdant)
  - Gain moyen observé
- 🧠 Rappels théoriques :
  - Probabilités mixtes optimales (toujours 50/50)
  - Valeur du jeu $V = 0$
- 🖥️ Interface graphique interactive avec images de mains et flux webcam
- ⌨️ Commandes clavier :
  - `P` / `I` – choisir la règle (Pair / Impair)
  - `Espace` – démarrer une manche
  - `L` – basculer entre le mode Minimax et le mode Markov
  - `Q` – quitter

---

## 🧠 Concept

Le jeu est un **jeu à somme nulle** :

- Le joueur montre un nombre de doigts (0 à 5).
- L'IA choisit également un nombre.
- La **somme** détermine le gagnant selon la **règle choisie par le joueur** au début :
  - Si le joueur a choisi **Pair** : somme paire → joueur gagne, somme impaire → IA gagne.
  - Si le joueur a choisi **Impair** : somme impaire → joueur gagne, somme paire → IA gagne.

En théorie classique des jeux, la **stratégie mixte optimale** pour les deux joueurs est de jouer **Pair avec probabilité 0,5** et **Impair avec probabilité 0,5**, ce qui donne une valeur de jeu attendue de **0** (jeu équitable).

L'IA implémente cette stratégie en mode **Minimax**. De plus, un mode **apprentissage** utilise une chaîne de Markov d'ordre 1 pour modéliser les transitions de l'humain entre les états Pair/Impair. Lorsque la probabilité prédite dépasse un seuil de confiance (0,6 par défaut), l'IA joue la meilleure réponse ; sinon, elle revient à la stratégie sûre 50/50.

---

## 🛠️ Technologies utilisées

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Vision par ordinateur
- Théorie des jeux

---

## 📂 Structure du projet
```
Projet_paire_impaire/
│
├── main.py                # Boucle principale du jeu
├── requirements.txt       # Dépendances
├── README.md              # Ce fichier
│
├── assets/                # Images des mains pour l'interface
│   ├── zero.png
│   ├── one.png
│   ├── two.png
│   ├── three.png
│   ├── four.png
│   └── five.png
│
└── src/                   # Modules sources
    ├── brain.py           # Stratégie IA (Minimax / Markov)
    ├── vision.py          # Détection des mains avec MediaPipe
    └── ui_manager.py      # Gestion de l'interface graphique
```

---

## ▶️ Installation

1. **Cloner le dépôt**
```bash
   git clone https://github.com/walidEHM/Pair-Odd-Game.git
   cd Pair-Odd-Game
```

2. **Installer les dépendances**
```bash
   pip install -r requirements.txt
```

3. **Lancer le jeu**
```bash
   python main.py
```

---

## 🎮 Comment jouer

**Choisissez votre règle au démarrage :**

- Appuyez sur `P` pour gagner sur **Pair** (somme paire)
- Appuyez sur `I` pour gagner sur **Impair** (somme impaire)

**Interface principale :**

- La main de l'IA est affichée à gauche.
- Votre flux webcam en direct apparaît à droite.
- Sous le flux, vous voyez le nombre de doigts détectés.

**Démarrer une manche :**

- Appuyez sur `Espace` – l'IA choisit immédiatement son nombre, et un compte à rebours de 3 secondes commence.
- Pendant le compte à rebours, la main de l'IA défile les nombres 1 à 5.
- Après 3 secondes, le nombre de doigts que vous montrez à cet instant est capturé et le résultat est affiché.
- Les statistiques sont mises à jour en temps réel en bas de l'écran.

**Changer le mode de l'IA** à tout moment en appuyant sur `L` :

- **Minimax** (par défaut) : joue aléatoirement 50/50.
- **Maximin (Markov)** : apprend votre schéma et tente de l'exploiter, mais revient au 50/50 en cas d'incertitude.

**Quitter le jeu** avec `Q`.

---

## 📊 Valeur pédagogique

Ce projet illustre la combinaison pratique de :

- **Vision par ordinateur** – détection et comptage de doigts en temps réel.
- **Théorie des jeux** – théorème du minimax, stratégies mixtes, valeur du jeu.
- **Apprentissage par renforcement** – modélisation de l'adversaire par chaîne de Markov.
- **Interaction humain–IA** – interface intuitive et retour en temps réel.

---

## 🔮 Améliorations futures

- Agent par apprentissage par renforcement (Q‑learning, policy gradient)
- Détection des mains plus robuste (occultations, arrière‑plans variés)
- Mode multijoueur (deux humains via deux caméras)
- Version web avec JavaScript/WebRTC
- Seuil de confiance adaptatif dans le mode apprentissage

---

## 👨‍💻 Auteur

**Abdoul-Walid EL-HADJ MAMA**

- Étudiant en IA
- Développeur Web & IA
- Membre du Club AIOT – IFRI (Institut de Formation et de Recherche en Informatique)

---

⭐ Si vous aimez ce projet, n'hésitez pas à mettre une étoile sur le dépôt !