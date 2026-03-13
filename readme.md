# 🎮 AI Pair–Odd Game (Computer Vision + Game Theory)

This project implements an **AI-based Pair/Odd game** where a human player competes against an AI.  
The system uses **computer vision to detect the number of fingers shown by the player through a webcam**, while the AI chooses its move using **game theory principles**.

The goal of the project is to explore how **mathematical strategies and visual perception can interact in an AI system**.

---

## 🚀 Features

- 👁️ Real-time finger detection using a webcam
- 🤖 AI opponent based on game theory
- 📊 Statistics tracking
  - Win rate
  - Number of rounds
  - Gain
- 🧠 Optimal mixed strategy computation
- 🖥️ Interactive graphical interface

---

## 🧠 Concept

The game is based on a **zero-sum game**:

- The player shows a number of fingers.
- The AI also selects a number.
- If the **sum is even → Pair wins**
- If the **sum is odd → Odd wins**

Game theory predicts the **optimal strategy** where both players randomize their actions.

In equilibrium:

```
P(Pair) = 50%
P(Odd)  = 50%
Game Value V = 0
```

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Computer Vision
- Game Theory

---

## 📂 Project Structure

```
Projet_paire_impaire/
│
├── main.py
├── requirements.txt
├── README.md
│
├── assets/                 # Images used in the interface
│   ├── zero.png
│   ├── one.png
│   ├── two.png
│   ├── three.png
│   ├── four.png
│   └── five.png
│
└── src/
    ├── brain.py            # AI strategy and game logic
    ├── vision.py           # Hand detection using MediaPipe
    └── ui_manager.py       # Graphical interface management
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/walidehm/ai-pair-odd-game.git
cd ai-pair-odd-game
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 🎥 Demo

The player shows fingers in front of the camera, and the AI responds with its own move.

The system then computes:

- the sum
- the winner
- updated game statistics

---

## 📊 Educational Value

This project demonstrates the combination of:

- Computer Vision
- Human–AI Interaction
- Game Theory
- Real-time decision systems

---

## 🔮 Future Improvements

- Reinforcement Learning opponent
- More robust hand detection models
- Multiplayer mode
- Web version of the game

---

## 👨‍💻 Author

**Abdoul-Walid EL-HADJ MAMA**

- AI Student
- Web & AI Developer
- Member of the **AIOT Club – IFRI (Institut de Formation et de Recherche en Informatique)**
---

⭐ If you like this project, feel free to star the repository!