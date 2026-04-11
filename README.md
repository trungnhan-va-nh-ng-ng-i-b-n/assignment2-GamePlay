# Othello (Reversi) agents

This project aims to create agents playing Othello. It includes an integrated Pygame GUI.

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv

# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. How to Run

To launch the graphical user interface (GUI):

```bash
python3 Game.py
```

When the GUI opens, a setup menu appears first:
- Adjust Search level (1-10) with the `+`/`-` buttons.
- Toggle Search side to choose whether SearchAgent is Player 1 (Black) or Player 2 (White).
- Adjust `Total Time (s)` with the `+`/`-` buttons.
- Click `Start Match` to run SearchAgent vs RandomAgent.
- Keyboard shortcuts: `Q/W` for Search level, `A/S` to toggle Search side, `Z/X` for total time, `Enter` to start.

During the match:
- Toggle run mode with `Mode: AUTO` / `Mode: STEP` button (or `Tab`).
- In step mode, click `Next Step` (or press `Space`) to execute exactly one move.
- Adjust auto speed with `-` / `+` speed buttons (`+` = faster, `-` = slower).

After game over:
- The GUI stays open on the final board state (no auto-close).
- Use `New Match` to go back to setup, or `Exit` to close.
- Press `R` for New Match or `Esc` to exit.

To run the headless batch simulation and quickly verify the AI's win rate:
```bash
python3 statistic.py
```
