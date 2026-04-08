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
- Adjust Agent 1 level (1-10) and Agent 2 level (1-10) with the `+`/`-` buttons.
- Click `Start Match` to run SearchAgent vs SearchAgent using selected levels.
- Keyboard shortcuts: `Q/W` for Agent 1, `A/S` for Agent 2, `Enter` to start.

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
