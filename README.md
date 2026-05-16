# Assignment 2 — Game Playing with Search (Othello)

**Course**: Introduction to Artificial Intelligence — Academic Year 2025–2026  
**Group 3** | Class CC01 | Advisor: Dr. Trần Hồng Tài

| Student | MSSV |
|---|---|
| Nguyễn Trung Nhân | 2352852 |
| Nguyễn Vĩnh Phú | 2352919 |
| Huỳnh Kim Nghĩa | 2352805 |
| Huỳnh Vương Khang | 2350011 |
| Trần Ngọc Bảo | 2352109 |

---

## Project Structure

```
assignment2-GamePlay/
│
├── core.py                        # GameState, board logic
├── Agents.py                      # BaseAgent interface
├── Game.py                        # Game runner (UI + headless)
├── Game_MLP.py                    # Assignment 3 GUI (MLP vs Random/Search)
├── statistic.py                   # Headless batch simulation helpers
│
├── agent_minimax/                 # Hand-crafted Minimax agent (HC baseline)
│   └── Agents.py                  #   SearchAgent with alpha-beta + heuristic
│
├── ml/                            # Assignment 3 MLP agent (OthelloMLP + Negamax)
│   ├── mlp_agent.py               #   MLPAgent: policy/negamax modes
│   ├── mlp_model.py               #   OthelloMLP architecture definition
│   ├── dataset.py                 #   Dataset loader
│   ├── encoding.py                #   4-channel board encoding
│   └── evaluate.py                #   Evaluation helpers
│
├── requirements.txt               # Core dependencies
└── requirements-ml.txt            # ML dependencies (torch, numpy...)
```

---

## Quick Start (Assignment 2)

### 1. Install dependencies
```bash
pip install -r requirements-ml.txt
```

### 2. Play a game (GUI)
```bash
python Game.py
```

This assignment focuses on SearchAgent vs RandomAgent/SearchAgent using the hand-crafted minimax (no MLP).


## Assignment 3 — MLP (Hybrid ML + Search)

### What we built
- A standalone GUI to play MLPAgent against RandomAgent or SearchAgent.
- The MLPAgent uses a trained MLP (policy + value heads) with Negamax + alpha-beta search.

### How to run
```bash
python Game_MLP.py
```

### Requirements
- Place the trained checkpoint at `ml/best_mlp_model.pt`.
- The GUI lets you choose opponent type, search level, MLP side, and time budget.

### Optional: Headless evaluation helper
```bash
python -m ml.evaluate --checkpoint ml/best_mlp_model.pt --opponent random --num-games 20
```

### Model summary

| | Value |
|---|---|
| Architecture | OthelloMLP — 6-layer FC, dual-head |
| Input | (4, 8, 8) board tensor |
| Output | policy logits (64) + value scalar |
| Parameters | ~1.3M |
| Checkpoint size | ~5 MB |
| Search | Negamax + alpha-beta + iterative deepening |
| Time budget | 3.0 s/move |

---

### Core files kept in the repo
- Agents.py, core.py, algorithms/
- Game.py, Game_MLP.py
- ml/ (mlp_agent.py, mlp_model.py, encoding.py, dataset.py, evaluate.py)
- requirements.txt, requirements-ml.txt
