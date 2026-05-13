# Assignment 2 — Game Playing with Machine Learning (Othello)

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

## 📁 Project Structure

```
assignment2-GamePlay/
│
├── core.py                        # GameState, board logic
├── Agents.py                      # BaseAgent interface
├── Game.py                        # Game runner (UI + headless)
│
├── agent_minimax/                 # Hand-crafted Minimax agent (HC baseline)
│   └── Agents.py                  #   SearchAgent with alpha-beta + heuristic
│
├── ml/                            # ML agent (OthelloMLP + Negamax)
│   ├── mlp_agent.py               #   MLPAgent: policy/negamax modes
│   ├── mlp_model.py               #   OthelloMLP architecture definition
│   ├── dataset.py                 #   Dataset loader
│   ├── encoding.py                #   4-channel board encoding
│   └── evaluate.py                #   Evaluation helpers
│
├── train/                         # Training scripts
│   ├── train_local.py             #   Train on local machine
│   ├── train_only_v2.py           #   Lightweight training
│   ├── train_mlp_colab.ipynb      #   Colab notebook (GPU recommended)
│   └── train_only_v2.ipynb        #   Colab notebook v2
│
├── benchmark/                     # Benchmark scripts + results
│   ├── parallel_benchmark.py      #   Main benchmark (parallel, fast)
│   ├── fair_benchmark.py          #   Single-process benchmark (debug)
│   ├── benchmark_chart.py         #   Chart generation
│   └── *.png / *.json             #   Benchmark results
│
├── models/                        # Trained model checkpoints
│   └── best_mlp_model (7).pt      #   Best model (~5MB, ~1.3M params)
│
├── data/                          # Training datasets (.npz)
│   └── data_l10_merged_4ch.npz    #   Main dataset (~400K samples, 4-channel)
│
├── generate_parallel.py           # Generate expert training data
│
├── report/                        # LaTeX report
│   ├── OUTLINE.md                 #   Report outline + section assignments
│   ├── Background.tex
│   ├── section3.tex
│   ├── ModelArchitecture.tex
│   └── Result_Analysis.tex
│
├── requirements.txt               # Core dependencies
└── requirements-ml.txt            # ML dependencies (torch, numpy...)
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements-ml.txt
```

### 2. Play a game (GUI)
```bash
python Game.py
```

### 3. Run benchmark (ML vs Minimax)
```bash
# Run from project root
python benchmark/parallel_benchmark.py \
    --model "models/best_mlp_model (7).pt" \
    --games 100 --workers 8 --time 3.0
```

### 4. Generate training data
```bash
python generate_parallel.py --level 10 --games 500 --workers 8 --out data/data_new.npz
```

### 5. Train model
```bash
# Local
python train/train_only_v2.py

# Colab (recommended — GPU)
# Open train/train_mlp_colab.ipynb in Google Colab
```

---

## 🧠 Model Summary

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

## 📄 Report

See `report/OUTLINE.md` for the full outline and section assignments.
