# ML Othello Agent — Technical Overview

## 1. Game: Othello (Reversi)

Othello is a two-player zero-sum board game played on an **8×8 grid**.

**Rules:**
- Players alternate placing discs (Black / White)
- A move is valid if it **flanks** ≥1 opponent discs in any direction (horizontal, vertical, diagonal)
- All flanked discs are **flipped** to the current player's color
- Game ends when neither player has a legal move; the player with **more discs wins**

**Why it's challenging:**
- State space: ~10^28 legal positions
- Long-horizon strategy: early positional play determines late-game outcomes
- Mobility and corner control are critical — a single corner capture can shift the entire game

---

## 2. Dataset Generation

### 2.1 Teacher Agent
Expert data is generated using the project's **SearchAgent** (Minimax + Alpha-Beta Pruning) as a teacher:

| Dataset | Teacher Level | Think Time | Purpose |
|---------|--------------|------------|---------|
| `data_best_*.npz` | L7 | **3.0s / game** | High-quality labels |
| `data_fast_*.npz` | L6–L7 | **0.2s / game** | Large-scale coverage |

### 2.2 Parallel Generation (48-Core Server)
Games were generated in parallel on a **48-core server** using Python `multiprocessing`:

```
generate_parallel.py
  └── Pool(48 workers)
        ├── Worker 1: generates games [seed=1, ...]
        ├── Worker 2: generates games [seed=2, ...]
        ├── ...
        └── Worker 48: generates games [seed=48, ...]
```

Each worker runs independently and saves its own `.npz` batch — no synchronization overhead.

### 2.3 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total `.npz` files | **38 files** (20 best + 18 fast) |
| Total games played | **24,948 games** |
| Total board positions | **~1.49M samples** |
| Teacher levels | L6, L7 |
| Think times | 0.2s, 3.0s |
| Opening randomization | 4 random moves per game |

### 2.4 Sample Attributes
Each sample in the `.npz` files contains:

| Attribute | Shape | Type | Description |
|-----------|-------|------|-------------|
| `states` | `(N, 4, 8, 8)` | float32 | Board encoding: **4 channels** per position |
| `legal_masks` | `(N, 64)` | float32 | Binary mask of valid moves (1=legal) |
| `policy_targets` | `(N,)` | int64 | Index (0–63) of the move chosen by teacher |
| `outcomes` | `(N,)` | float32 | Game result from current player's view (+1=win, -1=loss) |
| `players` | `(N,)` | int8 | Current player (+1=Black, -1=White) |

### 2.5 Board Encoding (4-Channel)
Each board state is encoded as a `(4, 8, 8)` tensor:

```
Channel 0 — own:        1.0 if current player's disc, else 0.0
Channel 1 — opp:        1.0 if opponent's disc, else 0.0
Channel 2 — empty:      1.0 if empty cell, else 0.0
Channel 3 — turn_plane: 1.0 everywhere if Black to move, 0.0 if White to move
```

This encoding is **player-relative** and matches the server-side format exactly.

---

## 3. Algorithms

### 3.1 Minimax & Negamax

**Minimax** is the foundation of adversarial game search. It assumes both players play optimally:
- **MAX player** (us) chooses the move with the **highest** score
- **MIN player** (opponent) chooses the move with the **lowest** score

**Negamax** is a cleaner implementation of Minimax exploiting the zero-sum property:

```
negamax(board, player, depth):
    if depth == 0 or game_over:
        return evaluate(board, player)   ← leaf node evaluation

    best = -∞
    for each legal move:
        apply move
        score = -negamax(board, -player, depth - 1)  ← negate (zero-sum)
        best = max(best, score)
        undo move
    return best
```

In our ML agent, the `evaluate()` function is replaced by the **neural network value head** instead of a hand-crafted formula.

---

### 3.2 Alpha-Beta Pruning

Alpha-Beta Pruning cuts branches that cannot affect the final decision, reducing the search space from **O(b^d)** to **O(b^(d/2))** in the best case:

```
negamax_ab(board, player, depth, α, β):
    if depth == 0: return evaluate(board, player)

    for each legal move:
        score = -negamax_ab(board, -player, depth-1, -β, -α)
        α = max(α, score)
        if α ≥ β:
            break          ← PRUNE: this branch can never be chosen
    return α
```

| | Without pruning | With Alpha-Beta |
|--|----------------|-----------------|
| Nodes at depth 6 | ~10^6 | ~10^3 (best case) |
| Branching factor | b | ~√b |

**Effectiveness depends heavily on move ordering** — better moves explored first → more cutoffs.

---

### 3.3 Supervised Imitation Learning

The neural network is trained via **imitation learning** — learning to replicate the decisions of an expert teacher (SearchAgent L7):

```
For each board position in dataset:
    teacher_move  = move chosen by SearchAgent-L7 (policy target)
    game_outcome  = +1 if teacher's color won, -1 if lost (value target)

Loss = 0.7 × CrossEntropy(policy_head(board), teacher_move)
     + 0.3 × MSE(value_head(board), game_outcome)
```

- **Policy loss** teaches *which move to make*
- **Value loss** teaches *how good this position is*

This is the same dual-head training approach used in **AlphaZero**, but with supervised (not self-play) data.

---

### 3.4 Policy-Guided Move Ordering (Key Innovation)

**Problem:** Neural net inference takes ~0.5ms/node vs ~0.001ms for hand-crafted heuristics. Without optimization, ML agent is capped at depth 4 (~5s/move).

**Insight:** Alpha-Beta pruning is most effective when the **best moves are explored first**. The policy head naturally ranks moves by quality.

**Solution:**
```python
# At each non-leaf node, use policy head to sort moves before searching
policy_logits = model.policy_head(board_tensor)       # one forward pass
legal_moves   = sorted(legal_moves,
                        key=lambda m: -policy_logits[m[0]*8 + m[1]])  # best-first

# Now search in sorted order → far more alpha-beta cutoffs
for move in legal_moves:
    score = -negamax_ab(apply(board, move), depth-1, -β, -α)
    ...
```

**Result:**

| Depth | Without ordering | With policy ordering | Speedup |
|-------|-----------------|---------------------|---------|
| 4 | ~5,000ms | ~70ms | **70×** |
| 5 | ~50,000ms | ~290ms | **170×** |
| 6 | timeout | ~830ms | **feasible!** |
| 7 | timeout | ~2,600ms | **feasible!** |
| 8 | timeout | ~10,000ms | **feasible!** |

This single optimization unlocks **depth 6–8** search, making the ML agent competitive at higher difficulty levels.

---

## 4. ML Pipeline

### 4.1 Overview

```
[Expert Games] ──► [4-Channel Encoding] ──► [OthelloMLP Training]
                                                      │
                                                      ▼
                               [Negamax Search + Alpha-Beta Pruning]
                                      │                    │
                              Policy head:           Value head:
                           sort moves best-first    evaluate leaf nodes
                                      │                    │
                                      └────────┬───────────┘
                                               ▼
                                        [Move Decision]
```

### 4.2 Model Architecture — OthelloMLP

```
Input: (4, 8, 8) → flatten → 256 features
          │
          ▼
    Linear(256 → 1024) + BatchNorm + ReLU + Dropout(0.1)
    Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.1)
    Linear(512  → 512) + BatchNorm + ReLU + Dropout(0.1)
    Linear(512  → 256) + BatchNorm + ReLU + Dropout(0.1)
    Linear(256  → 256) + BatchNorm + ReLU + Dropout(0.1)
          │
    ┌─────┴──────┐
    ▼            ▼
Policy head   Value head
→ 128 → 64   → 64 → 1 (Tanh)
(move logits) (position score ∈ [-1, 1])
```

**Total parameters:** ~3.2M

### 4.3 Training

| Hyperparameter | Value |
|----------------|-------|
| Framework | PyTorch (Colab T4 GPU) |
| Epochs | 100 |
| Batch size | 512 |
| Optimizer | Adam |
| Learning rate | 3e-4 with 5-epoch warmup |
| Loss | `0.7 × CrossEntropy(policy) + 0.3 × MSE(value)` |
| Best val_acc | **73.7%** (saved at epoch 93) |

### 4.4 Inference — Negamax + Policy Move Ordering

**Standard ML+Minimax issue:** Neural net inference is ~500× slower than hand-crafted heuristics per node (0.5ms vs 0.001ms), making deep search infeasible.

**Solution — Policy-Guided Move Ordering:**
```python
# Before searching children, sort moves using policy head
policy_probs = model.policy_head(board)
legal_moves  = sorted(legal_moves, key=lambda m: -policy_probs[m])  # best-first

# Alpha-beta now prunes far more branches → ~100× fewer nodes evaluated
```

**Impact:**

| Mode | Depth | Time/move | Nodes |
|------|-------|-----------|-------|
| Sequential (no ordering) | 4 | ~5s | ~50,000 |
| **+ Policy ordering** | **6** | **~0.8s** | **~1,400** |
| **+ Policy ordering** | **7** | **~2.6s** | **~4,700** |
| **+ Policy ordering** | **8** | **~10s** | **~16,000** |

### 4.5 Skill Levels

10 difficulty levels controlled by search depth:

| Level | Search Depth | Approx. time/move |
|-------|-------------|-------------------|
| L1 | 1 | ~0.01s |
| L2 | 2 | ~0.03s |
| L3 | 3 | ~0.1s |
| L4 | 4 | ~0.07s |
| L5 | 5 | ~0.3s |
| L6 | 6 | ~0.8s |
| L7 | 7 | ~2.6s |
| L8 | 8 | ~10s |

---

## 5. Benchmark Results

Fair benchmark: ML agent at depth D vs. SearchAgent at the same effective depth.

| Matchup | ML depth | ML Win Rate | Result |
|---------|----------|-------------|--------|
| vs Random (Policy) | — | **90%** | ✅ |
| vs Random (1-ply) | — | **100%** | ✅ |
| vs HC-L1 | 1 | **80%** | ✅ |
| vs HC-L2 | 2 | **70%** | ✅ |
| vs HC-L3 | 3 | **90%** | ✅ 🔥 |
| vs HC-L4 | 4 | **80%** | ✅ 🔥 |
| vs HC-L5 | 5 | **70%** | ✅ |
| vs HC-L6 | 6 | **60%** | ✅ |

**8 / 8 matchups won.** The ML-augmented Negamax outperforms the hand-crafted heuristic at every tested level under fair computational conditions.
