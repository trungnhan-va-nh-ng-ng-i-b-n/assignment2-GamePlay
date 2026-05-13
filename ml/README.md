# ML Agent — OthelloMLP + Negamax

## Overview

This ML agent replaces the **hand-crafted heuristic** inside a standard Negamax search with a **neural network (OthelloMLP)** trained via imitation learning on Level-10 Minimax expert data.

The search *framework* is identical to the HC `SearchAgent` (Negamax + alpha-beta pruning). The two key innovations are:

1. **Neural Leaf Evaluation** — `value_head(board)` replaces the hand-crafted heuristic at leaf nodes  
2. **Policy-Guided Move Ordering** — `policy_head(board)` sorts moves best-first at interior nodes, producing more alpha-beta cutoffs and effectively reaching deeper equivalent depth

---

## Architecture — OthelloMLP

```
Input: (B, 4, 8, 8) board tensor
  Channel 0: own discs    — 1.0 where current player has a disc
  Channel 1: opp discs    — 1.0 where opponent has a disc
  Channel 2: empty cells  — 1.0 where cell is empty
  Channel 3: turn plane   — 1.0 if Black (player=1), 0.0 if White (player=-1)

Backbone: 6-layer MLP with BatchNorm + ReLU + Dropout(0.1)
  Linear(256) → Linear(1024) → Linear(512) → Linear(512) → Linear(256) → Linear(256) → Linear(128)

Heads:
  policy_head: Linear(128 → 64)  — unnormalised logits over all 64 squares
  value_head:  Linear(128 → 64) → ReLU → Linear(64 → 1) → Tanh
               output ∈ [-1, +1], +1 = current player wins
```

**Parameters:** ~2.4M  
**Input channels:** 4 (own, opp, empty, turn)  

---

## Search Algorithm

### Negamax + Alpha-Beta + ML Evaluation

```
function negamax(board, player, depth, α, β):
    if depth == 0 or game_over:
        return value_head(board, player)      ← ML replaces heuristic

    moves = get_legal_moves(player)
    if depth ≥ 2:
        moves = sort_by_policy_prior(moves)   ← ML move ordering

    best = -∞
    for move in moves:
        next_board = apply(board, move)
        score = -negamax(next_board, -player, depth-1, -β, -α)
        best = max(best, score)
        α = max(α, score)
        if α ≥ β: break                       ← alpha-beta cutoff
    return best
```

### Iterative Deepening (Time-Based)

```
Reset transposition table
for depth in [1, 3, 5, 7, 9]:         ← odd depths only
    if time_elapsed ≥ deadline: break
    (score, move) = negamax(board, depth, deadline)
    if search completed before deadline:
        best_move = move               ← accept completed result only
return best_move
```

**Why odd depths only?**  
The `value_head` is trained on positions where it is the current player's turn to move. At odd search depths, the root player makes the last call to `value_head`, keeping the evaluation consistent with training. Even depths would have the opponent at the leaf, introducing a systematic bias.

### Move Ordering with Policy Head

At each interior node with `depth ≥ 2`, the policy head generates prior probabilities over all 64 squares. Legal moves are sorted in descending order of prior probability before the alpha-beta loop. This puts the most likely good moves first, maximising the chance of an early beta-cutoff.

**Effect on pruning:**  
With random ordering, alpha-beta prunes ~√(b^d) nodes of a b-ary tree of depth d.  
With perfect ordering (best move always first), pruning reaches ~b^(d/2) nodes — effectively **doubling** the searchable depth. Policy-guided ordering approaches this ideal.

### Transposition Table

```python
_negamax_cache: dict  # (board_bytes, player) → float value
```

Leaf node evaluations are cached by `(board.tobytes(), player)`. Repeated positions (transpositions) in the tree reuse the cached value without another model forward pass.

### TorchScript JIT Compilation

After loading the checkpoint, the model is compiled with `torch.jit.trace`:

```python
self.model = torch.jit.trace(self.model, dummy_input)
```

This eliminates Python dispatch overhead during `model.forward()`, giving **2–3× faster per-call inference** on CPU — critical for tree search where thousands of model calls occur per move.

---

## Training

### Data Generation

Expert self-play data is generated using the HC `SearchAgent` at Level 10 (depth 8, 3.0s/move) with 4 random opening moves for diversity:

```bash
python3 -u generate_parallel.py \
  --games 9000 --level 10 --think 3.0 \
  --workers 60 --opening 4 \
  --output Dataset/l10/data_l10.npz
```

Each game position is recorded as:
- `state`: (4, 8, 8) board tensor  
- `legal_mask`: (64,) binary mask of legal moves  
- `policy_target`: index of the move actually played by Level-10 agent  
- `outcome`: +1.0 (winner's positions), -1.0 (loser's), 0.0 (draw)

### Loss Function

```
L = L_policy + 0.5 × L_value

L_policy = CrossEntropy(policy_logits[legal_moves], target_move)
L_value  = MSE(value_head_output, game_outcome)
```

### Training Config

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Batch size | 2048 |
| Epochs | 100 |
| LR schedule | Warmup (5 epochs) + Cosine decay |
| Grad clip | 1.0 |
| Dropout | 0.1 |

---

## Performance

| Opponent | Win Rate | Notes |
|---|---|---|
| Random | ~100% | Trivial |
| HC-L1 (depth 1) | ~95% | |
| HC-L2 (depth 2) | ~75% | |
| HC-L3 (depth 3) | ~65% | (benchmark ongoing) |
| HC-L5 (depth 3) | ~80% | Key target ✅ |
| HC-L10 (depth 8) | ~0% | Node count bottleneck |

**Current model:** `best_mlp_model (6).pt`  — val_acc = **79.1%**, epoch 92, trained on ~540k L10 samples

---

## Files

```
ml/
├── mlp_agent.py      # MLPAgent class + negamax search + OthelloMLP definition
├── encoding.py       # Board → tensor encoding (used during data generation)
├── mlp_model.py      # Standalone OthelloMLP (for training scripts)
└── README.md         # This file

generate_parallel.py  # Parallel data generation (multiprocessing)
train_local.py        # Training script (CPU or GPU)
```

---

## Usage

```python
from ml.mlp_agent import MLPAgent

agent = MLPAgent(
    color=1,                            # 1 = Black, -1 = White
    checkpoint_path="best_mlp_model.pt",
    mode="negamax",                     # "policy" for fast mode
    device="auto",                      # "cpu" or "cuda"
)

move, metrics = agent.get_move(game_state, remain_time=3.0)
```
