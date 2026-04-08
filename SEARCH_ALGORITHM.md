# Search Algorithm Explanation for Othello

This document explains the search algorithm currently used in the project for the following assignment scope:
- Main agent
- 10 difficulty levels
- Metrics for agent capability evaluation

## 1. AI Problem in Othello

Othello is a two-player adversarial game with perfect information and no randomness.
Therefore, move selection is a search problem on a game tree:
- Node: a board state
- Edge: a legal move
- Goal: choose the best move for the agent

## 2. Algorithms Used

The project uses the following combination:
- Negamax (compact form of Minimax)
- Alpha-Beta Pruning (search-space reduction)
- Iterative Deepening (progressively deeper search)
- Heuristic Evaluation (state scoring function)

Code locations:
- select_move: algorithms/Minimax.py
- Agent integration: Agents.py

## 3. High-Level Search Flow

When it is the agent's turn:
1. Get all legal moves.
2. Map difficulty (1-10) to max_depth.
3. Run iterative deepening from depth = 1 to max_depth.
4. At each depth, evaluate moves using Negamax + Alpha-Beta.
5. Return the best move found within the time budget.

If time runs out at a deeper depth, the agent still keeps the best result from the last fully completed depth.

## 4. Negamax (Equivalent to Minimax)

Core idea:
- The opponent's value is the negative of your value.
- A single maximizing routine is enough (instead of separate max/min functions).

Formula:

```text
value(state, player) = max_{move} ( -value(next_state, -player) )
```

Terminal conditions:
- depth = 0
- or game is over

At terminal nodes, the algorithm returns heuristic_score.

## 5. Alpha-Beta Pruning

Purpose: skip branches that cannot influence the final decision.

Variables:
- alpha: best guaranteed value for the maximizing side
- beta: best guaranteed value for the minimizing side

If alpha >= beta, remaining sibling branches are pruned.

Benefits:
- Fewer explored nodes
- Deeper effective search under the same time limit

## 6. Iterative Deepening and Time Control

Instead of jumping directly to a large depth, the agent searches progressively:
- depth 1 -> depth 2 -> depth 3 -> ...

Each loop checks a deadline. If time is nearly exhausted:
- Stop searching
- Return the best move from the latest completed depth

This approach is more robust for per-move time constraints.

## 7. Heuristic Evaluation

The heuristic combines three components:
- Positional weights (board-square value map)
- Mobility (my legal moves minus opponent legal moves)
- Disc difference (my discs minus opponent discs)

General form:

```text
score = w_pos * positional + w_mob * mobility + w_disc * disc_diff
```

In this project, the positional map heavily rewards:
- Corners (very strong)
- Penalizes dangerous squares near corners

## 8. Ten Difficulty Levels

Difficulty is controlled by:
- DIFFICULTY_TO_DEPTH: higher level -> higher max_depth
- DIFFICULTY_TO_RANDOMNESS: lower level -> higher chance of random move

Interpretation:
- Low levels: faster, shallower, easier to beat
- High levels: slower, deeper, stronger

## 9. Agent Evaluation Metrics

Each move returns metrics; benchmark aggregates them across many games:
- move_time: time to compute one move
- nodes_explored: visited search nodes
- pruned_branches: branches cut by alpha-beta
- depth_reached: deepest fully reached depth
- heuristic_score: score of the selected move
- difficulty: current level

In benchmarking, these are averaged for level-to-level comparison.

## 10. How to Read Benchmark Results

For SearchAgent vs RandomAgent:
- Higher levels should generally show better win rate
- avg_nodes and avg_depth should usually increase with level
- avg_move_time should usually increase with level
- invalid_moves should be 0

If higher levels are not consistently stronger:
- Improve heuristic quality
- Retune depth/randomness mapping
- Improve move ordering quality

## 11. Current Limitations and Future Improvements

Current limitations:
- Heuristic still lacks advanced features (stable discs, frontier discs, parity)
- No transposition table yet

Potential improvements:
- Add Zobrist hashing + transposition table
- Use phase-dependent heuristic (opening/midgame/endgame)
- Auto-tune weights through self-play

---
This document is sufficient for report/video explanation of Search Agent + 10 difficulty levels + metrics.