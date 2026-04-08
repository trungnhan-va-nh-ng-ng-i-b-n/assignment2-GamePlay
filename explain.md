# Othello Framework - Updated Implementation Guide

This file documents the current codebase after integrating:
- Main Search Agent
- 10 difficulty levels
- Metrics-based evaluation
- Requirement check against Random Agent

## 1. Current Goal Coverage

The implementation now focuses on four core assignment goals:
1. Rule-correct gameplay through a shared game engine.
2. Search-only AI (no machine learning).
3. Multi-level skill behavior (difficulty 1-10).
4. Quantitative evaluation using strength and efficiency metrics.

## 2. Project Structure (What Each File Does)

### core.py
- Defines GameState and game rules.
- Provides legal move generation, move simulation, score calculation, and game-over checks.

### Agents.py
- BaseAgent: common interface get_move(game_state) -> (move, metrics).
- RandomAgent: baseline agent that selects a random legal move.
- SearchAgent: main AI using search with configurable difficulty.

### algorithms/Minimax.py
- Contains the full search engine used by SearchAgent.
- Implements:
  - Negamax
  - Alpha-Beta pruning
  - Iterative deepening
  - Heuristic evaluation with positional and tactical features
  - Difficulty mapping and controlled randomness for low levels

### statistic.py
- Headless simulation and benchmarking utilities.
- Supports batch matches, alternating first player, and aggregated metrics.
- Includes per-level benchmark for difficulty 1-10.

### Game.py
- Pygame UI runner.
- Uses the same agents and rule engine as headless mode.

## 3. Search Algorithm Design

The main search pipeline is:
1. Generate legal moves.
2. Map difficulty to a maximum depth.
3. Run iterative deepening from depth 1 up to max depth.
4. At each depth, evaluate moves via Negamax with Alpha-Beta pruning.
5. Stop at time budget and return the best completed result.

Why this design:
- Iterative deepening makes time-constrained play robust.
- Alpha-Beta reduces expanded nodes significantly.
- Negamax keeps implementation compact and consistent.

## 4. Heuristic Evaluation (What the Agent Optimizes)

The heuristic combines multiple factors:
- Positional weight matrix.
- Mobility difference (my legal moves minus opponent legal moves).
- Disc difference.
- Corner ownership bonus.
- Corner-neighbor risk penalty (X/C squares near empty corners).

The weighting is phase-aware:
- Opening: mobility and corner safety matter more.
- Midgame: balanced positional and tactical value.
- Endgame: disc advantage gets stronger weight.

## 5. Difficulty Levels 1-10

Difficulty is controlled by:
- DIFFICULTY_TO_DEPTH: higher levels search deeper.
- DIFFICULTY_TO_RANDOMNESS: lower levels inject random choices.

Extra behavior for high levels:
- Difficulty 9-10 uses stronger endgame handling.
- When empty squares are low, deeper exact endgame search is allowed.

## 6. Metrics Collected

Each move provides raw metrics such as:
- move_time
- nodes_explored
- pruned_branches
- depth_reached
- heuristic_score
- difficulty

Batch and level benchmarks aggregate these into:
- win_rate, draw_rate, loss_rate
- average score difference
- average time per move
- average time per game
- maximum time per move
- average expanded nodes
- average reached depth
- invalid move count

This supports both strength and computational efficiency analysis.

## 7. Evaluation Workflow

Two standard testing modes:

1. Requirement test (pass/fail)
- Run 10 games SearchAgent vs RandomAgent with alternating first player.
- Goal: verify 10/10 wins for the strongest level.

2. Research benchmark (analysis)
- Run all 10 levels against RandomAgent.
- Use 50-100 games per level for stable statistics.
- Compare strongest, fastest, and most efficient levels.

## 8. Current Status Summary

- Rule engine: integrated and stable.
- Random baseline agent: implemented.
- Search agent with 10 levels: implemented.
- Metrics pipeline: implemented.
- Search-only requirement: satisfied.
- Requirement-2 validation: tested with 10-game runs after search improvements.

## 9. Suggested Reading Order for New Members

1. core.py
2. Agents.py
3. algorithms/Minimax.py
4. statistic.py
5. Game.py

This order goes from rules -> agents -> search internals -> evaluation -> UI.

## 10. Notes for Report / Video

Recommended structure for presentation:
1. Game and state-space challenge.
2. Why search (not ML) fits this assignment.
3. Agent architecture and difficulty scaling.
4. Strength-vs-cost metrics.
5. Benchmark trends and selected best practical level.

This is the up-to-date technical baseline of the project.
