"""
convert_hf_dataset.py — Convert tdooms/othello game records to training data.

Each game record is a list of 64 move indices (0-63, -1 = padding).
We replay each game, and at every step record:
    - board state  (3, 8, 8)  [my_pieces, opp_pieces, valid_moves_mask_2d]
    - legal mask   (64,)
    - move index   (int)       ← the next move (label)
    - outcome      (float)     ← +1 / -1 / 0 from current player's perspective

Usage:
    python -m ml.convert_hf_dataset --input othello/data --output ml/data/hf_train.npz --max-games 200000
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Minimal Othello engine (standalone, no dependency on core.py)
# ---------------------------------------------------------------------------

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]


def _initial_board() -> np.ndarray:
    """Return 8x8 int8 board. 1=Black, -1=White, 0=empty."""
    board = np.zeros((8, 8), dtype=np.int8)
    board[3][3] = -1
    board[3][4] = 1
    board[4][3] = 1
    board[4][4] = -1
    return board


def _apply_move(board: np.ndarray, move_idx: int, player: int) -> np.ndarray:
    """Apply move and return new board."""
    new_board = board.copy()
    r, c = move_idx // 8, move_idx % 8
    new_board[r][c] = player

    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        to_flip = []
        while 0 <= nr < 8 and 0 <= nc < 8:
            if new_board[nr][nc] == 0:
                break
            elif new_board[nr][nc] == player:
                for fr, fc in to_flip:
                    new_board[fr][fc] = player
                break
            else:
                to_flip.append((nr, nc))
                nr += dr
                nc += dc
    return new_board


def _get_legal_mask(board: np.ndarray, player: int) -> np.ndarray:
    """Return (64,) float32 mask of legal moves."""
    mask = np.zeros(64, dtype=np.float32)
    for idx in range(64):
        r, c = idx // 8, idx % 8
        if board[r][c] != 0:
            continue
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            found = False
            while 0 <= nr < 8 and 0 <= nc < 8:
                if board[nr][nc] == 0:
                    break
                elif board[nr][nc] == player:
                    if found:
                        mask[idx] = 1.0
                    break
                else:
                    found = True
                    nr += dr
                    nc += dc
    return mask


def _board_to_tensor(board: np.ndarray, player: int) -> np.ndarray:
    """Convert board to (3, 8, 8) tensor: [my, opp, legal_2d]."""
    my = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    legal = _get_legal_mask(board, player).reshape(8, 8)
    return np.stack([my, opp, legal], axis=0)


def _winner(board: np.ndarray) -> int:
    total = int(board.sum())
    if total > 0:
        return 1
    elif total < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_game(moves: list[int]):
    """Replay one game and yield (state, legal_mask, policy_target, player) per step."""
    board = _initial_board()
    player = 1  # Black goes first

    samples = []
    for move_idx in moves:
        if move_idx < 0:
            break

        legal_mask = _get_legal_mask(board, player)

        # Only record if the move is actually legal
        if legal_mask[move_idx] > 0:
            state = _board_to_tensor(board, player)
            samples.append({
                "state": state,            # (3, 8, 8)
                "legal_mask": legal_mask,   # (64,)
                "policy_target": move_idx,
                "player": player,
            })

        board = _apply_move(board, move_idx, player)
        player = -player

        # Handle pass: if next player has no moves, switch back
        next_legal = _get_legal_mask(board, player)
        if next_legal.sum() == 0:
            player = -player

    # Determine winner and assign outcomes
    winner = _winner(board)
    for s in samples:
        if winner == 0:
            s["outcome"] = 0.0
        elif winner == s["player"]:
            s["outcome"] = 1.0
        else:
            s["outcome"] = -1.0

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert HF othello parquet to training .npz")
    parser.add_argument("--input", type=str, default="othello/data",
                        help="Directory containing .parquet files")
    parser.add_argument("--output", type=str, default="ml/data/hf_train.npz",
                        help="Output .npz file")
    parser.add_argument("--max-games", type=int, default=100_000,
                        help="Max number of games to convert")
    args = parser.parse_args()

    # Load parquet files
    parquet_files = sorted(
        [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".parquet")]
    )
    print(f"Found {len(parquet_files)} parquet files in {args.input}")

    all_states = []
    all_masks = []
    all_targets = []
    all_outcomes = []

    game_count = 0
    t0 = time.time()

    for pf in parquet_files:
        if game_count >= args.max_games:
            break

        print(f"Reading {os.path.basename(pf)}...")
        table = pq.read_table(pf)
        games_col = table.column("games").to_pylist()

        for game_moves in games_col:
            if game_count >= args.max_games:
                break

            samples = convert_game(game_moves)
            for s in samples:
                all_states.append(s["state"])
                all_masks.append(s["legal_mask"])
                all_targets.append(s["policy_target"])
                all_outcomes.append(s["outcome"])

            game_count += 1
            if game_count % 10_000 == 0:
                elapsed = time.time() - t0
                rate = game_count / elapsed
                print(f"  [{game_count:>7,}/{args.max_games:,}] "
                      f"samples={len(all_states):>9,}  "
                      f"speed={rate:.0f} games/s")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    states = np.array(all_states, dtype=np.float32)
    legal_masks = np.array(all_masks, dtype=np.float32)
    policy_targets = np.array(all_targets, dtype=np.int64)
    outcomes = np.array(all_outcomes, dtype=np.float32)

    np.savez_compressed(
        args.output,
        states=states,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        outcomes=outcomes,
    )

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output) / 1e6
    print(f"\n{'='*60}")
    print(f" Done in {elapsed:.1f}s")
    print(f" Games:   {game_count:,}")
    print(f" Samples: {len(all_states):,}")
    print(f" Output:  {args.output} ({file_size:.1f} MB)")
    print(f" Win/Loss/Draw: "
          f"{np.mean(outcomes==1):.1%} / {np.mean(outcomes==-1):.1%} / {np.mean(outcomes==0):.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
