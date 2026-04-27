from __future__ import annotations

import numpy as np

from algorithms.Minimax import get_valid_moves

BOARD_SIZE = 8
NUM_MOVES = BOARD_SIZE * BOARD_SIZE


def move_to_index(move: tuple[int, int] | None) -> int:
    if move is None:
        return -1
    row, col = move
    return int(row) * BOARD_SIZE + int(col)


def index_to_move(index: int) -> tuple[int, int]:
    index = int(index)
    if index < 0 or index >= NUM_MOVES:
        raise ValueError(f"Move index out of range: {index}")
    return index // BOARD_SIZE, index % BOARD_SIZE


def legal_moves_mask(board: list[list[int]], player_to_move: int) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=np.float32)
    valid_moves = get_valid_moves(board, int(player_to_move))
    for row, col in valid_moves:
        mask[row * BOARD_SIZE + col] = 1.0
    return mask


def board_to_tensor(board: list[list[int]], player_to_move: int) -> np.ndarray:
    board_np = np.asarray(board, dtype=np.int8)
    if board_np.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected board shape (8, 8), got {board_np.shape}")

    player = int(player_to_move)

    own_discs = (board_np == player).astype(np.float32)
    opp_discs = (board_np == -player).astype(np.float32)
    empty_cells = (board_np == 0).astype(np.float32)
    turn_plane = np.full((BOARD_SIZE, BOARD_SIZE), 1.0 if player == 1 else 0.0, dtype=np.float32)

    # 4 x 8 x 8 tensor: own, opponent, empty, and side-to-move plane.
    return np.stack((own_discs, opp_discs, empty_cells, turn_plane), axis=0)
