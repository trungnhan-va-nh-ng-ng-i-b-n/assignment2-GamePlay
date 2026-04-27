from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from Agents import SearchAgent
from core import GameState
from ml.encoding import board_to_tensor, legal_moves_mask, move_to_index


def _winner_from_state(game_state: GameState) -> int:
    black_score, white_score = game_state.calculate_score()
    if black_score > white_score:
        return 1
    if white_score > black_score:
        return -1
    return 0


def _teacher_for(color: int, level: int, think_time: float) -> SearchAgent:
    return SearchAgent(
        color=color,
        difficulty=level,
        name=f"Teacher-{('Black' if color == 1 else 'White')}-L{level}",
        remain_time=think_time,
    )


def _play_teacher_game(
    rng: random.Random,
    black_level: int,
    white_level: int,
    think_time: float,
    random_opening_moves: int,
):
    game_state = GameState()
    teacher_black = _teacher_for(color=1, level=black_level, think_time=think_time)
    teacher_white = _teacher_for(color=-1, level=white_level, think_time=think_time)

    samples = []
    invalid_fallbacks = 0
    played_moves = 0

    while not game_state.is_game_over():
        turn = game_state.current_turn
        legal_moves = game_state.get_legal_moves(turn)

        if not legal_moves:
            game_state = game_state.apply_move(None, turn)
            continue

        if played_moves < random_opening_moves:
            move = rng.choice(legal_moves)
        else:
            teacher = teacher_black if turn == 1 else teacher_white
            move, _ = teacher.get_move(game_state.get_board_copy())
            if move not in legal_moves:
                invalid_fallbacks += 1
                move = rng.choice(legal_moves)

        samples.append(
            {
                "state": board_to_tensor(game_state.board, turn),
                "legal_mask": legal_moves_mask(game_state.board, turn),
                "policy_target": move_to_index(move),
                "player_to_move": int(turn),
            }
        )

        game_state = game_state.apply_move(move, turn)
        played_moves += 1

    winner = _winner_from_state(game_state)
    for sample in samples:
        player_to_move = sample["player_to_move"]
        if winner == 0:
            sample["outcome"] = 0.0
        elif winner == player_to_move:
            sample["outcome"] = 1.0
        else:
            sample["outcome"] = -1.0

    return samples, winner, invalid_fallbacks


def parse_args():
    parser = argparse.ArgumentParser(description="Generate self-play teacher data for Othello ML agent")
    parser.add_argument("--output", type=str, default="ml/data/selfplay_teacher.npz", help="Output .npz dataset path")
    parser.add_argument("--num-games", type=int, default=200, help="Number of self-play games to generate")
    parser.add_argument("--min-level", type=int, default=6, help="Minimum teacher difficulty")
    parser.add_argument("--max-level", type=int, default=10, help="Maximum teacher difficulty")
    parser.add_argument("--think-time", type=float, default=0.8, help="Per-move time budget for teacher")
    parser.add_argument(
        "--random-opening-moves",
        type=int,
        default=4,
        help="Number of opening moves sampled randomly before teacher control",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-every", type=int, default=10, help="Progress print interval")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.min_level < 1 or args.max_level > 10 or args.min_level > args.max_level:
        raise ValueError("Teacher levels must satisfy 1 <= min-level <= max-level <= 10")
    if args.random_opening_moves < 0:
        raise ValueError("--random-opening-moves must be >= 0")

    rng = random.Random(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    wins = {1: 0, -1: 0, 0: 0}
    invalid_fallbacks = 0

    started_at = time.perf_counter()

    for game_index in range(1, args.num_games + 1):
        black_level = rng.randint(args.min_level, args.max_level)
        white_level = rng.randint(args.min_level, args.max_level)

        game_samples, winner, game_invalid = _play_teacher_game(
            rng=rng,
            black_level=black_level,
            white_level=white_level,
            think_time=args.think_time,
            random_opening_moves=args.random_opening_moves,
        )

        all_samples.extend(game_samples)
        wins[winner] += 1
        invalid_fallbacks += game_invalid

        if game_index % max(1, args.report_every) == 0 or game_index == args.num_games:
            print(
                f"[{game_index}/{args.num_games}] samples={len(all_samples)} "
                f"black_wins={wins[1]} white_wins={wins[-1]} draws={wins[0]}"
            )

    if not all_samples:
        raise RuntimeError("No samples were generated")

    states = np.stack([sample["state"] for sample in all_samples], axis=0).astype(np.float32)
    legal_masks = np.stack([sample["legal_mask"] for sample in all_samples], axis=0).astype(np.float32)
    policy_targets = np.asarray([sample["policy_target"] for sample in all_samples], dtype=np.int64)
    outcomes = np.asarray([sample["outcome"] for sample in all_samples], dtype=np.float32)
    players = np.asarray([sample["player_to_move"] for sample in all_samples], dtype=np.int8)

    np.savez_compressed(
        output_path,
        states=states,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        outcomes=outcomes,
        players=players,
    )

    elapsed = time.perf_counter() - started_at
    metadata = {
        "num_games": args.num_games,
        "num_samples": int(len(all_samples)),
        "wins": {"black": wins[1], "white": wins[-1], "draw": wins[0]},
        "invalid_fallbacks": int(invalid_fallbacks),
        "teacher_level_range": [args.min_level, args.max_level],
        "think_time": float(args.think_time),
        "random_opening_moves": int(args.random_opening_moves),
        "seed": int(args.seed),
        "generation_seconds": float(elapsed),
        "output": str(output_path),
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDataset generation complete")
    print(f"Output dataset: {output_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
