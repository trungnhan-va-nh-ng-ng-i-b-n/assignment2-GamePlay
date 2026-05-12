"""
generate_parallel.py — Generate Othello dataset song song với multiprocessing.

Dùng N worker processes, mỗi process chạy K games → merge lại.

Usage:
    python3 -u generate_parallel.py --games 2000 --level 7 --think 0.5 --workers 7
"""
import argparse, os, time, random
import numpy as np
import multiprocessing as mp
from functools import partial

# ── Worker function (chạy trong subprocess) ────────────────────────────
def _run_games(args):
    """Chạy một batch games, return (states, masks, targets, outcomes)."""
    worker_id, n_games, level, think_time, seed, random_opening = args

    # Import trong subprocess (tránh fork issues)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Agents import SearchAgent
    from core import GameState
    from ml.encoding import board_to_tensor, legal_moves_mask, move_to_index

    rng = random.Random(seed)
    np.random.seed(seed)

    all_states, all_masks, all_targets, all_outcomes = [], [], [], []

    for g in range(n_games):
        try:
            gs = GameState()
            b_agent = SearchAgent(color=1,  difficulty=level, remain_time=think_time)
            w_agent = SearchAgent(color=-1, difficulty=level, remain_time=think_time)
            turn = 1
            samples = []
            moves_played = 0

            for _ in range(130):
                legal = gs.get_legal_moves(turn)
                opp   = gs.get_legal_moves(-turn)
                if not legal and not opp:
                    break
                if not legal:
                    turn = -turn
                    continue

                # Random opening
                if moves_played < random_opening:
                    move = rng.choice(legal)
                else:
                    agent = b_agent if turn == 1 else w_agent
                    move, _ = agent.get_move(gs)
                    if move is None or move not in legal:
                        move = rng.choice(legal)

                # Record this position
                state_np = board_to_tensor(gs.board, turn)   # (4,8,8) from encoding.py
                # Convert to our 3-channel format (own, opp, legal_mask)
                board_arr = np.array(gs.board, dtype=np.int8)
                own  = (board_arr == turn).astype(np.float32)
                opp_ch = (board_arr == -turn).astype(np.float32)
                lmask = legal_moves_mask(gs.board, turn).reshape(8, 8)
                state_3ch = np.stack([own, opp_ch, lmask])   # (3,8,8)

                mask_flat = legal_moves_mask(gs.board, turn)  # (64,)
                move_idx  = move_to_index(move)

                # Only save if move is legal
                if mask_flat[move_idx] > 0:
                    samples.append({
                        "state"  : state_3ch,
                        "mask"   : mask_flat,
                        "target" : move_idx,
                        "player" : turn,
                    })

                gs = gs.apply_move(move, turn)
                turn = -turn
                moves_played += 1

            # Determine winner
            b_score, w_score = gs.calculate_score()
            if b_score > w_score:   winner = 1
            elif w_score > b_score: winner = -1
            else:                   winner = 0

            # Assign outcomes
            for s in samples:
                if winner == 0:
                    outcome = 0.0
                elif winner == s["player"]:
                    outcome = 1.0
                else:
                    outcome = -1.0
                all_states.append(s["state"])
                all_masks.append(s["mask"])
                all_targets.append(s["target"])
                all_outcomes.append(outcome)

        except Exception as e:
            pass  # Skip failed games silently

        if (g + 1) % 50 == 0:
            print(f"  Worker {worker_id}: {g+1}/{n_games} games done", flush=True)

    return (
        np.array(all_states,  dtype=np.float32),
        np.array(all_masks,   dtype=np.float32),
        np.array(all_targets, dtype=np.int64),
        np.array(all_outcomes,dtype=np.float32),
    )


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",   type=int,   default=2000)
    parser.add_argument("--level",   type=int,   default=7)
    parser.add_argument("--think",   type=float, default=0.5,
                        help="Think time per move (seconds)")
    parser.add_argument("--workers", type=int,   default=None,
                        help="Number of parallel workers (default: cpu_count-1)")
    parser.add_argument("--opening", type=int,   default=4,
                        help="Random opening moves")
    parser.add_argument("--output",  type=str,
                        default="ml/data/new_L7_parallel.npz")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 1)
    games_per_worker = args.games // n_workers
    extra = args.games % n_workers

    print(f"\n{'='*60}")
    print(f"  Parallel Dataset Generation")
    print(f"  Level     : {args.level}")
    print(f"  Think time: {args.think}s/move")
    print(f"  Games     : {args.games} ({n_workers} workers × ~{games_per_worker} games)")
    print(f"  Opening   : {args.opening} random moves")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}\n")

    # Estimate time
    # think_time * ~55 moves = time per game
    est_per_game = args.think * 60  # rough estimate
    est_total = games_per_worker * est_per_game
    print(f"Estimated: ~{est_total/60:.0f} min (rough, depends on search depth reached)")
    print(f"Starting {n_workers} workers...\n")

    # Build work args
    work_args = []
    for i in range(n_workers):
        n = games_per_worker + (1 if i < extra else 0)
        work_args.append((i, n, args.level, args.think, args.seed + i, args.opening))

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_games, work_args)

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Merge results
    all_states  = np.concatenate([r[0] for r in results], axis=0)
    all_masks   = np.concatenate([r[1] for r in results], axis=0)
    all_targets = np.concatenate([r[2] for r in results], axis=0)
    all_outcomes= np.concatenate([r[3] for r in results], axis=0)

    # Shuffle
    idx = np.random.default_rng(args.seed).permutation(len(all_states))
    all_states   = all_states[idx]
    all_masks    = all_masks[idx]
    all_targets  = all_targets[idx]
    all_outcomes = all_outcomes[idx]

    # Verify quality
    legal_at_target = all_masks[np.arange(len(all_targets)), all_targets]
    pct_legal = (legal_at_target > 0).mean()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output,
        states=all_states, legal_masks=all_masks,
        policy_targets=all_targets, outcomes=all_outcomes)

    sz = os.path.getsize(args.output) / 1024**2
    print(f"\n{'='*60}")
    print(f"  Saved: {args.output}")
    print(f"  Samples  : {len(all_states):,}")
    print(f"  Legal %  : {pct_legal:.1%}  (should be 100%)")
    print(f"  W/L/D    : {(all_outcomes==1).mean():.1%} / {(all_outcomes==-1).mean():.1%} / {(all_outcomes==0).mean():.1%}")
    print(f"  File size: {sz:.1f} MB")
    print(f"  Time     : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
