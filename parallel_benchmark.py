"""
parallel_benchmark.py — Parallel benchmark: ML+negamax vs HC levels.

Chạy N games song song với multiprocessing (safe trên Windows).
Mỗi worker tự load model → không có GPU contention.

Usage:
    py -u parallel_benchmark.py --model best_mlp_model.pt --games 200 --workers 60
    py -u parallel_benchmark.py --model best_mlp_model.pt --games 200 --workers 60 --time 3.0
"""
import argparse, os, sys, time, random
import multiprocessing as mp
import numpy as np


INITIAL_BOARD = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,-1,1,0,0,0],
    [0,0,0,1,-1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
]


def _play_one_game(args):
    """Worker function: play one game, return (win, loss, draw)."""
    worker_id, game_idx, level, ml_time, model_path, seed = args

    # Setup path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from core import GameState
    from ml.mlp_agent import MLPAgent
    from Agents import SearchAgent, RandomAgent

    random.seed(seed)

    # Load model (each worker loads independently)
    ml = MLPAgent(color=1, checkpoint_path=model_path, mode='negamax')

    if level == 0:
        opp = RandomAgent(color=-1)
    else:
        opp = SearchAgent(color=-1, difficulty=level, remain_time=3.0)

    # Alternate colors each game for fairness
    if game_idx % 2 == 0:
        ml_color, opp_color = 1, -1
    else:
        ml_color, opp_color = -1, 1

    ml.color = ml_color
    opp.color = opp_color

    board = [row[:] for row in INITIAL_BOARD]
    gs = GameState(board, 1)
    turn = 1

    for _ in range(130):
        legal = gs.get_legal_moves(turn)
        opp_legal = gs.get_legal_moves(-turn)
        if not legal and not opp_legal:
            break
        if not legal:
            turn = -turn
            continue

        if turn == ml_color:
            move, _ = ml.get_move(gs, ml_time)
        else:
            move, _ = opp.get_move(gs)

        if move is None or move not in legal:
            move = legal[0]

        gs = gs.apply_move(move, turn)
        turn = -turn

    b, w = gs.calculate_score()
    ml_score = b if ml_color == 1 else w
    opp_score = w if ml_color == 1 else b

    if ml_score > opp_score:
        return (1, 0, 0)
    elif ml_score < opp_score:
        return (0, 1, 0)
    else:
        return (0, 0, 1)


def run_matchup(label, level, n_games, ml_time, model_path, n_workers, base_seed):
    """Run n_games in parallel, return (wins, losses, draws)."""
    args_list = [
        (i, i, level, ml_time, model_path, base_seed + i)
        for i in range(n_games)
    ]

    with mp.Pool(processes=min(n_workers, n_games)) as pool:
        results = pool.map(_play_one_game, args_list)

    wins   = sum(r[0] for r in results)
    losses = sum(r[1] for r in results)
    draws  = sum(r[2] for r in results)
    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='best_mlp_model (7).pt')
    parser.add_argument('--games',   type=int,   default=200)
    parser.add_argument('--workers', type=int,   default=60)
    parser.add_argument('--time',    type=float, default=1.0,
                        help='ML think time per move (seconds)')
    parser.add_argument('--levels',  type=str,   default='0,1,2,3,4,5,6,7',
                        help='Comma-separated HC levels (0=Random)')
    parser.add_argument('--seed',    type=int,   default=42)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(',')]

    print(f'\n{"="*60}')
    print(f'  Parallel Benchmark — ML+Negamax')
    print(f'  Model   : {args.model}')
    print(f'  ML time : {args.time}s/move')
    print(f'  Games   : {args.games} per matchup')
    print(f'  Workers : {args.workers}')
    print(f'  Levels  : {levels}')
    print(f'{"="*60}\n')

    results = {}
    total_t0 = time.time()

    for level in levels:
        label = 'Random' if level == 0 else f'HC-L{level}'
        print(f'  Running {label} ({args.games} games × {args.workers} workers)...', flush=True)
        t0 = time.time()

        w, l, d = run_matchup(label, level, args.games, args.time,
                               args.model, args.workers, args.seed)
        elapsed = time.time() - t0

        wr = w / args.games * 100
        flag = '✅' if wr >= 60 else ('🟡' if wr >= 40 else '❌')
        ci = 1.96 * (wr/100 * (1-wr/100) / args.games) ** 0.5 * 100
        print(f'  {label:8s}  W={w:3d} L={l:3d} D={d:2d}  '
              f'WR={wr:5.1f}% ±{ci:.1f}%  {flag}  ({elapsed:.0f}s)')
        results[label] = (w, l, d, wr)

    total_elapsed = time.time() - total_t0

    print(f'\n{"="*60}')
    print(f'  SUMMARY  (model: {args.model}  time={args.time}s/move)')
    print(f'{"="*60}')
    for label, (w, l, d, wr) in results.items():
        ci = 1.96 * (wr/100 * (1-wr/100) / args.games) ** 0.5 * 100
        flag = '✅' if wr >= 60 else ('🟡' if wr >= 40 else '❌')
        print(f'  {label:8s}  WR = {wr:5.1f}% ± {ci:.1f}%  {flag}')
    avg = sum(r[3] for r in results.values()) / len(results)
    print(f'  {"Average":8s}  WR = {avg:5.1f}%')
    print(f'  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    mp.freeze_support()  # Required for Windows
    main()
