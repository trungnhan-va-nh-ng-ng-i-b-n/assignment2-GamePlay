"""
h2h_benchmark.py — Head-to-head: ML (level N) vs HC (level N).

Matchups:
  - ML max (search_level=7) vs Random
  - ML-L1  vs HC-L1
  - ML-L2  vs HC-L2
  ...
  - ML-L10 vs HC-L10

Usage (run from project root):
    py -u benchmark/h2h_benchmark.py --model "models/best_mlp_model (8).pt" --games 100 --workers 60
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
    worker_id, game_idx, ml_level, hc_level, ml_time, model_path, seed = args

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    from core import GameState
    from ml.mlp_agent import MLPAgent
    from Agents import SearchAgent, RandomAgent

    random.seed(seed)

    # Alternate colors each game for fairness
    if game_idx % 2 == 0:
        ml_color, opp_color = 1, -1
    else:
        ml_color, opp_color = -1, 1

    ml = MLPAgent(color=ml_color, checkpoint_path=model_path,
                  mode='negamax', search_level=ml_level)

    if hc_level == 0:
        opp = RandomAgent(color=opp_color)
    else:
        opp = SearchAgent(color=opp_color, difficulty=hc_level, remain_time=ml_time)

    board = [row[:] for row in INITIAL_BOARD]
    gs = GameState(board, 1)

    # FIX: apply_move returns a NEW GameState — must capture it!
    while not gs.is_game_over():
        current = gs.current_turn
        legal = gs.get_legal_moves(current)
        if not legal:
            gs.current_turn = -current
            continue

        if current == ml_color:
            move, _ = ml.get_move(gs, remain_time=ml_time)
        else:
            move, _ = opp.get_move(gs)

        gs = gs.apply_move(move, current)

    b = np.array(gs.board)
    black, white = int((b == 1).sum()), int((b == -1).sum())
    if black > white:
        winner = 1
    elif white > black:
        winner = -1
    else:
        winner = 0

    if winner == ml_color:
        return (1, 0, 0)
    elif winner == 0:
        return (0, 0, 1)
    else:
        return (0, 1, 0)


def run_matchup(label, ml_level, hc_level, n_games, workers, ml_time, model_path):
    print(f"  {label} ({n_games} games × {workers} workers)...")
    seeds = [random.randint(0, 2**31) for _ in range(n_games)]
    task_args = [
        (i % workers, i, ml_level, hc_level, ml_time, model_path, seeds[i])
        for i in range(n_games)
    ]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_play_one_game, task_args)

    W = sum(r[0] for r in results)
    L = sum(r[1] for r in results)
    D = sum(r[2] for r in results)
    n = W + L + D
    wr = W / n
    ci = 1.96 * (wr * (1 - wr) / n) ** 0.5
    elapsed = time.time() - t0
    result_str = f"W={W:3d} L={L:3d} D={D:2d}  WR={wr*100:.1f}% ±{ci*100:.1f}%"
    mark = "✅" if wr > 0.5 else "❌"
    print(f"  {label:<20} {result_str}  {mark}  ({elapsed:.0f}s)")
    return {"label": label, "W": W, "L": L, "D": D, "WR": wr, "CI": ci}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="models/best_mlp_model (8).pt")
    parser.add_argument("--games",   type=int,   default=100)
    parser.add_argument("--workers", type=int,   default=8)
    parser.add_argument("--time",    type=float, default=3.0)
    parser.add_argument("--depths",  type=int, nargs="+", default=[1, 3, 5, 7, 9])
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    print("=" * 65)
    print("  Head-to-Head Benchmark — ML(depth=D) vs HC(depth=D)")
    print(f"  Model  : {args.model}")
    print(f"  Games  : {args.games} per matchup")
    print(f"  Workers: {args.workers}")
    print(f"  Time   : {args.time}s/move")
    print(f"  Depths : {args.depths}  (odd only → no rounding)")
    print("=" * 65)

    all_results = []

    # 1. ML max (depth=9, 10s/move) vs Random — for 10/10 requirement
    all_results.append(
        run_matchup("ML-max vs Random", ml_level=9, hc_level=0,
                    n_games=args.games, workers=args.workers,
                    ml_time=10.0, model_path=args.model)
    )

    # 2. ML-D vs HC-D (same exact depth — odd only to avoid ML rounding)
    for d in args.depths:
        if d % 2 == 0:
            print(f"  [SKIP] depth={d} is even — ML would round to {d-1}, skipping.")
            continue
        label = f"ML-D{d:02d} vs HC-D{d:02d}"
        all_results.append(
            run_matchup(label, ml_level=d, hc_level=d,
                        n_games=args.games, workers=args.workers,
                        ml_time=args.time, model_path=args.model)
        )

    # Save chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["label"] for r in all_results]
        wrs    = [r["WR"] * 100 for r in all_results]
        cis    = [r["CI"] * 100 for r in all_results]
        colors = ["#2ecc71" if w >= 50 else "#e74c3c" for w in wrs]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels, wrs, xerr=cis, color=colors, alpha=0.85,
                       capsize=4, error_kw={"elinewidth": 1.5})
        ax.axvline(50, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Win Rate (%)")
        ax.set_title("Head-to-Head: ML (Level N) vs HC (Level N)")
        ax.set_xlim(0, 105)
        for bar, wr, ci in zip(bars, wrs, cis):
            ax.text(min(wr + ci + 1, 101), bar.get_y() + bar.get_height() / 2,
                    f"{wr:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2h_benchmark.png")
        plt.savefig(out, dpi=150)
        print(f"\nChart saved → {out}")
    except ImportError:
        print("matplotlib not installed, skipping chart.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
