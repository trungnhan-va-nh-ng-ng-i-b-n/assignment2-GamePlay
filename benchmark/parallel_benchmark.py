"""
parallel_benchmark.py — Parallel benchmark: ML+negamax vs HC levels.

Chạy N games song song với multiprocessing (safe trên Windows).
Mỗi worker tự load model → không có GPU contention.

Usage (run from project root):
    py -u benchmark/parallel_benchmark.py --model models/"best_mlp_model (7).pt" --games 200 --workers 8
    py -u benchmark/parallel_benchmark.py --model models/"best_mlp_model (7).pt" --games 200 --workers 8 --time 3.0
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

    # Setup path for imports — go up one level from benchmark/ to project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

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
    parser.add_argument('--model',   default='models/best_mlp_model (7).pt')
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

    _plot_results(results, args.model, args.games, args.time)


def _plot_results(results, model_name, n_games, ml_time):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print('[chart] matplotlib not installed, skipping chart.')
        return

    labels   = list(results.keys())
    wrs      = [r[3] for r in results.values()]
    wins     = [r[0] for r in results.values()]
    losses   = [r[1] for r in results.values()]
    draws    = [r[2] for r in results.values()]
    cis      = [1.96*(wr/100*(1-wr/100)/n_games)**0.5*100 for wr in wrs]
    loss_pct = [l/n_games*100 for l in losses]
    draw_pct = [d/n_games*100 for d in draws]

    # ── Color theme ──────────────────────────────────────────────────────
    BG    = '#0d1117'; PANEL = '#161b22'; GRID  = '#21262d'
    TEXT  = '#e6edf3'; ACCENT= '#58a6ff'
    WIN   = '#3fb950'; DRAW  = '#d29922'; LOSS  = '#f85149'

    x = np.arange(len(labels)); W = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    facecolor=BG,
                                    gridspec_kw={'width_ratios': [1.8, 1]})
    fig.suptitle(f'OthelloMLP + Negamax  ·  Benchmark  ({n_games} games each)',
                 color=ACCENT, fontsize=16, fontweight='bold', y=1.01)

    # ── Left: Stacked bar ────────────────────────────────────────────────
    ax1.set_facecolor(PANEL)
    for sp in ax1.spines.values(): sp.set_color(GRID)
    ax1.tick_params(colors=TEXT)

    ax1.bar(x, wrs,      W, color=WIN,  label='Win',  zorder=3)
    ax1.bar(x, draw_pct, W, color=DRAW, label='Draw', bottom=wrs, zorder=3)
    ax1.bar(x, loss_pct, W, color=LOSS, label='Loss',
            bottom=[w+d for w,d in zip(wrs,draw_pct)], zorder=3)

    # Win % label + CI error bar
    ax1.errorbar(x, wrs, yerr=cis, fmt='none', color='white',
                 capsize=5, capthick=1.5, elinewidth=1.5, zorder=5)
    for xi, wr in zip(x, wrs):
        if wr >= 10:
            ax1.text(xi, wr/2, f'{wr:.0f}%', ha='center', va='center',
                     color='#0d1117', fontsize=11, fontweight='bold')

    ax1.set_xticks(x); ax1.set_xticklabels(labels, color=TEXT, fontsize=11)
    ax1.set_ylim(0, 115); ax1.set_ylabel('Outcome (%)', color=TEXT, fontsize=11)
    ax1.set_title('Win / Draw / Loss  (error bars = 95% CI)', color=TEXT, fontsize=12, pad=10)
    ax1.yaxis.grid(True, color=GRID, linestyle='--', alpha=0.6, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(handles=[mpatches.Patch(color=WIN, label='Win'),
                         mpatches.Patch(color=DRAW, label='Draw'),
                         mpatches.Patch(color=LOSS, label='Loss')],
               facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
               fontsize=10, loc='lower left')

    # ── Right: Win-rate trend + CI band ─────────────────────────────────
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_color(GRID)
    ax2.tick_params(colors=TEXT)

    ci_arr = np.array(cis)
    wr_arr = np.array(wrs)
    ax2.fill_between(x, wr_arr-ci_arr, wr_arr+ci_arr,
                     color=ACCENT, alpha=0.15, zorder=2)
    ax2.plot(x, wr_arr, color=ACCENT, linewidth=2.5, zorder=3)

    dot_colors = [WIN if w>=60 else (DRAW if w>=40 else LOSS) for w in wrs]
    for xi, wr, c, ci in zip(x, wrs, dot_colors, cis):
        ax2.scatter(xi, wr, s=110, color=c, zorder=5,
                    edgecolors='white', linewidth=1.5)
        ax2.text(xi, wr+ci+4, f'{wr:.0f}%', ha='center',
                 color=c, fontsize=10, fontweight='bold')

    for ref, col, lbl in [(50,'#8b949e','50%'), (80,WIN,'80%'), (100,ACCENT,'100%')]:
        ax2.axhline(ref, color=col, linewidth=1, linestyle='--', alpha=0.6)
        ax2.text(len(labels)-0.6, ref+2, lbl, color=col, fontsize=9)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, color=TEXT, fontsize=10, rotation=10)
    ax2.set_ylim(0, 118); ax2.set_ylabel('Win rate (%)', color=TEXT, fontsize=11)
    ax2.set_title('Win Rate Trend  (shaded = 95% CI)', color=TEXT, fontsize=12, pad=10)
    ax2.yaxis.grid(True, color=GRID, linestyle='--', alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    fig.text(0.5, -0.04,
        f'Model: {os.path.basename(model_name)}  |  '
        f'Search: Negamax + alpha-beta + policy ordering  |  '
        f'{ml_time}s/move  |  n={n_games} games/matchup',
        ha='center', color='#8b949e', fontsize=9)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_parallel.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
    print(f'[chart] Saved → {out}')
    try:
        plt.show()
    except Exception:
        pass  # Headless server


if __name__ == '__main__':
    mp.freeze_support()  # Required for Windows
    main()

