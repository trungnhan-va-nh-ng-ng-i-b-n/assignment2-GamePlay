"""
benchmark_chart.py — Chạy ML+negamax vs Random + HC L1-L6, vẽ chart đẹp.
Usage:
    python3 benchmark_chart.py --model "best_mlp_model (6).pt" --games 20
"""
import argparse, time, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_match(agent1, agent2, n_games):
    from core import GameState
    INITIAL = [[0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,-1,1,0,0,0],
               [0,0,0,1,-1,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0]]
    w = l = d = 0
    for g in range(n_games):
        # Alternate colors each game
        if g % 2 == 0:
            a1_color, a2_color = 1, -1
        else:
            a1_color, a2_color = -1, 1
        agent1.color = a1_color
        agent2.color = a2_color

        gs = GameState(INITIAL, 1)
        turn = 1
        for _ in range(130):
            legal = gs.get_legal_moves(turn)
            opp   = gs.get_legal_moves(-turn)
            if not legal and not opp:
                break
            if not legal:
                turn = -turn
                continue
            if turn == a1_color:
                move, _ = agent1.get_move(gs)
            else:
                move, _ = agent2.get_move(gs)
            if move is None or move not in legal:
                move = legal[0]
            gs = gs.apply_move(move, turn)
            turn = -turn

        b, wh = gs.calculate_score()
        a1_score = b if a1_color == 1 else wh
        a2_score = wh if a1_color == 1 else b
        if a1_score > a2_score:   w += 1
        elif a1_score < a2_score: l += 1
        else:                     d += 1
        sys.stdout.write('.' if a1_score > a2_score else ('=' if a1_score == a2_score else 'x'))
        sys.stdout.flush()
    print()
    return w, l, d


def plot_chart(results, model_name, n_games):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    opponents  = list(results.keys())
    win_rates  = [results[o]['win']  for o in opponents]
    draw_rates = [results[o]['draw'] for o in opponents]
    loss_rates = [results[o]['loss'] for o in opponents]

    # ── Color palette ───────────────────────────────────────────────
    WIN_COLOR  = '#4ade80'   # green
    DRAW_COLOR = '#facc15'   # yellow
    LOSS_COLOR = '#f87171'   # red
    BG         = '#0f172a'   # dark navy
    PANEL      = '#1e293b'
    TEXT       = '#e2e8f0'
    ACCENT     = '#38bdf8'   # sky blue

    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=0.35)
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Left: stacked bar chart ──────────────────────────────────────
    ax.set_facecolor(PANEL)
    x = np.arange(len(opponents))
    bar_w = 0.55

    b1 = ax.bar(x, win_rates,  bar_w, label='Win',  color=WIN_COLOR,  zorder=3)
    b2 = ax.bar(x, draw_rates, bar_w, label='Draw', color=DRAW_COLOR, zorder=3,
                bottom=win_rates)
    b3 = ax.bar(x, loss_rates, bar_w, label='Loss', color=LOSS_COLOR, zorder=3,
                bottom=[w+d for w,d in zip(win_rates, draw_rates)])

    # Value labels on win bars
    for bar, wr in zip(b1, win_rates):
        if wr > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{wr:.0f}%', ha='center', va='center',
                    color='#052e16', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(opponents, color=TEXT, fontsize=12)
    ax.set_ylabel('Game outcome (%)', color=TEXT, fontsize=12)
    ax.set_ylim(0, 105)
    ax.tick_params(colors=TEXT, which='both')
    ax.spines[:].set_color('#334155')
    ax.yaxis.set_tick_params(color=TEXT)
    ax.set_title(f'ML + Negamax  vs  Opponents\n'
                 f'Model: {os.path.basename(model_name)}  |  {n_games} games each',
                 color=TEXT, fontsize=13, pad=14)

    # Grid
    ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(handles=[
        mpatches.Patch(color=WIN_COLOR,  label='Win'),
        mpatches.Patch(color=DRAW_COLOR, label='Draw'),
        mpatches.Patch(color=LOSS_COLOR, label='Loss'),
    ], loc='lower left', facecolor=PANEL, edgecolor='#334155',
       labelcolor=TEXT, fontsize=11)

    # ── Right: win rate line/dot chart ──────────────────────────────
    ax2.set_facecolor(PANEL)
    ax2.spines[:].set_color('#334155')
    ax2.tick_params(colors=TEXT)

    ax2.plot(x, win_rates, color=ACCENT, linewidth=2.5, zorder=3, marker='o',
             markersize=9, markerfacecolor=ACCENT, markeredgewidth=0)

    for xi, wr in zip(x, win_rates):
        ax2.text(xi, wr + 3, f'{wr:.0f}%', ha='center', color=ACCENT,
                 fontsize=11, fontweight='bold')

    # Reference lines
    for ref, lbl, col in [(50, '50%', '#94a3b8'), (80, '80%', WIN_COLOR)]:
        ax2.axhline(ref, color=col, linewidth=1, linestyle='--', alpha=0.7)
        ax2.text(len(opponents)-0.5, ref+2, lbl, color=col, fontsize=9, ha='right')

    ax2.set_xticks(x)
    ax2.set_xticklabels(opponents, color=TEXT, fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylabel('Win rate (%)', color=TEXT, fontsize=12)
    ax2.set_title('Win Rate Trend', color=TEXT, fontsize=12, pad=14)

    fig.suptitle('Othello ML Agent Benchmark', color=ACCENT,
                 fontsize=17, fontweight='bold', y=1.01)

    out = 'benchmark_chart.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'\n✅  Chart saved → {out}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='best_mlp_model (6).pt')
    parser.add_argument('--games',  type=int, default=20)
    parser.add_argument('--time',   type=float, default=1.0,
                        help='remain_time per move for ML (seconds)')
    args = parser.parse_args()

    from ml.mlp_agent import MLPAgent
    from Agents import SearchAgent, RandomAgent

    ml = MLPAgent(color=1, checkpoint_path=args.model, mode='negamax')
    print(f'\nModel loaded. Running {args.games} games per matchup (remain={args.time}s/move)\n')

    # Monkey-patch get_move để pass remain_time
    import functools
    orig_get_move = ml.get_move
    ml.get_move = functools.partial(orig_get_move, remain_time=args.time)

    opponents_cfg = [
        ('vs Random',  None),
        ('vs HC-L1',   1),
        ('vs HC-L2',   2),
        ('vs HC-L3',   3),
        ('vs HC-L4',   4),
        ('vs HC-L5',   5),
        ('vs HC-L6',   6),
    ]

    results = {}
    for label, level in opponents_cfg:
        if level is None:
            opp = RandomAgent(color=-1)
        else:
            opp = SearchAgent(color=-1, difficulty=level, remain_time=3.0)

        print(f'  {label:12s}', end='  ', flush=True)
        t0 = time.perf_counter()
        w, l, d = run_match(ml, opp, args.games)
        elapsed = time.perf_counter() - t0
        wr = w / args.games * 100
        lr = l / args.games * 100
        dr = d / args.games * 100
        results[label] = {'win': wr, 'draw': dr, 'loss': lr}
        flag = '✅' if wr >= 60 else ('🟡' if wr >= 40 else '❌')
        print(f'  W={w} D={d} L={l}  WR={wr:.0f}%  {flag}  ({elapsed:.0f}s)')

    print('\n' + '='*55)
    print('SUMMARY')
    print('='*55)
    for label, r in results.items():
        print(f'  {label:12s}  Win={r["win"]:.0f}%  Draw={r["draw"]:.0f}%  Loss={r["loss"]:.0f}%')

    plot_chart(results, args.model, args.games)


if __name__ == '__main__':
    main()
