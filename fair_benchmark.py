"""
fair_benchmark.py — Fair benchmark: ML+Negamax vs Minimax+HC

Fair = cùng search depth, HC remain_time giới hạn match với depth ML thực tế
"""
import argparse, sys, time, json, random
import numpy as np
sys.setrecursionlimit(10000)

from core import GameState
from Agents import SearchAgent, RandomAgent
from ml.mlp_agent import MLPAgent


def play_game(agent1, agent2, max_turns=130):
    state = GameState()
    turn  = 1
    for _ in range(max_turns):
        l1 = state.get_legal_moves(1)
        l2 = state.get_legal_moves(-1)
        if not l1 and not l2:
            break
        if not state.get_legal_moves(turn):
            turn = -turn
            continue
        agent = agent1 if turn == 1 else agent2
        move, _ = agent.get_move(state)
        if move is None or move not in state.get_legal_moves(turn):
            legal = state.get_legal_moves(turn)
            move = random.choice(legal) if legal else None
        state = state.apply_move(move, turn)
        turn  = -turn
    b, w = state.calculate_score()
    return 1 if b > w else (-1 if w > b else 0)


def run_match(fn_a, fn_b, n_games):
    wa = la = d = 0
    print("   ", end="", flush=True)
    for g in range(n_games):
        if g % 2 == 0:
            r = play_game(fn_a, fn_b); side = 1
        else:
            r = play_game(fn_b, fn_a); side = -1
        if r == side:    wa += 1; sym = "."
        elif r == 0:     d  += 1; sym = "="
        else:            la += 1; sym = "x"
        print(sym, end="", flush=True)
    print(f"  {wa}W/{la}L/{d}D", flush=True)
    return wa, la, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="best_mlp_model.pt")
    parser.add_argument("--games",  type=int, default=10)
    parser.add_argument("--levels", type=int, nargs="+", default=[1,2,3,4,5,6])
    parser.add_argument("--mode",   default="negamax",
                        choices=["negamax", "batch_negamax"],
                        help="negamax=seq depth<=4 | batch_negamax=depth<=6")
    args = parser.parse_args()

    # remain_time tương ứng với mỗi depth để HC search ngang bằng ML
    DEPTH_TIME = {1: 0.05, 2: 0.2, 3: 0.5, 4: 1.0, 5: 3.0, 6: 10.0, 7: 30.0, 8: 60.0}

    print(f"\n{'='*65}")
    print(f"  FAIR BENCHMARK: Negamax+ML  vs  Minimax+HC")
    print(f"  Model : {args.model}")
    print(f"  Mode  : {args.mode}")
    print(f"  Games : {args.games} per matchup (alternating colors)")
    print(f"  Rule  : HC remain_time khớp depth ML thực tế")
    print(f"{'='*65}\n")

    results = {}
    t_total = time.time()

    # ── A. vs Random ──────────────────────────────────────────────────
    print(f"{'─'*65}")
    print(f" A. ML (policy + 1ply) vs Random")
    print(f"{'─'*65}")

    ml_policy = MLPAgent(color=1, checkpoint_path=args.model, mode="policy")
    ml_1ply   = MLPAgent(color=1, checkpoint_path=args.model, mode="1ply")
    rand_ag   = RandomAgent(color=-1)

    for ag, label in [(ml_policy, "MLP-Policy"), (ml_1ply, "MLP-1ply")]:
        ag.color = 1; rand_ag.color = -1
        t0 = time.time()
        print(f"\n  {label} vs Random", flush=True)
        w, l, d = run_match(ag, rand_ag, args.games)
        wr = w / args.games
        flag = "✅" if wr > 0.5 else ("🟡" if wr >= 0.4 else "❌")
        print(f"  → ML: {wr:.0%}  {flag}  ({time.time()-t0:.0f}s)")
        results[f"{label} vs Random"] = {"w":w,"l":l,"d":d,"ml_win_rate":wr}

    # ── B. Fair Negamax+ML vs SearchAgent ─────────────────────────────
    print(f"\n{'─'*65}")
    print(f" B. FAIR: Negamax+ML(depth=D) vs SearchAgent(level=L, time~=D)")
    print(f"{'─'*65}")

    for level in args.levels:
        ml_neg = MLPAgent(
            color=1, checkpoint_path=args.model,
            mode=args.mode, search_level=level,
        )
        actual_depth = ml_neg.depth          # capped at 4
        rt = DEPTH_TIME.get(actual_depth, 1.0)

        hc_ag = SearchAgent(
            color=-1, difficulty=level,
            name=f"HC-L{level}", remain_time=rt
        )

        ml_neg.color = 1; hc_ag.color = -1
        t0 = time.time()
        print(f"\n  ML(depth={actual_depth}) vs HC-L{level}(remain={rt}s)", flush=True)
        w, l, d = run_match(ml_neg, hc_ag, args.games)
        ml_wr = w / args.games
        hc_wr = l / args.games
        elapsed = time.time() - t0
        flag = "✅ ML WINS!" if ml_wr > 0.5 else ("🟡 close" if ml_wr >= 0.4 else "❌ HC wins")
        print(f"  → ML:{ml_wr:.0%}  HC:{hc_wr:.0%}  {flag}  ({elapsed:.0f}s)")
        results[f"ML(d{actual_depth}) vs HC-L{level}"] = {
            "w":w,"l":l,"d":d,"ml_win_rate":ml_wr,"hc_win_rate":hc_wr,
            "level":level,"depth":actual_depth,"hc_time":rt
        }

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" SUMMARY")
    print(f"{'='*65}")
    for k, v in results.items():
        ml_wr = v.get("ml_win_rate", 0)
        hc_wr = v.get("hc_win_rate", 1 - ml_wr)
        flag  = "✅" if ml_wr > 0.5 else ("🟡" if ml_wr >= 0.4 else "❌")
        print(f"{flag}  {k:<45}  ML:{ml_wr:.0%}  HC:{hc_wr:.0%}")

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s")
    with open("fair_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → fair_benchmark.json")


if __name__ == "__main__":
    main()
