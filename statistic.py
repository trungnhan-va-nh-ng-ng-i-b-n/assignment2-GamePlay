from core import GameState
from Agents import RandomAgent, SearchAgent


def _new_bucket():
    return {
        "moves": 0,
        "total_move_time": 0.0,
        "max_move_time": 0.0,
        "total_nodes": 0,
        "total_pruned": 0,
        "total_depth": 0,
        "invalid_moves": 0,
    }


def _update_bucket(bucket, metrics):
    move_time = float(metrics.get("move_time", 0.0))
    bucket["moves"] += 1
    bucket["total_move_time"] += move_time
    bucket["max_move_time"] = max(bucket["max_move_time"], move_time)
    bucket["total_nodes"] += int(metrics.get("nodes_explored", 0))
    bucket["total_pruned"] += int(metrics.get("pruned_branches", 0))
    bucket["total_depth"] += int(metrics.get("depth_reached", 0))


def _merge_bucket(dst, src):
    dst["moves"] += src["moves"]
    dst["total_move_time"] += src["total_move_time"]
    dst["max_move_time"] = max(dst["max_move_time"], src["max_move_time"])
    dst["total_nodes"] += src["total_nodes"]
    dst["total_pruned"] += src["total_pruned"]
    dst["total_depth"] += src["total_depth"]
    dst["invalid_moves"] += src["invalid_moves"]


def _bucket_summary(bucket):
    moves = max(1, bucket["moves"])
    return {
        "moves": bucket["moves"],
        "avg_move_time": bucket["total_move_time"] / moves,
        "avg_game_time": 0.0,
        "max_move_time": bucket["max_move_time"],
        "avg_nodes": bucket["total_nodes"] / moves,
        "avg_pruned": bucket["total_pruned"] / moves,
        "avg_depth": bucket["total_depth"] / moves,
        "invalid_moves": bucket["invalid_moves"],
    }

def simulate_game(agent1, agent2):
    """
    Run a headless game between two agents.
    Returns: 1 if agent1 wins, -1 if agent2 wins, 0 if draw.
    """
    game_state = GameState()
    metrics = {
        "agent1": _new_bucket(),
        "agent2": _new_bucket(),
    }
    
    while not game_state.is_game_over():
        turn = game_state.current_turn
        active_agent = agent1 if turn == 1 else agent2
        
        move, move_metrics = active_agent.get_move(game_state)
        side_key = "agent1" if active_agent is agent1 else "agent2"
        _update_bucket(metrics[side_key], move_metrics)
        
        # Apply move
        valid_moves = game_state.get_legal_moves(turn)
        if not valid_moves:
            valid_moves.append(None)
            
        if move not in valid_moves:
            print(f"{active_agent.name} attempted invalid move {move}. Defaulting to loss.")
            metrics[side_key]["invalid_moves"] += 1
            black_score, white_score = game_state.calculate_score()
            return -turn, metrics, black_score, white_score
            
        game_state = game_state.apply_move(move, turn)

    black_score, white_score = game_state.calculate_score()
    if black_score > white_score:
        return 1, metrics, black_score, white_score
    elif white_score > black_score:
        return -1, metrics, black_score, white_score
    else:
        return 0, metrics, black_score, white_score

def run_batch(agent1_factory, agent2_factory, num_games=10):
    print(f"Running {num_games} matches...")
    results: dict = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}
    total_score_diff_agent1 = 0
    total_metrics = {
        "agent1": _new_bucket(),
        "agent2": _new_bucket(),
    }
    
    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            a1 = agent1_factory(color=1, name="Agent1 (Black)")
            a2 = agent2_factory(color=-1, name="Agent2 (White)")
            winner, game_metrics, black_score, white_score = simulate_game(a1, a2)
            total_score_diff_agent1 += black_score - white_score
            if winner == 1:
                results["agent1_wins"] += 1
            elif winner == -1:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
        else:
            # agent1 plays White (-1), agent2 plays Black (1)
            a2 = agent2_factory(color=1, name="Agent2 (Black)")
            a1 = agent1_factory(color=-1, name="Agent1 (White)")
            winner, swapped, black_score, white_score = simulate_game(a2, a1)
            total_score_diff_agent1 += white_score - black_score
            game_metrics = {
                "agent1": swapped["agent2"],
                "agent2": swapped["agent1"],
            }
            if winner == -1:
                results["agent1_wins"] += 1
            elif winner == 1:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1

        _merge_bucket(total_metrics["agent1"], game_metrics["agent1"])
        _merge_bucket(total_metrics["agent2"], game_metrics["agent2"])
                
        print(f"Game {i+1}/{num_games} finished.")
        
    print("\n--- RESULTS ---")
    print(f"Agent 1 Wins: {results['agent1_wins']}")
    print(f"Agent 2 Wins: {results['agent2_wins']}")
    print(f"Draws: {results['draws']}")

    total_games = max(1, num_games)
    results["win_rate"] = results["agent1_wins"] / total_games
    results["draw_rate"] = results["draws"] / total_games
    results["loss_rate"] = results["agent2_wins"] / total_games
    results["avg_score_diff"] = total_score_diff_agent1 / total_games

    agent1_stats = _bucket_summary(total_metrics["agent1"])
    agent2_stats = _bucket_summary(total_metrics["agent2"])
    agent1_stats["avg_game_time"] = total_metrics["agent1"]["total_move_time"] / total_games
    agent2_stats["avg_game_time"] = total_metrics["agent2"]["total_move_time"] / total_games

    print(
        f"Rates (Agent1): win={results['win_rate']:.2%}, "
        f"draw={results['draw_rate']:.2%}, loss={results['loss_rate']:.2%}"
    )
    print(f"Average score difference (Agent1 - Agent2): {results['avg_score_diff']:.2f}")
    print("Agent 1 Metrics:", agent1_stats)
    print("Agent 2 Metrics:", agent2_stats)
    return results, {"agent1": agent1_stats, "agent2": agent2_stats}


def benchmark_search_levels(num_games_per_level=20):
    report = []
    print(f"Running difficulty benchmark for {num_games_per_level} games/level")

    for level in range(1, 11):
        print(f"\n===== Difficulty {level} =====")
        results, stats = run_batch(
            lambda color, name: SearchAgent(color=color, difficulty=level, name=f"Search-L{level}"),
            lambda color, name: RandomAgent(color=color, name=name),
            num_games=num_games_per_level,
        )
        win_rate = results["agent1_wins"] / max(1, num_games_per_level)
        report.append(
            {
                "difficulty": level,
                "win_rate": win_rate,
                "draw_rate": results["draw_rate"],
                "loss_rate": results["loss_rate"],
                "wins": results["agent1_wins"],
                "draws": results["draws"],
                "losses": results["agent2_wins"],
                "avg_score_diff": results["avg_score_diff"],
                "agent1_metrics": stats["agent1"],
            }
        )

    print("\n=== SUMMARY (SearchAgent vs RandomAgent) ===")
    print("Depth | Win% | Draw% | Loss% | AvgScoreDiff | AvgTime/Move(s) | AvgTime/Game(s) | AvgNodes | AvgDepth")
    for row in report:
        print(
            f"{row['difficulty']:>5} | {row['win_rate']*100:>4.1f}% | "
            f"{row['draw_rate']*100:>5.1f}% | {row['loss_rate']*100:>5.1f}% | "
            f"{row['avg_score_diff']:>12.2f} | "
            f"{row['agent1_metrics']['avg_move_time']:>15.4f} | "
            f"{row['agent1_metrics']['avg_game_time']:>15.4f} | "
            f"{row['agent1_metrics']['avg_nodes']:>8.2f} | "
            f"{row['agent1_metrics']['avg_depth']:>8.2f}"
        )

    return report

if __name__ == "__main__":
    run_batch(
        lambda color, name: RandomAgent(color=color, name=name),
        lambda color, name: RandomAgent(color=color, name=name),
        num_games=10,
    )
